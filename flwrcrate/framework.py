"""Framework / software-environment detection.

Captures the software a Flower app declares in its ``pyproject.toml``, each with
the version spec the user pinned (from ``[project].dependencies``) and the
actually-installed version (``importlib.metadata``).
"""

import logging
import re
from importlib import metadata

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

logger = logging.getLogger("flwrcrate")

# Recognised ML / data-science frameworks: package -> (display name, homepage).
KNOWN_FRAMEWORKS = {
    "torch": ("PyTorch", "https://pytorch.org/"),
    "torchvision": ("TorchVision", "https://pytorch.org/vision/"),
    "pytorch-lightning": ("PyTorch Lightning", "https://lightning.ai/"),
    "lightning": ("Lightning", "https://lightning.ai/"),
    "scikit-learn": ("scikit-learn", "https://scikit-learn.org/"),
    "sklearn": ("scikit-learn", "https://scikit-learn.org/"),
    "tensorflow": ("TensorFlow", "https://www.tensorflow.org/"),
    "keras": ("Keras", "https://keras.io/"),
    "jax": ("JAX", "https://jax.readthedocs.io/"),
    "flax": ("Flax", "https://flax.readthedocs.io/"),
    "mlx": ("MLX", "https://ml-explore.github.io/mlx/"),
    "xgboost": ("XGBoost", "https://xgboost.readthedocs.io/"),
    "lightgbm": ("LightGBM", "https://lightgbm.readthedocs.io/"),
    "catboost": ("CatBoost", "https://catboost.ai/"),
    "statsmodels": ("statsmodels", "https://www.statsmodels.org/"),
    "prophet": ("Prophet", "https://facebook.github.io/prophet/"),
    "transformers": ("Transformers", "https://huggingface.co/docs/transformers"),
    "datasets": ("Hugging Face Datasets", "https://huggingface.co/docs/datasets"),
}

# Infrastructure / glue packages we never report as "software used" for the run.
# (Flower itself is recorded separately, as the FL framework in tracker.py.)
INFRA_DENYLIST = {
    "flwr", "flwr-datasets", "flwr-nightly", "ray",
    "flwrcrate", "fl-crate-generator", "rocrate", "tomli",
    "hatchling", "setuptools", "wheel", "pip", "build",
}


def _split_requirement(dep: str):
    """'torch==2.8.0' -> ('torch', '==2.8.0'); 'scikit-learn>=1.3' -> (..., '>=1.3')."""
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", dep)
    if not m:
        return dep.strip().lower(), ""
    name = m.group(1).lower()
    spec = dep[m.end():].strip()
    # Drop extras and environment markers for a clean spec, e.g. "flwr[simulation]>=1.28".
    spec = spec.split(";")[0].strip()
    if spec.startswith("["):
        spec = spec.split("]", 1)[-1].strip()
    return name, spec


def _installed_version(dist_name: str):
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def detect_frameworks(pyproject_path: str = "pyproject.toml") -> list:
    """Return the declared software dependencies (minus infrastructure) with
    declared + installed versions. Recognised ML frameworks are flagged via
    ``known_framework`` and given a friendly name + homepage."""
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
    except FileNotFoundError:
        logger.warning("pyproject.toml not found at %s; no software captured.", pyproject_path)
        return []

    deps = pyproject.get("project", {}).get("dependencies", []) or []
    found, seen = [], set()
    for dep in deps:
        name, spec = _split_requirement(dep)
        if not name or name in seen or name in INFRA_DENYLIST:
            continue
        seen.add(name)
        known = name in KNOWN_FRAMEWORKS
        if known:
            display, homepage = KNOWN_FRAMEWORKS[name]
        else:
            display, homepage = name, None
            logger.info(
                "Recording dependency %r as software used (not in the known-"
                "frameworks map; add it to KNOWN_FRAMEWORKS for a friendly "
                "name + homepage).", name,
            )
        found.append({
            "package": name,
            "name": display,
            "homepage": homepage,
            "declared": spec or None,                  # version spec from pyproject.toml
            "installed_version": _installed_version(name),  # version that actually ran
            "known_framework": known,
        })
    return found
