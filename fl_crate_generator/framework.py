"""Framework detection: capture the ML framework and its version.

Stian asked that the framework (PyTorch, scikit-learn, etc.) and its version be
recorded. We read the declared dependency from the app's pyproject.toml (the
version spec the user pinned) and also resolve the actually-installed version
via importlib.metadata, which is the precise version that ran.
"""

import logging
import re
from importlib import metadata

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

logger = logging.getLogger("fl_crate_generator")

# Distribution name -> (display name, homepage). Extend as needed.
KNOWN_FRAMEWORKS = {
    "torch": ("PyTorch", "https://pytorch.org/"),
    "torchvision": ("TorchVision", "https://pytorch.org/vision/"),
    "scikit-learn": ("scikit-learn", "https://scikit-learn.org/"),
    "sklearn": ("scikit-learn", "https://scikit-learn.org/"),
    "tensorflow": ("TensorFlow", "https://www.tensorflow.org/"),
    "jax": ("JAX", "https://jax.readthedocs.io/"),
    "mlx": ("MLX", "https://ml-explore.github.io/mlx/"),
    "xgboost": ("XGBoost", "https://xgboost.readthedocs.io/"),
    "transformers": ("Transformers", "https://huggingface.co/docs/transformers"),
}


def _split_requirement(dep: str):
    """'torch==2.8.0' -> ('torch', '==2.8.0'); 'scikit-learn>=1.3' -> (..., '>=1.3')."""
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", dep)
    if not m:
        return dep.strip().lower(), ""
    name = m.group(1).lower()
    spec = dep[m.end():].strip()
    # Drop extras and environment markers for a clean spec, e.g. "flwr[simulation]>=1.28" 
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
    """Return ML frameworks declared in pyproject.toml with declared + installed versions."""
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
    except FileNotFoundError:
        logger.warning("pyproject.toml not found at %s; no framework captured.", pyproject_path)
        return []

    deps = pyproject.get("project", {}).get("dependencies", []) or []
    found, seen = [], set()
    for dep in deps:
        name, spec = _split_requirement(dep)
        if name in KNOWN_FRAMEWORKS and name not in seen:
            seen.add(name)
            display, homepage = KNOWN_FRAMEWORKS[name]
            found.append({
                "package": name,
                "name": display,
                "homepage": homepage,
                "declared": spec or None,                 # version spec from pyproject.toml
                "installed_version": _installed_version(name),  # version that actually ran
            })
    return found
