"""Unit tests for flwrcrate.framework -- dependency / software-environment capture."""

from flwrcrate.framework import (
    _split_requirement,
    _installed_version,
    detect_frameworks,
    INFRA_DENYLIST,
    KNOWN_FRAMEWORKS,
)


# --- _split_requirement: parse a dependency string into (name, spec) ------

def test_split_pinned_version():
    assert _split_requirement("torch==2.8.0") == ("torch", "==2.8.0")


def test_split_lower_bound():
    assert _split_requirement("scikit-learn>=1.3") == ("scikit-learn", ">=1.3")


def test_split_name_lowercased():
    name, _ = _split_requirement("TorchVision==0.23.0")
    assert name == "torchvision"


def test_split_strips_extras():
    # extras in brackets must be dropped from the spec
    name, spec = _split_requirement("flwr[simulation]>=1.28")
    assert name == "flwr"
    assert spec == ">=1.28"


def test_split_strips_environment_marker():
    name, spec = _split_requirement("tomli>=2.0; python_version < '3.11'")
    assert name == "tomli"
    assert ";" not in spec


def test_split_bare_name_has_empty_spec():
    assert _split_requirement("numpy") == ("numpy", "")


# --- _installed_version: query the live environment -----------------------

def test_installed_version_found():
    # pytest is certainly installed in the test environment
    assert _installed_version("pytest") is not None


def test_installed_version_missing_returns_none():
    assert _installed_version("definitely-not-a-real-package-xyz") is None


# --- detect_frameworks: the whole capture, against a fixture pyproject -----

def test_detect_returns_declared_and_installed(app_pyproject):
    frameworks = detect_frameworks(str(app_pyproject))
    by_pkg = {f["package"]: f for f in frameworks}

    # torch is declared and recognised
    assert "torch" in by_pkg
    assert by_pkg["torch"]["declared"] == "==2.8.0"
    assert by_pkg["torch"]["name"] == "PyTorch"
    assert by_pkg["torch"]["known_framework"] is True

    # pytest is installed in this env, so the installed-version lookup resolves
    assert by_pkg["pytest"]["installed_version"] is not None
    assert by_pkg["pytest"]["known_framework"] is False  # not an ML framework


def test_detect_excludes_infrastructure(app_pyproject):
    packages = {f["package"] for f in detect_frameworks(str(app_pyproject))}
    # flwr is on the deny-list (captured separately as the FL framework)
    assert "flwr" not in packages


def test_detect_missing_pyproject_returns_empty(tmp_path):
    assert detect_frameworks(str(tmp_path / "nope.toml")) == []


def test_flwrcrate_denies_itself():
    # the tool must never list itself as software used in the run
    assert "flwrcrate" in INFRA_DENYLIST


def test_known_frameworks_have_name_and_homepage():
    for pkg, (name, homepage) in KNOWN_FRAMEWORKS.items():
        assert name and homepage.startswith("http")
