"""Tier A integration tests: real captured data from real Flower apps.

Unlike test_integration.py (which uses hand-built mocks), these feed the
*actual* ``captured_metadata.json`` produced by running each app in the
Tested-with table through ``build_crate`` and assert the resulting RO-Crate is
correct for that app. The fixtures under tests/fixtures/<app>/ were harvested
from real ``flwr run`` executions, so this proves flwrCrate handles the genuine
diversity of strategies, frameworks, and metrics across different apps --
deterministically and without needing Flower/Ray installed.

The heavier "actually run the app end to end" coverage lives in the Tier B
realapps.yml CI workflow (see tests/test_realapp_e2e.py).
"""

import json
from pathlib import Path

import pytest

from flwrcrate.crate_builder import build_crate
from flwrcrate.metrics import load_metric_uri_map, metricrecord_to_dict

FIXTURES = Path(__file__).parent / "fixtures"


# Expected shape per real app, keyed by fixture directory name.
REAL_APPS = [
    pytest.param(
        "quickstart-pytorch",
        {"strategy": "FedAvg", "frameworks": ["torch", "torchvision"], "metric": "accuracy"},
        id="quickstart-pytorch",
    ),
    pytest.param(
        "fed-engines",
        {"strategy": "FedProx", "frameworks": ["torch", "datasets"], "metric": "val_anomaly_recall"},
        id="fed-engines",
    ),
    pytest.param(
        "quickstart-sklearn",
        {"strategy": "FedAvg", "frameworks": ["scikit-learn"], "metric": "accuracy"},
        id="quickstart-sklearn",
    ),
]


def _load(app):
    captured = json.loads((FIXTURES / app / "captured_metadata.json").read_text())
    log = FIXTURES / app / "metrics_log.json"
    return captured, (log if log.exists() else None)


def _graph(crate_dir):
    data = json.loads((crate_dir / "ro-crate-metadata.json").read_text())
    return {e["@id"]: e for e in data["@graph"]}


@pytest.mark.parametrize("app, expect", REAL_APPS)
def test_real_app_builds_valid_crate(app, expect, tmp_path):
    captured, log = _load(app)
    crate_dir = build_crate(
        captured,
        crate_dir=tmp_path / "ro-crate",
        metrics_log_path=log,
        author={"name": "Test", "orcid": "https://orcid.org/0000-0000-0000-0000"},
        license="https://spdx.org/licenses/MIT.html",
    )
    g = _graph(crate_dir)

    # the run is discoverable and complete
    assert g["./"]["mentions"] == [{"@id": "#fl-run"}]
    assert g["#fl-run"]["actionStatus"]["@id"].endswith("CompletedActionStatus")

    # the strategy this app actually used (FedAvg vs FedProx -> not hardcoded)
    assert g["#fl-strategy"]["name"] == expect["strategy"]

    # every framework this app declared was captured as a SoftwareApplication
    for pkg in expect["frameworks"]:
        slug = pkg.replace(".", "-")
        assert f"#framework-{slug}" in g, f"{app}: missing #framework-{pkg}"

    # a representative metric for this app made it in as a PropertyValue
    metric_id = "#metric-" + expect["metric"].replace("_", "-")
    assert metric_id in g, f"{app}: missing {metric_id}"


@pytest.mark.parametrize("app, expect", REAL_APPS)
def test_real_app_captured_strategy_matches(app, expect):
    """The capture itself (not just the crate) records the right strategy."""
    captured, _ = _load(app)
    assert captured["strategy"]["class_name"] == expect["strategy"]


@pytest.mark.parametrize("app, expect", REAL_APPS)
def test_real_app_frameworks_have_versions(app, expect):
    """Each captured framework carries a declared spec and/or installed version."""
    captured, _ = _load(app)
    by_pkg = {f["package"]: f for f in captured.get("frameworks", [])}
    for pkg in expect["frameworks"]:
        assert pkg in by_pkg, f"{app}: {pkg} not captured"
        fw = by_pkg[pkg]
        assert fw.get("declared") or fw.get("installed_version"), \
            f"{app}: {pkg} has neither declared nor installed version"


def test_pytorch_accuracy_has_semantic_uri(tmp_path):
    """The pytorch app maps accuracy -> schema.org via [tool.flwrcrate.metric-uris];
    when that map is applied, the metric carries a propertyID."""
    captured, log = _load("quickstart-pytorch")
    uri_map = {"accuracy": "https://schema.org/Accuracy"}
    crate_dir = build_crate(captured, crate_dir=tmp_path / "ro-crate",
                            metrics_log_path=log, uri_map=uri_map)
    g = _graph(crate_dir)
    assert g["#metric-accuracy"]["propertyID"] == "https://schema.org/Accuracy"


def test_all_fixtures_present():
    """Guard against a half-added fixture set."""
    for app, _ in [(p.values[0], p.values[1]) for p in REAL_APPS]:
        assert (FIXTURES / app / "captured_metadata.json").exists(), f"missing fixture: {app}"
