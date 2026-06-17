"""Integration tests: the full FLCrateTracker lifecycle end to end.

These exercise __init__ -> wrap_evaluate -> record_result -> __exit__/build and
assert that a valid, complete ro-crate-metadata.json lands on disk -- the same
thing we otherwise verify by hand after a real ``flwr run``, but automated and
without Flower/Ray.
"""

import json

import pytest

from flwrcrate import FLCrateTracker
from tests.conftest import FakeContext, FakeResult, FedAvg


def _graph(out_dir):
    data = json.loads((out_dir / "ro-crate" / "ro-crate-metadata.json").read_text())
    return {e["@id"]: e for e in data["@graph"]}


def test_full_lifecycle_produces_complete_crate(
    tmp_path, fake_context, fake_strategy, fake_result, app_pyproject
):
    out = tmp_path / "out"
    model = tmp_path / "final_model.pt"
    model.write_bytes(b"fake-weights")

    with FLCrateTracker(
        fake_context, fake_strategy,
        output_dir=str(out),
        pyproject_path=str(app_pyproject),
        app_name="Integration run",
        author={"name": "Ali", "orcid": "https://orcid.org/0000-0000-0000-0001"},
        license="https://spdx.org/licenses/MIT.html",
    ) as tracker:
        ev = tracker.wrap_evaluate(lambda rnd, arrays: {"accuracy": 0.5 + rnd / 10})
        ev(1, None)
        ev(2, None)
        tracker.record_result(fake_result, model_path=str(model))

    g = _graph(out)

    # all five profile fixes present
    assert g["./"]["mentions"] == [{"@id": "#fl-run"}]                 # #1
    assert g["#fl-strategy"]["name"] == "FedAvg"                        # #2
    assert any(i["@id"] == "metrics_log.json" for i in g["#fl-run"]["result"])  # #3
    assert "#framework-torch" in g                                      # #4
    assert g["./"]["license"]["@id"].endswith("MIT.html")              # #5
    assert g["#fl-run"]["agent"]["@id"].endswith("0000-0001")

    # result-side capture happened (the record_result path)
    assert g["#fl-run"]["endTime"] is not None
    assert "final_model.pt" in g
    # accuracy got its semantic id from the fixture's metric-uri map
    assert g["#metric-accuracy"]["propertyID"] == "https://schema.org/Accuracy"
    assert g["#fl-run"]["actionStatus"]["@id"].endswith("CompletedActionStatus")


def test_wrap_evaluate_none_passthrough(tmp_path, fake_context, fake_strategy, app_pyproject):
    with FLCrateTracker(fake_context, fake_strategy, output_dir=str(tmp_path / "o"),
                        pyproject_path=str(app_pyproject)) as tracker:
        # apps without a server-side evaluate fn pass None
        assert tracker.wrap_evaluate(None) is None


def test_wrap_evaluate_records_per_round(tmp_path, fake_context, fake_strategy, app_pyproject):
    with FLCrateTracker(fake_context, fake_strategy, output_dir=str(tmp_path / "o"),
                        pyproject_path=str(app_pyproject)) as tracker:
        wrapped = tracker.wrap_evaluate(lambda rnd, arrays: {"accuracy": 0.42})
        returned = wrapped(1, None)
        # the wrapper returns the original metric record untouched
        assert returned == {"accuracy": 0.42}
        # ...and stashes it for the crate
        assert tracker._per_round["1"]["server_side_evaluate"] == {"accuracy": 0.42}


def test_no_server_evaluate_metrics_from_result_only(
    tmp_path, fake_context, fake_strategy, fake_result, app_pyproject
):
    """An app with no evaluate_fn (no wrap_evaluate) still captures metrics from
    the Result's client-side aggregates -- the fed-engines / FedProx pattern."""
    out = tmp_path / "out"
    with FLCrateTracker(fake_context, fake_strategy, output_dir=str(out),
                        pyproject_path=str(app_pyproject)) as tracker:
        tracker.record_result(fake_result)  # no model, no wrap_evaluate

    g = _graph(out)
    # final-round (round 2) client metrics made it into the crate
    assert "#metric-eval-acc" in g
    assert g["#metric-eval-acc"]["value"] == 0.8


def test_error_path_writes_failed_crate(tmp_path, fake_context, fake_strategy, app_pyproject):
    """A crash inside the with-block still yields a documented crate, and the
    original exception is NOT suppressed."""
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="boom"):
        with FLCrateTracker(fake_context, fake_strategy, output_dir=str(out),
                            pyproject_path=str(app_pyproject)):
            raise RuntimeError("boom")

    g = _graph(out)
    assert g["#fl-run"]["actionStatus"]["@id"].endswith("FailedActionStatus")
    assert "boom" in g["#fl-run"]["error"]


def test_federation_summary_picks_up_config_hints(
    tmp_path, fake_context, fake_strategy, app_pyproject
):
    """Federation-shaped run_config keys land in metrics_log.json's federation block."""
    out = tmp_path / "out"
    with FLCrateTracker(fake_context, fake_strategy, output_dir=str(out),
                        pyproject_path=str(app_pyproject)):
        pass

    log = json.loads((out / "metrics_log.json").read_text())
    fed = log["federation"]
    assert "fraction-evaluate" in fed
    assert "min-available-clients" in fed
    assert "learning-rate" not in fed  # not a federation key
