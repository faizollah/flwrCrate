"""Unit / component tests for flwrcrate.crate_builder -- RO-Crate assembly."""

import json

from rocrate.rocrate import ROCrate

from flwrcrate.crate_builder import _slug, _person, build_crate


# --- _slug ----------------------------------------------------------------

def test_slug_basic():
    assert _slug("num-server-rounds") == "num-server-rounds"


def test_slug_replaces_nonalnum_and_lowercases():
    assert _slug("Train Loss!") == "train-loss"


def test_slug_empty_falls_back():
    assert _slug("") == "x"


# --- _person --------------------------------------------------------------

def test_person_from_plain_name():
    crate = ROCrate()
    ent = _person(crate, "Ali", "#author")
    assert ent["@type"] == "Person"
    assert ent["name"] == "Ali"
    assert ent.id == "#author"


def test_person_from_dict_uses_orcid_as_id():
    crate = ROCrate()
    ent = _person(
        crate,
        {"name": "Ali", "orcid": "https://orcid.org/0000-0000-0000-0001", "affiliation": "UF"},
        "#author",
    )
    assert ent.id == "https://orcid.org/0000-0000-0000-0001"
    assert ent["name"] == "Ali"
    assert ent["affiliation"] == "UF"


# --- build_crate (component test against a hand-built capture) -------------

def _minimal_capture():
    return {
        "app_name": "Test run",
        "run_timing": {"start_time": "2026-01-01T00:00:00+00:00",
                       "end_time": "2026-01-01T00:01:00+00:00"},
        "environment_config": {"num-server-rounds": 3, "learning-rate": 0.1},
        "flower": {"version": "1.30.0"},
        "frameworks": [
            {"package": "torch", "name": "PyTorch", "homepage": "https://pytorch.org/",
             "declared": "==2.8.0", "installed_version": "2.8.0", "known_framework": True},
        ],
        "strategy": {"class_name": "FedAvg", "module": "flwr.serverapp.strategy.fedavg",
                     "attributes": {"fraction_train": 1.0, "min_available_nodes": 2}},
        "final_metrics": {"final_round": 3, "metrics": {"accuracy": 0.9, "loss": 0.2}},
    }


def _graph_by_id(crate_dir):
    data = json.loads((crate_dir / "ro-crate-metadata.json").read_text())
    return {e["@id"]: e for e in data["@graph"]}


def test_build_crate_writes_metadata_file(tmp_path):
    out = build_crate(_minimal_capture(), crate_dir=tmp_path / "ro-crate")
    assert (out / "ro-crate-metadata.json").exists()


def test_build_crate_core_entities(tmp_path):
    crate_dir = build_crate(_minimal_capture(), crate_dir=tmp_path / "ro-crate",
                            author={"name": "Ali", "orcid": "https://orcid.org/0000-0000-0000-0001"},
                            license="https://spdx.org/licenses/MIT.html")
    g = _graph_by_id(crate_dir)

    # #1 run discoverable from root
    assert g["./"]["mentions"] == [{"@id": "#fl-run"}]
    # #2 strategy as SoftwareApplication with hyperparameters
    assert g["#fl-strategy"]["name"] == "FedAvg"
    assert "additionalProperty" in g["#fl-strategy"]
    # #4 framework with declared + installed versions
    assert g["#framework-torch"]["softwareRequirements"] == "==2.8.0"
    assert g["#framework-torch"]["softwareVersion"] == "2.8.0"
    assert g["#flower"]["softwareVersion"] == "1.30.0"
    # #5 provenance
    assert g["./"]["license"]["@id"] == "https://spdx.org/licenses/MIT.html"
    assert g["./"]["author"]["@id"] == "https://orcid.org/0000-0000-0000-0001"
    assert g["#fl-run"]["agent"]["@id"] == "https://orcid.org/0000-0000-0000-0001"


def test_build_crate_action_status_completed(tmp_path):
    crate_dir = build_crate(_minimal_capture(), crate_dir=tmp_path / "ro-crate")
    g = _graph_by_id(crate_dir)
    assert g["#fl-run"]["actionStatus"]["@id"].endswith("CompletedActionStatus")


def test_build_crate_failed_status_on_error(tmp_path):
    cap = _minimal_capture()
    cap["error"] = "RuntimeError: boom"
    crate_dir = build_crate(cap, crate_dir=tmp_path / "ro-crate")
    g = _graph_by_id(crate_dir)
    assert g["#fl-run"]["actionStatus"]["@id"].endswith("FailedActionStatus")
    assert g["#fl-run"]["error"] == "RuntimeError: boom"


def test_build_crate_metrics_use_uri_mapping(tmp_path):
    crate_dir = build_crate(_minimal_capture(), crate_dir=tmp_path / "ro-crate",
                            uri_map={"accuracy": "https://schema.org/Accuracy"})
    g = _graph_by_id(crate_dir)
    assert g["#metric-accuracy"]["propertyID"] == "https://schema.org/Accuracy"
    assert "propertyID" not in g["#metric-loss"]  # unmapped stays plain


def test_build_crate_config_as_object(tmp_path):
    crate_dir = build_crate(_minimal_capture(), crate_dir=tmp_path / "ro-crate")
    g = _graph_by_id(crate_dir)
    obj_ids = {ref["@id"] for ref in g["#fl-run"]["object"]}
    assert "#param-num-server-rounds" in obj_ids
