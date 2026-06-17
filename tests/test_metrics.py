"""Unit tests for flwrcrate.metrics -- MetricRecord handling and URI mapping."""

import textwrap

from flwrcrate.metrics import (
    metricrecord_to_dict,
    load_metric_uri_map,
    metric_to_property_value,
)


# --- metricrecord_to_dict -------------------------------------------------

def test_metricrecord_from_plain_dict():
    assert metricrecord_to_dict({"accuracy": 0.9, "loss": 0.1}) == {
        "accuracy": 0.9,
        "loss": 0.1,
    }


def test_metricrecord_none_returns_empty():
    assert metricrecord_to_dict(None) == {}


def test_metricrecord_converts_arraylike_values():
    class ArrayLike:
        def tolist(self):
            return [1, 2, 3]

    out = metricrecord_to_dict({"per_class": ArrayLike()})
    assert out == {"per_class": [1, 2, 3]}


# --- load_metric_uri_map --------------------------------------------------

def test_load_uri_map_current_key(app_pyproject):
    mapping = load_metric_uri_map(str(app_pyproject))
    assert mapping == {"accuracy": "https://schema.org/Accuracy"}


def test_load_uri_map_legacy_key_fallback(tmp_path):
    # apps written against the old name must keep working
    legacy = tmp_path / "pyproject.toml"
    legacy.write_text(textwrap.dedent("""
        [tool.fedacrate.metric-uris]
        rmse = "http://example.org/RMSE"
    """).strip())
    assert load_metric_uri_map(str(legacy)) == {"rmse": "http://example.org/RMSE"}


def test_load_uri_map_prefers_current_over_legacy(tmp_path):
    both = tmp_path / "pyproject.toml"
    both.write_text(textwrap.dedent("""
        [tool.flwrcrate.metric-uris]
        accuracy = "https://schema.org/Accuracy"

        [tool.fedacrate.metric-uris]
        accuracy = "http://legacy.example/Accuracy"
    """).strip())
    assert load_metric_uri_map(str(both))["accuracy"] == "https://schema.org/Accuracy"


def test_load_uri_map_missing_file_returns_empty(tmp_path):
    assert load_metric_uri_map(str(tmp_path / "nope.toml")) == {}


def test_load_uri_map_no_mapping_section_returns_empty(tmp_path):
    p = tmp_path / "pyproject.toml"
    p.write_text("[project]\nname = 'x'\n")
    assert load_metric_uri_map(str(p)) == {}


# --- metric_to_property_value ---------------------------------------------

def test_property_value_with_uri_mapping():
    pv = metric_to_property_value("accuracy", 0.95, {"accuracy": "https://schema.org/Accuracy"})
    assert pv["@type"] == "PropertyValue"
    assert pv["name"] == "accuracy"
    assert pv["value"] == 0.95
    assert pv["propertyID"] == "https://schema.org/Accuracy"


def test_property_value_without_uri_warns(caplog):
    pv = metric_to_property_value("loss", 0.1, {})
    assert "propertyID" not in pv
    assert pv["value"] == 0.1
    # an unmapped metric should warn the user how to fix it
    assert any("loss" in r.message for r in caplog.records)
