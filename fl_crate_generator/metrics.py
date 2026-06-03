"""Metric handling: MetricRecord serialisation and metric->PropertyValue mapping.

This implements Eli's "generalisation" point: metric names come straight from
the user's training code, and a per-metric URI mapping (declared in pyproject.toml
under [tool.fedacrate.metric-uris]) is used as the PropertyValue propertyID when
available; otherwise a plain PropertyValue is emitted and a warning is raised.
"""

import logging

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

logger = logging.getLogger("fl_crate_generator")


def metricrecord_to_dict(mr) -> dict:
    """Convert a Flower MetricRecord (or plain mapping) to a JSON-friendly dict.

    Replaces the earlier repr() stringification: dict(mr) gives the real
    key/value pairs, and .tolist() unwraps any array-valued metrics.
    """
    if mr is None:
        return {}
    try:
        items = dict(mr)
    except (TypeError, ValueError):
        items = {k: mr[k] for k in mr}
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in items.items()}


def load_metric_uri_map(pyproject_path: str = "pyproject.toml") -> dict:
    """Read the metric->URI mapping from [tool.fedacrate.metric-uris]."""
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
    except FileNotFoundError:
        return {}
    return pyproject.get("tool", {}).get("fedacrate", {}).get("metric-uris", {})


def metric_to_property_value(name: str, value, uri_map: dict) -> dict:
    """One metric -> a schema.org PropertyValue dict.

    Uses propertyID when a URI mapping exists; otherwise emits a plain
    PropertyValue and warns.
    """
    pv = {"@type": "PropertyValue", "name": name, "value": value}
    uri = (uri_map or {}).get(name)
    if uri:
        pv["propertyID"] = uri
    else:
        logger.warning(
            "No URI mapping for metric %r; emitting a plain PropertyValue. "
            "Add it under [tool.fedacrate.metric-uris] in pyproject.toml for "
            "semantic metadata.",
            name,
        )
    return pv
