"""Shared pytest fixtures and lightweight fakes for the flwrCrate test suite.

The whole point of these fakes is that flwrCrate's capture logic depends only on
duck-typed attributes of the Flower ``context``, ``strategy`` and ``Result``
objects -- not on importing Flower itself. So the entire suite runs without
Flower, Ray, or any ML framework installed; only ``rocrate`` (a real dependency)
is needed for the crate-building tests.
"""

import textwrap

import pytest


# --- Fakes standing in for Flower's runtime objects -----------------------

class FakeContext:
    """Mimics flwr Context: only ``run_config`` is read by the tracker."""

    def __init__(self, run_config=None):
        self.run_config = run_config or {}


class FedAvg:
    """A stand-in strategy. The tracker reads its class name, module, and the
    public, non-callable attribute surface -- exactly what a real strategy
    exposes. ``_private`` and callables must be ignored by ``_strategy_attrs``."""

    def __init__(self):
        self.fraction_train = 1.0
        self.fraction_evaluate = 0.5
        self.min_available_nodes = 2
        self.arrayrecord_key = "arrays"
        self._private = "should-be-ignored"

    def start(self):  # callable -> must be ignored by _strategy_attrs
        return None


class FakeResult:
    """Mimics the Result returned by strategy.start(): dicts of round -> metrics.
    ``metricrecord_to_dict`` accepts plain dicts, so dicts stand in for
    MetricRecords."""

    def __init__(self, train=None, evaluate=None, server=None):
        self.train_metrics_clientapp = train or {}
        self.evaluate_metrics_clientapp = evaluate or {}
        self.evaluate_metrics_serverapp = server or {}


# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def fake_context():
    return FakeContext({
        "num-server-rounds": 3,
        "learning-rate": 0.1,
        "fraction-evaluate": 0.5,      # federation-shaped key
        "min-available-clients": 2,    # federation-shaped key
    })


@pytest.fixture
def fake_strategy():
    return FedAvg()


@pytest.fixture
def fake_result():
    return FakeResult(
        train={1: {"train_loss": 0.7}, 2: {"train_loss": 0.5}},
        evaluate={1: {"eval_acc": 0.6}, 2: {"eval_acc": 0.8}},
    )


@pytest.fixture
def app_pyproject(tmp_path):
    """A realistic app pyproject.toml with dependencies and a metric-URI map.
    ``pytest`` is listed as a dependency purely so an installed-version lookup
    has something guaranteed to resolve in the test environment."""
    content = textwrap.dedent("""
        [project]
        name = "demo-app"
        dependencies = [
            "torch==2.8.0",
            "scikit-learn>=1.3",
            "flwr[simulation]>=1.29",
            "pytest>=7.0",
        ]

        [tool.flwrcrate.metric-uris]
        accuracy = "https://schema.org/Accuracy"
    """).strip()
    path = tmp_path / "pyproject.toml"
    path.write_text(content)
    return path
