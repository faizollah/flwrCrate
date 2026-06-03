"""FLCrateTracker: capture a Flower run and emit an RO-Crate on exit.

Usage (inside server_app.py):

    from fl_crate_generator import FLCrateTracker

    with FLCrateTracker(context, strategy, output_dir="fl_crate_out") as tracker:
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=tracker.wrap_evaluate(global_evaluate),  # or None
        )
        torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
        tracker.record_result(result, model_path="final_model.pt")
    # On clean exit, the RO-Crate is written to output_dir/ro-crate/.
"""

import json
import logging
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .metrics import metricrecord_to_dict, load_metric_uri_map
from .framework import detect_frameworks
from .crate_builder import build_crate

logger = logging.getLogger("fl_crate_generator")


class FLCrateTracker:
    def __init__(self, context, strategy, output_dir=None,
                 pyproject_path="pyproject.toml", app_name=None):
        self.output_dir = Path(output_dir) if output_dir else (Path.cwd() / "fl_crate_out")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pyproject_path = pyproject_path
        self.capture_path = self.output_dir / "captured_metadata.json"
        self.metrics_log_path = self.output_dir / "metrics_log.json"
        self.crate_dir = self.output_dir / "ro-crate"
        self.model_path = None
        self._built = False

        self._per_round = {}  # round (str) -> {section: {...}, captured_at: ...}

        try:
            flwr_version = metadata.version("flwr")
        except metadata.PackageNotFoundError:
            flwr_version = None

        self._capture = {
            "app_name": app_name,
            "run_timing": {"start_time": datetime.now(timezone.utc).isoformat(), "end_time": None},
            "environment_config": dict(context.run_config),
            "flower": {"version": flwr_version},
            "frameworks": detect_frameworks(self.pyproject_path),
            "strategy": {
                "class_name": type(strategy).__name__,
                "module": type(strategy).__module__,
                "attributes": self._strategy_attrs(strategy),
            },
            "final_metrics": {},
            "metrics_log_file": self.metrics_log_path.name,
        }
        self._save_capture()

    # ---- capture helpers -------------------------------------------------

    @staticmethod
    def _strategy_attrs(strategy) -> dict:
        attrs = {}
        for k in dir(strategy):
            if k.startswith("_"):
                continue
            try:
                v = getattr(strategy, k)
            except Exception:
                continue
            if callable(v):
                continue
            attrs[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
        return attrs

    def _save_capture(self):
        tmp = self.capture_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._capture, indent=2, default=str))
        tmp.replace(self.capture_path)

    def _save_metrics_log(self):
        tmp = self.metrics_log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"per_round": self._per_round}, indent=2))
        tmp.replace(self.metrics_log_path)

    def _final_round_metrics(self) -> dict:
        if not self._per_round:
            return {}
        last = max(self._per_round, key=int)
        metrics = {}
        for section in ("server_side_evaluate", "train_clientapp", "evaluate_clientapp"):
            metrics.update(self._per_round[last].get(section, {}))
        return {"final_round": int(last), "metrics": metrics}

    # ---- run hooks -------------------------------------------------------

    def wrap_evaluate(self, user_evaluate_fn):
        """Intercept the server-side evaluate fn to capture per-round metrics.

        Tolerates None (e.g. scikit-learn apps without a global evaluate fn).
        """
        if user_evaluate_fn is None:
            return None

        def wrapped(server_round, arrays):
            mr = user_evaluate_fn(server_round, arrays)
            if mr is not None:
                slot = self._per_round.setdefault(str(server_round), {})
                slot["server_side_evaluate"] = metricrecord_to_dict(mr)
                slot["captured_at"] = datetime.now(timezone.utc).isoformat()
                self._save_metrics_log()
            return mr

        return wrapped

    def record_result(self, result, model_path=None):
        """Fold the Result object's per-round client metrics into the log and
        compute final metrics. Optionally record the saved model path."""
        if model_path is not None:
            self.model_path = Path(model_path)

        self._capture["run_timing"]["end_time"] = datetime.now(timezone.utc).isoformat()

        for attr, label in (("train_metrics_clientapp", "train_clientapp"),
                            ("evaluate_metrics_clientapp", "evaluate_clientapp")):
            for rnd, mr in getattr(result, attr, {}).items():
                self._per_round.setdefault(str(rnd), {})[label] = metricrecord_to_dict(mr)

        # Server-side from Result too, in case wrap_evaluate was not used.
        for rnd, mr in getattr(result, "evaluate_metrics_serverapp", {}).items():
            self._per_round.setdefault(str(rnd), {}).setdefault(
                "server_side_evaluate", metricrecord_to_dict(mr)
            )

        self._save_metrics_log()
        self._capture["final_metrics"] = self._final_round_metrics()
        self._save_capture()

    def build(self):
        """Write the RO-Crate. Called automatically on clean exit."""
        if self._built:
            return self.crate_dir
        uri_map = load_metric_uri_map(self.pyproject_path)
        build_crate(
            self._capture,
            crate_dir=self.crate_dir,
            metrics_log_path=self.metrics_log_path,
            model_path=self.model_path,
            uri_map=uri_map,
        )
        self._built = True
        logger.info("RO-Crate written to %s", self.crate_dir)
        return self.crate_dir

    # ---- context manager -------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._capture["error"] = f"{exc_type.__name__}: {exc_val}"
            self._save_capture()
        # Build the crate even on failure, so a partial run is still documented.
        try:
            self.build()
        except Exception as build_err:  # never mask the original error
            logger.error("Failed to write RO-Crate: %r", build_err)
        return False  # do not suppress exceptions