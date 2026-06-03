"""Build a real RO-Crate from the captured run metadata.

Models the federated learning run as a Process Run Crate-style CreateAction:
the run links to its software via instrument (Flower + the ML framework), to its
configuration inputs via object (PropertyValues), and to its outputs via result
(the final model file and the per-round metrics log file). Final metrics are
attached to the output model as additionalProperty PropertyValues, using the
metric->URI mapping where available.

The exact property names mandated by the Federated Learning RO-Crate profile
v0.1 may differ; conformsTo points at the profile and these choices follow the
Process Run Crate conventions the profile is built on. Reconcile with Eli's
profile before release.
"""

from pathlib import Path

from rocrate.rocrate import ROCrate
from rocrate.model.contextentity import ContextEntity

from .metrics import metric_to_property_value

FL_PROFILE = (
    "https://esciencelab.org.uk/federated-learning-ro-crate-profile/"
    "federated-learning-profile.html"
)
FLOWER_HOMEPAGE = "https://flower.ai/"
SCHEMA = "http://schema.org/"


def _slug(text) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(text)).strip("-").lower() or "x"


def build_crate(captured: dict, crate_dir, metrics_log_path=None,
                model_path=None, uri_map=None) -> Path:
    """Assemble and write an RO-Crate. Returns the crate directory path."""
    uri_map = uri_map or {}
    crate_dir = Path(crate_dir)
    crate = ROCrate()

    crate.name = captured.get("app_name") or "Federated learning run"
    crate.description = (
        "RO-Crate describing a federated learning run captured with "
        "fl-crate-generator."
    )

    # Conformance to the FL profile (RO-Crate spec conformance is set by ro-crate-py).
    profile = crate.add(ContextEntity(crate, FL_PROFILE, properties={
        "@type": "CreativeWork",
        "name": "Federated Learning RO-Crate profile v0.1",
    }))
    crate.root_dataset["conformsTo"] = {"@id": profile.id}

    # --- Software (instruments): Flower + ML framework, with versions (Stian) ---
    instruments = []
    flwr_version = (captured.get("flower") or {}).get("version")
    flower_props = {"@type": "SoftwareApplication", "name": "Flower", "url": FLOWER_HOMEPAGE}
    if flwr_version:
        flower_props["softwareVersion"] = flwr_version
    flower = crate.add(ContextEntity(crate, "#flower", properties=flower_props))
    instruments.append({"@id": flower.id})

    for fw in captured.get("frameworks", []) or []:
        props = {"@type": "SoftwareApplication", "name": fw["name"]}
        if fw.get("homepage"):
            props["url"] = fw["homepage"]
        if fw.get("installed_version"):
            props["softwareVersion"] = fw["installed_version"]
        if fw.get("declared"):
            props["softwareRequirements"] = fw["declared"]  # spec from pyproject.toml
        ent = crate.add(ContextEntity(crate, f"#framework-{_slug(fw['package'])}", properties=props))
        instruments.append({"@id": ent.id})

    # --- Outputs (results): model file + per-round metrics log file ---
    results = []
    model_entity = None
    if model_path and Path(model_path).exists():
        model_entity = crate.add_file(str(model_path), Path(model_path).name, properties={
            "@type": "File",
            "name": "Final aggregated model",
            "description": "Final global model produced by the federated learning run.",
        })
        results.append({"@id": model_entity.id})

    if metrics_log_path and Path(metrics_log_path).exists():
        log_entity = crate.add_file(str(metrics_log_path), Path(metrics_log_path).name, properties={
            "@type": "File",
            "name": "Per-round metrics log",
            "description": "Per-round training and evaluation metrics for the whole run.",
            "encodingFormat": "application/json",
        })
        results.append({"@id": log_entity.id})

    # --- Final metrics as PropertyValues (Eli task 2: URI mapping) ---
    final = captured.get("final_metrics", {}) or {}
    final_metrics = final.get("metrics", {}) if isinstance(final, dict) else {}
    metric_refs = []
    for name, value in final_metrics.items():
        pv = metric_to_property_value(name, value, uri_map)
        ent = crate.add(ContextEntity(crate, f"#metric-{_slug(name)}", properties=pv))
        metric_refs.append({"@id": ent.id})

    # --- Run configuration as PropertyValues (inputs / s:object) ---
    config = captured.get("environment_config", {}) or {}
    config_refs = []
    for name, value in config.items():
        ent = crate.add(ContextEntity(crate, f"#param-{_slug(name)}", properties={
            "@type": "PropertyValue", "name": name, "value": value,
        }))
        config_refs.append({"@id": ent.id})

    # --- The CreateAction: the FL run itself ---
    timing = captured.get("run_timing", {}) or {}
    strat = captured.get("strategy", {}) or {}
    action_props = {"@type": "CreateAction", "name": "Federated learning training run", "instrument": instruments}
    if timing.get("start_time"):
        action_props["startTime"] = timing["start_time"]
    if timing.get("end_time"):
        action_props["endTime"] = timing["end_time"]
    if config_refs:
        action_props["object"] = config_refs
    if results:
        action_props["result"] = results
    if strat:
        action_props["description"] = (
            f"Run using strategy {strat.get('class_name')} ({strat.get('module')}), "
            f"{config.get('num-server-rounds', '?')} rounds."
        )
    if captured.get("error"):
        action_props["actionStatus"] = {"@id": SCHEMA + "FailedActionStatus"}
        action_props["error"] = captured["error"]
    else:
        action_props["actionStatus"] = {"@id": SCHEMA + "CompletedActionStatus"}

    action = crate.add(ContextEntity(crate, "#fl-run", properties=action_props))

    # Final metrics attach to the output model, else to the action (Eli's choice).
    if metric_refs:
        host = model_entity if model_entity is not None else action
        host["additionalProperty"] = metric_refs

    crate.write(crate_dir)
    return crate_dir
