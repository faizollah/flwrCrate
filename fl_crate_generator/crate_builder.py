"""Build a real RO-Crate from the captured run metadata.

Models the federated learning run as a Process Run Crate-style CreateAction:
the run links to its software via ``instrument`` (Flower + the ML framework(s) +
the aggregation strategy), to its configuration inputs via ``object``
(PropertyValues), and to its outputs via ``result`` (the final model file and
the per-round / federation log file). Final metrics are attached to the output
model (or the action, if there is no model) as ``additionalProperty``
PropertyValues, using the metric->URI mapping where available. """

import logging
from pathlib import Path

from rocrate.rocrate import ROCrate
from rocrate.model.contextentity import ContextEntity

from .metrics import metric_to_property_value

logger = logging.getLogger("fl_crate_generator")

FL_PROFILE = (
    "https://esciencelab.org.uk/federated-learning-ro-crate-profile/"
    "federated-learning-profile.html"
)
FLOWER_HOMEPAGE = "https://flower.ai/"
SCHEMA = "http://schema.org/"


def _slug(text) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(text)).strip("-").lower() or "x"


def _person(crate, spec, fallback_id):
    """Add a Person entity from a name string or a dict.

    ``spec`` may be a plain name ("Ali") or a dict with optional keys
    ``name``, ``id``/``orcid`` and ``affiliation``. Returns the added entity.
    """
    if isinstance(spec, dict):
        name = spec.get("name")
        pid = spec.get("id") or spec.get("orcid") or fallback_id
        props = {"@type": "Person"}
        if name:
            props["name"] = name
        if spec.get("affiliation"):
            props["affiliation"] = spec["affiliation"]
    else:
        name = str(spec)
        pid = fallback_id
        props = {"@type": "Person", "name": name}
    return crate.add(ContextEntity(crate, pid, properties=props))


def build_crate(captured: dict, crate_dir, metrics_log_path=None, model_path=None,
                uri_map=None, author=None, license=None, agent=None) -> Path:
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

    # --- #5 license / author / agent scaffolding -------------------------------
    if license:
        if str(license).startswith("http"):
            lic = crate.add(ContextEntity(crate, str(license), properties={
                "@type": "CreativeWork", "name": str(license),
            }))
            crate.root_dataset["license"] = {"@id": lic.id}
        else:
            crate.root_dataset["license"] = str(license)
    else:
        logger.warning(
            "No license set for the RO-Crate. Pass license=... (e.g. an SPDX URL "
            "such as 'https://spdx.org/licenses/MIT.html') to satisfy RO-Crate "
            "completeness checks."
        )

    author_ref = None
    if author:
        author_ref = _person(crate, author, "#author")
        crate.root_dataset["author"] = {"@id": author_ref.id}
    else:
        logger.warning(
            "No author set for the RO-Crate. Pass author='Your Name' (or a dict "
            "with name/orcid/affiliation) for provenance completeness."
        )

    agent_ref = None
    if agent:
        agent_ref = _person(crate, agent, "#agent")
    elif author_ref is not None:
        agent_ref = author_ref  # the author ran it, unless told otherwise

    # --- Software (instruments): Flower + framework(s), with versions ---
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

    # --- #2 Aggregation strategy as a SoftwareApplication with hyperparameters ---
    strat = captured.get("strategy", {}) or {}
    if strat.get("class_name"):
        strat_props = {
            "@type": "SoftwareApplication",
            "name": strat["class_name"],
            "description": f"Federated aggregation strategy ({strat.get('module')}).",
        }
        hp_refs = []
        for k, v in (strat.get("attributes") or {}).items():
            pid = f"#strategy-param-{_slug(k)}"
            crate.add(ContextEntity(crate, pid, properties={
                "@type": "PropertyValue", "name": k, "value": v,
            }))
            hp_refs.append({"@id": pid})
        if hp_refs:
            strat_props["additionalProperty"] = hp_refs
        strategy = crate.add(ContextEntity(crate, "#fl-strategy", properties=strat_props))
        instruments.append({"@id": strategy.id})

    # --- Outputs (results): model file + per-round / federation log file ---
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
            "name": "Per-round metrics and federation log",
            "description": (
                "Per-round training and evaluation metrics, plus federation "
                "details (participant/supernode counts and configuration) for "
                "the whole run."
            ),
            "encodingFormat": "application/json",
        })
        results.append({"@id": log_entity.id})

    # --- Final metrics as PropertyValues ---
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
    action_props = {"@type": "CreateAction", "name": "Federated learning training run", "instrument": instruments}
    if timing.get("start_time"):
        action_props["startTime"] = timing["start_time"]
    if timing.get("end_time"):
        action_props["endTime"] = timing["end_time"]
    if config_refs:
        action_props["object"] = config_refs
    if results:
        action_props["result"] = results
    if agent_ref is not None:
        action_props["agent"] = {"@id": agent_ref.id}
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

    # --- #1 Link the action from the root so it is discoverable ---
    crate.root_dataset["mentions"] = [{"@id": action.id}]

    # Final metrics attach to the output model, else to the action.
    if metric_refs:
        host = model_entity if model_entity is not None else action
        host["additionalProperty"] = metric_refs

    crate.write(crate_dir)
    return crate_dir
