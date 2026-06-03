# fl-crate-generator (v0.3)

Capture a [Flower](https://flower.ai/) federated learning run and emit an
[RO-Crate](https://www.researchobject.org/ro-crate/) describing it. The run is
modelled as a Process Run Crate-style `CreateAction`: it links to the software
that produced it (Flower, the ML framework(s), and the aggregation strategy), to
its configuration inputs, and to its outputs (the final model and a per-round /
federation log file). Final metrics are attached as schema.org `PropertyValue`s.

This is the **v0.3** working copy. The original v0.2 library is left untouched in
the sibling `fl_crate_generator/` folder.

## Install

```bash
pip install -e .          # from this folder
# requires: flwr>=1.29, rocrate>=0.13 (tomli on Python < 3.11)
```

## Usage

Wrap the call to `strategy.start(...)` in your `server_app.py`:

```python
from fl_crate_generator import FLCrateTracker

with FLCrateTracker(
    context,
    strategy,
    output_dir="fl_crate_out",
    app_name="My federated run",
    author={"name": "Ali Faizollah", "orcid": "https://orcid.org/0000-0000-0000-0000"},
    license="https://spdx.org/licenses/MIT.html",
) as tracker:
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=tracker.wrap_evaluate(global_evaluate),  # pass None if you have no global evaluate fn
    )
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    tracker.record_result(result, model_path="final_model.pt")
# On exit the RO-Crate is written to output_dir/ro-crate/
```

That is the whole integration: a context manager, `wrap_evaluate(...)` around your
evaluate function, and one `record_result(...)` call.

### Generalisation across project types

Metric names are taken verbatim from your `MetricRecord`s, so the tool works for
any analysis (image classification, weather/soil regression, …) without
hardcoding `accuracy`/`loss`. To attach semantic identifiers, declare a mapping
in the **app's** `pyproject.toml`:

```toml
[tool.fedacrate.metric-uris]
rmse = "http://example.org/metric/RMSE"
accuracy = "https://schema.org/Accuracy"
```

Metrics without a mapping still emit a plain `PropertyValue` and log a warning.

## Output

```
output_dir/
├── captured_metadata.json   # full capture (config, frameworks, strategy, final metrics)
├── metrics_log.json         # per-round metrics + federation details (referenced by the crate)
└── ro-crate/
    ├── ro-crate-metadata.json
    ├── final_model.pt        # if a model_path was given
    └── metrics_log.json
```

## What changed in v0.3

- **#1** The root Dataset now `mentions` the `CreateAction`, so the run is
  discoverable from the root instead of being an orphan entity.
- **#2** The aggregation strategy is emitted as a `SoftwareApplication`
  (`#fl-strategy`) with its hyperparameters as `PropertyValue`s, linked from the
  action's `instrument`. (v0.2 captured these but dropped them from the crate.)
- **#3** Per-round metrics **and** federation details (participant/supernode
  counts and configuration) are written to `metrics_log.json`, which the crate
  references as a run output. Per-client metrics remain out of scope — Flower
  exposes only per-round aggregates.
- **#4** `framework.py` records **every** declared dependency (minus a small
  infrastructure deny-list), not just an allow-list, so frameworks like
  `lightgbm`, `statsmodels`, `keras`, etc. are no longer silently dropped.
  Recognised frameworks get a friendly name + homepage.
- **#5** `author` / `license` / `agent` scaffolding: a `Person` entity (ORCID
  used as its `@id` when provided), a root `license`, and `agent` on the action.

Verified end-to-end with a real `flwr run` (Flower 1.30, CIFAR-10, FedAvg) and
against the official `rocrate-validator` (see below).

## Notes / known follow-ups

- **Profile URL is a placeholder.** `FL_PROFILE` in `crate_builder.py` points at
  a guessed URL. Reconcile it with Eli's published Federated Learning RO-Crate
  profile permalink before release.
- **Flower 1.30 moved `num-supernodes`.** It now lives in `~/.flwr/config.toml`
  (the SuperLink connection config), not in the app's `pyproject.toml`, so the
  pyproject-based participant-count capture returns only run-config-derived
  hints under recent Flower. Reading the participant count from the active
  Flower config is a possible enhancement.
- **Validator version.** `ro-crate-py` 0.15 writes RO-Crate **1.2**;
  `rocrate-validator` 0.10 only ships a **1.1** profile, so it reports a single
  false-positive `MUST 5.3` (`conformsTo` version). The crate validates with
  **zero issues** when declared as 1.1, and passes all other REQUIRED checks.
  Use a 1.2-aware validator, or pin `ro-crate-py`, for a clean report.
- **Least user effort (Stian's point).** Integration still requires editing
  `server_app.py`. Fully automatic capture (e.g. a Flower plugin/hook) would
  need closer Flower integration and is left as a design question.
```
