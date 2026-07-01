# flwrCrate

[![tests](https://github.com/eScienceLab/flwrCrate/actions/workflows/tests.yml/badge.svg)](https://github.com/eScienceLab/flwrCrate/actions/workflows/tests.yml)
[![real-app e2e](https://github.com/eScienceLab/flwrCrate/actions/workflows/realapps.yml/badge.svg)](https://github.com/eScienceLab/flwrCrate/actions/workflows/realapps.yml)

Capture a [Flower](https://flower.ai/) federated learning run and emit an
[RO-Crate](https://www.researchobject.org/ro-crate/) describing it — a
machine-readable, FAIR provenance record of *who* ran *what*, with *which
software, configuration, and data flow*, and *what came out*.

The run is modelled as a [Process Run
Crate](https://www.researchobject.org/workflow-run-crate/profiles/process_run_crate/)-style
`CreateAction`, following the [Federated Learning RO-Crate
profile](https://esciencelab.org.uk/federated-learning-ro-crate-profile/federated-learning-profile.html):

- **instrument** — the software that did the work: Flower, the ML framework(s)
  (with declared *and* installed versions), and the aggregation strategy
  (FedAvg, FedProx, …) with its hyperparameters
- **object** — the run configuration as `PropertyValue` inputs
- **result** — the final aggregated model and a per-round metrics / federation
  log file
- **agent / author / license** — provenance: a `Person` entity (ORCID used as
  its `@id` when provided), a root license, and the agent on the action

Final performance metrics are attached to the output model as schema.org
`PropertyValue`s, optionally with semantic identifiers via a user-declared
metric→URI mapping.

## Requirements

> **⚠️ flwrCrate supports Flower's message-based ServerApp API only**
> (Flower **≥ 1.29**: `flwr.serverapp`, `@app.main()`, and
> `strategy.start(...)` returning a `Result`).
>
> Apps written against the classic API (`server_fn`,
> `ServerAppComponents`, `flwr.server.strategy`) are **not supported**.

Check an app's compatibility in seconds:

```bash
grep -rn "flwr.serverapp\|strategy.start\|server_fn\|ServerAppComponents" <app>/<pkg>/server_app.py
```

- `flwr.serverapp` **and** `strategy.start` → ✅ compatible
- `server_fn` or `ServerAppComponents` → ❌ classic API, not supported (yet)

Other requirements: Python ≥ 3.11, `rocrate ≥ 0.13`.

## Install

```bash
pip install flwrcrate
```

Install it into the **same environment that runs your Flower app** (the one
`flwr run` uses).

To work from source instead:

```bash
git clone https://github.com/eScienceLab/flwrCrate.git
pip install -e flwrCrate
```

## Usage

Integration is three touchpoints in your `server_app.py`: a context manager,
one `wrap_evaluate(...)`, and one `record_result(...)`.

### Before — a stock `flwr new` PyTorch app

```python
@app.main()
def main(grid: Grid, context: Context) -> None:
    ...
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
```

### After — with flwrCrate

```python
from flwrcrate import FLCrateTracker

@app.main()
def main(grid: Grid, context: Context) -> None:
    ...
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    with FLCrateTracker(
        context, strategy,
        output_dir="/absolute/path/to/your-app/fl_crate_out",        # absolute!
        pyproject_path="/absolute/path/to/your-app/pyproject.toml",  # absolute!
        app_name="My federated run",
        author={"name": "Your Name", "orcid": "https://orcid.org/0000-0000-0000-0000"},
        license="https://spdx.org/licenses/MIT.html",
    ) as tracker:
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=tracker.wrap_evaluate(global_evaluate),  # or None
        )
        torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
        tracker.record_result(result, model_path="final_model.pt")
    # On clean exit the RO-Crate is written to <output_dir>/ro-crate/
```

That is the whole integration. On exit — **including on failure** — the crate
is built; a failed run is recorded with `FailedActionStatus` and the error
message, so partial runs are still documented.

### Apps without a server-side evaluate function

If your app passes no `evaluate_fn` to `strategy.start(...)` (common for
client-side-only evaluation), skip `wrap_evaluate` entirely. Per-round metrics
are then read from the `Result` object's client-side aggregates
(`train_metrics_clientapp` / `evaluate_metrics_clientapp`) by
`record_result(...)`:

```python
    with FLCrateTracker(context, strategy, ...) as tracker:
        result = strategy.start(grid=grid, initial_arrays=arrays,
                                train_config=..., num_rounds=...)
        tracker.record_result(result)   # don't forget this line!
```

> **Don't forget `record_result(result)`.** Without it the crate is still
> written, but it will contain **no performance metrics, no end time, and no
> model** — only the static capture from the start of the run.

### Why absolute paths?

`flwr run` installs your app to `~/.flwr/apps/<hash>/` and executes the
ServerApp from there (in simulation, inside a Ray worker that does not inherit
your shell environment). Relative paths therefore resolve against the
installed copy, not your project: a relative `output_dir` "loses" the crate in
`~/.flwr/apps/...`, and a relative `pyproject_path` fails to find your
config — silently dropping the framework/dependency capture. Always pass both
as absolute paths.

## Configuration

### `FLCrateTracker(...)` parameters

| Parameter | Required | Description |
|---|---|---|
| `context` | yes | The Flower `Context` (gives access to `run_config`) |
| `strategy` | yes | Your strategy instance (FedAvg, FedProx, …) — class, module, and hyperparameters are captured |
| `output_dir` | recommended | Where to write outputs. **Use an absolute path.** Default: `./fl_crate_out` |
| `pyproject_path` | recommended | Path to your app's `pyproject.toml`. **Use an absolute path.** Default: `"pyproject.toml"` |
| `app_name` | no | Human-readable name for the crate's root dataset |
| `author` | no | `"Name"` or `{"name": ..., "orcid": ..., "affiliation": ...}` — becomes a `Person` entity (ORCID as `@id`) |
| `license` | no | License for the crate root, e.g. an SPDX URL `"https://spdx.org/licenses/MIT.html"` |
| `agent` | no | Who executed the run, same format as `author`. Defaults to the author |

### Semantic metric identifiers

Metric names are taken verbatim from your `MetricRecord`s, so any analysis
(classification, regression, anomaly detection, …) works without hardcoded
names. To attach semantic identifiers, declare a mapping in **your app's**
`pyproject.toml`:

```toml
[tool.flwrcrate.metric-uris]
accuracy = "https://schema.org/Accuracy"
rmse     = "http://www.wikidata.org/entity/Q1374913"
```

Mapped metrics get a `propertyID`; unmapped metrics still emit a plain
`PropertyValue` and log a warning.

## Output

```
<output_dir>/
├── captured_metadata.json   # full capture: config, frameworks, strategy, timing, final metrics
├── metrics_log.json         # per-round metrics + federation details (referenced by the crate)
└── ro-crate/
    ├── ro-crate-metadata.json
    ├── final_model.pt        # if a model_path was given
    └── metrics_log.json
```

### What's inside `ro-crate-metadata.json`

The crate's `@graph` contains, linked together:

| Entity | Type | Content |
|---|---|---|
| `./` | `Dataset` | Root: name, author, license, `conformsTo` the FL profile, `mentions` the run |
| `#fl-run` | `CreateAction` | The run: `agent`, `startTime`/`endTime`, `actionStatus`, instrument/object/result |
| `#flower` | `SoftwareApplication` | Flower with its installed version |
| `#framework-*` | `SoftwareApplication` | Every declared dependency (minus an infrastructure deny-list): `softwareRequirements` = the declared version spec, `softwareVersion` = the actually-installed version |
| `#fl-strategy` | `SoftwareApplication` | The aggregation strategy with its hyperparameters as `PropertyValue`s |
| `#param-*` | `PropertyValue` | Run configuration inputs (the action's `object`) |
| `#metric-*` | `PropertyValue` | Final-round metrics, attached to the output model (with `propertyID` when mapped) |
| `final_model.pt`, `metrics_log.json` | `File` | The run's outputs (the action's `result`) |

Per-round metric history and federation details (participant counts,
federation options) live in `metrics_log.json`, which the crate references as
a run output — keeping the metadata file lean while preserving the full time
series.

### Validation

Crates validate against the official
[`rocrate-validator`](https://github.com/crs4/rocrate-validator). Note:
`ro-crate-py` ≥ 0.15 writes RO-Crate **1.2**, while `rocrate-validator` 0.10
ships only a **1.1** profile, producing a single false-positive `MUST 5.3`
(`conformsTo` version). All other REQUIRED checks pass.

## Tested with

| App | Strategy | Stack | Notes |
|---|---|---|---|
| `@flwrlabs/quickstart-pytorch` | FedAvg | PyTorch, TorchVision | server- and client-side metrics |
| `@flwrlabs/quickstart-sklearn` | FedAvg | scikit-learn | no server-side evaluate fn |
| `@chongshenng/fed-engines` | FedProx | PyTorch, HF datasets | anomaly-detection metrics (balanced accuracy, per-class recall), client-side only |

👉 **See a real generated crate before you run anything:**
[`examples/quickstart-pytorch/`](examples/quickstart-pytorch/) contains the
integrated `server_app.py` and the
[`ro-crate-metadata.json`](examples/quickstart-pytorch/ro-crate-metadata.json)
it produced.

## Known limitations / roadmap

- **Classic API not supported.** Apps on `server_fn`/`ServerAppComponents`
  (still common in published Flower apps) need porting to the message API
  first. Supporting the classic API is an open design question.
- **Participant count under recent Flower.** `num-supernodes` moved from the
  app's `pyproject.toml` to `~/.flwr/config.toml` (the SuperLink connection
  config), so participant-count capture from pyproject returns only
  run-config-derived hints. Reading the active Flower config is a planned
  enhancement.
- **Dataset identifiers, ethics/governance lineage** are not reachable from
  Flower and must be supplied by the user (e.g. via config) — relevant for
  regulated domains.
- **Per-client metadata is out of scope by design.** Flower exposes only
  per-round aggregates, which aligns with FL's privacy premise.

## Changelog

- **0.4.0** — renamed to **flwrCrate** (dist/import `flwrcrate`); config key
  is now `[tool.flwrcrate.metric-uris]` (the legacy `[tool.fedacrate.*]` key
  is still read).
- **0.3.0** — run discoverable from the crate root (`mentions`); aggregation
  strategy emitted as a `SoftwareApplication` with hyperparameters; per-round
  metrics + federation details in `metrics_log.json`; all declared
  dependencies captured (deny-list instead of allow-list);
  author/license/agent provenance scaffolding.
- **0.2.0** — initial working version.

## Development

```bash
pip install -e ".[dev]"                      # installs pytest + pytest-cov
pytest                                        # run the full unit + integration suite
pytest --cov=flwrcrate --cov-report=term-missing   # with a coverage report
```

Testing has two tiers:

- **`tests` workflow (every push)** — fast **unit** tests + **integration**
  tests that drive the whole `FLCrateTracker` lifecycle, plus **real-data tests**
  that feed the *actual* captured output of the Tested-with apps
  (`tests/fixtures/<app>/`) through the crate builder. None of these need
  Flower, Ray, or an ML framework installed, so they run in seconds on Python
  3.11–3.12 and enforce ≥85% coverage (currently ~92%).
- **`real-app e2e` workflow (nightly + on demand)** — actually fetches a real
  Flower Hub app with `flwr new`, runs the federation end to end, and validates
  the produced crate (`.github/workflows/realapps.yml`). This is the only tier
  that exercises live capture from a running Flower simulation; it's heavier and
  network-dependent, hence separate from the per-push suite.

## License

[MIT](LICENSE)

## Acknowledgements

Developed within the ELIXIR **Fed-A-Crate** project (WP6) toward milestone
M6.7, building on the [Federated Learning RO-Crate
profile](https://esciencelab.org.uk/federated-learning-ro-crate-profile/federated-learning-profile.html)
by the [eScienceLab](https://esciencelab.org.uk/) (The University of
Manchester).
