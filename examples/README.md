# Examples

Real flwrCrate output, so you can see what a generated crate looks like before
running anything yourself.

## quickstart-pytorch (FedAvg, PyTorch)

The official [`@flwrlabs/quickstart-pytorch`](https://flower.ai/apps/flwrlabs/quickstart-pytorch/)
app (CIFAR-10 image classification, FedAvg over 2 supernodes, 3 rounds) with
the three flwrCrate touchpoints added.

| File | What it is |
|---|---|
| [`server_app.py`](quickstart-pytorch/server_app.py) | The integrated ServerApp — the stock example plus the flwrCrate context manager, `wrap_evaluate`, and `record_result` (the only changes; look for `# flwrCrate`) |
| [`ro-crate-metadata.json`](quickstart-pytorch/ro-crate-metadata.json) | **The generated crate** — open this to see the output: the run as a `CreateAction` linking Flower + PyTorch + the FedAvg strategy, the configuration, and the metrics |
| [`metrics_log.json`](quickstart-pytorch/metrics_log.json) | Per-round metrics and federation details, referenced by the crate |

Produced by:

```bash
flwr run . --stream
```

### What to look for in `ro-crate-metadata.json`

- `./` → `mentions` the `#fl-run` `CreateAction` (the run is discoverable from the root)
- `#flower`, `#framework-torch`, `#framework-torchvision` → the software (`instrument`), each with its declared *and* installed version
- `#fl-strategy` → the **FedAvg** strategy as a `SoftwareApplication` with its hyperparameters
- `#param-*` → the run configuration (the action's `object`)
- `#metric-accuracy` → carries `propertyID: https://schema.org/Accuracy` from the
  `[tool.flwrcrate.metric-uris]` mapping; other metrics are plain `PropertyValue`s
- `#fl-run` → `agent`, `startTime`/`endTime`, `actionStatus`, and the `result` files

### Notes

- **The model binary (`final_model.pt`) is omitted** from this example for size.
  A real run emits it, and the crate references it as a `result` File — you'll
  see that reference in the metadata here even though the binary isn't included.
- **The ORCID `0009-0000-0000-0000` is illustrative.** Pass your own ORCID via
  `author={"name": ..., "orcid": ...}` in your run.
- The paths in `server_app.py` are shown as `/absolute/path/to/...` placeholders;
  use real absolute paths for your machine (see the main README for why).
