"""pytorchexample: A Flower / PyTorch app, integrated with flwrCrate.

This is the stock `@flwrlabs/quickstart-pytorch` ServerApp with three flwrCrate
touchpoints added (see the comments marked `# flwrCrate`):

  1. the FLCrateTracker context manager wrapping strategy.start(...)
  2. tracker.wrap_evaluate(...) around the server-side evaluate function
  3. tracker.record_result(...) handing over the final Result

Everything else is unchanged from the upstream example. The crate produced by
running this file is committed alongside it as ro-crate-metadata.json.
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwrcrate import FLCrateTracker  # flwrCrate (1/3): import

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`.
    # flwrCrate (2/3): wrap strategy.start(...) in the tracker context manager.
    # Use ABSOLUTE paths: flwr runs the ServerApp from ~/.flwr/apps/<hash>/, so
    # relative paths resolve there, not in your project.
    with FLCrateTracker(
        context, strategy,
        output_dir="/absolute/path/to/quickstart-pytorch/fl_crate_out",
        pyproject_path="/absolute/path/to/quickstart-pytorch/pyproject.toml",
        app_name="Quickstart PyTorch (demo)",
        author={"name": "Ali Faizollah", "orcid": "https://orcid.org/0009-0000-0000-0000"},
        license="https://spdx.org/licenses/MIT.html",
    ) as tracker:
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=tracker.wrap_evaluate(global_evaluate),  # flwrCrate (3/3a)
        )
        torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
        tracker.record_result(result, model_path="final_model.pt")  # flwrCrate (3/3b)
    # On clean exit the RO-Crate is written to <output_dir>/ro-crate/


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
