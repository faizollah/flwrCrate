"""CI end-to-end variant of the quickstart-sklearn ServerApp.

Used by the real-app e2e workflow (.github/workflows/realapps.yml): the workflow
fetches the stock @flwrlabs/quickstart-sklearn app with `flwr new`, drops this
file in over its server_app.py, then actually runs the federation and validates
the produced crate.

The only differences from a normal integration are CI-friendly fixed absolute
paths under /tmp — required because Flower runs the ServerApp in a Ray worker
that doesn't inherit the shell's cwd or environment, so relative paths and env
vars don't reach it. The workflow copies the app's pyproject.toml to
/tmp/flcrate_pyproject.toml before running.
"""

import joblib
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwrcrate import FLCrateTracker

from sklearnexample.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    set_model_params,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds: int = context.run_config["num-server-rounds"]
    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)
    arrays = ArrayRecord(get_model_params(model))

    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    with FLCrateTracker(
        context, strategy,
        output_dir="/tmp/flcrate_out",
        pyproject_path="/tmp/flcrate_pyproject.toml",
        app_name="Quickstart scikit-learn (CI e2e)",
        author={"name": "flwrCrate CI", "orcid": "https://orcid.org/0000-0000-0000-0000"},
        license="https://spdx.org/licenses/MIT.html",
    ) as tracker:
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )
        ndarrays = result.arrays.to_numpy_ndarrays()
        set_model_params(model, ndarrays)
        joblib.dump(model, "logreg_model.pkl")
        tracker.record_result(result, model_path="logreg_model.pkl")
