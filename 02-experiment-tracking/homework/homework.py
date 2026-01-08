import subprocess
import sys

import mlflow


horizontal_bar = "⎯" * 40

# Q1. Install MLflow
try:
    mlFlow_version = subprocess.run(
        [
            "mlflow",
            "--version",
        ],
        capture_output=True,
        text=True,
    )
except FileNotFoundError:
    sys.exit("Error: please install mlflow")

print(f"A1. Installed: {mlFlow_version.stdout}")
print(horizontal_bar)

# Q2. Download and preprocess the data
print("""A2. After launching `preprocess_data.py` script, here is the output folder :

 output
├──  dv.pkl
├──  test.pkl
├──  train.pkl
└──  val.pkl
      """)
print(horizontal_bar)

# Q3. Train a model with autolog
print("A3. Modified `train.py`")
print("Run it with: `uv run train.py`")
print("Launch MLflow UI: `mlflow ui")
print("The value of the `min_samples_split` parameter is 2")
print("⎯" * 40)

# Q4. Launch the tracking server locally
print("""A4. Launch the tracking server locally:
      `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000`
""")
print(horizontal_bar)

# Q5. Tune model hyperparameters
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(
    experiment_names=["random-forest-hyperopt"],
    order_by=["metrics.rmse ASC"],  # ASC = croissant (meilleur RMSE en premier)
    max_results=1,
)

best_run = runs.iloc[0]
best_rmse = best_run["metrics.rmse"]
# print(f"Meilleur run ID: {best_run['run_id']}")
# print(f"Meilleur RMSE: {best_run['metrics.rmse']:.4f}")
# print(f"Run complet: {best_run.to_dict()}")
print(f"A5.After launching `hpo.py`, the best val RMSE obtained is {best_rmse:.3f}")
print(horizontal_bar)

# Q6. Promote the best model to the model registry
print("Test RMSE of the best model is 5.567")
