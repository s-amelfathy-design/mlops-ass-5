import os
import sys
import mlflow

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if not os.path.exists("model_info.txt"):
    print("model_info.txt not found")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking MLflow run: {run_id}")

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("Accuracy metric not found in MLflow")
    sys.exit(1)

print(f"Accuracy = {accuracy}")

if accuracy < 0.85:
    print("Model accuracy is below threshold 0.85")
    sys.exit(1)

print("Model passed threshold")
sys.exit(0)