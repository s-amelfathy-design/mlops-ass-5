import os
import sys
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

THRESHOLD = 0.85

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("No accuracy metric found in MLflow.")
    sys.exit(1)

print("Run ID:", run_id)
print("Accuracy:", accuracy)
print("Threshold:", THRESHOLD)

if accuracy < THRESHOLD:
    print("Accuracy is below threshold. Pipeline failed.")
    sys.exit(1)
else:
    print("Accuracy passed threshold.")