import mlflow
import subprocess

# Check if retraining is needed
latest_run = mlflow.search_runs(order_by=["start_time DESC"]).iloc[0]
if latest_run["tags.retrain_required"] == "true":
    print("Drift detected! Retraining model...")

    # Trigger SageMaker training
    subprocess.run(["python", "train/train.py"])
else:
    print("No drift detected. Model remains unchanged.")
