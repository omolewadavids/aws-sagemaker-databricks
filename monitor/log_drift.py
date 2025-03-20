import mlflow
import json
import boto3

s3_client = boto3.client("s3")
s3_bucket = "your-bucket"
drift_report_path = "monitoring-results/drift_report.json"

# Load Data Drift Report
response = s3_client.get_object(Bucket=s3_bucket, Key=drift_report_path)
drift_report = json.loads(response['Body'].read().decode('utf-8'))

data_drift_score = drift_report["data_drift"]["drift_score"]  # Example metric

# Compute SHAP Drift
from monitor.shap_drift import drift_score  # Import SHAP drift score

# Log both drift scores to MLflow
mlflow.set_experiment("/Shared/databricks_sagemaker_pipeline")
with mlflow.start_run():
    mlflow.log_metric("data_drift_score", data_drift_score)
    mlflow.log_metric("shap_drift_score", drift_score)

    # Trigger retraining if either data or SHAP drift is high
    if data_drift_score > 0.2 or drift_score > 0.2:
        mlflow.set_tag("retrain_required", "true")

print(f"Data Drift Score: {data_drift_score}, SHAP Drift Score: {drift_score}, Logged to MLflow.")
