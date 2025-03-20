import shap
import numpy as np
import mlflow
import torch
from model import Model
from preprocess.data_loader import load_data

# Load latest model
model = Model()
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

# Load incoming data
X_new, _, _, _ = load_data("s3://your-bucket/incoming-data.csv")
X_new = X_new.to_numpy()

# Load previous SHAP values
shap_train = np.load("shap_values_train.npy")

# Compute new SHAP values
explainer = shap.Explainer(model, X_new)
shap_new = explainer(X_new)

# Compute feature importance for new data
feature_importance_new = np.abs(shap_new.values).mean(axis=0)

# Compute drift score (absolute mean difference)
drift_score = np.mean(np.abs(feature_importance_new - np.mean(shap_train, axis=0)))

# Log drift metrics to MLflow
mlflow.set_experiment("/Shared/databricks_sagemaker_pipeline")
with mlflow.start_run():
    mlflow.log_metric("shap_drift_score", drift_score)

    # If drift score exceeds threshold, trigger retraining
    if drift_score > 0.2:
        mlflow.set_tag("retrain_required", "true")

print(f"SHAP Drift Score: {drift_score}, Logged to MLflow.")
