import shap
import mlflow
import torch
import numpy as np
import pandas as pd
from model import Model  # Import your PyTorch model
from preprocess.data_loader import load_data  # Load preprocessed data

# Load trained model
model = Model()
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

# Load dataset
X_train, _, _, _ = load_data("s3://your-bucket/training-data.csv")

# Convert to numpy
X_train = X_train.to_numpy()

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# Compute feature importance
feature_importance = np.abs(shap_values.values).mean(axis=0)

# Log to MLflow
mlflow.set_experiment("/Shared/databricks_sagemaker_pipeline")
with mlflow.start_run():
    for i, importance in enumerate(feature_importance):
        mlflow.log_metric(f"feature_{i}_importance", importance)

# Save SHAP values for comparison
np.save("shap_values_train.npy", shap_values.values)

print("SHAP values computed and logged in MLflow.")
