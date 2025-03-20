import mlflow
import boto3

# Get Best Model
best_model_uri = mlflow.search_runs(order_by=["metrics.final_loss ASC"]).iloc[0][
    "artifact_uri"
]

# Download Model
mlflow.pytorch.load_model(best_model_uri, dst_path="best_model")

# Deploy to SageMaker
sm_client = boto3.client("sagemaker")
response = sm_client.create_model(
    ModelName="MLflow-SageMaker-Model",
    PrimaryContainer={
        "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
        "ModelDataUrl": "s3://your-bucket/best_model",
    },
    ExecutionRoleArn="arn:aws:iam::123456789012:role/SageMakerRole",
)

print(f"Model deployed to SageMaker: {response['ModelArn']}")
