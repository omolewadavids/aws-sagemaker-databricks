import boto3

sm_client = boto3.client("sagemaker")

# Create a monitoring schedule
response = sm_client.create_monitoring_schedule(
    MonitoringScheduleName="ModelDriftDetectionSchedule",
    MonitoringScheduleConfig={
        "MonitoringJobDefinition": {
            "BaselineConfig": {
                "ConstraintsResource": {
                    "S3Uri": "s3://your-bucket/training-data-baseline.json"
                }
            },
            "MonitoringOutputs": [
                {
                    "S3Output": {
                        "S3Uri": "s3://your-bucket/monitoring-results/",
                        "LocalPath": "/opt/ml/processing/output"
                    }
                }
            ],
            "MonitoringResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 10
                }
            },
            "RoleArn": "arn:aws:iam::123456789012:role/SageMakerRole",
            "MonitoringAppSpecification": {
                "ImageUri": "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-model-monitor-analyzer"
            }
        }
    },
    MonitoringType="DataQuality"
)

print(f"Model Drift Detection Scheduled: {response['MonitoringScheduleArn']}")
