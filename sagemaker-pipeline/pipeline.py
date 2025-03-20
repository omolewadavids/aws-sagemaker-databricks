"""Author: Omolewa Adaramola omolewa.davids@gmail.com"""

from dotenv import load_dotenv
import os

import sagemaker
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.model import Model


sagemaker_session = sagemaker.Session()

load_dotenv()
role_arn = os.getenv("SAGEMAKER_ROLE_ARN")

s3_raw_data = "s3://sagemaker/input/train.csv"
s3_processed_data = "s3://sagemaker/output/processed-data.csv"
s3_model_output = "s3://sagemaker/output/model.pt"


# Define the input and output locations for your processing job
input_data = ProcessingInput(
    source="s3://sagemaker/input/train.csv",
    destination="/opt/ml/processing/input",
    input_name="train.csv",
    s3_data_distribution_type="FullyReplicated",
    s3_data_type="S3Prefix",
)

output_data = ProcessingOutput(
    source="/opt/ml/processing/output",
    destination="s3://sagemaker/output/",
    output_name="processed-data.csv",
)

output_model = ProcessingOutput(
    source="/opt/ml/processing/output",
    destination="s3://sagemaker/output/",
    output_name="model.pt",
)


# Define the processing job using a ScriptProcessor
processor = ScriptProcessor(
    image_uri="pytorch-inference:latest",
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role_arn,
)

# Data Preprocessing
preprocess_step = ProcessingStep(
    name="preprocess",
    processor=processor,
    inputs=[input_data],
    outputs=[output_data],
    code="preprocess/preprocess.py",
)

# Training Step
estimator = PyTorch(
    name="training",
    entry_point="train/train.py",
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.12",
    py_version="py3.12",
    output_path="s3://sagemaker/output",
    role=role_arn,
)

train_step = TrainingStep(
    name="TrainModel", estimator=estimator, inputs={"input_data": output_data}
)

# Model Deployment Step
model = Model(
    name="model",
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.0-cpu-py38",
    model_data="s3://sagemaker/output/model.pt",
    entry_point="deploy/inference.py",
    role=role_arn,
)

deploy_step = CreateModelStep(
    name="DeployModel",
    step_args=model.deploy(initial_instance_count=1, instance_type="ml.m5.large"),
)

# Create the pipeline
pipeline = Pipeline(
    name="ModelPipeline", steps=[preprocess_step, train_step, deploy_step]
)

pipeline.upsert(
    role_arn=role_arn,
)
