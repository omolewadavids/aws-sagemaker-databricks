"""Author: Omolewa Adaramola omolewa.davids@gmail.com"""

import sagemaker
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.model import Model


sagemaker_session = sagemaker.Session()
role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"

s3_raw_data = "s3://sagemaker/input/train.csv"
s3_processed_data = "s3://sagemaker/output/processed-data.csv"
s3_model_output = "s3://sagemaker/output/model.pt"


# Define the input and output locations for your processing job
input_data = ProcessingInput(
    source="s3://sagemaker/input/train.csv",
    destination='/opt/ml/processing/input',
    input_name="train.csv",
    s3_data_distribution_type='FullyReplicated',
    s3_data_type='S3Prefix'
)

output_data = ProcessingOutput(
    source='/opt/ml/processing/output',
    destination="s3://sagemaker/output/",
    output_name="processed-data.csv"
)


# Define the processing job using a ScriptProcessor
processor = ScriptProcessor(
    image_uri="pytorch-inference:latest",
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role_arn
)

# # Run the processing job
# processor.run(
#     inputs=[input_data],
#     outputs=[output_data],
#     code='your_processing_script.py'
# )

preprocess_step = ProcessingStep(
    name="preprocess",
    processor=processor,
    inputs=[input_data],
    outputs=[output_data],
    code="preprocess/preprocess.py"
)


