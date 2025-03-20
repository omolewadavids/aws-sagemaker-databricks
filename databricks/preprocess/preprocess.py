import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark
spark = SparkSession.builder.appName("DatabricksPreprocessing").getOrCreate()

# MLflow Experiment Setup
mlflow.set_experiment("/Shared/databricks_sagemaker_pipeline")

with mlflow.start_run():
    # Load raw data from S3
    s3_raw_data = "s3://your-bucket/raw-data.csv"
    df = spark.read.csv(s3_raw_data, header=True, inferSchema=True)

    # Example transformation
    df_transformed = df.withColumn("normalized_feature", col("feature") / 100)

    # Save processed data
    s3_processed_data = "s3://your-bucket/processed-data.csv"
    df_transformed.write.mode("overwrite").csv(s3_processed_data, header=True)

    # Log Preprocessing Metadata
    mlflow.log_param("dataset", s3_raw_data)
    mlflow.log_artifact(s3_processed_data)

    print("Preprocessing completed and logged to MLflow.")
