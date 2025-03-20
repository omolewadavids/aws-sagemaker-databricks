pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'
        S3_BUCKET = 'your-bucket'
        SAGEMAKER_ROLE = 'arn:aws:iam::123456789012:role/SageMakerRole'
        DATABRICKS_TOKEN = credentials('DATABRICKS_TOKEN')
    }

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/your-repo/sagemaker-pipeline.git'
            }
        }

        stage('Run Databricks Preprocessing') {
            steps {
                sh 'python databricks/databricks_job.py'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run SageMaker Training with MLflow') {
            steps {
                sh 'python train/train.py'
            }
        }

        stage('Deploy Best Model from MLflow') {
            steps {
                sh 'python deploy/deploy.py'
            }
        }

        stage('Monitor Data and SHAP Drift') {
            steps {
                sh 'python monitor/log_drift.py'
            }
        }

        stage('Retrain if Necessary') {
            steps {
                sh 'python train/retrain.py'
            }
        }

        stage('Deploy Best Model') {
            steps {
                sh 'python deploy/deploy.py'
            }
        }
    }

    post {
        success {
            echo 'Deployment Successful! 🚀'
        }
        failure {
            echo 'Deployment Failed! ❌'
        }
    }
}
