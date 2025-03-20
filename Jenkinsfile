
pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'
        S3_BUCKET = 'SageMaker'
        SAGEMAKER_ROLE = 'arn:aws:iam::123456789012:role/SageMakerRole'
    }

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/omolewadavids/aws-sagemaker-databricks.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run SageMaker Pipeline') {
            steps {
                sh 'python pipeline.py'
            }
        }

        stage('Deploy Model to Endpoint') {
            steps {
                sh 'python deploy/deploy.py'
            }
        }
    }
