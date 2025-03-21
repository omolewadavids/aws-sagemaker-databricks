# **AWS SageMaker Pipeline with Databricks MLflow & Model Drift Detection**  

## **📌 Overview**  
This project implements a **CI/CD pipeline** for training and deploying **PyTorch models on AWS SageMaker** with **data preprocessing in Databricks**. It includes:  
✅ **Automated model drift detection** using SageMaker Model Monitor  
✅ **SHAP-based model explainability** for tracking feature importance drift  
✅ **Databricks MLflow tracking** for experiment logging  
✅ **Jenkins CI/CD pipeline** for retraining and deployment  

---

## **📂 Project Structure**  
```angular2html
aws-sagemaker-databricks/
│── databricks/
│   ├── databricks_job.py          # Runs preprocessing on Databricks
│── deploy/
│   ├── deploy.py                  # Deploys the best model to SageMaker
│── explainability/
│   ├── shap_analysis.py           # Computes SHAP values for model explainability
│── monitor/
│   ├── model_monitor.py           # Sets up SageMaker Model Monitor
│   ├── log_drift.py               # Logs data & SHAP drift metrics to MLflow
│   ├── shap_drift.py              # Compares feature importance over time
│── preprocess/
│   ├── data_loader.py             # Loads data from S3
│── train/
│   ├── train.py                   # Trains a PyTorch model on SageMaker
│   ├── retrain.py                 # Retrains if drift is detected
│── models/
│   ├── model.py                   # PyTorch model definition
│── .github/
│   ├── workflows/
│       ├── cicd.yml               # GitHub Actions CI/CD pipeline
│── Jenkinsfile                     # Jenkins pipeline configuration
│── README.md                       # Project documentation
```


---

## **🚀 Key Features**  

### **1️⃣ Data Preprocessing in Databricks**  
- Uses **Apache Spark** for scalable data processing  
- Saves processed data to **AWS S3** for SageMaker training  

### **2️⃣ Model Training & Deployment in SageMaker**  
- Trains a **PyTorch model** on SageMaker  
- Deploys the trained model as a **SageMaker endpoint**  

### **3️⃣ Model Drift Detection & Explainability**  
- **SageMaker Model Monitor** detects data drift  
- **SHAP analysis** tracks feature importance drift  
- Logs **drift scores to Databricks MLflow**  

### **4️⃣ CI/CD Pipeline with Jenkins & GitHub Actions**  
- Automates preprocessing, training, monitoring, and deployment  
- Retrains model **only if drift is detected**  
- Ensures best-performing model is deployed  

---

## **🔧 Setup & Configuration**  

### **1️⃣ Prerequisites**  
- **AWS Account** with SageMaker, S3, and IAM configured  
- **Databricks Workspace** with MLflow tracking enabled  
- **Jenkins CI/CD Server** or GitHub Actions configured  

### **2️⃣ Clone the Repository**  
```sh
git clone https://github.com/omolewadavids/aws-sagemaker-databricks.git
cd aws-sagemaker-databricks

export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="your-region"

# preprocess the data
python databricks/databricks_job.py 

# train the model
python train/train.py 

# deploy the model
python deploy/deploy.py

# monitor data drift
python monitor/log_drift.py



