import requests
import json

DATABRICKS_INSTANCE = "https://your-databricks-instance"
TOKEN = "your-databricks-token"

JOB_ID = "1234"  # Your Databricks job ID


def trigger_databricks_job():
    url = f"{DATABRICKS_INSTANCE}/api/2.0/jobs/run-now"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {"job_id": JOB_ID}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("Databricks job started successfully!")
    else:
        print("Failed to start Databricks job:", response.text)


if __name__ == "__main__":
    trigger_databricks_job()
