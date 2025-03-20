import pandas as pd


def load_data(s3_path):
    print(f"Loading data from {s3_path}")
    return pd.read_csv("/opt/ml/processing/input/train.csv")
