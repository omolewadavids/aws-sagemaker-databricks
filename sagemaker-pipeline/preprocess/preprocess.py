import argparse

from data_loader import load_data
from cleaner import clean_data
from transformer import transform_data


def preprocess(input_path, output_path):
    # Load data
    df = load_data(input_path)

    # Clean data
    df = clean_data(df)

    # Transform data
    df = transform_data(df)

    # Save processed data to S3
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="S3 path to raw data")
    parser.add_argument(
        "--output", type=str, required=True, help="S3 path to save processed data"
    )
    args = parser.parse_args()

    preprocess(args.input, args.output)
