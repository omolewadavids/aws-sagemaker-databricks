import argparse
import torch
from dataset import load_dataset
from model import Model
from trainer import train_model

def train(data_path, model_path):
    # Load dataset
    train_loader = load_dataset(data_path)

    # Initialize model
    model = Model()

    # Train model
    train_model(model, train_loader)

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="S3 path to training data")
    parser.add_argument("--output", type=str, required=True, help="S3 path to save trained model")
    args = parser.parse_args()

    train(args.data, args.output)
