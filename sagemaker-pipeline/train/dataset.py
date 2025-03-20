
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(s3_path):
    print(f"Loading training data from {s3_path}")
    # Simulate dataset loading
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)
