import os
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def model_fn(model_dir):
    """Load the trained PyTorch model from SageMaker model directory."""
    model = MyModel()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    """Parse input JSON request into a PyTorch tensor."""
    import json
    import numpy as np

    if request_content_type == "application/json":
        data = json.loads(request_body)
        return torch.tensor(data["inputs"], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Generate predictions from the model."""
    with torch.no_grad():
        return model(input_data).tolist()
