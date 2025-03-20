import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# MLflow Experiment
mlflow.set_experiment("/Shared/databricks_sagemaker_pipeline")

with mlflow.start_run():
    # Define model
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)


    # Load Data
    transform = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(".", train=True, download=True, transform=transform),
        batch_size=32, shuffle=True
    )

    # Model Setup
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(3):  # Example: 3 epochs
        for images, labels in train_loader:
            images = images.view(-1, 784)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    # Save & Log Model
    model_path = "models/simple_nn.pth"
    torch.save(model.state_dict(), model_path)

    mlflow.log_param("epochs", 3)
    mlflow.log_metric("final_loss", loss.item())
    mlflow.pytorch.log_model(model, "model")

    print("Model training completed and logged to MLflow.")
