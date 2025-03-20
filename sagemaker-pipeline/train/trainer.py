
import torch
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, train_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss {loss.item()}")
