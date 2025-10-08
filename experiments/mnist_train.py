import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from optimizers import AdamCustom, AdaGradCustom, RMSPropCustom, AdaMaxCustom
import matplotlib.pyplot as plt
import numpy as np

# Simple MLP for MNIST
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.layers(x)

def train_model(optimizer_class, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), lr=1e-3)
    losses = []

    for epoch in range(3):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[{name}] Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        losses.append(avg_loss)

    plt.plot(losses, label=name)
    return losses

if __name__ == "__main__":
    optimizers = [AdamCustom, RMSPropCustom, AdaGradCustom, AdaMaxCustom]
    for opt in optimizers:
        train_model(opt, opt.__name__)
    plt.legend(); plt.title("MNIST Training Loss Comparison")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.savefig("results/1a.png")
