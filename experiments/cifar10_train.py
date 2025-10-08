import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from optimizers import AdamCustom, RMSPropCustom, AdaGradCustom, AdaMaxCustom
import matplotlib.pyplot as plt


def get_sgd(params, lr=1e-3, momentum=0.9, nesterov=False):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov)


class SimpleCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.use_dropout: x = self.drop(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        if self.use_dropout: x = self.drop(x)
        return self.fc2(x)

def train_model(optimizer_class, name, use_dropout=False, use_sgd=False, nesterov=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = SimpleCNN(use_dropout=use_dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    if use_sgd:
        optimizer = get_sgd(model.parameters(), lr=1e-3, nesterov=nesterov)
    else:
        optimizer = optimizer_class(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(3):
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        print(f"[{name}] Epoch {epoch+1}: {avg:.4f}")
        losses.append(avg)

    plt.plot(losses, label=name)
    return losses

if __name__ == "__main__":
    experiments = [
        (AdaGradCustom, "AdaGrad", False, False, False),
        (AdaGradCustom, "AdaGrad+Dropout", True, False, False),
        (None, "SGDNesterov", False, True, True),
        (None, "SGDNesterov+Dropout", True, True, True),
        (AdamCustom, "Adam", False, False, False),
        (AdamCustom, "Adam+Dropout", True, False, False)
    ]

    for opt_class, name, drop, sgd, nest in experiments:
        train_model(opt_class, name, drop, sgd, nest)

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("CIFAR-10 ConvNet — Optimizer Comparison (with Dropout)")
    plt.savefig("results/2a.png")
    plt.close()
