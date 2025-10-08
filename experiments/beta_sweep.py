import torch, matplotlib.pyplot as plt
from optimizers import AdamCustom
import numpy as np

def synthetic_test(beta1, beta2):
    # Simple 1D quadratic loss
    x = torch.tensor([5.0], requires_grad=True)
    optimizer = AdamCustom([x], lr=0.1, betas=(beta1,beta2))
    losses = []
    for t in range(50):
        loss = (x**2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses

if __name__ == "__main__":
    betas = [(0.8,0.9), (0.9,0.999), (0.95,0.9999)]
    for b1,b2 in betas:
        l = synthetic_test(b1,b2)
        plt.plot(l, label=f"β1={b1}, β2={b2}")
    plt.legend(); plt.xlabel("Steps"); plt.ylabel("Loss")
    plt.title("β₁/β₂ Sweep Stability Test")
    plt.savefig("results/4a.png")
