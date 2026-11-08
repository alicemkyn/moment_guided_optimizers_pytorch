"""Measure one real CPU step of the reference ResNet-18. Does not train CIFAR-10."""

from __future__ import annotations

import json
import time

import torch

from reconstruction.mgosc import MGOSC
from reconstruction.resnet import ResNet18


def measure(batch: int = 2, steps: int = 2, image: int = 32) -> dict:
    model = ResNet18(stem="cifar", num_classes=10)
    optimizer = MGOSC(model.parameters(), lr=3e-4, weight_decay=5e-4, gate_mode="mean_preserving")
    loss_fn = torch.nn.CrossEntropyLoss()
    images = torch.randn(batch, 3, image, image)
    labels = torch.randint(0, 10, (batch,))
    # One untimed step so initialization is not in the measurement.
    logits = model(images)
    loss_fn(logits, labels).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss_fn(logits, labels).backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    return {
        "device": "cpu",
        "stem": "cifar",
        "batch": batch,
        "image": image,
        "steps": steps,
        "seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "parameters": sum(p.numel() for p in model.parameters()),
        "note": "This is a measurement on the current machine. It is not a GPU epoch time and not a dollar cost.",
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
