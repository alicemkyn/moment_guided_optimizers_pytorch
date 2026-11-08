"""Cross entropy and generalized cross entropy.

q is never inferred from the archived logs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q: float):
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError(f"gce q must be in (0, 1], got {q}")
        self.q = float(q)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = F.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        probability = probability.clamp_min(torch.finfo(probability.dtype).eps)
        return ((1.0 - probability.pow(self.q)) / self.q).mean()


def build_loss(name: str, gce_q: float | None) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "gce":
        if gce_q is None:
            raise ValueError(
                "GCE requires --gce-q. The paper does not state q. "
                "0.7 is Zhang and Sabuncu 2018's reported example, not an archived value."
            )
        return GeneralizedCrossEntropy(gce_q)
    raise ValueError(f"Unknown loss: {name}")
