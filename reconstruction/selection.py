"""Checkpoint selection uses validation accuracy only."""

from __future__ import annotations


def consider(best: dict | None, epoch: int, val_acc: float, test_acc: float) -> dict:
    """Keep the earliest epoch that attains the highest validation accuracy.

    ``test_acc`` is stored for the published reporting rule. It never decides
    the checkpoint.
    """
    if best is None or val_acc > best["val_acc"]:
        return {"epoch": epoch, "val_acc": val_acc, "test_acc": test_acc}
    return best
