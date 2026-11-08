"""Paired summaries for reconstruction logs.

These functions describe a new run. They do not target the archived accuracies.
"""

from __future__ import annotations

import math
import random
import statistics


def wilcoxon_two_sided(deltas: list[float]) -> float | None:
    cleaned = [delta for delta in deltas if delta != 0.0]
    if not cleaned:
        return None
    order = sorted(range(len(cleaned)), key=lambda index: abs(cleaned[index]))
    ranks = [0.0] * len(cleaned)
    cursor = 0
    while cursor < len(order):
        stop = cursor
        while stop + 1 < len(order) and abs(cleaned[order[stop + 1]]) == abs(cleaned[order[cursor]]):
            stop += 1
        average = (cursor + 1 + stop + 1) / 2.0
        for position in range(cursor, stop + 1):
            ranks[order[position]] = average
        cursor = stop + 1
    observed = sum(rank for rank, delta in zip(ranks, cleaned) if delta > 0)
    total = sum(ranks)
    extreme = 0
    patterns = 1 << len(cleaned)
    for mask in range(patterns):
        positive = 0.0
        for index in range(len(cleaned)):
            if mask >> index & 1:
                positive += ranks[index]
        if abs(positive - total / 2.0) >= abs(observed - total / 2.0) - 1e-12:
            extreme += 1
    return extreme / patterns


def percentile_bootstrap(deltas: list[float], replicates: int = 20000, seed: int = 0) -> dict[str, float]:
    generator = random.Random(seed)
    width = len(deltas)
    means = []
    for _ in range(replicates):
        total = 0.0
        for _draw in range(width):
            total += deltas[generator.randrange(width)]
        means.append(total / width)
    means.sort()

    def quantile(probability: float) -> float:
        position = probability * (replicates - 1)
        lower = math.floor(position)
        upper = min(lower + 1, replicates - 1)
        weight = position - lower
        return means[lower] * (1.0 - weight) + means[upper] * weight

    return {
        "mean": statistics.mean(deltas),
        "low": quantile(0.025),
        "high": quantile(0.975),
        "replicates": float(replicates),
        "seed": float(seed),
    }
