"""Label noise and the 45k/5k split.

Symmetric modes are both implemented. The paper's sentence matches
``uniform_including_true``. ``always_wrong`` is the July trainer. Asymmetric
maps exist only as named presets.
"""

from __future__ import annotations

import random

import numpy as np

from reconstruction.assumptions import ASYMMETRIC_PRESETS


def train_val_split(n: int, val_size: int, seed: int, rng_name: str) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_size < n:
        raise ValueError(f"val_size must be strictly inside (0, {n}), got {val_size}")
    indices = np.arange(n)
    if rng_name == "numpy":
        generator = np.random.default_rng(seed)
        generator.shuffle(indices)
    elif rng_name == "python":
        order = indices.tolist()
        random.Random(seed).shuffle(order)
        indices = np.asarray(order, dtype=np.int64)
    else:
        raise ValueError(f"Unknown split rng: {rng_name}")
    val_idx = np.sort(indices[:val_size])
    train_idx = np.sort(indices[val_size:])
    return train_idx, val_idx


def corrupt_labels(
    labels: np.ndarray,
    eta: float,
    seed: int,
    noise_type: str,
    symmetric_mode: str,
    asymmetric_preset: str | None,
) -> np.ndarray:
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    labels = np.asarray(labels, dtype=np.int64).copy()
    if eta == 0.0 or noise_type == "none":
        return labels
    generator = np.random.default_rng(seed)
    draw = generator.random(labels.shape[0])
    hit = draw < eta
    if noise_type == "symmetric":
        if symmetric_mode == "uniform_including_true":
            replacement = generator.integers(0, 10, size=labels.shape[0])
        elif symmetric_mode == "always_wrong":
            replacement = generator.integers(0, 9, size=labels.shape[0])
            replacement = replacement + (replacement >= labels)
        else:
            raise ValueError(f"Unknown symmetric mode: {symmetric_mode}")
        labels[hit] = replacement[hit]
        return labels
    if noise_type == "asymmetric":
        if asymmetric_preset not in ASYMMETRIC_PRESETS:
            raise ValueError(
                "Asymmetric noise requires --asymmetric-preset "
                f"{tuple(ASYMMETRIC_PRESETS)}. The paper does not specify the map."
            )
        mapping = ASYMMETRIC_PRESETS[asymmetric_preset]
        for index in np.flatnonzero(hit):
            source = int(labels[index])
            if source in mapping:
                labels[index] = mapping[source]
        return labels
    raise ValueError(f"Unknown noise type: {noise_type}")
