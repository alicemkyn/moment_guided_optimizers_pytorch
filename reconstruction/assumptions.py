"""Paper choices versus explicit new assumptions.

Nothing in this module is a claim that the archived CSVs used that setting.
Filename tags are records of names, not a verified code path.
"""

from __future__ import annotations

PAPER_CHOICES = {
    "epochs": 50,
    "batch_size": 128,
    "weight_decay": 5e-4,
    "adam_family_lr": 3e-4,
    "sgd_lr": 0.1,
    "gamma": 0.5,
    "r_min": 0.2,
    "val_size": 5000,
    "cosine_schedule": True,
    "headline_noise": 0.4,
    "headline_seeds": 8,
    "gate_modes": ("shrink", "mean_preserving"),
    "headline_gate": "mean_preserving",
    "decoupled_weight_decay_for_mgosc": True,
    "selection": "test accuracy at the best validation checkpoint",
    "symmetric_noise_sentence": (
        "each training label is replaced with a uniformly random label "
        "from the 10 classes with probability eta"
    ),
}

# Present in CSV tags. The paper body does not state these values.
FILENAME_RECORDS = {
    "beta_s": 0.9,
    "bias_correction_s": False,
}

# Defaults below are reference assumptions for a new run. Each has an
# alternative flag. None of them is established as the archived protocol.
NEW_ASSUMPTIONS = {
    "stem": {
        "default": "cifar",
        "choices": ("cifar", "imagenet"),
        "why": (
            "The paper specifies a standard CIFAR-10 ResNet-18 recipe "
            "(random crop with padding 4, horizontal flip) and does not "
            "say whether conv1 is the CIFAR 3x3 stem or the ImageNet 7x7 stem. "
            "The July trainer uses the CIFAR stem. The later trainer uses "
            "stock resnet18. Default here is the CIFAR stem because that is "
            "the usual CIFAR-10 recipe. It is still an assumption."
        ),
    },
    "normalization": {
        "default": "cifar10_std_2470",
        "choices": ("cifar10_std_2470", "cifar10_std_2023", "none"),
        "why": (
            "The paper says per-channel normalization and does not give "
            "constants. Both constants appear in the surviving trainers. "
            "Default is mean (0.4914, 0.4822, 0.4465) and std "
            "(0.2470, 0.2435, 0.2616)."
        ),
    },
    "symmetric_mode": {
        "default": "uniform_including_true",
        "choices": ("uniform_including_true", "always_wrong"),
        "why": (
            "uniform_including_true follows the paper sentence and can keep "
            "the original label. always_wrong follows the July trainer and "
            "always draws a different class. The effective corruption rate "
            "differs by about a factor 0.9."
        ),
    },
    "sgd_momentum": {
        "default": 0.9,
        "why": "The paper says SGD with momentum and does not give the coefficient.",
    },
    "sgd_nesterov": {
        "default": False,
        "why": "The July trainer sets Nesterov. The paper does not.",
    },
    "gce_q": {
        "default": None,
        "preset_not_historical": {"name": "zhang2018_q07", "q": 0.7},
        "why": (
            "The paper cites generalized cross entropy and does not state q. "
            "A run with --loss gce must pass --gce-q. 0.7 is a reported "
            "setting in Zhang and Sabuncu 2018, not a value recovered from "
            "the archived logs."
        ),
    },
    "asymmetric_preset": {
        "default": None,
        "choices": ("patrini2017", "next_class"),
        "why": (
            "The paper says class-dependent noise and does not give the "
            "transition. Asymmetric runs require --asymmetric-preset. "
            "patrini2017 is the CIFAR-10 map bird->airplane, cat<->dog, "
            "deer->horse, truck->automobile. next_class sends y to (y+1) mod 10. "
            "Neither is established as the archived map."
        ),
    },
    "split_rng": {
        "default": "numpy",
        "choices": ("numpy", "python"),
        "why": "The two trainers shuffle the 45k/5k split with different RNGs.",
    },
    "noise_seed_offset": {
        "default": 0,
        "why": "The July trainer used seed+123. The later trainer used the run seed. Default 0 is new.",
    },
    "cosine_eta_min": {
        "default": 0.0,
        "why": (
            "The paper says a cosine schedule. eta_min=0 matches the archived "
            "epoch-1 learning rate under CosineAnnealingLR stepped once per epoch."
        ),
    },
    "foreach": {
        "default": False,
        "why": "Single-tensor updates keep MGO-SC and AdamW on the same floating-point path.",
    },
    "init": {
        "default": "kaiming_normal",
        "why": "The paper does not state initialization.",
    },
}

NORMALIZATION = {
    "cifar10_std_2470": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar10_std_2023": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "none": {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
}

# CIFAR-10 label indices.
ASYMMETRIC_PRESETS = {
    "patrini2017": {2: 0, 3: 5, 4: 7, 5: 3, 9: 1},
    "next_class": {i: (i + 1) % 10 for i in range(10)},
}


def assumption_rows() -> list[dict[str, str]]:
    rows = []
    for name, spec in NEW_ASSUMPTIONS.items():
        rows.append(
            {
                "choice": name,
                "default": repr(spec["default"]),
                "status": "new assumption, not a historical fact",
                "why": spec["why"],
            }
        )
    rows.append(
        {
            "choice": "beta_s / bias_correction_s",
            "default": "0.9 / False",
            "status": "filename record (betaS0.9, bcs0), not a verified code path",
            "why": "The paper says bias correction of the sign EMA is optional and does not give beta_s.",
        }
    )
    return rows
