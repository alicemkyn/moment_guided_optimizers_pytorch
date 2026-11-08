"""Commands for a fresh verification campaign.

Printing these commands does not run them and does not reproduce the archived CSVs.
"""

from __future__ import annotations

HEADLINE = (
    "python -m reconstruction.train "
    "--dataset cifar10 --accept-assumptions "
    "--stem cifar --normalization cifar10_std_2470 "
    "--noise-type symmetric --symmetric-mode uniform_including_true "
    "--label-noise 0.4 --loss ce "
    "--opt paired --gate-mode mean_preserving "
    "--gamma 0.5 --r-min 0.2 --beta-s 0.9 "
    "--epochs 50 --batch-size 128 --wd 5e-4 "
    "--seeds 8 --seed0 1 "
    "--out-dir reconstruction_runs/headline_eta04_n8"
)

# Each entry is one paired or single comparison still required for the paper's
# tables. Defaults that are new assumptions stay visible on the command line.
FULL = [
    "clean CE, n=3, Adam / AdamW / AMSGrad / SGD / AdamW+GCE / shrink / mean-preserving",
    "eta=0.2 symmetric, n=5, the same optimizer list",
    "eta=0.4 symmetric, n=5, Adam / AMSGrad / SGD / AdamW+GCE, plus the n=8 AdamW and MGO-SC files",
    "eta=0.4 gate ablation n=8, gamma in {0, 0.3, 0.5, 0.7}, shrink and mean-preserving",
    "eta=0.4 asymmetric n=5, only after an explicit --asymmetric-preset",
    "eta=0.4 learning-rate check at 3e-4 and 6e-4 for Adam and MGO-SC",
]


def headline_command() -> str:
    return HEADLINE


def describe() -> str:
    lines = [
        "Headline campaign (not a reproduction of the archived numbers):",
        HEADLINE,
        "",
        "The command refuses to start on CIFAR-10 until --accept-assumptions is present.",
        "Checkpoint selection uses validation accuracy only.",
        "Wall-clock fields are train_seconds, eval_seconds, and epoch_time_s.",
        "Gate statistics stay on device until one sync at the end of the epoch.",
        "",
        "Remaining comparisons, each with the same explicit stem, normalization,",
        "symmetric mode, and seed list written into provenance.json:",
    ]
    lines.extend(f"- {item}" for item in FULL)
    lines.extend(
        [
            "",
            "GCE requires --gce-q. Asymmetric noise requires --asymmetric-preset.",
            "GPU epoch time for this code was not measured in the reference environment.",
        ]
    )
    return "\n".join(lines)
