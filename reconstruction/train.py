"""Training harness for a new MGO-SC reference run.

Outputs are labeled ``artifact_class = reconstruction``. They are not a claim
that the archived CSVs have been reproduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from reconstruction import ARTIFACT_CLASS, PAPER_ID
from reconstruction.assumptions import FILENAME_RECORDS, NEW_ASSUMPTIONS, NORMALIZATION, PAPER_CHOICES, assumption_rows
from reconstruction.campaign import describe
from reconstruction.losses import build_loss
from reconstruction.mgosc import MGOSC
from reconstruction.noise import corrupt_labels, train_val_split
from reconstruction.resnet import ResNet18
from reconstruction.selection import consider

OPTIMIZERS = ("mgosc", "adam", "adamw", "amsgrad", "sgd", "paired")


class TensorClassification(Dataset):
    def __init__(self, images: torch.Tensor, labels: np.ndarray):
        self.images = images
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index])


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def default_lr(optimizer_name: str) -> float:
    if optimizer_name == "sgd":
        return float(PAPER_CHOICES["sgd_lr"])
    return float(PAPER_CHOICES["adam_family_lr"])


def build_model(stem: str, num_classes: int = 10) -> nn.Module:
    return ResNet18(stem=stem, num_classes=num_classes)


def build_optimizer(name: str, model: nn.Module, args: argparse.Namespace):
    lr = args.lr if args.lr is not None else default_lr(name)
    if name == "mgosc":
        return MGOSC(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            beta_s=args.beta_s,
            eps=1e-8,
            weight_decay=args.wd,
            gamma=args.gamma,
            r_min=args.r_min,
            gate_mode=args.gate_mode,
            bias_correction_s=args.bias_correction_s,
            decoupled_weight_decay=args.decoupled_weight_decay,
            collect_stats=True,
        ), lr
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wd, foreach=False), lr
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd, foreach=False), lr
    if name == "amsgrad":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True, foreach=False
        ), lr
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.sgd_momentum,
            weight_decay=args.wd,
            nesterov=args.sgd_nesterov,
        ), lr
    raise ValueError(f"Unknown optimizer: {name}")


def synthetic_split(args: argparse.Namespace, seed: int):
    generator = np.random.default_rng(seed)
    n = args.synthetic_n
    val_size = min(args.val_size, max(1, n // 5))
    images = torch.from_numpy(generator.standard_normal((n, 3, 8, 8)).astype(np.float32))
    labels = generator.integers(0, 10, size=n)
    train_idx, val_idx = train_val_split(n, val_size, seed, args.split_rng)
    noisy = corrupt_labels(
        labels[train_idx],
        args.label_noise,
        seed + args.noise_seed_offset,
        args.noise_type,
        args.symmetric_mode,
        args.asymmetric_preset,
    )
    test_images = torch.from_numpy(generator.standard_normal((max(8, n // 5), 3, 8, 8)).astype(np.float32))
    test_labels = generator.integers(0, 10, size=test_images.shape[0])
    train_ds = TensorClassification(images[train_idx], noisy)
    val_ds = TensorClassification(images[val_idx], labels[val_idx])
    test_ds = TensorClassification(test_images, test_labels)
    return train_ds, val_ds, test_ds


def cifar_split(args: argparse.Namespace, seed: int):
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit(
            "CIFAR-10 requires torchvision, and the import failed. "
            "Install a CPU torchvision build matching the installed torch before a full run. "
            "Synthetic smoke tests do not need it."
        ) from exc
    stats = NORMALIZATION[args.normalization]
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    root = args.data_dir
    train_full = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    train_eval = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
    test_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
    base_labels = np.asarray(train_full.targets, dtype=np.int64)
    train_idx, val_idx = train_val_split(len(train_full), args.val_size, seed, args.split_rng)
    noisy = corrupt_labels(
        base_labels[train_idx],
        args.label_noise,
        seed + args.noise_seed_offset,
        args.noise_type,
        args.symmetric_mode,
        args.asymmetric_preset,
    )

    class IndexedNoisy(Dataset):
        def __init__(self):
            self.base = Subset(train_full, train_idx.tolist())
            self.labels = noisy

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            image, _label = self.base[index]
            return image, int(self.labels[index])

    val_ds = Subset(train_eval, val_idx.tolist())
    return IndexedNoisy(), val_ds, test_ds


def make_loaders(args: argparse.Namespace, seed: int):
    if args.dataset == "synthetic":
        train_ds, val_ds, test_ds = synthetic_split(args, seed)
    elif args.dataset == "cifar10":
        train_ds, val_ds, test_ds = cifar_split(args, seed)
    else:
        raise ValueError(args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, loss_fn) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += float(loss.item()) * images.shape[0]
        correct += int((logits.argmax(1) == labels).sum().item())
        count += int(images.shape[0])
    return total_loss / max(count, 1), correct / max(count, 1)


def run_one(args: argparse.Namespace, optimizer_name: str, seed: int, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    train_loader, val_loader, test_loader = make_loaders(args, seed)
    model = build_model(args.stem).to(device)
    loss_fn = build_loss(args.loss, args.gce_q)
    # Validation and test stay on cross entropy so the selection metric does not
    # change with the training loss. This is a reporting choice, stated here.
    eval_loss = nn.CrossEntropyLoss()
    optimizer, lr = build_optimizer(optimizer_name, model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.cosine_eta_min
    )
    epoch_path = out_dir / (
        f"epochs_{optimizer_name}_seed{seed}_gate{args.gate_mode}_g{args.gamma}.csv"
    )
    fields = [
        "epoch", "seed", "optimizer", "lr",
        "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc",
        "gate_mean", "gate_active_ratio", "update_rms", "grad_rms",
        "train_seconds", "eval_seconds", "epoch_time_s",
    ]
    best = None
    epoch_rows = []
    with epoch_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_started = time.perf_counter()
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device)
            count = 0
            for step_index, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                batch = images.shape[0]
                loss_sum = loss_sum + loss.detach() * batch
                correct = correct + (logits.argmax(1) == labels).sum()
                count += batch
                if args.max_steps and step_index >= args.max_steps:
                    break
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds = time.perf_counter() - train_started
            train_loss = float(loss_sum.item()) / max(count, 1)
            train_acc = float(correct.item()) / max(count, 1)
            stats = optimizer.consume_stats() if hasattr(optimizer, "consume_stats") else None
            scheduler.step()
            eval_started = time.perf_counter()
            val_loss, val_acc = evaluate(model, val_loader, device, eval_loss)
            test_loss, test_acc = evaluate(model, test_loader, device, eval_loss)
            if device.type == "cuda":
                torch.cuda.synchronize()
            eval_seconds = time.perf_counter() - eval_started
            best = consider(best, epoch, val_acc, test_acc)
            row = {
                "epoch": epoch,
                "seed": seed,
                "optimizer": optimizer_name,
                "lr": scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "gate_mean": "" if stats is None else stats["gate_mean"],
                "gate_active_ratio": "" if stats is None else stats["gate_active_ratio"],
                "update_rms": "" if stats is None else stats["update_rms"],
                "grad_rms": "" if stats is None else stats["grad_rms"],
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "epoch_time_s": train_seconds + eval_seconds,
            }
            writer.writerow(row)
            epoch_rows.append(row)
            if args.max_steps:
                break
    times = [float(row["epoch_time_s"]) for row in epoch_rows]
    return {
        "artifact_class": ARTIFACT_CLASS,
        "optimizer": optimizer_name,
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_val": best["val_acc"],
        "test_acc_at_best_val": best["test_acc"],
        "mean_epoch_time_s": sum(times) / len(times),
        "lr": lr,
        "gate_mode": args.gate_mode,
        "gamma": args.gamma,
        "epoch_csv": epoch_path.name,
        "smoke": bool(args.max_steps),
    }


def write_provenance(args: argparse.Namespace, out_dir: Path) -> None:
    payload = {
        "artifact_class": ARTIFACT_CLASS,
        "not_historical": True,
        "paper": PAPER_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "config": {key: value for key, value in vars(args).items()},
        "paper_choices": PAPER_CHOICES,
        "filename_records": FILENAME_RECORDS,
        "new_assumptions_in_effect": {
            "stem": args.stem,
            "normalization": args.normalization,
            "symmetric_mode": args.symmetric_mode,
            "split_rng": args.split_rng,
            "noise_seed_offset": args.noise_seed_offset,
            "sgd_nesterov": args.sgd_nesterov,
            "cosine_eta_min": args.cosine_eta_min,
            "gce_q": args.gce_q,
            "asymmetric_preset": args.asymmetric_preset,
            "decoupled_weight_decay": args.decoupled_weight_decay,
        },
        "selection_rule": "highest validation accuracy, earliest epoch on a tie; test accuracy is recorded and not used to select",
        "timing_definition": "epoch_time_s = train_seconds + eval_seconds; gate stats synchronize once per epoch",
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="New reference MGO-SC run. Not the historical runner.")
    ap.add_argument("--dataset", choices=["synthetic", "cifar10"], default="synthetic")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--stem", choices=["cifar", "imagenet"], default=NEW_ASSUMPTIONS["stem"]["default"])
    ap.add_argument("--normalization", choices=tuple(NORMALIZATION), default=NEW_ASSUMPTIONS["normalization"]["default"])
    ap.add_argument("--noise-type", choices=["none", "symmetric", "asymmetric"], default="symmetric")
    ap.add_argument("--symmetric-mode", choices=NEW_ASSUMPTIONS["symmetric_mode"]["choices"], default=NEW_ASSUMPTIONS["symmetric_mode"]["default"])
    ap.add_argument("--asymmetric-preset", choices=["patrini2017", "next_class"], default=None)
    ap.add_argument("--label-noise", type=float, default=0.0)
    ap.add_argument("--opt", choices=OPTIMIZERS, default="mgosc")
    ap.add_argument("--loss", choices=["ce", "gce"], default="ce")
    ap.add_argument("--gce-q", type=float, default=None)
    ap.add_argument("--gate-mode", choices=["mean_preserving", "shrink"], default="mean_preserving")
    ap.add_argument("--gamma", type=float, default=PAPER_CHOICES["gamma"])
    ap.add_argument("--r-min", type=float, default=PAPER_CHOICES["r_min"])
    ap.add_argument("--beta-s", type=float, default=FILENAME_RECORDS["beta_s"])
    ap.add_argument("--bias-correction-s", action="store_true")
    ap.add_argument("--decoupled-weight-decay", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sgd-momentum", type=float, default=NEW_ASSUMPTIONS["sgd_momentum"]["default"])
    ap.add_argument("--sgd-nesterov", action="store_true")
    ap.add_argument("--epochs", type=int, default=PAPER_CHOICES["epochs"])
    ap.add_argument("--batch-size", type=int, default=PAPER_CHOICES["batch_size"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=PAPER_CHOICES["weight_decay"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--val-size", type=int, default=PAPER_CHOICES["val_size"])
    ap.add_argument("--split-rng", choices=["numpy", "python"], default="numpy")
    ap.add_argument("--noise-seed-offset", type=int, default=0)
    ap.add_argument("--cosine-eta-min", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--synthetic-n", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--accept-assumptions", action="store_true")
    ap.add_argument("--out-dir", default="reconstruction_runs/smoke")
    ap.add_argument("--list-campaign", action="store_true")
    return ap


def refuse_cifar_without_acceptance(args: argparse.Namespace) -> None:
    if args.dataset == "cifar10" and not args.accept_assumptions:
        print(describe())
        print("\nRefusing CIFAR-10 until --accept-assumptions is set.")
        print("The defaults include new assumptions. They are not archived facts.")
        for row in assumption_rows():
            print(f"- {row['choice']}: default {row['default']} ({row['status']})")
        raise SystemExit(2)
    if args.noise_type == "asymmetric" and args.asymmetric_preset is None:
        raise SystemExit("Asymmetric noise requires --asymmetric-preset. The paper does not specify the map.")
    if args.loss == "gce" and args.gce_q is None:
        raise SystemExit("GCE requires --gce-q. The paper does not state q.")


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.list_campaign:
        print(describe())
        return
    out_dir = Path(args.out_dir)
    if "artifacts/results_csv" in out_dir.as_posix():
        raise SystemExit("Refusing to write reconstruction logs into artifacts/results_csv.")
    refuse_cifar_without_acceptance(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(args, out_dir)
    names = ["adamw", "mgosc"] if args.opt == "paired" else [args.opt]
    rows = []
    for offset in range(args.seeds):
        seed = args.seed0 + offset
        for name in names:
            rows.append(run_one(args, name, seed, out_dir))
    summary = out_dir / "summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"artifact_class={ARTIFACT_CLASS} paper={PAPER_ID}")
    print(f"wrote {summary}")
    print("These numbers are from this process. They are not the archived table.")


if __name__ == "__main__":
    main()
