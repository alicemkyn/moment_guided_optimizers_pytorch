import argparse
import csv
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

from mgosc import MGOSC


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class NoisyLabels(torch.utils.data.Dataset):
    """Apply symmetric label noise on a fixed subset.

    Noise is precomputed deterministically from `seed`.
    """

    def __init__(self, base_subset, num_classes: int = 10, p_noise: float = 0.0, seed: int = 0):
        self.base = base_subset
        self.num_classes = num_classes
        self.p_noise = float(p_noise)
        rng = random.Random(seed)

        self.targets = []
        for i in range(len(self.base)):
            _, y = self.base[i]
            if self.p_noise > 0.0 and rng.random() < self.p_noise:
                y2 = rng.randrange(num_classes - 1)
                if y2 >= y:
                    y2 += 1
                self.targets.append(y2)
            else:
                self.targets.append(y)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return x, self.targets[idx]


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    m = resnet18(weights=None)
    # CIFAR-friendly stem
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return loss_sum / max(total, 1), correct / max(total, 1)


@dataclass
class RunRow:
    seed: int
    best_epoch: int
    best_val_acc: float
    test_acc_at_best_val: float
    opt: str
    lr: float
    wd: float
    batch_size: int
    label_noise: float
    beta_s: float
    gamma: float
    r_min: float
    bias_correction_s: int
    warmup_steps: int
    epochs: int


def make_tag(args: argparse.Namespace) -> str:
    return (
        f"cifar10_resnet18_{args.opt}"
        f"_lr{args.lr}_wd{args.wd}_bs{args.batch_size}"
        f"_noise{args.label_noise}_betaS{args.beta_s}_g{args.gamma}"
        f"_rmin{args.r_min}_bcs{int(args.bias_correction_s)}_wu{args.warmup_steps}"
        f"_E{args.epochs}_n{args.seeds}"
    )


def train_one_run(args: argparse.Namespace, seed: int, epoch_log_writer: Optional[csv.DictWriter] = None):
    set_seed(seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    tf_eval = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tf_train)
    train_full_eval = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=tf_eval)
    test_ds = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tf_eval)

    n = len(train_full)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(args.val_frac * n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_subset = Subset(train_full, tr_idx)
    va_subset = Subset(train_full_eval, val_idx)

    train_ds = NoisyLabels(tr_subset, num_classes=10, p_noise=args.label_noise, seed=seed + 123)
    val_ds = NoisyLabels(va_subset, num_classes=10, p_noise=0.0, seed=seed)

    tr_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    va_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_resnet18_cifar().to(device)
    crit = nn.CrossEntropyLoss()

    optimizer = None
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    elif args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, weight_decay=args.wd)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == "mgosc":
        optimizer = MGOSC(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            beta_s=args.beta_s,
            eps=1e-8,
            weight_decay=args.wd,
            gamma=args.gamma,
            r_min=args.r_min,
            bias_correction_s=args.bias_correction_s,
            warmup_steps=args.warmup_steps,
            collect_stats=args.log_epochs,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_test_at_best_val = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        loss_sum = 0.0
        correct = 0
        total = 0

        # epoch-level mechanism stats
        gate_mean_sum = 0.0
        active_ratio_sum = 0.0
        update_norm_sum = 0.0
        step_count = 0

        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            optimizer.step()

            # collect optimizer stats if available
            if args.log_epochs and hasattr(optimizer, "get_step_stats"):
                st = optimizer.get_step_stats()
                if st is not None:
                    gate_mean_sum += float(st.get("gate_mean", 0.0))
                    active_ratio_sum += float(st.get("active_ratio", 0.0))
                    update_norm_sum += float(st.get("update_norm", 0.0))
                    step_count += 1

            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        scheduler.step()

        tr_loss = loss_sum / max(total, 1)
        tr_acc = correct / max(total, 1)
        va_loss, va_acc = evaluate(model, va_loader, device)
        te_loss, te_acc = evaluate(model, te_loader, device)

        if va_acc > best_val:
            best_val = va_acc
            best_test_at_best_val = te_acc
            best_epoch = epoch

        if args.verbose:
            dt = time.time() - t0
            print(
                f"[seed {seed}] ep {epoch:03d} "
                f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} test_acc={te_acc:.4f} "
                f"loss(tr/va/te)={tr_loss:.3f}/{va_loss:.3f}/{te_loss:.3f} dt={dt:.1f}s"
            )

        if epoch_log_writer is not None:
            if step_count > 0:
                gate_mean = gate_mean_sum / step_count
                active_ratio = active_ratio_sum / step_count
                update_norm = update_norm_sum / step_count
            else:
                gate_mean = ""
                active_ratio = ""
                update_norm = ""

            epoch_log_writer.writerow(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "test_loss": te_loss,
                    "test_acc": te_acc,
                    "gate_mean": gate_mean,
                    "active_ratio": active_ratio,
                    "update_norm": update_norm,
                }
            )

    return RunRow(
        seed=seed,
        best_epoch=best_epoch,
        best_val_acc=float(best_val),
        test_acc_at_best_val=float(best_test_at_best_val),
        opt=args.opt,
        lr=args.lr,
        wd=args.wd,
        batch_size=args.batch_size,
        label_noise=args.label_noise,
        beta_s=args.beta_s,
        gamma=args.gamma,
        r_min=args.r_min,
        bias_correction_s=int(args.bias_correction_s),
        warmup_steps=int(args.warmup_steps),
        epochs=args.epochs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./results")
    ap.add_argument(
        "--opt",
        type=str,
        default="mgosc",
        choices=["mgosc", "adam", "adamw", "sgd", "rmsprop", "adagrad"],
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--label_noise", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # MGO-SC params
    ap.add_argument("--beta_s", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--r_min", type=float, default=0.2)
    ap.add_argument("--bias_correction_s", action="store_true")
    ap.add_argument("--warmup_steps", type=int, default=0)

    # Logging
    ap.add_argument("--log_epochs", action="store_true", help="Write epoch-level metrics CSV (for mechanism plots)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tag = make_tag(args)
    csv_path = os.path.join(args.out_dir, tag + ".csv")
    epoch_csv_path = os.path.join(args.out_dir, tag + "_epochs.csv")

    rows: List[RunRow] = []

    epoch_writer = None
    epoch_f = None
    if args.log_epochs:
        epoch_f = open(epoch_csv_path, "w", newline="")
        epoch_writer = csv.DictWriter(
            epoch_f,
            fieldnames=[
                "seed",
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "test_loss",
                "test_acc",
                "gate_mean",
                "active_ratio",
                "update_norm",
            ],
        )
        epoch_writer.writeheader()

    try:
        for i in range(args.seeds):
            seed = args.seed0 + i
            r = train_one_run(args, seed, epoch_writer)
            rows.append(r)
    finally:
        if epoch_f is not None:
            epoch_f.close()

    # Save per-run summary CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    # Print summary
    import numpy as np

    test_accs = np.array([r.test_acc_at_best_val for r in rows], dtype=float)
    print("\n=== SUMMARY ===")
    print("tag:", tag)
    print(f"test_acc_at_best_val: mean={test_accs.mean():.4f} std={test_accs.std(ddof=1):.4f} (n={len(test_accs)})")
    print("csv:", csv_path)
    if args.log_epochs:
        print("epoch_csv:", epoch_csv_path)


if __name__ == "__main__":
    main()
