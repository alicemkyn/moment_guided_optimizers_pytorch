# train_cifar10_resnet18.py
import os
import csv
import time
import math
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

from mgosc import MGOSC


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total += x.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


class NoisyCIFAR10(Dataset):
    def __init__(self, base_ds, noise_rate: float, seed: int):
        self.base = base_ds
        self.noise_rate = float(noise_rate)
        self.seed = int(seed)
        self.num_classes = 10

        # Precompute noisy labels (reproducible per seed)
        rng = np.random.RandomState(self.seed)
        self.noisy_targets = []
        for i in range(len(self.base)):
            _, y = self.base[i]
            if rng.rand() < self.noise_rate:
                y = int(rng.randint(0, self.num_classes))
            self.noisy_targets.append(y)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        y = self.noisy_targets[idx]
        return x, y


def build_loaders(batch_size: int, noise_rate: float, seed: int, num_workers: int = 2):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_base = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # Split train into train/val
    n = len(train_base)
    idxs = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idxs)
    val_n = 5000
    val_idxs = idxs[:val_n]
    tr_idxs = idxs[val_n:]

    tr_subset = torch.utils.data.Subset(train_base, tr_idxs)
    va_subset = torch.utils.data.Subset(torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_test), val_idxs)

    # Apply label noise only to training subset
    noisy_train = NoisyCIFAR10(tr_subset, noise_rate=noise_rate, seed=seed)

    tr_loader = DataLoader(noisy_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    va_loader = DataLoader(va_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr_loader, va_loader, te_loader


def make_model(device):
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)
    return model


def write_summary_csv(out_dir, tag, rows):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.csv")
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def train_one_seed(args, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_loader, va_loader, te_loader = build_loaders(
        batch_size=args.batch_size,
        noise_rate=args.label_noise,
        seed=seed,
        num_workers=args.num_workers,
    )

    model = make_model(device)

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == "mgosc":
        optimizer = MGOSC(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            beta_s=args.beta_s,
            weight_decay=args.wd,
            gamma=args.gamma,
            r_min=args.r_min,
            bias_correction_s=args.bias_correction_s,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    # Simple cosine schedule (optional but stable)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = nn.CrossEntropyLoss()

    # Epoch logging CSV (mechanism)
    epoch_csv_path = None
    epoch_writer = None
    epoch_fh = None
    if args.log_epochs:
        os.makedirs(args.out_dir, exist_ok=True)
        epoch_tag = (
            f"epochs_cifar10_resnet18_{args.opt}"
            f"_lr{args.lr}_wd{args.wd}_bs{args.batch_size}"
            f"_noise{args.label_noise}_betaS{args.beta_s}_g{args.gamma}"
            f"_rmin{args.r_min}_bcs{int(args.bias_correction_s)}"
            f"_E{args.epochs}_seed{seed}.csv"
        )
        epoch_csv_path = os.path.join(args.out_dir, epoch_tag)
        epoch_fh = open(epoch_csv_path, "w", newline="")
        epoch_writer = csv.DictWriter(epoch_fh, fieldnames=[
            "epoch", "seed", "lr",
            "train_loss", "train_acc",
            "val_acc", "test_acc",
            "gate_mean", "gate_active_ratio",
            "update_rms", "grad_rms",
        ])
        epoch_writer.writeheader()

    best_val = -1.0
    best_test_at_best_val = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        loss_sum = 0.0
        correct = 0
        total = 0

        # mechanism accumulators (epoch-level)
        gate_sum_ep = 0.0
        gate_cnt_ep = 0
        gate_active_ep = 0
        update_norm2_ep = 0.0
        grad_norm2_ep = 0.0
        steps_ep = 0

        for x, y in tr_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            # collect step stats if MGOSC
            if hasattr(optimizer, "get_last_step_stats"):
                st = optimizer.get_last_step_stats()
                if st is not None and st.get("gate_cnt", 0) > 0:
                    gate_sum_ep += float(st["gate_sum"])
                    gate_cnt_ep += int(st["gate_cnt"])
                    gate_active_ep += int(st["gate_active"])
                    update_norm2_ep += float(st["update_norm2_sum"])
                    grad_norm2_ep += float(st["grad_norm2_sum"])
                    steps_ep += 1

            loss_sum += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += x.size(0)

        scheduler.step()

        tr_loss = loss_sum / max(total, 1)
        tr_acc = correct / max(total, 1)

        _, va_acc = eval_loop(model, va_loader, device)
        _, te_acc = eval_loop(model, te_loader, device)

        # best-val tracking
        if va_acc > best_val:
            best_val = va_acc
            best_test_at_best_val = te_acc
            best_epoch = epoch

        # epoch-level write
        if epoch_writer is not None:
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr
            gate_mean = (gate_sum_ep / gate_cnt_ep) if gate_cnt_ep > 0 else float("nan")
            gate_active_ratio = (gate_active_ep / gate_cnt_ep) if gate_cnt_ep > 0 else float("nan")
            update_rms = math.sqrt(update_norm2_ep / max(steps_ep, 1)) if steps_ep > 0 else float("nan")
            grad_rms = math.sqrt(grad_norm2_ep / max(steps_ep, 1)) if steps_ep > 0 else float("nan")

            epoch_writer.writerow(dict(
                epoch=epoch, seed=seed, lr=lr_now,
                train_loss=tr_loss, train_acc=tr_acc,
                val_acc=va_acc, test_acc=te_acc,
                gate_mean=gate_mean, gate_active_ratio=gate_active_ratio,
                update_rms=update_rms, grad_rms=grad_rms,
            ))

        if args.verbose:
            dt = time.time() - t0
            print(f"[seed={seed}] epoch={epoch:03d} "
                  f"tr_acc={tr_acc:.4f} va_acc={va_acc:.4f} te_acc={te_acc:.4f} "
                  f"best_va={best_val:.4f} best_te@best_va={best_test_at_best_val:.4f} "
                  f"dt={dt:.1f}s")

    if epoch_fh is not None:
        epoch_fh.close()

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "test_acc_at_best_val": best_test_at_best_val,
        "epoch_csv": epoch_csv_path if epoch_csv_path is not None else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", type=str, required=True, choices=["adam", "mgosc"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--label_noise", type=float, default=0.0)
    ap.add_argument("--out_dir", type=str, default="./results")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_epochs", action="store_true")

    # MGOSC params
    ap.add_argument("--beta_s", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--r_min", type=float, default=0.2)
    ap.add_argument("--bias_correction_s", action="store_true")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device count: {torch.cuda.device_count()}")
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"device: {device}")

    # Build a tag like your existing outputs
    tag = (
        f"cifar10_resnet18_{args.opt}"
        f"_lr{args.lr}_wd{args.wd}_bs{args.batch_size}"
        f"_noise{args.label_noise}_betaS{args.beta_s}"
        f"_g{args.gamma}_rmin{args.r_min}_bcs{int(args.bias_correction_s)}"
        f"_E{args.epochs}_n{args.seeds}"
    )

    rows = []
    accs = []

    for i in range(args.seeds):
        seed = i + 1  # 1..n like your earlier plots
        r = train_one_seed(args, seed)
        rows.append({
            "tag": tag,
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "best_val": r["best_val"],
            "test_acc_at_best_val": r["test_acc_at_best_val"],
            "epoch_csv": r["epoch_csv"],
        })
        accs.append(r["test_acc_at_best_val"])

    accs = np.array(accs, dtype=np.float64)
    mean = float(accs.mean())
    std = float(accs.std(ddof=1)) if len(accs) > 1 else float("nan")

    csv_path = write_summary_csv(args.out_dir, tag, rows)

    print("\n\n=== SUMMARY ===")
    print(f"tag: {tag}")
    print(f"test_acc_at_best_val: mean={mean:.4f} std={std:.4f} (n={len(accs)})")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()

