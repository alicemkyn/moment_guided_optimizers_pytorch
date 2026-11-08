"""Plot helpers for IJCNN MGO-SC paper.

Usage (in Colab/Kaggle):
  python plot_from_csvs.py --results_dir ./results --out_dir ./figs \
    --adam_tag <...> --mgosc_tag <...>

It expects:
  - per-seed summary CSVs (produced by train script)
  - optional epoch logs: <tag>_epochs.csv

The script will create:
  - paired slopegraph
  - bootstrap CI plot (effect size)
  - raincloud/forest/ECDF
  - (if epoch logs exist) gate activity + update norm plots
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_summary_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seed" not in df.columns:
        raise ValueError(f"CSV must have a 'seed' column: {path}")
    return df


def _paired_merge(adam: pd.DataFrame, mg: pd.DataFrame) -> pd.DataFrame:
    m = adam.merge(mg, on="seed", suffixes=("_adam", "_mgosc")).sort_values("seed")
    m["delta"] = m["test_acc_at_best_val_mgosc"] - m["test_acc_at_best_val_adam"]
    return m


def _bootstrap_ci(values: np.ndarray, B: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(B, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def fig_paired_slopegraph(m: pd.DataFrame, out_path: str, title: str):
    plt.figure(figsize=(7.2, 4.2))
    for _, r in m.iterrows():
        plt.plot([0, 1], [r["test_acc_at_best_val_adam"], r["test_acc_at_best_val_mgosc"]], marker="o")
    plt.xticks([0, 1], ["Adam", "MGO-SC"])
    plt.ylabel("Test Acc @ best val")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def fig_estimation_delta(m: pd.DataFrame, out_path: str, title: str):
    deltas = m["delta"].to_numpy()
    mean_delta = deltas.mean()
    lo, hi = _bootstrap_ci(deltas)

    plt.figure(figsize=(7.2, 4.2))
    # dot per seed
    y = np.arange(len(deltas))
    plt.scatter(deltas, y, marker="o")
    plt.axvline(0.0, linestyle="--")
    plt.axvline(mean_delta, linestyle="-")
    plt.xlabel("Δ = MGO-SC − Adam")
    plt.yticks([])
    plt.title(f"{title}\nmean Δ={mean_delta:+.4f}, 95% CI [{lo:+.4f},{hi:+.4f}]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def fig_gate_update(epoch_csv: str, out_gate: str, out_update: str, title_prefix: str):
    df = pd.read_csv(epoch_csv)
    if "epoch" not in df.columns:
        return

    # Gate activity
    if "gate_active_ratio" in df.columns:
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(df["epoch"], df["gate_active_ratio"], marker="o")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Gate active ratio")
        plt.title(f"{title_prefix}: gate activity")
        plt.tight_layout()
        plt.savefig(out_gate, dpi=300)
        plt.close()

    # Update norm
    if "update_rms" in df.columns:
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(df["epoch"], df["update_rms"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Update RMS")
        plt.title(f"{title_prefix}: effective update magnitude")
        plt.tight_layout()
        plt.savefig(out_update, dpi=300)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./results")
    ap.add_argument("--out_dir", type=str, default="./figs")
    ap.add_argument("--adam_csv", type=str, required=True)
    ap.add_argument("--mgosc_csv", type=str, required=True)
    ap.add_argument("--title", type=str, default="Paired comparison")
    ap.add_argument("--prefix", type=str, default="noise04")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    adam = _read_summary_csv(args.adam_csv)
    mg = _read_summary_csv(args.mgosc_csv)
    m = _paired_merge(adam, mg)

    fig_paired_slopegraph(m, os.path.join(args.out_dir, f"fig_{args.prefix}_paired_slopegraph.png"), args.title)
    fig_estimation_delta(m, os.path.join(args.out_dir, f"fig_{args.prefix}_estimation_delta_bootstrap.png"), args.title)

    # Mechanism figs if epoch logs exist (optional)
    adam_epochs = os.path.splitext(args.adam_csv)[0] + "_epochs.csv"
    mg_epochs = os.path.splitext(args.mgosc_csv)[0] + "_epochs.csv"
    if os.path.exists(mg_epochs):
        fig_gate_update(
            mg_epochs,
            os.path.join(args.out_dir, "fig_gate_activity.png"),
            os.path.join(args.out_dir, "fig_update_norm.png"),
            title_prefix=args.title,
        )


if __name__ == "__main__":
    main()
