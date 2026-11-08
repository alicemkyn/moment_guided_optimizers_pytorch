# MGO-SC (IJCNN) — Experiments & Reproducibility

This folder contains the runnable reference code used to produce the CSV outputs and figures integrated into `main.tex`.

## 1) Minimal reproducible runs (the ones already in the paper)

### Adam (noise=0.4, E50, n=5)
```bash
python train_cifar10_resnet18.py --opt adam --epochs 50 --seeds 5 \
  --label_noise 0.4 --batch_size 128 --lr 3e-4 --wd 5e-4 \
  --out_dir ./results --verbose
```

### MGO-SC (noise=0.4, E50, n=5)
```bash
python train_cifar10_resnet18.py --opt mgosc --epochs 50 --seeds 5 \
  --label_noise 0.4 --batch_size 128 --lr 6e-4 --wd 5e-4 \
  --beta_s 0.9 --gamma 0.5 --r_min 0.2 --bias_correction_s \
  --out_dir ./results --verbose
```

The train script writes:
- a per-seed summary CSV: `./results/<tag>.csv`
- (optional) an epoch log CSV: `./results/<tag>_epochs.csv`

## 2) “B” and “C” additions to strengthen acceptance

### B) Noise-rate sweep (fast but reviewer-friendly)
Run Adam and MGO-SC at noise rates `{0.0, 0.2, 0.4}` with **n=3** seeds (minimum), then optionally n=5 for the final claim.

Recommended minimum commands:
```bash
for eta in 0.0 0.2 0.4; do
  python train_cifar10_resnet18.py --opt adam  --epochs 50 --seeds 3 --label_noise $eta \
    --batch_size 128 --lr 3e-4 --wd 5e-4 --out_dir ./results
  python train_cifar10_resnet18.py --opt mgosc --epochs 50 --seeds 3 --label_noise $eta \
    --batch_size 128 --lr 6e-4 --wd 5e-4 --beta_s 0.9 --gamma 0.5 --r_min 0.2 --bias_correction_s \
    --out_dir ./results
Done
```

### C) Mechanism diagnostics (gate activity + update norms)
Enable epoch logging and stats collection:
```bash
python train_cifar10_resnet18.py --opt mgosc --epochs 50 --seeds 3 --label_noise 0.4 \
  --batch_size 128 --lr 6e-4 --wd 5e-4 --beta_s 0.9 --gamma 0.5 --r_min 0.2 --bias_correction_s \
  --log_epochs --collect_opt_stats --out_dir ./results
```

## 3) Generate figures (PNG)
After you have the CSVs, generate figures:
```bash
python plot_from_csvs.py --results_dir ./results --out_dir ../figs \
  --adam_tag <ADAM_TAG> --mgosc_tag <MGOSC_TAG>
```

Replace `<ADAM_TAG>` and `<MGOSC_TAG>` with the exact tags printed in the `=== SUMMARY ===` block.

The paper is set up to compile cleanly while you iterate on figures and results.
