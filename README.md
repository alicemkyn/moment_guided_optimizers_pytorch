# Moment-Guided Optimizers for Noisy Deep Learning  
© 2025 Alicem Koyun  

This repository has two layers. They are not the same experiment.

The October 2025 scripts below are Adam, AdaGrad, RMSProp, and AdaMax, plus short MNIST, CIFAR-10, and IMDB runs. `experiments/cifar10_train.py` trains `SimpleCNN` for 3 epochs. Those scripts do not contain MGO-SC, ResNet-18, or the 50-epoch label-noise protocol.

The CIFAR-10 / ResNet-18 study in arXiv:2610.37712v1 is documented under `artifacts/`. Read `artifacts/AUDIT.md` before using any of it. The public git history through `84c613463f45b7f9093a1189817507b7493eee05` did not contain that study. Two later source snapshots are preserved there as distinct versions. Neither is established as the program that wrote the result CSVs, and neither implements the paper's mean-preserving gate (Equation 7). The CSVs do match the published table rounding. A proposed wording fix for Section III-C is in `PROPOSED_III_C_QUALIFICATION.md`. It has not been submitted to arXiv.

---

## Overview
Moment-Guided Optimizers extend classical first-order optimization by maintaining exponential moving  
averages of gradient means and variances, allowing per-parameter adaptive learning rates that  
improve convergence under noisy or sparse gradient conditions.

---

## Setup
1. Clone or unzip the project:
   ```bash
   git clone https://github.com/alicemkyn/moment_guided_optimizers_pytorch.git
   cd moment_guided_optimizers_pytorch
   pip install -r requirements.txt
   ```

## Reproducibility status (arXiv:2610.37712v1)

What is in this repository now:

- `artifacts/results_csv/` holds 36 per-seed summaries and 19 per-epoch logs from the author-provided zip `f1af1a3d268527110d6bcca9120a7760931bfffde02e7736749bb1160dc87752`. Table I–III rounding, the n=8 Wilcoxon value 0.0078125, and the logged checkpoints are checked by `tests/test_csv_tables.py`.
- `artifacts/source-archive-2026-07-05/code/` is the older snapshot from that zip. Its trainer tag contains `_wu` and does not contain loss, noise type, or `gmrenorm`. Its optimizer uses coupled weight decay and a shrink-style floor, with no mean renormalization.
- `artifacts/latest-remembered/` is the later pair the author still had, with the statement that nothing after it is remembered. Its trainer is closer to the CSV column names but still omits `mean_epoch_time_s`, `epoch_time_s`, `_loss`, `_nt`, and `_gmrenorm`. Its optimizer uses decoupled weight decay and clamps the gate to `[r_min, 1]`. It has no `--gate_mode`.
- `tests/test_eq7_norm.py` checks the Equation 7 counterexample with exact fractions: mean 1, squared norm inflated by 13/9.

What is not established:

- Which of the two sources, if either, wrote the CSVs. Do not describe either as the original runner.
- A source for `--gate_mode`, asymmetric noise, GCE, AMSGrad, or the mean-preserving division. Those appear in the paper and, except GCE's loss parameter and the asymmetric transition, in the CSV names. No surviving file implements them. No further file was inferred.
- The ResNet stem and CIFAR normalization. The two trainers disagree (CIFAR stem and std `0.2470, 0.2435, 0.2616` versus stock `resnet18` and std `0.2023, 0.1994, 0.2010`). The paper does not choose between them.
- A reproduction of the reported accuracies. The numbers below were not retrained.

## Run the October 2025 scripts

These commands run the short 2025 scripts only. They do not reproduce the ResNet-18 tables.

```bash
python experiments/mnist_train.py
python experiments/cifar10_train.py
python experiments/imdb_train.py
python experiments/beta_sweep.py
python experiments/plot_results.py
```

## Citation
If you use this code or findings in academic work, please cite:

@article{koyun2025moment,
  title={Moment-Guided Optimizers for Noisy Deep Learning},
  author={Koyun, Alicem},
  year={2025},
  publisher={Independent Research}
}

@article{koyun2026mgosc,
  title={Moment-Guided Optimizers for Noisy Deep Learning: MGO-SC as a Simple Consistency Gate for Adam-Style Updates},
  author={Koyun, Alicem},
  journal={arXiv:2610.37712},
  year={2026}
}

For questions or collaboration:
Alicem Koyun
alicemkoyun@gmail.com
