# Audit of arXiv:2610.37712v1 against the surviving sources

This record compares the published paper, the public git history, one author-provided zip, and the two files the author later identified as the latest copies still remembered. It does not rerun training and does not add an optimizer.

The emails that prompted the check ask where the ResNet-18 protocol and logs are, and whether Equation (7) preserves the squared step norm. They do not need to be read as an allegation about the tables.

## Public history before this commit

`main` at `84c613463f45b7f9093a1189817507b7493eee05` (8 October 2025) is a single line of history: license, then the 2025 scripts. There are no other branches, tags, releases, or dangling commits. A search of that history finds no `gate`, `ResNet`, sign-EMA, label noise, or MGO-SC.

`experiments/cifar10_train.py` trains `SimpleCNN` for 3 epochs. `optimizers/adam_custom.py` keeps `step`, `exp_avg`, and `exp_avg_sq` only. That matches the inspection reported against this commit. Those files are unchanged here.

## Two sources, kept separate

| | Zip `code/` (entry time 2026-07-05) | Latest remembered copies |
| --- | --- | --- |
| Optimizer SHA-256 | `bb499ffdcc5f2c0aa907178a99c02af563fc033adb4994fe1a5f0e4bdf471bc9` | `8f40a31c95a62ba9a6c5fd8124b8b19eac198dc33d4cce8a371bf88dbd7dcea3` |
| Trainer SHA-256 | `98a548a7219f43d0400564909bdc824c785ed7e8c1afd14c73ad36740d4d667c` | `fe3d36b382fc94b374d7e5e87a3afbd662cf047dc4b8408b8f731ae64b3933cb` |
| Gate | floor `clamp_(min=r_min)` only (`mgosc.py` lines 149–152) | `torch.clamp(..., min=r_min, max=1)` when `r_min > 0` (lines 148–151) |
| Mean-preserving division | absent | absent |
| `--gate_mode` | absent | absent |
| Weight decay | coupled into the gradient (lines 108–110) | decoupled `p.mul_(1 - lr * wd)` (lines 102–104) |
| Stem | CIFAR 3×3, `maxpool = Identity` (trainer lines 61–66) | stock `resnet18(num_classes=10)` (trainer lines 114–116) |
| Normalization std | `(0.2470, 0.2435, 0.2616)` (line 127) | `(0.2023, 0.1994, 0.2010)` (line 83) |
| Symmetric noise | always a different class (lines 45–49) | uniform label in `0..9`, which can be the original (lines 64–66) |
| Optimizers the trainer can launch | adam, adamw, sgd, rmsprop, adagrad, mgosc | adam, mgosc |
| Tag | has `_wu`, no loss / noise-type / `gmrenorm` (lines 108–115) | no `_wu`, still no loss / noise-type / `gmrenorm` (lines 316–322) |
| Summary columns | `best_val_acc` plus run metadata, no `mean_epoch_time_s` | `tag, seed, best_epoch, best_val, test_acc_at_best_val, epoch_csv` |
| Epoch columns | `active_ratio`, `update_norm`, losses; calls `get_step_stats`, which the optimizer does not define (trainer line 226) | `gate_mean`, `gate_active_ratio`, `update_rms`, `grad_rms`; no `epoch_time_s` |

The author said the second pair was the latest still on hand and that nothing afterward is remembered. This audit does not invent a third program. It also does not call either file the writer of the CSVs. The zip's own `README_experiments.md` says its code produced the CSVs; the schema comparison does not support that sentence.

The paper's symmetric-noise sentence (replace a label with a uniform draw from the 10 classes) matches the later trainer's noise function, not the July trainer's always-wrong flip. The paper's "decoupled weight decay" sentence matches the later optimizer, not the July optimizer. The paper never states the stem or the normalization constants, and the two trainers disagree on both. Those agreements are not identification. The column and tag mismatches below still separate both trainers from the files that were saved.

## What the CSVs actually contain

Zip SHA-256 `f1af1a3d268527110d6bcca9120a7760931bfffde02e7736749bb1160dc87752`. The 55 CSVs are in `artifacts/results_csv/`. Summary columns are:

`tag, seed, best_epoch, best_val, test_acc_at_best_val, mean_epoch_time_s, epoch_csv`

Epoch columns add `epoch_time_s` to the later trainer's epoch header. Nineteen summary rows point at `/content/drive/MyDrive/mgosc_results/<epoch file>`. All 19 files are in the directory, and each logged best-validation row matches that summary's epoch, validation accuracy, and test accuracy (`tests/test_csv_tables.py`).

The epoch-1 learning rate in those logs is `0.0002997040092642407`. That is `CosineAnnealingLR` with `T_max=50` and minimum 0, after one epoch step, at base learning rate `3e-4`. Both trainers call that scheduler once per epoch. The schedule matches the logs. It does not identify which trainer wrote them.

Names that neither trainer builds:

- `_lossce` / `_lossgce`
- `_ntsymmetric` / `_ntasymmetric`
- `_gmrenorm` after the seed count
- `mean_epoch_time_s` and `epoch_time_s`

The July trainer would also have written `_wu`. The saved names have no `_wu`.

For the 40% symmetric, learning rate `3e-4` files, the n=5 AdamW rows and the n=5 shrink MGO-SC rows are the seeds 1–5 of the corresponding n=8 files, not a second cohort.

`_gmrenorm` epoch logs have `gate_mean` within `1e-7` of 1 and `gate_active_ratio` exactly 1. The other epoch logs, which have no `gmrenorm` in the name, have `gate_mean` in `(0.277, 0.278)`. That is the pattern a shrink gate and a mean-one gate would log. It is evidence about the runs that were saved. It is not a source file. Neither `mgosc.py` divides by the coordinate mean.

## Published numbers that match these CSVs

Sample mean and sample standard deviation, times 100, rounded to two decimals, match Tables I–III entry for entry. The η=0.4 AdamW cell uses the n=8 file (73.61 ± 0.20), as the caption says. The same check covers the ablation grid and the asymmetric table.

Paired test accuracy, mean-preserving tag minus AdamW, at η=0.4, n=8, seeds 1–8, in percentage points:

`1.54, 1.13, 1.23, 1.02, 1.05, 1.56, 1.52, 1.81`

All eight are positive. The exact two-sided Wilcoxon signed-rank probability is `2/256 = 0.0078125`, which is the published `p = 0.0078`. The mean paired difference rounds to `+1.4`.

The same comparison against the `g0.0` file, rather than against AdamW, has mean `+1.48`, which rounds to the ablation's `+1.5`. A percentile bootstrap of those eight paired differences (20,000 resamples, Python `random.Random(0)`, linear quantile) rounds to the published interval `[+1.0, +2.0]`, and its Wilcoxon probability is again `0.0078125`.

η=0, n=3, all three paired differences are positive and the same bootstrap stays above 0. η=0.2, seeds 1–5:

`+0.75, -0.14, +0.87, +0.87, +1.35`

The mean rounds to `+0.7` and the bootstrap stays above 0, so the published "intervals exclude zero" sentence holds for this procedure. Seed 2 does not improve. "Consistently" is not true seed by seed at η=0.2. The Wilcoxon probability for this n=5 comparison is not the headline `0.0078`.

Asymmetric η=0.4, seeds 1–5:

`+0.77, +0.43, +1.88, -0.13, +0.56`

Mean rounds to `+0.7`. The same bootstrap rounds to `[+0.2, +1.4]`. Seed 4 is negative. The paper does not claim a Wilcoxon test for this table.

The bootstrap procedure is not stated in the paper. The checks above are one explicit percentile bootstrap, not every possible interval.

## Published sentences these files do not support

1. Section III-C, after Equation (7). Mean 1 does not keep the squared step the same. For `r = (1/5, 1)`, `s = (1/3, 5/3)` and `u = (1, 1)`, `mean(s) = 1` but `||s ⊙ u||^2 / ||u||^2 = 13/9`. If `r_i ∈ [r_min, 1]` and `r_min > 0`, the factor is at most `1/r_min^2`. The suggested wording is `PROPOSED_III_C_QUALIFICATION.md`. It has not been posted as a new arXiv version.

2. "γ = 0 coincides with AdamW" (Table II caption and the ablation paragraph). The `g0.0` n=8 accuracies are not the AdamW n=8 accuracies (73.49 ± 0.54 versus 73.61 ± 0.20). The `+1.5` figure is the gap versus `g0.0`. The `+1.4` figure is the gap versus AdamW. The paper treats those baselines as the same run.

3. The implementation paragraph: a `--gate_mode` flag, optional sign-EMA bias correction as the main setting, and decoupled weight decay inside MGO-SC. CSV tags say `bcs0`. The later optimizer can do decoupled decay and can turn bias correction on, but it has no mode flag and no mean division. The July optimizer is coupled and floors without an upper clamp.

4. "Adam and AdamW run at about 9.1 s/epoch and MGO-SC at about 14.6 s/epoch." AdamW n=8 averages 9.06 s. Several MGO summaries without epoch logs, including `g0.7` with `gmrenorm`, sit near 14.6 s. The n=8 `gmrenorm` rows that point at epoch logs average about 36.3 s. The paper's single 14.6 s figure does not describe those logged headline runs. Both gate families have summaries near 14–15 s, and the logged mean-preserving runs do not.

5. A binary sign-agreement ablation is mentioned in the method section. No CSV with that tag is in the zip.

6. A learning-rate check at `3e-4` and `6e-4` is mentioned. The zip has Adam at `6e-4` (78.71 ± 0.37) and MGO-SC at `6e-4` with no `gmrenorm` suffix (74.06 ± 0.44). Tables I–III do not print those two rows. There is no `6e-4` file with `gmrenorm`.

7. GCE's `q`, the asymmetric transition, the stem, and the normalization constants are not fixed by the paper, and the two trainers do not jointly fix the stem or the normalization.

## What a further release would have to add

A reconstruction can be written from Algorithm 1, but it would be a new program. It is not in either surviving file, and this repository does not add one under the original filenames. The open choices are the stem, the normalization, the symmetric-noise convention, whether γ=0 must match `torch.optim.AdamW` exactly, the GCE parameter, and the asymmetric matrix. The author does not remember a later script. Until one is found, the honest public release is the two sources, the CSVs, and the table checks above.
