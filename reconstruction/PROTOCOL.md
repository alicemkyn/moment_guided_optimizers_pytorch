# Decision matrix for a new preregistered run

This file describes the reconstruction in `reconstruction/`. It is not a claim that these defaults were used to write `artifacts/results_csv/`. The archived sources and CSVs are unchanged.

Mean 1 for Equation (7) does not preserve the squared update norm. See `PROPOSED_III_C_QUALIFICATION.md`.

## Choices the paper states

| Choice | Value |
| --- | --- |
| Data | CIFAR-10, official 10k test set |
| Split | 45,000 train / 5,000 validation, validation drawn from the training split |
| Selection | highest validation accuracy; report test accuracy at that checkpoint |
| Model | ResNet-18, random crop with padding 4, horizontal flip, per-channel normalization |
| Epochs, batch, weight decay | 50, 128, 5e-4 |
| Adam-family learning rate | 3e-4. SGD learning rate 0.1 |
| Schedule | cosine. Minimum learning rate is not stated |
| MGO-SC | gamma 0.5, r_min 0.2, decoupled weight decay, mean-preserving gate for the headline |
| Headline | 40% symmetric noise, n = 8, paired with AdamW |
| Symmetric noise sentence | with probability eta, replace the label by a uniform draw from the 10 classes |

## Choices that are still open

| Choice | Default used by a new run | Status |
| --- | --- | --- |
| Stem | `cifar` (3x3, stride 1, no max-pool) | New assumption. Alternative: `--stem imagenet` (7x7, stride 2, max-pool). The two surviving trainers disagree. |
| Normalization | `cifar10_std_2470` | New assumption. Alternative: `--normalization cifar10_std_2023`. The paper gives no constants. |
| Symmetric draw | `uniform_including_true` | Follows the paper sentence. Alternative: `--symmetric-mode always_wrong`, which is the July trainer. |
| beta_s, sign-EMA bias correction | 0.9, off | Filename tags `betaS0.9` and `bcs0`. The paper does not state them. Pass `--bias-correction-s` to turn correction on. |
| SGD momentum / Nesterov | 0.9 / off | New assumption. The July trainer uses Nesterov. Pass `--sgd-nesterov` to enable it. |
| GCE q | no default | Required via `--gce-q` when `--loss gce`. 0.7 is a Zhang and Sabuncu 2018 example, not an archived value. |
| Asymmetric map | no default | Required via `--asymmetric-preset patrini2017` or `next_class`. The paper does not give the map. |
| Split RNG | numpy | New assumption. Alternative: `--split-rng python`. |
| Noise-seed offset | 0 | New assumption. The July trainer used +123. |
| Cosine minimum | 0 | Matches the archived epoch-1 learning rate under one step of `CosineAnnealingLR`. The paper only says cosine. |
| Weight decay coupling for MGO-SC | decoupled | Paper statement. `--no-decoupled-weight-decay` selects the coupled alternative. |
| gamma = 0 versus AdamW | the new code matches `torch.optim.AdamW` when gamma is 0 and decay is decoupled | The archived gamma = 0 CSV is not the archived AdamW CSV. See the qualification note. |

Checkpoint selection reads validation accuracy only. Test accuracy is logged so the published reporting rule can be computed later. It is not a tuning target.

## Commands

List the campaign without running it:

```bash
python -m reconstruction.train --list-campaign
```

Headline, after the assumptions above are accepted. This was not executed here:

```bash
python -m reconstruction.train \
  --dataset cifar10 --accept-assumptions \
  --stem cifar --normalization cifar10_std_2470 \
  --noise-type symmetric --symmetric-mode uniform_including_true \
  --label-noise 0.4 --loss ce \
  --opt paired --gate-mode mean_preserving \
  --gamma 0.5 --r-min 0.2 --beta-s 0.9 \
  --epochs 50 --batch-size 128 --wd 5e-4 \
  --seeds 8 --seed0 1 \
  --out-dir reconstruction_runs/headline_eta04_n8
```

`--opt paired` runs AdamW and mean-preserving MGO-SC on the same seeds, with separate epoch clocks. Shrink at the same point, also not executed:

```bash
python -m reconstruction.train \
  --dataset cifar10 --accept-assumptions \
  --stem cifar --normalization cifar10_std_2470 \
  --noise-type symmetric --symmetric-mode uniform_including_true \
  --label-noise 0.4 --loss ce \
  --opt mgosc --gate-mode shrink \
  --epochs 50 --batch-size 128 --seeds 8 --seed0 1 \
  --out-dir reconstruction_runs/shrink_eta04_n8
```

Other intended comparisons, same explicit flags, not executed:

- Clean, and eta 0.2, for `adam`, `adamw`, `amsgrad`, `sgd`, `mgosc` with both gate modes. Clean uses `--seeds 3`. Eta 0.2 uses `--seeds 5`.
- Eta 0.4 baselines other than the n = 8 pair use `--seeds 5`.
- Gate ablation: `--label-noise 0.4 --seeds 8 --gamma` in `0`, `0.3`, `0.5`, `0.7`, once per gate mode. Gamma 0 is the gate-off reference, not a substitute for the AdamW logs.
- GCE: add `--loss gce --gce-q 0.7` only if that q is the preregistered choice. Do not treat 0.7 as the archived setting.
- Asymmetric: add `--noise-type asymmetric --asymmetric-preset patrini2017` or `next_class`.
- Learning-rate check: `--lr 6e-4` for Adam and for MGO-SC, and the matched `--lr 3e-4` runs.

Analyze a new summary only:

```bash
python -m reconstruction.analyze --summary reconstruction_runs/headline_eta04_n8/summary.csv
```

The analyzer rejects paths under `artifacts/results_csv/`.

## What this machine can say about cost

Measured on this VM on 2026-11-08, CPU only, no GPU:

- CIFAR-stem ResNet-18, 11,173,962 parameters, 32x32 input.
- Batch 2, two steps: 0.156 s total, 0.078 s/step.
- Batch 8, two steps: 0.191 s total, 0.096 s/step.
- Batch 128, one step: 0.898 s. That step is the optimizer update plus forward and backward. It does not include data loading or the validation and test passes.

A full 50-epoch CIFAR epoch was not run. A GPU epoch time was not measured. No dollar cost was measured. Do not convert the step time into a campaign budget without a new measurement on the machine that will run it.

`torchvision` 0.29.2+cpu imports in this environment. The CIFAR-10 files were not downloaded and the headline command was not started.
