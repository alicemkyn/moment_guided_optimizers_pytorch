"""Recompute published table numbers and paired tests from the archived CSVs.

Does not train, and does not treat either trainer as the writer of the CSVs.
"""

import csv
import math
import random
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "artifacts" / "results_csv"


def load(name):
    with (CSV_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def by_seed(name):
    rows = load(name)
    return {int(row["seed"]): row for row in rows}


def mean_std_percent(rows):
    values = [float(row["test_acc_at_best_val"]) for row in rows]
    mean = statistics.mean(values) * 100
    std = statistics.stdev(values) * 100 if len(values) > 1 else float("nan")
    return mean, std


def rounded_pair(rows):
    mean, std = mean_std_percent(rows)
    return round(mean, 2), round(std, 2)


def paired_deltas(left_name, right_name):
    left = by_seed(left_name)
    right = by_seed(right_name)
    seeds = sorted(set(left) & set(right))
    return [
        float(left[seed]["test_acc_at_best_val"]) - float(right[seed]["test_acc_at_best_val"])
        for seed in seeds
    ]


def wilcoxon_two_sided(deltas):
    """Exact two-sided Wilcoxon signed-rank p-value, dropping zeros."""
    cleaned = [delta for delta in deltas if delta != 0]
    order = sorted(range(len(cleaned)), key=lambda index: abs(cleaned[index]))
    ranks = [0.0] * len(cleaned)
    cursor = 0
    while cursor < len(order):
        stop = cursor
        while (
            stop + 1 < len(order)
            and abs(cleaned[order[stop + 1]]) == abs(cleaned[order[cursor]])
        ):
            stop += 1
        average = (cursor + 1 + stop + 1) / 2
        for position in range(cursor, stop + 1):
            ranks[order[position]] = average
        cursor = stop + 1
    observed = sum(rank for rank, delta in zip(ranks, cleaned) if delta > 0)
    total = sum(ranks)
    count = len(cleaned)
    extreme = 0
    patterns = 1 << count
    for mask in range(patterns):
        positive = 0.0
        for index in range(count):
            if mask >> index & 1:
                positive += ranks[index]
        if abs(positive - total / 2) >= abs(observed - total / 2) - 1e-12:
            extreme += 1
    return extreme / patterns


def percentile_bootstrap(deltas, replicates=20000, seed=0):
    generator = random.Random(seed)
    width = len(deltas)
    means = []
    for _ in range(replicates):
        total = 0.0
        for _draw in range(width):
            total += deltas[generator.randrange(width)]
        means.append(total / width)
    means.sort()

    def quantile(probability):
        position = probability * (replicates - 1)
        lower = math.floor(position)
        upper = min(lower + 1, replicates - 1)
        weight = position - lower
        return means[lower] * (1 - weight) + means[upper] * weight

    return statistics.mean(deltas), quantile(0.025), quantile(0.975)


class PublishedTables(unittest.TestCase):
    def test_table_i_matches_csv_rounding(self):
        expected = {
            "cifar10_resnet18_adam_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3.csv": (83.03, 0.41),
            "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3.csv": (82.06, 0.21),
            "cifar10_resnet18_amsgrad_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3.csv": (82.86, 0.28),
            "cifar10_resnet18_sgd_lossce_ntsymmetric_lr0.1_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3.csv": (86.61, 0.10),
            "cifar10_resnet18_adamw_lossgce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3.csv": (80.46, 0.15),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3.csv": (79.05, 0.17),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3_gmrenorm.csv": (82.87, 0.30),
            "cifar10_resnet18_adam_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (79.52, 0.41),
            "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (78.22, 0.48),
            "cifar10_resnet18_amsgrad_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (78.41, 0.68),
            "cifar10_resnet18_sgd_lossce_ntsymmetric_lr0.1_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (83.06, 0.20),
            "cifar10_resnet18_adamw_lossgce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (79.03, 0.46),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (74.54, 0.36),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5_gmrenorm.csv": (78.96, 0.73),
            "cifar10_resnet18_adam_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (76.00, 0.24),
            "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv": (73.61, 0.20),
            "cifar10_resnet18_amsgrad_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (74.23, 0.81),
            "cifar10_resnet18_sgd_lossce_ntsymmetric_lr0.1_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (78.43, 0.43),
            "cifar10_resnet18_adamw_lossgce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (77.09, 0.35),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv": (70.06, 0.25),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8_gmrenorm.csv": (74.97, 0.29),
        }
        for name, published in expected.items():
            self.assertEqual(rounded_pair(load(name)), published, name)

    def test_table_ii_and_iii_match_csv_rounding(self):
        ablation = {
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.0_rmin0.2_bcs0_E50_n8.csv": (73.49, 0.54),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.3_rmin0.2_bcs0_E50_n8.csv": (71.35, 0.51),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.3_rmin0.2_bcs0_E50_n8_gmrenorm.csv": (74.64, 0.41),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv": (70.06, 0.25),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8_gmrenorm.csv": (74.97, 0.29),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.7_rmin0.2_bcs0_E50_n8.csv": (68.69, 0.58),
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.7_rmin0.2_bcs0_E50_n8_gmrenorm.csv": (75.04, 0.42),
        }
        asymmetric = {
            "cifar10_resnet18_adam_lossce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (76.00, 0.48),
            "cifar10_resnet18_adamw_lossce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (73.70, 0.86),
            "cifar10_resnet18_adamw_lossgce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (71.94, 0.42),
            "cifar10_resnet18_mgosc_lossce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv": (69.89, 0.90),
            "cifar10_resnet18_mgosc_lossce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5_gmrenorm.csv": (74.40, 0.51),
        }
        for name, published in {**ablation, **asymmetric}.items():
            self.assertEqual(rounded_pair(load(name)), published, name)

    def test_gamma0_file_is_not_the_adamw_file(self):
        gamma0 = [
            float(row["test_acc_at_best_val"])
            for row in load(
                "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.0_rmin0.2_bcs0_E50_n8.csv"
            )
        ]
        adamw = [
            float(row["test_acc_at_best_val"])
            for row in load(
                "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv"
            )
        ]
        self.assertNotEqual(gamma0, adamw)
        self.assertEqual(rounded_pair(load(
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.0_rmin0.2_bcs0_E50_n8.csv"
        ))[0], 73.49)
        self.assertEqual(rounded_pair(load(
            "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv"
        ))[0], 73.61)

    def test_n5_files_are_prefixes_of_n8(self):
        pairs = [
            (
                "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv",
                "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv",
            ),
            (
                "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv",
                "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv",
            ),
        ]
        for short_name, long_name in pairs:
            short = by_seed(short_name)
            long = by_seed(long_name)
            for seed, row in short.items():
                self.assertEqual(row["test_acc_at_best_val"], long[seed]["test_acc_at_best_val"])
                self.assertEqual(row["best_epoch"], long[seed]["best_epoch"])

    def test_headline_paired_claims(self):
        renorm = "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8_gmrenorm.csv"
        adamw = "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv"
        gamma0 = "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.0_rmin0.2_bcs0_E50_n8.csv"
        versus_adamw = paired_deltas(renorm, adamw)
        versus_gamma0 = paired_deltas(renorm, gamma0)
        self.assertTrue(all(delta > 0 for delta in versus_adamw))
        self.assertTrue(all(delta > 0 for delta in versus_gamma0))
        self.assertEqual(wilcoxon_two_sided(versus_adamw), 2 / 256)
        self.assertEqual(wilcoxon_two_sided(versus_gamma0), 2 / 256)
        self.assertEqual(round(statistics.mean(versus_adamw) * 100, 1), 1.4)
        self.assertEqual(round(statistics.mean(versus_gamma0) * 100, 1), 1.5)
        _, low, high = percentile_bootstrap(versus_gamma0)
        self.assertEqual(round(low * 100, 1), 1.0)
        self.assertEqual(round(high * 100, 1), 2.0)

    def test_smaller_samples_are_not_uniform_wins(self):
        cases = [
            (
                "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3_gmrenorm.csv",
                "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.0_betaS0.9_g0.5_rmin0.2_bcs0_E50_n3.csv",
                True,
            ),
            (
                "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5_gmrenorm.csv",
                "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.2_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv",
                False,
            ),
            (
                "cifar10_resnet18_mgosc_lossce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5_gmrenorm.csv",
                "cifar10_resnet18_adamw_lossce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv",
                False,
            ),
        ]
        for left, right, all_positive in cases:
            deltas = paired_deltas(left, right)
            self.assertEqual(all(delta > 0 for delta in deltas), all_positive)
            _, low, _high = percentile_bootstrap(deltas)
            self.assertGreater(low, 0.0)

    def test_asymmetric_interval_rounds_to_the_published_one_decimal(self):
        deltas = paired_deltas(
            "cifar10_resnet18_mgosc_lossce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5_gmrenorm.csv",
            "cifar10_resnet18_adamw_lossce_ntasymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n5.csv",
        )
        mean, low, high = percentile_bootstrap(deltas)
        self.assertEqual(round(mean * 100, 1), 0.7)
        self.assertEqual(round(low * 100, 1), 0.2)
        self.assertEqual(round(high * 100, 1), 1.4)
        self.assertTrue(any(delta < 0 for delta in deltas))

    def test_epoch_logs_match_summary_checkpoints(self):
        linked = 0
        for summary_path in CSV_DIR.glob("cifar10*.csv"):
            for row in load(summary_path.name):
                pointer = row["epoch_csv"].strip()
                if not pointer:
                    continue
                epoch_path = CSV_DIR / Path(pointer).name
                self.assertTrue(epoch_path.exists(), pointer)
                with epoch_path.open(newline="") as handle:
                    epochs = list(csv.DictReader(handle))
                best = max(epochs, key=lambda item: float(item["val_acc"]))
                self.assertEqual(int(best["epoch"]), int(row["best_epoch"]))
                self.assertEqual(float(best["test_acc"]), float(row["test_acc_at_best_val"]))
                self.assertEqual(float(best["val_acc"]), float(row["best_val"]))
                linked += 1
        self.assertEqual(linked, 19)

    def test_logged_gate_means_match_filename_tags(self):
        renorm_gap = 0.0
        shrink_means = []
        for path in CSV_DIR.glob("epochs_*.csv"):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            means = [float(row["gate_mean"]) for row in rows]
            if "gmrenorm" in path.name:
                renorm_gap = max(renorm_gap, max(abs(value - 1.0) for value in means))
                self.assertTrue(all(float(row["gate_active_ratio"]) == 1.0 for row in rows))
            else:
                shrink_means.extend(means)
        self.assertLess(renorm_gap, 1e-6)
        self.assertTrue(all(0.27 < value < 0.28 for value in shrink_means))

    def test_epoch_time_split(self):
        def average(name):
            rows = load(name)
            return statistics.mean(float(row["mean_epoch_time_s"]) for row in rows)

        adamw = average(
            "cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv"
        )
        logged_renorm = average(
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8_gmrenorm.csv"
        )
        other_renorm = average(
            "cifar10_resnet18_mgosc_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.7_rmin0.2_bcs0_E50_n8_gmrenorm.csv"
        )
        self.assertGreater(adamw, 9.0)
        self.assertLess(adamw, 9.2)
        self.assertGreater(logged_renorm, 36.0)
        self.assertLess(logged_renorm, 37.0)
        self.assertGreater(other_renorm, 14.5)
        self.assertLess(other_renorm, 14.8)


if __name__ == "__main__":
    unittest.main()
