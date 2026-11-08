"""Harness tests: split, noise, selection, campaign guard, and a CPU smoke run."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from reconstruction.assumptions import ASYMMETRIC_PRESETS
from reconstruction.campaign import headline_command
from reconstruction.noise import corrupt_labels, train_val_split
from reconstruction.resnet import ResNet18
from reconstruction.selection import consider
from reconstruction.stats import percentile_bootstrap, wilcoxon_two_sided

ROOT = Path(__file__).resolve().parents[1]


class Protocol(unittest.TestCase):
    def test_split_is_a_partition(self):
        train_idx, val_idx = train_val_split(100, 20, seed=3, rng_name="numpy")
        self.assertEqual(len(train_idx), 80)
        self.assertEqual(len(val_idx), 20)
        self.assertEqual(set(train_idx).intersection(val_idx), set())
        self.assertEqual(set(train_idx).union(val_idx), set(range(100)))

    def test_symmetric_modes_differ_and_are_paired(self):
        labels = np.arange(10)
        uniform = corrupt_labels(labels, 1.0, 7, "symmetric", "uniform_including_true", None)
        flipped = corrupt_labels(labels, 1.0, 7, "symmetric", "always_wrong", None)
        again = corrupt_labels(labels, 1.0, 7, "symmetric", "uniform_including_true", None)
        self.assertTrue(np.array_equal(uniform, again))
        self.assertTrue(np.any(uniform == labels) or np.any(uniform != labels))
        self.assertTrue(np.all(flipped != labels))
        # The paper mode is allowed to keep a label. The July mode is not.
        kept = corrupt_labels(np.zeros(200, dtype=np.int64), 1.0, 1, "symmetric", "uniform_including_true", None)
        self.assertTrue(np.any(kept == 0))

    def test_asymmetric_preset_is_explicit(self):
        with self.assertRaises(ValueError):
            corrupt_labels(np.arange(10), 0.4, 0, "asymmetric", "uniform_including_true", None)
        labels = np.arange(10)
        mapped = corrupt_labels(labels, 1.0, 0, "asymmetric", "uniform_including_true", "patrini2017")
        for source, target in ASYMMETRIC_PRESETS["patrini2017"].items():
            self.assertEqual(int(mapped[source]), target)
        self.assertEqual(int(mapped[0]), 0)

    def test_selection_ignores_a_higher_test_score(self):
        best = consider(None, 1, val_acc=0.5, test_acc=0.9)
        best = consider(best, 2, val_acc=0.4, test_acc=0.99)
        best = consider(best, 3, val_acc=0.7, test_acc=0.1)
        self.assertEqual(best["epoch"], 3)
        self.assertEqual(best["test_acc"], 0.1)
        tied = consider(best, 4, val_acc=0.7, test_acc=0.8)
        self.assertEqual(tied["epoch"], 3)

    def test_headline_command_is_the_paired_n8_comparison(self):
        command = headline_command()
        self.assertIn("--label-noise 0.4", command)
        self.assertIn("--seeds 8", command)
        self.assertIn("--opt paired", command)
        self.assertIn("--gate-mode mean_preserving", command)
        self.assertIn("--accept-assumptions", command)
        self.assertNotIn("artifacts/results_csv", command)

    def test_stems_have_the_standard_parameter_counts(self):
        cifar = sum(p.numel() for p in ResNet18("cifar").parameters())
        imagenet = sum(p.numel() for p in ResNet18("imagenet").parameters())
        self.assertEqual(cifar, 11_173_962)
        self.assertEqual(imagenet, 11_181_642)
        self.assertNotEqual(cifar, imagenet)

    def test_exact_wilcoxon_on_a_known_sign_pattern(self):
        self.assertEqual(wilcoxon_two_sided([0.1] * 8), 2 / 256)
        report = percentile_bootstrap([0.1, -0.2, 0.3], replicates=1000, seed=0)
        self.assertLess(report["low"], report["mean"])
        self.assertLess(report["mean"], report["high"])


class Smoke(unittest.TestCase):
    def test_cpu_smoke_writes_reconstruction_provenance(self):
        out = ROOT / "reconstruction_runs" / "unit_smoke"
        if out.exists():
            for path in out.glob("*"):
                path.unlink()
        command = [
            sys.executable,
            "-m",
            "reconstruction.train",
            "--dataset",
            "synthetic",
            "--epochs",
            "1",
            "--max-steps",
            "2",
            "--seeds",
            "1",
            "--batch-size",
            "8",
            "--synthetic-n",
            "40",
            "--val-size",
            "8",
            "--opt",
            "paired",
            "--gate-mode",
            "mean_preserving",
            "--label-noise",
            "0.4",
            "--noise-type",
            "symmetric",
            "--out-dir",
            str(out),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("artifact_class=reconstruction", completed.stdout)
        provenance = json.loads((out / "provenance.json").read_text())
        self.assertEqual(provenance["artifact_class"], "reconstruction")
        self.assertTrue(provenance["not_historical"])
        self.assertEqual(provenance["new_assumptions_in_effect"]["stem"], "cifar")
        self.assertIn("git_commit", provenance)
        summary = (out / "summary.csv").read_text()
        self.assertIn("adamw", summary)
        self.assertIn("mgosc", summary)
        self.assertIn("smoke", summary)
        analysis = subprocess.run(
            [sys.executable, "-m", "reconstruction.analyze", "--summary", str(out / "summary.csv")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(analysis.stdout)
        self.assertTrue(report["not_a_historical_comparison"])
        self.assertEqual(len(report["deltas"]), 1)

    def test_cifar_without_acceptance_exits(self):
        completed = subprocess.run(
            [sys.executable, "-m", "reconstruction.train", "--dataset", "cifar10", "--epochs", "1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("not a historical fact", completed.stdout)

    def test_archived_directory_is_refused(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "reconstruction.analyze",
                "--summary",
                "artifacts/results_csv/cifar10_resnet18_adamw_lossce_ntsymmetric_lr0.0003_wd0.0005_bs128_noise0.4_betaS0.9_g0.5_rmin0.2_bcs0_E50_n8.csv",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
