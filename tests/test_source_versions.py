"""The two surviving sources stay distinct and neither matches the CSV schema."""

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_MGOSC = "bb499ffdcc5f2c0aa907178a99c02af563fc033adb4994fe1a5f0e4bdf471bc9"
ARCHIVE_TRAIN = "98a548a7219f43d0400564909bdc824c785ed7e8c1afd14c73ad36740d4d667c"
LATEST_MGOSC = "8f40a31c95a62ba9a6c5fd8124b8b19eac198dc33d4cce8a371bf88dbd7dcea3"
LATEST_TRAIN = "fe3d36b382fc94b374d7e5e87a3afbd662cf047dc4b8408b8f731ae64b3933cb"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceVersions(unittest.TestCase):
    def test_bytes_are_the_preserved_originals(self):
        archive = ROOT / "artifacts" / "source-archive-2026-07-05" / "code"
        latest = ROOT / "artifacts" / "latest-remembered"
        self.assertEqual(digest(archive / "mgosc.py"), ARCHIVE_MGOSC)
        self.assertEqual(digest(archive / "train_cifar10_resnet18.py"), ARCHIVE_TRAIN)
        self.assertEqual(digest(latest / "mgosc.py"), LATEST_MGOSC)
        self.assertEqual(digest(latest / "train_cifar10_resnet18.py"), LATEST_TRAIN)
        self.assertNotEqual(ARCHIVE_MGOSC, LATEST_MGOSC)
        self.assertNotEqual(ARCHIVE_TRAIN, LATEST_TRAIN)

    def test_neither_optimizer_implements_mean_preserving_mode(self):
        for relative in (
            "artifacts/source-archive-2026-07-05/code/mgosc.py",
            "artifacts/latest-remembered/mgosc.py",
        ):
            text = (ROOT / relative).read_text()
            self.assertNotIn("gate_mode", text)
            self.assertNotIn("gmrenorm", text)
            self.assertNotIn(".mean(", text)

    def test_tag_and_column_gaps(self):
        archive = (ROOT / "artifacts/source-archive-2026-07-05/code/train_cifar10_resnet18.py").read_text()
        latest = (ROOT / "artifacts/latest-remembered/train_cifar10_resnet18.py").read_text()
        for text in (archive, latest):
            self.assertNotIn("lossce", text)
            self.assertNotIn("ntsymmetric", text)
            self.assertNotIn("ntasymmetric", text)
            self.assertNotIn("gmrenorm", text)
            self.assertNotIn("mean_epoch_time_s", text)
            self.assertNotIn("epoch_time_s", text)
            self.assertNotIn("gate_mode", text)
        # The archive tag carries warmup. The CSV names do not.
        self.assertIn("_wu{args.warmup_steps}", archive)
        self.assertNotIn("warmup", latest)
        # The later trainer writes the column names that are actually in the epoch logs,
        # except epoch_time_s. The archive trainer writes different names.
        self.assertIn("gate_active_ratio", latest)
        self.assertIn("update_rms", latest)
        self.assertIn("grad_rms", latest)
        self.assertIn('"best_val"', latest)
        self.assertNotIn("best_val_acc", latest)
        self.assertIn("active_ratio", archive)
        self.assertIn("best_val_acc", archive)
        self.assertNotIn("gate_active_ratio", archive)

    def test_protocol_disagreements_are_still_in_the_sources(self):
        archive_opt = (ROOT / "artifacts/source-archive-2026-07-05/code/mgosc.py").read_text()
        latest_opt = (ROOT / "artifacts/latest-remembered/mgosc.py").read_text()
        archive_train = (ROOT / "artifacts/source-archive-2026-07-05/code/train_cifar10_resnet18.py").read_text()
        latest_train = (ROOT / "artifacts/latest-remembered/train_cifar10_resnet18.py").read_text()
        self.assertIn("g.add(p, alpha=wd)", archive_opt)
        self.assertIn("p.mul_(1.0 - lr * wd)", latest_opt)
        self.assertIn("m.conv1", archive_train)
        self.assertIn("m.maxpool = nn.Identity()", archive_train)
        self.assertNotIn("maxpool", latest_train)
        self.assertIn("resnet18(num_classes=10)", latest_train)
        self.assertIn("(0.2470, 0.2435, 0.2616)", archive_train)
        self.assertIn("(0.2023, 0.1994, 0.2010)", latest_train)
        self.assertIn("randrange(num_classes - 1)", archive_train)
        self.assertIn("randint(0, self.num_classes)", latest_train)
        self.assertIn('"adamw"', archive_train)
        self.assertNotIn("amsgrad", archive_train.lower())
        self.assertNotIn("adamw", latest_train)
        self.assertNotIn("gce", latest_train.lower())
        self.assertNotIn("asymmetric", latest_train.lower())


if __name__ == "__main__":
    unittest.main()
