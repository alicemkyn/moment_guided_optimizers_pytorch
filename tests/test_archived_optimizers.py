"""Behavior of the two preserved MGOSC sources.

These tests characterize the files as stored. They are not a new optimizer,
and a passing test does not mean either file wrote the paper tables.
"""

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def load_class(filename):
    spec = importlib.util.spec_from_file_location(filename.stem + "_under_test", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MGOSC


class ArchivedOptimizers(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("torch is not installed")

    def test_latest_gate_is_shrink_not_mean_one(self):
        cls = load_class(ROOT / "artifacts/latest-remembered/mgosc.py")
        parameter = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
        optimizer = cls(
            [parameter],
            lr=0.1,
            betas=(0.9, 0.999),
            beta_s=0.9,
            weight_decay=0.0,
            gamma=0.5,
            r_min=0.2,
            bias_correction_s=False,
        )
        parameter.grad = torch.tensor([1.0, -2.0])
        optimizer.step()
        stats = optimizer.get_last_step_stats()
        gate_mean = stats["gate_sum"] / stats["gate_cnt"]
        # |s| = 1 - beta_s = 0.1 on the first step, so the shrink gate is sqrt(0.1).
        self.assertAlmostEqual(gate_mean, 0.1 ** 0.5, places=6)
        self.assertNotAlmostEqual(gate_mean, 1.0, places=3)

    def test_latest_gamma_zero_uses_decoupled_decay_not_coupled_adam(self):
        cls = load_class(ROOT / "artifacts/latest-remembered/mgosc.py")
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = cls(
            [parameter],
            lr=0.1,
            betas=(0.9, 0.999),
            beta_s=0.9,
            eps=1e-8,
            weight_decay=0.5,
            gamma=0.0,
            r_min=0.2,
            bias_correction_s=False,
        )
        gradient = torch.tensor([1.0])
        parameter.grad = gradient.clone()
        optimizer.step()
        # Decoupled decay, then the bias-corrected first step, which equals the gradient direction.
        expected = 2.0 * (1.0 - 0.1 * 0.5) - 0.1 * (1.0 / (1.0 + 1e-8))
        self.assertAlmostEqual(parameter.item(), expected, places=6)

    def test_archive_clamp_has_no_upper_bound(self):
        archive = load_class(ROOT / "artifacts/source-archive-2026-07-05/code/mgosc.py")
        latest = load_class(ROOT / "artifacts/latest-remembered/mgosc.py")

        def step_size(cls):
            parameter = torch.nn.Parameter(torch.zeros(1))
            kwargs = dict(
                lr=1.0,
                betas=(0.9, 0.999),
                beta_s=0.999,
                eps=1e-8,
                weight_decay=0.0,
                gamma=1.0,
                r_min=0.2,
                bias_correction_s=False,
            )
            if "collect_stats" in cls.__init__.__code__.co_varnames:
                kwargs["collect_stats"] = True
            optimizer = cls([parameter], **kwargs)
            parameter.grad = torch.ones(1)
            optimizer.step()
            state = optimizer.state[parameter]
            state["s"].fill_(4.0)
            state["m"].zero_()
            state["v"].zero_()
            state["t"] = 0
            parameter.data.zero_()
            parameter.grad = torch.ones(1)
            optimizer.step()
            # t=1, m_hat = g, v_hat = g^2, direction = 1/(1+eps)
            # s = 0.999*4 + 0.001*1, then the gate is applied to a zero parameter.
            direction = 1.0 / (1.0 + 1e-8)
            s_value = 0.999 * 4.0 + 0.001 * 1.0
            return float((-parameter).item()) / direction, s_value

        latest_gate, s_value = step_size(latest)
        archive_gate, archive_s = step_size(archive)
        self.assertAlmostEqual(s_value, archive_s, places=6)
        self.assertAlmostEqual(latest_gate, 1.0, places=5)
        self.assertAlmostEqual(archive_gate, s_value, places=4)
        self.assertGreater(archive_gate, 1.0)

    def test_archive_weight_decay_is_coupled_into_the_gradient(self):
        archive = load_class(ROOT / "artifacts/source-archive-2026-07-05/code/mgosc.py")
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = archive(
            [parameter],
            lr=0.1,
            betas=(0.9, 0.999),
            beta_s=0.9,
            eps=1e-8,
            weight_decay=0.5,
            gamma=0.0,
            r_min=0.2,
            bias_correction_s=False,
        )
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        effective = 1.0 + 0.5 * 2.0
        expected = 2.0 - 0.1 * (effective / (abs(effective) + 1e-8))
        self.assertAlmostEqual(parameter.item(), expected, places=5)


if __name__ == "__main__":
    unittest.main()
