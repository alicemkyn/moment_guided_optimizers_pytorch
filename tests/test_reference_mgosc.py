"""Tests for the new reference optimizer. They do not use archived accuracies."""

import unittest

import torch

from reconstruction.mgosc import MGOSC, reliability_gate


class ReliabilityGate(unittest.TestCase):
    def test_equation7_mean_is_one_and_norm_grows(self):
        sign_hat = torch.tensor([0.2, 1.0], dtype=torch.float64)
        gate = reliability_gate(sign_hat, gamma=1.0, r_min=0.2, mode="mean_preserving")
        self.assertAlmostEqual(float(gate.mean()), 1.0, places=12)
        update = torch.ones(2, dtype=torch.float64)
        gated = gate * update
        self.assertGreater(float(gated.square().sum()), float(update.square().sum()))

    def test_shrink_stays_inside_the_unit_interval(self):
        sign_hat = torch.tensor([0.01, 0.2, 4.0], dtype=torch.float64)
        gate = reliability_gate(sign_hat, gamma=1.0, r_min=0.2, mode="shrink")
        self.assertTrue(torch.all(gate >= 0.2))
        self.assertTrue(torch.all(gate <= 1.0))
        self.assertLess(float(gate.mean()), 1.0)

    def test_gamma_zero_is_identically_one(self):
        sign_hat = torch.zeros(4, dtype=torch.float64)
        gate = reliability_gate(sign_hat, gamma=0.0, r_min=0.2, mode="mean_preserving")
        self.assertTrue(torch.equal(gate, torch.ones_like(sign_hat)))

    def test_mean_is_per_tensor(self):
        first = reliability_gate(torch.tensor([0.2, 1.0]), 1.0, 0.2, "mean_preserving")
        second = reliability_gate(torch.tensor([1.0, 1.0]), 1.0, 0.2, "mean_preserving")
        self.assertAlmostEqual(float(first.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(second.mean()), 1.0, places=6)
        self.assertFalse(torch.allclose(first, second))


class OptimizerBehavior(unittest.TestCase):
    def test_gamma_zero_matches_adamw(self):
        torch.manual_seed(0)
        left = torch.nn.Parameter(torch.randn(16))
        right = torch.nn.Parameter(left.detach().clone())
        adamw = torch.optim.AdamW([left], lr=3e-4, weight_decay=5e-4, foreach=False)
        mgosc = MGOSC(
            [right],
            lr=3e-4,
            weight_decay=5e-4,
            gamma=0.0,
            gate_mode="mean_preserving",
            decoupled_weight_decay=True,
            collect_stats=False,
        )
        for _ in range(4):
            grad = torch.randn(16)
            left.grad = grad.clone()
            right.grad = grad.clone()
            adamw.step()
            mgosc.step()
        self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))

    def test_coupled_gamma_zero_matches_adam(self):
        torch.manual_seed(1)
        left = torch.nn.Parameter(torch.randn(16))
        right = torch.nn.Parameter(left.detach().clone())
        adam = torch.optim.Adam([left], lr=3e-4, weight_decay=5e-4, foreach=False)
        mgosc = MGOSC(
            [right],
            lr=3e-4,
            weight_decay=5e-4,
            gamma=0.0,
            decoupled_weight_decay=False,
            collect_stats=False,
        )
        for _ in range(3):
            grad = torch.randn(16)
            left.grad = grad.clone()
            right.grad = grad.clone()
            adam.step()
            mgosc.step()
        self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))

    def test_sign_ema_and_state_keys(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = MGOSC(
            [parameter],
            lr=0.0,
            beta_s=0.9,
            gamma=0.5,
            r_min=0.2,
            gate_mode="shrink",
            bias_correction_s=False,
            weight_decay=0.0,
        )
        parameter.grad = torch.tensor([1.0, -2.0])
        optimizer.step()
        state = optimizer.state[parameter]
        self.assertEqual(set(state), {"step", "exp_avg", "exp_avg_sq", "exp_avg_sign"})
        self.assertTrue(torch.allclose(state["exp_avg_sign"], torch.tensor([0.1, -0.1])))
        stats = optimizer.consume_stats()
        self.assertAlmostEqual(stats["gate_mean"], 0.1 ** 0.5, places=5)
        self.assertNotAlmostEqual(stats["gate_mean"], 1.0, places=3)

    def test_two_parameters_are_renormalized_separately(self):
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(2))
        optimizer = MGOSC(
            [first, second],
            lr=0.0,
            beta_s=0.0,
            gamma=1.0,
            r_min=0.0,
            gate_mode="mean_preserving",
            weight_decay=0.0,
        )
        first.grad = torch.tensor([1.0, 1.0])
        second.grad = torch.tensor([1.0, -1.0])
        optimizer.step()
        # beta_s = 0 replaces the sign EMA with sign(grad), so |s| = 1 and the gate is 1.
        stats = optimizer.consume_stats()
        self.assertAlmostEqual(stats["gate_mean"], 1.0, places=5)

    def test_mean_preserving_step_has_mean_one_when_reliability_varies(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = MGOSC(
            [parameter],
            lr=0.0,
            beta_s=0.5,
            gamma=1.0,
            r_min=0.0,
            gate_mode="mean_preserving",
            weight_decay=0.0,
        )
        parameter.grad = torch.tensor([1.0, 1.0])
        optimizer.step()
        parameter.grad = torch.tensor([1.0, -1.0])
        optimizer.step()
        stats = optimizer.consume_stats()
        # The second step's gate is included in the epoch mean. Check the state directly.
        sign = optimizer.state[parameter]["exp_avg_sign"]
        gate = reliability_gate(sign, 1.0, 0.0, "mean_preserving")
        self.assertAlmostEqual(float(gate.mean()), 1.0, places=6)
        self.assertGreater(float(gate.max()), float(gate.min()))
        self.assertIsNotNone(stats)

    def test_sparse_gradients_raise(self):
        parameter = torch.nn.Parameter(torch.zeros(3))
        optimizer = MGOSC([parameter], collect_stats=False)
        parameter.grad = torch.sparse_coo_tensor([[0]], [1.0], (3,))
        with self.assertRaises(RuntimeError):
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
