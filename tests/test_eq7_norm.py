"""Exact rational check of arXiv:2610.37712v1 Equation (7).

mean(s) = 1 does not preserve ||s ⊙ u||^2. The rate is controlled by
the largest gate entry.
"""

import unittest
from fractions import Fraction


def mean(xs):
    return sum(xs) / len(xs)


class MeanPreservingGateNorm(unittest.TestCase):
    def test_counterexample_inflates_squared_norm(self):
        r = (Fraction(1, 5), Fraction(1))
        s = tuple(x / mean(r) for x in r)
        u = (Fraction(1), Fraction(1))
        self.assertEqual(s, (Fraction(1, 3), Fraction(5, 3)))
        self.assertEqual(mean(s), 1)
        norm_u = sum(v * v for v in u)
        norm_su = sum((a * b) * (a * b) for a, b in zip(s, u))
        self.assertEqual(norm_u, 2)
        self.assertEqual(norm_su, Fraction(26, 9))
        self.assertEqual(norm_su / norm_u, Fraction(13, 9))
        self.assertGreater(norm_su, norm_u)

    def test_equal_magnitude_update_strictly_increases_unless_gate_constant(self):
        s = (Fraction(1, 3), Fraction(5, 3))
        self.assertEqual(mean(s), 1)
        self.assertGreater(mean(tuple(v * v for v in s)), 1)
        # Constant gate is the equality case.
        constant = (Fraction(1), Fraction(1))
        self.assertEqual(mean(tuple(v * v for v in constant)), 1)

    def test_r_min_bound_covers_the_counterexample(self):
        r_min = Fraction(1, 5)
        r = (r_min, Fraction(1))
        s = tuple(x / mean(r) for x in r)
        u = (Fraction(1), Fraction(1))
        norm_u = sum(v * v for v in u)
        norm_su = sum((a * b) * (a * b) for a, b in zip(s, u))
        for value in s:
            self.assertGreaterEqual(value, r_min)
            self.assertLessEqual(value, 1 / r_min)
        self.assertLessEqual(norm_su, (1 / r_min) ** 2 * norm_u)
        # The mean-one property is strictly weaker than this bound:
        # the example exceeds ||u||^2 while respecting 1/r_min.
        self.assertGreater(norm_su, norm_u)


if __name__ == "__main__":
    unittest.main()
