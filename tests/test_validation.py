"""Generation-time parameter validation tests."""

import os
import sys

import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gauss_batch import (
    analytic_moments,
    sample_coupled_batches,
    sample_target,
)


class ScriptedRNG:
    """Minimal rng replacement serving queued standard_normal / random draws."""

    def __init__(self, normals=None, uniforms=None):
        self._normals = list(normals or [])
        self._uniforms = list(uniforms or [])

    def standard_normal(self, shape=None):
        value = np.asarray(self._normals.pop(0), dtype=float)
        if shape is not None:
            if value.shape == tuple(shape):
                return value
            return value.reshape(shape)
        return value

    def random(self):
        return float(self._uniforms.pop(0))


def make_target_proposal(d=2, s2=0.5):
    A = 0.8 * np.eye(d)
    b = np.full(d, 0.1)
    Ap = 0.5 * np.eye(d)
    bp = np.zeros(d)
    return (A, b, s2), (Ap, bp, s2)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.t, self.p = make_target_proposal(2)
        self.x0 = np.zeros(2)

    def test_nonpositive_variance_rejected(self):
        t_bad = (self.t[0], self.t[1], 0.0)
        with self.assertRaises(ValueError):
            sample_target(self.x0, t_bad, 3)
        with self.assertRaises(ValueError):
            sample_coupled_batches(self.x0, t_bad, self.p, 3, 2)
        t_neg = (self.t[0], self.t[1], -1.0)
        with self.assertRaises(ValueError):
            sample_target(self.x0, t_neg, 3)
        # variance becomes invalid only on a later step
        t_later = (np.repeat(self.t[0][None], 3, 0),
                   np.repeat(self.t[1][None], 3, 0),
                   np.array([0.5, 0.5, 0.0]))
        with self.assertRaises(ValueError):
            sample_target(self.x0, t_later, 3)

    def test_mismatched_variance_rejected(self):
        p_bad = (self.p[0], self.p[1], 0.6)
        with self.assertRaises(ValueError):
            sample_coupled_batches(self.x0, self.t, p_bad, 3, 2)
        p_later = (np.repeat(self.p[0][None], 3, 0),
                   np.repeat(self.p[1][None], 3, 0),
                   np.array([0.5, 0.5, 0.7]))
        with self.assertRaises(ValueError):
            sample_coupled_batches(self.x0, self.t, p_later, 3, 2)

    def test_nonfinite_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                sample_target(np.array([bad, 0.0]), self.t, 3)
            with self.assertRaises(ValueError):
                sample_target(self.x0, (np.full((2, 2), bad), self.t[1], self.t[2]), 3)
            with self.assertRaises(ValueError):
                sample_target(self.x0, (self.t[0], np.array([0.0, bad]), self.t[2]), 3)

    def test_dimension_conflicts_rejected(self):
        with self.assertRaises(ValueError):
            sample_target(np.zeros(3), self.t, 3)  # x0 vs 2x2 A
        bad_A = (np.eye(3), self.t[1], self.t[2])
        with self.assertRaises(ValueError):
            sample_target(self.x0, bad_A, 3)
        with self.assertRaises(ValueError):
            sample_coupled_batches(np.zeros(3), self.t, self.p, 3, 2)
        # mixing 2D and 3D inside a step array
        A_mix = np.zeros((3, 3, 2))
        with self.assertRaises(ValueError):
            sample_target(self.x0, (A_mix, self.t[1], self.t[2]), 3)

    def test_dimension_bounds(self):
        t1, p1 = make_target_proposal(1)
        out = sample_target(np.zeros(1), t1, 2, n_traj=1, seed=0)
        self.assertEqual(out.shape[-1], 1)
        t16, p16 = make_target_proposal(16)
        res = sample_coupled_batches(np.zeros(16), t16, p16, 2, 1,
                                     n_traj=1, seed=0)
        self.assertEqual(res.trajectories.shape[-1], 16)
        t0 = (np.zeros((0, 0)), np.zeros(0), 0.5)
        with self.assertRaises(ValueError):
            sample_target(np.zeros(0), t0, 2)

    def test_bad_counts(self):
        for kwargs in (dict(n_steps=0), dict(n_steps=-1), dict(n_steps=2.5)):
            with self.assertRaises(ValueError):
                sample_target(self.x0, self.t, **kwargs)
        for L in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                sample_coupled_batches(self.x0, self.t, self.p, 4, L)
        with self.assertRaises(ValueError):
            sample_coupled_batches(self.x0, self.t, self.p, 4, 2, n_traj=0)
        with self.assertRaises(ValueError):
            analytic_moments(self.x0, self.t, 0)

    def test_step_array_length_mismatch(self):
        s2 = np.array([0.5, 0.2, 0.9])
        with self.assertRaises(ValueError):
            sample_target(self.x0, (self.t[0], self.t[1], s2), 2)


if __name__ == "__main__":
    unittest.main()
