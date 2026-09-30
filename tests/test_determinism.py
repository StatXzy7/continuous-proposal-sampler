"""Seed reproducibility, accepted/unaccepted states, reflection geometry."""

import os
import sys

import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gauss_batch import (
    coupled_step,
    reflect_across_bisector,
    sample_coupled_batches,
    sample_target,
)
from gauss_batch.coupling import _run_one_coupled
from gauss_batch.kernels import prepare_kernel, prepare_kernels


class ScriptedRNG:
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


class TestReflectionGeometry(unittest.TestCase):
    def test_distance_swap_and_midpoint_hyperplane(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            d = int(rng.integers(1, 8))
            m1 = rng.normal(size=d)
            m2 = rng.normal(size=d)
            y = rng.normal(size=d)
            x = reflect_across_bisector(y, m1, m2)
            np.testing.assert_allclose(np.linalg.norm(x - m1),
                                       np.linalg.norm(y - m2), atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(x - m2),
                                       np.linalg.norm(y - m1), atol=1e-10)
            delta = m1 - m2
            # reflection preserves projection onto the bisecting hyperplane
            hp = 0.5 * (m1 + m2)
            np.testing.assert_allclose(np.dot(x - hp, delta),
                                       -np.dot(y - hp, delta), atol=1e-10)
            # reflection is an involution
            np.testing.assert_allclose(reflect_across_bisector(x, m1, m2),
                                       y, atol=1e-10)

    def test_reflection_midpoint_fixed_in_1d(self):
        m1, m2 = 3.0, 1.0
        x = reflect_across_bisector(np.array([2.0]),
                                    np.array([m1]), np.array([m2]))
        np.testing.assert_allclose(x, [2.0], atol=1e-12)
        x = reflect_across_bisector(np.array([5.0]),
                                    np.array([m1]), np.array([m2]))
        np.testing.assert_allclose(x, [-1.0], atol=1e-12)

    def test_identical_means_always_accepted(self):
        rng = np.random.default_rng(1)
        m = np.array([1.0, -2.0])
        y = m + 1e-12
        state, accepted = coupled_step(y, m, m.copy(), 1e-300, rng)
        self.assertTrue(accepted)
        np.testing.assert_array_equal(state, y)


class TestDeterministicPath(unittest.TestCase):
    """1-D, sigma2=1: m1=1, m2=0; acceptance region y > 0.5, reflection x=1-y."""

    def setUp(self):
        self.d = 1
        self.steps = 2
        # Constant means: target mean 1, proposal mean 0, unit variance,
        # so acceptance region is y > 0.5 and reflection is x = 1 - y.
        self.kt = prepare_kernel(np.zeros((1, 1, 1)), np.ones((1, 1)),
                                 np.ones(1), 1, self.steps, name="target")
        self.kp = prepare_kernel(np.zeros((1, 1, 1)), np.zeros((1, 1)),
                                 np.ones(1), 1, self.steps, name="proposal")

    def test_rejection_gives_reflected_state_not_target_draw(self):
        # step 0: draft y=-2, alpha=e^-2.5~=0.082, u=0.5 -> reject; x=3.
        # step 1: fresh batch, draft y=1, log-ratio=0.5 -> accepted.
        # first batch draws both draft noises at once; second batch one noise
        rng = ScriptedRNG(normals=[np.array([[-2.0], [42.0]]),
                                   np.array([[1.0]])],
                          uniforms=[0.5, 0.9])
        res = _run_one_coupled(np.zeros(1), self.kt, self.kp,
                               self.steps, 2, rng)
        traj = res["trajectory"]
        np.testing.assert_allclose(traj[0], [0.0])
        np.testing.assert_allclose(traj[1], [3.0])  # reflection: 1 + 0 - (-2)
        np.testing.assert_allclose(traj[2], [1.0])   # accepted draft unchanged
        self.assertEqual(res["rejection_step"], 0)
        self.assertEqual(res["n_rejections"], 1)
        self.assertEqual(res["n_batches"], 2)
        self.assertEqual(res["n_target_evaluations"], 3)
        self.assertEqual(res["accept_run_lengths"], [0, 1])

    def test_accepted_state_submitted_unchanged(self):
        # y=2 is in the acceptance region (log-ratio 1.5); submitted verbatim.
        rng = ScriptedRNG(normals=[np.array([[2.0], [1.0]])],
                          uniforms=[0.9, 0.9])
        res = _run_one_coupled(np.zeros(1), self.kt, self.kp,
                               self.steps, 2, rng)
        np.testing.assert_allclose(res["trajectory"][1], [2.0])
        np.testing.assert_allclose(res["trajectory"][2], [1.0])
        self.assertEqual(res["rejection_step"], -1)
        self.assertEqual(res["accept_run_lengths"], [2])

    def test_later_drafts_discarded_after_rejection(self):
        # The noise queued for the discarded second draft must never surface:
        # rejection at step 0 (y=-2, x=3), then a fresh batch; the queued
        # discarded draft noise 50.0 is never consumed.
        rng = ScriptedRNG(normals=[np.array([[-2.0], [50.0]]),
                                   np.array([[1.0]])],
                          uniforms=[0.5, 0.9])
        res = _run_one_coupled(np.zeros(1), self.kt, self.kp,
                               self.steps, 2, rng)
        np.testing.assert_allclose(res["trajectory"][1], [3.0])
        np.testing.assert_allclose(res["trajectory"][2], [1.0])
        self.assertEqual(res["n_batches"], 2)
        self.assertEqual(res["n_target_evaluations"], 3)  # 2 wasted + 1 used

    def test_end_to_end_same_seed_matches_baseline_when_kernels_equal(self):
        x0 = np.array([0.3, -0.8])
        A = 0.7 * np.eye(2)
        b = np.array([0.1, 0.4])
        s2 = 0.37
        target = (A, b, s2)
        base = sample_target(x0, target, 6, n_traj=3, seed=123)
        res = sample_coupled_batches(x0, target, target, 6, 2,
                                     n_traj=3, seed=123)
        np.testing.assert_allclose(res.trajectories, base, atol=1e-14)
        np.testing.assert_array_equal(res.rejection_steps, -1)
        self.assertEqual(res.n_rejections, 0)


class TestSeedReproducibility(unittest.TestCase):
    def setUp(self):
        d = 3
        self.x0 = np.linspace(-1, 1, d)
        self.target = (0.6 * np.eye(d), np.full(d, 0.2), 0.8)
        self.proposal = (0.4 * np.eye(d), np.zeros(d), 0.8)

    def test_same_seed_identical(self):
        a = sample_coupled_batches(self.x0, self.target, self.proposal,
                                   10, 3, n_traj=5, seed=99)
        b = sample_coupled_batches(self.x0, self.target, self.proposal,
                                   10, 3, n_traj=5, seed=99)
        np.testing.assert_array_equal(a.trajectories, b.trajectories)
        self.assertEqual(a.n_batches, b.n_batches)
        self.assertEqual(a.n_target_evaluations, b.n_target_evaluations)
        np.testing.assert_array_equal(a.rejection_steps, b.rejection_steps)
        base_a = sample_target(self.x0, self.target, 10, n_traj=5, seed=99)
        base_b = sample_target(self.x0, self.target, 10, n_traj=5, seed=99)
        np.testing.assert_array_equal(base_a, base_b)

    def test_different_seeds_differ(self):
        a = sample_coupled_batches(self.x0, self.target, self.proposal,
                                   12, 3, n_traj=4, seed=1)
        b = sample_coupled_batches(self.x0, self.target, self.proposal,
                                   12, 3, n_traj=4, seed=2)
        self.assertFalse(np.array_equal(a.trajectories, b.trajectories))

    def test_independent_trajectories_within_run(self):
        res = sample_coupled_batches(self.x0, self.target, self.proposal,
                                     8, 2, n_traj=20, seed=5)
        # trajectories must not be shifted copies of one another
        self.assertFalse(np.allclose(res.trajectories[0], res.trajectories[1]))


if __name__ == "__main__":
    unittest.main()
