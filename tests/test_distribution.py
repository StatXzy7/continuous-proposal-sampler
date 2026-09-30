"""Distributional validation against analytic target moments.

Pre-registered experiment (no threshold chosen after looking at data):

* sample size N = 10000 independent trajectories;
* 3-D affine Gaussian chain, T = 6, lookahead L = 3;
* target and proposal mean functions differ (rejections genuinely occur);
* mean test: every coordinate |z| < 4.5 at t = 3 and t = 6;
* covariance test: |empirical - analytic| max entry < 4.5 * SE of that entry
  (SE estimated analytically from the Gaussian fourth moments) at both steps;
* directional empirical CDFs: Kolmogorov-Smirnov distance against the exact
  projected N(v.mu, v.Cv) distribution on 6 fixed directions (coordinate
  axes + mixed diagonals), D < 0.023 (asymptotic p ~ 2.5e-5, N=10000).
"""

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

N_TRAJ = 10000
Z_LIMIT = 4.5
KS_LIMIT = 0.023


def ks_distance_normal(samples, mean, var):
    x = np.sort(samples)
    n = x.size
    cdf_emp = np.arange(1, n + 1) / n
    cdf_theory = 0.5 * (1.0 + _erf((x - mean) / np.sqrt(2.0 * var)))
    d_plus = np.max(cdf_emp - cdf_theory)
    d_minus = np.max(cdf_theory - (np.arange(0, n) / n))
    return max(d_plus, d_minus)


def _erf(x):
    # erf via the normal CDF: use scipy-free Abramowitz-Stegun 7.1.26
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
               - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return sign * y


def build_chain(d=3, seed_offset=0):
    rng = np.random.default_rng(2024 + seed_offset)
    A = np.array([[0.55, 0.10, -0.05],
                  [-0.12, 0.40, 0.08],
                  [0.03, -0.07, 0.60]])
    b = np.array([0.15, -0.10, 0.20])
    Ap = 0.35 * np.eye(d) + 0.05 * np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    bp = np.zeros(d)
    s2 = 0.9
    x0 = np.array([0.5, -0.5, 0.25])
    return x0, (A, b, s2), (Ap, bp, s2)


DIRECTIONS_3D = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, -1.0, 1.0]),
    np.array([0.5, 0.5, 1.0]),
]


def assert_gaussian_chain(test_case, sampler_name, sampler_fn, x0, target,
                          proposal, T, L):
    means, covs = analytic_moments(x0, target, T)
    result = sampler_fn()
    # sampler_fn returns (N, T+1, d)
    for t in (T // 2, T):
        samples = result[:, t, :]
        mu, C = means[t], covs[t]

        # mean: z-score per coordinate
        se_mean = np.sqrt(np.diag(C) / N_TRAJ)
        z = np.abs((samples.mean(axis=0) - mu) / se_mean)
        test_case.assertTrue(
            np.all(z < Z_LIMIT),
            msg=f"{sampler_name} t={t} mean z={z}",
        )

        # covariance: compare with analytic Gaussian sampling SE.
        emp_cov = np.cov(samples, rowvar=False)
        # Var of sample covariance entries for Gaussian data:
        # Var(C_hat_ij) = (C_ij^2 + C_ii C_jj) / N
        var_cov = (C ** 2 + np.outer(np.diag(C), np.diag(C))) / N_TRAJ
        cov_z = np.abs(emp_cov - C) / np.sqrt(var_cov)
        test_case.assertTrue(
            np.all(cov_z < Z_LIMIT),
            msg=f"{sampler_name} t={t} cov z max={cov_z.max()}",
        )

        # directional KS tests against the exact projected Gaussian
        for v in DIRECTIONS_3D:
            vn = v / np.linalg.norm(v)
            proj = samples @ vn
            pm = float(vn @ mu)
            pv = float(vn @ C @ vn)
            d = ks_distance_normal(proj, pm, pv)
            test_case.assertLess(
                d, KS_LIMIT,
                msg=f"{sampler_name} t={t} dir {v} KS D={d:.4f}",
            )


class TestTargetDistribution(unittest.TestCase):
    T = 6
    L = 3

    def test_baseline_matches_target(self):
        x0, target, proposal = build_chain()

        def run():
            return sample_target(x0, target, self.T, n_traj=N_TRAJ, seed=77)

        assert_gaussian_chain(self, "baseline", run, x0, target, proposal,
                              self.T, self.L)

    def test_coupled_batches_matches_target(self):
        x0, target, proposal = build_chain()

        def run():
            res = sample_coupled_batches(x0, target, proposal, self.T, self.L,
                                         n_traj=N_TRAJ, seed=77)
            # sanity: the proposal differs, so the coupling really rejects
            rej_frac = float(np.mean(res.rejection_steps >= 0))
            self.assertGreater(rej_frac, 0.5,
                               msg=f"rejection fraction only {rej_frac}")
            return res.trajectories

        assert_gaussian_chain(self, "coupled-batches", run, x0, target,
                              proposal, self.T, self.L)

    def test_baseline_and_coupled_equivalent_at_marginals(self):
        # Different seeds, same marginals: compare means/covs only.
        x0, target, proposal = build_chain()
        base = sample_target(x0, target, self.T, n_traj=N_TRAJ, seed=1)
        res = sample_coupled_batches(x0, target, proposal, self.T, self.L,
                                     n_traj=N_TRAJ, seed=2)
        for t in (3, 6):
            diff = np.abs(base[:, t, :].mean(0) - res.trajectories[:, t, :].mean(0))
            se = 3.0 * np.sqrt(np.diag(analytic_moments(x0, target, self.T)[1][t])
                               / N_TRAJ)
            self.assertTrue(np.all(diff < se), msg=f"t={t} {diff} vs {se}")


class TestOneDimensionalDistribution(unittest.TestCase):
    """1-D chain, tiny-ish variance, step-dependent parameters."""

    N = 6000

    def test_1d_ks_with_step_dependent_kernel(self):
        T = 4
        A = np.array([0.7, 0.5, 0.3, 0.1])
        b = np.array([0.2, -0.1, 0.0, 0.4])
        s2 = np.array([0.04, 0.09, 0.16, 0.25])
        Ap = 0.4 * np.ones(T)
        bp = np.zeros(T)
        x0 = np.array([1.0])
        target = (A.reshape(T, 1, 1), b.reshape(T, 1), s2)
        proposal = (Ap.reshape(T, 1, 1), bp.reshape(T, 1), s2)

        means, covs = analytic_moments(x0, target, T)
        res = sample_coupled_batches(x0, target, proposal, T, 2,
                                     n_traj=self.N, seed=4242)
        for t in range(1, T + 1):
            samples = res.trajectories[:, t, 0]
            z = abs(samples.mean() - means[t, 0]) / np.sqrt(covs[t, 0, 0] / self.N)
            self.assertLess(z, 4.5, msg=f"t={t} z={z}")
            d = ks_distance_normal(samples, means[t, 0], covs[t, 0, 0])
            self.assertLess(d, 0.030, msg=f"t={t} KS D={d}")

        self.assertGreater(float(np.mean(res.rejection_steps >= 0)), 0.2)


class TestTinyVarianceNumerics(unittest.TestCase):
    def test_no_underflow_or_division_by_zero(self):
        rng = np.random.default_rng(9)
        for s2 in (1e-12, 1e-60, 1e-200, 1e-300):
            d = 2
            x0 = np.zeros(d)
            target = (np.eye(d), np.array([0.0, 1e-8]), s2)
            proposal = (np.eye(d), np.zeros(d), s2)
            res = sample_coupled_batches(x0, target, proposal, 8, 3,
                                         n_traj=50, seed=3)
            self.assertTrue(np.all(np.isfinite(res.trajectories)))
            # states remain on the physically meaningful scale; no inf/nan,
            # no absurd blow-up
            self.assertLess(np.abs(res.trajectories).max(),
                            1e-4 + 1e-8 * 10)

        # identical means at extreme variance: all accepted, never divides
        target0 = (np.eye(2), np.zeros(2), 1e-300)
        res = sample_coupled_batches(np.zeros(2), target0, target0, 4, 2,
                                     n_traj=10, seed=1)
        np.testing.assert_array_equal(res.rejection_steps, -1)
        self.assertTrue(np.all(np.isfinite(res.trajectories)))


if __name__ == "__main__":
    unittest.main()
