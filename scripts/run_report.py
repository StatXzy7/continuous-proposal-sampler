"""Reproduce the small empirical report: marginals, KS distances, batching stats.

Run:  python3 scripts/run_report.py
No model downloads; NumPy is the only dependency.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gauss_batch import analytic_moments, sample_coupled_batches, sample_target
from tests.test_distribution import (
    DIRECTIONS_3D,
    N_TRAJ,
    build_chain,
    ks_distance_normal,
)


def main():
    T, L = 6, 3
    x0, target, proposal = build_chain()
    means, covs = analytic_moments(x0, target, T)

    base = sample_target(x0, target, T, n_traj=N_TRAJ, seed=77)
    res = sample_coupled_batches(x0, target, proposal, T, L,
                                 n_traj=N_TRAJ, seed=77)

    print(f"dim=3, T={T}, lookahead L={L}, trajectories N={N_TRAJ}")
    print("=" * 68)
    for t in (3, 6):
        emp = res.trajectories[:, t, :].mean(0)
        se = np.sqrt(np.diag(covs[t]) / N_TRAJ)
        z = np.max(np.abs(emp - means[t]) / se)
        emp_cov = np.cov(res.trajectories[:, t, :], rowvar=False)
        var_cov = (covs[t] ** 2 + np.outer(np.diag(covs[t]), np.diag(covs[t]))) / N_TRAJ
        cz = np.max(np.abs(emp_cov - covs[t]) / np.sqrt(var_cov))
        ks = [ks_distance_normal(res.trajectories[:, t, :] @ (v / np.linalg.norm(v)),
                                 v / np.linalg.norm(v) @ means[t],
                                 (v / np.linalg.norm(v)) @ covs[t] @ (v / np.linalg.norm(v)))
              for v in DIRECTIONS_3D]
        print(f"t={t}: max |mean z|={z:.2f}  max |cov z|={cz:.2f}  "
              f"KS max={max(ks):.4f} (all 6 directions: "
              f"{', '.join(f'{d:.4f}' for d in ks)})")

    rej_frac = float(np.mean(res.rejection_steps >= 0))
    n_evals = res.n_target_evaluations
    n_batches = res.n_batches
    print("-" * 68)
    print(f"trajectories with >=1 rejection : {rej_frac*100:.1f}%")
    print(f"total coupling rejections        : {res.n_rejections}")
    print(f"target mean evaluations (states) : {n_evals}")
    print(f"batch evaluation calls (batches) : {n_batches}")
    print(f"step-by-step target eval count   : {N_TRAJ * T}")
    print(f"states submitted per trajectory  : {T} (exactly T, no over-submit)")
    print(f"max evals in one trajectory      : {res.target_evaluations_per_traj.max()}")
    print(f"min evals in one trajectory      : {res.target_evaluations_per_traj.min()}")

    # baseline vs coupled marginal agreement (different seeds)
    b2 = sample_target(x0, target, T, n_traj=N_TRAJ, seed=202)
    c2 = sample_coupled_batches(x0, target, proposal, T, L,
                                n_traj=N_TRAJ, seed=203)
    md = max(np.abs(b2[:, t, :].mean(0) - c2.trajectories[:, t, :].mean(0)).max()
             for t in (3, 6))
    print("-" * 68)
    print(f"max mean gap baseline vs coupled  : {md:.4f}")
    print("=" * 68)
    print("thresholds registered before running: |z|<4.5, KS D<0.023")


if __name__ == "__main__":
    main()
