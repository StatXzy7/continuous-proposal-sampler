"""Continuous-state batch candidate sampler with reflection maximal coupling.

Public API
----------
sample_target(x0, target, n_steps, n_traj=, seed=)
    Step-by-step target-kernel baseline, returns (n_traj, T+1, d) trajectories.

sample_coupled_batches(x0, target, proposal, n_steps, lookahead,
                       n_traj=, seed=)
    Batch-lookahead mode; returns a :class:`CoupledBatchResult`.

analytic_moments(x0, target, n_steps)
    Exact mean and covariance of the target chain at every step.

Kernel specification
--------------------
``target`` / ``proposal`` are tuples ``(A, b, sigma2)`` defining

    X_{t+1} = A_t X_t + b_t + N(0, sigma2_t I).

``A`` has shape (d, d) or (T, d, d); ``b`` shape (d,) or (T, d);
``sigma2`` a positive scalar or length-T array, shared exactly by both
kernels.  ``d`` must be in 1..16.
"""

from .kernels import AffineKernel, MAX_DIM, prepare_kernel, prepare_kernels
from .coupling import (
    CoupledBatchResult,
    analytic_moments,
    coupled_step,
    gaussian_log_ratio,
    reflect_across_bisector,
    sample_coupled_batches,
    sample_target,
)

__all__ = [
    "AffineKernel",
    "MAX_DIM",
    "prepare_kernel",
    "prepare_kernels",
    "CoupledBatchResult",
    "analytic_moments",
    "coupled_step",
    "gaussian_log_ratio",
    "reflect_across_bisector",
    "sample_coupled_batches",
    "sample_target",
]
