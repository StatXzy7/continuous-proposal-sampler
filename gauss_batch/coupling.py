"""Batch candidate sampling with a Gaussian reflection maximal coupling.

The sampler implements a batch/lookahead Metropolis-Hastings accept-reject
scheme between two affine Gaussian transition kernels that share their
(spherical) noise variance:

* draft ``L`` proposal states and compute the corresponding target means in a
  single batch;
* walk the drafts in order, submitting each accepted state unchanged;
* on the first rejection, produce the correction with a maximal coupling via
  reflection of the draft across the bisecting hyperplane of the two Gaussian
  means, submit that state, and discard every later draft;
* continue the lookahead from the correction.  A target-kernel draw is never
  substituted for the coupling correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import (
    AffineKernel,
    prepare_kernels,
    validate_x0,
)


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------

def gaussian_log_ratio(y, target_mean, proposal_mean, sigma2: float) -> float:
    r"""log N(y; m1, s2 I) - log N(y; m2, s2 I) for equal spherical variances.

    Computed as (||y-m2||^2 - ||y-m1||^2) / (2 s2) via
    2 (y - midpoint) . (m1 - m2) / (2 s2), avoiding the cancellation that
    overflows at tiny variances.  No density is ever materialised, so density
    underflow is impossible.
    """

    delta = np.asarray(target_mean, dtype=float) - np.asarray(proposal_mean, dtype=float)
    midpoint = 0.5 * (np.asarray(target_mean, dtype=float) + np.asarray(proposal_mean, dtype=float))
    numerator = np.dot(np.asarray(y, dtype=float) - midpoint, delta)
    # delta == 0 (identical means) is handled by callers via an explicit path;
    # guard the division anyway to guarantee no division by zero.
    return float(numerator / sigma2)


def reflect_across_bisector(y, target_mean, proposal_mean):
    """Reflect ``y`` across the hyperplane bisecting the two Gaussian means.

    With delta = m1 - m2, reflection through {z : z . delta = (m1+m2).delta/2}
    gives a point X with ||X - m1|| = ||y - m2|| and ||X - m2|| = ||y - m1||,
    which is the maximal-coupling residual map for equal spherical Gaussians.
    """

    y = np.asarray(y, dtype=float)
    m1 = np.asarray(target_mean, dtype=float)
    m2 = np.asarray(proposal_mean, dtype=float)
    delta = m1 - m2
    norm2 = float(np.dot(delta, delta))
    if norm2 == 0.0:
        # Coincident means: the hyperplane is undefined, but a rejection is
        # impossible in that case (the acceptance probability is exactly 1).
        return y.copy()
    return y - (2.0 * np.dot(y - 0.5 * (m1 + m2), delta) / norm2) * delta


def coupled_step(y, target_mean, proposal_mean, sigma2: float, rng):
    """One reflection-maximal-coupling transition.

    Returns ``(state, accepted)``.  On acceptance ``state is y`` (returned
    unchanged); on rejection the reflected correction is returned.
    """

    delta = np.asarray(target_mean, dtype=float) - np.asarray(proposal_mean, dtype=float)
    if float(np.dot(delta, delta)) == 0.0:
        return np.asarray(y, dtype=float).copy(), True

    log_alpha = gaussian_log_ratio(y, target_mean, proposal_mean, sigma2)
    if log_alpha >= 0.0:
        return np.asarray(y, dtype=float).copy(), True

    # log(u) for the accept/reject test; guard against u == 0.
    log_u = np.log(max(rng.random(), np.finfo(float).tiny))
    if log_u <= log_alpha:
        return np.asarray(y, dtype=float).copy(), True

    x = reflect_across_bisector(y, target_mean, proposal_mean)
    return x, False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoupledBatchResult:
    trajectories: np.ndarray          # (n_traj, T+1, d)
    accept_run_lengths: list          # list[int] per trajectory (drafts accepted)
    rejection_steps: np.ndarray       # (n_traj,) int, -1 when never rejected
    n_target_evaluations: int         # target mean evaluations across all batches
    n_batches: int                    # number of batch target-mean evaluations
    n_rejections: int                 # total rejections (== coupling corrections)
    target_evaluations_per_traj: np.ndarray
    batches_per_traj: np.ndarray


# ---------------------------------------------------------------------------
# Single-trajectory runner (also used by tests with a scripted rng)
# ---------------------------------------------------------------------------

def _run_one_coupled(x0, target: AffineKernel, proposal: AffineKernel,
                     n_steps: int, lookahead: int, rng):
    """Run one trajectory in the batch lookahead mode."""

    d = target.dim
    traj = np.empty((n_steps + 1, d), dtype=float)
    traj[0] = x0

    state = np.array(x0, dtype=float)
    t = 0
    n_eval = 0
    n_batches = 0
    accept_run_lengths = []
    rejection_step = -1
    n_rej = 0

    while t < n_steps:
        size = min(lookahead, n_steps - t)

        # 1. Draft proposal states in one noise batch.  Proposal means depend
        #    on the previous draft, so they are unrolled along the path.
        step_idx = np.arange(t, t + size)
        scales = np.sqrt(proposal.sigma2[step_idx])[:, None]
        noises = scales * rng.standard_normal((size, d))
        drafts = np.empty((size, d), dtype=float)
        proposal_means = np.empty((size, d), dtype=float)
        anchor = state
        for j in range(size):
            proposal_means[j] = proposal.mean_at(t + j, anchor)
            drafts[j] = proposal_means[j] + noises[j]
            anchor = drafts[j]

        # 2. Batch target-mean evaluation along the drafted path: the anchors
        #    are the current state followed by the first size-1 drafts.
        anchors = np.concatenate((state[None, :], drafts[:-1]), axis=0)
        target_means = (
            np.einsum("tij,tj->ti", target.A[step_idx], anchors) + target.b[step_idx]
        )
        n_eval += size
        n_batches += 1

        # 3. Accept in order; the first rejection triggers the correction.
        accepted_here = 0
        j = 0
        while j < size:
            new_state, accepted = coupled_step(
                drafts[j], target_means[j], proposal_means[j],
                target.sigma2[t + j], rng,
            )
            if accepted:
                state = drafts[j]
                traj[t + j + 1] = state
                accepted_here += 1
                j += 1
            else:
                # Reflection correction; all later drafts are discarded.
                state = new_state
                traj[t + j + 1] = state
                if rejection_step == -1:
                    rejection_step = t + j
                n_rej += 1
                accept_run_lengths.append(accepted_here)
                t = t + j + 1
                break
        else:
            accept_run_lengths.append(accepted_here)
            t += size

    return {
        "trajectory": traj,
        "accept_run_lengths": accept_run_lengths,
        "rejection_step": rejection_step,
        "n_target_evaluations": n_eval,
        "n_batches": n_batches,
        "n_rejections": n_rej,
    }


def _validate_run_args(x0, target, proposal, n_steps, lookahead, n_traj):
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if not isinstance(lookahead, (int, np.integer)) or lookahead <= 0:
        raise ValueError("lookahead must be a positive integer")
    if not isinstance(n_traj, (int, np.integer)) or n_traj <= 0:
        raise ValueError("n_traj must be a positive integer")

    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1-D vector")
    d = x0.shape[0]
    x0 = validate_x0(x0, d)
    target_k, proposal_k = prepare_kernels(target, proposal, d, n_steps)
    return x0, target_k, proposal_k


def sample_target(x0, target, n_steps: int, *, n_traj: int = 1, seed=None):
    """Baseline: draw trajectories step by step directly from the target kernel.

    Returns an array of shape ``(n_traj, n_steps+1, d)``.
    """

    if np.asarray(x0, dtype=float).ndim != 1:
        raise ValueError("x0 must be a 1-D vector")
    d = np.asarray(x0, dtype=float).shape[0]
    x0v = validate_x0(x0, d)
    from .kernels import prepare_kernel

    target_k = prepare_kernel(target[0], target[1], target[2], d, n_steps,
                              name="target")
    if not isinstance(n_traj, (int, np.integer)) or n_traj <= 0:
        raise ValueError("n_traj must be a positive integer")
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")

    root = np.random.default_rng(seed)
    streams = root.spawn(n_traj)
    out = np.empty((n_traj, n_steps + 1, d), dtype=float)
    for i, rng in enumerate(streams):
        out[i, 0] = x0v
        state = x0v.copy()
        for t in range(n_steps):
            noise = np.sqrt(target_k.sigma2[t]) * rng.standard_normal(d)
            state = target_k.mean_at(t, state) + noise
            out[i, t + 1] = state
    return out


def sample_coupled_batches(x0, target, proposal, n_steps: int, lookahead: int,
                           *, n_traj: int = 1, seed=None):
    """Batch lookahead sampler with reflection maximal-coupling corrections."""

    x0v, target_k, proposal_k = _validate_run_args(
        x0, target, proposal, n_steps, lookahead, n_traj
    )

    root = np.random.default_rng(seed)
    streams = root.spawn(n_traj)

    trajectories = np.empty((n_traj, n_steps + 1, x0v.shape[0]), dtype=float)
    run_lengths = []
    rejection_steps = np.empty(n_traj, dtype=np.int64)
    evals = np.empty(n_traj, dtype=np.int64)
    batches = np.empty(n_traj, dtype=np.int64)
    total_rej = 0

    for i, rng in enumerate(streams):
        res = _run_one_coupled(x0v, target_k, proposal_k,
                               n_steps, lookahead, rng)
        trajectories[i] = res["trajectory"]
        run_lengths.append(res["accept_run_lengths"])
        rejection_steps[i] = res["rejection_step"]
        evals[i] = res["n_target_evaluations"]
        batches[i] = res["n_batches"]
        total_rej += res["n_rejections"]

    return CoupledBatchResult(
        trajectories=trajectories,
        accept_run_lengths=run_lengths,
        rejection_steps=rejection_steps,
        n_target_evaluations=int(evals.sum()),
        n_batches=int(batches.sum()),
        n_rejections=int(total_rej),
        target_evaluations_per_traj=evals,
        batches_per_traj=batches,
    )


# ---------------------------------------------------------------------------
# Analytic moments of the target chain
# ---------------------------------------------------------------------------

def analytic_moments(x0, target, n_steps: int):
    """Mean and covariance of X_t under the target affine Gaussian chain.

    Returns ``(means, covs)`` with shapes ``(n_steps+1, d)`` and
    ``(n_steps+1, d, d)``; the initial state is deterministic (cov 0).
    """

    if np.asarray(x0, dtype=float).ndim != 1:
        raise ValueError("x0 must be a 1-D vector")
    d = np.asarray(x0, dtype=float).shape[0]
    x0v = validate_x0(x0, d)
    from .kernels import prepare_kernel

    kernel = prepare_kernel(target[0], target[1], target[2], d, n_steps,
                            name="target")

    means = np.empty((n_steps + 1, d), dtype=float)
    covs = np.zeros((n_steps + 1, d, d), dtype=float)
    means[0] = x0v
    eye = np.eye(d)
    for t in range(n_steps):
        means[t + 1] = kernel.mean_at(t, means[t])
        covs[t + 1] = kernel.A[t] @ covs[t] @ kernel.A[t].T + kernel.sigma2[t] * eye
    return means, covs
