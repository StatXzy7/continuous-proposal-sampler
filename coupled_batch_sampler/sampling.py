"""Target baseline and look-ahead maximal-coupling batch sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .kernels import AffineMean

ArrayLike = Any


@dataclass(frozen=True)
class CoupledBatchResult:
    trajectories: np.ndarray
    accept_lengths: List[np.ndarray]
    rejection_positions: List[np.ndarray]
    target_evaluation_count: int
    target_batch_count: int
    target_evaluations_per_chain: np.ndarray


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _variance(variance: Any) -> float:
    try:
        value = float(variance)
    except (TypeError, ValueError) as exc:
        raise TypeError("variance must be a scalar float") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("variance must be a strictly positive finite scalar")
    return value


def _initial_states(initial_states: ArrayLike, dim: int) -> np.ndarray:
    try:
        states = np.asarray(initial_states, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("initial_states must be a NumPy array-like value") from exc
    if states.ndim == 1:
        states = states[None, :]
    if states.ndim != 2 or states.shape[1] != dim:
        raise ValueError(f"initial_states must have shape (n_chains, {dim})")
    if not np.all(np.isfinite(states)):
        raise ValueError("initial_states must contain only finite values")
    return states


def _checked_mean(mean: AffineMean, states: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(mean(states), dtype=np.float64)
    if values.shape != states.shape:
        raise ValueError(f"{name} returned shape {values.shape}, expected {states.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} returned non-finite values")
    return values


def _seed_sequence(seed: Optional[int | np.random.SeedSequence]) -> np.random.SeedSequence:
    if seed is None:
        return np.random.SeedSequence()
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be None, an int, or numpy.random.SeedSequence")
    return np.random.SeedSequence(seed)


def sample_target_baseline(
    initial_states: ArrayLike,
    target_mean: AffineMean | tuple,
    variance: float,
    n_steps: int,
    seed: Optional[int | np.random.SeedSequence] = None,
) -> np.ndarray:
    """Draw sequentially from the target affine Gaussian kernel."""
    target = AffineMean.coerce(target_mean, "target_mean")
    dim = target.dim
    states = _initial_states(initial_states, dim)
    noise_variance = _variance(variance)
    steps = _positive_int(n_steps, "n_steps")
    rng = np.random.default_rng(_seed_sequence(seed))

    n_chains = states.shape[0]
    trajectories = np.empty((n_chains, steps + 1, dim), dtype=np.float64)
    trajectories[:, 0, :] = states
    scale = np.sqrt(noise_variance)

    current = states.copy()
    for step in range(steps):
        means = _checked_mean(target, current, "target_mean")
        current = means + scale * rng.standard_normal(current.shape)
        if not np.all(np.isfinite(current)):
            raise FloatingPointError(f"non-finite target state generated at step {step}")
        trajectories[:, step + 1, :] = current
    return trajectories


def _stable_log_ratios(
    candidates: np.ndarray,
    target_means: np.ndarray,
    proposal_means: np.ndarray,
    variance: float,
) -> np.ndarray:
    delta = (proposal_means - target_means).astype(np.longdouble)
    delta_sq = np.sum(delta * delta, axis=-1)
    candidate_offset = (candidates - target_means).astype(np.longdouble)
    numerator = np.sum(delta * (delta - np.longdouble(2.0) * candidate_offset), axis=-1)
    ratios = np.zeros_like(delta_sq)
    np.divide(
        numerator,
        np.longdouble(2.0) * np.longdouble(variance),
        out=ratios,
        where=delta_sq > 0.0,
    )
    return ratios


def _reflect_across_equidensity_hyperplanes(
    candidates: np.ndarray,
    target_means: np.ndarray,
    proposal_means: np.ndarray,
) -> np.ndarray:
    candidate_ld = candidates.astype(np.longdouble)
    target_ld = target_means.astype(np.longdouble)
    proposal_ld = proposal_means.astype(np.longdouble)
    delta = proposal_ld - target_ld
    delta_sq = np.sum(delta * delta, axis=-1, keepdims=True)
    midpoint = np.longdouble(0.5) * (target_ld + proposal_ld)
    offset = candidate_ld - midpoint
    projection_coefficient = np.sum(offset * delta, axis=-1, keepdims=True) / delta_sq
    reflected = candidate_ld - np.longdouble(2.0) * projection_coefficient * delta
    equal_means = np.broadcast_to(delta_sq == 0.0, reflected.shape)
    reflected = np.where(equal_means, candidate_ld, reflected)
    result = np.asarray(reflected, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Gaussian reflection produced a non-finite state")
    return result


def _groups(positions: np.ndarray, n_steps: int, lookahead: int) -> list[tuple[int, int, np.ndarray]]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for chain_index, position in enumerate(positions):
        if position < n_steps:
            block_length = min(lookahead, n_steps - int(position))
            grouped.setdefault((int(position), block_length), []).append(chain_index)
    return [
        (start, block_length, np.asarray(indices, dtype=np.int64))
        for (start, block_length), indices in sorted(grouped.items())
    ]


def sample_coupled_batches(
    initial_states: ArrayLike,
    target_mean: AffineMean | tuple,
    proposal_mean: AffineMean | tuple,
    variance: float,
    n_steps: int,
    lookahead: int,
    seed: Optional[int | np.random.SeedSequence] = None,
) -> CoupledBatchResult:
    """Sample independent chains using batched look-ahead maximal coupling."""
    target = AffineMean.coerce(target_mean, "target_mean")
    dim = target.dim
    proposal = AffineMean.coerce(proposal_mean, "proposal_mean", dim)
    states = _initial_states(initial_states, dim)
    noise_variance = _variance(variance)
    steps = _positive_int(n_steps, "n_steps")
    lookahead_steps = _positive_int(lookahead, "lookahead")
    rng = np.random.default_rng(_seed_sequence(seed))

    n_chains = states.shape[0]
    trajectories = np.empty((n_chains, steps + 1, dim), dtype=np.float64)
    trajectories[:, 0, :] = states
    current = states.copy()
    positions = np.zeros(n_chains, dtype=np.int64)
    accept_lengths: List[List[int]] = [[] for _ in range(n_chains)]
    rejection_positions: List[List[int]] = [[] for _ in range(n_chains)]
    evaluations_per_chain = np.zeros(n_chains, dtype=np.int64)
    target_evaluation_count = 0
    target_batch_count = 0
    scale = np.sqrt(noise_variance)

    while np.any(positions < steps):
        for start, block_length, active in _groups(positions, steps, lookahead_steps):
            group_size = active.shape[0]
            drafts = np.empty((group_size, block_length, dim), dtype=np.float64)
            proposal_means = np.empty((group_size, block_length, dim), dtype=np.float64)

            predecessors = current[active]
            for draft_index in range(block_length):
                means = _checked_mean(proposal_mean, predecessors, "proposal_mean")
                proposal_means[:, draft_index, :] = means
                noise = rng.standard_normal((group_size, dim), dtype=np.float64)
                drafts[:, draft_index, :] = means + scale * noise
                predecessors = drafts[:, draft_index, :]

            if not np.all(np.isfinite(drafts)):
                raise FloatingPointError(f"non-finite draft generated at step {start}")

            target_predecessors = np.concatenate(
                (current[active, None, :], drafts[:, :-1, :]), axis=1
            ).reshape(group_size * block_length, dim)
            target_means = _checked_mean(
                target_mean, target_predecessors, "target_mean"
            ).reshape(group_size, block_length, dim)

            target_evaluation_count += group_size * block_length
            evaluations_per_chain[active] += block_length
            target_batch_count += 1

            uniforms = 1.0 - rng.random(size=(group_size, block_length))
            log_ratios = _stable_log_ratios(drafts, target_means, proposal_means, noise_variance)
            log_uniforms = np.log(uniforms.astype(np.longdouble))
            rejected = (log_ratios < 0.0) & (log_uniforms > log_ratios)
            has_rejection = np.any(rejected, axis=1)
            first_rejection = np.argmax(rejected, axis=1)

            all_accepted = ~has_rejection
            all_accepted_indices = active[all_accepted]
            if all_accepted_indices.size:
                trajectories[
                    all_accepted_indices, start + 1 : start + block_length + 1, :
                ] = drafts[all_accepted]
                current[all_accepted_indices] = drafts[all_accepted, -1, :]
                positions[all_accepted_indices] = start + block_length
                for chain_index in all_accepted_indices:
                    accept_lengths[int(chain_index)].append(block_length)
                    rejection_positions[int(chain_index)].append(-1)

            for rejection_index in range(block_length):
                row_mask = has_rejection & (first_rejection == rejection_index)
                if not np.any(row_mask):
                    continue
                chain_indices = active[row_mask]
                corrected = _reflect_across_equidensity_hyperplanes(
                    drafts[row_mask, rejection_index, :],
                    target_means[row_mask, rejection_index, :],
                    proposal_means[row_mask, rejection_index, :],
                )
                if rejection_index > 0:
                    trajectories[
                        chain_indices, start + 1 : start + rejection_index + 1, :
                    ] = drafts[row_mask, :rejection_index, :]
                trajectories[chain_indices, start + rejection_index + 1, :] = corrected
                current[chain_indices] = corrected
                positions[chain_indices] = start + rejection_index + 1
                for chain_index in chain_indices:
                    accept_lengths[int(chain_index)].append(rejection_index)
                    rejection_positions[int(chain_index)].append(rejection_index)

    return CoupledBatchResult(
        trajectories=trajectories,
        accept_lengths=[np.asarray(values, dtype=np.int64) for values in accept_lengths],
        rejection_positions=[
            np.asarray(values, dtype=np.int64) for values in rejection_positions
        ],
        target_evaluation_count=target_evaluation_count,
        target_batch_count=target_batch_count,
        target_evaluations_per_chain=evaluations_per_chain,
    )
