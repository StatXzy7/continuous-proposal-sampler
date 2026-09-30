# Continuous Proposal Sampler

A NumPy-only Linux/CPU implementation of batched look-ahead candidate sampling for continuous affine Gaussian transition kernels. It supports state dimensions 1 through 16 and multiple independent trajectories.

## Install

```bash
python3 -m pip install -r requirements.txt
```

NumPy is the only significant third-party dependency. No models or external datasets are downloaded.

## Model

For state `x` in `R^d`, both transition kernels use the same strictly positive spherical variance:

```text
q(. | x) = N(a_q x + b_q, variance * I)
p(. | x) = N(a_p x + b_p, variance * I)
```

The target and proposal mean functions may differ. Construct them with `AffineMean`:

```python
import numpy as np
from coupled_batch_sampler import AffineMean, sample_coupled_batches, sample_target_baseline

target = AffineMean(np.full(3, 0.72), [0.15, -0.20, 0.30])
proposal = AffineMean(np.full(3, 0.72), [0.70, 0.20, -0.50])
initial = np.zeros((1000, 3))

baseline = sample_target_baseline(
    initial_states=initial,
    target_mean=target,
    variance=0.25,
    n_steps=40,
    seed=123,
)

result = sample_coupled_batches(
    initial_states=initial,
    target_mean=target,
    proposal_mean=proposal,
    variance=0.25,
    n_steps=40,
    lookahead=7,
    seed=123,
)
```

A one-dimensional state may be supplied as shape `(n_chains, 1)` or `(1,)` for a single chain. `AffineMean(0.5)` denotes the scalar 1x1 matrix `[[0.5]]`.

## Look-ahead coupling mode

For every chain, one block generates up to `lookahead` proposal states. Their predecessors are then passed to one batched target mean evaluation. Candidates are checked in order:

1. Accepted proposal states are submitted unchanged.
2. At the first rejection, later drafts in that block are discarded.
3. The rejected proposal is reflected across the equidensity hyperplane of the two Gaussian densities to produce the correction state.
4. Sampling resumes from that correction state.

The implementation never replaces a rejection with an independent target-kernel draw.

The final block is clipped to the remaining number of steps. If a rejection happens there, only the required corrected state is submitted; no extra states beyond `n_steps` are committed.

## Returned fields

`sample_coupled_batches` returns `CoupledBatchResult`:

- `trajectories`: shape `(n_chains, n_steps + 1, d)`, including the supplied initial states.
- `accept_lengths`: per-chain arrays; each value is the number of accepted candidates before the block's first rejection.
- `rejection_positions`: per-chain arrays; zero-based local rejection position, or `-1` when the block was fully accepted.
- `target_evaluation_count`: number of rows for which the target mean was evaluated, including discarded drafts after a first rejection.
- `target_batch_count`: number of target mean function calls, which can evaluate many rows per call.
- `target_evaluations_per_chain`: target row-evaluation count attributed to each independent trajectory.

State evaluations and function calls are deliberately separate counters. For example, one call with 25 rows contributes one batch but 25 state evaluations.

## Validation

Invalid configurations are rejected before random numbers are generated or means are evaluated. This includes non-positive or non-finite variance, non-finite initial states or affine parameters, dimensions outside 1-16, target/proposal dimension mismatch, and non-positive step or look-ahead counts.

Acceptance uses log probabilities in extended precision. This avoids density underflow and division by zero at very small variance; Gaussian reflection also uses squared norm calculations without taking a square root.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests check seed reproduction, exact accepted states, reflection correction, draft discard behavior, final-block submission limits, the two distinct counters, invalid parameters, tiny variance, dimensions 1 and 16, baseline moments, coupled stationary mean/covariance, and directional empirical CDFs against analytic stationary Gaussian predictions.
