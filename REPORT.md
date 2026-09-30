# Small Technical Report

## Algorithm

At a current state `x`, a target and proposal kernel share covariance `vI`:

```text
p(y | x) = N(mu(x), vI)
q(y | x) = N(nu(x), vI)
```

A look-ahead block first rolls proposal states sequentially because each proposal state depends on the previous draft. It then evaluates target means for the current state and the accepted-prefix draft predecessors in a single two-dimensional NumPy call. Candidate `y` is accepted with probability

```text
min(1, p(y | x) / q(y | x)).
```

For the first rejected candidate, the correction is a maximal-coupling reflection:

```text
y* = y - 2 * <y - (mu + nu)/2, nu - mu> / ||nu - mu||^2 * (nu - mu).
```

This reflects through the equidensity hyperplane between the two Gaussian means. If the means coincide, the candidate is already distributed from both kernels and no reflection is needed. After a rejection, every later proposal draft in that block is discarded and the next block starts from `y*`. There is no fallback to an independent target Gaussian draw.

## Why the marginal remains the target kernel

Reflection is an involution with Jacobian determinant one. It maps the portion of the proposal density rejected by the ratio test onto the residual target mass not supplied by common proposals. Combining unchanged accepted proposals with reflected rejected proposals therefore gives exactly the target Gaussian marginal at every transition. Since the next target mean is evaluated at the committed correction state, subsequent transitions retain the target Markov law.

## Batching and counters

Trajectories at the same absolute step with the same remaining block length are processed together. For a group of `K` trajectories and look-ahead length `L`, target means are evaluated from a flattened `(K*L, d)` array:

- target state evaluations: `K*L`;
- target batch calls: `1`.

Rejected drafts remain target-evaluated because their means were required during the density-ratio test, but they are not committed to trajectories. The two counters therefore cannot be inferred from submitted trajectory length alone.

## Numerical safeguards

- variance is required to be strictly positive and finite before generation;
- log density ratios use `np.longdouble` on Linux/x86_64;
- the implementation compares `log U` with the log ratio rather than exponentiating a tiny Gaussian density;
- reflection divides by a squared norm, avoiding a square-root division by zero;
- zero mean difference is handled explicitly.

## Automated verification

The test suite is run with:

```bash
python3 -m unittest discover -s tests -v
```

Prespecified distribution settings use 10,000 independent coupled chains, 45 burn-in steps, and one additional target transition. For the stationary affine target, the report checks:

- mean norm error against `b / (1-a)` with threshold `0.05`;
- covariance Frobenius error against `v/(1-a^2) I` with threshold `0.12`;
- empirical CDF error in four unit directions at probabilities 0.10, 0.50, and 0.90 with threshold `0.04`.

A separate 4,000-chain baseline run checks the direct target transition's analytic stationary mean and covariance. Remaining tests verify deterministic seeds, unchanged accepted states, exact reflected states, discarded later drafts, no final over-submission, dimensions 1 and 16, and distinct evaluation/batch counters.
