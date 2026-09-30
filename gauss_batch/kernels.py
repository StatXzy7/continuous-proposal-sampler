"""Configurable affine Gaussian transition kernels (dimensions 1..16).

A kernel is the transition

    X_{t+1} = A_t X_t + b_t + noise,   noise ~ N(0, sigma2_t * I_d).

The target kernel and the proposal kernel may have different mean functions
(A, b) but must share the same strictly positive spherical variance at every
step.  Parameters may be scalars (held constant for all steps) or arrays of
length ``T`` holding the value for each transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_DIM = 16


def _finite_array(name, value, dtype=float):
    arr = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


@dataclass(frozen=True)
class AffineKernel:
    """An affine Gaussian kernel with parameters aligned to each step."""

    A: np.ndarray  # (T, d, d)
    b: np.ndarray  # (T, d)
    sigma2: np.ndarray  # (T,)

    @property
    def n_steps(self) -> int:
        return self.sigma2.shape[0]

    @property
    def dim(self) -> int:
        return self.A.shape[1]

    def means(self, x: np.ndarray) -> np.ndarray:
        """Mean A_t x + b_t for states ``x`` of shape (..., d)."""

        x = np.asarray(x, dtype=float)
        return np.einsum("tij,...j->...ti", self.A, x) + self.b

    def mean_at(self, t: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.A[t] @ x + self.b[t]


def _expand_step_array(name, value, steps, expected_shape):
    arr = _finite_array(name, value)
    expected = (steps,) + tuple(expected_shape)
    if arr.shape == tuple(expected):
        return np.array(arr, copy=True)
    # Broadcast a single-step specification across all steps.
    single_shape = (1,) + tuple(expected_shape)
    try:
        tiled = np.broadcast_to(arr, np.empty(single_shape).shape)
    except ValueError:
        raise ValueError(
            f"{name} has shape {arr.shape}, expected {tuple(expected_shape)} "
            f"or a single-step shape {single_shape}"
        )
    tiled = np.broadcast_to(tiled, (steps,) + tuple(expected_shape))
    out = np.array(tiled, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def prepare_kernel(A, b, sigma2, dim: int, steps: int, *, name: str = "kernel") -> AffineKernel:
    """Validate and align one kernel's parameters over ``steps`` transitions."""

    if not isinstance(steps, (int, np.integer)) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(dim, (int, np.integer)) or not (1 <= dim <= MAX_DIM):
        raise ValueError(f"dimension must be an integer in [1, {MAX_DIM}]")

    A_arr = _expand_step_array(f"{name} A", A, steps, (dim, dim))
    b_arr = _expand_step_array(f"{name} b", b, steps, (dim,))
    s_arr = _finite_array(f"{name} sigma2", sigma2, dtype=float).reshape(-1)

    if s_arr.shape[0] == 1:
        s_arr = np.repeat(s_arr, steps)
    elif s_arr.shape[0] != steps:
        raise ValueError(
            f"{name} sigma2 must be a scalar or a length-{steps} sequence"
        )

    if not np.all(np.isfinite(s_arr)):
        raise ValueError(f"{name} sigma2 must be finite")
    if np.any(s_arr <= 0.0):
        raise ValueError(f"{name} sigma2 must be strictly positive at every step")

    return AffineKernel(A=A_arr, b=b_arr, sigma2=s_arr)


def prepare_kernels(target, proposal, dim: int, steps: int):
    """Validate both kernels and enforce the shared-noise-variance contract."""

    k_t = prepare_kernel(target[0], target[1], target[2], dim, steps, name="target")
    k_p = prepare_kernel(proposal[0], proposal[1], proposal[2], dim, steps, name="proposal")
    if not np.array_equal(k_t.sigma2, k_p.sigma2):
        raise ValueError(
            "target and proposal kernels must share the exact noise variance "
            "at every step"
        )
    return k_t, k_p


def validate_x0(x0, dim: int) -> np.ndarray:
    arr = _finite_array("x0", x0).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"x0 must have length {dim}, got {arr.shape[0]}")
    return arr
