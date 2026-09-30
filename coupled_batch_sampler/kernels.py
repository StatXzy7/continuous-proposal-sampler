"""Affine Gaussian transition components."""

from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np

ArrayLike = Any
MeanLike = Union["AffineMean", Tuple[ArrayLike], Tuple[ArrayLike, ArrayLike]]


def _finite_float_array(value: ArrayLike, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric NumPy array-like value") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


class AffineMean:
    """Mean function ``x -> A @ x + b`` for one- to sixteen-dimensional states."""

    def __init__(self, a: ArrayLike, b: ArrayLike = 0.0) -> None:
        matrix = _finite_float_array(a, "affine matrix")
        if matrix.ndim == 0:
            dim = 1
            matrix = np.asarray([[float(matrix)]], dtype=np.float64)
        elif matrix.ndim == 1:
            dim = matrix.shape[0]
            matrix = np.diag(matrix)
        elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            dim = matrix.shape[0]
        else:
            raise ValueError("affine matrix must be a square matrix or a 1-D diagonal")
        if not 1 <= dim <= 16:
            raise ValueError("state dimension must be between 1 and 16")

        offset = _finite_float_array(b, "affine offset")
        if offset.ndim == 0:
            offset = np.full(dim, float(offset), dtype=np.float64)
        elif offset.shape != (dim,):
            raise ValueError(f"affine offset must have shape ({dim},)")

        self.a = matrix
        self.b = offset
        self.dim = dim

    @classmethod
    def coerce(cls, value: MeanLike, name: str, dim: int | None = None) -> "AffineMean":
        if isinstance(value, AffineMean):
            mean = value
        elif isinstance(value, tuple) and 1 <= len(value) <= 2:
            mean = cls(value[0], value[1] if len(value) == 2 else 0.0)
        else:
            raise TypeError(f"{name} must be an AffineMean or (A, b) tuple")
        if dim is not None and mean.dim != dim:
            raise ValueError(f"{name} dimension {mean.dim} does not match dimension {dim}")
        return mean

    def __call__(self, x: ArrayLike) -> np.ndarray:
        states = np.asarray(x, dtype=np.float64)
        return states @ self.a.T + self.b
