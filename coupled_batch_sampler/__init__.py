"""Batch candidate sampling with Gaussian reflection maximal coupling."""

from .kernels import AffineMean
from .sampling import CoupledBatchResult, sample_coupled_batches, sample_target_baseline

__all__ = [
    "AffineMean",
    "CoupledBatchResult",
    "sample_coupled_batches",
    "sample_target_baseline",
]
