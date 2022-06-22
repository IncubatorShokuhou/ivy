# global
import cupy as cp
from typing import Tuple
from collections import namedtuple
from packaging import version


def unique_all(x: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    UniqueAll = namedtuple(
        typename="unique_all",
        field_names=["values", "indices", "inverse_indices", "counts"],
    )

    values, indices, inverse_indices, counts = cp.unique(
        x, return_index=True, return_counts=True, return_inverse=True
    )
    nan_count = cp.sum(cp.isnan(x)).item()

    if (nan_count > 1) & (cp.sum(cp.isnan(values)).item() == 1):
        counts[cp.where(cp.isnan(values))[0]] = 1
        counts = cp.append(
            counts, cp.full(fill_value=1, shape=(nan_count - 1,))
        ).astype("int32")

        values = cp.append(
            values, cp.full(fill_value=cp.nan, shape=(nan_count - 1,)), axis=0
        )

        nan_idx = cp.where(cp.isnan(x.flatten()))[0]

        indices = cp.concatenate((indices[:-1], nan_idx), axis=0).astype("int32")
    else:
        pass

    return UniqueAll(
        values.astype(x.dtype),
        indices,
        cp.reshape(inverse_indices, x.shape).astype("int32"),
        counts,
    )


def unique_inverse(x: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    out = namedtuple("unique_inverse", ["values", "inverse_indices"])
    values, inverse_indices = cp.unique(x, return_inverse=True)
    nan_count = cp.count_nonzero(cp.isnan(x))
    if nan_count > 1:
        values = cp.append(values, cp.full(nan_count - 1, cp.nan)).astype(x.dtype)
    inverse_indices = inverse_indices.reshape(x.shape)
    return out(values, inverse_indices)


def unique_values(x: cp.ndarray) -> cp.ndarray:
    nan_count = cp.count_nonzero(cp.isnan(x))
    # cupy.count_nonzero() returns zero-dimensional array to reduce CPU-GPU synchronization.
    if nan_count.item() > 1: 
        unique = cp.append(
            cp.unique(x.flatten()), cp.full(nan_count - 1, cp.nan)
        ).astype(x.dtype)
    else:
        unique = cp.unique(x.flatten()).astype(x.dtype)
    return unique


def unique_counts(x: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    v, c = cp.unique(x, return_counts=True)
    nan_count = cp.count_nonzero(cp.isnan(x))
    if nan_count > 1:
        nan_idx = cp.where(cp.isnan(v))
        c[nan_idx] = 1
        v = cp.append(v, cp.full(nan_count - 1, cp.nan)).astype(x.dtype)
        c = cp.append(c, cp.full(nan_count - 1, 1)).astype("int32")
    uc = namedtuple("uc", ["values", "counts"])
    return uc(v, c)
