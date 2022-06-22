# global
import cupy as cp
from typing import Tuple, Union, Optional

# local
import ivy

# Array API Standard #
# -------------------#


def min(
    x: cp.ndarray, axis: Union[int, Tuple[int]] = None, keepdims: bool = False
) -> cp.ndarray:
    return cp.asarray(cp.amin(a=x, axis=axis, keepdims=keepdims))


def max(
    x: cp.ndarray, axis: Union[int, Tuple[int]] = None, keepdims: bool = False
) -> cp.ndarray:
    return cp.asarray(cp.amax(a=x, axis=axis, keepdims=keepdims))


def var(
    x: cp.ndarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> cp.ndarray:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return cp.asarray(cp.var(x, axis=axis, keepdims=keepdims))


def sum(
    x: cp.ndarray,
    *,
    axis: Union[int, Tuple[int]] = None,
    dtype: cp.dtype = None,
    keepdims: bool = False,
) -> cp.ndarray:

    if dtype is None and cp.issubdtype(x.dtype, cp.integer):
        if cp.issubdtype(x.dtype, cp.signedinteger) and x.dtype in [
            cp.int8,
            cp.int16,
            cp.int32,
        ]:
            dtype = cp.int32
        elif cp.issubdtype(x.dtype, cp.unsignedinteger) and x.dtype in [
            cp.uint8,
            cp.uint16,
            cp.uint32,
        ]:
            dtype = cp.uint32
        elif x.dtype == cp.int64:
            dtype = cp.int64
        else:
            dtype = cp.uint64
    dtype = ivy.as_native_dtype(dtype)
    return cp.asarray(cp.sum(a=x, axis=axis, dtype=dtype, keepdims=keepdims))


def prod(
    x: cp.ndarray,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: cp.dtype = None,
    keepdims: bool = False,
) -> cp.ndarray:

    if dtype is None and cp.issubdtype(x.dtype, cp.integer):
        if cp.issubdtype(x.dtype, cp.signedinteger) and x.dtype in [
            cp.int8,
            cp.int16,
            cp.int32,
        ]:
            dtype = cp.int32
        elif cp.issubdtype(x.dtype, cp.unsignedinteger) and x.dtype in [
            cp.uint8,
            cp.uint16,
            cp.uint32,
        ]:
            dtype = cp.uint32
        elif x.dtype == cp.int64:
            dtype = cp.int64
        else:
            dtype = cp.uint64
    dtype = ivy.as_native_dtype(dtype)
    return cp.asarray(cp.prod(a=x, axis=axis, dtype=dtype, keepdims=keepdims))


def mean(
    x: cp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> cp.ndarray:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return cp.asarray(cp.mean(x, axis=axis, keepdims=keepdims))


def std(
    x: cp.ndarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> cp.ndarray:
    return cp.asarray(cp.std(x, axis=axis, ddof=correction, keepdims=keepdims))


# Extra #
# ------#


def einsum(equation: str, *operands: cp.ndarray) -> cp.ndarray:
    return cp.asarray(cp.einsum(equation, *operands))
