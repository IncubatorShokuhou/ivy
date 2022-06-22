# global
import cupy as cp
from typing import Union, Optional, Tuple, Literal, List, NamedTuple


# local
from ivy import inf
from collections import namedtuple

# Array API Standard #
# -------------------#


def eigh(x: cp.ndarray) -> cp.ndarray:
    ret = cp.linalg.eigh(x)
    return ret


def inv(x: cp.ndarray) -> cp.ndarray:
    ret = cp.linalg.inv(x)
    return ret


def pinv(
    x: cp.ndarray, rtol: Optional[Union[float, Tuple[float]]] = None
) -> cp.ndarray:

    if rtol is None:
        ret = cp.linalg.pinv(x)
    else:
        ret = cp.linalg.pinv(x, rtol)
    return ret


def matrix_transpose(x: cp.ndarray) -> cp.ndarray:
    ret = cp.swapaxes(x, -1, -2)
    return ret


# noinspection PyUnusedLocal,PyShadowingBuiltins
def vector_norm(
    x: cp.ndarray,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
) -> cp.ndarray:
    if axis is None:
        np_normalized_vector = cp.linalg.norm(x.flatten(), ord, axis, keepdims)

    else:
        np_normalized_vector = cp.linalg.norm(x, ord, axis, keepdims)

    if np_normalized_vector.shape == tuple():
        ret = cp.expand_dims(np_normalized_vector, 0)
    else:
        ret = np_normalized_vector
    return ret


def matrix_norm(
    x: cp.ndarray,
    ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
    keepdims: bool = False,
) -> cp.ndarray:
    ret = cp.linalg.norm(x, ord=ord, axis=(-2, -1), keepdims=keepdims)
    return ret


def matrix_power(x: cp.ndarray, n: int) -> cp.ndarray:
    return cp.linalg.matrix_power(x, n)


def svd(
    x: cp.ndarray, full_matrices: bool = True
) -> Union[cp.ndarray, Tuple[cp.ndarray, ...]]:

    results = namedtuple("svd", "U S Vh")
    U, D, VT = cp.linalg.svd(x, full_matrices=full_matrices)
    ret = results(U, D, VT)
    return ret


def outer(
    x1: cp.ndarray, x2: cp.ndarray, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.outer(x1, x2, out=out)


def diagonal(
    x: cp.ndarray, offset: int = 0, axis1: int = -2, axis2: int = -1
) -> cp.ndarray:
    ret = cp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
    return ret


def svdvals(x: cp.ndarray) -> cp.ndarray:
    ret = cp.linalg.svd(x, compute_uv=False)
    return ret


def qr(x: cp.ndarray, mode: str = "reduced") -> NamedTuple:
    res = namedtuple("qr", ["Q", "R"])
    q, r = cp.linalg.qr(x, mode=mode)
    ret = res(q, r)
    return ret


def matmul(x1: cp.ndarray, x2: cp.ndarray, *, out=None) -> cp.ndarray:
    return cp.matmul(x1, x2, out)


def slogdet(x: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = cp.linalg.slogdet(x)
    ret = results(sign, logabsdet)
    return ret


def tensordot(
    x1: cp.ndarray, x2: cp.ndarray, axes: Union[int, Tuple[List[int], List[int]]] = 2
) -> cp.ndarray:

    ret = cp.tensordot(x1, x2, axes=axes)
    return ret


def trace(x: cp.ndarray, offset: int = 0, *, out=None) -> cp.ndarray:
    return cp.trace(x, offset=offset, axis1=-2, axis2=-1, dtype=x.dtype, out=out)


def vecdot(x1: cp.ndarray, x2: cp.ndarray, axis: int = -1) -> cp.ndarray:
    ret = cp.tensordot(x1, x2, axes=(axis, axis))
    return ret


def det(x: cp.ndarray) -> cp.ndarray:
    ret = cp.linalg.det(x)
    return ret


def cholesky(x: cp.ndarray, upper: bool = False) -> cp.ndarray:
    if not upper:
        ret = cp.linalg.cholesky(x)
    else:
        axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        ret = cp.transpose(cp.linalg.cholesky(cp.transpose(x, axes=axes)), axes=axes)
    return ret


def eigvalsh(x: cp.ndarray) -> cp.ndarray:
    ret = cp.linalg.eigvalsh(x)
    return ret


def cross(x1: cp.ndarray, x2: cp.ndarray, axis: int = -1) -> cp.ndarray:
    ret = cp.cross(a=x1, b=x2, axis=axis)
    return ret


def matrix_rank(
    x: cp.ndarray, rtol: Optional[Union[float, Tuple[float]]] = None
) -> cp.ndarray:
    if rtol is None:
        ret = cp.linalg.matrix_rank(x)
    ret = cp.linalg.matrix_rank(x, rtol)
    return ret


# Extra #
# ------#


def vector_to_skew_symmetric_matrix(vector: cp.ndarray) -> cp.ndarray:
    batch_shape = list(vector.shape[:-1])
    # BS x 3 x 1
    vector_expanded = cp.expand_dims(vector, -1)
    # BS x 1 x 1
    a1s = vector_expanded[..., 0:1, :]
    a2s = vector_expanded[..., 1:2, :]
    a3s = vector_expanded[..., 2:3, :]
    # BS x 1 x 1
    zs = cp.zeros(batch_shape + [1, 1])
    # BS x 1 x 3
    row1 = cp.concatenate((zs, -a3s, a2s), -1)
    row2 = cp.concatenate((a3s, zs, -a1s), -1)
    row3 = cp.concatenate((-a2s, a1s, zs), -1)
    # BS x 3 x 3
    ret = cp.concatenate((row1, row2, row3), -2)
    return ret


def solve(x1: cp.ndarray, x2: cp.ndarray) -> cp.ndarray:
    expanded_last = False
    if len(x2.shape) <= 1:
        if x2.shape[-1] == x1.shape[-1]:
            expanded_last = True
            x2 = cp.expand_dims(x2, axis=1)
    for i in range(len(x1.shape) - 2):
        x2 = cp.expand_dims(x2, axis=0)
    ret = cp.linalg.solve(x1, x2)
    if expanded_last:
        ret = cp.squeeze(ret, axis=-1)
    return ret
