import cupy as cp

from typing import Optional, Tuple


def argmax(
    x: cp.ndarray,
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> cp.ndarray:
    return cp.argmax(x, axis=axis, keepdims=keepdims)


def argmin(
    x: cp.ndarray,
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> cp.ndarray:
    return cp.argmin(x, axis=axis, keepdims=keepdims)


def nonzero(x: cp.ndarray) -> Tuple[cp.ndarray]:
    return cp.nonzero(x)


def where(condition: cp.ndarray, x1: cp.ndarray, x2: cp.ndarray) -> cp.ndarray:
    return cp.where(condition, x1, x2)
