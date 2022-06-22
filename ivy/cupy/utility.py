# global
import cupy as cp
from typing import Union, Tuple, Optional, List


# noinspection PyShadowingBuiltins
def all(
    x: cp.ndarray,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.asarray(cp.all(x, axis=axis, keepdims=keepdims, out=out))


# noinspection PyShadowingBuiltins
def any(
    x: cp.ndarray,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    keepdims: bool = False,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.asarray(cp.any(x, axis=axis, keepdims=keepdims, out=out))
