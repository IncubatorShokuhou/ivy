# global
import cupy as cp

from typing import Union, Tuple, Optional, List

# local
from .data_type import as_native_dtype
from ivy.functional.ivy import default_dtype

# noinspection PyProtectedMember
from ivy.functional.backends.cupy.device import _to_device


# Array API Standard #
# -------------------#


def asarray(object_in, *, copy=None, dtype: cp.dtype = None, device: str):
    # If copy=none then try using existing memory buffer
    if isinstance(object_in, cp.ndarray) and dtype is None:
        dtype = object_in.dtype
    elif (
        isinstance(object_in, (list, tuple, dict))
        and len(object_in) != 0
        and dtype is None
    ):
        dtype = default_dtype(item=object_in, as_native=True)
        if copy is True:
            return _to_device(
                cp.copy(cp.asarray(object_in, dtype=dtype)), device=device
            )
        else:
            return _to_device(cp.asarray(object_in, dtype=dtype), device=device)
    else:
        dtype = default_dtype(dtype, object_in)
    if copy is True:
        return _to_device(cp.copy(cp.asarray(object_in, dtype=dtype)), device=device)
    else:
        return _to_device(cp.asarray(object_in, dtype=dtype), device=device)


def zeros(
    shape: Union[int, Tuple[int], List[int]], *, dtype: cp.dtype, device: str
) -> cp.ndarray:
    return _to_device(cp.zeros(shape, dtype), device=device)


def ones(
    shape: Union[int, Tuple[int], List[int]], *, dtype: cp.dtype, device: str
) -> cp.ndarray:
    dtype = as_native_dtype(default_dtype(dtype))
    return _to_device(cp.ones(shape, dtype), device=device)


def full_like(
    x: cp.ndarray, fill_value: Union[int, float], *, dtype: cp.dtype, device: str
) -> cp.ndarray:
    if dtype:
        dtype = "bool_" if dtype == "bool" else dtype
    else:
        dtype = x.dtype
    return _to_device(cp.full_like(x, fill_value, dtype=dtype), device=device)


def ones_like(x: cp.ndarray, *, dtype: cp.dtype, device: str) -> cp.ndarray:

    if dtype:
        dtype = "bool_" if dtype == "bool" else dtype
        dtype = cp.dtype(dtype)
    else:
        dtype = x.dtype

    return _to_device(cp.ones_like(x, dtype=dtype), device=device)


def zeros_like(x: cp.ndarray, *, dtype: cp.dtype, device: str) -> cp.ndarray:
    if dtype:
        dtype = "bool_" if dtype == "bool" else dtype
    else:
        dtype = x.dtype
    return _to_device(cp.zeros_like(x, dtype=dtype), device=device)


def tril(x: cp.ndarray, k: int = 0) -> cp.ndarray:
    return cp.tril(x, k)


def triu(x: cp.ndarray, k: int = 0) -> cp.ndarray:
    return cp.triu(x, k)


def empty(
    shape: Union[int, Tuple[int], List[int]], *, dtype: cp.dtype, device: str
) -> cp.ndarray:
    return _to_device(
        cp.empty(shape, as_native_dtype(default_dtype(dtype))), device=device
    )


def empty_like(x: cp.ndarray, *, dtype: cp.dtype, device: str) -> cp.ndarray:

    if dtype:
        dtype = "bool_" if dtype == "bool" else dtype
        dtype = cp.dtype(dtype)
    else:
        dtype = x.dtype

    return _to_device(cp.empty_like(x, dtype=dtype), device=device)


def linspace(
    start, stop, num, axis=None, endpoint=True, *, dtype: cp.dtype, device: str
):
    if axis is None:
        axis = -1
    ans = cp.linspace(start, stop, num, endpoint, dtype=dtype, axis=axis)
    if dtype is None:
        ans = cp.float32(ans)
    # Waiting for fix when start is -0.0: https://github.com/numpy/numpy/issues/21513
    if (
        ans.shape[0] >= 1
        and (not isinstance(start, numpy.ndarray))
        and (not isinstance(stop, numpy.ndarray))
    ):
        ans[0] = start
    return _to_device(ans, device=device)


def meshgrid(*arrays: cp.ndarray, indexing: str = "xy") -> List[cp.ndarray]:
    return cp.meshgrid(*arrays, indexing=indexing)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    *,
    dtype: cp.dtype,
    device: str
) -> cp.ndarray:
    dtype = as_native_dtype(default_dtype(dtype))
    return _to_device(cp.eye(n_rows, n_cols, k, dtype), device=device)


# noinspection PyShadowingNames
def arange(start, stop=None, step=1, *, dtype: cp.dtype = None, device: str):
    if dtype:
        dtype = as_native_dtype(dtype)
    res = _to_device(cp.arange(start, stop, step=step, dtype=dtype), device=device)
    if not dtype:
        if res.dtype == cp.float64:
            return res.astype(cp.float32)
        elif res.dtype == cp.int64:
            return res.astype(cp.int32)
    return res


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: cp.dtype = None,
    device: str
) -> cp.ndarray:
    return _to_device(
        cp.full(shape, fill_value, as_native_dtype(default_dtype(dtype, fill_value))),
        device=device,
    )


def from_dlpack(x):
    return cp.from_dlpack(x)


# Extra #
# ------#

array = asarray


def logspace(start, stop, num, base=10.0, axis=None, *, device: str):
    if axis is None:
        axis = -1
    power_seq =  _to_device(cp.linspace(
        start, stop, num, axis=axis, dtype=None), device=device
    )
    return _to_device(
        base**power_seq, device=device
    )

