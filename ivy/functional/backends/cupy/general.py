"""Collection of Numpy general functions, wrapped to fit Ivy syntax and signature."""

# global
from typing import List, Optional, Union
import numpy as np
import cupy as cp
from operator import mul
from functools import reduce
import multiprocessing as _multiprocessing
from numbers import Number

# local
import ivy
from ivy.functional.backends.numpy.device import dev, _to_device

# Helpers #
# --------#


def copy_array(x: cp.ndarray) -> cp.ndarray:
    return x.copy()


def array_equal(x0: cp.ndarray, x1: cp.ndarray) -> bool:
    return cp.array_equal(x0, x1)


def to_numpy(x: np.ndarray) -> np.ndarray:
    return x


def to_scalar(x: cp.ndarray) -> Number:
    return x.item()


def to_list(x: cp.ndarray) -> list:
    return x.tolist()


def container_types():
    return []


inplace_arrays_supported = lambda: True
inplace_variables_supported = lambda: True


def inplace_update(
    x: Union[ivy.Array, cp.ndarray],
    val: Union[ivy.Array, cp.ndarray],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)

    # make both arrays contiguous if not already
    if not x_native.flags.c_contiguous:
        x_native = cp.ascontiguousarray(x_native)
    if not val_native.flags.c_contiguous:
        val_native = cp.ascontiguousarray(val_native)

    x_native.data = val_native

    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def is_native_array(x, exclusive=False):
    if isinstance(x, cp.ndarray):
        return True
    return False


def floormod(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    ret = cp.asarray(x % y)
    return ret


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    x_split = cp.split(x, x.shape[axis], axis)
    if keepdims:
        return x_split
    return [cp.squeeze(item, axis) for item in x_split]


def inplace_decrement(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_increment(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def cumsum(x: cp.ndarray, axis: int = 0) -> cp.ndarray:
    return cp.cumsum(x, axis)


def cumprod(
    x: cp.ndarray, axis: int = 0, exclusive: Optional[bool] = False
) -> cp.ndarray:
    if exclusive:
        x = cp.swapaxes(x, axis, -1)
        x = cp.concatenate((cp.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = cp.cumprod(x, -1)
        return cp.swapaxes(res, axis, -1)
    return cp.cumprod(x, axis)


def scatter_flat(indices, updates, size=None, tensor=None, reduction="sum", *, device):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if device is None:
        device = dev(updates)
    if reduction == "sum":
        if not target_given:
            target = cp.zeros([size], dtype=updates.dtype)
        cp.add.at(target, indices, updates)
    elif reduction == "replace":
        if not target_given:
            target = cp.zeros([size], dtype=updates.dtype)
        target = cp.asarray(target).copy()
        target.setflags(write=1)
        target[indices] = updates
    elif reduction == "min":
        if not target_given:
            target = cp.ones([size], dtype=updates.dtype) * 1e12
        cp.minimum.at(target, indices, updates)
        if not target_given:
            target = cp.where(target == 1e12, 0.0, target)
    elif reduction == "max":
        if not target_given:
            target = cp.ones([size], dtype=updates.dtype) * -1e12
        cp.maximum.at(target, indices, updates)
        if not target_given:
            target = cp.where(target == -1e12, 0.0, target)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return _to_device(target, device)


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape=None, tensor=None, reduction="sum", *, device):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.shape_to_tuple(target.shape) == ivy.shape_to_tuple(shape)
    if device is None:
        device = dev(updates)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if reduction == "sum":
        if not target_given:
            target = cp.zeros(shape, dtype=updates.dtype)
        cp.add.at(target, indices_tuple, updates)
    elif reduction == "replace":
        if not target_given:
            target = cp.zeros(shape, dtype=updates.dtype)
        target = cp.asarray(target).copy()
        target.setflags(write=1)
        target[indices_tuple] = updates
    elif reduction == "min":
        if not target_given:
            target = cp.ones(shape, dtype=updates.dtype) * 1e12
        cp.minimum.at(target, indices_tuple, updates)
        if not target_given:
            target = cp.where(target == 1e12, 0.0, target)
    elif reduction == "max":
        if not target_given:
            target = cp.ones(shape, dtype=updates.dtype) * -1e12
        cp.maximum.at(target, indices_tuple, updates)
        if not target_given:
            target = cp.where(target == -1e12, 0.0, target)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return _to_device(target, device)


def gather(
    params: cp.ndarray, indices: cp.ndarray, axis: Optional[int] = -1, *, device: str
) -> cp.ndarray:
    if device is None:
        device = dev(params)
    return _to_device(cp.take_along_axis(params, indices, axis), device)


def gather_nd(params, indices, *, device: str):
    if device is None:
        device = dev(params)
    indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        reduce(mul, params_shape[i + 1 :], 1) for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = cp.array(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = cp.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = cp.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = cp.tile(
        cp.reshape(cp.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)),
        (1, implicit_indices_factor),
    )
    implicit_indices = cp.tile(
        cp.expand_dims(cp.arange(implicit_indices_factor), 0),
        (indices_for_flat_tiled.shape[0], 1),
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = cp.reshape(indices_for_flat, (-1,)).astype(cp.int32)
    flat_gather = cp.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    res = cp.reshape(flat_gather, new_shape)
    return _to_device(res, device)


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def indices_where(x):
    where_x = cp.where(x)
    if len(where_x) == 1:
        return cp.expand_dims(where_x[0], -1)
    res = cp.concatenate([cp.expand_dims(item, -1) for item in where_x], -1)
    return res


# noinspection PyUnusedLocal
def one_hot(indices, depth, *, device):
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = cp.eye(depth)[cp.array(indices).reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


def shape(x: cp.ndarray, as_tensor: bool = False) -> Union[cp.ndarray, List[int]]:
    if as_tensor:
        return cp.asarray(cp.shape(x))
    else:
        return x.shape


def get_num_dims(x, as_tensor=False):
    return cp.asarray(len(cp.shape(x))) if as_tensor else len(x.shape)


def current_backend_str():
    return "cupy"