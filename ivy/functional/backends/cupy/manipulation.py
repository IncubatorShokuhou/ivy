# global
import cupy as cp
import math
from typing import Union, Tuple, Optional, List
from numbers import Number

# local


def squeeze(
    x: cp.ndarray,
    axis: Union[int, Tuple[int], List[int]],
) -> cp.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ValueError(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    ret = cp.squeeze(x, axis)
    return ret


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def flip(
    x: cp.ndarray,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
) -> cp.ndarray:
    num_dims = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        axis = list(range(num_dims))
    if type(axis) is int:
        axis = [axis]
    axis = [item + num_dims if item < 0 else item for item in axis]
    ret = cp.flip(x, axis)
    return ret


def expand_dims(x: cp.ndarray, axis: int = 0) -> cp.ndarray:
    ret = cp.expand_dims(x, axis)
    return ret


def permute_dims(x: cp.ndarray, axes: Tuple[int, ...]) -> cp.ndarray:
    ret = cp.transpose(x, axes)
    return ret


def concat(xs: List[cp.ndarray], axis: int = 0) -> cp.ndarray:
    is_tuple = type(xs) is tuple
    if axis is None:
        if is_tuple:
            xs = list(xs)
        for i in range(len(xs)):
            if xs[i].shape == ():
                xs[i] = cp.ravel(xs[i])
        if is_tuple:
            xs = tuple(xs)
    ret = cp.concatenate(xs, axis)
    highest_dtype = xs[0].dtype
    for i in xs:
        highest_dtype = cp.promote_types(highest_dtype, i.dtype)
    ret = ret.astype(highest_dtype)
    return ret


def stack(
    x: Union[Tuple[cp.ndarray], List[cp.ndarray]],
    axis: Optional[int] = 0,
    *,
    out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.stack(x, axis, out=out)


def reshape(
    x: cp.ndarray, shape: Tuple[int, ...], copy: Optional[bool] = None
) -> cp.ndarray:
    ret = cp.reshape(x, shape)
    return ret


# Extra #
# ------#


def roll(
    x: cp.ndarray,
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> cp.ndarray:
    return cp.roll(x, shift, axis)


def split(x, num_or_size_splits=None, axis=0, with_remainder=False):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
        return [x]
    if num_or_size_splits is None:
        num_or_size_splits = x.shape[axis]
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits] * num_chunks_int + [
                int(remainder * num_or_size_splits)
            ]
    if isinstance(num_or_size_splits, (list, tuple)):
        num_or_size_splits = cp.cumsum(num_or_size_splits[:-1])
    return cp.split(x, num_or_size_splits, axis)


def repeat(
    x: cp.ndarray, repeats: Union[int, List[int]], axis: int = None
) -> cp.ndarray:
    ret = cp.repeat(x, repeats, axis)
    return ret


def tile(x: cp.ndarray, reps, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    ret = cp.tile(x, reps)
    return ret


def constant_pad(
    x: cp.ndarray, pad_width: List[List[int]], value: Number = 0.0
) -> cp.ndarray:
    ret = cp.pad(_flat_array_to_1_dim_array(x), pad_width, constant_values=value)
    return ret


def zero_pad(
    x: cp.ndarray, pad_width: List[List[int]], out: Optional[cp.ndarray] = None
):
    ret = cp.pad(_flat_array_to_1_dim_array(x), pad_width)
    return ret


def swapaxes(
    x: cp.ndarray, axis0: int, axis1: int, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    ret = cp.swapaxes(x, axis0, axis1)
    return ret


def clip(
    x: cp.ndarray, x_min: Union[Number, cp.ndarray], x_max: Union[Number, cp.ndarray]
) -> cp.ndarray:
    ret = cp.asarray(cp.clip(x, x_min, x_max))
    return ret
