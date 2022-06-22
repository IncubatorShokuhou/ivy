# global
import cupy as cp
from typing import Optional, Callable
import functools

# local
import ivy

from cupyx.scipy.special import erf


# when inputs are 0 dimensional, numpy's functions return scalars
# so we use this wrapper to ensure outputs are always numpy arrays
def _handle_0_dim_output(function: Callable) -> Callable:
    @functools.wraps(function)
    def new_function(*args, **kwargs):
        ret = function(*args, **kwargs)
        return cp.asarray(ret) if not isinstance(ret, cp.ndarray) else ret

    return new_function


@_handle_0_dim_output
def add(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = cp.promote_types(x1.dtype, x2.dtype)
        x1, x2 = cp.asarray(x1), cp.asarray(x2)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    return cp.add(cp.asarray(x1), cp.asarray(x2), out=out)


@_handle_0_dim_output
def pow(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = cp.promote_types(x1.dtype, x2.dtype)
        x1, x2 = cp.asarray(x1), cp.asarray(x2)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not hasattr(x2, "dtype"):
        x2 = cp.array(x2, dtype=x1.dtype)
    return cp.power(x1, x2, out=out)


@_handle_0_dim_output
def bitwise_xor(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.bitwise_xor(x1, x2, out=out)


@_handle_0_dim_output
def exp(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.exp(x, out=out)


@_handle_0_dim_output
def expm1(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.expm1(x, out=out)


@_handle_0_dim_output
def bitwise_invert(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.invert(x, out=out)


@_handle_0_dim_output
def bitwise_and(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.bitwise_and(x1, x2, out=out)


@_handle_0_dim_output
def equal(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.equal(x1, x2, out=out)


@_handle_0_dim_output
def greater(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.greater(x1, x2, out=out)


@_handle_0_dim_output
def greater_equal(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.greater_equal(x1, x2, out=out)


@_handle_0_dim_output
def less_equal(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.less_equal(x1, x2, out=out)


@_handle_0_dim_output
def multiply(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = cp.promote_types(x1.dtype, x2.dtype)
        x1, x2 = cp.asarray(x1), cp.asarray(x2)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not hasattr(x2, "dtype"):
        x2 = cp.array(x2, dtype=x1.dtype)
    return cp.multiply(x1, x2, out=out)


@_handle_0_dim_output
def ceil(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if "int" in str(x.dtype):
        ret = cp.copy(x)
    else:
        return cp.ceil(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_0_dim_output
def floor(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if "int" in str(x.dtype):
        ret = cp.copy(x)
    else:
        return cp.floor(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_0_dim_output
def sign(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.sign(x, out=out)


@_handle_0_dim_output
def sqrt(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.sqrt(x, out=out)


@_handle_0_dim_output
def isfinite(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.isfinite(x, out=out)


@_handle_0_dim_output
def asin(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arcsin(x, out=out)


@_handle_0_dim_output
def isinf(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.isinf(x, out=out)


@_handle_0_dim_output
def asinh(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arcsinh(x, out=out)


@_handle_0_dim_output
def cosh(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.cosh(x, out=out)


@_handle_0_dim_output
def log10(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.log10(x, out=out)


@_handle_0_dim_output
def log(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.log(x, out=out)


@_handle_0_dim_output
def log2(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.log2(x, out=out)


@_handle_0_dim_output
def log1p(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.log1p(x, out=out)


@_handle_0_dim_output
def isnan(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.isnan(x, out=out)


@_handle_0_dim_output
def less(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.less(x1, x2, out=out)


@_handle_0_dim_output
def cos(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.cos(x, out=out)


@_handle_0_dim_output
def logical_not(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.logical_not(x, out=out)


@_handle_0_dim_output
def divide(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if isinstance(x1, cp.ndarray):
        if not isinstance(x2, cp.ndarray):
            x2 = cp.asarray(x2, dtype=x1.dtype)
        else:
            promoted_type = cp.promote_types(x1.dtype, x2.dtype)
            x1 = x1.astype(promoted_type)
            x2 = x2.astype(promoted_type)
    return cp.divide(x1, x2, out=out)


@_handle_0_dim_output
def acos(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arccos(x, out=out)


@_handle_0_dim_output
def logical_xor(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.logical_xor(x1, x2, out=out)


@_handle_0_dim_output
def logical_or(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.logical_or(x1, x2, out=out)


@_handle_0_dim_output
def logical_and(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.logical_and(x1, x2, out=out)


@_handle_0_dim_output
def acosh(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arccosh(x, out=out)


@_handle_0_dim_output
def sin(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.sin(x, out=out)


@_handle_0_dim_output
def negative(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.negative(x, out=out)


@_handle_0_dim_output
def not_equal(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.not_equal(x1, x2, out=out)


@_handle_0_dim_output
def tanh(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.tanh(x, out=out)


@_handle_0_dim_output
def floor_divide(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.floor_divide(x1, x2, out=out)


@_handle_0_dim_output
def sinh(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.sinh(x, out=out)


@_handle_0_dim_output
def positive(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.positive(x, out=out)


@_handle_0_dim_output
def square(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.square(x, out=out)


@_handle_0_dim_output
def remainder(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.remainder(x1, x2, out=out)


@_handle_0_dim_output
def round(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if "int" in str(x.dtype):
        ret = cp.copy(x)
    else:
        return cp.round(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_0_dim_output
def bitwise_or(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.bitwise_or(x1, x2, out=out)


@_handle_0_dim_output
def trunc(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if "int" in str(x.dtype):
        ret = cp.copy(x)
    else:
        return cp.trunc(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_0_dim_output
def abs(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.absolute(x, out=out)


@_handle_0_dim_output
def subtract(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not hasattr(x2, "dtype"):
        x2 = cp.array(x2, dtype=x1.dtype)
    return cp.subtract(x1, x2, out=out)


@_handle_0_dim_output
def logaddexp(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.logaddexp(x1, x2, out=out)


@_handle_0_dim_output
def bitwise_right_shift(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.right_shift(x1, x2, out=out)


@_handle_0_dim_output
def bitwise_left_shift(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.left_shift(x1, x2, out=out)


@_handle_0_dim_output
def tan(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.tan(x, out=out)


@_handle_0_dim_output
def atan(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arctan(x, out=out)


@_handle_0_dim_output
def atanh(x: cp.ndarray, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arctanh(x, out=out)


@_handle_0_dim_output
def atan2(
    x1: cp.ndarray, x2: cp.ndarray, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    if not isinstance(x2, cp.ndarray):
        x2 = cp.asarray(x2, dtype=x1.dtype)
    else:
        dtype = cp.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return cp.arctan2(x1, x2, out=out)


# Extra #
# ------#


@_handle_0_dim_output
def minimum(x1, x2, *, out: Optional[cp.ndarray] = None):
    return cp.minimum(x1, x2, out=out)


@_handle_0_dim_output
def maximum(x1, x2, *, out: Optional[cp.ndarray] = None):
    return cp.maximum(x1, x2, out=out)


@_handle_0_dim_output
def erf(x, *, out: Optional[cp.ndarray] = None):
    if _erf is None:
        raise Exception(
            "scipy must be installed in order to call ivy.erf with a numpy backend."
        )
    return _erf(x, out=out)
