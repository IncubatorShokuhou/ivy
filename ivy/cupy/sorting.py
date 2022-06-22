# global
import cupy as cp

# local


def argsort(
    x: cp.ndarray, axis: int = -1, descending: bool = False, stable: bool = True
) -> cp.ndarray:
    if descending:
        ret = cp.asarray(
            cp.argsort(-1 * cp.searchsorted(cp.unique(x), x), axis)
        )
    else:
        ret = cp.asarray(cp.argsort(x, axis))
    return ret


def sort(
    x: cp.ndarray, axis: int = -1, descending: bool = False, stable: bool = True
) -> cp.ndarray:
    # cupy.sort currently does not support kind and order parameters that
    # numpy.sort does support.
    ret = cp.asarray(cp.sort(x, axis=axis))
    if descending:
        ret = cp.asarray((cp.flip(ret, axis)))
    return ret

