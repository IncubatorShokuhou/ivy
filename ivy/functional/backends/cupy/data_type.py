# global
import cupy as cp
from typing import Union, Tuple, List

# local
import ivy


ivy_dtype_dict = {
    cp.dtype("int8"): "int8",
    cp.dtype("int16"): "int16",
    cp.dtype("int32"): "int32",
    cp.dtype("int64"): "int64",
    cp.dtype("uint8"): "uint8",
    cp.dtype("uint16"): "uint16",
    cp.dtype("uint32"): "uint32",
    cp.dtype("uint64"): "uint64",
    "bfloat16": "bfloat16",
    cp.dtype("float16"): "float16",
    cp.dtype("float32"): "float32",
    cp.dtype("float64"): "float64",
    cp.dtype("bool"): "bool",
    cp.int8: "int8",
    cp.int16: "int16",
    cp.int32: "int32",
    cp.int64: "int64",
    cp.uint8: "uint8",
    cp.uint16: "uint16",
    cp.uint32: "uint32",
    cp.uint64: "uint64",
    cp.float16: "float16",
    cp.float32: "float32",
    cp.float64: "float64",
    cp.bool_: "bool",
}

native_dtype_dict = {
    "int8": cp.dtype("int8"),
    "int16": cp.dtype("int16"),
    "int32": cp.dtype("int32"),
    "int64": cp.dtype("int64"),
    "uint8": cp.dtype("uint8"),
    "uint16": cp.dtype("uint16"),
    "uint32": cp.dtype("uint32"),
    "uint64": cp.dtype("uint64"),
    "bfloat16": "bfloat16",
    "float16": cp.dtype("float16"),
    "float32": cp.dtype("float32"),
    "float64": cp.dtype("float64"),
    "bool": cp.dtype("bool"),
}


# noinspection PyShadowingBuiltins
def iinfo(type: Union[cp.dtype, str, cp.ndarray]) -> cp.iinfo:
    return cp.iinfo(ivy.as_native_dtype(type))


class Finfo:
    def __init__(self, np_finfo):
        self._np_finfo = np_finfo

    @property
    def bits(self):
        return self._np_finfo.bits

    @property
    def eps(self):
        return float(self._np_finfo.eps)

    @property
    def max(self):
        return float(self._np_finfo.max)

    @property
    def min(self):
        return float(self._np_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._np_finfo.tiny)


def can_cast(from_: Union[cp.dtype, cp.ndarray], to: cp.dtype) -> bool:
    if isinstance(from_, cp.ndarray):
        from_ = str(from_.dtype)
    from_ = str(from_)
    to = str(to)
    if "bool" in from_ and (("int" in to) or ("float" in to)):
        return False
    if "int" in from_ and "float" in to:
        return False
    return cp.can_cast(from_, to)


# noinspection PyShadowingBuiltins
def finfo(type: Union[cp.dtype, str, cp.ndarray]) -> Finfo:
    return Finfo(cp.finfo(ivy.as_native_dtype(type)))


def result_type(*arrays_and_dtypes: Union[cp.ndarray, cp.dtype]) -> cp.dtype:
    if len(arrays_and_dtypes) <= 1:
        return cp.result_type(arrays_and_dtypes)

    result = cp.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = cp.result_type(result, arrays_and_dtypes[i])
    return result


def broadcast_to(x: cp.ndarray, shape: Tuple[int, ...]) -> cp.ndarray:
    return cp.broadcast_to(x, shape)


def broadcast_arrays(*arrays: cp.ndarray) -> List[cp.ndarray]:
    return cp.broadcast_arrays(*arrays)


def astype(x: cp.ndarray, dtype: cp.dtype, *, copy: bool = True) -> cp.ndarray:
    dtype = ivy.as_native_dtype(dtype)
    if copy:
        if x.dtype == dtype:
            new_tensor = cp.copy(x)
            return new_tensor
    else:
        if x.dtype == dtype:
            return x
        else:
            new_tensor = cp.copy(x)
            return new_tensor.astype(dtype)
    return x.astype(dtype)


def dtype_bits(dtype_in):
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
    )


def dtype(x, as_native=False):
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def as_ivy_dtype(dtype_in):
    if isinstance(dtype_in, str):
        return ivy.Dtype(dtype_in)
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


def as_native_dtype(dtype_in):
    if not isinstance(dtype_in, str):
        return dtype_in
    return native_dtype_dict[ivy.Dtype(dtype_in)]
