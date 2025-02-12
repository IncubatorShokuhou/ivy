# global
import sys
import cupy as cp

# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.backend_handler.ContextManager(sys.modules[__name__])

NativeArray = cp.ndarray
NativeVariable = cp.ndarray
NativeDevice = str
NativeDtype = cp.dtype

# data types (preventing cyclic imports)
int8 = ivy.IntDtype("int8")
int16 = ivy.IntDtype("int16")
int32 = ivy.IntDtype("int32")
int64 = ivy.IntDtype("int64")
uint8 = ivy.IntDtype("uint8")
uint16 = ivy.IntDtype("uint16")
uint32 = ivy.IntDtype("uint32")
uint64 = ivy.IntDtype("uint64")
bfloat16 = ivy.FloatDtype("bfloat16")
float16 = ivy.FloatDtype("float16")
float32 = ivy.FloatDtype("float32")
float64 = ivy.FloatDtype("float64")
# noinspection PyShadowingBuiltins
bool = ivy.Dtype("bool")

# native data types
native_int8 = cp.dtype("int8")
native_int16 = cp.dtype("int16")
native_int32 = cp.dtype("int32")
native_int64 = cp.dtype("int64")
native_uint8 = cp.dtype("uint8")
native_uint16 = cp.dtype("uint16")
native_uint32 = cp.dtype("uint32")
native_uint64 = cp.dtype("uint64")
native_float16 = cp.dtype("float16")
native_float32 = cp.dtype("float32")
native_float64 = cp.dtype("float64")
# noinspection PyShadowingBuiltins
native_bool = cp.dtype("bool")

# valid data types
valid_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    bool,
)
valid_numeric_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
)
valid_int_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
valid_float_dtypes = (float16, float32, float64)

# invalid data types
invalid_dtypes = (bfloat16,)
invalid_numeric_dtypes = (bfloat16,)
invalid_int_dtypes = ()
invalid_float_dtypes = (bfloat16,)


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.as_ivy_dtype(type)
    if type_str in invalid_dtypes:
        return {"bfloat16": float16}[type_str]
    return type


backend = "cupy"


# local sub-modules
from . import activations
from .activations import *
from . import compilation
from .compilation import *
from . import creation
from .creation import *
from . import data_type
from .data_type import *
from . import device
from .device import *
from . import elementwise
from .elementwise import *
from . import general
from .general import *
from . import gradients
from .gradients import *
from . import image
from .image import *
from . import layers
from .layers import *
from . import linear_algebra as linalg
from .linear_algebra import *
from . import manipulation
from .manipulation import *
from . import random
from .random import *
from . import searching
from .searching import *
from . import set
from .set import *
from . import sorting
from .sorting import *
from . import statistical
from .statistical import *
from . import utility
from .utility import *
