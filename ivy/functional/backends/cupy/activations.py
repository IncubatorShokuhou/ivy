"""Collection of Numpy activation functions, wrapped to fit Ivy syntax and signature."""

from typing import Optional

# global
import cupy as cp


from cupyx.scipy.special import erf


def relu(x: cp.ndarray, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.maximum(x, 0, out=out)


def leaky_relu(x: cp.ndarray, alpha: Optional[float] = 0.2) -> cp.ndarray:
    return cp.where(x > 0, x, x * alpha)


def gelu(x, approximate: Optional[bool] = True):
    if approximate:
        return 0.5 * x * (1 + cp.tanh(cp.sqrt(2 / cp.pi) * (x + 0.044715 * x**3)))
    return 0.5 * x * (1 + erf(x / cp.sqrt(2)))


def sigmoid(x: cp.ndarray) -> cp.ndarray:
    return 1 / (1 + cp.exp(-x))


def tanh(x: cp.ndarray) -> cp.ndarray:
    return (cp.exp(x) - cp.exp(-x)) / (cp.exp(x) + cp.exp(-x))


def softmax(x: cp.ndarray, axis: Optional[int] = None) -> cp.ndarray:
    exp_x = cp.exp(x)
    return exp_x / cp.sum(exp_x, axis, keepdims=True)


def softplus(x: cp.ndarray) -> cp.ndarray:
    return cp.log1p(cp.exp(-cp.abs(x))) + cp.maximum(x, 0)
