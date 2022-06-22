"""Collection of cupy random functions, wrapped to fit Ivy syntax and signature."""

# global
import cupy as cp
from typing import Optional, Union, Tuple, Sequence

# localf


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype=None,
    *,
    device: str,
) -> cp.ndarray:
    return cp.asarray(cp.random.uniform(low, high, shape), dtype=dtype)


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    device: str,
) -> cp.ndarray:
    return cp.asarray(cp.random.normal(mean, std, shape))


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[cp.ndarray] = None,
    replace=True,
    *,
    device: str,
) -> cp.ndarray:
    if probs is None:
        probs = (
            cp.ones(
                (
                    batch_size,
                    population_size,
                )
            )
            / population_size
        )
    orig_probs_shape = list(probs.shape)
    num_classes = orig_probs_shape[-1]
    probs_flat = cp.reshape(probs, (-1, orig_probs_shape[-1]))
    probs_flat = probs_flat / cp.sum(probs_flat, -1, keepdims=True, dtype="float64")
    probs_stack = cp.split(probs_flat, probs_flat.shape[0])
    samples_stack = [
        cp.random.choice(num_classes, num_samples, replace, p=prob[0])
        for prob in probs_stack
    ]
    samples_flat = cp.stack(samples_stack)
    return cp.asarray(cp.reshape(samples_flat, orig_probs_shape[:-1] + [num_samples]))


def randint(
    low: int, high: int, shape: Union[int, Sequence[int]], *, device: str
) -> cp.ndarray:
    return cp.random.randint(low, high, shape)


def seed(seed_value: int = 0) -> None:
    cp.random.seed(seed_value)


def shuffle(x: cp.ndarray) -> cp.ndarray:
    return cp.random.permutation(x)
