import math

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


class RunningMeanStd(flax.struct.PyTreeNode):
    count: jax.Array
    mean: PyTree
    summed_variance: PyTree


def init_running_mean_std(example: PyTree) -> RunningMeanStd:
    return RunningMeanStd(
        count=jnp.zeros((), dtype=jnp.int32),
        mean=jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=jnp.float32), example),
        summed_variance=jax.tree.map(
            lambda x: jnp.zeros(x.shape, dtype=jnp.float32), example
        ),
    )


def update_running_mean_std(state: RunningMeanStd, batch: PyTree) -> RunningMeanStd:
    if jax.tree.structure(batch) != jax.tree.structure(state.mean):
        raise ValueError("batch and running statistics must have the same tree")

    batch_leaf = jax.tree.leaves(batch)[0]
    mean_leaf = jax.tree.leaves(state.mean)[0]
    batch_shape = batch_leaf.shape[: batch_leaf.ndim - mean_leaf.ndim]
    batch_count = math.prod(batch_shape)
    if batch_count == 0:
        return state

    def mean(x, reference):
        axes = tuple(range(x.ndim - reference.ndim))
        return x.astype(reference.dtype).mean(axis=axes)

    batch_mean = jax.tree.map(mean, batch, state.mean)

    def summed_variance(x, reference, x_mean):
        axes = tuple(range(x.ndim - reference.ndim))
        difference = x.astype(reference.dtype) - x_mean
        return jnp.square(difference).sum(axis=axes)

    batch_summed_variance = jax.tree.map(summed_variance, batch, state.mean, batch_mean)

    count = state.count + batch_count
    old_count = state.count.astype(jnp.float32)
    batch_count_float = jnp.asarray(batch_count, dtype=jnp.float32)
    count_float = count.astype(jnp.float32)

    mean = jax.tree.map(
        lambda old, new: old + (new - old) * batch_count_float / count_float,
        state.mean,
        batch_mean,
    )
    combined_summed_variance = jax.tree.map(
        lambda old_variance, new_variance, old_mean, new_mean: (
            old_variance
            + new_variance
            + jnp.square(new_mean - old_mean)
            * old_count
            * batch_count_float
            / count_float
        ),
        state.summed_variance,
        batch_summed_variance,
        state.mean,
        batch_mean,
    )
    return RunningMeanStd(
        count=count,
        mean=mean,
        summed_variance=combined_summed_variance,
    )


def normalize(
    value: PyTree,
    state: RunningMeanStd,
    min_std: float = 1e-6,
    max_abs_value: float = 1e6,
) -> PyTree:
    count = state.count.astype(jnp.float32)

    def normalize_leaf(x, mean, summed_variance):
        variance = summed_variance / jnp.maximum(count, 1.0)
        std = jnp.maximum(jnp.sqrt(jnp.maximum(variance, 0.0)), min_std)
        std = jnp.where(state.count > 0, std, 1.0)
        normalized = (x - mean) / std
        return jnp.clip(normalized, -max_abs_value, max_abs_value)

    return jax.tree.map(normalize_leaf, value, state.mean, state.summed_variance)
