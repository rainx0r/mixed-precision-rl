from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import numpy as np
import pytest

from mixed_precision_rl.envs.base import (
    ContinuousActionSpace,
    DiscreteActionSpace,
    Environment,
    EnvStartMode,
    Timestep,
)
from mixed_precision_rl.types import Action, EnvState, LogDict, Observation


class FakeVectorEnvironment(Environment):
    def __init__(
        self,
        num_envs: int,
        max_episode_length: int,
        action_repeat: int = 1,
        discrete: bool = False,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            num_eval_envs=num_envs,
            max_episode_length=max_episode_length,
            action_repeat=action_repeat,
        )
        self.discrete = discrete

    def _init(self, rng: PRNGKeyArray, num_envs: int) -> tuple[EnvState, Observation]:
        del rng
        steps = jnp.zeros(num_envs, dtype=jnp.int32)
        action_total = jnp.zeros(num_envs, dtype=jnp.float32)
        state = {"steps": steps, "action_total": action_total}
        observations = jnp.stack((steps.astype(jnp.float32), action_total), axis=-1)
        return state, observations

    def step(self, state: EnvState, action: Action) -> tuple[EnvState, Timestep]:
        steps = state["steps"] + 1
        action_values = action.astype(jnp.float32) if action.ndim == 1 else action[:, 0]
        action_total = state["action_total"] + action_values
        next_state = {"steps": steps, "action_total": action_total}
        observations = jnp.stack((steps.astype(jnp.float32), action_total), axis=-1)
        terminated = jnp.zeros(steps.shape, dtype=jnp.bool_)
        truncated = steps * self.action_repeat >= self.max_episode_length
        return next_state, Timestep(
            observation=observations,
            reward=action_values,
            terminated=terminated,
            truncated=truncated,
            info={
                "episode_return": action_total,
                "episode_steps": steps * self.action_repeat,
            },
        )

    def make_evaluation(
        self,
        agent: Any,
        initial_agent_state: Any,
    ) -> Callable[[PRNGKeyArray, Any], LogDict]:
        del agent, initial_agent_state

        def evaluate(rng: PRNGKeyArray, params: Any) -> LogDict:
            del rng, params
            return {}

        return evaluate

    @property
    def action_space(self) -> DiscreteActionSpace | ContinuousActionSpace:
        if self.discrete:
            return DiscreteActionSpace(num_actions=4)
        return ContinuousActionSpace(action_dim=1)


def _expected_offsets_and_action_total(
    env: FakeVectorEnvironment,
    rng: PRNGKeyArray,
    rollout_length: int,
) -> tuple[jax.Array, jax.Array]:
    num_phases = (env.max_episode_length + env.action_repeat - 1) // env.action_repeat
    num_groups = max(num_phases // rollout_length, 1)
    group_offsets = jnp.arange(num_groups, dtype=jnp.int32) * rollout_length
    group_indices = jnp.arange(env.num_envs, dtype=jnp.int32) % num_groups
    assignment_rng, action_rng = jax.random.split(jax.random.fold_in(rng, 1))
    group_indices = jax.random.permutation(assignment_rng, group_indices)
    offsets = group_offsets[group_indices]
    action_total = jnp.zeros(env.num_envs, dtype=jnp.float32)

    for step in range((num_groups - 1) * rollout_length):
        action_rng, step_rng = jax.random.split(action_rng)
        if isinstance(env.action_space, ContinuousActionSpace):
            actions = jax.random.uniform(
                step_rng,
                (env.num_envs, env.action_space.action_dim),
                minval=-1.0,
                maxval=1.0,
            )[:, 0]
        else:
            actions = jax.random.randint(
                step_rng,
                (env.num_envs,),
                minval=0,
                maxval=env.action_space.num_actions,
                dtype=jnp.int32,
            ).astype(jnp.float32)
        action_total += jnp.where(step < offsets, actions, 0.0)

    return offsets, action_total


def test_init_is_synchronized_by_default() -> None:
    env = FakeVectorEnvironment(num_envs=8, max_episode_length=10)

    state, observations = env.init(jax.random.PRNGKey(0))

    np.testing.assert_array_equal(state["steps"], np.zeros(8, dtype=np.int32))
    np.testing.assert_array_equal(state["action_total"], np.zeros(8))
    np.testing.assert_array_equal(observations, np.zeros((8, 2)))


def test_init_staggers_environments_across_balanced_rollout_groups() -> None:
    env = FakeVectorEnvironment(num_envs=10, max_episode_length=1000)

    state, observations = env.init(
        jax.random.PRNGKey(11), "staggered", rollout_length=480
    )

    unique_offsets, counts = np.unique(np.asarray(state["steps"]), return_counts=True)
    np.testing.assert_array_equal(unique_offsets, [0, 480])
    np.testing.assert_array_equal(counts, [5, 5])
    np.testing.assert_array_equal(observations[:, 0], state["steps"])


def test_init_staggered_uses_deterministic_random_continuous_actions() -> None:
    env = FakeVectorEnvironment(num_envs=7, max_episode_length=6)
    rng = jax.random.PRNGKey(13)
    expected_offsets, expected_action_total = _expected_offsets_and_action_total(
        env, rng, rollout_length=2
    )

    state, observations = env.init(rng, "staggered", rollout_length=2)

    np.testing.assert_array_equal(state["steps"], expected_offsets)
    np.testing.assert_allclose(state["action_total"], expected_action_total)
    np.testing.assert_allclose(
        observations,
        jnp.stack(
            (expected_offsets.astype(jnp.float32), expected_action_total), axis=-1
        ),
    )


def test_init_staggered_samples_valid_discrete_actions() -> None:
    env = FakeVectorEnvironment(num_envs=7, max_episode_length=6, discrete=True)
    rng = jax.random.PRNGKey(17)
    expected_offsets, expected_action_total = _expected_offsets_and_action_total(
        env, rng, rollout_length=2
    )

    state, observations = env.init(rng, "staggered", rollout_length=2)

    np.testing.assert_array_equal(state["steps"], expected_offsets)
    np.testing.assert_array_equal(state["action_total"], expected_action_total)
    np.testing.assert_allclose(
        observations,
        jnp.stack(
            (expected_offsets.astype(jnp.float32), expected_action_total), axis=-1
        ),
    )


def test_init_staggered_respects_action_repeat_safe_maximum() -> None:
    env = FakeVectorEnvironment(num_envs=128, max_episode_length=10, action_repeat=4)

    state, _ = env.init(jax.random.PRNGKey(2), "staggered", rollout_length=2)

    np.testing.assert_array_equal(state["steps"], np.zeros(env.num_envs))


def test_init_staggered_is_jittable() -> None:
    env = FakeVectorEnvironment(num_envs=7, max_episode_length=6)
    rng = jax.random.PRNGKey(3)

    eager_state, eager_observations = env.init(rng, "staggered", rollout_length=2)
    jitted_state, jitted_observations = jax.jit(
        lambda key: env.init(key, "staggered", rollout_length=2)
    )(rng)

    np.testing.assert_array_equal(jitted_state["steps"], eager_state["steps"])
    np.testing.assert_allclose(
        jitted_state["action_total"], eager_state["action_total"]
    )
    np.testing.assert_allclose(jitted_observations, eager_observations)


@pytest.mark.parametrize("rollout_length", [None, 0, -1])
def test_init_staggered_requires_positive_rollout_length(
    rollout_length: int | None,
) -> None:
    env = FakeVectorEnvironment(num_envs=4, max_episode_length=6)

    with pytest.raises(ValueError, match="rollout_length must be at least 1"):
        env.init(
            jax.random.PRNGKey(0),
            "staggered",
            rollout_length=rollout_length,
        )


def test_init_rejects_unknown_start_mode() -> None:
    env = FakeVectorEnvironment(num_envs=4, max_episode_length=6)

    with pytest.raises(ValueError, match="Unknown environment start mode"):
        env.init(
            jax.random.PRNGKey(0),
            cast(EnvStartMode, "unknown"),
            rollout_length=2,
        )
