import abc
from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Literal, NamedTuple, NotRequired, TypedDict, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from mixed_precision_rl.types import (
    Action,
    Done,
    EnvState,
    LogDict,
    Observation,
    Reward,
)

Params = TypeVar("Params")
AgentState = TypeVar("AgentState")
type EnvStartMode = Literal["synchronized", "staggered"]


def reset_agent_state[State](
    initial_agent_state: State,
    agent_state: State,
    done: Done,
) -> State:
    def reset_if_done(initial_value, value):
        mask = done.reshape(done.shape + (1,) * (value.ndim - done.ndim))
        return jnp.where(mask, initial_value, value)

    return jax.tree.map(reset_if_done, initial_agent_state, agent_state)


class TimestepInfo(TypedDict):
    episode_return: Array
    episode_steps: Array
    achievements: NotRequired[dict[str, Array]]
    next_observation: NotRequired[Observation]


class Timestep(NamedTuple):
    observation: Observation
    reward: Reward
    terminated: Done
    truncated: Done
    info: TimestepInfo


@dataclass
class DiscreteActionSpace:
    num_actions: int


@dataclass
class ContinuousActionSpace:
    action_dim: int


class Environment(abc.ABC):
    def __init__(
        self,
        num_envs: int,
        num_eval_envs: int,
        max_episode_length: int,
        action_repeat: int,
        next_obs_in_extras: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.next_obs_in_extras = next_obs_in_extras

    @abc.abstractmethod
    def _init(
        self, rng: PRNGKeyArray, num_envs: int
    ) -> tuple[EnvState, Observation]: ...

    def init(
        self,
        rng: PRNGKeyArray,
        start_mode: EnvStartMode = "synchronized",
        rollout_length: int | None = None,
    ) -> tuple[EnvState, Observation]:
        """Initializes training environments, optionally at staggered phases."""
        env_states, observations = self._init(rng, self.num_envs)
        if start_mode == "synchronized":
            return env_states, observations
        if start_mode != "staggered":
            raise ValueError(f"Unknown environment start mode: {start_mode}")
        if rollout_length is None or rollout_length < 1:
            raise ValueError(
                "rollout_length must be at least 1 for staggered initialization"
            )
        return self._advance_to_offsets(
            env_states,
            observations,
            jax.random.fold_in(rng, 1),
            rollout_length,
        )

    @abc.abstractmethod
    def step(self, state: EnvState, action: Action) -> tuple[EnvState, Timestep]: ...

    def _advance_to_offsets(
        self,
        env_states: EnvState,
        observations: Observation,
        rng: PRNGKeyArray,
        rollout_length: int,
    ) -> tuple[EnvState, Observation]:
        """Assigns balanced offsets and reaches them using random actions."""
        num_phases = math.ceil(self.max_episode_length / self.action_repeat)
        num_groups = max(num_phases // rollout_length, 1)
        group_offsets = jnp.arange(num_groups, dtype=jnp.int32) * rollout_length
        group_indices = jnp.arange(self.num_envs, dtype=jnp.int32) % num_groups
        assignment_rng, action_rng = jax.random.split(rng)
        group_indices = jax.random.permutation(assignment_rng, group_indices)
        offsets: Int[Array, " env"] = group_offsets[group_indices]
        action_space = self.action_space

        def advance(carry, step):
            env_states, observations, action_rng = carry
            active = step < offsets
            action_rng, step_rng = jax.random.split(action_rng)
            if isinstance(action_space, ContinuousActionSpace):
                actions = jax.random.uniform(
                    step_rng,
                    (self.num_envs, action_space.action_dim),
                    minval=-1.0,
                    maxval=1.0,
                )
            else:
                actions = jax.random.randint(
                    step_rng,
                    (self.num_envs,),
                    minval=0,
                    maxval=action_space.num_actions,
                    dtype=jnp.int32,
                )
            next_env_states, timestep = self.step(env_states, actions)

            def take_active(next_value, value):
                if next_value.ndim == 0 or next_value.shape[0] != self.num_envs:
                    return next_value
                mask = active.reshape(
                    active.shape + (1,) * (next_value.ndim - active.ndim)
                )
                return jnp.where(mask, next_value, value)

            env_states = jax.tree.map(take_active, next_env_states, env_states)
            observations = jax.tree.map(take_active, timestep.observation, observations)
            return (env_states, observations, action_rng), None

        (env_states, observations, _), _ = jax.lax.scan(
            advance,
            (env_states, observations, action_rng),
            jnp.arange((num_groups - 1) * rollout_length, dtype=jnp.int32),
        )
        return env_states, observations

    @abc.abstractmethod
    def make_evaluation(
        self,
        agent: Callable[
            [Observation, PRNGKeyArray, Params, AgentState],
            tuple[AgentState, Action],
        ],
        initial_agent_state: AgentState,
    ) -> Callable[[PRNGKeyArray, Params], LogDict]: ...

    @property
    @abc.abstractmethod
    def action_space(self) -> DiscreteActionSpace | ContinuousActionSpace: ...

    @property
    def reward_bounds(self) -> tuple[float, float]:
        return -jnp.inf, jnp.inf
