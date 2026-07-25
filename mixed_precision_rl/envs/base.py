import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, NotRequired, TypedDict, TypeVar

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from mixed_precision_rl.types import (
    Action,
    Done,
    EnvState,
    LogDict,
    Observation,
    Reward,
)

Params = TypeVar("Params")


class TimestepInfo(TypedDict):
    episode_return: Array
    episode_steps: Array
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
    def init(self, rng: PRNGKeyArray) -> tuple[EnvState, Observation]: ...

    @abc.abstractmethod
    def step(self, state: EnvState, action: Action) -> tuple[EnvState, Timestep]: ...

    @abc.abstractmethod
    def make_evaluation(
        self, agent: Callable[[Observation, PRNGKeyArray, Params], Action]
    ) -> Callable[[PRNGKeyArray, Params], LogDict]: ...

    @property
    @abc.abstractmethod
    def action_space(self) -> DiscreteActionSpace | ContinuousActionSpace: ...

    @property
    def reward_bounds(self) -> tuple[float, float]:
        return -jnp.inf, jnp.inf
