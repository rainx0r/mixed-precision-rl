from collections.abc import Callable
from typing import NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from craftax.craftax_env import make_craftax_env_from_name

from mixed_precision_rl.envs.base import (
    DiscreteActionSpace,
    Environment,
    reset_agent_state,
    Timestep,
    TimestepInfo,
)
from mixed_precision_rl.types import Action, EnvState, LogDict, Observation

Params = TypeVar("Params")
AgentState = TypeVar("AgentState")


class CraftaxEnvState(NamedTuple):
    env_state: EnvState
    rng: PRNGKeyArray
    episode_return: Array
    episode_steps: Array


class CraftaxEnv(Environment):
    def __init__(
        self,
        env_name: str = "Craftax-Symbolic-v1",
        num_envs: int = 1024,
        num_eval_envs: int = 128,
        max_episode_length: int = 100_000,
        action_repeat: int = 1,
        reset_ratio: int = 16,
        next_obs_in_extras: bool = True,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            max_episode_length=max_episode_length,
            action_repeat=action_repeat,
            next_obs_in_extras=next_obs_in_extras,
        )
        if num_envs % reset_ratio:
            raise ValueError(
                f"reset_ratio ({reset_ratio}) must divide num_envs ({num_envs})"
            )
        if num_eval_envs % reset_ratio:
            raise ValueError(
                f"reset_ratio ({reset_ratio}) must divide num_eval_envs "
                f"({num_eval_envs})"
            )

        self.env_name = env_name
        self.reset_ratio = reset_ratio
        self._env = make_craftax_env_from_name(env_name, auto_reset=False)
        self._env_params = self._env.default_params.replace(
            max_timesteps=max_episode_length
        )
        self._vmap_reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self._vmap_step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

        if "Classic" in env_name:
            from craftax.craftax_classic.constants import Achievement
        else:
            from craftax.craftax.constants import Achievement

        self.achievement_names = tuple(
            f"Achievements/{achievement.name.lower()}" for achievement in Achievement
        )

    def _init(
        self, rng: PRNGKeyArray, num_envs: int
    ) -> tuple[CraftaxEnvState, Observation]:
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        observations, env_state = self._vmap_reset(reset_rngs, self._env_params)
        state = CraftaxEnvState(
            env_state=env_state,
            rng=rng,
            episode_return=jnp.zeros(num_envs, dtype=jnp.float32),
            episode_steps=jnp.zeros(num_envs, dtype=jnp.int32),
        )
        return state, observations

    def init(self, rng: PRNGKeyArray) -> tuple[CraftaxEnvState, Observation]:
        return self._init(rng, self.num_envs)

    def _step(
        self, state: CraftaxEnvState, action: Action, num_envs: int
    ) -> tuple[CraftaxEnvState, Timestep]:
        rng, step_rng, reset_rng, assignment_rng = jax.random.split(state.rng, 4)
        step_rngs = jax.random.split(step_rng, num_envs)
        (
            next_observation,
            stepped_env_state,
            reward,
            done,
            craftax_info,
        ) = self._vmap_step(
            step_rngs,
            state.env_state,
            action,
            self._env_params,
        )
        done = done.astype(jnp.bool_)

        episode_return = state.episode_return + reward
        episode_steps = state.episode_steps + 1

        reached_time_limit = (
            stepped_env_state.timestep >= self._env_params.max_timesteps
        )
        truncated = done & reached_time_limit
        terminated = done & ~truncated

        num_resets = num_envs // self.reset_ratio
        reset_rngs = jax.random.split(reset_rng, num_resets)
        reset_observation, reset_env_state = self._vmap_reset(
            reset_rngs, self._env_params
        )

        # TODO: Randomize the candidate-to-worker assignment. This intentionally
        # reproduces the index-based assignment used by the original Craftax
        # baseline and Stoa for now
        reset_indices = jnp.arange(num_resets).repeat(self.reset_ratio)
        randomly_selected_workers = jax.random.choice(
            assignment_rng,
            jnp.arange(num_envs),
            shape=(num_resets,),
            p=done,
            replace=False,
        )
        deterministically_selected_workers = jnp.argsort(done)[-num_resets:]
        selected_workers = jax.lax.select(
            done.astype(jnp.int32).sum() < num_resets,
            deterministically_selected_workers,
            randomly_selected_workers,
        )
        reset_indices = reset_indices.at[selected_workers].set(jnp.arange(num_resets))

        reset_env_state = jax.tree.map(lambda x: x[reset_indices], reset_env_state)
        reset_observation = reset_observation[reset_indices]

        def reset_if_done(reset_value, stepped_value):
            mask = done.reshape(done.shape + (1,) * (stepped_value.ndim - done.ndim))
            return jnp.where(mask, reset_value, stepped_value)

        env_state = jax.tree.map(reset_if_done, reset_env_state, stepped_env_state)
        observation = reset_if_done(reset_observation, next_observation)

        achievements = {
            name.lower(): craftax_info[name] for name in self.achievement_names
        }
        info: TimestepInfo = {
            "episode_return": episode_return,
            "episode_steps": episode_steps,
            "achievements": achievements,
        }
        if self.next_obs_in_extras:
            info["next_observation"] = next_observation

        next_state = CraftaxEnvState(
            env_state=env_state,
            rng=rng,
            episode_return=jnp.where(done, 0.0, episode_return),
            episode_steps=jnp.where(done, 0, episode_steps),
        )
        return next_state, Timestep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def step(
        self, state: CraftaxEnvState, action: Action
    ) -> tuple[CraftaxEnvState, Timestep]:
        return self._step(state, action, self.num_envs)

    def make_evaluation(
        self,
        agent: Callable[
            [Observation, PRNGKeyArray, Params, AgentState],
            tuple[AgentState, Action],
        ],
        initial_agent_state: AgentState,
    ) -> Callable[[PRNGKeyArray, Params], LogDict]:
        @jax.jit
        def evaluate(rng: PRNGKeyArray, params: Params) -> LogDict:
            env_state, observation = self._init(rng, self.num_eval_envs)
            action_rng = jax.random.fold_in(rng, 1)
            episode_returns = jnp.zeros(self.num_eval_envs, dtype=jnp.float32)
            episode_lengths = jnp.zeros(self.num_eval_envs, dtype=jnp.int32)
            achievements = {
                name.lower(): jnp.zeros(self.num_eval_envs, dtype=jnp.float32)
                for name in self.achievement_names
            }
            active = jnp.ones(self.num_eval_envs, dtype=jnp.bool_)

            def evaluation_not_finished(carry):
                step, _, _, _, _, _, active, _, _ = carry
                return (step < self.max_episode_length) & jnp.any(active)

            def evaluation_step(carry):
                (
                    step,
                    env_state,
                    observation,
                    episode_returns,
                    episode_lengths,
                    achievements,
                    active,
                    action_rng,
                    agent_state,
                ) = carry
                action_rng, sample_rng = jax.random.split(action_rng)
                agent_state, action = agent(
                    observation, sample_rng, params, agent_state
                )
                env_state, timestep = self._step(env_state, action, self.num_eval_envs)
                done = timestep.terminated | timestep.truncated
                agent_state = reset_agent_state(initial_agent_state, agent_state, done)
                episode_returns = episode_returns + active * timestep.reward
                episode_lengths = episode_lengths + active
                achievements = jax.tree.map(
                    lambda total, achieved: total + active * achieved,
                    achievements,
                    timestep.info["achievements"],
                )
                active = active & ~done
                return (
                    step + 1,
                    env_state,
                    timestep.observation,
                    episode_returns,
                    episode_lengths,
                    achievements,
                    active,
                    action_rng,
                    agent_state,
                )

            (
                _,
                _,
                _,
                episode_returns,
                episode_lengths,
                achievements,
                _,
                _,
                _,
            ) = jax.lax.while_loop(
                evaluation_not_finished,
                evaluation_step,
                (
                    jnp.array(0, dtype=jnp.int32),
                    env_state,
                    observation,
                    episode_returns,
                    episode_lengths,
                    achievements,
                    active,
                    action_rng,
                    initial_agent_state,
                ),
            )
            logs: LogDict = {
                "eval/episode_reward": episode_returns.mean(),
                "eval/episode_length": episode_lengths.mean(),
            }
            logs.update(
                {
                    f"eval/{name}": achievement.mean()
                    for name, achievement in achievements.items()
                }
            )
            return logs

        return evaluate

    @property
    def action_space(self) -> DiscreteActionSpace:
        action_space = self._env.action_space(self._env_params)
        return DiscreteActionSpace(num_actions=action_space.n)
