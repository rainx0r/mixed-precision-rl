from collections.abc import Callable
import math
from typing import Literal, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from mixed_precision_rl.envs.base import (
    ContinuousActionSpace,
    Environment,
    reset_agent_state,
    Timestep,
    TimestepInfo,
)
from mixed_precision_rl.types import Action, EnvState, LogDict, Observation

Params = TypeVar("Params")
AgentState = TypeVar("AgentState")


class DMCEnv(Environment):
    def __init__(
        self,
        env_name: str,
        num_envs: int,
        num_eval_envs: int,
        max_episode_length: int,
        action_repeat: int,
        impl: Literal["jax", "warp"] = "warp",
        warp_kernel_cache_dir: str | None = None,
        next_obs_in_extras: bool = False,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            max_episode_length=max_episode_length,
            action_repeat=action_repeat,
            next_obs_in_extras=next_obs_in_extras,
        )

        self.env_name = env_name
        self.impl = impl
        self.warp_kernel_cache_dir = warp_kernel_cache_dir

        import mujoco
        import mujoco_playground as mjp

        # Playground 0.2.0 compares MuJoCo versions as strings, so 3.10 is
        # incorrectly sent down the pre-3.3 Reacher compatibility path.
        mj_spec = vars(mujoco)["MjSpec"]
        if not hasattr(mj_spec, "find_body"):
            mj_spec.find_body = mj_spec.body

        if self.warp_kernel_cache_dir is not None:
            import warp as wp

            wp.config.kernel_cache_dir = self.warp_kernel_cache_dir

        config = mjp.registry.get_default_config(self.env_name)
        if self.impl == "warp" and self.env_name.startswith("Reacher"):
            config.njmax = max(config.njmax, 1)
        self._env = mjp.registry.load(
            self.env_name,
            config=config,
            config_overrides={"impl": self.impl},
        )

    def _init(self, rng: PRNGKeyArray, num_envs: int) -> tuple[EnvState, Observation]:
        rngs = jax.random.split(rng, num_envs)
        reset_rngs = jax.vmap(jax.random.split)(rngs)[:, 1]
        states = jax.vmap(self._env.reset)(reset_rngs)

        info = dict(states.info)
        info["_dmc_first_data"] = states.data
        info["_dmc_first_obs"] = states.obs
        info["_dmc_steps"] = jnp.zeros(num_envs, dtype=jnp.int32)
        info["_dmc_episode_return"] = jnp.zeros_like(states.reward)
        states = states.replace(info=info)  # ty: ignore[unresolved-attribute]
        return states, states.obs

    def init(self, rng: PRNGKeyArray) -> tuple[EnvState, Observation]:
        return self._init(rng, self.num_envs)

    def step(self, state: EnvState, action: Action) -> tuple[EnvState, Timestep]:
        state = state.replace(info=dict(state.info))
        was_done = state.done.astype(jnp.bool_)
        steps = jnp.where(was_done, 0, state.info["_dmc_steps"])
        episode_return = jnp.where(was_done, 0.0, state.info["_dmc_episode_return"])
        state = state.replace(done=jnp.zeros_like(state.done))

        def repeat_step(state, _):
            state = jax.vmap(self._env.step)(state, action)
            return state, state.reward

        state, rewards = jax.lax.scan(
            repeat_step, state, None, length=self.action_repeat
        )
        reward = rewards.sum(axis=0)
        next_observation = state.obs

        terminated = state.done.astype(jnp.bool_)
        steps = steps + self.action_repeat
        truncated = (steps >= self.max_episode_length) & ~terminated
        done = terminated | truncated
        episode_return = episode_return + reward

        state.info["_dmc_steps"] = steps
        state.info["_dmc_episode_return"] = episode_return

        def reset_if_done(reset_value, current_value):
            if current_value.ndim == 0 or current_value.shape[0] != done.shape[0]:
                return current_value
            mask = done.reshape(done.shape + (1,) * (current_value.ndim - done.ndim))
            return jnp.where(mask, reset_value, current_value)

        data = jax.tree.map(reset_if_done, state.info["_dmc_first_data"], state.data)
        observation = jax.tree.map(
            reset_if_done, state.info["_dmc_first_obs"], next_observation
        )
        state = state.replace(
            data=data,
            obs=observation,
            reward=reward,
            done=done.astype(state.done.dtype),
        )

        info: TimestepInfo = {
            **{
                key: value
                for key, value in state.info.items()
                if not key.startswith("_dmc_")
            },
            "episode_return": episode_return,
            "episode_steps": steps,
        }
        if self.next_obs_in_extras:
            info["next_observation"] = next_observation

        return state, Timestep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

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
            env_states, observations = self._init(rng, self.num_eval_envs)
            rng = jax.random.fold_in(rng, 1)
            episode_returns = jnp.zeros(self.num_eval_envs, dtype=jnp.float32)
            active = jnp.ones(self.num_eval_envs, dtype=jnp.bool_)

            def step(carry, _):
                (
                    env_states,
                    observations,
                    episode_returns,
                    active,
                    rng,
                    agent_state,
                ) = carry
                rng, action_rng = jax.random.split(rng)
                agent_state, actions = agent(
                    observations, action_rng, params, agent_state
                )
                env_states, timestep = self.step(env_states, actions)
                done = timestep.terminated | timestep.truncated
                agent_state = reset_agent_state(initial_agent_state, agent_state, done)
                episode_returns = episode_returns + active * timestep.reward
                active = active & ~done
                return (
                    env_states,
                    timestep.observation,
                    episode_returns,
                    active,
                    rng,
                    agent_state,
                ), None

            (_, _, episode_returns, _, _, _), _ = jax.lax.scan(
                step,
                (
                    env_states,
                    observations,
                    episode_returns,
                    active,
                    rng,
                    initial_agent_state,
                ),
                None,
                length=math.ceil(self.max_episode_length / self.action_repeat),
            )
            return {"eval/episode_reward": episode_returns.mean()}

        return evaluate

    @property
    def action_space(self) -> ContinuousActionSpace:
        return ContinuousActionSpace(action_dim=self._env.action_size)

    @property
    def reward_bounds(self) -> tuple[float, float]:
        return 0.0, float(self.action_repeat)
