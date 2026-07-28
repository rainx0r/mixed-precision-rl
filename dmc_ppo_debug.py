from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, cast, Literal, NamedTuple, TypedDict

import distrax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
import optax

import tyro

from mixed_precision_rl.envs.base import Environment
from mixed_precision_rl.envs.dmc import DMCEnv
from mixed_precision_rl.types import DType, EnvState, Observation
from mixed_precision_rl.utils import (
    init_running_mean_std,
    normalize,
    RunningMeanStd,
    update_running_mean_std,
)
import wandb


class RolloutExtras(TypedDict):
    log_probs: Float[Array, "time env"]
    episode_returns: Float[Array, "time env"]
    episode_steps: Int[Array, "time env"]


class Rollout(NamedTuple):
    observations: Float[Array, "time env observation"]
    actions: Float[Array, "time env action"]
    rewards: Float[Array, "time env"]
    terminations: Bool[Array, "time env"]
    truncations: Bool[Array, "time env"]
    next_observations: Float[Array, "time env observation"]
    extras: RolloutExtras


class ExperimentState(NamedTuple):
    policy: TrainState
    vf: TrainState
    normalizer: RunningMeanStd
    env_states: EnvState
    observations: Observation
    key: PRNGKeyArray


class Policy(nn.Module):
    output_dim: int
    width: int = 32
    depth: int = 4
    dtype: DTypeLike = jnp.float32
    activation_fn: Callable[[jax.Array], jax.Array] = nn.swish
    min_std: float = 1e-3
    var_scale: float = 1.0

    @nn.compact
    def __call__(
        self, x: Float[Array, "... observation"]
    ) -> distrax.MultivariateNormalDiag:
        for _ in range(self.depth):
            x = nn.Dense(
                self.width,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(x)
            x = self.activation_fn(x)

        x = nn.Dense(
            2 * self.output_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )(x)
        mean, std = jnp.split(x, 2, axis=-1)
        mean, std = mean.astype(jnp.float32), std.astype(jnp.float32)
        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale
        return distrax.MultivariateNormalDiag(mean, std)


class ValueFunction(nn.Module):
    width: int = 256
    depth: int = 5
    output_dim: int = 1
    dtype: DTypeLike = jnp.float32
    activation_fn: Callable[[jax.Array], jax.Array] = nn.swish

    @nn.compact
    def __call__(self, x: Float[Array, "... observation"]) -> Float[Array, "... value"]:
        for _ in range(self.depth):
            x = nn.Dense(
                self.width,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(x)
            x = self.activation_fn(x)

        x = nn.Dense(
            self.output_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )(x)
        return x.astype(jnp.float32)


def make_optimizer(
    learning_rate: float, max_grad_norm: float | None
) -> optax.GradientTransformation:
    transforms = []
    if max_grad_norm is not None:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(optax.adam(learning_rate))
    return optax.chain(*transforms)


def tree_l2_norm(tree: Any) -> jax.Array:
    return jnp.sqrt(
        sum(
            jnp.square(leaf.astype(jnp.float32)).sum() for leaf in jax.tree.leaves(tree)
        )
    )


def compute_gae(
    rewards: Float[Array, "time env"],
    values: Float[Array, "time env"],
    next_values: Float[Array, "time env"],
    gamma: float,
    gae_lambda: float,
    terminations: Bool[Array, "time env"],
    truncations: Bool[Array, "time env"],
) -> tuple[Float[Array, "time env"], Float[Array, "time env"]]:
    not_terminated = 1 - terminations.astype(values.dtype)
    not_truncated = 1 - truncations.astype(values.dtype)
    deltas = rewards + gamma * not_terminated * next_values - values

    def compute_advantage(acc, transition):
        delta, not_done = transition
        acc = delta + gamma * gae_lambda * not_done * acc
        return acc, acc

    _, advantages = jax.lax.scan(
        compute_advantage,
        jnp.zeros_like(values[-1]),
        (deltas, not_terminated * not_truncated),
        reverse=True,
    )
    return advantages + values, advantages


def ppo_update(
    policy: TrainState,
    vf: TrainState,
    data: Rollout,
    key: PRNGKeyArray,
    entropy_coeff: float,
    vf_coeff: float,
    gamma: float,
    gae_lambda: float,
    reward_scale: float,
    clip_eps: float,
    value_clip_eps: float | None,
    norm_advantage: bool,
    num_minibatches: int,
    num_epochs: int,
) -> tuple[
    tuple[TrainState, TrainState, PRNGKeyArray],
    dict[str, Float[Array, "epoch minibatch"]],
]:
    old_values = vf.apply_fn(vf.params, data.observations).squeeze(-1)
    next_values = vf.apply_fn(vf.params, data.next_observations).squeeze(-1)
    value_targets, advantages = compute_gae(
        data.rewards * reward_scale,
        old_values,
        next_values,
        gamma,
        gae_lambda,
        data.terminations,
        data.truncations,
    )

    def loss(
        policy_and_vf_params,
        data: Rollout,
        advantages: jax.Array,
        value_targets: jax.Array,
        old_values: jax.Array,
        entropy_key: PRNGKeyArray,
    ):
        policy_params, vf_params = policy_and_vf_params

        if norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        distribution = policy.apply_fn(policy_params, data.observations)
        forward_log_det_jacobian = 2.0 * (
            jnp.log(2.0) - data.actions - jax.nn.softplus(-2.0 * data.actions)
        )
        log_probs = distribution.log_prob(data.actions) - (
            forward_log_det_jacobian.sum(axis=-1)
        )
        log_ratio = log_probs - data.extras["log_probs"]
        ratio = jnp.exp(log_ratio)
        policy_loss = -jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages,
        ).mean()

        entropy_actions = distribution.sample(seed=entropy_key)
        entropy_forward_log_det_jacobian = 2.0 * (
            jnp.log(2.0) - entropy_actions - jax.nn.softplus(-2.0 * entropy_actions)
        )
        entropy = (
            distribution.entropy() + entropy_forward_log_det_jacobian.sum(axis=-1)
        ).mean()
        entropy_loss = -entropy_coeff * entropy

        predicted_values = vf.apply_fn(vf_params, data.observations).squeeze(-1)
        vf_loss = jnp.square(value_targets - predicted_values)
        if value_clip_eps is not None:
            clipped_values = old_values + jnp.clip(
                predicted_values - old_values,
                -value_clip_eps,
                value_clip_eps,
            )
            clipped_vf_loss = jnp.square(value_targets - clipped_values)
            vf_loss = jnp.maximum(vf_loss, clipped_vf_loss)
        vf_loss = vf_coeff * (0.5 * vf_loss.mean())

        total_loss = policy_loss + vf_loss + entropy_loss
        policy_std = distribution.stddev()
        return total_loss, {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "vf_loss": vf_loss,
            "entropy": entropy,
            "entropy_loss": entropy_loss,
            "debug/approx_kl": ((ratio - 1) - log_ratio).mean(),
            "debug/clip_fraction": (jnp.abs(ratio - 1) > clip_eps).mean(),
            "debug/ratio_max": ratio.max(),
            "debug/policy_std_min": policy_std.min(),
            "debug/policy_std_max": policy_std.max(),
            "debug/value_prediction_mean": predicted_values.mean(),
            "debug/value_prediction_std": predicted_values.std(),
            "debug/value_prediction_min": predicted_values.min(),
            "debug/value_prediction_max": predicted_values.max(),
        }

    def update_minibatch(carry, xs):
        policy, vf, key = carry
        key, entropy_key = jax.random.split(key)
        (
            minibatch,
            minibatch_advantages,
            minibatch_value_targets,
            minibatch_old_values,
        ) = xs

        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(
            (policy.params, vf.params),
            minibatch,
            minibatch_advantages,
            minibatch_value_targets,
            minibatch_old_values,
            entropy_key,
        )
        policy_param_norm = tree_l2_norm(policy.params)
        vf_param_norm = tree_l2_norm(vf.params)
        policy_grad_norm = tree_l2_norm(grads[0])
        vf_grad_norm = tree_l2_norm(grads[1])
        policy = policy.apply_gradients(grads=grads[0])
        vf = vf.apply_gradients(grads=grads[1])
        metrics.update(
            {
                "debug/policy_grad_norm": policy_grad_norm,
                "debug/vf_grad_norm": vf_grad_norm,
                "debug/policy_param_norm": policy_param_norm,
                "debug/vf_param_norm": vf_param_norm,
            }
        )
        return (policy, vf, key), metrics

    def update_epoch(carry, _):
        policy, vf, key = carry
        key, permutation_key = jax.random.split(key)

        def shuffle(x: jax.Array):
            x = x.reshape(-1, *x.shape[2:])
            x = jax.random.permutation(permutation_key, x, axis=0)
            return x.reshape(num_minibatches, -1, *x.shape[1:])

        (policy, vf, key), metrics = jax.lax.scan(
            update_minibatch,
            (policy, vf, key),
            jax.tree.map(shuffle, (data, advantages, value_targets, old_values)),
            length=num_minibatches,
        )
        return (policy, vf, key), metrics

    carry, metrics = jax.lax.scan(
        update_epoch,
        (policy, vf, key),
        None,
        length=num_epochs,
    )
    metric_shape = metrics["total_loss"].shape
    metrics.update(
        {
            "debug/advantage_std": jnp.broadcast_to(advantages.std(), metric_shape),
            "debug/advantage_min": jnp.broadcast_to(advantages.min(), metric_shape),
            "debug/advantage_max": jnp.broadcast_to(advantages.max(), metric_shape),
            "debug/value_target_std": jnp.broadcast_to(
                value_targets.std(), metric_shape
            ),
            "debug/value_target_min": jnp.broadcast_to(
                value_targets.min(), metric_shape
            ),
            "debug/value_target_max": jnp.broadcast_to(
                value_targets.max(), metric_shape
            ),
        }
    )
    return carry, metrics


def rollout(
    policy: TrainState,
    normalizer: RunningMeanStd,
    envs: Environment,
    env_states: EnvState,
    observations: Observation,
    key: PRNGKeyArray,
    rollout_length: int,
    observation_dtype: DTypeLike = jnp.float32,
) -> tuple[RunningMeanStd, EnvState, Observation, PRNGKeyArray, Rollout]:
    def step(carry, _):
        env_states, observations, key, next_normalizer = carry
        key, action_key = jax.random.split(key)

        normalized_observations = normalize(observations, normalizer)
        distribution = policy.apply_fn(policy.params, normalized_observations)
        raw_action, base_log_prob = distribution.sample_and_log_prob(seed=action_key)
        forward_log_det_jacobian = 2.0 * (
            jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)
        )
        log_prob = base_log_prob - forward_log_det_jacobian.sum(axis=-1)
        action = jnp.tanh(raw_action)

        next_state, timestep = envs.step(env_states, action)
        next_observation = timestep.info["next_observation"]
        data = Rollout(
            observations=normalized_observations.astype(observation_dtype),
            actions=raw_action,
            rewards=timestep.reward,
            terminations=timestep.terminated,
            truncations=timestep.truncated,
            next_observations=normalize(next_observation, normalizer).astype(
                observation_dtype
            ),
            extras={
                "log_probs": log_prob,
                "episode_returns": timestep.info["episode_return"],
                "episode_steps": timestep.info["episode_steps"],
            },
        )

        next_normalizer = update_running_mean_std(next_normalizer, observations)
        carry = next_state, timestep.observation, key, next_normalizer
        return carry, data

    (env_states, observations, key, normalizer), data = jax.lax.scan(
        step,
        (env_states, observations, key, normalizer),
        None,
        length=rollout_length,
    )
    return normalizer, env_states, observations, key, data


@dataclass
class Args:
    SEED: int = 43
    ENV_NAME: str = "CheetahRun"
    ENV_IMPL: Literal["jax", "warp"] = "warp"
    EPISODE_LENGTH: int = 1000
    ACTION_REPEAT: int = 1
    WARP_KERNEL_CACHE_DIR: str | None = "/tmp/warp-cache"
    MATMUL_PRECISION: Literal["default", "high", "highest"] = "highest"
    COMPUTE_DTYPE: DType = DType.float32
    ROLLOUT_OBSERVATION_DTYPE: DType = DType.float32

    TOTAL_TIMESTEPS: int = 60_000_000
    NUM_ENVS: int = 2048
    ROLLOUT_LENGTH: int = 480
    NUM_MINIBATCHES: int = 32
    NUM_EPOCHS: int = 16

    LEARNING_RATE: float = 1e-3
    MAX_GRAD_NORM: float | None = None
    ENTROPY_COEFF: float = 1e-2
    VF_COEFF: float = 0.5
    GAMMA: float = 0.995
    GAE_LAMBDA: float = 0.95
    REWARD_SCALE: float = 10.0
    CLIP_EPS: float = 0.3
    VALUE_CLIP_EPS: float | None = None
    NORM_ADVANTAGE: bool = True

    EVAL_FREQUENCY: int = 6_000_000
    NUM_EVAL_ENVS: int = 128

    WANDB_ENTITY: str = "evangelos-ch"
    WANDB_PROJECT: str = "mixed-precision-rl"
    WANDB_MODE: Literal["online", "offline", "disabled", "shared"] = "online"
    WANDB_RUN_NAME: str | None = None


def main(args: Args) -> None:
    jax.config.update("jax_default_matmul_precision", args.MATMUL_PRECISION)

    transitions_per_update = args.NUM_ENVS * args.ROLLOUT_LENGTH
    if transitions_per_update % args.NUM_MINIBATCHES:
        raise ValueError(
            "NUM_ENVS * ROLLOUT_LENGTH must be divisible by NUM_MINIBATCHES"
        )

    environment_steps_per_update = transitions_per_update * args.ACTION_REPEAT
    num_updates = math.ceil(args.TOTAL_TIMESTEPS / environment_steps_per_update)
    actual_timesteps = num_updates * environment_steps_per_update

    run = wandb.init(
        entity=args.WANDB_ENTITY,
        project=args.WANDB_PROJECT,
        name=args.WANDB_RUN_NAME
        or f"ppo_debug_{args.ENV_NAME}_{args.ENV_IMPL}_{args.SEED}",
        tags=["ppo", "debug", args.ENV_NAME, args.COMPUTE_DTYPE.name],
        mode=args.WANDB_MODE,
        config={
            **asdict(args),
            "ACTUAL_TIMESTEPS": actual_timesteps,
            "TRANSITIONS_PER_UPDATE": transitions_per_update,
            "DEBUG_INSTRUMENTATION_VERSION": 2,
        },
    )

    envs = DMCEnv(
        env_name=args.ENV_NAME,
        num_envs=args.NUM_ENVS,
        num_eval_envs=args.NUM_EVAL_ENVS,
        max_episode_length=args.EPISODE_LENGTH,
        action_repeat=args.ACTION_REPEAT,
        impl=args.ENV_IMPL,
        warp_kernel_cache_dir=args.WARP_KERNEL_CACHE_DIR,
        next_obs_in_extras=True,
    )

    key, key_policy, key_value, key_env, eval_key = jax.random.split(
        jax.random.PRNGKey(args.SEED), 5
    )
    env_states, observations = jax.jit(envs.init)(key_env)
    obs_spec = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), observations
    )

    policy_module = Policy(
        output_dim=envs.action_space.action_dim,
        dtype=args.COMPUTE_DTYPE(),
    )
    policy = TrainState.create(
        apply_fn=policy_module.apply,
        params=policy_module.lazy_init(key_policy, obs_spec),
        tx=make_optimizer(args.LEARNING_RATE, args.MAX_GRAD_NORM),
    )
    vf_module = ValueFunction(dtype=args.COMPUTE_DTYPE())
    vf = TrainState.create(
        apply_fn=vf_module.apply,
        params=vf_module.lazy_init(key_value, obs_spec),
        tx=make_optimizer(args.LEARNING_RATE, args.MAX_GRAD_NORM),
    )
    normalizer = init_running_mean_std(obs_spec)
    state = ExperimentState(
        policy=policy,
        vf=vf,
        normalizer=normalizer,
        env_states=env_states,
        observations=observations,
        key=key,
    )

    def evaluation_agent(observations, rng, params, agent_state):
        policy_params, observation_normalizer = params
        distribution = cast(
            distrax.MultivariateNormalDiag,
            policy_module.apply(
                policy_params,
                normalize(observations, observation_normalizer),
            ),
        )
        return agent_state, jnp.tanh(distribution.sample(seed=rng))

    def deterministic_evaluation_agent(observations, rng, params, agent_state):
        del rng
        policy_params, observation_normalizer = params
        distribution = cast(
            distrax.MultivariateNormalDiag,
            policy_module.apply(
                policy_params,
                normalize(observations, observation_normalizer),
            ),
        )
        return agent_state, jnp.tanh(distribution.mean())

    evaluation = (
        envs.make_evaluation(evaluation_agent, None)
        if args.EVAL_FREQUENCY > 0
        else None
    )
    deterministic_evaluation = (
        envs.make_evaluation(deterministic_evaluation_agent, None)
        if args.EVAL_FREQUENCY > 0
        else None
    )

    @jax.jit
    def train_step(
        state: ExperimentState,
    ) -> tuple[ExperimentState, dict[str, jax.Array]]:
        normalizer, env_states, observations, key, data = rollout(
            state.policy,
            state.normalizer,
            envs,
            state.env_states,
            state.observations,
            state.key,
            args.ROLLOUT_LENGTH,
            args.ROLLOUT_OBSERVATION_DTYPE(),
        )

        (policy, vf, key), metrics = ppo_update(
            state.policy,
            state.vf,
            data,
            key,
            args.ENTROPY_COEFF,
            args.VF_COEFF,
            args.GAMMA,
            args.GAE_LAMBDA,
            args.REWARD_SCALE,
            args.CLIP_EPS,
            args.VALUE_CLIP_EPS,
            args.NORM_ADVANTAGE,
            args.NUM_MINIBATCHES,
            args.NUM_EPOCHS,
        )

        done = data.terminations | data.truncations
        completed_episodes = done.sum()
        logs = jax.tree.map(jnp.mean, metrics)
        observation_values = data.observations.astype(jnp.float32)
        observation_norms = jnp.linalg.norm(observation_values, axis=-1)
        logs.update(
            {
                "debug/approx_kl_max_over_updates": metrics["debug/approx_kl"].max(),
                "debug/ratio_max": metrics["debug/ratio_max"].max(),
                "debug/policy_std_min": metrics["debug/policy_std_min"].min(),
                "debug/policy_std_max": metrics["debug/policy_std_max"].max(),
                "debug/advantage_min": metrics["debug/advantage_min"].min(),
                "debug/advantage_max": metrics["debug/advantage_max"].max(),
                "debug/value_prediction_min": metrics[
                    "debug/value_prediction_min"
                ].min(),
                "debug/value_prediction_max": metrics[
                    "debug/value_prediction_max"
                ].max(),
                "debug/value_target_min": metrics["debug/value_target_min"].min(),
                "debug/value_target_max": metrics["debug/value_target_max"].max(),
                "debug/observation_norm_mean": observation_norms.mean(),
                "debug/observation_norm_max": observation_norms.max(),
                "debug/observation_min": observation_values.min(),
                "debug/observation_max": observation_values.max(),
                "debug/observation_std": observation_values.std(),
            }
        )
        logs.update(
            completed_episodes=completed_episodes,
            episode_return_sum=jnp.where(
                done, data.extras["episode_returns"], 0.0
            ).sum(),
            episode_steps_sum=jnp.where(done, data.extras["episode_steps"], 0).sum(),
        )
        return (
            ExperimentState(
                policy=policy,
                vf=vf,
                normalizer=normalizer,
                env_states=env_states,
                observations=observations,
                key=key,
            ),
            logs,
        )

    if evaluation is not None and deterministic_evaluation is not None:
        eval_key, evaluation_key = jax.random.split(eval_key)
        eval_logs = jax.device_get(
            evaluation(evaluation_key, (state.policy.params, state.normalizer))
        )
        deterministic_eval_logs = jax.device_get(
            deterministic_evaluation(
                evaluation_key, (state.policy.params, state.normalizer)
            )
        )
        host_eval_logs = {name: float(value) for name, value in eval_logs.items()}
        host_eval_logs.update(
            {
                f"debug/deterministic_{name.removeprefix('eval/')}": float(value)
                for name, value in deterministic_eval_logs.items()
            }
        )
        run.log(
            host_eval_logs,
            step=0,
        )
        next_eval_step = args.EVAL_FREQUENCY
    else:
        next_eval_step = math.inf

    start_time = time.monotonic()
    environment_steps = 0
    for _ in range(num_updates):
        state, logs = train_step(state)
        logs = jax.device_get(logs)
        environment_steps += environment_steps_per_update
        elapsed = time.monotonic() - start_time

        completed = int(logs.pop("completed_episodes"))
        episode_return_sum = float(logs.pop("episode_return_sum"))
        episode_steps_sum = float(logs.pop("episode_steps_sum"))
        host_logs = {
            name if name.startswith("debug/") else f"training/{name}": float(value)
            for name, value in logs.items()
        }
        host_logs["training/sps"] = environment_steps / elapsed
        if completed:
            host_logs["episode/return"] = episode_return_sum / completed
            host_logs["episode/steps"] = episode_steps_sum / completed

        summary = (
            f"{environment_steps}: "
            f"loss={host_logs['training/total_loss']:.3f}, "
            f"sps={host_logs['training/sps']:.0f}"
        )
        if (
            environment_steps >= next_eval_step
            and evaluation is not None
            and deterministic_evaluation is not None
        ):
            eval_key, evaluation_key = jax.random.split(eval_key)
            eval_logs = jax.device_get(
                evaluation(evaluation_key, (state.policy.params, state.normalizer))
            )
            host_logs.update({name: float(value) for name, value in eval_logs.items()})
            deterministic_eval_logs = jax.device_get(
                deterministic_evaluation(
                    evaluation_key, (state.policy.params, state.normalizer)
                )
            )
            host_logs.update(
                {
                    f"debug/deterministic_{name.removeprefix('eval/')}": float(value)
                    for name, value in deterministic_eval_logs.items()
                }
            )
            summary += f", eval_reward={host_logs['eval/episode_reward']:.1f}"
            summary += (
                ", deterministic_eval_reward="
                f"{host_logs['debug/deterministic_episode_reward']:.1f}"
            )
            while next_eval_step <= environment_steps:
                next_eval_step += args.EVAL_FREQUENCY
        if completed:
            summary += f", episode_return={host_logs['episode/return']:.1f}"

        run.log(host_logs, step=environment_steps)
        print(summary)

    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
