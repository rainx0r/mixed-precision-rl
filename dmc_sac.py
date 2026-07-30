from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
import math
import time
from typing import Any, cast, Literal, NamedTuple, Self

import distrax
from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
from jax.experimental import io_callback
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

EXPENSIVE_METRICS = (
    "actor_grad_norm",
    "critic_grad_norm",
    "alpha_grad_norm",
    "actor_params_norm",
    "critic_params_norm",
)


class Transition(NamedTuple):
    observations: Float[Array, "batch observation"]
    actions: Float[Array, "batch action"]
    rewards: Float[Array, " batch"]
    terminations: Bool[Array, " batch"]
    next_observations: Float[Array, "batch observation"]


class ReplayBuffer(struct.PyTreeNode):
    data: Transition
    position: Int[Array, ""]
    size: Int[Array, ""]

    @classmethod
    def init(
        cls, observation_spec: jax.ShapeDtypeStruct, action_dim: int, capacity: int
    ) -> Self:
        return cls(
            data=Transition(
                observations=jnp.zeros(
                    (capacity, *observation_spec.shape), observation_spec.dtype
                ),
                actions=jnp.zeros((capacity, action_dim), jnp.float32),
                rewards=jnp.zeros((capacity,), jnp.float32),
                terminations=jnp.zeros((capacity,), jnp.bool_),
                next_observations=jnp.zeros(
                    (capacity, *observation_spec.shape), observation_spec.dtype
                ),
            ),
            position=jnp.zeros((), jnp.int32),
            size=jnp.zeros((), jnp.int32),
        )

    def insert(self, transitions: Transition) -> Self:
        batch_size = transitions.rewards.shape[0]
        capacity = self.data.rewards.shape[0]
        indices = (self.position + jnp.arange(batch_size)) % capacity
        data = jax.tree.map(
            lambda buffer, values: buffer.at[indices].set(values),
            self.data,
            transitions,
        )
        return self.replace(
            data=data,
            position=(self.position + batch_size) % capacity,
            size=jnp.minimum(self.size + batch_size, capacity),
        )

    def sample(self, key: PRNGKeyArray, batch_size: int) -> Transition:
        indices = jax.random.randint(key, (batch_size,), 0, self.size)
        return jax.tree.map(lambda x: x[indices], self.data)


class CriticTrainState(TrainState):
    target_params: Any = None


class ExperimentState(struct.PyTreeNode):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    normalizer: RunningMeanStd
    replay_buffer: ReplayBuffer
    env_states: EnvState
    observations: Observation
    key: PRNGKeyArray


class Policy(nn.Module):
    output_dim: int
    width: int = 256
    depth: int = 2
    dtype: DTypeLike = jnp.float32
    activation_fn: Callable[[jax.Array], jax.Array] = nn.relu
    min_std: float = 1e-3
    var_scale: float = 1.0
    kernel_init: Callable[..., jax.Array] = nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(
        self, x: Float[Array, "... observation"]
    ) -> distrax.MultivariateNormalDiag:
        for _ in range(self.depth):
            x = nn.Dense(
                self.width,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
            )(x)
            x = self.activation_fn(x)

        x = nn.Dense(
            2 * self.output_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
        )(x)
        mean, std = jnp.split(x, 2, axis=-1)
        mean, std = mean.astype(jnp.float32), std.astype(jnp.float32)
        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale
        return distrax.MultivariateNormalDiag(mean, std)


class QValueFunction(nn.Module):
    width: int = 256
    depth: int = 2
    dtype: DTypeLike = jnp.float32
    activation_fn: Callable[[jax.Array], jax.Array] = nn.relu
    layer_norm: bool = True
    kernel_init: Callable[..., jax.Array] = nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(
        self,
        observations: Float[Array, "... observation"],
        actions: Float[Array, "... action"],
    ) -> Float[Array, " ..."]:
        x = jnp.concatenate((observations, actions), axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(
                self.width,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
            )(x)
            x = self.activation_fn(x)
            if self.layer_norm:
                x = nn.LayerNorm(dtype=self.dtype)(x)

        x = nn.Dense(
            1,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
        )(x)
        return x.squeeze(-1).astype(jnp.float32)


class Ensemble(nn.Module):
    net_cls: Callable[..., nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args) -> jax.Array:
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    def setup(self) -> None:
        self.log_alpha = self.param(
            "log_alpha",
            lambda _: jnp.asarray(math.log(self.initial_temperature), jnp.float32),
        )

    def __call__(self) -> Float[Array, ""]:
        return jnp.exp(self.log_alpha)


def make_optimizer(
    learning_rate: float, max_grad_norm: float | None
) -> optax.GradientTransformation:
    transforms = []
    if max_grad_norm is not None:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(optax.adam(learning_rate))
    return optax.chain(*transforms)


def sac_update(
    actor: TrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    replay_buffer: ReplayBuffer,
    normalizer: RunningMeanStd,
    key: PRNGKeyArray,
    batch_size: int,
    num_updates: int,
    gamma: float,
    reward_scale: float,
    tau: float,
    target_entropy: float,
    compute_expensive_metrics: Bool[Array, ""],
) -> tuple[
    tuple[TrainState, CriticTrainState, TrainState, PRNGKeyArray],
    dict[str, Float[Array, " update"]],
]:
    key, sample_key = jax.random.split(key)
    data = replay_buffer.sample(sample_key, batch_size * num_updates)
    # Keep raw observations in replay so the whole batch uses the same current stats.
    data = data._replace(
        observations=normalize(data.observations, normalizer),
        next_observations=normalize(data.next_observations, normalizer),
    )
    data = jax.tree.map(
        lambda x: x.reshape(num_updates, batch_size, *x.shape[1:]), data
    )

    def update_minibatch(carry, data):
        actor, critic, alpha, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)
        alpha_value = jax.lax.stop_gradient(alpha.apply_fn(alpha.params))

        def critic_loss(params):
            next_distribution = actor.apply_fn(actor.params, data.next_observations)
            next_raw_actions, next_base_log_probs = (
                next_distribution.sample_and_log_prob(seed=critic_key)
            )
            next_forward_log_det_jacobian = 2.0 * (
                jnp.log(2.0)
                - next_raw_actions
                - jax.nn.softplus(-2.0 * next_raw_actions)
            )
            next_actions = jnp.tanh(next_raw_actions)
            next_log_probs = next_base_log_probs - next_forward_log_det_jacobian.sum(
                axis=-1
            )
            next_q_values = critic.apply_fn(
                critic.target_params, data.next_observations, next_actions
            )
            next_values = jnp.min(next_q_values, axis=0) - alpha_value * next_log_probs
            target_q_values = jax.lax.stop_gradient(
                data.rewards * reward_scale
                + gamma * (1.0 - data.terminations.astype(jnp.float32)) * next_values
            )

            q_values = critic.apply_fn(params, data.observations, data.actions)
            q_error = q_values - target_q_values
            loss = 0.5 * jnp.square(q_error).mean()
            return loss, {
                "q_value": q_values.mean(),
                "target_q_value": target_q_values.mean(),
                "q_error": jnp.abs(q_error).mean(),
            }

        (critic_loss_value, critic_metrics), critic_grads = jax.value_and_grad(
            critic_loss, has_aux=True
        )(critic.params)
        critic = critic.apply_gradients(grads=critic_grads)

        def actor_loss(params):
            distribution = actor.apply_fn(params, data.observations)
            raw_actions, base_log_probs = distribution.sample_and_log_prob(
                seed=actor_key
            )
            forward_log_det_jacobian = 2.0 * (
                jnp.log(2.0) - raw_actions - jax.nn.softplus(-2.0 * raw_actions)
            )
            actions = jnp.tanh(raw_actions)
            log_probs = base_log_probs - forward_log_det_jacobian.sum(axis=-1)
            q_values = critic.apply_fn(critic.params, data.observations, actions)
            min_q_values = jnp.min(q_values, axis=0)
            loss = (alpha_value * log_probs - min_q_values).mean()
            return loss, log_probs

        (actor_loss_value, log_probs), actor_grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(actor.params)
        actor = actor.apply_gradients(grads=actor_grads)

        def alpha_loss(params):
            log_alpha = params["params"]["log_alpha"]
            entropy_error = jax.lax.stop_gradient(log_probs + target_entropy)
            return -(log_alpha * entropy_error).mean()

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(alpha.params)
        alpha = alpha.apply_gradients(grads=alpha_grads)
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params, critic.target_params, tau
            )
        )

        expensive_metrics = jax.lax.cond(
            compute_expensive_metrics,
            lambda: {
                "actor_grad_norm": optax.tree.norm(actor_grads),
                "critic_grad_norm": optax.tree.norm(critic_grads),
                "alpha_grad_norm": optax.tree.norm(alpha_grads),
                "actor_params_norm": optax.tree.norm(actor.params),
                "critic_params_norm": optax.tree.norm(critic.params),
            },
            lambda: {name: jnp.zeros((), jnp.float32) for name in EXPENSIVE_METRICS},
        )
        metrics = {
            **critic_metrics,
            **expensive_metrics,
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": alpha.apply_fn(alpha.params),
            "entropy": -log_probs.mean(),
        }
        return (actor, critic, alpha, key), metrics

    return jax.lax.scan(
        update_minibatch,
        (actor, critic, alpha, key),
        data,
        length=num_updates,
    )


def collect_transition(
    state: ExperimentState,
    envs: Environment,
    random_actions: bool,
) -> tuple[ExperimentState, dict[str, Array]]:
    key, action_key = jax.random.split(state.key)
    if random_actions:
        action_dim = cast(Any, envs.action_space).action_dim
        actions = jax.random.uniform(
            action_key,
            (envs.num_envs, action_dim),
            minval=-1.0,
            maxval=1.0,
        )
    else:
        distribution = state.actor.apply_fn(
            state.actor.params, normalize(state.observations, state.normalizer)
        )
        actions = jnp.tanh(distribution.sample(seed=action_key))

    next_env_states, timestep = envs.step(state.env_states, actions)
    transitions = Transition(
        observations=state.observations,
        actions=actions,
        rewards=timestep.reward,
        terminations=timestep.terminated,
        next_observations=timestep.info["next_observation"],
    )
    normalizer = update_running_mean_std(state.normalizer, state.observations)
    done = timestep.terminated | timestep.truncated
    logs = {
        "completed_episodes": done.sum(),
        "episode_return_sum": jnp.where(
            done, timestep.info["episode_return"], 0.0
        ).sum(),
        "episode_steps_sum": jnp.where(done, timestep.info["episode_steps"], 0).sum(),
    }
    state = state.replace(
        normalizer=normalizer,
        replay_buffer=state.replay_buffer.insert(transitions),
        env_states=next_env_states,
        observations=timestep.observation,
        key=key,
    )
    return state, logs


def prefill_replay_buffer(
    state: ExperimentState,
    envs: Environment,
    num_steps: int,
    random_actions: bool,
) -> ExperimentState:
    def step(state: ExperimentState, _):
        state, _ = collect_transition(state, envs, random_actions)
        return state, None

    state, _ = jax.lax.scan(step, state, None, length=num_steps)
    return state


def sac_step(
    state: ExperimentState,
    envs: Environment,
    batch_size: int,
    num_updates: int,
    gamma: float,
    reward_scale: float,
    tau: float,
    target_entropy: float,
    compute_expensive_metrics: Bool[Array, ""],
) -> tuple[ExperimentState, dict[str, Array]]:
    state, episode_logs = collect_transition(state, envs, False)
    (actor, critic, alpha, key), metrics = sac_update(
        state.actor,
        state.critic,
        state.alpha,
        state.replay_buffer,
        state.normalizer,
        state.key,
        batch_size,
        num_updates,
        gamma,
        reward_scale,
        tau,
        target_entropy,
        compute_expensive_metrics,
    )
    logs = jax.tree.map(jnp.mean, metrics)
    logs.update(episode_logs)
    return state.replace(actor=actor, critic=critic, alpha=alpha, key=key), logs


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

    TOTAL_TIMESTEPS: int = 10_000_000
    NUM_ENVS: int = 128
    BATCH_SIZE: int = 512
    GRAD_UPDATES_PER_STEP: int = 8
    MIN_REPLAY_SIZE: int = 8192
    MAX_REPLAY_SIZE: int = 1_048_576 * 4
    WARMUP_POLICY: Literal["policy", "random"] = "policy"

    LEARNING_RATE: float = 1e-3
    ALPHA_LEARNING_RATE: float = 3e-4
    MAX_GRAD_NORM: float | None = None
    GAMMA: float = 0.99
    REWARD_SCALE: float = 1.0
    TAU: float = 0.005
    INITIAL_TEMPERATURE: float = 1.0
    TARGET_ENTROPY_SCALE: float = 0.5
    Q_LAYER_NORM: bool = True

    LOGGING_FREQUENCY: int = 100_000
    EVAL_FREQUENCY: int = 1_000_000
    NUM_EVAL_ENVS: int = 128
    DETERMINISTIC_EVAL: bool = False

    WANDB_ENTITY: str = "evangelos-ch"
    WANDB_PROJECT: str = "mixed-precision-rl"
    WANDB_MODE: Literal["online", "offline", "disabled", "shared"] = "online"
    WANDB_RUN_NAME: str | None = None

    def __post_init__(self) -> None:
        if self.MIN_REPLAY_SIZE % self.NUM_ENVS:
            raise ValueError("MIN_REPLAY_SIZE must be divisible by NUM_ENVS")
        if self.MIN_REPLAY_SIZE < self.BATCH_SIZE:
            raise ValueError("MIN_REPLAY_SIZE must be at least BATCH_SIZE")
        if self.MAX_REPLAY_SIZE < max(self.MIN_REPLAY_SIZE, self.NUM_ENVS):
            raise ValueError(
                "MAX_REPLAY_SIZE must hold the prefill and one vectorized env step"
            )
        if self.TOTAL_TIMESTEPS <= self.MIN_REPLAY_SIZE * self.ACTION_REPEAT:
            raise ValueError("TOTAL_TIMESTEPS must leave room to train after prefill")


def main(args: Args) -> None:
    jax.config.update("jax_default_matmul_precision", args.MATMUL_PRECISION)

    step_size = args.NUM_ENVS * args.ACTION_REPEAT
    num_prefill_steps = args.MIN_REPLAY_SIZE // args.NUM_ENVS
    prefill_environment_steps = args.MIN_REPLAY_SIZE * args.ACTION_REPEAT
    remaining_steps = args.TOTAL_TIMESTEPS - prefill_environment_steps

    num_sac_steps = math.ceil(
        remaining_steps / (args.EVAL_FREQUENCY or remaining_steps)
    )
    training_steps_per_scan = math.ceil((remaining_steps / step_size) / num_sac_steps)
    environment_steps_per_scan = training_steps_per_scan * step_size

    run = wandb.init(
        entity=args.WANDB_ENTITY,
        project=args.WANDB_PROJECT,
        name=args.WANDB_RUN_NAME
        or f"sac_{args.ENV_NAME}_{args.ENV_IMPL}_{args.COMPUTE_DTYPE.name}_{args.SEED}",
        tags=["sac", args.ENV_NAME, args.COMPUTE_DTYPE.name],
        mode=args.WANDB_MODE,
        config={
            **asdict(args),
            "ACTUAL_TIMESTEPS": prefill_environment_steps
            + (num_sac_steps * environment_steps_per_scan),
            "DEVICE_KIND": jax.local_devices()[0].device_kind,
            "NUM_DEVICES": len(jax.local_devices()),
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

    (
        key,
        actor_key,
        critic_key,
        alpha_key,
        env_key,
        eval_key,
    ) = jax.random.split(jax.random.PRNGKey(args.SEED), 6)
    env_states, observations = jax.jit(envs.init)(env_key)
    obs_spec = cast(
        jax.ShapeDtypeStruct,
        jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), observations
        ),
    )
    action_dim = cast(Any, envs.action_space).action_dim
    action_spec = jax.ShapeDtypeStruct((action_dim,), jnp.float32)

    actor_module = Policy(
        output_dim=action_dim,
        dtype=args.COMPUTE_DTYPE(),
    )
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.lazy_init(actor_key, obs_spec),
        tx=make_optimizer(args.LEARNING_RATE, args.MAX_GRAD_NORM),
    )
    critic_module = Ensemble(
        partial(
            QValueFunction,
            dtype=args.COMPUTE_DTYPE(),
            layer_norm=args.Q_LAYER_NORM,
        ),
    )
    critic_params = critic_module.lazy_init(critic_key, obs_spec, action_spec)
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_params,
        target_params=critic_params,
        tx=make_optimizer(args.LEARNING_RATE, args.MAX_GRAD_NORM),
    )
    alpha_module = Temperature(args.INITIAL_TEMPERATURE)
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=make_optimizer(args.ALPHA_LEARNING_RATE, None),
    )
    state = ExperimentState(
        actor=actor,
        critic=critic,
        alpha=alpha,
        normalizer=init_running_mean_std(obs_spec),
        replay_buffer=ReplayBuffer.init(obs_spec, action_dim, args.MAX_REPLAY_SIZE),
        env_states=env_states,
        observations=observations,
        key=key,
    )

    def evaluation_agent(observations, rng, params, agent_state):
        actor_params, observation_normalizer = params
        distribution = cast(
            distrax.MultivariateNormalDiag,
            actor_module.apply(
                actor_params,
                normalize(observations, observation_normalizer),
            ),
        )
        if args.DETERMINISTIC_EVAL:
            return agent_state, jnp.tanh(distribution.mean())
        return agent_state, jnp.tanh(distribution.sample(seed=rng))

    evaluation = (
        envs.make_evaluation(evaluation_agent, None)
        if args.EVAL_FREQUENCY > 0
        else None
    )
    target_entropy = -args.TARGET_ENTROPY_SCALE * action_dim
    run.config.update({"TARGET_ENTROPY": target_entropy})

    def log_training(environment_steps, logs) -> None:
        environment_steps = int(environment_steps)
        completed = int(logs["completed_episodes"])
        host_logs = {
            f"training/{name}": float(value)
            for name, value in logs.items()
            if name
            not in (
                "completed_episodes",
                "episode_return_sum",
                "episode_steps_sum",
            )
        }
        host_logs["training/sps"] = environment_steps / (time.monotonic() - start_time)
        if completed:
            host_logs["episode/return"] = float(logs["episode_return_sum"]) / completed
            host_logs["episode/steps"] = float(logs["episode_steps_sum"]) / completed

        run.log(host_logs, step=environment_steps)
        summary = (
            f"{environment_steps}: "
            f"actor_loss={host_logs['training/actor_loss']:.3f}, "
            f"critic_loss={host_logs['training/critic_loss']:.3f}, "
            f"alpha={host_logs['training/alpha']:.3f}, "
            f"sps={host_logs['training/sps']:.0f}"
        )
        if completed:
            summary += f", episode_return={host_logs['episode/return']:.1f}"
        print(summary)

    @jax.jit
    def prefill(state: ExperimentState) -> ExperimentState:
        return prefill_replay_buffer(
            state,
            envs,
            num_prefill_steps,
            args.WARMUP_POLICY == "random",
        )

    @jax.jit
    def train_step(
        state: ExperimentState, start_training_step: Int[Array, ""]
    ) -> ExperimentState:
        def step(state, training_step):
            environment_steps = (
                prefill_environment_steps + (training_step + 1) * step_size
            )
            if args.LOGGING_FREQUENCY > 0:
                should_log = (
                    environment_steps // args.LOGGING_FREQUENCY
                    > (environment_steps - step_size) // args.LOGGING_FREQUENCY
                )
            else:
                should_log = jnp.asarray(False)

            state, logs = sac_step(
                state,
                envs,
                args.BATCH_SIZE,
                args.GRAD_UPDATES_PER_STEP,
                args.GAMMA,
                args.REWARD_SCALE,
                args.TAU,
                target_entropy,
                should_log,
            )
            logs["replay_size"] = state.replay_buffer.size
            jax.lax.cond(
                should_log,
                lambda: io_callback(
                    log_training,
                    None,
                    environment_steps,
                    logs,
                    ordered=True,
                ),
                lambda: None,
            )
            return state, None

        training_steps = start_training_step + jnp.arange(training_steps_per_scan)
        state, _ = jax.lax.scan(step, state, training_steps)
        return state

    if evaluation is not None:
        eval_key, evaluation_key = jax.random.split(eval_key)
        eval_logs = jax.device_get(
            evaluation(
                evaluation_key,
                (state.actor.params, state.normalizer),
            )
        )
        run.log(
            {name: float(value) for name, value in eval_logs.items()},
            step=0,
        )

    start_time = time.monotonic()
    state = prefill(state)
    jax.block_until_ready(state)
    for step in range(num_sac_steps):
        state = train_step(
            state, jnp.asarray(step * training_steps_per_scan, jnp.int32)
        )
        jax.block_until_ready(state)
        jax.effects_barrier()
        if evaluation is not None:
            eval_key, evaluation_key = jax.random.split(eval_key)
            eval_logs = jax.device_get(
                evaluation(
                    evaluation_key,
                    (state.actor.params, state.normalizer),
                )
            )
            eval_logs = {name: float(value) for name, value in eval_logs.items()}
            total_steps = (
                prefill_environment_steps + (step + 1) * environment_steps_per_scan
            )
            run.log(eval_logs, step=total_steps)
            print(f"{total_steps}: eval_reward={eval_logs['eval/episode_reward']:.1f}")

    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
