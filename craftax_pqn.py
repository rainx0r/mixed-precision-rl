"""PQN-RNN for Craftax.

This follows the Craftax baseline from ``mttga/purejaxql`` while using this
project's environment, logging, evaluation, and mixed-precision conventions.
PQN trains a recurrent Q-network directly on fresh parallel rollouts, without
a replay buffer or target network.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
import math
import time
from typing import Any, Literal, NamedTuple, TypedDict

import flax.linen as nn
from flax.linen.normalization import (
    _canonicalize_axes,
    _compute_stats,
    _normalize,
)
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
import optax

import tyro
import wandb

from mixed_precision_rl.envs.craftax import CraftaxEnv, CraftaxEnvState
from mixed_precision_rl.types import DType, Observation

type LSTMState = tuple[
    Float[Array, "batch hidden"],
    Float[Array, "batch hidden"],
]
type RecurrentState = tuple[LSTMState, ...]


class RolloutExtras(TypedDict):
    episode_returns: Float[Array, "time env"]
    episode_steps: Int[Array, "time env"]
    achievements: dict[str, Float[Array, "time env"]]


class Rollout(NamedTuple):
    observations: Float[Array, "time env observation"]
    previous_actions: Int[Array, "time env"]
    actions: Int[Array, "time env"]
    rewards: Float[Array, "time env"]
    terminations: Bool[Array, "time env"]
    truncations: Bool[Array, "time env"]
    extras: RolloutExtras


class SequenceBatch(NamedTuple):
    initial_hidden_states: RecurrentState
    episode_starts: Bool[Array, "time sequence"]
    rollout: Rollout


class PQNTrainState(TrainState):
    batch_stats: Any


class ExperimentState(NamedTuple):
    train_state: PQNTrainState
    env_states: CraftaxEnvState
    observations: Observation
    episode_starts: Bool[Array, " env"]
    previous_actions: Int[Array, " env"]
    hidden_states: RecurrentState
    key: PRNGKeyArray


class EvaluationState(NamedTuple):
    hidden_states: RecurrentState
    previous_actions: Int[Array, " env"]


class BatchRenorm(nn.Module):
    """Batch renormalization used for observations in the PQN baseline."""

    use_running_average: bool
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = jax.nn.initializers.zeros
    scale_init: Callable = jax.nn.initializers.ones

    @nn.compact
    def __call__(
        self, inputs: Float[Array, "... feature"]
    ) -> Float[Array, "... feature"]:
        feature_axes = _canonicalize_axes(inputs.ndim, -1)
        reduction_axes = tuple(
            axis for axis in range(inputs.ndim) if axis not in feature_axes
        )
        feature_shape = tuple(inputs.shape[axis] for axis in feature_axes)

        running_mean = self.variable(
            "batch_stats",
            "mean",
            lambda: jnp.zeros(feature_shape, dtype=jnp.float32),
        )
        running_variance = self.variable(
            "batch_stats",
            "variance",
            lambda: jnp.ones(feature_shape, dtype=jnp.float32),
        )
        steps = self.variable(
            "batch_stats",
            "steps",
            lambda: jnp.zeros((), dtype=jnp.int32),
        )

        if self.use_running_average:
            mean = running_mean.value
            variance = running_variance.value
        else:
            mean, variance = _compute_stats(
                inputs,
                reduction_axes,
                dtype=jnp.float32,
                use_fast_variance=True,
            )
            normalized_mean = mean
            normalized_variance = variance
            if not self.is_initializing():
                standard_deviation = jnp.sqrt(variance + self.epsilon)
                running_standard_deviation = jnp.sqrt(
                    running_variance.value + self.epsilon
                )
                scale = jax.lax.stop_gradient(
                    standard_deviation / running_standard_deviation
                )
                scale = jnp.clip(scale, 1 / 3, 3)
                offset = jax.lax.stop_gradient(
                    (mean - running_mean.value) / running_standard_deviation
                )
                offset = jnp.clip(offset, -5, 5)

                renormalized_variance = variance / jnp.square(scale)
                renormalized_mean = (
                    mean - offset * jnp.sqrt(normalized_variance) / scale
                )
                warmed_up = steps.value >= 1_000
                normalized_variance = jnp.where(
                    warmed_up,
                    renormalized_variance,
                    normalized_variance,
                )
                normalized_mean = jnp.where(
                    warmed_up,
                    renormalized_mean,
                    normalized_mean,
                )

                running_mean.value = (
                    self.momentum * running_mean.value + (1 - self.momentum) * mean
                )
                running_variance.value = (
                    self.momentum * running_variance.value
                    + (1 - self.momentum) * variance
                )
                steps.value += 1

            mean = normalized_mean
            variance = normalized_variance

        return _normalize(
            self,
            inputs,
            mean,
            variance,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class ScannedLSTM(nn.Module):
    hidden_size: int
    dtype: DTypeLike = jnp.float32

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},  # ty: ignore[invalid-argument-type]
    )
    @nn.compact
    def __call__(
        self,
        hidden_states: LSTMState,
        inputs: tuple[
            Float[Array, "batch feature"],
            Bool[Array, " batch"],
        ],
    ) -> tuple[LSTMState, Float[Array, "batch hidden"]]:
        embeddings, episode_starts = inputs
        hidden_states = jax.tree.map(
            lambda state: jnp.where(
                episode_starts[:, None],
                jnp.zeros_like(state),
                state,
            ),
            hidden_states,
        )
        return nn.OptimizedLSTMCell(
            features=self.hidden_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(hidden_states, embeddings)

    @staticmethod
    def initialize_carry(
        batch_size: int,
        hidden_size: int,
        dtype: DTypeLike = jnp.float32,
    ) -> LSTMState:
        zeros = jnp.zeros((batch_size, hidden_size), dtype=dtype)
        return zeros, zeros


class QNetworkRNN(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_hidden_layers: int = 1
    num_rnn_layers: int = 1
    input_normalization: bool = True
    hidden_normalization: Literal["layer_norm", "batch_norm", "none"] = "layer_norm"
    add_previous_action: bool = True
    dtype: DTypeLike = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: RecurrentState,
        observations: Float[Array, "time batch observation"],
        episode_starts: Bool[Array, "time batch"],
        previous_actions: Int[Array, "time batch"],
        training: bool = False,
    ) -> tuple[RecurrentState, Float[Array, "time batch action"]]:
        if self.input_normalization:
            embeddings = BatchRenorm(
                use_running_average=not training,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                name="observation_norm",
            )(observations)
        else:
            embeddings = observations.astype(self.dtype)

        for layer in range(self.num_hidden_layers):
            embeddings = nn.Dense(
                self.hidden_size,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                name=f"encoder_{layer}",
            )(embeddings)
            if self.hidden_normalization == "layer_norm":
                embeddings = nn.LayerNorm(
                    dtype=self.dtype,
                    param_dtype=jnp.float32,
                    name=f"encoder_norm_{layer}",
                )(embeddings)
            elif self.hidden_normalization == "batch_norm":
                embeddings = BatchRenorm(
                    use_running_average=not training,
                    dtype=self.dtype,
                    param_dtype=jnp.float32,
                    name=f"encoder_norm_{layer}",
                )(embeddings)
            embeddings = nn.relu(embeddings)

        if self.add_previous_action:
            previous_action_one_hot = jax.nn.one_hot(
                previous_actions,
                self.action_dim,
                dtype=self.dtype,
            )
            embeddings = jnp.concatenate(
                (embeddings, previous_action_one_hot),
                axis=-1,
            )

        next_hidden_states = []
        for layer in range(self.num_rnn_layers):
            hidden_state, embeddings = ScannedLSTM(
                hidden_size=self.hidden_size,
                dtype=self.dtype,
                name=f"lstm_{layer}",
            )(hidden_states[layer], (embeddings, episode_starts))
            next_hidden_states.append(hidden_state)

        q_values = nn.Dense(
            self.action_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="q_values",
        )(embeddings)
        return tuple(next_hidden_states), q_values.astype(jnp.float32)

    def initialize_carry(
        self,
        batch_size: int,
    ) -> RecurrentState:
        return tuple(
            ScannedLSTM.initialize_carry(
                batch_size,
                self.hidden_size,
                self.dtype,
            )
            for _ in range(self.num_rnn_layers)
        )


def epsilon_greedy_actions(
    key: PRNGKeyArray,
    q_values: Float[Array, "batch action"],
    epsilon: Float[Array, ""] | float,
) -> Int[Array, " batch"]:
    random_action_key, exploration_key = jax.random.split(key)
    greedy_actions = jnp.argmax(q_values, axis=-1)
    random_actions = jax.random.randint(
        random_action_key,
        greedy_actions.shape,
        minval=0,
        maxval=q_values.shape[-1],
    )
    explore = jax.random.uniform(exploration_key, greedy_actions.shape) < epsilon
    return jnp.where(explore, random_actions, greedy_actions).astype(jnp.int32)


def compute_q_lambda_targets(
    q_values: Float[Array, "time batch action"],
    rewards: Float[Array, "time batch"],
    done: Bool[Array, "time batch"],
    gamma: float,
    q_lambda: float,
) -> Float[Array, "train_time batch"]:
    """Compute Peng's Q(lambda) targets, dropping the final bootstrap step."""
    if rewards.shape[0] < 2:
        raise ValueError("Q(lambda) targets require at least two rollout steps")

    next_q_values = jnp.max(q_values[1:], axis=-1)

    def compute_target(next_lambda_return, transition):
        reward, transition_done, next_q_value = transition
        continues = 1 - transition_done.astype(q_values.dtype)
        target = reward + gamma * continues * (
            (1 - q_lambda) * next_q_value + q_lambda * next_lambda_return
        )
        return target, target

    _, targets = jax.lax.scan(
        compute_target,
        next_q_values[-1],
        (
            rewards[:-1].astype(q_values.dtype),
            done[:-1],
            next_q_values,
        ),
        reverse=True,
    )
    return targets


def make_optimizer(
    learning_rate: float,
    max_grad_norm: float | None,
    anneal_learning_rate: bool,
    num_updates: int,
    num_epochs: int,
    num_minibatches: int,
) -> optax.GradientTransformation:
    if anneal_learning_rate:
        learning_rate_or_schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=1e-20,
            transition_steps=num_updates * num_epochs * num_minibatches,
        )
    else:
        learning_rate_or_schedule = learning_rate

    transforms = []
    if max_grad_norm is not None:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(optax.radam(learning_rate_or_schedule))
    return optax.chain(*transforms)


def pqn_update(
    train_state: PQNTrainState,
    initial_hidden_states: RecurrentState,
    initial_episode_starts: Bool[Array, " env"],
    data: Rollout,
    key: PRNGKeyArray,
    gamma: float,
    q_lambda: float,
    reward_scale: float,
    num_minibatches: int,
    num_epochs: int,
) -> tuple[
    tuple[PQNTrainState, PRNGKeyArray],
    dict[str, Float[Array, "epoch minibatch"]],
]:
    rollout_length, num_envs = data.rewards.shape
    sequences_per_minibatch = num_envs // num_minibatches
    done = data.terminations | data.truncations
    episode_starts = jnp.concatenate((initial_episode_starts[None], done[:-1]), axis=0)

    def update_minibatch(train_state, batch: SequenceBatch):
        def loss(params):
            (
                (_, q_values),
                variable_updates,
            ) = train_state.apply_fn(
                {
                    "params": params,
                    "batch_stats": train_state.batch_stats,
                },
                batch.initial_hidden_states,
                batch.rollout.observations,
                batch.episode_starts,
                batch.rollout.previous_actions,
                training=True,
                mutable=["batch_stats"],
            )
            target_q_values = compute_q_lambda_targets(
                jax.lax.stop_gradient(q_values),
                batch.rollout.rewards * reward_scale,
                batch.rollout.terminations | batch.rollout.truncations,
                gamma,
                q_lambda,
            )
            chosen_q_values = jnp.take_along_axis(
                q_values[:-1],
                batch.rollout.actions[:-1, :, None],
                axis=-1,
            ).squeeze(-1)
            td_errors = chosen_q_values - target_q_values
            td_loss = 0.5 * jnp.square(td_errors).mean()
            metrics = {
                "td_loss": td_loss,
                "q_value": chosen_q_values.mean(),
                "target_q_value": target_q_values.mean(),
                "max_abs_td_error": jnp.abs(td_errors).max(),
            }
            return td_loss, (variable_updates, metrics)

        (
            (
                _,
                (variable_updates, metrics),
            ),
            grads,
        ) = jax.value_and_grad(loss, has_aux=True)(train_state.params)
        metrics["grad_norm"] = optax.global_norm(grads)
        train_state = train_state.apply_gradients(grads=grads)
        # Flax returns mutable collections separately from the model output.
        train_state = train_state.replace(
            batch_stats=variable_updates.get(
                "batch_stats",
                train_state.batch_stats,
            )
        )
        return train_state, metrics

    def update_epoch(carry, _):
        train_state, key = carry
        key, permutation_key = jax.random.split(key)
        permutation = jax.random.permutation(permutation_key, num_envs)

        def pack_rollout_leaf(x: jax.Array) -> jax.Array:
            x = jnp.take(x, permutation, axis=1)
            x = x.reshape(
                rollout_length,
                num_minibatches,
                sequences_per_minibatch,
                *x.shape[2:],
            )
            return jnp.swapaxes(x, 0, 1)

        def pack_hidden_state_leaf(x: jax.Array) -> jax.Array:
            x = jnp.take(x, permutation, axis=0)
            return x.reshape(
                num_minibatches,
                sequences_per_minibatch,
                *x.shape[1:],
            )

        minibatches = SequenceBatch(
            initial_hidden_states=jax.tree.map(
                pack_hidden_state_leaf,
                initial_hidden_states,
            ),
            episode_starts=pack_rollout_leaf(episode_starts),
            rollout=jax.tree.map(pack_rollout_leaf, data),
        )
        train_state, metrics = jax.lax.scan(
            update_minibatch,
            train_state,
            minibatches,
            length=num_minibatches,
        )
        return (train_state, key), metrics

    return jax.lax.scan(
        update_epoch,
        (train_state, key),
        None,
        length=num_epochs,
    )


def rollout(
    train_state: PQNTrainState,
    envs: CraftaxEnv,
    env_states: CraftaxEnvState,
    observations: Observation,
    episode_starts: Bool[Array, " env"],
    previous_actions: Int[Array, " env"],
    hidden_states: RecurrentState,
    key: PRNGKeyArray,
    epsilon: Float[Array, ""] | float,
    rollout_length: int,
    observation_dtype: DTypeLike = jnp.float32,
) -> tuple[
    CraftaxEnvState,
    Observation,
    Bool[Array, " env"],
    Int[Array, " env"],
    RecurrentState,
    PRNGKeyArray,
    Rollout,
]:
    def environment_step(carry, _):
        (
            env_states,
            observations,
            episode_starts,
            previous_actions,
            hidden_states,
            key,
        ) = carry
        key, action_key = jax.random.split(key)

        hidden_states, q_values = train_state.apply_fn(
            {
                "params": train_state.params,
                "batch_stats": train_state.batch_stats,
            },
            hidden_states,
            observations[None],
            episode_starts[None],
            previous_actions[None],
            training=False,
        )
        actions = epsilon_greedy_actions(
            action_key,
            q_values.squeeze(0),
            epsilon,
        )

        env_states, timestep = envs.step(env_states, actions)
        next_episode_starts = timestep.terminated | timestep.truncated
        transition = Rollout(
            observations=observations.astype(observation_dtype),
            previous_actions=previous_actions,
            actions=actions,
            rewards=timestep.reward,
            terminations=timestep.terminated,
            truncations=timestep.truncated,
            extras={
                "episode_returns": timestep.info["episode_return"],
                "episode_steps": timestep.info["episode_steps"],
                "achievements": timestep.info["achievements"],
            },
        )
        carry = (
            env_states,
            timestep.observation,
            next_episode_starts,
            actions,
            hidden_states,
            key,
        )
        return carry, transition

    carry = (
        env_states,
        observations,
        episode_starts,
        previous_actions,
        hidden_states,
        key,
    )
    carry, data = jax.lax.scan(
        environment_step,
        carry,
        None,
        length=rollout_length,
    )
    return (*carry, data)


@dataclass
class Args:
    SEED: int = 43
    ENV_NAME: str = "Craftax-Symbolic-v1"
    EPISODE_LENGTH: int = 10_000
    RESET_RATIO: int = 16
    MATMUL_PRECISION: Literal["default", "high", "highest"] = "highest"
    COMPUTE_DTYPE: DType = DType.float32
    ROLLOUT_OBSERVATION_DTYPE: DType = DType.float32

    TOTAL_TIMESTEPS: int = 1_000_000_000
    ROLLOUT_LENGTH: int = 128
    NUM_ENVS: int = 1024
    NUM_MINIBATCHES: int = 4
    NUM_EPOCHS: int = 4

    HIDDEN_SIZE: int = 512
    NUM_HIDDEN_LAYERS: int = 1
    NUM_RNN_LAYERS: int = 1
    INPUT_NORMALIZATION: bool = True
    HIDDEN_NORMALIZATION: Literal["layer_norm", "batch_norm", "none"] = "layer_norm"
    ADD_PREVIOUS_ACTION: bool = True

    LEARNING_RATE: float = 3e-4
    ANNEAL_LEARNING_RATE: bool = True
    MAX_GRAD_NORM: float | None = 0.5
    GAMMA: float = 0.99
    Q_LAMBDA: float = 0.5
    REWARD_SCALE: float = 1.0
    EPSILON_START: float = 1.0
    EPSILON_FINISH: float = 0.005
    EPSILON_DECAY_FRACTION: float = 0.1

    EVAL_FREQUENCY: int = 10_000_000
    NUM_EVAL_ENVS: int = 128
    EVAL_EPSILON: float = 0.0

    WANDB_ENTITY: str = "evangelos-ch"
    WANDB_PROJECT: str = "mixed-precision-rl"
    WANDB_MODE: Literal["online", "offline", "disabled", "shared"] = "online"
    WANDB_RUN_NAME: str | None = None

    def __post_init__(self) -> None:
        if self.ROLLOUT_LENGTH < 2:
            raise ValueError("ROLLOUT_LENGTH must be at least 2 for Q(lambda)")
        if self.NUM_ENVS < 1:
            raise ValueError("NUM_ENVS must be at least 1")
        if self.NUM_MINIBATCHES < 1:
            raise ValueError("NUM_MINIBATCHES must be at least 1")
        if self.NUM_ENVS % self.NUM_MINIBATCHES:
            raise ValueError("NUM_MINIBATCHES must divide NUM_ENVS")
        if self.NUM_EPOCHS < 1:
            raise ValueError("NUM_EPOCHS must be at least 1")
        if self.NUM_HIDDEN_LAYERS < 1:
            raise ValueError("NUM_HIDDEN_LAYERS must be at least 1")
        if self.NUM_RNN_LAYERS < 1:
            raise ValueError("NUM_RNN_LAYERS must be at least 1")
        if not 0 <= self.Q_LAMBDA <= 1:
            raise ValueError("Q_LAMBDA must be between 0 and 1")
        if not 0 <= self.EPSILON_DECAY_FRACTION <= 1:
            raise ValueError("EPSILON_DECAY_FRACTION must be between 0 and 1")
        if self.TOTAL_TIMESTEPS < self.NUM_ENVS * self.ROLLOUT_LENGTH:
            raise ValueError(
                "TOTAL_TIMESTEPS must cover at least one full parallel rollout"
            )


def main(args: Args) -> None:
    jax.config.update("jax_default_matmul_precision", args.MATMUL_PRECISION)

    transitions_per_update = args.NUM_ENVS * args.ROLLOUT_LENGTH
    num_updates = args.TOTAL_TIMESTEPS // transitions_per_update
    actual_timesteps = num_updates * transitions_per_update
    epsilon_decay_updates = max(args.EPSILON_DECAY_FRACTION * num_updates, 1.0)

    run = wandb.init(
        entity=args.WANDB_ENTITY,
        project=args.WANDB_PROJECT,
        name=args.WANDB_RUN_NAME
        or (f"pqn_rnn_{args.ENV_NAME}_{args.COMPUTE_DTYPE.name}_{args.SEED}"),
        tags=[
            "pqn",
            "q_learning",
            "rnn",
            args.ENV_NAME,
            args.COMPUTE_DTYPE.name,
        ],
        mode=args.WANDB_MODE,
        config={
            **asdict(args),
            "ACTUAL_TIMESTEPS": actual_timesteps,
            "TRANSITIONS_PER_UPDATE": transitions_per_update,
            "NUM_UPDATES": num_updates,
        },
    )

    envs = CraftaxEnv(
        env_name=args.ENV_NAME,
        num_envs=args.NUM_ENVS,
        num_eval_envs=args.NUM_EVAL_ENVS,
        max_episode_length=args.EPISODE_LENGTH,
        reset_ratio=args.RESET_RATIO,
        next_obs_in_extras=False,
    )

    key, network_key, env_key, eval_key = jax.random.split(
        jax.random.PRNGKey(args.SEED), 4
    )
    env_states, observations = jax.jit(envs.init)(env_key)
    episode_starts = jnp.zeros(args.NUM_ENVS, dtype=jnp.bool_)
    previous_actions = jnp.zeros(args.NUM_ENVS, dtype=jnp.int32)

    network = QNetworkRNN(
        action_dim=envs.action_space.num_actions,
        hidden_size=args.HIDDEN_SIZE,
        num_hidden_layers=args.NUM_HIDDEN_LAYERS,
        num_rnn_layers=args.NUM_RNN_LAYERS,
        input_normalization=args.INPUT_NORMALIZATION,
        hidden_normalization=args.HIDDEN_NORMALIZATION,
        add_previous_action=args.ADD_PREVIOUS_ACTION,
        dtype=args.COMPUTE_DTYPE(),
    )
    hidden_states = network.initialize_carry(args.NUM_ENVS)
    variables = network.init(
        network_key,
        hidden_states,
        observations[None],
        episode_starts[None],
        previous_actions[None],
        training=False,
    )
    train_state = PQNTrainState.create(
        apply_fn=network.apply,
        params=variables["params"],
        batch_stats=variables.get("batch_stats", {}),
        tx=make_optimizer(
            args.LEARNING_RATE,
            args.MAX_GRAD_NORM,
            args.ANNEAL_LEARNING_RATE,
            num_updates,
            args.NUM_EPOCHS,
            args.NUM_MINIBATCHES,
        ),
    )
    state = ExperimentState(
        train_state=train_state,
        env_states=env_states,
        observations=observations,
        episode_starts=episode_starts,
        previous_actions=previous_actions,
        hidden_states=hidden_states,
        key=key,
    )

    @jax.jit
    def random_warmup(state: ExperimentState) -> ExperimentState:
        (
            env_states,
            observations,
            episode_starts,
            previous_actions,
            hidden_states,
            key,
            _,
        ) = rollout(
            state.train_state,
            envs,
            state.env_states,
            state.observations,
            state.episode_starts,
            state.previous_actions,
            state.hidden_states,
            state.key,
            1.0,
            args.ROLLOUT_LENGTH,
            args.ROLLOUT_OBSERVATION_DTYPE(),
        )
        return ExperimentState(
            train_state=state.train_state,
            env_states=env_states,
            observations=observations,
            episode_starts=episode_starts,
            previous_actions=previous_actions,
            hidden_states=hidden_states,
            key=key,
        )

    # Upstream seeds its rollout memory with random interaction before learning.
    # With the baseline MEMORY_WINDOW=0 those samples are discarded, but the
    # environment and recurrent state are still advanced; retain that for fidelity.
    state = random_warmup(state)

    def evaluation_agent(
        observations,
        rng,
        variables,
        evaluation_state: EvaluationState,
    ):
        hidden_states, q_values = train_state.apply_fn(
            variables,
            evaluation_state.hidden_states,
            observations[None],
            jnp.zeros((1, observations.shape[0]), dtype=jnp.bool_),
            evaluation_state.previous_actions[None],
            training=False,
        )
        actions = epsilon_greedy_actions(
            rng,
            q_values.squeeze(0),
            args.EVAL_EPSILON,
        )
        return EvaluationState(hidden_states, actions), actions

    initial_evaluation_state = EvaluationState(
        hidden_states=network.initialize_carry(args.NUM_EVAL_ENVS),
        previous_actions=jnp.zeros(args.NUM_EVAL_ENVS, dtype=jnp.int32),
    )
    evaluation = (
        envs.make_evaluation(
            evaluation_agent,
            initial_evaluation_state,
        )
        if args.EVAL_FREQUENCY > 0
        else None
    )

    @jax.jit
    def train_step(
        state: ExperimentState,
    ) -> tuple[ExperimentState, dict[str, jax.Array]]:
        initial_episode_starts = state.episode_starts
        initial_hidden_states = state.hidden_states
        update = state.train_state.step // (args.NUM_EPOCHS * args.NUM_MINIBATCHES)
        epsilon_fraction = jnp.minimum(update / epsilon_decay_updates, 1.0)
        epsilon = args.EPSILON_START + epsilon_fraction * (
            args.EPSILON_FINISH - args.EPSILON_START
        )
        (
            env_states,
            observations,
            episode_starts,
            previous_actions,
            hidden_states,
            key,
            data,
        ) = rollout(
            state.train_state,
            envs,
            state.env_states,
            state.observations,
            state.episode_starts,
            state.previous_actions,
            state.hidden_states,
            state.key,
            epsilon,
            args.ROLLOUT_LENGTH,
            args.ROLLOUT_OBSERVATION_DTYPE(),
        )
        (train_state, key), metrics = pqn_update(
            state.train_state,
            initial_hidden_states,
            initial_episode_starts,
            data,
            key,
            args.GAMMA,
            args.Q_LAMBDA,
            args.REWARD_SCALE,
            args.NUM_MINIBATCHES,
            args.NUM_EPOCHS,
        )

        done = data.terminations | data.truncations
        logs = jax.tree.map(jnp.mean, metrics)
        logs.update(
            epsilon=epsilon,
            completed_episodes=done.sum(),
            episode_return_sum=jnp.where(
                done, data.extras["episode_returns"], 0.0
            ).sum(),
            episode_steps_sum=jnp.where(done, data.extras["episode_steps"], 0).sum(),
        )
        logs.update(
            {
                f"achievement_sum/{name}": achievement.sum()
                for name, achievement in data.extras["achievements"].items()
            }
        )
        return (
            ExperimentState(
                train_state=train_state,
                env_states=env_states,
                observations=observations,
                episode_starts=episode_starts,
                previous_actions=previous_actions,
                hidden_states=hidden_states,
                key=key,
            ),
            logs,
        )

    if evaluation is not None:
        eval_key, evaluation_key = jax.random.split(eval_key)
        evaluation_variables = {
            "params": state.train_state.params,
            "batch_stats": state.train_state.batch_stats,
        }
        eval_logs = jax.device_get(evaluation(evaluation_key, evaluation_variables))
        run.log(
            {name: float(value) for name, value in eval_logs.items()},
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
        environment_steps += transitions_per_update
        elapsed = time.monotonic() - start_time

        completed = int(logs.pop("completed_episodes"))
        episode_return_sum = float(logs.pop("episode_return_sum"))
        episode_steps_sum = float(logs.pop("episode_steps_sum"))
        achievement_sums = {
            name.removeprefix("achievement_sum/"): float(logs.pop(name))
            for name in tuple(logs)
            if name.startswith("achievement_sum/")
        }

        host_logs = {f"training/{name}": float(value) for name, value in logs.items()}
        host_logs["training/sps"] = environment_steps / elapsed
        if completed:
            host_logs["episode/return"] = episode_return_sum / completed
            host_logs["episode/steps"] = episode_steps_sum / completed
            host_logs.update(
                {
                    f"episode/{name}": total / completed
                    for name, total in achievement_sums.items()
                }
            )

        summary = (
            f"{environment_steps}: "
            f"loss={host_logs['training/td_loss']:.3f}, "
            f"epsilon={host_logs['training/epsilon']:.3f}, "
            f"sps={host_logs['training/sps']:.0f}"
        )
        if environment_steps >= next_eval_step and evaluation is not None:
            eval_key, evaluation_key = jax.random.split(eval_key)
            evaluation_variables = {
                "params": state.train_state.params,
                "batch_stats": state.train_state.batch_stats,
            }
            eval_logs = jax.device_get(evaluation(evaluation_key, evaluation_variables))
            host_logs.update({name: float(value) for name, value in eval_logs.items()})
            summary += f", eval_reward={host_logs['eval/episode_reward']:.1f}"
            while next_eval_step <= environment_steps:
                next_eval_step += args.EVAL_FREQUENCY
        if completed:
            summary += f", episode_return={host_logs['episode/return']:.1f}"

        run.log(host_logs, step=environment_steps)
        print(summary)

    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
