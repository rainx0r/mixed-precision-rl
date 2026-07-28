from dataclasses import asdict, dataclass
from functools import partial
import math
import time
from typing import Literal, NamedTuple, TypedDict

import distrax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
import optax

import tyro

from mixed_precision_rl.envs.craftax import CraftaxEnv, CraftaxEnvState
from mixed_precision_rl.types import DType, Observation
import wandb


class RolloutExtras(TypedDict):
    episode_returns: Float[Array, "time env"]
    episode_steps: Int[Array, "time env"]
    achievements: dict[str, Float[Array, "time env"]]


class Rollout(NamedTuple):
    observations: Float[Array, "time env observation"]
    actions: Int[Array, "time env"]
    rewards: Float[Array, "time env"]
    terminations: Bool[Array, "time env"]
    truncations: Bool[Array, "time env"]
    values: Float[Array, "time env"]
    next_values: Float[Array, "time env"]
    log_probs: Float[Array, "time env"]
    extras: RolloutExtras


class SequenceBatch(NamedTuple):
    initial_hidden_states: Float[Array, "minibatch sequence hidden"]
    episode_starts: Bool[Array, "minibatch time sequence"]
    rollout: Rollout
    advantages: Float[Array, "minibatch time sequence"]
    value_targets: Float[Array, "minibatch time sequence"]


class ExperimentState(NamedTuple):
    train_state: TrainState
    env_states: CraftaxEnvState
    observations: Observation
    episode_starts: Bool[Array, " env"]
    hidden_states: Float[Array, "env hidden"]
    key: PRNGKeyArray


class ScannedRNN(nn.Module):
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
        hidden_states: Float[Array, "batch hidden"],
        inputs: tuple[
            Float[Array, "batch hidden"],
            Bool[Array, " batch"],
        ],
    ) -> tuple[
        Float[Array, "batch hidden"],
        Float[Array, "batch hidden"],
    ]:
        embeddings, episode_starts = inputs
        hidden_states = jnp.where(
            episode_starts[:, None],
            jnp.zeros_like(hidden_states),
            hidden_states,
        )
        return nn.GRUCell(
            features=self.hidden_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(hidden_states, embeddings)

    @staticmethod
    def initialize_carry(
        batch_size: int,
        hidden_size: int,
        dtype: DTypeLike = jnp.float32,
    ) -> Float[Array, "batch hidden"]:
        return jnp.zeros((batch_size, hidden_size), dtype=dtype)


class ActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 512
    dtype: DTypeLike = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: Float[Array, "batch hidden"],
        inputs: tuple[
            Float[Array, "time batch observation"],
            Bool[Array, "time batch"],
        ],
    ) -> tuple[
        Float[Array, "batch hidden"],
        Float[Array, "time batch hidden"],
        distrax.Categorical,
        Float[Array, "time batch"],
    ]:
        observations, episode_starts = inputs
        embeddings = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=jax.nn.initializers.orthogonal(math.sqrt(2)),
            bias_init=jax.nn.initializers.constant(0.0),
        )(observations)
        embeddings = nn.relu(embeddings)
        hidden_states, embeddings = ScannedRNN(
            hidden_size=self.hidden_size,
            dtype=self.dtype,
        )(hidden_states, (embeddings, episode_starts))
        recurrent_states = embeddings

        actor = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=jax.nn.initializers.orthogonal(2.0),
            bias_init=jax.nn.initializers.constant(0.0),
        )(embeddings)
        actor = nn.relu(actor)
        actor = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=jax.nn.initializers.orthogonal(2.0),
            bias_init=jax.nn.initializers.constant(0.0),
        )(actor)
        actor = nn.relu(actor)
        logits = nn.Dense(
            self.action_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=jax.nn.initializers.orthogonal(0.01),
            bias_init=jax.nn.initializers.constant(0.0),
        )(actor)

        critic = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=jax.nn.initializers.orthogonal(2.0),
            bias_init=jax.nn.initializers.constant(0.0),
        )(embeddings)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=jax.nn.initializers.orthogonal(2.0),
            bias_init=jax.nn.initializers.constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        values = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=jax.nn.initializers.orthogonal(1.0),
            bias_init=jax.nn.initializers.constant(0.0),
        )(critic)
        return (
            hidden_states,
            recurrent_states,
            distrax.Categorical(logits=logits.astype(jnp.float32)),
            values.astype(jnp.float32).squeeze(-1),
        )


def make_optimizer(
    learning_rate: float,
    adam_eps: float,
    max_grad_norm: float | None,
    anneal_learning_rate: bool,
    num_updates: int,
    num_epochs: int,
    num_minibatches: int,
) -> optax.GradientTransformation:
    if anneal_learning_rate:

        def schedule(count):
            update = count // (num_epochs * num_minibatches)
            fraction = 1.0 - update / num_updates
            return learning_rate * fraction

        learning_rate_or_schedule = schedule
    else:
        learning_rate_or_schedule = learning_rate

    transforms = []
    if max_grad_norm is not None:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(optax.adam(learning_rate_or_schedule, eps=adam_eps))
    return optax.chain(*transforms)


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
    not_done = 1 - (terminations | truncations).astype(values.dtype)
    deltas = rewards + gamma * not_terminated * next_values - values

    def compute_advantage(acc, transition):
        delta, continues = transition
        acc = delta + gamma * gae_lambda * continues * acc
        return acc, acc

    _, advantages = jax.lax.scan(
        compute_advantage,
        jnp.zeros_like(values[-1]),
        (deltas, not_done),
        reverse=True,
        unroll=16,
    )
    return advantages + values, advantages


def ppo_update(
    train_state: TrainState,
    initial_hidden_states: Float[Array, "env hidden"],
    initial_episode_starts: Bool[Array, " env"],
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
    bptt_length: int,
    num_minibatches: int,
    num_epochs: int,
) -> tuple[
    tuple[TrainState, PRNGKeyArray],
    dict[str, Float[Array, "epoch minibatch"]],
]:
    rollout_length, num_envs = data.rewards.shape
    num_chunks = rollout_length // bptt_length
    done = data.terminations | data.truncations
    episode_starts = jnp.concatenate((initial_episode_starts[None], done[:-1]), axis=0)
    value_targets, advantages = compute_gae(
        data.rewards * reward_scale,
        data.values,
        data.next_values,
        gamma,
        gae_lambda,
        data.terminations,
        data.truncations,
    )
    if num_chunks == 1:
        chunk_initial_hidden_states = initial_hidden_states[None]
    else:
        _, recurrent_states, _, _ = train_state.apply_fn(
            train_state.params,
            initial_hidden_states,
            (data.observations, episode_starts),
        )
        hidden_states_before_step = jnp.concatenate(
            (initial_hidden_states[None], recurrent_states[:-1]), axis=0
        )
        chunk_initial_hidden_states = hidden_states_before_step[::bptt_length]

    def loss(params, batch: SequenceBatch):
        _, _, distribution, predicted_values = train_state.apply_fn(
            params,
            batch.initial_hidden_states,
            (batch.rollout.observations, batch.episode_starts),
        )
        log_probs = distribution.log_prob(batch.rollout.actions)
        advantages = batch.advantages
        if norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = jnp.exp(log_probs - batch.rollout.log_probs)
        policy_loss = -jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages,
        ).mean()

        value_loss = jnp.square(batch.value_targets - predicted_values)
        if value_clip_eps is not None:
            clipped_values = batch.rollout.values + jnp.clip(
                predicted_values - batch.rollout.values,
                -value_clip_eps,
                value_clip_eps,
            )
            clipped_value_loss = jnp.square(batch.value_targets - clipped_values)
            value_loss = jnp.maximum(value_loss, clipped_value_loss)
        value_loss = 0.5 * value_loss.mean()

        entropy = distribution.entropy().mean()
        total_loss = policy_loss + vf_coeff * value_loss - entropy_coeff * entropy
        return total_loss, {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "vf_loss": value_loss,
            "entropy": entropy,
        }

    def update_minibatch(train_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(
            train_state.params, batch
        )
        return train_state.apply_gradients(grads=grads), metrics

    def update_epoch(carry, _):
        train_state, key = carry
        key, permutation_key = jax.random.split(key)
        num_sequences = num_chunks * num_envs
        sequences_per_minibatch = num_sequences // num_minibatches
        permutation = jax.random.permutation(permutation_key, num_sequences)

        def pack_rollout_leaf(x: jax.Array) -> jax.Array:
            x = x.reshape(num_chunks, bptt_length, num_envs, *x.shape[2:])
            x = jnp.swapaxes(x, 1, 2)
            x = x.reshape(num_sequences, bptt_length, *x.shape[3:])
            x = jnp.take(x, permutation, axis=0)
            x = x.reshape(
                num_minibatches,
                sequences_per_minibatch,
                bptt_length,
                *x.shape[2:],
            )
            return jnp.swapaxes(x, 1, 2)

        minibatch_initial_hidden_states = chunk_initial_hidden_states.reshape(
            num_sequences, *chunk_initial_hidden_states.shape[2:]
        )
        minibatch_initial_hidden_states = jnp.take(
            minibatch_initial_hidden_states, permutation, axis=0
        )
        minibatch_initial_hidden_states = minibatch_initial_hidden_states.reshape(
            num_minibatches,
            sequences_per_minibatch,
            *minibatch_initial_hidden_states.shape[1:],
        )
        minibatches = SequenceBatch(
            initial_hidden_states=minibatch_initial_hidden_states,
            episode_starts=pack_rollout_leaf(episode_starts),
            rollout=jax.tree.map(pack_rollout_leaf, data),
            advantages=pack_rollout_leaf(advantages),
            value_targets=pack_rollout_leaf(value_targets),
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
    train_state: TrainState,
    envs: CraftaxEnv,
    env_states: CraftaxEnvState,
    observations: Observation,
    episode_starts: Bool[Array, " env"],
    hidden_states: Float[Array, "env hidden"],
    key: PRNGKeyArray,
    rollout_length: int,
    observation_dtype: DTypeLike = jnp.float32,
) -> tuple[
    CraftaxEnvState,
    Observation,
    Bool[Array, " env"],
    Float[Array, "env hidden"],
    PRNGKeyArray,
    Rollout,
]:
    def environment_step(carry, _):
        env_states, observations, episode_starts, hidden_states, key = carry
        key, action_key = jax.random.split(key)

        hidden_states, _, distribution, values = train_state.apply_fn(
            train_state.params,
            hidden_states,
            (observations[None], episode_starts[None]),
        )
        actions, log_probs = distribution.sample_and_log_prob(seed=action_key)
        actions = actions.squeeze(0)
        log_probs = log_probs.squeeze(0)
        values = values.squeeze(0)

        env_states, timestep = envs.step(env_states, actions)
        next_observations = timestep.info["next_observation"]
        _, _, _, next_values = train_state.apply_fn(
            train_state.params,
            hidden_states,
            (
                next_observations[None],
                jnp.zeros_like(episode_starts)[None],
            ),
        )
        next_values = next_values.squeeze(0)
        next_episode_starts = timestep.terminated | timestep.truncated

        transition = Rollout(
            observations=observations.astype(observation_dtype),
            actions=actions,
            rewards=timestep.reward,
            terminations=timestep.terminated,
            truncations=timestep.truncated,
            values=values,
            next_values=next_values,
            log_probs=log_probs,
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
            hidden_states,
            key,
        )
        return carry, transition

    carry = (env_states, observations, episode_starts, hidden_states, key)
    carry, data = jax.lax.scan(
        environment_step,
        carry,
        None,
        length=rollout_length,
    )
    env_states, observations, episode_starts, hidden_states, key = carry
    return (
        env_states,
        observations,
        episode_starts,
        hidden_states,
        key,
        data,
    )


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
    NUM_ENVS: int = 1024
    ROLLOUT_LENGTH: int = 64
    BPTT_LENGTH: int = 64
    NUM_MINIBATCHES: int = 8
    NUM_EPOCHS: int = 4

    HIDDEN_SIZE: int = 512
    LEARNING_RATE: float = 2e-4
    ADAM_EPS: float = 1e-5
    ANNEAL_LEARNING_RATE: bool = True
    MAX_GRAD_NORM: float | None = 1.0
    ENTROPY_COEFF: float = 0.01
    VF_COEFF: float = 0.5
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.8
    REWARD_SCALE: float = 1.0
    CLIP_EPS: float = 0.2
    VALUE_CLIP_EPS: float | None = 0.2
    NORM_ADVANTAGE: bool = True

    EVAL_FREQUENCY: int = 10_000_000
    NUM_EVAL_ENVS: int = 128
    GREEDY_EVALUATION: bool = True

    WANDB_ENTITY: str = "evangelos-ch"
    WANDB_PROJECT: str = "mixed-precision-rl"
    WANDB_MODE: Literal["online", "offline", "disabled", "shared"] = "online"
    WANDB_RUN_NAME: str | None = None


def main(args: Args) -> None:
    jax.config.update("jax_default_matmul_precision", args.MATMUL_PRECISION)

    if args.ROLLOUT_LENGTH % args.BPTT_LENGTH:
        raise ValueError("BPTT_LENGTH must divide ROLLOUT_LENGTH")
    num_chunks = args.ROLLOUT_LENGTH // args.BPTT_LENGTH
    num_sequences = num_chunks * args.NUM_ENVS
    if num_sequences % args.NUM_MINIBATCHES:
        raise ValueError(
            "(NUM_ENVS * ROLLOUT_LENGTH / BPTT_LENGTH) must be divisible by "
            "NUM_MINIBATCHES"
        )

    transitions_per_update = args.NUM_ENVS * args.ROLLOUT_LENGTH
    num_updates = math.ceil(args.TOTAL_TIMESTEPS / transitions_per_update)
    actual_timesteps = num_updates * transitions_per_update

    run = wandb.init(
        entity=args.WANDB_ENTITY,
        project=args.WANDB_PROJECT,
        name=args.WANDB_RUN_NAME
        or (f"ppo_rnn_{args.ENV_NAME}_{args.COMPUTE_DTYPE.name}_{args.SEED}"),
        tags=["ppo", "rnn", args.ENV_NAME, args.COMPUTE_DTYPE.name],
        mode=args.WANDB_MODE,
        config={
            **asdict(args),
            "ACTUAL_TIMESTEPS": actual_timesteps,
            "TRANSITIONS_PER_UPDATE": transitions_per_update,
            "NUM_SEQUENCES": num_sequences,
        },
    )

    envs = CraftaxEnv(
        env_name=args.ENV_NAME,
        num_envs=args.NUM_ENVS,
        num_eval_envs=args.NUM_EVAL_ENVS,
        max_episode_length=args.EPISODE_LENGTH,
        reset_ratio=args.RESET_RATIO,
        next_obs_in_extras=True,
    )

    key, network_key, env_key, eval_key = jax.random.split(
        jax.random.PRNGKey(args.SEED), 4
    )
    env_states, observations = jax.jit(envs.init)(env_key)
    hidden_states = ScannedRNN.initialize_carry(
        args.NUM_ENVS,
        args.HIDDEN_SIZE,
        args.COMPUTE_DTYPE(),
    )
    episode_starts = jnp.zeros(args.NUM_ENVS, dtype=jnp.bool_)

    network = ActorCriticRNN(
        action_dim=envs.action_space.num_actions,
        hidden_size=args.HIDDEN_SIZE,
        dtype=args.COMPUTE_DTYPE(),
    )
    params = network.lazy_init(
        network_key,
        hidden_states,
        (observations[None], episode_starts[None]),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=make_optimizer(
            args.LEARNING_RATE,
            args.ADAM_EPS,
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
        hidden_states=hidden_states,
        key=key,
    )

    def evaluation_agent(observations, rng, params, hidden_states):
        hidden_states, _, distribution, _ = train_state.apply_fn(
            params,
            hidden_states,
            (
                observations[None],
                jnp.zeros((1, observations.shape[0]), dtype=jnp.bool_),
            ),
        )
        if args.GREEDY_EVALUATION:
            actions = distribution.mode().squeeze(0)
        else:
            actions = distribution.sample(seed=rng).squeeze(0)
        return hidden_states, jnp.asarray(actions, dtype=jnp.int32)

    initial_evaluation_hidden_states = ScannedRNN.initialize_carry(
        args.NUM_EVAL_ENVS,
        args.HIDDEN_SIZE,
        args.COMPUTE_DTYPE(),
    )
    evaluation = (
        envs.make_evaluation(
            evaluation_agent,
            initial_evaluation_hidden_states,
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
        (
            env_states,
            observations,
            episode_starts,
            hidden_states,
            key,
            data,
        ) = rollout(
            state.train_state,
            envs,
            state.env_states,
            state.observations,
            state.episode_starts,
            state.hidden_states,
            state.key,
            args.ROLLOUT_LENGTH,
            args.ROLLOUT_OBSERVATION_DTYPE(),
        )
        (train_state, key), metrics = ppo_update(
            state.train_state,
            initial_hidden_states,
            initial_episode_starts,
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
            args.BPTT_LENGTH,
            args.NUM_MINIBATCHES,
            args.NUM_EPOCHS,
        )

        done = data.terminations | data.truncations
        logs = jax.tree.map(jnp.mean, metrics)
        logs.update(
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
                hidden_states=hidden_states,
                key=key,
            ),
            logs,
        )

    if evaluation is not None:
        eval_key, evaluation_key = jax.random.split(eval_key)
        eval_logs = jax.device_get(evaluation(evaluation_key, state.train_state.params))
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
            f"loss={host_logs['training/total_loss']:.3f}, "
            f"sps={host_logs['training/sps']:.0f}"
        )
        if environment_steps >= next_eval_step and evaluation is not None:
            eval_key, evaluation_key = jax.random.split(eval_key)
            eval_logs = jax.device_get(
                evaluation(evaluation_key, state.train_state.params)
            )
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
