from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb
from brax import envs as brax_envs
from brax.envs.base import Env
from flax.training.train_state import TrainState
from jaxtyping import PRNGKeyArray


class Policy(nn.Module):
    width: int = 32
    depth: int = 4

    output_dim: int = 8
    dtype: jnp.dtype = jnp.float32
    activation_fn: Callable[[jax.Array], jax.Array] = nn.swish

    min_std: float = 1e-3
    var_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.width, dtype=self.dtype)(x)
            x = self.activation_fn(x)

        # NOTE: STD could be different in the brax code

        x = nn.Dense(2 * self.output_dim, dtype=self.dtype)(x)
        mean, std = jnp.split(x, 2, axis=-1)
        mean, std = mean.astype(jnp.float32), std.astype(jnp.float32)

        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale

        return distrax.MultivariateNormalDiag(mean, std)


class ValueFunction(nn.Module):
    width: int = 256
    depth: int = 5

    output_dim: int = 1
    dtype: jnp.dtype = jnp.float32
    activation_fn: Callable[[jax.Array], jax.Array] = nn.swish

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.width, dtype=self.dtype)(x)
            x = self.activation_fn(x)

        x = nn.Dense(self.output_dim, dtype=self.dtype)(x)
        return x.astype(jnp.float32)


class Transition(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    values: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    next_observations: jax.Array
    log_probs: jax.Array
    extra: dict


def compute_gae(
    rewards, values, last_values, gamma, gae_lambda, terminations, truncations
):
    truncation_mask = 1 - truncations
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate([values[1:], last_values[None, ...]], axis=0)
    deltas = rewards + gamma * (1 - terminations) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(last_values)

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + gamma * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs,
        (gae_lambda, acc),
        (truncation_mask, deltas, terminations),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate([vs[1:], last_values[None, ...]], axis=0)
    advantages = (
        rewards + gamma * (1 - terminations) * vs_t_plus_1 - values
    ) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def update(
    policy: TrainState,
    vf: TrainState,
    data: Transition,
    key: PRNGKeyArray,
    entropy_coeff,
    vf_coeff,
    gamma,
    gae_lambda,
    reward_scale,
    clip_eps,
    norm_advantage,
    num_minibatches,
    num_epochs,
):
    final_obs = jax.tree.map(lambda x: x[-1], data.next_observations)
    last_values = vf.apply_fn(vf.params, final_obs).squeeze(-1)
    rewards = data.rewards * reward_scale

    value_targets, advantages = compute_gae(
        rewards,
        data.values,
        last_values,
        gamma,
        gae_lambda,
        data.terminations,
        data.truncations,
    )

    def loss(
        policy_and_vf_params,
        data: Transition,
        advantages: jax.Array,
        value_targets: jax.Array,
    ):
        policy_params, vf_params = policy_and_vf_params

        values = vf.apply_fn(vf_params, data.observations).squeeze(-1)

        if norm_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy
        dist = policy.apply_fn(policy_params, data.observations)
        log_probs = dist.log_prob(data.actions)
        ratio = jnp.exp(log_probs - data.log_probs)
        policy_loss = -jnp.minimum(
            ratio * advantages, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        ).mean()

        entropy_loss = entropy_coeff * dist.entropy().mean()

        # VF
        vf_loss = vf_coeff * 0.5 * jnp.power(value_targets - values, 2).mean()

        total_loss = policy_loss + vf_loss + entropy_loss

        return total_loss, {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "vf_loss": vf_loss,
            "entropy_loss": entropy_loss,
        }

    def update_minibatch(carry, xs):
        policy, vf = carry
        data, advantages, value_targets = xs

        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(
            (policy.params, vf.params), data, advantages, value_targets
        )

        policy = policy.apply_gradients(grads=grads[0])
        vf = vf.apply_gradients(grads=grads[1])

        return (policy, vf), metrics

    def update_epoch(carry, _):
        policy, vf, key = carry
        key, perm_key = jax.random.split(key)

        def shuffle(x: jax.Array):
            x = x.reshape(-1, *x.shape[2:])
            x = jax.random.permutation(perm_key, x, axis=0)
            x = jnp.reshape(x, (num_minibatches, -1, *x.shape[1:]))
            return x

        (policy, vf), metrics = jax.lax.scan(
            update_minibatch,
            (policy, vf),
            jax.tree.map(shuffle, (data, advantages, value_targets)),
            length=num_minibatches,
        )

        return (policy, vf, key), metrics

    (policy, vf, key), metrics = jax.lax.scan(
        update_epoch, (policy, vf, key), None, length=num_epochs
    )

    return (policy, vf, key), metrics


def rollout(
    policy: TrainState,
    vf: TrainState,
    envs: Env,
    env_states,
    key: PRNGKeyArray,
    rollout_size: int,
    num_envs: int,
):
    def step(carry, _):
        env_states, key = carry
        key, action_key = jax.random.split(key)
        action, log_prob = policy.apply_fn(
            policy.params, env_states.obs
        ).sample_and_log_prob(seed=action_key)
        values = vf.apply_fn(vf.params, env_states.obs).squeeze(-1)
        next_state = envs.step(env_states, action)

        truncations = next_state.info["truncation"]
        data = Transition(
            observations=env_states.obs,
            actions=action,
            rewards=next_state.reward,
            values=values,
            terminations=next_state.done * (1 - truncations),
            truncations=truncations,
            next_observations=next_state.obs,  # ty: ignore[invalid-argument-type]
            log_probs=log_prob,
            extra={
                "sum_reward": next_state.info["episode_metrics"]["sum_reward"],
                "episode_done": next_state.info["episode_done"],
            },
        )

        return (next_state, key), data

    (env_states, key), data = jax.lax.scan(
        step,
        (env_states, key),
        None,
        length=rollout_size // num_envs,
    )

    return env_states, key, data  # (T, Envs, D)


@dataclass
class Args:
    SEED: int = 43

    TOTAL_TIMESTEPS = 50_000_000

    BATCH_SIZE = 2048
    NUM_ENVS = 4096
    NUM_MINIBATCHES = 32
    UNROLL_LENGTH = 5
    action_repeat = 1

    LEARNING_RATE = 3e-4

    # PPO
    ENTROPY_COEFF = 1e-2
    VF_COEFF = 0.5
    GAMMA = 0.97
    GAE_LAMBDA = 0.95
    REWARD_SCALE = 10.0
    CLIP_EPS = 0.3
    NORM_ADVANTAGE = True
    NUM_EPOCHS = 4


def main(args: Args):
    wandb.init(
        project="debug-runs",
        name=f"ppo_from_scratch_ant_{args.SEED}",
        tags=["ppo", "ant"],
        # mode="disabled",
    )

    envs = brax_envs.training.wrap(
        brax_envs.get_environment(env_name="ant"),
        episode_length=1000,
        action_repeat=args.action_repeat,
    )

    rollout_size = (
        args.BATCH_SIZE * args.NUM_MINIBATCHES * args.UNROLL_LENGTH * args.action_repeat
    )  # Keep rollout size same

    key = jax.random.PRNGKey(args.SEED)
    key, key_policy, key_value, key_env = jax.random.split(key, 4)

    reset_fn = jax.jit(envs.reset)
    key_envs = jax.random.split(key_env, args.NUM_ENVS)
    env_states = reset_fn(key_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], env_states.obs)
    obs_spec = jax.ShapeDtypeStruct(obs_shape, jnp.float32)

    policy_module = Policy()
    policy = TrainState.create(
        apply_fn=policy_module.apply,
        params=policy_module.lazy_init(key_policy, obs_spec),
        tx=optax.adam(learning_rate=args.LEARNING_RATE),
    )

    vf_module = ValueFunction()
    vf = TrainState.create(
        apply_fn=vf_module.apply,
        params=vf_module.lazy_init(key_value, obs_spec),
        tx=optax.adam(learning_rate=args.LEARNING_RATE),
    )

    @jax.jit
    def train_step(policy, vf, env_states, i, key):
        env_states, key, data = rollout(
            policy, vf, envs, env_states, key, rollout_size, args.NUM_ENVS
        )

        (policy, vf, key), metrics = update(
            policy,
            vf,
            data,
            key,
            args.ENTROPY_COEFF,
            args.VF_COEFF,
            args.GAMMA,
            args.GAE_LAMBDA,
            args.REWARD_SCALE,
            args.CLIP_EPS,
            args.NORM_ADVANTAGE,
            args.NUM_MINIBATCHES,
            args.NUM_EPOCHS,
        )

        logs = jax.tree.map(jnp.mean, metrics)

        return policy, vf, env_states, i + rollout_size, key, logs, data.extra

    i = 0
    for _ in range(0, args.TOTAL_TIMESTEPS, rollout_size):
        policy, vf, env_states, i, key, logs, extras = train_step(
            policy, vf, env_states, i, key
        )
        logs = logs | {
            "episode_return": extras["sum_reward"][
                extras["episode_done"].astype(bool)
            ].mean()
        }
        wandb.log(logs, step=i)

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
