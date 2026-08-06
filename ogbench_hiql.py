from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from functools import partial
import math
import time
from typing import Any, cast, Literal, NamedTuple

import distrax
from flax.core import freeze, FrozenDict
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float
import optax

import gymnasium
import numpy as np
import ogbench
import tyro
import wandb

from mixed_precision_rl.types import DType


class HIQLBatch(NamedTuple):
    observations: Float[np.ndarray | Array, "batch observation"]
    actions: Float[np.ndarray | Array, "batch action"]
    next_observations: Float[np.ndarray | Array, "batch observation"]
    value_goals: Float[np.ndarray | Array, "batch observation"]
    low_actor_goals: Float[np.ndarray | Array, "batch observation"]
    high_actor_goals: Float[np.ndarray | Array, "batch observation"]
    high_actor_targets: Float[np.ndarray | Array, "batch observation"]
    rewards: Float[np.ndarray | Array, " batch"]
    masks: Float[np.ndarray | Array, " batch"]


class HIQLDataset:
    """Host-side sampler for OGBench's compact trajectory dataset."""

    def __init__(
        self,
        dataset: Mapping[str, np.ndarray],
        batch_size: int,
        seed: int,
        discount: float,
        subgoal_steps: int,
        value_current_goal_probability: float,
        value_trajectory_goal_probability: float,
        actor_random_goal_probability: float,
        value_geometric_sampling: bool,
        actor_geometric_sampling: bool,
    ) -> None:
        required_keys = {"observations", "actions", "terminals", "valids"}
        missing_keys = required_keys.difference(dataset)
        if missing_keys:
            raise ValueError(
                "HIQL requires an OGBench compact dataset; missing "
                f"{sorted(missing_keys)}"
            )

        self.observations = np.asarray(dataset["observations"], dtype=np.float32)
        self.actions = np.asarray(dataset["actions"], dtype=np.float32)
        terminals = np.asarray(dataset["terminals"])
        valids = np.asarray(dataset["valids"])
        if self.observations.ndim != 2:
            raise ValueError(
                "This first HIQL implementation supports state observations only; "
                f"received shape {self.observations.shape}"
            )
        if self.actions.ndim != 2:
            raise ValueError(
                "This first HIQL implementation supports continuous vector actions "
                f"only; received shape {self.actions.shape}"
            )
        if not (
            len(self.observations) == len(self.actions) == len(terminals) == len(valids)
        ):
            raise ValueError("OGBench dataset fields have inconsistent lengths")

        self.valid_indices = np.flatnonzero(valids > 0)
        self.terminal_locations = np.flatnonzero(terminals > 0)
        if not len(self.valid_indices):
            raise ValueError("OGBench dataset has no valid transitions")
        if not len(self.terminal_locations) or self.terminal_locations[-1] != (
            len(self.observations) - 1
        ):
            raise ValueError("OGBench compact dataset has malformed trajectories")

        self.batch_size = batch_size
        self.discount = discount
        self.subgoal_steps = subgoal_steps
        self.value_current_goal_probability = value_current_goal_probability
        self.value_trajectory_goal_probability = value_trajectory_goal_probability
        self.actor_random_goal_probability = actor_random_goal_probability
        self.value_geometric_sampling = value_geometric_sampling
        self.actor_geometric_sampling = actor_geometric_sampling
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[HIQLBatch]:
        while True:
            yield self.sample()

    def random_indices(self, size: int) -> np.ndarray:
        positions = self.rng.integers(len(self.valid_indices), size=size)
        return self.valid_indices[positions]

    def trajectory_goal_indices(
        self,
        indices: np.ndarray,
        final_indices: np.ndarray,
        geometric: bool,
    ) -> np.ndarray:
        if geometric:
            offsets = self.rng.geometric(1.0 - self.discount, size=len(indices))
            return np.minimum(indices + offsets, final_indices)

        distances = self.rng.random(len(indices))
        first_future_indices = np.minimum(indices + 1, final_indices)
        return np.round(
            first_future_indices * distances + final_indices * (1.0 - distances)
        ).astype(np.int64)

    def sample(self) -> HIQLBatch:
        indices = self.random_indices(self.batch_size)
        final_indices = self.terminal_locations[
            np.searchsorted(self.terminal_locations, indices)
        ]

        random_value_goal_indices = self.random_indices(self.batch_size)
        trajectory_value_goal_indices = self.trajectory_goal_indices(
            indices, final_indices, self.value_geometric_sampling
        )
        if self.value_current_goal_probability == 1.0:
            value_goal_indices = indices
        else:
            trajectory_probability = self.value_trajectory_goal_probability / (
                1.0 - self.value_current_goal_probability
            )
            value_goal_indices = np.where(
                self.rng.random(self.batch_size) < trajectory_probability,
                trajectory_value_goal_indices,
                random_value_goal_indices,
            )
            value_goal_indices = np.where(
                self.rng.random(self.batch_size) < self.value_current_goal_probability,
                indices,
                value_goal_indices,
            )

        low_actor_goal_indices = np.minimum(indices + self.subgoal_steps, final_indices)
        high_trajectory_goal_indices = self.trajectory_goal_indices(
            indices, final_indices, self.actor_geometric_sampling
        )
        high_trajectory_target_indices = np.minimum(
            indices + self.subgoal_steps, high_trajectory_goal_indices
        )
        high_random_goal_indices = self.random_indices(self.batch_size)
        high_random_target_indices = np.minimum(
            indices + self.subgoal_steps, final_indices
        )
        use_random_high_goal = (
            self.rng.random(self.batch_size) < self.actor_random_goal_probability
        )
        high_actor_goal_indices = np.where(
            use_random_high_goal,
            high_random_goal_indices,
            high_trajectory_goal_indices,
        )
        high_actor_target_indices = np.where(
            use_random_high_goal,
            high_random_target_indices,
            high_trajectory_target_indices,
        )

        successes = (indices == value_goal_indices).astype(np.float32)
        return HIQLBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            next_observations=self.observations[indices + 1],
            value_goals=self.observations[value_goal_indices],
            low_actor_goals=self.observations[low_actor_goal_indices],
            high_actor_goals=self.observations[high_actor_goal_indices],
            high_actor_targets=self.observations[high_actor_target_indices],
            rewards=successes - 1.0,
            masks=1.0 - successes,
        )


class MLP(nn.Module):
    width: int
    depth: int
    output_dim: int
    layer_norm: bool
    dtype: DTypeLike
    output_kernel_init: Callable[..., Array] = nn.initializers.variance_scaling(
        1.0, "fan_avg", "uniform"
    )

    @nn.compact
    def __call__(self, inputs: Float[Array, "... input"]) -> Float[Array, "... output"]:
        x = inputs
        kernel_init = nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")
        for _ in range(self.depth):
            x = nn.Dense(
                self.width,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                kernel_init=kernel_init,
            )(x)
            x = nn.gelu(x)
            if self.layer_norm:
                x = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x)
        return nn.Dense(
            self.output_dim,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=self.output_kernel_init,
        )(x)


class HIQLNetworks(nn.Module):
    action_dim: int
    width: int
    depth: int
    representation_dim: int
    layer_norm: bool
    dtype: DTypeLike

    def setup(self) -> None:
        self.goal_encoder = MLP(
            width=self.width,
            depth=self.depth,
            output_dim=self.representation_dim,
            layer_norm=self.layer_norm,
            dtype=self.dtype,
        )
        value_ensemble = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        self.value_ensemble = value_ensemble(
            width=self.width,
            depth=self.depth,
            output_dim=1,
            layer_norm=self.layer_norm,
            dtype=self.dtype,
        )
        actor_output_kernel_init = nn.initializers.variance_scaling(
            1e-2, "fan_avg", "uniform"
        )
        self.low_actor = MLP(
            width=self.width,
            depth=self.depth,
            output_dim=self.action_dim,
            layer_norm=False,
            dtype=self.dtype,
            output_kernel_init=actor_output_kernel_init,
        )
        self.high_actor = MLP(
            width=self.width,
            depth=self.depth,
            output_dim=self.representation_dim,
            layer_norm=False,
            dtype=self.dtype,
            output_kernel_init=actor_output_kernel_init,
        )

    def encode_goal(
        self,
        observations: Float[Array, "... observation"],
        goals: Float[Array, "... observation"],
    ) -> Float[Array, "... representation"]:
        representation = self.goal_encoder(
            jnp.concatenate((observations, goals), axis=-1)
        ).astype(jnp.float32)
        norm = jnp.linalg.norm(representation, axis=-1, keepdims=True)
        return (
            representation
            / jnp.maximum(norm, 1e-6)
            * math.sqrt(self.representation_dim)
        )

    def values(
        self,
        observations: Float[Array, "... observation"],
        goals: Float[Array, "... observation"],
    ) -> Float[Array, "ensemble ..."]:
        goal_representations = self.encode_goal(observations, goals)
        inputs = jnp.concatenate((observations, goal_representations), axis=-1)
        return self.value_ensemble(inputs).squeeze(-1).astype(jnp.float32)

    def low_actor_means(
        self,
        observations: Float[Array, "... observation"],
        goal_representations: Float[Array, "... representation"],
    ) -> Float[Array, "... action"]:
        return self.low_actor(
            jnp.concatenate((observations, goal_representations), axis=-1)
        ).astype(jnp.float32)

    def high_actor_means(
        self,
        observations: Float[Array, "... observation"],
        goals: Float[Array, "... observation"],
    ) -> Float[Array, "... representation"]:
        return self.high_actor(jnp.concatenate((observations, goals), axis=-1)).astype(
            jnp.float32
        )

    def __call__(
        self,
        observations: Float[Array, "... observation"],
        goals: Float[Array, "... observation"],
    ) -> tuple[Array, Array, Array]:
        goal_representations = self.encode_goal(observations, goals)
        return (
            self.values(observations, goals),
            self.low_actor_means(observations, goal_representations),
            self.high_actor_means(observations, goals),
        )


class HIQLTrainState(TrainState):
    target_value_params: FrozenDict[str, Any]


@partial(
    jax.jit,
    static_argnames=(
        "discount",
        "tau",
        "expectile",
        "low_alpha",
        "high_alpha",
        "low_actor_representation_gradient",
        "logs",
    ),
    donate_argnums=(0,),
)
def hiql_update(
    state: HIQLTrainState,
    batch: HIQLBatch,
    discount: float,
    tau: float,
    expectile: float,
    low_alpha: float,
    high_alpha: float,
    low_actor_representation_gradient: bool,
    logs: bool,
) -> tuple[HIQLTrainState, dict[str, Array]]:
    target_variables = {"params": state.target_value_params}
    fixed_variables = {"params": state.params}

    next_target_values = state.apply_fn(
        target_variables,
        batch.next_observations,
        batch.value_goals,
        method=HIQLNetworks.values,
    )
    target_values = state.apply_fn(
        target_variables,
        batch.observations,
        batch.value_goals,
        method=HIQLNetworks.values,
    )
    minimum_next_target_value = next_target_values.min(axis=0)
    q = batch.rewards + discount * batch.masks * minimum_next_target_value
    advantage = q - target_values.mean(axis=0)
    q_ensemble = batch.rewards[None] + discount * batch.masks[None] * next_target_values

    low_values = state.apply_fn(
        fixed_variables,
        batch.observations,
        batch.low_actor_goals,
        method=HIQLNetworks.values,
    ).mean(axis=0)
    next_low_values = state.apply_fn(
        fixed_variables,
        batch.next_observations,
        batch.low_actor_goals,
        method=HIQLNetworks.values,
    ).mean(axis=0)
    low_advantage = next_low_values - low_values
    low_weights = jnp.minimum(jnp.exp(low_alpha * low_advantage), 100.0)

    high_values = state.apply_fn(
        fixed_variables,
        batch.observations,
        batch.high_actor_goals,
        method=HIQLNetworks.values,
    ).mean(axis=0)
    next_high_values = state.apply_fn(
        fixed_variables,
        batch.high_actor_targets,
        batch.high_actor_goals,
        method=HIQLNetworks.values,
    ).mean(axis=0)
    high_advantage = next_high_values - high_values
    high_weights = jnp.minimum(jnp.exp(high_alpha * high_advantage), 100.0)
    high_targets = state.apply_fn(
        fixed_variables,
        batch.observations,
        batch.high_actor_targets,
        method=HIQLNetworks.encode_goal,
    )

    def loss(params):
        variables = {"params": params}
        values = state.apply_fn(
            variables,
            batch.observations,
            batch.value_goals,
            method=HIQLNetworks.values,
        )
        expectile_weights = jnp.where(advantage >= 0.0, expectile, 1.0 - expectile)
        value_loss = (
            (expectile_weights[None] * jnp.square(q_ensemble - values))
            .mean(axis=1)
            .sum()
        )

        low_goal_representations = state.apply_fn(
            variables,
            batch.observations,
            batch.low_actor_goals,
            method=HIQLNetworks.encode_goal,
        )
        if not low_actor_representation_gradient:
            low_goal_representations = jax.lax.stop_gradient(low_goal_representations)
        low_means = state.apply_fn(
            variables,
            batch.observations,
            low_goal_representations,
            method=HIQLNetworks.low_actor_means,
        )
        low_log_probabilities = distrax.MultivariateNormalDiag(
            low_means, jnp.ones_like(low_means)
        ).log_prob(batch.actions)
        low_actor_loss = -(low_weights * low_log_probabilities).mean()

        high_means = state.apply_fn(
            variables,
            batch.observations,
            batch.high_actor_goals,
            method=HIQLNetworks.high_actor_means,
        )
        high_log_probabilities = distrax.MultivariateNormalDiag(
            high_means, jnp.ones_like(high_means)
        ).log_prob(high_targets)
        high_actor_loss = -(high_weights * high_log_probabilities).mean()

        total_loss = value_loss + low_actor_loss + high_actor_loss
        if logs:
            metrics = {
                "total_loss": total_loss,
                "value/loss": value_loss,
                "value/mean": values.mean(),
                "value/max": values.max(),
                "value/min": values.min(),
                "low_actor/loss": low_actor_loss,
                "low_actor/advantage": low_advantage.mean(),
                "low_actor/bc_log_probability": low_log_probabilities.mean(),
                "low_actor/mse": jnp.square(low_means - batch.actions).mean(),
                "high_actor/loss": high_actor_loss,
                "high_actor/advantage": high_advantage.mean(),
                "high_actor/bc_log_probability": high_log_probabilities.mean(),
                "high_actor/mse": jnp.square(high_means - high_targets).mean(),
            }
        else:
            metrics = {}
        return total_loss, metrics

    (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(state.params)
    if logs:
        metrics["gradient_norm"] = optax.global_norm(gradients)
    state = state.apply_gradients(grads=gradients)
    online_value_params = freeze(
        {name: state.params[name] for name in ("goal_encoder", "value_ensemble")}
    )
    target_value_params = optax.incremental_update(
        online_value_params, state.target_value_params, tau
    )
    return state.replace(target_value_params=target_value_params), metrics


@dataclass
class Args:
    SEED: int = 43
    DATASET_NAME: str = "antmaze-large-navigate-v0"
    DATASET_DIR: str = "~/.ogbench/data"
    DATASET_PATH: str | None = None
    MATMUL_PRECISION: Literal["default", "high", "highest"] = "highest"
    COMPUTE_DTYPE: DType = DType.float32

    TRAIN_STEPS: int = 1_000_000
    BATCH_SIZE: int = 1024
    WIDTH: int = 512
    DEPTH: int = 3
    REPRESENTATION_DIM: int = 10
    LAYER_NORM: bool = True

    LEARNING_RATE: float = 3e-4
    MAX_GRAD_NORM: float | None = None
    DISCOUNT: float = 0.99
    TAU: float = 0.005
    EXPECTILE: float = 0.7
    LOW_ALPHA: float = 3.0
    HIGH_ALPHA: float = 3.0
    SUBGOAL_STEPS: int = 25
    LOW_ACTOR_REPRESENTATION_GRADIENT: bool = False

    VALUE_CURRENT_GOAL_PROBABILITY: float = 0.2
    VALUE_TRAJECTORY_GOAL_PROBABILITY: float = 0.5
    VALUE_RANDOM_GOAL_PROBABILITY: float = 0.3
    ACTOR_RANDOM_GOAL_PROBABILITY: float = 0.0
    VALUE_GEOMETRIC_SAMPLING: bool = True
    ACTOR_GEOMETRIC_SAMPLING: bool = False

    LOG_INTERVAL: int = 5_000
    EVAL_INTERVAL: int = 100_000
    EVAL_TASKS: int | None = None
    EVAL_EPISODES: int = 20
    EVAL_TEMPERATURE: float = 0.0
    EVAL_ON_CPU: bool = True

    WANDB_ENTITY: str = "evangelos-ch"
    WANDB_PROJECT: str = "mixed-precision-rl"
    WANDB_MODE: Literal["online", "offline", "disabled", "shared"] = "online"
    WANDB_RUN_NAME: str | None = None

    def __post_init__(self) -> None:
        if self.TRAIN_STEPS < 1:
            raise ValueError("TRAIN_STEPS must be positive")
        if self.BATCH_SIZE < 1 or self.WIDTH < 1 or self.DEPTH < 1:
            raise ValueError("BATCH_SIZE, WIDTH, and DEPTH must be positive")
        if self.REPRESENTATION_DIM < 1 or self.SUBGOAL_STEPS < 1:
            raise ValueError("REPRESENTATION_DIM and SUBGOAL_STEPS must be positive")
        if not 0.0 < self.DISCOUNT < 1.0:
            raise ValueError("DISCOUNT must be between zero and one")
        if not 0.0 < self.EXPECTILE < 1.0:
            raise ValueError("EXPECTILE must be between zero and one")
        if not 0.0 <= self.TAU <= 1.0:
            raise ValueError("TAU must be between zero and one")
        value_goal_probabilities = (
            self.VALUE_CURRENT_GOAL_PROBABILITY,
            self.VALUE_TRAJECTORY_GOAL_PROBABILITY,
            self.VALUE_RANDOM_GOAL_PROBABILITY,
        )
        if any(
            probability < 0.0 or probability > 1.0
            for probability in value_goal_probabilities
        ):
            raise ValueError("value goal probabilities must be in [0, 1]")
        if not math.isclose(sum(value_goal_probabilities), 1.0):
            raise ValueError("value goal probabilities must sum to one")
        if not 0.0 <= self.ACTOR_RANDOM_GOAL_PROBABILITY <= 1.0:
            raise ValueError("ACTOR_RANDOM_GOAL_PROBABILITY must be in [0, 1]")
        if self.LOG_INTERVAL < 0 or self.EVAL_INTERVAL < 0:
            raise ValueError("logging and evaluation intervals cannot be negative")
        if self.EVAL_TASKS is not None and self.EVAL_TASKS < 1:
            raise ValueError("EVAL_TASKS must be positive when set")
        if self.EVAL_EPISODES < 1:
            raise ValueError("EVAL_EPISODES must be positive")
        if self.EVAL_TEMPERATURE < 0.0:
            raise ValueError("EVAL_TEMPERATURE cannot be negative")


def evaluate(
    env: gymnasium.Env,
    params: FrozenDict[str, Any],
    apply_fn: Callable[..., Any],
    representation_dim: int,
    key: Array,
    num_tasks: int | None,
    num_episodes: int,
    temperature: float,
    on_cpu: bool,
) -> dict[str, float]:
    unwrapped_env = cast(Any, env.unwrapped)
    task_infos = cast(list[dict[str, Any]], unwrapped_env.task_infos)
    evaluation_tasks = len(task_infos) if num_tasks is None else num_tasks
    if evaluation_tasks > len(task_infos):
        raise ValueError(
            f"EVAL_TASKS={evaluation_tasks} exceeds the environment's "
            f"{len(task_infos)} tasks"
        )

    evaluation_device = jax.devices("cpu")[0] if on_cpu else None
    evaluation_params = jax.device_put(params, evaluation_device)
    key = jax.device_put(key, evaluation_device)

    def act(
        actor_params: FrozenDict[str, Any],
        observation: Array,
        goal: Array,
        action_key: Array,
    ) -> Array:
        high_key, low_key = jax.random.split(action_key)
        variables = {"params": actor_params}
        high_means = apply_fn(
            variables,
            observation,
            goal,
            method=HIQLNetworks.high_actor_means,
        )
        if temperature:
            high_actions = high_means + temperature * jax.random.normal(
                high_key, high_means.shape
            )
        else:
            high_actions = high_means
        high_norms = jnp.linalg.norm(high_actions, axis=-1, keepdims=True)
        goal_representations = (
            high_actions / jnp.maximum(high_norms, 1e-6) * math.sqrt(representation_dim)
        )
        low_means = apply_fn(
            variables,
            observation,
            goal_representations,
            method=HIQLNetworks.low_actor_means,
        )
        if temperature:
            actions = low_means + temperature * jax.random.normal(
                low_key, low_means.shape
            )
        else:
            actions = low_means
        return jnp.clip(actions, -1.0, 1.0)

    action_fn = jax.jit(act)
    logs: dict[str, float] = {}
    task_successes = []
    for task_id in range(1, evaluation_tasks + 1):
        task_name = task_infos[task_id - 1]["task_name"]
        successes = []
        for episode in range(num_episodes):
            key, reset_key = jax.random.split(key)
            reset_seed = int(
                jax.device_get(jax.random.randint(reset_key, (), 0, 2**31 - 1))
            )
            observation, info = env.reset(seed=reset_seed, options={"task_id": task_id})
            goal = np.asarray(info["goal"], dtype=np.float32)
            done = False
            while not done:
                key, action_key = jax.random.split(key)
                action = np.asarray(
                    action_fn(
                        evaluation_params,
                        np.asarray(observation, dtype=np.float32),
                        goal,
                        action_key,
                    )
                )
                observation, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            successes.append(float(info["success"]))
        success = float(np.mean(successes))
        logs[f"evaluation/{task_name}_success"] = success
        task_successes.append(success)
        print(f"evaluation task {task_id}/{evaluation_tasks}: success={success:.3f}")
    logs["evaluation/overall_success"] = float(np.mean(task_successes))
    return logs


def main(args: Args) -> None:
    jax.config.update("jax_default_matmul_precision", args.MATMUL_PRECISION)

    env, train_data, validation_data = ogbench.make_env_and_datasets(
        args.DATASET_NAME,
        dataset_dir=args.DATASET_DIR,
        dataset_path=args.DATASET_PATH,
        compact_dataset=True,
    )
    del validation_data
    if not isinstance(env.action_space, gymnasium.spaces.Box):
        env.close()
        raise TypeError(
            "This first HIQL implementation supports continuous actions only; "
            f"received {type(env.action_space).__name__}"
        )

    dataset = HIQLDataset(
        train_data,
        batch_size=args.BATCH_SIZE,
        seed=args.SEED,
        discount=args.DISCOUNT,
        subgoal_steps=args.SUBGOAL_STEPS,
        value_current_goal_probability=args.VALUE_CURRENT_GOAL_PROBABILITY,
        value_trajectory_goal_probability=args.VALUE_TRAJECTORY_GOAL_PROBABILITY,
        actor_random_goal_probability=args.ACTOR_RANDOM_GOAL_PROBABILITY,
        value_geometric_sampling=args.VALUE_GEOMETRIC_SAMPLING,
        actor_geometric_sampling=args.ACTOR_GEOMETRIC_SAMPLING,
    )
    del train_data
    action_dim = int(np.prod(env.action_space.shape))

    init_key, evaluation_key = jax.random.split(jax.random.PRNGKey(args.SEED))
    example_batch = dataset.sample()
    example_observations = jnp.asarray(example_batch.observations[:1])
    example_goals = jnp.asarray(example_batch.value_goals[:1])
    network = HIQLNetworks(
        action_dim=action_dim,
        width=args.WIDTH,
        depth=args.DEPTH,
        representation_dim=args.REPRESENTATION_DIM,
        layer_norm=args.LAYER_NORM,
        dtype=args.COMPUTE_DTYPE(),
    )
    params = network.init(init_key, example_observations, example_goals)["params"]
    target_value_params = freeze(
        {
            name: jax.tree.map(jnp.copy, params[name])
            for name in ("goal_encoder", "value_ensemble")
        }
    )
    transforms = []
    if args.MAX_GRAD_NORM is not None:
        transforms.append(optax.clip_by_global_norm(args.MAX_GRAD_NORM))
    transforms.append(optax.adam(args.LEARNING_RATE))
    state = HIQLTrainState.create(
        apply_fn=network.apply,
        params=params,
        target_value_params=target_value_params,
        tx=optax.chain(*transforms),
    )

    run = wandb.init(
        entity=args.WANDB_ENTITY,
        project=args.WANDB_PROJECT,
        name=args.WANDB_RUN_NAME
        or (f"hiql_{args.DATASET_NAME}_{args.COMPUTE_DTYPE.name}_{args.SEED}"),
        tags=["hiql", "ogbench", args.DATASET_NAME, args.COMPUTE_DTYPE.name],
        mode=args.WANDB_MODE,
        config={
            **asdict(args),
            "DEVICE_KIND": jax.local_devices()[0].device_kind,
            "NUM_DEVICES": len(jax.local_devices()),
            "OBSERVATION_DIM": dataset.observations.shape[-1],
            "ACTION_DIM": action_dim,
            "DATASET_SIZE": len(dataset.observations),
        },
    )

    start_time = time.monotonic()
    evaluation_time = 0.0

    def log_training(step: int, device_metrics: Mapping[str, Array]) -> None:
        host_metrics = jax.device_get(device_metrics)
        elapsed = max(time.monotonic() - start_time - evaluation_time, 1e-6)
        logs = {
            f"training/{name}": float(value) for name, value in host_metrics.items()
        }
        logs["time/updates_per_second"] = step / elapsed
        run.log(logs, step=step)
        print(
            f"{step}: loss={logs['training/total_loss']:.3f}, "
            f"value={logs['training/value/loss']:.3f}, "
            f"low_actor={logs['training/low_actor/loss']:.3f}, "
            f"high_actor={logs['training/high_actor/loss']:.3f}, "
            f"ups={logs['time/updates_per_second']:.1f}"
        )

    # Both calls are asynchronous. Each update is enqueued before the previous
    # logged metrics are materialized, and host sampling/device transfer of the
    # next batch happens while the accelerator executes the current update.
    device_batches = map(jax.device_put, iter(dataset))
    batch = next(device_batches)
    pending_metrics: tuple[int, Mapping[str, Array]] | None = None
    for step in range(1, args.TRAIN_STEPS + 1):
        should_log = args.LOG_INTERVAL > 0 and (
            step == 1 or step % args.LOG_INTERVAL == 0
        )
        should_evaluate = args.EVAL_INTERVAL > 0 and (
            step == 1 or step % args.EVAL_INTERVAL == 0
        )
        state, metrics = hiql_update(
            state,
            batch,
            discount=args.DISCOUNT,
            tau=args.TAU,
            expectile=args.EXPECTILE,
            low_alpha=args.LOW_ALPHA,
            high_alpha=args.HIGH_ALPHA,
            low_actor_representation_gradient=(args.LOW_ACTOR_REPRESENTATION_GRADIENT),
            logs=should_log,
        )
        batch = next(device_batches)

        if pending_metrics is not None:
            log_training(*pending_metrics)
            pending_metrics = None

        if should_evaluate:
            if should_log:
                log_training(step, metrics)
            jax.block_until_ready(state)
            evaluation_start = time.monotonic()
            evaluation_key, current_evaluation_key = jax.random.split(evaluation_key)
            evaluation_logs = evaluate(
                env,
                state.params,
                state.apply_fn,
                args.REPRESENTATION_DIM,
                current_evaluation_key,
                num_tasks=args.EVAL_TASKS,
                num_episodes=args.EVAL_EPISODES,
                temperature=args.EVAL_TEMPERATURE,
                on_cpu=args.EVAL_ON_CPU,
            )
            run.log(evaluation_logs, step=step)
            evaluation_time += time.monotonic() - evaluation_start
            print(
                f"{step}: "
                f"overall_success={evaluation_logs['evaluation/overall_success']:.3f}"
            )
        elif should_log:
            pending_metrics = (step, metrics)

    jax.block_until_ready(state)
    if pending_metrics is not None:
        log_training(*pending_metrics)
    env.close()
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
