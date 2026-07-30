import jax.numpy as jnp

import numpy as np

from craftax_pqn import compute_q_lambda_targets


def test_q_lambda_targets_mix_one_step_and_multistep_returns() -> None:
    q_values = jnp.array(
        [
            [[0.0, 0.0]],
            [[8.0, 10.0]],
            [[20.0, 15.0]],
        ]
    )
    rewards = jnp.array([[1.0], [2.0], [0.0]])
    done = jnp.zeros_like(rewards, dtype=jnp.bool_)

    targets = compute_q_lambda_targets(
        q_values,
        rewards,
        done,
        gamma=0.9,
        q_lambda=0.5,
    )

    np.testing.assert_allclose(targets, [[14.5], [20.0]])


def test_q_lambda_targets_do_not_bootstrap_across_episode_boundaries() -> None:
    q_values = jnp.array(
        [
            [[0.0, 0.0]],
            [[8.0, 10.0]],
            [[20.0, 15.0]],
        ]
    )
    rewards = jnp.array([[1.0], [2.0], [0.0]])
    done = jnp.array([[True], [False], [False]])

    targets = compute_q_lambda_targets(
        q_values,
        rewards,
        done,
        gamma=0.9,
        q_lambda=0.5,
    )

    np.testing.assert_allclose(targets, [[1.0], [20.0]])
