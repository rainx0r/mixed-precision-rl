#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
suite_start_seconds=$SECONDS

# Run the three noisy SAC tasks with the MuJoCo Playground paper's
# hyperparameters, changing only the environment backend from Warp to JAX.
env_names=(
  FingerSpin
  FingerTurnEasy
  PendulumSwingup
)

num_env_names=${#env_names[@]}
for env_index in "${!env_names[@]}"; do
  env_name=${env_names[$env_index]}
  env_number=$((env_index + 1))
  action_repeat=1

  if [[ "$env_name" == "PendulumSwingup" ]]; then
    action_repeat=4
  fi

  for seed in 0 1 2 3 4; do
    echo "===== ${env_name} (${env_number}/${num_env_names}) JAX Seed ${seed} ====="
    uv run dmc_sac.py \
      --SEED "$seed" \
      --ENV-NAME "$env_name" \
      --ENV-IMPL jax \
      --EPISODE-LENGTH 1000 \
      --ACTION-REPEAT "$action_repeat" \
      --MATMUL-PRECISION highest \
      --COMPUTE-DTYPE float32 \
      --TOTAL-TIMESTEPS 10000000 \
      --NUM-ENVS 128 \
      --BATCH-SIZE 512 \
      --GRAD-UPDATES-PER-STEP 8 \
      --MIN-REPLAY-SIZE 8192 \
      --MAX-REPLAY-SIZE 4194304 \
      --WARMUP-POLICY policy \
      --LEARNING-RATE 0.001 \
      --ALPHA-LEARNING-RATE 0.0003 \
      --GAMMA 0.99 \
      --REWARD-SCALE 1.0 \
      --TAU 0.005 \
      --INITIAL-TEMPERATURE 1.0 \
      --TARGET-ENTROPY-SCALE 0.5 \
      --Q-LAYER-NORM \
      --LOGGING-FREQUENCY 100000 \
      --EVAL-FREQUENCY 1111112 \
      --NUM-EVAL-ENVS 128 \
      --WANDB-ENTITY evangelos-ch \
      --WANDB-PROJECT mixed-precision-rl \
      --WANDB-MODE online \
      --WANDB-RUN-NAME "sac_${env_name}_jax_float32_${seed}_backend_check"
  done
done

suite_elapsed_seconds=$((SECONDS - suite_start_seconds))
printf '===== DMC SAC JAX backend check completed in %02d:%02d:%02d =====\n' \
  "$((suite_elapsed_seconds / 3600))" \
  "$(((suite_elapsed_seconds % 3600) / 60))" \
  "$((suite_elapsed_seconds % 60))"
