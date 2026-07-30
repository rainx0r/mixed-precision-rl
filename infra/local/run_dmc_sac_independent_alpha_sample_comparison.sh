#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
comparison_start_seconds=$SECONDS
env_impl=${ENV_IMPL:-warp}

if [[ "$env_impl" != "warp" && "$env_impl" != "jax" ]]; then
  printf 'ENV_IMPL must be "warp" or "jax", got: %s\n' "$env_impl" >&2
  exit 2
fi

# These are the MuJoCo Playground paper SAC hyperparameters. Each task runs
# baseline and independent-alpha-sample SAC for seeds 0-4. Variant order is
# reversed on odd seeds to counterbalance order-dependent machine effects.
env_names=(
  PendulumSwingup
  FingerTurnEasy
)

for env_name in "${env_names[@]}"; do
  action_repeat=1
  if [[ "$env_name" == "PendulumSwingup" ]]; then
    action_repeat=4
  fi

  for seed in 0 1 2 3 4; do
    variants=(baseline independent_alpha_sample)
    if ((seed % 2 == 1)); then
      variants=(independent_alpha_sample baseline)
    fi

    for variant in "${variants[@]}"; do
      case "$variant" in
      baseline)
        entrypoint=dmc_sac.py
        run_name="sac_paper_compare_baseline_${env_name}_${env_impl}_float32_${seed}"
        ;;
      independent_alpha_sample)
        entrypoint=dmc_sac_independent_alpha_sample.py
        run_name="sac_paper_compare_independent_alpha_sample_${env_name}_${env_impl}_float32_${seed}"
        ;;
      esac

      echo "===== ${env_name} | ${variant} | ${env_impl} | seed ${seed} ====="
      uv run "$entrypoint" \
        --SEED "$seed" \
        --ENV-NAME "$env_name" \
        --ENV-IMPL "$env_impl" \
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
        --WANDB-RUN-NAME "$run_name"
    done
  done
done

comparison_elapsed_seconds=$((SECONDS - comparison_start_seconds))
printf '===== SAC comparison completed: 20 runs in %02d:%02d:%02d =====\n' \
  "$((comparison_elapsed_seconds / 3600))" \
  "$(((comparison_elapsed_seconds % 3600) / 60))" \
  "$((comparison_elapsed_seconds % 60))"
