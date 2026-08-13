#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

env_file=${ENV_FILE:-.env}
if [[ ! -f $env_file ]]; then
  echo "Environment file not found: $env_file" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

dmc_start_run=${DMC_START_RUN:-0}
craftax_seed=${CRAFTAX_SEED:-0}
dry_run=${DRY_RUN:-0}
total_dmc_runs=125

if [[ ! $dmc_start_run =~ ^[0-9]+$ ]] || ((dmc_start_run > total_dmc_runs)); then
  echo "DMC_START_RUN must be an integer from 0 to $total_dmc_runs" >&2
  exit 2
fi
if [[ ! $craftax_seed =~ ^[0-9]+$ ]]; then
  echo "CRAFTAX_SEED must be a non-negative integer" >&2
  exit 2
fi
if [[ $dry_run != 0 && $dry_run != 1 ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi

uv_run=(uv run --frozen --env-file "$env_file")

run_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ $dry_run == 0 ]]; then
    "$@"
  fi
}

env_names=(
  AcrobotSwingup
  AcrobotSwingupSparse
  BallInCup
  CartpoleBalance
  CartpoleBalanceSparse
  CartpoleSwingup
  CartpoleSwingupSparse
  CheetahRun
  FingerSpin
  FingerTurnEasy
  FingerTurnHard
  FishSwim
  HopperHop
  HopperStand
  HumanoidRun
  HumanoidStand
  HumanoidWalk
  PendulumSwingup
  PointMass
  ReacherEasy
  ReacherHard
  SwimmerSwimmer6
  WalkerRun
  WalkerStand
  WalkerWalk
)

pipeline_start_seconds=$SECONDS
run_index=0

echo "===== DMC SAC BF16 suite: ${#env_names[@]} tasks x 5 seeds ====="
echo "GPU index: $CUDA_VISIBLE_DEVICES"
echo "Starting from DMC run index: $dmc_start_run/$total_dmc_runs"

for env_index in "${!env_names[@]}"; do
  env_name=${env_names[$env_index]}
  env_number=$((env_index + 1))
  total_timesteps=5000000
  action_repeat=1

  case "$env_name" in
    AcrobotSwingup | AcrobotSwingupSparse | CheetahRun | FingerSpin | FingerTurnEasy | FingerTurnHard | HopperHop | HopperStand | HumanoidWalk | SwimmerSwimmer6 | WalkerRun)
      total_timesteps=10000000
      ;;
    PendulumSwingup)
      total_timesteps=10000000
      action_repeat=4
      ;;
  esac

  # The paper's ten evaluations include the one before training begins.
  eval_frequency=$(((total_timesteps + 8) / 9))

  for seed in 0 1 2 3 4; do
    if ((run_index < dmc_start_run)); then
      ((run_index += 1))
      continue
    fi

    echo "===== DMC run $((run_index + 1))/$total_dmc_runs: ${env_name} (${env_number}/${#env_names[@]}) seed ${seed} ====="
    run_command "${uv_run[@]}" dmc_sac.py \
      --SEED "$seed" \
      --ENV-NAME "$env_name" \
      --ENV-IMPL warp \
      --EPISODE-LENGTH 1000 \
      --ACTION-REPEAT "$action_repeat" \
      --MATMUL-PRECISION highest \
      --COMPUTE-DTYPE bfloat16 \
      --TOTAL-TIMESTEPS "$total_timesteps" \
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
      --EVAL-FREQUENCY "$eval_frequency" \
      --NUM-EVAL-ENVS 128 \
      --WANDB-ENTITY evangelos-ch \
      --WANDB-PROJECT mixed-precision-rl \
      --WANDB-MODE online

    ((run_index += 1))
  done
done

dmc_elapsed_seconds=$((SECONDS - pipeline_start_seconds))
printf '===== DMC SAC BF16 suite completed in %02d:%02d:%02d =====\n' \
  "$((dmc_elapsed_seconds / 3600))" \
  "$(((dmc_elapsed_seconds % 3600) / 60))" \
  "$((dmc_elapsed_seconds % 60))"

echo "===== Craftax PPO-RNN BF16: 100M transitions, seed $craftax_seed ====="
run_command "${uv_run[@]}" craftax_ppo.py \
  --SEED "$craftax_seed" \
  --ENV-NAME Craftax-Symbolic-v1 \
  --EPISODE-LENGTH 100000 \
  --MATMUL-PRECISION default \
  --COMPUTE-DTYPE bfloat16 \
  --ROLLOUT-OBSERVATION-DTYPE bfloat16 \
  --TOTAL-TIMESTEPS 100000000 \
  --EVAL-FREQUENCY 0 \
  --WANDB-ENTITY evangelos-ch \
  --WANDB-PROJECT mixed-precision-rl \
  --WANDB-MODE online \
  --WANDB-RUN-NAME "craftax_ppo_bf16_100m_seed${craftax_seed}_5090"

pipeline_elapsed_seconds=$((SECONDS - pipeline_start_seconds))
printf '===== All local BF16 experiments completed in %02d:%02d:%02d =====\n' \
  "$((pipeline_elapsed_seconds / 3600))" \
  "$(((pipeline_elapsed_seconds % 3600) / 60))" \
  "$((pipeline_elapsed_seconds % 60))"
