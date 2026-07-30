#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
suite_start_seconds=$SECONDS

# These are the 25 state-based DM Control Suite tasks reported in the MuJoCo
# Playground paper. Seeds 0-4 give five independent runs per task.
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

num_env_names=${#env_names[@]}
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
    echo "===== ${env_name} (${env_number}/${num_env_names}) Seed ${seed} ====="
    uv run dmc_sac.py \
      --SEED "$seed" \
      --ENV-NAME "$env_name" \
      --ENV-IMPL warp \
      --EPISODE-LENGTH 1000 \
      --ACTION-REPEAT "$action_repeat" \
      --MATMUL-PRECISION highest \
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
  done
done

suite_elapsed_seconds=$((SECONDS - suite_start_seconds))
printf '===== DMC SAC suite completed in %02d:%02d:%02d =====\n' \
  "$((suite_elapsed_seconds / 3600))" \
  "$(((suite_elapsed_seconds % 3600) / 60))" \
  "$((suite_elapsed_seconds % 60))"
