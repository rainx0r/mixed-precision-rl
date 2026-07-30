#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

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
  total_timesteps=60000000
  gamma=0.995
  action_repeat=1
  num_epochs=16

  case "$env_name" in
    AcrobotSwingup | AcrobotSwingupSparse | SwimmerSwimmer6 | WalkerRun)
      total_timesteps=100000000
      ;;
    BallInCup | FingerSpin)
      gamma=0.95
      ;;
    PendulumSwingup)
      action_repeat=4
      num_epochs=4
      ;;
  esac

  # The paper's ten evaluations include the one before training begins.
  eval_frequency=$(((total_timesteps + 8) / 9))

  for seed in 0 1 2 3 4; do
    echo "===== ${env_name} (${env_number}/${num_env_names}) Seed ${seed} ====="
    uv run dmc_ppo.py \
      --SEED "$seed" \
      --ENV-NAME "$env_name" \
      --ENV-IMPL warp \
      --EPISODE-LENGTH 1000 \
      --ACTION-REPEAT "$action_repeat" \
      --TOTAL-TIMESTEPS "$total_timesteps" \
      --NUM-ENVS 2048 \
      --ROLLOUT-LENGTH 480 \
      --NUM-MINIBATCHES 32 \
      --NUM-EPOCHS "$num_epochs" \
      --LEARNING-RATE 0.001 \
      --ENTROPY-COEFF 0.01 \
      --GAMMA "$gamma" \
      --REWARD-SCALE 10.0 \
      --EVAL-FREQUENCY "$eval_frequency" \
      --WANDB-ENTITY evangelos-ch \
      --WANDB-PROJECT mixed-precision-rl \
      --WANDB-MODE online
  done
done
