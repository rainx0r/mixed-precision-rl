#!/usr/bin/env bash
#SBATCH --job-name=dmc-suite
#SBATCH --partition=workq
#SBATCH --dependency=singleton
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/dmc-suite-sac-%j.out
#SBATCH --open-mode=append

set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?}
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

if [[ ${1:-} == run ]]; then
  uv_bin=${UV_BIN:?UV_BIN must be exported by the parent batch process}
  run_index=${2:?}
  num_seeds=5
  env_index=$((run_index / num_seeds))
  seed=$((run_index % num_seeds))
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

  echo "===== ${env_name} (${env_number}/${#env_names[@]}) Seed ${seed} ====="
  echo "Slurm job step: ${SLURM_JOB_ID}.${SLURM_STEP_ID}"
  echo "Host: $(hostname)"
  echo "Started: $(date --iso-8601=seconds)"

  "$uv_bin" run --frozen --no-group cuda13 --group cuda12 dmc_sac.py \
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
    --MATMUL-PRECISION "default" \
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

  echo "Finished: $(date --iso-8601=seconds)"
  exit 0
fi

if (($# > 0)); then
  echo "Unknown command: $1" >&2
  exit 2
fi

mkdir -p logs
echo "Preparing environment in ${repo_root}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
if [[ -n ${WANDB_API_KEY:-} ]]; then
  echo "W&B authentication: WANDB_API_KEY supplied"
else
  echo "W&B authentication: no WANDB_API_KEY supplied" >&2
fi
source "$repo_root/infra/isambard/bootstrap_uv.sh"
uv_bin=$UV_BIN
"$uv_bin" sync --frozen --no-group cuda13 --group cuda12

max_concurrent=8
total_runs=$((${#env_names[@]} * 5))
active_steps=0

run_step() {
  local run_index=$1
  local log_path="logs/dmc-suite-sac-${SLURM_JOB_ID}-${run_index}.out"
  local failure_path="logs/dmc-suite-sac-${SLURM_JOB_ID}-${run_index}.failed"
  local status

  if srun --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=72 --exclusive \
    bash "$repo_root/infra/isambard/run_dmc_suite_sac.sbatch" run "$run_index" \
    >"$log_path" 2>&1; then
    return 0
  else
    status=$?
    printf "Run %d exited with status %d\n" "$run_index" "$status" >"$failure_path"
    return "$status"
  fi
}

echo "Launching ${total_runs} runs across ${max_concurrent} GPUs"
for ((run_index = 0; run_index < total_runs; run_index++)); do
  while ((active_steps >= max_concurrent)); do
    wait -n || true
    active_steps=$((active_steps - 1))
  done

  run_step "$run_index" &
  ((active_steps += 1))
done

while ((active_steps > 0)); do
  wait -n || true
  active_steps=$((active_steps - 1))
done

shopt -s nullglob
failure_files=(logs/dmc-suite-sac-"${SLURM_JOB_ID}"-*.failed)
if ((${#failure_files[@]} > 0)); then
  echo "${#failure_files[@]} training run(s) failed:" >&2
  cat "${failure_files[@]}" >&2
  exit 1
fi

echo "All ${total_runs} runs finished successfully"
echo "Finished: $(date --iso-8601=seconds)"
