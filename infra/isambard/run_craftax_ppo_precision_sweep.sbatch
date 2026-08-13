#!/usr/bin/env bash
#SBATCH --job-name=craftax-ppo-precision
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:30:00
#SBATCH --array=0-14%8
#SBATCH --output=logs/craftax-ppo-precision-%A-%a.out
#SBATCH --open-mode=append

set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?}
task_id=${SLURM_ARRAY_TASK_ID:?}
seed=$((task_id % 5))
precision_index=$((task_id / 5))
wandb_run_name=

case "$precision_index" in
0)
  precision_name=fp32
  matmul_precision=highest
  compute_dtype=float32
  rollout_observation_dtype=float32
  ;;
1)
  precision_name=tf32
  matmul_precision=default
  compute_dtype=float32
  rollout_observation_dtype=float32
  wandb_run_name="ppo_rnn_Craftax-Symbolic-v1_tf32_${seed}"
  ;;
2)
  precision_name=bfloat16
  matmul_precision=default
  compute_dtype=bfloat16
  rollout_observation_dtype=bfloat16
  ;;
*)
  echo "Unexpected precision index: ${precision_index}" >&2
  exit 2
  ;;
esac

cd "$repo_root"
mkdir -p logs
echo "===== Craftax PPO ${precision_name} Seed ${seed} ====="
echo "Slurm array task: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

training_command=(
  "$uv_bin" run --frozen --no-group cuda13 --group cuda12 craftax_ppo.py
  --SEED "$seed"
  --MATMUL-PRECISION "$matmul_precision"
  --COMPUTE-DTYPE "$compute_dtype"
  --ROLLOUT-OBSERVATION-DTYPE "$rollout_observation_dtype"
  --EVAL-FREQUENCY 0
  --WANDB-ENTITY evangelos-ch
  --WANDB-PROJECT mixed-precision-rl
  --WANDB-MODE online
)
if [[ -n $wandb_run_name ]]; then
  training_command+=(--WANDB-RUN-NAME "$wandb_run_name")
fi
"${training_command[@]}"

echo "Finished: $(date --iso-8601=seconds)"
