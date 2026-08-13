#!/usr/bin/env bash
#SBATCH --job-name=craftax-ppo
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:30:00
#SBATCH --array=0-4%8
#SBATCH --output=logs/craftax-ppo-%A-%a.out
#SBATCH --open-mode=append

set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?}
seed=${SLURM_ARRAY_TASK_ID:?}
cd "$repo_root"

mkdir -p logs
echo "===== Craftax PPO Seed ${seed} ====="
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

"$uv_bin" run --frozen --no-group cuda13 --group cuda12 craftax_ppo.py \
  --SEED "$seed" \
  --EVAL-FREQUENCY 100000000 \
  --WANDB-ENTITY evangelos-ch \
  --WANDB-PROJECT mixed-precision-rl \
  --WANDB-MODE online

echo "Finished: $(date --iso-8601=seconds)"
