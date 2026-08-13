#!/usr/bin/env bash
#SBATCH --job-name=ogbench-hiql-precision
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --array=0-14%8
#SBATCH --output=logs/ogbench-hiql-precision-%A-%a.out
#SBATCH --open-mode=append

set -euo pipefail

if (($# > 1)); then
  echo "Usage: $0 [DATASET_NAME]" >&2
  exit 2
fi

repo_root=${SLURM_SUBMIT_DIR:?}
dataset_name=${1:-antmaze-large-navigate-v0}
if [[ ! $dataset_name =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid OGBench dataset name: ${dataset_name}" >&2
  exit 2
fi

task_id=${SLURM_ARRAY_TASK_ID:?}
num_seeds=5
seed=$((task_id % num_seeds))
precision_index=$((task_id / num_seeds))

case "$precision_index" in
0)
  precision_name=fp32
  matmul_precision=highest
  compute_dtype=float32
  ;;
1)
  precision_name=tf32
  matmul_precision=default
  compute_dtype=float32
  ;;
2)
  precision_name=bfloat16
  matmul_precision=default
  compute_dtype=bfloat16
  ;;
*)
  echo "Unexpected precision index: ${precision_index}" >&2
  exit 2
  ;;
esac

cd "$repo_root"
mkdir -p logs
echo "===== OGBench HIQL ${dataset_name} ${precision_name} Seed ${seed} ====="
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

dataset_dir=${OGBENCH_DATASET_DIR:-"${PROJECTDIR:?PROJECTDIR is not set}/${USER:?USER is not set}/ogbench/data"}
mkdir -p "$dataset_dir"
if ! command -v flock >/dev/null; then
  echo "flock is required to coordinate the shared OGBench dataset cache" >&2
  exit 1
fi

# OGBench downloads through a fixed .tmp filename. Serialize this short setup
# step so concurrent array tasks cannot corrupt a first-time dataset download.
dataset_lock="${dataset_dir}/.${dataset_name}.lock"
flock "$dataset_lock" \
  "$uv_bin" run --frozen --no-group cuda13 --group cuda12 python -c \
  'import sys, ogbench; ogbench.download_datasets([sys.argv[1]], sys.argv[2])' \
  "$dataset_name" "$dataset_dir"

"$uv_bin" run --frozen --no-group cuda13 --group cuda12 ogbench_hiql.py \
  --SEED "$seed" \
  --DATASET-NAME "$dataset_name" \
  --DATASET-DIR "$dataset_dir" \
  --MATMUL-PRECISION "$matmul_precision" \
  --COMPUTE-DTYPE "$compute_dtype" \
  --WANDB-ENTITY evangelos-ch \
  --WANDB-PROJECT mixed-precision-rl \
  --WANDB-MODE online \
  --WANDB-RUN-NAME "hiql_${dataset_name}_${precision_name}_${seed}"

echo "Finished: $(date --iso-8601=seconds)"
