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

dataset_name=${OGBENCH_DATASET_NAME:-antmaze-large-navigate-v0}
dataset_dir=${OGBENCH_DATASET_DIR:-"${HOME}/.ogbench/data"}
dataset_url=${OGBENCH_DATASET_URL:-https://rail.eecs.berkeley.edu/datasets/ogbench}
start_run=${START_RUN:-0}
dry_run=${DRY_RUN:-0}
train_steps=${TRAIN_STEPS:-1000000}
total_runs=15

if [[ ! $dataset_name =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid OGBench dataset name: $dataset_name" >&2
  exit 2
fi
if [[ ! $start_run =~ ^[0-9]+$ ]] || ((start_run > total_runs)); then
  echo "START_RUN must be an integer from 0 to $total_runs" >&2
  exit 2
fi
if [[ $dry_run != 0 && $dry_run != 1 ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ ! $train_steps =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_STEPS must be a positive integer" >&2
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

if [[ $dry_run == 0 ]]; then
  if ! command -v curl >/dev/null; then
    echo "curl is required to download OGBench datasets" >&2
    exit 1
  fi
  mkdir -p "$dataset_dir"
  for dataset_suffix in "" -val; do
    dataset_file="${dataset_name}${dataset_suffix}.npz"
    dataset_path="${dataset_dir}/${dataset_file}"
    if [[ -f $dataset_path ]]; then
      continue
    fi

    temporary_path=$(mktemp "${dataset_path}.tmp.XXXXXX")
    echo "Downloading ${dataset_url}/${dataset_file}"
    if ! curl --fail --location --show-error --output "$temporary_path" \
      "${dataset_url}/${dataset_file}"; then
      rm -f -- "$temporary_path"
      exit 1
    fi
    mv -- "$temporary_path" "$dataset_path"
  done
fi

precisions=(fp32 tf32 bf16)
seeds=(0 1 2 3 4)
sweep_start_seconds=$SECONDS
run_index=0

echo "===== OGBench HIQL precision sweep: ${total_runs} runs ====="
echo "Dataset: $dataset_name"
echo "Dataset directory: $dataset_dir"
echo "GPU index: $CUDA_VISIBLE_DEVICES"
echo "Training steps per run: $train_steps"
echo "Starting from run index: $start_run/$total_runs"

for precision in "${precisions[@]}"; do
  case "$precision" in
  fp32)
    precision_name=float32
    matmul_precision=highest
    compute_dtype=float32
    ;;
  tf32)
    precision_name=tf32
    matmul_precision=default
    compute_dtype=float32
    ;;
  bf16)
    precision_name=bfloat16
    matmul_precision=default
    compute_dtype=bfloat16
    ;;
  *)
    echo "Unexpected precision: $precision" >&2
    exit 2
    ;;
  esac

  for seed in "${seeds[@]}"; do
    if ((run_index < start_run)); then
      ((run_index += 1))
      continue
    fi

    wandb_run_name="hiql_${dataset_name}_${precision_name}_${seed}_5090"
    echo "===== Run $((run_index + 1))/$total_runs: HIQL ${precision} seed ${seed} ====="
    run_command "${uv_run[@]}" ogbench_hiql.py \
      --SEED "$seed" \
      --DATASET-NAME "$dataset_name" \
      --DATASET-DIR "$dataset_dir" \
      --MATMUL-PRECISION "$matmul_precision" \
      --COMPUTE-DTYPE "$compute_dtype" \
      --TRAIN-STEPS "$train_steps" \
      --WANDB-ENTITY evangelos-ch \
      --WANDB-PROJECT mixed-precision-rl \
      --WANDB-MODE online \
      --WANDB-RUN-NAME "$wandb_run_name"

    ((run_index += 1))
  done
done

sweep_elapsed_seconds=$((SECONDS - sweep_start_seconds))
printf '===== OGBench HIQL precision sweep completed in %02d:%02d:%02d =====\n' \
  "$((sweep_elapsed_seconds / 3600))" \
  "$(((sweep_elapsed_seconds % 3600) / 60))" \
  "$((sweep_elapsed_seconds % 60))"
