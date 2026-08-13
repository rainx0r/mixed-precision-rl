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

start_run=${START_RUN:-0}
dry_run=${DRY_RUN:-0}
total_runs=30

if [[ ! $start_run =~ ^[0-9]+$ ]] || ((start_run > total_runs)); then
  echo "START_RUN must be an integer from 0 to $total_runs" >&2
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

algorithms=(ppo pqn)
precisions=(fp32 tf32 bf16)
seeds=(0 1 2 3 4)

sweep_start_seconds=$SECONDS
run_index=0

echo "===== Craftax PPO/PQN precision sweep: ${total_runs} runs ====="
echo "GPU index: $CUDA_VISIBLE_DEVICES"
echo "Starting from run index: $start_run/$total_runs"

for algorithm in "${algorithms[@]}"; do
  entrypoint="craftax_${algorithm}.py"

  for precision in "${precisions[@]}"; do
    case "$precision" in
    fp32)
      precision_name=float32
      matmul_precision=highest
      compute_dtype=float32
      rollout_observation_dtype=float32
      ;;
    tf32)
      precision_name=tf32
      matmul_precision=default
      compute_dtype=float32
      rollout_observation_dtype=float32
      ;;
    bf16)
      precision_name=bfloat16
      matmul_precision=default
      compute_dtype=bfloat16
      rollout_observation_dtype=bfloat16
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

      wandb_run_name="${algorithm}_rnn_Craftax-Symbolic-v1_${precision_name}_${seed}_5090"
      echo "===== Run $((run_index + 1))/$total_runs: Craftax ${algorithm^^} ${precision} seed ${seed} ====="
      run_command "${uv_run[@]}" "$entrypoint" \
        --SEED "$seed" \
        --ENV-NAME Craftax-Symbolic-v1 \
        --MATMUL-PRECISION "$matmul_precision" \
        --COMPUTE-DTYPE "$compute_dtype" \
        --ROLLOUT-OBSERVATION-DTYPE "$rollout_observation_dtype" \
        --TOTAL-TIMESTEPS 1000000000 \
        --EVAL-FREQUENCY 0 \
        --WANDB-ENTITY evangelos-ch \
        --WANDB-PROJECT mixed-precision-rl \
        --WANDB-MODE online \
        --WANDB-RUN-NAME "$wandb_run_name"

      ((run_index += 1))
    done
  done
done

sweep_elapsed_seconds=$((SECONDS - sweep_start_seconds))
printf '===== Craftax precision sweep completed in %02d:%02d:%02d =====\n' \
  "$((sweep_elapsed_seconds / 3600))" \
  "$(((sweep_elapsed_seconds % 3600) / 60))" \
  "$((sweep_elapsed_seconds % 60))"
