#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: infra/isambard/submit.sh [--dry-run] [--env-file FILE] JOB.sbatch [JOB_ARGUMENT ...]

Create an immutable snapshot of the repository on Isambard and submit the
specified repository-local batch script. Any remaining arguments are passed to
the batch script. Variables from FILE are passed directly to Slurm and are not
copied into the remote snapshot. FILE uses dotenv-style NAME=VALUE lines.

Environment variables:
  ISAMBARD_HOST         Clifton SSH alias. Auto-detected when exactly one
                        <project>.aip2.isambard alias is configured.
  ISAMBARD_REMOTE_BASE  Remote directory that will contain submissions.
                        Defaults to $PROJECTDIR/$USER/mixed-precision-rl.
EOF
}

dry_run=false
env_file_input=
while (($# > 0)); do
  case "$1" in
  --dry-run)
    dry_run=true
    shift
    ;;
  --env-file)
    if (($# < 2)); then
      echo "--env-file requires a path." >&2
      exit 2
    fi
    env_file_input=$2
    shift 2
    ;;
  --env-file=*)
    env_file_input=${1#*=}
    if [[ -z "$env_file_input" ]]; then
      echo "--env-file requires a path." >&2
      exit 2
    fi
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  --)
    shift
    break
    ;;
  -*)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
  *)
    break
    ;;
  esac
done

if (($# == 0)); then
  echo "A batch script is required." >&2
  usage >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
job_input=$1
shift
job_arguments=("$@")

forwarded_env_names=()
forwarded_env_values=()
if [[ -n "$env_file_input" ]]; then
  if ! env_file_path=$(realpath -e -- "$env_file_input") || [[ ! -f "$env_file_path" ]]; then
    echo "Environment file not found: ${env_file_input}" >&2
    exit 1
  fi

  declare -A seen_env_names=()
  line_number=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    line=${line%$'\r'}
    if [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]]; then
      continue
    fi
    if [[ ! "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      echo "Invalid dotenv assignment at ${env_file_input}:${line_number}." >&2
      exit 1
    fi

    name=${BASH_REMATCH[2]}
    value=${BASH_REMATCH[3]}
    value=${value#"${value%%[![:space:]]*}"}
    value=${value%"${value##*[![:space:]]}"}
    if [[ "$value" == \"* ]]; then
      if [[ ${value: -1} != '"' ]]; then
        echo "Unterminated double quote at ${env_file_input}:${line_number}." >&2
        exit 1
      fi
      value=${value:1:${#value}-2}
    elif [[ "$value" == \'* ]]; then
      if [[ ${value: -1} != "'" ]]; then
        echo "Unterminated single quote at ${env_file_input}:${line_number}." >&2
        exit 1
      fi
      value=${value:1:${#value}-2}
    fi

    if [[ -v "seen_env_names[$name]" ]]; then
      echo "Duplicate variable ${name} in ${env_file_input}." >&2
      exit 1
    fi
    seen_env_names[$name]=1
    forwarded_env_names+=("$name")
    forwarded_env_values+=("$value")
  done <"$env_file_path"
fi

if [[ "$job_input" = /* ]]; then
  job_path=$job_input
else
  job_path=${repo_root}/${job_input}
fi
if ! job_path=$(realpath -e -- "$job_path") || [[ ! -f "$job_path" ]]; then
  echo "Batch script not found: ${job_input}" >&2
  exit 1
fi
if [[ "$job_path" != "${repo_root}/"* ]]; then
  echo "Batch script must be inside the repository: ${job_input}" >&2
  exit 1
fi
job_relative_path=${job_path#"${repo_root}/"}

isambard_host=${ISAMBARD_HOST:-}
if [[ -z "$isambard_host" ]]; then
  clifton_config=${HOME}/.ssh/config_clifton
  if [[ -r "$clifton_config" ]]; then
    mapfile -t aip2_hosts < <(
      awk '
        $1 == "Host" {
          for (field = 2; field <= NF; field++) {
            if ($field ~ /^[[:alnum:]_-]+[.]aip2[.]isambard$/) {
              print $field
            }
          }
        }
      ' "$clifton_config"
    )
  else
    aip2_hosts=()
  fi

  if ((${#aip2_hosts[@]} == 1)); then
    isambard_host=${aip2_hosts[0]}
  else
    echo "Could not uniquely detect an Isambard-AI Phase 2 SSH alias." >&2
    echo "Set ISAMBARD_HOST=<project>.aip2.isambard and try again." >&2
    exit 1
  fi
fi

if ! command -v rsync >/dev/null; then
  echo "rsync is required but was not found." >&2
  exit 1
fi

remote_info=$(ssh "$isambard_host" \
  'printf "%s\n%s\n" "${PROJECTDIR:?PROJECTDIR is not set}" "${USER:?USER is not set}"')
mapfile -t remote_values <<<"$remote_info"
if ((${#remote_values[@]} != 2)); then
  echo "Could not determine PROJECTDIR and USER on ${isambard_host}." >&2
  exit 1
fi
remote_project_dir=${remote_values[0]}
remote_user=${remote_values[1]}

remote_base=${ISAMBARD_REMOTE_BASE:-"${remote_project_dir}/${remote_user}/mixed-precision-rl"}
if [[ "$remote_base" != /* ]]; then
  echo "ISAMBARD_REMOTE_BASE must be an absolute remote path." >&2
  exit 1
fi

git_commit=unversioned
dirty_suffix=
if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_commit=$(git -C "$repo_root" rev-parse --short=12 HEAD)
  if [[ -n $(git -C "$repo_root" status --porcelain --untracked-files=normal) ]]; then
    dirty_suffix=-dirty
  fi
fi

submission_id="$(date -u +%Y%m%dT%H%M%SZ)-${git_commit}${dirty_suffix}-$$"
remote_snapshot="${remote_base}/submissions/${submission_id}"
printf -v remote_snapshot_quoted '%q' "$remote_snapshot"
printf -v job_relative_path_quoted '%q' "$job_relative_path"

sbatch_command="sbatch --parsable"
if ((${#forwarded_env_names[@]} > 0)); then
  sbatch_command+=" --export-file=3"
fi
sbatch_command+=" ${job_relative_path_quoted}"
for argument in "${job_arguments[@]}"; do
  printf -v argument_quoted '%q' "$argument"
  sbatch_command+=" ${argument_quoted}"
done

if ((${#forwarded_env_names[@]} > 0)); then
  remote_env_command=env
  for name in "${forwarded_env_names[@]}"; do
    remote_env_command+=" -u ${name}"
  done
  remote_command="cd ${remote_snapshot_quoted} && { ${remote_env_command} -0; cat; } | ${sbatch_command} 3<&0"
else
  remote_command="cd ${remote_snapshot_quoted} && ${sbatch_command}"
fi

echo "Host:       ${isambard_host}"
echo "Snapshot:   ${remote_snapshot}"
echo "Batch file: ${job_relative_path}"
if ((${#forwarded_env_names[@]} > 0)); then
  printf "Environment:"
  printf " %s" "${forwarded_env_names[@]}"
  echo
fi
if ((${#job_arguments[@]} > 0)); then
  printf "Arguments:  "
  printf "%q " "${job_arguments[@]}"
  echo
fi

if "$dry_run"; then
  echo
  echo "Dry run: no files will be uploaded and no jobs will be submitted."
  echo "Would sync ${repo_root}/ to ${isambard_host}:${remote_snapshot}/"
  echo "Would run: ${remote_command}"
  exit 0
fi

ssh "$isambard_host" "mkdir -p -- ${remote_snapshot_quoted}/logs"

rsync \
  --archive \
  --compress \
  --human-readable \
  --itemize-changes \
  --exclude=/.git/ \
  --exclude=/.venv/ \
  --exclude=/.env \
  --exclude='/.env.*' \
  --exclude=/__pycache__/ \
  --exclude=/logs/ \
  --exclude=/wandb/ \
  --exclude=/.agents/ \
  --exclude=/.codex/ \
  --exclude=/.worktrees/ \
  "${repo_root}/" \
  "${isambard_host}:${remote_snapshot}/"

if ((${#forwarded_env_names[@]} > 0)); then
  submission=$(
    for ((env_index = 0; env_index < ${#forwarded_env_names[@]}; env_index++)); do
      printf '%s=%s\0' "${forwarded_env_names[$env_index]}" "${forwarded_env_values[$env_index]}"
    done | ssh "$isambard_host" "$remote_command"
  )
else
  submission=$(ssh "$isambard_host" "$remote_command")
fi
job_id=${submission%%;*}
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected job ID: ${submission}" >&2
  exit 1
fi

echo
echo "Job:      ${job_id}"
echo "Status:   ssh ${isambard_host} 'squeue --jobs=${job_id}'"
echo "Cancel:   ssh ${isambard_host} 'scancel ${job_id}'"
echo "Snapshot: ${isambard_host}:${remote_snapshot}/"
