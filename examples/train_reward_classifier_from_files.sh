#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/train_reward_classifier_from_files.sh \
    --success classifier_data/pick_cube_sim_30_success_images_2026-07-14_14-55-13.pkl \
    --failure classifier_data/pick_cube_sim_failure_images_2026-07-14_14-55-13.pkl

Options:
  --success PATH          Success pkl file. Can be passed multiple times.
  --failure PATH          Failure pkl file. Can be passed multiple times.
  --success-glob GLOB     Add success pkl files matched by a glob.
  --failure-glob GLOB     Add failure pkl files matched by a glob.
  --exp-name NAME         Experiment name in examples/experiments/mappings.py.
                          Default: pick_cube_sim
  --epochs N              Number of training epochs. Default: 150
  --batch-size N          Batch size. Default: 32
  --eval-split FLOAT      Fraction of data used for eval. Default: 0.15
  --env NAME              Conda env to run in. Default: hilserl_reward_yt
  --repo-dir PATH         Repo root. Default: inferred from this script.
  --output-root PATH      Parent directory for run outputs.
                          Default: <repo>/classifier_runs
  --run-dir PATH          Exact run directory. Overrides --output-root.
  --conda-sh PATH         Conda profile script.
                          Default: ~/miniconda3/etc/profile.d/conda.sh
  --debug                 Pass --debug to Python script, disabling wandb.
                          Default: enabled
  --wandb | --no-debug    Do not pass --debug, allowing wandb logging.
  -h, --help              Show this help.

Output:
  The trained checkpoint is saved under:
    <run-dir>/classifier_ckpt/
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXP_NAME="pick_cube_sim"
NUM_EPOCHS="150"
BATCH_SIZE="32"
EVAL_SPLIT="0.15"
CONDA_ENV="hilserl_reward_yt"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
OUTPUT_ROOT=""
RUN_DIR=""
DEBUG=1

SUCCESS_FILES=()
FAILURE_FILES=()

add_glob_matches() {
  local kind="$1"
  local pattern="$2"
  local matches=()

  while IFS= read -r match; do
    matches+=("$match")
  done < <(compgen -G "$pattern" || true)

  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "No files matched ${kind} glob: ${pattern}" >&2
    exit 1
  fi

  local match
  for match in "${matches[@]}"; do
    if [[ "$kind" == "success" ]]; then
      SUCCESS_FILES+=("$match")
    else
      FAILURE_FILES+=("$match")
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --success|--success-pkl)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      SUCCESS_FILES+=("$2")
      shift 2
      ;;
    --failure|--failure-pkl)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      FAILURE_FILES+=("$2")
      shift 2
      ;;
    --success-glob)
      [[ $# -ge 2 ]] || { echo "$1 requires a glob" >&2; exit 1; }
      add_glob_matches "success" "$2"
      shift 2
      ;;
    --failure-glob)
      [[ $# -ge 2 ]] || { echo "$1 requires a glob" >&2; exit 1; }
      add_glob_matches "failure" "$2"
      shift 2
      ;;
    --exp-name)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      EXP_NAME="$2"
      shift 2
      ;;
    --epochs|--num-epochs)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      BATCH_SIZE="$2"
      shift 2
      ;;
    --eval-split)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      EVAL_SPLIT="$2"
      shift 2
      ;;
    --env|--conda-env)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      CONDA_ENV="$2"
      shift 2
      ;;
    --repo-dir)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      REPO_DIR="$(cd "$2" && pwd)"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-dir)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      RUN_DIR="$2"
      shift 2
      ;;
    --conda-sh)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      CONDA_SH="$2"
      shift 2
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --wandb|--no-debug)
      DEBUG=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#SUCCESS_FILES[@]} -eq 0 ]]; then
  echo "At least one --success file is required." >&2
  usage >&2
  exit 1
fi

if [[ ${#FAILURE_FILES[@]} -eq 0 ]]; then
  echo "At least one --failure file is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="${REPO_DIR}/classifier_runs"
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="${OUTPUT_ROOT}/${EXP_NAME}_reward_classifier_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${RUN_DIR}/classifier_data"

link_file() {
  local src="$1"
  local prefix="$2"
  local idx="$3"

  if [[ ! -f "$src" ]]; then
    echo "Input file not found: $src" >&2
    exit 1
  fi

  local abs_src
  abs_src="$(realpath "$src")"
  local base
  base="$(basename "$abs_src")"
  ln -s "$abs_src" "${RUN_DIR}/classifier_data/${prefix}_${idx}_${base}"
}

idx=0
for file in "${SUCCESS_FILES[@]}"; do
  link_file "$file" "success" "$idx"
  idx=$((idx + 1))
done

idx=0
for file in "${FAILURE_FILES[@]}"; do
  link_file "$file" "failure" "$idx"
  idx=$((idx + 1))
done

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
elif ! command -v conda >/dev/null 2>&1; then
  echo "Could not find conda. Pass --conda-sh /path/to/conda.sh." >&2
  exit 1
fi

TRAIN_SCRIPT="${REPO_DIR}/examples/train_reward_classifier.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Could not find training script: $TRAIN_SCRIPT" >&2
  exit 1
fi

export PYTHONPATH="${REPO_DIR}/examples:${REPO_DIR}/serl_launcher:${REPO_DIR}/franka_sim:${REPO_DIR}/serl_robot_infra:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TRAIN_ARGS=(
  "--exp_name=${EXP_NAME}"
  "--num_epochs=${NUM_EPOCHS}"
  "--batch_size=${BATCH_SIZE}"
  "--eval_split=${EVAL_SPLIT}"
)

if [[ "$DEBUG" -eq 1 ]]; then
  TRAIN_ARGS+=("--debug")
fi

cat <<EOF
Reward classifier training run
  repo:       ${REPO_DIR}
  run dir:    ${RUN_DIR}
  env:        ${CONDA_ENV}
  exp:        ${EXP_NAME}
  epochs:     ${NUM_EPOCHS}
  batch size: ${BATCH_SIZE}
  eval split: ${EVAL_SPLIT}
  debug:      ${DEBUG}

Linked classifier_data:
EOF
find "${RUN_DIR}/classifier_data" -maxdepth 1 -type l -printf '  %f -> %l\n' | sort

cd "$RUN_DIR"

conda run --no-capture-output -n "$CONDA_ENV" python -c '
import runpy
import sys
import jax

if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree_util.tree_leaves

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"

cat <<EOF

Done.
Checkpoint directory:
  ${RUN_DIR}/classifier_ckpt
EOF
