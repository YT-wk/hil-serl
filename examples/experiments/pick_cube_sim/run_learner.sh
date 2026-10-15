#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_learner.sh --demo_path PATH [options] [extra train_rlpd_sim.py flags]

Common options:
  --demo_path PATH         Demo pkl path. Can be passed multiple times.
  --demo-path PATH         Alias for --demo_path.
  --checkpoint_path PATH   Checkpoint directory. Default: six_run
  --checkpoint-path PATH   Alias for --checkpoint_path.
  --exp_name NAME          Experiment name. Default: pick_cube_sim
  --exp-name NAME          Alias for --exp_name.
  --conda-env NAME         Conda env to run in. Default: hilserl_reward_yt
  --env NAME               Alias for --conda-env.
  --classifier_ckpt PATH   Reward classifier checkpoint dir.
                          Default: <repo>/classifier_ckpt
  --classifier-ckpt PATH   Alias for --classifier_ckpt.
  --no-conda               Use the current python instead of conda run.
  --conda-sh PATH          Conda profile script. Default: ~/miniconda3/etc/profile.d/conda.sh
  --mem-fraction FLOAT     XLA memory fraction. Default: .3
  --debug                  Disable wandb logging.
  -h, --help               Show this help.

Examples:
  bash run_learner.sh \
    --checkpoint_path pick_cube_rlpd_30demos_2026_07_14 \
    --demo_path /home/ubuntu/yt_ws/Codes/hil-serl/demo_data/pick_cube_sim_30_demos_2026-07-14_16-45-18.pkl \
    --debug
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${REPO_DIR}/examples/train_rlpd_sim.py"

EXP_NAME="pick_cube_sim"
CHECKPOINT_PATH="six_run"
CONDA_ENV="${CONDA_ENV:-hilserl_reward_yt}"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
MEM_FRACTION=".3"
DEBUG=0
CLASSIFIER_CKPT="${REPO_DIR}/classifier_ckpt"
DEMO_PATHS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --demo_path|--demo-path)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      DEMO_PATHS+=("$2")
      shift 2
      ;;
    --demo_path=*|--demo-path=*)
      DEMO_PATHS+=("${1#*=}")
      shift
      ;;
    --checkpoint_path|--checkpoint-path)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --checkpoint_path=*|--checkpoint-path=*)
      CHECKPOINT_PATH="${1#*=}"
      shift
      ;;
    --exp_name|--exp-name)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      EXP_NAME="$2"
      shift 2
      ;;
    --exp_name=*|--exp-name=*)
      EXP_NAME="${1#*=}"
      shift
      ;;
    --conda-env|--env)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      CONDA_ENV="$2"
      shift 2
      ;;
    --conda-env=*|--env=*)
      CONDA_ENV="${1#*=}"
      shift
      ;;
    --no-conda)
      CONDA_ENV=""
      shift
      ;;
    --classifier_ckpt|--classifier-ckpt)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      CLASSIFIER_CKPT="$2"
      shift 2
      ;;
    --classifier_ckpt=*|--classifier-ckpt=*)
      CLASSIFIER_CKPT="${1#*=}"
      shift
      ;;
    --conda-sh)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      CONDA_SH="$2"
      shift 2
      ;;
    --conda-sh=*)
      CONDA_SH="${1#*=}"
      shift
      ;;
    --mem-fraction)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      MEM_FRACTION="$2"
      shift 2
      ;;
    --mem-fraction=*)
      MEM_FRACTION="${1#*=}"
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --no-debug|--wandb)
      DEBUG=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#DEMO_PATHS[@]} -eq 0 ]]; then
  echo "At least one --demo_path is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Could not find train script: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [[ ! -e "$CLASSIFIER_CKPT" ]]; then
  echo "Reward classifier checkpoint not found: $CLASSIFIER_CKPT" >&2
  echo "Pass --classifier_ckpt /path/to/classifier_ckpt." >&2
  exit 1
fi

CLASSIFIER_CKPT="$(realpath "$CLASSIFIER_CKPT")"
if [[ "$CLASSIFIER_CKPT" != "${REPO_DIR}/classifier_ckpt" ]]; then
  ln -sfn "$CLASSIFIER_CKPT" "${REPO_DIR}/classifier_ckpt"
fi

export PYTHONPATH="${REPO_DIR}/examples:${REPO_DIR}/serl_launcher:${REPO_DIR}/franka_sim:${REPO_DIR}/serl_robot_infra:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

ARGS=(
  "--exp_name=${EXP_NAME}"
  "--checkpoint_path=${CHECKPOINT_PATH}"
  "--learner"
)

for demo_path in "${DEMO_PATHS[@]}"; do
  if [[ ! -f "$demo_path" ]]; then
    echo "Demo file not found: $demo_path" >&2
    exit 1
  fi
  ARGS+=("--demo_path=$(realpath "$demo_path")")
done

if [[ "$DEBUG" -eq 1 ]]; then
  ARGS+=("--debug")
fi

ARGS+=("${EXTRA_ARGS[@]}")

cat <<EOF
Starting pick_cube_sim learner
  repo:            ${REPO_DIR}
  checkpoint_path: ${CHECKPOINT_PATH}
  classifier_ckpt: ${CLASSIFIER_CKPT}
  exp_name:        ${EXP_NAME}
  conda_env:       ${CONDA_ENV:-<current python>}
  debug:           ${DEBUG}
  demo_paths:
EOF
for demo_path in "${DEMO_PATHS[@]}"; do
  echo "    - $(realpath "$demo_path")"
done

cd "$REPO_DIR"

if [[ -n "$CONDA_ENV" ]]; then
  if [[ -f "$CONDA_SH" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_SH"
  fi
  ENV_PREFIX="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.prefix)')"
  NVIDIA_LIBS="$(find "${ENV_PREFIX}/lib/python3.10/site-packages/nvidia" -path '*/lib' -type d 2>/dev/null | paste -sd: - || true)"
  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${NVIDIA_LIBS:+:${NVIDIA_LIBS}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  exec env \
    PATH="${ENV_PREFIX}/bin:${PATH}" \
    CONDA_PREFIX="$ENV_PREFIX" \
    CONDA_DEFAULT_ENV="$CONDA_ENV" \
    "${ENV_PREFIX}/bin/python" -c '
import runpy
import sys
import jax

if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree_util.tree_leaves

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' "$TRAIN_SCRIPT" "${ARGS[@]}"
else
  exec python -c '
import runpy
import sys
import jax

if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree_util.tree_leaves

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' "$TRAIN_SCRIPT" "${ARGS[@]}"
fi
