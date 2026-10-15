#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_actor.sh [options] [extra train_rlpd_sim.py flags]

Common options:
  --checkpoint_path PATH   Checkpoint directory. Default: six_run
  --checkpoint-path PATH   Alias for --checkpoint_path.
  --exp_name NAME          Experiment name. Default: pick_cube_sim
  --exp-name NAME          Alias for --exp_name.
  --ip HOST                Learner host/IP. Default: localhost
  --conda-env NAME         Conda env to run in. Default: hilserl_reward_yt
  --env NAME               Alias for --conda-env.
  --classifier_ckpt PATH   Reward classifier checkpoint dir.
                          Default: <repo>/classifier_ckpt
  --classifier-ckpt PATH   Alias for --classifier_ckpt.
  --no-conda               Use the current python instead of conda run.
  --conda-sh PATH          Conda profile script. Default: ~/miniconda3/etc/profile.d/conda.sh
  --mem-fraction FLOAT     XLA memory fraction. Default: .1
  --controller TYPE        Intervention device: xbox, ps5, or keyboard.
                          Default: experiment configuration (xbox).
  --no-viewer              Run headless with EGL rendering.
  --viewer                 Launch the native MuJoCo GLFW viewer.
  --opencv-viewer          Show front/wrist camera images in an OpenCV window.
  --image-viewer           Alias for --opencv-viewer.
                          Default: OpenCV viewer when DISPLAY is available,
                          otherwise headless.
  --intervention-latch     Hold after joystick/keyboard input until success.
  --no-intervention-latch  Disable the hold behavior. Default.
  --intervention-toggle    Start (gamepad) / Space (keyboard) toggles autonomous
                          and intervention-demo modes.
  --no-intervention-toggle Disable explicit mode switching. Default.
  --debug                  Disable wandb logging.
  -h, --help               Show this help.

Examples:
  bash run_actor.sh --checkpoint_path pick_cube_rlpd_30demos_2026_07_14 --debug
  bash run_actor.sh --checkpoint_path pick_cube_rlpd_30demos_2026_07_14 --opencv-viewer
  bash run_actor.sh --checkpoint_path pick_cube_rlpd_eval --eval_checkpoint_step 5000 --eval_n_trajs 10
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${REPO_DIR}/examples/train_rlpd_sim.py"

EXP_NAME="pick_cube_sim"
CHECKPOINT_PATH="six_run"
IP="localhost"
CONDA_ENV="${CONDA_ENV:-hilserl_reward_yt}"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
MEM_FRACTION=".1"
DEBUG=0
CLASSIFIER_CKPT="${REPO_DIR}/classifier_ckpt"
VIEWER_MODE="auto"
INTERVENTION_LATCH=0
INTERVENTION_TOGGLE=0
CONTROLLER=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --ip)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      IP="$2"
      shift 2
      ;;
    --ip=*)
      IP="${1#*=}"
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
    --controller)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      CONTROLLER="$2"
      shift 2
      ;;
    --controller=*)
      CONTROLLER="${1#*=}"
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --no-viewer)
      VIEWER_MODE="headless"
      shift
      ;;
    --viewer)
      VIEWER_MODE="mujoco"
      shift
      ;;
    --opencv-viewer|--image-viewer)
      VIEWER_MODE="opencv"
      shift
      ;;
    --intervention-latch|--intervention_latch)
      INTERVENTION_LATCH=1
      shift
      ;;
    --no-intervention-latch|--no-intervention_latch)
      INTERVENTION_LATCH=0
      shift
      ;;
    --intervention-toggle|--intervention_toggle)
      INTERVENTION_TOGGLE=1
      shift
      ;;
    --no-intervention-toggle|--no-intervention_toggle)
      INTERVENTION_TOGGLE=0
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

if [[ "$VIEWER_MODE" == "auto" ]]; then
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    VIEWER_MODE="headless"
  else
    VIEWER_MODE="opencv"
  fi
fi

case "$VIEWER_MODE" in
  headless)
    HIL_SERL_NO_VIEWER=1
    HIL_SERL_IMAGE_VIEWER=0
    export MUJOCO_GL="egl"
    ;;
  opencv)
    HIL_SERL_NO_VIEWER=0
    HIL_SERL_IMAGE_VIEWER=1
    export MUJOCO_GL="egl"
    ;;
  mujoco)
    HIL_SERL_NO_VIEWER=0
    HIL_SERL_IMAGE_VIEWER=0
    export MUJOCO_GL="glfw"
    ;;
  *)
    echo "Unknown viewer mode: $VIEWER_MODE" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${REPO_DIR}/examples:${REPO_DIR}/serl_launcher:${REPO_DIR}/franka_sim:${REPO_DIR}/serl_robot_infra:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"

ARGS=(
  "--exp_name=${EXP_NAME}"
  "--checkpoint_path=${CHECKPOINT_PATH}"
  "--ip=${IP}"
  "--actor"
)

if [[ "$DEBUG" -eq 1 ]]; then
  ARGS+=("--debug")
fi

if [[ "$INTERVENTION_LATCH" -eq 1 ]]; then
  ARGS+=("--intervention_latch=true")
fi

if [[ "$INTERVENTION_TOGGLE" -eq 1 ]]; then
  ARGS+=("--intervention_toggle=true")
fi

if [[ -n "$CONTROLLER" ]]; then
  case "$CONTROLLER" in
    keyboard|xbox|ps5) ARGS+=("--controller=${CONTROLLER}") ;;
    *)
      echo "Unsupported controller: $CONTROLLER (use keyboard, xbox, or ps5)" >&2
      exit 1
      ;;
  esac
fi

ARGS+=("${EXTRA_ARGS[@]}")

cat <<EOF
Starting pick_cube_sim actor
  repo:            ${REPO_DIR}
  checkpoint_path: ${CHECKPOINT_PATH}
  classifier_ckpt: ${CLASSIFIER_CKPT}
  exp_name:        ${EXP_NAME}
  ip:              ${IP}
  conda_env:       ${CONDA_ENV:-<current python>}
  viewer_mode:     ${VIEWER_MODE}
  mujoco_gl:       ${MUJOCO_GL}
  intervention_latch: ${INTERVENTION_LATCH}
  intervention_toggle: ${INTERVENTION_TOGGLE}
  controller:      ${CONTROLLER:-<experiment default>}
  debug:           ${DEBUG}
EOF

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
    HIL_SERL_NO_VIEWER="$HIL_SERL_NO_VIEWER" \
    HIL_SERL_IMAGE_VIEWER="$HIL_SERL_IMAGE_VIEWER" \
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
  exec env HIL_SERL_NO_VIEWER="$HIL_SERL_NO_VIEWER" HIL_SERL_IMAGE_VIEWER="$HIL_SERL_IMAGE_VIEWER" python -c '
import runpy
import sys
import jax

if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree_util.tree_leaves

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' "$TRAIN_SCRIPT" "${ARGS[@]}"
fi
