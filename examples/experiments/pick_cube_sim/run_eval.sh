#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_eval.sh [options] [extra train_rlpd_sim.py flags]

Defaults are set for the current pick_cube_sim run on the lab server.

Options:
  --checkpoint_path PATH       Checkpoint directory.
  --checkpoint-path PATH       Alias for --checkpoint_path.
  --step STEP                  Checkpoint step to evaluate. Default: latest.
  --eval_checkpoint_step STEP  Alias for --step.
  --latest                    Force using the latest checkpoint_* directory.
  --eval_n_trajs N             Number of eval episodes. Default: 1.
  --eval-n-trajs N             Alias for --eval_n_trajs.
  --conda-env NAME             Conda env. Default: hilserl_reward_yt.
  --env NAME                   Alias for --conda-env.
  --classifier_ckpt PATH       Reward classifier checkpoint dir.
  --classifier-ckpt PATH       Alias for --classifier_ckpt.
  --mem-fraction FLOAT         XLA memory fraction. Default: .1.
  --viewer                     Launch MuJoCo viewer.
  --no-viewer                  Disable MuJoCo viewer. Default.
  --num-viewers N              MuJoCo viewer windows when --viewer is used. Default: 1.
  --num_viewers N              Alias for --num-viewers.
  --deterministic              Use argmax actions. Default.
  --stochastic                 Sample stochastic actions.
  --reset-after-eval           Reset env to home after eval. Default.
  --no-reset-after-eval        Leave env at the final eval pose.
  --reset-hold-steps N         Viewer sync frames after final reset. Default: 20.
  --debug                      Disable wandb logging. Default.
  --no-debug                   Keep wandb logging enabled.
  --timeout-sec SECONDS        Kill eval after this many seconds. 0 disables.
  --skip-viewer-check          Skip DISPLAY/GLFW preflight checks.
  --dry-run                    Print the resolved command without running it.
  -h, --help                   Show this help.

Examples:
  bash run_eval.sh
  bash run_eval.sh --eval_n_trajs 10 --viewer
  bash run_eval.sh --step 880000 --timeout-sec 180
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/ssd/yt_ws/0714/pick_cube_rlpd_30demos_2026_07_14}"
EVAL_STEP="latest"
EVAL_N_TRAJS=1
CONDA_ENV="${CONDA_ENV:-hilserl_reward_yt}"
MEM_FRACTION=".1"
NO_VIEWER=1
NUM_VIEWERS=1
DETERMINISTIC=1
RESET_AFTER_EVAL=1
RESET_HOLD_STEPS=20
DEBUG=1
TIMEOUT_SEC=0
SKIP_VIEWER_CHECK=0
DRY_RUN=0
CLASSIFIER_CKPT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_path|--checkpoint-path)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --checkpoint_path=*|--checkpoint-path=*)
      CHECKPOINT_PATH="${1#*=}"
      shift
      ;;
    --step|--eval_checkpoint_step)
      [[ $# -ge 2 ]] || { echo "$1 requires a step" >&2; exit 1; }
      EVAL_STEP="$2"
      shift 2
      ;;
    --step=*|--eval_checkpoint_step=*)
      EVAL_STEP="${1#*=}"
      shift
      ;;
    --latest)
      EVAL_STEP="latest"
      shift
      ;;
    --eval_n_trajs|--eval-n-trajs)
      [[ $# -ge 2 ]] || { echo "$1 requires a count" >&2; exit 1; }
      EVAL_N_TRAJS="$2"
      shift 2
      ;;
    --eval_n_trajs=*|--eval-n-trajs=*)
      EVAL_N_TRAJS="${1#*=}"
      shift
      ;;
    --conda-env|--env)
      [[ $# -ge 2 ]] || { echo "$1 requires an env name" >&2; exit 1; }
      CONDA_ENV="$2"
      shift 2
      ;;
    --conda-env=*|--env=*)
      CONDA_ENV="${1#*=}"
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
    --mem-fraction)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      MEM_FRACTION="$2"
      shift 2
      ;;
    --mem-fraction=*)
      MEM_FRACTION="${1#*=}"
      shift
      ;;
    --viewer)
      NO_VIEWER=0
      shift
      ;;
    --no-viewer)
      NO_VIEWER=1
      shift
      ;;
    --num-viewers|--num_viewers)
      [[ $# -ge 2 ]] || { echo "$1 requires a count" >&2; exit 1; }
      NUM_VIEWERS="$2"
      shift 2
      ;;
    --num-viewers=*|--num_viewers=*)
      NUM_VIEWERS="${1#*=}"
      shift
      ;;
    --deterministic)
      DETERMINISTIC=1
      shift
      ;;
    --stochastic)
      DETERMINISTIC=0
      shift
      ;;
    --reset-after-eval)
      RESET_AFTER_EVAL=1
      shift
      ;;
    --no-reset-after-eval)
      RESET_AFTER_EVAL=0
      shift
      ;;
    --reset-hold-steps|--reset_hold_steps)
      [[ $# -ge 2 ]] || { echo "$1 requires a count" >&2; exit 1; }
      RESET_HOLD_STEPS="$2"
      shift 2
      ;;
    --reset-hold-steps=*|--reset_hold_steps=*)
      RESET_HOLD_STEPS="${1#*=}"
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --no-debug)
      DEBUG=0
      shift
      ;;
    --timeout-sec)
      [[ $# -ge 2 ]] || { echo "$1 requires seconds" >&2; exit 1; }
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --timeout-sec=*)
      TIMEOUT_SEC="${1#*=}"
      shift
      ;;
    --skip-viewer-check)
      SKIP_VIEWER_CHECK=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint path not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ "$EVAL_STEP" == "latest" ]]; then
  EVAL_STEP="$(
    find "$CHECKPOINT_PATH" -maxdepth 1 -type d -name 'checkpoint_*' -printf '%f\n' \
      | sed 's/^checkpoint_//' \
      | sort -n \
      | tail -1
  )"
fi

if [[ -z "$EVAL_STEP" ]]; then
  echo "No checkpoint_* directories found in $CHECKPOINT_PATH" >&2
  exit 1
fi

CMD=(
  bash "${SCRIPT_DIR}/run_actor.sh"
  --checkpoint_path "$CHECKPOINT_PATH"
  --eval_checkpoint_step "$EVAL_STEP"
  --eval_n_trajs "$EVAL_N_TRAJS"
  --conda-env "$CONDA_ENV"
  --mem-fraction "$MEM_FRACTION"
  --num_viewers "$NUM_VIEWERS"
  --reset_after_eval="$([[ "$RESET_AFTER_EVAL" -eq 1 ]] && echo true || echo false)"
  --reset_hold_steps "$RESET_HOLD_STEPS"
)

if [[ -n "$CLASSIFIER_CKPT" ]]; then
  CMD+=(--classifier_ckpt "$CLASSIFIER_CKPT")
fi

if [[ "$NO_VIEWER" -eq 1 ]]; then
  CMD+=(--no-viewer)
else
  CMD+=(--viewer)
fi

if [[ "$DEBUG" -eq 1 ]]; then
  CMD+=(--debug)
else
  CMD+=(--no-debug)
fi

if [[ "$DETERMINISTIC" -eq 1 ]]; then
  CMD+=(--eval_deterministic=true)
else
  CMD+=(--eval_deterministic=false)
fi

CMD+=("${EXTRA_ARGS[@]}")

cat <<EOF
Resolved pick_cube_sim eval
  checkpoint_path: ${CHECKPOINT_PATH}
  eval_step:       ${EVAL_STEP}
  eval_n_trajs:    ${EVAL_N_TRAJS}
  conda_env:       ${CONDA_ENV}
  no_viewer:       ${NO_VIEWER}
  num_viewers:     ${NUM_VIEWERS}
  deterministic:   ${DETERMINISTIC}
  reset_after_eval:${RESET_AFTER_EVAL}
  reset_hold_steps:${RESET_HOLD_STEPS}
  timeout_sec:     ${TIMEOUT_SEC}
  display:         ${DISPLAY:-<empty>}
  wayland_display: ${WAYLAND_DISPLAY:-<empty>}
EOF

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

if [[ "$NO_VIEWER" -eq 0 && "$SKIP_VIEWER_CHECK" -eq 0 ]]; then
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    cat >&2 <<'EOF'
Viewer requested, but neither DISPLAY nor WAYLAND_DISPLAY is set.

Run from a graphical session, or connect with X11 forwarding, for example:
  ssh -Y ubuntu@192.168.0.80

On Windows, start an X server first, such as VcXsrv or MobaXterm.
Use --no-viewer for headless evaluation.
EOF
    exit 2
  fi

  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi

  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
  if ! conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
import sys

try:
    import glfw
except Exception as exc:
    print(f"Failed to import glfw: {exc}", file=sys.stderr)
    sys.exit(3)

if not glfw.init():
    print("glfw.init() failed. The DISPLAY exists, but no usable OpenGL/GLFW display was found.", file=sys.stderr)
    sys.exit(4)

glfw.terminate()
print("GLFW preflight ok")
PY
  then
    cat >&2 <<'EOF'
Viewer preflight failed.

This usually means X11 forwarding or the local X server is not working.
First verify a simple GUI app works, for example:
  xclock
or:
  python -c "import glfw; print(glfw.init())"

Use --skip-viewer-check only if you are sure the display is valid.
EOF
    exit 2
  fi
fi

if [[ "$TIMEOUT_SEC" != "0" ]]; then
  printf '\n' | timeout "${TIMEOUT_SEC}s" "${CMD[@]}"
else
  printf '\n' | "${CMD[@]}"
fi
