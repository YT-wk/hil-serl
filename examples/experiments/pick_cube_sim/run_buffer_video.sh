#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_buffer_video.sh [options]

Convert a selected segment of one HIL-SERL buffer pkl into an MP4.

Options:
  --file PATH             Specific transitions_*.pkl file to read.
  --buffer-dir PATH       Directory containing transitions_*.pkl files.
                          Default: /ssd/yt_ws/0714/pick_cube_rlpd_30demos_2026_07_14/buffer
  --step STEP             Step to read from --buffer-dir. Default: latest.
  --latest                Alias for --step latest.
  --output PATH           Output .mp4 path. Default: /tmp/<file>_idx<range>_<camera>.mp4
  --start N               Start transition index. Default: 0.
  --end N                 End transition index, exclusive.
  --count N               Max frames to write. Default: 120.
  --stride N              Transition stride. Default: 1.
  --fps N                 Output FPS. Default: 20.
  --camera NAME           front, wrist, both, or comma-separated keys. Default: both.
  --obs-key KEY           observations or next_observations. Default: observations.
  --layout NAME           horizontal or vertical. Default: horizontal.
  --scale FLOAT           Resize output frames. Default: 1.0.
  --no-overlay            Do not draw index/reward/done text.
  --conda-env NAME        Conda env. Default: hilserl_reward_yt.
  --dry-run               Print the command without running it.
  -h, --help              Show this help.

Examples:
  bash run_buffer_video.sh --step latest --start 0 --count 120 --stride 2
  bash run_buffer_video.sh --step 998000 --start 600 --count 80 --camera front
  bash run_buffer_video.sh --file /path/to/transitions_998000.pkl --start 100 --end 180
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_SCRIPT="${REPO_DIR}/examples/buffer_to_video.py"

BUFFER_DIR="${BUFFER_DIR:-/ssd/yt_ws/0714/pick_cube_rlpd_30demos_2026_07_14/buffer}"
CONDA_ENV="${CONDA_ENV:-hilserl_reward_yt}"
STEP="latest"
DRY_RUN=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --buffer-dir)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 1; }
      BUFFER_DIR="$2"
      shift 2
      ;;
    --buffer-dir=*)
      BUFFER_DIR="${1#*=}"
      shift
      ;;
    --step)
      [[ $# -ge 2 ]] || { echo "$1 requires a step" >&2; exit 1; }
      STEP="$2"
      shift 2
      ;;
    --step=*)
      STEP="${1#*=}"
      shift
      ;;
    --latest)
      STEP="latest"
      shift
      ;;
    --file|--output|--start|--end|--count|--stride|--fps|--camera|--obs-key|--layout|--scale)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --file=*|--output=*|--start=*|--end=*|--count=*|--stride=*|--fps=*|--camera=*|--obs-key=*|--layout=*|--scale=*)
      ARGS+=("$1")
      shift
      ;;
    --no-overlay)
      ARGS+=("$1")
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Could not find converter: $PY_SCRIPT" >&2
  exit 1
fi

has_file_arg=0
for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--file" || "$arg" == --file=* ]]; then
    has_file_arg=1
    break
  fi
done

CMD=(python "$PY_SCRIPT")
if [[ "$has_file_arg" -eq 0 ]]; then
  CMD+=(--buffer-dir "$BUFFER_DIR" --step "$STEP")
fi
CMD+=("${ARGS[@]}")

cat <<EOF
Resolved buffer video conversion
  repo:       ${REPO_DIR}
  conda_env:  ${CONDA_ENV}
  buffer_dir: ${BUFFER_DIR}
  step:       ${STEP}
EOF

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

conda run --no-capture-output -n "$CONDA_ENV" "${CMD[@]}"
