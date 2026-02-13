#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Train MaskablePPO for multiple benchmark scales with a unified configuration.

Usage:
  scripts/train_all_scales.sh [options] [-- <extra args for train_maskable_ppo.py>]

Options:
  --scales "S M L XL"            Space/comma separated scale list (default: "S M L XL")
  --manifest-json PATH           Manifest path (default: data/instances/manifest.json)
  --output-root DIR              Output root for scale-specific logs (default: results/rl)
  --python BIN                   Python executable (default: python3)
  --seed INT                     Global seed per scale run (default: 42)
  --total-timesteps INT          PPO total timesteps (default: 1000000)
  --eval-freq INT                Evaluation frequency in steps (default: 50000)
  --eval-episodes INT            Evaluation episodes per checkpoint (default: 3)
  --max-time-s FLOAT             Episode horizon in seconds (default: 28800)
  --max-no-progress-steps INT    No-progress guard (default: 512)
  --no-progress-time-epsilon F   No-progress epsilon (default: 1e-9)
  --dry-run                      Print commands without executing
  -h, --help                     Show this help and exit

Notes:
  - This wrapper enforces consistent core training arguments across scales.
  - Additional args after `--` are forwarded verbatim to train_maskable_ppo.py.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_JSON="${MANIFEST_JSON:-${ROOT_DIR}/data/instances/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/rl}"
SEED="${SEED:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
EVAL_FREQ="${EVAL_FREQ:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
MAX_TIME_S="${MAX_TIME_S:-28800}"
MAX_NO_PROGRESS_STEPS="${MAX_NO_PROGRESS_STEPS:-512}"
NO_PROGRESS_TIME_EPSILON="${NO_PROGRESS_TIME_EPSILON:-1e-9}"
SCALES=(S M L XL)
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --scales)
      shift
      if [[ $# -eq 0 ]]; then
        echo "missing value for --scales" >&2
        exit 2
      fi
      scale_arg="${1//,/ }"
      # shellcheck disable=SC2206
      SCALES=($scale_arg)
      ;;
    --manifest-json)
      shift
      MANIFEST_JSON="${1:?missing value for --manifest-json}"
      ;;
    --output-root)
      shift
      OUTPUT_ROOT="${1:?missing value for --output-root}"
      ;;
    --python)
      shift
      PYTHON_BIN="${1:?missing value for --python}"
      ;;
    --seed)
      shift
      SEED="${1:?missing value for --seed}"
      ;;
    --total-timesteps)
      shift
      TOTAL_TIMESTEPS="${1:?missing value for --total-timesteps}"
      ;;
    --eval-freq)
      shift
      EVAL_FREQ="${1:?missing value for --eval-freq}"
      ;;
    --eval-episodes)
      shift
      EVAL_EPISODES="${1:?missing value for --eval-episodes}"
      ;;
    --max-time-s)
      shift
      MAX_TIME_S="${1:?missing value for --max-time-s}"
      ;;
    --max-no-progress-steps)
      shift
      MAX_NO_PROGRESS_STEPS="${1:?missing value for --max-no-progress-steps}"
      ;;
    --no-progress-time-epsilon)
      shift
      NO_PROGRESS_TIME_EPSILON="${1:?missing value for --no-progress-time-epsilon}"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -f "${MANIFEST_JSON}" ]]; then
  echo "manifest not found: ${MANIFEST_JSON}" >&2
  exit 1
fi

if [[ ${#SCALES[@]} -eq 0 ]]; then
  echo "no scales selected" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

for scale in "${SCALES[@]}"; do
  log_dir="${OUTPUT_ROOT}/train_${scale}"
  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_maskable_ppo.py"
    --manifest-json "${MANIFEST_JSON}"
    --scale "${scale}"
    --seed "${SEED}"
    --total-timesteps "${TOTAL_TIMESTEPS}"
    --eval-freq "${EVAL_FREQ}"
    --eval-episodes "${EVAL_EPISODES}"
    --max-time-s "${MAX_TIME_S}"
    --max-no-progress-steps "${MAX_NO_PROGRESS_STEPS}"
    --no-progress-time-epsilon "${NO_PROGRESS_TIME_EPSILON}"
    --log-dir "${log_dir}"
  )
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "[train_all_scales] scale=${scale} log_dir=${log_dir}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  %q' "${cmd[@]}"
    printf '\n'
    continue
  fi
  "${cmd[@]}"
done
