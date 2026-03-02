#!/usr/bin/env bash
set -euo pipefail

# Master pipeline script for paper experiments (Route A)
# Runs: Phase 2 (benchmark eval) → Phase 4 (paper assets) → Phase 5 (validation)
#
# Prerequisites: All training must be complete (S/M/L/XL best_model.zip exist)
#
# Usage:
#   scripts/run_paper_pipeline.sh [--skip-eval] [--skip-multiseed] [--only-scale S]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

SKIP_EVAL=0
SKIP_MULTISEED=0
ONLY_SCALE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-eval) SKIP_EVAL=1 ;;
    --skip-multiseed) SKIP_MULTISEED=1 ;;
    --only-scale) shift; ONLY_SCALE="$1" ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

echo "================================================================"
echo "Paper Experiment Pipeline (Route A)"
echo "================================================================"
echo ""

# ------------------------------------------------------------------
# Pre-flight: check all best_models exist
# ------------------------------------------------------------------
echo "--- Pre-flight checks ---"
MISSING=0
for scale in S M L XL; do
  model="results/rl/train_${scale}/best_model/best_model.zip"
  if [[ -f "${model}" ]]; then
    echo "  [OK] ${model}"
  else
    echo "  [MISSING] ${model}"
    MISSING=1
  fi
done
if [[ ${MISSING} -eq 1 ]]; then
  echo ""
  echo "ERROR: Some best_model.zip files are missing. Wait for training to complete."
  exit 1
fi
echo ""

# ------------------------------------------------------------------
# Phase 2: Benchmark evaluation (all scales)
# ------------------------------------------------------------------
if [[ ${SKIP_EVAL} -eq 0 ]]; then
  echo "--- Phase 2: Benchmark Evaluation ---"
  mkdir -p results/benchmark

  # Scale-specific evaluation parameters (must match training)
  _eval_max_time() {
    case "$1" in
      S)  echo 20000 ;; M)  echo 24000 ;; L)  echo 26000 ;; XL) echo 25000 ;; *)  echo 28800 ;;
    esac
  }
  _eval_terminal_penalty() {
    case "$1" in
      S)  echo 3000 ;; M)  echo 2500 ;; L)  echo 2000 ;; XL) echo 1500 ;; *)  echo 1000 ;;
    esac
  }
  _eval_tardiness_scale() {
    case "$1" in
      S)  echo 0.2 ;; M)  echo 0.25 ;; L)  echo 0.3 ;; XL) echo 0.2 ;; *)  echo 0.5 ;;
    esac
  }

  run_eval() {
    local scale="$1"
    local algos="$2"
    local mt; mt="$(_eval_max_time "${scale}")"
    local tp; tp="$(_eval_terminal_penalty "${scale}")"
    local ts; ts="$(_eval_tardiness_scale "${scale}")"
    echo ""
    echo "[eval] Scale ${scale} (max_time=${mt}, terminal_penalty=${tp}, tardiness_scale=${ts})"
    python3 scripts/evaluate_all.py \
      --scale "${scale}" --split test \
      --algorithms "${algos}" \
      --ppo-model-path "results/rl/train_${scale}/best_model" \
      --max-time-s "${mt}" \
      --terminal-penalty "${tp}" --tardiness-scale "${ts}" \
      --output-csv "results/benchmark/evaluate_${scale}.csv" \
      --output-summary-json "results/benchmark/evaluate_${scale}_summary.json"
    echo "[eval] Scale ${scale} done."
  }

  SCALES_TO_EVAL=(S M L XL)
  if [[ -n "${ONLY_SCALE}" ]]; then
    SCALES_TO_EVAL=("${ONLY_SCALE}")
  fi

  for scale in "${SCALES_TO_EVAL[@]}"; do
    case "${scale}" in
      S|M)
        run_eval "${scale}" "rl_apc,greedy_fr,greedy_pr,random_rule,alns_fr,alns_pr,mip_hind"
        ;;
      L|XL)
        run_eval "${scale}" "rl_apc,greedy_fr,greedy_pr,random_rule,alns_fr,alns_pr"
        ;;
    esac
  done
  echo ""
fi

# ------------------------------------------------------------------
# Phase 3: S multi-seed (optional)
# ------------------------------------------------------------------
if [[ ${SKIP_MULTISEED} -eq 0 ]]; then
  echo "--- Phase 3: S Multi-Seed Training ---"
  for seed in 43 44; do
    log_dir="results/rl/train_S_seed${seed}"
    if [[ -f "${log_dir}/best_model/best_model.zip" ]]; then
      echo "[multiseed] Seed ${seed} already trained, skipping."
      continue
    fi
    echo "[multiseed] Training S with seed ${seed}..."
    python3 scripts/train_maskable_ppo.py \
      --manifest-json data/instances/manifest.json \
      --scale S --seed "${seed}" \
      --total-timesteps 1000000 \
      --eval-freq 50000 --eval-episodes 3 \
      --max-time-s 20000 \
      --terminal-penalty 3000 --tardiness-scale 0.2 \
      --log-dir "${log_dir}"
    echo "[multiseed] Seed ${seed} done."
  done
  echo ""
fi

# ------------------------------------------------------------------
# Phase 4: Generate paper assets
# ------------------------------------------------------------------
echo "--- Phase 4: Paper Tables & Figures ---"
python3 scripts/generate_paper_results.py \
  --scales S,M,L,XL \
  --results-root results/rl \
  --benchmark-root results/benchmark \
  --output-dir results/paper
echo ""

# ------------------------------------------------------------------
# Phase 5: Validation
# ------------------------------------------------------------------
echo "--- Phase 5: Validation ---"
python3 scripts/validate_results.py
echo ""

echo "================================================================"
echo "Pipeline complete. Results in: results/paper/"
echo "================================================================"
