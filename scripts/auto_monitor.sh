#!/usr/bin/env bash
set -uo pipefail

# Auto-monitor training and trigger evaluations as scales complete.
# Runs as a background daemon, checks every 5 minutes.

cd /mnt/f/simulation3
LOG="/mnt/f/simulation3/results/auto_monitor.log"
mkdir -p results/benchmark results/paper

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Track which evaluations have been triggered
declare -A EVAL_DONE
EVAL_DONE=([S]=1)  # S already evaluated

# Scale-specific eval parameters (must match training)
declare -A MAX_TIME=([S]=20000 [M]=24000 [L]=26000 [XL]=25000)
declare -A TERM_PEN=([S]=3000 [M]=2500 [L]=2000 [XL]=1500)
declare -A TARD_SC=([S]=0.2 [M]=0.25 [L]=0.3 [XL]=0.2)
declare -A TARGET=([S]=1000000 [M]=1000000 [L]=1000000 [XL]=500000)

run_eval() {
    local scale="$1"
    local algos="rl_apc,greedy_fr,greedy_pr,random_rule,alns_fr,alns_pr"
    # Add MIP for S and M only
    if [[ "$scale" == "S" || "$scale" == "M" ]]; then
        algos="${algos},mip_hind"
    fi
    log "EVAL START: ${scale} (algos: ${algos})"
    python3 scripts/evaluate_all.py \
      --scale "${scale}" --split test \
      --algorithms "${algos}" \
      --ppo-model-path "results/rl/train_${scale}/best_model" \
      --max-time-s "${MAX_TIME[$scale]}" \
      --terminal-penalty "${TERM_PEN[$scale]}" \
      --tardiness-scale "${TARD_SC[$scale]}" \
      --output-csv "results/benchmark/evaluate_${scale}.csv" \
      --output-summary-json "results/benchmark/evaluate_${scale}_summary.json" \
      >> "$LOG" 2>&1
    log "EVAL DONE: ${scale}"
    EVAL_DONE[$scale]=1
}

check_training_done() {
    local scale="$1"
    local path="results/rl/train_${scale}"
    # Check if process is still running
    if ps aux | grep "train_maskable_ppo" | grep -v grep | grep "train_${scale}" | grep -q "python3"; then
        return 1  # still running
    fi
    # Process not running - check if we have a model
    if [[ -f "${path}/best_model/best_model.zip" ]]; then
        return 0  # done
    fi
    return 1  # no model yet
}

log "=== Auto Monitor Started ==="
log "S evaluation already complete."

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))

    # Check each scale
    ALL_TRAIN_DONE=true
    ALL_EVAL_DONE=true

    for scale in M L XL; do
        if [[ -z "${EVAL_DONE[$scale]+x}" ]]; then
            ALL_EVAL_DONE=false
            if check_training_done "$scale"; then
                log "TRAINING COMPLETE: ${scale} - triggering evaluation"
                run_eval "$scale"
            else
                ALL_TRAIN_DONE=false
                # Report progress from NPZ
                python3 -c "
import numpy as np, os
npz = 'results/rl/train_${scale}/eval_logs/evaluations.npz'
if os.path.exists(npz):
    d = np.load(npz)
    ts = d['timesteps']
    rew = d['results'].mean(axis=1)
    bi = rew.argmax()
    print(f'${scale}: {int(ts[-1])}/${TARGET[$scale]} ({int(ts[-1])/${TARGET[$scale]}*100:.0f}%), best={rew[bi]:.0f}')
else:
    print('${scale}: no eval data')
" 2>/dev/null | tee -a "$LOG"
            fi
        fi
    done

    # Check multi-seed
    for seed in 43 44; do
        sname="S_seed${seed}"
        spath="results/rl/train_S_seed${seed}"
        if ! ps aux | grep "train_maskable_ppo" | grep -v grep | grep "train_S_seed${seed}" | grep -q "python3"; then
            if [[ -f "${spath}/best_model/best_model.zip" ]]; then
                log "S seed${seed} training complete"
            fi
        else
            python3 -c "
import numpy as np, os
npz = '${spath}/eval_logs/evaluations.npz'
if os.path.exists(npz):
    d = np.load(npz)
    ts = d['timesteps']
    rew = d['results'].mean(axis=1)
    bi = rew.argmax()
    print(f'S_seed${seed}: {int(ts[-1])}/1000000 ({int(ts[-1])/1000000*100:.0f}%), best={rew[bi]:.0f}')
" 2>/dev/null | tee -a "$LOG"
        fi
    done

    # If all main evals done, generate paper assets
    if [[ "$ALL_EVAL_DONE" == true ]]; then
        log "ALL EVALUATIONS COMPLETE - generating paper assets"
        python3 scripts/generate_paper_results.py \
          --scales S,M,L,XL \
          --results-root results/rl \
          --benchmark-root results/benchmark \
          --output-dir results/paper >> "$LOG" 2>&1
        log "Paper assets generated"

        python3 scripts/validate_results.py >> "$LOG" 2>&1
        log "Validation complete"
        log "=== ALL DONE ==="
        break
    fi

    log "--- iteration ${ITERATION}, sleeping 5min ---"
    sleep 300
done
