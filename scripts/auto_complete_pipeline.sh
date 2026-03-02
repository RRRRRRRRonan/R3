#!/usr/bin/env bash
# Auto-completion pipeline: monitors training, evaluates, generates paper assets.
# Runs unattended until all phases complete.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="results/auto_pipeline.log"
mkdir -p results/benchmark results/paper

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Scale-specific eval parameters (must match training)
eval_params() {
    local scale=$1
    case "$scale" in
        S)  echo "--terminal-penalty 3000 --tardiness-scale 0.2" ;;
        M)  echo "--terminal-penalty 2500 --tardiness-scale 0.25" ;;
        L)  echo "--terminal-penalty 2000 --tardiness-scale 0.3" ;;
        XL) echo "--terminal-penalty 1500 --tardiness-scale 0.2" ;;
    esac
}

# Wait for training by watching for the process to exit.
# Uses process-based detection since final_model.zip may be stale.
wait_for_training() {
    local name=$1 dir=$2 keyword=$3

    # Check if any matching python process is running
    local running
    running=$(ps aux | grep "train_maskable_ppo" | grep "$keyword" | grep -v grep | wc -l)

    if [ "$running" -eq 0 ]; then
        # No process running - check if we have a model
        if [ -d "$dir/best_model" ]; then
            log "$name: process finished, best_model exists"
            return 0
        else
            log "$name: process not running and no best_model - may need manual intervention"
            return 1
        fi
    fi

    log "$name: training in progress (process detected), waiting..."
    while true; do
        sleep 120
        running=$(ps aux | grep "train_maskable_ppo" | grep "$keyword" | grep -v grep | wc -l)
        if [ "$running" -eq 0 ]; then
            log "$name: training process completed!"
            sleep 5  # Let files flush
            return 0
        fi
        # Show progress
        if [ -f "$dir/eval_logs/evaluations.npz" ]; then
            python3 -c "
import numpy as np
d=np.load('$dir/eval_logs/evaluations.npz')
ts=d['timesteps']; r=d['results']
print(f'  $name: {ts[-1]/1e6:.2f}M steps, best={r.mean(axis=1).min():.0f}')
" 2>/dev/null | tee -a "$LOG" || true
        fi
    done
}

# Run evaluation for a scale
run_eval() {
    local scale=$1 model_dir=$2
    local params
    params=$(eval_params "$scale")
    local csv="results/benchmark/evaluate_${scale}_30.csv"
    local json="results/benchmark/evaluate_${scale}_30_summary.json"

    if [ -f "$json" ]; then
        log "eval $scale: already exists ($json), skipping"
        return 0
    fi

    if [ ! -d "$model_dir" ]; then
        log "eval $scale: model dir $model_dir not found, SKIPPING"
        return 1
    fi

    log "eval $scale: starting 30-instance evaluation..."
    python3 scripts/evaluate_all.py \
        --scale "$scale" --split test \
        --algorithms "rl_apc,greedy_fr,greedy_pr,random_rule,alns_fr,alns_pr" \
        --ppo-model-path "$model_dir" \
        $params \
        --output-csv "$csv" \
        --output-summary-json "$json" \
        2>&1 | grep -E "\[info\]|saved|Error|status=" | tee -a "$LOG"
    log "eval $scale: DONE → $json"
}

# ============ MAIN PIPELINE ============

log "===== Auto-completion pipeline started ====="

# Phase 1: Wait for all training to complete
log "--- Phase 1: Waiting for training ---"

# Wait for each training job (use grep keywords to identify processes)
wait_for_training "S_seed43" "results/rl/train_S_seed43" "seed 43" &
PID_S43=$!
wait_for_training "S_seed44" "results/rl/train_S_seed44" "seed 44" &
PID_S44=$!
wait_for_training "M" "results/rl/train_M" "scale M" &
PID_M=$!
wait_for_training "L" "results/rl/train_L" "scale L" &
PID_L=$!

# Wait for all training to finish
wait $PID_S43 $PID_S44 $PID_M $PID_L || true
log "All training complete!"

# Phase 2: Evaluate all scales (30 test instances each)
log "--- Phase 2: Evaluating all scales ---"

run_eval S "results/rl/train_S/best_model"
run_eval M "results/rl/train_M/best_model"
run_eval L "results/rl/train_L/best_model"
run_eval XL "results/rl/train_XL/best_model"

log "All evaluations complete!"

# Phase 3: Statistical analysis
log "--- Phase 3: Statistical analysis ---"
# Determine which scales have results
FOUND_SCALES=""
for scale in S M L XL; do
    if [ -f "results/benchmark/evaluate_${scale}_30_summary.json" ]; then
        [ -n "$FOUND_SCALES" ] && FOUND_SCALES="$FOUND_SCALES,"
        FOUND_SCALES="${FOUND_SCALES}${scale}"
    fi
done

if [ -n "$FOUND_SCALES" ]; then
    python3 scripts/statistical_analysis.py \
        --scales "$FOUND_SCALES" \
        --benchmark-root results/benchmark \
        --output-dir results/paper \
        --suffix _30 \
        2>&1 | tee -a "$LOG"
    log "Statistical analysis complete"
else
    log "WARNING: No evaluation results found for statistical analysis"
fi

# Phase 4: Generate paper figures (training curves)
log "--- Phase 4: Generating paper assets ---"
python3 scripts/generate_paper_results.py \
    --scales S,M,L,XL \
    --results-root results/rl \
    --benchmark-root results/benchmark \
    --output-dir results/paper \
    2>&1 | tee -a "$LOG" || log "Warning: generate_paper_results.py had errors"

# Phase 5: Multi-seed analysis for S
log "--- Phase 5: Multi-seed S analysis ---"
python3 -c "
import numpy as np

seeds = {'seed42': 'results/rl/train_S/eval_logs/evaluations.npz',
         'seed43': 'results/rl/train_S_seed43/eval_logs/evaluations.npz',
         'seed44': 'results/rl/train_S_seed44/eval_logs/evaluations.npz'}

print('S-scale multi-seed training results:')
print(f'{\"Seed\":<10} {\"Best Reward\":>14} {\"Final Reward\":>14} {\"Steps\":>10}')
print('-'*52)
bests = []
for name, path in seeds.items():
    try:
        d = np.load(path)
        ts = d['timesteps']
        r = d['results'].mean(axis=1)
        best = r.min()
        bests.append(best)
        print(f'{name:<10} {best:>14,.0f} {r[-1]:>14,.0f} {ts[-1]:>10,}')
    except Exception as e:
        print(f'{name:<10} ERROR: {e}')

if len(bests) >= 2:
    bests = np.array(bests)
    print(f'\nMean best reward: {bests.mean():,.0f} (std={bests.std():,.0f}, CV={bests.std()/abs(bests.mean())*100:.1f}%)')
" 2>&1 | tee -a "$LOG"

# Phase 6: Final summary
log "--- Phase 6: Final Summary ---"
python3 -c "
import json, os

print('='*60)
print('  FINAL RESULTS SUMMARY')
print('='*60)

for scale in ['S','M','L','XL']:
    f = f'results/benchmark/evaluate_{scale}_30_summary.json'
    if not os.path.exists(f):
        print(f'\n{scale}: NOT EVALUATED')
        continue
    d = json.load(open(f))
    algos = d['algorithms']
    print(f'\n{scale} scale ({d[\"total_rows\"]} rows):')
    for algo, info in sorted(algos.items(), key=lambda x: x[1]['avg_cost']):
        marker = ' ★' if algo == 'rl_apc' else ''
        print(f'  {algo:<15} avg_cost={info[\"avg_cost\"]:>12,.0f} (n={info[\"count\"]}){marker}')

    # RL vs Greedy comparison
    if 'rl_apc' in algos and 'greedy_fr' in algos:
        rl = algos['rl_apc']['avg_cost']
        gr = algos['greedy_fr']['avg_cost']
        if rl < gr:
            imp = (gr - rl) / gr * 100
            print(f'  → RL-APC beats Greedy by {imp:.1f}%')
        else:
            deg = (rl - gr) / gr * 100
            print(f'  → RL-APC WORSE than Greedy by {deg:.1f}%')

print('\n' + '='*60)
print('Paper assets in: results/paper/')
for f in sorted(os.listdir('results/paper')) if os.path.isdir('results/paper') else []:
    size = os.path.getsize(f'results/paper/{f}')
    print(f'  {f} ({size:,} bytes)')
print('='*60)
" 2>&1 | tee -a "$LOG"

log "===== Pipeline complete! ====="
