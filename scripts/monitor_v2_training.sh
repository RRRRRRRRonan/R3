#!/usr/bin/env bash
# Monitor v2 training (L_v2 and XL_v2 with larger network + VecNormalize fix)
# and auto-evaluate when done.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="results/v2_pipeline.log"
mkdir -p results/benchmark results/paper

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

eval_params() {
    local scale=$1
    case "$scale" in
        S)  echo "--terminal-penalty 3000 --tardiness-scale 0.2" ;;
        M)  echo "--terminal-penalty 2500 --tardiness-scale 0.25" ;;
        L)  echo "--terminal-penalty 2000 --tardiness-scale 0.3" ;;
        XL) echo "--terminal-penalty 1500 --tardiness-scale 0.2" ;;
    esac
}

wait_for_process() {
    local name=$1 keyword=$2
    while true; do
        local running
        running=$(ps aux | grep "train_maskable_ppo" | grep "$keyword" | grep -v grep | wc -l)
        if [ "$running" -eq 0 ]; then
            log "$name: training process completed!"
            sleep 5
            return 0
        fi
        # Show progress
        local dir="results/rl/${name}"
        if [ -f "$dir/eval_logs/evaluations.npz" ]; then
            python3 -c "
import numpy as np
d=np.load('$dir/eval_logs/evaluations.npz')
ts=d['timesteps']; r=d['results'].mean(axis=1)
print(f'  $name: {ts[-1]/1e6:.2f}M steps, latest={r[-1]:.0f}, best={r.max():.0f}')
" 2>/dev/null | tee -a "$LOG" || true
        fi
        sleep 120
    done
}

run_eval() {
    local scale=$1 model_dir=$2 suffix=$3
    local params
    params=$(eval_params "$scale")
    local csv="results/benchmark/evaluate_${scale}${suffix}.csv"
    local json="results/benchmark/evaluate_${scale}${suffix}_summary.json"

    if [ -f "$json" ]; then
        log "eval $scale ($suffix): already exists, skipping"
        return 0
    fi

    # Use the vecnormalize.pkl from best_model dir (v2 fix saves it there)
    local vecnorm_dir="$model_dir"
    if [ ! -f "$vecnorm_dir/vecnormalize.pkl" ]; then
        # Fallback to parent dir
        vecnorm_dir="$(dirname "$model_dir")"
    fi

    log "eval $scale ($suffix): starting 30-instance evaluation..."
    python3 scripts/evaluate_all.py \
        --scale "$scale" --split test \
        --algorithms "rl_apc,greedy_fr,greedy_pr,random_rule,alns_fr,alns_pr" \
        --ppo-model-path "$model_dir" \
        $params \
        --output-csv "$csv" \
        --output-summary-json "$json" \
        2>&1 | tee -a "$LOG"
    log "eval $scale ($suffix): DONE"
}

# ============ MAIN ============
log "===== V2 Training Monitor Started ====="

# Wait for both v2 trainings
wait_for_process "train_L_v2" "train_L_v2" &
PID_L=$!
wait_for_process "train_XL_v2" "train_XL_v2" &
PID_XL=$!

wait $PID_L $PID_XL || true
log "All v2 training complete!"

# Evaluate with _v2_30 suffix
run_eval L "results/rl/train_L_v2/best_model" "_v2_30"
run_eval XL "results/rl/train_XL_v2/best_model" "_v2_30"

# Compare v1 vs v2
log "--- Comparison: v1 vs v2 ---"
python3 -c "
import json, os

print('='*70)
print('  V1 vs V2 COMPARISON (with VecNormalize fix + [512,256] network)')
print('='*70)

for scale in ['L', 'XL']:
    v1_f = f'results/benchmark/evaluate_{scale}_30_summary.json'
    v2_f = f'results/benchmark/evaluate_{scale}_v2_30_summary.json'

    v1_rl = v1_gr = v2_rl = v2_gr = None
    if os.path.exists(v1_f):
        d = json.load(open(v1_f))
        v1_rl = d['algorithms'].get('rl_apc', {}).get('avg_cost')
        v1_gr = d['algorithms'].get('greedy_fr', {}).get('avg_cost')
    if os.path.exists(v2_f):
        d = json.load(open(v2_f))
        v2_rl = d['algorithms'].get('rl_apc', {}).get('avg_cost')
        v2_gr = d['algorithms'].get('greedy_fr', {}).get('avg_cost')

    print(f'\n{scale} scale:')
    if v1_rl: print(f'  v1 RL:     {v1_rl:>12,.0f}  (net=[256,128], VecNorm bug)')
    if v2_rl: print(f'  v2 RL:     {v2_rl:>12,.0f}  (net=[512,256], VecNorm fix)')
    if v1_gr: print(f'  Greedy:    {v1_gr:>12,.0f}')
    if v1_rl and v2_rl:
        imp = (v1_rl - v2_rl) / v1_rl * 100
        print(f'  v2 vs v1:  {imp:+.1f}% improvement')
    if v2_rl and v2_gr:
        if v2_rl < v2_gr:
            print(f'  v2 vs Greedy: RL wins by {(v2_gr-v2_rl)/v2_gr*100:.1f}%')
        else:
            print(f'  v2 vs Greedy: RL loses by {(v2_rl-v2_gr)/v2_gr*100:.1f}%')

print()
" 2>&1 | tee -a "$LOG"

log "===== V2 Pipeline Complete ====="
