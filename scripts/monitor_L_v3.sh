#!/usr/bin/env bash
# Monitor L_v3 training and auto-evaluate when done.
# L_v3 uses: terminal_penalty=1500, ent=0.08, net=[512,256], 2M steps
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="results/L_v3_pipeline.log"
mkdir -p results/benchmark results/paper

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "===== L_v3 Monitor Started ====="

# Wait for L_v3 process to finish
while true; do
    running=$(ps aux | grep "train_maskable_ppo" | grep "train_L_v3" | grep -v grep | wc -l)
    if [ "$running" -eq 0 ]; then
        log "L_v3: training process completed!"
        sleep 5
        break
    fi
    # Show progress
    if [ -f "results/rl/train_L_v3/eval_logs/evaluations.npz" ]; then
        python3 -c "
import numpy as np
d=np.load('results/rl/train_L_v3/eval_logs/evaluations.npz')
ts=d['timesteps']; r=d['results'].mean(axis=1)
print(f'  L_v3: {ts[-1]/1e6:.2f}M/{2.0:.1f}M steps ({ts[-1]/2e6*100:.0f}%), latest={r[-1]:.0f}, best={r.max():.0f} @ {ts[r.argmax()]:,}')
" 2>/dev/null | tee -a "$LOG" || true
    fi
    sleep 120
done

# Evaluate — NOTE: terminal_penalty=1500 matches L_v3 training (NOT the default L=2000)
CSV="results/benchmark/evaluate_L_v3_30.csv"
JSON="results/benchmark/evaluate_L_v3_30_summary.json"

if [ -f "$JSON" ]; then
    log "eval L_v3: already exists, skipping"
else
    log "eval L_v3: starting 30-instance evaluation..."
    python3 scripts/evaluate_all.py \
        --scale L --split test \
        --algorithms "rl_apc,greedy_fr,greedy_pr,random_rule,alns_fr,alns_pr" \
        --ppo-model-path "results/rl/train_L_v3/best_model" \
        --terminal-penalty 1500 --tardiness-scale 0.3 \
        --output-csv "$CSV" \
        --output-summary-json "$JSON" \
        2>&1 | tee -a "$LOG"
    log "eval L_v3: DONE"
fi

# Compare all L versions
log "--- L Version Comparison ---"
python3 -c "
import json, os

print('='*70)
print('  L SCALE: v1 vs v2 vs v3')
print('='*70)

versions = {
    'v1 [256,128] tp=2000 ent=0.05': 'results/benchmark/evaluate_L_30_summary.json',
    'v1 final_model (synced)':        'results/benchmark/evaluate_L_final_30_summary.json',
    'v2 [512,256] tp=2000 ent=0.05': 'results/benchmark/evaluate_L_v2_30_summary.json',
    'v3 [512,256] tp=1500 ent=0.08': 'results/benchmark/evaluate_L_v3_30_summary.json',
}

greedy = None
for label, path in versions.items():
    if not os.path.exists(path):
        print(f'  {label:<38} N/A')
        continue
    d = json.load(open(path))
    rl = d['algorithms']['rl_apc']['avg_cost']
    gr = d['algorithms']['greedy_fr']['avg_cost']
    if greedy is None:
        greedy = gr
    diff = (rl - gr) / gr * 100
    status = 'WINS' if rl < gr else 'TIES' if abs(diff) < 5 else 'LOSES'
    print(f'  {label:<38} RL={rl:>12,.0f}  Greedy={gr:>10,.0f}  {diff:+6.1f}%  {status}')

print()
" 2>&1 | tee -a "$LOG"

# Regenerate full comparison report with latest results
log "Regenerating full_comparison_report.txt..."
python3 scripts/generate_comparison_report.py 2>&1 | tail -3 | tee -a "$LOG"

# Regenerate ALL paper results (ejor_tables.tex, CSVs, figures)
# generate_ejor_tables.py auto-picks v3 for L scale via _pick_best_csv()
log "Regenerating EJOR paper tables (ejor_tables.tex + CSVs)..."
python3 scripts/generate_ejor_tables.py 2>&1 | tee -a "$LOG"

log "Regenerating paper results (table1-3, wilcoxon, training curves, vecnorm)..."
python3 scripts/generate_paper_results_synced.py 2>&1 | tee -a "$LOG"

# Summary: show which CSV was picked for each scale
log "--- Final Data Sources ---"
python3 -c "
import os
from pathlib import Path

BENCHMARK = Path('results/benchmark')
for scale in ['S', 'M', 'L', 'XL']:
    candidates = [
        (f'evaluate_{scale}_v3_30.csv', 'v3'),
        (f'evaluate_{scale}_synced_30.csv', 'synced'),
        (f'evaluate_{scale}_v2_30.csv', 'v2'),
        (f'evaluate_{scale}_30.csv', 'v1'),
    ]
    for fname, label in candidates:
        if (BENCHMARK / fname).exists():
            print(f'  {scale}: {fname} ({label})')
            break
    else:
        print(f'  {scale}: N/A')
" 2>&1 | tee -a "$LOG"

log "===== L_v3 Pipeline Complete ====="
