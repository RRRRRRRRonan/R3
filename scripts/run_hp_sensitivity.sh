#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# Hyperparameter Sensitivity: 3×3 grid (η × β) on M-scale, 500K steps
# ══════════════════════════════════════════════════════════════════════════
#
# Grid:
#   η  (learning rate) ∈ {1e-4, 3e-4, 1e-3}
#   β  (entropy coef)  ∈ {0.01, 0.05, 0.10}
#
# Base config ★ = (3e-4, 0.05) already trained as train_M.
# This script trains the remaining 8 configs, 2 at a time.
#
# Usage:  bash scripts/run_hp_sensitivity.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL_STEPS=500000
SCALE=M
SEED=42
MANIFEST=data/instances/manifest.json
MAX_TIME=24000
TERMINAL=2500
TARDINESS=0.25
NET_ARCH="256,128"
EVAL_FREQ=50000
EVAL_EPS=3
MAX_PARALLEL=2

# ── Grid definition ──────────────────────────────────────────────────────
# Format: "lr ent label"
CONFIGS=(
    "1e-4  0.01  lr1e4_ent001"
    "1e-4  0.05  lr1e4_ent005"
    "1e-4  0.10  lr1e4_ent010"
    "3e-4  0.01  lr3e4_ent001"
    # "3e-4  0.05  lr3e4_ent005"  ← base, skip (use train_M)
    "3e-4  0.10  lr3e4_ent010"
    "1e-3  0.01  lr1e3_ent001"
    "1e-3  0.05  lr1e3_ent005"
    "1e-3  0.10  lr1e3_ent010"
)

echo "═══════════════════════════════════════════════════════════════"
echo "  HP Sensitivity: ${#CONFIGS[@]} configs × ${TOTAL_STEPS} steps"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "═══════════════════════════════════════════════════════════════"

PIDS=()
LABELS=()

wait_slot() {
    # Wait until fewer than MAX_PARALLEL jobs are running
    while true; do
        local running=0
        local new_pids=()
        local new_labels=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                running=$((running + 1))
                new_pids+=("${PIDS[$i]}")
                new_labels+=("${LABELS[$i]}")
            else
                wait "${PIDS[$i]}" || true
                echo "  ✓ Finished: ${LABELS[$i]} (PID ${PIDS[$i]})"
            fi
        done
        PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        LABELS=("${new_labels[@]+"${new_labels[@]}"}")
        if [ "$running" -lt "$MAX_PARALLEL" ]; then
            break
        fi
        sleep 30
    done
}

wait_all() {
    for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            wait "${PIDS[$i]}" || true
            echo "  ✓ Finished: ${LABELS[$i]} (PID ${PIDS[$i]})"
        fi
    done
    PIDS=()
    LABELS=()
}

# ── Launch training ──────────────────────────────────────────────────────
for cfg in "${CONFIGS[@]}"; do
    read -r lr ent label <<< "$cfg"
    log_dir="results/rl/train_M_hp_${label}"
    log_file="results/rl/train_M_hp_${label}.log"

    if [ -f "${log_dir}/final_model.zip" ]; then
        echo "[SKIP] ${label} — final_model.zip already exists"
        continue
    fi

    wait_slot

    echo "[LAUNCH] ${label}: lr=${lr}, ent=${ent} → ${log_dir}"

    nohup python3 -u scripts/train_maskable_ppo.py \
        --scale "$SCALE" \
        --manifest-json "$MANIFEST" \
        --log-dir "$log_dir" \
        --total-timesteps "$TOTAL_STEPS" \
        --seed "$SEED" \
        --max-time-s "$MAX_TIME" \
        --terminal-penalty "$TERMINAL" \
        --tardiness-scale "$TARDINESS" \
        --ent-coef "$ent" \
        --learning-rate "$lr" \
        --net-arch "$NET_ARCH" \
        --eval-freq "$EVAL_FREQ" \
        --eval-episodes "$EVAL_EPS" \
        > "$log_file" 2>&1 &

    PIDS+=("$!")
    LABELS+=("$label")
    echo "  PID $!"
    sleep 2  # stagger startup slightly
done

echo ""
echo "All jobs launched. Waiting for completion ..."
wait_all

# ── Evaluate each config ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Evaluation phase"
echo "═══════════════════════════════════════════════════════════════"

# Base config (train_M) — use existing eval if available
BASE_EVAL="results/benchmark/evaluate_M_hp_lr3e4_ent005.csv"
if [ ! -f "$BASE_EVAL" ]; then
    echo "[EVAL] base (lr=3e-4, ent=0.05) → ${BASE_EVAL}"
    python3 -u scripts/evaluate_all.py \
        --scale M --split test \
        --algorithms rl_apc \
        --ppo-model-path results/rl/train_M/best_model \
        --terminal-penalty "$TERMINAL" --tardiness-scale "$TARDINESS" \
        --max-time-s "$MAX_TIME" \
        --output-csv "$BASE_EVAL" 2>&1 | tail -3
else
    echo "[SKIP] base eval already exists: ${BASE_EVAL}"
fi

for cfg in "${CONFIGS[@]}"; do
    read -r lr ent label <<< "$cfg"
    model_dir="results/rl/train_M_hp_${label}/best_model"
    eval_csv="results/benchmark/evaluate_M_hp_${label}.csv"

    if [ -f "$eval_csv" ]; then
        echo "[SKIP] ${label} eval already exists"
        continue
    fi

    if [ ! -d "$model_dir" ]; then
        echo "[WARN] ${label} — no best_model found, skipping eval"
        continue
    fi

    echo "[EVAL] ${label} → ${eval_csv}"
    python3 -u scripts/evaluate_all.py \
        --scale M --split test \
        --algorithms rl_apc \
        --ppo-model-path "$model_dir" \
        --terminal-penalty "$TERMINAL" --tardiness-scale "$TARDINESS" \
        --max-time-s "$MAX_TIME" \
        --output-csv "$eval_csv" 2>&1 | tail -3
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All done! Run the heatmap generator next:"
echo "  python3 scripts/generate_hp_sensitivity_heatmap.py"
echo "═══════════════════════════════════════════════════════════════"
