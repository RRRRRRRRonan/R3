#!/bin/bash
# Week 1, Day 1-3: Baseline Collection
#
# This script collects baseline performance data for the current Q-learning
# implementation (zero initialization) across 10 seeds and 3 scales.
#
# Expected runtime: ~30 minutes (10 seeds × 3 scales)
# Expected output: 30 JSON files

set -e  # Exit on error

# Configuration
SEEDS=(2025 2026 2027 2028 2029 2030 2031 2032 2033 2034)
SCENARIOS=("small" "medium" "large")
INIT_STRATEGY="zero"  # Current baseline
OUTPUT_DIR="results/week1/baseline"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print header
echo "======================================================================"
echo "Week 1: Baseline Collection (Zero Initialization)"
echo "======================================================================"
echo "Start time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Total runs: $((${#SEEDS[@]} * ${#SCENARIOS[@]}))"
echo ""

# Run experiments
total=0
success=0
failed=0

for scenario in "${SCENARIOS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Running ${scenario} scenario..."
    echo "----------------------------------------------------------------------"

    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))
        output_file="${OUTPUT_DIR}/baseline_${scenario}_seed${seed}.json"

        echo -n "[$total] ${scenario} seed ${seed}... "

        if python3 scripts/week1/run_experiment.py \
            --scenario "$scenario" \
            --init_strategy "$INIT_STRATEGY" \
            --seed "$seed" \
            --output "$output_file" 2>&1 | grep -q "Results saved"; then

            success=$((success + 1))
            echo "✓ Done"
        else
            failed=$((failed + 1))
            echo "✗ FAILED"
        fi
    done
    echo ""
done

# Summary
echo "======================================================================"
echo "Baseline Collection Complete"
echo "======================================================================"
echo "End time: $(date)"
echo "Total runs: $total"
echo "Successful: $success"
echo "Failed: $failed"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Next step: Run analysis script"
echo "  python3 scripts/week1/analyze_baseline.py"
