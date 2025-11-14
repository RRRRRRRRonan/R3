#!/bin/bash
# Week 1, Day 4-7: Q-table Initialization Experiments
#
# This script tests 4 different initialization strategies:
#   - zero: Current baseline (all Q-values = 0.0)
#   - uniform: Uniform positive bias (all Q-values = 50.0)
#   - action_specific: Higher bias for matheuristic operators
#   - state_specific: Higher bias for stuck states
#
# Expected runtime: ~2 hours (4 strategies × 3 scales × 10 seeds = 120 runs)
# Expected output: 120 JSON files

set -e  # Exit on error

# Configuration
SEEDS=(2025 2026 2027 2028 2029 2030 2031 2032 2033 2034)
SCENARIOS=("small" "medium" "large")
STRATEGIES=("zero" "uniform" "action_specific" "state_specific")
OUTPUT_DIR="results/week1/init_experiments"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print header
echo "======================================================================"
echo "Week 1: Q-table Initialization Experiments"
echo "======================================================================"
echo "Start time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Strategies: ${STRATEGIES[*]}"
echo "Total runs: $((${#STRATEGIES[@]} * ${#SCENARIOS[@]} * ${#SEEDS[@]}))"
echo ""

# Run experiments
total=0
success=0
failed=0

for strategy in "${STRATEGIES[@]}"; do
    echo "======================================================================"
    echo "Strategy: $strategy"
    echo "======================================================================"

    for scenario in "${SCENARIOS[@]}"; do
        echo "----------------------------------------------------------------------"
        echo "Running ${scenario} scenario with ${strategy} initialization..."
        echo "----------------------------------------------------------------------"

        for seed in "${SEEDS[@]}"; do
            total=$((total + 1))
            output_file="${OUTPUT_DIR}/init_${strategy}_${scenario}_seed${seed}.json"

            echo -n "[$total] ${strategy}/${scenario}/seed${seed}... "

            if python scripts/week1/run_experiment.py \
                --scenario "$scenario" \
                --init_strategy "$strategy" \
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
done

# Summary
echo "======================================================================"
echo "Initialization Experiments Complete"
echo "======================================================================"
echo "End time: $(date)"
echo "Total runs: $total"
echo "Successful: $success"
echo "Failed: $failed"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Next step: Run analysis script"
echo "  python scripts/week1/analyze_init_strategies.py"
