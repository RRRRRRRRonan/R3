#!/usr/bin/env python3
"""Run ONLY large scale Q-learning to get operator statistics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from tests.optimization.common import get_scale_config, get_solver_iterations
from tests.optimization.q_learning.utils import run_q_learning_trial

# Run large scale with full iterations (44) and seed 2026
config = get_scale_config('large')
iterations = get_solver_iterations('large', solver='q_learning')

print("="*70)
print("Large Scale Q-learning ONLY (seed 2026, full iterations)")
print("="*70)
print(f"Tasks: {config.num_tasks}")
print(f"Iterations: {iterations}")
print("This will take 5-7 minutes...")
print()

planner, baseline_cost, optimized_cost = run_q_learning_trial(
    config=config,
    iterations=iterations,
    seed=2026
)

improvement_ratio = (baseline_cost - optimized_cost) / baseline_cost if baseline_cost > 0 else 0

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Baseline cost: {baseline_cost:.2f}")
print(f"Optimized cost: {optimized_cost:.2f}")
print(f"Improvement: {improvement_ratio*100:.2f}%")

# Print Q-learning operator selection statistics
if hasattr(planner, '_q_agent') and planner._q_agent is not None:
    print("\n" + "="*70)
    print("Q-LEARNING OPERATOR STATISTICS (Large Scale)")
    print("="*70)
    print(planner._q_agent.format_statistics())

    # Print detailed usage by repair operator
    print("\n" + "="*70)
    print("REPAIR OPERATOR USAGE SUMMARY")
    print("="*70)
    stats = planner._q_agent.statistics()
    repair_totals = {}
    for state, state_stats in stats.items():
        for stat in state_stats:
            _, repair = stat.action
            repair_totals[repair] = repair_totals.get(repair, 0) + stat.total_usage

    total_actions = sum(repair_totals.values())
    print(f"Total operator selections: {total_actions}")
    for repair, count in sorted(repair_totals.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total_actions if total_actions > 0 else 0
        print(f"  {repair:12s}: {count:5d} times ({pct:5.1f}%)")

    # Detailed analysis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    lp_count = repair_totals.get('lp', 0)
    lp_pct = 100 * lp_count / total_actions if total_actions > 0 else 0

    print(f"LP usage rate: {lp_pct:.1f}%")
    if lp_pct < 10:
        print("❌ CRITICAL: LP使用率极低 (<10%)")
        print("   Q-learning几乎不选择LP operator")
    elif lp_pct < 30:
        print("⚠️  LP使用率偏低 (<30%)")
        print("   LP被选择但频率不足")
    elif lp_pct < 50:
        print("✓ LP使用率中等 (30-50%)")
    else:
        print(f"✓ LP使用率正常 (>50%: {lp_pct:.1f}%)")

    # Check initial Q-values for large scale
    print("\n" + "="*70)
    print("INITIAL Q-VALUES VERIFICATION (Large Scale)")
    print("="*70)
    q_table = planner._q_agent.q_table
    first_destroy = planner._q_agent.destroy_operators[0]

    print("Expected large scale LP bonuses: explore +9, stuck +12, deep_stuck +14")
    print("Expected LP Q-values: explore=21, stuck=27, deep_stuck=34\n")

    for state in ['explore', 'stuck', 'deep_stuck']:
        if state in q_table:
            print(f"State: {state}")
            for repair in planner._q_agent.repair_operators:
                action = (first_destroy, repair)
                if action in q_table[state]:
                    q_val = q_table[state][action]
                    print(f"  ({first_destroy}, {repair}): Q={q_val:.1f}")
else:
    print("\n⚠️  No Q-learning agent found!")
