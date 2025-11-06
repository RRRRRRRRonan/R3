#!/usr/bin/env python3
"""Quick Q-learning operator selection diagnostic (reduced iterations)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from tests.optimization.common import get_scale_config
from tests.optimization.q_learning.utils import run_q_learning_trial

# Test large scale with seed 2026, but REDUCED iterations for quick diagnosis
config = get_scale_config('large')
iterations = 15  # Reduced from 44 to get quick operator statistics

print("="*70)
print("Quick Q-learning Diagnostic (Large Scale, seed 2026)")
print("="*70)
print(f"Tasks: {config.num_tasks}")
print(f"Iterations: {iterations} (reduced for quick diagnosis)")
print()

planner, baseline_cost, optimized_cost = run_q_learning_trial(
    config=config,
    iterations=iterations,
    seed=2026
)

improvement_ratio = (baseline_cost - optimized_cost) / baseline_cost if baseline_cost > 0 else 0

print("\n" + "="*70)
print("RESULTS (Note: reduced iterations, not final performance)")
print("="*70)
print(f"Baseline cost: {baseline_cost:.2f}")
print(f"Optimized cost: {optimized_cost:.2f}")
print(f"Improvement: {improvement_ratio*100:.2f}%")

# Print Q-learning operator selection statistics
if hasattr(planner, '_q_agent') and planner._q_agent is not None:
    print("\n" + "="*70)
    print("Q-LEARNING OPERATOR STATISTICS")
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

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    lp_count = repair_totals.get('lp', 0)
    lp_pct = 100 * lp_count / total_actions if total_actions > 0 else 0

    if lp_pct < 10:
        print("⚠️  LP使用率很低 (<10%)!")
        print("   问题: Q-learning没有学会选择LP operator")
        print("   可能原因: reward计算问题、状态判断错误、或Q值更新异常")
    elif lp_pct < 30:
        print("⚠️  LP使用率偏低 (<30%)")
        print("   LP被使用但不够频繁，可能需要调整initial Q-values")
    else:
        print(f"✓ LP使用率正常 ({lp_pct:.1f}%)")
        print("  如果结果仍然差，问题可能在reward计算或状态转换")
