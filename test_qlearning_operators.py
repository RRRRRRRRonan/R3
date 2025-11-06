#!/usr/bin/env python3
"""Test Q-learning operator selection on large scale."""

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

# Test large scale with seed 2026
config = get_scale_config('large')
iterations = get_solver_iterations('large', solver='q_learning')

print("="*70)
print("Testing Q-learning on Large Scale (seed 2026)")
print("="*70)
print(f"Tasks: {config.num_tasks}")
print(f"Iterations: {iterations}")
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
    for repair, count in sorted(repair_totals.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total_actions if total_actions > 0 else 0
        print(f"  {repair:12s}: {count:5d} times ({pct:5.1f}%)")
