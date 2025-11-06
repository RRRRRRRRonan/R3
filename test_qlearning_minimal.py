#!/usr/bin/env python3
"""Minimal Q-learning diagnostic - SMALL scale, 5 iterations only."""

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

# Use SMALL scale for fastest diagnosis
config = get_scale_config('small')
iterations = 5  # Minimal iterations just to see operator selection pattern

print("="*70)
print("MINIMAL Q-learning Diagnostic (Small Scale, 5 iterations)")
print("="*70)
print(f"Tasks: {config.num_tasks}")
print(f"Iterations: {iterations} (minimal for quick pattern check)")
print("Note: This is NOT for performance testing, only for operator statistics")
print()

planner, baseline_cost, optimized_cost = run_q_learning_trial(
    config=config,
    iterations=iterations,
    seed=2026
)

improvement_ratio = (baseline_cost - optimized_cost) / baseline_cost if baseline_cost > 0 else 0

print("\n" + "="*70)
print("RESULTS (minimal run - ignore performance numbers)")
print("="*70)
print(f"Improvement: {improvement_ratio*100:.2f}% (not meaningful with only 5 iterations)")

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

    # Check initial Q-values to verify our fix worked
    print("\n" + "="*70)
    print("INITIAL Q-VALUES VERIFICATION (checking our duplicate function fix)")
    print("="*70)
    q_table = planner._q_agent.q_table
    for state in ['explore', 'stuck', 'deep_stuck']:
        if state in q_table:
            print(f"\nState: {state}")
            # Get Q-values for first destroy operator
            first_destroy = planner._q_agent.destroy_operators[0]
            for repair in planner._q_agent.repair_operators:
                action = (first_destroy, repair)
                if action in q_table[state]:
                    q_val = q_table[state][action]
                    print(f"  ({first_destroy}, {repair}): Q={q_val:.1f}")
else:
    print("\n⚠️  No Q-learning agent found!")
