#!/usr/bin/env python3
"""Diagnose seed variance by running multiple seeds and comparing results."""

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

# Test multiple seeds to identify variance
seeds_to_test = [2026, 42, 7, 123, 456]
scale = 'large'

config = get_scale_config(scale)
iterations = get_solver_iterations(scale, solver='q_learning')

print("="*80)
print(f"SEED VARIANCE DIAGNOSTIC ({scale} scale)")
print("="*80)
print(f"Testing seeds: {seeds_to_test}")
print(f"Tasks: {config.num_tasks}, Iterations: {iterations}")
print()

results = []

for seed in seeds_to_test:
    print(f"\n{'='*80}")
    print(f"Running seed={seed}...")
    print('='*80)

    planner, baseline_cost, optimized_cost = run_q_learning_trial(
        config=config,
        iterations=iterations,
        seed=seed
    )

    improvement_ratio = (baseline_cost - optimized_cost) / baseline_cost if baseline_cost > 0 else 0

    # Collect statistics
    if hasattr(planner, '_q_agent') and planner._q_agent is not None:
        stats = planner._q_agent.statistics()
        repair_totals = {}
        for state, state_stats in stats.items():
            for stat in state_stats:
                _, repair = stat.action
                repair_totals[repair] = repair_totals.get(repair, 0) + stat.total_usage

        total_actions = sum(repair_totals.values())
        lp_count = repair_totals.get('lp', 0)
        lp_pct = 100 * lp_count / total_actions if total_actions > 0 else 0

        # Get top Q-values
        explore_stats = stats.get('explore', [])
        top_q_action = explore_stats[0] if explore_stats else None

        results.append({
            'seed': seed,
            'baseline': baseline_cost,
            'optimized': optimized_cost,
            'improvement': improvement_ratio,
            'lp_usage_pct': lp_pct,
            'total_actions': total_actions,
            'top_q_action': top_q_action.action if top_q_action else None,
            'top_q_value': top_q_action.average_q_value if top_q_action else 0,
            'epsilon': planner._q_agent.epsilon,
        })

        print(f"\nSeed {seed} summary:")
        print(f"  Improvement: {improvement_ratio*100:.2f}%")
        print(f"  LP usage: {lp_pct:.1f}%")
        print(f"  Top Q (explore): {top_q_action.action if top_q_action else 'N/A'} = {top_q_action.average_q_value if top_q_action else 0:.1f}")
        print(f"  Final epsilon: {planner._q_agent.epsilon:.3f}")

# Summary comparison
print("\n" + "="*80)
print("VARIANCE ANALYSIS SUMMARY")
print("="*80)
print(f"{'Seed':<8} {'Improvement':<12} {'LP Usage':<12} {'Top Q Action':<25} {'Top Q Value':<12}")
print("-"*80)

for r in results:
    action_str = f"{r['top_q_action']}" if r['top_q_action'] else "N/A"
    print(f"{r['seed']:<8} {r['improvement']*100:>6.2f}%      {r['lp_usage_pct']:>6.1f}%      {action_str:<25} {r['top_q_value']:>8.1f}")

# Statistics
improvements = [r['improvement'] for r in results]
lp_usages = [r['lp_usage_pct'] for r in results]

import statistics
print("\n" + "="*80)
print("VARIANCE METRICS")
print("="*80)
print(f"Improvement: mean={statistics.mean(improvements)*100:.2f}%, std={statistics.stdev(improvements)*100:.2f}%")
print(f"LP usage: mean={statistics.mean(lp_usages):.1f}%, std={statistics.stdev(lp_usages):.1f}%")
print(f"Variance coefficient: {statistics.stdev(improvements)/statistics.mean(improvements)*100:.1f}%")

# Identify outliers
mean_imp = statistics.mean(improvements)
for r in results:
    deviation = abs(r['improvement'] - mean_imp)
    if deviation > statistics.stdev(improvements):
        print(f"\n⚠️  Seed {r['seed']} is an outlier (deviation={deviation*100:.2f}%)")
        print(f"    This seed may have an unusual task distribution")
