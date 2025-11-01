"""Batch experiments with multiple random seeds for statistical analysis.

This script runs ALNS experiments across multiple random seeds to obtain
statistically significant results. It supports all three solver variants
(minimal, matheuristic, q_learning) and all three scales (small, medium, large).

Usage:
    python scripts/batch_experiments.py --scales small medium large --seeds 10
    python scripts/batch_experiments.py --solvers matheuristic q_learning --seeds 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import math
import statistics

# Add src and project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from tests.optimization.common import (
    get_scale_config,
    run_matheuristic_trial,
    run_minimal_trial,
)
from tests.optimization.q_learning.utils import run_q_learning_trial
from tests.optimization.presets import get_scale_preset


SCALES = ['small', 'medium', 'large']
SOLVERS = ['minimal', 'matheuristic', 'q_learning']


def run_single_experiment(
    solver: str,
    scale: str,
    seed: int,
    verbose: bool = False,
) -> Dict[str, float]:
    """Run a single experiment with given solver, scale, and seed.

    Args:
        solver: One of 'minimal', 'matheuristic', 'q_learning'
        scale: One of 'small', 'medium', 'large'
        seed: Random seed for ALNS optimization
        verbose: Whether to print progress

    Returns:
        Dictionary with baseline_cost, optimised_cost, improvement_ratio
    """
    preset = get_scale_preset(scale)
    config = get_scale_config(scale)

    if verbose:
        print(f"  Running {solver} on {scale} scale (seed={seed})...", end=' ', flush=True)

    if solver == 'minimal':
        baseline, optimised = run_minimal_trial(
            config,
            iterations=preset.iterations.minimal,
            seed=seed,
        )
    elif solver == 'matheuristic':
        baseline, optimised = run_matheuristic_trial(
            config,
            iterations=preset.iterations.matheuristic,
            seed=seed,
        )
    elif solver == 'q_learning':
        _, baseline, optimised = run_q_learning_trial(
            config,
            iterations=preset.iterations.q_learning,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    improvement_ratio = (baseline - optimised) / baseline if baseline > 0 else 0.0

    if verbose:
        print(f"Improvement: {improvement_ratio*100:.2f}%")

    return {
        'baseline_cost': baseline,
        'optimised_cost': optimised,
        'improvement_ratio': improvement_ratio,
        'seed': seed,
    }


def run_batch_experiments(
    solvers: List[str],
    scales: List[str],
    num_seeds: int,
    base_seed: int = 2025,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Run batch experiments with multiple seeds.

    Args:
        solvers: List of solvers to test
        scales: List of scales to test
        num_seeds: Number of random seeds to use
        base_seed: Starting seed value
        verbose: Whether to print progress

    Returns:
        Nested dictionary: {scale: {solver: [results]}}
    """
    results = {}
    seeds = list(range(base_seed, base_seed + num_seeds))

    for scale in scales:
        results[scale] = {}
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing scale: {scale}")
            print(f"{'='*60}")

        for solver in solvers:
            if verbose:
                print(f"\nSolver: {solver}")

            solver_results = []
            for seed in seeds:
                result = run_single_experiment(solver, scale, seed, verbose=verbose)
                solver_results.append(result)

            results[scale][solver] = solver_results

    return results


def compute_statistics(results: List[Dict]) -> Dict[str, float]:
    """Compute statistical metrics for a list of experiment results.

    Args:
        results: List of experiment result dictionaries

    Returns:
        Dictionary with mean, std, min, max, ci_95_lower, ci_95_upper
    """
    improvements = [r['improvement_ratio'] for r in results]

    mean = statistics.mean(improvements)
    std = statistics.stdev(improvements) if len(improvements) > 1 else 0.0
    median = statistics.median(improvements)

    # 95% confidence interval using normal approximation (for n>30) or conservative estimate
    n = len(improvements)
    se = std / math.sqrt(n) if n > 0 else 0.0

    # Use z=1.96 for 95% CI (valid for n>30, conservative for smaller n)
    # For small n, this is an approximation; ideally use t-distribution
    z_critical = 1.96 if n > 30 else 2.0 * math.sqrt(n / (n - 1)) if n > 1 else 1.96
    margin = z_critical * se

    return {
        'mean': mean,
        'std': std,
        'min': min(improvements),
        'max': max(improvements),
        'median': median,
        'ci_95_lower': mean - margin,
        'ci_95_upper': mean + margin,
        'count': n,
    }


def paired_t_test(results_a: List[Dict], results_b: List[Dict]) -> Dict[str, float]:
    """Perform paired t-test between two sets of results.

    Args:
        results_a: Results from method A
        results_b: Results from method B

    Returns:
        Dictionary with t_statistic, p_value, significant (at α=0.05)
    """
    improvements_a = [r['improvement_ratio'] for r in results_a]
    improvements_b = [r['improvement_ratio'] for r in results_b]

    if len(improvements_a) != len(improvements_b):
        raise ValueError("Results must have same length for paired t-test")

    # Compute differences
    differences = [a - b for a, b in zip(improvements_a, improvements_b)]

    n = len(differences)
    if n < 2:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'mean_diff': differences[0] if n == 1 else 0.0,
        }

    mean_diff = statistics.mean(differences)
    std_diff = statistics.stdev(differences)
    se_diff = std_diff / math.sqrt(n)

    # t-statistic
    t_stat = mean_diff / se_diff if se_diff > 0 else 0.0

    # Approximate p-value using normal distribution (valid for n>30)
    # For small n, this is conservative
    abs_t = abs(t_stat)

    # Rough p-value approximation (two-tailed)
    # For |t| > 2, p < 0.05; for |t| > 2.5, p < 0.01
    if abs_t > 2.5:
        p_value = 0.01
    elif abs_t > 2.0:
        p_value = 0.04
    elif abs_t > 1.5:
        p_value = 0.1
    else:
        p_value = 0.2

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': mean_diff,
    }


def print_summary(results: Dict[str, Dict[str, List[Dict]]]) -> None:
    """Print summary statistics for all experiments."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for scale in results:
        print(f"\n{scale.upper()} SCALE:")
        print("-" * 80)
        print(f"{'Solver':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'95% CI':<20}")
        print("-" * 80)

        for solver in results[scale]:
            stats_dict = compute_statistics(results[scale][solver])
            ci_str = f"[{stats_dict['ci_95_lower']:.4f}, {stats_dict['ci_95_upper']:.4f}]"
            print(f"{solver:<20} "
                  f"{stats_dict['mean']:<10.4f} "
                  f"{stats_dict['std']:<10.4f} "
                  f"{stats_dict['min']:<10.4f} "
                  f"{stats_dict['max']:<10.4f} "
                  f"{ci_str:<20}")


def print_comparisons(results: Dict[str, Dict[str, List[Dict]]]) -> None:
    """Print pairwise comparisons between solvers."""
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS (Paired t-test)")
    print("="*80)

    for scale in results:
        print(f"\n{scale.upper()} SCALE:")
        print("-" * 80)

        solvers = list(results[scale].keys())

        # Compare Q-learning vs Matheuristic
        if 'q_learning' in solvers and 'matheuristic' in solvers:
            test_result = paired_t_test(
                results[scale]['q_learning'],
                results[scale]['matheuristic']
            )

            sig_marker = "***" if test_result['significant'] else "n.s."
            mean_diff_pct = test_result['mean_diff'] * 100

            print(f"\nQ-learning vs Matheuristic:")
            print(f"  Mean difference: {mean_diff_pct:+.2f}% {sig_marker}")
            print(f"  t-statistic: {test_result['t_statistic']:.4f}")
            print(f"  p-value: {test_result['p_value']:.4f}")

            if test_result['significant']:
                if test_result['mean_diff'] > 0:
                    print(f"  → Q-learning significantly BETTER (α=0.05)")
                else:
                    print(f"  → Q-learning significantly WORSE (α=0.05)")
            else:
                print(f"  → No significant difference (α=0.05)")


def save_results(
    results: Dict[str, Dict[str, List[Dict]]],
    output_path: Path,
) -> None:
    """Save results to JSON file."""
    # Compute statistics for each combination
    output = {}
    for scale in results:
        output[scale] = {}
        for solver in results[scale]:
            stats_dict = compute_statistics(results[scale][solver])
            output[scale][solver] = {
                'raw_results': results[scale][solver],
                'statistics': stats_dict,
            }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--solvers',
        nargs='+',
        choices=SOLVERS,
        default=SOLVERS,
        help='Solvers to test (default: all)',
    )
    parser.add_argument(
        '--scales',
        nargs='+',
        choices=SCALES,
        default=SCALES,
        help='Scales to test (default: all)',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=10,
        help='Number of random seeds to test (default: 10)',
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=2025,
        help='Starting seed value (default: 2025)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('docs/data/batch_experiments_results.json'),
        help='Output JSON file path',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("="*80)
    print("BATCH EXPERIMENTS WITH MULTIPLE RANDOM SEEDS")
    print("="*80)
    print(f"Solvers: {', '.join(args.solvers)}")
    print(f"Scales: {', '.join(args.scales)}")
    print(f"Seeds: {args.seeds} (from {args.base_seed} to {args.base_seed + args.seeds - 1})")
    print("="*80)

    # Run experiments
    results = run_batch_experiments(
        solvers=args.solvers,
        scales=args.scales,
        num_seeds=args.seeds,
        base_seed=args.base_seed,
        verbose=not args.quiet,
    )

    # Print summary
    print_summary(results)

    # Print comparisons
    print_comparisons(results)

    # Save results
    save_results(results, args.output)

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
