#!/usr/bin/env python3
"""
Week 5 Results Analysis: Scale-Aware Reward Normalization

Analyzes experimental results comparing OLD (ROI-aware baseline) vs. NEW (scale-aware) rewards.

Statistical tests:
- Paired t-test
- Wilcoxon signed-rank test
- Cohen's d effect size
- Reward variance analysis

Usage:
    python scripts/week5/analyze_rewards.py
    python scripts/week5/analyze_rewards.py --results-dir results/week5/reward_experiments
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_results(results_dir: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load all experimental results from JSON files.

    Returns:
        Nested dict: {reward_type: {scale: [result_dicts]}}
    """
    results = {
        "old": {"small": [], "medium": [], "large": []},
        "new": {"small": [], "medium": [], "large": []},
    }

    for json_file in results_dir.glob("reward_*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        reward_type = data["reward_type"]
        scale = data["scenario_scale"]

        if reward_type in results and scale in results[reward_type]:
            results[reward_type][scale].append(data)

    return results


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {}

    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "count": len(values),
    }


def paired_t_test_simple(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """
    Simple paired t-test implementation.

    Returns:
        (t_statistic, p_value_approx)
    """
    if len(group1) != len(group2) or len(group1) < 2:
        return 0.0, 1.0

    # Compute differences
    differences = [g2 - g1 for g1, g2 in zip(group1, group2)]

    # Mean and std of differences
    mean_diff = statistics.mean(differences)
    if len(differences) == 1:
        return 0.0, 1.0

    std_diff = statistics.stdev(differences)

    if std_diff == 0:
        return 0.0, 1.0

    # t-statistic
    n = len(differences)
    t_stat = mean_diff / (std_diff / (n ** 0.5))

    # Approximate p-value (two-tailed) using simplified lookup
    df = n - 1
    abs_t = abs(t_stat)

    # Rough p-value approximation
    if abs_t > 4.0:
        p_value = 0.001
    elif abs_t > 3.0:
        p_value = 0.01
    elif abs_t > 2.5:
        p_value = 0.02
    elif abs_t > 2.0:
        p_value = 0.05
    elif abs_t > 1.5:
        p_value = 0.15
    else:
        p_value = 0.30

    return t_stat, p_value


def wilcoxon_test_simple(group1: List[float], group2: List[float]) -> Tuple[float, str]:
    """
    Simplified Wilcoxon signed-rank test.

    Returns:
        (W_statistic, significance_level)
    """
    if len(group1) != len(group2) or len(group1) < 2:
        return 0.0, "ns"

    # Compute differences and ranks
    differences = [(g2 - g1, i) for i, (g1, g2) in enumerate(zip(group1, group2)) if g2 != g1]

    if not differences:
        return 0.0, "ns"

    # Sort by absolute difference
    differences.sort(key=lambda x: abs(x[0]))

    # Assign ranks
    ranks_pos = sum(i + 1 for i, (diff, _) in enumerate(differences) if diff > 0)
    ranks_neg = sum(i + 1 for i, (diff, _) in enumerate(differences) if diff < 0)

    W = min(ranks_pos, ranks_neg)
    n = len(differences)

    # Critical values for Wilcoxon test (approximation)
    if n >= 10:
        if W <= n * (n + 1) * 0.05:
            return W, "p<0.01"
        elif W <= n * (n + 1) * 0.10:
            return W, "p<0.05"
        elif W <= n * (n + 1) * 0.15:
            return W, "p<0.10"
        else:
            return W, "ns"
    else:
        return W, "ns (n<10)"


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    d = mean_diff / std_diff
    """
    if len(group1) != len(group2) or len(group1) < 2:
        return 0.0

    differences = [g2 - g1 for g1, g2 in zip(group1, group2)]
    mean_diff = statistics.mean(differences)

    if len(differences) == 1:
        return 0.0

    std_diff = statistics.stdev(differences)

    if std_diff == 0:
        return 0.0

    return mean_diff / std_diff


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def analyze_scale(
    old_results: List[Dict],
    new_results: List[Dict],
    scale: str,
) -> Dict:
    """
    Analyze results for a specific scale.

    Returns:
        Dictionary with statistical analysis results
    """
    # Extract improvement ratios
    old_improvements = [r["improvement_ratio"] for r in old_results]
    new_improvements = [r["improvement_ratio"] for r in new_results]

    # Descriptive statistics
    old_stats = compute_statistics(old_improvements)
    new_stats = compute_statistics(new_improvements)

    # Paired t-test
    t_stat, p_value = paired_t_test_simple(old_improvements, new_improvements)

    # Wilcoxon test
    w_stat, w_sig = wilcoxon_test_simple(old_improvements, new_improvements)

    # Cohen's d
    d = cohens_d(old_improvements, new_improvements)
    d_interp = interpret_cohens_d(d)

    # Mean difference
    mean_diff = new_stats["mean"] - old_stats["mean"]

    return {
        "scale": scale,
        "old_stats": old_stats,
        "new_stats": new_stats,
        "mean_difference": mean_diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "wilcoxon_W": w_stat,
        "wilcoxon_sig": w_sig,
        "cohens_d": d,
        "cohens_d_interpretation": d_interp,
    }


def checkpoint_2_decision(large_scale_analysis: Dict) -> Tuple[str, str]:
    """
    Apply Checkpoint 2 decision criteria.

    Success criteria:
    - Large-scale improvement ≥8%
    - p < 0.05
    - Cohen's d > 0.5

    Returns:
        (decision, reason)
    """
    mean_diff = large_scale_analysis["mean_difference"]
    p_value = large_scale_analysis["p_value"]
    cohens_d = large_scale_analysis["cohens_d"]

    if mean_diff >= 8.0 and p_value < 0.05 and cohens_d > 0.5:
        return "✅ FULL SUCCESS", "All criteria met: Adopt scale-aware rewards"
    elif mean_diff >= 5.0 and p_value < 0.10 and cohens_d > 0.3:
        return "⚠️ PARTIAL SUCCESS", "Moderate improvement: Consider adoption with tuning"
    elif mean_diff >= 3.0:
        return "⚠️ MARGINAL", "Weak improvement: Investigate further"
    else:
        return "❌ FAILURE", "Insufficient improvement: Major pivot needed"


def print_summary(analysis_results: Dict[str, Dict]) -> None:
    """Print formatted summary of analysis results."""

    print("\n" + "="*80)
    print("WEEK 5 EXPERIMENTAL RESULTS ANALYSIS")
    print("Scale-Aware Reward Normalization")
    print("="*80)

    for scale in ["small", "medium", "large"]:
        result = analysis_results[scale]

        print(f"\n{'='*80}")
        print(f"SCALE: {scale.upper()}")
        print(f"{'='*80}")

        print(f"\nOLD (ROI-aware baseline):")
        print(f"  Mean: {result['old_stats']['mean']:.2f}% ± {result['old_stats']['std']:.2f}%")
        print(f"  Range: [{result['old_stats']['min']:.2f}%, {result['old_stats']['max']:.2f}%]")

        print(f"\nNEW (Scale-aware):")
        print(f"  Mean: {result['new_stats']['mean']:.2f}% ± {result['new_stats']['std']:.2f}%")
        print(f"  Range: [{result['new_stats']['min']:.2f}%, {result['new_stats']['max']:.2f}%]")

        print(f"\nComparison:")
        print(f"  Mean Difference: {result['mean_difference']:+.2f}%")
        print(f"  Paired t-test: t={result['t_statistic']:.3f}, p={result['p_value']:.3f}")
        print(f"  Wilcoxon: W={result['wilcoxon_W']:.1f}, {result['wilcoxon_sig']}")
        print(f"  Cohen's d: {result['cohens_d']:.3f} ({result['cohens_d_interpretation']})")

    # Checkpoint 2 decision
    print(f"\n{'='*80}")
    print("CHECKPOINT 2 DECISION (Large Scale)")
    print(f"{'='*80}")

    decision, reason = checkpoint_2_decision(analysis_results["large"])
    print(f"\nDecision: {decision}")
    print(f"Reason: {reason}")

    large = analysis_results["large"]
    print(f"\nCriteria Assessment:")
    print(f"  1. Large-scale improvement ≥8%: {large['mean_difference']:.2f}% "
          f"{'✅' if large['mean_difference'] >= 8.0 else '❌'}")
    print(f"  2. Statistical significance p<0.05: {large['p_value']:.3f} "
          f"{'✅' if large['p_value'] < 0.05 else '❌'}")
    print(f"  3. Cohen's d > 0.5: {large['cohens_d']:.3f} "
          f"{'✅' if large['cohens_d'] > 0.5 else '❌'}")

    print(f"\n{'='*80}\n")


def save_summary(analysis_results: Dict[str, Dict], output_file: Path) -> None:
    """Save analysis summary to text file."""

    with open(output_file, 'w') as f:
        f.write("Week 5 Experimental Results Analysis\n")
        f.write("="*80 + "\n\n")

        for scale in ["small", "medium", "large"]:
            result = analysis_results[scale]

            f.write(f"\n{'='*80}\n")
            f.write(f"SCALE: {scale.upper()}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"OLD (ROI-aware baseline):\n")
            f.write(f"  Mean: {result['old_stats']['mean']:.2f}% ± {result['old_stats']['std']:.2f}%\n")
            f.write(f"  Range: [{result['old_stats']['min']:.2f}%, {result['old_stats']['max']:.2f}%]\n\n")

            f.write(f"NEW (Scale-aware):\n")
            f.write(f"  Mean: {result['new_stats']['mean']:.2f}% ± {result['new_stats']['std']:.2f}%\n")
            f.write(f"  Range: [{result['new_stats']['min']:.2f}%, {result['new_stats']['max']:.2f}%]\n\n")

            f.write(f"Comparison:\n")
            f.write(f"  Mean Difference: {result['mean_difference']:+.2f}%\n")
            f.write(f"  Paired t-test: t={result['t_statistic']:.3f}, p={result['p_value']:.3f}\n")
            f.write(f"  Wilcoxon: W={result['wilcoxon_W']:.1f}, {result['wilcoxon_sig']}\n")
            f.write(f"  Cohen's d: {result['cohens_d']:.3f} ({result['cohens_d_interpretation']})\n")

        # Checkpoint 2 decision
        f.write(f"\n{'='*80}\n")
        f.write("CHECKPOINT 2 DECISION (Large Scale)\n")
        f.write(f"{'='*80}\n\n")

        decision, reason = checkpoint_2_decision(analysis_results["large"])
        f.write(f"Decision: {decision}\n")
        f.write(f"Reason: {reason}\n\n")

        large = analysis_results["large"]
        f.write(f"Criteria Assessment:\n")
        f.write(f"  1. Large-scale improvement ≥8%: {large['mean_difference']:.2f}% "
                f"{'✅' if large['mean_difference'] >= 8.0 else '❌'}\n")
        f.write(f"  2. Statistical significance p<0.05: {large['p_value']:.3f} "
                f"{'✅' if large['p_value'] < 0.05 else '❌'}\n")
        f.write(f"  3. Cohen's d > 0.5: {large['cohens_d']:.3f} "
                f"{'✅' if large['cohens_d'] > 0.5 else '❌'}\n")


def main():
    parser = argparse.ArgumentParser(description="Week 5 Results Analysis")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/week5/reward_experiments"),
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/week5"),
        help="Directory for analysis outputs"
    )

    args = parser.parse_args()

    # Check if results directory exists
    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}", file=sys.stderr)
        return 1

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)

    # Verify data completeness
    total_files = sum(len(results[reward][scale]) for reward in results for scale in results[reward])
    print(f"Loaded {total_files} result files")

    expected = 60  # 2 reward types × 3 scales × 10 seeds
    if total_files < expected:
        print(f"WARNING: Expected {expected} files, found {total_files}")

    # Analyze each scale
    analysis_results = {}
    for scale in ["small", "medium", "large"]:
        old_results = results["old"][scale]
        new_results = results["new"][scale]

        if len(old_results) == 0 or len(new_results) == 0:
            print(f"WARNING: No results for {scale} scale", file=sys.stderr)
            continue

        analysis_results[scale] = analyze_scale(old_results, new_results, scale)

    # Print summary
    print_summary(analysis_results)

    # Save summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir / "analysis_summary.txt"
    save_summary(analysis_results, summary_file)
    print(f"Analysis summary saved to: {summary_file}")

    # Save detailed results as JSON
    json_file = args.output_dir / "analysis_results.json"
    with open(json_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
