"""Week 2 Analysis: Epsilon Strategy Statistical Comparison

Analyzes epsilon strategy experiments and compares against CURRENT baseline.

Usage:
    python scripts/week2/analyze_epsilon.py
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_results(results_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Load all experiment results.

    Returns:
        Dict[strategy][scale] -> List[improvement_ratios]
    """
    data = defaultdict(lambda: defaultdict(list))

    for json_file in results_dir.glob("epsilon_*.json"):
        with open(json_file) as f:
            result = json.load(f)

        strategy = result["epsilon_strategy"]
        scale = result["scenario"]
        improvement_ratio = result["improvement_ratio"]

        data[strategy][scale].append(improvement_ratio)

    return data


def mean(values: List[float]) -> float:
    """Calculate mean."""
    return sum(values) / len(values) if values else 0.0


def variance(values: List[float]) -> float:
    """Calculate sample variance."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def std(values: List[float]) -> float:
    """Calculate standard deviation."""
    return math.sqrt(variance(values))


def wilcoxon_test_simple(baseline: List[float], strategy: List[float]) -> Tuple[float, float]:
    """Simplified Wilcoxon signed-rank test.

    Returns:
        (p_value, W_statistic)
    """
    if len(baseline) != len(strategy):
        raise ValueError("Arrays must have same length")

    differences = [s - b for s, b in zip(strategy, baseline)]
    non_zero_diffs = [(abs(d), 1 if d > 0 else -1) for d in differences if abs(d) > 1e-10]

    if not non_zero_diffs:
        return 1.0, 0.0

    # Rank by absolute value
    ranked = sorted(enumerate(non_zero_diffs, 1), key=lambda x: x[1][0])

    W_plus = sum(rank for rank, (_, sign) in ranked if sign > 0)
    W_minus = sum(rank for rank, (_, sign) in ranked if sign < 0)

    W = min(W_plus, W_minus)
    n = len(ranked)

    # Normal approximation
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if sigma == 0:
        return 1.0, W

    z = (W - mu) / sigma

    # Two-tailed p-value (approximate)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return p_value, W


def cohens_d(baseline: List[float], strategy: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    mean_diff = mean(strategy) - mean(baseline)

    n1, n2 = len(baseline), len(strategy)
    var1, var2 = variance(baseline), variance(strategy)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return mean_diff / pooled_std


def analyze_experiments():
    """Main analysis function."""
    results_dir = Path("results/week2/epsilon_experiments")

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        print("Please run experiments first!")
        return

    print("=" * 80)
    print("Week 2: Epsilon Strategy Statistical Analysis")
    print("=" * 80)
    print()

    # Load data
    data = load_results(results_dir)

    # Data completeness check
    print("Data Completeness Check:")
    for strategy in ["current", "scale_adaptive", "high_uniform"]:
        for scale in ["small", "medium", "large"]:
            n = len(data[strategy][scale])
            status = "✓" if n == 10 else "✗"
            print(f"  {strategy:20} {scale:10}: {n}/10 {status}")
    print()

    # Descriptive statistics
    print("=" * 80)
    print("Descriptive Statistics (Improvement Ratio)")
    print("=" * 80)
    print()

    print(f"{'Strategy':<20} {'Small':<20} {'Medium':<20} {'Large':<20}")
    print("-" * 80)

    for strategy in ["current", "scale_adaptive", "high_uniform"]:
        row = f"{strategy:<20} "
        for scale in ["small", "medium", "large"]:
            values = data[strategy][scale]
            if values:
                m = mean(values)
                s = std(values)
                row += f"{m*100:.2f}% ± {s*100:.2f}%{' ':<8}"
            else:
                row += f"{'N/A':<20}"
        print(row)

    print()

    # Statistical comparison vs. CURRENT
    print("=" * 80)
    print("Comparison vs. CURRENT Baseline")
    print("=" * 80)
    print()

    baseline_strategy = "current"

    for scale in ["small", "medium", "large"]:
        print(f"\n{scale.upper()} Scenario:")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Mean Δ':<15} {'Wilcoxon':<20} {'Cohen\\'s d':<15} {'Effect':<10}")
        print("-" * 80)

        baseline_values = data[baseline_strategy][scale]

        for strategy in ["scale_adaptive", "high_uniform"]:
            strategy_values = data[strategy][scale]

            if not baseline_values or not strategy_values:
                print(f"{strategy:<20} {'N/A':<15} {'N/A':<20} {'N/A':<15} {'N/A':<10}")
                continue

            mean_diff = (mean(strategy_values) - mean(baseline_values)) * 100

            # Wilcoxon test
            p_value, _ = wilcoxon_test_simple(baseline_values, strategy_values)

            if p_value < 0.01:
                sig_str = "*** (p<0.01)"
            elif p_value < 0.05:
                sig_str = "** (p<0.05)"
            elif p_value < 0.10:
                sig_str = "* (p<0.10)"
            else:
                sig_str = "ns (p>0.10)"

            # Cohen's d
            d = cohens_d(baseline_values, strategy_values)

            if abs(d) < 0.2:
                effect = "Negligible"
            elif abs(d) < 0.5:
                effect = "Small"
            elif abs(d) < 0.8:
                effect = "Medium"
            else:
                effect = "Large"

            sign = "+" if mean_diff >= 0 else ""
            print(f"{strategy:<20} {sign}{mean_diff:>5.2f}%{' ':<9} {sig_str:<20} {d:>+6.3f}{' ':<8} {effect:<10}")

    print()
    print("=" * 80)
    print("Decision Recommendation")
    print("=" * 80)
    print()

    # Checkpoint 1 decision logic
    for scale in ["small", "medium", "large"]:
        baseline = data[baseline_strategy][scale]
        scale_adaptive = data["scale_adaptive"][scale]

        if not baseline or not scale_adaptive:
            continue

        mean_diff = (mean(scale_adaptive) - mean(baseline)) * 100
        p_value, _ = wilcoxon_test_simple(baseline, scale_adaptive)
        d = cohens_d(baseline, scale_adaptive)

        print(f"{scale.upper()}:")
        print(f"  SCALE_ADAPTIVE vs CURRENT: {mean_diff:+.2f}% (p={p_value:.3f}, d={d:+.3f})")

        if scale == "large":
            if mean_diff >= 5.0 and p_value < 0.05 and d > 0.3:
                print(f"  ✅ SUCCESS: Large-scale improvement ≥5%, significant, moderate effect")
                print(f"  → Adopt SCALE_ADAPTIVE epsilon")
                print(f"  → Proceed to Week 5 (Reward Normalization)")
            elif mean_diff >= 3.0 and p_value < 0.10:
                print(f"  ⚠️  PARTIAL: Modest improvement, marginally significant")
                print(f"  → Consider adoption with caution")
                print(f"  → Proceed to Week 5, but epsilon may not be key factor")
            else:
                print(f"  ❌ NO IMPROVEMENT: Epsilon not the bottleneck")
                print(f"  → Skip adaptive epsilon")
                print(f"  → Proceed directly to Week 5 (Reward Normalization)")
        print()

    # Save summary
    summary_path = Path("results/week2/analysis_summary.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Week 2 Epsilon Strategy Analysis Summary\n")
        f.write("=" * 80 + "\n\n")

        for scale in ["small", "medium", "large"]:
            f.write(f"{scale.upper()} Scenario:\n")
            f.write("-" * 40 + "\n")

            baseline = data[baseline_strategy][scale]
            f.write(f"CURRENT (baseline): {mean(baseline)*100:.2f}%\n")

            for strategy in ["scale_adaptive", "high_uniform"]:
                values = data[strategy][scale]
                mean_diff = (mean(values) - mean(baseline)) * 100
                p_value, _ = wilcoxon_test_simple(baseline, values)
                d = cohens_d(baseline, values)

                sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else "ns"

                f.write(f"{strategy}: {mean(values)*100:.2f}% ({mean_diff:+.2f}%), "
                       f"{sig} (p={p_value:.3f}), d={d:+.3f}\n")

            f.write("\n")

    print(f"Analysis summary saved to: {summary_path}")
    print()


if __name__ == "__main__":
    analyze_experiments()
