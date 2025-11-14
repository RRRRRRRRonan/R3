"""Analyze baseline Q-learning performance across seeds and scales.

This script analyzes the results from baseline collection to understand
the current Q-learning performance and variance.

Usage:
    python scripts/week1/analyze_baseline.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_results(results_dir: str = "results/week1/baseline") -> Dict[str, List[float]]:
    """Load baseline results from JSON files.

    Args:
        results_dir: Directory containing baseline results

    Returns:
        Dictionary mapping scale to list of improvement ratios
    """
    results = {"small": [], "medium": [], "large": []}

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for file in results_path.glob("baseline_*.json"):
        with open(file) as f:
            data = json.load(f)
            scale = data["scenario"]
            improvement = data["improvement_ratio"]
            results[scale].append(improvement)

    # Verify we have all expected results
    for scale in ["small", "medium", "large"]:
        if not results[scale]:
            print(f"WARNING: No results found for {scale} scale")
        elif len(results[scale]) != 10:
            print(f"WARNING: Expected 10 seeds for {scale}, found {len(results[scale])}")

    return results


def analyze_baseline(results_dir: str = "results/week1/baseline"):
    """Analyze and print baseline performance statistics.

    Args:
        results_dir: Directory containing baseline results
    """
    print("=" * 70)
    print("Week 1 Baseline Analysis: Q-learning with Zero Initialization")
    print("=" * 70)
    print()

    # Load results
    results = load_results(results_dir)

    # Analyze each scale
    summary = {}

    for scale in ["small", "medium", "large"]:
        improvements = np.array(results[scale])

        if len(improvements) == 0:
            print(f"{scale.upper()} Scale: NO DATA")
            continue

        mean_imp = improvements.mean()
        std_imp = improvements.std()
        min_imp = improvements.min()
        max_imp = improvements.max()
        cv = std_imp / mean_imp if mean_imp > 0 else 0.0

        print(f"{scale.upper()} Scale:")
        print(f"  Sample size: {len(improvements)}")
        print(f"  Mean improvement: {mean_imp:.2%} ± {std_imp:.2%}")
        print(f"  Range: [{min_imp:.2%}, {max_imp:.2%}]")
        print(f"  Coefficient of Variation (CV): {cv:.3f}")
        print()

        summary[scale] = {
            "n": int(len(improvements)),
            "mean": float(mean_imp),
            "std": float(std_imp),
            "min": float(min_imp),
            "max": float(max_imp),
            "cv": float(cv),
        }

    # Cross-scale comparison
    print("-" * 70)
    print("Cross-scale Comparison:")
    print("-" * 70)

    if all(scale in summary for scale in ["small", "medium", "large"]):
        small_mean = summary["small"]["mean"]
        medium_mean = summary["medium"]["mean"]
        large_mean = summary["large"]["mean"]

        print(f"  Small → Medium: {(medium_mean / small_mean - 1) * 100:+.1f}% change")
        print(f"  Medium → Large: {(large_mean / medium_mean - 1) * 100:+.1f}% change")
        print(f"  Small → Large: {(large_mean / small_mean - 1) * 100:+.1f}% change")
        print()

        # Highlight the large-scale degradation issue
        if large_mean < 0.10:  # Less than 10% improvement
            print("  ⚠ ALERT: Large-scale performance is very poor (< 10%)")
            print("     This confirms the need for scale-aware improvements!")
        elif large_mean < small_mean * 0.5:
            print("  ⚠ WARNING: Large-scale performance degrades significantly")
            print(f"     ({large_mean:.1%} vs {small_mean:.1%} for small scale)")

    print()

    # Variance analysis
    print("-" * 70)
    print("Variance Analysis:")
    print("-" * 70)

    for scale in ["small", "medium", "large"]:
        if scale not in summary:
            continue

        cv = summary[scale]["cv"]
        if cv > 0.30:
            status = "HIGH (problematic)"
        elif cv > 0.20:
            status = "MODERATE"
        else:
            status = "LOW (good)"

        print(f"  {scale.capitalize():8s}: CV = {cv:.3f} [{status}]")

    print()

    # Key findings
    print("=" * 70)
    print("Key Findings:")
    print("=" * 70)

    findings = []

    # Performance degradation
    if "large" in summary and "small" in summary:
        large_mean = summary["large"]["mean"]
        small_mean = summary["small"]["mean"]

        if large_mean < small_mean * 0.3:
            findings.append(
                f"1. SEVERE performance degradation at large scale "
                f"({large_mean:.1%} vs {small_mean:.1%})"
            )
        elif large_mean < small_mean * 0.5:
            findings.append(
                f"1. Significant performance degradation at large scale "
                f"({large_mean:.1%} vs {small_mean:.1%})"
            )

    # High variance
    high_variance_scales = [
        scale for scale, data in summary.items() if data.get("cv", 0) > 0.30
    ]
    if high_variance_scales:
        findings.append(
            f"2. High seed variance detected in: {', '.join(high_variance_scales)}"
        )

    # Print findings
    if findings:
        for finding in findings:
            print(f"  {finding}")
    else:
        print("  No critical issues detected")

    print()

    # Save summary
    summary_file = Path(results_dir) / "baseline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    analyze_baseline()
