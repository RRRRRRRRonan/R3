"""Compare Q-learning parameter tuning results before and after.

This script analyzes the impact of parameter tuning on Q-learning performance
across all 10 seeds (2025-2034) and three scales (small, medium, large).

Usage:
    python scripts/compare_tuning_results.py

The script expects JSON result files in the results/ directory following
the naming convention: seed_XXXX_scale_YYYY.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_results(seed: int, scale: str, branch: str = "main") -> Dict:
    """Load result JSON for a specific seed and scale.

    Args:
        seed: Seed number (2025-2034)
        scale: Problem scale ('small', 'medium', 'large')
        branch: Git branch name ('main' or 'tuned')

    Returns:
        Dictionary containing results, or None if file not found
    """
    results_dir = project_root / "results"
    filename = f"seed_{seed}_scale_{scale}.json"
    filepath = results_dir / filename

    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    return data


def extract_improvements(results: Dict) -> Dict[str, float]:
    """Extract improvement percentages for each solver.

    Args:
        results: Results dictionary from JSON

    Returns:
        Dictionary mapping solver name to improvement percentage
    """
    improvements = {}

    if "solvers" not in results:
        return improvements

    for solver_name, solver_data in results["solvers"].items():
        if "improvement" in solver_data:
            # Improvement is stored as a ratio (0.0 to 1.0)
            improvements[solver_name] = solver_data["improvement"] * 100.0
        elif "final_cost" in solver_data and "initial_cost" in solver_data:
            # Calculate improvement
            initial = solver_data["initial_cost"]
            final = solver_data["final_cost"]
            if initial > 0:
                improvements[solver_name] = ((initial - final) / initial) * 100.0

    return improvements


def compare_seed_results(seed: int) -> Dict:
    """Compare results for a specific seed across all scales.

    Args:
        seed: Seed number

    Returns:
        Dictionary with comparison data
    """
    comparison = {
        "seed": seed,
        "scales": {},
    }

    for scale in ["small", "medium", "large"]:
        results = load_results(seed, scale)

        if results is None:
            continue

        improvements = extract_improvements(results)

        comparison["scales"][scale] = {
            "q_learning": improvements.get("q_learning", 0.0),
            "matheuristic": improvements.get("matheuristic", 0.0),
            "difference": improvements.get("q_learning", 0.0)
            - improvements.get("matheuristic", 0.0),
        }

    return comparison


def analyze_all_seeds() -> None:
    """Analyze and compare results for all 10 seeds."""

    print("=" * 80)
    print("Q-LEARNING PARAMETER TUNING ANALYSIS")
    print("=" * 80)
    print()

    all_comparisons = []
    q_learning_improvements = []
    matheuristic_improvements = []
    differences = []

    # Collect data for all seeds
    for seed in range(2025, 2035):
        comparison = compare_seed_results(seed)

        if not comparison["scales"]:
            print(f"⚠️  Warning: No results found for seed {seed}")
            continue

        all_comparisons.append(comparison)

        # Collect all improvements
        for scale, data in comparison["scales"].items():
            q_learning_improvements.append(data["q_learning"])
            matheuristic_improvements.append(data["matheuristic"])
            differences.append(data["difference"])

    if not all_comparisons:
        print("❌ No results found! Please run experiments first.")
        return

    # Print detailed results
    print("\n📊 DETAILED RESULTS BY SEED AND SCALE")
    print("-" * 80)
    print(
        f"{'Seed':<6} {'Scale':<8} {'Q-learning':<12} {'Matheuristic':<14} {'Difference':<12}"
    )
    print("-" * 80)

    for comp in all_comparisons:
        seed = comp["seed"]
        for scale in ["small", "medium", "large"]:
            if scale not in comp["scales"]:
                continue

            data = comp["scales"][scale]
            q_imp = data["q_learning"]
            m_imp = data["matheuristic"]
            diff = data["difference"]

            # Mark problematic cases
            marker = ""
            if diff < -5:
                marker = " ❌ WORSE"
            elif diff < 0:
                marker = " ⚠️"
            elif diff > 5:
                marker = " ✅ BETTER"

            print(
                f"{seed:<6} {scale:<8} {q_imp:>10.2f}%  {m_imp:>12.2f}%  {diff:>10.2f}%{marker}"
            )

    # Statistical summary
    print("\n" + "=" * 80)
    print("📈 STATISTICAL SUMMARY")
    print("=" * 80)

    if q_learning_improvements:
        print(f"\nQ-learning Performance:")
        print(f"  Mean:         {statistics.mean(q_learning_improvements):>8.2f}%")
        print(f"  Std Dev:      {statistics.stdev(q_learning_improvements):>8.2f}%")
        print(f"  Min:          {min(q_learning_improvements):>8.2f}%")
        print(f"  Max:          {max(q_learning_improvements):>8.2f}%")

    if matheuristic_improvements:
        print(f"\nMatheuristic Performance:")
        print(f"  Mean:         {statistics.mean(matheuristic_improvements):>8.2f}%")
        print(f"  Std Dev:      {statistics.stdev(matheuristic_improvements):>8.2f}%")
        print(f"  Min:          {min(matheuristic_improvements):>8.2f}%")
        print(f"  Max:          {max(matheuristic_improvements):>8.2f}%")

    if differences:
        print(f"\nDifference (Q-learning - Matheuristic):")
        print(f"  Mean:         {statistics.mean(differences):>8.2f}%")
        print(f"  Std Dev:      {statistics.stdev(differences):>8.2f}%")
        print(f"  Min:          {min(differences):>8.2f}%")
        print(f"  Max:          {max(differences):>8.2f}%")

        # Win/loss statistics
        wins = sum(1 for d in differences if d > 0)
        losses = sum(1 for d in differences if d < 0)
        ties = sum(1 for d in differences if d == 0)

        print(f"\nWin/Loss Record:")
        print(f"  Q-learning wins:    {wins:>3} ({wins/len(differences)*100:.1f}%)")
        print(f"  Matheuristic wins:  {losses:>3} ({losses/len(differences)*100:.1f}%)")
        print(f"  Ties:               {ties:>3} ({ties/len(differences)*100:.1f}%)")

    # Identify problem cases
    print("\n" + "=" * 80)
    print("⚠️  PROBLEM CASES (Q-learning underperforms by >5%)")
    print("=" * 80)

    problem_cases = []
    for comp in all_comparisons:
        seed = comp["seed"]
        for scale, data in comp["scales"].items():
            if data["difference"] < -5:
                problem_cases.append((seed, scale, data["difference"]))

    if problem_cases:
        problem_cases.sort(key=lambda x: x[2])  # Sort by difference
        for seed, scale, diff in problem_cases:
            print(f"  Seed {seed} ({scale}): {diff:>7.2f}%")
    else:
        print("  None! 🎉")

    # Success cases
    print("\n" + "=" * 80)
    print("✅ SUCCESS CASES (Q-learning outperforms by >5%)")
    print("=" * 80)

    success_cases = []
    for comp in all_comparisons:
        seed = comp["seed"]
        for scale, data in comp["scales"].items():
            if data["difference"] > 5:
                success_cases.append((seed, scale, data["difference"]))

    if success_cases:
        success_cases.sort(key=lambda x: x[2], reverse=True)  # Sort by difference
        for seed, scale, diff in success_cases:
            print(f"  Seed {seed} ({scale}): +{diff:>6.2f}%")
    else:
        print("  None.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_all_seeds()
