"""Analyze and compare Q-table initialization strategies.

This script compares the performance of different initialization strategies
and provides statistical analysis including hypothesis testing.

Usage:
    python scripts/week1/analyze_init_strategies.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_results(
    results_dir: str = "results/week1/init_experiments",
) -> pd.DataFrame:
    """Load all initialization experiment results into a DataFrame.

    Args:
        results_dir: Directory containing experiment results

    Returns:
        DataFrame with columns: strategy, scenario, seed, improvement, runtime
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    data = []
    for file in results_path.glob("init_*.json"):
        with open(file) as f:
            result = json.load(f)
            data.append({
                "strategy": result["init_strategy"],
                "scenario": result["scenario"],
                "seed": result["seed"],
                "improvement": result["improvement_ratio"],
                "runtime": result["runtime"],
                "final_epsilon": result.get("final_epsilon", 0.0),
            })

    if not data:
        raise ValueError(f"No experiment results found in {results_dir}")

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} experiment results")
    print(f"  Strategies: {df['strategy'].unique()}")
    print(f"  Scenarios: {df['scenario'].unique()}")
    print(f"  Seeds per config: {df.groupby(['strategy', 'scenario']).size().iloc[0]}")
    print()

    return df


def statistical_comparison(
    baseline_data: np.ndarray,
    strategy_data: np.ndarray,
    strategy_name: str,
) -> Dict:
    """Perform statistical comparison between baseline and strategy.

    Args:
        baseline_data: Improvement ratios for baseline (zero init)
        strategy_data: Improvement ratios for strategy
        strategy_name: Name of the strategy

    Returns:
        Dictionary with statistical test results
    """
    # Paired Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(baseline_data, strategy_data)

    # Effect size (Cohen's d)
    mean_diff = strategy_data.mean() - baseline_data.mean()
    pooled_std = np.sqrt((baseline_data.std()**2 + strategy_data.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    # Significance
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"

    return {
        "strategy": strategy_name,
        "mean_diff": mean_diff,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size": effect_size,
        "significance": significance,
    }


def analyze_by_scale(df: pd.DataFrame, results_dir: str):
    """Analyze initialization strategies by scale.

    Args:
        df: DataFrame with experiment results
        results_dir: Directory to save analysis results
    """
    print("=" * 80)
    print("Initialization Strategy Analysis by Scale")
    print("=" * 80)
    print()

    strategies = ["zero", "uniform", "action_specific", "state_specific"]
    strategy_labels = {
        "zero": "Zero (Baseline)",
        "uniform": "Uniform(50.0)",
        "action_specific": "Action-Specific",
        "state_specific": "State-Specific",
    }

    statistical_results = []

    for scenario in ["small", "medium", "large"]:
        print(f"{'='*80}")
        print(f"{scenario.upper()} Scale")
        print(f"{'='*80}")
        print()

        scenario_df = df[df["scenario"] == scenario]

        # Summary statistics
        print("Summary Statistics:")
        print("-" * 80)

        summary = scenario_df.groupby("strategy")["improvement"].agg([
            ("Mean", "mean"),
            ("Std", "std"),
            ("Min", "min"),
            ("Max", "max"),
            ("CV", lambda x: x.std() / x.mean()),
        ])

        # Reorder rows
        summary = summary.reindex(strategies)
        print(summary.to_string())
        print()

        # Statistical comparisons with baseline (zero)
        print("Statistical Comparisons with Baseline (Zero):")
        print("-" * 80)

        zero_data = scenario_df[scenario_df["strategy"] == "zero"]["improvement"].values

        for strategy in ["uniform", "action_specific", "state_specific"]:
            strategy_data = scenario_df[scenario_df["strategy"] == strategy]["improvement"].values

            if len(strategy_data) == 0 or len(zero_data) == 0:
                print(f"{strategy_labels[strategy]}: NO DATA")
                continue

            result = statistical_comparison(zero_data, strategy_data, strategy)

            print(f"\n{strategy_labels[strategy]}:")
            print(f"  Mean difference: {result['mean_diff']:+.2%}")
            print(f"  p-value: {result['p_value']:.4f} {result['significance']}")
            print(f"  Cohen's d: {result['cohens_d']:.3f} ({result['effect_size']} effect)")

            # Add scale to result
            result["scale"] = scenario
            statistical_results.append(result)

        print("\n")

    # Save statistical results
    stats_df = pd.DataFrame(statistical_results)
    stats_file = Path(results_dir) / "statistical_comparison.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"Statistical results saved to: {stats_file}")
    print()

    return stats_df


def create_visualizations(df: pd.DataFrame, results_dir: str):
    """Create visualization plots.

    Args:
        df: DataFrame with experiment results
        results_dir: Directory to save plots
    """
    print("Creating visualizations...")

    strategies = ["zero", "uniform", "action_specific", "state_specific"]
    strategy_labels = {
        "zero": "Zero",
        "uniform": "Uniform",
        "action_specific": "Action-Spec",
        "state_specific": "State-Spec",
    }

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 5)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, scenario in enumerate(["small", "medium", "large"]):
        scenario_df = df[df["scenario"] == scenario]

        # Prepare data for boxplot
        plot_data = []
        plot_labels = []

        for strategy in strategies:
            strategy_data = scenario_df[scenario_df["strategy"] == strategy]["improvement"]
            if len(strategy_data) > 0:
                plot_data.append(strategy_data.values)
                plot_labels.append(strategy_labels[strategy])

        # Create boxplot
        bp = axes[idx].boxplot(
            plot_data,
            labels=plot_labels,
            patch_artist=True,
            showmeans=True,
        )

        # Color boxes
        colors = ['lightgray', 'lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
            patch.set_facecolor(color)

        axes[idx].set_title(f"{scenario.capitalize()} Scale", fontsize=14, fontweight='bold')
        axes[idx].set_ylabel("Improvement Ratio", fontsize=12)
        axes[idx].set_xlabel("Initialization Strategy", fontsize=12)
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=15)

        # Add horizontal line at current baseline mean
        zero_mean = scenario_df[scenario_df["strategy"] == "zero"]["improvement"].mean()
        axes[idx].axhline(y=zero_mean, color='red', linestyle='--', alpha=0.5, label='Zero mean')

    plt.tight_layout()

    # Save figure
    plot_file = Path(results_dir) / "init_strategies_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_file}")
    plt.close()


def recommend_strategy(stats_df: pd.DataFrame) -> Dict[str, str]:
    """Recommend best initialization strategy for each scale.

    Args:
        stats_df: DataFrame with statistical comparison results

    Returns:
        Dictionary mapping scale to recommended strategy
    """
    print("=" * 80)
    print("Recommended Initialization Strategies")
    print("=" * 80)
    print()

    recommendations = {}

    for scale in ["small", "medium", "large"]:
        scale_stats = stats_df[stats_df["scale"] == scale]

        # Filter for statistically significant improvements
        significant = scale_stats[scale_stats["p_value"] < 0.05]

        if len(significant) == 0:
            recommendations[scale] = "zero (no significant improvement found)"
            print(f"{scale.capitalize()}: {recommendations[scale]}")
            continue

        # Select strategy with largest positive effect
        best = significant.loc[significant["mean_diff"].idxmax()]
        recommendations[scale] = best["strategy"]

        print(f"{scale.capitalize()}: {best['strategy']}")
        print(f"  Mean improvement over zero: {best['mean_diff']:+.2%}")
        print(f"  Effect size: {best['effect_size']} (d={best['cohens_d']:.3f})")
        print(f"  p-value: {best['p_value']:.4f} {best['significance']}")
        print()

    return recommendations


def main():
    """Main analysis function."""
    results_dir = "results/week1/init_experiments"

    # Load data
    df = load_experiment_results(results_dir)

    # Analyze by scale
    stats_df = analyze_by_scale(df, results_dir)

    # Create visualizations
    create_visualizations(df, results_dir)

    # Recommendations
    recommendations = recommend_strategy(stats_df)

    # Save recommendations
    rec_file = Path(results_dir) / "recommendations.json"
    with open(rec_file, "w") as f:
        json.dump(recommendations, f, indent=2)

    print(f"Recommendations saved to: {rec_file}")
    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
