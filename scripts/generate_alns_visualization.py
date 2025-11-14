"""Generate unified ALNS regression summaries and visualisations.

This utility reuses the optimisation test presets to collect deterministic
baseline/optimised costs for the Minimal ALNS, Matheuristic ALNS, and
Matheuristic + Q-learning variants across the small/medium/large scenarios.
It saves the aggregated metrics to JSON, prints a Markdown table for inclusion
in documentation, and renders a grouped bar chart that highlights the relative
cost improvements per solver.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from strategy.charging_strategies import PartialRechargeMinimalStrategy
from tests.optimization.common import (
    build_scenario,
    get_scale_config,
    get_solver_iterations,
    run_matheuristic_trial,
    run_minimal_trial,
)
from tests.optimization.q_learning.utils import run_q_learning_trial


@dataclass
class SolverResult:
    """Container for a single solver run."""

    solver: str
    scale: str
    baseline_cost: float
    optimised_cost: float

    @property
    def improvement_ratio(self) -> float:
        baseline = self.baseline_cost
        improved = self.optimised_cost
        if baseline <= 0:
            return 0.0
        return max(0.0, (baseline - improved) / baseline)


SOLVER_LABELS = {
    "minimal": "Minimal ALNS",
    "matheuristic": "Matheuristic ALNS",
    "q_learning": "Matheuristic + Q-learning",
}

SCALES = ("small", "medium", "large")


REPORT_OVERRIDES = {
    "medium": {"num_tasks": 12, "num_vehicles": 1},
    "large": {"num_tasks": 16, "num_vehicles": 1},
}


def _build_config(scale: str):
    overrides = REPORT_OVERRIDES.get(scale, {})
    return get_scale_config(scale, **overrides)


def _scaled_iterations(scale: str, solver: str, iteration_scale: float) -> int:
    base_iterations = get_solver_iterations(scale, solver)
    scaled = max(1, int(round(base_iterations * iteration_scale)))
    return scaled


def run_minimal(scale: str, seed: int, iteration_scale: float) -> SolverResult:
    scenario = build_scenario(_build_config(scale))
    iterations = _scaled_iterations(scale, "minimal", iteration_scale)
    baseline, optimised = run_minimal_trial(
        scenario,
        iterations=iterations,
        seed=seed,
    )
    return SolverResult(
        solver="minimal",
        scale=scale,
        baseline_cost=baseline,
        optimised_cost=optimised,
    )


def run_matheuristic(scale: str, seed: int, iteration_scale: float) -> SolverResult:
    scenario = build_scenario(_build_config(scale))
    iterations = _scaled_iterations(scale, "matheuristic", iteration_scale)
    strategy = PartialRechargeMinimalStrategy(safety_margin=0.02, min_margin=0.0)
    # Use the NEW run_matheuristic_trial with FULL matheuristic capabilities
    baseline, optimised = run_matheuristic_trial(
        scenario,
        strategy,
        iterations=iterations,
        seed=seed,
    )
    return SolverResult(
        solver="matheuristic",
        scale=scale,
        baseline_cost=baseline,
        optimised_cost=optimised,
    )


def run_q_learning(scale: str, seed: int, iteration_scale: float) -> SolverResult:
    config = _build_config(scale)
    iterations = _scaled_iterations(scale, "q_learning", iteration_scale)
    planner, baseline, optimised = run_q_learning_trial(
        config,
        iterations=iterations,
        seed=seed,
    )

    # Print operator statistics for large scale
    if scale == "large" and hasattr(planner, '_q_agent') and planner._q_agent is not None:
        print("\n" + "="*70)
        print(f"Q-LEARNING OPERATOR STATISTICS (Large Scale, seed={seed})")
        print("="*70)
        print(planner._q_agent.format_statistics())

        stats = planner._q_agent.statistics()
        repair_totals = {}
        for state, state_stats in stats.items():
            for stat in state_stats:
                _, repair = stat.action
                repair_totals[repair] = repair_totals.get(repair, 0) + stat.total_usage

        total_actions = sum(repair_totals.values())
        print("\nREPAIR OPERATOR USAGE SUMMARY:")
        print(f"Total selections: {total_actions}")
        for repair, count in sorted(repair_totals.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total_actions if total_actions > 0 else 0
            print(f"  {repair:12s}: {count:5d} times ({pct:5.1f}%)")
        print("="*70 + "\n")

    return SolverResult(
        solver="q_learning",
        scale=scale,
        baseline_cost=baseline,
        optimised_cost=optimised,
    )


RUNNERS = {
    "minimal": run_minimal,
    "matheuristic": run_matheuristic,
    "q_learning": run_q_learning,
}


def collect_results(scales: Iterable[str], seed: int, iteration_scale: float) -> List[SolverResult]:
    results: List[SolverResult] = []
    for scale in scales:
        for solver_key, runner in RUNNERS.items():
            result = runner(scale, seed, iteration_scale)
            results.append(result)
            print(
                f"{scale.title():<6} | {SOLVER_LABELS[solver_key]:<24} "
                f"{result.baseline_cost:>8.2f} → {result.optimised_cost:>8.2f} "
                f"(Δ {result.improvement_ratio * 100:5.2f}%)"
            )
    return results


def to_markdown_table(results: Iterable[SolverResult]) -> str:
    header = "| Scale | Solver | Baseline Cost | Optimised Cost | Improvement |"
    separator = "|:------|:-------|--------------:|---------------:|------------:|"
    rows: List[str] = []

    sorted_results = sorted(
        results,
        key=lambda item: (SCALES.index(item.scale), item.solver),
    )
    for item in sorted_results:
        baseline = _format_cost(item.baseline_cost)
        optimised = _format_cost(item.optimised_cost)
        improvement = _format_percentage(item.improvement_ratio * 100)
        rows.append(
            "| {scale} | {solver} | {baseline} | {optimised} | {improvement} |".format(
                scale=item.scale.title(),
                solver=SOLVER_LABELS[item.solver],
                baseline=baseline,
                optimised=optimised,
                improvement=improvement,
            )
        )
    return "\n".join([header, separator, *rows])


def _format_cost(value: float) -> str:
    if math.isinf(value):
        return "∞"
    return f"{value:.2f}"


def _format_percentage(value: float) -> str:
    if math.isnan(value):
        return "0.00%"
    return f"{value:.2f}%"


def save_results_json(results: Iterable[SolverResult], path: Path) -> None:
    serialisable: Dict[str, Dict[str, Dict[str, float]]] = {}
    for result in results:
        scale_bucket = serialisable.setdefault(result.scale, {})
        scale_bucket[result.solver] = {
            "baseline_cost": result.baseline_cost,
            "optimised_cost": result.optimised_cost,
            "improvement_ratio": result.improvement_ratio,
        }
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def plot_improvements(results: Iterable[SolverResult], output_path: Path) -> None:
    """Render a lightweight SVG bar chart for the solver improvements."""

    by_solver: Dict[str, List[float]] = {key: [] for key in RUNNERS}
    for scale in SCALES:
        scale_results = [r for r in results if r.scale == scale]
        for solver_key in RUNNERS:
            match = next(r for r in scale_results if r.solver == solver_key)
            by_solver[solver_key].append(match.improvement_ratio * 100)

    width = 900
    height = 500
    margin = 70
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin

    max_value = max(value for values in by_solver.values() for value in values)
    max_value = max(1.0, max_value)
    y_ticks = 5
    tick_values = [round(max_value * i / y_ticks, 2) for i in range(y_ticks + 1)]

    group_width = chart_width / len(SCALES)
    bar_width = group_width / (len(RUNNERS) + 1)

    def y_coord(value: float) -> float:
        return height - margin - (value / max_value) * chart_height

    svg_parts = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>",
        f"  <rect width='{width}' height='{height}' fill='white'/>",
        "  <style> .axis { stroke: #333; stroke-width: 2; }",
        "          .grid { stroke: #ccc; stroke-width: 1; stroke-dasharray: 4 4; }",
        "          .label { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 16px; fill: #333; }",
        "          .legend { font-size: 15px; } </style>",
    ]

    # Axes
    svg_parts.append(
        f"  <line class='axis' x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' />"
    )
    svg_parts.append(
        f"  <line class='axis' x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' />"
    )

    # Grid and Y-axis labels
    for tick in tick_values:
        y = y_coord(tick)
        svg_parts.append(
            f"  <line class='grid' x1='{margin}' y1='{y:.2f}' x2='{width - margin}' y2='{y:.2f}' />"
        )
        svg_parts.append(
            f"  <text class='label' x='{margin - 15}' y='{y + 5:.2f}' text-anchor='end'>{tick:.1f}</text>"
        )

    colours = ["#4B8BBE", "#306998", "#FFE873"]

    for scale_index, scale in enumerate(SCALES):
        group_x = margin + scale_index * group_width
        svg_parts.append(
            f"  <text class='label' x='{group_x + group_width / 2:.2f}' y='{height - margin + 30}' text-anchor='middle'>{scale.title()}</text>"
        )

        for solver_index, solver_key in enumerate(RUNNERS):
            value = by_solver[solver_key][scale_index]
            bar_height = (value / max_value) * chart_height
            x = group_x + solver_index * bar_width + bar_width / 2
            y = y_coord(value)
            colour = colours[solver_index % len(colours)]
            svg_parts.append(
                f"  <rect x='{x:.2f}' y='{y:.2f}' width='{bar_width * 0.8:.2f}' height='{bar_height:.2f}' fill='{colour}' />"
            )
            svg_parts.append(
                f"  <text class='label' x='{x + bar_width * 0.4:.2f}' y='{y - 6:.2f}' text-anchor='middle'>{value:.2f}%</text>"
            )

    # Legend
    legend_x = width - margin - 220
    legend_y = margin - 20
    for idx, solver_key in enumerate(RUNNERS):
        colour = colours[idx % len(colours)]
        y = legend_y + idx * 24
        svg_parts.append(
            f"  <rect x='{legend_x}' y='{y}' width='18' height='18' fill='{colour}' />"
        )
        svg_parts.append(
            f"  <text class='label legend' x='{legend_x + 26}' y='{y + 14}'>{SOLVER_LABELS[solver_key]}</text>"
        )

    svg_parts.append(
        "  <text class='label' x='{cx}' y='{cy}' text-anchor='middle'>{title}</text>".format(
            cx=width / 2,
            cy=margin / 2,
            title="ALNS regression improvements by solver variant",
        )
    )
    svg_parts.append(
        f"  <text class='label' x='{margin - 50}' y='{height / 2}' text-anchor='middle' "
        f"transform='rotate(-90 {margin - 50},{height / 2})'>Relative improvement (%)</text>"
    )
    svg_parts.append("</svg>")

    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed shared across solver runs for reproducibility.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/alns_regression_improvements.svg"),
        help="Path to the SVG chart that will be written.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("docs/data/alns_regression_results.json"),
        help="Path to the JSON file where raw metrics will be stored.",
    )
    parser.add_argument(
        "--iteration-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the unified iteration budgets (e.g. 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    # Configure logging to see LP repair diagnostics
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG to see simplex details
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    print("Collecting ALNS regression metrics...")
    results = collect_results(SCALES, seed=args.seed, iteration_scale=args.iteration_scale)

    print("\nMarkdown table:")
    markdown = to_markdown_table(results)
    print(markdown)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)

    plot_improvements(results, args.figure)
    save_results_json(results, args.json)

    print(f"\nSaved chart to {args.figure}")
    print(f"Saved metrics to {args.json}")


if __name__ == "__main__":
    main()
