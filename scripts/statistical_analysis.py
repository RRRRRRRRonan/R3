"""Statistical analysis for EJOR paper: paired comparisons, Wilcoxon tests, CI.

Usage:
    python3 scripts/statistical_analysis.py \
        --scales S,M,L,XL \
        --benchmark-root results/benchmark \
        --output-dir results/paper
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None  # type: ignore[assignment]

ONLINE_ALGORITHMS = ["rl_apc", "greedy_fr", "greedy_pr", "random_rule"]
OFFLINE_ALGORITHMS = ["alns_fr", "alns_pr"]
ALL_ALGORITHMS = ONLINE_ALGORITHMS + OFFLINE_ALGORITHMS

ALGO_DISPLAY = {
    "rl_apc": "RL-APC",
    "greedy_fr": "Greedy-FR",
    "greedy_pr": "Greedy-PR",
    "random_rule": "Random",
    "alns_fr": "ALNS-FR",
    "alns_pr": "ALNS-PR",
}


def load_benchmark_csv(path: Path) -> Dict[str, Dict[int, dict]]:
    """Load benchmark CSV, return {algo_id -> {seed -> row_dict}}."""
    result: Dict[str, Dict[int, dict]] = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            algo = row["algorithm_id"]
            seed = int(row["seed"])
            result[algo][seed] = row
    return dict(result)


def paired_costs(
    data: Dict[str, Dict[int, dict]], algo_a: str, algo_b: str
) -> tuple:
    """Return paired cost arrays for common seeds."""
    seeds_a = set(data.get(algo_a, {}).keys())
    seeds_b = set(data.get(algo_b, {}).keys())
    common = sorted(seeds_a & seeds_b)
    if not common:
        return np.array([]), np.array([]), []
    costs_a = np.array([float(data[algo_a][s]["cost"]) for s in common])
    costs_b = np.array([float(data[algo_b][s]["cost"]) for s in common])
    return costs_a, costs_b, common


def analyze_scale(data: Dict[str, Dict[int, dict]], scale: str) -> dict:
    """Compute statistics for one scale."""
    results = {"scale": scale, "algorithms": {}, "comparisons": []}

    # Per-algorithm summary stats
    for algo in ALL_ALGORITHMS:
        if algo not in data:
            continue
        costs = np.array([float(r["cost"]) for r in data[algo].values()])
        n = len(costs)
        mean = costs.mean()
        std = costs.std(ddof=1) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        results["algorithms"][algo] = {
            "n": n,
            "mean": mean,
            "std": std,
            "min": costs.min(),
            "max": costs.max(),
            "median": np.median(costs),
            "ci95": ci95,
        }

    # Paired comparisons: RL vs each baseline
    for baseline in ["greedy_fr", "greedy_pr", "random_rule", "alns_fr", "alns_pr"]:
        rl_costs, bl_costs, seeds = paired_costs(data, "rl_apc", baseline)
        if len(rl_costs) == 0:
            continue
        improvement = (bl_costs - rl_costs) / bl_costs * 100
        n_wins = int((rl_costs < bl_costs).sum())
        n_ties = int((rl_costs == bl_costs).sum())
        n_losses = int((rl_costs > bl_costs).sum())

        p_val = None
        if wilcoxon is not None and len(rl_costs) >= 10:
            try:
                _, p_val = wilcoxon(rl_costs, bl_costs, alternative="less")
            except Exception:
                p_val = None

        results["comparisons"].append({
            "rl_vs": baseline,
            "n_pairs": len(seeds),
            "mean_improvement_pct": improvement.mean(),
            "std_improvement_pct": improvement.std(ddof=1) if len(improvement) > 1 else 0.0,
            "n_wins": n_wins,
            "n_ties": n_ties,
            "n_losses": n_losses,
            "wilcoxon_p": p_val,
        })

    return results


def format_p_value(p) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return f"{p:.2e}***"
    if p < 0.01:
        return f"{p:.4f}**"
    if p < 0.05:
        return f"{p:.4f}*"
    return f"{p:.4f}"


def print_report(all_results: List[dict]) -> str:
    lines = []
    for res in all_results:
        scale = res["scale"]
        lines.append(f"\n{'='*80}")
        lines.append(f"  Scale {scale} — Summary Statistics")
        lines.append(f"{'='*80}")
        hdr = f"{'Algorithm':<15} {'n':>4} {'Mean':>12} {'Std':>12} {'95% CI':>12} {'Min':>12} {'Median':>12} {'Max':>12}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for algo in ALL_ALGORITHMS:
            if algo not in res["algorithms"]:
                continue
            s = res["algorithms"][algo]
            name = ALGO_DISPLAY.get(algo, algo)
            lines.append(
                f"{name:<15} {s['n']:>4} {s['mean']:>12,.0f} {s['std']:>12,.0f} "
                f"±{s['ci95']:>10,.0f} {s['min']:>12,.0f} {s['median']:>12,.0f} {s['max']:>12,.0f}"
            )

        if res["comparisons"]:
            lines.append(f"\n  Paired Comparisons: RL-APC vs Baselines")
            lines.append(f"  {'Baseline':<15} {'n':>4} {'Improv%':>10} {'Wins':>6} {'Losses':>8} {'Wilcoxon p':>16}")
            lines.append("  " + "-" * 70)
            for c in res["comparisons"]:
                bl_name = ALGO_DISPLAY.get(c["rl_vs"], c["rl_vs"])
                lines.append(
                    f"  {bl_name:<15} {c['n_pairs']:>4} "
                    f"{c['mean_improvement_pct']:>+9.1f}% "
                    f"{c['n_wins']:>4}/{c['n_pairs']:<4} "
                    f"{c['n_losses']:>4}/{c['n_pairs']:<4} "
                    f"{format_p_value(c['wilcoxon_p']):>16}"
                )

    return "\n".join(lines)


def generate_latex_table(all_results: List[dict], output_path: Path):
    """Generate LaTeX table for EJOR paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm comparison across problem scales (average total cost). "
        r"Lower is better. Best online method in \textbf{bold}.}",
        r"\label{tab:comparison}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(ALL_ALGORITHMS) + "}",
        r"\toprule",
    ]

    # Header
    headers = [ALGO_DISPLAY.get(a, a) for a in ALL_ALGORITHMS]
    lines.append("Scale & " + " & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for res in all_results:
        scale = res["scale"]
        algos = res["algorithms"]
        # Find best online algorithm
        online_costs = {
            a: algos[a]["mean"] for a in ONLINE_ALGORITHMS if a in algos
        }
        best_online = min(online_costs, key=online_costs.get) if online_costs else None

        cells = [scale]
        for algo in ALL_ALGORITHMS:
            if algo not in algos:
                cells.append("—")
                continue
            mean = algos[algo]["mean"]
            ci = algos[algo]["ci95"]
            val = f"{mean:,.0f}"
            if algo == best_online:
                val = r"\textbf{" + val + "}"
            cells.append(val)

        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_wilcoxon_latex(all_results: List[dict], output_path: Path):
    """Generate LaTeX table for Wilcoxon test results."""
    baselines = ["greedy_fr", "random_rule", "alns_fr", "alns_pr"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Wilcoxon signed-rank test: RL-APC vs.\ baselines. "
        r"Improvement (\%) and $p$-values reported. "
        r"***: $p<0.001$, **: $p<0.01$, *: $p<0.05$.}",
        r"\label{tab:wilcoxon}",
        r"\small",
        r"\begin{tabular}{ll" + "cc" * len(baselines) + "}",
        r"\toprule",
    ]

    # Multi-row header
    hdr1 = "& "
    hdr2 = "Scale & $n$ "
    for bl in baselines:
        name = ALGO_DISPLAY.get(bl, bl)
        hdr1 += r"& \multicolumn{2}{c}{" + name + "} "
        hdr2 += r"& Improv.(\%) & $p$-value "
    lines.append(hdr1 + r"\\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}")
    lines.append(hdr2 + r"\\")
    lines.append(r"\midrule")

    for res in all_results:
        comp_map = {c["rl_vs"]: c for c in res["comparisons"]}
        n = comp_map.get("greedy_fr", {}).get("n_pairs", "—")
        cells = [res["scale"], str(n)]
        for bl in baselines:
            c = comp_map.get(bl)
            if c is None:
                cells.extend(["—", "—"])
            else:
                imp = f"{c['mean_improvement_pct']:+.1f}"
                p = c["wilcoxon_p"]
                if p is not None and p < 0.001:
                    pstr = f"{p:.1e}***"
                elif p is not None and p < 0.01:
                    pstr = f"{p:.3f}**"
                elif p is not None and p < 0.05:
                    pstr = f"{p:.3f}*"
                elif p is not None:
                    pstr = f"{p:.3f}"
                else:
                    pstr = "—"
                cells.extend([imp, pstr])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="EJOR-quality statistical analysis")
    parser.add_argument("--scales", default="S,M,L,XL")
    parser.add_argument("--benchmark-root", default="results/benchmark")
    parser.add_argument("--output-dir", default="results/paper")
    parser.add_argument("--suffix", default="_30", help="CSV filename suffix (e.g. _30)")
    args = parser.parse_args()

    scales = [s.strip().upper() for s in args.scales.split(",")]
    bench_root = Path(args.benchmark_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for scale in scales:
        csv_path = bench_root / f"evaluate_{scale}{args.suffix}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue
        data = load_benchmark_csv(csv_path)
        res = analyze_scale(data, scale)
        all_results.append(res)

    if not all_results:
        print("No data found.")
        return 1

    # Text report
    report = print_report(all_results)
    print(report)
    (out_dir / "statistical_report.txt").write_text(report, encoding="utf-8")

    # LaTeX tables
    generate_latex_table(all_results, out_dir / "table_comparison.tex")
    generate_wilcoxon_latex(all_results, out_dir / "table_wilcoxon.tex")

    print(f"\nSaved to: {out_dir}/")
    print(f"  statistical_report.txt")
    print(f"  table_comparison.tex")
    print(f"  table_wilcoxon.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
