"""Generate paper tables and figures from training and benchmark results.

Outputs:
  - Figure 1: Training curves (mean eval reward vs timesteps) for all scales
  - Table 1: Algorithm comparison across scales (CSV + LaTeX)
  - Table 2: Reward shaping ablation (before/after tuning)
  - Table 3: Cost decomposition by algorithm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Figure 1: Training Curves
# ---------------------------------------------------------------------------

def generate_training_curves(
    scales: List[str],
    results_root: Path,
    output_path: Path,
    *,
    show: bool = False,
) -> None:
    """Plot mean eval reward vs timesteps for each scale."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"S": "#1f77b4", "M": "#ff7f0e", "L": "#2ca02c", "XL": "#d62728"}
    markers = {"S": "o", "M": "s", "L": "^", "XL": "D"}

    for scale in scales:
        npz_path = results_root / f"train_{scale}" / "eval_logs" / "evaluations.npz"
        if not npz_path.exists():
            print(f"[warn] {npz_path} not found, skipping scale {scale}")
            continue
        data = np.load(str(npz_path))
        timesteps = data["timesteps"]
        results = data["results"]
        mean_reward = results.mean(axis=1)
        std_reward = results.std(axis=1)

        color = colors.get(scale, "gray")
        marker = markers.get(scale, "o")
        ax.plot(
            timesteps / 1000,
            mean_reward,
            label=f"Scale {scale}",
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.5,
        )
        ax.fill_between(
            timesteps / 1000,
            mean_reward - std_reward,
            mean_reward + std_reward,
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Training Timesteps (×1000)", fontsize=12)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=12)
    ax.set_title("RL-APC Training Convergence Across Scales", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="y")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"[fig1] saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1b: Multi-seed training curves (S scale)
# ---------------------------------------------------------------------------

def generate_multiseed_curves(
    results_root: Path,
    output_path: Path,
    *,
    show: bool = False,
) -> None:
    """Plot S-scale training curves for multiple seeds."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_dirs = {
        "Seed 42": results_root / "train_S",
        "Seed 43": results_root / "train_S_seed43",
        "Seed 44": results_root / "train_S_seed44",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    found = False
    for (label, path), color in zip(seed_dirs.items(), colors):
        npz_path = path / "eval_logs" / "evaluations.npz"
        if not npz_path.exists():
            print(f"[warn] {npz_path} not found, skipping {label}")
            continue
        found = True
        data = np.load(str(npz_path))
        timesteps = data["timesteps"]
        results = data["results"]
        mean_reward = results.mean(axis=1)
        ax.plot(timesteps / 1000, mean_reward, label=label, color=color, linewidth=1.5)

    if not found:
        print("[warn] No multi-seed data found, skipping Figure 1b")
        plt.close(fig)
        return

    ax.set_xlabel("Training Timesteps (×1000)", fontsize=12)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=12)
    ax.set_title("S-Scale Training Stability (Multi-Seed)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"[fig1b] saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table 1: Algorithm Comparison
# ---------------------------------------------------------------------------

ALGORITHM_ORDER = [
    "rl_apc", "greedy_fr", "greedy_pr", "random_rule",
    "alns_fr", "alns_pr", "mip_hind",
]
ALGO_LABELS = {
    "rl_apc": "RL-APC",
    "greedy_fr": "Greedy-FR",
    "greedy_pr": "Greedy-PR",
    "random_rule": "Random",
    "alns_fr": "ALNS-FR",
    "alns_pr": "ALNS-PR",
    "mip_hind": "MIP-Hind",
}


def _load_benchmark_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _aggregate_by_algorithm(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate benchmark rows by algorithm_id. Returns mean metrics per algorithm."""
    by_algo: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        algo_id = row.get("algorithm_id", "")
        by_algo.setdefault(algo_id, []).append(row)

    result = {}
    for algo_id, items in by_algo.items():
        ok_items = [
            item for item in items
            if str(item.get("status", "")).upper() in ("OK", "OPTIMAL", "FEASIBLE")
        ]
        if not ok_items:
            result[algo_id] = {
                "count": len(items),
                "success_count": 0,
                "avg_cost": None,
                "avg_runtime": None,
                "completed_tasks": None,
                "rejected_tasks": None,
                "avg_delay": None,
            }
            continue

        def _safe_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        costs = [_safe_float(item.get("cost")) for item in ok_items]
        costs = [c for c in costs if c is not None]

        runtimes = [_safe_float(item.get("runtime_s")) for item in ok_items]
        runtimes = [r for r in runtimes if r is not None]

        completed = [_safe_float(item.get("completed_tasks")) for item in ok_items]
        completed = [c for c in completed if c is not None]

        rejected = [_safe_float(item.get("rejected_tasks")) for item in ok_items]
        rejected = [r for r in rejected if r is not None]

        delays = [_safe_float(item.get("metrics_total_delay")) for item in ok_items]
        delays = [d for d in delays if d is not None]

        result[algo_id] = {
            "count": len(items),
            "success_count": len(ok_items),
            "avg_cost": sum(costs) / len(costs) if costs else None,
            "avg_runtime": sum(runtimes) / len(runtimes) if runtimes else None,
            "completed_tasks": sum(completed) / len(completed) if completed else None,
            "rejected_tasks": sum(rejected) / len(rejected) if rejected else None,
            "avg_delay": sum(delays) / len(delays) if delays else None,
        }
    return result


def generate_comparison_table(
    scales: List[str],
    benchmark_root: Path,
    output_csv: Path,
    output_latex: Optional[Path] = None,
    benchmark_suffix: str = "_30",
) -> None:
    """Generate Table 1: Algorithm comparison across scales."""
    all_data: Dict[str, Dict[str, Dict[str, Any]]] = {}  # scale -> algo -> metrics
    for scale in scales:
        csv_path = benchmark_root / f"evaluate_{scale}{benchmark_suffix}.csv"
        if not csv_path.exists():
            print(f"[warn] {csv_path} not found, skipping scale {scale}")
            continue
        rows = _load_benchmark_csv(csv_path)
        all_data[scale] = _aggregate_by_algorithm(rows)

    if not all_data:
        print("[warn] No benchmark data found, skipping Table 1")
        return

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["Scale"] + [ALGO_LABELS.get(a, a) for a in ALGORITHM_ORDER]
    csv_rows = []
    for scale in scales:
        if scale not in all_data:
            continue
        row_data = {"Scale": scale}
        for algo in ALGORITHM_ORDER:
            label = ALGO_LABELS.get(algo, algo)
            metrics = all_data[scale].get(algo)
            if metrics is None or metrics.get("avg_cost") is None:
                row_data[label] = "—"
            else:
                row_data[label] = f"{metrics['avg_cost']:.0f}"
        csv_rows.append(row_data)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[table1] saved CSV: {output_csv}")

    # Write LaTeX
    if output_latex:
        _write_comparison_latex(scales, all_data, output_latex)


def _write_comparison_latex(
    scales: List[str],
    all_data: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
) -> None:
    cols = "l" + "r" * len(ALGORITHM_ORDER)
    header_labels = " & ".join([ALGO_LABELS.get(a, a) for a in ALGORITHM_ORDER])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm comparison: average total cost across scales.}",
        r"\label{tab:comparison}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        rf"Scale & {header_labels} \\",
        r"\midrule",
    ]
    for scale in scales:
        if scale not in all_data:
            continue
        values = []
        # Find best (lowest) cost for bold formatting
        costs = {}
        for algo in ALGORITHM_ORDER:
            metrics = all_data[scale].get(algo)
            if metrics and metrics.get("avg_cost") is not None:
                costs[algo] = metrics["avg_cost"]
        best_algo = min(costs, key=costs.get) if costs else None

        for algo in ALGORITHM_ORDER:
            metrics = all_data[scale].get(algo)
            if metrics is None or metrics.get("avg_cost") is None:
                values.append("—")
            else:
                cost_str = f"{metrics['avg_cost']:,.0f}"
                if algo == best_algo:
                    cost_str = rf"\textbf{{{cost_str}}}"
                values.append(cost_str)
        lines.append(f"{scale} & {' & '.join(values)} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table1] saved LaTeX: {output_path}")


# ---------------------------------------------------------------------------
# Table 2: Detailed metrics comparison
# ---------------------------------------------------------------------------

def generate_detailed_table(
    scales: List[str],
    benchmark_root: Path,
    output_csv: Path,
    benchmark_suffix: str = "_30",
) -> None:
    """Generate detailed metrics table: cost, completed_tasks, rejected_tasks, delay, runtime."""
    header = [
        "Scale", "Algorithm", "Avg Cost", "Completed Tasks",
        "Rejected Tasks", "Avg Delay", "Runtime (s)",
    ]
    csv_rows = []
    for scale in scales:
        csv_path = benchmark_root / f"evaluate_{scale}{benchmark_suffix}.csv"
        if not csv_path.exists():
            continue
        rows = _load_benchmark_csv(csv_path)
        agg = _aggregate_by_algorithm(rows)
        for algo in ALGORITHM_ORDER:
            metrics = agg.get(algo)
            if metrics is None:
                continue
            csv_rows.append({
                "Scale": scale,
                "Algorithm": ALGO_LABELS.get(algo, algo),
                "Avg Cost": f"{metrics['avg_cost']:.0f}" if metrics.get("avg_cost") is not None else "—",
                "Completed Tasks": f"{metrics['completed_tasks']:.1f}" if metrics.get("completed_tasks") is not None else "—",
                "Rejected Tasks": f"{metrics['rejected_tasks']:.1f}" if metrics.get("rejected_tasks") is not None else "—",
                "Avg Delay": f"{metrics['avg_delay']:.1f}" if metrics.get("avg_delay") is not None else "—",
                "Runtime (s)": f"{metrics['avg_runtime']:.2f}" if metrics.get("avg_runtime") is not None else "—",
            })

    if not csv_rows:
        print("[warn] No data for detailed table")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[table2] saved: {output_csv}")


# ---------------------------------------------------------------------------
# Table 3: Cost Decomposition
# ---------------------------------------------------------------------------

COST_COMPONENTS = [
    ("metrics_total_distance", "Travel"),
    ("metrics_total_charging", "Charging"),
    ("metrics_total_delay", "Tardiness"),
    ("metrics_total_waiting", "Waiting"),
    ("metrics_total_standby", "Standby"),
]


def generate_cost_decomposition(
    scales: List[str],
    benchmark_root: Path,
    output_csv: Path,
    algos: Optional[List[str]] = None,
    benchmark_suffix: str = "_30",
) -> None:
    """Generate Table 3: Cost decomposition by component for selected algorithms."""
    if algos is None:
        algos = ["rl_apc", "greedy_fr", "alns_fr"]

    header = ["Scale", "Algorithm"] + [label for _, label in COST_COMPONENTS] + ["Total Cost"]
    csv_rows = []

    for scale in scales:
        csv_path = benchmark_root / f"evaluate_{scale}{benchmark_suffix}.csv"
        if not csv_path.exists():
            continue
        rows = _load_benchmark_csv(csv_path)
        by_algo: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            algo_id = row.get("algorithm_id", "")
            if algo_id in algos:
                status = str(row.get("status", "")).upper()
                if status in ("OK", "OPTIMAL", "FEASIBLE"):
                    by_algo.setdefault(algo_id, []).append(row)

        for algo in algos:
            items = by_algo.get(algo, [])
            if not items:
                continue
            row_data = {
                "Scale": scale,
                "Algorithm": ALGO_LABELS.get(algo, algo),
            }
            for key, label in COST_COMPONENTS:
                vals = []
                for item in items:
                    try:
                        vals.append(float(item.get(key, 0)))
                    except (TypeError, ValueError):
                        pass
                avg = sum(vals) / len(vals) if vals else 0
                row_data[label] = f"{avg:.1f}"

            costs = []
            for item in items:
                try:
                    costs.append(float(item.get("cost", 0)))
                except (TypeError, ValueError):
                    pass
            row_data["Total Cost"] = f"{sum(costs)/len(costs):.0f}" if costs else "—"
            csv_rows.append(row_data)

    if not csv_rows:
        print("[warn] No data for cost decomposition")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[table3] saved: {output_csv}")


# ---------------------------------------------------------------------------
# Training Summary Table
# ---------------------------------------------------------------------------

def generate_training_summary(
    scales: List[str],
    results_root: Path,
    output_csv: Path,
) -> None:
    """Generate training summary: best reward, convergence step, total steps per scale."""
    header = ["Scale", "Total Steps", "Best Mean Reward", "Best Step", "Final Mean Reward"]
    csv_rows = []
    for scale in scales:
        npz_path = results_root / f"train_{scale}" / "eval_logs" / "evaluations.npz"
        if not npz_path.exists():
            continue
        data = np.load(str(npz_path))
        timesteps = data["timesteps"]
        results = data["results"]
        mean_reward = results.mean(axis=1)
        best_idx = int(mean_reward.argmax())
        csv_rows.append({
            "Scale": scale,
            "Total Steps": int(timesteps[-1]),
            "Best Mean Reward": f"{mean_reward[best_idx]:.1f}",
            "Best Step": int(timesteps[best_idx]),
            "Final Mean Reward": f"{mean_reward[-1]:.1f}",
        })

    if not csv_rows:
        print("[warn] No training data found")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[training_summary] saved: {output_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures.")
    parser.add_argument(
        "--scales", type=str, default="S,M,L,XL",
        help="Comma-separated scales to include.",
    )
    parser.add_argument(
        "--results-root", type=str, default="results/rl",
        help="Root directory for training results.",
    )
    parser.add_argument(
        "--benchmark-root", type=str, default="results/benchmark",
        help="Root directory for benchmark evaluation CSVs.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/paper",
        help="Output directory for paper assets.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation (e.g. in headless env without display).",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Only generate specific outputs: fig1,fig1b,table1,table2,table3,summary (comma-separated).",
    )
    parser.add_argument(
        "--benchmark-suffix", type=str, default="_30",
        help="Suffix for benchmark CSV filenames (e.g. _30 for evaluate_S_30.csv).",
    )
    args = parser.parse_args()

    scales = [s.strip().upper() for s in args.scales.split(",") if s.strip()]
    results_root = Path(args.results_root)
    benchmark_root = Path(args.benchmark_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_suffix = args.benchmark_suffix

    targets = None
    if args.only:
        targets = set(t.strip().lower() for t in args.only.split(",") if t.strip())

    def _should_run(name: str) -> bool:
        return targets is None or name in targets

    # Figure 1: Training curves
    if not args.skip_figures and _should_run("fig1"):
        generate_training_curves(
            scales, results_root,
            output_dir / "fig1_training_curves.png",
            show=args.show,
        )

    # Figure 1b: Multi-seed curves
    if not args.skip_figures and _should_run("fig1b"):
        generate_multiseed_curves(
            results_root,
            output_dir / "fig1b_multiseed_curves.png",
            show=args.show,
        )

    # Table 1: Algorithm comparison
    if _should_run("table1"):
        generate_comparison_table(
            scales, benchmark_root,
            output_csv=output_dir / "table1_comparison.csv",
            output_latex=output_dir / "table1_comparison.tex",
            benchmark_suffix=bench_suffix,
        )

    # Table 2: Detailed metrics
    if _should_run("table2"):
        generate_detailed_table(
            scales, benchmark_root,
            output_csv=output_dir / "table2_detailed_metrics.csv",
            benchmark_suffix=bench_suffix,
        )

    # Table 3: Cost decomposition
    if _should_run("table3"):
        generate_cost_decomposition(
            scales, benchmark_root,
            output_csv=output_dir / "table3_cost_decomposition.csv",
            benchmark_suffix=bench_suffix,
        )

    # Training summary
    if _should_run("summary"):
        generate_training_summary(
            scales, results_root,
            output_csv=output_dir / "training_summary.csv",
        )

    print(f"\n[done] All outputs in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
