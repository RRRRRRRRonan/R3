"""Generate paper tables and figures using VecNormalize-synced evaluation results.

Uses final_model (synced with VecNormalize) for M and XL, best_model for S and L.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _load_csv_rows(path: str) -> List[Dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _get_eval_csv(scale: str, benchmark_root: str) -> str:
    """Return the best available evaluation CSV for each scale.

    Priority: v3 > synced > v2 > v1 (standard).
    This ensures retrained models are automatically picked up.
    """
    candidates = [
        Path(benchmark_root) / f"evaluate_{scale}_v3_30.csv",
        Path(benchmark_root) / f"evaluate_{scale}_synced_30.csv",
        Path(benchmark_root) / f"evaluate_{scale}_v2_30.csv",
        Path(benchmark_root) / f"evaluate_{scale}_30.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""


ALGO_DISPLAY = {
    "rl_apc": "RL-APC",
    "greedy_fr": "Greedy-FR",
    "greedy_pr": "Greedy-PR",
    "random_rule": "Random",
    "alns_fr": "ALNS-FR",
    "alns_pr": "ALNS-PR",
    "mip_hind": "MIP-Hind",
}

ALGO_ORDER = ["rl_apc", "greedy_fr", "greedy_pr", "random_rule", "alns_fr", "alns_pr"]


def generate_table1_comparison(scales, benchmark_root, output_dir):
    """Table 1: Average total cost across scales."""
    rows_csv = []
    rows_tex = []

    for scale in scales:
        csv_path = _get_eval_csv(scale, benchmark_root)
        if not csv_path:
            continue

        rows = _load_csv_rows(csv_path)
        algo_costs = {}
        for row in rows:
            algo_id = row.get("algorithm_id", "")
            cost = float(row.get("cost", 0))
            algo_costs.setdefault(algo_id, []).append(cost)

        csv_row = {"Scale": scale}
        tex_values = {}
        for algo in ALGO_ORDER:
            if algo in algo_costs:
                avg = np.mean(algo_costs[algo])
                csv_row[ALGO_DISPLAY.get(algo, algo)] = f"{avg:.0f}"
                tex_values[algo] = avg
            else:
                csv_row[ALGO_DISPLAY.get(algo, algo)] = "—"
                tex_values[algo] = None
        csv_row["MIP-Hind"] = "—"
        rows_csv.append(csv_row)

        # Find best for bold
        online_costs = {a: tex_values[a] for a in ALGO_ORDER if tex_values.get(a) is not None}
        best_algo = min(online_costs, key=online_costs.get) if online_costs else None

        tex_row_parts = [scale]
        for algo in ALGO_ORDER:
            v = tex_values.get(algo)
            if v is None:
                tex_row_parts.append("—")
            elif algo == best_algo:
                tex_row_parts.append(f"\\textbf{{{v:,.0f}}}")
            else:
                tex_row_parts.append(f"{v:,.0f}")
        tex_row_parts.append("—")  # MIP
        rows_tex.append(" & ".join(tex_row_parts) + " \\\\")

    # Write CSV
    csv_path = Path(output_dir) / "table1_comparison.csv"
    headers = ["Scale"] + [ALGO_DISPLAY[a] for a in ALGO_ORDER] + ["MIP-Hind"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_csv)

    # Write LaTeX
    tex_path = Path(output_dir) / "table1_comparison.tex"
    algo_headers = " & ".join([ALGO_DISPLAY[a] for a in ALGO_ORDER] + ["MIP-Hind"])
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Algorithm comparison: average total cost across scales (30 test instances each).}\n")
        f.write("\\label{tab:comparison}\n")
        f.write(f"\\begin{{tabular}}{{l{'r' * 7}}}\n")
        f.write("\\toprule\n")
        f.write(f"Scale & {algo_headers} \\\\\n")
        f.write("\\midrule\n")
        for row in rows_tex:
            f.write(row + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}")

    print(f"  table1: {csv_path}, {tex_path}")


def generate_table2_detailed(scales, benchmark_root, output_dir):
    """Table 2: Detailed metrics (cost, tasks, delay, runtime)."""
    rows_csv = []

    for scale in scales:
        csv_path = _get_eval_csv(scale, benchmark_root)
        if not csv_path:
            continue

        rows = _load_csv_rows(csv_path)
        algo_data = {}
        for row in rows:
            algo_id = row.get("algorithm_id", "")
            algo_data.setdefault(algo_id, []).append(row)

        for algo in ALGO_ORDER:
            if algo not in algo_data:
                continue
            data = algo_data[algo]
            costs = [float(r["cost"]) for r in data]
            runtimes = [float(r.get("runtime_s", 0)) for r in data]
            completed = [float(r.get("completed_tasks", 0)) for r in data if r.get("completed_tasks")]
            rejected = [float(r.get("rejected_tasks", 0)) for r in data if r.get("rejected_tasks")]
            delays = [float(r.get("metrics_total_delay", 0)) for r in data if r.get("metrics_total_delay")]

            rows_csv.append({
                "Scale": scale,
                "Algorithm": ALGO_DISPLAY.get(algo, algo),
                "Avg Cost": f"{np.mean(costs):.0f}",
                "Std Cost": f"{np.std(costs):.0f}",
                "Completed Tasks": f"{np.mean(completed):.1f}" if completed else "—",
                "Rejected Tasks": f"{np.mean(rejected):.1f}" if rejected else "—",
                "Avg Delay": f"{np.mean(delays):.1f}" if delays else "—",
                "Runtime (s)": f"{np.mean(runtimes):.2f}",
            })

    csv_path = Path(output_dir) / "table2_detailed_metrics.csv"
    headers = ["Scale", "Algorithm", "Avg Cost", "Std Cost", "Completed Tasks", "Rejected Tasks", "Avg Delay", "Runtime (s)"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_csv)

    print(f"  table2: {csv_path}")


def generate_table3_cost_decomposition(scales, benchmark_root, output_dir):
    """Table 3: Cost decomposition (travel, charging, tardiness, etc.)."""
    rows_csv = []

    cost_cols = {
        "Travel": "metrics_total_distance",
        "Charging": "metrics_total_charging",
        "Tardiness": "metrics_total_delay",
        "Waiting": "metrics_total_waiting",
        "Standby": "metrics_total_standby",
    }

    for scale in scales:
        csv_path = _get_eval_csv(scale, benchmark_root)
        if not csv_path:
            continue

        rows = _load_csv_rows(csv_path)
        algo_data = {}
        for row in rows:
            algo_id = row.get("algorithm_id", "")
            algo_data.setdefault(algo_id, []).append(row)

        for algo in ["rl_apc", "greedy_fr", "alns_fr"]:
            if algo not in algo_data:
                continue
            data = algo_data[algo]

            csv_row = {
                "Scale": scale,
                "Algorithm": ALGO_DISPLAY.get(algo, algo),
            }

            for label, col in cost_cols.items():
                vals = [float(r.get(col, 0) or 0) for r in data]
                csv_row[label] = f"{np.mean(vals):.1f}"

            costs = [float(r["cost"]) for r in data]
            csv_row["Total Cost"] = f"{np.mean(costs):.0f}"
            rows_csv.append(csv_row)

    csv_path = Path(output_dir) / "table3_cost_decomposition.csv"
    headers = ["Scale", "Algorithm", "Travel", "Charging", "Tardiness", "Waiting", "Standby", "Total Cost"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_csv)

    print(f"  table3: {csv_path}")


def generate_wilcoxon_table(scales, benchmark_root, output_dir):
    """Statistical significance table with Wilcoxon signed-rank tests."""
    from scipy import stats as scipy_stats

    rows_csv = []
    tex_rows = []

    for scale in scales:
        csv_path = _get_eval_csv(scale, benchmark_root)
        if not csv_path:
            continue

        rows = _load_csv_rows(csv_path)
        algo_costs = {}
        for row in rows:
            algo_id = row.get("algorithm_id", "")
            cost = float(row.get("cost", 0))
            algo_costs.setdefault(algo_id, []).append(cost)

        if "rl_apc" not in algo_costs:
            continue

        rl = np.array(algo_costs["rl_apc"])

        for baseline in ["greedy_fr", "random_rule", "alns_fr"]:
            if baseline not in algo_costs:
                continue
            bl = np.array(algo_costs[baseline])
            if len(rl) != len(bl):
                continue

            diff_pct = (rl.mean() - bl.mean()) / bl.mean() * 100
            try:
                stat, p = scipy_stats.wilcoxon(rl, bl, alternative="two-sided")
            except Exception:
                p = 1.0

            wins = int(np.sum(rl < bl))
            ties = int(np.sum(rl == bl))
            losses = len(rl) - wins - ties
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            rows_csv.append({
                "Scale": scale,
                "Baseline": ALGO_DISPLAY.get(baseline, baseline),
                "RL Mean": f"{rl.mean():.0f}",
                "Baseline Mean": f"{bl.mean():.0f}",
                "Diff %": f"{diff_pct:+.1f}",
                "p-value": f"{p:.2e}",
                "Significance": sig,
                "W/T/L": f"{wins}/{ties}/{losses}",
            })

            tex_rows.append(
                f"{scale} & {ALGO_DISPLAY.get(baseline, baseline)} & {rl.mean():,.0f} & "
                f"{bl.mean():,.0f} & {diff_pct:+.1f}\\% & {p:.2e} & {sig} & {wins}/{ties}/{losses} \\\\"
            )

    csv_path = Path(output_dir) / "table_wilcoxon.csv"
    headers = ["Scale", "Baseline", "RL Mean", "Baseline Mean", "Diff %", "p-value", "Significance", "W/T/L"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_csv)

    tex_path = Path(output_dir) / "table_wilcoxon.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Wilcoxon signed-rank tests: RL-APC vs baselines (30 instances per scale).}\n")
        f.write("\\label{tab:wilcoxon}\n")
        f.write("\\begin{tabular}{llrrrrcr}\n")
        f.write("\\toprule\n")
        f.write("Scale & Baseline & RL Mean & BL Mean & Diff \\% & $p$-value & Sig. & W/T/L \\\\\n")
        f.write("\\midrule\n")
        for row in tex_rows:
            f.write(row + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}")

    print(f"  wilcoxon: {csv_path}, {tex_path}")


def generate_training_summary(scales, results_root, output_dir):
    """Training summary: steps, best reward, convergence."""
    rows = []
    for scale in scales:
        npz = Path(results_root) / f"train_{scale}" / "eval_logs" / "evaluations.npz"
        if not npz.exists():
            continue
        data = np.load(str(npz))
        ts = data["timesteps"]
        rew = data["results"].mean(axis=1)
        rows.append({
            "Scale": scale,
            "Total Steps": f"{ts[-1]}",
            "Best Mean Reward": f"{rew.max():.1f}",
            "Best Step": f"{ts[rew.argmax()]}",
            "Final Mean Reward": f"{rew[-1]:.1f}",
        })

    csv_path = Path(output_dir) / "training_summary.csv"
    headers = ["Scale", "Total Steps", "Best Mean Reward", "Best Step", "Final Mean Reward"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  training: {csv_path}")


def generate_training_curves(scales, results_root, output_dir):
    """Figure 1: Training curves across scales with std bands and best markers.

    Enhanced version: includes std deviation bands, best_model star markers,
    and automatically picks the latest version (v3 > v2 > v1) for L/XL.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not installed")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"S": "#2196F3", "M": "#4CAF50", "L": "#FF9800", "XL": "#F44336"}
    linestyles = {"v1": "-", "v2": "--", "v3": "-."}

    plotted_labels = set()

    for scale in scales:
        # Try multiple training versions — prefer latest
        versions = [
            (f"train_{scale}_v3", "v3"),
            (f"train_{scale}_v2", "v2"),
            (f"train_{scale}", "v1"),
        ]
        for train_dir, version in versions:
            npz = Path(results_root) / train_dir / "eval_logs" / "evaluations.npz"
            if not npz.exists():
                continue

            data = np.load(str(npz))
            ts = data["timesteps"] / 1e6
            results = data["results"]  # [n_evals, n_episodes]
            mean_rew = results.mean(axis=1)
            std_rew = results.std(axis=1)

            color = colors.get(scale, "gray")
            ls = linestyles.get(version, "-")
            label = f"{scale}" if version == "v1" else f"{scale}-{version}"

            # For L/XL, only show the latest version (first found)
            scale_key = scale
            if scale_key in plotted_labels:
                continue
            plotted_labels.add(scale_key)

            ax.plot(ts, mean_rew, label=label, color=color, linewidth=2, linestyle=ls)
            ax.fill_between(ts, mean_rew - std_rew, mean_rew + std_rew,
                            color=color, alpha=0.15)

            # Mark best model point with star
            best_idx = mean_rew.argmax()
            ax.scatter([ts[best_idx]], [mean_rew[best_idx]], color=color,
                       s=120, zorder=5, edgecolors="black", linewidths=1,
                       marker="*")
            # Annotate best step
            ax.annotate(f"{ts[best_idx]:.1f}M",
                        xy=(ts[best_idx], mean_rew[best_idx]),
                        xytext=(5, 8), textcoords="offset points",
                        fontsize=7, color=color, alpha=0.8)

            break  # Only plot the latest version per scale

    ax.set_xlabel("Training Steps (millions)", fontsize=12)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=12)
    ax.set_title("RL-APC Training Convergence", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Save PNG and PDF
    for ext in ["png", "pdf"]:
        fig_path = Path(output_dir) / f"fig1_training_curves.{ext}"
        dpi = 150 if ext == "png" else 300
        fig.savefig(str(fig_path), dpi=dpi, bbox_inches="tight")
        print(f"  fig1: {fig_path}")
    plt.close(fig)


def generate_vecnorm_impact_table(benchmark_root, output_dir):
    """Table showing VecNormalize mismatch impact — key methodological finding."""
    rows = []

    for scale in ["S", "M", "L", "XL"]:
        best_f = Path(benchmark_root) / f"evaluate_{scale}_30_summary.json"
        final_f = Path(benchmark_root) / f"evaluate_{scale}_final_30_summary.json"
        synced_f = Path(benchmark_root) / f"evaluate_{scale}_synced_30_summary.json"

        if not best_f.exists():
            continue

        d_best = json.load(open(best_f))
        rl_best = d_best["algorithms"]["rl_apc"]["avg_cost"]
        greedy = d_best["algorithms"]["greedy_fr"]["avg_cost"]

        rl_synced = None
        source = "best_model"
        if synced_f.exists():
            d_synced = json.load(open(synced_f))
            rl_synced = d_synced["algorithms"]["rl_apc"]["avg_cost"]
            source = "final_model (synced)"
        elif final_f.exists():
            d_final = json.load(open(final_f))
            rl_synced = d_final["algorithms"]["rl_apc"]["avg_cost"]
            source = "final_model (synced)"

        row = {
            "Scale": scale,
            "Greedy": f"{greedy:.0f}",
            "RL (mismatched)": f"{rl_best:.0f}",
            "RL vs Greedy (mis)": f"{(rl_best-greedy)/greedy*100:+.1f}%",
        }
        if rl_synced is not None:
            row["RL (synced)"] = f"{rl_synced:.0f}"
            row["RL vs Greedy (sync)"] = f"{(rl_synced-greedy)/greedy*100:+.1f}%"
            row["Sync Improvement"] = f"{(rl_best-rl_synced)/rl_best*100:+.1f}%"
        else:
            row["RL (synced)"] = "—"
            row["RL vs Greedy (sync)"] = "—"
            row["Sync Improvement"] = "—"
        rows.append(row)

    csv_path = Path(output_dir) / "table_vecnorm_impact.csv"
    headers = ["Scale", "Greedy", "RL (mismatched)", "RL vs Greedy (mis)",
               "RL (synced)", "RL vs Greedy (sync)", "Sync Improvement"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  vecnorm_impact: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="S,M,L,XL")
    parser.add_argument("--results-root", default="results/rl")
    parser.add_argument("--benchmark-root", default="results/benchmark")
    parser.add_argument("--output-dir", default="results/paper")
    args = parser.parse_args()

    scales = [s.strip() for s in args.scales.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating paper results (VecNormalize-synced)...")
    generate_table1_comparison(scales, args.benchmark_root, args.output_dir)
    generate_table2_detailed(scales, args.benchmark_root, args.output_dir)
    generate_table3_cost_decomposition(scales, args.benchmark_root, args.output_dir)
    generate_wilcoxon_table(scales, args.benchmark_root, args.output_dir)
    generate_training_summary(scales, args.results_root, args.output_dir)
    generate_training_curves(scales, args.results_root, args.output_dir)
    generate_vecnorm_impact_table(args.benchmark_root, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
