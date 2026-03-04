"""Generate publication-quality figures for the EJOR paper.

Produces:
  - Figure 1: Grouped bar chart of individual rule costs across scales
  - Figure 2: Cost distribution boxplots (RL-APC vs baselines vs top rules)
  - Figure 3: Rule selection heatmap (delegated to analyze_rule_selection.py)

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --scales S,M
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "results" / "benchmark"
PAPER = PROJECT_ROOT / "results" / "paper"

# Rule display names matching evaluate_individual_rules.py
RULE_NAMES_SHORT = {
    1: "STTF", 2: "EDD", 3: "MST", 4: "HPF",
    5: "Ch-Urg", 6: "Ch-Low", 7: "Ch-Med", 8: "Ch-High", 9: "Ch-Opp",
    10: "Stby-LC", 11: "Stby-Lz", 12: "Stby-HM",
    13: "Acc-Feas", 14: "Acc-Val", 15: "Ins-MC",
}

RULE_CATEGORIES = {
    "Dispatch": [1, 2, 3, 4, 15],
    "Charge": [5, 6, 7, 8, 9],
    "Standby": [10, 11, 12],
    "Accept": [13, 14],
}

CATEGORY_COLORS = {
    "Dispatch": "#2196F3",
    "Charge": "#FF9800",
    "Standby": "#4CAF50",
    "Accept": "#9C27B0",
}

SCALE_ORDER = ["S", "M", "L", "XL"]


def _load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _pick_best_eval_csv(scale: str) -> Optional[str]:
    candidates = [
        BENCHMARK / f"evaluate_{scale}_v3_30.csv",
        BENCHMARK / f"evaluate_{scale}_synced_30.csv",
        BENCHMARK / f"evaluate_{scale}_v2_30.csv",
        BENCHMARK / f"evaluate_{scale}_30.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _load_individual_rules(scale: str) -> Optional[List[Dict]]:
    path = BENCHMARK / f"individual_rules_{scale}_30.csv"
    if not path.exists():
        return None
    return _load_csv(str(path))


def _get_rule_costs(rows: List[Dict], rule_id: int) -> List[float]:
    return [
        float(r["cost"])
        for r in rows
        if int(r["rule_id"]) == rule_id and r.get("status") == "OK"
    ]


def _get_algo_costs(rows: List[Dict], algo_id: str) -> List[float]:
    return [
        float(r["cost"])
        for r in rows
        if r.get("algorithm_id") == algo_id and r.get("status") == "OK"
    ]


def fig_rule_bar_chart(scales: List[str], output_dir: Path):
    """Figure 1: Grouped bar chart — avg cost per rule per scale.

    Groups rules by category (Dispatch/Charge/Standby/Accept), with
    RL-APC shown as a separate highlighted bar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect data (cost + rejected tasks)
    scale_data = {}
    for scale in scales:
        ir_rows = _load_individual_rules(scale)
        if ir_rows is None:
            print(f"  [skip] no individual_rules_{scale}_30.csv")
            continue
        eval_csv = _pick_best_eval_csv(scale)
        rl_cost = None
        rl_rej = None
        if eval_csv:
            eval_rows = _load_csv(eval_csv)
            rl_costs = _get_algo_costs(eval_rows, "rl_apc")
            if rl_costs:
                rl_cost = np.mean(rl_costs)
            rl_rej_vals = [float(r.get("rejected_tasks", 0)) for r in eval_rows
                           if r.get("algorithm_id") == "rl_apc" and r.get("status") == "OK"]
            if rl_rej_vals:
                rl_rej = np.mean(rl_rej_vals)

        rule_avgs = {}
        rule_rejs = {}
        for rid in range(1, 16):
            ok_rows = [r for r in ir_rows
                       if int(r["rule_id"]) == rid and r.get("status") == "OK"]
            costs = [float(r["cost"]) for r in ok_rows]
            if costs:
                rule_avgs[rid] = np.mean(costs)
                rule_rejs[rid] = np.mean([float(r.get("rejected_tasks", 0)) for r in ok_rows])

        scale_data[scale] = {"rules": rule_avgs, "rejs": rule_rejs,
                             "rl": rl_cost, "rl_rej": rl_rej}

    available_scales = [s for s in scales if s in scale_data]
    if not available_scales:
        print("  [skip] no individual rule data available")
        return

    n_scales = len(available_scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(5 * n_scales, 6), sharey=False)
    if n_scales == 1:
        axes = [axes]

    for ax, scale in zip(axes, available_scales):
        data = scale_data[scale]
        rule_avgs = data["rules"]
        rule_rejs = data["rejs"]
        rl_cost = data["rl"]
        rl_rej = data.get("rl_rej")

        # Sort rule IDs by category order
        rule_ids = list(range(1, 16))
        costs = [rule_avgs.get(rid, 0) for rid in rule_ids]
        rejs = [rule_rejs.get(rid, 0) for rid in rule_ids]
        colors = []
        for rid in rule_ids:
            for cat, rids in RULE_CATEGORIES.items():
                if rid in rids:
                    colors.append(CATEGORY_COLORS[cat])
                    break

        labels = [RULE_NAMES_SHORT.get(rid, str(rid)) for rid in rule_ids]

        bars = ax.bar(range(len(rule_ids)), costs, color=colors, alpha=0.8, width=0.7)

        # Annotate rejection count on bars with high rejection (>=5)
        y_max = max(costs) if costs else 1
        for i, (bar, rej_val) in enumerate(zip(bars, rejs)):
            if rej_val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_max * 0.01,
                        f"R:{rej_val:.0f}", ha="center", va="bottom",
                        fontsize=5.5, color="red", fontweight="bold")

        # Add RL-APC reference line with rejection annotation
        if rl_cost is not None:
            rl_label = f"RL-APC ({rl_cost:,.0f})"
            if rl_rej is not None:
                rl_label += f" [R:{rl_rej:.1f}]"
            ax.axhline(y=rl_cost, color="red", linewidth=2, linestyle="--",
                        label=rl_label)
            ax.legend(fontsize=7, loc="upper right")

        ax.set_xticks(range(len(rule_ids)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Scale {scale}", fontsize=13)
        ax.set_ylabel("Average Cost" if ax == axes[0] else "", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Highlight best rule
        if costs:
            best_idx = np.argmin(costs)
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2)

    # Add category legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Individual Rule Performance Across Scales", fontsize=14, y=1.05)
    fig.tight_layout()

    out_path = output_dir / "fig_rule_bar_chart.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path}")

    # Also save a PDF version for paper
    out_pdf = output_dir / "fig_rule_bar_chart.pdf"
    fig2, axes2 = plt.subplots(1, n_scales, figsize=(5 * n_scales, 6), sharey=False)
    if n_scales == 1:
        axes2 = [axes2]
    for ax, scale in zip(axes2, available_scales):
        data = scale_data[scale]
        rule_avgs = data["rules"]
        rule_rejs = data["rejs"]
        rl_cost = data["rl"]
        rl_rej = data.get("rl_rej")
        rule_ids = list(range(1, 16))
        costs = [rule_avgs.get(rid, 0) for rid in rule_ids]
        rejs = [rule_rejs.get(rid, 0) for rid in rule_ids]
        colors = []
        for rid in rule_ids:
            for cat, rids in RULE_CATEGORIES.items():
                if rid in rids:
                    colors.append(CATEGORY_COLORS[cat])
                    break
        labels = [RULE_NAMES_SHORT.get(rid, str(rid)) for rid in rule_ids]
        bars = ax.bar(range(len(rule_ids)), costs, color=colors, alpha=0.8, width=0.7)
        # Annotate rejection counts
        y_max = max(costs) if costs else 1
        for i, (bar, rej_val) in enumerate(zip(bars, rejs)):
            if rej_val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_max * 0.01,
                        f"R:{rej_val:.0f}", ha="center", va="bottom",
                        fontsize=5.5, color="red", fontweight="bold")
        if rl_cost is not None:
            rl_label = f"RL-APC ({rl_cost:,.0f})"
            if rl_rej is not None:
                rl_label += f" [R:{rl_rej:.1f}]"
            ax.axhline(y=rl_cost, color="red", linewidth=2, linestyle="--",
                        label=rl_label)
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xticks(range(len(rule_ids)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Scale {scale}", fontsize=13)
        ax.set_ylabel("Average Cost" if ax == axes2[0] else "", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if costs:
            best_idx = np.argmin(costs)
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2)
    fig2.legend(handles=legend_elements, loc="upper center", ncol=4,
                fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig2.suptitle("Individual Rule Performance Across Scales", fontsize=14, y=1.05)
    fig2.tight_layout()
    fig2.savefig(str(out_pdf), dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Figure (PDF): {out_pdf}")


def fig_cost_boxplots(scales: List[str], output_dir: Path):
    """Figure 2: Cost distribution boxplots — RL-APC vs baselines vs top rules.

    For each scale, shows boxplots comparing RL-APC, Greedy-FR, and the
    top-3 individual rules.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available_scales = []
    scale_boxdata = {}

    for scale in scales:
        eval_csv = _pick_best_eval_csv(scale)
        if not eval_csv:
            continue
        eval_rows = _load_csv(eval_csv)

        methods = {}
        rl_costs = _get_algo_costs(eval_rows, "rl_apc")
        if rl_costs:
            methods["RL-APC"] = rl_costs
        gr_costs = _get_algo_costs(eval_rows, "greedy_fr")
        if gr_costs:
            methods["Greedy-FR"] = gr_costs

        # Add top-3 individual rules
        ir_rows = _load_individual_rules(scale)
        if ir_rows:
            rule_means = {}
            for rid in range(1, 16):
                costs = _get_rule_costs(ir_rows, rid)
                if costs:
                    rule_means[rid] = (np.mean(costs), costs)

            top3 = sorted(rule_means.items(), key=lambda x: x[1][0])[:3]
            for rid, (_, costs) in top3:
                name = RULE_NAMES_SHORT.get(rid, f"R{rid}")
                methods[name] = costs

        if methods:
            scale_boxdata[scale] = methods
            available_scales.append(scale)

    if not available_scales:
        print("  [skip] no data for boxplots")
        return

    n = len(available_scales)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, scale in zip(axes, available_scales):
        methods = scale_boxdata[scale]
        labels = list(methods.keys())
        data = [methods[l] for l in labels]

        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)

        # Color RL-APC red, Greedy blue, rules gray
        for i, (patch, label) in enumerate(zip(bp["boxes"], labels)):
            if label == "RL-APC":
                patch.set_facecolor("#EF5350")
                patch.set_alpha(0.7)
            elif label == "Greedy-FR":
                patch.set_facecolor("#42A5F5")
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor("#BDBDBD")
                patch.set_alpha(0.7)

        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Scale {scale}", fontsize=12)
        ax.set_ylabel("Total Cost" if ax == axes[0] else "", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Cost Distribution: RL-APC vs Baselines and Top Rules", fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = output_dir / "fig_cost_boxplots.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path}")

    out_pdf = output_dir / "fig_cost_boxplots.pdf"
    # Re-create for PDF
    fig2, axes2 = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes2 = [axes2]
    for ax, scale in zip(axes2, available_scales):
        methods = scale_boxdata[scale]
        labels = list(methods.keys())
        data = [methods[l] for l in labels]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
        for i, (patch, label) in enumerate(zip(bp["boxes"], labels)):
            if label == "RL-APC":
                patch.set_facecolor("#EF5350")
                patch.set_alpha(0.7)
            elif label == "Greedy-FR":
                patch.set_facecolor("#42A5F5")
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor("#BDBDBD")
                patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Scale {scale}", fontsize=12)
        ax.set_ylabel("Total Cost" if ax == axes2[0] else "", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig2.suptitle("Cost Distribution: RL-APC vs Baselines and Top Rules", fontsize=13, y=1.02)
    fig2.tight_layout()
    fig2.savefig(str(out_pdf), dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Figure (PDF): {out_pdf}")


def fig_training_curves_enhanced(scales: List[str], output_dir: Path):
    """Enhanced training curves with std bands and best_model markers.

    Includes v2/v3 variants for L and XL if available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_root = PROJECT_ROOT / "results" / "rl"
    colors = {"S": "#2196F3", "M": "#4CAF50", "L": "#FF9800", "XL": "#F44336"}
    linestyles = {"v1": "-", "v2": "--", "v3": ":"}

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    for scale in scales:
        # Try multiple training versions
        versions = [
            (f"train_{scale}", "v1"),
            (f"train_{scale}_v2", "v2"),
            (f"train_{scale}_v3", "v3"),
        ]
        for train_dir, version in versions:
            npz_path = results_root / train_dir / "eval_logs" / "evaluations.npz"
            if not npz_path.exists():
                continue

            data = np.load(str(npz_path))
            ts = data["timesteps"] / 1e6
            results = data["results"]  # shape: [n_evals, n_episodes]
            mean_rew = results.mean(axis=1)
            std_rew = results.std(axis=1)

            color = colors.get(scale, "gray")
            ls = linestyles.get(version, "-")
            label = f"{scale}" if version == "v1" else f"{scale}-{version}"

            # For L/XL, only show the latest version to avoid clutter
            if scale in ("L", "XL") and version == "v1":
                # Check if v2 or v3 exists
                has_newer = any(
                    (results_root / f"train_{scale}_{v}" / "eval_logs" / "evaluations.npz").exists()
                    for v in ["v2", "v3"]
                )
                if has_newer:
                    continue

            ax.plot(ts, mean_rew, label=label, color=color, linewidth=2, linestyle=ls)
            ax.fill_between(ts, mean_rew - std_rew, mean_rew + std_rew,
                            color=color, alpha=0.15)

            # Mark best model point
            best_idx = mean_rew.argmax()
            ax.scatter([ts[best_idx]], [mean_rew[best_idx]], color=color,
                       s=100, zorder=5, edgecolors="black", linewidths=1,
                       marker="*")

            plotted = True

    if not plotted:
        print("  [skip] no training data found")
        return

    ax.set_xlabel("Training Steps (millions)", fontsize=12)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=12)
    ax.set_title("RL-APC Training Convergence", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        out_path = output_dir / f"fig_training_curves.{ext}"
        dpi = 150 if ext == "png" else 300
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
        print(f"  Figure: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate EJOR paper figures.")
    parser.add_argument("--scales", default="S,M,L,XL")
    parser.add_argument("--output-dir", default="results/paper")
    args = parser.parse_args()

    scales = [s.strip() for s in args.scales.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating EJOR paper figures...")

    print("\n[1/3] Rule performance bar chart")
    fig_rule_bar_chart(scales, output_dir)

    print("\n[2/3] Cost distribution boxplots")
    fig_cost_boxplots(scales, output_dir)

    print("\n[3/3] Enhanced training curves")
    fig_training_curves_enhanced(scales, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
