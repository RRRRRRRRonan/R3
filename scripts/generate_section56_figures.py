"""Generate EJOR-quality figures for Section 5.6 (Robustness & Sensitivity).

Produces:
  - Figure A: Training convergence curves (2×2 layout, one panel per scale)
  - Figure B: Cost distribution boxplots (2×2 layout, RL vs Best Rule vs Greedy)

All costs use Option B: Total = Oper + Reject + Terminal.

Usage:
    python scripts/generate_section56_figures.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Font: STIX (≈ Times New Roman, standard for academic journals) ────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "results" / "benchmark"
RESULTS_RL = PROJECT_ROOT / "results" / "rl"
PAPER = PROJECT_ROOT / "results" / "paper"
OUT_DIR = PROJECT_ROOT / "docs" / "ejor" / "section5.6"

# ── Cost coefficients for Option B clean cost ─────────────────────────────
REJECTION_PENALTY = 10000
C_TR = 1.0; C_TIME = 0.1; C_CH = 0.6; C_DELAY = 2.0
C_WAIT = 0.05; C_CONFLICT = 0.05; C_STANDBY = 0.05
C_TERMINAL_PER_SCALE = {"S": 3000, "M": 2500, "L": 2000, "XL": 1500}

SCALE_ORDER = ["S", "M", "L", "XL"]

# Best rule per scale (from Table 4, determined by original cost ranking)
BEST_RULE_PER_SCALE = {
    "S": (11, "Standby-Lazy"),
    "M": (11, "Standby-Lazy"),
    "L": (8, "Charge-High"),
    "XL": (11, "Standby-Lazy"),
}

# Second-best rule per scale (fallback when best rule = Greedy-FR)
SECOND_BEST_RULE = {
    "S": (8, "Charge-High"),
    "M": (8, "Charge-High"),
    "L": (11, "Standby-Lazy"),  # L: best=Charge-High=Greedy-FR, so use 2nd
    "XL": (15, "Insert-MC"),
}

# Preferred eval CSV per scale (newest first)
EVAL_CSV_PRIORITY = {
    "S": ["evaluate_S_30.csv"],
    "M": ["evaluate_M_synced_30.csv", "evaluate_M_30.csv"],
    "L": ["evaluate_L_v3_30.csv", "evaluate_L_v2_30.csv", "evaluate_L_30.csv"],
    "XL": ["evaluate_XL_synced_30.csv", "evaluate_XL_30.csv"],
}

# Training data: which npz to load for each scale
TRAIN_NPZ_PRIORITY = {
    "S": ["train_S/eval_logs/evaluations.npz"],
    "M": ["train_M/eval_logs/evaluations.npz"],
    "L": ["train_L_v4/eval_logs/evaluations.npz",
           "train_L_v3/eval_logs/evaluations.npz",
           "train_L/eval_logs/evaluations.npz"],
    "XL": ["train_XL/eval_logs/evaluations.npz"],
}


def _clean_cost_row(r: Dict, scale: str) -> float:
    """Compute Option B clean cost from a CSV row."""
    travel = float(r.get("metrics_total_distance", 0) or 0)
    travel_time = float(r.get("metrics_total_travel_time", 0) or 0)
    charging = float(r.get("metrics_total_charging", 0) or 0)
    delay = float(r.get("metrics_total_delay", 0) or 0)
    waiting = float(r.get("metrics_total_waiting", 0) or 0)
    conflict = float(r.get("metrics_total_conflict_waiting", 0) or 0)
    standby = float(r.get("metrics_total_standby", 0) or 0)
    rejected = float(r.get("rejected_tasks", 0) or 0)
    num_tasks = float(r.get("num_tasks_manifest", 0) or 0)
    completed = float(r.get("completed_tasks", 0) or 0)

    w_travel = C_TR * travel + C_TIME * travel_time
    w_charging = C_CH * charging
    w_tardiness = C_DELAY * delay
    w_idle = C_WAIT * waiting + C_CONFLICT * (conflict + waiting) + C_STANDBY * standby
    w_oper = w_travel + w_charging + w_tardiness + w_idle
    w_rejection = REJECTION_PENALTY * rejected
    unfinished = max(0, num_tasks - completed - rejected)
    w_terminal = C_TERMINAL_PER_SCALE.get(scale, 1000) * unfinished
    return w_oper + w_rejection + w_terminal


def _load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _pick_eval_csv(scale: str) -> Optional[Path]:
    for name in EVAL_CSV_PRIORITY.get(scale, []):
        p = BENCHMARK / name
        if p.exists():
            return p
    return None


def _pick_train_npz(scale: str) -> Optional[Path]:
    for name in TRAIN_NPZ_PRIORITY.get(scale, []):
        p = RESULTS_RL / name
        if p.exists():
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════
# Figure A: Training Convergence (2×2)
# ══════════════════════════════════════════════════════════════════════════

def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Exponential moving average smoothing."""
    if len(arr) <= window:
        return arr
    alpha = 2.0 / (window + 1)
    result = np.empty_like(arr, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _get_greedy_mean_return(scale: str) -> Optional[float]:
    """Get Greedy-FR mean return from evaluation CSV (as negative cost proxy)."""
    eval_path = _pick_eval_csv(scale)
    if eval_path is None:
        return None
    rows = _load_csv(str(eval_path))
    costs = [float(r["cost"]) for r in rows
             if r.get("algorithm_id") == "greedy_fr" and r.get("status") == "OK"]
    if not costs:
        return None
    # Return is approximately negative cost (reward = -cost + shaping)
    # Use raw cost as an approximate reference
    return -np.mean(costs)


def fig_training_curves():
    """2×2 training curves — one panel per scale, smoothed, with Greedy baseline."""
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.flatten()

    scale_colors = {"S": "#1976D2", "M": "#388E3C", "L": "#F57C00", "XL": "#D32F2F"}
    # Adaptive smoothing: more data points → lighter smoothing
    SMOOTH_WINDOWS = {"S": 5, "M": 5, "L": 7, "XL": 3}

    for idx, scale in enumerate(SCALE_ORDER):
        ax = axes[idx]
        npz_path = _pick_train_npz(scale)
        if npz_path is None:
            ax.text(0.5, 0.5, f"No training data\nfor {scale}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray")
            ax.set_title(f"({chr(97+idx)}) Scale {scale}", fontsize=12, fontweight="bold")
            continue

        data = np.load(str(npz_path))
        ts = data["timesteps"]
        results = data["results"]  # [n_evals, n_episodes]
        mean_rew = results.mean(axis=1)
        std_rew = results.std(axis=1)

        color = scale_colors[scale]
        win = SMOOTH_WINDOWS.get(scale, 5)
        n_evals = len(ts)

        if n_evals >= 15:
            # Enough points: scatter raw + smooth curve
            ax.scatter(ts, mean_rew, color=color, s=12, alpha=0.35, zorder=2,
                       edgecolors="none")
            smooth_mean = _smooth(mean_rew, win)
            smooth_lo = _smooth(mean_rew - std_rew, win)
            smooth_hi = _smooth(mean_rew + std_rew, win)
            ax.plot(ts, smooth_mean, color=color, linewidth=2.2, zorder=4)
            ax.fill_between(ts, smooth_lo, smooth_hi,
                            color=color, alpha=0.10, zorder=1)
        else:
            # Sparse evals (e.g. XL=10): connected markers, no fake smoothing
            ax.plot(ts, mean_rew, color=color, linewidth=1.5, zorder=3,
                    marker="o", markersize=5, markeredgecolor="white",
                    markeredgewidth=0.6)
            ax.fill_between(ts, mean_rew - std_rew, mean_rew + std_rew,
                            color=color, alpha=0.10, zorder=1)
            # Note on panel
            ax.text(0.97, 0.03, f"n={n_evals} evals",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="gray", fontstyle="italic")

        # Greedy-FR reference line (approximate: negative of raw cost)
        greedy_ret = _get_greedy_mean_return(scale)
        if greedy_ret is not None:
            ax.axhline(y=greedy_ret, color="gray", linewidth=1.2,
                       linestyle="--", alpha=0.7, zorder=3)
            # Place label on right side
            ax.text(ts[-1], greedy_ret, " Greedy-FR",
                    va="bottom", ha="right", fontsize=7.5,
                    color="gray", fontstyle="italic")

        # Mark best checkpoint (on raw data)
        best_idx = mean_rew.argmax()
        best_step = ts[best_idx]
        best_val = mean_rew[best_idx]
        ax.scatter([best_step], [best_val], color=color, s=100, zorder=6,
                   edgecolors="black", linewidths=1.0, marker="*")
        # Annotate best — adaptive placement
        offset_y = 12 if best_val < np.median(mean_rew) else -18
        ax.annotate(f"Best: {best_step/1e3:.0f}K",
                    xy=(best_step, best_val),
                    xytext=(12, offset_y), textcoords="offset points",
                    fontsize=8.5, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=color, alpha=0.8, lw=0.5))

        ax.set_title(f"({chr(97+idx)}) Scale {scale}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Steps", fontsize=10)
        ax.set_ylabel("Mean Evaluation Return", fontsize=10)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.tick_params(labelsize=9)

        # Format x-axis as "100K", "500K", "1M"
        def step_formatter(x, pos):
            if x >= 1e6:
                return f"{x/1e6:.1f}M"
            elif x >= 1e3:
                return f"{x/1e3:.0f}K"
            return f"{x:.0f}"
        ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))

    fig.suptitle("RL-APC Training Convergence", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Caption note at bottom
    fig.text(0.5, 0.005,
             "Solid line: EMA-smoothed mean return; dots: raw evaluation mean; "
             "shaded: smoothed ±1 std.  Star: best checkpoint selected for deployment.  "
             "Dashed grey: Greedy-FR return (approximate baseline).  "
             "Higher return = lower simulation cost.",
             ha="center", fontsize=7.5, fontstyle="italic", color="gray")

    for ext, dpi in [("png", 300), ("pdf", 300)]:
        out = OUT_DIR / f"fig_training_curves.{ext}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure B: Cost Distribution Boxplots (2×2)
# ══════════════════════════════════════════════════════════════════════════

def fig_cost_boxplots():
    """2×2 boxplots — RL-APC vs Greedy-FR vs Best Rule, Option B clean cost."""
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Consistent colour scheme
    METHOD_COLORS = {
        "RL-APC": "#E53935",      # red
        "Greedy-FR": "#1E88E5",   # blue
        "best_rule": "#43A047",   # green
    }

    for idx, scale in enumerate(SCALE_ORDER):
        ax = axes[idx]
        eval_path = _pick_eval_csv(scale)
        if eval_path is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(f"({chr(97+idx)}) Scale {scale}", fontsize=12, fontweight="bold")
            continue

        eval_rows = _load_csv(str(eval_path))

        # RL-APC costs
        rl_costs = [_clean_cost_row(r, scale) for r in eval_rows
                     if r.get("algorithm_id") == "rl_apc" and r.get("status") == "OK"]

        # Greedy-FR costs
        gr_costs = [_clean_cost_row(r, scale) for r in eval_rows
                     if r.get("algorithm_id") == "greedy_fr" and r.get("status") == "OK"]

        # Best rule costs (from individual_rules CSV)
        best_rid, best_name = BEST_RULE_PER_SCALE[scale]
        ir_path = BENCHMARK / f"individual_rules_{scale}_30.csv"
        br_costs = []
        br_label = best_name

        if ir_path.exists():
            ir_rows = _load_csv(str(ir_path))
            br_costs = [_clean_cost_row(r, scale) for r in ir_rows
                        if int(r["rule_id"]) == best_rid and r.get("status") == "OK"]

            # If best rule ≈ Greedy-FR (L-scale: Charge-High=Greedy-FR),
            # keep the best rule but mark with dagger to signal equivalence
            if gr_costs and br_costs:
                gr_mean = np.mean(gr_costs)
                br_mean = np.mean(br_costs)
                if abs(gr_mean - br_mean) / max(gr_mean, 1) < 0.02:
                    br_label = best_name + "*"

        # Build boxplot data
        box_data = []
        box_labels = []
        box_colors = []

        if rl_costs:
            box_data.append(rl_costs)
            box_labels.append("RL-APC")
            box_colors.append(METHOD_COLORS["RL-APC"])
        if gr_costs:
            box_data.append(gr_costs)
            box_labels.append("Greedy-FR")
            box_colors.append(METHOD_COLORS["Greedy-FR"])
        if br_costs:
            box_data.append(br_costs)
            box_labels.append(br_label)
            box_colors.append(METHOD_COLORS["best_rule"])

        if not box_data:
            continue

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                        showfliers=True, flierprops=dict(marker="o", markersize=4,
                                                          markerfacecolor="gray",
                                                          markeredgecolor="gray",
                                                          alpha=0.5))

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(0.8)
        for line in bp["medians"]:
            line.set_color("black")
            line.set_linewidth(1.2)

        ax.set_xticklabels(box_labels, fontsize=9.5)
        ax.set_title(f"({chr(97+idx)}) Scale {scale}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Total Cost" if idx % 2 == 0 else "", fontsize=10)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=9)

        # Add mean markers
        for i, data_i in enumerate(box_data):
            mean_val = np.mean(data_i)
            ax.scatter([i + 1], [mean_val], color="black", marker="D",
                       s=25, zorder=5, linewidths=0.5)

        # Format y-axis with comma separators
        ax.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Shared legend
    legend_elements = [
        Patch(facecolor=METHOD_COLORS["RL-APC"], edgecolor="black",
              alpha=0.6, label="RL-APC"),
        Patch(facecolor=METHOD_COLORS["Greedy-FR"], edgecolor="black",
              alpha=0.6, label="Greedy-FR"),
        Patch(facecolor=METHOD_COLORS["best_rule"], edgecolor="black",
              alpha=0.6, label="Best Fixed Rule"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 0.99),
               frameon=True, edgecolor="gray")

    fig.suptitle("Cost Distribution: RL-APC vs Baselines (Option B, 30 Test Instances)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    fig.text(0.5, 0.005,
             "Total Cost = Operational + Rejection + Terminal (Option B).  Diamond = mean.  "
             "Third method: best fixed rule per scale.  "
             "*On Scale L, Charge-High and Greedy-FR are equivalent; "
             "both are shown for completeness.",
             ha="center", fontsize=7.5, fontstyle="italic", color="gray")

    for ext, dpi in [("png", 300), ("pdf", 300)]:
        out = OUT_DIR / f"fig_cost_boxplots.{ext}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Section 5.6 Figures — EJOR Quality")
    print("=" * 60)

    print("\n[1/2] Training convergence curves (2×2)")
    fig_training_curves()

    print("\n[2/2] Cost distribution boxplots (2×2)")
    fig_cost_boxplots()

    print("\nDone!")
