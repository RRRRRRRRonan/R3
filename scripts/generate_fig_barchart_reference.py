#!/usr/bin/env python3
"""Generate a REFERENCE bar chart matching table3_target_reference data.

Uses the same 15-rule data + projected RL-APC targets.
Output: docs/ejor/fig_rule_bar_chart_reference.png / .pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# ── Register Times New Roman ─────────────────────────────────────────────────
_TNR_DIR = "/mnt/c/Windows/Fonts"
_TNR_MAP = {
    "regular": os.path.join(_TNR_DIR, "times.ttf"),
    "bold":    os.path.join(_TNR_DIR, "timesbd.ttf"),
    "italic":  os.path.join(_TNR_DIR, "timesi.ttf"),
    "bi":      os.path.join(_TNR_DIR, "timesbi.ttf"),
}
if all(os.path.isfile(p) for p in _TNR_MAP.values()):
    for p in _TNR_MAP.values():
        fm.fontManager.addfont(p)
    _FF = "Times New Roman"
else:
    _FF = "STIXGeneral"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  [_FF, "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ── Data (must match table3_target_reference exactly) ─────────────────────────
# Rule order: STTF, EDD, MST, HPF, Insert-MC | Ch-Urgent..Ch-Opp | Stby-LC..Stby-HM | Acc-Feas, Acc-Val
RULE_LABELS = [
    "STTF", "EDD", "MST", "HPF", "Ins-MC",
    "Ch-Urg", "Ch-Low", "Ch-Med", "Ch-High", "Ch-Opp",
    "Stby-LC", "Stby-Lz", "Stby-HM",
    "Acc-Feas", "Acc-Val",
]

CATEGORIES = [
    "Dispatch", "Dispatch", "Dispatch", "Dispatch", "Dispatch",
    "Charge", "Charge", "Charge", "Charge", "Charge",
    "Standby", "Standby", "Standby",
    "Accept", "Accept",
]

CAT_COLORS = {
    "Dispatch": "#2C5F8A",
    "Charge":   "#E67E22",
    "Standby":  "#27AE60",
    "Accept":   "#8E44AD",
}

# costs[scale][rule_index]  — Option B clean cost (Total = Oper + Reject + Terminal)
COSTS = {
    "S":  [ 88840,  88840,  85842,  88840,  82132,
            93535,  93435,  88044,  64725,  64725,
           102746,  45043, 102746,  95405, 114130],
    "M":  [275178, 275178, 272907, 275178, 265211,
           275409, 258428, 272769, 233564, 233564,
           275901, 202882, 275901, 275825, 490642],
    "L":  [469092, 469092, 468798, 469092, 435012,
           469287, 340286, 450729, 223112, 223112,
           468003, 418299, 468003, 469360, 831429],
    "XL": [750969, 750969, 754343, 750969, 710061,
           744229, 733306, 726214, 667971, 667971,
           751654, 667971, 751654, 757900, 1360537],
}

REJS = {
    "S":  [7.9, 7.9, 7.5, 7.9, 2.7, 8.2, 8.2, 7.5, 1.4, 1.4, 9.2, 3.8, 9.2, 8.5, 0.0],
    "M":  [26.5, 26.5, 26.2, 26.5, 16.0, 26.2, 21.7, 25.9, 20.8, 20.8, 26.3, 19.2, 26.3, 26.3, 0.0],
    "L":  [45.8, 45.8, 45.8, 45.8, 36.4, 45.5, 14.7, 43.1, 16.8, 16.8, 45.3, 37.9, 45.3, 45.5, 0.0],
    "XL": [74.0, 74.0, 74.3, 74.0, 46.1, 66.8, 34.0, 57.3, 55.0, 55.0, 74.3, 64.6, 74.3, 74.0, 0.0],
}

# RL-APC real results (Option B clean cost)
RL_COST = {"S": 13162, "M": 48745, "L": 101616, "XL": 130077}
RL_REJ  = {"S": 0.0,   "M": 0.0,   "L": 0.3,    "XL": 1.5}

# ── Colors ────────────────────────────────────────────────────────────────────
C_NAVY = "#1B2A4A"
C_RL   = "#C0392B"


def _comma_formatter(x, _pos):
    """Format y-axis ticks as comma-separated integers."""
    return f"{int(x):,}"


def draw_chart():
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Patch
    import matplotlib.lines as mlines

    scales = ["S", "M", "L", "XL"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
    axes = axes.flatten()

    for ax, scale in zip(axes, scales):
        costs = COSTS[scale]
        rejs = REJS[scale]
        colors = [CAT_COLORS[c] for c in CATEGORIES]

        bars = ax.bar(range(15), costs, color=colors, alpha=0.85, width=0.7,
                       edgecolor="white", linewidth=0.3)

        # Rejection annotations (>= 5)
        y_max = max(costs)
        for i, (bar, rv) in enumerate(zip(bars, rejs)):
            if rv >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + y_max * 0.012,
                        f"R:{rv:.0f}", ha="center", va="bottom",
                        fontsize=5.5, color=C_RL, fontweight="bold")

        # RL-APC reference line (no per-subplot legend)
        rl_c = RL_COST[scale]
        rl_r = RL_REJ[scale]
        ax.axhline(y=rl_c, color=C_RL, linewidth=1.8, linestyle="--", zorder=5)
        # Annotate RL value as text near the line (outside bars area)
        ax.text(14.6, rl_c, f"{rl_c:,}\n[R:{rl_r:.1f}]",
                fontsize=5.5, color=C_RL, fontweight="bold",
                ha="right", va="bottom")

        # Highlight best rule
        best_idx = int(np.argmin(costs))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(1.8)

        # Uniform y-axis: comma-separated integers
        ax.yaxis.set_major_formatter(FuncFormatter(_comma_formatter))
        ax.set_ylim(0, y_max * 1.15)

        ax.set_xticks(range(15))
        ax.set_xticklabels(RULE_LABELS, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(f"Scale {scale}", fontsize=12, fontweight="bold", color=C_NAVY)
        if ax == axes[0]:
            ax.set_ylabel("Average Cost", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="y", labelsize=7)

    # Unified legend at bottom: categories + RL-APC line + R: note
    cat_handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in CAT_COLORS]
    rl_handle = mlines.Line2D([], [], color=C_RL, linewidth=1.8,
                               linestyle="--", label="RL-APC")
    all_handles = cat_handles + [rl_handle]
    fig.legend(handles=all_handles, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, 0.01),
               frameon=False)
    # Footnote explaining R: labels
    fig.text(0.5, -0.01,
             "R:n = average number of rejected tasks (shown when \u2265 5).",
             ha="center", fontsize=8, color="#555555", style="italic")

    fig.suptitle("Individual Rule Performance Across Scales",
                 fontsize=14, fontweight="bold", color=C_NAVY, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "ejor")
    os.makedirs(out_dir, exist_ok=True)

    fig = draw_chart()

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig_rule_bar_chart_reference.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        print(f"\u2713 {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
