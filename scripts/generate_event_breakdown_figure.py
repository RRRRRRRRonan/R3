"""Generate event-type breakdown stacked bar chart for Section 5.4.

Shows which rules RL-APC selects under different event types (TASK_ARRIVAL,
ROBOT_IDLE, DEADLOCK_RISK, CHARGE_DONE, SOC_LOW), with 4 scale panels.

Each of the 15 rules is shown individually, colored by category
(Dispatch / Charge / Standby / Accept). Dominant segments (>8%) are
annotated with rule name + percentage for readability.

Usage:
    python scripts/generate_event_breakdown_figure.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "results" / "paper"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "ejor" / "section5.4"

SCALES = ["S", "M", "L", "XL"]
EVENT_ORDER = ["TASK_ARRIVAL", "ROBOT_IDLE", "DEADLOCK_RISK", "CHARGE_DONE", "SOC_LOW"]
EVENT_LABELS = {
    "TASK_ARRIVAL": "Task\nArrival",
    "ROBOT_IDLE": "Robot\nIdle",
    "DEADLOCK_RISK": "Deadlock\nRisk",
    "CHARGE_DONE": "Charge\nDone",
    "SOC_LOW": "SOC\nLow",
}

# ── 15 rules: id → name ──
RULE_NAMES = {
    1: "STTF", 2: "EDD", 3: "MST", 4: "HPF",
    5: "Charge-Urgent", 6: "Charge-Low", 7: "Charge-Med",
    8: "Charge-High", 9: "Charge-Opp",
    10: "Standby-LowCost", 11: "Standby-Lazy", 12: "Standby-Heatmap",
    13: "Accept-Feasible", 14: "Accept-Feasible",
    15: "Insert-MinCost",
}

# ── Category membership ──
CATEGORIES = {
    "Dispatch":  [1, 2, 3, 4, 15],
    "Charge":    [5, 6, 7, 8, 9],
    "Standby":   [10, 11, 12],
    "Accept":    [13, 14],
}

# ── Category base colors (used to generate per-rule shades) ──
# Dispatch = grey family, Charge = orange family, Standby = blue family, Accept = red family
CATEGORY_PALETTES = {
    "Dispatch": ["#bdbdbd", "#969696", "#737373", "#525252", "#d9d9d9"],
    "Charge":   ["#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#d95f0e"],
    "Standby":  ["#9ecae1", "#4292c6", "#08519c"],
    "Accept":   ["#fc9272", "#de2d26"],
}

# Build per-rule color map
RULE_COLORS = {}
for cat, rids in CATEGORIES.items():
    palette = CATEGORY_PALETTES[cat]
    for i, rid in enumerate(rids):
        RULE_COLORS[rid] = palette[i % len(palette)]

# Stacking order: bottom → top = Dispatch, Charge, Standby, Accept
# Within each category: by rule_id ascending
STACK_ORDER = (
    CATEGORIES["Dispatch"] +
    CATEGORIES["Charge"] +
    CATEGORIES["Standby"] +
    CATEGORIES["Accept"]
)


def load_event_breakdown(scale: str) -> dict:
    """Load event breakdown CSV → {event: {rule_id: frequency}}."""
    path = PAPER_DIR / f"rule_event_breakdown_{scale}.csv"
    if not path.exists():
        return {}

    raw = defaultdict(lambda: defaultdict(int))
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = row["event_type"]
            rid = int(row["rule_id"])
            count = int(row["count"])
            raw[event][rid] += count

    result = {}
    for event in EVENT_ORDER:
        if event not in raw:
            result[event] = {}
            continue
        event_total = sum(raw[event].values())
        freqs = {}
        for rid in STACK_ORDER:
            cnt = raw[event].get(rid, 0)
            freqs[rid] = cnt / event_total if event_total > 0 else 0.0
        result[event] = freqs
    return result


def _short_name(rid: int) -> str:
    """Abbreviated name for annotation."""
    mapping = {
        1: "STTF", 2: "EDD", 3: "MST", 4: "HPF",
        5: "Ch-Urg", 6: "Ch-Low", 7: "Ch-Med",
        8: "Ch-High", 9: "Ch-Opp",
        10: "Sb-LC", 11: "Sb-Lazy", 12: "Sb-Heat",
        13: "Ac-Feas", 14: "Ac-Feas",
        15: "Ins-MC",
    }
    return mapping.get(rid, f"R{rid}")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=False)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, scale in zip(axes_flat, SCALES):
        data = load_event_breakdown(scale)
        if not data:
            ax.set_title(f"{scale}-scale (no data)")
            continue

        events = [e for e in EVENT_ORDER if e in data]
        x = np.arange(len(events))
        bar_width = 0.62

        bottom = np.zeros(len(events))
        for rid in STACK_ORDER:
            vals = np.array([data[e].get(rid, 0.0) for e in events])
            if vals.sum() < 1e-6:
                continue
            ax.bar(x, vals, bar_width, bottom=bottom,
                   color=RULE_COLORS[rid], edgecolor="white", linewidth=0.4)

            # Annotate segments >= 10%
            for i, v in enumerate(vals):
                if v >= 0.10:
                    cy = bottom[i] + v / 2
                    # For large segments show name+%, for medium just %
                    if v >= 0.20:
                        label = f"{_short_name(rid)}\n{v:.0%}"
                        fs = 7.0
                    else:
                        label = f"{_short_name(rid)} {v:.0%}"
                        fs = 6.0
                    txt_color = "white" if v > 0.25 else "black"
                    ax.text(x[i], cy, label, ha="center", va="center",
                            fontsize=fs, fontweight="bold", color=txt_color,
                            linespacing=0.9)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([EVENT_LABELS.get(e, e) for e in events], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Selection Frequency" if scale in ("S", "L") else "")
        ax.set_title(f"{scale}-scale", fontweight="bold", fontsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8.5)

    # ── Legend: one patch per category (not per rule) for cleanliness ──
    # Plus key individual rules as sub-items
    legend_entries = []

    # Category-level patches
    cat_representative = {
        "Dispatch": (RULE_COLORS[1], "Dispatch (STTF/EDD/MST/HPF/Ins)"),
        "Charge-Opp": (RULE_COLORS[9], "Charge-Opp"),
        "Charge-Other": (RULE_COLORS[7], "Charge-Other (Urg/Low/Med/High)"),
        "Standby-Lazy": (RULE_COLORS[11], "Standby-Lazy"),
        "Standby-Other": (RULE_COLORS[10], "Standby-Other (LowCost/Heatmap)"),
        "Accept-Feasible": (RULE_COLORS[14], "Accept-Feasible"),
    }

    for key in ["Dispatch", "Charge-Opp", "Charge-Other",
                "Standby-Lazy", "Standby-Other", "Accept-Feasible"]:
        color, label = cat_representative[key]
        legend_entries.append(mpatches.Patch(facecolor=color, edgecolor="white",
                                             linewidth=0.5, label=label))

    fig.legend(handles=legend_entries, loc="upper center", ncol=4,
               fontsize=8.5, frameon=True, fancybox=True, framealpha=0.95,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, 1.005),
               handlelength=1.2, handletextpad=0.5, columnspacing=1.5)

    fig.suptitle("RL-APC Rule Selection by Event Type",
                 fontsize=14, fontweight="bold", y=1.045)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        out = OUTPUT_DIR / f"fig_event_breakdown.{fmt}"
        fig.savefig(str(out), dpi=250, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
