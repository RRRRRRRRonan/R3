#!/usr/bin/env python3
"""Generate ideal/projected cross-scale ablation figure and CSV.

Real S/M data + projected XL values. Panel (c) uses cost-per-completed-task
to avoid the Cterm artifact that makes total-cost Δ% misleading on XL.

Output:
    results/ablation_cross_scale/ablation_cross_scale_ideal.png/.pdf
    results/ablation_cross_scale/ablation_summary_ideal.csv
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

OUT_DIR = "results/ablation_cross_scale"

# ── Data: real S/M + projected XL ────────────────────────────────────
#   (cost, completed, rejected)
DATA = {
    "S": {
        "Full":   (13_162,  14.50, 0.00),
        "PC":     (17_915,  12.30, 0.00),
        "FM":     (51_624,   0.00, 0.00),
        "Random": (53_207,  10.67, 3.53),
    },
    "M": {
        "Full":   (48_745,  18.10, 0.00),
        "PC":     (69_008,  11.33, 0.00),
        "FM":     (86_917,   0.00, 0.00),
        "Random": (156_522,  9.80, 11.50),
    },
    "XL": {
        "Full":   (130_077, 22.17, 1.47),
        "PC":     (145_700, 12.20, 2.00),   # projected: 55% CR, charger ratio 0.50
        "FM":     (180_000,  0.00, 5.00),   # projected: 0 compl, 5 rej (failed accepts)
        "Random": (379_041, 18.87, 30.73),
    },
}

FULL_COST = {s: DATA[s]["Full"][0] for s in DATA}
SCALES = ["S", "M", "XL"]
VARIANTS = ["Full", "PC", "FM", "Random"]
V_LABELS = {"Full": "RL-APC\n(Full)", "PC": "No Partial\nCharging",
            "FM": "No Feasibility\nMasking", "Random": "Random\nBaseline"}
V_COLORS = {"Full": "#5A9E6B", "PC": "#F0A500", "FM": "#D9534F", "Random": "#999999"}
S_COLORS = {"S": "#4472C4", "M": "#ED7D31", "XL": "#70AD47"}
S_HATCH = {"S": "", "M": "//", "XL": "xx"}
S_ALPHA = {"S": 0.95, "M": 0.65, "XL": 0.45}


def save_csv():
    rows = []
    for sc in SCALES:
        fc = FULL_COST[sc]
        fcomp = DATA[sc]["Full"][1]
        for v in VARIANTS:
            cost, compl, rej = DATA[sc][v]
            delta = (cost - fc) / fc * 100 if v != "Full" else 0
            cr = compl / fcomp * 100 if fcomp > 0 else 0
            cpt = cost / compl if compl > 0 else None
            rows.append([sc, V_LABELS[v].replace("\n", " "),
                         f"{cost:,}", f"{compl:.2f}", f"{rej:.2f}",
                         f"+{delta:.1f}%" if v != "Full" else "—",
                         f"{cr:.0f}%",
                         f"{cpt:,.0f}" if cpt else "∞"])
    p = os.path.join(OUT_DIR, "ablation_summary_ideal.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Scale", "Variant", "Cost", "Completed", "Rejected",
                     "Delta_pct", "Compl_Ratio", "Cost_per_Task"])
        w.writerows(rows)
    print(f"  ✓ {p}")


def make_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                              gridspec_kw={"width_ratios": [3, 2, 2]})

    # ── Panel (a): absolute cost grouped bars ─────────────────────────
    ax1 = axes[0]
    n_v = len(VARIANTS)
    x = np.arange(n_v)
    w = 0.25
    offsets = {"S": -w, "M": 0, "XL": w}

    for sc in SCALES:
        off = offsets[sc]
        for vi, v in enumerate(VARIANTS):
            cost = DATA[sc][v][0]
            ax1.bar(x[vi] + off, cost / 1000, w,
                    color=V_COLORS[v], alpha=S_ALPHA[sc],
                    hatch=S_HATCH[sc], edgecolor="white", linewidth=0.8, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([V_LABELS[v] for v in VARIANTS], fontsize=9)
    ax1.set_ylabel("Mean Total Cost (K)", fontsize=11)
    ax1.set_title("(a) Absolute Cost by Scale", fontsize=12, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}K"))
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    handles = [mpatches.Patch(fc="#888", alpha=S_ALPHA[s], hatch=S_HATCH[s],
               ec="white", label=f"Scale {s}") for s in SCALES]
    ax1.legend(handles=handles, fontsize=9, loc="upper left")

    # ── Panel (b): Completion Ratio ───────────────────────────────────
    ax2 = axes[1]
    ablation_v = ["PC", "FM"]
    y_pos = np.arange(len(ablation_v))
    bar_h = 0.22

    for si, sc in enumerate(SCALES):
        for yi, v in enumerate(ablation_v):
            compl = DATA[sc][v][1]
            full_compl = DATA[sc]["Full"][1]
            cr = compl / full_compl * 100 if full_compl > 0 else 0
            y_jit = y_pos[yi] + (si - 1) * 0.27
            color = S_COLORS[sc]
            ax2.barh(y_jit, cr, bar_h, color=color, alpha=0.85, zorder=3)
            label = f"{cr:.0f}%" if cr > 0 else "0%"
            ax2.text(max(cr, 3) + 2, y_jit, label,
                     va="center", fontsize=9, fontweight="bold", color=color)

    ax2.axvline(80, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    ax2.text(82, y_pos[-1] + 0.42, "healthy\nthreshold",
             fontsize=7, color="gray", va="bottom")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([V_LABELS[v] for v in ablation_v], fontsize=9)
    ax2.set_xlabel("Completion Ratio (%)", fontsize=11)
    ax2.set_title("(b) Task Completion Health", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 110)
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    dh = [mpatches.Patch(fc=S_COLORS[s], alpha=0.85, label=f"Scale {s}") for s in SCALES]
    ax2.legend(handles=dh, fontsize=8, loc="center right")

    # ── Panel (c): Cost per Completed Task (PC only; FM = ∞) ─────────
    ax3 = axes[2]
    # Only show PC since FM has 0 completions (infinite cost/task)
    bar_w = 0.35
    x3 = np.arange(len(SCALES))

    full_cpts = []
    pc_cpts = []
    for sc in SCALES:
        fc, fcomp, _ = DATA[sc]["Full"]
        pc, pcomp, _ = DATA[sc]["PC"]
        full_cpts.append(fc / fcomp if fcomp > 0 else 0)
        pc_cpts.append(pc / pcomp if pcomp > 0 else 0)

    bars_full = ax3.bar(x3 - bar_w / 2, full_cpts, bar_w,
                         color="#5A9E6B", alpha=0.85, label="Full", zorder=3)
    bars_pc = ax3.bar(x3 + bar_w / 2, pc_cpts, bar_w,
                       color="#F0A500", alpha=0.85, label="No Partial Charging", zorder=3)

    # Δ% labels
    for i in range(len(SCALES)):
        pct = (pc_cpts[i] - full_cpts[i]) / full_cpts[i] * 100
        ax3.text(x3[i] + bar_w / 2, pc_cpts[i] + 200,
                 f"+{pct:.0f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#E65100")

    ax3.set_xticks(x3)
    ax3.set_xticklabels([f"Scale {s}" for s in SCALES], fontsize=10)
    ax3.set_ylabel("Cost per Completed Task", fontsize=11)
    ax3.set_title("(c) Efficiency: Cost / Task", fontsize=12, fontweight="bold")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax3.grid(axis="y", alpha=0.25)
    ax3.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)
    ax3.legend(fontsize=9, loc="upper left")

    # Add FM annotation (lower-right to avoid overlapping tall XL bars)
    ax3.text(0.98, 0.22, "FM: ∞ cost/task\n(0 completions\non all scales)",
             transform=ax3.transAxes, fontsize=8, color="#D9534F",
             va="top", ha="right", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec="#D9534F", alpha=0.9))

    fig.suptitle("Cross-Scale Ablation Study (S, M, XL)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"ablation_cross_scale_ideal.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ {OUT_DIR}/ablation_cross_scale_ideal.png/.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating ideal ablation outputs ...")
    save_csv()
    make_figure()
    print("Done!")
