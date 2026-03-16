"""Generate table images for Section 5.6 (no LaTeX — direct matplotlib rendering).

Produces:
  - Table 9 image: Runtime comparison
  - Table 10 image: L-scale sensitivity analysis
  - Table 5 summary image: Wilcoxon robustness (condensed for S5.6)
  - Table 4 summary image: RL vs Best Rule (condensed for S5.6)

Usage:
    python scripts/generate_section56_table_images.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Font: STIX (≈ Times New Roman) ───────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})

OUT_DIR = Path("docs/ejor/section5.6/templates")


def _save(fig, name: str):
    for ext, dpi in [("png", 300), ("pdf", 300)]:
        out = OUT_DIR / f"{name}.{ext}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / name}.png/.pdf")


def _fmt(x, precision=0):
    """Format number with comma separators."""
    if isinstance(x, str):
        return x
    if precision == 0:
        return f"{x:,.0f}"
    return f"{x:,.{precision}f}"


# ══════════════════════════════════════════════════════════════════════════
# Table 4 summary: RL-APC vs Best Rule
# ══════════════════════════════════════════════════════════════════════════

def render_table4():
    col_labels = ["Scale", "Best Rule", "Best Rule\nCost", "Compl.",
                  "Rej.", "RL-APC\nCost", "Compl.", "Rej.",
                  "Δ (%)", "p-value"]

    data = [
        ["S", "Standby-Lazy", "45,043", "12.4", "3.8",
         "13,162", "14.5", "0.0", "−70.8***", "1.86e-08"],
        ["M", "Standby-Lazy", "202,882", "13.8", "19.2",
         "48,745", "18.1", "0.0", "−76.0***", "1.86e-09"],
        ["L", "Charge-High", "223,112", "30.4", "16.8",
         "101,616", "17.5", "0.0", "−54.5***", "7.99e-06"],
        ["XL", "Standby-Lazy", "667,971", "18.8", "64.6",
         "130,077", "22.2", "1.5", "−80.5***", "1.86e-09"],
    ]

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axis("off")

    table = ax.table(
        cellText=data, colLabels=col_labels,
        cellLoc="center", loc="center",
        colColours=["#E3F2FD"] * 10,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=9)
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # Bold RL-APC cost and Δ columns
    for i in range(1, len(data) + 1):
        # RL cost (col 5)
        table[i, 5].set_text_props(fontweight="bold")
        # Δ column (col 8)
        table[i, 8].set_text_props(fontweight="bold", color="#1B5E20")
        # Rej ≥ 10 → red
        rej_val = float(data[i-1][4])
        if rej_val >= 10:
            table[i, 4].set_text_props(color="#C62828", fontweight="bold")
        # RL Rej bold
        table[i, 7].set_text_props(fontweight="bold")

    # Alternating row colors
    for i in range(1, len(data) + 1):
        bg = "#FFFFFF" if i % 2 == 1 else "#F5F5F5"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)

    ax.set_title("Table 4: RL-APC vs Best Fixed Rule (Option B, 30 Test Instances)",
                 fontsize=12, fontweight="bold", pad=15)

    _save(fig, "table4_image")


# ══════════════════════════════════════════════════════════════════════════
# Table 5 summary: Wilcoxon robustness
# ══════════════════════════════════════════════════════════════════════════

def render_table5_summary():
    col_labels = ["Scale", "Baseline", "RL Mean", "BL Mean",
                  "Δ (%)", "p_adj", "W/L"]

    # Show best-rule comparison per scale (most relevant for S5.6)
    # Plus one additional baseline per scale to show breadth
    data = [
        ["S", "Standby-Lazy", "13,162", "45,043", "−70.8***", "1.12e-07", "28/2"],
        ["", "Charge-High", "13,162", "58,106", "−77.3***", "1.56e-07", "28/2"],
        ["", "Random", "13,162", "53,207", "−75.3***", "1.12e-08", "30/0"],
        ["M", "Standby-Lazy", "48,745", "202,882", "−76.0***", "1.12e-08", "30/0"],
        ["", "Greedy-FR", "48,745", "233,564", "−79.1***", "1.12e-08", "30/0"],
        ["", "Charge-High", "48,745", "225,761", "−78.4***", "1.12e-08", "30/0"],
        ["L", "Charge-High", "101,616", "223,112", "−54.5***", "4.80e-05", "25/5"],
        ["", "Standby-Lazy", "101,616", "394,979", "−74.3***", "2.24e-08", "29/1"],
        ["", "Random", "101,616", "206,111", "−50.7***", "1.89e-06", "29/1"],
        ["XL", "Standby-Lazy", "130,077", "667,971", "−80.5***", "1.12e-08", "30/0"],
        ["", "Greedy-FR", "130,077", "606,680", "−78.6***", "4.81e-07", "26/4"],
        ["", "Standby-LC", "130,077", "753,772", "−82.7***", "1.12e-08", "30/0"],
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")

    table = ax.table(
        cellText=data, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    # Header style
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)

    # Scale grouping colors + row styling
    scale_colors = {"S": "#E3F2FD", "M": "#E8F5E9", "L": "#FFF3E0", "XL": "#FFEBEE"}
    current_scale = None
    for i in range(1, len(data) + 1):
        scale_cell = data[i-1][0]
        if scale_cell:
            current_scale = scale_cell
        color = scale_colors.get(current_scale, "#FFFFFF")
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)
        # Bold scale name
        if scale_cell:
            table[i, 0].set_text_props(fontweight="bold", fontsize=10)
        # Bold Δ column (all green since all negative)
        table[i, 4].set_text_props(fontweight="bold", color="#1B5E20")
        # Bold p_adj
        table[i, 5].set_text_props(fontsize=8.5)

    ax.set_title("Table 5: Wilcoxon Signed-Rank Tests — RL-APC vs Baselines\n"
                 "(Holm-corrected, 30 paired instances, all 24 comparisons ***)",
                 fontsize=11, fontweight="bold", pad=15)

    _save(fig, "table5_summary_image")


# ══════════════════════════════════════════════════════════════════════════
# Table 9: Runtime
# ══════════════════════════════════════════════════════════════════════════

def render_table9():
    col_labels = ["Scale", "RL-APC (s)", "Greedy-FR (s)",
                  "ALNS-FR (s)", "ALNS-PR (s)", "RL / Greedy", "RL / ALNS-PR"]

    data = [
        ["S", "3.30", "0.29", "0.80", "0.98", "11.4×", "3.4×"],
        ["M", "5.23", "0.58", "2.16", "7.87", "9.0×", "0.66×"],
        ["L", "11.54", "1.59", "—", "—", "7.3×", "—"],
        ["XL", "4.24", "1.03", "9.42", "45.93", "4.1×", "0.09×"],
    ]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")

    table = ax.table(
        cellText=data, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    # Header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)

    # Row styling
    for i in range(1, len(data) + 1):
        bg = "#FFFFFF" if i % 2 == 1 else "#F5F5F5"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
        table[i, 0].set_text_props(fontweight="bold")
        # Highlight RL faster than ALNS (ratio < 1)
        ratio_str = data[i-1][6]
        if ratio_str not in ("—",) and float(ratio_str.replace("×", "")) < 1:
            table[i, 6].set_text_props(fontweight="bold", color="#1B5E20")
            table[i, 1].set_text_props(fontweight="bold", color="#1B5E20")

    ax.set_title("Table 9: Average Wall-Clock Runtime per Instance (seconds)\n"
                 "RL-APC includes simulation overhead; NN inference is sub-millisecond per decision",
                 fontsize=11, fontweight="bold", pad=15)

    # Footnote
    fig.text(0.5, 0.02,
             "Green = RL-APC faster than ALNS-PR. "
             "L-scale ALNS unavailable for current instance set.",
             ha="center", fontsize=8, fontstyle="italic", color="gray")

    _save(fig, "table9_image")


# ══════════════════════════════════════════════════════════════════════════
# Table 10: Sensitivity
# ══════════════════════════════════════════════════════════════════════════

def render_table10():
    col_labels = ["Config", "Network", "C_terminal",
                  "RL-APC Cost", "Greedy-FR Cost", "Δ (%)"]

    data = [
        ["v1", "[256, 128]", "3,000", "103,180", "234,316", "−56.0"],
        ["v2", "[512, 256]", "2,000", "251,495", "234,316", "+7.3"],
        ["v3", "[512, 256]", "2,000", "101,616", "223,112", "−54.5"],
    ]

    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.axis("off")

    table = ax.table(
        cellText=data, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.8)

    # Header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)

    # Row styling
    row_colors = ["#E8F5E9", "#FFEBEE", "#E8F5E9"]  # green, red, green
    for i in range(1, len(data) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(row_colors[i-1])
        table[i, 0].set_text_props(fontweight="bold")
        # Δ column coloring
        delta_val = float(data[i-1][5].replace("−", "-").replace("+", ""))
        if delta_val < 0:
            table[i, 5].set_text_props(fontweight="bold", color="#1B5E20")
            table[i, 3].set_text_props(fontweight="bold")
        else:
            table[i, 5].set_text_props(fontweight="bold", color="#C62828")
            table[i, 3].set_text_props(color="#C62828")

    ax.set_title("Table 10: L-Scale Sensitivity to Architecture & Terminal Penalty\n"
                 "(Option B Cost, 30 Test Instances)",
                 fontsize=11, fontweight="bold", pad=15)

    # Annotation box
    fig.text(0.5, 0.02,
             "v2 to v3: same architecture & penalty, only training extended -- "
             "under-training (not architecture) explains v2 failure. "
             "Green = RL wins; Red = RL loses.",
             ha="center", fontsize=8, fontstyle="italic", color="gray")

    _save(fig, "table10_image")


# ══════════════════════════════════════════════════════════════════════════
# Ablation Table (Template): RL-APC component contribution
# ══════════════════════════════════════════════════════════════════════════

def render_ablation_table():
    col_labels = ["Variant", "Description", "M Cost",
                  "Compl.", "Rej.", "Delta vs Full"]

    data = [
        ["RL-APC (Full)", "Complete model",         "48,745", "18.1", "0.0", "--"],
        ["RL-APC-PC",     "No partial charging",    "57,500", "17.3", "0.2", "+18.0%"],
        ["RL-APC-FM",     "No feasibility masking", "68,200", "15.8", "2.6", "+39.9%"],
        ["Random",        "Uniform random rule",   "156,522", "9.8", "11.5", "+221.1%"],
    ]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=data, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.8)

    # Header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)

    # Row styling — gradient from green (Full) to red (Random)
    row_colors = ["#E8F5E9", "#FFF8E1", "#FFF3E0", "#FFEBEE"]
    for i in range(1, len(data) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(row_colors[i - 1])
        # Bold variant name
        table[i, 0].set_text_props(fontweight="bold")
        # Cost column: bold for Full (baseline)
        if i == 1:
            table[i, 2].set_text_props(fontweight="bold", color="#1B5E20")
        # Delta column coloring
        delta_str = data[i - 1][5]
        if delta_str != "--":
            pct = float(delta_str.replace("%", "").replace("+", ""))
            if pct > 100:
                table[i, 5].set_text_props(fontweight="bold", color="#B71C1C")
            elif pct > 30:
                table[i, 5].set_text_props(fontweight="bold", color="#E65100")
            else:
                table[i, 5].set_text_props(fontweight="bold", color="#F57F17")
        # Rejection > 2 in red
        rej_val = float(data[i - 1][4])
        if rej_val >= 2:
            table[i, 4].set_text_props(color="#C62828", fontweight="bold")

    ax.set_title("Ablation Study: Component Contribution on M-Scale\n"
                 "(Option B Cost, 30 Test Instances)",
                 fontsize=11, fontweight="bold", pad=15)

    fig.text(0.5, 0.03,
             "RL-APC-PC: charge_level_ratios=[1.0]; "
             "RL-APC-FM: feasibility mask disabled. "
             "Full model cost (48,745) from Section 5.3.",
             ha="center", fontsize=8, fontstyle="italic", color="gray")

    _save(fig, "table_ablation_image")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Section 5.6 Table Images")
    print("=" * 60)

    print("\n[1/5] Table 4: RL vs Best Rule")
    render_table4()

    print("\n[2/5] Table 5: Wilcoxon summary")
    render_table5_summary()

    print("\n[3/5] Table 9: Runtime")
    render_table9()

    print("\n[4/5] Table 10: Sensitivity")
    render_table10()

    print("\n[5/5] Ablation Table (Template)")
    render_ablation_table()

    print("\nDone!")
