#!/usr/bin/env python3
"""Generate a publication-quality image of Table 7 (Cost Decomposition)
with the redesigned layout: Oper. subtotal + Terminal merged column.

Output: docs/ejor/section5.3/table7_cost_decomposition.png / .pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import textwrap
import numpy as np
import os

# ── Register Times New Roman from Windows fonts ──────────────────────────────
_TNR_DIR = "/mnt/c/Windows/Fonts"
_TNR_MAP = {
    "regular": os.path.join(_TNR_DIR, "times.ttf"),
    "bold":    os.path.join(_TNR_DIR, "timesbd.ttf"),
    "italic":  os.path.join(_TNR_DIR, "timesi.ttf"),
    "bi":      os.path.join(_TNR_DIR, "timesbi.ttf"),
}
_TNR_AVAILABLE = all(os.path.isfile(p) for p in _TNR_MAP.values())

if _TNR_AVAILABLE:
    for p in _TNR_MAP.values():
        fm.fontManager.addfont(p)
    _FONT_FAMILY = "Times New Roman"
else:
    _FONT_FAMILY = "STIXGeneral"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  [_FONT_FAMILY, "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ── Colors ────────────────────────────────────────────────────────────────────
C_NAVY     = "#1B2A4A"
C_BLUE     = "#2C5F8A"
C_WHITE    = "#FFFFFF"
C_GRAY     = "#F4F5F7"
C_RED      = "#C0392B"
C_TEXT     = "#1A1A1A"
C_MTEXT    = "#444444"
C_SEP      = "#B0BEC5"
C_RL_BG    = "#E3EBF6"
C_OPER_BG  = "#EBF5EB"  # light green for Oper. column

# ── Data ──────────────────────────────────────────────────────────────────────
# Option B: (Scale, Method, Travel, Charg., Tard., Idle, Oper., Reject., Terminal, Total)
# From ejor_table8_decomposition.csv (clean cost = Oper + Reject + Terminal)
DATA = [
    ("S",  "RL-APC",       1487,    78,    273,  3323,   5162,       0,   8000,  13162),
    ("S",  "Greedy-FR",    8729,  1110,  34274,  4279,  48391,   15333,   1000,  64725),
    ("S",  "Standby-Lazy", 1053,     9,      0,  2982,   4043,   38000,   3000,  45043),
    ("M",  "RL-APC",       3123,   199,     44,  3629,   6995,       0,  41750,  48745),
    ("M",  "Greedy-FR",    4087,   336,   1210,  6182,  11814,  218000,   3750, 233564),
    ("M",  "Standby-Lazy", 1686,     6,      0,  4440,   6132,  192333,   4417, 202882),
    ("L",  "RL-APC",       4598,   276,  11487,  7322,  23682,       0,  77933, 101616),
    ("L",  "Greedy-FR",   12329,  1341,  16913,  6064,  36646,  168000,  18467, 223112),
    ("XL", "RL-APC",       7035,   392,   6731,  2052,  16210,   14667,  99200, 130077),
    ("XL", "Greedy-FR",    7889,   509,    354,  2545,  11297,  577333,  18050, 606680),
    ("XL", "Standby-Lazy", 3672,     4,      0,  8462,  12138,  646333,   9500, 667971),
]

# Bold Oper. per scale: RL wins S/M/L (lowest Oper.), Greedy wins XL
BOLD_OPER = {0: True, 3: True, 6: True, 9: True}

# Rejection >= 30% of total → red
def rej_is_red(rej, total):
    return total > 0 and (rej / total) >= 0.30

# Scale group boundaries (row index where new scale starts)
SCALE_BREAKS = [3, 6, 8]  # M starts at 3, L at 6, XL at 8


def fmt_cost(v):
    return f"{v:,.0f}"


def draw_table():
    n_rows = len(DATA)

    # ── Column geometry ──────────────────────────────────────────────────
    # Scale | Method | Travel | Charg. | Tard. | Idle | Oper. | Reject. | Terminal | Total
    col_widths = [0.5, 1.3, 0.85, 0.7, 0.85, 0.7, 0.85, 0.85, 0.85, 0.85]
    TW = sum(col_widths)

    col_x = np.cumsum([0] + col_widths[:-1])
    col_cx = col_x + np.array(col_widths) / 2

    # ── Row geometry ─────────────────────────────────────────────────────
    ROW     = 0.38
    HDR     = 0.42
    SEP     = 0.14
    n_seps  = len(SCALE_BREAKS)

    body_h = n_rows * ROW + n_seps * SEP

    # Caption
    CAPTION_TEXT = (
        "Cost decomposition into weighted components (averages over 30 test instances). "
        "Oper. = Travel + Charging + Tardiness + Idle (Section 3 operational cost). "
        "Reject. = 10,000 \u00D7 n_rej. "
        "Terminal = C_term \u00D7 n_unfin at episode end "
        "(tasks accepted but not completed within the horizon). "
        "Best Oper. per scale in bold. "
        "Rejection cost in red when \u2265 30% of total. "
        "L-scale: Greedy-FR = Charge-High (best fixed rule)."
    )
    cap_lines = textwrap.wrap(CAPTION_TEXT, width=120)
    CAP_LINE_H = 0.16
    CAP_PAD_TOP = 0.14
    CAP_PAD_BOT = 0.06
    cap_block_h = len(cap_lines) * CAP_LINE_H + CAP_PAD_TOP + CAP_PAD_BOT

    TITLE_H  = 0.50
    BOTTOM_H = 0.10

    total_h = TITLE_H + HDR + body_h + cap_block_h + BOTTOM_H

    # ── Figure ───────────────────────────────────────────────────────────
    SCALE = 0.72
    fig_w = TW * SCALE
    fig_h = total_h * SCALE

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, TW)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_WHITE)

    # ── Helpers ──────────────────────────────────────────────────────────
    def rect(x, y, w, h, fc, ec=None, lw=0):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec or fc, lw=lw))

    def txt(x, y, s, fs=8, ha="center", va="center",
            color=C_TEXT, weight="normal", style="normal"):
        return ax.text(x, y, s, fontsize=fs, ha=ha, va=va,
                       color=color, fontweight=weight, fontstyle=style)

    # ── Title ────────────────────────────────────────────────────────────
    y_cur = total_h - TITLE_H / 2
    txt(TW / 2, y_cur, "Table 7  Cost Decomposition",
        fs=11, weight="bold", color=C_NAVY)

    # ── Column header ────────────────────────────────────────────────────
    y_hdr_top = total_h - TITLE_H
    rect(0, y_hdr_top - HDR, TW, HDR, C_BLUE)

    col_labels = ["Scale", "Method", "Travel", "Charg.", "Tard.", "Idle",
                  "Oper.", "Reject.", "Terminal", "Total"]
    for ci, lab in enumerate(col_labels):
        ha = "left" if ci <= 1 else "center"
        xp = col_x[ci] + 0.06 if ci <= 1 else col_cx[ci]
        txt(xp, y_hdr_top - HDR / 2, lab, fs=8, weight="bold",
            color=C_WHITE, ha=ha)

    # ── Operational bracket in header ────────────────────────────────────
    # Draw a bracket above Travel-Idle columns indicating they sum to Oper.
    bracket_y = y_hdr_top - 0.06
    bracket_x1 = col_x[2] + 0.06
    bracket_x2 = col_x[5] + col_widths[5] - 0.06
    ax.annotate("", xy=(bracket_x1, bracket_y), xytext=(bracket_x2, bracket_y),
                arrowprops=dict(arrowstyle="-", color=C_WHITE, lw=0.6))

    # ── Data body ────────────────────────────────────────────────────────
    y_body_top = y_hdr_top - HDR
    ax.plot([0, TW], [y_body_top, y_body_top], color=C_NAVY, lw=0.3)

    y = y_body_top
    cur_scale = None

    for ri, row in enumerate(DATA):
        scale, method = row[0], row[1]

        # Scale separator
        if ri in SCALE_BREAKS:
            y -= SEP
            ax.plot([0, TW], [y + SEP / 2] * 2, color=C_SEP, lw=0.4)

        y -= ROW
        ym = y + ROW / 2

        is_rl = ("RL" in method)

        # Background
        bg = C_RL_BG if is_rl else (C_WHITE if ri % 2 == 0 else C_GRAY)
        rect(0, y, TW, ROW, bg)

        # Light green highlight for Oper. column (skip on RL rows to keep blue bg visible)
        if not is_rl:
            rect(col_x[6], y, col_widths[6], ROW, C_OPER_BG)

        # Scale label (only first row of each group)
        if scale != cur_scale:
            txt(col_x[0] + 0.06, ym, scale, fs=8, ha="left",
                weight="bold", color=C_NAVY)
            cur_scale = scale

        # Method
        txt(col_x[1] + 0.06, ym, method, fs=7.5, ha="left",
            weight="bold" if is_rl else "normal",
            color=C_NAVY if is_rl else C_TEXT)

        # Travel, Charg., Tard., Idle (cols 2-5)
        vals = row[2:6]
        for ci_off, v in enumerate(vals):
            ci = 2 + ci_off
            txt(col_cx[ci], ym, fmt_cost(v), fs=7.5)

        # Oper. (col 6) — bold if best
        oper_val = row[6]
        oper_bold = BOLD_OPER.get(ri, False)
        txt(col_cx[6], ym, fmt_cost(oper_val), fs=7.5,
            weight="bold" if oper_bold else "normal",
            color=C_NAVY if oper_bold else C_TEXT)

        # Reject. (col 7) — red if >= 30%
        rej_val = row[7]
        total_val = row[9]
        rej_color = C_RED if rej_is_red(rej_val, total_val) else C_TEXT
        txt(col_cx[7], ym, fmt_cost(rej_val), fs=7.5, color=rej_color)

        # Terminal (col 8)
        epiend_val = row[8]
        txt(col_cx[8], ym, fmt_cost(epiend_val), fs=7.5)

        # Total (col 9)
        txt(col_cx[9], ym, fmt_cost(total_val), fs=7.5,
            weight="bold" if is_rl else "normal")

    # ── Bottom rule ──────────────────────────────────────────────────────
    ax.plot([0, TW], [y, y], color=C_NAVY, lw=1.2)

    # ── Vertical separator for Oper. column ──────────────────────────────
    for vx in [col_x[6], col_x[6] + col_widths[6]]:
        ax.plot([vx, vx], [y_body_top, y], color=C_SEP, lw=0.5, alpha=0.6)

    # ── Caption ──────────────────────────────────────────────────────────
    CAP_FS = 6.0
    MARGIN = 0.10
    y_cap = y - CAP_PAD_TOP

    for i, line in enumerate(cap_lines):
        yp = y_cap - i * CAP_LINE_H
        txt(MARGIN, yp, line, fs=CAP_FS, color=C_MTEXT, ha="left")

    return fig


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "ejor", "section5.3")
    os.makedirs(out_dir, exist_ok=True)

    fig = draw_table()

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"table7_cost_decomposition.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), pad_inches=0.08)
        print(f"\u2713 {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
