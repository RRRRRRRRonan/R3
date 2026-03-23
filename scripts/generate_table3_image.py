#!/usr/bin/env python3
"""Generate a publication-quality image of Table 3 (Individual Rule Performance)
with EJOR-style formatting.  Times New Roman throughout.

Output: docs/ejor/table3_individual_rules.png / .pdf
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
    # Fallback: STIX is metrically close to Times
    _FONT_FAMILY = "STIXGeneral"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  [_FONT_FAMILY, "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ── Data ──────────────────────────────────────────────────────────────────────
RULES = [
    # (name, category, S_cost, S_rej, M_cost, M_rej, L_cost, L_rej, XL_cost, XL_rej)
    ("STTF\u2020",       "Dispatch",  88840, 7.9, 275178, 26.5, 469092, 45.8, 750969,  74.0),
    ("EDD\u2020",        "Dispatch",  88840, 7.9, 275178, 26.5, 469092, 45.8, 750969,  74.0),
    ("MST",              "Dispatch",  85842, 7.5, 272907, 26.2, 468798, 45.8, 754343,  74.3),
    ("HPF\u2020",        "Dispatch",  88840, 7.9, 275178, 26.5, 469092, 45.8, 750969,  74.0),
    ("Insert-MC",        "Dispatch",  82132, 2.7, 265211, 16.0, 435012, 36.4, 710061,  46.1),
    ("Charge-Urgent",    "Charge",    93535, 8.2, 275409, 26.2, 469287, 45.5, 744229,  66.8),
    ("Charge-Low",       "Charge",    93435, 8.2, 258428, 21.7, 340286, 14.7, 733306,  34.0),
    ("Charge-Med",       "Charge",    88044, 7.5, 272769, 25.9, 450729, 43.1, 726214,  57.3),
    ("Charge-High\u2021","Charge",    64725, 1.4, 233564, 20.8, 223112, 16.8, 667971,  55.0),
    ("Charge-Opp\u2021", "Charge",    64725, 1.4, 233564, 20.8, 223112, 16.8, 667971,  55.0),
    ("Standby-LC\u00A7", "Standby",  102746, 9.2, 275901, 26.3, 468003, 45.3, 751654,  74.3),
    ("Standby-Lazy",     "Standby",   45043, 3.8, 202882, 19.2, 418299, 37.9, 667971,  64.6),
    ("Standby-HM\u00A7", "Standby",  102746, 9.2, 275901, 26.3, 468003, 45.3, 751654,  74.3),
    ("Accept-Feas",      "Accept",    95405, 8.5, 275825, 26.3, 469360, 45.5, 757900,  74.0),
    ("Accept-Val",       "Accept",   114130, 0.0, 490642,  0.0, 831429,  0.0, 1360537,  0.0),
]

RL_ROW = ("RL-APC", "Adaptive", 13162, 0.0, 48745, 0.0, 101616, 0.3, 130077, 1.5)

CAT_BOUNDARIES = [5, 10, 13]
BEST_RULE_ROWS = {"S": 11, "M": 11, "L": 8, "XL": [8, 11]}

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


def fmt_cost(v):
    return f"{v:,.0f}"

def fmt_rej(v):
    return f"{v:.1f}"


def draw_table():
    n_rules = len(RULES)
    n_rows  = n_rules + 1

    # ── Column geometry (data-coords) ─────────────────────────────────────
    col_widths = [1.4, 0.9, 1.0, 0.55, 1.0, 0.55, 1.0, 0.55, 1.1, 0.55]
    TW = sum(col_widths)                   # table width in data-coords

    col_x = np.cumsum([0] + col_widths[:-1])
    col_cx = col_x + np.array(col_widths) / 2

    # ── Row geometry ──────────────────────────────────────────────────────
    ROW     = 0.38
    HDR     = 0.42
    SCALE_H = 0.38
    SEP     = 0.12
    RL_SEP  = 0.18

    body_h = n_rows * ROW + len(CAT_BOUNDARIES) * SEP + RL_SEP

    # Caption = wrapped paragraph between title and table, aligned to table width
    CAPTION_TEXT = (
        "Average total cost and number of rejected tasks (Rej) for 15 individual "
        "dispatch rules and RL-APC across four problem scales (30 test instances "
        "per scale, Option B clean cost). Rules are grouped by category. "
        "Underlined: best single-rule cost per scale. "
        "Red Rej: \u2265 10 rejected tasks. "
        "\u2020 STTF = EDD = HPF;  \u2021 Charge-High = Charge-Opp;  "
        "\u00A7 Standby-LC = Standby-HM (identical performance), "
        "reducing 15 rules to 11 independent strategies. "
        "The best rule is Standby-Lazy (S, M) and Charge-High (L); "
        "on XL, Charge-High and Standby-Lazy tie at 667,971."
    )
    cap_lines = textwrap.wrap(CAPTION_TEXT, width=130)
    CAP_LINE_H = 0.16       # very tight line spacing
    CAP_PAD_TOP = 0.14
    CAP_PAD_BOT = 0.06
    cap_block_h = len(cap_lines) * CAP_LINE_H + CAP_PAD_TOP + CAP_PAD_BOT

    TITLE_H  = 0.50
    BOTTOM_H = 0.10

    # Caption goes BELOW the table body
    total_h = TITLE_H + SCALE_H + HDR + body_h + cap_block_h + BOTTOM_H

    # ── Figure ────────────────────────────────────────────────────────────
    SCALE = 0.72          # data-units → inches
    fig_w = TW * SCALE
    fig_h = total_h * SCALE

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, TW)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])      # axes fill entire figure
    fig.patch.set_facecolor(C_WHITE)

    # ── Helpers ───────────────────────────────────────────────────────────
    def rect(x, y, w, h, fc, ec=None, lw=0):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec or fc, lw=lw))

    def txt(x, y, s, fs=8, ha="center", va="center",
            color=C_TEXT, weight="normal", style="normal"):
        return ax.text(x, y, s, fontsize=fs, ha=ha, va=va,
                       color=color, fontweight=weight, fontstyle=style)

    def txt_uline(x, y, s, fs=8, color=C_TEXT, weight="normal"):
        """Draw text with a manual underline."""
        t = txt(x, y, s, fs=fs, color=color, weight=weight)
        fig.canvas.draw()                # needed for get_window_extent
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        ax.plot([bb.x0, bb.x1], [bb.y0 - 0.02, bb.y0 - 0.02],
                color=color, lw=0.8)
        return t

    # ── (A) Title ─────────────────────────────────────────────────────────
    y_cur = total_h - TITLE_H / 2
    txt(TW / 2, y_cur, "Table 3  Individual Rule Performance",
        fs=11, weight="bold", color=C_NAVY)

    # ── (C) Scale header row  (S  M  L  XL) ──────────────────────────────
    y_sh_top = total_h - TITLE_H
    rect(0, y_sh_top - SCALE_H, TW, SCALE_H, C_NAVY)

    scale_labels = ["S", "M", "L", "XL"]
    for si, lab in enumerate(scale_labels):
        ci = 2 + si * 2
        mid = (col_x[ci] + col_x[ci] + col_widths[ci] + col_widths[ci + 1]) / 2
        txt(mid, y_sh_top - SCALE_H / 2, lab, fs=9, weight="bold", color=C_WHITE)
        # bracket
        x1 = col_x[ci] + 0.08
        x2 = col_x[ci] + col_widths[ci] + col_widths[ci + 1] - 0.08
        ax.plot([x1, x2], [y_sh_top - SCALE_H + 0.04] * 2,
                color=C_WHITE, lw=0.6, alpha=0.5)

    # ── (D) Column header row ─────────────────────────────────────────────
    y_hdr_top = y_sh_top - SCALE_H
    rect(0, y_hdr_top - HDR, TW, HDR, C_BLUE)

    col_labels = ["Rule", "Category",
                  "Cost", "Rej", "Cost", "Rej", "Cost", "Rej", "Cost", "Rej"]
    for ci, lab in enumerate(col_labels):
        ha = "left" if ci == 0 else "center"
        xp = col_x[ci] + 0.10 if ci == 0 else col_cx[ci]
        txt(xp, y_hdr_top - HDR / 2, lab, fs=8, weight="bold",
            color=C_WHITE, ha=ha)

    # ── (E) Data body ─────────────────────────────────────────────────────
    y_body_top = y_hdr_top - HDR
    ax.plot([0, TW], [y_body_top, y_body_top], color=C_NAVY, lw=0.3)

    all_rows = list(RULES) + [RL_ROW]
    sc_cost_idx = [2, 4, 6, 8]
    sc_rej_idx  = [3, 5, 7, 9]
    sc_names    = ["S", "M", "L", "XL"]

    y = y_body_top
    for ri, row in enumerate(all_rows):
        is_rl = (ri == n_rules)

        # category separator
        if ri in CAT_BOUNDARIES:
            y -= SEP
            ax.plot([0, TW], [y + SEP / 2] * 2, color=C_SEP, lw=0.4)

        # RL separator
        if is_rl:
            y -= RL_SEP
            ax.plot([0, TW], [y + RL_SEP / 2 + 0.02] * 2, color=C_NAVY, lw=1.2)

        y -= ROW
        ym = y + ROW / 2

        # background
        bg = C_RL_BG if is_rl else (C_WHITE if ri % 2 == 0 else C_GRAY)
        rect(0, y, TW, ROW, bg)

        # rule name
        txt(col_x[0] + 0.10, ym, row[0], fs=7.5, ha="left",
            weight="bold" if is_rl else "normal",
            color=C_NAVY if is_rl else C_TEXT)

        # category
        txt(col_cx[1], ym, row[1], fs=7.5,
            weight="bold" if is_rl else "normal",
            style="normal" if is_rl else "italic",
            color=C_NAVY if is_rl else C_MTEXT)

        # cost & rej per scale
        for si, sc in enumerate(sc_names):
            cv = row[sc_cost_idx[si]]
            rv = row[sc_rej_idx[si]]
            ci_c = 2 + si * 2
            ci_r = 3 + si * 2

            best_val = BEST_RULE_ROWS[sc]
            is_best = (not is_rl and (ri == best_val if isinstance(best_val, int) else ri in best_val))
            # cost
            cw = "bold" if (is_best or is_rl) else "normal"
            cc = C_NAVY if is_rl else C_TEXT
            if is_best:
                txt_uline(col_cx[ci_c], ym, fmt_cost(cv), fs=7.5,
                          color=cc, weight=cw)
            else:
                txt(col_cx[ci_c], ym, fmt_cost(cv), fs=7.5,
                    color=cc, weight=cw)

            # rej
            rc = C_RED if (rv >= 10 and not is_rl) else (C_NAVY if is_rl else C_TEXT)
            rw = "bold" if is_rl else "normal"
            txt(col_cx[ci_r], ym, fmt_rej(rv), fs=7.5, color=rc, weight=rw)

    # ── Bottom rule ───────────────────────────────────────────────────────
    ax.plot([0, TW], [y, y], color=C_NAVY, lw=1.2)

    # ── Subtle vertical separators between scale groups ───────────────────
    for si in range(4):
        x = col_x[2 + si * 2]
        ax.plot([x, x], [y_sh_top, y], color=C_SEP, lw=0.3, alpha=0.4)

    # ── (F) Caption — justified, compact spacing ────────────────────────
    CAP_FS = 6.0            # smaller font for compact caption
    MARGIN = 0.10
    avail_w = TW - 2 * MARGIN
    y_cap = y - CAP_PAD_TOP

    # Pre-measure a typical space width for gap capping
    _sp = txt(0, -10, "x x", fs=CAP_FS, color=C_MTEXT, ha="left")
    _sp2 = txt(0, -10, "xx", fs=CAP_FS, color=C_MTEXT, ha="left")
    fig.canvas.draw()
    _w1 = _sp.get_window_extent().transformed(ax.transData.inverted()).width
    _w2 = _sp2.get_window_extent().transformed(ax.transData.inverted()).width
    _sp.remove(); _sp2.remove()
    natural_space = _w1 - _w2
    max_gap = natural_space * 1.8   # cap gap to avoid overly loose lines

    for i, line in enumerate(cap_lines):
        is_last = (i == len(cap_lines) - 1)
        yp = y_cap - i * CAP_LINE_H

        if is_last or len(line.split()) <= 1:
            txt(MARGIN, yp, line, fs=CAP_FS, color=C_MTEXT, ha="left")
            continue

        words = line.split()
        word_widths = []
        for w in words:
            t = txt(0, -10, w, fs=CAP_FS, color=C_MTEXT, ha="left")
            fig.canvas.draw()
            ww = t.get_window_extent().transformed(
                ax.transData.inverted()).width
            t.remove()
            word_widths.append(ww)

        total_word_w = sum(word_widths)
        n_gaps = len(words) - 1
        gap = (avail_w - total_word_w) / n_gaps if n_gaps > 0 else 0
        gap = min(gap, max_gap)     # keep words close together

        xp = MARGIN
        for wi, w in enumerate(words):
            txt(xp, yp, w, fs=CAP_FS, color=C_MTEXT, ha="left")
            xp += word_widths[wi] + gap

    return fig


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "ejor")
    os.makedirs(out_dir, exist_ok=True)

    fig = draw_table()

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"table3_individual_rules.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), pad_inches=0.08)
        print(f"\u2713 {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
