#!/usr/bin/env python3
"""Generate a REFERENCE Table 3 image with projected L-v4 target results.

This is NOT real data — it is a target visualization for future training.
Output: docs/ejor/table3_target_reference.png / .pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import textwrap
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

# ── Data (actual for S/M/XL, PROJECTED for L) ────────────────────────────────
# L-scale RL-APC: projected v4 target based on:
#   - v3→v4 expected ~37% cost reduction (662K → 415K)
#   - Less conservative standby → more completed tasks (17.5 → 25.5)
#   - Maintain near-zero rejection (0.5)
RULES = [
    ("STTF\u2020",       "Dispatch", 110551, 7.9, 294978, 26.5, 491192, 45.8, 764969,  74.0),
    ("EDD\u2020",        "Dispatch", 110551, 7.9, 294978, 26.5, 491192, 45.8, 764969,  74.0),
    ("MST",              "Dispatch", 108707, 7.5, 294407, 26.2, 490298, 45.8, 768743,  74.3),
    ("HPF\u2020",        "Dispatch", 110551, 7.9, 294978, 26.5, 491192, 45.8, 764969,  74.0),
    ("Insert-MC",        "Dispatch", 103048, 2.7, 351911, 16.0, 447292, 36.4, 747761,  46.1),
    ("Charge-Urgent",    "Charge",   118337, 8.2, 301809, 26.2, 498087, 45.5, 841829,  66.8),
    ("Charge-Low",       "Charge",   118135, 8.2, 335128, 21.7, 737186, 14.7, 1141406, 34.0),
    ("Charge-Med",       "Charge",   112644, 7.5, 299669, 25.9, 490829, 43.1, 915014,  57.3),
    ("Charge-High\u2021","Charge",    83962, 1.4, 266218, 20.8, 336386, 16.8, 790373,  55.0),
    ("Charge-Opp\u2021", "Charge",    83962, 1.4, 266218, 20.8, 336386, 16.8, 790373,  55.0),
    ("Standby-LC\u00A7", "Standby",  126546, 9.2, 301201, 26.3, 497103, 45.3, 762554,  74.3),
    ("Standby-Lazy",     "Standby",   60731, 3.8, 224316, 19.2, 418299, 37.9, 696148,  64.6),
    ("Standby-HM\u00A7", "Standby",  126546, 9.2, 301201, 26.3, 497103, 45.3, 762554,  74.3),
    ("Accept-Feas",      "Accept",   118405, 8.5, 300925, 26.3, 497360, 45.5, 780760,  74.0),
    ("Accept-Val",       "Accept",   157360, 0.0, 490642,  0.0, 831429,  0.0, 1360537,  0.0),
]

# RL-APC row: ALL scales = projected best-achievable targets
#   S: keep actual (already -34.4% vs best rule, dominant)
#   M: 286,665 → 251,438 (-12.3%): [512,256] + 2M steps, Δ +12.1% vs Standby-Lazy
#   L: 662,267 → 397,214 (-40.0%): v4 training (2M steps + tuned params), Δ +18.1% vs Charge-High
#   XL: 814,344 → 728,591 (-10.5%): 2M steps + gamma tuning, Δ +4.7% vs Standby-Lazy
RL_ROW = ("RL-APC", "Adaptive", 39846, 0.0, 251438, 0.0, 397214, 0.3, 728591, 0.7)

CAT_BOUNDARIES = [5, 10, 13]
BEST_RULE_ROWS = {"S": 11, "M": 11, "L": 8, "XL": 11}

# ── Colors ────────────────────────────────────────────────────────────────────
C_NAVY   = "#1B2A4A"
C_BLUE   = "#2C5F8A"
C_WHITE  = "#FFFFFF"
C_GRAY   = "#F4F5F7"
C_RED    = "#C0392B"
C_TEXT   = "#1A1A1A"
C_MTEXT  = "#444444"
C_SEP    = "#B0BEC5"
C_RL_BG  = "#E3EBF6"
C_TARGET = "#E67E22"   # orange for projected values


def fmt_cost(v):
    return f"{v:,.0f}"

def fmt_rej(v):
    return f"{v:.1f}"


def draw_table():
    n_rules = len(RULES)
    n_rows  = n_rules + 1

    col_widths = [1.4, 0.9, 1.0, 0.55, 1.0, 0.55, 1.0, 0.55, 1.1, 0.55]
    TW = sum(col_widths)
    col_x  = np.cumsum([0] + col_widths[:-1])
    col_cx = col_x + np.array(col_widths) / 2

    ROW = 0.38;  HDR = 0.42;  SCALE_H = 0.38
    SEP = 0.12;  RL_SEP = 0.18

    body_h = n_rows * ROW + len(CAT_BOUNDARIES) * SEP + RL_SEP

    # Caption config — tighter than before
    CAPTION_TEXT = (
        "Average total cost and number of rejected tasks (Rej) for 15 individual "
        "dispatch rules and RL-APC across four problem scales (30 test instances "
        "per scale). Rules are grouped by category. Underlined: best single-rule "
        "cost per scale. Bold RL-APC cost: lower than all single rules. "
        "Red Rej: \u2265 10 rejected tasks. "
        "\u2020 STTF = EDD = HPF; \u2021 Charge-High = Charge-Opp; "
        "\u00A7 Standby-LC = Standby-HM (identical performance), "
        "reducing 15 rules to 11 independent strategies. "
        "The best rule shifts from Standby-Lazy (S, M, XL) to Charge-High (L); "
        "cost ratio between worst and best: 1.95\u00D7 (XL) to 2.59\u00D7 (S). "
        "RL-APC values in orange (M, L, XL) are projected targets from optimized retraining, not measured results."
    )
    cap_lines = textwrap.wrap(CAPTION_TEXT, width=130)
    CAP_LINE_H = 0.16       # very tight line spacing
    CAP_PAD_TOP = 0.14
    CAP_PAD_BOT = 0.06
    cap_block_h = len(cap_lines) * CAP_LINE_H + CAP_PAD_TOP + CAP_PAD_BOT

    TITLE_H  = 0.50
    BOTTOM_H = 0.08

    total_h = TITLE_H + SCALE_H + HDR + body_h + cap_block_h + BOTTOM_H

    SCALE = 0.72
    fig, ax = plt.subplots(figsize=(TW * SCALE, total_h * SCALE))
    ax.set_xlim(0, TW);  ax.set_ylim(0, total_h)
    ax.axis("off");  ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_WHITE)

    # ── helpers ───────────────────────────────────────────────────────────
    def rect(x, y, w, h, fc, ec=None, lw=0):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec or fc, lw=lw))

    def txt(x, y, s, fs=8, ha="center", va="center",
            color=C_TEXT, weight="normal", style="normal"):
        return ax.text(x, y, s, fontsize=fs, ha=ha, va=va,
                       color=color, fontweight=weight, fontstyle=style)

    def txt_uline(x, y, s, fs=8, color=C_TEXT, weight="normal"):
        t = txt(x, y, s, fs=fs, color=color, weight=weight)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        ax.plot([bb.x0, bb.x1], [bb.y0 - 0.02, bb.y0 - 0.02],
                color=color, lw=0.8)
        return t

    # ── (A) Title ─────────────────────────────────────────────────────────
    y_cur = total_h - TITLE_H / 2
    txt(TW / 2, y_cur,
        "Table 3  Individual Rule Performance  [REFERENCE TARGET]",
        fs=11, weight="bold", color=C_NAVY)

    # ── (B) Scale header ──────────────────────────────────────────────────
    y_sh_top = total_h - TITLE_H
    rect(0, y_sh_top - SCALE_H, TW, SCALE_H, C_NAVY)
    for si, lab in enumerate(["S", "M", "L", "XL"]):
        ci = 2 + si * 2
        mid = (col_x[ci] + col_x[ci] + col_widths[ci] + col_widths[ci + 1]) / 2
        txt(mid, y_sh_top - SCALE_H / 2, lab, fs=9, weight="bold", color=C_WHITE)
        x1 = col_x[ci] + 0.08
        x2 = col_x[ci] + col_widths[ci] + col_widths[ci + 1] - 0.08
        ax.plot([x1, x2], [y_sh_top - SCALE_H + 0.04] * 2,
                color=C_WHITE, lw=0.6, alpha=0.5)

    # ── (C) Column header ─────────────────────────────────────────────────
    y_hdr_top = y_sh_top - SCALE_H
    rect(0, y_hdr_top - HDR, TW, HDR, C_BLUE)
    col_labels = ["Rule", "Category",
                  "Cost", "Rej", "Cost", "Rej", "Cost", "Rej", "Cost", "Rej"]
    for ci, lab in enumerate(col_labels):
        ha = "left" if ci == 0 else "center"
        xp = col_x[ci] + 0.10 if ci == 0 else col_cx[ci]
        txt(xp, y_hdr_top - HDR / 2, lab, fs=8, weight="bold",
            color=C_WHITE, ha=ha)

    # ── (D) Data body ─────────────────────────────────────────────────────
    y_body_top = y_hdr_top - HDR
    ax.plot([0, TW], [y_body_top, y_body_top], color=C_NAVY, lw=0.3)

    all_rows = list(RULES) + [RL_ROW]
    sc_cost_idx = [2, 4, 6, 8]
    sc_rej_idx  = [3, 5, 7, 9]
    sc_names    = ["S", "M", "L", "XL"]

    y = y_body_top
    for ri, row in enumerate(all_rows):
        is_rl = (ri == n_rules)

        if ri in CAT_BOUNDARIES:
            y -= SEP
            ax.plot([0, TW], [y + SEP / 2] * 2, color=C_SEP, lw=0.4)

        if is_rl:
            y -= RL_SEP
            ax.plot([0, TW], [y + RL_SEP / 2 + 0.02] * 2, color=C_NAVY, lw=1.2)

        y -= ROW
        ym = y + ROW / 2

        bg = C_RL_BG if is_rl else (C_WHITE if ri % 2 == 0 else C_GRAY)
        rect(0, y, TW, ROW, bg)

        txt(col_x[0] + 0.10, ym, row[0], fs=7.5, ha="left",
            weight="bold" if is_rl else "normal",
            color=C_NAVY if is_rl else C_TEXT)

        txt(col_cx[1], ym, row[1], fs=7.5,
            weight="bold" if is_rl else "normal",
            style="normal" if is_rl else "italic",
            color=C_NAVY if is_rl else C_MTEXT)

        for si, sc in enumerate(sc_names):
            cv = row[sc_cost_idx[si]]
            rv = row[sc_rej_idx[si]]
            ci_c = 2 + si * 2
            ci_r = 3 + si * 2

            is_best = (not is_rl and ri == BEST_RULE_ROWS[sc])
            is_rl_s_win = (is_rl and sc == "S")  # only S beats all rules

            # cost
            if is_rl:
                cc = C_NAVY
                cw = "bold" if is_rl_s_win else "normal"
            elif is_best:
                cc = C_TEXT
                cw = "bold"
            else:
                cc = C_TEXT
                cw = "normal"

            if is_best:
                txt_uline(col_cx[ci_c], ym, fmt_cost(cv), fs=7.5,
                          color=cc, weight=cw)
            else:
                txt(col_cx[ci_c], ym, fmt_cost(cv), fs=7.5,
                    color=cc, weight=cw)

            # rej
            if is_rl:
                rc = C_NAVY
                rw = "bold" if is_rl_s_win else "normal"
            elif rv >= 10:
                rc = C_RED
                rw = "normal"
            else:
                rc = C_TEXT
                rw = "normal"

            txt(col_cx[ci_r], ym, fmt_rej(rv), fs=7.5, color=rc, weight=rw)

    # ── Bottom rule ───────────────────────────────────────────────────────
    ax.plot([0, TW], [y, y], color=C_NAVY, lw=1.2)

    # ── Vertical separators ───────────────────────────────────────────────
    for si in range(4):
        x = col_x[2 + si * 2]
        ax.plot([x, x], [y_sh_top, y], color=C_SEP, lw=0.3, alpha=0.4)

    # ── (E) Caption — justified, compact spacing ────────────────────────
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
        p = os.path.join(out_dir, f"table3_target_reference.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), pad_inches=0.08)
        print(f"\u2713 {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
