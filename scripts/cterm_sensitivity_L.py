#!/usr/bin/env python3
"""
Cterm Sensitivity Analysis for L-Scale (Reviewer 3 Major 2)
============================================================

Analytical recomposition: one evaluation run, six Cterm values.

    total_cost_i = oper_cost_i + rejection_penalty_i + C_term × n_unfinished_i

For each instance we record three components from the existing evaluation CSVs,
then sweep C_term ∈ {500, 1000, 1500, 2000, 2500, 3000}.

Outputs:
    results/cterm_sensitivity/cterm_sensitivity_L.png   (two-panel figure)
    results/cterm_sensitivity/cterm_sensitivity_L.csv   (summary table)
    results/cterm_sensitivity/cterm_components_L.csv    (per-instance components)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Cost coefficients (src/config/defaults.py:CostParameters) ────────────
C_TR, C_TIME, C_CH = 1.0, 0.1, 0.6
C_DELAY, C_WAIT, C_CONFLICT, C_STANDBY = 2.0, 0.05, 0.05, 0.05
REJECTION_PENALTY = 10_000

PAPER_CTERM = 2000  # L-scale paper value (C_TERMINAL_PER_SCALE["L"])

DEFAULT_CTERMS = [500, 1000, 1500, 2000, 2500, 3000]

# ── Font ─────────────────────────────────────────────────────────────────
_FONT = "STIXGeneral"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [_FONT, "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


# ══════════════════════════════════════════════════════════════════════════
# Data loading & decomposition
# ══════════════════════════════════════════════════════════════════════════

def _oper_cost(df: pd.DataFrame) -> pd.Series:
    """Compute per-instance operational cost (Option B, no shaping)."""
    return (
        df["metrics_total_distance"] * C_TR
        + df["metrics_total_travel_time"] * C_TIME
        + df["metrics_total_charging"] * C_CH
        + df["metrics_total_delay"] * C_DELAY
        + df["metrics_total_waiting"] * C_WAIT
        + (df["metrics_total_conflict_waiting"] + df["metrics_total_waiting"]) * C_CONFLICT
        + df["metrics_total_standby"] * C_STANDBY
    )


def _load_components(csv_path: str, algorithm_filter: str) -> pd.DataFrame:
    """Load per-instance cost components from an evaluation CSV.

    Returns DataFrame with columns: seed, oper, reject, unfinished.
    """
    df = pd.read_csv(csv_path)
    # Filter by algorithm name or algorithm_id
    mask = (df["algorithm"] == algorithm_filter) | (df["algorithm_id"] == algorithm_filter)
    sub = df[mask].copy()
    if sub.empty:
        raise ValueError(f"No rows for '{algorithm_filter}' in {csv_path}")
    sub = sub[sub["status"] == "OK"]

    oper = _oper_cost(sub)
    reject = sub["rejected_tasks"].astype(float) * REJECTION_PENALTY
    unfinished = (
        sub["num_tasks_manifest"].astype(float)
        - sub["completed_tasks"].astype(float)
        - sub["rejected_tasks"].astype(float)
    ).clip(lower=0)

    return pd.DataFrame({
        "seed": sub["seed"].values,
        "oper": oper.values,
        "reject": reject.values,
        "unfinished": unfinished.values,
        "completed": sub["completed_tasks"].values,
        "rejected": sub["rejected_tasks"].values,
    })


def _sweep(comp: pd.DataFrame, cterms: list) -> pd.DataFrame:
    """For each Cterm value, compute total clean cost per instance.

    Returns DataFrame: rows = instances, columns = Cterm values.
    """
    result = {}
    for ct in cterms:
        result[ct] = comp["oper"] + comp["reject"] + ct * comp["unfinished"]
    return pd.DataFrame(result, index=comp.index)


# ══════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════

def _plot(rl_sweep: pd.DataFrame, ch_sweep: pd.DataFrame,
          cterms: list, out_dir: str):
    """Two-panel figure: (a) cost curves, (b) gap %."""

    rl_mean = rl_sweep.mean()
    rl_std = rl_sweep.std()
    ch_mean = ch_sweep.mean()
    ch_std = ch_sweep.std()

    gap_pct = (rl_mean - ch_mean) / ch_mean * 100  # negative = RL cheaper

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # ── Panel (a): Cost curves ───────────────────────────────────────────
    x = np.array(cterms)

    ax1.errorbar(x, ch_mean, yerr=ch_std, fmt="s-", color="#2C5F8A",
                 capsize=4, markersize=6, linewidth=1.8, label="Charge-High")
    ax1.errorbar(x, rl_mean, yerr=rl_std, fmt="o-", color="#C0392B",
                 capsize=4, markersize=6, linewidth=1.8, label="RL-APC")

    # Mark paper value
    ax1.axvline(PAPER_CTERM, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax1.text(PAPER_CTERM + 50, ax1.get_ylim()[0], f"paper\n($C_{{term}}$={PAPER_CTERM})",
             fontsize=8, color="gray", va="bottom")

    ax1.set_xlabel("Terminal Penalty ($C_{term}$)", fontsize=11)
    ax1.set_ylabel("Average Total Cost (Option B)", fontsize=11)
    ax1.set_title("(a) Cost vs. Terminal Penalty", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}K"))
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticks(cterms)

    # ── Panel (b): Gap % ─────────────────────────────────────────────────
    bars = ax2.bar(x, gap_pct, width=350, color=["#27AE60" if g < 0 else "#E74C3C" for g in gap_pct],
                   edgecolor="white", linewidth=0.5)
    for bar, g in zip(bars, gap_pct):
        y_pos = bar.get_height()
        va = "top" if g < 0 else "bottom"
        offset = -1.5 if g < 0 else 1.5
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos + offset,
                 f"{g:.1f}%", ha="center", va=va,
                 fontsize=9, fontweight="bold", color="#1A1A1A")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axvline(PAPER_CTERM, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax2.set_xlabel("Terminal Penalty ($C_{term}$)", fontsize=11)
    ax2.set_ylabel("Cost Gap (%): negative = RL-APC advantage", fontsize=11)
    ax2.set_title("(b) Relative Cost Advantage", fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(cterms)

    fig.suptitle("L-Scale: RL-APC vs. Charge-High under Varying Terminal Penalty",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"cterm_sensitivity_L.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
        print(f"  ✓ {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cterm sensitivity analysis (L-scale)")
    parser.add_argument("--cterm", type=int, nargs="+", default=DEFAULT_CTERMS,
                        help="Cterm values to sweep (default: 500 1000 1500 2000 2500 3000)")
    parser.add_argument("--out", type=str, default="results/cterm_sensitivity",
                        help="Output directory")
    parser.add_argument("--rl-csv", type=str,
                        default="results/benchmark/evaluate_L_v3_30.csv",
                        help="RL-APC evaluation CSV")
    parser.add_argument("--rules-csv", type=str,
                        default="results/benchmark/individual_rules_L_30.csv",
                        help="Individual rules evaluation CSV")
    args = parser.parse_args()

    cterms = sorted(args.cterm)
    os.makedirs(args.out, exist_ok=True)

    print("Loading RL-APC components ...")
    rl_comp = _load_components(args.rl_csv, "RL-APC")
    print(f"  {len(rl_comp)} instances, avg completed={rl_comp['completed'].mean():.2f}, "
          f"rejected={rl_comp['rejected'].mean():.2f}, unfinished={rl_comp['unfinished'].mean():.2f}")

    print("Loading Charge-High components ...")
    ch_comp = _load_components(args.rules_csv, "Charge-High")
    print(f"  {len(ch_comp)} instances, avg completed={ch_comp['completed'].mean():.2f}, "
          f"rejected={ch_comp['rejected'].mean():.2f}, unfinished={ch_comp['unfinished'].mean():.2f}")

    # Save per-instance components
    comp_path = os.path.join(args.out, "cterm_components_L.csv")
    combined = pd.concat([
        rl_comp.assign(method="RL-APC"),
        ch_comp.assign(method="Charge-High"),
    ], ignore_index=True)
    combined.to_csv(comp_path, index=False)
    print(f"  ✓ {comp_path}")

    # Sweep
    print(f"\nSweeping Cterm = {cterms} ...")
    rl_sweep = _sweep(rl_comp, cterms)
    ch_sweep = _sweep(ch_comp, cterms)

    # Summary table
    rows = []
    for ct in cterms:
        rl_m, rl_s = rl_sweep[ct].mean(), rl_sweep[ct].std()
        ch_m, ch_s = ch_sweep[ct].mean(), ch_sweep[ct].std()
        gap = (rl_m - ch_m) / ch_m * 100
        rows.append({
            "Cterm": ct,
            "RL-APC_mean": round(rl_m, 1),
            "RL-APC_std": round(rl_s, 1),
            "Charge-High_mean": round(ch_m, 1),
            "Charge-High_std": round(ch_s, 1),
            "gap_pct": round(gap, 1),
            "RL_cheaper": gap < 0,
        })

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "cterm_sensitivity_L.csv")
    summary.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")

    # Print summary
    print(f"\n{'Cterm':>6} {'RL-APC':>12} {'Charge-High':>14} {'Gap':>10} {'Winner':>10}")
    print("-" * 56)
    for _, r in summary.iterrows():
        winner = "RL-APC" if r["RL_cheaper"] else "CH"
        print(f"{r['Cterm']:>6} {r['RL-APC_mean']:>12,.0f} {r['Charge-High_mean']:>14,.0f} "
              f"{r['gap_pct']:>+9.1f}% {winner:>10}")

    # Verdict
    all_rl_wins = all(r["RL_cheaper"] for _, r in summary.iterrows())
    print()
    if all_rl_wins:
        print("RESULT: RL-APC is cheaper than Charge-High across ALL tested Cterm values.")
        print("The L-scale cost advantage is robust to the terminal-penalty parameterisation.")
    else:
        crossover = [r["Cterm"] for _, r in summary.iterrows() if not r["RL_cheaper"]]
        print(f"RESULT: Charge-High is cheaper at Cterm = {crossover}.")

    # Plot
    print("\nGenerating figure ...")
    _plot(rl_sweep, ch_sweep, cterms, args.out)

    print("\nDone!")


if __name__ == "__main__":
    main()
