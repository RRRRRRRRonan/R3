#!/usr/bin/env python3
"""Generate 3×3 hyperparameter sensitivity heatmap (η × β) for M-scale.

Reads evaluation CSVs from results/benchmark/evaluate_M_hp_*.csv and computes
Option B clean cost for each (learning_rate, ent_coef) configuration.

Output: results/hp_sensitivity/hp_sensitivity_M.png / .csv
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Cost coefficients ────────────────────────────────────────────────────
C_TR, C_TIME, C_CH = 1.0, 0.1, 0.6
C_DELAY, C_WAIT, C_CONFLICT, C_STANDBY = 2.0, 0.05, 0.05, 0.05
REJECTION_PENALTY = 10_000
C_TERMINAL = 2500  # M-scale

# ── Font ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ── Grid ─────────────────────────────────────────────────────────────────
LRS = [1e-4, 3e-4, 1e-3]
ENTS = [0.01, 0.05, 0.10]
LR_LABELS = ["1e-4", "3e-4", "1e-3"]
ENT_LABELS = ["0.01", "0.05", "0.10"]
BASE_LR, BASE_ENT = 3e-4, 0.05

LABEL_MAP = {
    (1e-4, 0.01): "lr1e4_ent001",
    (1e-4, 0.05): "lr1e4_ent005",
    (1e-4, 0.10): "lr1e4_ent010",
    (3e-4, 0.01): "lr3e4_ent001",
    (3e-4, 0.05): "lr3e4_ent005",
    (3e-4, 0.10): "lr3e4_ent010",
    (1e-3, 0.01): "lr1e3_ent001",
    (1e-3, 0.05): "lr1e3_ent005",
    (1e-3, 0.10): "lr1e3_ent010",
}

BASE_COST = 48_745  # RL-APC (Full) M-scale from Table 4


def _clean_cost(df: pd.DataFrame) -> pd.Series:
    """Option B clean cost per instance."""
    oper = (
        df["metrics_total_distance"] * C_TR
        + df["metrics_total_travel_time"] * C_TIME
        + df["metrics_total_charging"] * C_CH
        + df["metrics_total_delay"] * C_DELAY
        + df["metrics_total_waiting"] * C_WAIT
        + (df["metrics_total_conflict_waiting"] + df["metrics_total_waiting"]) * C_CONFLICT
        + df["metrics_total_standby"] * C_STANDBY
    )
    reject = df["rejected_tasks"].astype(float) * REJECTION_PENALTY
    unfinished = (
        df["num_tasks_manifest"].astype(float)
        - df["completed_tasks"].astype(float)
        - df["rejected_tasks"].astype(float)
    ).clip(lower=0)
    terminal = unfinished * C_TERMINAL
    return oper + reject + terminal


def load_grid():
    """Load evaluation results for all 9 grid cells.

    Returns:
        costs: 3×3 array of mean clean costs (rows=LR, cols=ENT)
        stds: 3×3 array of std
        missing: list of (lr, ent) pairs with no data
    """
    costs = np.full((3, 3), np.nan)
    stds = np.full((3, 3), np.nan)
    missing = []

    for i, lr in enumerate(LRS):
        for j, ent in enumerate(ENTS):
            label = LABEL_MAP[(lr, ent)]
            csv_path = f"results/benchmark/evaluate_M_hp_{label}.csv"

            if not os.path.exists(csv_path):
                print(f"  [MISS] ({lr}, {ent}) — {csv_path} not found")
                missing.append((lr, ent))
                continue

            df = pd.read_csv(csv_path)
            ok = df[(df["status"] == "OK") & (df["algorithm"].isin(["RL-APC"]))]
            if ok.empty:
                ok = df[df["status"] == "OK"]
            if ok.empty:
                print(f"  [WARN] ({lr}, {ent}) — no OK rows in {csv_path}")
                missing.append((lr, ent))
                continue

            cc = _clean_cost(ok)
            costs[i, j] = cc.mean()
            stds[i, j] = cc.std()
            print(f"  ({lr}, {ent}): cost={cc.mean():,.0f} ± {cc.std():,.0f} "
                  f"[{len(ok)} instances]")

    return costs, stds, missing


def plot_heatmap(costs, stds, out_dir):
    """Generate annotated heatmap of cost increment vs base."""
    # Compute % increment relative to base
    delta_pct = (costs - BASE_COST) / BASE_COST * 100

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Color map: green (0%) → yellow → red (higher %)
    im = ax.imshow(delta_pct, cmap="RdYlGn_r", aspect="auto",
                   vmin=-5, vmax=max(50, np.nanmax(delta_pct) * 1.1))

    # Annotate each cell
    for i in range(3):
        for j in range(3):
            if np.isnan(delta_pct[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=11, color="gray")
                continue

            pct = delta_pct[i, j]
            cost = costs[i, j]
            is_base = (LRS[i] == BASE_LR and ENTS[j] == BASE_ENT)

            # Cell text
            marker = " ★" if is_base else ""
            txt = f"{pct:+.1f}%{marker}\n({cost:,.0f})"
            color = "white" if abs(pct) > 25 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, fontweight="bold" if is_base else "normal",
                    color=color)

    # Labels
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"β = {e}" for e in ENT_LABELS], fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"η = {l}" for l in LR_LABELS], fontsize=10)
    ax.set_xlabel("Entropy Coefficient (β)", fontsize=12)
    ax.set_ylabel("Learning Rate (η)", fontsize=12)
    ax.set_title("(c) Hyperparameter Sensitivity: Cost Increment vs. Base\n"
                 "(M-Scale, 500K steps, 30 test instances)",
                 fontsize=12, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.08)
    cbar.set_label("Δ Cost vs. Base (%)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))

    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"hp_sensitivity_M.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  ✓ {p}")
    plt.close(fig)


def main():
    out_dir = "results/hp_sensitivity"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading grid results ...")
    costs, stds, missing = load_grid()

    if missing:
        print(f"\n⚠ {len(missing)} configs missing: {missing}")
        print("  Run scripts/run_hp_sensitivity.sh first.")

    # Save summary CSV
    rows = []
    for i, lr in enumerate(LRS):
        for j, ent in enumerate(ENTS):
            is_base = (lr == BASE_LR and ent == BASE_ENT)
            delta = (costs[i, j] - BASE_COST) / BASE_COST * 100 if not np.isnan(costs[i, j]) else None
            rows.append({
                "learning_rate": lr,
                "ent_coef": ent,
                "label": LABEL_MAP[(lr, ent)],
                "mean_cost": round(costs[i, j], 1) if not np.isnan(costs[i, j]) else None,
                "std_cost": round(stds[i, j], 1) if not np.isnan(stds[i, j]) else None,
                "delta_pct": round(delta, 1) if delta is not None else None,
                "is_base": is_base,
            })
    summary = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "hp_sensitivity_M.csv")
    summary.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")

    # Print summary table
    print(f"\n{'LR':>8} {'Ent':>6} {'Cost':>10} {'Δ%':>8} {'Note':>6}")
    print("-" * 42)
    for _, r in summary.iterrows():
        note = "★ base" if r["is_base"] else ""
        cost_s = f"{r['mean_cost']:,.0f}" if r["mean_cost"] else "N/A"
        delta_s = f"{r['delta_pct']:+.1f}%" if r["delta_pct"] is not None else "N/A"
        print(f"{r['learning_rate']:>8} {r['ent_coef']:>6} {cost_s:>10} {delta_s:>8} {note:>6}")

    # Robustness check
    valid = summary.dropna(subset=["delta_pct"])
    if len(valid) > 0:
        reasonable = valid[(valid["learning_rate"] <= 3e-4) & (valid["ent_coef"] <= 0.05)]
        if len(reasonable) > 0:
            max_delta = reasonable["delta_pct"].max()
            print(f"\nIn reasonable range (η ≤ 3e-4, β ≤ 0.05): max Δ = {max_delta:+.1f}%")

    # Plot
    if not all(np.isnan(costs.flat)):
        print("\nGenerating heatmap ...")
        plot_heatmap(costs, stds, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
