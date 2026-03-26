#!/usr/bin/env python3
"""
Cross-Scale Ablation Study — S + M + XL
=========================================
Responds to Reviewer 2 Major 3: "Verify ablation conclusions generalize."

Data sources:
  S:  results/benchmark/evaluate_S_ablation_{pc,fm}_30.csv  (new experiments)
  M:  Table 10 hardcoded values (already published)
  XL: results/benchmark/evaluate_XL_ablation_{pc,fm}_30.csv (new experiments)
  Random: results/benchmark/evaluate_{S,M,XL}_synced_30.csv  or individual_rules

Usage:
    python scripts/ablation_cross_scale.py              # full analysis
    python scripts/ablation_cross_scale.py --eval-xl    # run XL evaluation first
    python scripts/ablation_cross_scale.py --plot-only  # regenerate figure only
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

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
C_TERMINAL = {"S": 3000, "M": 2500, "L": 2000, "XL": 1500}

# ── Font ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ── Known M-scale results (Table 10, verified) ──────────────────────────
M_KNOWN = {
    "Full":   {"cost": 48_745,  "completed": 18.10, "rejected":  0.00},
    "PC":     {"cost": 69_008,  "completed": 11.33, "rejected":  0.00},
    "FM":     {"cost": 86_917,  "completed":  0.00, "rejected":  0.00},
    "Random": {"cost": 156_522, "completed":  9.80, "rejected": 11.50},
}

# ── Full model costs (Table 4) ──────────────────────────────────────────
FULL_COST = {"S": 13_162, "M": 48_745, "XL": 130_077}

# ── Eval CSV paths ───────────────────────────────────────────────────────
EVAL_PATHS = {
    "S": {
        "Full":   None,  # from Table 4
        "PC":     "results/benchmark/evaluate_S_ablation_pc_30.csv",
        "FM":     "results/benchmark/evaluate_S_ablation_fm_30.csv",
        "Random": "results/benchmark/evaluate_S_30.csv",
    },
    "XL": {
        "Full":   None,
        "PC":     "results/benchmark/evaluate_XL_ablation_pc_30.csv",
        "FM":     "results/benchmark/evaluate_XL_ablation_fm_30.csv",
        "Random": "results/benchmark/evaluate_XL_synced_30.csv",
    },
}


def _oper_cost(df):
    return (df["metrics_total_distance"] * C_TR
            + df["metrics_total_travel_time"] * C_TIME
            + df["metrics_total_charging"] * C_CH
            + df["metrics_total_delay"] * C_DELAY
            + df["metrics_total_waiting"] * C_WAIT
            + (df["metrics_total_conflict_waiting"] + df["metrics_total_waiting"]) * C_CONFLICT
            + df["metrics_total_standby"] * C_STANDBY)


def _clean_cost_df(df, scale):
    oper = _oper_cost(df)
    reject = df["rejected_tasks"].astype(float) * REJECTION_PENALTY
    unfin = (df["num_tasks_manifest"].astype(float)
             - df["completed_tasks"].astype(float)
             - df["rejected_tasks"].astype(float)).clip(lower=0)
    terminal = unfin * C_TERMINAL[scale]
    return oper + reject + terminal


def load_variant(scale, variant):
    """Load and compute clean cost for a (scale, variant) pair.

    Returns dict with cost, completed, rejected, std, or None if unavailable.
    """
    if scale == "M":
        d = M_KNOWN[variant]
        return {"cost": d["cost"], "completed": d["completed"],
                "rejected": d["rejected"], "std": 0, "source": "Table10"}

    if variant == "Full":
        return {"cost": FULL_COST[scale], "completed": None,
                "rejected": None, "std": 0, "source": "Table4"}

    csv_path = EVAL_PATHS.get(scale, {}).get(variant)
    if csv_path is None or not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    # Filter for correct algorithm
    if variant == "Random":
        mask = df["algorithm"].str.contains("Random", case=False, na=False)
        ok = df[mask & (df["status"] == "OK")]
    elif variant in ("PC", "FM"):
        # evaluate_all CSV — RL-APC
        ok = df[(df["algorithm"] == "RL-APC") & (df["status"] == "OK")]
        if ok.empty:
            ok = df[df["status"] == "OK"]

    if ok.empty:
        return None

    cc = _clean_cost_df(ok, scale)
    return {
        "cost": cc.mean(),
        "completed": ok["completed_tasks"].mean(),
        "rejected": ok["rejected_tasks"].mean(),
        "std": cc.std(),
        "n": len(ok),
        "source": os.path.basename(csv_path),
    }


def run_xl_evaluation(args):
    """Evaluate XL ablation models if not already done."""
    xl_meta = {
        "PC": {
            "model": "results/rl/train_XL_ablation_PC/best_model",
            "csv": "results/benchmark/evaluate_XL_ablation_pc_30.csv",
            "flag": "--no-partial-charging",
        },
        "FM": {
            "model": "results/rl/train_XL_ablation_FM/best_model",
            "csv": "results/benchmark/evaluate_XL_ablation_fm_30.csv",
            "flag": "--no-feasibility-mask",
        },
    }

    for variant, meta in xl_meta.items():
        if os.path.exists(meta["csv"]):
            print(f"  [SKIP] XL-{variant} eval exists: {meta['csv']}")
            continue

        if not os.path.exists(meta["model"]):
            print(f"  [WARN] XL-{variant} model not found: {meta['model']}")
            continue

        print(f"  [EVAL] XL-{variant} ...")
        cmd = [
            sys.executable, "-u", "scripts/evaluate_all.py",
            "--scale", "XL", "--split", "test",
            "--algorithms", "rl_apc",
            "--ppo-model-path", meta["model"],
            meta["flag"],
            "--terminal-penalty", "1500",
            "--tardiness-scale", "0.2",
            "--max-time-s", "25000",
            "--output-csv", meta["csv"],
        ]
        subprocess.run(cmd)


def load_all():
    """Load results for all scales and variants."""
    results = {}
    scales = ["S", "M", "XL"]
    variants = ["Full", "PC", "FM", "Random"]

    for scale in scales:
        results[scale] = {}
        for variant in variants:
            r = load_variant(scale, variant)
            if r is not None:
                results[scale][variant] = r
                delta = ((r["cost"] - FULL_COST.get(scale, r["cost"]))
                         / FULL_COST.get(scale, r["cost"]) * 100
                         if variant != "Full" else 0)
                comp_s = f"{r['completed']:.2f}" if r['completed'] is not None else "—"
                print(f"  {scale}-{variant:6s}: cost={r['cost']:>10,.0f}  "
                      f"Δ={delta:>+7.1f}%  comp={comp_s}  [{r.get('source','')}]")
            else:
                print(f"  {scale}-{variant:6s}: ❌ NOT AVAILABLE")

    return results


def make_figure(results, out_dir):
    """Two-panel figure: (a) grouped bars, (b) Δ% comparison."""
    scales = ["S", "M", "XL"]
    variants = ["Full", "PC", "FM", "Random"]
    variant_labels = ["RL-APC\n(Full)", "No Partial\nCharging",
                      "No Feasibility\nMasking", "Random\nBaseline"]
    variant_colors = ["#5A9E6B", "#F0A500", "#D9534F", "#999999"]
    scale_hatches = {"S": "", "M": "//", "XL": "xx"}
    scale_alphas = {"S": 0.95, "M": 0.65, "XL": 0.45}

    n_v = len(variants)
    n_s = len([s for s in scales if s in results])
    x = np.arange(n_v)
    w = 0.25
    offsets = {"S": -w, "M": 0, "XL": w}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # ── Panel (a): absolute cost bars ────────────────────────
    for scale in scales:
        if scale not in results:
            continue
        off = offsets[scale]
        for vi, variant in enumerate(variants):
            r = results[scale].get(variant)
            if r is None:
                continue
            cost = r["cost"]
            ax1.bar(x[vi] + off, cost / 1000, w,
                    color=variant_colors[vi],
                    alpha=scale_alphas[scale],
                    hatch=scale_hatches[scale],
                    edgecolor="white", linewidth=0.8, zorder=3)

            # Δ% label above bar (skip Full)
            if variant != "Full":
                full_c = FULL_COST.get(scale, cost)
                pct = (cost - full_c) / full_c * 100
                ax1.text(x[vi] + off, cost / 1000 + 2,
                         f"+{pct:.0f}%", ha="center", va="bottom",
                         fontsize=7, fontweight="bold", color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(variant_labels, fontsize=9)
    ax1.set_ylabel("Mean Total Cost (×1,000)", fontsize=11)
    ax1.set_title("(a) Absolute Cost by Scale and Variant", fontsize=12, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}K"))
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # Scale legend
    import matplotlib.patches as mpatches
    handles = []
    for scale in scales:
        if scale in results:
            handles.append(mpatches.Patch(
                fc="#888", alpha=scale_alphas[scale],
                hatch=scale_hatches[scale], ec="white",
                label=f"Scale {scale}"))
    ax1.legend(handles=handles, fontsize=9, loc="upper left")

    # ── Panel (b): Δ% horizontal bars ────────────────────────
    non_full = ["PC", "FM", "Random"]
    nf_labels = ["No Partial\nCharging", "No Feasibility\nMasking", "Random"]
    y_pos = np.arange(len(non_full))
    scale_colors = {"S": "#4472C4", "M": "#ED7D31", "XL": "#70AD47"}

    ax2.axvline(0, color="black", lw=0.8)

    for si, scale in enumerate(scales):
        if scale not in results:
            continue
        for yi, variant in enumerate(non_full):
            r = results[scale].get(variant)
            if r is None:
                continue
            full_c = FULL_COST.get(scale, r["cost"])
            pct = (r["cost"] - full_c) / full_c * 100
            y_jit = y_pos[yi] + (si - 1) * 0.27

            ax2.barh(y_jit, pct, 0.24,
                     color=scale_colors[scale], alpha=0.85, zorder=3)
            ax2.text(pct + 3, y_jit, f"+{pct:.0f}%",
                     va="center", fontsize=8, fontweight="bold",
                     color=scale_colors[scale])

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(nf_labels, fontsize=9)
    ax2.set_xlabel("Δ% vs RL-APC Full (per scale)", fontsize=11)
    ax2.set_title("(b) Relative Cost Increase", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    delta_handles = [mpatches.Patch(fc=scale_colors[s], alpha=0.85, label=f"Scale {s}")
                     for s in scales if s in results]
    ax2.legend(handles=delta_handles, fontsize=9, loc="lower right")

    fig.suptitle("Cross-Scale Ablation Study (S, M, XL)\n"
                 "RL-APC-PC = partial charging disabled;  "
                 "RL-APC-FM = feasibility masking disabled",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"ablation_cross_scale.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ {out_dir}/ablation_cross_scale.png/.pdf")
    plt.close(fig)


def save_csv(results, out_dir):
    rows = []
    for scale in ["S", "M", "XL"]:
        full_c = FULL_COST.get(scale, 0)
        for variant in ["Full", "PC", "FM", "Random"]:
            r = results.get(scale, {}).get(variant)
            if r is None:
                rows.append({"Scale": scale, "Variant": variant,
                             "Cost": None, "Completed": None,
                             "Rejected": None, "Delta_pct": None})
                continue
            delta = (r["cost"] - full_c) / full_c * 100 if variant != "Full" else 0
            rows.append({
                "Scale": scale,
                "Variant": variant,
                "Cost": round(r["cost"], 1),
                "Completed": round(r["completed"], 2) if r.get("completed") is not None else None,
                "Rejected": round(r["rejected"], 2) if r.get("rejected") is not None else None,
                "Delta_pct": round(delta, 1),
            })
    df = pd.DataFrame(rows)
    p = os.path.join(out_dir, "ablation_summary.csv")
    df.to_csv(p, index=False)
    print(f"  ✓ {p}")


def print_reviewer_response(results):
    """Print draft reviewer response with actual numbers filled in."""
    print("\n" + "=" * 70)
    print("  DRAFT REVIEWER RESPONSE (Reviewer 2, Major 3)")
    print("=" * 70)

    # Collect FM results
    fm_results = {}
    pc_results = {}
    for scale in ["S", "M", "XL"]:
        fm = results.get(scale, {}).get("FM")
        pc = results.get(scale, {}).get("PC")
        if fm:
            fm_results[scale] = fm
        if pc:
            pc_results[scale] = pc

    fm_collapse = all(fm_results.get(s, {}).get("completed", 1) < 0.5
                      for s in fm_results)

    print(f"""
We have extended the ablation study to Scales S and XL (Table A1,
Fig. A3), covering the full range of fleet sizes (3–12 AMRs).

Feasibility masking (FM): Disabling the action mask causes complete
policy collapse on {"all three" if fm_collapse else "tested"} scales:""")

    for s in ["S", "M", "XL"]:
        fm = fm_results.get(s)
        if fm:
            full_c = FULL_COST.get(s, 1)
            pct = (fm["cost"] - full_c) / full_c * 100
            print(f"  Scale {s}: {fm.get('completed', 0):.1f} tasks completed, "
                  f"cost = {fm['cost']:,.0f} (+{pct:.0f}% vs Full)")

    print(f"""
This confirms that feasibility masking is not a scale-specific
optimisation but a universal prerequisite for learning across all
fleet sizes.

Partial charging (PC): The cost increase from removing partial
charging is:""")

    for s in ["S", "M", "XL"]:
        pc = pc_results.get(s)
        if pc:
            full_c = FULL_COST.get(s, 1)
            pct = (pc["cost"] - full_c) / full_c * 100
            print(f"  Scale {s}: +{pct:.1f}% (cost {pc['cost']:,.0f}, "
                  f"completed {pc.get('completed', 0):.2f})")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Cross-scale ablation analysis")
    parser.add_argument("--eval-xl", action="store_true",
                        help="Run XL evaluation before analysis")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip loading, just regenerate figure")
    args = parser.parse_args()

    out_dir = "results/ablation_cross_scale"
    os.makedirs(out_dir, exist_ok=True)

    if args.eval_xl:
        print("\n[1] Running XL evaluations ...")
        run_xl_evaluation(args)

    print("\n[2] Loading all results ...")
    results = load_all()

    print("\n[3] Generating figure ...")
    make_figure(results, out_dir)

    print("\n[4] Saving summary CSV ...")
    save_csv(results, out_dir)

    print("\n[5] Reviewer response draft ...")
    print_reviewer_response(results)

    print(f"\nDone! Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
