"""Generate results/full_comparison_report.txt from the latest evaluation CSVs.

Automatically picks the best available evaluation for each scale:
  - L: v3 > synced > best_model (in priority order)
  - M/XL: synced > best_model
  - S: best_model

Re-run this script whenever new evaluation results are available.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "results" / "benchmark"
OUTPUT = ROOT / "results" / "full_comparison_report.txt"

ALGO_ORDER = ["rl_apc", "greedy_fr", "greedy_pr", "random_rule", "alns_fr", "alns_pr"]
ALGO_DISPLAY = {
    "rl_apc": "RL-APC",
    "greedy_fr": "Greedy-FR",
    "greedy_pr": "Greedy-PR",
    "random_rule": "Random",
    "alns_fr": "ALNS-FR",
    "alns_pr": "ALNS-PR",
}
TASK_COUNTS = {"S": "15-20", "M": "30-40", "L": "50-60", "XL": "80-100"}


def _pick_best_csv(scale: str) -> tuple[str | None, str]:
    """Return (csv_path, label) for the best available evaluation of *scale*."""
    candidates = [
        (f"evaluate_{scale}_v3_30.csv", "v3 best_model"),
        (f"evaluate_{scale}_synced_30.csv", "final_model (synced)"),
        (f"evaluate_{scale}_v2_30.csv", "v2 best_model"),
        (f"evaluate_{scale}_30.csv", "best_model"),
    ]
    for fname, label in candidates:
        p = BENCHMARK / fname
        if p.exists():
            return str(p), label
    return None, "N/A"


def _load_rows(path: str):
    with open(path) as f:
        return list(csv.DictReader(f))


def _group_by_algo(rows):
    by_algo: dict[str, list[dict]] = {}
    for row in rows:
        aid = row["algorithm_id"]
        d = {
            "cost": float(row["cost"]),
            "completed": float(row.get("completed_tasks", 0) or 0),
            "rejected": float(row.get("rejected_tasks", 0) or 0),
            "delay": float(row.get("metrics_total_delay", 0) or 0),
            "travel": float(row.get("metrics_total_distance", 0) or 0),
            "charging": float(row.get("metrics_total_charging", 0) or 0),
            "standby": float(row.get("metrics_total_standby", 0) or 0),
            "waiting": float(row.get("metrics_total_waiting", 0) or 0),
            "runtime": float(row.get("runtime_s", 0) or 0),
            "steps": int(row.get("steps", 0) or 0),
        }
        by_algo.setdefault(aid, []).append(d)
    return by_algo


def generate_report() -> str:
    out: list[str] = []

    # Resolve best CSV per scale
    scale_sources: dict[str, tuple[str, str]] = {}
    all_data: dict[str, dict] = {}
    for scale in ["S", "M", "L", "XL"]:
        path, label = _pick_best_csv(scale)
        if path is None:
            continue
        scale_sources[scale] = (path, label)
        all_data[scale] = _group_by_algo(_load_rows(path))

    out.append("=" * 120)
    out.append("FULL COMPARISON REPORT — RL-APC vs Baselines (auto-generated)")
    out.append("=" * 120)
    out.append("")
    out.append("Data sources:")
    for scale in ["S", "M", "L", "XL"]:
        if scale in scale_sources:
            path, label = scale_sources[scale]
            out.append(f"  {scale}: {os.path.basename(path)}  ({label})")
    out.append("")

    # ---- TABLE 1 ----
    out.append("=" * 120)
    out.append("TABLE 1: Algorithm Comparison — Average Total Cost (30 test instances per scale)")
    out.append("=" * 120)
    hdr = f"{'Scale':<6} {'RL-APC':>12} {'Greedy-FR':>12} {'Greedy-PR':>12} {'Random':>12} {'ALNS-FR':>12} {'ALNS-PR':>12}  {'RL vs Greedy':>14} {'RL Rank':>8}"
    out.append(hdr)
    out.append("-" * 120)

    for scale in ["S", "M", "L", "XL"]:
        if scale not in all_data:
            continue
        by_algo = all_data[scale]
        costs = {a: np.mean([d["cost"] for d in by_algo[a]]) for a in ALGO_ORDER if a in by_algo}
        sorted_a = sorted(costs.items(), key=lambda x: x[1])
        rl_rank = next(i + 1 for i, (a, _) in enumerate(sorted_a) if a == "rl_apc")
        gr = costs.get("greedy_fr", 1)
        rl = costs.get("rl_apc", 0)
        diff = (rl - gr) / gr * 100
        parts = [f"{scale:<6}"]
        for a in ALGO_ORDER:
            v = costs.get(a)
            parts.append(f"{v:>12,.0f}" if v is not None else f"{'—':>12}")
        parts.append(f"  {diff:>+12.1f}%")
        parts.append(f"{rl_rank:>6}/6")
        out.append("".join(parts))

    out.append("")
    out.append("Note: ALNS-FR/PR are offline batch optimizers (unfair comparison with online methods)")
    out.append("")

    # ---- TABLE 2 ----
    out.append("=" * 120)
    out.append("TABLE 2: Detailed Metrics per Algorithm")
    out.append("=" * 120)
    out.append(f"{'Scale':<6} {'Algorithm':<12} {'Avg Cost':>12} {'Std Cost':>12} {'Completed':>10} {'Rejected':>10} {'Avg Delay':>12} {'Runtime(s)':>10}")
    out.append("-" * 120)

    for scale in ["S", "M", "L", "XL"]:
        if scale not in all_data:
            continue
        by_algo = all_data[scale]
        for a in ALGO_ORDER:
            if a not in by_algo:
                continue
            vals = by_algo[a]
            c = [d["cost"] for d in vals]
            comp = [d["completed"] for d in vals]
            rej = [d["rejected"] for d in vals]
            delay = [d["delay"] for d in vals]
            rt = [d["runtime"] for d in vals]
            comp_s = f"{np.mean(comp):>10.1f}" if any(v > 0 for v in comp) else f"{'—':>10}"
            rej_s = (
                f"{np.mean(rej):>10.1f}"
                if any(v > 0 for v in rej) or a in ("rl_apc", "greedy_fr", "greedy_pr", "random_rule")
                else f"{'—':>10}"
            )
            delay_s = f"{np.mean(delay):>12.1f}" if any(v > 0 for v in delay) else f"{'—':>12}"
            out.append(
                f"{scale:<6} {ALGO_DISPLAY[a]:<12} {np.mean(c):>12,.0f} {np.std(c):>12,.0f} "
                f"{comp_s} {rej_s} {delay_s} {np.mean(rt):>10.2f}"
            )
        out.append("")

    # ---- TABLE 3 ----
    out.append("=" * 120)
    out.append("TABLE 3: Cost Decomposition (RL-APC vs Greedy-FR)")
    out.append("=" * 120)
    out.append(f"{'Scale':<6} {'Algorithm':<12} {'Travel':>10} {'Charging':>10} {'Tardiness':>10} {'Waiting':>10} {'Standby':>10} {'Total Cost':>12}")
    out.append("-" * 120)

    for scale in ["S", "M", "L", "XL"]:
        if scale not in all_data:
            continue
        by_algo = all_data[scale]
        for a in ["rl_apc", "greedy_fr"]:
            if a not in by_algo:
                continue
            vals = by_algo[a]
            out.append(
                f"{scale:<6} {ALGO_DISPLAY[a]:<12} "
                f"{np.mean([d['travel'] for d in vals]):>10,.1f} "
                f"{np.mean([d['charging'] for d in vals]):>10,.1f} "
                f"{np.mean([d['delay'] for d in vals]):>10,.1f} "
                f"{np.mean([d['waiting'] for d in vals]):>10,.1f} "
                f"{np.mean([d['standby'] for d in vals]):>10,.0f} "
                f"{np.mean([d['cost'] for d in vals]):>12,.0f}"
            )
        out.append("")

    # ---- TABLE 4 ----
    out.append("=" * 120)
    out.append("TABLE 4: Wilcoxon Signed-Rank Tests (RL-APC vs each baseline, two-sided)")
    out.append("=" * 120)
    out.append(f"{'Scale':<6} {'Baseline':<12} {'RL Mean':>12} {'BL Mean':>12} {'Diff%':>8} {'p-value':>12} {'Sig':>5} {'W/T/L':>8}")
    out.append("-" * 120)

    for scale in ["S", "M", "L", "XL"]:
        if scale not in all_data:
            continue
        by_algo = all_data[scale]
        rl_costs = np.array([d["cost"] for d in by_algo["rl_apc"]])
        for bl in ["greedy_fr", "random_rule", "alns_fr", "alns_pr"]:
            if bl not in by_algo:
                continue
            bl_costs = np.array([d["cost"] for d in by_algo[bl]])
            if len(rl_costs) != len(bl_costs):
                continue
            diff = (rl_costs.mean() - bl_costs.mean()) / bl_costs.mean() * 100
            try:
                _, p = scipy_stats.wilcoxon(rl_costs, bl_costs)
            except Exception:
                p = 1.0
            w = int(np.sum(rl_costs < bl_costs))
            t = int(np.sum(rl_costs == bl_costs))
            l = len(rl_costs) - w - t
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            out.append(
                f"{scale:<6} {ALGO_DISPLAY[bl]:<12} {rl_costs.mean():>12,.0f} {bl_costs.mean():>12,.0f} "
                f"{diff:>+7.1f}% {p:>12.2e} {sig:>5} {w}/{t}/{l}"
            )
        out.append("")

    # ---- TABLE 5 ----
    out.append("=" * 120)
    out.append("TABLE 5: Service Quality Comparison (RL-APC vs Greedy-FR)")
    out.append("=" * 120)
    out.append(f"{'Scale':<6} {'Tasks':>6} {'':2} {'RL Completed':>13} {'GR Completed':>13} {'RL Rejected':>12} {'GR Rejected':>12} {'RL Delay':>10} {'GR Delay':>10}")
    out.append("-" * 120)

    for scale in ["S", "M", "L", "XL"]:
        if scale not in all_data:
            continue
        by_algo = all_data[scale]
        rl = by_algo["rl_apc"]
        gr = by_algo["greedy_fr"]
        out.append(
            f"{scale:<6} {TASK_COUNTS[scale]:>6}   "
            f"{np.mean([d['completed'] for d in rl]):>13.1f} "
            f"{np.mean([d['completed'] for d in gr]):>13.1f} "
            f"{np.mean([d['rejected'] for d in rl]):>12.1f} "
            f"{np.mean([d['rejected'] for d in gr]):>12.1f} "
            f"{np.mean([d['delay'] for d in rl]):>10.1f} "
            f"{np.mean([d['delay'] for d in gr]):>10.1f}"
        )

    out.append("")

    # ---- TABLE 6 ----
    out.append("=" * 120)
    out.append("TABLE 6: VecNormalize Synchronization Impact")
    out.append("=" * 120)
    out.append(f"{'Scale':<6} {'Greedy':>12} {'RL(mismatched)':>16} {'RL(synced)':>14} {'Mismatch Gap':>14} {'Sync Improvement':>18}")
    out.append("-" * 120)

    for scale in ["S", "M", "L", "XL"]:
        mis_f = BENCHMARK / f"evaluate_{scale}_30_summary.json"
        if not mis_f.exists():
            continue
        d_mis = json.load(open(mis_f))
        gr = d_mis["algorithms"]["greedy_fr"]["avg_cost"]
        rl_mis = d_mis["algorithms"]["rl_apc"]["avg_cost"]
        gap = f"{(rl_mis - gr) / gr * 100:>+.1f}%"

        # Try synced
        syn_f = BENCHMARK / f"evaluate_{scale}_synced_30_summary.json"
        if syn_f.exists():
            d_syn = json.load(open(syn_f))
            rl_syn = d_syn["algorithms"]["rl_apc"]["avg_cost"]
            imp = f"{(rl_mis - rl_syn) / rl_mis * 100:>+.1f}%"
            out.append(f"{scale:<6} {gr:>12,.0f} {rl_mis:>16,.0f} {rl_syn:>14,.0f} {gap:>14} {imp:>18}")
        else:
            out.append(f"{scale:<6} {gr:>12,.0f} {rl_mis:>16,.0f} {'—':>14} {gap:>14} {'—':>18}")

    out.append("")
    out.append("=" * 120)

    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out.append(f"Generated: {now} | 30 test instances per scale | Wilcoxon two-sided test")
    out.append("Best available evaluation per scale (auto-selected)")
    out.append("=" * 120)

    return "\n".join(out)


def main():
    text = generate_report()
    print(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"\n→ Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
