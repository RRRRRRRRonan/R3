"""Generate EJOR-quality paper tables (LaTeX + CSV).

Restructured for EJOR submission with progressive argumentation:
  T1: Problem instances
  T2: RL training configuration
  T3: Individual rule performance (15 rules × 4 scales)
  T4: RL-APC vs best fixed rule per scale
  T5: Wilcoxon tests (enhanced with Bonferroni)
  T6: Service quality (enhanced with completion rate)
  T7: Online-offline gap
  T8: Cost decomposition (with rejection penalty)
  T9: Computational efficiency

Output: results/paper/ejor_tables.tex + individual CSVs
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "results" / "benchmark"
PAPER = ROOT / "results" / "paper"

ALGO_DISPLAY = {
    "rl_apc": "RL-APC",
    "greedy_fr": "Greedy-FR",
    "greedy_pr": "Greedy-PR",
    "random_rule": "Random",
    "alns_fr": "ALNS-FR",
    "alns_pr": "ALNS-PR",
}
TASK_COUNTS = {"S": "15--20", "M": "30--40", "L": "50--60", "XL": "80--100"}
VEHICLE_COUNTS = {"S": 3, "M": 5, "L": 8, "XL": 12}
CHARGER_COUNTS = {"S": 2, "M": 3, "L": 4, "XL": 6}

# ── Training parameters (from MEMORY + training scripts) ──────────────────
TRAINING_CONFIG = {
    "S": {
        "total_timesteps": "500K", "net_arch": "[256, 128]",
        "gamma": 0.995, "ent_coef": 0.05,
        "terminal_penalty": 3000, "tardiness_scale": 0.2,
        "max_time_s": 20000, "n_seeds": 3,
    },
    "M": {
        "total_timesteps": "1M", "net_arch": "[256, 128]",
        "gamma": 0.995, "ent_coef": 0.05,
        "terminal_penalty": 2500, "tardiness_scale": 0.25,
        "max_time_s": 24000, "n_seeds": 1,
    },
    "L": {
        "total_timesteps": "500K", "net_arch": "[512, 256]",
        "gamma": 0.998, "ent_coef": 0.05,
        "terminal_penalty": 2000, "tardiness_scale": 0.3,
        "max_time_s": 26000, "n_seeds": 1,
    },
    "XL": {
        "total_timesteps": "500K", "net_arch": "[512, 256]",
        "gamma": 0.998, "ent_coef": 0.08,
        "terminal_penalty": 1500, "tardiness_scale": 0.2,
        "max_time_s": 25000, "n_seeds": 1,
    },
}

RULE_NAMES = {
    1: "STTF", 2: "EDD", 3: "MST", 4: "HPF",
    5: "Charge-Urgent", 6: "Charge-Low", 7: "Charge-Med",
    8: "Charge-High", 9: "Charge-Opp",
    10: "Standby-LC", 11: "Standby-Lazy", 12: "Standby-HM",
    13: "Accept-Feas", 14: "Accept-Val", 15: "Insert-MC",
}
RULE_CATEGORIES = {
    1: "Dispatch", 2: "Dispatch", 3: "Dispatch", 4: "Dispatch", 15: "Dispatch",
    5: "Charge", 6: "Charge", 7: "Charge", 8: "Charge", 9: "Charge",
    10: "Standby", 11: "Standby", 12: "Standby",
    13: "Accept", 14: "Accept",
}

SCALES = ["S", "M", "L", "XL"]
REJECTION_PENALTY = 10000  # per rejected task

# Cost coefficients from CostParameters (src/config/defaults.py)
C_TR = 1.0          # travel distance
C_TIME = 0.1        # travel time
C_CH = 0.6          # charging time
C_DELAY = 2.0       # tardiness
C_WAIT = 0.05       # waiting time
C_CONFLICT = 0.05   # conflict waiting (includes waiting)
C_STANDBY = 0.05    # standby time
C_TERMINAL = 1000.0 # per unfinished task at episode end


def _pick_best_csv(scale: str) -> Tuple[Optional[str], str]:
    candidates = [
        (f"evaluate_{scale}_v3_30.csv", "v3"),
        (f"evaluate_{scale}_synced_30.csv", "synced"),
        (f"evaluate_{scale}_v2_30.csv", "v2"),
        (f"evaluate_{scale}_30.csv", "v1"),
    ]
    for fname, label in candidates:
        p = BENCHMARK / fname
        if p.exists():
            return str(p), label
    return None, "N/A"


def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _group(rows):
    by = {}
    for r in rows:
        aid = r["algorithm_id"]
        d = {
            "cost": float(r["cost"]),
            "completed": float(r.get("completed_tasks", 0) or 0),
            "rejected": float(r.get("rejected_tasks", 0) or 0),
            "delay": float(r.get("metrics_total_delay", 0) or 0),
            "travel": float(r.get("metrics_total_distance", 0) or 0),
            "travel_time": float(r.get("metrics_total_travel_time", 0) or 0),
            "charging": float(r.get("metrics_total_charging", 0) or 0),
            "standby": float(r.get("metrics_total_standby", 0) or 0),
            "waiting": float(r.get("metrics_total_waiting", 0) or 0),
            "conflict_waiting": float(r.get("metrics_total_conflict_waiting", 0) or 0),
            "runtime": float(r.get("runtime_s", 0) or 0),
            "num_tasks": float(r.get("num_tasks_manifest", 0) or 0),
        }
        by.setdefault(aid, []).append(d)
    return by


def _load_individual_rules(scale: str) -> Optional[List[Dict]]:
    path = BENCHMARK / f"individual_rules_{scale}_30.csv"
    if not path.exists():
        return None
    return _load(str(path))


def _bold_min(values: dict, bold_min=True) -> dict:
    if not values:
        return {}
    target = min(values.values()) if bold_min else max(values.values())
    return {k: (v == target) for k, v in values.items()}


def _write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    all_data = {}
    sources = {}
    for scale in SCALES:
        path, label = _pick_best_csv(scale)
        if path:
            all_data[scale] = _group(_load(path))
            sources[scale] = (path, label)

    # Load individual rule data
    ir_data = {}
    for scale in SCALES:
        rows = _load_individual_rules(scale)
        if rows:
            ir_data[scale] = rows

    tex_lines: List[str] = []
    tex = tex_lines.append

    tex("% EJOR Paper Tables — Auto-generated " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    tex("% Data sources: " + ", ".join(
        f"{s}={os.path.basename(p)} ({l})" for s, (p, l) in sources.items()))
    if ir_data:
        tex("% Individual rules: " + ", ".join(
            f"{s}=individual_rules_{s}_30.csv" for s in sorted(ir_data)))
    tex("")

    # ================================================================
    # Table 1: Problem Instance Description
    # ================================================================
    tex("% " + "=" * 70)
    tex("% Table 1: Problem Instances")
    tex("% " + "=" * 70)
    tex(r"\begin{table}[htbp]")
    tex(r"\centering")
    tex(r"\caption{Benchmark instance characteristics.}")
    tex(r"\label{tab:instances}")
    tex(r"\begin{tabular}{lcccc}")
    tex(r"\toprule")
    tex(r"Scale & Tasks & Vehicles & Chargers & Test Instances \\")
    tex(r"\midrule")
    for s in SCALES:
        tex(f"{s} & {TASK_COUNTS[s]} & {VEHICLE_COUNTS[s]} & {CHARGER_COUNTS[s]} & 30 \\\\")
    tex(r"\bottomrule")
    tex(r"\end{tabular}")
    tex(r"\end{table}")
    tex("")

    # ================================================================
    # Table 2: RL Training Configuration [NEW]
    # ================================================================
    tex("% " + "=" * 70)
    tex("% Table 2: RL-APC Training Configuration")
    tex("% " + "=" * 70)
    tex(r"\begin{table}[htbp]")
    tex(r"\centering")
    tex(r"\caption{RL-APC training hyperparameters per scale. All models use MaskablePPO with " +
        r"learning rate $3 \times 10^{-4}$ and 10 training instances.}")
    tex(r"\label{tab:training}")
    tex(r"\begin{tabular}{lcccccc}")
    tex(r"\toprule")
    tex(r"Scale & Steps & Network & $\gamma$ & $H_\text{ent}$ & $C_\text{terminal}$ & $\alpha_\text{tard}$ \\")
    tex(r"\midrule")

    csv_rows_train = []
    for s in SCALES:
        cfg = TRAINING_CONFIG[s]
        tex(f"{s} & {cfg['total_timesteps']} & {cfg['net_arch']} & "
            f"{cfg['gamma']} & {cfg['ent_coef']} & "
            f"{cfg['terminal_penalty']:,} & {cfg['tardiness_scale']} \\\\")
        csv_rows_train.append({
            "Scale": s, "Total Steps": cfg["total_timesteps"],
            "Net Arch": cfg["net_arch"], "Gamma": cfg["gamma"],
            "Ent Coef": cfg["ent_coef"],
            "Terminal Penalty": cfg["terminal_penalty"],
            "Tardiness Scale": cfg["tardiness_scale"],
            "Max Time (s)": cfg["max_time_s"],
        })
    tex(r"\bottomrule")
    tex(r"\end{tabular}")
    tex(r"\end{table}")
    tex("")

    _write_csv(PAPER / "ejor_table2_training.csv",
               ["Scale", "Total Steps", "Net Arch", "Gamma", "Ent Coef",
                "Terminal Penalty", "Tardiness Scale", "Max Time (s)"],
               csv_rows_train)

    # ================================================================
    # Table 3: Individual Rule Performance [NEW]
    # ================================================================
    if ir_data:
        tex("% " + "=" * 70)
        tex("% Table 3: Individual Rule Performance")
        tex("% " + "=" * 70)
        tex(r"\begin{table}[htbp]")
        tex(r"\centering")
        tex(r"\caption{Average total cost and service quality of individual dispatch rules and RL-APC "
            r"(30 test instances). Best single-rule cost per scale is underlined; "
            r"RL-APC in bold when it outperforms all single rules. "
            r"Rej = average number of rejected tasks.}")
        tex(r"\label{tab:rules}")
        tex(r"\small")
        tex(r"\begin{tabular}{ll" + "rr" * len(SCALES) + "}")
        tex(r"\toprule")
        # Multi-row header: Scale on top, Cost / Rej underneath
        tex(r" & & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{s}}}" for s in SCALES) + r" \\")
        tex(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}")
        tex(r"Rule & Category & " + " & ".join(
            r"Cost & Rej" for _ in SCALES) + r" \\")
        tex(r"\midrule")

        csv_rows_rules = []

        # Compute per-rule averages per scale (cost + service quality + components)
        rule_avgs: Dict[str, Dict[int, float]] = {}
        rule_completed: Dict[str, Dict[int, float]] = {}
        rule_rejected: Dict[str, Dict[int, float]] = {}
        rule_delay: Dict[str, Dict[int, float]] = {}
        rule_travel: Dict[str, Dict[int, float]] = {}
        rule_travel_time: Dict[str, Dict[int, float]] = {}
        rule_charging: Dict[str, Dict[int, float]] = {}
        rule_standby: Dict[str, Dict[int, float]] = {}
        rule_waiting: Dict[str, Dict[int, float]] = {}
        rule_conflict: Dict[str, Dict[int, float]] = {}
        for scale in SCALES:
            rule_avgs[scale] = {}
            rule_completed[scale] = {}
            rule_rejected[scale] = {}
            rule_delay[scale] = {}
            rule_travel[scale] = {}
            rule_travel_time[scale] = {}
            rule_charging[scale] = {}
            rule_standby[scale] = {}
            rule_waiting[scale] = {}
            rule_conflict[scale] = {}
            if scale in ir_data:
                for rid in range(1, 16):
                    ok_rows = [r for r in ir_data[scale]
                               if int(r["rule_id"]) == rid and r.get("status") == "OK"]
                    costs = [float(r["cost"]) for r in ok_rows]
                    if costs:
                        rule_avgs[scale][rid] = np.mean(costs)
                        rule_completed[scale][rid] = np.mean(
                            [float(r.get("completed_tasks", 0)) for r in ok_rows])
                        rule_rejected[scale][rid] = np.mean(
                            [float(r.get("rejected_tasks", 0)) for r in ok_rows])
                        rule_delay[scale][rid] = np.mean(
                            [float(r.get("metrics_total_delay", 0) or 0) for r in ok_rows])
                        rule_travel[scale][rid] = np.mean(
                            [float(r.get("metrics_total_distance", 0) or 0) for r in ok_rows])
                        rule_travel_time[scale][rid] = np.mean(
                            [float(r.get("metrics_total_travel_time", 0) or 0) for r in ok_rows])
                        rule_charging[scale][rid] = np.mean(
                            [float(r.get("metrics_total_charging", 0) or 0) for r in ok_rows])
                        rule_standby[scale][rid] = np.mean(
                            [float(r.get("metrics_total_standby", 0) or 0) for r in ok_rows])
                        rule_waiting[scale][rid] = np.mean(
                            [float(r.get("metrics_total_waiting", 0) or 0) for r in ok_rows])
                        rule_conflict[scale][rid] = np.mean(
                            [float(r.get("metrics_total_conflict_waiting", 0) or 0) for r in ok_rows])

        # Find best single rule per scale
        best_rule_per_scale = {}
        for scale in SCALES:
            avgs = rule_avgs.get(scale, {})
            if avgs:
                best_rule_per_scale[scale] = min(avgs, key=avgs.get)

        # RL-APC costs per scale
        rl_avgs = {}
        for scale in SCALES:
            if scale in all_data and "rl_apc" in all_data[scale]:
                rl_avgs[scale] = np.mean([d["cost"] for d in all_data[scale]["rl_apc"]])

        for rid in range(1, 16):
            name = RULE_NAMES.get(rid, f"Rule {rid}")
            cat = RULE_CATEGORIES.get(rid, "—")
            parts = [name, cat]
            csv_row = {"Rule": name, "Category": cat}

            for scale in SCALES:
                avg = rule_avgs.get(scale, {}).get(rid)
                rej = rule_rejected.get(scale, {}).get(rid)
                if avg is None:
                    parts.extend(["—", "—"])
                    csv_row[f"{scale}_Cost"] = "—"
                    csv_row[f"{scale}_Rej"] = "—"
                else:
                    is_best = best_rule_per_scale.get(scale) == rid
                    cost_str = f"{avg:,.0f}"
                    if is_best:
                        cost_str = r"\underline{" + cost_str + "}"
                    parts.append(cost_str)
                    # Highlight high rejection (>=10) in red
                    rej_val = rej if rej is not None else 0
                    if rej_val >= 10:
                        parts.append(r"\textcolor{red}{" + f"{rej_val:.1f}" + "}")
                    else:
                        parts.append(f"{rej_val:.1f}")
                    csv_row[f"{scale}_Cost"] = f"{avg:.0f}"
                    csv_row[f"{scale}_Rej"] = f"{rej_val:.1f}"

            tex(" & ".join(parts) + r" \\")
            csv_rows_rules.append(csv_row)

        # Add separator and RL-APC row
        tex(r"\midrule")
        parts_rl = ["\\textbf{RL-APC}", "\\textbf{Adaptive}"]
        csv_row_rl = {"Rule": "RL-APC", "Category": "Adaptive"}
        for scale in SCALES:
            v = rl_avgs.get(scale)
            if v is None:
                parts_rl.extend(["—", "—"])
                csv_row_rl[f"{scale}_Cost"] = "—"
                csv_row_rl[f"{scale}_Rej"] = "—"
            else:
                # Bold if better than best single rule
                best_rid = best_rule_per_scale.get(scale)
                best_val = rule_avgs.get(scale, {}).get(best_rid, float("inf")) if best_rid else float("inf")
                if v < best_val:
                    parts_rl.append(r"\textbf{" + f"{v:,.0f}" + "}")
                else:
                    parts_rl.append(f"{v:,.0f}")
                csv_row_rl[f"{scale}_Cost"] = f"{v:.0f}"
                # RL rejected tasks from eval data (key from _group: "rejected")
                rl_rej = 0.0
                if scale in all_data and "rl_apc" in all_data[scale]:
                    rl_rej = np.mean([d.get("rejected", 0)
                                      for d in all_data[scale]["rl_apc"]])
                parts_rl.append(r"\textbf{" + f"{rl_rej:.1f}" + "}")
                csv_row_rl[f"{scale}_Rej"] = f"{rl_rej:.1f}"
        tex(" & ".join(parts_rl) + r" \\")
        csv_rows_rules.append(csv_row_rl)

        tex(r"\bottomrule")
        tex(r"\end{tabular}")
        tex(r"\end{table}")
        tex("")

        t3_headers = ["Rule", "Category"]
        for s in SCALES:
            t3_headers.extend([f"{s}_Cost", f"{s}_Rej"])
        _write_csv(PAPER / "ejor_table3_rules.csv", t3_headers, csv_rows_rules)

    # ================================================================
    # Table 4: RL-APC vs Best Fixed Rule Per Scale [ENHANCED]
    # Multi-dimensional comparison: cost + service quality
    # ================================================================
    if ir_data:
        tex("% " + "=" * 70)
        tex("% Table 4: RL-APC vs Best Fixed Rule (multi-dimensional)")
        tex("% " + "=" * 70)
        tex(r"\begin{table*}[htbp]")
        tex(r"\centering")
        tex(r"\caption{Multidimensional comparison of RL-APC against the best-performing "
            r"fixed rule (by cost) for each scale. Wilcoxon signed-rank tests on 30 paired "
            r"instances. Cost/task = average cost per completed task. While the best fixed "
            r"rule achieves lower total cost in M/L/XL, it does so by rejecting a large "
            r"fraction of incoming tasks (see Rej column).}")
        tex(r"\label{tab:rl_vs_best}")
        tex(r"\small")
        tex(r"\begin{tabular}{llrrrrrrrc}")
        tex(r"\toprule")
        tex(r" & & \multicolumn{3}{c}{Best Fixed Rule} & "
            r"\multicolumn{3}{c}{RL-APC} & & \\")
        tex(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
        tex(r"Scale & Best Rule & Cost & Compl & Rej & "
            r"Cost & Compl & Rej & $\Delta_\text{cost}$ (\%) & $p$ \\")
        tex(r"\midrule")

        csv_rows_rlvsbest = []
        for scale in SCALES:
            if scale not in ir_data or scale not in all_data:
                continue
            avgs = rule_avgs.get(scale, {})
            if not avgs:
                continue
            best_rid = min(avgs, key=avgs.get)
            best_name = RULE_NAMES.get(best_rid, f"Rule {best_rid}")

            # Best rule: service quality
            best_comp = rule_completed.get(scale, {}).get(best_rid, 0)
            best_rej = rule_rejected.get(scale, {}).get(best_rid, 0)

            # RL-APC: service quality (keys from _group: "completed", "rejected")
            rl_comp, rl_rej = 0.0, 0.0
            if scale in all_data and "rl_apc" in all_data[scale]:
                rl_comp = np.mean([d.get("completed", 0)
                                   for d in all_data[scale]["rl_apc"]])
                rl_rej = np.mean([d.get("rejected", 0)
                                  for d in all_data[scale]["rl_apc"]])

            # Get paired costs for Wilcoxon test
            ir_rows = ir_data[scale]
            best_costs_sorted = sorted(
                [(r["seed"], float(r["cost"])) for r in ir_rows
                 if int(r["rule_id"]) == best_rid and r.get("status") == "OK"],
                key=lambda x: x[0]
            )
            best_arr = np.array([c for _, c in best_costs_sorted])
            rl_arr = (np.array([d["cost"] for d in all_data[scale]["rl_apc"]])
                      if "rl_apc" in all_data.get(scale, {}) else np.array([]))

            if len(best_arr) > 0 and len(rl_arr) > 0:
                n = min(len(best_arr), len(rl_arr))
                best_arr = best_arr[:n]
                rl_arr = rl_arr[:n]

                diff_pct = (rl_arr.mean() - best_arr.mean()) / best_arr.mean() * 100
                try:
                    _, p = scipy_stats.wilcoxon(rl_arr, best_arr)
                except Exception:
                    p = 1.0

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                p_str = f"${p:.2e}$" if p < 0.01 else f"${p:.3f}$"

                # Format rejection: red if >=10
                def _rej_fmt(v):
                    if v >= 10:
                        return r"\textcolor{red}{" + f"{v:.1f}" + "}"
                    return f"{v:.1f}"

                # Bold RL cost when it wins
                rl_cost_str = f"{rl_arr.mean():,.0f}"
                if diff_pct < 0:
                    rl_cost_str = r"\textbf{" + rl_cost_str + "}"
                best_cost_str = f"{best_arr.mean():,.0f}"
                if diff_pct > 0:
                    best_cost_str = r"\textbf{" + best_cost_str + "}"

                tex(f"{scale} & {best_name} & {best_cost_str} & "
                    f"{best_comp:.1f} & {_rej_fmt(best_rej)} & "
                    f"{rl_cost_str} & {rl_comp:.1f} & "
                    rf"\textbf{{{rl_rej:.1f}}} & "
                    f"${diff_pct:+.1f}${sig} & {p_str} \\\\")

                csv_rows_rlvsbest.append({
                    "Scale": scale, "Best Rule": best_name,
                    "Best Rule Cost": f"{best_arr.mean():.0f}",
                    "Best Compl": f"{best_comp:.1f}",
                    "Best Rej": f"{best_rej:.1f}",
                    "RL-APC Cost": f"{rl_arr.mean():.0f}",
                    "RL Compl": f"{rl_comp:.1f}",
                    "RL Rej": f"{rl_rej:.1f}",
                    "Diff %": f"{diff_pct:+.1f}",
                    "p-value": f"{p:.2e}", "Sig": sig or "ns",
                })

        tex(r"\bottomrule")
        tex(r"\end{tabular}")
        tex(r"\end{table*}")
        tex("")

        _write_csv(PAPER / "ejor_table4_rl_vs_best.csv",
                   ["Scale", "Best Rule", "Best Rule Cost", "Best Compl",
                    "Best Rej", "RL-APC Cost", "RL Compl", "RL Rej",
                    "Diff %", "p-value", "Sig"],
                   csv_rows_rlvsbest)

    # ================================================================
    # Table 5: Wilcoxon Tests — Enhanced with Bonferroni [ENHANCED]
    # ================================================================
    tex("% " + "=" * 70)
    tex("% Table 5: Statistical Tests — Enhanced with Bonferroni Correction")
    tex("% " + "=" * 70)
    tex(r"\begin{table}[htbp]")
    tex(r"\centering")
    tex(r"\caption{Wilcoxon signed-rank tests: RL-APC versus baselines (30 paired instances). " +
        r"$p$-values adjusted via Bonferroni correction for multiple comparisons within each scale.}")
    tex(r"\label{tab:wilcoxon}")
    tex(r"\begin{tabular}{llrrrcr}")
    tex(r"\toprule")
    tex(r"Scale & Baseline & RL Mean & BL Mean & $\Delta$ (\%) & $p_\text{adj}$ & W/L \\")
    tex(r"\midrule")

    csv_rows_wil = []
    baselines_online = ["greedy_fr", "greedy_pr", "random_rule"]
    # Add top individual rules if available
    for scale in SCALES:
        by = all_data.get(scale, {})
        if "rl_apc" not in by:
            continue
        rl = np.array([d["cost"] for d in by["rl_apc"]])
        first_in_scale = True

        # Collect all comparisons for this scale
        comparisons = []
        for bl in baselines_online:
            if bl not in by:
                continue
            bl_arr = np.array([d["cost"] for d in by[bl]])
            comparisons.append((ALGO_DISPLAY[bl], rl, bl_arr))

        # Add top-3 individual rules for this scale
        if scale in ir_data:
            avgs = rule_avgs.get(scale, {})
            if avgs:
                top3_rids = sorted(avgs, key=avgs.get)[:3]
                for rid in top3_rids:
                    name = RULE_NAMES.get(rid, f"Rule{rid}")
                    ir_costs = sorted(
                        [float(r["cost"]) for r in ir_data[scale]
                         if int(r["rule_id"]) == rid and r.get("status") == "OK"]
                    )
                    if len(ir_costs) == len(rl):
                        comparisons.append((name, rl, np.array(ir_costs)))

        n_tests = len(comparisons)

        for bl_name, rl_arr, bl_arr in comparisons:
            diff = (rl_arr.mean() - bl_arr.mean()) / bl_arr.mean() * 100
            try:
                _, p_raw = scipy_stats.wilcoxon(rl_arr, bl_arr)
            except Exception:
                p_raw = 1.0

            p_adj = min(p_raw * n_tests, 1.0)  # Bonferroni
            w = int(np.sum(rl_arr < bl_arr))
            l = len(rl_arr) - w - int(np.sum(rl_arr == bl_arr))
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""

            scale_col = scale if first_in_scale else ""
            first_in_scale = False
            p_str = f"${p_adj:.2e}$" if p_adj < 0.01 else f"${p_adj:.3f}$"
            diff_str = f"${diff:+.1f}$"

            tex(f"{scale_col} & {bl_name} & {rl_arr.mean():,.0f} & "
                f"{bl_arr.mean():,.0f} & {diff_str}{sig} & {p_str} & {w}/{l} \\\\")

            csv_rows_wil.append({
                "Scale": scale, "Baseline": bl_name,
                "RL Mean": f"{rl_arr.mean():.0f}",
                "BL Mean": f"{bl_arr.mean():.0f}",
                "Diff %": f"{diff:+.1f}",
                "p-raw": f"{p_raw:.2e}", "p-adj": f"{p_adj:.2e}",
                "Sig": sig or "ns", "W/L": f"{w}/{l}",
                "n_tests": n_tests,
            })

        if scale != SCALES[-1]:
            tex(r"\addlinespace")

    tex(r"\bottomrule")
    tex(r"\end{tabular}")
    tex(r"\end{table}")
    tex("")

    _write_csv(PAPER / "ejor_table5_wilcoxon.csv",
               ["Scale", "Baseline", "RL Mean", "BL Mean", "Diff %",
                "p-raw", "p-adj", "Sig", "W/L", "n_tests"],
               csv_rows_wil)

    # ================================================================
    # Table 6: Service Quality — Three-Way Comparison [ENHANCED]
    # RL-APC vs Greedy-FR vs Best Fixed Rule, with cost-per-completed-task
    # ================================================================
    tex("% " + "=" * 70)
    tex("% Table 6: Service Quality — Three-Way Comparison")
    tex("% " + "=" * 70)
    tex(r"\begin{table*}[htbp]")
    tex(r"\centering")
    tex(r"\caption{Service quality comparison: RL-APC, Greedy-FR, and the best fixed rule "
        r"(by cost) per scale. Cost/task = total cost $\div$ completed tasks, measuring "
        r"operational efficiency independent of workload volume. "
        r"Best values per scale in \textbf{bold}; rejected $\geq 10$ in "
        r"\textcolor{red}{red}. RL-APC is the only method achieving near-zero "
        r"rejection across all scales.}")
    tex(r"\label{tab:service}")
    tex(r"\small")
    tex(r"\begin{tabular}{llrrrrrr}")
    tex(r"\toprule")
    tex(r"Scale & Method & Cost & Completed & Rejected & Delay & Cost/Task \\")
    tex(r"\midrule")

    csv_rows_sq = []
    for scale in SCALES:
        by = all_data.get(scale, {})
        if "rl_apc" not in by:
            continue

        # Collect stats for 3 methods: RL, Greedy, Best Rule
        methods = []  # list of (name, cost, compl, rej, delay)

        # RL-APC
        rl = by["rl_apc"]
        rl_cost = np.mean([d["cost"] for d in rl])
        rl_comp = np.mean([d["completed"] for d in rl])
        rl_rej = np.mean([d["rejected"] for d in rl])
        rl_del = np.mean([d["delay"] for d in rl])
        methods.append(("RL-APC", rl_cost, rl_comp, rl_rej, rl_del))

        # Greedy-FR
        if "greedy_fr" in by:
            gr = by["greedy_fr"]
            gr_cost = np.mean([d["cost"] for d in gr])
            gr_comp = np.mean([d["completed"] for d in gr])
            gr_rej = np.mean([d["rejected"] for d in gr])
            gr_del = np.mean([d["delay"] for d in gr])
            methods.append(("Greedy-FR", gr_cost, gr_comp, gr_rej, gr_del))

        # Best Fixed Rule (from ir_data) — skip if same as Greedy-FR
        avgs = rule_avgs.get(scale, {})
        if avgs:
            best_rid = min(avgs, key=avgs.get)
            best_name = RULE_NAMES.get(best_rid, f"Rule {best_rid}")
            br_cost = rule_avgs[scale][best_rid]
            br_comp = rule_completed.get(scale, {}).get(best_rid, 0)
            br_rej = rule_rejected.get(scale, {}).get(best_rid, 0)
            br_del = rule_delay.get(scale, {}).get(best_rid, 0)
            # Greedy-FR uses Charge-Opp (rid=9); Charge-High(8)=Charge-Opp(9)
            # If best rule is effectively the same as Greedy-FR, annotate instead
            gr_cost_val = gr_cost if "greedy_fr" in by else None
            if gr_cost_val is not None and abs(br_cost - gr_cost_val) < 1:
                # Rename Greedy-FR row to include best rule identity
                methods = [(f"Greedy-FR (={best_name})" if n == "Greedy-FR" else n,
                            c, co, r, d) for n, c, co, r, d in methods]
            else:
                methods.append((best_name, br_cost, br_comp, br_rej, br_del))

        # Find best values (min cost, max compl, min rej, min delay, min cost/task)
        costs = [m[1] for m in methods]
        compls = [m[2] for m in methods]
        rejs = [m[3] for m in methods]
        delays = [m[4] for m in methods]
        cpts = [(m[1] / m[2]) if m[2] > 0 else float('inf') for m in methods]
        best_cost = min(costs)
        best_compl = max(compls)
        best_rej = min(rejs)
        best_delay = min(delays)
        best_cpt = min(cpts)

        first = True
        for i, (mname, mcost, mcomp, mrej, mdel) in enumerate(methods):
            scale_col = scale if first else ""
            first = False

            # Cost/task
            cpt = mcost / mcomp if mcomp > 0 else float('inf')

            # Format with bold for best, red for high rejection
            def _fmt_cost(v, is_best):
                s = f"{v:,.0f}"
                return r"\textbf{" + s + "}" if is_best else s

            def _fmt_compl(v, is_best):
                s = f"{v:.1f}"
                return r"\textbf{" + s + "}" if is_best else s

            def _fmt_rej(v, is_best):
                s = f"{v:.1f}"
                if v >= 10:
                    s = r"\textcolor{red}{" + s + "}"
                if is_best:
                    s = r"\textbf{" + f"{v:.1f}" + "}"
                return s

            def _fmt_delay(v, is_best):
                s = f"{v:,.0f}"
                return r"\textbf{" + s + "}" if is_best else s

            def _fmt_cpt(v, is_best):
                s = f"{v:,.0f}"
                return r"\textbf{" + s + "}" if is_best else s

            tex(f"{scale_col} & {mname} & "
                f"{_fmt_cost(mcost, mcost == best_cost)} & "
                f"{_fmt_compl(mcomp, mcomp == best_compl)} & "
                f"{_fmt_rej(mrej, mrej == best_rej)} & "
                f"{_fmt_delay(mdel, mdel == best_delay)} & "
                f"{_fmt_cpt(cpt, cpt == best_cpt)} \\\\")

            csv_rows_sq.append({
                "Scale": scale, "Method": mname,
                "Cost": f"{mcost:.0f}", "Completed": f"{mcomp:.1f}",
                "Rejected": f"{mrej:.1f}", "Delay": f"{mdel:.1f}",
                "Cost/Task": f"{cpt:.0f}",
            })

        if scale != SCALES[-1]:
            tex(r"\addlinespace")

    tex(r"\bottomrule")
    tex(r"\end{tabular}")
    tex(r"\end{table*}")
    tex("")

    _write_csv(PAPER / "ejor_table6_service.csv",
               ["Scale", "Method", "Cost", "Completed", "Rejected",
                "Delay", "Cost/Task"],
               csv_rows_sq)

    # ================================================================
    # Table 7: Online-Offline Gap [ENHANCED]
    # ================================================================
    tex("% " + "=" * 70)
    tex("% Table 7: Online vs Offline — Gap Analysis")
    tex("% " + "=" * 70)
    tex(r"\begin{table}[htbp]")
    tex(r"\centering")
    tex(r"\caption{Online--offline performance gap. The gap reflects the inherent cost of " +
        r"real-time decision-making without future task knowledge, not algorithm deficiency.}")
    tex(r"\label{tab:offline}")
    tex(r"\begin{tabular}{lrrrrr}")
    tex(r"\toprule")
    tex(r"Scale & RL-APC & Greedy-FR & Best Offline & Gap$_\text{RL}$ (\%) & Gap$_\text{Greedy}$ (\%) \\")
    tex(r"\midrule")

    csv_rows_off = []
    for scale in SCALES:
        by = all_data.get(scale, {})
        rl_cost = np.mean([d["cost"] for d in by["rl_apc"]]) if "rl_apc" in by else None
        gr_cost = np.mean([d["cost"] for d in by["greedy_fr"]]) if "greedy_fr" in by else None

        # Best offline = min(ALNS-FR, ALNS-PR)
        offline_costs = {}
        for a in ["alns_fr", "alns_pr"]:
            if a in by:
                offline_costs[a] = np.mean([d["cost"] for d in by[a]])
        best_offline = min(offline_costs.values()) if offline_costs else None

        parts = [scale]
        parts.append(f"{rl_cost:,.0f}" if rl_cost is not None else "—")
        parts.append(f"{gr_cost:,.0f}" if gr_cost is not None else "—")
        parts.append(f"{best_offline:,.0f}" if best_offline is not None else "—")

        gap_rl = ((rl_cost - best_offline) / best_offline * 100) if (rl_cost and best_offline) else None
        gap_gr = ((gr_cost - best_offline) / best_offline * 100) if (gr_cost and best_offline) else None
        parts.append(f"${gap_rl:+.1f}$" if gap_rl is not None else "—")
        parts.append(f"${gap_gr:+.1f}$" if gap_gr is not None else "—")

        tex(" & ".join(parts) + r" \\")
        csv_rows_off.append({
            "Scale": scale,
            "RL-APC": f"{rl_cost:.0f}" if rl_cost else "—",
            "Greedy-FR": f"{gr_cost:.0f}" if gr_cost else "—",
            "Best Offline": f"{best_offline:.0f}" if best_offline else "—",
            "Gap RL (%)": f"{gap_rl:+.1f}" if gap_rl is not None else "—",
            "Gap Greedy (%)": f"{gap_gr:+.1f}" if gap_gr is not None else "—",
        })

    tex(r"\bottomrule")
    tex(r"\end{tabular}")
    tex(r"\end{table}")
    tex("")

    _write_csv(PAPER / "ejor_table7_offline_gap.csv",
               ["Scale", "RL-APC", "Greedy-FR", "Best Offline",
                "Gap RL (%)", "Gap Greedy (%)"],
               csv_rows_off)

    # ================================================================
    # Table 8: Cost Decomposition — Three-Way with Weighted Costs [ENHANCED]
    # Shows weighted cost contributions (coefficient × raw metric),
    # not raw physical quantities.
    # ================================================================
    tex("% " + "=" * 70)
    tex("% Table 8: Cost Decomposition — Weighted Cost Contributions")
    tex("% " + "=" * 70)
    tex(r"\begin{table*}[htbp]")
    tex(r"\centering")
    tex(r"\caption{Cost decomposition into weighted components (averages over 30 instances). "
        r"Travel = $C_\text{tr} \times d + C_\text{time} \times t$; "
        r"Charging = $C_\text{ch} \times t_\text{ch}$; "
        r"Tardiness = $C_\text{delay} \times \tau$; "
        r"Idle = $C_\text{wait} \times t_w + C_\text{cf} \times (t_w + t_\text{cf}) "
        r"+ C_\text{sb} \times t_\text{sb}$; "
        r"Rejection = $10{,}000 \times n_\text{rej}$; "
        r"Other = terminal penalties and reward shaping. "
        r"Rejection cost (in \textcolor{red}{red} when $\geq 30\%$ of total) "
        r"dominates baselines; RL-APC shifts cost to idle (conservative waiting).}")
    tex(r"\label{tab:cost_decomp}")
    tex(r"\small")
    tex(r"\begin{tabular}{llrrrrrrrr}")
    tex(r"\toprule")
    tex(r"Scale & Method & Travel & Charg. & Tard. & Idle & Reject. & Other & Total & \% Rej. \\")
    tex(r"\midrule")

    def _weighted_costs(travel, travel_time, charging, delay, waiting,
                        conflict, standby, rejected, total_cost):
        """Compute weighted cost contributions and residual."""
        w_travel = C_TR * travel + C_TIME * travel_time
        w_charging = C_CH * charging
        w_tardiness = C_DELAY * delay
        w_idle = C_WAIT * waiting + C_CONFLICT * (conflict + waiting) + C_STANDBY * standby
        w_rejection = REJECTION_PENALTY * rejected
        w_known = w_travel + w_charging + w_tardiness + w_idle + w_rejection
        w_other = total_cost - w_known  # terminal penalties + reward shaping
        rej_pct = (w_rejection / total_cost * 100) if total_cost > 0 else 0
        return w_travel, w_charging, w_tardiness, w_idle, w_rejection, w_other, rej_pct

    csv_rows_cd = []
    for scale in SCALES:
        by = all_data.get(scale, {})
        first = True

        # Build method list with all raw metrics
        # tuple: (name, travel, travel_time, charging, delay, waiting, conflict, standby, rejected, cost)
        method_rows = []
        for a in ["rl_apc", "greedy_fr"]:
            if a not in by:
                continue
            vals = by[a]
            method_rows.append((
                ALGO_DISPLAY[a],
                np.mean([d["travel"] for d in vals]),
                np.mean([d.get("travel_time", 0) for d in vals]),
                np.mean([d["charging"] for d in vals]),
                np.mean([d["delay"] for d in vals]),
                np.mean([d["waiting"] for d in vals]),
                np.mean([d["conflict_waiting"] for d in vals]),
                np.mean([d["standby"] for d in vals]),
                np.mean([d["rejected"] for d in vals]),
                np.mean([d["cost"] for d in vals]),
            ))

        # Best Fixed Rule from ir_data — skip if same as Greedy-FR
        avgs = rule_avgs.get(scale, {})
        if avgs:
            best_rid = min(avgs, key=avgs.get)
            best_name = RULE_NAMES.get(best_rid, f"Rule {best_rid}")
            br_cost = rule_avgs[scale][best_rid]
            gr_cost_val = None
            for row in method_rows:
                if row[0] == "Greedy-FR":
                    gr_cost_val = row[-1]
            if gr_cost_val is not None and abs(br_cost - gr_cost_val) < 1:
                method_rows = [(f"Greedy-FR (={best_name})" if n == "Greedy-FR" else n,
                                *rest) for n, *rest in method_rows]
            else:
                method_rows.append((
                    best_name,
                    rule_travel.get(scale, {}).get(best_rid, 0),
                    rule_travel_time.get(scale, {}).get(best_rid, 0),
                    rule_charging.get(scale, {}).get(best_rid, 0),
                    rule_delay.get(scale, {}).get(best_rid, 0),
                    rule_waiting.get(scale, {}).get(best_rid, 0),
                    rule_conflict.get(scale, {}).get(best_rid, 0),
                    rule_standby.get(scale, {}).get(best_rid, 0),
                    rule_rejected.get(scale, {}).get(best_rid, 0),
                    br_cost,
                ))

        for (mname, travel, travel_time, charging, delay, waiting,
             conflict, standby, rejected, cost) in method_rows:
            scale_col = scale if first else ""
            first = False

            w_tr, w_ch, w_td, w_idle, w_rej, w_other, rej_pct = _weighted_costs(
                travel, travel_time, charging, delay, waiting,
                conflict, standby, rejected, cost)

            # Highlight rejection when ≥ 30% of total
            rej_str = f"{w_rej:,.0f}"
            if rej_pct >= 30:
                rej_str = r"\textcolor{red}{" + rej_str + "}"

            # Other can be negative if reward shaping credits exist
            other_str = f"{w_other:,.0f}"

            tex(f"{scale_col} & {mname} & {w_tr:,.0f} & {w_ch:,.0f} & "
                f"{w_td:,.0f} & {w_idle:,.0f} & {rej_str} & "
                f"{other_str} & {cost:,.0f} & {rej_pct:.0f}\\% \\\\")

            csv_rows_cd.append({
                "Scale": scale, "Method": mname,
                "Travel": f"{w_tr:.0f}", "Charging": f"{w_ch:.0f}",
                "Tardiness": f"{w_td:.0f}", "Idle": f"{w_idle:.0f}",
                "Rejection": f"{w_rej:.0f}",
                "Other": f"{w_other:.0f}",
                "Total": f"{cost:.0f}",
                "Rej %": f"{rej_pct:.0f}",
            })
        if scale != SCALES[-1]:
            tex(r"\addlinespace")

    tex(r"\bottomrule")
    tex(r"\end{tabular}")
    tex(r"\end{table*}")
    tex("")

    _write_csv(PAPER / "ejor_table8_decomposition.csv",
               ["Scale", "Method", "Travel", "Charging", "Tardiness",
                "Idle", "Rejection", "Other", "Total", "Rej %"],
               csv_rows_cd)

    # ================================================================
    # Table 9: Computational Efficiency
    # ================================================================
    tex("% " + "=" * 70)
    tex("% Table 9: Computational Efficiency")
    tex("% " + "=" * 70)
    tex(r"\begin{table}[htbp]")
    tex(r"\centering")
    tex(r"\caption{Average runtime per instance (seconds).}")
    tex(r"\label{tab:runtime}")
    tex(r"\begin{tabular}{lrrrrrr}")
    tex(r"\toprule")
    tex(r"Scale & RL-APC & Greedy-FR & Greedy-PR & Random & ALNS-FR & ALNS-PR \\")
    tex(r"\midrule")

    all_algos = ["rl_apc", "greedy_fr", "greedy_pr", "random_rule", "alns_fr", "alns_pr"]
    for scale in SCALES:
        by = all_data.get(scale, {})
        parts = [scale]
        for a in all_algos:
            if a in by:
                rt = np.mean([d["runtime"] for d in by[a]])
                parts.append(f"{rt:.2f}")
            else:
                parts.append("—")
        tex(" & ".join(parts) + r" \\")

    tex(r"\bottomrule")
    tex(r"\end{tabular}")
    tex(r"\end{table}")
    tex("")

    # ================================================================
    # Table 10: L-Scale Sensitivity Analysis [NICE-TO-HAVE]
    # ================================================================
    # Check if multiple L training versions exist
    l_versions = []
    for tag, label in [("evaluate_L_30.csv", "v1"), ("evaluate_L_v2_30.csv", "v2"),
                        ("evaluate_L_v3_30.csv", "v3")]:
        p = BENCHMARK / tag
        if p.exists():
            l_versions.append((str(p), label))

    if len(l_versions) > 1:
        tex("% " + "=" * 70)
        tex("% Table 10: L-Scale Sensitivity Analysis")
        tex("% " + "=" * 70)
        tex(r"\begin{table}[htbp]")
        tex(r"\centering")
        tex(r"\caption{L-scale training sensitivity. Different hyperparameter configurations " +
            r"show progressive improvement in RL-APC performance.}")
        tex(r"\label{tab:sensitivity}")
        tex(r"\begin{tabular}{lrrrrr}")
        tex(r"\toprule")
        tex(r"Version & Net Arch & $C_\text{terminal}$ & RL Cost & Greedy Cost & $\Delta$ (\%) \\")
        tex(r"\midrule")

        l_configs = {
            "v1": {"net_arch": "[256, 128]", "terminal": "3000"},
            "v2": {"net_arch": "[512, 256]", "terminal": "2000"},
            "v3": {"net_arch": "[512, 256]", "terminal": "2000"},
        }

        csv_rows_sens = []
        for path, label in l_versions:
            rows = _load(path)
            by = _group(rows)
            rl_cost = np.mean([d["cost"] for d in by["rl_apc"]]) if "rl_apc" in by else None
            gr_cost = np.mean([d["cost"] for d in by["greedy_fr"]]) if "greedy_fr" in by else None

            cfg = l_configs.get(label, {"net_arch": "—", "terminal": "—"})
            diff = ((rl_cost - gr_cost) / gr_cost * 100) if (rl_cost and gr_cost) else None
            diff_str = f"${diff:+.1f}$" if diff is not None else "—"

            tex(f"{label} & {cfg['net_arch']} & {cfg['terminal']} & "
                f"{rl_cost:,.0f} & {gr_cost:,.0f} & {diff_str} \\\\")

            csv_rows_sens.append({
                "Version": label, "Net Arch": cfg["net_arch"],
                "Terminal Penalty": cfg["terminal"],
                "RL Cost": f"{rl_cost:.0f}" if rl_cost else "—",
                "Greedy Cost": f"{gr_cost:.0f}" if gr_cost else "—",
                "Diff %": f"{diff:+.1f}" if diff is not None else "—",
            })

        tex(r"\bottomrule")
        tex(r"\end{tabular}")
        tex(r"\end{table}")
        tex("")

        _write_csv(PAPER / "ejor_table10_sensitivity.csv",
                   ["Version", "Net Arch", "Terminal Penalty",
                    "RL Cost", "Greedy Cost", "Diff %"],
                   csv_rows_sens)

    # Write main tex file
    tex_path = PAPER / "ejor_tables.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"LaTeX tables -> {tex_path}")
    print(f"CSVs -> {PAPER}/ejor_table*.csv")

    # Print summary
    print(f"\nGenerated tables:")
    print(f"  T1: Problem instances")
    print(f"  T2: Training configuration")
    if ir_data:
        print(f"  T3: Individual rule performance ({len(ir_data)} scales)")
        print(f"  T4: RL-APC vs best fixed rule")
    print(f"  T5: Wilcoxon tests (Bonferroni-corrected)")
    print(f"  T6: Service quality (with completion rate)")
    print(f"  T7: Online-offline gap")
    print(f"  T8: Cost decomposition (with rejection penalty)")
    print(f"  T9: Computational efficiency")
    if len(l_versions) > 1:
        print(f"  T10: L-scale sensitivity ({len(l_versions)} versions)")


if __name__ == "__main__":
    main()
