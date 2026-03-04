"""Analyze RL-APC rule selection behavior from decision logs.

Parses decision_log CSV files generated during RL-APC evaluation to produce:
1. Per-scale rule selection frequency tables
2. Per-event-type rule selection breakdown
3. Heatmap data (15 rules × 4 scales)
4. Temporal rule selection patterns within episodes

Usage:
    python scripts/analyze_rule_selection.py --scale S
    python scripts/analyze_rule_selection.py               # all available

To generate decision logs, run evaluate_all.py with logging enabled
(see --enable-decision-log), or use the companion runner script.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from strategy.rule_gating import (
    RULE_STTF,
    RULE_EDD,
    RULE_MST,
    RULE_HPF,
    RULE_CHARGE_URGENT,
    RULE_CHARGE_TARGET_LOW,
    RULE_CHARGE_TARGET_MED,
    RULE_CHARGE_TARGET_HIGH,
    RULE_CHARGE_OPPORTUNITY,
    RULE_STANDBY_LOW_COST,
    RULE_STANDBY_LAZY,
    RULE_STANDBY_HEATMAP,
    RULE_ACCEPT_FEASIBLE,
    RULE_ACCEPT_VALUE,
    RULE_INSERT_MIN_COST,
)

RULE_NAMES = {
    RULE_STTF: "STTF",
    RULE_EDD: "EDD",
    RULE_MST: "MST",
    RULE_HPF: "HPF",
    RULE_CHARGE_URGENT: "Charge-Urgent",
    RULE_CHARGE_TARGET_LOW: "Charge-Low",
    RULE_CHARGE_TARGET_MED: "Charge-Med",
    RULE_CHARGE_TARGET_HIGH: "Charge-High",
    RULE_CHARGE_OPPORTUNITY: "Charge-Opp",
    RULE_STANDBY_LOW_COST: "Standby-LowCost",
    RULE_STANDBY_LAZY: "Standby-Lazy",
    RULE_STANDBY_HEATMAP: "Standby-Heatmap",
    RULE_ACCEPT_FEASIBLE: "Accept-Feasible",
    RULE_ACCEPT_VALUE: "Accept-Value",
    RULE_INSERT_MIN_COST: "Insert-MinCost",
}

RULE_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

RULE_CATEGORIES = {
    "Dispatch": [1, 2, 3, 4, 15],
    "Charge": [5, 6, 7, 8, 9],
    "Standby": [10, 11, 12],
    "Accept": [13, 14],
}


def _load_decision_log(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_decision_logs(eval_dir: str, scale: str) -> List[str]:
    """Find decision log CSVs for a given scale."""
    pattern = os.path.join(eval_dir, f"decision_log_*.csv")
    logs = sorted(glob.glob(pattern))
    # Filter out anomaly reports
    logs = [l for l in logs if "anomaly" not in l]
    return logs


def analyze_scale(decision_log_paths: List[str], scale: str) -> Dict[str, Any]:
    """Aggregate rule selection statistics across all episodes for one scale."""
    all_rows = []
    for path in decision_log_paths:
        rows = _load_decision_log(path)
        all_rows.extend(rows)

    if not all_rows:
        return {"scale": scale, "total_decisions": 0}

    # Count rule selections overall
    rule_counter = Counter()
    event_rule_counter = defaultdict(Counter)  # event_type → {rule_id: count}
    total_decisions = 0
    masked_count = 0

    for row in all_rows:
        selected = row.get("selected_rule", "").strip()
        if not selected:
            continue
        try:
            rule_id = int(selected)
        except ValueError:
            continue

        event_type = row.get("event_type", "UNKNOWN").strip()
        was_masked = row.get("masked", "").strip().lower() == "true"

        rule_counter[rule_id] += 1
        event_rule_counter[event_type][rule_id] += 1
        total_decisions += 1
        if was_masked:
            masked_count += 1

    # Build frequency table
    freq_table = {}
    for rid in RULE_ORDER:
        count = rule_counter.get(rid, 0)
        freq_table[rid] = {
            "rule_name": RULE_NAMES.get(rid, f"Rule{rid}"),
            "count": count,
            "frequency": count / total_decisions if total_decisions > 0 else 0.0,
        }

    # Build event-type breakdown
    event_breakdown = {}
    for event_type, counter in sorted(event_rule_counter.items()):
        event_total = sum(counter.values())
        event_breakdown[event_type] = {
            "total": event_total,
            "rules": {
                rid: {
                    "count": counter.get(rid, 0),
                    "frequency": counter.get(rid, 0) / event_total if event_total > 0 else 0.0,
                }
                for rid in RULE_ORDER
            },
        }

    return {
        "scale": scale,
        "total_decisions": total_decisions,
        "masked_decisions": masked_count,
        "masked_rate": masked_count / total_decisions if total_decisions > 0 else 0.0,
        "rule_frequencies": freq_table,
        "event_breakdown": event_breakdown,
    }


def write_frequency_csv(analysis: Dict, output_path: Path):
    """Write rule selection frequency table."""
    freq = analysis.get("rule_frequencies", {})
    total = analysis.get("total_decisions", 0)

    rows = []
    for rid in RULE_ORDER:
        info = freq.get(rid, {})
        rows.append({
            "rule_id": rid,
            "rule_name": info.get("rule_name", f"Rule{rid}"),
            "count": info.get("count", 0),
            "frequency": f"{info.get('frequency', 0):.4f}",
            "percentage": f"{info.get('frequency', 0) * 100:.1f}%",
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rule_id", "rule_name", "count", "frequency", "percentage"])
        writer.writeheader()
        writer.writerows(rows)


def write_event_breakdown_csv(analysis: Dict, output_path: Path):
    """Write per-event-type rule selection breakdown."""
    event_data = analysis.get("event_breakdown", {})
    rows = []
    for event_type, data in sorted(event_data.items()):
        for rid in RULE_ORDER:
            rule_info = data["rules"].get(rid, {"count": 0, "frequency": 0.0})
            if rule_info["count"] > 0:
                rows.append({
                    "event_type": event_type,
                    "rule_id": rid,
                    "rule_name": RULE_NAMES.get(rid, f"Rule{rid}"),
                    "count": rule_info["count"],
                    "frequency": f"{rule_info['frequency']:.4f}",
                })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_type", "rule_id", "rule_name", "count", "frequency"])
        writer.writeheader()
        writer.writerows(rows)


def write_heatmap_csv(all_analyses: Dict[str, Dict], output_path: Path):
    """Write 15×4 heatmap data (rule × scale frequency matrix)."""
    scales = sorted(all_analyses.keys())
    rows = []
    for rid in RULE_ORDER:
        row = {
            "rule_id": rid,
            "rule_name": RULE_NAMES.get(rid, f"Rule{rid}"),
        }
        for scale in scales:
            freq = all_analyses[scale].get("rule_frequencies", {}).get(rid, {})
            row[f"freq_{scale}"] = f"{freq.get('frequency', 0):.4f}"
        rows.append(row)

    headers = ["rule_id", "rule_name"] + [f"freq_{s}" for s in scales]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def generate_heatmap_figure(all_analyses: Dict[str, Dict], output_path: Path):
    """Generate rule selection frequency heatmap (15 rules × N scales)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("  [skip] matplotlib not installed")
        return

    scales = sorted(all_analyses.keys())
    if not scales:
        return

    # Build frequency matrix
    matrix = np.zeros((len(RULE_ORDER), len(scales)))
    for j, scale in enumerate(scales):
        freq = all_analyses[scale].get("rule_frequencies", {})
        for i, rid in enumerate(RULE_ORDER):
            matrix[i, j] = freq.get(rid, {}).get("frequency", 0.0)

    rule_labels = [RULE_NAMES.get(rid, f"Rule{rid}") for rid in RULE_ORDER]

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(scales, fontsize=11)
    ax.set_yticks(range(len(RULE_ORDER)))
    ax.set_yticklabels(rule_labels, fontsize=9)

    # Annotate cells
    for i in range(len(RULE_ORDER)):
        for j in range(len(scales)):
            val = matrix[i, j]
            if val > 0.005:
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Selection Frequency")
    ax.set_xlabel("Scale", fontsize=12)
    ax.set_title("RL-APC Rule Selection Frequency", fontsize=13)

    # Add category separators
    for y in [4.5, 9.5, 12.5]:  # after Dispatch(5), Charge(5), Standby(3)
        ax.axhline(y=y, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RL-APC rule selection behavior.")
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        default=None,
        help="Directories containing decision_log CSVs. Auto-detected if not set.",
    )
    parser.add_argument("--scale", default=None, help="Filter to specific scale.")
    parser.add_argument("--output-dir", default="results/paper")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect eval dirs
    if args.eval_dirs:
        eval_dirs_by_scale = {}
        for d in args.eval_dirs:
            # Try to infer scale from directory name
            for s in ["S", "M", "L", "XL"]:
                if f"train_{s}" in d:
                    eval_dirs_by_scale[s] = d
                    break
    else:
        eval_dirs_by_scale = {}
        results_root = PROJECT_ROOT / "results" / "rl"
        for s in ["S", "M", "L", "XL"]:
            # Try v3 > v2 > v1 eval directory
            candidates = [
                results_root / f"train_{s}_v3" / "eval",
                results_root / f"train_{s}_v2" / "eval",
                results_root / f"train_{s}" / "eval",
            ]
            for c in candidates:
                if c.exists():
                    logs = _find_decision_logs(str(c), s)
                    if logs:
                        eval_dirs_by_scale[s] = str(c)
                        break

    if args.scale:
        eval_dirs_by_scale = {k: v for k, v in eval_dirs_by_scale.items() if k == args.scale}

    if not eval_dirs_by_scale:
        print("No decision logs found. Run RL-APC evaluation with logging enabled first.")
        print("Hint: ensure eval directories have decision_log_*.csv files")
        return 1

    all_analyses: Dict[str, Dict] = {}

    for scale, eval_dir in sorted(eval_dirs_by_scale.items()):
        logs = _find_decision_logs(eval_dir, scale)
        if not logs:
            print(f"Scale {scale}: no decision logs in {eval_dir}")
            continue

        print(f"\nScale {scale}: analyzing {len(logs)} decision log(s) from {eval_dir}")
        analysis = analyze_scale(logs, scale)
        all_analyses[scale] = analysis

        print(f"  Total decisions: {analysis['total_decisions']:,}")
        print(f"  Masked rate: {analysis.get('masked_rate', 0):.1%}")

        # Print top-5 rules
        freq = analysis.get("rule_frequencies", {})
        sorted_rules = sorted(freq.items(), key=lambda x: x[1].get("count", 0), reverse=True)
        print("  Top rules:")
        for rid, info in sorted_rules[:5]:
            print(f"    {info['rule_name']:20s}: {info['frequency']:.1%} ({info['count']:,})")

        # Write per-scale CSVs
        write_frequency_csv(analysis, output_dir / f"rule_freq_{scale}.csv")
        write_event_breakdown_csv(analysis, output_dir / f"rule_event_breakdown_{scale}.csv")

    # Write cross-scale heatmap data
    if all_analyses:
        write_heatmap_csv(all_analyses, output_dir / "rule_selection_heatmap.csv")
        generate_heatmap_figure(all_analyses, output_dir / "fig_rule_selection_heatmap.png")
        print(f"\nHeatmap CSV: {output_dir / 'rule_selection_heatmap.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
