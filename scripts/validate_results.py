"""Validation checklist for paper experiment results.

Checks:
  1. All best_model.zip files exist for S/M/L/XL
  2. evaluate_*.csv files exist for all scales
  3. RL-APC beats greedy baselines on S and M
  4. Training curves show convergence trend
  5. Multi-seed variance is acceptable (if available)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALES = ["S", "M", "L", "XL"]


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def check_best_models(results_root: Path) -> int:
    print("\n1. Best model files")
    failures = 0
    for scale in SCALES:
        model_path = results_root / f"train_{scale}" / "best_model" / "best_model.zip"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            _pass(f"{scale}: best_model.zip ({size_mb:.2f} MB)")
        else:
            _fail(f"{scale}: best_model.zip not found at {model_path}")
            failures += 1
    return failures


def check_evaluations_npz(results_root: Path) -> int:
    print("\n2. Training evaluation logs")
    failures = 0
    for scale in SCALES:
        npz_path = results_root / f"train_{scale}" / "eval_logs" / "evaluations.npz"
        if not npz_path.exists():
            _fail(f"{scale}: evaluations.npz not found")
            failures += 1
            continue
        data = np.load(str(npz_path))
        timesteps = data["timesteps"]
        results = data["results"]
        mean_rew = results.mean(axis=1)
        best_idx = int(mean_rew.argmax())
        _pass(
            f"{scale}: {len(timesteps)} evals, "
            f"best={mean_rew[best_idx]:.0f} @ {int(timesteps[best_idx])}steps, "
            f"final={mean_rew[-1]:.0f} @ {int(timesteps[-1])}steps"
        )
    return failures


def check_benchmark_csvs(benchmark_root: Path) -> int:
    print("\n3. Benchmark evaluation CSVs")
    failures = 0
    for scale in SCALES:
        csv_path = benchmark_root / f"evaluate_{scale}.csv"
        if not csv_path.exists():
            _fail(f"{scale}: evaluate_{scale}.csv not found")
            failures += 1
            continue
        rows = _load_csv(csv_path)
        algos = set(row.get("algorithm_id", "") for row in rows)
        ok_count = sum(1 for row in rows if str(row.get("status", "")).upper() in ("OK", "OPTIMAL", "FEASIBLE"))
        _pass(f"{scale}: {len(rows)} rows, {ok_count} OK, algorithms: {sorted(algos)}")
    return failures


def check_rl_beats_greedy(benchmark_root: Path) -> int:
    print("\n4. RL-APC vs Greedy baselines")
    failures = 0
    for scale in ["S", "M"]:
        csv_path = benchmark_root / f"evaluate_{scale}.csv"
        if not csv_path.exists():
            _warn(f"{scale}: CSV not found, cannot check")
            continue
        rows = _load_csv(csv_path)

        def _avg_cost(algo_id: str) -> float | None:
            vals = []
            for row in rows:
                if row.get("algorithm_id") == algo_id:
                    status = str(row.get("status", "")).upper()
                    if status in ("OK", "OPTIMAL", "FEASIBLE"):
                        try:
                            vals.append(float(row["cost"]))
                        except (KeyError, TypeError, ValueError):
                            pass
            return sum(vals) / len(vals) if vals else None

        rl_cost = _avg_cost("rl_apc")
        gfr_cost = _avg_cost("greedy_fr")
        gpr_cost = _avg_cost("greedy_pr")

        if rl_cost is None:
            _fail(f"{scale}: RL-APC cost not available")
            failures += 1
            continue

        for name, greedy_cost in [("Greedy-FR", gfr_cost), ("Greedy-PR", gpr_cost)]:
            if greedy_cost is None:
                _warn(f"{scale}: {name} cost not available")
                continue
            if rl_cost < greedy_cost:
                improvement = (greedy_cost - rl_cost) / greedy_cost * 100
                _pass(f"{scale}: RL-APC ({rl_cost:.0f}) < {name} ({greedy_cost:.0f}), improvement: {improvement:.1f}%")
            else:
                gap = (rl_cost - greedy_cost) / greedy_cost * 100
                _fail(f"{scale}: RL-APC ({rl_cost:.0f}) >= {name} ({greedy_cost:.0f}), gap: +{gap:.1f}%")
                failures += 1
    return failures


def check_convergence(results_root: Path) -> int:
    print("\n5. Training convergence check")
    failures = 0
    for scale in SCALES:
        npz_path = results_root / f"train_{scale}" / "eval_logs" / "evaluations.npz"
        if not npz_path.exists():
            continue
        data = np.load(str(npz_path))
        results = data["results"]
        mean_rew = results.mean(axis=1)
        if len(mean_rew) < 3:
            _warn(f"{scale}: too few eval points ({len(mean_rew)})")
            continue

        # Check: best reward in first half vs second half
        mid = len(mean_rew) // 2
        best_first_half = mean_rew[:mid].max()
        best_second_half = mean_rew[mid:].max()
        best_overall = mean_rew.max()
        best_idx = int(mean_rew.argmax())

        if best_second_half >= best_first_half * 0.95:  # within 5%
            _pass(f"{scale}: convergence OK (best={best_overall:.0f} @ eval#{best_idx})")
        else:
            _warn(
                f"{scale}: best in 1st half ({best_first_half:.0f}) > 2nd half ({best_second_half:.0f}); "
                f"may indicate instability"
            )
    return failures


def check_multiseed(results_root: Path) -> int:
    print("\n6. Multi-seed stability (S scale)")
    seed_dirs = {
        42: results_root / "train_S",
        43: results_root / "train_S_seed43",
        44: results_root / "train_S_seed44",
    }
    best_rewards = {}
    for seed, path in seed_dirs.items():
        npz_path = path / "eval_logs" / "evaluations.npz"
        if not npz_path.exists():
            _warn(f"Seed {seed}: not found")
            continue
        data = np.load(str(npz_path))
        mean_rew = data["results"].mean(axis=1)
        best_rewards[seed] = float(mean_rew.max())

    if len(best_rewards) < 2:
        _warn("Not enough seeds for variance check")
        return 0

    values = list(best_rewards.values())
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = abs(std_val / mean_val) * 100 if mean_val != 0 else 0

    for seed, reward in sorted(best_rewards.items()):
        print(f"    Seed {seed}: best={reward:.0f}")
    print(f"    Mean={mean_val:.0f}, Std={std_val:.0f}, CV={cv:.1f}%")

    if cv < 20:
        _pass(f"Multi-seed CV={cv:.1f}% (< 20% threshold)")
        return 0
    else:
        _warn(f"Multi-seed CV={cv:.1f}% (>= 20% threshold, high variance)")
        return 0  # warn, not fail


def main() -> int:
    results_root = PROJECT_ROOT / "results" / "rl"
    benchmark_root = PROJECT_ROOT / "results" / "benchmark"

    print("=" * 60)
    print("Paper Results Validation Checklist")
    print("=" * 60)

    total_failures = 0
    total_failures += check_best_models(results_root)
    total_failures += check_evaluations_npz(results_root)
    total_failures += check_benchmark_csvs(benchmark_root)
    total_failures += check_rl_beats_greedy(benchmark_root)
    total_failures += check_convergence(results_root)
    total_failures += check_multiseed(results_root)

    print("\n" + "=" * 60)
    if total_failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILURES: {total_failures}")
    print("=" * 60)

    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
