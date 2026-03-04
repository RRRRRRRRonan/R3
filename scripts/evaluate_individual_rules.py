"""Evaluate each of the 15 dispatch rules independently across all scales.

Runs every individual rule as a fixed policy on 30 test instances per scale,
producing CSV results comparable to evaluate_all.py output.

Usage:
    python scripts/evaluate_individual_rules.py --scale S
    python scripts/evaluate_individual_rules.py              # all scales
    python scripts/evaluate_individual_rules.py --max-instances 5  # quick test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from config.benchmark_manifest import (
    BenchmarkManifestEntry,
    list_manifest_entries,
    load_manifest,
    resolve_entry_path,
)
from baselines.mip.scenario_io import ExperimentScenario, load_experiment_json
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

# ── Rule metadata ──────────────────────────────────────────────────────────
RULES = [
    (RULE_STTF, "STTF", "Dispatch"),
    (RULE_EDD, "EDD", "Dispatch"),
    (RULE_MST, "MST", "Dispatch"),
    (RULE_HPF, "HPF", "Dispatch"),
    (RULE_CHARGE_URGENT, "Charge-Urgent", "Charge"),
    (RULE_CHARGE_TARGET_LOW, "Charge-Low", "Charge"),
    (RULE_CHARGE_TARGET_MED, "Charge-Med", "Charge"),
    (RULE_CHARGE_TARGET_HIGH, "Charge-High", "Charge"),
    (RULE_CHARGE_OPPORTUNITY, "Charge-Opp", "Charge"),
    (RULE_STANDBY_LOW_COST, "Standby-LowCost", "Standby"),
    (RULE_STANDBY_LAZY, "Standby-Lazy", "Standby"),
    (RULE_STANDBY_HEATMAP, "Standby-Heatmap", "Standby"),
    (RULE_ACCEPT_FEASIBLE, "Accept-Feasible", "Accept"),
    (RULE_ACCEPT_VALUE, "Accept-Value", "Accept"),
    (RULE_INSERT_MIN_COST, "Insert-MinCost", "Dispatch"),
]

RULE_ID_TO_NAME = {rid: name for rid, name, _ in RULES}
RULE_ID_TO_CATEGORY = {rid: cat for rid, _, cat in RULES}


def _resolve_instances(args):
    manifest = load_manifest(args.manifest_json)
    entries = list_manifest_entries(
        manifest,
        split=args.split,
        scale=args.scale,
        seed=args.seed_id,
    )
    if not entries:
        raise ValueError(f"No entries matched: split={args.split!r}, scale={args.scale!r}")
    if args.max_instances is not None:
        entries = entries[: max(0, int(args.max_instances))]

    instances_root = (
        Path(args.instances_root).resolve()
        if args.instances_root
        else Path(args.manifest_json).resolve().parent
    )
    out = []
    for entry in entries:
        path = resolve_entry_path(entry, instances_root=instances_root).resolve()
        experiment = load_experiment_json(path)
        out.append((entry, path, experiment))
    return out


def _run_fixed_rule(experiment, *, scenario_index, args, fixed_rule_id, seed):
    """Run a single fixed-rule episode via the same path as evaluate_all.py."""
    # Import here to avoid heavy startup cost at module level
    from strategy.baseline_policies import FixedRulePolicy
    from strategy.action_mask import ALL_RULES
    from strategy.rule_env import RuleSelectionEnv
    from strategy.simulator import EventDrivenSimulator
    from strategy.execution_layer import ExecutionLayer
    from config import CostParameters
    from config.instance_generator import generate_warehouse_instance
    from coordinator.traffic_manager import TrafficManager
    from core.task import TaskPool
    from core.vehicle import create_vehicle
    from baselines.mip import MIPBaselineSolverConfig

    instance = generate_warehouse_instance(experiment.layout)
    pool = TaskPool()
    pool.add_tasks(instance.tasks)

    num_vehicles = int(experiment.metadata.get("num_vehicles", args.num_vehicles))
    initial_battery = args.battery_capacity_kwh
    vehicles = [
        create_vehicle(
            vehicle_id=i + 1,
            capacity=args.vehicle_capacity_kg,
            battery_capacity=args.battery_capacity_kwh,
            speed=args.vehicle_speed_m_s,
            initial_location=instance.depot.coordinates,
            initial_battery=initial_battery,
        )
        for i in range(num_vehicles)
    ]

    traffic = TrafficManager(headway_s=args.headway_s)
    simulator = EventDrivenSimulator(
        task_pool=pool,
        vehicles=vehicles,
        chargers=instance.charging_nodes,
        traffic_manager=traffic,
    )
    solver_config = MIPBaselineSolverConfig(min_soc_threshold=args.min_soc_threshold)

    selected_scenario = None
    if experiment.scenarios and scenario_index < len(experiment.scenarios):
        selected_scenario = experiment.scenarios[scenario_index]
    scenarios = [selected_scenario] if selected_scenario is not None else None

    cost_params = None
    if args.terminal_penalty is not None or args.tardiness_scale is not None:
        from dataclasses import replace as dc_replace
        overrides = {}
        if args.terminal_penalty is not None:
            overrides["C_terminal_unfinished"] = args.terminal_penalty
        if args.tardiness_scale is not None:
            overrides["C_tardiness_shaping_scale"] = args.tardiness_scale
        cost_params = dc_replace(CostParameters(), **overrides)

    env = RuleSelectionEnv(
        simulator=simulator,
        execution_layer=ExecutionLayer(
            task_pool=pool,
            simulator=simulator,
            traffic_manager=traffic,
            min_soc_threshold=solver_config.min_soc_threshold,
        ),
        max_decision_steps=args.max_decision_steps,
        max_time_s=args.max_time_s if args.max_time_s is not None else experiment.episode_length_s,
        max_no_progress_steps=args.max_no_progress_steps,
        no_progress_time_epsilon=args.no_progress_time_epsilon,
        cost_params=cost_params,
        mip_solver_config=solver_config,
        scenarios=scenarios,
        scenario_seed=seed,
        fixed_scenario=True,
        auto_synthesize_scenarios=False,
        cost_log_path=os.devnull,
        cost_log_csv_path=os.devnull,
        decision_log_path=os.devnull,
        decision_log_csv_path=os.devnull,
        task_log_path=os.devnull,
        task_log_csv_path=os.devnull,
        robot_log_path=os.devnull,
        robot_log_csv_path=os.devnull,
    )

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    fixed_policy = FixedRulePolicy(rule_id=int(fixed_rule_id))
    terminated = False
    truncated = False
    last_info = dict(info)

    while True:
        mask = list(env.action_masks())
        action_rule = fixed_policy.select_action(mask)
        if action_rule in ALL_RULES:
            action_rule = ALL_RULES.index(int(action_rule))

        result = env.step(action_rule)
        total_reward += float(result.reward)
        steps += 1
        obs = result.obs
        last_info = dict(result.info)
        terminated = bool(result.terminated)
        truncated = bool(result.truncated)
        if terminated or truncated:
            break

    metrics = env.simulator.metrics
    stats = env.simulator.task_pool.get_statistics()
    return {
        "status": "OK",
        "cost": -total_reward,
        "total_reward": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "terminated_reason": last_info.get("terminated_reason"),
        "completed_tasks": int(stats.get("completed", 0)),
        "rejected_tasks": int(stats.get("rejected", 0)),
        "metrics_total_distance": metrics.total_distance,
        "metrics_total_travel_time": metrics.total_travel_time,
        "metrics_total_charging": metrics.total_charging,
        "metrics_total_delay": metrics.total_delay,
        "metrics_total_waiting": metrics.total_waiting,
        "metrics_total_conflict_waiting": metrics.total_conflict_waiting,
        "metrics_total_standby": metrics.total_standby,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate all 15 individual rules.")
    parser.add_argument("--manifest-json", default="data/instances/manifest.json")
    parser.add_argument("--instances-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scale", default=None, help="S/M/L/XL or None for all.")
    parser.add_argument("--seed-id", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenario-index", type=int, default=0)

    parser.add_argument("--headway-s", type=float, default=2.0)
    parser.add_argument("--min-soc-threshold", type=float, default=None)
    parser.add_argument("--max-decision-steps", type=int, default=None)
    parser.add_argument("--max-time-s", type=float, default=None)
    parser.add_argument("--max-no-progress-steps", type=int, default=512)
    parser.add_argument("--no-progress-time-epsilon", type=float, default=1e-9)
    parser.add_argument("--num-vehicles", type=int, default=2)
    parser.add_argument("--vehicle-capacity-kg", type=float, default=None)
    parser.add_argument("--battery-capacity-kwh", type=float, default=None)
    parser.add_argument("--vehicle-speed-m-s", type=float, default=None)
    parser.add_argument("--terminal-penalty", type=float, default=None)
    parser.add_argument("--tardiness-scale", type=float, default=None)

    # Select which rules to evaluate (default: all 15)
    parser.add_argument(
        "--rule-ids",
        default=None,
        help="Comma-separated rule IDs to evaluate (default: all 1-15).",
    )

    args = parser.parse_args()

    scales = [args.scale] if args.scale else ["S", "M", "L", "XL"]
    rule_ids_to_run = None
    if args.rule_ids:
        rule_ids_to_run = [int(x.strip()) for x in args.rule_ids.split(",")]

    for scale in scales:
        args_copy = argparse.Namespace(**vars(args))
        args_copy.scale = scale

        instances = _resolve_instances(args_copy)
        print(f"\n{'='*60}")
        print(f"Scale {scale}: {len(instances)} instances")
        print(f"{'='*60}")

        rows: List[Dict[str, Any]] = []
        rules_to_eval = [(rid, name, cat) for rid, name, cat in RULES
                         if rule_ids_to_run is None or rid in rule_ids_to_run]

        total = len(instances) * len(rules_to_eval)
        done = 0

        for idx, (entry, path, experiment) in enumerate(instances):
            for rid, rname, rcat in rules_to_eval:
                per_run_seed = int(args.seed) + idx * 100 + rid
                t0 = time.perf_counter()
                try:
                    result = _run_fixed_rule(
                        experiment,
                        scenario_index=args.scenario_index,
                        args=args_copy,
                        fixed_rule_id=rid,
                        seed=per_run_seed,
                    )
                except Exception as exc:
                    result = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
                runtime_s = time.perf_counter() - t0

                row: Dict[str, Any] = {
                    "rule_id": rid,
                    "rule_name": rname,
                    "rule_category": rcat,
                    "algorithm_id": f"fixed_{rid}",
                    "algorithm": rname,
                    "status": result.get("status"),
                    "runtime_s": runtime_s,
                    "scale": entry.scale,
                    "seed": entry.seed,
                    "split": entry.split,
                    "instance_path": entry.path,
                    "num_tasks_manifest": entry.num_tasks,
                    "num_vehicles_manifest": entry.num_vehicles,
                    "num_charging_manifest": entry.num_charging_stations,
                }
                for key, value in result.items():
                    if key not in row:
                        row[key] = value
                rows.append(row)
                done += 1

                status = result.get("status", "?")
                cost = result.get("cost", "?")
                cost_str = f"{cost:,.0f}" if isinstance(cost, (int, float)) else str(cost)
                print(
                    f"[{done}/{total}] Rule {rid:2d} ({rname:16s}) "
                    f"{scale}-seed{entry.seed} "
                    f"status={status} cost={cost_str} "
                    f"runtime={runtime_s:.2f}s"
                )

        out_csv = Path(args.output_dir) / f"individual_rules_{scale}_30.csv"
        _write_csv(out_csv, rows)
        print(f"\nSaved: {out_csv} ({len(rows)} rows)")

        # Write summary JSON
        summary: Dict[str, Any] = {"scale": scale, "instances": len(instances), "rules": {}}
        for rid, rname, rcat in rules_to_eval:
            rule_rows = [r for r in rows if r["rule_id"] == rid and r.get("status") == "OK"]
            costs = [float(r["cost"]) for r in rule_rows]
            summary["rules"][rname] = {
                "rule_id": rid,
                "category": rcat,
                "n": len(costs),
                "avg_cost": sum(costs) / len(costs) if costs else None,
                "std_cost": float(__import__("numpy").std(costs)) if costs else None,
                "min_cost": min(costs) if costs else None,
                "max_cost": max(costs) if costs else None,
            }
        summary_path = Path(args.output_dir) / f"individual_rules_{scale}_30_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
