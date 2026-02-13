"""Run MIP baseline on minimal or manifest-selected benchmark instances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baselines.mip import MIPBaselineResult, MIPBaselineSolverConfig, build_instance, get_default_solver, solve_minimal_instance
from baselines.mip.scenario_io import ExperimentScenario, load_experiment_json
from config.benchmark_manifest import list_manifest_entries, load_manifest, resolve_entry_path
from config.instance_generator import generate_warehouse_instance
from core.vehicle import create_vehicle


def _build_instance_from_experiment(
    experiment: ExperimentScenario,
    args: argparse.Namespace,
):
    layout_instance = generate_warehouse_instance(experiment.layout)
    task_pool = layout_instance.create_task_pool()

    num_vehicles = int(experiment.metadata.get("num_vehicles", args.num_vehicles))
    depot_xy = layout_instance.depot.coordinates
    initial_battery = args.initial_battery_kwh
    if initial_battery is None and args.battery_capacity_kwh is not None:
        initial_battery = args.battery_capacity_kwh

    vehicles = [
        create_vehicle(
            vehicle_id=index + 1,
            capacity=args.vehicle_capacity_kg,
            battery_capacity=args.battery_capacity_kwh,
            speed=args.vehicle_speed_m_s,
            initial_location=depot_xy,
            initial_battery=initial_battery,
        )
        for index in range(num_vehicles)
    ]

    mip_instance = build_instance(
        task_pool=task_pool,
        vehicles=vehicles,
        depot=layout_instance.depot,
        charging_stations=layout_instance.charging_nodes,
        distance_matrix=layout_instance.distance_matrix,
        rule_count=args.rule_count,
        decision_epochs=args.decision_epochs,
        scenarios=experiment.scenarios,
    )
    return mip_instance


def _resolve_experiments(args: argparse.Namespace) -> List[Tuple[str, ExperimentScenario]]:
    if args.experiment_json:
        path = Path(args.experiment_json).resolve()
        return [(path.stem, load_experiment_json(path))]

    if not args.manifest_json:
        return []

    manifest = load_manifest(args.manifest_json)
    entries = list_manifest_entries(
        manifest,
        split=args.split,
        scale=args.scale,
        seed=args.seed_id,
    )
    if not entries:
        raise ValueError(
            "No manifest entries matched filters: split={!r}, scale={!r}, seed={!r}".format(
                args.split,
                args.scale,
                args.seed_id,
            )
        )

    if args.all_matches:
        selected = entries
    else:
        idx = int(args.entry_index)
        if idx < 0 or idx >= len(entries):
            raise IndexError(f"entry-index {idx} out of range for {len(entries)} matched entries")
        selected = [entries[idx]]

    if args.max_instances is not None:
        selected = selected[: max(0, int(args.max_instances))]

    instances_root = (
        Path(args.instances_root).resolve()
        if args.instances_root
        else Path(args.manifest_json).resolve().parent
    )
    resolved: List[Tuple[str, ExperimentScenario]] = []
    for entry in selected:
        path = resolve_entry_path(entry, instances_root=instances_root).resolve()
        run_id = f"{entry.scale}_seed{entry.seed}_{entry.split}"
        resolved.append((run_id, load_experiment_json(path)))
    return resolved


def _result_to_record(run_id: str, result: MIPBaselineResult) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "status": result.status,
        "objective_value": result.objective_value,
        "details": dict(result.details or {}),
    }


def _print_result(run_id: str, result: MIPBaselineResult) -> None:
    print(f"[{run_id}] status: {result.status}")
    if result.objective_value is not None:
        print(f"[{run_id}] objective: {result.objective_value:.4f}")
    if result.details:
        for key in sorted(result.details.keys()):
            print(f"[{run_id}] {key}: {result.details[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve MIP baseline instances.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Time limit (seconds).")
    parser.add_argument("--mip-gap", type=float, default=0.0, help="Relative MIP gap.")
    parser.add_argument(
        "--solver-name",
        type=str,
        default="ortools",
        choices=("ortools", "gurobi"),
        help="MIP backend: 'ortools' (CBC) or 'gurobi' via OR-Tools.",
    )
    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="minimal",
        choices=("minimal", "medium"),
        help="Scenario-mode switch in MIPBaselineSolverConfig.",
    )

    parser.add_argument(
        "--experiment-json",
        type=str,
        default=None,
        help="Single replayable ExperimentScenario JSON path.",
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default=None,
        help="Benchmark manifest JSON. Used when --experiment-json is omitted.",
    )
    parser.add_argument(
        "--instances-root",
        type=str,
        default=None,
        help="Root directory for entry paths in manifest; defaults to manifest parent.",
    )
    parser.add_argument("--split", type=str, default="test", help="Manifest split filter (train/test).")
    parser.add_argument("--scale", type=str, default=None, help="Manifest scale filter (S/M/L/XL).")
    parser.add_argument("--seed-id", type=int, default=None, help="Manifest seed filter.")
    parser.add_argument(
        "--entry-index",
        type=int,
        default=0,
        help="When not using --all-matches, pick this index among filtered entries.",
    )
    parser.add_argument(
        "--all-matches",
        action="store_true",
        help="Run every matched manifest entry instead of a single entry-index.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional upper bound when --all-matches is set.",
    )

    parser.add_argument("--num-vehicles", type=int, default=2, help="Fallback fleet size if metadata missing.")
    parser.add_argument("--vehicle-capacity-kg", type=float, default=None)
    parser.add_argument("--battery-capacity-kwh", type=float, default=None)
    parser.add_argument("--initial-battery-kwh", type=float, default=None)
    parser.add_argument("--vehicle-speed-m-s", type=float, default=None)
    parser.add_argument("--rule-count", type=int, default=13)
    parser.add_argument("--decision-epochs", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    solver_config = MIPBaselineSolverConfig(
        solver_name=args.solver_name,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        scenario_mode=args.scenario_mode,
    )
    solver = get_default_solver(solver_config)

    experiments = _resolve_experiments(args)
    results: List[Dict[str, Any]] = []
    if not experiments:
        result = solve_minimal_instance(solver_config=solver_config)
        _print_result("minimal_instance", result)
        results.append(_result_to_record("minimal_instance", result))
    else:
        for run_id, experiment in experiments:
            mip_instance = _build_instance_from_experiment(experiment, args)
            result = solver.solve(mip_instance)
            _print_result(run_id, result)
            results.append(_result_to_record(run_id, result))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "count": len(results),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"saved_summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
