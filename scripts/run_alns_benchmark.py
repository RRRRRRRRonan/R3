"""Run ALNS solvers on replayable benchmark experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baselines.mip.model import MIPBaselineScenario
from baselines.mip.scenario_io import ExperimentScenario, load_experiment_json
from config import CostParameters
from config.benchmark_manifest import list_manifest_entries, load_manifest, resolve_entry_path
from config.instance_generator import generate_warehouse_instance
from core.node import NodeType, TaskNode
from core.task import Task, TaskPool
from core.vehicle import create_vehicle
from physics.energy import EnergyConfig
from physics.time import TimeWindow, TimeWindowType
from planner.alns import MinimalALNS
from planner.alns_matheuristic import MatheuristicALNS
from planner.fleet import FleetPlanner


def _resolve_time_window(
    base_window: Optional[TimeWindow],
    override: Optional[Tuple[float, float]],
) -> Optional[TimeWindow]:
    if override is None:
        return base_window
    window_type = base_window.window_type if base_window is not None else TimeWindowType.SOFT
    return TimeWindow(
        earliest=float(override[0]),
        latest=float(override[1]),
        window_type=window_type,
    )


def _materialize_tasks(
    base_tasks: List[Task],
    scenario: Optional[MIPBaselineScenario],
) -> List[Task]:
    if scenario is None:
        return list(base_tasks)

    materialized: List[Task] = []
    for task in sorted(base_tasks, key=lambda item: item.task_id):
        if scenario.task_availability.get(task.task_id, 1) <= 0:
            continue

        demand = float(scenario.task_demands.get(task.task_id, task.demand))
        release_time = float(scenario.task_release_times.get(task.task_id, task.arrival_time))

        pickup_tw = _resolve_time_window(
            task.pickup_node.time_window,
            scenario.node_time_windows.get(task.pickup_id),
        )
        delivery_tw = _resolve_time_window(
            task.delivery_node.time_window,
            scenario.node_time_windows.get(task.delivery_id),
        )
        pickup_service = float(
            scenario.node_service_times.get(task.pickup_id, task.pickup_node.service_time)
        )
        delivery_service = float(
            scenario.node_service_times.get(task.delivery_id, task.delivery_node.service_time)
        )

        pickup_node = TaskNode(
            node_id=task.pickup_id,
            node_type=NodeType.PICKUP,
            coordinates=task.pickup_coordinates,
            task_id=task.task_id,
            time_window=pickup_tw,
            service_time=pickup_service,
            demand=demand,
        )
        delivery_node = TaskNode(
            node_id=task.delivery_id,
            node_type=NodeType.DELIVERY,
            coordinates=task.delivery_coordinates,
            task_id=task.task_id,
            time_window=delivery_tw,
            service_time=delivery_service,
            demand=demand,
        )
        materialized.append(
            Task(
                task_id=task.task_id,
                pickup_node=pickup_node,
                delivery_node=delivery_node,
                demand=demand,
                priority=task.priority,
                arrival_time=release_time,
            )
        )

    return materialized


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


def _solver_class(name: str) -> Type[MinimalALNS]:
    key = str(name).strip().lower()
    if key == "minimal":
        return MinimalALNS
    if key == "matheuristic":
        return MatheuristicALNS
    raise ValueError(f"Unsupported algorithm: {name}")


def _run_single(
    run_id: str,
    experiment: ExperimentScenario,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    warehouse = generate_warehouse_instance(experiment.layout)
    scenario: Optional[MIPBaselineScenario] = None
    if experiment.scenarios:
        if args.scenario_index < 0 or args.scenario_index >= len(experiment.scenarios):
            raise IndexError(
                f"scenario-index {args.scenario_index} out of range for {len(experiment.scenarios)} scenarios"
            )
        scenario = experiment.scenarios[args.scenario_index]

    tasks = _materialize_tasks(warehouse.tasks, scenario)
    task_pool = TaskPool()
    task_pool.add_tasks(tasks)

    num_vehicles = int(experiment.metadata.get("num_vehicles", args.num_vehicles))
    depot_xy = warehouse.depot.coordinates
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

    planner = FleetPlanner(
        distance_matrix=warehouse.distance_matrix,
        depot=warehouse.depot,
        vehicles=vehicles,
        task_pool=task_pool,
        energy_config=EnergyConfig(),
        cost_params=CostParameters(),
        repair_mode=args.repair_mode,
        use_adaptive=(not args.no_adaptive),
        verbose=args.verbose,
        alns_class=_solver_class(args.algorithm),
    )
    plan = planner.plan_routes(max_iterations=args.max_iterations)

    nonempty_routes = sum(1 for route in plan.routes.values() if not route.is_empty())
    improvement = plan.initial_cost - plan.optimised_cost
    improvement_ratio = (improvement / plan.initial_cost) if plan.initial_cost > 0 else 0.0

    record: Dict[str, Any] = {
        "run_id": run_id,
        "algorithm": args.algorithm,
        "split": experiment.metadata.get("split"),
        "scale": experiment.metadata.get("scale"),
        "seed": experiment.metadata.get("seed"),
        "num_tasks": len(tasks),
        "num_vehicles": num_vehicles,
        "scenario_id": scenario.scenario_id if scenario is not None else None,
        "initial_cost": plan.initial_cost,
        "optimised_cost": plan.optimised_cost,
        "improvement": improvement,
        "improvement_ratio": improvement_ratio,
        "unassigned_tasks": len(plan.unassigned_tasks),
        "nonempty_routes": nonempty_routes,
    }
    return record


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "avg_initial_cost": 0.0,
            "avg_optimised_cost": 0.0,
            "avg_improvement": 0.0,
            "avg_improvement_ratio": 0.0,
        }
    return {
        "count": len(results),
        "avg_initial_cost": mean(item["initial_cost"] for item in results),
        "avg_optimised_cost": mean(item["optimised_cost"] for item in results),
        "avg_improvement": mean(item["improvement"] for item in results),
        "avg_improvement_ratio": mean(item["improvement_ratio"] for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ALNS benchmark instances.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="minimal",
        choices=("minimal", "matheuristic"),
        help="ALNS solver variant.",
    )
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--repair-mode", type=str, default="mixed")
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Scenario index to apply when ExperimentScenario includes multiple scenarios.",
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

    parser.add_argument("--num-vehicles", type=int, default=1, help="Fallback fleet size if metadata missing.")
    parser.add_argument("--vehicle-capacity-kg", type=float, default=None)
    parser.add_argument("--battery-capacity-kwh", type=float, default=None)
    parser.add_argument("--initial-battery-kwh", type=float, default=None)
    parser.add_argument("--vehicle-speed-m-s", type=float, default=None)

    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    experiments = _resolve_experiments(args)
    if not experiments:
        raise ValueError("No input selected. Provide --experiment-json or --manifest-json.")

    results: List[Dict[str, Any]] = []
    for run_id, experiment in experiments:
        record = _run_single(run_id, experiment, args)
        results.append(record)
        print(
            "[{run_id}] {algo} tasks={tasks} vehicles={vehicles} initial={initial:.3f} "
            "optimized={optimized:.3f} improvement={improvement:.3f} ({ratio:.2%})".format(
                run_id=record["run_id"],
                algo=record["algorithm"],
                tasks=record["num_tasks"],
                vehicles=record["num_vehicles"],
                initial=record["initial_cost"],
                optimized=record["optimised_cost"],
                improvement=record["improvement"],
                ratio=record["improvement_ratio"],
            )
        )

    summary = _summarize(results)
    print(
        "summary: count={count} avg_initial={avg_initial:.3f} avg_optimized={avg_optimized:.3f} "
        "avg_improvement={avg_improvement:.3f} avg_ratio={avg_ratio:.2%}".format(
            count=summary["count"],
            avg_initial=summary["avg_initial_cost"],
            avg_optimized=summary["avg_optimised_cost"],
            avg_improvement=summary["avg_improvement"],
            avg_ratio=summary["avg_improvement_ratio"],
        )
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"saved_summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
