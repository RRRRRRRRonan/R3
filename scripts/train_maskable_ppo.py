"""Train MaskablePPO on the rule-selection environment."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import random
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "torch is required for MaskablePPO reproducibility. Install via `python3 -m pip install torch`."
    ) from exc

from baselines.mip.config import MIPBaselineSolverConfig
from coordinator.traffic_manager import TrafficManager
from core.node import ChargingNode, NodeType, create_task_node_pair
from core.task import Task, TaskPool
from core.vehicle import create_vehicle
from physics.time import TimeWindow, TimeWindowType
from strategy.execution_layer import ExecutionLayer
from strategy.rule_env import RuleSelectionEnv
from strategy.ppo_trainer import PPOTrainingConfig, make_masked_env, train_maskable_ppo
from strategy.scenario_synthesizer import synthesize_scenarios
from strategy.simulator import EventDrivenSimulator


def build_synthetic_tasks(num_tasks: int, seed: int) -> Tuple[List[Task], dict]:
    rng = random.Random(seed)
    tasks: List[Task] = []
    coordinates = {0: (0.0, 0.0)}

    for task_id in range(1, num_tasks + 1):
        pickup_xy = (rng.uniform(5.0, 45.0), rng.uniform(5.0, 45.0))
        delivery_xy = (rng.uniform(5.0, 45.0), rng.uniform(5.0, 45.0))
        pickup, delivery = create_task_node_pair(
            task_id=task_id,
            pickup_id=task_id * 2 - 1,
            delivery_id=task_id * 2,
            pickup_coords=pickup_xy,
            delivery_coords=delivery_xy,
            demand=5.0,
            service_time=10.0,
            pickup_time_window=TimeWindow(0.0, 10_000.0, TimeWindowType.SOFT),
            delivery_time_window=TimeWindow(0.0, 10_000.0, TimeWindowType.SOFT),
        )
        task = Task(task_id=task_id, pickup_node=pickup, delivery_node=delivery, demand=5.0, arrival_time=0.0)
        tasks.append(task)
        coordinates[pickup.node_id] = pickup.coordinates
        coordinates[delivery.node_id] = delivery.coordinates
    return tasks, coordinates


def build_env(args, *, seed: int, log_dir: Path) -> RuleSelectionGymEnv:
    tasks, coords = build_synthetic_tasks(args.num_tasks, seed)
    pool = TaskPool()
    pool.add_tasks(tasks)

    vehicles = [
        create_vehicle(vehicle_id=i + 1, initial_location=(0.0, 0.0), battery_capacity=100.0, initial_battery=100.0)
        for i in range(args.num_vehicles)
    ]

    chargers = []
    for idx in range(args.num_chargers):
        node_id = 10_000 + idx
        coord = (10.0 + idx * 20.0, 10.0 + idx * 20.0)
        chargers.append(ChargingNode(node_id=node_id, coordinates=coord, node_type=NodeType.CHARGING))

    traffic = TrafficManager(headway_s=args.headway_s)
    simulator = EventDrivenSimulator(task_pool=pool, vehicles=vehicles, chargers=chargers, traffic_manager=traffic)
    solver_config = MIPBaselineSolverConfig(min_soc_threshold=args.min_soc_threshold)
    executor = ExecutionLayer(
        task_pool=pool,
        simulator=simulator,
        traffic_manager=traffic,
        min_soc_threshold=solver_config.min_soc_threshold,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_log_path = _resolve_heatmap_log_path(
        args.heatmap_log_path,
        pool,
        chargers,
        solver_config,
        seed=seed,
        log_dir=log_dir,
        auto_synth=args.auto_heatmap_from_synth,
    )
    core_env = RuleSelectionEnv(
        simulator=simulator,
        execution_layer=executor,
        max_decision_steps=args.max_decision_steps,
        max_time_s=args.max_time_s,
        mip_solver_config=solver_config,
        heatmap_log_path=heatmap_log_path,
        cost_log_path=str(log_dir / f"cost_log_{stamp}.jsonl"),
        cost_log_csv_path=str(log_dir / f"cost_log_{stamp}.csv"),
        decision_log_path=str(log_dir / f"decision_log_{stamp}.jsonl"),
        decision_log_csv_path=str(log_dir / f"decision_log_{stamp}.csv"),
        task_log_path=str(log_dir / f"task_log_{stamp}.jsonl"),
        task_log_csv_path=str(log_dir / f"task_log_{stamp}.csv"),
        robot_log_path=str(log_dir / f"robot_log_{stamp}.jsonl"),
        robot_log_csv_path=str(log_dir / f"robot_log_{stamp}.csv"),
    )

    return make_masked_env(core_env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="results/rl")
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--min-soc-threshold", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--num-vehicles", type=int, default=2)
    parser.add_argument("--num-chargers", type=int, default=2)
    parser.add_argument("--max-decision-steps", type=int, default=200)
    parser.add_argument("--max-time-s", type=float, default=10_000.0)
    parser.add_argument("--headway-s", type=float, default=2.0)
    parser.add_argument("--heatmap-log-path", type=str, default=None)
    parser.add_argument("--auto-heatmap-from-synth", action="store_true", default=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_env = build_env(args, seed=args.seed, log_dir=log_dir / "train")
    eval_env = build_env(args, seed=args.seed + 1, log_dir=log_dir / "eval")

    config = PPOTrainingConfig(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        log_dir=str(log_dir),
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        deterministic_eval=True,
    )
    train_maskable_ppo(train_env, eval_env, config=config)


def _resolve_heatmap_log_path(
    path: str | None,
    pool: TaskPool,
    chargers: List[ChargingNode],
    solver_config: MIPBaselineSolverConfig,
    *,
    seed: int,
    log_dir: Path,
    auto_synth: bool,
) -> str | None:
    if path:
        return path
    if auto_synth:
        return _generate_heatmap_log_from_synth(
            pool,
            chargers,
            solver_config,
            seed=seed,
            log_dir=log_dir,
        )
    return _resolve_latest_task_log()


def _resolve_latest_task_log() -> str | None:
    logs_dir = ROOT / "results" / "logs"
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("task_log_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return None


def _generate_heatmap_log_from_synth(
    pool: TaskPool,
    chargers: List[ChargingNode],
    solver_config: MIPBaselineSolverConfig,
    *,
    seed: int,
    log_dir: Path,
) -> str:
    scenarios = synthesize_scenarios(
        pool,
        chargers=chargers,
        seed=seed,
        config=solver_config.scenario_synth_config,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"task_log_heatmap_{stamp}.csv"
    fieldnames = [
        "time",
        "task_id",
        "prev_status",
        "status",
        "prev_assigned_vehicle_id",
        "assigned_vehicle_id",
        "pickup_node_id",
        "delivery_node_id",
        "demand",
        "priority",
        "arrival_time",
        "scenario_id",
    ]
    rows = []
    tasks = {task.task_id: task for task in pool.get_all_tasks()}
    for scenario in scenarios:
        for task_id, task in tasks.items():
            if scenario.task_availability.get(task_id, 1) <= 0:
                continue
            arrival_time = scenario.task_release_times.get(task_id, task.arrival_time)
            demand = scenario.task_demands.get(task_id, task.demand)
            rows.append(
                {
                    "time": float(arrival_time),
                    "task_id": task_id,
                    "prev_status": "pending",
                    "status": "pending",
                    "prev_assigned_vehicle_id": "",
                    "assigned_vehicle_id": "",
                    "pickup_node_id": task.pickup_node.node_id,
                    "delivery_node_id": task.delivery_node.node_id,
                    "demand": float(demand),
                    "priority": int(task.priority),
                    "arrival_time": float(arrival_time),
                    "scenario_id": scenario.scenario_id,
                }
            )
    rows.sort(key=lambda row: row["time"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


if __name__ == "__main__":
    main()
