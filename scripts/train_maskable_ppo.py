"""Train MaskablePPO on the rule-selection environment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import random
import sys

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

from config.instance_generator import PPO_TRAINING, generate_warehouse_instance
from config.benchmark_manifest import (
    load_manifest,
    resolve_entry_path,
    select_manifest_entry,
)
from baselines.mip.config import MIPBaselineSolverConfig
from baselines.mip.scenario_io import ExperimentScenario, load_experiment_json
from coordinator.traffic_manager import TrafficManager
from core.task import TaskPool
from core.vehicle import create_vehicle
from strategy.execution_layer import ExecutionLayer
from strategy.rule_env import RuleSelectionEnv
from strategy.ppo_trainer import PPOTrainingConfig, make_masked_env, train_maskable_ppo
from strategy.scenario_synthesizer import synthesize_scenarios
from strategy.simulator import EventDrivenSimulator


def build_env(
    args,
    *,
    seed: int,
    log_dir: Path,
    experiment: ExperimentScenario | None = None,
) -> RuleSelectionGymEnv:
    if experiment is None:
        layout = replace(
            PPO_TRAINING,
            num_tasks=args.num_tasks,
            num_charging_stations=args.num_chargers,
            seed=seed,
        )
        scenarios = None
        max_time_s = args.max_time_s
    else:
        # Replay: keep the exact layout/scenarios stored on disk.
        layout = experiment.layout
        scenarios = experiment.scenarios
        # Keep 8h horizon as the default benchmark boundary unless explicitly overridden.
        max_time_s = args.max_time_s if args.max_time_s is not None else experiment.episode_length_s
    instance = generate_warehouse_instance(layout)

    pool = TaskPool()
    pool.add_tasks(instance.tasks)

    depot_xy = instance.depot.coordinates
    num_vehicles = args.num_vehicles
    if experiment is not None:
        num_vehicles = int(experiment.metadata.get("num_vehicles", num_vehicles))

    vehicles = [
        create_vehicle(
            vehicle_id=i + 1,
            initial_location=depot_xy,
            battery_capacity=100.0,
            initial_battery=100.0,
        )
        for i in range(num_vehicles)
    ]
    chargers = instance.charging_nodes

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
        max_time_s=max_time_s,
        max_no_progress_steps=args.max_no_progress_steps,
        no_progress_time_epsilon=args.no_progress_time_epsilon,
        mip_solver_config=solver_config,
        scenarios=scenarios,
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="results/rl")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Run periodic evaluation every N environment steps.",
    )
    parser.add_argument("--min-soc-threshold", type=float, default=None)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Number of episodes per periodic evaluation.",
    )
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--num-vehicles", type=int, default=2)
    parser.add_argument("--num-chargers", type=int, default=2)
    parser.add_argument(
        "--max-decision-steps",
        type=int,
        default=None,
        help="Optional hard cap on decision steps; default is unset (time horizon controls episode end).",
    )
    # Paper Section 5.1: 8h episode horizon.
    parser.add_argument(
        "--max-time-s",
        type=float,
        default=28_800.0,
        help="Episode horizon in seconds (paper Section 5.1: 8h).",
    )
    parser.add_argument(
        "--max-no-progress-steps",
        type=int,
        default=512,
        help="Truncate an episode after this many consecutive no-progress decision steps (<=0 disables).",
    )
    parser.add_argument(
        "--no-progress-time-epsilon",
        type=float,
        default=1e-9,
        help="Treat step as no-progress when next_state.t - prev_state.t <= epsilon.",
    )
    parser.add_argument(
        "--experiment-json",
        type=str,
        default=None,
        help="Optional ExperimentScenario JSON for deterministic replay (layout + scenarios).",
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default=None,
        help="Optional benchmark manifest path. Used when --experiment-json is omitted.",
    )
    parser.add_argument(
        "--instances-root",
        type=str,
        default=None,
        help="Optional root for manifest entry paths. Defaults to manifest parent.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Manifest split filter for train env (train/test).",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        help="Manifest split filter for eval env (train/test).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Optional manifest scale filter (S/M/L/XL).",
    )
    parser.add_argument(
        "--seed-id",
        type=int,
        default=None,
        help="Optional manifest seed filter.",
    )
    parser.add_argument(
        "--entry-index",
        type=int,
        default=0,
        help="Entry index among filtered train manifest entries.",
    )
    parser.add_argument(
        "--eval-entry-index",
        type=int,
        default=0,
        help="Entry index among filtered eval manifest entries.",
    )
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

    train_experiment = None
    eval_experiment = None
    if args.experiment_json:
        train_experiment = load_experiment_json(args.experiment_json)
        eval_experiment = train_experiment
    elif args.manifest_json:
        manifest = load_manifest(args.manifest_json)
        instances_root = args.instances_root or str(Path(args.manifest_json).resolve().parent)

        train_entry = select_manifest_entry(
            manifest,
            split=args.split,
            scale=args.scale,
            seed=args.seed_id,
            entry_index=args.entry_index,
        )
        eval_entry = select_manifest_entry(
            manifest,
            split=args.eval_split,
            scale=args.scale,
            seed=args.seed_id,
            entry_index=args.eval_entry_index,
        )
        train_path = resolve_entry_path(train_entry, instances_root=instances_root)
        eval_path = resolve_entry_path(eval_entry, instances_root=instances_root)
        train_experiment = load_experiment_json(train_path)
        eval_experiment = load_experiment_json(eval_path)

    train_env = build_env(args, seed=args.seed, log_dir=log_dir / "train", experiment=train_experiment)
    eval_env = build_env(args, seed=args.seed + 1, log_dir=log_dir / "eval", experiment=eval_experiment)

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
