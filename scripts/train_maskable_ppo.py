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

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "gymnasium is required. Install via `python3 -m pip install gymnasium`."
    ) from exc

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from sb3_contrib.common.maskable.wrappers import ActionMasker
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "sb3-contrib is required. Install via `python3 -m pip install sb3-contrib stable-baselines3`."
    ) from exc

from baselines.mip.config import MIPBaselineSolverConfig
from coordinator.traffic_manager import TrafficManager
from core.node import ChargingNode, NodeType, create_task_node_pair
from core.task import Task, TaskPool
from core.vehicle import create_vehicle
from physics.time import TimeWindow, TimeWindowType
from strategy.execution_layer import ExecutionLayer
from strategy.gym_env import RuleSelectionGymEnv
from strategy.rule_env import RuleSelectionEnv
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
    core_env = RuleSelectionEnv(
        simulator=simulator,
        execution_layer=executor,
        max_decision_steps=args.max_decision_steps,
        max_time_s=args.max_time_s,
        mip_solver_config=solver_config,
        cost_log_path=str(log_dir / f"cost_log_{stamp}.jsonl"),
        cost_log_csv_path=str(log_dir / f"cost_log_{stamp}.csv"),
        decision_log_path=str(log_dir / f"decision_log_{stamp}.jsonl"),
        decision_log_csv_path=str(log_dir / f"decision_log_{stamp}.csv"),
        task_log_path=str(log_dir / f"task_log_{stamp}.jsonl"),
        task_log_csv_path=str(log_dir / f"task_log_{stamp}.csv"),
        robot_log_path=str(log_dir / f"robot_log_{stamp}.jsonl"),
        robot_log_csv_path=str(log_dir / f"robot_log_{stamp}.csv"),
    )

    gym_env = RuleSelectionGymEnv(core_env)
    return ActionMasker(gym_env, lambda env: env.action_masks())


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

    model = MaskablePPO("MlpPolicy", train_env, verbose=1, seed=args.seed)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(str(log_dir / "final_model"))


if __name__ == "__main__":
    main()
