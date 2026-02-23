"""Unified benchmark evaluator for 7 comparison algorithms.

Evaluates instance entries from ``data/instances/manifest.json`` and writes a
standardized CSV with one row per (instance, algorithm) run.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baselines.mip import MIPBaselineSolverConfig, build_instance, get_default_solver
from baselines.mip.model import MIPBaselineScenario
from baselines.mip.scenario_io import ExperimentScenario, load_experiment_json
from config import CostParameters
from config.benchmark_manifest import BenchmarkManifestEntry, list_manifest_entries, load_manifest, resolve_entry_path
from config.instance_generator import generate_warehouse_instance
from coordinator.traffic_manager import TrafficManager
from core.node import NodeType, TaskNode
from core.task import Task, TaskPool
from core.vehicle import create_vehicle
from physics.energy import EnergyConfig
from physics.time import TimeWindow, TimeWindowType
from planner.alns import MinimalALNS
from planner.fleet import FleetPlanner
from strategy.charging_strategies import FullRechargeStrategy, PartialRechargeFixedStrategy
from strategy.execution_layer import ExecutionLayer
from strategy.baseline_policies import (
    FixedRulePolicy,
    GREEDY_FR_RULE_ID,
    GREEDY_PR_RULE_ID,
    RandomMaskedRulePolicy,
)
from strategy.rule_env import RuleSelectionEnv
from strategy.simulator import EventDrivenSimulator

ALGORITHM_ORDER = (
    "rl_apc",
    "alns_fr",
    "alns_pr",
    "mip_hind",
    "greedy_fr",
    "greedy_pr",
    "random_rule",
)

ALGORITHM_LABELS = {
    "rl_apc": "RL-APC",
    "alns_fr": "ALNS-FR",
    "alns_pr": "ALNS-PR",
    "mip_hind": "MIP-Hind",
    "greedy_fr": "Greedy-FR",
    "greedy_pr": "Greedy-PR",
    "random_rule": "Random-Rule",
}

_RL_MODEL_CACHE: Dict[Tuple[str, int, int], Any] = {}


@dataclass(frozen=True)
class InstanceRun:
    entry: BenchmarkManifestEntry
    path: Path
    experiment: ExperimentScenario


def _parse_algorithms(raw: str) -> List[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("No algorithms specified.")
    unknown = [name for name in values if name not in ALGORITHM_ORDER]
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}. Allowed: {list(ALGORITHM_ORDER)}")
    return values


def _parse_scale_set(raw: str) -> set[str]:
    values = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    return set(values)


def _resolve_instances(args: argparse.Namespace) -> List[InstanceRun]:
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
    if args.entry_index is not None:
        idx = int(args.entry_index)
        if idx < 0 or idx >= len(entries):
            raise IndexError(f"entry-index {idx} out of range for {len(entries)} entries")
        entries = [entries[idx]]
    elif args.max_instances is not None:
        entries = entries[: max(0, int(args.max_instances))]

    instances_root = (
        Path(args.instances_root).resolve()
        if args.instances_root
        else Path(args.manifest_json).resolve().parent
    )
    out: List[InstanceRun] = []
    for entry in entries:
        path = resolve_entry_path(entry, instances_root=instances_root).resolve()
        experiment = load_experiment_json(path)
        out.append(InstanceRun(entry=entry, path=path, experiment=experiment))
    return out


def _select_scenario(
    experiment: ExperimentScenario,
    scenario_index: int,
) -> Optional[MIPBaselineScenario]:
    if not experiment.scenarios:
        return None
    if scenario_index < 0 or scenario_index >= len(experiment.scenarios):
        raise IndexError(
            f"scenario-index {scenario_index} out of range for {len(experiment.scenarios)} scenarios"
        )
    return experiment.scenarios[scenario_index]


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
    base_tasks: Sequence[Task],
    scenario: Optional[MIPBaselineScenario],
) -> List[Task]:
    if scenario is None:
        return list(base_tasks)

    tasks: List[Task] = []
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
        tasks.append(
            Task(
                task_id=task.task_id,
                pickup_node=pickup_node,
                delivery_node=delivery_node,
                demand=demand,
                priority=task.priority,
                arrival_time=release_time,
            )
        )
    return tasks


def _build_fleet_from_experiment(
    experiment: ExperimentScenario,
    depot_xy: Tuple[float, float],
    args: argparse.Namespace,
) -> List[Any]:
    num_vehicles = int(experiment.metadata.get("num_vehicles", args.num_vehicles))
    initial_battery = args.initial_battery_kwh
    if initial_battery is None and args.battery_capacity_kwh is not None:
        initial_battery = args.battery_capacity_kwh
    return [
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


def _build_rule_env(
    experiment: ExperimentScenario,
    *,
    scenario_index: int,
    seed: int,
    args: argparse.Namespace,
) -> RuleSelectionEnv:
    instance = generate_warehouse_instance(experiment.layout)
    pool = TaskPool()
    pool.add_tasks(instance.tasks)
    vehicles = _build_fleet_from_experiment(experiment, instance.depot.coordinates, args)
    traffic = TrafficManager(headway_s=args.headway_s)
    simulator = EventDrivenSimulator(
        task_pool=pool,
        vehicles=vehicles,
        chargers=instance.charging_nodes,
        traffic_manager=traffic,
    )
    solver_config = MIPBaselineSolverConfig(min_soc_threshold=args.min_soc_threshold)
    selected_scenario = _select_scenario(experiment, scenario_index)
    scenarios = [selected_scenario] if selected_scenario is not None else None
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
    return env


def _run_rule_env_policy(
    experiment: ExperimentScenario,
    *,
    scenario_index: int,
    args: argparse.Namespace,
    policy_kind: str,
    model=None,
    fixed_rule_id: Optional[int] = None,
    seed: int,
) -> Dict[str, Any]:
    env = _build_rule_env(experiment, scenario_index=scenario_index, seed=seed, args=args)
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    random_policy = RandomMaskedRulePolicy(seed=seed)
    fixed_policy = FixedRulePolicy(rule_id=int(fixed_rule_id)) if fixed_rule_id is not None else None
    terminated = False
    truncated = False
    last_info = dict(info)

    while True:
        mask = list(env.action_masks())
        if policy_kind == "rl":
            import numpy as np

            flat = np.asarray(
                obs.get("vehicles", []) + obs.get("tasks", []) + obs.get("chargers", []) + obs.get("meta", []),
                dtype=np.float32,
            )
            if isinstance(model, str):
                model = _get_cached_rl_model(
                    model,
                    observation_dim=int(flat.shape[0]),
                    action_dim=len(mask),
                )
            action, _ = model.predict(
                flat,
                action_masks=np.asarray(mask, dtype=bool),
                deterministic=bool(args.rl_deterministic),
            )
            action_rule_or_index = int(action)
        elif policy_kind == "fixed":
            if fixed_policy is None:
                raise ValueError("fixed_rule_id is required for fixed policy")
            action_rule_or_index = fixed_policy.select_action(mask)
        elif policy_kind == "random":
            action_rule_or_index = random_policy.select_action(mask)
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported policy kind: {policy_kind}")

        result = env.step(action_rule_or_index)
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


def _run_alns(
    experiment: ExperimentScenario,
    *,
    scenario_index: int,
    args: argparse.Namespace,
    charge_strategy,
) -> Dict[str, Any]:
    def _build_planner(tasks: Sequence[Task], strategy):
        pool = TaskPool()
        pool.add_tasks(tasks)
        vehicles = _build_fleet_from_experiment(experiment, warehouse.depot.coordinates, args)
        return FleetPlanner(
            distance_matrix=warehouse.distance_matrix,
            depot=warehouse.depot,
            vehicles=vehicles,
            task_pool=pool,
            energy_config=EnergyConfig(),
            cost_params=CostParameters(),
            charging_strategy=strategy,
            repair_mode=args.alns_repair_mode,
            use_adaptive=(not args.alns_no_adaptive),
            verbose=args.alns_verbose,
            alns_class=MinimalALNS,
        )

    def _plan_to_result(plan, *, num_tasks: int) -> Dict[str, Any]:
        initial_cost = float(plan.initial_cost)
        optimised_cost = float(plan.optimised_cost)
        improvement = float(initial_cost - optimised_cost)
        # Guard against non-finite outputs (e.g., inf from infeasible ALNS runs).
        if not all(math.isfinite(value) for value in (initial_cost, optimised_cost, improvement)):
            return {
                "status": "ERROR",
                "error": "non_finite_alns_cost",
                "cost": optimised_cost,
                "initial_cost": initial_cost,
                "improvement": improvement,
                "improvement_ratio": None,
                "num_tasks": num_tasks,
                "unassigned_tasks": len(plan.unassigned_tasks),
            }

        improvement_ratio = (improvement / initial_cost) if initial_cost > 0 else 0.0
        if not math.isfinite(improvement_ratio):
            return {
                "status": "ERROR",
                "error": "non_finite_alns_improvement_ratio",
                "cost": optimised_cost,
                "initial_cost": initial_cost,
                "improvement": improvement,
                "improvement_ratio": None,
                "num_tasks": num_tasks,
                "unassigned_tasks": len(plan.unassigned_tasks),
            }

        return {
            "status": "OK",
            "cost": optimised_cost,
            "initial_cost": initial_cost,
            "improvement": improvement,
            "improvement_ratio": improvement_ratio,
            "num_tasks": num_tasks,
            "unassigned_tasks": len(plan.unassigned_tasks),
        }

    def _parse_charge_ratios(raw: str) -> List[float]:
        ratios: List[float] = []
        for token in str(raw).split(","):
            value = token.strip()
            if not value:
                continue
            try:
                ratio = float(value)
            except ValueError:
                continue
            if 0.0 < ratio <= 1.0:
                ratios.append(ratio)
        return ratios

    warehouse = generate_warehouse_instance(experiment.layout)
    scenario = _select_scenario(experiment, scenario_index)
    tasks = _materialize_tasks(warehouse.tasks, scenario)

    # For ALNS-PR, progressively raise charge ratio and finally fall back to FR
    # when non-finite costs indicate infeasible partial-recharge schedules.
    strategy_chain = [charge_strategy]
    if isinstance(charge_strategy, PartialRechargeFixedStrategy) and not bool(
        getattr(args, "alns_pr_disable_fallback", False)
    ):
        base_ratio = float(charge_strategy.charge_ratio)
        parsed_ratios = _parse_charge_ratios(
            getattr(args, "alns_pr_fallback_charge_ratios", "0.9,1.0")
        )
        seen_ratios = {round(base_ratio, 6)}
        for ratio in parsed_ratios:
            key = round(ratio, 6)
            if key in seen_ratios:
                continue
            seen_ratios.add(key)
            strategy_chain.append(PartialRechargeFixedStrategy(charge_ratio=ratio))
        strategy_chain.append(FullRechargeStrategy())

    attempted = []
    last_result: Dict[str, Any] | None = None
    for idx, strategy in enumerate(strategy_chain):
        planner = _build_planner(tasks, strategy)
        plan = planner.plan_routes(max_iterations=args.alns_iterations)
        result = _plan_to_result(plan, num_tasks=len(tasks))
        strategy_name = (
            strategy.get_strategy_name() if hasattr(strategy, "get_strategy_name") else type(strategy).__name__
        )
        attempted.append(strategy_name)

        if result.get("status") == "OK":
            result["fallback_used"] = idx > 0
            result["initial_strategy"] = (
                charge_strategy.get_strategy_name()
                if hasattr(charge_strategy, "get_strategy_name")
                else type(charge_strategy).__name__
            )
            result["effective_strategy"] = strategy_name
            if hasattr(strategy, "charge_ratio"):
                result["effective_charge_ratio"] = float(strategy.charge_ratio)
            result["fallback_attempts"] = idx
            if idx > 0 and hasattr(charge_strategy, "charge_ratio"):
                result["initial_charge_ratio"] = float(charge_strategy.charge_ratio)
            return result
        last_result = result

    if last_result is None:  # pragma: no cover - defensive guard
        return {
            "status": "ERROR",
            "error": "alns_internal_no_result",
            "num_tasks": len(tasks),
            "unassigned_tasks": None,
        }
    last_result = dict(last_result)
    last_result["fallback_used"] = len(strategy_chain) > 1
    last_result["attempted_strategies"] = ";".join(attempted)
    return last_result


def _run_mip(
    experiment: ExperimentScenario,
    args: argparse.Namespace,
    *,
    time_limit_s: float,
    decision_epochs: int,
) -> Dict[str, Any]:
    layout_instance = generate_warehouse_instance(experiment.layout)
    task_pool = layout_instance.create_task_pool()
    vehicles = _build_fleet_from_experiment(experiment, layout_instance.depot.coordinates, args)
    mip_instance = build_instance(
        task_pool=task_pool,
        vehicles=vehicles,
        depot=layout_instance.depot,
        charging_stations=layout_instance.charging_nodes,
        distance_matrix=layout_instance.distance_matrix,
        rule_count=args.mip_rule_count,
        decision_epochs=decision_epochs,
        scenarios=experiment.scenarios,
    )
    solver_config = MIPBaselineSolverConfig(
        solver_name=args.mip_solver_name,
        time_limit_s=time_limit_s,
        mip_gap=args.mip_gap,
        scenario_mode=args.mip_scenario_mode,
    )
    solver = get_default_solver(solver_config)
    result = solver.solve(mip_instance)
    data: Dict[str, Any] = {
        "status": result.status,
        "cost": result.objective_value,
        "mip_time_limit_s": time_limit_s,
        "mip_decision_epochs": decision_epochs,
    }
    if result.details:
        for key, value in result.details.items():
            data[f"details_{key}"] = value
    return data


def _resolve_mip_runtime_budget(scale: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Return MIP runtime policy by scale.

    Default policy:
    - S: use ``--mip-time-limit-s``.
    - M: use ``--mip-time-limit-s-medium``.
    - L/XL: skip by default (configurable via ``--mip-skip-scales``).
    """

    scale_key = str(scale).strip().upper()
    skip_scales = _parse_scale_set(args.mip_skip_scales)
    if scale_key in skip_scales:
        return {
            "skip": True,
            "skip_reason": f"mip_budget_disabled_for_scale_{scale_key}",
            "time_limit_s": None,
            "decision_epochs": None,
        }

    if scale_key == "M":
        time_limit_s = float(args.mip_time_limit_s_medium)
    else:
        time_limit_s = float(args.mip_time_limit_s)

    return {
        "skip": False,
        "time_limit_s": time_limit_s,
        "decision_epochs": int(args.mip_decision_epochs),
    }


def _install_numpy_pickle_compat_aliases() -> None:
    """Install numpy module aliases for cross-version pickle compatibility.

    Some models are pickled against ``numpy._core.*`` (newer layout) while
    others expect ``numpy.core.*`` (older layout). We install both aliases when
    possible so ``MaskablePPO.load`` can deserialize across environments.
    """

    pairs = [
        ("numpy.core", "numpy._core"),
        ("numpy._core", "numpy.core"),
        ("numpy.core.numeric", "numpy._core.numeric"),
        ("numpy._core.numeric", "numpy.core.numeric"),
        ("numpy.core.multiarray", "numpy._core.multiarray"),
        ("numpy._core.multiarray", "numpy.core.multiarray"),
        ("numpy.core._multiarray_umath", "numpy._core._multiarray_umath"),
        ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"),
    ]
    for source, alias in pairs:
        if alias in sys.modules:
            continue
        try:
            sys.modules[alias] = importlib.import_module(source)
        except Exception:
            continue


def _install_numpy_random_pickle_ctor_compat() -> None:
    """Patch numpy bit-generator ctor for cross-version pickle payloads."""

    try:
        np_pickle = importlib.import_module("numpy.random._pickle")
    except Exception:
        return

    if getattr(np_pickle, "_codex_bitgen_ctor_patched", False):
        return

    original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    if original_ctor is None:
        return

    def _compat_ctor(bit_generator_name="MT19937"):
        value = bit_generator_name
        if isinstance(value, type):
            cls_name = getattr(value, "__name__", None)
            if cls_name:
                value = cls_name
        if not isinstance(value, str):
            value = str(value)
        if value.startswith("<class '") and value.endswith("'>"):
            value = value[len("<class '") : -len("'>")]
        if "." in value:
            value = value.split(".")[-1]
        return original_ctor(value)

    setattr(np_pickle, "__bit_generator_ctor", _compat_ctor)
    setattr(np_pickle, "_codex_bitgen_ctor_patched", True)


def _load_maskable_ppo_model(
    path: str,
    *,
    observation_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
):
    from sb3_contrib import MaskablePPO
    custom_objects = None
    if observation_dim is not None and action_dim is not None:
        import gymnasium as gym
        import numpy as np

        custom_objects = {
            "observation_space": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(observation_dim),),
                dtype=np.float32,
            ),
            "action_space": gym.spaces.Discrete(int(action_dim)),
        }

    try:
        return MaskablePPO.load(path, custom_objects=custom_objects)
    except ModuleNotFoundError as exc:
        # Windows/Linux env drift often shows up as numpy.core vs numpy._core.
        if "numpy._core" not in str(exc) and "numpy.core" not in str(exc):
            raise
        _install_numpy_pickle_compat_aliases()
        _install_numpy_random_pickle_ctor_compat()
        return MaskablePPO.load(path, custom_objects=custom_objects)


def _get_cached_rl_model(model_path: str, *, observation_dim: int, action_dim: int):
    key = (str(model_path), int(observation_dim), int(action_dim))
    model = _RL_MODEL_CACHE.get(key)
    if model is None:
        model = _load_maskable_ppo_model(
            str(model_path),
            observation_dim=int(observation_dim),
            action_dim=int(action_dim),
        )
        _RL_MODEL_CACHE[key] = model
    return model


def _evaluate_algorithm(
    algo: str,
    instance: InstanceRun,
    args: argparse.Namespace,
    *,
    rl_model=None,
    seed: int,
) -> Dict[str, Any]:
    if algo == "rl_apc":
        if rl_model is None:
            return {"status": "SKIPPED", "skip_reason": "missing_rl_model_or_dependency"}
        return _run_rule_env_policy(
            instance.experiment,
            scenario_index=args.scenario_index,
            args=args,
            policy_kind="rl",
            model=rl_model,
            seed=seed,
        )
    if algo == "greedy_fr":
        return _run_rule_env_policy(
            instance.experiment,
            scenario_index=args.scenario_index,
            args=args,
            policy_kind="fixed",
            fixed_rule_id=GREEDY_FR_RULE_ID,
            seed=seed,
        )
    if algo == "greedy_pr":
        return _run_rule_env_policy(
            instance.experiment,
            scenario_index=args.scenario_index,
            args=args,
            policy_kind="fixed",
            fixed_rule_id=GREEDY_PR_RULE_ID,
            seed=seed,
        )
    if algo == "random_rule":
        return _run_rule_env_policy(
            instance.experiment,
            scenario_index=args.scenario_index,
            args=args,
            policy_kind="random",
            seed=seed,
        )
    if algo == "alns_fr":
        return _run_alns(
            instance.experiment,
            scenario_index=args.scenario_index,
            args=args,
            charge_strategy=FullRechargeStrategy(),
        )
    if algo == "alns_pr":
        return _run_alns(
            instance.experiment,
            scenario_index=args.scenario_index,
            args=args,
            charge_strategy=PartialRechargeFixedStrategy(charge_ratio=args.alns_pr_charge_ratio),
        )
    if algo == "mip_hind":
        budget = _resolve_mip_runtime_budget(instance.entry.scale, args)
        if budget["skip"]:
            return {
                "status": "SKIPPED",
                "skip_reason": budget["skip_reason"],
            }
        return _run_mip(
            instance.experiment,
            args,
            time_limit_s=float(budget["time_limit_s"]),
            decision_epochs=int(budget["decision_epochs"]),
        )
    raise ValueError(f"Unknown algorithm: {algo}")


def _row_from_result(
    instance: InstanceRun,
    algo: str,
    runtime_s: float,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "algorithm": ALGORITHM_LABELS[algo],
        "algorithm_id": algo,
        "status": result.get("status"),
        "runtime_s": runtime_s,
        "scale": instance.entry.scale,
        "seed": instance.entry.seed,
        "split": instance.entry.split,
        "instance_path": instance.entry.path,
        "num_tasks_manifest": instance.entry.num_tasks,
        "num_vehicles_manifest": instance.entry.num_vehicles,
        "num_charging_manifest": instance.entry.num_charging_stations,
    }
    for key, value in result.items():
        if key in row:
            continue
        row[key] = value
    return row


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


def _write_summary_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    by_algo: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_algo.setdefault(str(row["algorithm_id"]), []).append(row)
    summary: Dict[str, Any] = {"total_rows": len(rows), "algorithms": {}}
    for algo, items in by_algo.items():
        success = [item for item in items if str(item.get("status")) in ("OK", "OPTIMAL", "FEASIBLE")]
        costs = [float(item["cost"]) for item in success if item.get("cost") is not None]
        summary["algorithms"][algo] = {
            "count": len(items),
            "success_count": len(success),
            "avg_cost": (sum(costs) / len(costs)) if costs else None,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified 7-algorithm benchmark evaluation.")
    parser.add_argument("--manifest-json", type=str, default="data/instances/manifest.json")
    parser.add_argument("--instances-root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, help="Optional split filter (train/test).")
    parser.add_argument("--scale", type=str, default=None, help="Optional scale filter (S/M/L/XL).")
    parser.add_argument("--seed-id", type=int, default=None, help="Optional seed filter.")
    parser.add_argument(
        "--entry-index",
        type=int,
        default=None,
        help="Evaluate a single filtered manifest entry by index.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on number of filtered entries (ignored with --entry-index).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(ALGORITHM_ORDER),
        help="Comma-separated algorithm ids.",
    )
    parser.add_argument("--output-csv", type=str, default="results/benchmark/evaluate_all.csv")
    parser.add_argument("--output-summary-json", type=str, default="results/benchmark/evaluate_all_summary.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenario-index", type=int, default=0)

    parser.add_argument("--headway-s", type=float, default=2.0)
    parser.add_argument("--min-soc-threshold", type=float, default=None)
    parser.add_argument(
        "--max-decision-steps",
        type=int,
        default=None,
        help="Optional hard cap on decision steps; default is unset (no step-based truncation).",
    )
    parser.add_argument("--max-time-s", type=float, default=None)
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
    parser.add_argument("--num-vehicles", type=int, default=2)
    parser.add_argument("--vehicle-capacity-kg", type=float, default=None)
    parser.add_argument("--battery-capacity-kwh", type=float, default=None)
    parser.add_argument("--initial-battery-kwh", type=float, default=None)
    parser.add_argument("--vehicle-speed-m-s", type=float, default=None)

    parser.add_argument("--ppo-model-path", type=str, default=None)
    parser.add_argument("--rl-deterministic", action="store_true")
    parser.add_argument("--rl-stochastic", action="store_true")

    parser.add_argument("--alns-iterations", type=int, default=40)
    parser.add_argument("--alns-pr-charge-ratio", type=float, default=0.8)
    parser.add_argument(
        "--alns-pr-fallback-charge-ratios",
        type=str,
        default="0.9,1.0",
        help="Comma-separated ALNS-PR fallback charge ratios before FR fallback.",
    )
    parser.add_argument(
        "--alns-pr-disable-fallback",
        action="store_true",
        help="Disable ALNS-PR feasibility fallback (higher charge ratio / FR).",
    )
    parser.add_argument("--alns-repair-mode", type=str, default="mixed")
    parser.add_argument("--alns-no-adaptive", action="store_true")
    parser.add_argument("--alns-verbose", action="store_true")

    parser.add_argument("--mip-time-limit-s", type=float, default=5.0)
    parser.add_argument(
        "--mip-time-limit-s-medium",
        type=float,
        default=30.0,
        help="M-scale (30-40 tasks) MIP time limit in seconds.",
    )
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument(
        "--mip-solver-name",
        type=str,
        default="ortools",
        choices=("ortools", "gurobi"),
        help="MIP backend: 'ortools' (CBC) or 'gurobi' via OR-Tools.",
    )
    parser.add_argument("--mip-scenario-mode", type=str, default="minimal", choices=("minimal", "medium"))
    parser.add_argument("--mip-rule-count", type=int, default=15)
    parser.add_argument("--mip-decision-epochs", type=int, default=3)
    parser.add_argument(
        "--mip-skip-scales",
        type=str,
        default="L,XL",
        help="Comma-separated scales to skip for MIP-Hind due to runtime budget.",
    )

    args = parser.parse_args()
    if args.rl_stochastic:
        args.rl_deterministic = False
    else:
        # Deterministic by default for benchmark repeatability.
        args.rl_deterministic = True

    algorithms = _parse_algorithms(args.algorithms)
    instances = _resolve_instances(args)

    rl_model = None
    if "rl_apc" in algorithms and args.ppo_model_path:
        # Lazy-load with per-env observation/action dimensions for better
        # cross-version compatibility of serialized spaces.
        rl_model = str(args.ppo_model_path)
    elif "rl_apc" in algorithms:
        print("[warn] rl_apc selected but --ppo-model-path not set; RL rows will be marked SKIPPED.")

    rows: List[Dict[str, Any]] = []
    total = len(instances) * len(algorithms)
    done = 0
    for idx, instance in enumerate(instances):
        for algo_pos, algo in enumerate(algorithms):
            per_run_seed = int(args.seed) + idx * 100 + algo_pos
            t0 = time.perf_counter()
            try:
                result = _evaluate_algorithm(
                    algo,
                    instance,
                    args,
                    rl_model=rl_model,
                    seed=per_run_seed,
                )
            except Exception as exc:  # noqa: BLE001
                result = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
            runtime_s = time.perf_counter() - t0
            row = _row_from_result(instance, algo, runtime_s, result)
            rows.append(row)
            done += 1
            print(
                "[{done}/{total}] {algo} {scale}-seed{seed} ({split}) status={status} runtime={runtime:.2f}s".format(
                    done=done,
                    total=total,
                    algo=ALGORITHM_LABELS[algo],
                    scale=instance.entry.scale,
                    seed=instance.entry.seed,
                    split=instance.entry.split,
                    status=row.get("status"),
                    runtime=runtime_s,
                )
            )

    out_csv = Path(args.output_csv)
    _write_csv(out_csv, rows)
    print(f"saved_csv: {out_csv} rows={len(rows)}")
    if args.output_summary_json:
        out_json = Path(args.output_summary_json)
        _write_summary_json(out_json, rows)
        print(f"saved_summary: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
