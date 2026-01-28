"""Minimal RL environment wrapper for rule-selection policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from config import DEFAULT_COST_PARAMETERS

from physics.energy import EnergyConfig
from strategy.action_mask import ALL_RULES, action_masks
from strategy.rule_gating import get_available_rules, RULE_ACCEPT_FEASIBLE, RULE_CHARGE_URGENT
from core.task import TaskStatus
from baselines.mip.config import MIPBaselineSolverConfig
from strategy.reward import compute_delta_cost, snapshot_metrics, to_info_dict
from strategy.rules import apply as apply_rule

if TYPE_CHECKING:
    from strategy.execution_layer import ExecutionLayer
from strategy.simulator import Event, EventDrivenSimulator, EVENT_ROBOT_IDLE, EVENT_TASK_ARRIVAL
from strategy.state import SimulatorState, env_get_obs


@dataclass
class StepResult:
    obs: Dict[str, List[float]]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class RuleSelectionEnv:
    """Event-driven rule-selection environment (stub for RL integration)."""

    def __init__(
        self,
        simulator: EventDrivenSimulator,
        *,
        top_k_tasks: int = 5,
        top_k_chargers: int = 3,
        soc_threshold: float = 0.2,
        energy_config: Optional[EnergyConfig] = None,
        rule_executor: Optional[
            Callable[[int, Optional[Event], SimulatorState, EventDrivenSimulator], None]
        ] = None,
        execution_layer: Optional["ExecutionLayer"] = None,
        max_decision_steps: Optional[int] = None,
        max_time_s: Optional[float] = None,
        cost_params=None,
        mip_solver_config: Optional[MIPBaselineSolverConfig] = None,
        cost_log_path: Optional[str] = None,
        cost_log_csv_path: Optional[str] = None,
        decision_log_path: Optional[str] = None,
        decision_log_csv_path: Optional[str] = None,
        task_log_path: Optional[str] = None,
        task_log_csv_path: Optional[str] = None,
        robot_log_path: Optional[str] = None,
        robot_log_csv_path: Optional[str] = None,
    ) -> None:
        self.simulator = simulator
        self.top_k_tasks = top_k_tasks
        self.top_k_chargers = top_k_chargers
        self.soc_threshold = soc_threshold
        self.energy_config = energy_config
        self.rule_executor = rule_executor
        self.execution_layer = execution_layer
        self.max_decision_steps = max_decision_steps
        self.max_time_s = max_time_s
        if cost_params is None and mip_solver_config is not None:
            cost_params = mip_solver_config.cost_params
        self.cost_params = cost_params or DEFAULT_COST_PARAMETERS
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path("results") / "logs"
        default_decision_log = base_dir / f"decision_log_{stamp}.jsonl"
        default_decision_csv = base_dir / f"decision_log_{stamp}.csv"
        default_task_log = base_dir / f"task_log_{stamp}.jsonl"
        default_task_csv = base_dir / f"task_log_{stamp}.csv"
        default_robot_log = base_dir / f"robot_log_{stamp}.jsonl"
        default_robot_csv = base_dir / f"robot_log_{stamp}.csv"

        self.cost_log_path = Path(cost_log_path) if cost_log_path else None
        self.cost_log_csv_path = Path(cost_log_csv_path) if cost_log_csv_path else None
        self.decision_log_path = Path(decision_log_path) if decision_log_path else default_decision_log
        self.decision_log_csv_path = (
            Path(decision_log_csv_path) if decision_log_csv_path else default_decision_csv
        )
        self.task_log_path = Path(task_log_path) if task_log_path else default_task_log
        self.task_log_csv_path = Path(task_log_csv_path) if task_log_csv_path else default_task_csv
        self.robot_log_path = Path(robot_log_path) if robot_log_path else default_robot_log
        self.robot_log_csv_path = Path(robot_log_csv_path) if robot_log_csv_path else default_robot_csv
        self._current_event: Optional[Event] = None
        self._current_state: Optional[SimulatorState] = None
        self._decision_steps = 0
        self._task_snapshot: Dict[int, Tuple[str, Optional[int]]] = {}

    def reset(self) -> Tuple[Dict[str, List[float]], Dict[str, object]]:
        """Reset environment and return initial observation + info."""
        self.simulator.reset()
        event, _ = self.simulator.advance_to_next_decision_epoch()
        self._current_event = event
        self._current_state = self.simulator.build_state(event=event)
        self._decision_steps = 0
        self._task_snapshot = self._snapshot_task_states()
        obs = env_get_obs(
            self._current_state,
            event,
            top_k_tasks=self.top_k_tasks,
            top_k_chargers=self.top_k_chargers,
        )
        available_rules = get_available_rules(
            self._current_event,
            self._current_state,
            soc_threshold=self.soc_threshold,
        )
        mask = action_masks(
            self._current_event,
            self._current_state,
            soc_threshold=self.soc_threshold,
            energy_config=self.energy_config,
            return_numpy=False,
        )
        info = self._build_info(
            event,
            selected_rule_id=None,
            masked=False,
            fallback_rule_id=None,
            available_rules=available_rules,
            mask=mask,
            action_index=None,
        )
        self._log_decision_event(info, cost_breakdown=None)
        self._log_robot_snapshot(self._current_state)
        return obs, info

    def action_masks(self) -> Sequence[bool]:
        """Return the action mask for the current decision epoch."""
        if self._current_state is None:
            return [True] * len(ALL_RULES)
        return action_masks(
            self._current_event,
            self._current_state,
            soc_threshold=self.soc_threshold,
            energy_config=self.energy_config,
            return_numpy=False,
        )

    def step(self, action_rule_id: int) -> StepResult:
        """Apply a rule choice, then advance to the next decision epoch."""
        if self._current_state is None:
            obs = {"vehicles": [], "tasks": [], "chargers": [], "meta": []}
            return StepResult(obs, 0.0, True, False, {"terminated_reason": "reset_required"})

        current_event = self._current_event
        current_state = self._current_state
        prev_state = current_state
        prev_metrics = snapshot_metrics(current_state.metrics)
        available_rules = get_available_rules(
            current_event,
            current_state,
            soc_threshold=self.soc_threshold,
        )
        mask = action_masks(
            current_event,
            current_state,
            soc_threshold=self.soc_threshold,
            energy_config=self.energy_config,
            return_numpy=False,
        )

        selected_rule_id, action_index = _normalize_action(action_rule_id)
        masked = False
        fallback_rule_id: Optional[int] = None

        if action_index is None or not mask[action_index]:
            masked = True
            fallback_rule_id = _choose_fallback_rule(current_event, mask)
            selected_rule_id = fallback_rule_id
            action_index = _action_index(selected_rule_id)

        if self.execution_layer is None and self.rule_executor is not None:
            self.rule_executor(selected_rule_id, current_event, current_state, self.simulator)

        atomic_action = apply_rule(selected_rule_id, current_event, current_state)
        if self.execution_layer is not None:
            self.execution_layer.execute(atomic_action, current_state, current_event)

        event, _ = self.simulator.advance_to_next_decision_epoch()
        self._current_event = event
        self._current_state = self.simulator.build_state(event=event)
        self._decision_steps += 1
        obs = env_get_obs(
            self._current_state,
            event,
            top_k_tasks=self.top_k_tasks,
            top_k_chargers=self.top_k_chargers,
        )
        dt = self._current_state.t - prev_state.t
        cost_breakdown = compute_delta_cost(
            prev_state,
            self._current_state,
            atomic_action,
            dt,
            prev_metrics=prev_metrics,
            cost_params=self.cost_params,
        )
        info = self._build_info(
            current_event,
            selected_rule_id=selected_rule_id,
            masked=masked,
            fallback_rule_id=fallback_rule_id,
            available_rules=available_rules,
            mask=mask,
            action_index=action_index,
        )
        info["atomic_action"] = {"kind": atomic_action.kind, "payload": dict(atomic_action.payload)}
        info["cost_breakdown"] = to_info_dict(cost_breakdown)
        self._log_cost(cost_breakdown, info)
        terminated, truncated, reason = self._check_termination()
        if reason is not None:
            info["terminated_reason"] = reason
        reward = -cost_breakdown.total
        self._log_decision_event(
            info,
            cost_breakdown=cost_breakdown,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        self._log_task_changes(self._current_state)
        self._log_robot_snapshot(self._current_state)
        return StepResult(obs, reward, terminated, truncated, info)

    def _build_info(
        self,
        event: Optional[Event],
        *,
        selected_rule_id: Optional[int],
        masked: bool,
        fallback_rule_id: Optional[int],
        available_rules: Optional[Sequence[int]] = None,
        mask: Optional[Sequence[bool]] = None,
        action_index: Optional[int] = None,
    ) -> Dict[str, object]:
        info: Dict[str, object] = {
            "selected_rule": selected_rule_id,
            "selected_action_index": action_index,
            "available_rules": list(available_rules) if available_rules is not None else [],
            "masked": masked,
            "fallback_rule": fallback_rule_id,
            "event_type": event.event_type if event else None,
            "event_time": event.time if event else None,
        }
        if mask is not None:
            info["mask"] = list(mask)
        return info

    def _check_termination(self) -> Tuple[bool, bool, Optional[str]]:
        if self._current_state is None:
            return True, False, "reset_required"

        if self.max_decision_steps is not None and self._decision_steps >= self.max_decision_steps:
            return False, True, "max_decision_steps"

        if self.max_time_s is not None and self._current_state.t >= self.max_time_s:
            return False, True, "max_time"

        if self._current_event is None:
            return True, False, "event_queue_empty"

        if not self._has_pending_work():
            return True, False, "all_tasks_finished"

        return False, False, None

    def _has_pending_work(self) -> bool:
        pool = self.simulator.task_pool
        for tracker in pool.trackers.values():
            if tracker.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                return True
        # No pending tasks; if all robots idle, we can terminate.
        return any(not vehicle.is_idle() for vehicle in self.simulator.vehicles)

    def _log_cost(self, cost_breakdown, info: Dict[str, object]) -> None:
        record = {
            "step": self._decision_steps,
            "time": self._current_state.t if self._current_state is not None else None,
            "event_type": info.get("event_type"),
            "selected_rule": info.get("selected_rule"),
            "masked": info.get("masked"),
            "cost_breakdown": to_info_dict(cost_breakdown),
        }
        if self.cost_log_path is not None:
            try:
                self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cost_log_path.open("a", encoding="utf-8") as handle:
                    import json

                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            except OSError:
                pass

        if self.cost_log_csv_path is not None:
            self._log_cost_csv(cost_breakdown, info)

    def _log_cost_csv(self, cost_breakdown, info: Dict[str, object]) -> None:
        if self.cost_log_csv_path is None:
            return
        row = {
            "step": self._decision_steps,
            "time": self._current_state.t if self._current_state is not None else None,
            "event_type": info.get("event_type"),
            "selected_rule": info.get("selected_rule"),
            "masked": info.get("masked"),
            **to_info_dict(cost_breakdown),
        }
        try:
            self.cost_log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.cost_log_csv_path.exists()
            write_header = not file_exists or self.cost_log_csv_path.stat().st_size == 0
            with self.cost_log_csv_path.open("a", encoding="utf-8", newline="") as handle:
                import csv

                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError:
            return

    def _log_decision_event(
        self,
        info: Dict[str, object],
        *,
        cost_breakdown=None,
        reward: Optional[float] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:
        cost_dict = (
            to_info_dict(cost_breakdown)
            if cost_breakdown is not None
            else _empty_cost_breakdown()
        )
        record = {
            "step": self._decision_steps,
            "time": self._current_state.t if self._current_state is not None else None,
            "event_type": info.get("event_type"),
            "event_time": info.get("event_time"),
            "selected_rule": info.get("selected_rule"),
            "selected_action_index": info.get("selected_action_index"),
            "available_rules": info.get("available_rules"),
            "masked": info.get("masked"),
            "fallback_rule": info.get("fallback_rule"),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "cost_breakdown": cost_dict,
        }
        if self.decision_log_path is not None:
            _append_jsonl(self.decision_log_path, record)
        if self.decision_log_csv_path is not None:
            _append_csv(self.decision_log_csv_path, _flatten_record(record))

    def _snapshot_task_states(self) -> Dict[int, Tuple[str, Optional[int]]]:
        snapshot: Dict[int, Tuple[str, Optional[int]]] = {}
        for task_id, tracker in self.simulator.task_pool.trackers.items():
            status = tracker.status.value if hasattr(tracker.status, "value") else str(tracker.status)
            snapshot[task_id] = (status, tracker.assigned_vehicle_id)
        return snapshot

    def _log_task_changes(self, state: SimulatorState) -> None:
        current = self._snapshot_task_states()
        time = state.t
        records = []
        for task_id, (status, vehicle_id) in current.items():
            prev = self._task_snapshot.get(task_id)
            if prev is None or prev != (status, vehicle_id):
                prev_status = prev[0] if prev else None
                prev_vehicle = prev[1] if prev else None
                task = self.simulator.task_pool.get_task(task_id)
                if task is None:
                    continue
                records.append(
                    {
                        "time": time,
                        "task_id": task_id,
                        "prev_status": prev_status,
                        "status": status,
                        "prev_assigned_vehicle_id": prev_vehicle,
                        "assigned_vehicle_id": vehicle_id,
                        "pickup_node_id": task.pickup_node.node_id,
                        "delivery_node_id": task.delivery_node.node_id,
                        "demand": task.demand,
                        "priority": task.priority,
                        "arrival_time": task.arrival_time,
                    }
                )
        if records:
            if self.task_log_path is not None:
                for record in records:
                    _append_jsonl(self.task_log_path, record)
            if self.task_log_csv_path is not None:
                for record in records:
                    _append_csv(self.task_log_csv_path, record)
        self._task_snapshot = current

    def _log_robot_snapshot(self, state: SimulatorState) -> None:
        records = []
        for vehicle_id, vehicle in state.robots.items():
            soc = (
                vehicle.current_battery / vehicle.battery_capacity
                if vehicle.battery_capacity > 0
                else 0.0
            )
            status = vehicle.status.value if hasattr(vehicle.status, "value") else str(vehicle.status)
            x, y = vehicle.current_location
            load_ratio = (
                vehicle.current_load / vehicle.capacity if vehicle.capacity > 0 else 0.0
            )
            idle_flag = 1 if vehicle.is_idle() else 0
            records.append(
                {
                    "time": state.t,
                    "vehicle_id": vehicle_id,
                    "status": status,
                    "x": x,
                    "y": y,
                    "soc": soc,
                    "battery": vehicle.current_battery,
                    "battery_capacity": vehicle.battery_capacity,
                    "load": vehicle.current_load,
                    "load_ratio": load_ratio,
                    "capacity": vehicle.capacity,
                    "speed": vehicle.speed,
                    "idle": idle_flag,
                    "current_time": vehicle.current_time,
                }
            )
        if self.robot_log_path is not None:
            for record in records:
                _append_jsonl(self.robot_log_path, record)
        if self.robot_log_csv_path is not None:
            for record in records:
                _append_csv(self.robot_log_csv_path, record)


def _append_jsonl(path: Path, record: Dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            import json

            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        return


def _append_csv(path: Path, row: Dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        write_header = not file_exists or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            import csv

            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        return


def _flatten_record(record: Dict[str, object]) -> Dict[str, object]:
    flat = dict(record)
    cost = record.get("cost_breakdown")
    if isinstance(cost, dict):
        flat.pop("cost_breakdown", None)
        for key, value in cost.items():
            flat[f"cost_{key}"] = value
    available = record.get("available_rules")
    if isinstance(available, list):
        flat["available_rules"] = ",".join(str(item) for item in available)
    return flat


def _empty_cost_breakdown() -> Dict[str, float]:
    return {
        "travel": 0.0,
        "time": 0.0,
        "charging": 0.0,
        "tardiness": 0.0,
        "waiting": 0.0,
        "conflict_wait": 0.0,
        "rejection": 0.0,
        "infeasible": 0.0,
        "standby": 0.0,
        "total": 0.0,
    }


def _normalize_action(action_rule_id: int) -> Tuple[int, Optional[int]]:
    if action_rule_id in ALL_RULES:
        return action_rule_id, _action_index(action_rule_id)
    if 0 <= action_rule_id < len(ALL_RULES):
        rule_id = ALL_RULES[action_rule_id]
        return rule_id, action_rule_id
    return ALL_RULES[0], None


def _action_index(rule_id: int) -> Optional[int]:
    try:
        return ALL_RULES.index(rule_id)
    except ValueError:
        return None


def _choose_fallback_rule(event: Optional[Event], mask: Sequence[bool]) -> int:
    allowed_rules = [rule_id for rule_id, allowed in zip(ALL_RULES, mask) if allowed]
    if not allowed_rules:
        return ALL_RULES[0]
    if event and event.event_type == EVENT_ROBOT_IDLE and RULE_CHARGE_URGENT in allowed_rules:
        return RULE_CHARGE_URGENT
    if event and event.event_type == EVENT_TASK_ARRIVAL and RULE_ACCEPT_FEASIBLE in allowed_rules:
        return RULE_ACCEPT_FEASIBLE
    return allowed_rules[0]


__all__ = ["RuleSelectionEnv", "StepResult"]
