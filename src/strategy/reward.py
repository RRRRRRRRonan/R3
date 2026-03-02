"""Reward/cost utilities for rule-selection episodes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Optional

from config import DEFAULT_COST_PARAMETERS
from strategy.rules import AtomicAction
from strategy.state import EpisodeMetrics, SimulatorState

_LOW_SOC_IDLE_THRESHOLD = 0.2
_REWARD_SHAPING_ENABLED = os.getenv("RL_ENABLE_REWARD_SHAPING", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}


def _task_earliest_due(task) -> float:
    """Return the earliest (most urgent) due time across all soft time windows."""
    due_candidates = []
    for node_attr in ("pickup_node", "delivery_node"):
        node = getattr(task, node_attr, None)
        if node is None:
            continue
        tw = getattr(node, "time_window", None)
        if tw is None:
            continue
        # Only charge continuous tardiness for SOFT windows (hard windows block dispatch)
        tw_type = getattr(tw, "window_type", None)
        type_name = str(getattr(tw_type, "value", tw_type)).upper()
        if "HARD" in type_name:
            continue
        latest = getattr(tw, "latest", None)
        if latest is not None:
            due_candidates.append(float(latest))
    return min(due_candidates) if due_candidates else float("inf")


def _continuous_tardiness_cost(prev_state, dt: float, cost_params) -> float:
    """Per-step tardiness for tasks whose soft due-date has already passed.

    This distributes the delay penalty continuously (instead of as a lump sum
    at dispatch time), which dramatically shortens the credit-assignment chain.
    A scaling factor of 0.5 is applied so the per-step shaping complements rather
    than overwhelms the existing dispatch-time delta_delay signal.
    """
    if prev_state is None or dt <= 0.0 or not prev_state.open_tasks:
        return 0.0
    t = prev_state.t
    scale = getattr(cost_params, "C_tardiness_shaping_scale", 0.5)
    coeff = getattr(cost_params, "C_delay", 2.0) * scale
    total = 0.0
    for task in prev_state.open_tasks.values():
        due = _task_earliest_due(task)
        if due < float("inf") and t > due:
            total += coeff * dt
    return total


@dataclass(frozen=True)
class CostBreakdown:
    travel: float
    time: float
    charging: float
    tardiness: float
    waiting: float
    conflict_wait: float
    rejection: float
    infeasible: float
    standby: float
    total: float


def compute_delta_cost(
    prev_state: Optional[SimulatorState],
    new_state: Optional[SimulatorState],
    action: Optional[AtomicAction],
    dt: float,
    *,
    cost_params=DEFAULT_COST_PARAMETERS,
    prev_metrics: Optional[EpisodeMetrics] = None,
    new_metrics: Optional[EpisodeMetrics] = None,
) -> CostBreakdown:
    """Compute cost increment based on metric deltas."""

    if prev_metrics is None:
        prev = prev_state.metrics if prev_state and prev_state.metrics else EpisodeMetrics()
    else:
        prev = prev_metrics
    if new_metrics is None:
        new = new_state.metrics if new_state and new_state.metrics else EpisodeMetrics()
    else:
        new = new_metrics

    delta_distance = max(0.0, new.total_distance - prev.total_distance)
    delta_time = max(0.0, new.total_travel_time - prev.total_travel_time)
    delta_charging = max(0.0, new.total_charging - prev.total_charging)
    delta_delay = max(0.0, new.total_delay - prev.total_delay)
    weighted_waiting_delta = max(0.0, new.total_waiting_weighted - prev.total_waiting_weighted)
    raw_waiting_delta = max(0.0, new.total_waiting - prev.total_waiting)
    # Backward compatibility: older paths may only populate raw waiting metrics.
    delta_waiting = weighted_waiting_delta if weighted_waiting_delta > 0.0 else raw_waiting_delta
    delta_conflict = max(0.0, new.total_conflict_waiting - prev.total_conflict_waiting)
    delta_standby = max(0.0, new.total_standby - prev.total_standby)
    delta_rejected = max(0, new.rejected_tasks - prev.rejected_tasks)

    travel_cost = cost_params.C_tr * delta_distance
    time_cost = cost_params.C_time * delta_time
    charging_cost = cost_params.C_ch * delta_charging
    tardiness_cost = cost_params.C_delay * delta_delay
    waiting_cost = cost_params.C_wait * delta_waiting
    conflict_cost = cost_params.C_conflict * delta_conflict
    standby_cost = cost_params.C_standby * delta_standby
    rejection_cost = cost_params.C_missing_task * delta_rejected

    # Fix-1: continuous tardiness for tasks already past their soft due-date.
    # This reshapes credit assignment: the agent feels tardiness as it accumulates,
    # not as a lump-sum spike when the overdue task is finally dispatched.
    if _REWARD_SHAPING_ENABLED and prev_state is not None and dt > 0.0:
        tardiness_cost += _continuous_tardiness_cost(prev_state, dt, cost_params)

    # Round-2 shaping:
    # 1) penalize standby when backlog exists / SOC is low;
    # 2) penalize zero-progress operational actions under backlog.
    if _REWARD_SHAPING_ENABLED and prev_state is not None and action is not None:
        if action.kind == "DWELL" and dt > 0.0:
            backlog = len(prev_state.open_tasks)
            if backlog > 0:
                standby_cost += getattr(cost_params, "C_idle_backlog", 0.0) * dt * backlog
            low_soc_exists = any(
                vehicle.battery_capacity > 0
                and (vehicle.current_battery / vehicle.battery_capacity) <= _LOW_SOC_IDLE_THRESHOLD
                for vehicle in prev_state.robots.values()
            )
            if low_soc_exists:
                standby_cost += getattr(cost_params, "C_low_soc_idle", 0.0) * dt
        if (
            dt <= 1e-9
            and prev_state.open_tasks
            and action.kind in {"DISPATCH", "CHARGE", "DWELL"}
        ):
            waiting_cost += getattr(cost_params, "C_no_progress", 0.0)

    # P2-A: small obligation cost on ACCEPT to break "accept=free" symmetry.
    if (
        _REWARD_SHAPING_ENABLED
        and action is not None
        and action.kind == "ACCEPT"
        and int(action.payload.get("accept", 0)) == 1
    ):
        waiting_cost += getattr(cost_params, "C_accept_obligation", 0.0)

    # Hard constraints are handled by masking/shielding, not by reward penalties.
    infeasible_cost = 0.0
    total = (
        travel_cost
        + time_cost
        + charging_cost
        + tardiness_cost
        + waiting_cost
        + conflict_cost
        + standby_cost
        + rejection_cost
        + infeasible_cost
    )

    return CostBreakdown(
        travel=travel_cost,
        time=time_cost,
        charging=charging_cost,
        tardiness=tardiness_cost,
        waiting=waiting_cost,
        conflict_wait=conflict_cost,
        rejection=rejection_cost,
        infeasible=infeasible_cost,
        standby=standby_cost,
        total=total,
    )


def snapshot_metrics(metrics: EpisodeMetrics) -> EpisodeMetrics:
    """Create a detached copy of episode metrics."""
    return EpisodeMetrics(
        total_distance=metrics.total_distance,
        total_travel_time=metrics.total_travel_time,
        total_charging=metrics.total_charging,
        total_delay=metrics.total_delay,
        total_waiting=metrics.total_waiting,
        total_waiting_weighted=metrics.total_waiting_weighted,
        total_conflict_waiting=metrics.total_conflict_waiting,
        total_standby=metrics.total_standby,
        rejected_tasks=metrics.rejected_tasks,
        infeasible_actions=metrics.infeasible_actions,
        mask_total=metrics.mask_total,
        mask_blocked=metrics.mask_blocked,
        mask_fallbacks=metrics.mask_fallbacks,
    )


def to_info_dict(cost: CostBreakdown) -> Dict[str, float]:
    return {
        "travel": cost.travel,
        "time": cost.time,
        "charging": cost.charging,
        "tardiness": cost.tardiness,
        "waiting": cost.waiting,
        "conflict_wait": cost.conflict_wait,
        "rejection": cost.rejection,
        "infeasible": cost.infeasible,
        "standby": cost.standby,
        "total": cost.total,
    }


__all__ = ["CostBreakdown", "compute_delta_cost", "snapshot_metrics", "to_info_dict"]
