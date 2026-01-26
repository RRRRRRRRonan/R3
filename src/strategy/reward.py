"""Reward/cost utilities for rule-selection episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from config import DEFAULT_COST_PARAMETERS
from strategy.rules import AtomicAction
from strategy.state import EpisodeMetrics, SimulatorState


@dataclass(frozen=True)
class CostBreakdown:
    travel: float
    charging: float
    tardiness: float
    conflict_wait: float
    rejection: float
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
    delta_charging = max(0.0, new.total_charging - prev.total_charging)
    delta_delay = max(0.0, new.total_delay - prev.total_delay)
    delta_conflict = max(0.0, new.total_conflict_waiting - prev.total_conflict_waiting)
    delta_standby = max(0.0, new.total_standby - prev.total_standby)
    delta_rejected = max(0, new.rejected_tasks - prev.rejected_tasks)

    travel_cost = cost_params.C_tr * delta_distance
    charging_cost = cost_params.C_ch * delta_charging
    tardiness_cost = cost_params.C_delay * delta_delay
    conflict_cost = cost_params.C_conflict * delta_conflict
    standby_cost = cost_params.C_standby * delta_standby
    rejection_cost = cost_params.C_missing_task * delta_rejected

    total = (
        travel_cost
        + charging_cost
        + tardiness_cost
        + conflict_cost
        + standby_cost
        + rejection_cost
    )

    return CostBreakdown(
        travel=travel_cost,
        charging=charging_cost,
        tardiness=tardiness_cost,
        conflict_wait=conflict_cost,
        rejection=rejection_cost,
        standby=standby_cost,
        total=total,
    )


def snapshot_metrics(metrics: EpisodeMetrics) -> EpisodeMetrics:
    """Create a detached copy of episode metrics."""
    return EpisodeMetrics(
        total_distance=metrics.total_distance,
        total_charging=metrics.total_charging,
        total_delay=metrics.total_delay,
        total_conflict_waiting=metrics.total_conflict_waiting,
        total_standby=metrics.total_standby,
        rejected_tasks=metrics.rejected_tasks,
        mask_total=metrics.mask_total,
        mask_blocked=metrics.mask_blocked,
        mask_fallbacks=metrics.mask_fallbacks,
    )


def to_info_dict(cost: CostBreakdown) -> Dict[str, float]:
    return {
        "travel": cost.travel,
        "charging": cost.charging,
        "tardiness": cost.tardiness,
        "conflict_wait": cost.conflict_wait,
        "rejection": cost.rejection,
        "standby": cost.standby,
        "total": cost.total,
    }


__all__ = ["CostBreakdown", "compute_delta_cost", "snapshot_metrics", "to_info_dict"]
