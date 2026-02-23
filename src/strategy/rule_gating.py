"""Rule gating logic for event-driven RL/heuristic selection."""

from __future__ import annotations

from typing import List, Optional

from strategy.simulator import (
    EVENT_CHARGE_DONE,
    EVENT_DEADLOCK_RISK,
    EVENT_ROBOT_IDLE,
    EVENT_SOC_LOW,
    EVENT_TASK_ARRIVAL,
    Event,
)
from strategy.state import SimulatorState

RULE_STTF = 1
RULE_EDD = 2
RULE_MST = 3
RULE_HPF = 4
RULE_CHARGE_URGENT = 5
RULE_CHARGE_TARGET_LOW = 6
RULE_CHARGE_TARGET_MED = 7
RULE_CHARGE_TARGET_HIGH = 8
RULE_CHARGE_OPPORTUNITY = 9
RULE_STANDBY_LOW_COST = 10
RULE_STANDBY_LAZY = 11
RULE_STANDBY_HEATMAP = 12
RULE_ACCEPT_FEASIBLE = 13
RULE_ACCEPT_VALUE = 14
RULE_INSERT_MIN_COST = 15

# Backward-compatible alias: legacy callers mapped to the medium target level.
RULE_CHARGE_TARGET = RULE_CHARGE_TARGET_MED

DISPATCH_RULES = [RULE_STTF, RULE_EDD, RULE_MST, RULE_HPF, RULE_INSERT_MIN_COST]
CHARGE_RULES = [
    RULE_CHARGE_URGENT,
    RULE_CHARGE_TARGET_LOW,
    RULE_CHARGE_TARGET_MED,
    RULE_CHARGE_TARGET_HIGH,
    RULE_CHARGE_OPPORTUNITY,
]
STANDBY_RULES = [RULE_STANDBY_LOW_COST, RULE_STANDBY_LAZY, RULE_STANDBY_HEATMAP]
ACCEPT_RULES = [RULE_ACCEPT_FEASIBLE, RULE_ACCEPT_VALUE]


def get_available_rules(
    event: Optional[Event],
    state: SimulatorState,
    *,
    soc_threshold: float = 0.2,
) -> List[int]:
    """Return the available rule IDs for the given event and state.

    This matches the gating logic described in Step 3:
    - TASK_ARRIVAL: allow accept rules + dispatch rules
    - ROBOT_IDLE: if energy risk -> charge rules
                  elif pending tasks -> dispatch rules
                  else -> standby rules
    """

    event_type = event.event_type if event else None

    available: List[int] = []
    has_tasks = bool(state.open_tasks)
    has_energy_risk = _has_energy_risk(state, soc_threshold)

    if event_type == EVENT_TASK_ARRIVAL:
        # Accept/reject is only meaningful at arrival epochs.
        if has_tasks:
            available.extend(ACCEPT_RULES)
        # On arrival, prioritize dispatch; allow charging if energy risk is present.
        available.extend(DISPATCH_RULES)
        if has_energy_risk:
            available.extend(CHARGE_RULES)
    elif event_type in {EVENT_ROBOT_IDLE, EVENT_CHARGE_DONE, EVENT_SOC_LOW, EVENT_DEADLOCK_RISK}:
        # Idle-like events: allow charging when energy is risky, otherwise dispatch if tasks exist.
        if has_energy_risk:
            available.extend(CHARGE_RULES)
        if has_tasks:
            available.extend(DISPATCH_RULES)
        if not has_tasks:
            available.extend(STANDBY_RULES)
    else:
        # Fallback: allow dispatch if tasks exist; otherwise standby.
        if has_tasks:
            available.extend(DISPATCH_RULES)
        else:
            available.extend(STANDBY_RULES)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(available))


def _has_energy_risk(state: SimulatorState, soc_threshold: float) -> bool:
    for vehicle in state.robots.values():
        if vehicle.battery_capacity <= 0:
            continue
        soc = vehicle.current_battery / vehicle.battery_capacity
        if soc <= soc_threshold:
            return True
    return False


__all__ = [
    "get_available_rules",
    "RULE_STTF",
    "RULE_EDD",
    "RULE_MST",
    "RULE_HPF",
    "RULE_CHARGE_URGENT",
    "RULE_CHARGE_TARGET_LOW",
    "RULE_CHARGE_TARGET_MED",
    "RULE_CHARGE_TARGET_HIGH",
    "RULE_CHARGE_OPPORTUNITY",
    "RULE_STANDBY_LOW_COST",
    "RULE_STANDBY_LAZY",
    "RULE_STANDBY_HEATMAP",
    "RULE_ACCEPT_FEASIBLE",
    "RULE_ACCEPT_VALUE",
    "RULE_INSERT_MIN_COST",
]
