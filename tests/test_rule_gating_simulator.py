"""Regression tests for rule gating and decision-epoch generation."""

from __future__ import annotations

from types import SimpleNamespace

from core.task import TaskPool
from strategy.rule_gating import (
    RULE_ACCEPT_FEASIBLE,
    RULE_ACCEPT_VALUE,
    RULE_EDD,
    RULE_STANDBY_LOW_COST,
    get_available_rules,
)
from strategy.simulator import EVENT_ROBOT_IDLE, EVENT_TASK_ARRIVAL, Event, EventDrivenSimulator


def _state(*, has_tasks: bool, low_soc: bool = False):
    battery = 10.0 if low_soc else 100.0
    tasks = {1: object()} if has_tasks else {}
    vehicles = {
        1: SimpleNamespace(
            current_battery=battery,
            battery_capacity=100.0,
        )
    }
    return SimpleNamespace(open_tasks=tasks, robots=vehicles)


def test_accept_rules_only_enabled_on_task_arrival():
    arrival_event = Event(time=0.0, event_type=EVENT_TASK_ARRIVAL, payload={"task_id": 1})
    idle_event = Event(time=0.0, event_type=EVENT_ROBOT_IDLE, payload={"vehicle_id": 1})

    rules_on_arrival = set(get_available_rules(arrival_event, _state(has_tasks=True)))
    rules_on_idle = set(get_available_rules(idle_event, _state(has_tasks=True)))

    assert RULE_ACCEPT_FEASIBLE in rules_on_arrival
    assert RULE_ACCEPT_VALUE in rules_on_arrival
    assert RULE_EDD in rules_on_arrival

    assert RULE_ACCEPT_FEASIBLE not in rules_on_idle
    assert RULE_ACCEPT_VALUE not in rules_on_idle
    assert RULE_EDD in rules_on_idle


def test_idle_without_tasks_falls_back_to_standby():
    idle_event = Event(time=0.0, event_type=EVENT_ROBOT_IDLE, payload={"vehicle_id": 1})
    rules = set(get_available_rules(idle_event, _state(has_tasks=False)))
    assert RULE_STANDBY_LOW_COST in rules


def test_simulator_generates_synthetic_idle_event_when_queue_drains_but_work_remains():
    simulator = EventDrivenSimulator(task_pool=TaskPool(), vehicles=[])
    simulator.current_time = 123.0
    simulator._pending_task_ids = {1}
    simulator._idle_vehicle_ids = {7}

    event, _ = simulator.advance_to_next_decision_epoch()

    assert event is not None
    assert event.event_type == EVENT_ROBOT_IDLE
    assert event.payload.get("vehicle_id") == 7
    assert event.time == 123.0


def test_simulator_returns_none_when_queue_drains_without_pending_work():
    simulator = EventDrivenSimulator(task_pool=TaskPool(), vehicles=[])
    event, _ = simulator.advance_to_next_decision_epoch()
    assert event is None

