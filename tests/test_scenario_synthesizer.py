"""Tests for the paper-aligned demand model + scenario synthesis.

These tests cover the "Section 5.1 mapping" requirements:
- NHPP-based release times are generated (not all zero).
- Truncated-normal demands stay within configured bounds.
- Time windows are anchored to the per-task release time and are applied to the
  simulator via ``node_time_windows`` overrides.
"""

from __future__ import annotations

from dataclasses import replace

from baselines.mip.config import ScenarioSynthConfig
from core.task import TaskPool
from core.vehicle import create_vehicle
from strategy.scenario_synthesizer import synthesize_scenarios
from strategy.simulator import EventDrivenSimulator

from config.warehouse_layout import TimeWindowMode, WarehouseLayoutConfig, generate_warehouse_instance


def test_synthesizer_generates_release_times_demands_and_time_windows():
    layout = WarehouseLayoutConfig(width=50.0, height=50.0)
    # Use a small instance with no preset time windows so scenario overrides are visible.
    layout = replace(
        layout,
        num_tasks=3,
        num_charging_stations=0,
        time_window_mode=TimeWindowMode.NONE,
        seed=1,
    )
    instance = generate_warehouse_instance(layout)
    pool = TaskPool()
    pool.add_tasks(instance.tasks)

    cfg = ScenarioSynthConfig(
        num_scenarios=1,
        episode_length_s=1_000.0,
        availability_prob=1.0,
        use_nhpp_arrivals=True,
        use_truncnorm_demands=True,
        demand_min_kg=0.0,
        demand_max_kg=150.0,
        use_release_time_windows=True,
        pickup_tw_width_s=1800.0,
        delivery_tw_width_s=3600.0,
        release_jitter_s=0.0,
    )
    scenarios = synthesize_scenarios(pool, seed=123, config=cfg)
    assert len(scenarios) == 1
    scenario = scenarios[0]

    # Release times are per-task and within horizon.
    assert set(scenario.task_release_times.keys()) == {1, 2, 3}
    assert any(t > 0.0 for t in scenario.task_release_times.values())
    assert all(0.0 <= t <= cfg.episode_length_s for t in scenario.task_release_times.values())

    # Demands stay within truncation bounds.
    assert all(cfg.demand_min_kg <= d <= cfg.demand_max_kg for d in scenario.task_demands.values())

    # Time windows are anchored to release time for pickup nodes.
    for task in instance.tasks:
        release = scenario.task_release_times[task.task_id]
        pickup_tw = scenario.node_time_windows.get(task.pickup_node.node_id)
        assert pickup_tw is not None
        assert pickup_tw[0] == release
        assert pickup_tw[1] == release + cfg.pickup_tw_width_s

    # Simulator applies the time-window overrides to its task pool.
    simulator = EventDrivenSimulator(task_pool=pool, vehicles=[create_vehicle(1)], chargers=[])
    simulator.apply_scenario(
        task_availability=scenario.task_availability,
        task_release_times=scenario.task_release_times,
        task_demands=scenario.task_demands,
        node_time_windows=scenario.node_time_windows,
        node_service_times=scenario.node_service_times,
    )

    for task in pool.get_all_tasks():
        assert task.pickup_node.time_window is not None
        assert task.delivery_node.time_window is not None
