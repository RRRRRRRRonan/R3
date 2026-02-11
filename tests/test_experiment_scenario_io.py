"""Tests for unified experiment scenario serialization."""

from __future__ import annotations

from baselines.mip.model import MIPBaselineScenario
from baselines.mip.scenario_io import (
    ExperimentScenario,
    layout_config_from_dict,
    layout_config_to_dict,
    load_experiment_json,
    save_experiment_json,
)
from config.warehouse_layout import (
    ChargingPlacement,
    DepotPosition,
    TimeWindowMode,
    WarehouseLayoutConfig,
    ZoneStrategy,
)
from physics.time import TimeWindowType


def _build_layout() -> WarehouseLayoutConfig:
    return WarehouseLayoutConfig(
        width=123.0,
        height=77.0,
        depot_position=DepotPosition.CUSTOM,
        depot_custom_xy=(9.0, 11.0),
        num_tasks=2,
        zone_strategy=ZoneStrategy.UNIFORM,
        uniform_margin=0.2,
        demand_range=(5.0, 25.0),
        time_window_mode=TimeWindowMode.SEQUENTIAL,
        time_window_type=TimeWindowType.HARD,
        num_charging_stations=2,
        charging_placement=ChargingPlacement.CUSTOM,
        charging_custom_coords=((3.0, 4.0), (7.0, 8.0)),
        seed=314,
    )


def _build_scenario() -> MIPBaselineScenario:
    return MIPBaselineScenario(
        scenario_id=0,
        probability=1.0,
        task_availability={1: 1, 2: 0},
        task_release_times={1: 120.0, 2: 540.0},
        node_time_windows={1: (120.0, 1920.0), 3: (180.0, 3780.0)},
        node_service_times={1: 60.0, 3: 60.0},
        task_demands={1: 80.0, 2: 65.0},
        queue_estimates_s={9: 15.0},
        charging_availability={9: 1},
        decision_epoch_times=[0.0, 120.0],
    )


def test_layout_config_dict_round_trip_preserves_enums_and_tuples():
    layout = _build_layout()
    data = layout_config_to_dict(layout)
    restored = layout_config_from_dict(data)
    assert restored == layout


def test_experiment_json_round_trip(tmp_path):
    layout = _build_layout()
    experiment = ExperimentScenario(
        layout=layout,
        episode_length_s=28_800.0,
        scenarios=[_build_scenario()],
        metadata={"scale": "S", "seed": 1001, "split": "train", "num_vehicles": 3},
    )
    path = tmp_path / "sample_experiment.json"
    save_experiment_json(path, experiment)
    loaded = load_experiment_json(path)

    assert loaded.layout == experiment.layout
    assert loaded.episode_length_s == experiment.episode_length_s
    assert loaded.metadata == experiment.metadata
    assert loaded.scenarios == experiment.scenarios
