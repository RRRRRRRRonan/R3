"""Scenario IO helpers for the MIP baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from baselines.mip.model import MIPBaselineScenario
from config.warehouse_layout import (
    ChargingPlacement,
    DepotPosition,
    TimeWindowMode,
    WarehouseLayoutConfig,
    ZoneStrategy,
)
from physics.time import TimeWindowType


def scenario_to_dict(scenario: MIPBaselineScenario) -> Dict:
    data = asdict(scenario)
    # Convert tuple time windows to list for JSON.
    if data.get("node_time_windows"):
        data["node_time_windows"] = {
            str(node_id): list(window) for node_id, window in data["node_time_windows"].items()
        }
    return data


def scenario_from_dict(data: Dict) -> MIPBaselineScenario:
    node_time_windows = data.get("node_time_windows", {})
    node_time_windows = {
        int(node_id): (float(window[0]), float(window[1]))
        for node_id, window in node_time_windows.items()
    }
    return MIPBaselineScenario(
        scenario_id=int(data["scenario_id"]),
        probability=float(data.get("probability", 1.0)),
        task_availability={int(k): int(v) for k, v in data.get("task_availability", {}).items()},
        task_release_times={int(k): float(v) for k, v in data.get("task_release_times", {}).items()},
        node_time_windows=node_time_windows,
        node_service_times={int(k): float(v) for k, v in data.get("node_service_times", {}).items()},
        task_demands={int(k): float(v) for k, v in data.get("task_demands", {}).items()},
        arrival_time_shift_s=float(data.get("arrival_time_shift_s", 0.0)),
        time_window_scale=float(data.get("time_window_scale", 1.0)),
        priority_boost=int(data.get("priority_boost", 0)),
        queue_estimates_s={int(k): float(v) for k, v in data.get("queue_estimates_s", {}).items()},
        travel_time_factor=float(data.get("travel_time_factor", 1.0)),
        charging_availability={
            int(k): int(v) for k, v in data.get("charging_availability", {}).items()
        },
        decision_epoch_times=[float(t) for t in data.get("decision_epoch_times", [])],
    )


def save_scenarios_json(path: str | Path, scenarios: Iterable[MIPBaselineScenario]) -> None:
    payload = {"scenarios": [scenario_to_dict(s) for s in scenarios]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_scenarios_json(path: str | Path) -> List[MIPBaselineScenario]:
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    return [scenario_from_dict(item) for item in content.get("scenarios", [])]


def save_scenarios_jsonl(path: str | Path, scenarios: Iterable[MIPBaselineScenario]) -> None:
    lines = [json.dumps(scenario_to_dict(s), ensure_ascii=True) for s in scenarios]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_scenarios_jsonl(path: str | Path) -> List[MIPBaselineScenario]:
    scenarios: List[MIPBaselineScenario] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scenarios.append(scenario_from_dict(json.loads(line)))
    return scenarios


def save_scenarios_csv(
    path: str | Path,
    scenarios: Iterable[MIPBaselineScenario],
    *,
    task_pairs: Optional[Dict[int, Tuple[int, int]]] = None,
) -> None:
    import csv

    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        for task_id, availability in scenario.task_availability.items():
            pickup_id = None
            delivery_id = None
            if task_pairs and task_id in task_pairs:
                pickup_id, delivery_id = task_pairs[task_id]
            row = {
                "scenario_id": scenario.scenario_id,
                "probability": scenario.probability,
                "task_id": task_id,
                "availability": availability,
                "release_time": scenario.task_release_times.get(task_id, 0.0),
                "demand": scenario.task_demands.get(task_id, ""),
                "pickup_node_id": pickup_id,
                "delivery_node_id": delivery_id,
            }
            if pickup_id is not None:
                pickup_tw = scenario.node_time_windows.get(pickup_id)
                row["pickup_tw_earliest"] = pickup_tw[0] if pickup_tw else ""
                row["pickup_tw_latest"] = pickup_tw[1] if pickup_tw else ""
                row["pickup_service_time"] = scenario.node_service_times.get(pickup_id, "")
            if delivery_id is not None:
                delivery_tw = scenario.node_time_windows.get(delivery_id)
                row["delivery_tw_earliest"] = delivery_tw[0] if delivery_tw else ""
                row["delivery_tw_latest"] = delivery_tw[1] if delivery_tw else ""
                row["delivery_service_time"] = scenario.node_service_times.get(delivery_id, "")
            rows.append(row)

    headers = [
        "scenario_id",
        "probability",
        "task_id",
        "availability",
        "release_time",
        "demand",
        "pickup_node_id",
        "delivery_node_id",
        "pickup_tw_earliest",
        "pickup_tw_latest",
        "pickup_service_time",
        "delivery_tw_earliest",
        "delivery_tw_latest",
        "delivery_service_time",
    ]

    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = [
    "ExperimentScenario",
    "experiment_from_dict",
    "experiment_to_dict",
    "scenario_to_dict",
    "scenario_from_dict",
    "layout_config_from_dict",
    "layout_config_to_dict",
    "save_scenarios_json",
    "load_scenarios_json",
    "save_scenarios_jsonl",
    "load_scenarios_jsonl",
    "save_scenarios_csv",
    "save_experiment_json",
    "load_experiment_json",
]


def layout_config_to_dict(config: WarehouseLayoutConfig) -> Dict[str, Any]:
    """Return a JSON-serialisable representation of ``WarehouseLayoutConfig``."""

    data = asdict(config)
    data["depot_position"] = config.depot_position.value
    data["zone_strategy"] = config.zone_strategy.value
    data["charging_placement"] = config.charging_placement.value
    data["time_window_mode"] = config.time_window_mode.value
    data["time_window_type"] = config.time_window_type.value
    return data


def layout_config_from_dict(data: Dict[str, Any]) -> WarehouseLayoutConfig:
    """Parse ``WarehouseLayoutConfig`` from ``layout_config_to_dict`` output."""

    kwargs: Dict[str, Any] = {}
    for key in WarehouseLayoutConfig.__dataclass_fields__:
        if key in data:
            kwargs[key] = data[key]

    if "depot_position" in kwargs:
        kwargs["depot_position"] = DepotPosition(str(kwargs["depot_position"]))
    if "zone_strategy" in kwargs:
        kwargs["zone_strategy"] = ZoneStrategy(str(kwargs["zone_strategy"]))
    if "charging_placement" in kwargs:
        kwargs["charging_placement"] = ChargingPlacement(str(kwargs["charging_placement"]))
    if "time_window_mode" in kwargs:
        kwargs["time_window_mode"] = TimeWindowMode(str(kwargs["time_window_mode"]))
    if "time_window_type" in kwargs:
        kwargs["time_window_type"] = TimeWindowType(str(kwargs["time_window_type"]))

    # Normalise common tuple-like fields loaded as lists.
    if kwargs.get("depot_custom_xy") is not None:
        kwargs["depot_custom_xy"] = tuple(float(x) for x in kwargs["depot_custom_xy"])
    if kwargs.get("pickup_x_range") is not None:
        kwargs["pickup_x_range"] = tuple(float(x) for x in kwargs["pickup_x_range"])
    if kwargs.get("pickup_y_range") is not None:
        kwargs["pickup_y_range"] = tuple(float(x) for x in kwargs["pickup_y_range"])
    if kwargs.get("delivery_x_range") is not None:
        kwargs["delivery_x_range"] = tuple(float(x) for x in kwargs["delivery_x_range"])
    if kwargs.get("delivery_y_range") is not None:
        kwargs["delivery_y_range"] = tuple(float(x) for x in kwargs["delivery_y_range"])
    if kwargs.get("demand_range") is not None:
        kwargs["demand_range"] = tuple(float(x) for x in kwargs["demand_range"])

    if kwargs.get("charging_custom_coords") is not None:
        kwargs["charging_custom_coords"] = tuple(
            tuple(float(x) for x in pair) for pair in kwargs["charging_custom_coords"]
        )

    return WarehouseLayoutConfig(**kwargs)


@dataclass(frozen=True)
class ExperimentScenario:
    """Unified, replayable experiment bundle (layout + dynamic scenarios)."""

    layout: WarehouseLayoutConfig
    episode_length_s: float = 28_800.0
    scenarios: List[MIPBaselineScenario] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def experiment_to_dict(experiment: ExperimentScenario) -> Dict[str, Any]:
    return {
        "layout": layout_config_to_dict(experiment.layout),
        "episode_length_s": float(experiment.episode_length_s),
        "scenarios": [scenario_to_dict(s) for s in experiment.scenarios],
        "metadata": dict(experiment.metadata),
    }


def experiment_from_dict(data: Dict[str, Any]) -> ExperimentScenario:
    layout = layout_config_from_dict(data.get("layout", {}))
    scenarios = [scenario_from_dict(item) for item in data.get("scenarios", [])]
    return ExperimentScenario(
        layout=layout,
        episode_length_s=float(data.get("episode_length_s", 28_800.0)),
        scenarios=scenarios,
        metadata=dict(data.get("metadata", {})),
    )


def save_experiment_json(path: str | Path, experiment: ExperimentScenario) -> None:
    Path(path).write_text(
        json.dumps(experiment_to_dict(experiment), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def load_experiment_json(path: str | Path) -> ExperimentScenario:
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    return experiment_from_dict(content)
