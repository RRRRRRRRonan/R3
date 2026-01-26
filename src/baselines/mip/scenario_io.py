"""Scenario IO helpers for the MIP baseline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from baselines.mip.model import MIPBaselineScenario


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
    "scenario_to_dict",
    "scenario_from_dict",
    "save_scenarios_json",
    "load_scenarios_json",
    "save_scenarios_jsonl",
    "load_scenarios_jsonl",
    "save_scenarios_csv",
]
