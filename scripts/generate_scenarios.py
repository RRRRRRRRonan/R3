"""Generate reproducible MIP scenarios from task pools."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from baselines.mip.model import build_minimal_instance
from baselines.mip.scenario_io import save_scenarios_json, save_scenarios_csv
from core.task import TaskPool
from strategy.scenario_bridge import build_scenario_from_task_pool, record_event_epochs, attach_epoch_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=str, default="results/scenarios/scenarios.json")
    parser.add_argument("--output-csv", type=str, default="results/scenarios/scenarios.csv")
    parser.add_argument("--epoch-mode", choices=["release", "simulate"], default="release")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    instance = build_minimal_instance()
    task_pool = TaskPool()
    for task in instance.tasks:
        task_pool.add_task(task)

    scenario = build_scenario_from_task_pool(
        task_pool,
        scenario_id=0,
        probability=1.0,
        chargers=instance.charging_stations,
    )

    if args.epoch_mode == "simulate":
        epoch_times = record_event_epochs(
            task_pool,
            instance.vehicles,
            instance.charging_stations,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        scenario = attach_epoch_times(scenario, epoch_times)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_scenarios_json(output_json, [scenario])

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    task_pairs = {
        task.task_id: (task.pickup_node.node_id, task.delivery_node.node_id)
        for task in instance.tasks
    }
    save_scenarios_csv(output_csv, [scenario], task_pairs=task_pairs)

    print(f"Wrote {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
