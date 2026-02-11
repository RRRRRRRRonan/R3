"""Generate fixed-task-count benchmark instances for S/M/L/XL scales.

Outputs 40 JSON files (4 scales x 10 seeds) under:
    data/instances/{scale}/

Each JSON is an ``ExperimentScenario`` payload containing:
- layout config
- episode length
- one synthesized dynamic scenario
- metadata (scale, seed, split, vehicle count, etc.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baselines.mip.config import ScenarioSynthConfig
from baselines.mip.scenario_io import (
    ExperimentScenario,
    save_experiment_json,
)
from config.benchmark_instances import (
    BENCHMARK_SCALE_ORDER,
    build_benchmark_layout,
    get_benchmark_scale,
)
from config.instance_generator import generate_warehouse_instance
from strategy.scenario_synthesizer import synthesize_scenarios


def _parse_scales(raw: str) -> List[str]:
    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("No scales provided")
    return values


def _build_synth_config(scale, *, episode_length_s: float) -> ScenarioSynthConfig:
    peak, normal, offpeak = scale.arrival_rates_per_s
    if normal <= 0.0:
        raise ValueError(f"{scale.scale} has non-positive normal arrival rate: {normal}")
    return ScenarioSynthConfig(
        episode_length_s=episode_length_s,
        num_scenarios=1,
        use_nhpp_arrivals=True,
        arrival_time_sampling_mode="fixed_count",
        nhpp_base_rate_per_s=normal,
        nhpp_peak_multiplier=peak / normal,
        nhpp_normal_multiplier=1.0,
        nhpp_offpeak_multiplier=offpeak / normal,
        use_truncnorm_demands=True,
        use_release_time_windows=True,
    )


def generate_instances(
    *,
    output_root: Path,
    scales: List[str],
    episode_length_s: float,
    overwrite: bool,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, object]] = []
    split_map: Dict[str, List[str]] = {"train": [], "test": []}

    for scale_name in scales:
        scale = get_benchmark_scale(scale_name)
        synth_cfg = _build_synth_config(scale, episode_length_s=episode_length_s)
        scale_dir = output_root / scale.scale
        scale_dir.mkdir(parents=True, exist_ok=True)

        for seed in scale.seeds():
            num_tasks = scale.sample_task_count(seed)
            split = scale.split_for_seed(seed)
            layout = build_benchmark_layout(scale, seed=seed, num_tasks=num_tasks)
            instance = generate_warehouse_instance(layout)
            scenarios = synthesize_scenarios(
                instance.create_task_pool(),
                chargers=instance.charging_nodes,
                seed=seed,
                config=synth_cfg,
            )

            payload = ExperimentScenario(
                layout=layout,
                episode_length_s=episode_length_s,
                scenarios=scenarios,
                metadata={
                    "scale": scale.scale,
                    "seed": seed,
                    "split": split,
                    "num_vehicles": scale.num_vehicles,
                    "num_charging_stations": scale.num_charging_stations,
                    "num_tasks": num_tasks,
                    "warehouse_size_m": list(scale.warehouse_size_m),
                    "arrival_rates_per_s": list(scale.arrival_rates_per_s),
                    "mode": "fixed_task_count",
                },
            )

            out_path = scale_dir / f"{scale.scale}_seed{seed}.json"
            if out_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{out_path} already exists. Use --overwrite to regenerate."
                )
            save_experiment_json(out_path, payload)

            rel_path = str(out_path.relative_to(output_root))
            split_map[split].append(rel_path)
            manifest_entries.append(
                {
                    "scale": scale.scale,
                    "seed": seed,
                    "split": split,
                    "path": rel_path,
                    "num_tasks": num_tasks,
                    "num_vehicles": scale.num_vehicles,
                    "num_charging_stations": scale.num_charging_stations,
                }
            )

    manifest = {
        "mode": "fixed_task_count",
        "episode_length_s": episode_length_s,
        "total_instances": len(manifest_entries),
        "scales": scales,
        "entries": manifest_entries,
        "splits": split_map,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixed-count benchmark instance JSON files.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/instances",
        help="Output directory root.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=",".join(BENCHMARK_SCALE_ORDER),
        help="Comma-separated scales, e.g. S,M,L,XL.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=28_800.0,
        help="Episode horizon in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files.",
    )
    args = parser.parse_args()

    scales = _parse_scales(args.scales)
    manifest = generate_instances(
        output_root=Path(args.output_root),
        scales=scales,
        episode_length_s=float(args.episode_length_s),
        overwrite=bool(args.overwrite),
    )
    print(
        "Generated {count} instances across scales {scales}. Manifest: {path}".format(
            count=manifest["total_instances"],
            scales=",".join(scales),
            path=str(Path(args.output_root) / "manifest.json"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
