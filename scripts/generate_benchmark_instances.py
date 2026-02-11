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
from typing import Dict, List, Tuple

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
from config import DEFAULT_VEHICLE_DEFAULTS
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


def _arrival_count_semantics(mode: str) -> str:
    key = str(mode).strip().lower()
    if key == "fixed_count":
        return "conditional_on_N_tasks"
    if key == "thinning":
        return "nhpp_thinning_with_fallback_to_fixed_count"
    raise ValueError(f"Unsupported arrival_time_sampling_mode: {mode}")


def _build_synth_config(
    scale,
    *,
    episode_length_s: float,
    arrival_time_sampling_mode: str,
    demand_max_kg: float | None,
    demand_max_ratio_of_vehicle_capacity: float,
) -> Tuple[ScenarioSynthConfig, Dict[str, float]]:
    peak, normal, offpeak = scale.arrival_rates_per_s
    if normal <= 0.0:
        raise ValueError(f"{scale.scale} has non-positive normal arrival rate: {normal}")

    mode = str(arrival_time_sampling_mode).strip().lower()
    if mode not in {"fixed_count", "thinning"}:
        raise ValueError(
            "Unsupported arrival_time_sampling_mode '{}'; expected 'fixed_count' or 'thinning'".format(
                arrival_time_sampling_mode
            )
        )

    ratio = float(demand_max_ratio_of_vehicle_capacity)
    if ratio <= 0.0:
        raise ValueError(
            "demand_max_ratio_of_vehicle_capacity must be > 0, got {}".format(
                demand_max_ratio_of_vehicle_capacity
            )
        )

    vehicle_capacity = float(DEFAULT_VEHICLE_DEFAULTS.capacity_kg)
    configured_upper = float(scale.demand_max_kg if demand_max_kg is None else demand_max_kg)
    if configured_upper <= 0.0:
        raise ValueError(f"demand_max_kg must be > 0, got {configured_upper}")
    ratio_cap = vehicle_capacity * ratio
    effective_upper = min(configured_upper, ratio_cap, vehicle_capacity)
    if effective_upper < float(scale.demand_min_kg):
        raise ValueError(
            "Effective demand upper bound {} is below demand_min_kg {}".format(
                effective_upper,
                scale.demand_min_kg,
            )
        )

    cfg = ScenarioSynthConfig(
        episode_length_s=episode_length_s,
        num_scenarios=1,
        use_nhpp_arrivals=True,
        arrival_time_sampling_mode=mode,
        nhpp_base_rate_per_s=normal,
        nhpp_peak_multiplier=peak / normal,
        nhpp_normal_multiplier=1.0,
        nhpp_offpeak_multiplier=offpeak / normal,
        use_truncnorm_demands=True,
        demand_mean_kg=float(scale.demand_mean_kg),
        demand_std_kg=float(scale.demand_std_kg),
        demand_min_kg=float(scale.demand_min_kg),
        demand_max_kg=float(effective_upper),
        use_release_time_windows=True,
    )
    metadata = {
        "vehicle_capacity_kg": vehicle_capacity,
        "demand_max_kg_effective": float(effective_upper),
        "demand_max_ratio_of_vehicle_capacity_effective": (
            float(effective_upper) / vehicle_capacity if vehicle_capacity > 0 else 0.0
        ),
    }
    return cfg, metadata


def generate_instances(
    *,
    output_root: Path,
    scales: List[str],
    episode_length_s: float,
    overwrite: bool,
    arrival_time_sampling_mode: str = "fixed_count",
    demand_max_kg: float | None = None,
    demand_max_ratio_of_vehicle_capacity: float = 1.0,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, object]] = []
    split_map: Dict[str, List[str]] = {"train": [], "test": []}
    per_scale_demand_caps: Dict[str, float] = {}

    for scale_name in scales:
        scale = get_benchmark_scale(scale_name)
        synth_cfg, synth_meta = _build_synth_config(
            scale,
            episode_length_s=episode_length_s,
            arrival_time_sampling_mode=arrival_time_sampling_mode,
            demand_max_kg=demand_max_kg,
            demand_max_ratio_of_vehicle_capacity=demand_max_ratio_of_vehicle_capacity,
        )
        per_scale_demand_caps[scale.scale] = float(synth_meta["demand_max_kg_effective"])
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
                    "arrival_time_sampling_mode": synth_cfg.arrival_time_sampling_mode,
                    "arrival_count_semantics": _arrival_count_semantics(
                        synth_cfg.arrival_time_sampling_mode
                    ),
                    "demand_model": "truncnorm",
                    "demand_params_kg": {
                        "mean": synth_cfg.demand_mean_kg,
                        "std": synth_cfg.demand_std_kg,
                        "min": synth_cfg.demand_min_kg,
                        "max": synth_cfg.demand_max_kg,
                    },
                    "vehicle_capacity_kg": synth_meta["vehicle_capacity_kg"],
                    "demand_max_ratio_of_vehicle_capacity_effective": synth_meta[
                        "demand_max_ratio_of_vehicle_capacity_effective"
                    ],
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
                    "arrival_time_sampling_mode": synth_cfg.arrival_time_sampling_mode,
                    "demand_max_kg_effective": synth_cfg.demand_max_kg,
                }
            )

    manifest = {
        "mode": "fixed_task_count",
        "episode_length_s": episode_length_s,
        "arrival_time_sampling_mode": str(arrival_time_sampling_mode).strip().lower(),
        "arrival_count_semantics": _arrival_count_semantics(arrival_time_sampling_mode),
        "demand_model": "truncnorm",
        "demand_max_kg_override": (None if demand_max_kg is None else float(demand_max_kg)),
        "demand_max_ratio_of_vehicle_capacity": float(demand_max_ratio_of_vehicle_capacity),
        "effective_demand_max_kg_by_scale": per_scale_demand_caps,
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
    parser.add_argument(
        "--arrival-time-sampling-mode",
        type=str,
        default="fixed_count",
        choices=("fixed_count", "thinning"),
        help=(
            "NHPP release-time sampling mode. "
            "'fixed_count' = conditional-on-N-tasks; 'thinning' = NHPP thinning with fallback."
        ),
    )
    parser.add_argument(
        "--demand-max-kg",
        type=float,
        default=None,
        help=(
            "Optional absolute cap for truncnorm demand upper bound (kg). "
            "If omitted, uses scale default."
        ),
    )
    parser.add_argument(
        "--demand-max-ratio-of-vehicle-capacity",
        type=float,
        default=1.0,
        help=(
            "Safety cap ratio for demand upper bound relative to vehicle capacity. "
            "Final max demand is min(scale/default cap, ratio*capacity)."
        ),
    )
    args = parser.parse_args()

    scales = _parse_scales(args.scales)
    manifest = generate_instances(
        output_root=Path(args.output_root),
        scales=scales,
        episode_length_s=float(args.episode_length_s),
        overwrite=bool(args.overwrite),
        arrival_time_sampling_mode=str(args.arrival_time_sampling_mode),
        demand_max_kg=args.demand_max_kg,
        demand_max_ratio_of_vehicle_capacity=float(args.demand_max_ratio_of_vehicle_capacity),
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
