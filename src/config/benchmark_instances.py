"""Unified benchmark scale presets for fixed-task-count instance generation.

This module encodes the 4-scale benchmark table used by experiments:
S/M/L/XL, each with 10 seeds and shared train/test split rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Tuple

from config.instance_generator import (
    ChargingPlacement,
    DepotPosition,
    TimeWindowMode,
    WarehouseLayoutConfig,
    ZoneStrategy,
)


@dataclass(frozen=True)
class BenchmarkScaleConfig:
    """Benchmark scale definition."""

    scale: str
    num_vehicles: int
    num_charging_stations: int
    warehouse_size_m: Tuple[float, float]
    task_count_range: Tuple[int, int]
    seed_range: Tuple[int, int]
    # (peak, normal, off-peak) rates in events/second.
    arrival_rates_per_s: Tuple[float, float, float]
    # Truncated-normal demand model parameters (kg).
    demand_mean_kg: float = 75.0
    demand_std_kg: float = 22.5
    demand_min_kg: float = 0.0
    demand_max_kg: float = 150.0

    def seeds(self) -> Tuple[int, ...]:
        start, end = self.seed_range
        if end < start:
            raise ValueError(f"Invalid seed range for {self.scale}: {self.seed_range}")
        return tuple(range(start, end + 1))

    def sample_task_count(self, seed: int) -> int:
        """Deterministically sample task count in the configured range."""

        lo, hi = self.task_count_range
        if hi < lo:
            raise ValueError(f"Invalid task_count_range for {self.scale}: {self.task_count_range}")
        return random.Random(seed).randint(lo, hi)

    def split_for_seed(self, seed: int) -> str:
        """Shared split rule: first 5 seeds train, remaining 5 test."""

        seeds = self.seeds()
        try:
            idx = seeds.index(seed)
        except ValueError as exc:
            raise ValueError(f"Seed {seed} not in {self.scale} seed range {self.seed_range}") from exc
        return "train" if idx < 5 else "test"


BENCHMARK_SCALE_ORDER: Tuple[str, ...] = ("S", "M", "L", "XL")


BENCHMARK_SCALES: Dict[str, BenchmarkScaleConfig] = {
    "S": BenchmarkScaleConfig(
        scale="S",
        num_vehicles=3,
        num_charging_stations=2,
        warehouse_size_m=(100.0, 80.0),
        task_count_range=(15, 20),
        seed_range=(1001, 1010),
        arrival_rates_per_s=(0.003, 0.001, 0.0005),
    ),
    "M": BenchmarkScaleConfig(
        scale="M",
        num_vehicles=5,
        num_charging_stations=3,
        warehouse_size_m=(150.0, 120.0),
        task_count_range=(30, 40),
        seed_range=(2001, 2010),
        arrival_rates_per_s=(0.0045, 0.0015, 0.00075),
    ),
    "L": BenchmarkScaleConfig(
        scale="L",
        num_vehicles=8,
        num_charging_stations=4,
        warehouse_size_m=(200.0, 160.0),
        task_count_range=(50, 60),
        seed_range=(3001, 3010),
        arrival_rates_per_s=(0.006, 0.002, 0.001),
    ),
    "XL": BenchmarkScaleConfig(
        scale="XL",
        num_vehicles=12,
        num_charging_stations=6,
        warehouse_size_m=(250.0, 200.0),
        task_count_range=(80, 100),
        seed_range=(4001, 4010),
        arrival_rates_per_s=(0.009, 0.003, 0.0015),
    ),
}


def get_benchmark_scale(scale: str) -> BenchmarkScaleConfig:
    """Return benchmark scale config by name (case-insensitive)."""

    key = str(scale).strip().upper()
    try:
        return BENCHMARK_SCALES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark scale '{scale}'") from exc


def build_benchmark_layout(
    scale_config: BenchmarkScaleConfig,
    *,
    seed: int,
    num_tasks: int | None = None,
) -> WarehouseLayoutConfig:
    """Build a layout config for one benchmark instance."""

    tasks = scale_config.sample_task_count(seed) if num_tasks is None else int(num_tasks)
    width, height = scale_config.warehouse_size_m
    return WarehouseLayoutConfig(
        width=width,
        height=height,
        depot_position=DepotPosition.CENTER,
        num_tasks=tasks,
        zone_strategy=ZoneStrategy.LEFT_RIGHT,
        # Time windows are scenario-driven (release-time anchored), keep base nodes unconstrained.
        time_window_mode=TimeWindowMode.NONE,
        num_charging_stations=scale_config.num_charging_stations,
        charging_placement=ChargingPlacement.PERIMETER,
        seed=seed,
    )


__all__ = [
    "BENCHMARK_SCALE_ORDER",
    "BENCHMARK_SCALES",
    "BenchmarkScaleConfig",
    "build_benchmark_layout",
    "get_benchmark_scale",
]
