"""Unified warehouse layout configuration and instance generation.

Provides a single source of truth for generating warehouse layouts with
math-model-compliant node IDs (pickup 1..n, delivery n+1..2n, charging
2n+1..2n+m), reproducible random generation, and paper-ready layout
descriptions.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.node import (
    ChargingNode,
    DepotNode,
    create_charging_node,
    create_task_node_pair,
)
from core.task import Task, TaskPool
from physics.distance import DistanceMatrix
from physics.time import TimeWindow, TimeWindowType


# ── Enums ──────────────────────────────────────────────────────────────


class DepotPosition(Enum):
    """Where to place the depot node."""

    CENTER = "center"
    ORIGIN = "origin"
    CUSTOM = "custom"


class ZoneStrategy(Enum):
    """How pickup and delivery coordinates are generated."""

    LEFT_RIGHT = "left_right"
    UNIFORM = "uniform"


class ChargingPlacement(Enum):
    """Strategy for placing charging stations in the warehouse."""

    CORNER = "corner"
    PERIMETER = "perimeter"
    DIAGONAL = "diagonal"
    UNIFORM = "uniform"
    CUSTOM = "custom"


class TimeWindowMode(Enum):
    """How time windows are assigned to pickup/delivery nodes."""

    NONE = "none"
    SEQUENTIAL = "sequential"
    WIDE_OPEN = "wide_open"


# ── Config ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WarehouseLayoutConfig:
    """Immutable specification of a warehouse layout for instance generation."""

    # Dimensions
    width: float = 100.0
    height: float = 80.0

    # Depot
    depot_position: DepotPosition = DepotPosition.CENTER
    depot_custom_xy: Optional[Tuple[float, float]] = None

    # Tasks
    num_tasks: int = 20
    zone_strategy: ZoneStrategy = ZoneStrategy.LEFT_RIGHT

    # Zone ratios for LEFT_RIGHT (fractions of width/height)
    pickup_x_range: Tuple[float, float] = (0.15, 0.45)
    pickup_y_range: Tuple[float, float] = (0.15, 0.85)
    delivery_x_range: Tuple[float, float] = (0.55, 0.85)
    delivery_y_range: Tuple[float, float] = (0.15, 0.85)
    grid_noise_ratio: float = 0.05

    # For UNIFORM
    uniform_margin: float = 0.1

    # Demand
    demand_range: Tuple[float, float] = (10.0, 40.0)
    demand_fixed: Optional[float] = None
    service_time: float = 60.0

    # Time windows
    time_window_mode: TimeWindowMode = TimeWindowMode.NONE
    time_window_type: TimeWindowType = TimeWindowType.SOFT
    tw_pickup_width: float = 120.0
    tw_delivery_gap: float = 30.0
    tw_stagger_interval: float = 45.0
    tw_horizon: float = 10_000.0

    # Charging
    num_charging_stations: int = 2
    charging_placement: ChargingPlacement = ChargingPlacement.CORNER
    charging_custom_coords: Optional[Tuple[Tuple[float, float], ...]] = None

    # Seed
    seed: int = 42


# ── Result ─────────────────────────────────────────────────────────────


@dataclass
class WarehouseInstance:
    """Generated warehouse layout with all node objects and distance data."""

    depot: DepotNode
    tasks: List[Task]
    charging_nodes: List[ChargingNode]
    distance_matrix: DistanceMatrix
    coordinates: Dict[int, Tuple[float, float]]
    config: WarehouseLayoutConfig

    def create_task_pool(self) -> TaskPool:
        """Build a fresh TaskPool from this instance's tasks."""
        pool = TaskPool()
        pool.add_tasks(self.tasks)
        return pool


# ── Generator ──────────────────────────────────────────────────────────


def generate_warehouse_instance(
    config: WarehouseLayoutConfig,
) -> WarehouseInstance:
    """Create a reproducible warehouse instance from a layout configuration."""

    rng = _random.Random(config.seed)

    n = config.num_tasks
    m = config.num_charging_stations

    # 1. Depot
    depot = _resolve_depot(config)
    coordinates: Dict[int, Tuple[float, float]] = {0: depot.coordinates}

    # 2. Task locations
    pickup_coords, delivery_coords = _generate_task_locations(config, rng)

    # 3. Demand per task
    demands = _resolve_demands(config, rng)

    # 4. Time windows per task
    pickup_tws, delivery_tws = _resolve_time_windows(config)

    # 5. Build Task objects with math-model node IDs
    tasks: List[Task] = []
    for i in range(n):
        task_id = i + 1
        pickup_id = task_id          # 1..n
        delivery_id = task_id + n    # n+1..2n

        pickup, delivery = create_task_node_pair(
            task_id=task_id,
            pickup_id=pickup_id,
            delivery_id=delivery_id,
            pickup_coords=pickup_coords[i],
            delivery_coords=delivery_coords[i],
            demand=demands[i],
            service_time=config.service_time,
            pickup_time_window=pickup_tws[i],
            delivery_time_window=delivery_tws[i],
        )
        task = Task(
            task_id=task_id,
            pickup_node=pickup,
            delivery_node=delivery,
            demand=demands[i],
        )
        tasks.append(task)
        coordinates[pickup_id] = pickup_coords[i]
        coordinates[delivery_id] = delivery_coords[i]

    # 6. Charging stations
    charging_coords = _generate_charging_locations(config, rng)
    charging_nodes: List[ChargingNode] = []
    for j, coord in enumerate(charging_coords):
        node_id = 2 * n + 1 + j   # 2n+1..2n+m
        cn = create_charging_node(node_id=node_id, coordinates=coord)
        charging_nodes.append(cn)
        coordinates[node_id] = coord

    # 7. Distance matrix
    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=n,
        num_charging_stations=m,
    )

    return WarehouseInstance(
        depot=depot,
        tasks=tasks,
        charging_nodes=charging_nodes,
        distance_matrix=distance_matrix,
        coordinates=coordinates,
        config=config,
    )


# ── Private helpers ────────────────────────────────────────────────────


def _resolve_depot(config: WarehouseLayoutConfig) -> DepotNode:
    if config.depot_position == DepotPosition.CENTER:
        return DepotNode(coordinates=(config.width / 2, config.height / 2))
    if config.depot_position == DepotPosition.ORIGIN:
        return DepotNode(coordinates=(0.0, 0.0))
    if config.depot_position == DepotPosition.CUSTOM:
        if config.depot_custom_xy is None:
            raise ValueError(
                "depot_custom_xy is required when depot_position is CUSTOM"
            )
        return DepotNode(coordinates=config.depot_custom_xy)
    raise ValueError(f"Unknown depot_position: {config.depot_position}")


def _generate_task_locations(
    config: WarehouseLayoutConfig,
    rng: _random.Random,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if config.zone_strategy == ZoneStrategy.LEFT_RIGHT:
        return _generate_left_right(config, rng)
    if config.zone_strategy == ZoneStrategy.UNIFORM:
        return _generate_uniform(config, rng)
    raise ValueError(f"Unknown zone_strategy: {config.zone_strategy}")


def _generate_left_right(
    config: WarehouseLayoutConfig,
    rng: _random.Random,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Grid-based layout with pickup left, delivery right, plus noise."""

    n = config.num_tasks
    w, h = config.width, config.height
    noise = config.grid_noise_ratio

    grid_size = int(n ** 0.5) + 1
    px_lo, px_hi = config.pickup_x_range
    py_lo, py_hi = config.pickup_y_range
    dx_lo, dx_hi = config.delivery_x_range
    dy_lo, dy_hi = config.delivery_y_range

    pickups: List[Tuple[float, float]] = []
    for i in range(n):
        row = i // grid_size
        col = i % grid_size
        if grid_size > 1:
            x = w * px_lo + col * w * (px_hi - px_lo) / max(1, grid_size - 1)
            y = h * py_lo + row * h * (py_hi - py_lo) / max(1, grid_size - 1)
        else:
            x = w * (px_lo + px_hi) / 2
            y = h * (py_lo + py_hi) / 2
        x += rng.uniform(-w * noise, w * noise)
        y += rng.uniform(-h * noise, h * noise)
        pickups.append((x, y))

    deliveries: List[Tuple[float, float]] = []
    for i in range(n):
        row = i // grid_size
        col = i % grid_size
        if grid_size > 1:
            x = w * dx_lo + col * w * (dx_hi - dx_lo) / max(1, grid_size - 1)
            y = h * dy_lo + row * h * (dy_hi - dy_lo) / max(1, grid_size - 1)
        else:
            x = w * (dx_lo + dx_hi) / 2
            y = h * (dy_lo + dy_hi) / 2
        x += rng.uniform(-w * noise, w * noise)
        y += rng.uniform(-h * noise, h * noise)
        deliveries.append((x, y))

    return pickups, deliveries


def _generate_uniform(
    config: WarehouseLayoutConfig,
    rng: _random.Random,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Uniform random layout with configurable margin."""

    n = config.num_tasks
    w, h = config.width, config.height
    margin = config.uniform_margin

    pickups: List[Tuple[float, float]] = []
    deliveries: List[Tuple[float, float]] = []
    # Interleaved generation to match the per-task loop order used by
    # the old build_scenario / build_synthetic_tasks functions so that
    # the same seed produces comparable coordinate sequences.
    for _ in range(n):
        pickups.append((
            rng.uniform(w * margin, w * (1 - margin)),
            rng.uniform(h * margin, h * (1 - margin)),
        ))
        deliveries.append((
            rng.uniform(w * margin, w * (1 - margin)),
            rng.uniform(h * margin, h * (1 - margin)),
        ))
    return pickups, deliveries


def _generate_charging_locations(
    config: WarehouseLayoutConfig,
    rng: _random.Random,
) -> List[Tuple[float, float]]:
    m = config.num_charging_stations
    if m == 0:
        return []

    w, h = config.width, config.height
    placement = config.charging_placement

    if placement == ChargingPlacement.CORNER:
        positions = [
            (w * 0.15, h * 0.15),
            (w * 0.85, h * 0.15),
            (w * 0.15, h * 0.85),
            (w * 0.85, h * 0.85),
        ]
        return positions[:m]

    if placement == ChargingPlacement.DIAGONAL:
        coords: List[Tuple[float, float]] = []
        for idx in range(m):
            offset_x = (idx + 1) / (m + 1)
            coords.append((w * offset_x, h * (0.3 + 0.4 * offset_x)))
        return coords

    if placement == ChargingPlacement.PERIMETER:
        coords = []
        perimeter = 2 * (w + h)
        for idx in range(m):
            d = perimeter * (idx + 1) / (m + 1)
            if d <= w:
                coords.append((d, 0.0))
            elif d <= w + h:
                coords.append((w, d - w))
            elif d <= 2 * w + h:
                coords.append((2 * w + h - d, h))
            else:
                coords.append((0.0, perimeter - d))
        return coords

    if placement == ChargingPlacement.UNIFORM:
        margin = config.uniform_margin
        return [
            (
                rng.uniform(w * margin, w * (1 - margin)),
                rng.uniform(h * margin, h * (1 - margin)),
            )
            for _ in range(m)
        ]

    if placement == ChargingPlacement.CUSTOM:
        if config.charging_custom_coords is None:
            raise ValueError(
                "charging_custom_coords is required when placement is CUSTOM"
            )
        return list(config.charging_custom_coords)[:m]

    raise ValueError(f"Unknown charging_placement: {placement}")


def _resolve_demands(
    config: WarehouseLayoutConfig,
    rng: _random.Random,
) -> List[float]:
    if config.demand_fixed is not None:
        return [config.demand_fixed] * config.num_tasks
    lo, hi = config.demand_range
    return [rng.uniform(lo, hi) for _ in range(config.num_tasks)]


def _resolve_time_windows(
    config: WarehouseLayoutConfig,
) -> Tuple[List[Optional[TimeWindow]], List[Optional[TimeWindow]]]:
    n = config.num_tasks
    tw_type = config.time_window_type

    if config.time_window_mode == TimeWindowMode.NONE:
        return [None] * n, [None] * n

    if config.time_window_mode == TimeWindowMode.SEQUENTIAL:
        pickup_tws: List[Optional[TimeWindow]] = []
        delivery_tws: List[Optional[TimeWindow]] = []
        for task_id in range(1, n + 1):
            p_start = task_id * config.tw_stagger_interval
            p_end = p_start + config.tw_pickup_width
            pickup_tws.append(
                TimeWindow(earliest=p_start, latest=p_end, window_type=tw_type)
            )
            d_start = p_end + config.tw_delivery_gap
            d_end = d_start + config.tw_pickup_width
            delivery_tws.append(
                TimeWindow(earliest=d_start, latest=d_end, window_type=tw_type)
            )
        return pickup_tws, delivery_tws

    if config.time_window_mode == TimeWindowMode.WIDE_OPEN:
        tw = TimeWindow(earliest=0.0, latest=config.tw_horizon, window_type=tw_type)
        return [tw] * n, [tw] * n

    raise ValueError(f"Unknown time_window_mode: {config.time_window_mode}")


# ── Description ────────────────────────────────────────────────────────


def describe_layout(config: WarehouseLayoutConfig) -> str:
    """Return a paper-ready description of the layout configuration."""

    depot_desc = {
        DepotPosition.CENTER: "depot at center",
        DepotPosition.ORIGIN: "depot at origin",
        DepotPosition.CUSTOM: f"depot at {config.depot_custom_xy}",
    }[config.depot_position]

    if config.num_charging_stations == 0:
        cs_desc = "no charging stations"
    else:
        placement_desc = {
            ChargingPlacement.CORNER: "corner",
            ChargingPlacement.DIAGONAL: "diagonal",
            ChargingPlacement.PERIMETER: "perimeter",
            ChargingPlacement.UNIFORM: "random",
            ChargingPlacement.CUSTOM: "custom",
        }[config.charging_placement]
        suffix = "s" if config.num_charging_stations > 1 else ""
        cs_desc = (
            f"{config.num_charging_stations} charging station{suffix} "
            f"at {placement_desc} positions"
        )

    return (
        f"{config.width:.0f}m x {config.height:.0f}m warehouse, "
        f"{depot_desc}, "
        f"{config.num_tasks} tasks ({config.zone_strategy.value} zones), "
        f"{cs_desc}"
    )


def layouts_to_markdown_table(layouts: Dict[str, WarehouseLayoutConfig]) -> str:
    """Render a Markdown table describing one or more layouts.

    This is intended for paper Section 5.1 style tables so the reported layout
    parameters can be derived directly from code defaults/presets.
    """

    header = (
        "| Layout | Warehouse (W x H) | Depot | Tasks | Task placement | Charging stations | Seed |"
    )
    sep = "|:--|:--|:--|--:|:--|:--|--:|"
    rows: List[str] = []

    for name, cfg in layouts.items():
        rows.append(
            "| {name} | {wh} | {depot} | {tasks} | {task_place} | {charging} | {seed} |".format(
                name=name,
                wh=f"{cfg.width:.0f}m x {cfg.height:.0f}m",
                depot=_depot_label(cfg),
                tasks=str(int(cfg.num_tasks)),
                task_place=_task_placement_label(cfg),
                charging=_charging_label(cfg),
                seed=str(int(cfg.seed)),
            )
        )

    return "\n".join([header, sep, *rows])


def _depot_label(config: WarehouseLayoutConfig) -> str:
    if config.depot_position == DepotPosition.CENTER:
        return "center"
    if config.depot_position == DepotPosition.ORIGIN:
        return "origin"
    if config.depot_position == DepotPosition.CUSTOM:
        return f"custom {config.depot_custom_xy}"
    return config.depot_position.value


def _task_placement_label(config: WarehouseLayoutConfig) -> str:
    if config.zone_strategy == ZoneStrategy.LEFT_RIGHT:
        px_lo, px_hi = config.pickup_x_range
        py_lo, py_hi = config.pickup_y_range
        dx_lo, dx_hi = config.delivery_x_range
        dy_lo, dy_hi = config.delivery_y_range
        return (
            "pickup x[{px_lo:.2f},{px_hi:.2f}]W y[{py_lo:.2f},{py_hi:.2f}]H; "
            "delivery x[{dx_lo:.2f},{dx_hi:.2f}]W y[{dy_lo:.2f},{dy_hi:.2f}]H; "
            "grid noise {noise:.2f}"
        ).format(
            px_lo=px_lo,
            px_hi=px_hi,
            py_lo=py_lo,
            py_hi=py_hi,
            dx_lo=dx_lo,
            dx_hi=dx_hi,
            dy_lo=dy_lo,
            dy_hi=dy_hi,
            noise=config.grid_noise_ratio,
        )
    if config.zone_strategy == ZoneStrategy.UNIFORM:
        return f"uniform (margin {config.uniform_margin:.2f})"
    return config.zone_strategy.value


def _charging_label(config: WarehouseLayoutConfig) -> str:
    if config.num_charging_stations <= 0:
        return "0"
    return f"{config.num_charging_stations} ({config.charging_placement.value})"


# ── Presets ────────────────────────────────────────────────────────────


PAPER_DEFAULT_LAYOUT = WarehouseLayoutConfig(
    width=100.0,
    height=80.0,
    depot_position=DepotPosition.CENTER,
    num_tasks=20,
    zone_strategy=ZoneStrategy.LEFT_RIGHT,
    num_charging_stations=2,
    charging_placement=ChargingPlacement.CORNER,
    seed=42,
)

# ── Warehouse regression presets ───────────────────────────────────────

WAREHOUSE_SMALL_5 = WarehouseLayoutConfig(
    width=50.0,
    height=50.0,
    depot_position=DepotPosition.CENTER,
    num_tasks=5,
    zone_strategy=ZoneStrategy.LEFT_RIGHT,
    demand_range=(10.0, 25.0),
    num_charging_stations=0,
    seed=42,
)

WAREHOUSE_SMALL_10 = WarehouseLayoutConfig(
    width=60.0,
    height=60.0,
    depot_position=DepotPosition.CENTER,
    num_tasks=10,
    zone_strategy=ZoneStrategy.LEFT_RIGHT,
    demand_range=(15.0, 30.0),
    num_charging_stations=1,
    charging_placement=ChargingPlacement.CORNER,
    seed=42,
)

WAREHOUSE_MEDIUM_20 = WarehouseLayoutConfig(
    width=100.0,
    height=100.0,
    depot_position=DepotPosition.CENTER,
    num_tasks=20,
    zone_strategy=ZoneStrategy.LEFT_RIGHT,
    demand_range=(20.0, 40.0),
    num_charging_stations=2,
    charging_placement=ChargingPlacement.CORNER,
    seed=42,
)

WAREHOUSE_MEDIUM_30 = WarehouseLayoutConfig(
    width=120.0,
    height=120.0,
    depot_position=DepotPosition.CENTER,
    num_tasks=30,
    zone_strategy=ZoneStrategy.LEFT_RIGHT,
    demand_range=(15.0, 35.0),
    num_charging_stations=2,
    charging_placement=ChargingPlacement.CORNER,
    seed=42,
)

WAREHOUSE_LARGE_50 = WarehouseLayoutConfig(
    width=150.0,
    height=150.0,
    depot_position=DepotPosition.CENTER,
    num_tasks=50,
    zone_strategy=ZoneStrategy.LEFT_RIGHT,
    demand_range=(20.0, 50.0),
    num_charging_stations=3,
    charging_placement=ChargingPlacement.CORNER,
    seed=42,
)

WAREHOUSE_LARGE_100 = WarehouseLayoutConfig(
    width=200.0,
    height=200.0,
    depot_position=DepotPosition.CENTER,
    num_tasks=100,
    zone_strategy=ZoneStrategy.LEFT_RIGHT,
    demand_range=(15.0, 45.0),
    num_charging_stations=4,
    charging_placement=ChargingPlacement.CORNER,
    seed=42,
)

# ── Optimization test presets ──────────────────────────────────────────

OPTIMIZATION_SMALL = WarehouseLayoutConfig(
    width=1000.0,
    height=1000.0,
    depot_position=DepotPosition.ORIGIN,
    num_tasks=10,
    zone_strategy=ZoneStrategy.UNIFORM,
    uniform_margin=0.1,
    service_time=45.0,
    time_window_mode=TimeWindowMode.SEQUENTIAL,
    tw_pickup_width=120.0,
    tw_delivery_gap=30.0,
    tw_stagger_interval=45.0,
    num_charging_stations=1,
    charging_placement=ChargingPlacement.DIAGONAL,
    seed=7,
)

OPTIMIZATION_MEDIUM = WarehouseLayoutConfig(
    width=2000.0,
    height=2000.0,
    depot_position=DepotPosition.ORIGIN,
    num_tasks=30,
    zone_strategy=ZoneStrategy.UNIFORM,
    uniform_margin=0.1,
    service_time=45.0,
    time_window_mode=TimeWindowMode.SEQUENTIAL,
    tw_pickup_width=120.0,
    tw_delivery_gap=30.0,
    tw_stagger_interval=45.0,
    num_charging_stations=2,
    charging_placement=ChargingPlacement.DIAGONAL,
    seed=11,
)

OPTIMIZATION_LARGE = WarehouseLayoutConfig(
    width=3000.0,
    height=3000.0,
    depot_position=DepotPosition.ORIGIN,
    num_tasks=50,
    zone_strategy=ZoneStrategy.UNIFORM,
    uniform_margin=0.1,
    service_time=45.0,
    time_window_mode=TimeWindowMode.SEQUENTIAL,
    tw_pickup_width=120.0,
    tw_delivery_gap=30.0,
    tw_stagger_interval=45.0,
    num_charging_stations=3,
    charging_placement=ChargingPlacement.DIAGONAL,
    seed=17,
)

# ── PPO training preset ───────────────────────────────────────────────

PPO_TRAINING = WarehouseLayoutConfig(
    width=50.0,
    height=50.0,
    depot_position=DepotPosition.ORIGIN,
    num_tasks=8,
    zone_strategy=ZoneStrategy.UNIFORM,
    uniform_margin=0.1,
    demand_fixed=5.0,
    service_time=10.0,
    time_window_mode=TimeWindowMode.WIDE_OPEN,
    tw_horizon=10_000.0,
    num_charging_stations=2,
    charging_placement=ChargingPlacement.DIAGONAL,
    seed=42,
)


# Unified preset registry used by tests/scripts/docs.
LAYOUT_PRESETS: Dict[str, WarehouseLayoutConfig] = {
    "paper_default": PAPER_DEFAULT_LAYOUT,
    "warehouse_small_5": WAREHOUSE_SMALL_5,
    "warehouse_small_10": WAREHOUSE_SMALL_10,
    "warehouse_medium_20": WAREHOUSE_MEDIUM_20,
    "warehouse_medium_30": WAREHOUSE_MEDIUM_30,
    "warehouse_large_50": WAREHOUSE_LARGE_50,
    "warehouse_large_100": WAREHOUSE_LARGE_100,
    "optimization_small": OPTIMIZATION_SMALL,
    "optimization_medium": OPTIMIZATION_MEDIUM,
    "optimization_large": OPTIMIZATION_LARGE,
    "ppo_training": PPO_TRAINING,
}
