"""Unified warehouse layout configuration and instance generation.

This module is the *preferred* import location for layout generation going
forward. It re-exports the implementation from ``config.warehouse_layout`` to
avoid circular imports with ``config.__init__`` (which is imported widely by
core modules such as ``physics.time``).
"""

from __future__ import annotations

# NOTE: Do NOT import this module from config/__init__.py. physics.time imports
# config, and the layout generator imports physics.time; exporting it from
# config/__init__.py would create an import cycle.

from .warehouse_layout import (  # noqa: F401
    ChargingPlacement,
    DepotPosition,
    LAYOUT_PRESETS,
    PAPER_DEFAULT_LAYOUT,
    PPO_TRAINING,
    OPTIMIZATION_LARGE,
    OPTIMIZATION_MEDIUM,
    OPTIMIZATION_SMALL,
    TimeWindowMode,
    WAREHOUSE_LARGE_100,
    WAREHOUSE_LARGE_50,
    WAREHOUSE_MEDIUM_20,
    WAREHOUSE_MEDIUM_30,
    WAREHOUSE_SMALL_10,
    WAREHOUSE_SMALL_5,
    WarehouseInstance,
    WarehouseLayoutConfig,
    ZoneStrategy,
    describe_layout,
    generate_warehouse_instance,
    layouts_to_markdown_table,
)

__all__ = [
    "ChargingPlacement",
    "DepotPosition",
    "LAYOUT_PRESETS",
    "PAPER_DEFAULT_LAYOUT",
    "PPO_TRAINING",
    "OPTIMIZATION_LARGE",
    "OPTIMIZATION_MEDIUM",
    "OPTIMIZATION_SMALL",
    "TimeWindowMode",
    "WAREHOUSE_LARGE_100",
    "WAREHOUSE_LARGE_50",
    "WAREHOUSE_MEDIUM_20",
    "WAREHOUSE_MEDIUM_30",
    "WAREHOUSE_SMALL_10",
    "WAREHOUSE_SMALL_5",
    "WarehouseInstance",
    "WarehouseLayoutConfig",
    "ZoneStrategy",
    "describe_layout",
    "generate_warehouse_instance",
    "layouts_to_markdown_table",
]
