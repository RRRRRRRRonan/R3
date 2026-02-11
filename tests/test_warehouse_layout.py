"""Smoke tests for the unified warehouse layout generator.

Verifies that every preset produces a valid WarehouseInstance with
math-model-compliant node IDs, deterministic seeding, correct zone
placement, and a readable layout description.
"""

import pytest

from config.warehouse_layout import (
    WarehouseLayoutConfig,
    generate_warehouse_instance,
    describe_layout,
    layouts_to_markdown_table,
    DepotPosition,
    ZoneStrategy,
    PAPER_DEFAULT_LAYOUT,
    WAREHOUSE_SMALL_5,
    WAREHOUSE_SMALL_10,
    WAREHOUSE_MEDIUM_20,
    WAREHOUSE_MEDIUM_30,
    WAREHOUSE_LARGE_50,
    WAREHOUSE_LARGE_100,
    OPTIMIZATION_SMALL,
    OPTIMIZATION_MEDIUM,
    OPTIMIZATION_LARGE,
    PPO_TRAINING,
)

ALL_PRESETS = [
    PAPER_DEFAULT_LAYOUT,
    WAREHOUSE_SMALL_5,
    WAREHOUSE_SMALL_10,
    WAREHOUSE_MEDIUM_20,
    WAREHOUSE_MEDIUM_30,
    WAREHOUSE_LARGE_50,
    WAREHOUSE_LARGE_100,
    OPTIMIZATION_SMALL,
    OPTIMIZATION_MEDIUM,
    OPTIMIZATION_LARGE,
    PPO_TRAINING,
]


@pytest.mark.parametrize("config", ALL_PRESETS, ids=lambda c: f"n{c.num_tasks}_cs{c.num_charging_stations}")
def test_preset_produces_valid_instance(config):
    """Each preset yields an instance with correct sizes and ID ranges."""
    instance = generate_warehouse_instance(config)
    n = config.num_tasks
    m = config.num_charging_stations

    # Coordinate dict: 1 depot + 2n task nodes + m charging nodes
    assert len(instance.coordinates) == 1 + 2 * n + m

    # Depot at node 0
    assert 0 in instance.coordinates
    assert instance.depot.node_id == 0

    # Math-model node IDs
    for task in instance.tasks:
        assert 1 <= task.pickup_id <= n, f"pickup_id {task.pickup_id} not in [1, {n}]"
        assert n + 1 <= task.delivery_id <= 2 * n, f"delivery_id {task.delivery_id} not in [{n+1}, {2*n}]"

    for j, cn in enumerate(instance.charging_nodes):
        expected = 2 * n + 1 + j
        assert cn.node_id == expected, f"charging node_id {cn.node_id} != {expected}"

    # Task and charging node counts
    assert len(instance.tasks) == n
    assert len(instance.charging_nodes) == m


def test_seed_determinism():
    """Same config (same seed) produces identical instances."""
    config = PAPER_DEFAULT_LAYOUT
    inst1 = generate_warehouse_instance(config)
    inst2 = generate_warehouse_instance(config)

    assert inst1.coordinates == inst2.coordinates
    for t1, t2 in zip(inst1.tasks, inst2.tasks):
        assert t1.pickup_coordinates == t2.pickup_coordinates
        assert t1.delivery_coordinates == t2.delivery_coordinates
        assert t1.demand == t2.demand


def test_different_seed_produces_different_instance():
    """Changing the seed produces a different layout."""
    from dataclasses import replace

    config_a = replace(PAPER_DEFAULT_LAYOUT, seed=1)
    config_b = replace(PAPER_DEFAULT_LAYOUT, seed=2)
    inst_a = generate_warehouse_instance(config_a)
    inst_b = generate_warehouse_instance(config_b)

    # At least some coordinates should differ
    diffs = sum(
        1 for k in inst_a.coordinates
        if inst_a.coordinates[k] != inst_b.coordinates.get(k)
    )
    assert diffs > 0


def test_left_right_zone_separation():
    """LEFT_RIGHT pickup x-coords are in the left band, delivery in the right."""
    config = WAREHOUSE_MEDIUM_20
    instance = generate_warehouse_instance(config)
    w = config.width

    for task in instance.tasks:
        px = task.pickup_coordinates[0]
        dx = task.delivery_coordinates[0]
        # With 5% grid noise on a 100m warehouse, pickup should stay < 55m
        assert px < w * 0.55, f"Pickup x={px:.1f} outside left band"
        # Delivery should stay > 45m
        assert dx > w * 0.45, f"Delivery x={dx:.1f} outside right band"


def test_describe_layout():
    """describe_layout returns a human-readable string with key parameters."""
    desc = describe_layout(PAPER_DEFAULT_LAYOUT)
    assert "100m" in desc
    assert "80m" in desc
    assert "warehouse" in desc.lower()
    assert "center" in desc.lower()
    assert "20 tasks" in desc


def test_create_task_pool():
    """WarehouseInstance.create_task_pool builds a usable TaskPool."""
    instance = generate_warehouse_instance(WAREHOUSE_SMALL_5)
    pool = instance.create_task_pool()
    assert len(pool) == 5

    # All tasks are pending
    pending = pool.get_pending_tasks()
    assert len(pending) == 5


def test_depot_positions():
    """Different DepotPosition values produce correct coordinates."""
    from dataclasses import replace

    center = replace(PAPER_DEFAULT_LAYOUT, depot_position=DepotPosition.CENTER)
    origin = replace(PAPER_DEFAULT_LAYOUT, depot_position=DepotPosition.ORIGIN)
    custom = replace(
        PAPER_DEFAULT_LAYOUT,
        depot_position=DepotPosition.CUSTOM,
        depot_custom_xy=(7.0, 13.0),
    )

    assert generate_warehouse_instance(center).depot.coordinates == (50.0, 40.0)
    assert generate_warehouse_instance(origin).depot.coordinates == (0.0, 0.0)
    assert generate_warehouse_instance(custom).depot.coordinates == (7.0, 13.0)


def test_no_charging_stations():
    """Zero charging stations produces an empty charging list."""
    instance = generate_warehouse_instance(WAREHOUSE_SMALL_5)
    assert instance.charging_nodes == []
    assert instance.config.num_charging_stations == 0


def test_distance_matrix_lookups():
    """Distance matrix covers all generated node IDs."""
    instance = generate_warehouse_instance(OPTIMIZATION_SMALL)
    dm = instance.distance_matrix

    # Depot to first pickup
    d = dm.get_distance(0, 1)
    assert d >= 0.0

    # Self-distance is zero
    assert dm.get_distance(0, 0) == 0.0

    # Symmetry check for a pair
    n = instance.config.num_tasks
    assert abs(dm.get_distance(1, n + 1) - dm.get_distance(n + 1, 1)) < 1e-9


def test_layouts_to_markdown_table():
    """Table export is stable and includes key paper default parameters."""
    table = layouts_to_markdown_table({"paper_default": PAPER_DEFAULT_LAYOUT})
    assert "| Layout |" in table
    assert "100m x 80m" in table
    assert "center" in table.lower()
    assert "2 (corner)" in table
