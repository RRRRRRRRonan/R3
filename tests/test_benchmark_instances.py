"""Tests for fixed-task-count benchmark scale presets."""

from __future__ import annotations

from config.benchmark_instances import (
    BENCHMARK_SCALE_ORDER,
    BENCHMARK_SCALES,
    build_benchmark_layout,
    get_benchmark_scale,
)


def test_benchmark_scales_have_expected_structure():
    assert tuple(BENCHMARK_SCALES.keys()) == BENCHMARK_SCALE_ORDER
    for name in BENCHMARK_SCALE_ORDER:
        scale = BENCHMARK_SCALES[name]
        seeds = scale.seeds()
        assert len(seeds) == 10
        assert seeds[0] == scale.seed_range[0]
        assert seeds[-1] == scale.seed_range[1]
        assert scale.task_count_range[0] <= scale.task_count_range[1]


def test_seed_split_rule_is_5_train_5_test():
    scale = get_benchmark_scale("s")
    seeds = scale.seeds()
    train = [seed for seed in seeds if scale.split_for_seed(seed) == "train"]
    test = [seed for seed in seeds if scale.split_for_seed(seed) == "test"]
    assert len(train) == 5
    assert len(test) == 5
    assert train == list(seeds[:5])
    assert test == list(seeds[5:])


def test_layout_builder_respects_scale_and_seed():
    scale = get_benchmark_scale("XL")
    layout = build_benchmark_layout(scale, seed=4001, num_tasks=90)
    assert layout.width == 250.0
    assert layout.height == 200.0
    assert layout.num_charging_stations == 6
    assert layout.num_tasks == 90
    assert layout.seed == 4001

