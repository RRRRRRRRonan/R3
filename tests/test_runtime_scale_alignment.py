"""Regression checks for benchmark runtime scale alignment."""

from __future__ import annotations

from baselines.mip.config import MIPBaselineScale
from config.benchmark_instances import BENCHMARK_SCALES
from config.defaults import OPTIMIZATION_SCENARIO_PRESETS
from tests.optimization.presets import ALNS_TEST_PRESETS


def test_defaults_include_xlarge_preset():
    assert "xlarge" in OPTIMIZATION_SCENARIO_PRESETS
    preset = OPTIMIZATION_SCENARIO_PRESETS["xlarge"]
    assert preset.num_tasks >= 80
    assert preset.num_charging >= 4


def test_optimization_test_presets_include_xlarge():
    assert "xlarge" in ALNS_TEST_PRESETS
    preset = ALNS_TEST_PRESETS["xlarge"]
    assert int(preset.scenario_overrides["num_tasks"]) >= 80
    assert int(preset.scenario_overrides["num_charging"]) >= 4


def test_mip_scale_covers_benchmark_xl_limits():
    xl = BENCHMARK_SCALES["XL"]
    mip_scale = MIPBaselineScale()
    assert mip_scale.max_tasks >= xl.task_count_range[1]
    assert mip_scale.max_vehicles >= xl.num_vehicles
    assert mip_scale.max_charging_stations >= xl.num_charging_stations
