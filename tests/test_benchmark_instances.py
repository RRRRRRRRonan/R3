"""Tests for fixed-task-count benchmark scale presets and generation outputs."""

from __future__ import annotations

import json

from config.benchmark_instances import (
    BENCHMARK_SCALE_ORDER,
    BENCHMARK_SCALES,
    build_benchmark_layout,
    get_benchmark_scale,
)
from scripts.generate_benchmark_instances import generate_instances


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


def test_generate_instances_builds_40_json_and_manifest(tmp_path):
    manifest = generate_instances(
        output_root=tmp_path,
        scales=list(BENCHMARK_SCALE_ORDER),
        episode_length_s=28_800.0,
        overwrite=True,
    )

    assert manifest["total_instances"] == 40
    assert manifest["scales"] == list(BENCHMARK_SCALE_ORDER)
    assert manifest["arrival_time_sampling_mode"] == "fixed_count"
    assert manifest["arrival_count_semantics"] == "conditional_on_N_tasks"
    assert manifest["demand_model"] == "truncnorm"
    assert len(manifest["entries"]) == 40

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

    all_json = sorted(path for path in tmp_path.rglob("*.json") if path.name != "manifest.json")
    assert len(all_json) == 40

    by_scale = {name: 0 for name in BENCHMARK_SCALE_ORDER}
    by_scale_split = {(name, "train"): 0 for name in BENCHMARK_SCALE_ORDER}
    by_scale_split.update({(name, "test"): 0 for name in BENCHMARK_SCALE_ORDER})

    for entry in manifest["entries"]:
        by_scale[entry["scale"]] += 1
        by_scale_split[(entry["scale"], entry["split"])] += 1

    assert all(value == 10 for value in by_scale.values())
    assert all(by_scale_split[(name, "train")] == 5 for name in BENCHMARK_SCALE_ORDER)
    assert all(by_scale_split[(name, "test")] == 5 for name in BENCHMARK_SCALE_ORDER)


def test_generated_s_seed1001_and_seed1009_match_expected_profile(tmp_path):
    manifest = generate_instances(
        output_root=tmp_path,
        scales=["S"],
        episode_length_s=28_800.0,
        overwrite=True,
    )
    entries = manifest["entries"]
    assert len(entries) == 10

    def load_seed(seed: int) -> dict:
        entry = [item for item in entries if item["seed"] == seed][0]
        path = tmp_path / entry["path"]
        return json.loads(path.read_text(encoding="utf-8"))

    s1001 = load_seed(1001)
    assert s1001["metadata"]["split"] == "train"
    assert s1001["metadata"]["num_vehicles"] == 3
    assert s1001["metadata"]["num_charging_stations"] == 2
    assert s1001["layout"]["num_tasks"] == 15
    assert s1001["metadata"]["arrival_time_sampling_mode"] == "fixed_count"
    assert s1001["metadata"]["arrival_count_semantics"] == "conditional_on_N_tasks"
    assert s1001["metadata"]["demand_params_kg"]["max"] == 150.0

    demands_1001 = [float(v) for v in s1001["scenarios"][0]["task_demands"].values()]
    assert 30.0 <= min(demands_1001) <= 45.0
    assert 120.0 <= max(demands_1001) <= 130.0

    s1009 = load_seed(1009)
    assert s1009["metadata"]["split"] == "test"
    assert s1009["layout"]["num_tasks"] == 16
    demands_1009 = [float(v) for v in s1009["scenarios"][0]["task_demands"].values()]
    assert 20.0 <= min(demands_1009) <= 30.0
    assert 90.0 <= max(demands_1009) <= 105.0

    for payload in (s1001, s1009):
        scenario = payload["scenarios"][0]
        release_times = [float(v) for v in scenario["task_release_times"].values()]
        assert all(0.0 <= t <= 28_800.0 for t in release_times)
        assert all(0.0 <= float(d) <= 150.0 for d in scenario["task_demands"].values())


def test_generate_instances_supports_demand_cap_and_thinning_mode(tmp_path):
    manifest = generate_instances(
        output_root=tmp_path,
        scales=["S"],
        episode_length_s=28_800.0,
        overwrite=True,
        arrival_time_sampling_mode="thinning",
        demand_max_ratio_of_vehicle_capacity=0.8,
    )
    assert manifest["arrival_time_sampling_mode"] == "thinning"
    assert manifest["arrival_count_semantics"] == "nhpp_thinning_with_fallback_to_fixed_count"
    assert manifest["effective_demand_max_kg_by_scale"]["S"] == 120.0

    for entry in manifest["entries"]:
        payload = json.loads((tmp_path / entry["path"]).read_text(encoding="utf-8"))
        assert payload["metadata"]["arrival_time_sampling_mode"] == "thinning"
        assert payload["metadata"]["demand_params_kg"]["max"] == 120.0
        demands = [float(v) for v in payload["scenarios"][0]["task_demands"].values()]
        assert demands
        assert max(demands) <= 120.0
