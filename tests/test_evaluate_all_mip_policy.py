"""Tests for MIP runtime budget policy in unified evaluator."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from baselines.mip.scenario_io import ExperimentScenario
from config.benchmark_manifest import BenchmarkManifestEntry
from config.warehouse_layout import WarehouseLayoutConfig
from scripts.evaluate_all import (
    InstanceRun,
    _evaluate_algorithm,
    _parse_scale_set,
    _resolve_mip_runtime_budget,
)


def _base_args() -> Namespace:
    return Namespace(
        mip_skip_scales="L,XL",
        mip_time_limit_s=5.0,
        mip_time_limit_s_medium=30.0,
        mip_decision_epochs=3,
        mip_rule_count=13,
        mip_solver_name="ortools",
        mip_gap=0.0,
        mip_scenario_mode="minimal",
        num_vehicles=2,
        vehicle_capacity_kg=None,
        battery_capacity_kwh=None,
        initial_battery_kwh=None,
        vehicle_speed_m_s=None,
    )


def _instance(scale: str) -> InstanceRun:
    entry = BenchmarkManifestEntry(
        scale=scale,
        seed=1001,
        split="test",
        path=f"{scale}/{scale}_seed1001.json",
        num_tasks=20,
        num_vehicles=3,
        num_charging_stations=2,
    )
    experiment = ExperimentScenario(layout=WarehouseLayoutConfig(num_tasks=1, num_charging_stations=0))
    return InstanceRun(entry=entry, path=Path(entry.path), experiment=experiment)


def test_parse_scale_set_normalizes_tokens():
    assert _parse_scale_set(" l, xl ,m ") == {"L", "XL", "M"}


def test_resolve_mip_runtime_budget_by_scale_defaults():
    args = _base_args()
    budget_s = _resolve_mip_runtime_budget("S", args)
    assert budget_s["skip"] is False
    assert budget_s["time_limit_s"] == 5.0

    budget_m = _resolve_mip_runtime_budget("M", args)
    assert budget_m["skip"] is False
    assert budget_m["time_limit_s"] == 30.0
    assert budget_m["decision_epochs"] == 3

    budget_l = _resolve_mip_runtime_budget("L", args)
    assert budget_l["skip"] is True
    assert "mip_budget_disabled_for_scale_L" in budget_l["skip_reason"]


def test_evaluate_algorithm_skips_mip_for_large_scale():
    args = _base_args()
    result = _evaluate_algorithm("mip_hind", _instance("XL"), args, seed=7)
    assert result["status"] == "SKIPPED"
    assert result["skip_reason"] == "mip_budget_disabled_for_scale_XL"


def test_evaluate_algorithm_uses_medium_time_budget_for_m(monkeypatch):
    args = _base_args()
    captured = {}

    def fake_run_mip(experiment, run_args, *, time_limit_s, decision_epochs):
        captured["time_limit_s"] = time_limit_s
        captured["decision_epochs"] = decision_epochs
        return {"status": "OK", "cost": 0.0}

    import scripts.evaluate_all as eval_mod

    monkeypatch.setattr(eval_mod, "_run_mip", fake_run_mip)
    result = _evaluate_algorithm("mip_hind", _instance("M"), args, seed=11)
    assert result["status"] == "OK"
    assert captured["time_limit_s"] == 30.0
    assert captured["decision_epochs"] == 3

