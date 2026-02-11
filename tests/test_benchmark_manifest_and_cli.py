"""Tests for benchmark manifest helpers and CLI input options."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from config.benchmark_manifest import list_manifest_entries, load_manifest, resolve_entry_path


def _write_manifest(path: Path) -> Path:
    payload = {
        "mode": "fixed_task_count",
        "entries": [
            {
                "scale": "L",
                "seed": 3001,
                "split": "test",
                "path": "L/L_seed3001.json",
                "num_tasks": 55,
                "num_vehicles": 8,
                "num_charging_stations": 4,
            },
            {
                "scale": "S",
                "seed": 1001,
                "split": "train",
                "path": "S/S_seed1001.json",
                "num_tasks": 18,
                "num_vehicles": 3,
                "num_charging_stations": 2,
            },
            {
                "scale": "S",
                "seed": 1002,
                "split": "test",
                "path": "S/S_seed1002.json",
                "num_tasks": 20,
                "num_vehicles": 3,
                "num_charging_stations": 2,
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def _run_help(script_name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / script_name
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_manifest_filter_and_resolve_path(tmp_path):
    manifest_path = _write_manifest(tmp_path / "manifest.json")
    manifest = load_manifest(manifest_path)

    entries = list_manifest_entries(manifest, scale="S")
    assert [item.seed for item in entries] == [1001, 1002]

    test_entries = list_manifest_entries(manifest, split="test", scale="S", seed=1002)
    assert len(test_entries) == 1
    assert test_entries[0].path == "S/S_seed1002.json"

    resolved = resolve_entry_path(test_entries[0], instances_root=tmp_path / "instances")
    assert resolved == tmp_path / "instances" / "S/S_seed1002.json"


def test_train_maskable_ppo_cli_exposes_experiment_and_manifest_options():
    output = _run_help("train_maskable_ppo.py")
    assert "--experiment-json" in output
    assert "--manifest-json" in output


def test_run_alns_benchmark_cli_exposes_experiment_and_manifest_options():
    output = _run_help("run_alns_benchmark.py")
    assert "--experiment-json" in output
    assert "--manifest-json" in output


def test_run_mip_baseline_cli_exposes_experiment_and_manifest_options():
    output = _run_help("run_mip_baseline.py")
    assert "--experiment-json" in output
    assert "--manifest-json" in output


def test_evaluate_all_cli_exposes_mip_budget_options():
    output = _run_help("evaluate_all.py")
    assert "--mip-time-limit-s" in output
    assert "--mip-time-limit-s-medium" in output
    assert "--mip-skip-scales" in output


def test_train_maskable_ppo_cli_default_max_time_is_28800s():
    output = _run_help("train_maskable_ppo.py")
    assert "--max-time-s MAX_TIME_S" in output
    assert "(default: 28800.0)" in output


def test_generate_benchmark_instances_cli_default_episode_length_is_28800s():
    output = _run_help("generate_benchmark_instances.py")
    assert "--episode-length-s EPISODE_LENGTH_S" in output
    assert "(default: 28800.0)" in output
