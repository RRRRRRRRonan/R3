"""Utilities for building simple demand heatmaps from historical logs."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class HeatmapConfig:
    """Configuration for historical heatmap construction."""

    decay_half_life_s: Optional[float] = None
    priority_weight: float = 0.1


def build_heatmap_from_task_log(
    path: str | Path | None,
    *,
    config: Optional[HeatmapConfig] = None,
) -> Dict[int, float]:
    """Build a node-level heatmap from a historical task log CSV.

    Expected columns: pickup_node_id, arrival_time (optional), priority (optional).
    """

    if path is None:
        return {}
    log_path = Path(path)
    if not log_path.exists():
        return {}

    cfg = config or HeatmapConfig()
    rows = _read_rows(log_path)
    if not rows:
        return {}

    time_key = _detect_time_key(rows)
    max_time = _max_time(rows, time_key) if time_key else None
    heatmap: Dict[int, float] = {}
    for row in rows:
        node_id = _safe_int(row.get("pickup_node_id"))
        if node_id is None:
            continue
        weight = 1.0
        priority = _safe_float(row.get("priority"))
        if priority is not None:
            weight += cfg.priority_weight * priority
        if time_key and cfg.decay_half_life_s:
            t = _safe_float(row.get(time_key))
            if t is not None and max_time is not None and cfg.decay_half_life_s > 0:
                age = max(0.0, max_time - t)
                weight *= 0.5 ** (age / cfg.decay_half_life_s)
        heatmap[node_id] = heatmap.get(node_id, 0.0) + weight

    return heatmap


def _read_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [row for row in reader]
    except OSError:
        return []


def _detect_time_key(rows: Iterable[Dict[str, str]]) -> Optional[str]:
    for key in ("arrival_time", "time"):
        if rows and key in rows[0]:
            return key
    return None


def _max_time(rows: Iterable[Dict[str, str]], key: str) -> Optional[float]:
    max_val = None
    for row in rows:
        val = _safe_float(row.get(key))
        if val is None:
            continue
        if max_val is None or val > max_val:
            max_val = val
    return max_val


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["HeatmapConfig", "build_heatmap_from_task_log"]
