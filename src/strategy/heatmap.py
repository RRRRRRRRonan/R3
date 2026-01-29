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
    bucket_size_s: float = 3600.0
    lookback_buckets: int = 3


@dataclass
class HeatmapModel:
    """Bucketed heatmap for time-dependent scoring."""

    bucket_size_s: float
    decay_half_life_s: Optional[float]
    priority_weight: float
    counts: Dict[int, Dict[int, float]]
    max_time: float

    def scores_at_time(self, t: float, *, lookback_buckets: Optional[int] = None) -> Dict[int, float]:
        if self.bucket_size_s <= 0:
            return {}
        bucket = int(max(0.0, t) // self.bucket_size_s)
        lookback = max(1, int(lookback_buckets or 1))
        scores: Dict[int, float] = {}
        for idx in range(max(0, bucket - lookback + 1), bucket + 1):
            weight = self._bucket_weight(bucket - idx)
            for node_id, value in self.counts.get(idx, {}).items():
                scores[node_id] = scores.get(node_id, 0.0) + value * weight
        return scores

    def _bucket_weight(self, age_buckets: int) -> float:
        if self.decay_half_life_s is None or self.decay_half_life_s <= 0:
            return 1.0
        age_seconds = age_buckets * self.bucket_size_s
        return 0.5 ** (age_seconds / self.decay_half_life_s)


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

    model = build_heatmap_model(path, config=config)
    return model.scores_at_time(model.max_time)


def build_heatmap_model(
    path: str | Path | None,
    *,
    config: Optional[HeatmapConfig] = None,
) -> HeatmapModel:
    if path is None:
        return HeatmapModel(3600.0, None, 0.0, {}, 0.0)
    log_path = Path(path)
    if not log_path.exists():
        return HeatmapModel(3600.0, None, 0.0, {}, 0.0)

    cfg = config or HeatmapConfig()
    rows = _read_rows(log_path)
    if not rows:
        return HeatmapModel(cfg.bucket_size_s, cfg.decay_half_life_s, cfg.priority_weight, {}, 0.0)

    time_key = _detect_time_key(rows)
    max_time = _max_time(rows, time_key) if time_key else 0.0
    bucket_size = max(1.0, cfg.bucket_size_s)
    counts: Dict[int, Dict[int, float]] = {}
    for row in rows:
        node_id = _safe_int(row.get("pickup_node_id"))
        if node_id is None:
            continue
        t = _safe_float(row.get(time_key)) if time_key else None
        if t is None:
            t = 0.0
        bucket = int(max(0.0, t) // bucket_size)
        weight = 1.0
        priority = _safe_float(row.get("priority"))
        if priority is not None:
            weight += cfg.priority_weight * priority
        counts.setdefault(bucket, {})
        counts[bucket][node_id] = counts[bucket].get(node_id, 0.0) + weight

    return HeatmapModel(
        bucket_size_s=bucket_size,
        decay_half_life_s=cfg.decay_half_life_s,
        priority_weight=cfg.priority_weight,
        counts=counts,
        max_time=max_time,
    )


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


__all__ = ["HeatmapConfig", "HeatmapModel", "build_heatmap_model", "build_heatmap_from_task_log"]
