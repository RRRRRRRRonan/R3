"""Helpers for loading and querying benchmark instance manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BenchmarkManifestEntry:
    """One instance entry from ``data/instances/manifest.json``."""

    scale: str
    seed: int
    split: str
    path: str
    num_tasks: int
    num_vehicles: int
    num_charging_stations: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkManifestEntry":
        return cls(
            scale=str(data["scale"]).upper(),
            seed=int(data["seed"]),
            split=str(data["split"]).lower(),
            path=str(data["path"]),
            num_tasks=int(data.get("num_tasks", 0)),
            num_vehicles=int(data.get("num_vehicles", 1)),
            num_charging_stations=int(data.get("num_charging_stations", 0)),
        )


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Load a benchmark manifest JSON payload."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "entries" not in payload or not isinstance(payload["entries"], list):
        raise ValueError(f"Invalid manifest structure: missing list field 'entries' in {path}")
    return payload


def list_manifest_entries(
    manifest: Dict[str, Any],
    *,
    split: Optional[str] = None,
    scale: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[BenchmarkManifestEntry]:
    """Return entries filtered by split/scale/seed and sorted by (scale, seed)."""

    split_norm = str(split).lower() if split is not None else None
    scale_norm = str(scale).upper() if scale is not None else None
    seed_norm = int(seed) if seed is not None else None

    out: List[BenchmarkManifestEntry] = []
    for raw in manifest.get("entries", []):
        entry = BenchmarkManifestEntry.from_dict(raw)
        if split_norm is not None and entry.split != split_norm:
            continue
        if scale_norm is not None and entry.scale != scale_norm:
            continue
        if seed_norm is not None and entry.seed != seed_norm:
            continue
        out.append(entry)

    out.sort(key=lambda item: (item.scale, item.seed))
    return out


def resolve_entry_path(
    entry: BenchmarkManifestEntry,
    *,
    instances_root: str | Path,
) -> Path:
    """Resolve an entry's relative path against an instances root directory."""

    return Path(instances_root) / entry.path


def select_manifest_entry(
    manifest: Dict[str, Any],
    *,
    split: Optional[str] = None,
    scale: Optional[str] = None,
    seed: Optional[int] = None,
    entry_index: int = 0,
) -> BenchmarkManifestEntry:
    """Select one filtered entry by index."""

    matches = list_manifest_entries(
        manifest,
        split=split,
        scale=scale,
        seed=seed,
    )
    if not matches:
        raise ValueError(
            "No manifest entries matched filters: split={!r}, scale={!r}, seed={!r}".format(
                split,
                scale,
                seed,
            )
        )
    idx = int(entry_index)
    if idx < 0 or idx >= len(matches):
        raise IndexError(f"entry_index {idx} out of range for {len(matches)} matches")
    return matches[idx]


__all__ = [
    "BenchmarkManifestEntry",
    "list_manifest_entries",
    "load_manifest",
    "resolve_entry_path",
    "select_manifest_entry",
]

