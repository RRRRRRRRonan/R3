#!/usr/bin/env python3
"""Scan decision logs and highlight suspicious segments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"


@dataclass(frozen=True)
class DecisionRow:
    row_index: int
    step: Optional[int]
    time: Optional[float]
    event_type: Optional[str]
    selected_rule: Optional[int]
    masked: Optional[bool]
    infeasible_cost: float
    total_cost: float


@dataclass(frozen=True)
class Anomaly:
    kind: str
    severity: str
    start_row: int
    end_row: int
    length: int
    start_step: Optional[int]
    end_step: Optional[int]
    start_time: Optional[float]
    end_time: Optional[float]
    message: str


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_optional_int(value: Any) -> Optional[int]:
    parsed = _parse_optional_float(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except (TypeError, ValueError):
        return None


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_row(raw: Dict[str, Any], row_index: int) -> DecisionRow:
    cost = raw.get("cost_breakdown")
    infeasible = 0.0
    total = 0.0
    if isinstance(cost, dict):
        infeasible = float(_parse_optional_float(cost.get("infeasible")) or 0.0)
        total = float(_parse_optional_float(cost.get("total")) or 0.0)
    else:
        infeasible = float(
            _parse_optional_float(raw.get("cost_infeasible"))
            or _parse_optional_float(raw.get("infeasible"))
            or 0.0
        )
        total = float(_parse_optional_float(raw.get("cost_total")) or _parse_optional_float(raw.get("total")) or 0.0)

    return DecisionRow(
        row_index=row_index,
        step=_parse_optional_int(raw.get("step")),
        time=_parse_optional_float(raw.get("time")),
        event_type=(str(raw.get("event_type")) if raw.get("event_type") is not None else None),
        selected_rule=_parse_optional_int(raw.get("selected_rule")),
        masked=_parse_optional_bool(raw.get("masked")),
        infeasible_cost=infeasible,
        total_cost=total,
    )


def read_decision_log(path: Path) -> List[DecisionRow]:
    rows: List[DecisionRow] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    continue
                rows.append(_normalize_row(raw, idx))
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, raw in enumerate(reader):
                rows.append(_normalize_row(raw, idx))
        return rows

    raise ValueError(f"Unsupported decision-log format: {path}")


def _segment_event_summary(rows: Sequence[DecisionRow], start: int, end: int) -> str:
    counter = Counter(row.event_type or "None" for row in rows[start : end + 1])
    return ", ".join(f"{name}:{count}" for name, count in counter.most_common(3))


def detect_anomalies(
    rows: Sequence[DecisionRow],
    *,
    same_time_threshold: int,
    charge_done_threshold: int,
    cost_spike_threshold: float,
    time_epsilon: float,
) -> List[Anomaly]:
    anomalies: List[Anomaly] = []
    if not rows:
        return anomalies

    # 1) Same-time streak (possible no-progress / livelock segment).
    seg_start: Optional[int] = None
    for i in range(1, len(rows)):
        t_prev = rows[i - 1].time
        t_curr = rows[i].time
        same_time = (
            t_prev is not None
            and t_curr is not None
            and abs(t_curr - t_prev) <= time_epsilon
        )
        if same_time:
            if seg_start is None:
                seg_start = i - 1
            continue
        if seg_start is not None:
            seg_end = i - 1
            length = seg_end - seg_start + 1
            if length >= same_time_threshold:
                summary = _segment_event_summary(rows, seg_start, seg_end)
                anomalies.append(
                    Anomaly(
                        kind="same_time_streak",
                        severity="critical",
                        start_row=seg_start,
                        end_row=seg_end,
                        length=length,
                        start_step=rows[seg_start].step,
                        end_step=rows[seg_end].step,
                        start_time=rows[seg_start].time,
                        end_time=rows[seg_end].time,
                        message=f"同时间戳连续 {length} 条，事件分布: {summary}",
                    )
                )
            seg_start = None
    if seg_start is not None:
        seg_end = len(rows) - 1
        length = seg_end - seg_start + 1
        if length >= same_time_threshold:
            summary = _segment_event_summary(rows, seg_start, seg_end)
            anomalies.append(
                Anomaly(
                    kind="same_time_streak",
                    severity="critical",
                    start_row=seg_start,
                    end_row=seg_end,
                    length=length,
                    start_step=rows[seg_start].step,
                    end_step=rows[seg_end].step,
                    start_time=rows[seg_start].time,
                    end_time=rows[seg_end].time,
                    message=f"同时间戳连续 {length} 条，事件分布: {summary}",
                )
            )

    # 2) CHARGE_DONE streak (often associated with zero-duration loops).
    seg_start = None
    for i, row in enumerate(rows):
        is_charge_done = row.event_type == "CHARGE_DONE"
        if is_charge_done:
            if seg_start is None:
                seg_start = i
            continue
        if seg_start is not None:
            seg_end = i - 1
            length = seg_end - seg_start + 1
            if length >= charge_done_threshold:
                anomalies.append(
                    Anomaly(
                        kind="charge_done_streak",
                        severity="critical",
                        start_row=seg_start,
                        end_row=seg_end,
                        length=length,
                        start_step=rows[seg_start].step,
                        end_step=rows[seg_end].step,
                        start_time=rows[seg_start].time,
                        end_time=rows[seg_end].time,
                        message=f"CHARGE_DONE 连续 {length} 条",
                    )
                )
            seg_start = None
    if seg_start is not None:
        seg_end = len(rows) - 1
        length = seg_end - seg_start + 1
        if length >= charge_done_threshold:
            anomalies.append(
                Anomaly(
                    kind="charge_done_streak",
                    severity="critical",
                    start_row=seg_start,
                    end_row=seg_end,
                    length=length,
                    start_step=rows[seg_start].step,
                    end_step=rows[seg_end].step,
                    start_time=rows[seg_start].time,
                    end_time=rows[seg_end].time,
                    message=f"CHARGE_DONE 连续 {length} 条",
                )
            )

    # 3) Unmasked infeasible step(s): mask failed to shield an infeasible action.
    seg_start = None
    for i, row in enumerate(rows):
        cond = (row.masked is False) and (row.infeasible_cost > 0.0)
        if cond:
            if seg_start is None:
                seg_start = i
            continue
        if seg_start is not None:
            seg_end = i - 1
            length = seg_end - seg_start + 1
            anomalies.append(
                Anomaly(
                    kind="unmasked_infeasible",
                    severity="critical",
                    start_row=seg_start,
                    end_row=seg_end,
                    length=length,
                    start_step=rows[seg_start].step,
                    end_step=rows[seg_end].step,
                    start_time=rows[seg_start].time,
                    end_time=rows[seg_end].time,
                    message=f"masked=False 但 infeasible_cost>0，连续 {length} 条",
                )
            )
            seg_start = None
    if seg_start is not None:
        seg_end = len(rows) - 1
        length = seg_end - seg_start + 1
        anomalies.append(
            Anomaly(
                kind="unmasked_infeasible",
                severity="critical",
                start_row=seg_start,
                end_row=seg_end,
                length=length,
                start_step=rows[seg_start].step,
                end_step=rows[seg_end].step,
                start_time=rows[seg_start].time,
                end_time=rows[seg_end].time,
                message=f"masked=False 但 infeasible_cost>0，连续 {length} 条",
            )
        )

    # 4) Cost spike: usually indicates unstable reward/penalty configuration.
    if cost_spike_threshold > 0.0:
        seg_start = None
        for i, row in enumerate(rows):
            cond = row.total_cost >= cost_spike_threshold
            if cond:
                if seg_start is None:
                    seg_start = i
                continue
            if seg_start is not None:
                seg_end = i - 1
                length = seg_end - seg_start + 1
                max_cost = max(rows[j].total_cost for j in range(seg_start, seg_end + 1))
                severity = "critical" if max_cost >= cost_spike_threshold * 2.0 else "warning"
                anomalies.append(
                    Anomaly(
                        kind="cost_spike",
                        severity=severity,
                        start_row=seg_start,
                        end_row=seg_end,
                        length=length,
                        start_step=rows[seg_start].step,
                        end_step=rows[seg_end].step,
                        start_time=rows[seg_start].time,
                        end_time=rows[seg_end].time,
                        message=(
                            f"total_cost >= {cost_spike_threshold:.2f}，"
                            f"连续 {length} 条，段内最大值 {max_cost:.2f}"
                        ),
                    )
                )
                seg_start = None
        if seg_start is not None:
            seg_end = len(rows) - 1
            length = seg_end - seg_start + 1
            max_cost = max(rows[j].total_cost for j in range(seg_start, seg_end + 1))
            severity = "critical" if max_cost >= cost_spike_threshold * 2.0 else "warning"
            anomalies.append(
                Anomaly(
                    kind="cost_spike",
                    severity=severity,
                    start_row=seg_start,
                    end_row=seg_end,
                    length=length,
                    start_step=rows[seg_start].step,
                    end_step=rows[seg_end].step,
                    start_time=rows[seg_start].time,
                    end_time=rows[seg_end].time,
                    message=(
                        f"total_cost >= {cost_spike_threshold:.2f}，"
                        f"连续 {length} 条，段内最大值 {max_cost:.2f}"
                    ),
                )
            )

    return anomalies


def _format_time(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{value:.6f}"


def print_anomalies(
    path: Path,
    anomalies: Sequence[Anomaly],
    *,
    use_color: bool,
) -> None:
    if not anomalies:
        line = f"[OK] {path} 未检测到潜在异常段"
        if use_color:
            line = f"{GREEN}{line}{RESET}"
        print(line)
        return

    header = f"[WARN] {path} 检测到 {len(anomalies)} 个潜在异常段"
    if use_color:
        header = f"{RED}{header}{RESET}"
    print(header)
    for anomaly in anomalies:
        line = (
            f"  - [{anomaly.severity}] {anomaly.kind} "
            f"rows={anomaly.start_row}-{anomaly.end_row} "
            f"steps={anomaly.start_step}-{anomaly.end_step} "
            f"time={_format_time(anomaly.start_time)}->{_format_time(anomaly.end_time)} "
            f"len={anomaly.length} | {anomaly.message}"
        )
        if use_color:
            color = RED if anomaly.severity == "critical" else YELLOW
            line = f"{color}{line}{RESET}"
        print(line)


def write_report(path: Path, anomalies: Sequence[Anomaly], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "log_path": str(path),
            "anomaly_type": item.kind,
            "severity": item.severity,
            "start_row": item.start_row,
            "end_row": item.end_row,
            "length": item.length,
            "start_step": item.start_step,
            "end_step": item.end_step,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "message": item.message,
        }
        for item in anomalies
    ]

    fieldnames = [
        "log_path",
        "anomaly_type",
        "severity",
        "start_row",
        "end_row",
        "length",
        "start_step",
        "end_step",
        "start_time",
        "end_time",
        "message",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_decision_logs(target: Path) -> Iterable[Path]:
    if target.is_file():
        yield target
        return
    if not target.exists():
        return
    # Prefer JSONL when both JSONL/CSV exist for the same log stamp.
    selected: Dict[Path, Path] = {}
    for path in sorted(target.rglob("decision_log_*")):
        if path.name.endswith("_anomaly_report.csv"):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".jsonl", ".csv"}:
            continue
        key = path.with_suffix("").resolve()
        existing = selected.get(key)
        if existing is None:
            selected[key] = path
            continue
        if existing.suffix.lower() == ".csv" and suffix == ".jsonl":
            selected[key] = path
    for path in sorted(selected.values()):
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="自动扫描 decision_log 并标红潜在异常段")
    parser.add_argument(
        "target",
        type=str,
        help="decision_log 文件路径，或包含多个日志的目录",
    )
    parser.add_argument(
        "--same-time-threshold",
        type=int,
        default=20,
        help="同时间戳连续长度阈值（超过判为异常）",
    )
    parser.add_argument(
        "--charge-done-threshold",
        type=int,
        default=10,
        help="CHARGE_DONE 连续长度阈值（超过判为异常）",
    )
    parser.add_argument(
        "--time-epsilon",
        type=float,
        default=1e-9,
        help="判定同时间戳的容差",
    )
    parser.add_argument(
        "--cost-spike-threshold",
        type=float,
        default=5000.0,
        help="单步 total_cost 异常阈值（<=0 关闭该检测）",
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        default=None,
        help="单文件模式下自定义报告路径；目录模式下忽略（每个日志单独输出）",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="关闭终端彩色输出",
    )
    args = parser.parse_args()

    target = Path(args.target)
    logs = list(iter_decision_logs(target))
    if not logs:
        print(f"[ERROR] 未找到 decision_log: {target}")
        return 1

    total_anomalies = 0
    for log_path in logs:
        rows = read_decision_log(log_path)
        anomalies = detect_anomalies(
            rows,
            same_time_threshold=max(1, int(args.same_time_threshold)),
            charge_done_threshold=max(1, int(args.charge_done_threshold)),
            cost_spike_threshold=max(0.0, float(args.cost_spike_threshold)),
            time_epsilon=max(0.0, float(args.time_epsilon)),
        )
        total_anomalies += len(anomalies)
        print_anomalies(log_path, anomalies, use_color=not args.no_color)

        if args.report_csv and target.is_file():
            report_path = Path(args.report_csv)
        else:
            report_path = log_path.with_name(f"{log_path.stem}_anomaly_report.csv")
        write_report(log_path, anomalies, report_path)

    if total_anomalies == 0:
        print("扫描完成：未发现潜在异常段。")
    else:
        print(f"扫描完成：共发现 {total_anomalies} 个潜在异常段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
