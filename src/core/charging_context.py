"""Context helpers for local (partial) charging decisions.

The Week 5 plan requires every charging decision to be aware of the local
state: current battery ratio, the energy deficit to the next opportunity, the
time-window pressure of upcoming tasks, and how dense the remaining charging
stations are.  This module provides a lightweight data container plus a
discretiser that maps the continuous ratios to coarse levels (4 battery × 3
slack × 3 density = 36 states) so that both heuristic rules and RL agents can
reason about "when / where / how much" to recharge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from physics.energy import EnergyConfig


@dataclass(frozen=True)
class ChargingContext:
    """Continuous ratios describing the local charging situation."""

    battery_ratio: float
    demand_ratio: float
    time_slack_ratio: float
    station_density: float


@dataclass(frozen=True)
class ChargingStateLevels:
    """Discrete identifiers for the tri-layer Week 5 state grid."""

    battery_level: int
    slack_level: int
    density_level: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return self.battery_level, self.slack_level, self.density_level


class ChargingContextDiscretizer:
    """Maps :class:`ChargingContext` ratios to coarse integer levels."""

    def __init__(self, energy_config: EnergyConfig,
                 slack_thresholds: Tuple[float, float] = (0.0, 0.3),
                 density_thresholds: Tuple[float, float] = (0.0, 0.2)) -> None:
        self.energy_config = energy_config
        self.slack_thresholds = slack_thresholds
        self.density_thresholds = density_thresholds

    def discretize(self, context: ChargingContext) -> ChargingStateLevels:
        return ChargingStateLevels(
            battery_level=self._bucket_battery(context.battery_ratio),
            slack_level=self._bucket_slack(context.time_slack_ratio),
            density_level=self._bucket_density(context.station_density),
        )

    # ---- bucket helpers -------------------------------------------------
    def _bucket_battery(self, ratio: float) -> int:
        safety = self.energy_config.safety_threshold
        warning = self.energy_config.warning_threshold
        comfort = self.energy_config.comfort_threshold

        if ratio <= max(0.0, safety):
            return 0
        if ratio <= warning:
            return 1
        if ratio <= comfort:
            return 2
        return 3

    def _bucket_slack(self, ratio: float) -> int:
        lower, upper = self.slack_thresholds
        if ratio <= lower:
            return 0
        if ratio <= upper:
            return 1
        return 2

    def _bucket_density(self, ratio: float) -> int:
        lower, upper = self.density_thresholds
        if ratio <= lower:
            return 0
        if ratio <= upper:
            return 1
        return 2

