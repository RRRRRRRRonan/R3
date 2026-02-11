"""Arrival process models for dynamic request generation.

This module implements the (paper-aligned) *Nonhomogeneous Poisson Process*
(NHPP) arrival model used to generate task release times over an episode.

Key requirement (Section 5.1 mapping):
- Nonhomogeneous Poisson arrival process with 3 piecewise-constant segments.
- Peak/normal/off-peak intensity ratio defaults to 3 : 1 : 0.5.
- Provide a thinning-based sampler (NHPP) plus a fixed-count sampler useful
  when the number of tasks is controlled elsewhere (e.g., layout presets).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RateSegment:
    """Piecewise-constant NHPP rate segment.

    Attributes:
        start_s: Segment start time (seconds), inclusive.
        end_s: Segment end time (seconds), exclusive.
        multiplier: Multiplier applied to the base rate during this segment.
    """

    start_s: float
    end_s: float
    multiplier: float

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"Invalid segment bounds: start_s={self.start_s}, end_s={self.end_s}"
            )
        if self.multiplier < 0.0:
            raise ValueError(f"Segment multiplier must be non-negative: {self.multiplier}")

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s)

    def contains(self, t_s: float) -> bool:
        t = float(t_s)
        return self.start_s <= t < self.end_s


@dataclass(frozen=True)
class NonhomogeneousPoissonArrivalModel:
    """NHPP with piecewise-constant rates.

    The rate function is defined as:
        lambda(t) = base_rate_per_s * multiplier(segment(t))
    """

    base_rate_per_s: float
    segments: Tuple[RateSegment, ...]

    def __post_init__(self) -> None:
        if self.base_rate_per_s < 0.0:
            raise ValueError(f"base_rate_per_s must be non-negative: {self.base_rate_per_s}")
        if not self.segments:
            raise ValueError("segments must be non-empty")
        # Enforce ascending segments.
        for prev, nxt in zip(self.segments, self.segments[1:]):
            if nxt.start_s < prev.end_s:
                raise ValueError(
                    "segments must be non-overlapping and sorted by time: "
                    f"{prev} overlaps {nxt}"
                )

    @property
    def horizon_s(self) -> float:
        return float(self.segments[-1].end_s)

    def rate(self, t_s: float) -> float:
        """Return lambda(t) (events/second) at time t_s."""

        t = float(t_s)
        if t < 0.0:
            return 0.0
        for seg in self.segments:
            if seg.contains(t):
                return float(self.base_rate_per_s * seg.multiplier)
        return 0.0

    @property
    def max_rate_per_s(self) -> float:
        return float(self.base_rate_per_s * max(seg.multiplier for seg in self.segments))

    def sample_arrivals_thinning(
        self,
        rng: random.Random,
        *,
        horizon_s: Optional[float] = None,
    ) -> List[float]:
        """Sample arrival times using the classic thinning algorithm.

        Returns a sorted list of event times in [0, horizon_s).
        """

        horizon = self.horizon_s if horizon_s is None else float(horizon_s)
        if horizon <= 0.0:
            return []

        lambda_max = self.max_rate_per_s
        if lambda_max <= 0.0:
            return []

        t = 0.0
        arrivals: List[float] = []
        while True:
            # Proposal via homogeneous Poisson with rate lambda_max.
            t += rng.expovariate(lambda_max)
            if t >= horizon:
                break
            accept_prob = self.rate(t) / lambda_max
            if accept_prob <= 0.0:
                continue
            if rng.random() <= accept_prob:
                arrivals.append(t)
        return arrivals

    def sample_arrivals_fixed_count(
        self,
        count: int,
        rng: random.Random,
        *,
        horizon_s: Optional[float] = None,
    ) -> List[float]:
        """Sample exactly ``count`` arrival times with density proportional to lambda(t).

        This is equivalent to sampling NHPP arrival times conditioned on having
        exactly ``count`` events within the horizon for piecewise-constant
        rates.
        """

        n = int(count)
        if n <= 0:
            return []
        horizon = self.horizon_s if horizon_s is None else float(horizon_s)
        if horizon <= 0.0:
            raise ValueError("horizon_s must be positive for fixed-count sampling")

        usable: List[RateSegment] = []
        weights: List[float] = []
        for seg in self.segments:
            start = max(0.0, seg.start_s)
            end = min(horizon, seg.end_s)
            if end <= start or seg.multiplier <= 0.0:
                continue
            usable.append(RateSegment(start_s=start, end_s=end, multiplier=seg.multiplier))
            weights.append(seg.multiplier * (end - start))

        if not usable or sum(weights) <= 0.0:
            # No positive-rate segments: collapse all arrivals to 0.
            return [0.0] * n

        times = [
            rng.uniform(seg.start_s, seg.end_s)
            for seg in rng.choices(usable, weights=weights, k=n)
        ]
        times.sort()
        return times


def build_default_nhpp_model(
    *,
    episode_length_s: float = 28_800.0,
    base_rate_per_s: float = 0.0005,
    peak_multiplier: float = 3.0,
    normal_multiplier: float = 1.0,
    offpeak_multiplier: float = 0.5,
    segment_fractions: Tuple[float, float, float, float] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
) -> NonhomogeneousPoissonArrivalModel:
    """Build the paper-aligned 3-segment NHPP arrival model.

    The default ratio is peak : normal : off-peak = 3 : 1 : 0.5.

    Segment order (default):
    - off-peak  : [0, 1/3) of the episode
    - peak      : [1/3, 2/3)
    - normal    : [2/3, 1)
    """

    horizon = float(episode_length_s)
    if horizon <= 0.0:
        raise ValueError("episode_length_s must be positive")
    if len(segment_fractions) != 4:
        raise ValueError("segment_fractions must have 4 entries (0, b1, b2, 1)")
    b0, b1, b2, b3 = (float(x) for x in segment_fractions)
    if not (0.0 <= b0 <= b1 <= b2 <= b3 <= 1.0):
        raise ValueError(f"Invalid segment_fractions: {segment_fractions}")

    t0 = horizon * b0
    t1 = horizon * b1
    t2 = horizon * b2
    t3 = horizon * b3

    segments = (
        RateSegment(start_s=t0, end_s=t1, multiplier=offpeak_multiplier),
        RateSegment(start_s=t1, end_s=t2, multiplier=peak_multiplier),
        RateSegment(start_s=t2, end_s=t3, multiplier=normal_multiplier),
    )
    return NonhomogeneousPoissonArrivalModel(
        base_rate_per_s=float(base_rate_per_s),
        segments=segments,
    )


__all__ = [
    "RateSegment",
    "NonhomogeneousPoissonArrivalModel",
    "build_default_nhpp_model",
]

