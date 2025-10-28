"""Utilities for selecting and tracking ALNS operators."""

"""Adaptive operator scoring utilities shared by the planning pipeline.

The ALNS planner routes all destroy and repair operator choices through this
module.  It maintains exponentially-decayed performance weights, computes
roulette-wheel sampling probabilities, and exposes human-readable summaries so
we can inspect which neighbourhood moves drive improvements.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class OperatorStats:
    """Simple data structure to expose operator statistics."""

    usage_count: int
    success_count: int
    success_rate: float
    avg_improvement: float
    weight: float


class AdaptiveOperatorSelector:
    """Adaptive roulette-wheel selector used by the ALNS optimizer."""

    def __init__(self, operators: Iterable[str], initial_weight: float = 1.0, decay_factor: float = 0.8) -> None:
        self.operators: List[str] = list(operators)
        if not self.operators:
            raise ValueError("operators must contain at least one entry")

        self.decay_factor = decay_factor
        self.weights = defaultdict(lambda: initial_weight)
        self.usage_count = defaultdict(int)
        self.success_count = defaultdict(int)
        self.total_improvement = defaultdict(float)

        # Reward configuration inspired by Ropke & Pisinger (2006)
        self.sigma_best = 33
        self.sigma_accept = 9
        self.sigma_improve = 13

    def select(self) -> str:
        """Select an operator using roulette-wheel sampling."""

        total_weight = sum(self.weights[op] for op in self.operators)
        if total_weight <= 0:
            probability = 1 / len(self.operators)
            thresholds = [probability * (i + 1) for i in range(len(self.operators))]
        else:
            probability = 1 / total_weight
            cumulative = 0.0
            thresholds = []
            for op in self.operators:
                cumulative += self.weights[op] * probability
                thresholds.append(cumulative)

        rand_val = random.random()
        for op, threshold in zip(self.operators, thresholds):
            if rand_val <= threshold:
                self.usage_count[op] += 1
                return op

        final_operator = self.operators[-1]
        self.usage_count[final_operator] += 1
        return final_operator

    def update(self, operator: str, improvement: float, *, is_new_best: bool, is_accepted: bool) -> None:
        """Update the weight of an operator based on the observed outcome."""

        reward = 0
        if is_new_best:
            reward = self.sigma_best
        elif is_accepted:
            reward = self.sigma_accept
        elif improvement > 0:
            reward = self.sigma_improve

        if improvement > 0:
            self.success_count[operator] += 1
            self.total_improvement[operator] += improvement

        old_weight = self.weights[operator]
        self.weights[operator] = old_weight * self.decay_factor + reward * (1 - self.decay_factor)

    def statistics(self) -> Dict[str, OperatorStats]:
        """Return a snapshot of tracked statistics for each operator."""

        stats: Dict[str, OperatorStats] = {}
        for op in self.operators:
            usage = self.usage_count[op]
            success = self.success_count[op]
            success_rate = success / usage if usage else 0.0
            avg_improvement = (self.total_improvement[op] / success) if success else 0.0
            stats[op] = OperatorStats(
                usage_count=usage,
                success_count=success,
                success_rate=success_rate,
                avg_improvement=avg_improvement,
                weight=self.weights[op],
            )
        return stats

    def format_statistics(self) -> str:
        """Format statistics in a human-readable table."""

        rows = ["算子        使用次数  成功次数  成功率    平均改进    当前权重"]
        for op, data in self.statistics().items():
            rows.append(
                f"{op:<10} {data.usage_count:<8} {data.success_count:<8} "
                f"{data.success_rate:>8.2%} {data.avg_improvement:>10.2f} {data.weight:>10.2f}"
            )
        return "\n".join(rows)
