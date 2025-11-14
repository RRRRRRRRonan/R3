"""Epsilon (exploration rate) strategies for Q-learning-based ALNS.

This module defines different epsilon initialization and decay strategies
to control the exploration-exploitation trade-off in Q-learning operator
selection.

Author: R3 Research Team
Date: 2025-11-12
Related: Week 2 experiments (adaptive epsilon)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EpsilonStrategy:
    """Defines epsilon (exploration rate) strategy for Q-learning.

    Attributes:
        name: Strategy identifier
        initial_epsilon: Starting exploration rate (0.0 to 1.0)
        decay_rate: Multiplicative decay per iteration (typically 0.99-0.999)
        min_epsilon: Minimum epsilon floor to maintain some exploration
    """

    name: str
    initial_epsilon: float
    decay_rate: float = 0.995
    min_epsilon: float = 0.01

    def __post_init__(self):
        """Validate epsilon parameters."""
        if not 0.0 <= self.initial_epsilon <= 1.0:
            raise ValueError(
                f"initial_epsilon must be in [0, 1], got {self.initial_epsilon}"
            )
        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError(
                f"decay_rate must be in [0, 1], got {self.decay_rate}"
            )
        if not 0.0 <= self.min_epsilon <= 1.0:
            raise ValueError(
                f"min_epsilon must be in [0, 1], got {self.min_epsilon}"
            )
        if self.min_epsilon > self.initial_epsilon:
            raise ValueError(
                f"min_epsilon ({self.min_epsilon}) cannot exceed "
                f"initial_epsilon ({self.initial_epsilon})"
            )

    @staticmethod
    def current() -> "EpsilonStrategy":
        """Current baseline strategy (Week 1).

        Returns:
            EpsilonStrategy with current default parameters

        Notes:
            - Initial epsilon: 0.12 (fixed for all scales)
            - Decay rate: 0.995 (standard exponential decay)
            - Min epsilon: 0.01 (maintains 1% exploration)
        """
        return EpsilonStrategy(
            name="current",
            initial_epsilon=0.12,
            decay_rate=0.995,
            min_epsilon=0.01,
        )

    @staticmethod
    def scale_adaptive(num_requests: int) -> "EpsilonStrategy":
        """Scale-adaptive strategy (Week 2 primary hypothesis).

        Adjusts initial epsilon based on problem scale (number of requests).
        Larger problems have exponentially larger solution spaces, requiring
        more exploration to avoid premature convergence.

        Args:
            num_requests: Number of customer requests in the scenario

        Returns:
            EpsilonStrategy with scale-dependent initial epsilon

        Theory:
            - Small (≤12 requests): ~10^9 solutions → modest exploration (0.30)
            - Medium (13-30 requests): ~10^30 solutions → medium exploration (0.50)
            - Large (>30 requests): ~10^60 solutions → high exploration (0.70)

        Notes:
            - Min epsilon increased to 0.05 (5%) for sustained exploration
            - Decay rate unchanged (0.995)
            - Thresholds based on Schneider instance sizes
        """
        if num_requests <= 12:
            initial = 0.30
            scale = "small"
        elif num_requests <= 30:
            initial = 0.50
            scale = "medium"
        else:
            initial = 0.70
            scale = "large"

        return EpsilonStrategy(
            name=f"scale_adaptive_{scale}",
            initial_epsilon=initial,
            decay_rate=0.995,
            min_epsilon=0.05,  # Higher floor for sustained exploration
        )

    @staticmethod
    def high_uniform() -> "EpsilonStrategy":
        """High uniform exploration strategy (Week 2 control).

        Uses high epsilon (0.50) uniformly across all scales. Simpler than
        scale-adaptive but tests whether high exploration universally beneficial.

        Returns:
            EpsilonStrategy with high uniform exploration

        Notes:
            - Initial epsilon: 0.50 (4.2× current baseline)
            - Useful for testing if scale-adaptation is necessary
            - May degrade small-scale performance if exploration excessive
        """
        return EpsilonStrategy(
            name="high_uniform",
            initial_epsilon=0.50,
            decay_rate=0.995,
            min_epsilon=0.05,
        )

    @staticmethod
    def from_name(
        name: str, num_requests: Optional[int] = None
    ) -> "EpsilonStrategy":
        """Factory method to create strategy by name.

        Args:
            name: Strategy name ("current", "scale_adaptive", "high_uniform")
            num_requests: Number of requests (required for scale_adaptive)

        Returns:
            EpsilonStrategy instance

        Raises:
            ValueError: If name is invalid or num_requests missing for scale_adaptive

        Example:
            >>> strategy = EpsilonStrategy.from_name("current")
            >>> strategy = EpsilonStrategy.from_name("scale_adaptive", num_requests=24)
        """
        if name == "current":
            return EpsilonStrategy.current()
        elif name == "scale_adaptive":
            if num_requests is None:
                raise ValueError(
                    "num_requests required for scale_adaptive strategy"
                )
            return EpsilonStrategy.scale_adaptive(num_requests)
        elif name == "high_uniform":
            return EpsilonStrategy.high_uniform()
        else:
            raise ValueError(
                f"Unknown epsilon strategy: {name}. "
                f"Valid options: current, scale_adaptive, high_uniform"
            )

    def compute_epsilon(self, iteration: int) -> float:
        """Compute epsilon value at given iteration.

        Args:
            iteration: Current iteration number (0-indexed)

        Returns:
            Epsilon value after decay, bounded by min_epsilon

        Formula:
            epsilon(t) = max(min_epsilon, initial_epsilon * decay_rate^t)

        Example:
            >>> strategy = EpsilonStrategy.current()
            >>> strategy.compute_epsilon(0)   # 0.12 (initial)
            >>> strategy.compute_epsilon(100) # ~0.067
            >>> strategy.compute_epsilon(500) # ~0.01 (at floor)
        """
        decayed = self.initial_epsilon * (self.decay_rate**iteration)
        return max(self.min_epsilon, decayed)

    def iterations_to_floor(self) -> int:
        """Compute iterations until epsilon reaches floor (min_epsilon).

        Returns:
            Number of iterations to reach min_epsilon

        Formula:
            t = log(min_epsilon / initial_epsilon) / log(decay_rate)

        Example:
            >>> strategy = EpsilonStrategy.current()
            >>> strategy.iterations_to_floor()  # ~470 iterations
        """
        import math

        if self.initial_epsilon <= self.min_epsilon:
            return 0

        t = math.log(self.min_epsilon / self.initial_epsilon) / math.log(
            self.decay_rate
        )
        return int(math.ceil(t))

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"EpsilonStrategy(name='{self.name}', "
            f"initial={self.initial_epsilon:.2f}, "
            f"decay={self.decay_rate:.3f}, "
            f"min={self.min_epsilon:.2f})"
        )

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"EpsilonStrategy(name='{self.name}', "
            f"initial_epsilon={self.initial_epsilon}, "
            f"decay_rate={self.decay_rate}, "
            f"min_epsilon={self.min_epsilon})"
        )
