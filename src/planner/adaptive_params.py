"""Adaptive parameter management for Q-learning based on problem characteristics.

This module implements scale-adaptive parameter selection for Q-learning to address
the "No Free Lunch" problem where unified static parameters cannot optimize all
problem instances effectively.

Key insight: Different problem scales require different exploration-exploitation
trade-offs:
- Small: Fast convergence, less exploration
- Medium: Balanced approach
- Large: More exploration, slower learning

Reference: ADAPTIVE_SOLUTION_IMPLEMENTATION.md
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from config import QLearningParams


ScaleType = Literal["small", "medium", "large"]


class AdaptiveQLearningParamsManager:
    """Manages scale-adaptive Q-learning parameters.

    This manager provides problem-scale-specific parameter configurations that
    adapt the exploration-exploitation balance to the characteristics of different
    problem sizes.

    Design rationale (from 10-seed analysis):
    - Small problems (15 tasks): Need fast convergence, risk over-exploration
    - Medium problems (24 tasks): Need balanced exploration and exploitation
    - Large problems (30 tasks): Need more exploration due to larger search space

    Example:
        >>> manager = AdaptiveQLearningParamsManager()
        >>> small_params = manager.get_params_for_scale("small")
        >>> # Use small_params to initialize Q-learning agent for small instance
    """

    def __init__(self, base_params: QLearningParams | None = None):
        """Initialize the adaptive params manager.

        Args:
            base_params: Base parameters to use as template. If None, uses default
                        QLearningParams from config.
        """
        self.base_params = base_params or QLearningParams()

    def get_params_for_scale(self, scale: ScaleType) -> QLearningParams:
        """Get Q-learning parameters optimized for the given problem scale.

        Args:
            scale: Problem scale ('small', 'medium', or 'large')

        Returns:
            QLearningParams object with scale-specific parameter values

        Raises:
            ValueError: If scale is not one of the supported values
        """
        if scale not in ("small", "medium", "large"):
            raise ValueError(
                f"Unknown scale '{scale}'. Must be 'small', 'medium', or 'large'"
            )

        adjustments = self._get_scale_adjustments(scale)
        return replace(self.base_params, **adjustments)

    def _get_scale_adjustments(self, scale: ScaleType) -> dict:
        """Get parameter adjustments for the specified scale.

        Parameter rationale:

        Small scale (15 tasks):
        - Higher alpha (0.3): Learn quickly from limited search space
        - Lower epsilon_min (0.05): Reduce exploration once patterns found
        - Lower stagnation_ratio (0.15): Enter stuck state earlier to leverage LP

        Medium scale (24 tasks):
        - Moderate alpha (0.2): Balance learning speed and stability
        - Moderate epsilon_min (0.1): Maintain steady exploration
        - Moderate stagnation_ratio (0.25): Standard stuck threshold

        Large scale (30+ tasks):
        - Lower alpha (0.15): Avoid Q-value oscillation in large state space
        - Higher epsilon_min (0.15): Maintain exploration in complex landscape
        - Higher stagnation_ratio (0.35): Give more time before declaring stuck

        Args:
            scale: Problem scale

        Returns:
            Dictionary of parameter adjustments
        """
        if scale == "small":
            return {
                "alpha": 0.3,
                "epsilon_min": 0.05,
                "stagnation_ratio": 0.15,
                "deep_stagnation_ratio": 0.35,  # Proportional to stagnation_ratio
            }
        elif scale == "medium":
            return {
                "alpha": 0.2,
                "epsilon_min": 0.1,
                "stagnation_ratio": 0.25,
                "deep_stagnation_ratio": 0.45,
            }
        else:  # large
            return {
                "alpha": 0.15,
                "epsilon_min": 0.15,
                "stagnation_ratio": 0.35,
                "deep_stagnation_ratio": 0.55,
            }

    def get_performance_adjusted_params(
        self,
        scale: ScaleType,
        improvement_rate: float,
        iteration_ratio: float,
    ) -> QLearningParams:
        """Get parameters with additional performance-based adjustments.

        This is a Phase 2 extension that adjusts parameters based on current
        search performance. Can be enabled after validating Phase 1 (scale-only).

        Args:
            scale: Problem scale
            improvement_rate: Current improvement ratio (0.0 to 1.0)
            iteration_ratio: Progress through iterations (0.0 to 1.0)

        Returns:
            QLearningParams with both scale and performance adjustments

        Note:
            Currently returns same as get_params_for_scale. Performance-based
            adjustment logic can be implemented here in Phase 2.
        """
        # Phase 1: Only use scale-based params
        # Phase 2: Add performance-based dynamic adjustment logic here
        base_adjusted = self.get_params_for_scale(scale)

        # Placeholder for Phase 2 dynamic adjustments:
        # if improvement_rate < 0.2 and iteration_ratio < 0.5:
        #     # Poor performance early on: increase exploration
        #     return replace(base_adjusted, epsilon_min=base_adjusted.epsilon_min * 1.2)
        # elif improvement_rate > 0.4 and iteration_ratio > 0.5:
        #     # Good performance late: reduce exploration for convergence
        #     return replace(base_adjusted, epsilon_min=base_adjusted.epsilon_min * 0.8)

        return base_adjusted


# Convenience function for quick access
def get_adaptive_params(scale: ScaleType) -> QLearningParams:
    """Get adaptive Q-learning parameters for the specified scale.

    This is a convenience function that creates a manager instance and
    returns parameters for the requested scale.

    Args:
        scale: Problem scale ('small', 'medium', or 'large')

    Returns:
        QLearningParams object with scale-specific parameters

    Example:
        >>> params = get_adaptive_params("medium")
        >>> agent = QLearningOperatorAgent(..., params=params)
    """
    manager = AdaptiveQLearningParamsManager()
    return manager.get_params_for_scale(scale)


__all__ = [
    "AdaptiveQLearningParamsManager",
    "ScaleType",
    "get_adaptive_params",
]
