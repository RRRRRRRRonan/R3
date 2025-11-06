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

        Parameter rationale (Phase 1.5c - Ultra-conservative Large + Optimized Medium):

        Small scale (15 tasks):
        - Higher alpha (0.35): Fast learning for limited search space
        - Moderate epsilon_min (0.08): Balanced exploration
        - Lower stagnation_ratio (0.12): Enter stuck earlier to leverage LP repair

        Medium scale (24 tasks):
        - Higher alpha (0.35): INCREASED from 0.30 - maximize learning speed
        - Moderate epsilon_min (0.12): Adequate exploration
        - Lower stagnation_ratio (0.15): DECREASED from 0.18 - even earlier LP repair

        Large scale (30+ tasks) - ULTRA-CONSERVATIVE to prevent collapse:
        - Lower alpha (0.12): DECREASED from 0.25 - prevent Q-value oscillation
        - Higher epsilon_min (0.15): Maintain exploration
        - Higher stagnation_ratio (0.40): INCREASED from 0.22 - much more exploration time

        Key insights from Phase 1.5 failures:
        - Seed 2034 Large collapsed (4.45%) with aggressive params (α=0.25, stag=0.22)
        - 2/3 Large seeds performed better with conservative params
        - Medium scale improvements need even more aggressive optimization
        - Large scale requires ULTRA-conservative approach (even more than Phase 1)

        Phase 1.5c changes:
        - Small: Keep stable (no change)
        - Medium: More aggressive (α 0.30→0.35, stag 0.18→0.15)
        - Large: ULTRA-conservative (α 0.25→0.12, stag 0.22→0.40)

        Args:
            scale: Problem scale

        Returns:
            Dictionary of parameter adjustments
        """
        if scale == "small":
            return {
                "alpha": 0.35,
                "epsilon_min": 0.08,
                "stagnation_ratio": 0.12,
                "deep_stagnation_ratio": 0.30,
            }
        elif scale == "medium":
            return {
                "alpha": 0.35,              # INCREASED from 0.30
                "epsilon_min": 0.12,
                "stagnation_ratio": 0.15,   # DECREASED from 0.18
                "deep_stagnation_ratio": 0.35,
            }
        else:  # large - ULTRA-CONSERVATIVE
            return {
                "alpha": 0.12,              # DECREASED from 0.25 (prevent collapse)
                "epsilon_min": 0.15,
                "stagnation_ratio": 0.40,   # INCREASED from 0.22 (more exploration)
                "deep_stagnation_ratio": 0.60,
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
