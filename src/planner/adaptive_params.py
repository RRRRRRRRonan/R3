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

        Parameter rationale (Phase 1.5 - Recalibrated after initial testing):

        Small scale (15 tasks):
        - Higher alpha (0.35): Fast learning for limited search space
        - Moderate epsilon_min (0.08): Balanced exploration
        - Lower stagnation_ratio (0.12): Enter stuck earlier to leverage LP repair

        Medium scale (24 tasks):
        - Higher alpha (0.30): INCREASED from 0.2 - faster learning is critical
        - Moderate epsilon_min (0.12): Adequate exploration
        - Lower stagnation_ratio (0.18): DECREASED from 0.25 - earlier LP repair

        Large scale (30+ tasks):
        - Moderate alpha (0.25): INCREASED from 0.15 - balance learning speed
        - Higher epsilon_min (0.15): Maintain exploration in complex landscape
        - Lower stagnation_ratio (0.22): DECREASED from 0.35 - earlier LP repair

        Key changes from Phase 1:
        - All scales: DECREASED stagnation_ratio to trigger LP repair earlier
        - Medium/Large: INCREASED alpha for faster learning (critical fix)

        Args:
            scale: Problem scale

        Returns:
            Dictionary of parameter adjustments
        """
        if scale == "small":
            return {
                "alpha": 0.35,
                "epsilon_min": 0.08,
                "initial_epsilon": max(self.base_params.initial_epsilon, 0.12),
                "stagnation_ratio": 0.12,
                "deep_stagnation_ratio": 0.30,
            }
        elif scale == "medium":
            return {
                "alpha": 0.30,
                "epsilon_min": 0.12,
                "initial_epsilon": max(self.base_params.initial_epsilon, 0.15),
                "stagnation_ratio": 0.18,
                "deep_stagnation_ratio": 0.40,
            }
        else:  # large
            return {
                "alpha": 0.25,
                "epsilon_min": 0.15,
                "initial_epsilon": max(self.base_params.initial_epsilon, 0.18),
                "stagnation_ratio": 0.22,
                "deep_stagnation_ratio": 0.48,
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
        base_adjusted = self.get_params_for_scale(scale)

        improvement_rate = self._clamp(improvement_rate, 0.0, 1.0)
        iteration_ratio = self._clamp(iteration_ratio, 0.0, 1.0)

        alpha = base_adjusted.alpha
        epsilon_min = base_adjusted.epsilon_min
        epsilon_decay = base_adjusted.epsilon_decay
        stagnation_ratio = base_adjusted.stagnation_ratio
        deep_stagnation_ratio = base_adjusted.deep_stagnation_ratio

        # Empirical observation (10-seed regression, seeds 2025-2034): even强的求解
        # 组合整体改进幅度通常只有 3%-8%。此前以 18%/45% 为门槛的判据会把所有
        # 运行都当成“表现不佳”，导致 epsilon 长期维持在较高水平，而 alpha 反而被
        # 压低到 0.18，Q-learning 难以沉淀学习成果。这里把进展阈值对齐到真实数值
        # 范围，并修正调度逻辑：
        #   - 进展不足时提高学习率、延缓衰减并适度拉低卡滞阈值
        #   - 有持续改进时收紧探索并适度放宽阈值
        #   - 后期停滞时再度加大探索，同时保持学习率不要崩塌

        if improvement_rate < 0.03:
            # 明显欠收敛：提高探索和学习步长，提前触发LP修复。
            exploration_boost = 1.35 if iteration_ratio < 0.4 else 1.2
            epsilon_min = min(
                base_adjusted.initial_epsilon, epsilon_min * exploration_boost
            )
            epsilon_decay = min(0.999, epsilon_decay + 0.003)
            alpha = min(0.55, alpha * 1.12)
            stagnation_ratio = max(0.06, stagnation_ratio * 0.85)
            deep_stagnation_ratio = max(
                stagnation_ratio + 0.08, deep_stagnation_ratio * 0.88
            )
        elif improvement_rate > 0.12 and iteration_ratio > 0.35:
            # 稳定改善：降低随机性并逐步延长停滞阈值，保护已学偏好。
            epsilon_min = max(0.015, epsilon_min * 0.65)
            epsilon_decay = max(0.984, epsilon_decay - 0.0045)
            alpha = min(0.48, alpha * 1.05)
            stagnation_ratio = min(0.55, stagnation_ratio * 1.12)
            deep_stagnation_ratio = min(
                0.82,
                max(deep_stagnation_ratio * 1.08, stagnation_ratio + 0.14),
            )
        elif iteration_ratio > 0.65 and improvement_rate < 0.06:
            # 后期停滞：温和增加探索、维持足够学习率，防止陷入局部最优。
            epsilon_min = min(base_adjusted.initial_epsilon, epsilon_min * 1.15)
            epsilon_decay = min(0.999, epsilon_decay + 0.002)
            alpha = max(0.22, alpha * 0.97)
            stagnation_ratio = max(0.08, stagnation_ratio * 0.9)
            deep_stagnation_ratio = max(
                stagnation_ratio + 0.1, deep_stagnation_ratio * 0.93
            )

        return replace(
            base_adjusted,
            alpha=alpha,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            stagnation_ratio=stagnation_ratio,
            deep_stagnation_ratio=deep_stagnation_ratio,
        )

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


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
