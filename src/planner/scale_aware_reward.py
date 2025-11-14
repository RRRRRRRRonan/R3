"""
Scale-Aware Reward Normalization for Q-Learning ALNS (MVP VERSION)

This module implements reward normalization strategies that adapt to problem scale,
enabling stable Q-learning convergence across small, medium, and large problem instances.

MVP VERSION - Simplified to core features:
1. Scale-dependent reward amplification (core innovation)
2. Baseline-cost normalization for stability (changed from previous-cost)
3. Adaptive bonus/penalty scaling

Removed components (had issues in initial testing):
- Convergence bonus (gave unconditional rewards, polluted Q-learning signal)
- Variance penalty (unclear benefit, added complexity)

Author: SAQL Research Team
Date: 2025-11-13
Updated: 2025-11-14 (MVP simplification based on experimental feedback)
"""

from dataclasses import dataclass
from typing import Literal, Dict, Any


ScaleCategory = Literal["small", "medium", "large"]


@dataclass
class ScaleFactors:
    """Scale-dependent parameters for reward normalization."""

    scale_category: ScaleCategory
    reward_amplification: float
    variance_penalty_factor: float
    bonus_scale: float

    @staticmethod
    def from_num_requests(num_requests: int) -> "ScaleFactors":
        """
        Determine scale factors based on number of requests.

        Thresholds:
        - Small: ≤18 tasks (covers typical 15-task instances)
        - Medium: 19-27 tasks (covers typical 24-task instances)
        - Large: ≥28 tasks (covers typical 30-task instances)

        Args:
            num_requests: Number of pickup-delivery request pairs

        Returns:
            ScaleFactors instance with appropriate parameters
        """
        if num_requests <= 18:
            return ScaleFactors(
                scale_category="small",
                reward_amplification=1.0,
                variance_penalty_factor=1.0,
                bonus_scale=1.0,
            )
        elif num_requests <= 27:
            return ScaleFactors(
                scale_category="medium",
                reward_amplification=1.3,
                variance_penalty_factor=1.2,
                bonus_scale=1.3,
            )
        else:  # num_requests >= 28
            return ScaleFactors(
                scale_category="large",
                reward_amplification=1.6,
                variance_penalty_factor=1.5,
                bonus_scale=1.6,
            )


@dataclass
class RewardComponents:
    """Breakdown of reward components for analysis."""

    improvement_reward: float
    global_best_bonus: float
    convergence_bonus: float
    time_penalty: float
    variance_penalty: float
    total_reward: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "improvement_reward": self.improvement_reward,
            "global_best_bonus": self.global_best_bonus,
            "convergence_bonus": self.convergence_bonus,
            "time_penalty": self.time_penalty,
            "variance_penalty": self.variance_penalty,
            "total_reward": self.total_reward,
        }


class ScaleAwareRewardCalculator:
    """
    Computes scale-aware normalized rewards for Q-learning operator selection.

    This class implements the reward normalization strategy that addresses
    scale-dependent performance degradation in Q-learning ALNS.
    """

    def __init__(
        self,
        num_requests: int,
        baseline_cost: float,
        max_iterations: int,
        *,
        base_scale: float = 100_000.0,
        enable_variance_penalty: bool = True,
        enable_convergence_bonus: bool = True,
    ):
        """
        Initialize reward calculator.

        Args:
            num_requests: Number of pickup-delivery pairs in the problem
            baseline_cost: Initial solution cost (for convergence tracking)
            max_iterations: Maximum ALNS iterations (for time penalty)
            base_scale: Base scaling factor for rewards (default: 100,000)
            enable_variance_penalty: Whether to apply variance penalty
            enable_convergence_bonus: Whether to apply convergence bonus
        """
        self.num_requests = num_requests
        self.baseline_cost = baseline_cost
        self.max_iterations = max_iterations
        self.base_scale = base_scale
        self.enable_variance_penalty = enable_variance_penalty
        self.enable_convergence_bonus = enable_convergence_bonus

        # Determine scale factors
        self.scale_factors = ScaleFactors.from_num_requests(num_requests)

    def compute_reward(
        self,
        improvement: float,
        previous_cost: float,
        is_new_global_best: bool,
        iteration: int,
    ) -> float:
        """
        Compute scale-aware reward for a single ALNS iteration.

        Args:
            improvement: Cost reduction (previous_cost - new_cost)
            previous_cost: Solution cost before this iteration
            is_new_global_best: Whether this iteration found new global best
            iteration: Current iteration number (0-indexed)

        Returns:
            Normalized reward value
        """
        components = self.compute_reward_components(
            improvement=improvement,
            previous_cost=previous_cost,
            is_new_global_best=is_new_global_best,
            iteration=iteration,
        )
        return components.total_reward

    def compute_reward_components(
        self,
        improvement: float,
        previous_cost: float,
        is_new_global_best: bool,
        iteration: int,
    ) -> RewardComponents:
        """
        Compute reward with detailed component breakdown.

        MVP VERSION: Simplified to only 3 core components

        Key changes from original design:
        - Changed: Use baseline_cost instead of previous_cost for normalization (more stable)
        - Removed: convergence_bonus (had design flaw - gave bonus unconditionally)
        - Removed: variance_penalty (unclear benefit, added complexity)

        This MVP focuses on the core hypothesis: scale-dependent reward amplification
        improves Q-learning performance on large-scale problems.

        Returns:
            RewardComponents with individual component values
        """
        # Component 1: Normalized improvement reward
        # MVP: Changed from previous_cost to baseline_cost for stability
        if improvement > 0:
            relative_improvement = improvement / self.baseline_cost  # MVP: fixed denominator
            improvement_reward = (
                relative_improvement
                * self.base_scale
                * self.scale_factors.reward_amplification
            )
        else:
            improvement_reward = 0.0

        # Component 2: Global best bonus
        # MVP: Kept as-is, scaled by problem size
        if is_new_global_best:
            base_bonus = 50.0
            global_best_bonus = base_bonus * self.scale_factors.bonus_scale
        else:
            global_best_bonus = 0.0

        # Component 3: Time penalty
        # MVP: Kept as-is, encourages early improvements
        iteration_progress = iteration / self.max_iterations
        base_time_penalty = iteration_progress * 15.0
        time_penalty = base_time_penalty * self.scale_factors.bonus_scale

        # MVP: Removed convergence bonus (always 0)
        # Original issue: gave ~20 reward even for tiny improvements, polluted Q-learning signal
        convergence_bonus = 0.0

        # MVP: Removed variance penalty (always 0)
        # Original issue: unclear benefit, added complexity without proven value
        variance_penalty = 0.0

        # Total reward (MVP: simplified to 3 components)
        total_reward = (
            improvement_reward
            + global_best_bonus
            - time_penalty
        )

        return RewardComponents(
            improvement_reward=improvement_reward,
            global_best_bonus=global_best_bonus,
            convergence_bonus=convergence_bonus,  # Always 0 in MVP
            time_penalty=time_penalty,
            variance_penalty=variance_penalty,  # Always 0 in MVP
            total_reward=total_reward,
        )

    def compute_no_change_reward(self, iteration: int) -> float:
        """
        Compute reward for iterations with no cost change.

        Args:
            iteration: Current iteration number

        Returns:
            Small positive reward (scaled to problem size)
        """
        base_reward = 10.0
        return base_reward * self.scale_factors.bonus_scale

    def compute_worsening_penalty(self, iteration: int) -> float:
        """
        Compute penalty for iterations that worsen the solution.

        Args:
            iteration: Current iteration number

        Returns:
            Negative reward (penalty, scaled to problem size)
        """
        base_penalty = -5.0
        return base_penalty * self.scale_factors.bonus_scale

    def get_scale_category(self) -> ScaleCategory:
        """Return the problem scale category."""
        return self.scale_factors.scale_category

    def get_scale_info(self) -> Dict[str, Any]:
        """Return scale configuration for logging."""
        return {
            "num_requests": self.num_requests,
            "scale_category": self.scale_factors.scale_category,
            "reward_amplification": self.scale_factors.reward_amplification,
            "variance_penalty_factor": self.scale_factors.variance_penalty_factor,
            "bonus_scale": self.scale_factors.bonus_scale,
            "baseline_cost": self.baseline_cost,
            "max_iterations": self.max_iterations,
        }
