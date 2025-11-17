"""Charging policies used by the ALNS planner and simulator benchmarks.

The module defines three baseline strategies—full recharge, fixed-ratio partial
recharge, and minimum feasible recharge—each exposing a unified interface for
charging amount selection and warning threshold feedback.  Optimisation tests
and documentation rely on these implementations to compare how strategy choice
impacts route cost, charging frequency, and robustness.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from core.charging_context import ChargingContext, ChargingStateLevels
from physics.energy import calculate_minimum_charging_needed
from physics.energy import EnergyConfig

if TYPE_CHECKING:  # Avoid heavy import cycles at runtime
    from planner.q_learning import ChargingQLearningAgent


class ChargingStrategy(ABC):
    """
    充电策略抽象基类

    定义充电策略的接口规范
    """

    @abstractmethod
    def determine_charging_amount(self,
                                  current_battery: float,
                                  remaining_demand: float,
                                  battery_capacity: float,
                                  context: Optional[ChargingContext] = None,
                                  context_levels: Optional[ChargingStateLevels] = None) -> float:
        """
        确定充电量

        参数:
            current_battery: 当前电量 (kWh)
            remaining_demand: 剩余路径的能量需求 (kWh)
            battery_capacity: 电池容量 (kWh)

        返回:
            float: 应该充电的量 (kWh)
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """返回策略名称"""
        pass

    def get_warning_threshold(self) -> float:
        """
        获取策略感知的警告阈值（相对于电池容量的比例）

        Week 4新增：每个充电策略根据自身特性返回合适的警告阈值

        返回:
            float: 警告阈值比例 (0-1)，低于此值建议充电

        设计思想:
            - FR策略: 低阈值(10%)，因为充满后续航长
            - PR-Fixed策略: 中等阈值(10-12%)，根据充电比例调整
            - PR-Minimal策略: 高阈值(15%)，因为每次只充刚好够用
        """
        return 0.15  # 默认值：15%


class FullRechargeStrategy(ChargingStrategy):
    """
    完全充电策略 (FR)

    策略描述:
        每次访问充电站都充满100%

    特点:
        - 最简单的策略
        - 充电时间最长
        - 充电次数可能最少
        - 适合充电站稀疏的场景

    用途:
        作为基准策略进行对比
    """

    def determine_charging_amount(self,
                                  current_battery: float,
                                  remaining_demand: float,
                                  battery_capacity: float,
                                  context: Optional[ChargingContext] = None,
                                  context_levels: Optional[ChargingStateLevels] = None) -> float:
        """
        充满电池

        返回:
            充电量 = 电池容量 - 当前电量
        """
        return battery_capacity - current_battery

    def get_strategy_name(self) -> str:
        return "Full Recharge (FR)"

    def get_warning_threshold(self) -> float:
        """
        FR策略：低警告阈值(10%)

        原因: 每次充满电，续航时间长，可以容忍较低的电量
        """
        return 0.10


class PartialRechargeFixedStrategy(ChargingStrategy):
    """
    固定比例局部充电 (PR-Fixed) - Keskin 2016

    策略描述:
        每次充电到电池容量的固定百分比
        例如: charge_ratio=0.3 表示充到30%

    特点:
        - 充电时间固定且较短
        - 可能需要更频繁的充电
        - 适合充电站密集的场景

    参数:
        charge_ratio: 充电比例 (0-1)
            - 0.3: 充到30% (快速充电)
            - 0.5: 充到50% (平衡)
            - 0.8: 充到80% (接近完全充电)
    """

    def __init__(self, charge_ratio: float = 0.3):
        """
        初始化

        参数:
            charge_ratio: 充电比例，默认0.3 (30%)
        """
        if not 0 < charge_ratio <= 1.0:
            raise ValueError(f"charge_ratio必须在(0, 1]范围内，当前值: {charge_ratio}")

        self.charge_ratio = charge_ratio

    def determine_charging_amount(self,
                                  current_battery: float,
                                  remaining_demand: float,
                                  battery_capacity: float,
                                  context: Optional[ChargingContext] = None,
                                  context_levels: Optional[ChargingStateLevels] = None) -> float:
        """
        充电到电池容量的固定百分比

        示例:
            电池容量 100kWh, 当前20kWh, 目标30%
            → 充到30kWh → 充电量10kWh
        """
        target_level = battery_capacity * self.charge_ratio

        # 如果当前电量已经高于目标，不充电
        if current_battery >= target_level:
            return 0.0

        return target_level - current_battery

    def get_strategy_name(self) -> str:
        return f"Partial Recharge Fixed (PR-Fixed {self.charge_ratio*100:.0f}%)"

    def get_warning_threshold(self) -> float:
        """
        PR-Fixed策略：动态警告阈值

        根据充电比例调整:
            - 充电比例高(80%) → 低阈值(10%)，因为每次充电多
            - 充电比例低(30%) → 高阈值(12%)，因为每次充电少

        计算公式: 基础10% + (1 - 充电比例) * 5%
        """
        # 充电比例越低，阈值越高
        return 0.10 + (1.0 - self.charge_ratio) * 0.05


class PartialRechargeMinimalStrategy(ChargingStrategy):
    """
    最小充电策略 (PR-Minimal) - Keskin 2016基础版

    策略描述:
        只充刚好够用的电量 + 安全余量

    特点:
        - 充电时间最短
        - 根据实际需求动态调整
        - 需要准确的能量预测
        - 适合已知路径的静态规划

    参数:
        safety_margin: 基础安全余量比例 (0-1)
            - 作为动态余量的上限，结合剩余需求自动缩放
        min_margin: 自适应最小安全余量比例
        max_margin: 自适应最大安全余量比例

    示例:
        当前电量: 20kWh
        剩余需求: 80kWh
        安全余量: 10kWh
        → 充电量 = 80 - 20 + 10 = 70kWh
    """

    def __init__(self, safety_margin: float = 0.1,
                 min_margin: float = 0.015,
                 max_margin: float = 0.18,
                 *,
                 charging_agent: Optional["ChargingQLearningAgent"] = None,
                 energy_config: Optional[EnergyConfig] = None):
        """
        初始化

        参数:
            safety_margin: 基础安全余量比例，默认0.1 (10%)
            min_margin: 自适应最小安全余量（默认1.5%）
            max_margin: 自适应最大安全余量（默认18%）
        """
        if not 0 <= safety_margin < 1.0:
            raise ValueError(f"safety_margin必须在[0, 1)范围内，当前值: {safety_margin}")
        if min_margin < 0 or max_margin <= 0 or min_margin >= max_margin:
            raise ValueError("min_margin/max_margin必须满足 0 ≤ min < max")

        self.base_margin = safety_margin
        self.min_margin = min_margin
        self.max_margin = max(max_margin, safety_margin)
        self.charging_agent = charging_agent
        self.energy_config = energy_config or EnergyConfig()

    BATTERY_LEVEL_MARGINS = (0.18, 0.12, 0.08, 0.05)
    SLACK_MULTIPLIERS = (1.25, 1.0, 0.85)
    DENSITY_MULTIPLIERS = (1.25, 1.0, 0.9)
    RL_LATENESS_WEIGHT = 20.0
    RL_SAFETY_BONUS = 4.0

    def determine_charging_amount(self,
                                  current_battery: float,
                                  remaining_demand: float,
                                  battery_capacity: float,
                                  context: Optional[ChargingContext] = None,
                                  context_levels: Optional[ChargingStateLevels] = None) -> float:
        """上下文感知的最小充电量。"""

        if context is None or context_levels is None:
            base_charge = self._determine_baseline(current_battery, remaining_demand, battery_capacity)
            return base_charge

        base_need = max(0.0, remaining_demand - current_battery)

        margin_ratio = self._margin_ratio_from_levels(context_levels)
        slack_multiplier = self.SLACK_MULTIPLIERS[min(context_levels.slack_level, len(self.SLACK_MULTIPLIERS) - 1)]
        density_multiplier = self.DENSITY_MULTIPLIERS[min(context_levels.density_level, len(self.DENSITY_MULTIPLIERS) - 1)]

        contextual_margin_ratio = margin_ratio * slack_multiplier * density_multiplier
        contextual_margin_ratio = max(self.min_margin, min(self.max_margin, contextual_margin_ratio))

        margin = battery_capacity * contextual_margin_ratio
        charge = max(0.0, base_need + margin)
        base_charge = min(charge, battery_capacity - current_battery)

        if not self.charging_agent:
            return base_charge

        return self._apply_q_learning_override(
            base_charge=base_charge,
            current_battery=current_battery,
            battery_capacity=battery_capacity,
            remaining_demand=remaining_demand,
            context=context,
            context_levels=context_levels,
        )

    def _margin_ratio_from_levels(self, context_levels: ChargingStateLevels) -> float:
        level = min(context_levels.battery_level, len(self.BATTERY_LEVEL_MARGINS) - 1)
        return self.BATTERY_LEVEL_MARGINS[level]

    def _determine_baseline(self,
                            current_battery: float,
                            remaining_demand: float,
                            battery_capacity: float) -> float:
        demand_ratio = 0.0 if battery_capacity <= 0 else remaining_demand / battery_capacity
        demand_ratio = max(0.0, min(1.0, demand_ratio))

        adaptive_ratio = self.base_margin * (0.5 + 0.5 * demand_ratio)
        adaptive_ratio = max(self.min_margin, min(self.max_margin, adaptive_ratio))

        margin = battery_capacity * adaptive_ratio

        return calculate_minimum_charging_needed(
            current_battery=current_battery,
            remaining_energy_demand=remaining_demand,
            battery_capacity=battery_capacity,
            safety_margin=margin
        )

    # ------------------------------------------------------------------
    def _apply_q_learning_override(self,
                                   *,
                                   base_charge: float,
                                   current_battery: float,
                                   battery_capacity: float,
                                   remaining_demand: float,
                                   context: ChargingContext,
                                   context_levels: ChargingStateLevels) -> float:
        """Blend heuristic + RL advice (Week5 §2.2)."""

        agent = self.charging_agent
        if agent is None:
            return base_charge

        state_label = agent.encode_state(context_levels)
        action_index, action_level = agent.select_action(state_label)
        rl_cap = action_level * battery_capacity

        if action_level <= 0.0:
            final_amount = 0.0
        elif action_level >= 0.999:
            final_amount = max(base_charge, rl_cap)
        else:
            final_amount = min(base_charge, max(0.0, rl_cap))

        final_amount = max(0.0, min(final_amount, battery_capacity - current_battery))

        reward = self._calculate_rl_reward(
            context=context,
            final_amount=final_amount,
            remaining_demand=remaining_demand,
            battery_capacity=battery_capacity,
            current_battery=current_battery,
        )
        agent.update(state_label, action_index, reward, state_label)
        agent.decay_epsilon()
        return final_amount

    def _calculate_rl_reward(self,
                             *,
                             context: ChargingContext,
                             final_amount: float,
                             remaining_demand: float,
                             battery_capacity: float,
                             current_battery: float) -> float:
        """Reward shaping mirroring Week5 §2.3 instructions."""

        charging_rate = getattr(self.energy_config, "charging_rate", 1.0)
        charging_time_penalty = final_amount / max(1e-6, charging_rate)

        lateness_penalty = max(0.0, -context.time_slack_ratio) * self.RL_LATENESS_WEIGHT

        next_battery_ratio = 0.0
        if battery_capacity > 0:
            next_battery_ratio = min(1.0, (current_battery + final_amount) / battery_capacity)

        demand_ratio = 0.0 if battery_capacity <= 0 else remaining_demand / battery_capacity
        projected_buffer = next_battery_ratio - demand_ratio
        safety_threshold = getattr(self.energy_config, "safety_threshold", 0.05)
        safety_reward = self.RL_SAFETY_BONUS if projected_buffer >= safety_threshold else -self.RL_SAFETY_BONUS

        return -charging_time_penalty - lateness_penalty + safety_reward

    def get_strategy_name(self) -> str:
        return f"Partial Recharge Minimal (PR-Minimal {self.base_margin*100:.0f}%)"

    def get_warning_threshold(self) -> float:
        """
        PR-Minimal策略：高警告阈值(15%)

        原因:
            - 每次只充刚好够用的电量，没有太多余量
            - 需要更早触发充电站插入，避免电量不足
            - 需要较高的安全余量以应对能量预测误差

        计算: 基础15% + 安全余量的一半
        """
        return max(0.05, min(0.16, self.base_margin * 0.3 + 0.05))


# ========== 便捷工厂函数 ==========

def create_charging_strategy(strategy_type: str, **kwargs) -> ChargingStrategy:
    """
    工厂函数：根据类型创建充电策略

    参数:
        strategy_type: 策略类型
            - 'FR': Full Recharge
            - 'PR-Fixed': Partial Recharge Fixed
            - 'PR-Minimal': Partial Recharge Minimal
        **kwargs: 策略特定参数
            - charge_ratio: PR-Fixed的充电比例
            - safety_margin: PR-Minimal的安全余量

    返回:
        ChargingStrategy: 充电策略实例

    示例:
        >>> fr = create_charging_strategy('FR')
        >>> pr_fixed = create_charging_strategy('PR-Fixed', charge_ratio=0.3)
        >>> pr_minimal = create_charging_strategy('PR-Minimal', safety_margin=0.1)
    """
    strategy_map = {
        'FR': FullRechargeStrategy,
        'PR-Fixed': PartialRechargeFixedStrategy,
        'PR-Minimal': PartialRechargeMinimalStrategy
    }

    if strategy_type not in strategy_map:
        raise ValueError(f"未知策略类型: {strategy_type}. 可选: {list(strategy_map.keys())}")

    return strategy_map[strategy_type](**kwargs)


# ========== 策略对比工具 ==========

def compare_strategies(current_battery: float,
                      remaining_demand: float,
                      battery_capacity: float) -> dict:
    """
    对比不同策略的充电量

    参数:
        current_battery: 当前电量
        remaining_demand: 剩余需求
        battery_capacity: 电池容量

    返回:
        dict: 各策略的充电量对比
    """
    strategies = [
        FullRechargeStrategy(),
        PartialRechargeFixedStrategy(charge_ratio=0.3),
        PartialRechargeFixedStrategy(charge_ratio=0.5),
        PartialRechargeMinimalStrategy(safety_margin=0.1)
    ]

    results = {}
    for strategy in strategies:
        amount = strategy.determine_charging_amount(
            current_battery, remaining_demand, battery_capacity
        )
        results[strategy.get_strategy_name()] = {
            'charging_amount': amount,
            'final_battery': current_battery + amount
        }

    return results
