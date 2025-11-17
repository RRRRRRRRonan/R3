"""Unit tests for the Week 5 charging Q-learning components."""

from __future__ import annotations

from dataclasses import replace

from config import QLearningParams
from core.charging_context import ChargingContext, ChargingStateLevels
from physics.energy import EnergyConfig
from planner.q_learning import ChargingQLearningAgent
from strategy.charging_strategies import PartialRechargeMinimalStrategy


def _make_params() -> QLearningParams:
    """Return deterministic parameters for reproducible unit tests."""

    return replace(
        QLearningParams(),
        alpha=0.5,
        gamma=0.9,
        initial_epsilon=0.0,
        epsilon_decay=1.0,
        epsilon_min=0.0,
    )


def _make_uniform_q_values(best_index: int = 2) -> dict[str, list[float]]:
    """Populate every discrete state with the same preference vector."""

    q_values: dict[str, list[float]] = {}
    for battery in range(4):
        for slack in range(3):
            for density in range(3):
                state = f"b{battery}|s{slack}|d{density}"
                scores = [0.0 for _ in ChargingQLearningAgent.ACTION_LEVELS]
                scores[best_index] = 1.0
                q_values[state] = scores
    return q_values


def test_encode_state_label_matches_levels() -> None:
    levels = ChargingStateLevels(battery_level=0, slack_level=2, density_level=1)
    assert ChargingQLearningAgent.encode_state(levels) == "b0|s2|d1"


def test_agent_selects_highest_value_action_when_greedy() -> None:
    agent = ChargingQLearningAgent(
        _make_params(),
        initial_q_values=_make_uniform_q_values(best_index=3),
    )
    state = "b1|s1|d0"
    action_index, action_ratio = agent.select_action(state)
    assert action_index == 3
    assert action_ratio == ChargingQLearningAgent.ACTION_LEVELS[3]


def test_partial_recharge_strategy_caps_amount_and_logs_experience() -> None:
    params = _make_params()
    agent = ChargingQLearningAgent(
        params,
        initial_q_values=_make_uniform_q_values(best_index=2),
    )
    strategy = PartialRechargeMinimalStrategy(
        safety_margin=0.05,
        min_margin=0.02,
        charging_agent=agent,
        energy_config=EnergyConfig(battery_capacity=100.0, charging_rate=0.5),
    )
    context = ChargingContext(
        battery_ratio=0.1,
        demand_ratio=0.6,
        time_slack_ratio=0.1,
        station_density=0.05,
    )
    levels = ChargingStateLevels(battery_level=0, slack_level=1, density_level=1)

    amount = strategy.determine_charging_amount(
        current_battery=5.0,
        remaining_demand=70.0,
        battery_capacity=100.0,
        context=context,
        context_levels=levels,
    )

    assert 0.0 < amount <= 100.0 * ChargingQLearningAgent.ACTION_LEVELS[2] + 1e-6
    experiences = agent.consume_experiences()
    assert len(experiences) == 1
    state, action_index, reward, next_state = experiences[0]
    assert state == next_state == ChargingQLearningAgent.encode_state(levels)
    assert action_index == 2
    assert reward != 0.0
