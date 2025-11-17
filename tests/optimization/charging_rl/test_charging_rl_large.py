"""Regression test for contextual charging RL on the large scenario."""

from planner.q_learning import ChargingQLearningAgent
from tests.optimization.charging_rl.utils import run_contextual_charging_trial


def test_charging_rl_large_executes_and_limits_charge() -> None:
    result = run_contextual_charging_trial("large")

    assert result.executed_route.is_feasible
    assert result.charging_visits, "Expected at least one charging visit"

    charged = result.charging_visits[0].battery_after_service - result.charging_visits[0].battery_after_travel
    cap = result.vehicle.battery_capacity * ChargingQLearningAgent.ACTION_LEVELS[2]
    assert 0.0 < charged <= cap + 1e-6

    experiences = result.agent.consume_experiences()
    assert len(experiences) >= 1
