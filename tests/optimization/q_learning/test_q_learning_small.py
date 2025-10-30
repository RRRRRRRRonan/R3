"""Validate Q-learning adaptation on the small optimisation scenario."""

from tests.optimization.common import (
    ScenarioConfig,
    summarize_improvements,
)
from config import OPTIMIZATION_SCENARIO_PRESETS

from tests.optimization.q_learning.utils import run_q_learning_trial

SMALL_Q_CONFIG = ScenarioConfig.from_defaults(
    OPTIMIZATION_SCENARIO_PRESETS["small"],
    num_tasks=10,
)

ITERATIONS = 10


def test_q_learning_updates_values_and_improves_small_cost():
    alns, baseline_cost, optimised_cost = run_q_learning_trial(
        SMALL_Q_CONFIG,
        iterations=ITERATIONS,
    )

    assert alns._use_q_learning is True
    q_agent = getattr(alns, "_q_agent")
    assert q_agent is not None

    stats = q_agent.statistics()
    explore_usage = sum(stat.total_usage for stat in stats["explore"])
    assert explore_usage >= ITERATIONS
    assert any(stat.average_q_value > 0.0 for stat in stats["explore"])
    assert q_agent.epsilon < q_agent.params.initial_epsilon

    improvement = summarize_improvements([(baseline_cost, optimised_cost)])
    assert improvement > 0.0
