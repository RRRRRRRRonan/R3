"""Stress the Q-learning agent on the large optimisation scenario."""

from tests.optimization.common import ScenarioConfig
from config import OPTIMIZATION_SCENARIO_PRESETS

from tests.optimization.q_learning.utils import run_q_learning_trial

LARGE_Q_CONFIG = ScenarioConfig.from_defaults(
    OPTIMIZATION_SCENARIO_PRESETS["large"],
    num_tasks=30,
)

ITERATIONS = 18


def test_q_learning_scales_to_large_scenario():
    alns, baseline_cost, optimised_cost = run_q_learning_trial(
        LARGE_Q_CONFIG,
        iterations=ITERATIONS,
    )

    q_agent = getattr(alns, "_q_agent")
    assert q_agent is not None and alns._use_q_learning is True

    stats = q_agent.statistics()
    explore_stats = stats["explore"]
    explore_usage = sum(stat.total_usage for stat in explore_stats)
    assert explore_usage >= ITERATIONS
    assert any(stat.average_q_value > 0.0 for stat in explore_stats)
    assert q_agent.epsilon < q_agent.params.initial_epsilon

    assert optimised_cost <= baseline_cost
