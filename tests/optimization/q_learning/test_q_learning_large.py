"""Stress the Q-learning agent on the large optimisation scenario."""

from tests.optimization.common import get_scale_config, get_solver_iterations

from tests.optimization.q_learning.utils import run_q_learning_trial

SCALE = "large"
LARGE_Q_CONFIG = get_scale_config(SCALE)
ITERATIONS = get_solver_iterations(SCALE, "q_learning")


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
