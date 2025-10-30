"""Ensure the Minimal ALNS solver improves the unified medium scenario."""

from tests.optimization.common import (
    build_scenario,
    get_scale_config,
    get_solver_iterations,
    run_minimal_trial,
    summarize_improvements,
)

SCALE = "medium"
CONFIG = get_scale_config(SCALE)
ITERATIONS = get_solver_iterations(SCALE, "minimal")


def test_minimal_alns_improves_medium_cost():
    scenario = build_scenario(CONFIG)
    initial_cost, optimised_cost = run_minimal_trial(
        scenario,
        iterations=ITERATIONS,
    )

    assert optimised_cost <= initial_cost
    improvement = summarize_improvements([(initial_cost, optimised_cost)])
    assert improvement > 0.0
