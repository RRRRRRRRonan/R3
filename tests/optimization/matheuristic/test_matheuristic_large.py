"""Regression test covering the large optimisation scenario.

Exercises a sixteen-request scenario to confirm that optimisation still
delivers significant cost savings across charging strategies despite the larger
search space.
"""

from tests.optimization.common import (
    build_scenario,
    get_scale_config,
    get_solver_iterations,
    run_alns_trial,
    summarize_improvements,
)

from strategy.charging_strategies import (
    FullRechargeStrategy,
    PartialRechargeFixedStrategy,
    PartialRechargeMinimalStrategy,
)

SCALE = "large"
LARGE_CONFIG = get_scale_config(SCALE)
ITERATIONS = get_solver_iterations(SCALE, "matheuristic")

STRATEGIES = [
    ("Full", FullRechargeStrategy()),
    ("Fixed-60%", PartialRechargeFixedStrategy(charge_ratio=0.6)),
    (
        "Minimal",
        PartialRechargeMinimalStrategy(safety_margin=0.035, min_margin=0.015),
    ),
]


def test_large_scenario_strategies_improve_cost():
    scenario = build_scenario(LARGE_CONFIG)
    cost_pairs = []

    for name, strategy in STRATEGIES:
        initial_cost, optimized_cost = run_alns_trial(
            scenario,
            strategy,
            iterations=ITERATIONS,
        )
        assert optimized_cost < initial_cost, (
            f"{name} strategy failed to improve cost: "
            f"{initial_cost:.2f} → {optimized_cost:.2f}"
        )
        cost_pairs.append((initial_cost, optimized_cost))

    average_gain = summarize_improvements(cost_pairs)
    assert average_gain > 0.02
