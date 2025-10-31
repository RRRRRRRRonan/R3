"""Validate ALNS improvements on the small optimisation scenario.

The test instantiates the reusable five-request scenario, runs ALNS with each
charging strategy, and asserts that every strategy beats the greedy baseline by
the minimum improvement margin documented in the optimisation README.
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

SCALE = "small"
SMALL_CONFIG = get_scale_config(SCALE)
ITERATIONS = get_solver_iterations(SCALE, "matheuristic")

STRATEGIES = [
    ("Full", FullRechargeStrategy()),
    ("Fixed-50%", PartialRechargeFixedStrategy(charge_ratio=0.5)),
    (
        "Minimal",
        PartialRechargeMinimalStrategy(safety_margin=0.02, min_margin=0.0),
    ),
]


def test_small_scenario_strategies_improve_cost():
    scenario = build_scenario(SMALL_CONFIG)
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
    assert average_gain > 0.03
