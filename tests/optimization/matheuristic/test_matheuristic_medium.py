"""Regression test covering the medium optimisation scenario.

Runs a trimmed twelve-request scenario, expecting each charging strategy to
deliver cost reductions while staying within the documented runtime budget.
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

SCALE = "medium"
MEDIUM_CONFIG = get_scale_config(SCALE)
ITERATIONS = get_solver_iterations(SCALE, "matheuristic")

STRATEGIES = [
    ("Full", FullRechargeStrategy()),
    ("Fixed-60%", PartialRechargeFixedStrategy(charge_ratio=0.6)),
    (
        "Minimal",
        PartialRechargeMinimalStrategy(safety_margin=0.03, min_margin=0.01),
    ),
]


def test_medium_scenario_strategies_improve_cost():
    scenario = build_scenario(MEDIUM_CONFIG)
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
    assert average_gain > 0.025
