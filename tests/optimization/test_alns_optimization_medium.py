"""Regression test covering the medium optimisation scenario.

Runs a trimmed twelve-request scenario, expecting each charging strategy to
deliver cost reductions while staying within the documented runtime budget.
"""

from tests.optimization.common import (
    ScenarioConfig,
    build_scenario,
    run_alns_trial,
    summarize_improvements,
)
from config import OPTIMIZATION_SCENARIO_PRESETS

from strategy.charging_strategies import (
    FullRechargeStrategy,
    PartialRechargeFixedStrategy,
    PartialRechargeMinimalStrategy,
)

MEDIUM_CONFIG = ScenarioConfig.from_defaults(
    OPTIMIZATION_SCENARIO_PRESETS["medium"],
    num_tasks=12,
)

STRATEGIES = [
    ("Full", FullRechargeStrategy(), 6),
    ("Fixed-60%", PartialRechargeFixedStrategy(charge_ratio=0.6), 8),
    (
        "Minimal",
        PartialRechargeMinimalStrategy(safety_margin=0.03, min_margin=0.01),
        10,
    ),
]


def test_medium_scenario_strategies_improve_cost():
    scenario = build_scenario(MEDIUM_CONFIG)
    cost_pairs = []

    for name, strategy, iterations in STRATEGIES:
        initial_cost, optimized_cost = run_alns_trial(
            scenario,
            strategy,
            iterations=iterations,
        )
        assert optimized_cost < initial_cost, (
            f"{name} strategy failed to improve cost: "
            f"{initial_cost:.2f} → {optimized_cost:.2f}"
        )
        cost_pairs.append((initial_cost, optimized_cost))

    average_gain = summarize_improvements(cost_pairs)
    assert average_gain > 0.025
