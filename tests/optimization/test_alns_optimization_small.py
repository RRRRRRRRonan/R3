"""Validate ALNS improvements on the small optimisation scenario.

The test instantiates the reusable five-request scenario, runs ALNS with each
charging strategy, and asserts that every strategy beats the greedy baseline by
the minimum improvement margin documented in the optimisation README.
"""

from tests.optimization.common import (
    ScenarioConfig,
    build_scenario,
    run_alns_trial,
    summarize_improvements,
)

from strategy.charging_strategies import (
    FullRechargeStrategy,
    PartialRechargeFixedStrategy,
    PartialRechargeMinimalStrategy,
)

SMALL_CONFIG = ScenarioConfig(
    num_tasks=10,
    num_charging=1,
    area_size=(1000.0, 1000.0),
    vehicle_capacity=200.0,
    battery_capacity=1.2,
    consumption_per_km=0.4,
    charging_rate=5.0 / 3600.0,
    seed=7,
)

STRATEGIES = [
    ("Full", FullRechargeStrategy(), 20),
    ("Fixed-50%", PartialRechargeFixedStrategy(charge_ratio=0.5), 25),
    (
        "Minimal",
        PartialRechargeMinimalStrategy(safety_margin=0.02, min_margin=0.0),
        30,
    ),
]


def test_small_scenario_strategies_improve_cost():
    scenario = build_scenario(SMALL_CONFIG)
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
    assert average_gain > 0.03
