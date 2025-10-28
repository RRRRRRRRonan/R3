"""Regression test covering the large optimisation scenario.

Exercises the fifteen-request scenario to confirm that optimisation still
delivers significant cost savings across charging strategies despite the larger
search space.
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

LARGE_CONFIG = ScenarioConfig(
    num_tasks=50,
    num_charging=3,
    area_size=(3000.0, 3000.0),
    vehicle_capacity=240.0,
    battery_capacity=2.0,
    consumption_per_km=0.5,
    charging_rate=6.0 / 3600.0,
    seed=17,
)

STRATEGIES = [
    ("Full", FullRechargeStrategy(), 70),
    ("Fixed-60%", PartialRechargeFixedStrategy(charge_ratio=0.6), 80),
    (
        "Minimal",
        PartialRechargeMinimalStrategy(safety_margin=0.035, min_margin=0.015),
        90,
    ),
]


def test_large_scenario_strategies_improve_cost():
    scenario = build_scenario(LARGE_CONFIG)
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
    assert average_gain > 0.02
