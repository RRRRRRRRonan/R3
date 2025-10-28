"""Regression test covering the medium optimisation scenario.

Runs the shared ten-request scenario, expecting each charging strategy to
deliver cost reductions while staying within the documented runtime budget.
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

MEDIUM_CONFIG = ScenarioConfig(
    num_tasks=30,
    num_charging=2,
    area_size=(2000.0, 2000.0),
    vehicle_capacity=220.0,
    battery_capacity=1.6,
    consumption_per_km=0.45,
    charging_rate=5.5 / 3600.0,
    seed=11,
)

STRATEGIES = [
    ("Full", FullRechargeStrategy(), 55),
    ("Fixed-60%", PartialRechargeFixedStrategy(charge_ratio=0.6), 65),
    (
        "Minimal",
        PartialRechargeMinimalStrategy(safety_margin=0.03, min_margin=0.01),
        75,
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
