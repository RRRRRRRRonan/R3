"""Scenario synthesizer for event-driven simulation."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence

from baselines.mip.config import ScenarioSynthConfig
from baselines.mip.model import MIPBaselineScenario
from core.node import ChargingNode
from core.task import TaskPool


def synthesize_scenarios(
    task_pool: TaskPool,
    *,
    chargers: Optional[Sequence[ChargingNode]] = None,
    seed: Optional[int] = None,
    config: Optional[ScenarioSynthConfig] = None,
) -> List[MIPBaselineScenario]:
    """Create a list of synthetic scenarios from the current task pool."""

    cfg = config or ScenarioSynthConfig()
    rng = random.Random(seed)
    tasks = list(task_pool.get_all_tasks())
    if not tasks:
        return [
            MIPBaselineScenario(
                scenario_id=0,
                probability=1.0,
                task_availability={},
            )
        ]

    num = max(1, int(cfg.num_scenarios))
    scenarios: List[MIPBaselineScenario] = []
    prob = 1.0 / num

    for scenario_id in range(num):
        availability: Dict[int, int] = {}
        release_times: Dict[int, float] = {}
        demands: Dict[int, float] = {}

        for task in tasks:
            is_available = 1 if rng.random() <= cfg.availability_prob else 0
            availability[task.task_id] = is_available
            base_release = float(task.arrival_time)
            jitter = 0.0
            if cfg.release_jitter_s > 0.0:
                jitter = rng.uniform(-cfg.release_jitter_s, cfg.release_jitter_s)
            release_times[task.task_id] = max(0.0, base_release + jitter)

            base_demand = float(task.demand)
            if cfg.demand_noise_ratio > 0.0:
                noise = rng.uniform(-cfg.demand_noise_ratio, cfg.demand_noise_ratio)
                base_demand = max(0.0, base_demand * (1.0 + noise))
            demands[task.task_id] = base_demand

        # Ensure at least one task is available.
        if all(value == 0 for value in availability.values()):
            availability[rng.choice(tasks).task_id] = 1

        queue_estimates: Dict[int, float] = {}
        charging_availability: Dict[int, int] = {}
        if chargers:
            for charger in chargers:
                queue_estimates[charger.node_id] = rng.uniform(
                    cfg.queue_time_range_s[0], cfg.queue_time_range_s[1]
                )
                charging_availability[charger.node_id] = (
                    1 if rng.random() <= cfg.charging_available_prob else 0
                )

        travel_time_factor = rng.uniform(
            cfg.travel_time_factor_range[0], cfg.travel_time_factor_range[1]
        )

        decision_epoch_times = sorted(set(release_times.values()))
        if 0.0 not in decision_epoch_times:
            decision_epoch_times.insert(0, 0.0)

        scenarios.append(
            MIPBaselineScenario(
                scenario_id=scenario_id,
                probability=prob,
                task_availability=availability,
                task_release_times=release_times,
                task_demands=demands,
                queue_estimates_s=queue_estimates,
                travel_time_factor=travel_time_factor,
                charging_availability=charging_availability,
                decision_epoch_times=decision_epoch_times,
            )
        )

    return scenarios


__all__ = ["ScenarioSynthConfig", "synthesize_scenarios"]
