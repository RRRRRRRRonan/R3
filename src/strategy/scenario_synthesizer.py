"""Scenario synthesizer for event-driven simulation.

This module turns a static ``TaskPool`` (known coordinates + node IDs) into one
or more *dynamic* demand scenarios by sampling:
- task availability indicators
- task release times (NHPP arrival model)
- task demands (truncated normal)
- node time windows anchored to release time

The defaults are chosen to align with the paper Section 5.1 experimental setup.
"""

from __future__ import annotations

import random
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

from baselines.mip.config import ScenarioSynthConfig
from baselines.mip.model import MIPBaselineScenario
from config.arrival_model import build_default_nhpp_model
from core.node import ChargingNode
from core.task import TaskPool
from physics.distance import euclidean_distance
from physics.time import calculate_travel_time


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
        node_time_windows: Dict[int, Tuple[float, float]] = {}
        node_service_times: Dict[int, float] = {}
        demands: Dict[int, float] = {}

        sampled_release_times = _sample_release_times(tasks, rng, cfg)

        for task in tasks:
            is_available = 1 if rng.random() <= cfg.availability_prob else 0
            availability[task.task_id] = is_available
            release_times[task.task_id] = float(
                sampled_release_times.get(task.task_id, task.arrival_time)
            )
            demands[task.task_id] = float(_sample_task_demand(task, rng, cfg))

        if cfg.use_release_time_windows:
            node_time_windows, node_service_times = _build_time_window_overrides(
                tasks,
                release_times,
                cfg,
            )

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

        decision_epoch_times = sorted(
            set(
                release_times[task_id]
                for task_id, is_available in availability.items()
                if is_available > 0
            )
        )
        if 0.0 not in decision_epoch_times:
            decision_epoch_times.insert(0, 0.0)

        scenarios.append(
            MIPBaselineScenario(
                scenario_id=scenario_id,
                probability=prob,
                task_availability=availability,
                task_release_times=release_times,
                node_time_windows=node_time_windows,
                node_service_times=node_service_times,
                task_demands=demands,
                queue_estimates_s=queue_estimates,
                travel_time_factor=travel_time_factor,
                charging_availability=charging_availability,
                decision_epoch_times=decision_epoch_times,
            )
        )

    return scenarios


def _sample_release_times(
    tasks,
    rng: random.Random,
    cfg: ScenarioSynthConfig,
) -> Dict[int, float]:
    """Return per-task release times for a single scenario."""

    if not cfg.use_nhpp_arrivals:
        release_times: Dict[int, float] = {}
        for task in tasks:
            base_release = float(task.arrival_time)
            jitter = 0.0
            if cfg.release_jitter_s > 0.0:
                jitter = rng.uniform(-cfg.release_jitter_s, cfg.release_jitter_s)
            release_times[task.task_id] = max(0.0, base_release + jitter)
        return release_times

    model = build_default_nhpp_model(
        episode_length_s=cfg.episode_length_s,
        base_rate_per_s=cfg.nhpp_base_rate_per_s,
        peak_multiplier=cfg.nhpp_peak_multiplier,
        normal_multiplier=cfg.nhpp_normal_multiplier,
        offpeak_multiplier=cfg.nhpp_offpeak_multiplier,
        segment_fractions=cfg.nhpp_segment_fractions,
    )

    mode = str(cfg.arrival_time_sampling_mode).strip().lower()
    if mode == "fixed_count":
        times = model.sample_arrivals_fixed_count(
            len(tasks),
            rng,
            horizon_s=cfg.episode_length_s,
        )
    elif mode == "thinning":
        sampled = model.sample_arrivals_thinning(rng, horizon_s=cfg.episode_length_s)
        if len(sampled) >= len(tasks):
            times = sorted(sampled)[: len(tasks)]
        else:
            # Fallback to fixed-count to guarantee one release time per task.
            times = model.sample_arrivals_fixed_count(
                len(tasks),
                rng,
                horizon_s=cfg.episode_length_s,
            )
    else:
        raise ValueError(
            "Unsupported arrival_time_sampling_mode '{}'; expected 'fixed_count' or 'thinning'".format(
                cfg.arrival_time_sampling_mode
            )
        )

    # Deterministic mapping: sort by task_id then assign in time order.
    sorted_tasks = sorted(tasks, key=lambda t: t.task_id)
    release_times = {task.task_id: float(times[idx]) for idx, task in enumerate(sorted_tasks)}

    if cfg.release_jitter_s > 0.0:
        for task_id in list(release_times.keys()):
            jitter = rng.uniform(-cfg.release_jitter_s, cfg.release_jitter_s)
            release_times[task_id] = min(
                max(0.0, release_times[task_id] + jitter),
                float(cfg.episode_length_s),
            )

    return release_times


def _sample_task_demand(task, rng: random.Random, cfg: ScenarioSynthConfig) -> float:
    if cfg.use_truncnorm_demands:
        return _sample_truncnorm(
            rng,
            mean=cfg.demand_mean_kg,
            std=cfg.demand_std_kg,
            lower=cfg.demand_min_kg,
            upper=cfg.demand_max_kg,
        )

    base_demand = float(task.demand)
    if cfg.demand_noise_ratio > 0.0:
        noise = rng.uniform(-cfg.demand_noise_ratio, cfg.demand_noise_ratio)
        base_demand = max(0.0, base_demand * (1.0 + noise))
    return base_demand


def _sample_truncnorm(
    rng: random.Random,
    *,
    mean: float,
    std: float,
    lower: float,
    upper: float,
) -> float:
    """Sample a truncated normal using inverse-CDF (no SciPy dependency)."""

    lo = float(lower)
    hi = float(upper)
    if hi < lo:
        raise ValueError(f"upper must be >= lower (got {upper} < {lower})")
    if std <= 0.0:
        return min(max(float(mean), lo), hi)

    dist = NormalDist(mu=float(mean), sigma=float(std))
    a = dist.cdf(lo)
    b = dist.cdf(hi)
    if b <= a:
        return min(max(float(mean), lo), hi)
    u = a + (b - a) * rng.random()
    return float(dist.inv_cdf(u))


def _build_time_window_overrides(
    tasks,
    release_times: Dict[int, float],
    cfg: ScenarioSynthConfig,
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, float]]:
    """Compute pickup/delivery time windows anchored to release time."""

    node_time_windows: Dict[int, Tuple[float, float]] = {}
    node_service_times: Dict[int, float] = {}

    speed = float(cfg.vehicle_speed_m_s)
    pickup_width = float(cfg.pickup_tw_width_s)
    delivery_width = float(cfg.delivery_tw_width_s)

    for task in tasks:
        release = float(release_times.get(task.task_id, task.arrival_time))
        pickup = task.pickup_node
        delivery = task.delivery_node

        # Pickup window: [t_r, t_r + 1800]
        node_time_windows[pickup.node_id] = (release, release + pickup_width)

        # Delivery window: [t_r + t_min, t_r + t_min + 3600]
        dist_pd = euclidean_distance(
            pickup.coordinates[0],
            pickup.coordinates[1],
            delivery.coordinates[0],
            delivery.coordinates[1],
        )
        travel_pd = calculate_travel_time(dist_pd, speed)
        t_min = float(pickup.service_time) + float(travel_pd)
        node_time_windows[delivery.node_id] = (
            release + t_min,
            release + t_min + delivery_width,
        )

        node_service_times[pickup.node_id] = float(pickup.service_time)
        node_service_times[delivery.node_id] = float(delivery.service_time)

    return node_time_windows, node_service_times


__all__ = ["ScenarioSynthConfig", "synthesize_scenarios"]
