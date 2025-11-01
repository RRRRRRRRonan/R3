"""Test Large scale with verbose ALNS output to see where it hangs.

This modifies the ALNS to be verbose so we can see each iteration.
"""

import sys
from pathlib import Path
import time
import random
from dataclasses import replace

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from tests.optimization.common import get_scale_config, build_scenario
from planner.alns_matheuristic import MatheuristicALNS
from strategy.charging_strategies import PartialRechargeMinimalStrategy
from config import (
    DEFAULT_ALNS_HYPERPARAMETERS,
    CostParameters,
    DestroyRepairParams,
    LPRepairParams,
    MatheuristicParams,
    SegmentOptimizationParams,
)

print("="*60, flush=True)
print("VERBOSE ALNS TEST - See each iteration", flush=True)
print("="*60, flush=True)

config = get_scale_config('large')
seed = 2026
iterations = 10  # Start with just 10 to see what happens

print(f"\nConfiguration:", flush=True)
print(f"  Scale: large", flush=True)
print(f"  Seed: {seed}", flush=True)
print(f"  Iterations: {iterations}", flush=True)
print(f"  VERBOSE: True (will show each iteration)", flush=True)
print(f"\n", flush=True)

# Build scenario
scenario = build_scenario(config)
task_pool = scenario.create_task_pool()

# Setup (same as run_q_learning_trial)
tuned_hyper = replace(
    DEFAULT_ALNS_HYPERPARAMETERS,
    destroy_repair=DestroyRepairParams(
        random_removal_q=2,
        partial_removal_q=2,
        remove_cs_probability=0.2,
    ),
    matheuristic=MatheuristicParams(
        elite_pool_size=4,
        intensification_interval=25,
        segment_frequency=6,
        max_elite_trials=2,
        segment_optimization=SegmentOptimizationParams(
            max_segment_tasks=3,
            candidate_pool_size=3,
            improvement_tolerance=1e-3,
            max_permutations=12,
            lookahead_window=2,
        ),
        lp_repair=LPRepairParams(
            time_limit_s=0.3,
            max_plans_per_task=4,
            improvement_tolerance=1e-4,
            skip_penalty=5_000.0,
            fractional_threshold=1e-3,
        ),
    ),
)

charging_strategy = PartialRechargeMinimalStrategy(safety_margin=0.02, min_margin=0.0)
cost_params = CostParameters()

# Set random seed
rng = random.Random(seed)
state = random.getstate()
random.setstate(rng.getstate())

print(f"[{time.strftime('%H:%M:%S')}] Creating ALNS with VERBOSE=True...", flush=True)

try:
    # Create with VERBOSE=True (KEY CHANGE!)
    alns = MatheuristicALNS(
        distance_matrix=scenario.distance,
        task_pool=task_pool,
        repair_mode="adaptive",
        cost_params=cost_params,
        charging_strategy=charging_strategy,
        use_adaptive=True,
        verbose=True,  # ← VERBOSE!
        adaptation_mode="q_learning",
        hyper_params=tuned_hyper,
    )

    print(f"[{time.strftime('%H:%M:%S')}] ALNS created successfully", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Starting optimization...", flush=True)
    print(f"\nYou should see iteration-by-iteration output below:", flush=True)
    print("-"*60, flush=True)

    # Get baseline
    vehicle = scenario.create_vehicles()[0]
    baseline_route = alns._create_initial_solution(vehicle)
    baseline_cost = alns.evaluate_route(baseline_route)

    print(f"\nBaseline cost: {baseline_cost:.2f}", flush=True)
    print(f"Starting {iterations} iterations...\n", flush=True)

    # Run optimization
    start = time.time()
    result_route = alns.optimize(vehicle, iterations=iterations)
    elapsed = time.time() - start

    optimised_cost = alns.evaluate_route(result_route)
    improvement = (baseline_cost - optimised_cost) / baseline_cost * 100

    print(f"\n" + "-"*60, flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Optimization completed!", flush=True)
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"  Baseline: {baseline_cost:.2f}", flush=True)
    print(f"  Optimised: {optimised_cost:.2f}", flush=True)
    print(f"  Improvement: {improvement:.2f}%", flush=True)

    print("\n" + "="*60, flush=True)
    print("✅ SUCCESS - ALNS completed without hanging", flush=True)
    print("="*60, flush=True)

except KeyboardInterrupt:
    print(f"\n\n⚠️  Interrupted by user", flush=True)
    print(f"Last seen output was above ↑", flush=True)
    sys.exit(1)

except Exception as e:
    print(f"\n\n❌ ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    random.setstate(state)
