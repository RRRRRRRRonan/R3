# Matheuristic ALNS upgrade (Singh et al., 2022)

This document captures the design notes for the matheuristic extension of the
single-AMR ALNS solver.  The implementation follows the key components from
Singh, N., et al. (2022) *"A matheuristic for AGV scheduling with battery
constraints"* while staying compatible with the existing task, energy, and
cost infrastructure in this project.

## Architectural additions

| Paper component | Implementation | Notes |
| --- | --- | --- |
| Large neighbourhood search backbone | `MatheuristicALNS` inherits from `MinimalALNS` | Reuses destroy/repair operators, adaptive scoring, and SA acceptance. |
| Elite pool memory | `_EliteRoute` entries maintained in `_update_elite_pool` | Preserves best routes for intensification; configurable pool size. |
| Matheuristic repair (MILP subproblem) | `_SegmentOptimizer._optimise_segment` | Enumerates permutations for a bounded set of requests using `regret2` insertion, mimicking the MILP reconstruction from the paper. |
| Battery-aware segment selection | `_SegmentOptimizer._identify_segments` | Detects low-battery hotspots via route visit data; falls back to pickup windows when energy telemetry is unavailable. |
| Intensification phase | `MatheuristicALNS._intensify` | Periodically selects an elite route, re-optimises its hotspot segments, and replaces the incumbent when better. |

## Key configuration knobs

The new dataclasses in `config/defaults.py` expose the matheuristic tuning
parameters:

* `SegmentOptimizationParams`: bounds the neighbourhood re-optimisation effort
  (max tasks per segment, candidate pool size, permutation cap, etc.).
* `MatheuristicParams`: controls the elite pool size, how often segment
  rebuilding is attempted after accepted moves, and the intensification cadence.

These defaults can be overridden via the existing configuration factory, so the
new solver is ready for experimentation without touching the algorithmic code.

## Usage

```python
from planner.alns_matheuristic import MatheuristicALNS
from config import DEFAULT_COST_PARAMETERS, DEFAULT_ALNS_HYPERPARAMETERS

alns = MatheuristicALNS(
    distance_matrix=distance,
    task_pool=task_pool,
    cost_params=DEFAULT_COST_PARAMETERS,
    hyper_params=DEFAULT_ALNS_HYPERPARAMETERS,
    charging_strategy=strategy,
    use_adaptive=True,
    verbose=False,
)

alns.vehicle = vehicle
alns.energy_config = energy_config
route = alns.optimize(initial_route, max_iterations=300)
```

When vehicles and energy configurations are supplied, the optimiser will use
the recorded visit telemetry to detect energy-critical segments and run the
MILP-inspired reconstruction routine; otherwise it falls back to pure ALNS
behaviour.
