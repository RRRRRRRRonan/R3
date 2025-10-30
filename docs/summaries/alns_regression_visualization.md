# ALNS Regression Visualisation

This note summarises the unified regression presets across the Minimal ALNS,
Matheuristic ALNS, and Matheuristic + Q-learning solvers.  The metrics were
produced with [`scripts/generate_alns_visualization.py`](../../scripts/generate_alns_visualization.py)
using the shared optimisation presets scaled to 60% of their iteration budgets
(`--iteration-scale 0.6`).  To keep runtimes modest, the helper narrows the
medium and large scenarios to single-vehicle cases with 12 and 16 requests
respectively while preserving the relative charging density from the tests.

The script emits the Markdown table below, saves a JSON dump of the raw metrics
(`docs/data/alns_regression_results.json`), and renders an SVG bar chart
(`docs/figures/alns_regression_improvements.svg`).

| Scale | Solver | Baseline Cost | Optimised Cost | Improvement |
|:------|:-------|--------------:|---------------:|------------:|
| Small | Matheuristic ALNS | 35353.07 | 20746.39 | 41.32% |
| Small | Minimal ALNS | 35791.99 | 32965.82 | 7.90% |
| Small | Matheuristic + Q-learning | 35791.99 | 23905.07 | 33.21% |
| Medium | Matheuristic ALNS | 24698.03 | 24698.03 | 0.00% |
| Medium | Minimal ALNS | 38842.80 | 35619.06 | 8.30% |
| Medium | Matheuristic + Q-learning | 38842.80 | 35678.08 | 8.15% |
| Large | Matheuristic ALNS | 52400.92 | 52400.92 | 0.00% |
| Large | Minimal ALNS | 60709.91 | 59709.45 | 1.65% |
| Large | Matheuristic + Q-learning | ∞ | 55654.49 | 0.00% |

> **Note:** The greedy baseline for the large Q-learning scenario is infeasible,
> producing an infinite cost.  The optimiser still finds a finite solution, but
> the relative improvement is reported as zero because the baseline cost is
> unbounded.  This mirrors the stress the large-scale preset puts on the
> destroy/repair pair selection when only one vehicle is available.

![Relative improvement comparison](../figures/alns_regression_improvements.svg)
