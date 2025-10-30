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
| Small | Matheuristic ALNS | 35353.07 | 32941.42 | 6.82% |
| Small | Minimal ALNS | 35791.99 | 31751.32 | 11.29% |
| Small | Matheuristic + Q-learning | 35791.99 | 24215.64 | 32.34% |
| Medium | Matheuristic ALNS | 24698.03 | 24324.44 | 1.51% |
| Medium | Minimal ALNS | 38842.80 | 38842.80 | 0.00% |
| Medium | Matheuristic + Q-learning | 38842.80 | 38257.55 | 1.51% |
| Large | Matheuristic ALNS | 52400.92 | 33600.06 | 35.88% |
| Large | Minimal ALNS | 60709.91 | 59230.27 | 2.44% |
| Large | Matheuristic + Q-learning | 60709.91 | 54340.95 | 10.49% |

> **Note:** Dynamic stagnation thresholds for the Q-learning agent now adapt to
> each scenario's iteration budget, keeping the state machine reachable on small
> regression runs while preventing the large-scale preset from reporting an
> infinite baseline cost.

![Relative improvement comparison](../figures/alns_regression_improvements.svg)
