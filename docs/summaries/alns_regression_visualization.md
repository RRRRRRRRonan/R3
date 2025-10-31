# ALNS Regression Visualisation

This note summarises the unified regression presets across the Minimal ALNS,
Matheuristic ALNS, and Matheuristic + Q-learning solvers.  The metrics were
produced with [`scripts/generate_alns_visualization.py`](../../scripts/generate_alns_visualization.py)
using the shared optimisation presets at their full iteration budgets so the
reported improvements mirror the regression tests exactly.  The medium scale now
uses a tougher 24-request, single-station scenario which exposes the benefits of
state-aware destroy strengths and the reinforcement-learning agent.

The script emits the Markdown table below, saves a JSON dump of the raw metrics
(`docs/data/alns_regression_results.json`), and renders an SVG bar chart
(`docs/figures/alns_regression_improvements.svg`).

| Scale | Solver | Baseline Cost | Optimised Cost | Improvement |
|:------|:-------|--------------:|---------------:|------------:|
| Small | Matheuristic ALNS | 35353.07 | 14662.86 | 58.52% |
| Small | Minimal ALNS | 35791.99 | 32263.47 | 9.86% |
| Small | Matheuristic + Q-learning | 35791.99 | 13439.54 | 62.45% |
| Medium | Matheuristic ALNS | 35102.80 | 18815.16 | 46.40% |
| Medium | Minimal ALNS | 39317.52 | 36648.89 | 6.79% |
| Medium | Matheuristic + Q-learning | 39317.52 | 27711.66 | 29.52% |
| Large | Matheuristic ALNS | 52400.92 | 38227.26 | 27.05% |
| Large | Minimal ALNS | 60709.91 | 56409.99 | 7.08% |
| Large | Matheuristic + Q-learning | 60709.91 | 56508.74 | 6.92% |

> **Note:** Dynamic stagnation thresholds for the Q-learning agent adapt to each
> scenario's iteration budget, keeping the state machine reachable on the shorter
> regression runs while preventing the large-scale preset from reporting an
> infinite baseline cost.  The new action-masking rules steer aggressive
> destroyers toward LP-backed repairs in stagnation states, while the
> time-aware reward penalties discourage slow operators that fail to deliver
> high-quality improvements.  Together with the restored segment optimisation
> cadence, the reinforcement learner now overtakes the matheuristic baseline on
> the small scenario and narrows the gap substantially on the tougher
> medium/large presets.

![Relative improvement comparison](../figures/alns_regression_improvements.svg)
