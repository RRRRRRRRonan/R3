# Optimization Scenario Regression Tests

These tests keep the three scenario scales that demonstrate the ALNS planner's
design goals while trimming the legacy notebooks and manual walkthroughs that
used to live in this folder.

## Available scenarios

| Test module | Scenario size | Charging stations | Purpose |
|-------------|---------------|-------------------|---------|
| `test_alns_optimization_small.py` | 10 tasks | 1 | Quick confidence check that every charging strategy improves a compact instance. |
| `test_alns_optimization_medium.py` | 30 tasks | 2 | Regression harness for the default demo instance showcased in Week 3. |
| `test_alns_optimization_large.py` | 50 tasks | 3 | Stress test for the adaptive operator logic at higher scale. |

Each suite exercises the three supported charging strategies (full recharge,
fixed partial recharge, and adaptive minimal recharge) and asserts that the ALNS
search reduces the weighted cost from the greedy baseline solution.

## Running the tests

Run a specific scale with:

```bash
python -m pytest tests/optimization/test_alns_optimization_small.py
python -m pytest tests/optimization/test_alns_optimization_medium.py
python -m pytest tests/optimization/test_alns_optimization_large.py
```

The large-scale scenario may take a few minutes. When iterating on the ALNS
operators you can temporarily lower the iteration counts inside the test to
probe behaviour without waiting for the full regression cycle.
