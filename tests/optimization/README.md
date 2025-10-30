# Optimization Scenario Regression Tests

These tests keep the three scenario scales that demonstrate the ALNS planner's
behaviour. They are now organised by solver variant to make it explicit which
features are being exercised at each scale.

```
tests/optimization/
├── common.py                  # Shared scenario factories/utilities
├── matheuristic/              # Matheuristic ALNS regression by scale
│   ├── test_matheuristic_small.py
│   ├── test_matheuristic_medium.py
│   └── test_matheuristic_large.py
├── q_learning/                # Matheuristic + Q-learning regression by scale
│   ├── test_q_learning_small.py
│   ├── test_q_learning_medium.py
│   ├── test_q_learning_large.py
│   └── utils.py               # Lightweight runner with tuned hyper-parameters
└── test_alns_matheuristic.py  # Targeted checks for LP repair & elite pool
```

## Matheuristic ALNS regression suites

| Test module | Scenario size | Charging stations | Purpose |
|-------------|---------------|-------------------|---------|
| `matheuristic/test_matheuristic_small.py` | 10 tasks | 1 | Quick confidence check that every charging strategy improves a compact instance. |
| `matheuristic/test_matheuristic_medium.py` | 30 tasks | 2 | Regression harness for the default demo instance introduced during the warehouse regression iteration. |
| `matheuristic/test_matheuristic_large.py` | 50 tasks | 3 | Stress test for the adaptive operator logic at higher scale. |

Each suite exercises the three supported charging strategies (full recharge,
fixed partial recharge, and adaptive minimal recharge) and asserts that the ALNS
search reduces the weighted cost from the greedy baseline solution.

## Q-learning enhanced matheuristic regression

| Test module | Scenario size | Iterations | Focus |
|-------------|---------------|------------|-------|
| `q_learning/test_q_learning_small.py` | 10 tasks | 10 | Confirms Q-learning updates operator values while improving cost. |
| `q_learning/test_q_learning_medium.py` | 20 tasks | 14 | Ensures the Q-agent continues to gather positive rewards at medium scale. |
| `q_learning/test_q_learning_large.py` | 30 tasks | 18 | Stresses the learning pipeline on a larger request set. |

These tests reuse the deterministic scenario factory but run a single solver
configuration. They assert both cost improvements and that the Q-learning agent
receives non-zero rewards, decays its exploration rate, and records operator
usage, providing coverage for the reinforcement-learning adaptation.

## Running the tests

Run a specific suite with:

```bash
python -m pytest tests/optimization/matheuristic/test_matheuristic_small.py
python -m pytest tests/optimization/q_learning/test_q_learning_small.py
```

Large-scale scenarios may take a few minutes. When iterating on the ALNS
operators you can temporarily lower the iteration counts inside the tests to
probe behaviour without waiting for the full regression cycle.
