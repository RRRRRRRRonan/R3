# Optimization Scenario Regression Tests

These tests keep the three scenario scales that demonstrate the ALNS planner's
behaviour. They are now organised by solver variant to make it explicit which
features are being exercised at each scale.

```
tests/optimization/
├── common.py                  # Shared scenario factories/utilities
├── minimal/                   # Minimal ALNS regression by scale
│   ├── test_minimal_small.py
│   ├── test_minimal_medium.py
│   └── test_minimal_large.py
├── matheuristic/              # Matheuristic ALNS regression by scale
│   ├── test_matheuristic_small.py
│   ├── test_matheuristic_medium.py
│   └── test_matheuristic_large.py
├── presets.py                 # Unified scenario + iteration settings
├── q_learning/                # Matheuristic + Q-learning regression by scale
│   ├── test_q_learning_small.py
│   ├── test_q_learning_medium.py
│   ├── test_q_learning_large.py
│   └── utils.py               # Lightweight runner with tuned hyper-parameters
└── test_alns_matheuristic.py  # Targeted checks for LP repair & elite pool
```

## Matheuristic ALNS regression suites

| Test module | Scenario size | Charging stations | Iterations |
|-------------|---------------|-------------------|------------|
| `matheuristic/test_matheuristic_small.py` | 10 tasks | 1 | 28 |
| `matheuristic/test_matheuristic_medium.py` | 24 tasks | 1 | 44 |
| `matheuristic/test_matheuristic_large.py` | 30 tasks | 3 | 44 |

Each suite exercises the three supported charging strategies (full recharge,
fixed partial recharge, and adaptive minimal recharge) and asserts that the ALNS
search reduces the weighted cost from the greedy baseline solution.

## Minimal ALNS regression

| Test module | Scenario size | Charging stations | Iterations |
|-------------|---------------|-------------------|------------|
| `minimal/test_minimal_small.py` | 10 tasks | 1 | 16 |
| `minimal/test_minimal_medium.py` | 24 tasks | 1 | 32 |
| `minimal/test_minimal_large.py` | 30 tasks | 3 | 32 |

These tests run the baseline single-vehicle ALNS solver without matheuristic
enhancements. The deterministic scenarios match the ones used by the other
variants so we can compare solver behaviour under identical conditions.

## Q-learning enhanced matheuristic regression

| Test module | Scenario size | Charging stations | Iterations |
|-------------|---------------|-------------------|------------|
| `q_learning/test_q_learning_small.py` | 10 tasks | 1 | 18 |
| `q_learning/test_q_learning_medium.py` | 24 tasks | 1 | 36 |
| `q_learning/test_q_learning_large.py` | 30 tasks | 3 | 30 |

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
operators you can temporarily adjust the iteration budgets in `presets.py` to
shrink runtime across all solver variants, ensuring the suites stay in sync.
