# Week 5: Q-Learning for Adaptive Partial Recharge Strategy

## Overview

Week 5 investigates using **Q-learning to learn optimal partial recharge strategies** for electric vehicle routing with battery constraints. Unlike Weeks 1-4 (operator selection), this targets the **unique charging characteristics** of EV routing problems.

## Research Question

> Can Q-learning learn better partial recharge strategies than rule-based heuristics, reducing charging time while maintaining solution feasibility?

## Quick Start

### Prerequisites

```bash
# Ensure you have completed Weeks 1-4
# Python 3.9+ with required packages
```

### Run Single Experiment

```bash
# Baseline (rule-based strategy)
python scripts/week5/run_charging_experiment.py \
    --scale small \
    --strategy baseline \
    --seed 2025 \
    --verbose

# Q-Learning strategy
python scripts/week5/run_charging_experiment.py \
    --scale small \
    --strategy qlearning \
    --seed 2025 \
    --verbose
```

### 可视化 DRL 局部充电优化过程

> **新功能**：`run_charging_rl_progress.py` 会调用 Week 5 的 `ChargingQLearningAgent`，
> 输出每次访问充电站时的状态、动作、奖励和 Q 值更新，方便观察“局部补能”策略的学习过程。

```bash
# 小规模：查看 5 个回合的学习过程并打印访问状态的 Q 表
python -m scripts.week5.run_charging_rl_progress --scale small --episodes 5 --show-q-table

# 中等规模：仅查看 3 个回合（默认）
python -m scripts.week5.run_charging_rl_progress --scale medium

# 大规模：快速巡检 2 个回合
python -m scripts.week5.run_charging_rl_progress --scale large --episodes 2

# 一次性运行小/中/大，并展示 Q 表
python -m scripts.week5.run_charging_rl_progress --scale all --episodes 4 --show-q-table
```

### Run Full Experimental Suite (60 experiments)

**Windows**:
```powershell
# Run all small-scale experiments
.\scripts\week5\batch_small_baseline.bat
.\scripts\week5\batch_small_qlearning.bat

# Medium scale
.\scripts\week5\batch_medium_baseline.bat
.\scripts\week5\batch_medium_qlearning.bat

# Large scale
.\scripts\week5\batch_large_baseline.bat
.\scripts\week5\batch_large_qlearning.bat
```

**Linux/Mac**:
```bash
# Run all experiments
bash scripts/week5/run_all_experiments.sh
```

## File Structure

```
scripts/week5/
├── README.md                          # This file
├── run_charging_experiment.py         # Main experiment runner
├── analyze_charging.py                # Analysis script
├── batch_*.bat                        # Windows batch scripts
└── run_all_experiments.sh             # Unix shell script

results/week5/
├── charging_baseline_small_seed2025.json
├── charging_qlearning_small_seed2025.json
├── ...
└── summary.csv                        # Aggregated results

src/strategy/
├── q_learning_charging.py             # Q-learning charging strategy
├── charging_state.py                  # State representation
└── charging_reward.py                 # Reward calculator
```

## Experimental Design

### Factors

1. **Charging Strategy** (2 levels):
   - `baseline`: Rule-based `PartialRechargeMinimalStrategy`
   - `qlearning`: Q-learning adaptive strategy

2. **Problem Scale** (3 levels):
   - `small`: 15 tasks, 5 charging stations
   - `medium`: 24 tasks, 7 charging stations
   - `large`: 30 tasks, 10 charging stations

3. **Random Seeds** (10 per scale):
   - 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034

**Total**: 2 × 3 × 10 = **60 experiments**

### Key Metrics

- **Total route cost** (primary)
- **Charging time** (secondary)
- **Number of charging stops**
- **Solution feasibility**
- **Q-learning diagnostics** (Q-table, action distribution)

## Analysis

### Generate Summary Statistics

```bash
python scripts/week5/analyze_charging.py --summarize
```

### Generate Comparison Plots

```bash
python scripts/week5/analyze_charging.py --plot
```

### Statistical Testing

```bash
python scripts/week5/analyze_charging.py --test
```

## Expected Results

**Hypotheses**:
- **H1**: Q-learning reduces route cost by ≥8%
- **H2**: Q-learning reduces charging time by ≥15%
- **H3**: Improvement increases with scale (small < medium < large)

**Success Criteria**:
- H1 confirmed with p < 0.05
- Feasibility maintained (no infeasible solutions)
- Consistent improvement across seeds (low variance)

## Implementation Status

- [x] Design document complete
- [ ] Core Q-learning charging strategy
- [ ] Integration with ALNS
- [ ] Experiment runner script
- [ ] Batch execution scripts
- [ ] Analysis scripts
- [ ] Pilot run (6 experiments)
- [ ] Full run (60 experiments)
- [ ] Results summary

## Troubleshooting

### Q-Learning Not Converging

- Increase learning rate: `alpha=0.1 → 0.3`
- Simplify state space: Reduce discretization granularity
- Extend training: `max_iterations=1000 → 2000`

### Infeasible Solutions

- Increase feasibility penalty in reward function
- Add action masking for risky actions
- Use hybrid strategy (Q-learning + rule-based validation)

### Slow Execution

- Run experiments in parallel (10 processes)
- Reduce iterations for pilot: `1000 → 500`
- Use smaller state space

## References

See `docs/experiments/WEEK5_DESIGN.md` for:
- Literature review (2023-2024 papers)
- Detailed methodology
- Theoretical foundations
- Risk mitigation strategies

## Contact

For questions about Week 5 experiments, refer to the design document or raise an issue in the repository.

---

**Status**: Implementation in progress
**Last Updated**: 2025-11-16
