# Week 5: Scale-Aware Reward Normalization Experiments

## Overview

Week 5 tests the **Scale-Aware Reward Normalization** framework, which is the core contribution of the SAQL research. This experiment compares:

- **OLD** (baseline): ROI-aware reward function (current implementation)
- **NEW** (proposed): Scale-aware normalized rewards with amplification factors

**Hypothesis**: Scale-aware reward normalization will improve large-scale Q-learning performance by ≥8%.

## Experiment Design

| Factor | Levels | Values |
|--------|--------|--------|
| Reward Type | 2 | OLD (baseline), NEW (scale-aware) |
| Problem Scale | 3 | Small (15 tasks), Medium (24 tasks), Large (30 tasks) |
| Random Seed | 10 | 2025-2034 |

**Total Experiments**: 2 × 3 × 10 = **60 runs**

## Quick Start

### 1. Test Single Experiment

```bash
# Test OLD reward on small scale
python scripts/week5/run_reward_experiment.py --scale small --reward old --seed 2025

# Test NEW reward on large scale
python scripts/week5/run_reward_experiment.py --scale large --reward new --seed 2025
```

### 2. Run All Experiments (Parallel Execution)

Open **3 terminal windows** and run simultaneously:

**Terminal 1 (Small Scale)**:
```bash
# Run OLD baseline
batch_small_old.bat

# Then run NEW
batch_small_new.bat
```

**Terminal 2 (Medium Scale)**:
```bash
batch_medium_old.bat
batch_medium_new.bat
```

**Terminal 3 (Large Scale)**:
```bash
batch_large_old.bat
batch_large_new.bat
```

**Expected Duration**: ~6-8 hours total (2-3 hours per terminal)

### 3. Analyze Results

After all experiments complete:

```bash
python scripts/week5/analyze_rewards.py
```

This generates:
- `results/week5/analysis_summary.txt` - Human-readable summary
- `results/week5/analysis_results.json` - Detailed statistical results
- **Checkpoint 2 Decision** - Go/no-go for scale-aware rewards

## Files Structure

```
scripts/week5/
├── run_reward_experiment.py    # Single experiment runner
├── analyze_rewards.py           # Statistical analysis script
├── batch_small_old.bat          # Batch: Small scale, OLD reward (10 seeds)
├── batch_small_new.bat          # Batch: Small scale, NEW reward (10 seeds)
├── batch_medium_old.bat         # Batch: Medium scale, OLD reward
├── batch_medium_new.bat         # Batch: Medium scale, NEW reward
├── batch_large_old.bat          # Batch: Large scale, OLD reward
├── batch_large_new.bat          # Batch: Large scale, NEW reward
└── README.md                    # This file

results/week5/
├── reward_experiments/          # Raw experiment outputs (60 JSON files)
│   ├── reward_old_small_seed2025.json
│   ├── reward_new_small_seed2025.json
│   └── ...
├── analysis_summary.txt         # Statistical analysis summary
└── analysis_results.json        # Detailed analysis results
```

## Success Criteria (Checkpoint 2)

**Primary**: Large-scale improvement ≥8%
- OLD (baseline): ~17% improvement
- NEW (target): ≥25% improvement
- Δ requirement: ≥8 percentage points

**Secondary**:
1. Statistical significance: p < 0.05
2. Effect size: Cohen's d > 0.5 (medium to large)
3. No degradation on small/medium scales (within 2% margin)

## Decision Matrix

| Outcome | Large Δ | p-value | Cohen's d | Decision |
|---------|---------|---------|-----------|----------|
| **Full Success** | ≥8% | <0.05 | >0.5 | ✅ Adopt NEW, proceed to Week 6 |
| **Partial Success** | 5-8% | <0.10 | >0.3 | ⚠️ Adopt with tuning |
| **Marginal** | 3-5% | Any | <0.3 | ⚠️ Investigate, may skip Week 3-4 |
| **Failure** | <3% | Any | Any | ❌ Major pivot needed |

## Implementation Details

### Scale-Aware Reward Normalization

**Key Innovations**:
1. **Previous-cost normalization**: `reward = (improvement / previous_cost) × scale_factor`
2. **Scale-dependent amplification**:
   - Small: 1.0× (baseline)
   - Medium: 1.3× (30% boost)
   - Large: 1.6× (60% boost)
3. **Adaptive bonuses**: Global best bonus scales with problem size
4. **Variance penalty**: Filters noise on large problems (<0.1% improvements)
5. **Convergence bonus**: Rewards progress toward baseline

### Code Locations

- `src/planner/scale_aware_reward.py` - Reward calculator implementation
- `src/planner/alns.py` - Integration with ALNS (line 631-743)
- `src/planner/alns_matheuristic.py` - Parameter pass-through

## Troubleshooting

### Issue: Missing dependencies

```bash
# Verify installation
python scripts/week5/run_reward_experiment.py --help
```

### Issue: "No module named 'scenario'"

```bash
# Ensure you're in the project root
cd /home/user/R3
python scripts/week5/run_reward_experiment.py --scale small --reward old --seed 2025
```

### Issue: Experiments taking too long

**Expected times** (per experiment):
- Small: 3-5 minutes
- Medium: 5-8 minutes
- Large: 8-12 minutes

If significantly slower, check:
- CPU usage (should be ~100% per process)
- LP solver timeout (set to 5 seconds)

### Issue: Analysis script fails

```bash
# Check if all 60 files exist
ls results/week5/reward_experiments/*.json | wc -l
# Should output: 60

# If missing files, check batch script logs for errors
```

## Next Steps After Week 5

### If Checkpoint 2 Passes (≥8% improvement):

1. **Document results** in `docs/experiments/WEEK5_RESULTS.md`
2. **Week 6**: Combined ablation study (epsilon + reward)
3. **Consider Week 3-4**: 7-state MDP (conditional on Week 5 success)

### If Checkpoint 2 Fails (<5% improvement):

1. **Investigate failure modes**: Analyze reward distributions, Q-value convergence
2. **Try alternative scale factors** (see WEEK5_DESIGN.md Appendix A)
3. **Pivot to Dynamic E-VRP** (Phase 3) if reward normalization insufficient

## References

- **Design Document**: `docs/experiments/WEEK5_DESIGN.md`
- **SAQL Plan**: `docs/SAQL_IMPLEMENTATION_PLAN_2025-11-09.md`
- **Week 1 Results**: `docs/experiments/WEEK1_RESULTS.md` (Q-init baseline)
- **Week 2 Results**: `docs/experiments/WEEK2_RESULTS.md` (Epsilon analysis)

## Contact

For questions or issues, refer to the detailed design document or the SAQL implementation plan.
