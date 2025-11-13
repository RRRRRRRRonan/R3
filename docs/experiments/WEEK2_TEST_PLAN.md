# Week 2 测试方案：自适应Epsilon策略实验

**日期**: 2025-11-12
**状态**: 🔄 设计中
**目标**: 测试自适应exploration rate是否能改善大规模Q-learning性能
**预计时间**: 2-3天
**预计运行次数**: 90次 (3策略 × 3规模 × 10种子)

---

## 📋 实验动机

### Week 1 的发现

Week 1实验表明Q-table初始化对性能影响很小，但提出了一个重要假设：

**当前epsilon策略问题**:
- 固定初始值：0.12（所有规模）
- 这对大规模问题可能不够探索

**证据**:
1. 大规模性能远低于中小规模（25% vs 37%）
2. Q-learning在大规模场景下方差更高（CV=0.595）
3. Matheuristic（无Q-learning）在大规模表现更好（27% vs 7%）

**假设**: 大规模问题的解空间更大，需要更多探索才能找到好解。

---

## 🎯 研究问题

**Primary Question**: 提高大规模问题的exploration rate是否能显著改善Q-learning性能？

**Hypotheses**:
- H1: Scale-adaptive epsilon（根据规模调整）将改善大规模性能
- H2: 统一提高epsilon至0.50将改善所有规模的性能
- H3: 当前epsilon（0.12）对小规模足够，但对大规模不足

---

## 🔬 实验设计

### Epsilon策略

#### Strategy 1: CURRENT (Baseline)
```python
initial_epsilon = 0.12  # All scales
epsilon_decay = 0.995
min_epsilon = 0.01
```
**Rationale**: Current implementation, baseline for comparison

#### Strategy 2: SCALE_ADAPTIVE
```python
# Scale-dependent initial epsilon
scale_epsilon_map = {
    "small": 0.30,   # 2.5× current
    "medium": 0.50,  # 4.2× current
    "large": 0.70    # 5.8× current
}
epsilon_decay = 0.995
min_epsilon = 0.05  # Higher floor for sustained exploration
```
**Rationale**: Larger problems need more exploration due to exponentially larger search spaces

**Justification**:
- Small (8 requests): ~8! × 5^8 ≈ 10^9 possible solutions
- Medium (24 requests): ~24! × 15^24 ≈ 10^30 possible solutions
- Large (40 requests): ~40! × 25^40 ≈ 10^60 possible solutions

#### Strategy 3: HIGH_UNIFORM
```python
initial_epsilon = 0.50  # All scales
epsilon_decay = 0.995
min_epsilon = 0.05
```
**Rationale**: Test if high exploration universally beneficial (simpler than scale-adaptive)

---

## 📊 Experimental Protocol

### Problem Configuration

**Scenarios** (Same as Week 1):
- Small: 8 requests, 5 stations (seed 11)
- Medium: 24 requests, 15 stations (seed 19)
- Large: 40 requests, 25 stations (seed 17)

**Optimization Seeds**: 10 per configuration (2025-2034)

**Fixed Parameters** (Consistent with Week 1):
- Iterations: 40 (small/medium), 44 (large)
- Alpha (learning rate): 0.40
- Gamma (discount): 0.95
- Q-table initialization: ZERO (based on Week 1 recommendation)

**Experiment Matrix**:
```
3 epsilon strategies × 3 scales × 10 seeds = 90 experiments
```

### Execution Plan

**Day 1**: Implementation
- Modify `src/planner/alns_matheuristic.py` to support scale-adaptive epsilon
- Add `epsilon_strategy` parameter to ALNS initialization
- Create `scripts/week2/run_experiment.py` (adapted from week1)

**Day 2**: Execution
- Run experiments in parallel (3 processes: small/medium/large)
- Estimated time: 6-8 hours

**Day 3**: Analysis
- Statistical comparison vs. CURRENT baseline
- Convergence curve analysis (epsilon decay trajectory)
- Decision: Continue with best strategy or revert to CURRENT

---

## 📈 Evaluation Metrics

### Primary Metrics

1. **Improvement Ratio** (main metric)
   ```
   improvement_ratio = (baseline_cost - optimised_cost) / baseline_cost
   ```

2. **Statistical Significance**
   - Wilcoxon signed-rank test vs. CURRENT baseline
   - Significance levels: * p<0.10, ** p<0.05, *** p<0.01
   - Cohen's d effect size (|d| > 0.3 considered meaningful)

### Secondary Metrics

3. **Convergence Analysis**
   - Q-value stabilization iteration
   - Epsilon trajectory (initial → final)
   - Exploration vs. exploitation ratio

4. **Variance Analysis**
   - Standard deviation across seeds
   - Coefficient of variation (CV)
   - Compare to Week 1 ZERO baseline

5. **Runtime**
   - Total runtime per experiment
   - Iterations per second

---

## 🎯 Success Criteria

### Minimal Success
- **Large scale improvement**: +5% over CURRENT (25% → 30%)
- **Statistical significance**: p < 0.05, |Cohen's d| > 0.3
- **No degradation on small/medium**: Within 2% of CURRENT

### Full Success
- **Large scale improvement**: +8% or more (25% → 33%+)
- **Medium scale improvement**: +3% or more (31% → 34%+)
- **Statistical significance**: p < 0.01, |Cohen's d| > 0.5
- **Reduced variance**: CV reduction by 20%+

### Expected Outcome
Based on literature and theory, we expect:
- SCALE_ADAPTIVE to improve large-scale performance by 3-7%
- HIGH_UNIFORM to improve large-scale but may hurt small-scale
- Trade-off: More exploration → Better solutions but higher variance

---

## 🔧 Implementation Details

### Code Changes

**New File**: `src/planner/epsilon_strategy.py`
```python
@dataclass
class EpsilonStrategy:
    """Defines epsilon (exploration rate) strategy."""
    name: str
    initial_epsilon: float
    decay_rate: float = 0.995
    min_epsilon: float = 0.01

    @staticmethod
    def current() -> "EpsilonStrategy":
        """Current baseline strategy."""
        return EpsilonStrategy("current", 0.12, 0.995, 0.01)

    @staticmethod
    def scale_adaptive(num_requests: int) -> "EpsilonStrategy":
        """Scale-adaptive strategy."""
        if num_requests <= 12:
            initial = 0.30
        elif num_requests <= 30:
            initial = 0.50
        else:
            initial = 0.70
        return EpsilonStrategy("scale_adaptive", initial, 0.995, 0.05)

    @staticmethod
    def high_uniform() -> "EpsilonStrategy":
        """High uniform exploration strategy."""
        return EpsilonStrategy("high_uniform", 0.50, 0.995, 0.05)
```

**Modified File**: `src/planner/alns_matheuristic.py`
- Add `epsilon_strategy` parameter to `__init__`
- Pass strategy to Q-learning agent initialization

**Modified File**: `src/planner/q_learning.py`
- Accept epsilon parameters from strategy
- Update initialization to use strategy values

### Experiment Script

**Location**: `scripts/week2/run_experiment.py`

**Usage**:
```powershell
python scripts\week2\run_experiment.py \
    --scenario small \
    --epsilon_strategy scale_adaptive \
    --seed 2025 \
    --output results\week2\epsilon_scale_adaptive_small_seed2025.json
```

**Batch Scripts**:
- `scripts/week2/01_current_baseline_*.bat` (baseline, should match Week 1 ZERO)
- `scripts/week2/02_scale_adaptive_*.bat`
- `scripts/week2/03_high_uniform_*.bat`

---

## 📊 Analysis Plan

### Statistical Analysis Script

**Location**: `scripts/week2/analyze_epsilon.py`

**Features**:
1. Load all 90 experiment results
2. Compute descriptive statistics per strategy/scale
3. Wilcoxon tests (each strategy vs. CURRENT)
4. Cohen's d effect sizes
5. Convergence curve plotting (epsilon over iterations)
6. Generate summary report

**Output**:
- `results/week2/analysis_summary.txt`
- `results/week2/convergence_curves.png`
- `results/week2/improvement_comparison.png`

### Decision Tree

```
If SCALE_ADAPTIVE improves large by ≥5% (p<0.05):
  ✅ Adopt SCALE_ADAPTIVE for all future experiments
  ✅ Proceed to Week 5 (Reward Normalization)

Else if HIGH_UNIFORM improves large by ≥5% (p<0.05):
  ⚠️ Consider trade-offs (may hurt small-scale)
  ⚠️ May need scale-specific configurations

Else (no improvement):
  ❌ Epsilon is not the bottleneck
  ❌ Skip directly to Week 5 (Reward Normalization)
  ❌ Hypothesis: Reward signal quality matters more than exploration
```

---

## 🔗 Relation to Week 1

**What Week 1 Told Us**:
- Q-table initialization has minimal impact (|d| < 0.2)
- ACTION_SPECIFIC with domain bias was harmful
- ZERO baseline is most robust

**What Week 2 Tests**:
- Whether exploration rate (not initialization) is the key factor
- Whether scale-dependent behavior requires scale-dependent parameters
- Quick test (2-3 days) before committing to Week 5 (5-7 days)

**Combined Insight** (if Week 2 succeeds):
- Initialization: Use ZERO (Week 1)
- Exploration: Use SCALE_ADAPTIVE epsilon (Week 2)
- Next: Test if reward normalization adds further benefit (Week 5)

---

## 📚 Literature Support

**Sutton & Barto (2018)**: "Reinforcement Learning: An Introduction"
- Chapter 2.7: Optimistic initial values encourage exploration
- Chapter 6.5: Importance of exploration in large state spaces
- Finding: Exploration matters more when value estimates are uncertain

**Silva et al. (2019)**: "Q-learning for ALNS"
- Used fixed epsilon=0.10, did not test scale-adaptive strategies
- Noted performance degradation on large instances (but did not address)

**Auer et al. (2002)**: "UCB for multi-armed bandits"
- Theoretical justification for exploration in large action spaces
- Suggests exploration rate should scale with log(action_space_size)

**Contribution**: First systematic study of scale-adaptive exploration for Q-learning in metaheuristics.

---

## 🚨 Risk Mitigation

### Risk 1: Higher Epsilon Increases Variance
**Probability**: Medium
**Impact**: Medium
**Mitigation**:
- 10 seeds provide statistical power to detect true signal
- Wilcoxon test robust to outliers
- If variance too high, can adjust decay rate (faster decay)

### Risk 2: No Improvement Despite Higher Exploration
**Probability**: Medium
**Impact**: Low (quick experiment, low cost)
**Mitigation**:
- Confirms that exploration is not the bottleneck
- Pivots focus to Week 5 (reward normalization)
- Not a wasted effort—negative results are informative

### Risk 3: Trade-off Between Small and Large Scale
**Probability**: Low
**Impact**: Medium
**Mitigation**:
- SCALE_ADAPTIVE avoids this by using different epsilon per scale
- If trade-off exists, document and justify scale-specific configs

---

## 📅 Timeline

| Day | Task | Hours | Deliverable |
|-----|------|-------|-------------|
| 1 | Design & Implementation | 4-6h | Code ready, scripts written |
| 2 | Run experiments | 6-8h | 90 result files |
| 3 | Analysis & Report | 3-4h | Summary document, decision |

**Total**: 13-18 hours over 2-3 days

---

## ✅ Checklist

### Pre-Experiment
- [ ] Create `src/planner/epsilon_strategy.py`
- [ ] Modify `src/planner/q_learning.py` to accept strategy
- [ ] Modify `src/planner/alns_matheuristic.py` to pass strategy
- [ ] Create `scripts/week2/run_experiment.py`
- [ ] Create batch scripts (×9: 3 strategies × 3 scales)
- [ ] Test single run to verify correctness

### Execution
- [ ] Run CURRENT baseline (30 experiments)
- [ ] Run SCALE_ADAPTIVE (30 experiments)
- [ ] Run HIGH_UNIFORM (30 experiments)
- [ ] Verify all 90 result files generated
- [ ] Spot-check epsilon values in output JSON

### Analysis
- [ ] Run `analyze_epsilon.py`
- [ ] Generate convergence curves
- [ ] Statistical tests (Wilcoxon, Cohen's d)
- [ ] Write WEEK2_RESULTS.md (similar to Week 1)
- [ ] Make decision: adopt strategy or proceed to Week 5

### Documentation
- [ ] Update `SAQL_IMPLEMENTATION_PLAN_2025-11-09.md` with Week 2 results
- [ ] Commit and push all code/results
- [ ] Update Option A timeline based on findings

---

## 📖 References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.

2. Silva, T. L. C., de Queiroz, T. A., & Munari, P. (2019). An improved adaptive large neighborhood search algorithm for the resource constrained project scheduling problem. Engineering Optimization, 51(12), 2063-2088.

3. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2-3), 235-256.

4. Ropke, S., & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. Transportation Science, 40(4), 455-472.

---

## Document Metadata

**Version**: 1.0
**Created**: 2025-11-12
**Status**: Design Complete, Ready for Implementation
**Dependencies**: Week 1 results (ZERO initialization recommendation)
**Next**: Week 5 (Reward Normalization) or Week 3-4 (7-State MDP, conditional)

---

**End of Document**
