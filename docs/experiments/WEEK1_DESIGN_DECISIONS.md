# Week 1 设计决策与注意事项

本文档记录Week 1实验过程中的关键设计决策，包括原因、权衡和未来建议。

---

## 1. 场景生成：固定场景 vs 随机场景

### 当前设计（已采用）

**场景种子固定**：
- Small场景：seed = 11（来自`tests/optimization/presets.py`）
- Medium场景：seed = 19
- Large场景：seed = 17

**实验种子变化**：
- 10个实验种子：2025-2034
- 这些种子只控制ALNS优化过程的随机性，不影响场景生成

**实际效果**：
```
每个scale使用1个固定场景
该场景运行10次（不同的ALNS随机化）
所有4种初始化策略在相同的30个配置上测试
```

### 设计原因

1. **时间效率**：保留已完成的30个baseline实验结果
2. **科学有效性**：对照组一致性保证了公平比较
3. **Week 1目标**：找出相对最佳的初始化策略，不需要证明绝对泛化性

### 实验解释

**正确解释**：
> 在3个代表性场景上，测试4种Q-table初始化策略的性能。每个场景运行10次以评估在不同ALNS随机化下的稳定性。

**不是**：
> 在30个不同场景上测试...

### 科学有效性

虽然场景固定，但实验设计仍然科学有效，因为：

1. ✅ **所有策略测试条件一致**
   - ZERO、UNIFORM、ACTION_SPECIFIC、STATE_SPECIFIC都在相同30个配置上运行
   - Wilcoxon配对检验仍然有效

2. ✅ **10个seed提供足够的统计样本**
   - 评估算法在不同随机化下的鲁棒性
   - ALNS的随机性本身就很强（CV>50%说明）

3. ✅ **3个scale代表不同难度**
   - Small (15任务), Medium (24任务), Large (30任务)
   - 覆盖了不同复杂度的场景

### 观察到的现象

**baseline结果分析**（所有实验使用ZERO初始化）：

| 场景 | 平均改进率 | 标准差 | CV | 范围 |
|------|-----------|--------|-----|------|
| Small | 37.7% | 6.0% | 15.9% | 29.7% - 49.5% |
| Medium | 31.5% | 17.1% | **54.4%** | 10.6% - 49.7% |
| Large | 25.5% | 14.4% | **56.4%** | 12.5% - 48.9% |

**关键发现**：
- Medium/Large场景的变异系数>50%，说明即使在固定场景下，ALNS的随机性也导致结果高度变化
- 某些seed收敛到相同的optimised_cost（Medium有4个实验都是38626.91，Large有5个实验都是69389.40）
- 这说明存在多个局部最优，不同随机化可能收敛到不同解

---

## 2. 未来修改建议

### Week 2/3：增加场景多样性

当实施Scale-Aware Q-learning初始化时，建议修改场景生成逻辑以测试泛化性。

**修改位置**：`scripts/week1/run_experiment.py`

**修改方法**：

```python
# 当前代码（第140-141行）
config = get_scale_config(scenario_scale)  # 使用固定seed
scenario = build_scenario(config)

# 建议修改为
from dataclasses import replace
config = get_scale_config(scenario_scale)
config = replace(config, seed=seed)  # 用实验seed覆盖固定seed
scenario = build_scenario(config)
```

**效果**：
- 每个实验seed（2025-2034）会生成不同的场景
- 真正测试算法在多个场景上的泛化能力

### 论文发表：补充验证实验

如果投稿到顶级会议/期刊，可能需要：
- 在更多场景（20-50个）上验证最佳策略
- 或者重新运行所有150个实验，使用随机场景
- 在附录中说明Week 1是"探索性研究"阶段

---

## 3. 性能优化记录

### 已实施的优化

为了让Medium/Large场景能在合理时间内完成，进行了以下简化：

**Matheuristic参数调整**（`run_experiment.py:87-114`）：
- Segment optimization：**完全禁用**（frequency = 0）
- Elite pool size：4 → 2
- Max permutations：12 → 4
- LP timeout：0.3s → 2.0s
- Max plans per task：4 → 2

**影响**：
- ✅ 运行速度大幅提升（避免卡死）
- ⚠️ 优化质量可能下降（但对Q-learning比较影响有限）

### 运行时间统计

实际运行时间（baseline，ZERO初始化）：

| 场景 | 最小 | 最大 | 平均估算 |
|------|------|------|----------|
| Small | 55s | 149s | ~90s |
| Medium | 71s | 6275s | ~2000s |
| Large | 162s | 8133s | ~1800s |

**总时间估算**：
- 30个baseline：约15-20小时
- 150个全部实验：约75-100小时

### 未来优化方向

如果需要进一步加速：

1. **实验级并行**（推荐）
   - 使用多进程同时运行多个实验
   - 4核CPU可缩短75%时间
   - 不影响实验结果的可重现性

2. **减少迭代次数**
   - Small: 40 → 30
   - Medium/Large: 44 → 30
   - 约节省30%时间

3. **进一步简化Matheuristic**
   - 考虑禁用LP repair
   - 只使用greedy/regret2/random算子
   - 可能快5-10倍，但影响实验有效性

---

## 4. CPU利用率低的原因

### 问题现象

运行实验时，CPU利用率只有10-15%（在多核CPU上）。

### 根本原因

1. **Python GIL限制**：
   - Python的全局解释器锁导致单线程执行
   - 即使是8核CPU，单个Python进程只能用满1个核心
   - CPU总利用率 = 100% / 核心数（如8核CPU上12.5%）

2. **LP求解器是纯Python实现**：
   - 使用自己实现的Simplex算法（`src/planner/repair_lp.py`）
   - 没有使用商业求解器（Gurobi/CPLEX）
   - 完全串行执行，无法并行化

3. **time_limit_s参数未实际生效**：
   - 虽然配置了timeout，但SimplexSolver没有时间检查
   - 复杂的LP问题可能运行很久

### 解决方案

**推荐**：实验级多进程并行
- 同时运行N个独立实验（N=CPU核心数）
- 每个实验仍是单线程，但总CPU利用率接近100%
- 可以在Week 2实施

**不推荐**：修改LP求解器为多线程
- 工程量大，收益有限
- LP问题本身不大，并行化收益不明显

---

## 5. 数据质量检查

### Baseline数据完整性

✅ **已确认**：
- 30个实验全部成功完成
- 所有实验的final_epsilon都正常收敛（~0.096）
- Q-learning学习正常（Q值有明显分化）
- 数据格式正确，无缺失字段

⚠️ **注意事项**：
- Medium/Large场景的高CV值是真实现象，不是数据错误
- 某些seed收敛到相同解是局部最优的表现，不是bug

### Q-learning学习模式

**Small场景**：主要在explore状态学习，避免使用LP repair

**Medium场景**：explore和stuck状态都有学习

**Large场景**：**LP repair在explore状态显示正Q值**
- 与Small/Medium相反
- 说明大场景确实需要不同策略
- 为Scale-Aware初始化提供了证据

---

## 6. 下一步行动

### 立即执行

1. ✅ 保留当前30个baseline结果
2. 🔄 运行120个其他策略实验：
   - UNIFORM: 30个实验
   - ACTION_SPECIFIC: 30个实验
   - STATE_SPECIFIC: 30个实验
3. 📊 完成Week 1统计分析
4. 📝 撰写Week 1总结报告

### Week 2计划

1. 📌 修改场景生成逻辑（使用随机场景）
2. 🎯 基于Week 1最佳策略开发Scale-Aware初始化
3. 🔬 在多个场景上验证泛化性

---

## 文档版本

- 创建日期：2025-11-10
- 最后更新：2025-11-10
- 相关分支：`claude/week1-q-init-experiments-011CUvXevjUyvvvvDkBspLeJ`
- 相关提交：3bc3bda (baseline结果)
