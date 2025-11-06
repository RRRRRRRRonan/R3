# Phase 1 Q-Learning Stability Fix - Testing Guide

## ✅ 已完成的改动

Phase 1的所有改进已成功实施：

### 1. 配置参数更新 (src/config/defaults.py)

| 参数 | 原值 | 新值 | 改进目标 |
|------|------|------|---------|
| initial_epsilon | 0.12 | 0.20 | 更多初始探索 |
| epsilon_decay | 0.995 | 0.998 | 更慢衰减 |
| epsilon_min | 0.01 | 0.05 | 持续探索 |
| reward_improvement | 36.0 | 50.0 | 简化奖励 |
| reward_accepted | 10.0 | 5.0 | 降低噪音 |
| reward_rejected | -6.0 | -5.0 | 更温和惩罚 |
| time_penalty_threshold | 0.18 | 0.5 | 只惩罚真正慢的 |
| stagnation_ratio | 0.16 | 0.10 | 延迟stuck转换 |
| deep_stagnation_ratio | 0.40 | 0.18 | 延迟deep_stuck转换 |

**移除的参数** (简化奖励函数):
- ❌ `roi_positive_scale: 220.0`
- ❌ `roi_negative_scale: 260.0`
- ❌ `time_penalty_positive_scale: 1.1`
- ❌ `time_penalty_negative_scale: 6.0`
- ❌ `standard_time_penalty_scale: 0.2`

### 2. 保守初始Q值 (src/planner/alns.py)

| 状态 | 操作符 | 原值 | 新值 | LP vs greedy差距 |
|------|-------|------|------|-----------------|
| explore | lp | 15.0 | 12.0 | 1.5x → 1.3x |
| explore | greedy | 10.0 | 9.0 | - |
| stuck | lp | 30.0 | 15.0 | 3.0x → 1.5x |
| stuck | greedy | 10.0 | 10.0 | - |
| deep_stuck | lp | 35.0 | 20.0 | 3.5x → 2.0x |
| deep_stuck | greedy | 10.0 | 10.0 | - |

### 3. 简化奖励函数 (src/planner/alns.py)

**原始复杂度:**
```python
quality = base_reward + (improvement/cost) × 220 × scenario_multiplier
penalty = action_cost × scale(quality) × scenario_factor
reward = quality - penalty
# 涉及7个超参数
```

**新简化版本:**
```python
quality = min(50, relative_improvement × 500)  # 自然缩放
penalty = min(20, action_cost × 10) if matheuristic else 0
reward = quality - penalty
# 0个超参数，完全自适应
```

---

## 🧪 测试方法

### 快速验证（关键2个seed）

测试失败case和成功case，验证改进是否有效：

```bash
# 1. 测试失败case (seed 2026)
python scripts/generate_alns_visualization.py --seed 2026

# 预期改进:
# - Large规模: 从2.52%提升到15%+
# - Medium规模: 从40.08%提升到至少45%+

# 2. 测试成功case (seed 2028)
python scripts/generate_alns_visualization.py --seed 2028

# 预期结果:
# - Small规模: 保持57.74%的高性能
# - 确保改动没有破坏已经好的case
```

### 完整测试（10个seed）

运行完整的种子测试，验证整体方差降低：

```bash
# 测试所有10个seed
for seed in 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034; do
    echo "Testing seed $seed..."
    python scripts/generate_alns_visualization.py --seed $seed
done

# 或使用批量测试脚本（如果有）
python scripts/run_10seeds_test.py
```

---

## 📊 评估指标

### 主要成功标准

| 指标 | Phase 1前 | Phase 1目标 | 如何计算 |
|------|----------|-----------|---------|
| seed 2026 large | 2.52% | ≥15% | 直接从结果读取 |
| 10-seed方差 | ~50% | ≤30% | std(improvements) / mean(improvements) |
| vs Matheuristic | ~0.8x | ≥0.95x | q_learning_avg / matheuristic_avg |
| 最差case性能 | 2.52% | ≥10% | min(all improvements) |

### 详细评估清单

**1. 性能改进 (Performance)**
- [ ] seed 2026 large提升到15%+
- [ ] seed 2026 medium保持或提升
- [ ] 其他成功seed性能未下降

**2. 稳定性 (Stability)**
- [ ] 10个seed中，至少8个 ≥ matheuristic的80%
- [ ] 最好和最差的差距 < 30%
- [ ] 没有极端失败case (<5%)

**3. 探索行为 (Exploration)**
- [ ] epsilon在300次迭代后仍 ≥ 0.06 (vs原来的0.027)
- [ ] LP在explore阶段的使用率 < 70% (vs原来的88%)
- [ ] 各操作符都有机会被尝试

---

## 🔍 调试检查

如果结果不理想，检查以下内容：

### 1. 验证参数生效

```python
# 在Python中验证新参数
from config import DEFAULT_Q_LEARNING_PARAMS

params = DEFAULT_Q_LEARNING_PARAMS
print(f"initial_epsilon: {params.initial_epsilon}")  # 应该是0.20
print(f"epsilon_decay: {params.epsilon_decay}")      # 应该是0.998
print(f"epsilon_min: {params.epsilon_min}")          # 应该是0.05
print(f"stagnation_ratio: {params.stagnation_ratio}") # 应该是0.10

# 检查ROI参数是否已移除
try:
    print(params.roi_positive_scale)
    print("❌ ERROR: ROI parameters still exist!")
except AttributeError:
    print("✓ ROI parameters successfully removed")
```

### 2. 检查初始Q值

在运行开始时，查看日志中的Q值分布：

```
Q-Learning算子统计
epsilon=0.200
状态 explore:
  (random_removal, lp) -> 使用 X 次, Q=  12.000  ← 应该是12而不是15
  (random_removal, greedy) -> 使用 Y 次, Q=   9.000  ← 应该是9而不是10
```

### 3. 查看epsilon衰减曲线

添加调试输出（可选）：

```python
# 在alns.py的optimize函数中，每50次迭代打印epsilon
if (iteration + 1) % 50 == 0:
    print(f"Iteration {iteration+1}: epsilon={self._q_agent.epsilon:.4f}")

# 预期输出:
# Iteration 50: epsilon≈0.18 (vs原来的0.072)
# Iteration 100: epsilon≈0.17 (vs原来的0.044)
# Iteration 150: epsilon≈0.16 (vs原来的0.027)
```

### 4. 验证奖励计算

在`_compute_q_reward`中添加临时日志：

```python
# 在return前添加
if iteration < 10:  # 只打印前10次
    print(f"Reward: quality={quality:.1f}, penalty={penalty:.1f}, "
          f"total={quality-penalty:.1f}, is_new_best={is_new_best}")
```

预期应该看到：
- 奖励值在 -10 到 +100 之间（原来可能 -200 到 +300）
- 时间惩罚 ≤ 20（原来可能 > 100）

---

## 📈 结果分析

### 收集结果

所有测试结果会保存在：
- `docs/data/alns_regression_results [seed].json`
- `docs/figures/alns_regression_improvements [seed].svg`

### 生成汇总报告

```bash
# 如果有分析脚本
python scripts/analyze_10seeds_results.py

# 或手动汇总
cat docs/data/alns_regression_results*.json | grep "improvement_ratio"
```

### 对比Phase 1前后

创建对比表格：

| Seed | Scale | Phase 1前 | Phase 1后 | 改进 |
|------|-------|----------|----------|------|
| 2026 | large | 2.52% | ???% | +??? |
| 2026 | medium | 40.08% | ???% | +??? |
| 2028 | small | 57.74% | ???% | +??? |

---

## 🚨 常见问题

### Q1: 结果没有改善怎么办？

**可能原因:**
1. 旧的配置缓存：删除 `__pycache__` 并重新运行
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   ```

2. 使用了旧的默认参数：确保代码中没有硬编码的旧参数

3. 需要更多迭代：检查是否large规模使用了足够的迭代次数（430次）

### Q2: 某些seed反而变差了？

这是正常的，Phase 1的目标是降低方差，不是提升所有seed：
- 允许个别seed轻微下降（<5%）
- 关注整体方差和最差case的改进
- 如果多数seed都变差，需要回滚调查

### Q3: 探索率衰减还是太快？

可以进一步调整：
```python
# 在defaults.py中
initial_epsilon: float = 0.25  # 进一步提高
epsilon_decay: float = 0.999   # 进一步减缓
epsilon_min: float = 0.08      # 进一步提高最小值
```

### Q4: LP还是被过度使用？

可以进一步降低初始Q值：
```python
# 在alns.py的_default_q_learning_initial_q中
'explore': {
    'lp': 10.0,      # 进一步降低（原12.0）
    'regret2': 10.0,
    'greedy': 9.0,
    'random': 5.0,
},
```

---

## 📋 报告模板

测试完成后，用以下模板报告结果：

```markdown
# Phase 1测试报告

## 环境
- 分支: claude/investigate-qlearning-seed-variance-011CUr3KaWkShxPhYokPR6xe
- Commit: e9c184b
- 测试日期: YYYY-MM-DD

## 关键结果

### seed 2026 (主要失败case)
- Large规模: 2.52% → ???% (目标≥15%)
- Medium规模: 40.08% → ???%

### seed 2028 (主要成功case)
- Small规模: 57.74% → ???% (期望保持)

### 10-seed汇总
- 平均改进率: ???%
- 性能方差: ???% (目标≤30%)
- vs Matheuristic: ???x (目标≥0.95x)
- 最差case: ???% (目标≥10%)

## 评估

✅/❌ 达到Phase 1目标
✅/❌ seed 2026 large ≥15%
✅/❌ 方差 ≤30%
✅/❌ 无性能大幅下降

## 建议

[是否需要进一步调整参数/进入Phase 2]
```

---

## 🎯 下一步

### 如果Phase 1成功 (达到目标)

1. **进入Phase 2**: 实施动态状态转换
   - 添加学习进展监控
   - 实现自适应状态阈值
   - 预期方差进一步降至15%

2. **准备论文材料**:
   - 整理实验数据
   - 生成对比图表
   - 撰写方法论部分

### 如果Phase 1部分成功 (有改进但未达标)

1. **参数微调**:
   - 调整epsilon参数（见Q3）
   - 调整初始Q值（见Q4）
   - 重新测试

2. **深入分析**:
   - 查看哪些seed改进，哪些未改进
   - 分析改进/未改进的特征
   - 针对性优化

### 如果Phase 1失败 (无明显改进)

1. **回滚并诊断**:
   ```bash
   git revert e9c184b
   ```

2. **重新分析问题**:
   - 是否还有其他算法设计缺陷
   - Phase 1的假设是否正确
   - 考虑替代方案

---

## 📞 获取帮助

如果遇到问题，可以：

1. 查看 `QLEARNING_STABILITY_ANALYSIS.md` 的理论分析
2. 检查代码中的详细注释
3. 对比 commit e9c184b 前后的差异
4. 提交issue描述具体问题

---

**祝测试顺利！期待看到方差大幅降低的好结果！** 🚀
