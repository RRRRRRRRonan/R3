# 自适应Q-Learning参数框架实现方案

**创建日期**: 2025-11-05
**分支**: claude/experiment-results-seeds-2025-2026-011CUhJ2dCiVnBt3HEiNW3oY
**状态**: 实现中

---

## 📋 问题总结

### 静态参数调优失败的根本原因

经过完整的10-seed测试，参数调优不仅无效，反而使性能下降3.12%：

| 指标 | 调优前 | 调优后 | 变化 | 评价 |
|:-----|-------:|-------:|-----:|:----:|
| Q-learning平均 | 36.34% | 33.22% | **-3.12%** | ❌ 恶化 |
| 标准差 | 10.41% | 13.00% | +2.59% | ❌ 更不稳定 |

**关键发现**：
- ✅ 修复了1个案例（Seed 2027 Medium: +14.76%）
- ❌ 破坏了6个案例（最严重：2026 Large: -35.79%）
- 📊 "No Free Lunch"问题：统一参数无法同时优化所有seeds

---

## 🎯 解决方案：自适应参数框架

### 核心思想

**从静态统一参数 → 动态自适应参数**

根据以下因素实时调整参数：
1. **问题规模特征** (Small/Medium/Large)
2. **当前性能表现** (improvement_rate)
3. **搜索进度** (iteration_ratio)

---

## 🔧 实现方案

### Phase 1: 基于规模的参数自适应 (本次实现)

**原理**：不同规模问题需要不同的探索-利用平衡

```python
class AdaptiveQLearningParams:
    """自适应Q-learning参数管理器"""

    def __init__(self, scale: str):
        self.scale = scale
        self.base_params = self._get_base_params()

    def _get_base_params(self) -> dict:
        """根据问题规模返回基础参数"""
        if self.scale == 'small':
            # Small: 快速收敛，较少探索
            return {
                'alpha': 0.3,           # 较高学习率
                'epsilon_min': 0.05,    # 较低探索下限
                'stagnation_ratio': 0.15,  # 更早进入stuck
            }
        elif self.scale == 'medium':
            # Medium: 平衡探索与利用
            return {
                'alpha': 0.2,           # 中等学习率
                'epsilon_min': 0.1,     # 中等探索下限
                'stagnation_ratio': 0.25,  # 标准stuck阈值
            }
        else:  # large
            # Large: 更多探索，更慢学习
            return {
                'alpha': 0.15,          # 较低学习率
                'epsilon_min': 0.15,    # 较高探索下限
                'stagnation_ratio': 0.35,  # 更晚进入stuck
            }
```

### Phase 2: 基于性能的动态调整 (后续扩展)

```python
def adjust_by_performance(self, improvement_rate: float, iteration_ratio: float):
    """根据当前性能动态调整参数"""

    # 如果性能很差(<20%)且还在前半段迭代，增加探索
    if improvement_rate < 0.2 and iteration_ratio < 0.5:
        self.current_epsilon_min *= 1.2  # 增加20%探索
        self.current_alpha *= 0.9         # 降低10%学习率

    # 如果性能很好(>40%)且在后半段迭代，减少探索加快收敛
    elif improvement_rate > 0.4 and iteration_ratio > 0.5:
        self.current_epsilon_min *= 0.8  # 减少20%探索
        self.current_alpha *= 1.1         # 增加10%学习率
```

---

## 📐 实现细节

### 修改文件清单

1. **src/config/defaults.py**
   - 保留原有QLearningParams作为基准
   - 添加scale_adaptive参数开关

2. **新建 src/planner/adaptive_params.py**
   - AdaptiveParamsManager类
   - 根据规模和性能动态计算参数

3. **修改 src/planner/q_learning.py**
   - 在初始化时接收scale信息
   - 使用AdaptiveParamsManager获取参数

4. **修改 tests/optimization/*.py**
   - 传递scale信息到Q-learning agent

---

## 📊 预期效果

### 成功指标

| 指标 | 当前值 | 目标值 | 说明 |
|:-----|-------:|-------:|:-----|
| **平均差异** | +3.80% | **>6%** | Q vs Matheuristic |
| **t统计量** | 1.516 | **>2.045** | 统计显著性 |
| **灾难级失败** | 3个 | **0个** | <20%的案例 |
| **Large规模CV** | 31.23% | **<20%** | 稳定性 |

### 按规模预期改进

**Small规模**：
- 快速收敛策略应能保持或提升当前40.99%的表现
- 预期：41-43%

**Medium规模**：
- 平衡策略应能改善当前40.30%的表现
- 修复Seed 2027的失败（17.01% → >35%）
- 预期：42-45%

**Large规模**：
- 增加探索应能改善当前27.73%和高波动性(CV=31.23%)
- 修复Seed 2031的失败（8.34% → >25%）
- 预期：30-35%，CV<20%

---

## 🚀 实施步骤

### Step 1: 创建自适应参数管理器 ✅
```bash
# 新建 src/planner/adaptive_params.py
```

### Step 2: 修改Q-learning agent接口
```bash
# 修改 src/planner/q_learning.py
# 添加scale参数，集成adaptive_params
```

### Step 3: 更新测试套件
```bash
# 修改 tests/optimization/test_alns_qlearning.py
# 修改 tests/optimization/test_alns_matheuristic.py
```

### Step 4: 单seed验证
```bash
# 测试之前的失败案例
python scripts/generate_alns_visualization.py --seed 2027
python scripts/generate_alns_visualization.py --seed 2031
```

### Step 5: 完整10-seed测试
```bash
# 运行完整测试
for seed in {2025..2034}; do
    python scripts/generate_alns_visualization.py --seed $seed
done
```

### Step 6: 统计分析
```bash
python scripts/analyze_10seeds_results.py
```

---

## ⚖️ 风险与备选方案

### 潜在风险

1. **实现复杂度增加**
   - 缓解：保持原有static params作为fallback
   - 通过config开关控制是否启用adaptive

2. **可能仍然无法完全解决某些seeds**
   - 缓解：接受算法局限性
   - 在论文中诚实讨论

3. **调试和验证时间**
   - 缓解：先实现Phase 1（规模自适应）
   - Phase 2（性能自适应）作为后续改进

### 备选方案

如果自适应框架仍然不能达到统计显著性（t>2.045）：

**Plan B: 论文重新定位**
- 重点：Q-learning的自适应能力，而非绝对性能
- 对比实验：Q-learning vs Random vs Roulette Wheel
- 讨论：Q-learning在某些情况下优于Matheuristic的原因
- 目标期刊：Tier 2 (Computers & OR, Journal of Heuristics)

---

## 📅 时间估算

| 阶段 | 时间 | 说明 |
|:-----|:----:|:-----|
| 实现代码 | 3-4小时 | adaptive_params.py + 集成 |
| 单seed验证 | 30分钟 | 测试2027和2031 |
| 完整10-seed | 3-4小时 | 运行时间 |
| 分析结果 | 30分钟 | 统计分析 |
| **总计** | **1-2天** | 可以在周末完成 |

---

## 💡 关键创新点（论文角度）

1. **首次提出规模自适应的Q-learning参数框架**
   - 不同于传统的统一参数设置
   - 根据问题特征动态调整

2. **理论支撑**
   - "No Free Lunch"定理的实际应用
   - 探索-利用平衡的规模依赖性

3. **实验验证**
   - 10-seed稳健性测试
   - 与静态参数的对比实验

4. **普适性**
   - 框架可扩展到其他问题
   - 不限于EVRP

---

## 🎯 立即开始

接下来开始实现：

1. ✅ 创建 `src/planner/adaptive_params.py`
2. ⏳ 修改 `src/planner/q_learning.py`
3. ⏳ 更新测试文件
4. ⏳ 运行验证

**目标**：在1-2天内完成实现和测试，获得统计显著的结果！

---

## 📌 References

- DEEP_DIAGNOSIS_TUNING_FAILURE.md - 失败原因分析
- PARAMETER_TUNING_GUIDE.md - 之前的调参尝试
- 10seeds_analysis_and_publication_roadmap.md - 统计分析

---

**让我们开始实现！** 🚀
