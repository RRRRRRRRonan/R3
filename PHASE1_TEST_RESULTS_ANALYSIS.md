# Phase 1 自适应参数测试结果分析

**日期**: 2025-11-05
**测试种子**: 2027, 2031, 2034 (3个之前的问题seeds)
**分支**: claude/fix-qlearning-failures-20251103-011CUhJ2dCiVnBt3HEiNW3oY

---

## 📊 测试结果汇总

### Seed 2027 表现

| 规模 | Q-learning | Matheuristic | 差值 | 评价 |
|:-----|----------:|-----------:|-----:|:----:|
| Small | 40.13% | 35.92% | **+4.21%** | ✅ 赢 |
| Medium | **29.66%** | **48.31%** | **-18.65%** | ❌❌❌ 严重输 |
| Large | 36.11% | 35.20% | +0.91% | ✅ 赢 |

**分析**：
- ✅ Small和Large规模有改善
- ❌ **Medium规模灾难性失败**：从17.01%提升到29.66%（+12.65%），但仍远低于Matheuristic的48.31%
- 🔍 Medium参数(α=0.2, ε_min=0.1)可能不适合这个seed

---

### Seed 2031 表现

| 规模 | Q-learning | Matheuristic | 差值 | 评价 |
|:-----|----------:|-----------:|-----:|:----:|
| Small | 40.49% | 3.34% | **+37.15%** | ✅✅✅ 大胜 |
| Medium | **40.12%** | **53.87%** | **-13.75%** | ❌❌ 严重输 |
| Large | **8.34%** | **28.34%** | **-20.00%** | ❌❌❌ 灾难性失败 |

**分析**：
- ✅ Small规模表现优异（+37%）
- ❌ **Medium规模严重失败**：输13.75%
- ❌ **Large规模几乎无改善**：仍然只有8.34%，与原始值相同

---

### Seed 2034 表现

| 规模 | Q-learning | Matheuristic | 差值 | 评价 |
|:-----|----------:|-----------:|-----:|:----:|
| Small | 44.15% | 48.07% | -3.92% | ⚠️ 小输 |
| Medium | **29.29%** | **39.41%** | **-10.12%** | ❌❌ 严重输 |
| Large | 30.35% | 35.16% | -4.81% | ⚠️ 小输 |

**分析**：
- ⚠️ Small和Large规模略输但差距不大
- ❌ **Medium规模严重失败**：输10.12%

---

## 🔍 核心问题诊断

### 问题1: Medium规模全面失败 🚨

**数据**：
- Seed 2027 Medium: -18.65%
- Seed 2031 Medium: -13.75%
- Seed 2034 Medium: -10.12%
- **平均输**: -14.17%

**当前参数**：
```python
Medium: α=0.2, ε_min=0.1, stagnation_ratio=0.25
```

**可能原因**：
1. ❌ **学习率α=0.2太低**：无法快速学习有效模式
2. ❌ **探索率ε_min=0.1可能不够**：Medium问题搜索空间比想象的大
3. ❌ **stagnation_ratio=0.25可能太晚**：错过了LP修复的最佳时机

---

### 问题2: Large规模Seed 2031仍然灾难性失败

**数据**：
- Seed 2031 Large: 8.34% (几乎无改善)
- 输给Matheuristic: -20.00%

**当前参数**：
```python
Large: α=0.15, ε_min=0.15, stagnation_ratio=0.35
```

**可能原因**：
1. ❌ **学习率α=0.15太低**：Large问题也需要一定的学习速度
2. ❌ **stagnation_ratio=0.35太晚**：这个seed可能需要更早使用LP
3. ❌ **问题特性**：可能不只是规模问题，还有其他特征（空间分布、时间窗紧度等）

---

### 问题3: Small规模表现不稳定

**数据**：
- Seed 2027: +4.21% ✅
- Seed 2031: +37.15% ✅✅✅
- Seed 2034: -3.92% ⚠️

**分析**：
- Small规模的参数相对较好，但Seed 2034仍然输
- 可能需要微调α和ε_min

---

## 💡 根本原因分析

### 发现1: 规模不是唯一因素

**证据**：
- 同样是Medium规模，三个seeds都失败，但失败程度不同
- 同样是Large规模，Seed 2027成功但Seed 2031失败

**结论**：
- 问题的难度不仅取决于规模（任务数量）
- 还取决于**问题特征**：
  - 空间分布（任务点分散程度）
  - 时间窗紧度
  - 充电站密度
  - Seed的随机性影响

### 发现2: 当前参数过于保守

**Medium规模参数分析**：
```python
α=0.2      # 学习速度太慢
ε_min=0.1  # 探索不够
stag=0.25  # 进入stuck太晚
```

**对比Matheuristic的优势**：
- Matheuristic使用Roulette Wheel，权重更新更激进
- Matheuristic可能在LP修复上投入更多
- Q-learning的三态设计（explore→stuck→deep_stuck）太僵化

### 发现3: Large问题的特殊性

**Seed 2031 Large只有8.34%的原因**：
1. 可能陷入了特定的局部最优
2. LP修复启动太晚（stagnation_ratio=0.35意味着要等35%迭代无改进）
3. 学习率太低，无法快速调整策略

---

## 🎯 方案建议

### ❌ 不建议：继续Phase 2

**理由**：
1. Phase 1的基础参数选择就有严重问题
2. Medium规模全面失败说明参数方向错误
3. 在错误的基础上叠加性能自适应只会更复杂

### ✅ 推荐：Phase 1.5 参数重新校准

**方案A: 调整参数值** (推荐，2-3天)

基于测试结果，调整参数：

```python
# 当前参数（失败）
Small:  α=0.3,  ε_min=0.05, stag=0.15
Medium: α=0.2,  ε_min=0.1,  stag=0.25  # ❌ 失败
Large:  α=0.15, ε_min=0.15, stag=0.35  # ❌ 部分失败

# 建议新参数（更激进）
Small:  α=0.35, ε_min=0.08, stag=0.12  # 更快学习
Medium: α=0.30, ε_min=0.12, stag=0.18  # ⬆️ 提高学习率，提早stuck
Large:  α=0.25, ε_min=0.15, stag=0.22  # ⬆️ 提高学习率，提早stuck
```

**调整逻辑**：
1. **提高alpha**：Medium和Large都需要更快的学习速度
2. **提早stuck**：降低stagnation_ratio，更早利用LP修复
3. **适度探索**：epsilon_min略微增加，但不过度

---

**方案B: 问题特征驱动** (更复杂，1-2周)

不只看规模，还看问题特征：

```python
def select_params(problem):
    num_tasks = len(problem.tasks)

    # 计算额外特征
    spatial_variance = calc_spatial_variance(problem)  # 任务分散度
    tw_tightness = calc_tw_tightness(problem)          # 时间窗紧度

    # 基础规模参数
    if num_tasks <= 18:
        base_alpha, base_epsilon = 0.35, 0.08
    elif num_tasks <= 26:
        base_alpha, base_epsilon = 0.30, 0.12
    else:
        base_alpha, base_epsilon = 0.25, 0.15

    # 根据特征微调
    if spatial_variance > threshold:
        base_alpha *= 1.15  # 更分散→更快学习

    if tw_tightness > threshold:
        stagnation_ratio *= 0.85  # 更紧→更早LP

    return params
```

---

**方案C: 简化为2规模** (快速，1天)

当前3规模可能过于细分：

```python
# Small (≤20任务)
α=0.35, ε_min=0.08, stag=0.15

# Large (>20任务)
α=0.28, ε_min=0.13, stag=0.20
```

---

**方案D: 回归统一参数，但更优化** (最快，1天)

基于测试数据，找一组对所有seeds都相对好的统一参数：

```python
# 折中参数（针对所有规模）
α=0.28, ε_min=0.12, stag=0.18
```

这可能不会在所有seeds上都最优，但避免灾难性失败。

---

## 📋 我的推荐行动

### 立即行动：方案A（参数重新校准）

**步骤**：
1. 修改 `adaptive_params.py` 中的参数值
2. 重新测试Seeds 2027, 2031, 2034
3. 如果改善，测试完整10 seeds
4. 如果仍不理想，考虑方案B或D

**时间**: 2-3天（含测试）

**成功标准**：
- Medium规模：Q-learning vs Matheuristic差距 < 5%
- Large问题Seed 2031: 改进到 > 20%
- 无灾难性失败（<15%）

---

## 🔧 具体参数修改建议

### 修改src/planner/adaptive_params.py

```python
def _get_scale_adjustments(self, scale: ScaleType) -> dict:
    if scale == "small":
        return {
            "alpha": 0.35,              # ⬆️ 从0.3提高
            "epsilon_min": 0.08,        # ⬆️ 从0.05提高
            "stagnation_ratio": 0.12,   # ⬇️ 从0.15降低（更早stuck）
            "deep_stagnation_ratio": 0.30,
        }
    elif scale == "medium":
        return {
            "alpha": 0.30,              # ⬆️ 从0.2提高（关键！）
            "epsilon_min": 0.12,        # ⬆️ 从0.1提高
            "stagnation_ratio": 0.18,   # ⬇️ 从0.25降低（更早stuck，关键！）
            "deep_stagnation_ratio": 0.40,
        }
    else:  # large
        return {
            "alpha": 0.25,              # ⬆️ 从0.15提高（关键！）
            "epsilon_min": 0.15,        # 保持
            "stagnation_ratio": 0.22,   # ⬇️ 从0.35降低（更早stuck，关键！）
            "deep_stagnation_ratio": 0.48,
        }
```

**调整原理**：
1. **Medium规模**：大幅提高alpha（0.2→0.3），降低stagnation（0.25→0.18）
2. **Large规模**：提高alpha（0.15→0.25），降低stagnation（0.35→0.22）
3. **Small规模**：微调即可

---

## 📊 预期效果（调整后）

### Seed 2027
- Medium: 29.66% → 目标 42-45% (接近Matheuristic的48%)

### Seed 2031
- Medium: 40.12% → 目标 50-52% (接近Matheuristic的54%)
- Large: 8.34% → 目标 22-26% (接近Matheuristic的28%)

### Seed 2034
- Medium: 29.29% → 目标 35-38% (接近Matheuristic的39%)

---

## ⚠️ 如果调整后仍然失败

考虑以下可能性：

1. **Q-learning三态设计的根本缺陷**
   - explore→stuck→deep_stuck太僵化
   - 需要重新设计状态机

2. **Matheuristic本身优势太大**
   - LP修复的效果远超Q-learning的学习
   - 可能需要接受这个现实

3. **论文重新定位**（Plan E）
   - 强调Q-learning的**在线学习能力**
   - 讨论**算法设计的trade-off**
   - 诚实分析**适用场景**

---

## 🎯 总结

### 核心发现
1. ❌ Phase 1参数选择过于保守，尤其是Medium和Large
2. ❌ Medium规模全面失败（平均-14%）
3. ⚠️ 规模不是唯一因素，问题特征也很重要

### 下一步
1. ✅ **推荐**：实施方案A，重新校准参数（更激进的alpha，更早的stuck）
2. ⏳ 测试3个seeds验证
3. ✅ 如果改善，继续完整10-seed
4. ❌ **不推荐**：直接进入Phase 2

### 时间估算
- 修改参数：30分钟
- 测试3 seeds：1-2小时
- 完整10 seeds：3-4小时
- **总计**：1天内可完成验证

---

**请告诉我是否实施方案A？我会立即修改参数并推送到分支。**
