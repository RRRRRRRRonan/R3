# 本周工作计划 (Week 1)

## 📋 总体目标

**主要目标**: 理解并验证Large scale性能，实现多随机种子测试框架

**背景**: 您观察到seed=2026时Large scale达到38.1%，但seed=2025时只有23.33%。这表明性能对随机种子高度敏感。

## ✅ 已完成

### 1. 创建多随机种子测试框架
- ✅ `scripts/batch_experiments.py` - 完整的批量实验脚本
  - 支持多种子、多规模、多solver测试
  - 自动计算统计指标（均值、标准差、置信区间）
  - 自动进行统计显著性检验（配对t检验）
  - 结果保存为JSON格式

- ✅ `scripts/quick_seed_test.py` - 快速种子对比脚本
  - 适合快速验证少量种子
  - 直接对比Q-learning vs Matheuristic

### 2. 创建分析文档
- ✅ `docs/seed_variance_analysis.md` - 种子方差分析
  - 解释为什么会出现种子敏感性
  - 分析可能原因
  - 提出解决方案

### 3. 创建新分支
- ✅ `claude/multi-seed-experiments-*` - 多种子实验分支

## 🔄 进行中

### 测试运行
-运行中 `batch_experiments.py` 测试（small scale, 2 seeds）
  - 预计完成时间：5-10分钟
  - 将验证脚本功能是否正常

## 📅 本周剩余任务

### Day 1-2: 验证与数据收集

**任务1.1: 验证seed=2026的Large scale性能** ⚡
```bash
python scripts/quick_seed_test.py --scale large --seeds 2025 2026 2027
```
**目的**: 确认您观察到的38.1%是否可复现

**任务1.2: 运行完整的10种子实验**
```bash
# Small + Medium + Large，每个10个种子
python scripts/batch_experiments.py \
  --scales small medium large \
  --solvers matheuristic q_learning \
  --seeds 10 \
  --output docs/data/multi_seed_results_10.json
```
**预计时间**: 2-3小时
**输出**:
- JSON文件包含所有原始数据
- 统计摘要（均值、标准差、置信区间）
- 统计检验结果

### Day 3-4: 分析与理解

**任务2.1: 分析Large scale真实性能**

基于10种子实验结果，回答：
1. Large scale Q-learning的**真实均值**是多少？
2. **标准差**有多大？（方差有多严重）
3. 与Matheuristic相比是否有统计显著差异？

**关键问题**:
```
如果均值 > 30%:
  ✅ 性能可接受
  → 重点：降低方差，提高稳定性
  → 论文中强调均值性能，讨论方差问题

如果均值 < 30%:
  ⚠️ 需要进一步优化
  → 重点：提升均值性能
  → 可能需要调整Q-learning参数或状态设计
```

**任务2.2: 创建性能分析脚本**
```python
# scripts/analyze_variance.py
# 功能：
# 1. 识别"好种子"和"坏种子"
# 2. 对比它们的scenario特征
# 3. 分析Q-learning在不同种子下的行为差异
```

### Day 5-7: 决策与行动

**情景A: 如果Large scale均值表现良好（>30%）**

**优先级**: 降低方差 + 准备论文数据

1. **参数微调**（可选）
   - 尝试降低epsilon以减少随机性
   - 测试是否能降低标准差而不损失均值

2. **准备论文数据**
   - 运行所有对比方法（至少10种子）
   - 制作表格和图表
   - 计算统计显著性

3. **下周开始**: Solomon benchmark实现

---

**情景B: 如果Large scale均值表现不佳（<30%）**

**优先级**: 提升性能

1. **诊断问题**
   - 分析Q-learning的算子选择
   - 检查状态转换是否合理
   - 查看stuck/deep_stuck触发时机

2. **尝试改进**
   - 调整stagnation_threshold
   - 修改reward function scaling
   - 优化状态设计

3. **验证改进**
   - 再次运行10种子测试
   - 对比改进前后

## 📊 预期输出（本周末）

### 数据产出
1. ✅ `docs/data/multi_seed_results_10.json` - 10种子完整数据
2. ✅ 统计分析报告

### 文档产出
1. ✅ 种子方差分析文档（已完成）
2. ✅ Large scale性能评估报告（待完成）
3. ✅ 下周工作计划（待完成）

### 代码产出
1. ✅ 批量实验脚本（已完成）
2. ✅ 快速测试脚本（已完成）
3. ⏳ 方差分析脚本（待完成）

## 🎯 关键问题待回答

本周结束时，我们需要回答：

### 1. Large scale性能评估
- Q-learning在Large scale的**真实均值性能**？
- 性能方差有多大？
- 是否稳定可靠？

### 2. 与Matheuristic对比
- Q-learning是否优于Matheuristic？
- 差异是否具有统计显著性？
- 在哪些规模上有优势/劣势？

### 3. 发表策略
- 当前数据是否足够支撑期刊论文？
- 需要哪些补充实验？
- 预计何时可以投稿？

## 💻 快速命令参考

### 运行实验
```bash
# 快速测试（3个种子）
python scripts/quick_seed_test.py --scale large --seeds 2025 2026 2027

# 完整测试（10种子，所有规模）
python scripts/batch_experiments.py --seeds 10

# 只测试Large scale
python scripts/batch_experiments.py --scales large --seeds 10

# 测试所有三个solver
python scripts/batch_experiments.py --solvers minimal matheuristic q_learning --seeds 10
```

### 查看结果
```bash
# 查看JSON结果
cat docs/data/batch_experiments_results.json | python -m json.tool

# 提取统计摘要
python -c "
import json
with open('docs/data/batch_experiments_results.json') as f:
    data = json.load(f)
    for scale in data:
        print(f'{scale}:')
        for solver in data[scale]:
            stats = data[scale][solver]['statistics']
            print(f'  {solver}: {stats[\"mean\"]*100:.2f}% ± {stats[\"std\"]*100:.2f}%')
"
```

## 📝 日志

### 2025-11-01 (Today)
- ✅ 创建批量实验框架
- ✅ 创建分析文档
- 🔄 运行初步测试（small scale, 2 seeds）
- ⏳ 待验证seed=2026的Large scale结果

### 下一个工作日
- ⏳ 运行10种子完整实验
- ⏳ 分析结果
- ⏳ 决定优化策略

---

**注意**: 所有实验都在新分支 `claude/multi-seed-experiments-*` 上进行，不影响当前的工作分支。
