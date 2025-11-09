# Week 1 测试方案：Q-table初始化策略实验

**日期**: 2025-11-09
**目标**: 测试4种Q-table初始化策略，确定最优方案
**预计时间**: 5-7天
**预计运行次数**: 150次 (30基线 + 120实验)

---

## 📋 实验概述

### 问题背景

当前Q-learning实现将所有Q值初始化为0.0，导致：
- 早期没有探索偏好
- 容易过早收敛到局部最优
- 性能不稳定，种子方差大

**代码位置**: `src/planner/q_learning.py:64-66`

```python
# 当前实现（问题）
self.q_table: Dict[State, Dict[Action, float]] = {
    state: {action: 0.0 for action in self.actions} for state in self.states
}
```

### 实验目标

1. **收集基线数据**：了解当前零初始化的性能和方差
2. **测试新策略**：比较4种初始化策略的效果
3. **统计验证**：使用Wilcoxon检验和Cohen's d量化改进
4. **选择最优策略**：为Week 2-7的实验确定初始化方案

---

## 🧪 实验设计

### 测试的4种策略

| 策略 | 描述 | 理论依据 | 实现 |
|------|------|---------|------|
| **A: Zero** | 全部为0.0（基线） | 当前实现 | `QInitStrategy.ZERO` |
| **B: Uniform** | 全部为50.0 | 正偏置鼓励探索 | `QInitStrategy.UNIFORM` |
| **C: Action-Specific** | Matheuristic算子100.0，其他50.0 | 利用领域知识 | `QInitStrategy.ACTION_SPECIFIC` |
| **D: State-Specific** | 根据状态设置30.0-120.0 | 困住时更激进 | `QInitStrategy.STATE_SPECIFIC` |

### 实验参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 场景规模 | small, medium, large | 3种规模 |
| 随机种子 | 2025-2034 | 10个种子 |
| 初始化策略 | 4种 | 见上表 |
| 总运行次数 | 150 | 30(基线) + 120(实验) |

### 数据收集

每次运行收集：
- `baseline_cost`: 初始解成本
- `optimised_cost`: 优化后成本
- `improvement_ratio`: 改进率 = (baseline - optimised) / baseline
- `runtime`: 运行时间（秒）
- `final_epsilon`: 最终epsilon值
- `q_values`: 各状态的Q值（用于分析）

---

## 📅 执行时间表

### Day 1-3: 基线收集（30次运行）

**目标**: 建立当前性能基准

**执行步骤**:

```bash
# 1. 确保环境准备就绪
cd /home/user/R3
python -m pytest tests/optimization/q_learning/ -v  # 验证现有功能

# 2. 运行基线收集
chmod +x scripts/week1/01_baseline_collection.sh
./scripts/week1/01_baseline_collection.sh

# 预计时间：30-45分钟
# 输出：results/week1/baseline/baseline_*.json (30个文件)
```

**分析步骤**:

```bash
# 3. 分析基线数据
python scripts/week1/analyze_baseline.py

# 输出：
#   - results/week1/baseline/baseline_summary.json
#   - 控制台打印统计信息
```

**预期发现**:
- 小规模：改进率 ~60%，CV ~0.15
- 中规模：改进率 ~30%，CV ~0.25
- **大规模：改进率 ~7%，CV ~0.40 ← 确认问题存在**

### Day 4-7: 初始化策略实验（120次运行）

**目标**: 测试4种初始化策略

**执行步骤**:

```bash
# 1. 运行完整实验
chmod +x scripts/week1/02_init_experiments.sh
./scripts/week1/02_init_experiments.sh

# 预计时间：1.5-2小时
# 输出：results/week1/init_experiments/init_*.json (120个文件)
```

**分析步骤**:

```bash
# 2. 统计分析和可视化
python scripts/week1/analyze_init_strategies.py

# 输出：
#   - results/week1/init_experiments/statistical_comparison.csv
#   - results/week1/init_experiments/init_strategies_comparison.png
#   - results/week1/init_experiments/recommendations.json
#   - 控制台打印详细分析
```

---

## 📊 预期结果

### 性能改进预期

| 规模 | 基线（Zero） | Uniform预期 | 改进幅度 |
|------|-------------|------------|---------|
| Small | ~62% | ~63% | +1pp |
| Medium | ~30% | ~35% | +5pp |
| **Large** | **~7%** | **~12-15%** | **+5-8pp** |

### 方差降低预期

| 规模 | 基线CV | 预期CV | 降低幅度 |
|------|--------|--------|---------|
| Small | 0.15 | 0.12 | -20% |
| Medium | 0.25 | 0.20 | -20% |
| Large | 0.40 | 0.30 | -25% |

### 统计显著性预期

对于**Uniform vs Zero**对比：
- Small: p < 0.05, Cohen's d ~ 0.3 (小效应)
- Medium: p < 0.01, Cohen's d ~ 0.5 (中效应)
- **Large: p < 0.001, Cohen's d ~ 0.8+ (大效应)**

---

## ✅ 成功标准

Week 1实验成功需满足：

### 必须达成（Critical）

- [x] 所有150次实验成功运行
- [x] 至少1种策略在大规模上有统计显著改进（p < 0.05）
- [x] 大规模改进率从7%提升至≥10%
- [x] 生成完整的统计分析报告

### 期望达成（Desired）

- [x] 找到在所有规模都优于基线的策略
- [x] 大规模方差（CV）降低≥20%
- [x] 有清晰的策略推荐（基于统计证据）

### 可选达成（Optional）

- [x] 理解不同策略的适用场景
- [x] Q值分布的可视化分析
- [x] 初始Q值与最终性能的相关性分析

---

## 🔬 统计方法

### 1. Wilcoxon Signed-Rank Test（配对样本）

**用途**: 比较两种策略在相同种子下的性能差异

**原假设**: 两种策略的性能分布无显著差异

**拒绝域**: p < 0.05

**Python实现**:
```python
from scipy import stats
statistic, p_value = stats.wilcoxon(baseline_data, strategy_data)
```

### 2. Cohen's d（效应量）

**用途**: 量化改进的实际大小

**计算公式**:
```
d = (mean_strategy - mean_baseline) / pooled_std
```

**解释**:
- |d| < 0.2: 可忽略效应
- 0.2 ≤ |d| < 0.5: 小效应
- 0.5 ≤ |d| < 0.8: 中效应
- |d| ≥ 0.8: 大效应

### 3. 变异系数（Coefficient of Variation）

**用途**: 衡量性能稳定性

**计算公式**:
```
CV = std / mean
```

**解释**:
- CV < 0.15: 稳定性好
- 0.15 ≤ CV < 0.30: 稳定性中等
- CV ≥ 0.30: 稳定性差（需改进）

---

## 📁 输出文件结构

```
results/week1/
├── baseline/                          # Day 1-3输出
│   ├── baseline_small_seed2025.json
│   ├── baseline_small_seed2026.json
│   ├── ... (30个文件)
│   └── baseline_summary.json          # 汇总统计
│
└── init_experiments/                  # Day 4-7输出
    ├── init_zero_small_seed2025.json
    ├── init_uniform_small_seed2025.json
    ├── ... (120个文件)
    ├── statistical_comparison.csv     # 统计检验结果
    ├── init_strategies_comparison.png # 可视化图表
    └── recommendations.json            # 策略推荐
```

---

## 🚨 故障排查

### 问题1: 实验运行失败

**症状**: 脚本报错或超时

**解决方案**:
```bash
# 检查Python环境
python --version  # 应为3.9+

# 检查依赖
pip list | grep -E "scipy|numpy|pandas|matplotlib"

# 单独运行一个实验测试
python scripts/week1/run_experiment.py \
    --scenario small \
    --init_strategy uniform \
    --seed 2025 \
    --output test.json \
    --verbose
```

### 问题2: 结果文件缺失

**症状**: 分析脚本找不到文件

**解决方案**:
```bash
# 检查文件数量
ls results/week1/baseline/*.json | wc -l  # 应为30
ls results/week1/init_experiments/*.json | wc -l  # 应为120

# 查找缺失的配置
cd results/week1/init_experiments
for strategy in zero uniform action_specific state_specific; do
    for scenario in small medium large; do
        count=$(ls init_${strategy}_${scenario}_*.json 2>/dev/null | wc -l)
        echo "${strategy}/${scenario}: $count files (expected 10)"
    done
done
```

### 问题3: 统计分析报错

**症状**: `analyze_init_strategies.py` 报错

**解决方案**:
```bash
# 检查数据完整性
python -c "
import json
from pathlib import Path

files = list(Path('results/week1/init_experiments').glob('init_*.json'))
print(f'Found {len(files)} files')

for f in files[:5]:  # 检查前5个文件
    with open(f) as fp:
        data = json.load(fp)
        print(f'{f.name}: improvement = {data[\"improvement_ratio\"]:.2%}')
"
```

---

## 📈 下一步（Week 2）

Week 1完成后，将：
1. **选定最优初始化策略**（如Uniform或Action-Specific）
2. **在Week 2-7的所有实验中使用该策略**
3. **开始Week 2: Epsilon策略分析**

---

## 📝 检查清单

### 实验前准备
- [ ] 代码已提交到分支 `claude/week1-q-init-experiments-011CUvXevjUyvvvvDkBspLeJ`
- [ ] 所有脚本有执行权限（chmod +x）
- [ ] 创建results/week1目录
- [ ] Python环境依赖完整

### Day 1-3 基线收集
- [ ] 运行基线收集脚本
- [ ] 30个JSON文件生成
- [ ] 运行分析脚本
- [ ] 确认大规模性能问题（~7%）

### Day 4-7 初始化实验
- [ ] 运行完整实验脚本
- [ ] 120个JSON文件生成
- [ ] 运行统计分析脚本
- [ ] 生成可视化图表

### 实验后总结
- [ ] 至少1种策略有显著改进
- [ ] 统计报告完整
- [ ] 确定Week 2-7使用的策略
- [ ] 更新文档记录发现

---

**文档版本**: 1.0
**最后更新**: 2025-11-09
**状态**: 准备就绪，待执行
