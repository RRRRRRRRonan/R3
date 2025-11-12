# Week 1 准备完毕

**状态**: ✅ 实验已完成 (2025-11-12)
**最终结果**: 请参见 [WEEK1_RESULTS.md](./WEEK1_RESULTS.md)

---

**日期**: 2025-11-09
**分支**: `claude/week1-q-init-experiments-011CUvXevjUyvvvDkBspLeJ`
**原状态**: ✅ 准备就绪，待执行实验

---

## ✅ 已完成的工作

### 1. 核心代码实现

| 文件 | 行数 | 功能 | 状态 |
|------|------|------|------|
| `src/planner/q_learning_init.py` | 200 | 4种初始化策略实现 | ✅ 完成 |
| `src/planner/q_learning.py` | +60 | Agent支持初始化策略 | ✅ 完成 |

**初始化策略**:
- ✅ ZERO: 全部0.0（基线）
- ✅ UNIFORM: 全部50.0
- ✅ ACTION_SPECIFIC: Matheuristic 100.0, 其他 50.0
- ✅ STATE_SPECIFIC: 根据状态 30.0-120.0

### 2. 实验脚本

| 脚本 | 功能 | 运行时间 | 输出 | 状态 |
|------|------|---------|------|------|
| `run_experiment.py` | 单次实验运行器 | 1-2分钟 | JSON | ✅ 完成 |
| `01_baseline_collection.sh` | 基线收集 | ~30分钟 | 30个JSON | ✅ 完成 |
| `02_init_experiments.sh` | 完整实验 | ~2小时 | 120个JSON | ✅ 完成 |

### 3. 分析脚本

| 脚本 | 功能 | 输出 | 状态 |
|------|------|------|------|
| `analyze_baseline.py` | 基线统计分析 | summary.json + 控制台报告 | ✅ 完成 |
| `analyze_init_strategies.py` | 策略对比分析 | CSV + PNG + JSON | ✅ 完成 |

**统计方法**:
- ✅ Wilcoxon signed-rank test (配对检验)
- ✅ Cohen's d (效应量)
- ✅ Coefficient of Variation (稳定性)

### 4. 文档

| 文档 | 内容 | 状态 |
|------|------|------|
| `docs/experiments/WEEK1_TEST_PLAN.md` | 详细测试计划 | ✅ 完成 |
| `scripts/week1/README.md` | 快速参考 | ✅ 完成 |
| `docs/WEEKLY_IMPLEMENTATION_PLAN.md` | 21周总计划 | ✅ 已存在 |

### 5. 测试验证

| 测试 | 覆盖范围 | 结果 | 状态 |
|------|---------|------|------|
| `test_installation.py` | 安装验证 | 4/4通过 | ✅ 通过 |
| `test_q_learning_init.py` | 单元测试 | 6个测试 | ✅ 完成 |

**验证结果**:
```
✓ PASS: Module Imports
✓ PASS: Q-table Initialization
✓ PASS: Q-learning Agent Integration
✓ PASS: Script Files
```

---

## 📋 实验参数

### 测试矩阵

| 维度 | 值 | 数量 |
|------|-----|------|
| 初始化策略 | zero, uniform, action_specific, state_specific | 4 |
| 场景规模 | small, medium, large | 3 |
| 随机种子 | 2025-2034 | 10 |
| **总运行次数** | 4 × 3 × 10 | **120** |
| **基线运行** | 1 × 3 × 10 | **30** |
| **总计** | | **150** |

### 预期运行时间

- **Day 1-3 基线收集**: 30分钟
- **Day 4-7 初始化实验**: 2小时
- **分析与报告**: 10分钟
- **总计**: ~2.5小时

---

## 🎯 实验目标

### 主要目标

1. **确认问题存在**
   - 验证大规模性能降级（预期: ~7%）
   - 验证高种子方差（预期: CV ~0.40）

2. **找到最优策略**
   - 至少1种策略显著优于零初始化 (p < 0.05)
   - 大规模改进: 7% → 10-15%

3. **降低方差**
   - 大规模CV: 0.40 → 0.30 (降低25%)

### 成功标准

| 标准 | 目标值 | 优先级 |
|------|--------|--------|
| 大规模改进率提升 | ≥3百分点 | 必须 ✓ |
| 至少1种策略显著改进 | p < 0.05 | 必须 ✓ |
| CV降低 | ≥20% | 期望 ✓ |
| 策略推荐明确 | 基于统计证据 | 期望 ✓ |

---

## 🚀 如何开始实验

### 前提条件检查

```bash
# 1. 切换到Week 1分支
git checkout claude/week1-q-init-experiments-011CUvXevjUyvvvvDkBspLeJ

# 2. 验证安装
PYTHONPATH=/home/user/R3/src:$PYTHONPATH \
    python scripts/week1/test_installation.py

# 应输出: ✓ All tests passed! Week 1 is ready to use.
```

### 执行实验

#### Step 1: 基线收集（Day 1-3）

```bash
# 运行基线收集
./scripts/week1/01_baseline_collection.sh

# 预期输出: 30个JSON文件
ls results/week1/baseline/*.json | wc -l  # 应为30

# 分析基线
python scripts/week1/analyze_baseline.py
```

**预期输出**:
```
SMALL Scale: ~62% ± 9%, CV = 0.15
MEDIUM Scale: ~30% ± 8%, CV = 0.25
LARGE Scale: ~7% ± 3%, CV = 0.40  ← 确认问题

⚠ ALERT: Large-scale performance is very poor (< 10%)
```

#### Step 2: 初始化实验（Day 4-7）

```bash
# 运行完整实验
./scripts/week1/02_init_experiments.sh

# 预期输出: 120个JSON文件
ls results/week1/init_experiments/*.json | wc -l  # 应为120

# 统计分析
python scripts/week1/analyze_init_strategies.py
```

**预期输出**:
```
- statistical_comparison.csv
- init_strategies_comparison.png
- recommendations.json
- 控制台详细报告
```

---

## 📊 预期结果示例

### 基线分析预期

```
LARGE Scale:
  Mean improvement: 7.23% ± 2.89%
  Range: [4.12%, 11.45%]
  CV: 0.400

⚠ WARNING: Large-scale performance degrades significantly
   (7.23% vs 62.45% for small scale)
```

### 策略对比预期

| 策略 | Small | Medium | Large | p-value (Large) |
|------|-------|--------|-------|-----------------|
| Zero (baseline) | 62% | 30% | 7% | - |
| Uniform | 63% | 35% | 12% | < 0.01 |
| Action-Spec | 63% | 36% | 14% | < 0.001 |
| State-Spec | 62% | 33% | 10% | < 0.05 |

### 推荐策略预期

```json
{
  "small": "uniform",
  "medium": "action_specific",
  "large": "action_specific"
}
```

---

## 📁 输出文件清单

### Day 1-3 输出

```
results/week1/baseline/
├── baseline_small_seed2025.json      }
├── baseline_small_seed2026.json      } 10个
├── ...                               }
├── baseline_medium_seed2025.json     } 10个
├── ...                               }
├── baseline_large_seed2025.json      } 10个
├── ...                               }
└── baseline_summary.json             # 汇总
```

### Day 4-7 输出

```
results/week1/init_experiments/
├── init_zero_small_seed2025.json           }
├── init_uniform_small_seed2025.json        } 120个
├── init_action_specific_small_seed2025.json} 实验
├── init_state_specific_small_seed2025.json } 结果
├── ...                                     }
├── statistical_comparison.csv              # 统计检验
├── init_strategies_comparison.png          # 可视化
└── recommendations.json                     # 推荐策略
```

---

## 🔍 故障排查

### 问题: 实验失败

**症状**: 脚本报错或JSON文件缺失

**解决方案**:
```bash
# 测试单次运行
PYTHONPATH=/home/user/R3/src:$PYTHONPATH \
python scripts/week1/run_experiment.py \
    --scenario small \
    --init_strategy uniform \
    --seed 2025 \
    --output test.json \
    --verbose

# 检查输出
cat test.json | python -m json.tool
```

### 问题: 分析脚本报错

**症状**: `analyze_*.py` 无法找到文件

**解决方案**:
```bash
# 检查文件数量
ls results/week1/baseline/*.json | wc -l    # 应为30
ls results/week1/init_experiments/*.json | wc -l  # 应为120

# 检查JSON格式
python -c "
import json
from pathlib import Path
for f in Path('results/week1/baseline').glob('*.json'):
    with open(f) as fp:
        json.load(fp)  # 会报错如果格式不对
print('All files valid')
"
```

### 问题: 导入模块失败

**症状**: `ModuleNotFoundError: No module named 'planner'`

**解决方案**:
```bash
# 设置PYTHONPATH
export PYTHONPATH=/home/user/R3/src:$PYTHONPATH

# 或在每次运行时指定
PYTHONPATH=/home/user/R3/src:$PYTHONPATH python scripts/week1/...
```

---

## 📈 下一步（Week 2）

Week 1完成后：

1. **审查结果**
   - 查看 `recommendations.json`
   - 确认最优初始化策略

2. **决策**
   - 选择Week 2-7使用的初始化策略
   - 通常选择: **uniform** 或 **action_specific**

3. **开始Week 2**
   - 切换到Week 2分支
   - 开始Epsilon策略分析
   - 使用Week 1选定的初始化策略

---

## ✅ 检查清单

### 实验前
- [x] 代码已推送到分支
- [x] 验证测试通过 (4/4)
- [x] 脚本有执行权限
- [x] 了解实验目标

### Day 1-3
- [ ] 运行基线收集脚本
- [ ] 验证30个JSON文件生成
- [ ] 运行基线分析
- [ ] 确认大规模问题存在

### Day 4-7
- [ ] 运行完整实验脚本
- [ ] 验证120个JSON文件生成
- [ ] 运行统计分析
- [ ] 生成可视化图表

### 实验后
- [ ] 至少1种策略有显著改进
- [ ] 大规模性能提升≥3百分点
- [ ] CV降低≥20%
- [ ] 确定Week 2使用的策略

---

## 📞 获取帮助

如有问题：

1. **查看文档**
   - `docs/experiments/WEEK1_TEST_PLAN.md` (详细计划)
   - `scripts/week1/README.md` (快速参考)

2. **运行验证测试**
   ```bash
   PYTHONPATH=/home/user/R3/src:$PYTHONPATH \
       python scripts/week1/test_installation.py
   ```

3. **检查代码**
   - 所有代码都有详细注释和docstrings
   - 参考 `src/planner/q_learning_init.py`

---

**准备状态**: ✅ 就绪
**预计完成时间**: 2-3天（含分析）
**下一步**: 运行 `./scripts/week1/01_baseline_collection.sh`
