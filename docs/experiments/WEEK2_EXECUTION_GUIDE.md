# Week 2 实验执行指南

**日期**: 2025-11-12
**实验**: 自适应Epsilon策略对比
**预计时间**: 2-3天（包含运行+分析）

---

## 📋 实验概述

### 目标
测试3种epsilon（探索率）策略，判断提高探索率是否能改善大规模Q-learning性能。

### 策略对比
| 策略 | 初始Epsilon | Small | Medium | Large | 用途 |
|------|-------------|-------|--------|-------|------|
| **CURRENT** | 0.12 | 0.12 | 0.12 | 0.12 | Baseline |
| **SCALE_ADAPTIVE** | 规模自适应 | 0.30 | 0.50 | 0.70 | 主要假设 |
| **HIGH_UNIFORM** | 0.50 | 0.50 | 0.50 | 0.50 | 对照组 |

### 实验规模
- **总实验数**: 90（3策略 × 3规模 × 10种子）
- **预计时间**: 6-8小时（并行运行）
- **输出目录**: `results/week2/epsilon_experiments/`

---

## 🚀 Step 1: 准备环境

### 1.1 确认代码最新
```powershell
git status
git log -1  # 应该看到 "Complete Week 1 and prepare for Week 2"
```

### 1.2 创建结果目录
```powershell
mkdir results\week2\epsilon_experiments
```

### 1.3 测试单个实验
```powershell
# 测试CURRENT策略（应该与Week 1 ZERO结果接近）
python scripts\week2\run_experiment.py `
    --scenario small `
    --epsilon_strategy current `
    --seed 2025 `
    --output results\week2\test.json `
    --verbose

# 检查输出
type results\week2\test.json | findstr epsilon
# 应该看到 "epsilon_config": {"initial": 0.12, "decay": 0.995, "min": 0.01}
```

### 1.4 验证epsilon值
```powershell
python -c "import json; d=json.load(open('results/week2/test.json')); print(f'Initial: {d[\"epsilon_config\"][\"initial\"]}, Final: {d[\"final_epsilon\"]}')"
```

预期输出：`Initial: 0.12, Final: ~0.098`

---

## 🔄 Step 2: 并行运行实验

### 选项A：全并行（推荐，需要3个窗口）

**窗口1 - CURRENT策略（baseline）**:
```powershell
# Small
scripts\week2\01_current_small.bat

# Medium
scripts\week2\01_current_medium.bat

# Large
scripts\week2\01_current_large.bat
```

**窗口2 - SCALE_ADAPTIVE（主要测试）**:
```powershell
# Small
scripts\week2\02_scale_adaptive_small.bat

# Medium
scripts\week2\02_scale_adaptive_medium.bat

# Large
scripts\week2\02_scale_adaptive_large.bat
```

**窗口3 - HIGH_UNIFORM（对照）**:
```powershell
# Small
scripts\week2\03_high_uniform_small.bat

# Medium
scripts\week2\03_high_uniform_medium.bat

# Large
scripts\week2\03_high_uniform_large.bat
```

### 选项B：按规模并行（需要3个窗口）

每个窗口运行一个规模的所有3种策略：

**窗口1 - Small**:
```powershell
scripts\week2\01_current_small.bat
scripts\week2\02_scale_adaptive_small.bat
scripts\week2\03_high_uniform_small.bat
```

**窗口2 - Medium**:
```powershell
scripts\week2\01_current_medium.bat
scripts\week2\02_scale_adaptive_medium.bat
scripts\week2\03_high_uniform_medium.bat
```

**窗口3 - Large**:
```powershell
scripts\week2\01_current_large.bat
scripts\week2\02_scale_adaptive_large.bat
scripts\week2\03_high_uniform_large.bat
```

### 选项C：单独运行特定实验

```powershell
python scripts\week2\run_experiment.py `
    --scenario <SCALE> `
    --epsilon_strategy <STRATEGY> `
    --seed <SEED> `
    --output results\week2\epsilon_experiments\epsilon_<STRATEGY>_<SCALE>_seed<SEED>.json
```

**示例**:
```powershell
# 运行large规模，scale_adaptive策略，种子2030
python scripts\week2\run_experiment.py `
    --scenario large `
    --epsilon_strategy scale_adaptive `
    --seed 2030 `
    --output results\week2\epsilon_experiments\epsilon_scale_adaptive_large_seed2030.json
```

---

## 📊 Step 3: 监控进度

### 3.1 检查已完成实验数量
```powershell
dir results\week2\epsilon_experiments\*.json | measure
# 目标：90个文件
```

### 3.2 按策略分组统计
```powershell
# CURRENT
dir results\week2\epsilon_experiments\epsilon_current_*.json | measure

# SCALE_ADAPTIVE
dir results\week2\epsilon_experiments\epsilon_scale_adaptive_*.json | measure

# HIGH_UNIFORM
dir results\week2\epsilon_experiments\epsilon_high_uniform_*.json | measure
```

### 3.3 快速查看某个结果
```powershell
python -c "import json; d=json.load(open('results/week2/epsilon_experiments/epsilon_scale_adaptive_large_seed2025.json')); print(f'{d[\"scenario\"]} {d[\"epsilon_strategy\"]}: {d[\"improvement_ratio\"]*100:.2f}% (epsilon: {d[\"epsilon_config\"][\"initial\"]} -> {d[\"final_epsilon\"]:.3f})')"
```

---

## 📈 Step 4: 运行分析

### 4.1 确保所有实验完成
```powershell
dir results\week2\epsilon_experiments\*.json | measure
# 必须是90个文件！
```

### 4.2 运行统计分析
```powershell
python scripts\week2\analyze_epsilon.py
```

### 4.3 查看结果摘要
```powershell
type results\week2\analysis_summary.txt
```

### 4.4 预期输出格式
```
Week 2 Epsilon Strategy Analysis Summary
================================================================================

SMALL Scenario:
----------------------------------------
CURRENT (baseline): 37.70%
scale_adaptive: 38.50% (+0.80%), ns (p=0.150), d=+0.120
high_uniform: 37.20% (-0.50%), ns (p=0.250), d=-0.080

MEDIUM Scenario:
----------------------------------------
CURRENT (baseline): 31.46%
scale_adaptive: 35.20% (+3.74%), * (p=0.080), d=+0.350
high_uniform: 33.10% (+1.64%), ns (p=0.120), d=+0.180

LARGE Scenario:
----------------------------------------
CURRENT (baseline): 25.46%
scale_adaptive: 30.80% (+5.34%), ** (p=0.030), d=+0.520
high_uniform: 28.20% (+2.74%), * (p=0.090), d=+0.280
```

---

## ✅ Step 5: Checkpoint 1 决策

### 判断标准

查看Large规模的SCALE_ADAPTIVE结果：

**✅ 成功（采纳SCALE_ADAPTIVE）**:
- 改进率 ≥ 5%（例如：25% → 30%+）
- 统计显著性：p < 0.05
- 效应量：Cohen's d > 0.3

**⚠️ 部分成功**:
- 改进率：3-5%
- 统计显著性：p < 0.10
- → 可以考虑采纳，但epsilon可能不是主要因素

**❌ 失败（跳过epsilon，直接Week 5）**:
- 改进率 < 3%
- 统计显著性：p > 0.10
- → Epsilon不是瓶颈，重点转向Week 5（奖励归一化）

### 决策行动

**如果成功**:
1. 记录结果到`docs/experiments/WEEK2_RESULTS.md`
2. 更新计划文档：标记Week 2为✅
3. 开始Week 5设计（奖励归一化）
4. 后续实验使用SCALE_ADAPTIVE epsilon

**如果失败**:
1. 记录负面结果（同样有价值！）
2. 更新计划文档：标记Week 2为❌，但有信息
3. **直接跳到Week 5**（奖励归一化更可能是关键）
4. 后续实验继续使用CURRENT epsilon（0.12）

---

## 🐛 故障排除

### 问题1：实验运行很慢
**症状**: 单个实验超过10分钟
**原因**: LP求解器或segment优化耗时
**解决**:
```powershell
# 使用--disable_matheuristic_adaptation加速（降低质量）
python scripts\week2\run_experiment.py `
    --scenario medium `
    --epsilon_strategy scale_adaptive `
    --seed 2025 `
    --output results\week2\test_fast.json `
    --disable_matheuristic_adaptation
```

### 问题2：结果文件缺失
**症状**: 某些seed的结果没有生成
**排查**:
```powershell
# 列出所有结果文件
dir results\week2\epsilon_experiments\ | sort

# 找出缺失的seed
python -c "
import os
from pathlib import Path
expected = [(s, sc, sd) for s in ['current', 'scale_adaptive', 'high_uniform'] for sc in ['small', 'medium', 'large'] for sd in range(2025, 2035)]
existing = [f.stem for f in Path('results/week2/epsilon_experiments').glob('*.json')]
for strategy, scale, seed in expected:
    fname = f'epsilon_{strategy}_{scale}_seed{seed}'
    if fname not in existing:
        print(f'Missing: {fname}')"
```

**重跑缺失实验**:
```powershell
python scripts\week2\run_experiment.py `
    --scenario <SCALE> `
    --epsilon_strategy <STRATEGY> `
    --seed <SEED> `
    --output results\week2\epsilon_experiments\epsilon_<STRATEGY>_<SCALE>_seed<SEED>.json
```

### 问题3：CURRENT结果与Week 1 ZERO不一致
**预期**: Week 2 CURRENT应该与Week 1 ZERO baseline接近（相同epsilon，相同Q-init）

**检查**:
```powershell
# Week 1 ZERO baseline (small, seed 2025)
python -c "import json; d=json.load(open('results/week1/baseline/baseline_small_seed2025.json')); print(f'Week 1 ZERO: {d[\"improvement_ratio\"]*100:.2f}%')"

# Week 2 CURRENT (small, seed 2025)
python -c "import json; d=json.load(open('results/week2/epsilon_experiments/epsilon_current_small_seed2025.json')); print(f'Week 2 CURRENT: {d[\"improvement_ratio\"]*100:.2f}%')"
```

如果差异 > 2%，可能有随机性问题（可接受）。如果差异 > 5%，需要排查代码变更。

### 问题4：epsilon值不正确
**检查epsilon配置**:
```powershell
python -c "
import json
from pathlib import Path
for f in Path('results/week2/epsilon_experiments').glob('epsilon_scale_adaptive_large_*.json'):
    d = json.load(open(f))
    print(f'{f.name}: initial={d[\"epsilon_config\"][\"initial\"]}, final={d[\"final_epsilon\"]:.3f}')
"
# Large的SCALE_ADAPTIVE应该是initial=0.70
```

---

## 📝 实验日志模板

在运行实验时，建议记录以下信息：

```markdown
## Week 2 实验日志

**日期**: YYYY-MM-DD
**操作员**: [你的名字]

### 实验运行
- 开始时间: HH:MM
- 结束时间: HH:MM
- 运行方式: [选项A/B/C]
- 使用窗口数: [1/2/3]

### 完成情况
- CURRENT: [X]/30
- SCALE_ADAPTIVE: [X]/30
- HIGH_UNIFORM: [X]/30
- **总计**: [X]/90

### 初步观察
- Large规模SCALE_ADAPTIVE平均改进: ~X%
- 是否显著优于CURRENT: [是/否]
- 是否有异常值: [描述]

### Checkpoint 1决策
- [ ] ✅ 采纳SCALE_ADAPTIVE（≥5%改进，显著）
- [ ] ⚠️ 部分采纳（3-5%改进）
- [ ] ❌ 不采纳，直接Week 5

### 下一步
[记录决策后的行动计划]
```

---

## 📚 参考文档

- **实验设计**: `docs/experiments/WEEK2_TEST_PLAN.md`
- **Week 1结果**: `docs/experiments/WEEK1_RESULTS.md`
- **总计划**: `docs/SAQL_IMPLEMENTATION_PLAN_2025-11-09.md`（查看Option A调整）

---

## ⏱️ 预计时间分配

| 任务 | 预计时间 | 说明 |
|------|----------|------|
| 环境准备+测试 | 30分钟 | Step 1 |
| 实验运行（并行） | 6-8小时 | Step 2（可挂机） |
| 数据检查 | 15分钟 | Step 3 |
| 统计分析 | 10分钟 | Step 4 |
| 结果讨论+决策 | 30分钟 | Step 5 |
| **总计** | **~8-10小时** | 主要是计算时间 |

---

**祝实验顺利！有问题随时查阅本指南或Week 2测试计划文档。**
