# Week 1: Q-table初始化策略实验

本目录包含Week 1实验所需的所有脚本和工具。

## 快速开始

### 1. 基线收集（Day 1-3）

```bash
# 运行基线收集（30次运行，~30分钟）
chmod +x scripts/week1/01_baseline_collection.sh
./scripts/week1/01_baseline_collection.sh

# 分析基线数据
python scripts/week1/analyze_baseline.py
```

**输出**:
- `results/week1/baseline/baseline_*.json` (30个文件)
- `results/week1/baseline/baseline_summary.json`
- 控制台统计报告

### 2. 初始化实验（Day 4-7）

```bash
# 运行完整实验（120次运行，~2小时）
chmod +x scripts/week1/02_init_experiments.sh
./scripts/week1/02_init_experiments.sh

# 统计分析和可视化
python scripts/week1/analyze_init_strategies.py
```

**输出**:
- `results/week1/init_experiments/init_*.json` (120个文件)
- `results/week1/init_experiments/statistical_comparison.csv`
- `results/week1/init_experiments/init_strategies_comparison.png`
- `results/week1/init_experiments/recommendations.json`

## 文件说明

| 文件 | 用途 | 运行时间 |
|------|------|---------|
| `run_experiment.py` | 单次实验运行器 | ~1-2分钟/次 |
| `01_baseline_collection.sh` | 基线数据收集脚本 | ~30分钟 |
| `02_init_experiments.sh` | 完整初始化实验脚本 | ~2小时 |
| `analyze_baseline.py` | 基线数据分析 | <1分钟 |
| `analyze_init_strategies.py` | 初始化策略对比分析 | <1分钟 |

## 单独运行实验

如果需要单独运行某个配置：

```bash
python scripts/week1/run_experiment.py \
    --scenario small \
    --init_strategy uniform \
    --seed 2025 \
    --output results/test.json \
    --verbose
```

**参数说明**:
- `--scenario`: 场景规模 (small/medium/large)
- `--init_strategy`: 初始化策略 (zero/uniform/action_specific/state_specific)
- `--seed`: 随机种子 (整数)
- `--output`: 输出JSON文件路径
- `--verbose`: 打印详细进度

## 故障排查

### 实验失败

```bash
# 测试环境
python scripts/week1/run_experiment.py \
    --scenario small \
    --init_strategy zero \
    --seed 2025 \
    --output test.json \
    --verbose

# 检查输出文件
cat test.json | python -m json.tool
```

### 分析脚本报错

```bash
# 检查文件数量
ls results/week1/baseline/*.json | wc -l  # 应为30
ls results/week1/init_experiments/*.json | wc -l  # 应为120

# 检查JSON格式
python -c "
import json
from pathlib import Path
for f in Path('results/week1/baseline').glob('*.json'):
    with open(f) as fp:
        json.load(fp)
print('All JSON files valid')
"
```

## 预期结果

### 基线性能（Zero初始化）

| 规模 | 改进率 | 标准差 | CV |
|------|--------|--------|-----|
| Small | ~62% | ~9% | 0.15 |
| Medium | ~30% | ~8% | 0.25 |
| Large | ~7% | ~3% | 0.40 |

### 实验目标

- 大规模改进率：从 7% 提升到 ≥10%
- 大规模CV：从 0.40 降低到 ≤0.30
- 至少1种策略有统计显著改进 (p < 0.05)

## 下一步

Week 1完成后：
1. 查看 `results/week1/init_experiments/recommendations.json` 了解推荐策略
2. 将推荐策略用于Week 2-7的实验
3. 开始Week 2: Epsilon策略分析

## 参考文档

- 详细测试方案: `docs/experiments/WEEK1_TEST_PLAN.md`
- 周计划: `docs/WEEKLY_IMPLEMENTATION_PLAN.md`
- SAQL实施计划: `docs/SAQL_IMPLEMENTATION_PLAN_2025-11-09.md`
