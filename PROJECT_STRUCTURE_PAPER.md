# R3 Project Structure - Paper Writing Ready

**版本**: Phase 1 Baseline
**状态**: ✅ Ready for Paper Writing
**分支**: `claude/fix-qlearning-failures-20251103-011CUhJ2dCiVnBt3HEiNW3oY`

---

## 📁 核心目录结构

```
R3/
│
├── 📄 论文写作核心文档
│   ├── PAPER_WRITING_GUIDE.md      ⭐⭐⭐ 完整论文写作指南（844行）
│   ├── README_PAPER.md             ⭐⭐ 快速开始指南（277行）
│   ├── README.md                   项目主README
│   └── PROJECT_STRUCTURE.md        旧的项目结构（可选）
│
├── 📂 src/ - 核心源代码
│   ├── core/                       数据结构
│   │   ├── task.py                 任务模型（Pickup-Delivery）
│   │   ├── route.py                路径模型（含电池约束）
│   │   ├── vehicle.py              车辆模型
│   │   └── node.py                 节点模型（Depot/Task/Charging）
│   │
│   ├── planner/                    算法实现 ⭐
│   │   ├── alns.py                 Minimal ALNS (Phase 1 baseline)
│   │   ├── alns_matheuristic.py    Matheuristic ALNS
│   │   ├── q_learning.py           Q-learning agent ⭐⭐⭐
│   │   ├── operators.py            Destroy/Repair operators
│   │   ├── repair_lp.py            LP-based repair
│   │   ├── fleet.py                Multi-vehicle planner
│   │   └── adaptive_params.py.DISABLED  (Phase 1.5, 已禁用)
│   │
│   ├── physics/                    物理模型
│   │   ├── energy.py               电池和充电模型 ⭐
│   │   ├── distance.py             距离矩阵
│   │   └── time.py                 时间窗模型
│   │
│   ├── strategy/                   充电策略
│   │   └── charging_strategies.py PR-Minimal策略 ⭐
│   │
│   └── config/                     配置参数
│       ├── defaults.py             Q-learning参数（Phase 1）⭐
│       └── __init__.py
│
├── 📂 tests/ - 测试和实验
│   ├── optimization/               主要实验 ⭐⭐⭐
│   │   ├── presets.py              场景配置（Small/Medium/Large）
│   │   ├── common.py               实验工具函数
│   │   ├── q_learning/             Q-learning实验
│   │   │   └── utils.py            Q-learning实验工具
│   │   ├── test_alns_*.py          各种ALNS测试
│   │   └── README.md
│   │
│   ├── planner/                    单元测试
│   │   ├── test_alns.py
│   │   └── test_q_learning.py
│   │
│   └── conftest.py                 pytest配置
│
├── 📂 scripts/ - 运行脚本
│   └── generate_alns_visualization.py  ⭐ 主实验脚本
│
├── 📂 docs/ - 参考文档
│   ├── ARCHITECTURE.md             系统架构说明
│   ├── README.md                   技术文档
│   ├── 10seeds_analysis_and_publication_roadmap.md  ⭐ 实验分析
│   ├── data/                       实验数据
│   └── figures/                    实验图表
│
├── 📂 archive_debugging_docs/ - 归档（可忽略）
│   └── (10个调试分析文档)
│
└── 📂 experiments/ - 实验结果（待生成）
    └── seed_2025_2034/
        ├── seed_2025_small_minimal.json
        ├── seed_2025_small_matheuristic.json
        ├── seed_2025_small_q_learning.json
        └── ...（其他规模和seeds）
```

---

## 🎯 关键文件说明

### 论文写作必读

| 文件 | 重要性 | 用途 |
|:-----|:-------|:-----|
| **PAPER_WRITING_GUIDE.md** | ⭐⭐⭐ | 完整论文写作指南（数学模型、创新点、结构） |
| **README_PAPER.md** | ⭐⭐ | 快速开始（实验运行、结果摘要） |
| **docs/10seeds_analysis_and_publication_roadmap.md** | ⭐⭐ | 10-seed实验分析和发表路线图 |

### 核心算法实现

| 文件 | 代码行数 | 说明 |
|:-----|:---------|:-----|
| **src/planner/q_learning.py** | ~600行 | Q-learning agent（三状态系统） |
| **src/planner/alns.py** | ~2400行 | Minimal ALNS（Phase 1 baseline） |
| **src/planner/alns_matheuristic.py** | ~800行 | Matheuristic ALNS（LP + 段优化） |
| **src/strategy/charging_strategies.py** | ~330行 | PR-Minimal充电策略 |
| **src/physics/energy.py** | ~380行 | 电池和充电模型 |

### 实验配置

| 文件 | 说明 |
|:-----|:-----|
| **tests/optimization/presets.py** | 场景配置（15/24/30任务） |
| **src/config/defaults.py** | Q-learning参数（Phase 1: α=0.35, ε=0.01） |
| **scripts/generate_alns_visualization.py** | 主实验脚本 |

---

## 🔧 Phase 1 配置

### Q-learning参数 (src/config/defaults.py)
```python
alpha = 0.35              # 学习率
gamma = 0.95              # 折扣因子
epsilon_min = 0.01        # 最小探索率
stagnation_ratio = 0.16   # stuck触发阈值（16%）
deep_stagnation_ratio = 0.4  # deep_stuck触发阈值（40%）
```

### 实验规模 (tests/optimization/presets.py)
```python
Small:  15 tasks, 1 charging station, 40 iterations
Medium: 24 tasks, 1 charging station, 44 iterations
Large:  30 tasks, 3 charging stations, 44 iterations
```

### 充电策略 (src/strategy/charging_strategies.py)
```python
PR-Minimal: safety_margin=0.02 (2%)
只充刚好够用的电量 + 2%安全余量
```

---

## 🚀 运行实验

### 单个seed
```bash
python scripts/generate_alns_visualization.py --seed 2025
```

### 批量运行（10 seeds）
```bash
for seed in {2025..2034}; do
    python scripts/generate_alns_visualization.py --seed $seed
done
```

### 验证Phase 1还原
```bash
# 验证Seed 2034 Large是否恢复到30.35%
python scripts/generate_alns_visualization.py --seed 2034 --scale large --solver q_learning
```

---

## 📊 预期结果（Phase 1）

| 指标 | 值 |
|:-----|:---|
| 平均成本降低 | 36.34% |
| 胜率 | 60% (18/30) |
| t统计量 | -1.516 (不显著) |
| 标准差 | 18.5% |

**关键seeds**:
- ✅ Seed 2034 Large: 30.35% (Phase 1 best)
- ⚠️ Seed 2027 Medium: 17.01% (失败案例)
- ⚠️ Seed 2031 Large: 8.34% (失败案例)

---

## 📝 论文写作步骤

### Step 1: 阅读指南
```bash
cat PAPER_WRITING_GUIDE.md
```

### Step 2: 完成实验
运行10个seeds（如未完成）

### Step 3: 撰写论文
参考 `PAPER_WRITING_GUIDE.md` Section 5:
1. Introduction (3-4页)
2. Literature Review (4-5页)
3. Problem Formulation (3-4页)
4. Solution Methodology (6-7页)
5. Computational Experiments (5-6页)
6. Discussion (3-4页)
7. Conclusion (1-2页)

### Step 4: 投稿
推荐期刊（Q1-Q2）:
- Computers & Operations Research
- European Journal of Operational Research
- Transportation Research Part C

---

## 🗑️ 已清理内容

### 删除的文档 (移至 archive_debugging_docs/)
- ADAPTIVE_SOLUTION_IMPLEMENTATION.md
- ALGORITHM_OPTIMIZATION_PLAN.md
- COMPREHENSIVE_3SEEDS_ANALYSIS.md
- DEEP_DIAGNOSIS_TUNING_FAILURE.md
- PHASE1.5_TESTING_INSTRUCTIONS.md
- PHASE1_TEST_RESULTS_ANALYSIS.md
- SEED_*_ANALYSIS.md (多个)
- PARAMETER_TUNING_GUIDE.md
- PHASE1.5C_TESTING_GUIDE.md
- TESTING_GUIDE.md

### 删除的测试
- tests/warehouse_regression/ (7个文件)
- tests/charging/ (1个文件)

### 禁用的代码
- src/planner/adaptive_params.py → adaptive_params.py.DISABLED

---

## ✅ 验证清单

- [x] Phase 1参数已还原（alpha=0.35, epsilon_min=0.01）
- [x] 论文写作指南已创建（PAPER_WRITING_GUIDE.md）
- [x] 快速开始指南已创建（README_PAPER.md）
- [x] 调试文档已归档
- [x] 多余测试已删除
- [x] adaptive_params已禁用
- [x] 代码可正常导入
- [ ] 实验已完成（10 seeds × 3 scales × 3 solvers）
- [ ] 论文已开始撰写

---

## 📞 快速帮助

### 查看完整论文指南
```bash
cat /home/user/R3/PAPER_WRITING_GUIDE.md
```

### 查看快速开始
```bash
cat /home/user/R3/README_PAPER.md
```

### 查看实验配置
```bash
cat /home/user/R3/tests/optimization/presets.py
cat /home/user/R3/src/config/defaults.py
```

---

**状态**: ✅ 项目已准备好进行论文写作
**版本**: Phase 1 Baseline
**日期**: 2025-11-06
