# Electric Vehicle Routing with Q-learning - Paper Repository

**状态**: Phase 1 Baseline (最稳定版本)

---

## 📁 项目结构（论文写作相关）

```
R3/
├── PAPER_WRITING_GUIDE.md           ⭐ 论文写作完整指南
├── README_PAPER.md                  📖 本文件（快速开始）
│
├── src/                             💻 核心算法实现
│   ├── core/                        数据结构（Task, Route, Vehicle）
│   ├── planner/                     ALNS算法
│   │   ├── alns.py                  Minimal ALNS (baseline)
│   │   ├── alns_matheuristic.py     Matheuristic ALNS
│   │   ├── q_learning.py            Q-learning agent ⭐
│   │   └── adaptive_params.py       (Phase 1.5, 已禁用)
│   ├── physics/                     物理模型（energy, distance）
│   │   └── energy.py                电池和充电模型
│   ├── strategy/                    充电策略
│   │   └── charging_strategies.py  PR-Minimal ⭐
│   └── config/                      配置参数
│       └── defaults.py              Q-learning参数 (Phase 1)
│
├── tests/                           🧪 实验和测试
│   ├── optimization/                主要实验（10-seed测试）
│   │   ├── presets.py               场景配置（Small/Medium/Large）
│   │   ├── common.py                实验工具函数
│   │   └── q_learning/              Q-learning实验
│   └── planner/                     单元测试
│
├── scripts/                         🚀 运行脚本
│   └── generate_alns_visualization.py  主实验脚本（生成结果）
│
├── docs/                            📚 参考文档
│   ├── ARCHITECTURE.md              系统架构说明
│   ├── README.md                    技术文档
│   ├── 10seeds_analysis_and_publication_roadmap.md  实验分析
│   └── data/                        实验数据
│
└── experiments/                     📊 实验结果（如有）
    └── seed_2025_2034/              10个种子的完整结果
```

---

## 🎯 论文核心信息

### 问题定义
**Multi-Vehicle Electric Vehicle Routing Problem with Partial Recharging and Time Windows (mE-VRP-PR-TW)**

### 创新点
1. ✨ **Q-learning驱动的ALNS算子选择**（三状态系统：explore/stuck/deep_stuck）
2. ✨ **Matheuristic框架**（ALNS + LP repair + 段优化）
3. ✨ **No Free Lunch现象的实证研究**（10 seeds完整数据）

### 充电策略
**Partial Recharge Minimal (PR-Minimal)** - Keskin & Çatay (2016)
- 只充刚好够用的电量 + 2%安全余量
- 节省充电时间

### Phase 1 参数（当前版本）
```python
# src/config/defaults.py
alpha = 0.35              # 学习率
epsilon_min = 0.01        # 最小探索率
stagnation_ratio = 0.16   # stuck触发阈值
```

---

## 🚀 快速开始

### 运行10-seed实验

```bash
# 运行单个seed的完整实验（3种规模 × 3种求解器）
python scripts/generate_alns_visualization.py --seed 2025

# 批量运行所有seeds（2025-2034）
for seed in {2025..2034}; do
    python scripts/generate_alns_visualization.py --seed $seed
done
```

### 实验结果位置
```
experiments/seed_2025_2034/
├── seed_2025_small_minimal.json
├── seed_2025_small_matheuristic.json
├── seed_2025_small_q_learning.json
├── ...（其他规模和seeds）
```

---

## 📊 Phase 1 实验结果摘要

| 指标 | Q-learning | Matheuristic | 差异 |
|:-----|:-----------|:-------------|:-----|
| **平均成本降低** | 36.34% | 38.50% | -2.16% |
| **胜率** | 60% (18/30) | 40% (12/30) | +20% |
| **t统计量** | - | - | -1.516 (NS) |
| **标准差** | 18.5% | 16.2% | +2.3% |

**结论**:
- ✅ Q-learning具有竞争力（胜率60%）
- ⚠️ 统计不显著（t=1.516 < 2.045, p>0.05）
- ⚠️ 高方差（NFL现象）

**关键失败案例**:
- Seed 2034 Large: 4.45% (vs 30.35% in Phase 1)
- Seed 2027 Medium: 17.01% (vs Matheuristic 48.52%)

---

## 📝 论文写作流程

### Step 1: 阅读写作指南
```bash
cat PAPER_WRITING_GUIDE.md
```

**重点章节**:
- 第2节: 创新点总结
- 第3节: 算法框架
- 第5节: 论文结构建议
- 第6节: 写作策略

### Step 2: 完成实验（如未完成）
```bash
# 确保10个seeds都已运行
python scripts/generate_alns_visualization.py --seed 2025
# ... (seeds 2026-2034)
```

### Step 3: 统计分析
```bash
# 运行统计分析脚本（如有）
python scripts/analyze_10seeds_results.py
```

### Step 4: 撰写论文
参考 `PAPER_WRITING_GUIDE.md` 第5节的结构：

1. **Introduction** (3-4页)
2. **Literature Review** (4-5页)
3. **Problem Formulation** (3-4页) ⭐ 数学模型
4. **Solution Methodology** (6-7页) ⭐ Q-learning + Matheuristic
5. **Computational Experiments** (5-6页) ⭐ 实验结果
6. **Discussion** (3-4页) ⭐ NFL现象
7. **Conclusion** (1-2页)

### Step 5: 投稿建议
**推荐期刊**（Q1-Q2）:
1. ✅ Computers & Operations Research (IF ~4.5)
2. ✅ European Journal of Operational Research (IF ~6.0)
3. ✅ Transportation Research Part C (IF ~8.3)
4. ✅ Expert Systems with Applications (IF ~8.5)

---

## 🔬 关键技术细节

### 数学模型
详见 `PAPER_WRITING_GUIDE.md` 第1.3节

**目标函数**:
$$
\min Z = \sum_{v \in V} \left( C_{tr} \cdot D_v + C_{ch} \cdot Q_v + C_{time} \cdot T_v + C_{delay} \cdot \Delta_v + C_{wait} \cdot W_v \right)
$$

**关键约束**:
1. 任务分配约束
2. Pickup-Delivery优先级
3. 载重约束
4. 时间窗约束（软）
5. 电池约束（含安全阈值5%）
6. **局部充电约束** (Partial Recharging)

### Q-learning设计
详见 `PAPER_WRITING_GUIDE.md` 第3.2节

**三状态系统**:
```python
State = {
    "explore":      # 正常搜索
    "stuck",        # 停滞（触发LP repair）
    "deep_stuck"    # 深度停滞
}
```

**奖励函数**:
- 基础奖励：new_best(+100), improvement(+36), accepted(+10), rejected(-6)
- ROI奖励：基于成本改进比例
- 时间惩罚：避免过慢算子

---

## 📚 重要文献

1. **Keskin & Çatay (2016)**: Partial recharge策略
2. **Ropke & Pisinger (2006)**: ALNS原始论文
3. **Wolpert & Macready (1997)**: No Free Lunch定理
4. **Singh et al.**: LP-based repair

完整文献列表见 `PAPER_WRITING_GUIDE.md` 第7节。

---

## ⚠️ 已知问题和局限

1. **统计不显著**: t=1.516 < 2.045 (p>0.05)
2. **高方差**: 某些seeds表现极差（NFL现象）
3. **规模限制**: 最大30任务（适合单车规划）
4. **参数敏感性**: Phase 1.5/1.5c调参失败

**如何在论文中处理**:
- ✅ 诚实报告负面结果
- ✅ 强调NFL现象的学术价值
- ✅ 提供详细的per-seed分析
- ✅ 讨论局限性和未来工作

详见 `PAPER_WRITING_GUIDE.md` 第6.1节。

---

## 🛠️ 故障排查

### 问题1: 导入错误
```bash
ModuleNotFoundError: No module named 'core'
```
**解决**: 在项目根目录运行，或添加：
```python
import sys
sys.path.insert(0, '/home/user/R3/src')
```

### 问题2: 实验结果不一致
**原因**: 随机种子未固定
**解决**: 确保使用相同的seed参数

### 问题3: 内存不足
**原因**: Large规模 + 多次迭代
**解决**: 减少迭代次数或使用批处理

---

## 📞 联系信息

- **代码仓库**: `/home/user/R3/`
- **论文指南**: `PAPER_WRITING_GUIDE.md`
- **实验配置**: `tests/optimization/presets.py`
- **参数设置**: `src/config/defaults.py`

---

## ✅ 检查清单（论文提交前）

- [ ] 完成10-seed实验（seeds 2025-2034）
- [ ] 统计分析（t-test, p-value）
- [ ] 所有图表完成（至少6个图+7个表）
- [ ] 数学符号一致性检查
- [ ] 英文语法检查（Grammarly）
- [ ] 代码开源并获得DOI（Zenodo）
- [ ] 避免过度宣称（"first", "best"）
- [ ] 包含limitations部分
- [ ] 参考文献格式正确
- [ ] 实验可复现（提供完整参数）

---

**祝论文写作顺利！** 🎓📄✨
