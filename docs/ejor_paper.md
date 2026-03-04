# EJOR Paper: Results Section 重构计划

## 论文定位
- 目标期刊: European Journal of Operational Research (EJOR)
- 方法: RL-APC (Reinforcement Learning — Adaptive Policy Composition)
- 场景: 仓储 AGV 在线任务调度，15 条调度规则的自适应选择

## 核心问题：原稿弱点
- Greedy-FR = Greedy-PR，实际只有 **1 个有效在线对比算法**
- Random 仅为 sanity check，不构成有意义对比
- 审稿人会判定为"稻草人对比 (straw-man comparison)"
- 缺少：规则行为分析、训练收敛证据、灵敏度分析、统计严谨性

---

## 六步递进论证架构

### Q1: 单条规则够用吗？→ 不够
- **证据**: T3 (15规则×4scale 全表) + 柱状图 (F1)
- **论点**: 没有任何单规则在所有 scale 都最优，规则选择依赖于问题规模
- **脚本**: `scripts/evaluate_individual_rules.py`
- **输出**: `results/benchmark/individual_rules_{S,M,L,XL}_30.csv`
- **数据就绪**: ❌ 全缺，需运行评估（~60分钟）
- **对应表/图**: T3, F1

### Q2: 自适应选择有效吗？→ 有效
- **证据**: T4 (RL-APC vs 每个 scale 的最优单规则) + Wilcoxon 检验
- **论点**: RL-APC 在多数 scale 优于任何固定规则，对比从 1 个扩展到 15+ 个
- **脚本**: `scripts/generate_ejor_tables.py` (T4)
- **数据就绪**: ❌ 依赖 Q1 的 individual_rules CSV
- **对应表/图**: T4, T5 (部分)

### Q3: RL 学到了什么？→ 可解释策略
- **证据**: 规则选择热力图 (15×4) + per-event-type 频率
- **论点**: RL 在不同 scale/事件类型选择不同规则，策略可解释非黑盒
- **脚本**: `scripts/analyze_rule_selection.py`
- **输出**: `results/paper/fig_rule_selection_heatmap.png`, `rule_selection_heatmap.csv`
- **数据就绪**: ✅ 4/4 scale 均有 decision log (S, M, L_v3, XL_v2)
- **对应表/图**: F3

### Q4: 成本之外还有什么优势？→ 服务质量
- **证据**: T6 (完成率、拒绝率) + T8 (拒绝惩罚成本分解)
- **论点**: 即使总成本相近 (M, XL)，RL-APC 的服务质量远优于 Greedy
- **数据就绪**: ✅ 4/4 scale eval CSV 就绪
- **对应表/图**: T6, T8
- **关键数据**:
  - S: RL 完成13.0 / 0拒绝 vs Greedy 17.0 / 0拒绝
  - M: RL 完成 18.1 / 0 拒绝 vs Greedy 11.5 / 21.8 拒绝 → **RL 服务质量远优**
  - L: RL 完成 17.5 / 0 拒绝 vs Greedy 30.4 / 16.8 拒绝 → RL 完成更少但不拒绝
  - XL: RL 完成 22.2 / 1.5 拒绝 vs Greedy 20.0 / 57.7 拒绝 → **RL 服务质量远优**

### Q5: 与离线最优差多远？→ 量化在线代价
- **证据**: T7 (online-offline gap%)
- **论点**: gap 反映的是"没有未来信息"的固有代价，不是算法缺陷
- **数据就绪**: ✅ S/M/XL 有 ALNS 数据; ⚠️ L_v3 缺 ALNS（可用 L_v1 补）
- **对应表/图**: T7

### Q6: 方法是否鲁棒？→ 统计严谨 + 灵敏度
- **证据**: T5 (Bonferroni 校正多重比较) + T10 (L-scale v1→v3 灵敏度)
- **论点**: 结果统计显著、超参数可调、性能可改善
- **数据就绪**: ✅ L v1/v2/v3 三版 eval CSV 均存在
- **对应表/图**: T5, T10

---

## 完整表格/图表清单

### 表格 (Tables)

| 编号 | 标题 | 状态 | 数据来源 |
|------|------|------|---------|
| T1 | Problem Instance Characteristics | ✅ 已有 | 静态 |
| T2 | RL Training Configuration | ✅ 新增 | MEMORY.md + 训练脚本 |
| T3 | Individual Rule Performance (15×4) | ⏳ 待数据 | `individual_rules_*.csv` |
| T4 | RL-APC vs Best Fixed Rule | ⏳ 待数据 | T3 + 现有 eval CSV |
| T5 | Wilcoxon Tests (Bonferroni) | ✅ 增强 | 现有 eval CSV + T3 |
| T6 | Service Quality (含完成率) | ✅ 增强 | 现有 eval CSV |
| T7 | Online-Offline Gap | ✅ 增强 | 现有 eval CSV |
| T8 | Cost Decomposition (含拒绝惩罚) | ✅ 增强 | 现有 eval CSV |
| T9 | Computational Efficiency | ✅ 已有 | 现有 eval CSV |
| T10 | L-scale Sensitivity | ✅ 自动检测 | L v1/v2/v3 eval CSV |

### 图表 (Figures)

| 编号 | 标题 | 状态 | 脚本 |
|------|------|------|------|
| F1 | Rule Performance Bar Chart | ⏳ 待数据 | `generate_paper_figures.py` |
| F2 | Cost Distribution Boxplots | ⏳ 待数据 | `generate_paper_figures.py` |
| F3 | Rule Selection Heatmap | ⏳ 待 decision log | `analyze_rule_selection.py` |
| F4 | Training Curves (enhanced) | ✅ 可生成 | `generate_paper_results_synced.py` |

---

## 脚本依赖关系

```
evaluate_individual_rules.py  ──→ individual_rules_*.csv
        │                               │
        ▼                               ▼
generate_paper_figures.py      generate_ejor_tables.py (T3, T4, T5)
  (F1 bar chart, F2 boxplots)    (所有 10 张表)

analyze_rule_selection.py  ──→ heatmap CSV + F3 图
  (需要 decision log 数据)

generate_paper_results_synced.py ──→ F4 训练曲线
```

---

## L-Scale 弱点应对策略

L-scale 是论文最大短板，四层防线：

1. **T3 提供背景**: 展示 L-scale 哪些单规则也差，RL 不是唯一失败者
2. **T8 解释根因**: RL 拒绝 0 任务但仅完成 17.5/56 → 过度保守 → standby 爆炸
3. **T10 展示改进**: v1→v2→v3 的性能变化，证明可通过调参改善
4. **诚实讨论**: 定位为中等规模组合优化的 exploration-exploitation 挑战

### L-scale 版本对比 (Mar 3 更新)

| 版本 | RL Cost | Greedy Cost | Δ% | 说明 |
|------|---------|-------------|-----|------|
| v1 (best_model, vecnorm mismatch) | 972,882 | 377,493 | +157.7% | vecnorm 不匹配 |
| v2 ([512,256], 500K步) | 471,513 | ~377K | ~+25% | 网络加大 |
| **v3 best_model (50K步)** | **662,267** | **336,386** | **+96.9%** | 30-instance 实测 |

注意: v3 训练 eval 显示 50K步 cost=372K（仅5个episode），但 30-instance 实测为 662K（方差大、泛化弱）。

当前 L_v3 训练状态：650K/2M 步 (32%)，预计 ~22:00 完成

---

## 执行顺序

```bash
# 1. 跑 15 规则独立评估（最高优先级，~60分钟）
python scripts/evaluate_individual_rules.py

# 2. 生成表格（T3/T4 需要步骤1的输出）
python scripts/generate_ejor_tables.py

# 3. 生成图表
python scripts/generate_paper_figures.py

# 4. 训练曲线（可随时运行）
python scripts/generate_paper_results_synced.py

# 5. 规则选择分析（需要带 log 的 RL 评估）
python scripts/analyze_rule_selection.py

# 6. L_v3 完成后：重新评估 + 更新所有表格
python scripts/evaluate_all.py --scale L --algorithms rl_apc,greedy_fr ...
python scripts/generate_ejor_tables.py
```

---

## 关键文件位置

| 文件 | 用途 |
|------|------|
| `scripts/evaluate_individual_rules.py` | 15 规则独立评估 |
| `scripts/analyze_rule_selection.py` | 规则选择行为分析 |
| `scripts/generate_paper_figures.py` | 论文图表 (F1-F3) |
| `scripts/generate_ejor_tables.py` | 论文表格 (T1-T10) |
| `scripts/generate_paper_results_synced.py` | 训练曲线 (F4) |
| `results/paper/ejor_tables.tex` | LaTeX 表格合集 |
| `results/paper/` | 所有论文资产输出目录 |
| `results/benchmark/individual_rules_*.csv` | 单规则评估结果 |
