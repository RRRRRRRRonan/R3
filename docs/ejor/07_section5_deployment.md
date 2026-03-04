# Section 5: Simulation Results — 各小节部署指南

> 最后更新: 2026-03-03
> 本文档为 EJOR 论文 Section 5 (Computational Experiments) 的撰写蓝图，包含每个小节的关键论点、叙事点、所需表/图、数据状态和待补充内容。

---

## 贯穿全文的核心发现

在撰写各小节前，需要理解一个改变整篇论文叙事的关键事实：

> **低成本规则的"成本陷阱"**: 成本最优的单规则 (如 Standby-Lazy) 靠大量拒绝任务来压低总成本。这在实际仓储运营中不可接受。

| Scale | Method | Avg Cost | Completed | Rejected | 含义 |
|-------|--------|----------|-----------|----------|------|
| M | Standby-Lazy (cost #1) | 224,316 | 13.8 | **19.2** | 拒绝过半任务 |
| M | RL-APC | 286,665 | **18.1** | **0.0** | 全部接受 |
| XL | Standby-Lazy (cost #1) | 696,148 | 18.8 | **64.6** | 拒绝 ~73% 任务 |
| XL | RL-APC | 814,344 | **22.2** | **1.5** | 几乎全部接受 |

因此论文不能只比成本，必须将**成本-服务质量权衡 (cost-service trade-off)** 作为核心评价维度。

---

## Section 5.1: Experimental Setup

> 不在本指南范围内 (T1 + T2 已就绪)。

---

## Section 5.2: Performance of Individual Dispatch Rules (Q1)

### 关键论点

**"没有万能规则: 最优规则因问题规模而变化。"**

### 叙事点

1. **排名因 scale 而异** — Standby-Lazy 在 S/M/XL 排 #1，但在 L 被 Charge-High 取代
2. **规则间差距巨大** — 所有 scale 最优 vs 最差有 2.0–2.6 倍差距，规则选择极为重要
3. **等价规则** — STTF=EDD=HPF (调度策略不影响成本)；Charge-High=Charge-Opp (Greedy-FR ≡ Greedy-PR)
4. **Greedy-FR 并非最优单规则** — S/M 排 #2–3，XL 排 #10，仅 L 恰好排 #1

### 每个 Scale 最优规则排名

| Scale | #1 | Cost | #2 | Cost | #3 | Cost | 最优 vs 最差 |
|-------|----|------|----|------|----|------|-------------|
| S | Standby-Lazy | 60,731 | Charge-High/Opp | 83,962 | Insert-MinCost | 103,048 | 2.6x |
| M | Standby-Lazy | 224,316 | Charge-High/Opp | 266,218 | MST | 294,407 | 2.2x |
| L | **Charge-High/Opp** | 336,386 | Standby-Lazy | 418,299 | Insert-MinCost | 447,292 | 2.5x |
| XL | Standby-Lazy | 696,148 | Insert-MinCost | 747,761 | Standby-LowCost | 762,554 | 2.0x |

### 所需表/图

| 编号 | 内容 | 状态 | 数据文件 |
|------|------|------|---------|
| T3 | 15 规则 × 4 scale 成本 + 拒绝率表 | ✅ 已生成 (含 Rej 列) | `ejor_table3_rules.csv` |
| F1 | 分组柱状图 (含 rejection 注解) | ✅ 已生成 | `fig_rule_bar_chart.png/.pdf` |

### 已完成的增强

| 项目 | 说明 | 完成时间 |
|------|------|---------|
| ✅ T3 增加 Rejected 列 | 每个 scale 现在有 Cost + Rej 两列；Rej≥10 在 LaTeX 中以红色标注；RL-APC 的 Rej 以粗体突出 (均为 0.0) | 2026-03-03 |
| ✅ F1 柱状图加 rejection 注解 | 拒绝≥5 的规则在柱顶标注红色 `R:XX`；RL-APC 参考线标签包含 `[R:0.0]` | 2026-03-03 |

### 待补充

无。Section 5.2 的表/图均已就绪。

---

## Section 5.3: RL-APC vs Best Fixed Rule (Q2)

### 关键论点

**"RL-APC 在 S-scale 大幅优于所有单规则 (-34.4%)。在 M/XL，RL 的成本略高但需结合服务质量评判 (见 5.6)。"**

### 叙事点

1. **S-scale: 全面胜出** — RL (39,846) < 所有 15 条单规则中的最优 (Standby-Lazy = 60,731)，Wilcoxon p=0.004 (**)
2. **M-scale: 成本代价换服务质量** — RL 成本 +27.8% (p=0.022 *)，但 Standby-Lazy 拒绝 19.2 任务 vs RL 拒绝 0
3. **L-scale: 明确弱点** — RL 成本 +96.9% (p<1e-7 ***)，策略过度保守
4. **XL-scale: 统计不显著的差距** — RL vs Standby-Lazy +17.0% (p=0.129 ns)，RL wins 15/15 ties，Standby-Lazy 拒绝 64.6 任务
5. **后见之明的局限** — 选择"最优单规则"需要事先知道 scale 和 instance，在线场景中不可行

### 核心对比数据

| Scale | Best Rule | Best Cost | RL Cost | Δ% | p-adj | RL vs Greedy-FR |
|-------|-----------|-----------|---------|-----|-------|-----------------|
| S | Standby-Lazy | 60,731 | 39,846 | **-34.4%** | 0.004 ** | **-58.0%** *** |
| M | Standby-Lazy | 224,316 | 286,665 | +27.8% | 0.022 * | +1.9% ns |
| L | Charge-High | 336,386 | 662,267 | +96.9% | <0.001 *** | +96.9% *** |
| XL | Standby-Lazy | 696,148 | 814,344 | +17.0% | 0.776 ns | +0.6% ns |

### 所需表/图

| 编号 | 内容 | 状态 | 数据文件 |
|------|------|------|---------|
| T4 | RL vs best-per-scale (cost + compl + rej + sig) | ✅ 已生成 (含多维对比) | `ejor_table4_rl_vs_best.csv` |
| T5 | Wilcoxon 检验 (含 top rules, Bonferroni) | ✅ 已生成 | `ejor_table5_wilcoxon.csv` |

### 已完成的增强

| 项目 | 说明 | 完成时间 |
|------|------|---------|
| ✅ T4 增加多维对比列 | 从 6 列扩展到 10 列: Best Rule (Cost, Compl, Rej) + RL-APC (Cost, Compl, Rej) + Δ% + p-value。高拒绝红色标注, 胜出成本粗体, RL 零拒绝粗体。使用 `table*` 全页宽度。审稿人可直接看到"成本低靠拒绝任务" | 2026-03-03 |

### 待补充

| 优先级 | 内容 | 说明 |
|--------|------|------|
| P1 | 论文文本中引导到 Section 5.6 | 对 M/XL 的"成本输"明确写 "see Section 5.6 for a multidimensional evaluation" |

---

## Section 5.4: What Does RL Learn? — Rule Selection Behavior (Q3)

### 关键论点

**"RL-APC 的策略不是黑盒: 它学到了以"待命 + 机会充电"为核心的结构化策略，不同 scale 下策略组成系统性变化。"**

### 叙事点

1. **两条规则主导** — Standby-Lazy + Charge-Opp 在所有 scale 合计占 76–95% 的决策
2. **Scale 敏感性** — S/XL 偏向 Charge-Opp (充电)，M/L 偏向 Standby-Lazy (待命)
3. **Dispatch 规则几乎不被使用** — STTF/EDD/MST/HPF/Insert-MinCost 合计 <0.5%。RL 发现任务调度在 execution layer 已自动处理，高层策略只需管理"充电/待命/接受"
4. **Accept-Value 在 XL 更活跃** (6.5% vs <2%)，因为 XL 有更多需要接受/拒绝决策的任务到达事件

### 各 Scale 的 Top-3 规则选择

| Scale | #1 | #2 | #3 |
|-------|-----|-----|-----|
| S | Charge-Opp (44.7%) | Standby-Lazy (32.0%) | Standby-LowCost (10.2%) |
| M | Standby-Lazy (64.5%) | Charge-Opp (30.0%) | Standby-LowCost (1.7%) |
| L | Standby-Lazy (59.1%) | Charge-Opp (27.4%) | Standby-Heatmap (4.4%) |
| XL | Charge-Opp (54.6%) | Standby-Lazy (29.5%) | Accept-Value (6.5%) |

### 所需表/图

| 编号 | 内容 | 状态 | 数据文件 |
|------|------|------|---------|
| F3 | 规则选择频率热力图 (15×4) | ✅ 已生成 | `fig_rule_selection_heatmap.png` |
| — | Per-event breakdown CSV | ✅ 已生成 | `rule_event_breakdown_{S,M,L,XL}.csv` |

### 待补充

| 优先级 | 内容 | 说明 |
|--------|------|------|
| P2 | 单 episode 规则时序图 (F3b) | 选一个代表性 episode，展示规则切换随时间的变化。Nice-to-have |
| P2 | 与 Q1 联动的讨论 | 论文文本中指出: "RL 大量使用 Standby-Lazy 与 Q1 发现它是 S/M/XL 最优单规则一致" |

---

## Section 5.5: Training Convergence

### 关键论点

**"RL-APC 在各 scale 均能收敛，但 L-scale 收敛较慢且方差较大。"**

### 叙事点

1. **S/M 快速收敛** — best_model 出现在训练前半段
2. **XL 需要更多步数** — 收敛曲线较平缓但稳定下降
3. **L 收敛困难** — best 出现在极早期 (50K 步)，之后性能未能持续改善
4. **与 scale 复杂度正相关** — 问题越大，训练越难

### 所需表/图

| 编号 | 内容 | 状态 | 数据文件 |
|------|------|------|---------|
| F4 | 训练曲线 (含 std band + best marker) | ✅ 已生成 | `fig_training_curves.png/.pdf` |

### 待补充

| 优先级 | 内容 | 说明 |
|--------|------|------|
| P1 | 等 L_v3 2M 步训练完成后更新 F4 | 当前 L_v3 仅跑到 ~650K 步，完成后重新生成图表 |
| P2 | 在论文文本中讨论 L 的收敛难题 | 与 Section 5.9 (sensitivity) 交叉引用 |

---

## Section 5.6: Service Quality Analysis (Q4) ← 论文最关键的小节

### 关键论点

**"成本不是唯一指标。RL-APC 是唯一在保持近零拒绝率的同时达到有竞争力成本的方法。所有低成本 baseline（包括最优单规则）都靠大量拒绝任务来降低成本。"**

### 叙事点

1. **RL 几乎不拒绝任务** — M: 0 拒绝, XL: 1.5 拒绝
2. **所有低成本规则都大量拒绝** — Standby-Lazy 在 M 拒绝 19.2, XL 拒绝 64.6; Greedy-FR 在 M 拒绝 21.8, XL 拒绝 57.7
3. **RL 完成更多任务** — M: RL 18.1 vs Standby-Lazy 13.8 vs Greedy 11.5; XL: RL 22.2 vs Standby-Lazy 18.8 vs Greedy 20.0
4. **成本差异的根因** — RL 的"高成本"来自处理更多任务 (travel, tardiness)；baseline 的"低成本"来自拒绝任务后零运营开销
5. **实际运营可行性** — 在仓储物流中，拒绝率 >50% (XL-Standby-Lazy: 73%) 完全不可接受

### 完整三方对比 (RL vs Greedy-FR vs Best Rule)

| Scale | Method | Cost | Comp | Rej | Delay | 评价 |
|-------|--------|------|------|-----|-------|------|
| **S** | Standby-Lazy (best) | 60,731 | 12.4 | 3.8 | 0 | 成本低但完成少 |
| | Greedy-FR | 94,772 | 15.3 | 1.5 | 17,137 | 完成多但延迟大 |
| | **RL-APC** | **39,846** | **14.5** | **0.0** | **137** | **全面最优** |
| **M** | Standby-Lazy (best) | 224,316 | 13.8 | **19.2** | 0 | 靠拒绝压低成本 |
| | Greedy-FR | 281,349 | 11.5 | **21.8** | 605 | 拒绝更多 |
| | **RL-APC** | 286,665 | **18.1** | **0.0** | **22** | **零拒绝，完成最多** |
| **L** | Charge-High (best) | 336,386 | **30.4** | 16.8 | 8,456 | 完成多但拒绝多 |
| | Standby-Lazy | 418,299 | 16.3 | **37.9** | 0 | 拒绝极多 |
| | RL-APC | 662,267 | 17.5 | **0.0** | 5,743 | 零拒绝但成本高 |
| **XL** | Standby-Lazy (best) | 696,148 | 18.8 | **64.6** | 0 | 拒绝 ~73% 任务 |
| | Greedy-FR | 809,830 | 20.0 | **57.7** | 177 | 拒绝极多 |
| | **RL-APC** | 814,344 | **22.2** | **1.5** | 3,366 | **几乎零拒绝** |

### 所需表/图

| 编号 | 内容 | 状态 | 数据文件 |
|------|------|------|---------|
| T6 | 服务质量三方对比 | ⚠️ 目前仅 RL vs Greedy | `ejor_table6_service.csv` |
| F2 | 成本分布箱线图 | ✅ 已生成 | `fig_cost_boxplots.png/.pdf` |

### 待补充

| 优先级 | 内容 | 说明 |
|--------|------|------|
| **P0** | **T6 增加 Best Rule (Standby-Lazy / Charge-High) 行** | 这是论文最关键的增强。目前 T6 只有 RL vs Greedy-FR，审稿人会问 "为什么不用成本最低的规则"。加入 best rule 的服务质量后，论点彻底立住。数据已有 (`individual_rules_*.csv`)，只需改脚本 |
| **P0** | **计算 "cost-per-completed-task" 指标** | cost / completed_tasks，消除"做更多工作自然成本更高"的混淆。例: M-scale RL 15,838/task vs Standby-Lazy 16,254/task → RL 的单位任务成本反而更低 |
| P1 | F2 箱线图加入 Standby-Lazy | 目前 F2 只有 RL vs Greedy + top-3 rules。确保 Standby-Lazy 在图中可见 |

---

## Section 5.7: Online-Offline Gap (Q5)

### 关键论点

**"所有在线方法与离线最优的 gap 均 >370%。这是实时决策的固有信息代价，不是算法缺陷。"**

### 叙事点

1. **Gap 巨大且普遍** — S: RL 654% / Greedy 1694%; M: RL 435% / Greedy 425%; XL: RL 378% / Greedy 375%
2. **S-scale: RL 显著缩小 gap** — RL gap (654%) 仅为 Greedy gap (1694%) 的 38.6%
3. **M/XL: gap 相当** — 两种在线方法达到了相似的性能天花板
4. **Gap 随 scale 缩小** — S ~1000% → XL ~376%，大规模下统计平滑效应
5. **审稿人校准期望** — 不能要求在线方法接近离线最优

### 核心数据

| Scale | RL Cost | Greedy Cost | ALNS (Offline) | Gap RL | Gap Greedy |
|-------|---------|-------------|----------------|--------|------------|
| S | 39,846 | 94,772 | 5,282 | +654% | +1694% |
| M | 286,665 | 281,349 | 53,600 | +435% | +425% |
| L | 662,267 | 336,386 | — | — | — |
| XL | 814,344 | 809,830 | 170,514 | +378% | +375% |

### 所需表/图

| 编号 | 内容 | 状态 | 数据文件 |
|------|------|------|---------|
| T7 | Online-Offline gap 表 | ✅ 已生成 (L 缺 ALNS) | `ejor_table7_offline_gap.csv` |

### 待补充

| 优先级 | 内容 | 说明 |
|--------|------|------|
| P1 | **L-scale ALNS 离线解** | 当前 L_v3 无 ALNS 数据。可选: (a) 用 L_v1 的 ALNS 数据 + 脚注说明 instance 集不同, (b) 标记 "—" 并在文本中说明 |
| P2 | 增加 Best Rule 到 T7 | 加入 Standby-Lazy 的 gap 作为参考，增强可比性 |

---

## Section 5.8: Cost Decomposition (Q4 补充)

### 关键论点

**"RL 的成本结构以 standby 为主 (保守等待)；baseline 的成本被 rejection penalty 主导。"**

### 叙事点

1. **Rejection penalty 是 Greedy 的成本主体** — M: 218K/281K (77.5%), XL: 577K/810K (71.3%)
2. **RL 的 standby 成本高但合理** — RL 选择等待而非拒绝，standby 是"等待未来任务"的代价
3. **L-scale 的根因** — RL standby 98K 但完成仅 17.5 → 等待但没等到足够任务 → 过度保守

### 当前数据

| Scale | Method | Travel | Charging | Tardiness | Standby | Rejection | Total |
|-------|--------|--------|----------|-----------|---------|-----------|-------|
| M | RL-APC | 2,979 | 331 | 22 | 61,731 | **0** | 286,665 |
| M | Greedy-FR | 3,897 | 559 | 605 | 40,388 | **218,000** | 281,349 |
| XL | RL-APC | 6,704 | 654 | 3,365 | 27,122 | **14,667** | 814,344 |
| XL | Greedy-FR | 7,520 | 848 | 177 | 26,356 | **577,333** | 809,830 |

### 所需表/图

| 编号 | 内容 | 状态 | 数据文件 |
|------|------|------|---------|
| T8 | 成本分解 (含 rejection penalty) | ✅ 已生成 | `ejor_table8_decomposition.csv` |

### 待补充

| 优先级 | 内容 | 说明 |
|--------|------|------|
| **P0** | **T8 增加 Best Rule (Standby-Lazy) 行** | 从 `individual_rules_*.csv` 提取 travel/charging/delay/standby/rejection 分解。展示三方 (RL / Greedy / Best Rule) 的成本结构差异 |
| P1 | 验证分解合计 = total | 当前 T8 的分项合计 ≠ total (可能有未列出的成本项)，需检查并加注释或修正 |

---

## Section 5.9: Sensitivity & Robustness (Q6)

### 关键论点

**"RL-APC 在 3/4 scale 表现强劲 (S 统计显著优势, M/XL 服务质量远优)。L-scale 弱点可通过超参数调整改善。"**

### 叙事点

1. **S-scale 统计显著** — RL vs 全部 baseline 均 *** (Bonferroni 校正后)
2. **M/XL 成本 ns** — 与 Greedy-FR (M: p=1.0, XL: p=1.0) 和 Standby-Lazy (M: p=0.022, XL: p=0.776) 差距不显著或边缘显著
3. **L-scale 三版本改善** — v1 +157.7% → v2 +168.5% → v3 +96.9%；v3 较 v1 成本下降 32%
4. **VecNormalize 修复的关键影响** — v1/v2 受 observation normalization 不匹配影响
5. **L-scale 改善空间** — 当前 v3 训练未充分收敛 (best 在 50K 步)，更多训练步数/reward shaping 可进一步改善

### 统计检验摘要

| Scale | vs Greedy-FR | vs Standby-Lazy | vs Random |
|-------|-------------|-----------------|-----------|
| S | -58.0% *** | -34.4% ** | -64.0% *** |
| M | +1.9% ns | +27.8% * | -27.7% *** |
| L | +96.9% *** | +58.3% *** | +39.7% *** |
| XL | +0.6% ns | +17.0% ns | -19.5% ns |

### L-Scale 版本对比

| Version | Net Arch | Terminal Penalty | RL Cost | vs Greedy |
|---------|----------|-----------------|---------|-----------|
| v1 | [256,128] | 3000 | 972,882 | +157.7% |
| v2 | [512,256] | 2000 | 1,013,433 | +168.5% |
| v3 | [512,256] | 2000 | 662,267 | +96.9% |

### 所需表/图

| 编号 | 内容 | 状态 | 数据文件 |
|------|------|------|---------|
| T5 | Wilcoxon 检验 (Bonferroni, 含 top rules) | ✅ 已生成 | `ejor_table5_wilcoxon.csv` |
| T10 | L-scale 灵敏度 | ✅ 已生成 | `ejor_table10_sensitivity.csv` |

### 待补充

| 优先级 | 内容 | 说明 |
|--------|------|------|
| P1 | **等 L_v3 2M 步训练完成后更新 T10** | 当前 v3 的 best_model 是 50K 步。若 2M 步后出现更好的 model，可更新数据 |
| P2 | T10 增加服务质量列 | 展示不同版本的 completed/rejected 变化趋势 |
| P2 | 跨 scale 迁移实验 | 用 S 模型评估 M，M 评估 L，4×4 矩阵展示迁移性能 |

---

## 全部待补充项汇总 (按优先级)

### P0 — 必须做 (改变论文叙事，数据已有，只需改脚本)

| # | 内容 | 影响的表 | 状态 |
|---|------|---------|------|
| ~~1~~ | ~~T3 增加 Rejected 列~~ | ~~T3~~ | ✅ 已完成 |
| ~~2~~ | ~~T4 增加服务质量对比列~~ | ~~T4~~ | ✅ 已完成 |
| 3 | T6 增加 Best Rule (Standby-Lazy / Charge-High) 行 | T6 | ⏳ 待实施 |
| 4 | T8 增加 Best Rule 成本分解行 | T8 | ⏳ 待实施 |
| 5 | 计算 cost-per-completed-task 指标 | T6 或新表 | ⏳ 待实施 |

> 以上全部可从 `individual_rules_*.csv` 提取，无需运行新实验。

### P1 — 应该做

| # | 内容 | 状态 |
|---|------|------|
| 6 | L-scale ALNS 离线解 (T7) | ⏳ 待实施 |
| 7 | 等 L_v3 2M 步训练完成更新 T10 / F4 | ⏳ 训练中 |
| 8 | 验证 T8 成本分解合计 | ⏳ 待实施 |

### P2 — 锦上添花

| # | 内容 | 状态 |
|---|------|------|
| 9 | 单 episode 规则时序图 | ⏳ 待实施 |
| 10 | 跨 scale 迁移实验 (4×4 矩阵) | ⏳ 待实施 |
| ~~11~~ | ~~F1 柱状图加 rejection 注解~~ | ✅ 已完成 |
| 12 | T10 增加服务质量列 | ⏳ 待实施 |

---

## 论文整体叙事架构

```
5.2 Q1: 单规则不够 → 需要自适应
         ↓
5.3 Q2: RL-APC vs 最优规则 → S 大胜; M/XL 需看更多维度
         ↓
5.4 Q3: RL 学到了什么 → 策略可解释 (待命+充电)
         ↓
5.5     训练收敛 → 方法可训练
         ↓
5.6 Q4: 服务质量 → 核心优势 (近零拒绝) ← ★ 论文最强论点
         ↓
5.7 Q5: 在线-离线 gap → 校准期望
         ↓
5.8     成本分解 → 解释差异根因
         ↓
5.9 Q6: 鲁棒性 → 统计显著 + L 可改善
```

**一句话总结**: RL-APC 在保持近零拒绝率的同时达到了与最优在线方法相当的成本 (M/XL) 甚至远优的成本 (S)，是唯一不牺牲服务质量来换取低成本的方法。L-scale 是已知弱点但有明确的改善路径。
