# EJOR Paper — Simulation Results: 六步论证总览

> 最后更新: 2026-03-03 (15规则全 scale 评估完成)
> 数据来源: evaluate_{S,M_synced,L_v3,XL_synced}_30.csv + individual_rules_{S,M,L,XL}_30.csv

## 论证主线

```
Q1 → Q2 → Q3 → Q4 → Q5 → Q6
没有万能规则 → RL 自适应有效 → RL 策略可解释 → 服务质量优势 → 在线代价量化 → 方法鲁棒
```

## 各步结论一览

| 步骤 | 研究问题 | 结论 | 强度 | 详见 |
|------|---------|------|------|------|
| Q1 | 单条规则够用吗？ | 不够。S/M/XL 最优=Standby-Lazy，**L 最优=Charge-High** → 排名变化 | **强** | `01_Q1_single_rules.md` |
| Q2 | RL-APC 优于最佳单规则？ | S 大胜 (-34.4%)；M +27.8%；**L +96.9%**；**XL +17.0%** | **中偏弱** | `02_Q2_rl_vs_best_rule.md` |
| Q3 | RL 学到了什么？ | 以 Standby-Lazy + Charge-Opp 为核心，不同 scale 比例不同 | **强** | `03_Q3_rule_selection.md` |
| Q4 | 成本之外的优势？ | M/XL 服务质量远优（拒绝率从 21.8/57.7 降到 0/1.5） | **强** | `04_Q4_service_quality.md` |
| Q5 | 在线-离线 gap？ | 所有方法 gap 均 >370%，gap 是问题固有的 | **强** | `05_Q5_online_offline_gap.md` |
| Q6 | 方法鲁棒吗？ | L-scale 三版本改善趋势明确；S 统计显著 | **中** | `06_Q6_robustness.md` |

## 数据就绪状态

| 步骤 | 数据状态 | 缺失项 |
|------|---------|--------|
| Q1 | ✅ 完成 (4/4 scale) | — |
| Q2 | ✅ 完成 (4/4 scale) | — |
| Q3 | ✅ 完成 | — |
| Q4 | ✅ 完成 | — |
| Q5 | ✅ 完成 (L 缺 ALNS) | L-scale ALNS 离线解 |
| Q6 | ✅ 完成 | — |

## 总体评价

**论文核心叙事 (per scale)**:

| Scale | 成本 | 服务质量 | 综合判定 |
|-------|------|---------|---------|
| S | RL 大胜 (-58%) *** | 优 | **全面胜出** |
| M | 持平 (+1.9%) ns | RL 远优 (0 vs 21.8 拒绝) | **RL 综合更优** |
| L | RL 大输 (+96.9%) *** | 弱 | **已知弱点** |
| XL | 持平 (+0.6%) ns | RL 远优 (1.5 vs 57.7 拒绝) | **RL 综合更优** |

**论文可写的方向**: RL-APC 在 S-scale 成本大胜，在 M/XL-scale 以成本持平换取极大服务质量提升，L-scale 是已知 limitation。六步论证提供了远超"稻草人对比"的实验深度。

**主要风险**:
1. **Q2 论点偏弱**: RL 仅在 S 胜过最优单规则，M/L/XL 均输。但 vs Greedy-FR（公平在线对比）: S 大胜, M/XL 持平
2. **L-scale 明确失败**: 成本 +96.9%，且 Greedy-FR 恰好是 L 的最优单规则
3. **新发现 — L 排名反转**: L 的最优规则是 Charge-High (非 Standby-Lazy)，强化了 Q1 论点

## 文件清单

```
docs/ejor/
├── 00_overview.md          ← 本文件
├── 01_Q1_single_rules.md   ← 单规则排名 (S/M 完成, L/XL 待数据)
├── 02_Q2_rl_vs_best_rule.md ← RL vs 最优单规则
├── 03_Q3_rule_selection.md  ← 规则选择热力图分析
├── 04_Q4_service_quality.md ← 服务质量 + 成本分解
├── 05_Q5_online_offline_gap.md ← 在线-离线 gap 分析
└── 06_Q6_robustness.md      ← 统计检验 + L-scale 灵敏度
```
