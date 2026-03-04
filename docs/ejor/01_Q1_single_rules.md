# Q1: 单条规则够用吗？

> **结论: 不够。没有任何单规则在所有 scale 都是最优的。L-scale 最优规则与 S/M/XL 不同，证明规则排名随 scale 变化。**

## 证据

### S-Scale 排名 (30 instances)

| Rank | Rule | Category | Avg Cost | Std |
|------|------|----------|----------|-----|
| 1 | **Standby-Lazy** | Standby | **60,731** | 18,969 |
| 2 | Charge-High | Charge | 83,962 | 30,289 |
| 3 | Charge-Opp | Charge | 83,962 | 30,289 |
| 4 | Insert-MinCost | Dispatch | 103,048 | 37,263 |
| 5 | MST | Dispatch | 108,707 | 26,740 |
| 6 | STTF / EDD / HPF | Dispatch | 110,551 | 25,238 |
| 9 | Charge-Med | Charge | 112,644 | 28,605 |
| 10 | Charge-Urgent | Charge | 118,337 | 24,625 |
| 11 | Charge-Low | Charge | 118,135 | 23,552 |
| 12 | Accept-Feasible | Accept | 118,405 | 24,589 |
| 13 | Standby-LowCost / Standby-Heatmap | Standby | 126,546 | 19,780 |
| 15 | Accept-Value | Accept | 157,360 | 45,840 |

- 最优 vs 最差: 60,731 vs 157,360 (**2.6x 差距**)
- Greedy-FR (=Charge-Opp) 排 #3，不是最优

### M-Scale 排名 (30 instances)

| Rank | Rule | Category | Avg Cost | Std |
|------|------|----------|----------|-----|
| 1 | **Standby-Lazy** | Standby | **224,316** | 29,499 |
| 2 | Charge-High | Charge | 266,218 | 47,455 |
| 3 | Charge-Opp | Charge | 266,218 | 47,455 |
| 4 | MST | Dispatch | 294,407 | 34,838 |
| 5 | STTF / EDD / HPF | Dispatch | 294,978 | 32,337 |
| 8 | Charge-Med | Charge | 299,669 | 33,231 |
| 9 | Accept-Feasible | Accept | 300,925 | 31,993 |
| 10 | Standby-LowCost / Standby-Heatmap | Standby | 301,201 | 30,232 |
| 12 | Charge-Urgent | Charge | 301,809 | 31,462 |
| 13 | Charge-Low | Charge | 335,128 | 89,983 |
| 14 | Insert-MinCost | Dispatch | 351,911 | 76,469 |
| 15 | Accept-Value | Accept | 490,642 | 66,308 |

- 最优 vs 最差: 224,316 vs 490,642 (**2.2x 差距**)
- Greedy-FR (=Charge-Opp) 排 #3，不是最优

### L-Scale 排名 (30 instances) ← NEW

| Rank | Rule | Category | Avg Cost | Std |
|------|------|----------|----------|-----|
| 1 | **Charge-High** | Charge | **336,386** | 89,646 |
| 2 | **Charge-Opp** | Charge | **336,386** | 89,646 |
| 3 | Standby-Lazy | Standby | 418,299 | 38,469 |
| 4 | Insert-MinCost | Dispatch | 447,292 | 119,773 |
| 5 | MST | Dispatch | 490,298 | 34,003 |
| 6 | Charge-Med | Charge | 490,829 | 43,791 |
| 7 | STTF / EDD / HPF | Dispatch | 491,192 | 37,627 |
| 10 | Standby-LowCost / Standby-Heatmap | Standby | 497,103 | 36,502 |
| 12 | Accept-Feasible | Accept | 497,360 | 35,833 |
| 13 | Charge-Urgent | Charge | 498,087 | 35,976 |
| 14 | Charge-Low | Charge | 737,186 | 197,302 |
| 15 | Accept-Value | Accept | 831,429 | 140,804 |

- 最优 vs 最差: 336,386 vs 831,429 (**2.5x 差距**)
- **最优规则变了！** Charge-High/Charge-Opp 超越 Standby-Lazy 成为 #1
- Standby-Lazy 降到 #3 (成本高出 24.3%)
- Greedy-FR (=Charge-Opp) 在 L-scale 恰好是最优单规则

### XL-Scale 排名 (30 instances) ← NEW

| Rank | Rule | Category | Avg Cost | Std |
|------|------|----------|----------|-----|
| 1 | **Standby-Lazy** | Standby | **696,148** | 67,334 |
| 2 | Insert-MinCost | Dispatch | 747,761 | 166,357 |
| 3 | Standby-LowCost / Standby-Heatmap | Standby | 762,554 | 68,749 |
| 5 | STTF / EDD / HPF | Dispatch | 764,969 | 64,185 |
| 8 | MST | Dispatch | 768,743 | 63,159 |
| 9 | Accept-Feasible | Accept | 780,760 | 65,479 |
| 10 | Charge-High | Charge | 790,373 | 212,838 |
| 11 | Charge-Opp | Charge | 790,373 | 212,838 |
| 12 | Charge-Urgent | Charge | 841,829 | 161,572 |
| 13 | Charge-Med | Charge | 915,014 | 230,302 |
| 14 | Charge-Low | Charge | 1,141,406 | 348,448 |
| 15 | Accept-Value | Accept | 1,360,537 | 225,075 |

- 最优 vs 最差: 696,148 vs 1,360,537 (**2.0x 差距**)
- Standby-Lazy 回到 #1，但 Charge-High/Charge-Opp 降到 #10
- Greedy-FR (=Charge-Opp) 排 #10，远非最优 (+13.5%)

## 关键发现

### 1. 最优规则随 scale 变化 — Q1 的核心证据

| Scale | #1 Rule | #1 Cost | #2 Rule | #2 Cost |
|-------|---------|---------|---------|---------|
| S | Standby-Lazy | 60,731 | Charge-High/Opp | 83,962 |
| M | Standby-Lazy | 224,316 | Charge-High/Opp | 266,218 |
| **L** | **Charge-High/Opp** | **336,386** | Standby-Lazy | 418,299 |
| XL | Standby-Lazy | 696,148 | Insert-MinCost | 747,761 |

**L-scale 的排名反转是关键发现**: Standby-Lazy 在 S/M/XL 均排 #1，但在 L 降到 #3。这直接证明"没有万能规则"。

### 2. Charge-High = Charge-Opp (全 scale 确认)

在所有 4 个 scale 上，Charge-High 和 Charge-Opp 的成本完全相同，证实 Greedy-FR ≡ Greedy-PR。

### 3. 规则间差距巨大

| Scale | Best | Worst | Ratio |
|-------|------|-------|-------|
| S | 60,731 | 157,360 | 2.6x |
| M | 224,316 | 490,642 | 2.2x |
| L | 336,386 | 831,429 | 2.5x |
| XL | 696,148 | 1,360,537 | 2.0x |

所有 scale 上最优 vs 最差均有 **2-2.6 倍差距**，规则选择极为重要。

### 4. Dispatch 规则 (STTF/EDD/HPF) 在所有 scale 上成本相同

STTF = EDD = HPF 在所有 4 个 scale 上成本完全相同。在当前仿真中，调度策略差异不影响总成本。

### 5. Accept-Value 始终最差

Accept-Value 在所有 4 个 scale 上排名最后，说明"挑选性接受任务"的策略在仿真环境中不合适。

### 6. Insert-MinCost 排名跳跃

- S: #4 (103,048)
- M: #14 (351,911)
- L: #4 (447,292)
- XL: #2 (747,761)

Insert-MinCost 在 XL 表现突出 (#2)，但在 M 很差 (#14)。

## 对 EJOR 论文的意义

1. **Q1 核心论点成立**: L-scale 的排名反转（Charge-High 取代 Standby-Lazy 成为 #1）直接证明没有万能规则
2. **Greedy-FR 不是最优 baseline**: 在 S/M/XL 上，Greedy-FR (=Charge-Opp) 都不是最优单规则
3. **但在 L-scale**: Greedy-FR 恰好是最优单规则 — 这解释了为什么 RL 在 L 上输得最惨
4. **规则选择的价值**: 2-2.6x 的最优-最差差距证明规则选择决策的重要性

## 对应图表

- T3: `individual_rules_{S,M,L,XL}_30.csv` (Section 5.2)
- F1: 分组柱状图 (Section 5.2)
- 数据: `results/benchmark/individual_rules_*_30.csv`
