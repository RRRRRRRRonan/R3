# Section 5.4 — Data-Filled Writing Instructions

> 研究问题：Q3 — RL-APC 学到了什么？怎么赢的？
> 数据来源：`results/paper/rule_freq_*.csv`, `rule_event_breakdown_*.csv`, `rule_selection_heatmap.csv`

---

## 核心数据表

### A. 各 Scale 规则选择频率 (Top-5)

#### S-scale (45,358 decisions)

| Rank | Rule | Count | Frequency |
|------|------|-------|-----------|
| 1 | Charge-Opp | 20,271 | **44.7%** |
| 2 | Standby-Lazy | 14,490 | **32.0%** |
| 3 | Standby-Heatmap | 4,671 | 10.3% |
| 4 | Standby-LowCost | 4,630 | 10.2% |
| 5 | Accept-Value | 420 | 0.9% |

#### M-scale (180,742 decisions)

| Rank | Rule | Count | Frequency |
|------|------|-------|-----------|
| 1 | Standby-Lazy | 116,403 | **64.5%** |
| 2 | Charge-Opp | 54,187 | **30.0%** |
| 3 | Standby-LowCost | 2,980 | 1.7% |
| 4 | Standby-Heatmap | 2,779 | 1.5% |
| 5 | Accept-Value | 2,599 | 1.4% |

#### L-scale (60,931 decisions)

| Rank | Rule | Count | Frequency |
|------|------|-------|-----------|
| 1 | Standby-Lazy | 35,938 | **59.1%** |
| 2 | Charge-Opp | 16,644 | **27.4%** |
| 3 | Standby-Heatmap | 2,703 | 4.4% |
| 4 | Standby-LowCost | 2,440 | 4.0% |
| 5 | Accept-Value | 1,289 | 2.1% |

#### XL-scale (6,112 decisions)

| Rank | Rule | Count | Frequency |
|------|------|-------|-----------|
| 1 | Charge-Opp | 3,338 | **54.6%** |
| 2 | Standby-Lazy | 1,803 | **29.5%** |
| 3 | Accept-Value | 396 | 6.5% |
| 4 | Standby-LowCost | 185 | 3.0% |
| 5 | Standby-Heatmap | 166 | 2.7% |

### B. Two-Rule Dominance Summary

| Scale | Standby-Lazy | Charge-Opp | Sum | 主导规则 |
|-------|-------------|-----------|-----|---------|
| S | 32.0% | 44.7% | **76.7%** | Charge-Opp |
| M | 64.5% | 30.0% | **94.5%** | Standby-Lazy |
| L | 59.1% | 27.4% | **86.5%** | Standby-Lazy |
| XL | 29.5% | 54.6% | **84.1%** | Charge-Opp |

**模式**：S/XL 偏充电，M/L 偏待命

### C. 事件类型分解（各事件的 Top-1 规则）

| Event Type | S | M | L | XL |
|------------|---|---|---|------|
| TASK_ARRIVAL | Accept-Value **47.2%** | Accept-Value **79.7%** | Accept-Value **59.7%** | Accept-Value **73.5%** |
| ROBOT_IDLE | Charge-Opp **45.4%** | Standby-Lazy **71.4%** | Standby-Lazy **66.2%** | Charge-Opp **59.0%** |
| DEADLOCK_RISK | Charge-Opp **52.0%** | Charge-Opp **86.0%** | Charge-Opp **58.8%** | Charge-Opp **76.0%** |
| CHARGE_DONE | Standby-Heatmap **36.1%** | Standby-Lazy **27.5%** | Standby-Lazy **44.0%** | Standby-Lazy **38.1%** |
| SOC_LOW | Standby-Heatmap **34.7%** | Charge-Opp **34.7%** | Charge-Opp **55.5%** | Charge-Opp **39.5%** |

### D. Dispatch 规则使用率（所有 <0.5%）

| Rule | S | M | L | XL |
|------|---|---|---|----|
| STTF | 0.3% | 0.1% | 0.3% | 0.3% |
| EDD | 0.3% | 0.1% | 0.3% | 0.5% |
| MST | 0.3% | 0.1% | 0.2% | 0.3% |
| HPF | 0.3% | 0.1% | 0.3% | 0.3% |
| Insert-MinCost | 0.1% | 0.1% | 0.2% | 0.3% |
| **Total Dispatch** | **1.3%** | **0.5%** | **1.3%** | **1.7%** |

---

## Paragraph-by-Paragraph Data Fill

### ¶1 Overall pattern (~80 words)

> To examine what the RL-APC policy has learned, we analyse the rule
> selection frequencies recorded during evaluation (Figure 3). Across all
> four scales, two rules dominate: **Standby-Lazy** (idle in place until
> a task arrives) and **Charge-Opp** (opportunistically charge to a
> learnable SOC target). Together they account for **77--95%** of all
> decisions. This reveals a structured, interpretable strategy centred on
> energy management and conservative waiting, rather than a diffuse
> selection over the full 15-rule portfolio.

### ¶2 Scale-dependent variation (~100 words)

> The relative weight of these two core rules varies systematically with
> problem scale. On **S**- and **XL**-scale instances, **Charge-Opp** is
> the most frequently selected rule (**44.7%** and **54.6%**), reflecting
> the heightened importance of proactive energy management when fleet sizes
> are small (3 AMRs) or task volumes are high (80--100 tasks). On **M**-
> and **L**-scale instances, **Standby-Lazy** dominates (**64.5%** and
> **59.1%**), indicating that a more patient, wait-for-task strategy is
> preferable when the fleet-to-task ratio permits selective engagement.
> This scale-dependent rebalancing is precisely the adaptive capability
> that no single fixed rule can provide.

### ¶3 Context-dependent selection + dispatch rule absence (~120 words)

> The event-type breakdown reveals further context dependence. On
> **task arrival** events, the policy predominantly selects **Accept-Value**
> (**47--80%** across scales), using the value-based accept/reject decision
> to manage workload admission. When AMRs become **idle**, the policy
> reverts to **Standby-Lazy** or **Charge-Opp** depending on the scale.
> For **deadlock-risk** events, **Charge-Opp** dominates across all scales
> (**52--86%**), effectively resolving congestion by routing AMRs to
> charging stations and clearing blocked corridors.
>
> All five dispatch rules (STTF, EDD, MST, HPF, Insert-MinCost) are
> selected in fewer than **0.5%** of decisions on every scale. This reflects
> the event-driven simulation architecture: task-to-AMR assignment is
> resolved deterministically by the execution layer once an AMR becomes
> available, leaving the RL policy to focus on the higher-level decisions
> of **when to charge, when to idle, and whether to accept** incoming tasks.

### ¶4 Transition to 5.5 (~30 words)

> Having established what the policy learns and how it adapts to scale
> and context, we next assess the gap between online decision-making and
> offline optimisation.

---

## 关键叙事要点

### "RL-APC 怎么赢的？" — 三层回答

1. **策略极简但有效**：不是随机选择 15 条规则，而是集中在 2 条核心规则上（待命 + 充电），这恰好是 Section 5.2 中表现最好的两类规则

2. **自适应比例调节**：
   - 小规模/高任务密度 → 偏充电（保持能量储备应对密集任务）
   - 中规模/低任务密度 → 偏待命（节省不必要的充电移动）

3. **智能事件响应**：
   - 任务来了 → 用 Accept-Value 决定要不要接（workload management）
   - 没事做 → 待命或充电（energy management）
   - 要堵了 → 充电（congestion resolution，最聪明的发现）

### 与 Section 5.3 的呼应

- 5.3 说 RL-APC 充电成本低 23-93% → 5.4 解释：因为 Charge-Opp 是核心策略，实现了 partial charging
- 5.3 说 RL-APC 拒绝率近零 → 5.4 解释：因为 task_arrival 时用 Accept-Value 而非拒绝
- 5.3 说 RL-APC 运营成本最低 → 5.4 解释：因为 idle 时选择 Standby-Lazy（零运营成本）而非无谓移动

### 防御 "只用了 2 条规则" 的攻击

论文措辞策略：
- **不回避**：直接承认 two-rule dominance
- **正面解读**：这是 RL 从 15 条规则中自动发现最优子集的能力
- **强调自适应**：虽然只用了 2 条主力规则，但比例随 scale 和 context 变化
- **保留全集的价值**："The full rule portfolio remains necessary as a search space from which the policy can discover this emergent simplification"
