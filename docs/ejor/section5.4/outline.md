# Section 5.4 详细方案
# Policy Interpretability — Rule Selection Behaviour
# "RL-APC 是怎么赢的？"

> 对齐 `section5_subsection_guide_detailed.md` 中 5.4 的完整规划。
> 研究问题：Q3（RL策略可解释性 + 规则选择行为）
> 目标篇幅：~400 词，4 段正文 + 1 张热力图 + 1 张事件分解图（可选）
> 前置依赖：Section 5.3 已建立 RL-APC 全面胜出的事实

---

## 一、核心叙事："RL-APC 怎么赢的？"

### 三层递进回答

1. **策略结构清晰**（不是黑盒）：两条规则主导 77-95% 的决策
   - Standby-Lazy（原地待命）+ Charge-Opp（机会充电）
   - RL 自动发现了"少做决策反而更好"的策略

2. **上下文自适应**（这是固定规则做不到的）：
   - 不同 scale → 不同策略比例（S/XL 偏充电，M/L 偏待命）
   - 不同事件类型 → 选择不同规则（task_arrival → Accept-Value, idle → Standby-Lazy, deadlock → Charge-Opp）

3. **架构层面的洞察**（dispatch 规则消失的解释）：
   - 5 条 dispatch 规则使用率 < 0.5%
   - 不是冗余设计，而是 RL 发现了：任务分配由 execution layer 处理，高层策略只需管理 充电/待命/接受

---

## 二、数据来源

### 2.1 已就绪数据

| 数据文件 | 描述 | 位置 |
|----------|------|------|
| `rule_selection_heatmap.csv` | 15 rules × 4 scales 频率矩阵 | `results/paper/` |
| `rule_freq_{S,M,L,XL}.csv` | 各 scale 规则频率详表 | `results/paper/` |
| `rule_event_breakdown_{S,M,L,XL}.csv` | 事件类型 × 规则分解 | `results/paper/` |
| `fig_rule_selection_heatmap.png` | 热力图（F3） | `results/paper/` |
| `fig_rule_bar_chart.png/pdf` | 柱状图（F1, Section 5.2） | `results/paper/` |

### 2.2 需要新增/补充

| 补充项 | 描述 | 优先级 |
|--------|------|--------|
| 事件分解堆叠柱状图 | 横轴=event type，纵轴=频率，颜色=规则 | **P1** |
| 热力图 PDF 版本 | 当前只有 PNG | P2 |

---

## 三、Figure 3: Rule Selection Frequency Heatmap

### 3.1 当前版本
- 15 rules (rows) × 4 scales (columns)
- YlOrRd 颜色映射，标注频率百分比
- 按 Category 分组（Dispatch / Charge / Standby / Accept）

### 3.2 关键数据（来自 heatmap.csv）

| Rule | S | M | L | XL | 类别 |
|------|---|---|---|----|----|
| **Charge-Opp** | **44.7%** | 30.0% | 27.4% | **54.6%** | Charge |
| **Standby-Lazy** | 32.0% | **64.5%** | **59.1%** | 29.5% | Standby |
| Standby-LowCost | 10.2% | 1.7% | 4.0% | 3.0% | Standby |
| Standby-Heatmap | 10.3% | 1.5% | 4.4% | 2.7% | Standby |
| Accept-Value | 0.9% | 1.4% | 2.1% | **6.5%** | Accept |
| Charge-High | 0.3% | 0.1% | 0.7% | 0.6% | Charge |
| 5 Dispatch rules | <0.3% each | <0.1% each | <0.3% each | <0.5% each | Dispatch |

### 3.3 Two-rule dominance

| Scale | Standby-Lazy + Charge-Opp | 说明 |
|-------|--------------------------|------|
| S | 32.0% + 44.7% = **76.7%** | Charge-Opp 主导 |
| M | 64.5% + 30.0% = **94.5%** | Standby-Lazy 主导 |
| L | 59.1% + 27.4% = **86.5%** | Standby-Lazy 主导 |
| XL | 29.5% + 54.6% = **84.1%** | Charge-Opp 主导 |

---

## 四、事件类型分解（核心可解释性证据）

### 4.1 各事件类型的主导规则

| 事件类型 | S-scale 主导 | M-scale 主导 | L-scale 主导 | XL-scale 主导 |
|----------|-------------|-------------|-------------|--------------|
| **TASK_ARRIVAL** | Accept-Value (47.2%) | Accept-Value (79.7%) | Accept-Value (59.7%) | Accept-Value (73.5%) |
| **ROBOT_IDLE** | Charge-Opp (45.4%) | Standby-Lazy (71.4%) | Standby-Lazy (66.2%) | Charge-Opp (59.0%) |
| **DEADLOCK_RISK** | Charge-Opp (52.0%) | Charge-Opp (86.0%) | Charge-Opp (58.8%) | Charge-Opp (76.0%) |
| **CHARGE_DONE** | Standby-Heatmap (36.1%) | Standby-Lazy (27.5%) | Standby-Lazy (44.0%) | Standby-Lazy (38.1%) |
| **SOC_LOW** | Standby-Heatmap (34.7%) | Charge-Opp (34.7%) | Charge-Opp (55.5%) | Charge-Opp (39.5%) |

### 4.2 核心发现：Context-Dependent Policy

1. **TASK_ARRIVAL → Accept-Value 主导**（47-80%）
   - RL 在任务到达时主要使用 Accept-Value 决定是否接受
   - 这是唯一大量使用 Accept 类规则的场景
   - M-scale 最高（79.7%），因为 M 有更多任务需要筛选

2. **ROBOT_IDLE → Standby-Lazy 或 Charge-Opp**
   - M/L: Standby-Lazy 主导（66-71%）→ 保守等待
   - S/XL: Charge-Opp 主导（45-59%）→ 主动充电

3. **DEADLOCK_RISK → Charge-Opp 压倒性主导**（52-86%）
   - 所有 scale 的 deadlock 事件都选择 Charge-Opp
   - 解释：Charge-Opp 将 AMR 送去充电，天然解除拥堵
   - **这是 RL 发现的最聪明策略之一**

4. **Dispatch 规则几乎只在 TASK_ARRIVAL 出现**
   - STTF/EDD/MST/HPF 在 task_arrival 中有 7-13% 的使用率
   - 在其他事件中接近 0%
   - 说明 dispatch 规则仅在需要分配任务时偶尔被选用

---

## 五、正文 4 段逐段方案

### ¶1 总体模式 (~80 词)

**论点**：RL 策略有清晰结构，两条规则主导 77-95% 的决策。

**模板**：
> To examine what the RL-APC policy has learned, we analyse the rule
> selection frequencies recorded during evaluation (Figure 3). Across all
> four scales, two rules dominate: **Standby-Lazy** and **Charge-Opp**,
> jointly accounting for **77--95%** of all decisions (Table 8). This
> reveals a structured, interpretable strategy centred on energy management
> and conservative waiting, rather than a diffuse selection over the full
> 15-rule portfolio.

**数据来源**：
- Two-rule sum: S=76.7%, M=94.5%, L=86.5%, XL=84.1%

### ¶2 Scale-dependent variation (~100 词)

**论点**：不同 scale 下策略比例有系统性变化，这是固定规则做不到的自适应。

**模板**：
> The relative weight of these two rules varies systematically with problem
> scale. On S- and XL-scale instances, **Charge-Opp** is the most frequently
> selected rule (**44.7%** and **54.6%**, respectively), reflecting the
> heightened importance of proactive energy management when AMR fleets are
> small or task arrival rates are high. On M- and L-scale instances,
> **Standby-Lazy** becomes dominant (**64.5%** and **59.1%**), indicating
> that a more patient strategy is preferable when the fleet-to-task ratio
> permits selective engagement. This scale-dependent rebalancing is precisely
> the adaptive capability that no fixed rule can provide.

**数据来源**：
- S: Charge-Opp=44.7%, Standby-Lazy=32.0%
- M: Standby-Lazy=64.5%, Charge-Opp=30.0%
- L: Standby-Lazy=59.1%, Charge-Opp=27.4%
- XL: Charge-Opp=54.6%, Standby-Lazy=29.5%

### ¶3 Context-dependent selection + Dispatch 规则消失 (~120 词)

**论点**：RL 在不同事件类型下选择不同规则（上下文自适应），且 dispatch 规则的消失是合理的。

**模板**：
> The event-type breakdown reveals further context dependence. On task
> arrival events, the policy predominantly selects **Accept-Value**
> (**47--80%** across scales), using the accept/reject decision to manage
> workload. When AMRs become idle, the policy reverts to **Standby-Lazy**
> or **Charge-Opp** depending on scale. For deadlock-risk events,
> **Charge-Opp** dominates on all scales (**52--86%**), effectively
> resolving congestion by routing AMRs to charging stations.
>
> A striking finding is that all five dispatch rules (STTF, EDD, MST, HPF,
> Insert-MinCost) are selected in fewer than **0.5%** of decisions. This
> reflects the event-driven architecture: task-to-AMR assignment is
> resolved deterministically by the execution layer, leaving the RL policy
> to focus on higher-level decisions — when to charge, when to idle, and
> whether to accept incoming tasks.

**数据来源**：
- TASK_ARRIVAL: Accept-Value 47-80%
- DEADLOCK_RISK: Charge-Opp 52-86%
- ROBOT_IDLE: Standby-Lazy (M/L 66-71%) or Charge-Opp (S/XL 45-59%)
- Dispatch rules: all <0.5%

### ¶4 过渡到 5.5 (~30 词)

**模板**：
> Having established what the policy learns, we next assess the gap
> between online decision-making and offline optimisation.

---

## 六、Table 8 (论文编号): Rule Selection Summary

### 设计（可选，作为 Figure 3 的数据补充）

如果篇幅允许，可在正文中用一个简洁的 inline 表总结 top-3 rules：

| Scale | #1 Rule (%) | #2 Rule (%) | #3 Rule (%) | Top-2 Sum |
|-------|-------------|-------------|-------------|-----------|
| S | Charge-Opp (44.7) | Standby-Lazy (32.0) | Standby-LowCost (10.2) | 76.7% |
| M | Standby-Lazy (64.5) | Charge-Opp (30.0) | Standby-LowCost (1.7) | 94.5% |
| L | Standby-Lazy (59.1) | Charge-Opp (27.4) | Standby-Heatmap (4.4) | 86.5% |
| XL | Charge-Opp (54.6) | Standby-Lazy (29.5) | Accept-Value (6.5) | 84.1% |

判断：**Figure 3 (热力图) 已经包含全部信息，不需要额外表格。**
如果审稿人要求数字精度，可在表注或附录中列出。

---

## 七、写作检查清单

### 数据一致性
- [ ] ¶1 的 "77-95%" 与 heatmap.csv 计算一致
- [ ] ¶2 的百分比与 rule_freq_*.csv 一致
- [ ] ¶3 的事件分解百分比与 rule_event_breakdown_*.csv 一致
- [ ] ¶3 的 "< 0.5%" dispatch 规则声明与 4 个 scale 的数据一致

### 叙事一致性
- [ ] ¶1 的 "two rules dominate" 与 5.3 的 Option B 结果呼应
- [ ] ¶2 的 scale-dependent 解释与 5.2 的 L-scale 排名反转一致
- [ ] ¶3 的 dispatch 规则消失与 5.2 的等价规则发现一致
- [ ] ¶4 过渡到 5.5 "online-offline gap" 与 Q5 内容吻合

### EJOR 风格
- [ ] 无 "interestingly" / "notably" / "it is worth noting"
- [ ] 结论先行：每段第一句给结论
- [ ] 不逐规则分析（聚焦 3 个发现）
- [ ] 主动化解 "只用了2条规则" 的攻击面

---

## 八、版面预算

| 元素 | 预估版面 |
|------|---------|
| Figure 3 (热力图) | ~1/3 page |
| 正文 4 段 (~400 词) | ~2/3 page |
| **合计** | **~1 page** |

EJOR 单节 1 页是紧凑合理的。本节不需要表格（热力图已足够）。
