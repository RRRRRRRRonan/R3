# Section 5 各小节：数据补充 + 展示方式 + 写作策略

> 基于 docs/ejor/ 全部文档的审查，结合 EJOR 审稿标准编写
> 建议结构：5.1 Setup (已完成) → 5.2–5.7 共六个实验小节

---

# 5.2 Individual Rule Performance (对应 Q1)

## 一、需要补充的实验数据/结果

### 已就绪 ✅
- 15 规则 × 4 scale × 30 instances 的完整评估数据 (`individual_rules_{S,M,L,XL}_30.csv`)
- F1 柱状图已生成 (含 rejection 注解)

### 需补充 ⚠️

| 补充项 | 原因 | 优先级 |
|--------|------|--------|
| 声明有效独立规则数 | STTF=EDD=HPF 成本完全相同，Charge-High=Charge-Opp 完全相同。15 条规则中实际独立的约 10–11 条。不声明则审稿人会指控"人为膨胀规则集" | **P0** |
| T3 表注中标注等价规则 | 在表格脚注说明 "STTF, EDD, HPF yield identical costs due to the event-driven execution layer resolving dispatch order deterministically" | **P0** |

**不需要额外实验**，仅需在现有数据的呈现上做补充说明。

## 二、展示方式

### Table 3: Individual Rule Performance (15 rules × 4 scales)
**展示格式：表格（必需）**

当前结构（行=规则，列=Scale×{Cost, Rej}）已合理。建议以下调整：

```
调整1: 等价规则合并为单行
  STTF / EDD / HPF → 合并为一行 "STTF†" 并在表注标 "† STTF, EDD, HPF yield identical results"
  Charge-High / Charge-Opp → 合并为 "Charge-High‡"，标 "‡ Charge-High ≡ Charge-Opp (= Greedy-FR)"
  合并后表格从 15 行缩减为 11 行，更紧凑

调整2: 每列最优值加下划线，RL-APC 单独列在表底部作参照
调整3: 按 Cost 升序排列（每个 scale 分别排）或按 Category 分组
  → EJOR 偏好按 Category 分组（Dispatch / Charge / Standby / Accept），组内按 Cost 排
```

**EJOR格式注意：** 这张表很大（11行 × 8数据列），建议用 `table*`（full-width）。如果版面实在紧张，可把 Rej 列移到 T6 的服务质量表中，T3 只保留 Cost。

### Figure 1 (F1): Rule Performance Bar Chart
**展示格式：图（推荐保留）**

四面板柱状图（每个scale一个面板），已有 rejection ≥ 5 的红色标注。建议：
- RL-APC 参照线用红色虚线，与柱子区分
- 柱子按 Cost 升序排列（而非按 rule ID），让视觉排名一目了然
- 图注中标明等价规则

## 三、文字描述策略

**本节目标：** 用一页左右建立"没有万能规则"的事实基础，为后续 RL 的自适应选择提供动机。

### 段落结构（建议 4 段，~400 词）

**第1段：开门见山给结论 + 引出 T3（~60词）**

> 直接陈述核心发现：没有任何单规则在所有 scale 上都最优。Table 3 reports the average cost...  The best-performing rule varies across scales: Standby-Lazy ranks first on S, M, and XL, but Charge-High takes the lead on L.

写法要点：EJOR 偏好"结论先行"，不要用 "In order to investigate..." 这类铺垫句式。

**第2段：规则间差距 + 等价规则发现（~100词）**

> 量化最优-最差差距（2.0–2.6x across scales）。说明规则选择的重要性。
> 坦率承认等价规则现象："Three dispatch rules (STTF, EDD, HPF) produce identical costs across all scales, indicating that task dispatch order has negligible impact under the event-driven execution model. Similarly, Charge-High and Charge-Opp coincide because..."
> 这段话既是诚实声明，也是为 Q3（RL 不使用 dispatch 规则）的发现铺垫。

**第3段：L-scale 排名反转 — 本节最关键的观察（~120词）**

> 专门讨论 L-scale 的排名反转。Charge-High 在 L 上取代 Standby-Lazy，说明最优规则对问题规模敏感。
> 解释可能原因："In the L-scale configuration (8 AMRs, 50–60 tasks), the higher task density increases the opportunity cost of prolonged idling, making proactive charging more advantageous."
> 这段是 Q1 论证的核心证据——如果所有 scale 的排名完全一致，Q1 的论点就弱得多。

**第4段：过渡到 5.3（~40词）**

> "The absence of a universally dominant rule motivates an adaptive selection mechanism. Section 5.3 evaluates whether the RL-APC policy can exploit this variability to achieve competitive performance without requiring manual rule selection."

### 写作禁忌
- ❌ 不要逐规则分析（15 条规则逐一讨论会让审稿人昏睡）
- ❌ 不要说 "interestingly" / "it is worth noting that"（EJOR 不喜欢的AI/教科书式表达）
- ✅ 聚焦 3 个关键发现：排名变化、差距幅度、等价规则

---

# 5.3 RL-APC vs Fixed Rules + Service Quality (Q2 + Q4 合并)

## 一、需要补充的实验数据/结果

### 已就绪 ✅
- T4: RL vs best-per-scale（含 cost + completed + rejected + p-value）
- T5: Wilcoxon 检验（Bonferroni 校正）
- 部分 T6 数据（RL vs Greedy-FR）
- 部分 T8 成本分解数据

### 需补充 ⚠️⚠️⚠️ （本节补充量最大）

| 补充项 | 具体内容 | 优先级 |
|--------|---------|--------|
| **T6 三方对比补全** | 加入 Best Rule (Standby-Lazy / Charge-High) 的 Completed + Rejected + Delay 数据。数据源: `individual_rules_*.csv`，只需改 `generate_ejor_tables.py` | **P0** |
| **Cost-per-completed-task 指标** | T6 增加一列 "Cost/Task = Total Cost ÷ Completed Tasks"。这是化解"RL 成本更高因为做了更多工作"困惑的关键指标 | **P0** |
| **T8 分项合计 ≠ Total 的修复** | S-scale RL: 1,417+131+137+64,734+0 = 66,419 ≠ 39,846。必须查清缺失/多余的成本项，加 "Other" 列或修正数据 | **P0** |
| **Rejection penalty sensitivity** | 测试 penalty = {5000, 10000, 20000, 50000} 下各算法总成本排名变化（不需重训RL，仅重新计算评估成本）。用 M-scale 即可 | **P1** |

### 需新增实验 🔬

**Rejection penalty sensitivity（P1）：** 这不需要重新训练RL或重新跑仿真。只需要在已有的评估结果上，按不同的 penalty 值重新计算总成本。具体做法：

```python
# 伪代码：从现有 eval CSV 读取各分项成本，按新 penalty 重算
for penalty in [0, 5000, 10000, 20000, 50000]:
    total_cost = travel + charging + tardiness + standby + penalty * rejected_tasks
```

产出一个小表（5 penalty × 3 methods × M-scale），展示RL的成本优势如何随 penalty 增大而增大。

## 二、展示方式

### Table 4 (扩展版): RL-APC vs Best Fixed Rule — 多维对比
**展示格式：表格（必需，full-width `table*`）**

建议将当前 T4 和 T6 合并为一张综合表：

```
| Scale | Best Rule | Rule Cost | Rule Comp | Rule Rej | RL Cost | RL Comp | RL Rej | RL Cost/Task | Rule Cost/Task | Δ_cost% | p-adj |
```

这样审稿人在一张表里就能同时看到 cost、service quality、efficiency 三个维度，不需要在 T4 和 T6 之间来回翻。

### Table 5: Wilcoxon Tests
**展示格式：表格（必需）**

当前结构合理。如果版面紧张，可以把 p-value 直接标注在 T4 的 Δ 列旁边（用上标符号 */**/***），省掉独立的 T5。EJOR 常见做法是在主结果表中嵌入显著性标注。

### Table 8 (修复版): Cost Decomposition
**展示格式：表格（必需）**

修复分项合计后，建议结构：

```
| Scale | Method | Travel | Charging | Tardiness | Standby | Rejection | Other | Total |
```

关键：每行的分项之和必须精确等于 Total。如果有 reward shaping credits 或 rounding，在表注中说明。

### Figure 2 (F2): Cost Distribution Boxplots
**展示格式：图（推荐保留）**

已有，但建议加入 Standby-Lazy。最终每个 scale 面板应包含：RL-APC、Greedy-FR、Standby-Lazy（或当scale最优规则）、1-2 条其他有代表性的规则。

### Figure (新增建议): Rejection Penalty Sensitivity
**展示格式：折线图（如果做了 penalty sensitivity 实验）**

横轴 = penalty 值 {0, 5K, 10K, 20K, 50K}，纵轴 = 总成本。3 条线（RL / Greedy-FR / Standby-Lazy）。应能清楚看到：penalty 增大时 Greedy/Standby-Lazy 的成本陡增（因为拒绝多），RL 几乎不变。交叉点的位置说明"在什么 penalty 水平下 RL 开始占优"。

## 三、文字描述策略

**本节目标：** 建立 RL-APC 的综合竞争力论证——不仅看成本，更看"成本-服务质量"的综合权衡。这是全文最需要精心措辞的小节。

### 段落结构（建议 6–7 段，~600–700 词）

**第1段：Overall cost comparison，引出合并后的 T4（~80词）**

> 以 S-scale 的明确优势开场："RL-APC achieves the lowest cost on S-scale instances, outperforming every individual rule including the best fixed rule (Standby-Lazy) by 34.4% (Wilcoxon p = 0.004)."
> 然后坦率承认："On M, L, and XL scales, the total cost of RL-APC exceeds that of the best fixed rule. However, a purely cost-based comparison is misleading, as detailed below."

写法要点：先给读者最强的正面结果（S），再诚实地引出需要进一步分析的部分。不要回避坏消息。

**第2段：引入 "cost trap" 概念（~100词）— 本段是论文叙事的转折点**

> 这一段需要清晰地建立一个核心论点：**成本最低的规则靠大量拒绝任务来实现低成本，在实际仓储运营中不可接受。**
>
> 措辞建议："The cost advantage of Standby-Lazy stems from a fundamentally different operating strategy: on M-scale, it rejects an average of 19.2 out of 34.8 incoming tasks (55%), effectively reducing its workload — and hence its cost — by declining service. In XL-scale, rejection reaches 64.6 tasks (73%). Such rejection rates are operationally infeasible in warehouse logistics, where order fulfilment commitments are contractually binding."
>
> 关键句：将拒绝率转化为业务语言（"contractually binding fulfilment"），让 EJOR 审稿人（含OR practitioner）直觉认同。

**第3段：Cost-per-completed-task 分析（~100词）**

> 引入 cost/task 指标来"拉平"工作量差异。
> "To account for the unequal workload, we compute cost per completed task. On M-scale, RL-APC achieves 15,838 per task versus 16,254 for Standby-Lazy — a 2.6% advantage despite higher total cost. On XL-scale, RL-APC completes 22.2 tasks at 36,683 per task, while Standby-Lazy completes only 18.8 tasks at 37,029 per task."
> 这个指标翻转了 M/XL 的叙事：RL 的单位效率实际上更优。

**第4段：M-scale 和 XL-scale 的详细服务质量论证（~100词）**

> 聚焦 M 和 XL（论文的两个"胜利 scale"）。
> M: "RL-APC completes 18.1 tasks with zero rejection versus 11.5 tasks and 21.8 rejections for Greedy-FR. The cost difference is a statistically insignificant +1.9% (p = 0.271)."
> XL: "RL-APC achieves 22.2 completions with 1.5 rejections, compared to 20.0 completions and 57.7 rejections for Greedy-FR (Δcost = +0.6%, p = 0.655)."
> 让数字说话，不做过度修饰。

**第5段：L-scale 的坦率讨论（~80词）**

> 不要试图辩解。直接承认并分析根因。
> "On L-scale, RL-APC incurs 96.9% higher cost than Charge-High. Cost decomposition (Table 8) reveals that standby cost accounts for XX% of RL-APC's total, indicating an overly conservative policy that idles rather than dispatches. This represents the primary limitation of the current training configuration; Section 5.7 discusses ongoing improvements."
> 简短有力，不超过 4 句话。

**第6段：成本分解（~80词）**

> 引出 T8，解释成本结构差异。
> "Table 8 decomposes total cost into operational components. For Greedy-FR on M-scale, rejection penalties constitute 77.5% of total cost, confirming that its low operating cost is an artefact of workload avoidance. RL-APC's cost is dominated by standby (XX%), reflecting its strategy of maintaining availability for incoming tasks rather than rejecting them."

**第7段：统计显著性 + 小结过渡（~60词）**

> 一句话概括 T5 的 Wilcoxon 结果，然后过渡。
> "Wilcoxon tests (Table 5) confirm the S-scale advantage at the 1% level. M- and XL-scale cost differences are statistically insignificant, while service quality differences are substantively large. To understand how RL-APC achieves this balance, we next examine its rule selection behaviour."

### 写作禁忌
- ❌ 不要把 Q2 和 Q4 分成两个 subsection。合并论证更有力——读者不需要翻来翻去
- ❌ 不要用 "interestingly" / "notably" / "it should be noted"
- ❌ 不要在 L-scale 上花超过一段的篇幅辩解
- ✅ 每段聚焦一个明确论点，用 2-3 个具体数字支撑
- ✅ 使用 business-relevant 的表述（"contractually binding", "operationally infeasible"）

---

# 5.4 Policy Interpretability — Rule Selection Behaviour (Q3)

## 一、需要补充的实验数据/结果

### 已就绪 ✅
- F3 热力图已生成（15 rules × 4 scales frequency matrix）
- Per-event-type breakdown CSV（4 scales）
- Top-3 规则选择数据

### 需补充 ⚠️

| 补充项 | 具体内容 | 优先级 |
|--------|---------|--------|
| **Per-event-type breakdown 可视化** | 目前只有 CSV，没有图。建议做一个堆叠柱状图：横轴 = event type（task_arrival / idle / low_battery / ...），纵轴 = 频率，颜色 = 规则。按 scale 分面板。展示"不同事件类型触发不同规则" | **P1** |
| **2-rule ablation 实验** | 只保留 Standby-Lazy + Charge-Opp 两条规则重训 RL，看性能是否下降。这是防御"为什么需要15条规则"的关键实验 | **P1** |

### 2-rule ablation 实验详情

这个实验的做法：修改 `rule_env.py` 中的规则集，只保留 Standby-Lazy 和 Charge-Opp（action_space=2），用相同超参数训练一个新 agent。在 M-scale 上跑 30 instances 评估。

可能的结果及应对：
- **如果 2-rule 性能接近 15-rule（差距 <3%）**：诚实承认，但论证"RL 自动发现这一简化本身就是 hyper-heuristic 的价值——免去了人工筛选规则的领域知识需求"
- **如果 2-rule 性能明显下降（>5%）**：证明完整规则集提供了"保险价值"——在某些边缘情况下其他规则有用，即使平均频率很低
- **无论哪种结果都可以写进论文**，这正是 EJOR 看重的诚实实验设计

## 二、展示方式

### Figure 3 (F3): Rule Selection Frequency Heatmap
**展示格式：图（必需，核心可视化）**

当前版本可用。建议调整：
- 合并等价规则（与 T3 保持一致）
- 使用 log-scale 颜色映射（否则 85-95% 的 Standby-Lazy + Charge-Opp 会淹没其他规则的细节）
- 或者用两层颜色：深色底表示 >10% 的规则，浅色表示 <10%
- 行按频率降序排列（最常用的规则在最上面）

### Figure (新增建议): Event-Type Breakdown
**展示格式：堆叠柱状图（P1 优先级）**

每个 scale 一个面板。横轴 = event type（如 task_arrival, amr_idle, low_battery, task_complete），纵轴 = 占比。颜色 = 规则。

核心叙事意图：展示 RL 在不同事件类型下选择不同规则——这是"context-dependent policy"的直接证据。例如：
- task_arrival → 主要选 Accept-Feasible 或 Standby-Lazy
- amr_idle → 主要选 Standby-Lazy 或 Charge-Opp
- low_battery → 主要选 Charge-High

### Table (可选): 2-Rule Ablation Results
**展示格式：小表格（如果做了实验）**

```
| Variant      | Rules | M Cost | M Comp | M Rej | Δ vs Full |
|--------------|-------|--------|--------|-------|-----------|
| Full (15)    | 15    | 286,665| 18.1   | 0.0   | —         |
| Reduced (2)  | 2     | xxx    | xx.x   | x.x   | +x.x%    |
```

如果只在 M-scale 做，一个小表即可。

## 三、文字描述策略

**本节目标：** 展示 RL 策略的可解释性，回应"black-box RL"的潜在质疑，同时主动化解"只用了2条规则"的攻击面。

### 段落结构（建议 4 段，~350 词）

**第1段：总体模式（~80词）**

> "To examine what the RL-APC policy has learned, we analyse the rule selection frequencies recorded during evaluation (Figure 3). Across all four scales, two rules dominate: Standby-Lazy and Charge-Opp, jointly accounting for 77–95% of all decisions. This reveals a structured, interpretable strategy centred on energy management and conservative waiting, rather than a diffuse selection over the full portfolio."

**第2段：Scale-dependent variation（~100词）**

> 讨论 S/XL 偏向 Charge-Opp（充电主导）vs. M/L 偏向 Standby-Lazy（待命主导）的差异。
> "On S- and XL-scale instances, Charge-Opp is the most frequently selected rule (44.7% and 54.6%, respectively), suggesting that proactive energy management is critical when AMR fleets are small or task arrival rates are high. On M- and L-scale, Standby-Lazy becomes dominant (64.5% and 59.1%), indicating that a more patient strategy is preferable when the fleet-to-task ratio allows for selective engagement."
> 给出合理的因果解释，而非仅描述频率数字。

**第3段：Dispatch 规则的"消失"——主动化解攻击面（~100词）**

> 这一段至关重要。不要等审稿人来攻击，自己先讨论。
> "A striking finding is that all five dispatch rules (STTF, EDD, MST, HPF, Insert-MinCost) are selected in fewer than 0.5% of decisions. This does not indicate design redundancy. Rather, it reflects the event-driven architecture: task-to-AMR assignment is resolved deterministically by the execution layer once an AMR becomes available, leaving the RL policy to focus on the higher-level decisions of when to charge, when to idle, and whether to accept incoming tasks. The full rule portfolio remains necessary as a search space from which the policy can discover this emergent simplification."

**第4段：过渡（~30词）**

> "Having established what the policy learns, we next assess the gap between online decision-making and offline optimisation."

### 写作禁忌
- ❌ 不要逐规则解释为什么使用或不使用每条规则
- ❌ 不要回避"只用了2条规则"的事实
- ✅ 主动承认 + 给出结构化解释 + 如有 ablation 数据则引用

---

# 5.5 Online-Offline Gap (Q5)

## 一、需要补充的实验数据/结果

### 已就绪 ✅
- S/M/XL 均有 ALNS 离线解
- Gap 计算完成

### 需补充 ⚠️

| 补充项 | 具体内容 | 优先级 |
|--------|---------|--------|
| **L-scale ALNS 数据** | 当前 L_v3 缺 ALNS 离线解。两种处理：(a) 跑 ALNS on L_v3 instances（最优）；(b) 在 T7 中标 "—" 并注明原因 | **P1** |
| **明确 gap 计算公式** | 在正文或表注中写清 gap = (online − offline) / offline × 100%。ALNS 不是 exact optimal，应称 "offline heuristic upper bound" | **P1** |
| **增加 Best Rule 到 T7** | 除了 RL 和 Greedy-FR，加入 Standby-Lazy 的 gap，让三方比较更完整 | **P2** |

### 不需要额外实验
除非要补 L-scale ALNS（需要运行 `evaluate_all.py --scale L --algorithms alns_fr`），其他只是数据呈现调整。

## 二、展示方式

### Table 7: Online-Offline Gap
**展示格式：表格（必需）**

建议结构：

```
| Scale | ALNS (Offline) | RL-APC | Greedy-FR | Best Rule | Gap_RL | Gap_Greedy | Gap_Best |
```

每列 Gap = (online − offline) / offline × 100%。这样审稿人一眼看到三种在线方法与离线解的距离。

### Figure (可选): Gap Bar Chart
**展示格式：分组柱状图（P2 优先级）**

如果数据简洁（4 scales × 3 methods），一张紧凑的柱状图可以直观展示"所有在线方法的 gap 都很大"这一事实。但 T7 本身已足够，图是锦上添花。

**总体判断：本节不需要图，一张表 + 文字解释即可。** 节约版面给更重要的 5.3 和 5.4。

## 三、文字描述策略

**本节目标：** 校准审稿人对在线方法的性能期望，说明在线-离线 gap 是问题固有的信息代价。本节篇幅应最短（~250 词）。

### 段落结构（建议 3 段，~250 词）

**第1段：Gap 的量级和普遍性（~80词）**

> "Table 7 compares all online methods against the ALNS offline heuristic. The gap exceeds 370% for every online method on every scale, confirming that the absence of future information imposes a substantial cost penalty inherent to the online setting."
> 不需要修饰，直接给出数字。

**第2段：S-scale 亮点 + M/XL 的趋同（~100词）**

> "On S-scale, RL-APC reduces the online-offline gap from 1694% (Greedy-FR) to 654%, a 61% relative reduction. This indicates that the adaptive policy captures regularities in small-scale instances that fixed rules miss. On M- and XL-scale, all online methods converge to similar gap levels (375–435%), suggesting that the online performance ceiling is largely determined by the information structure of the problem rather than the sophistication of the decision policy."
> 关键措辞："information structure of the problem"——这是OR文献中标准的表述，审稿人会认同。

**第3段：ALNS 不是 exact optimal 的声明 + 过渡（~70词）**

> "We note that ALNS provides a high-quality heuristic solution, not a certified lower bound. The true gap to optimality may be larger. Despite this caveat, the uniformly large gap across all online methods validates the inherent difficulty of the online scheduling problem and contextualises the RL-APC results presented in the preceding sections."

### 写作禁忌
- ❌ 不要在这一节花太多篇幅。250 词足够
- ❌ 不要称 ALNS 为 "optimal" 或 "lower bound"（它是上界/近似解）
- ✅ 要用 "online performance ceiling" / "information rent" 这类 OR 术语

---

# 5.6 Training and Computational Efficiency (Q5 补充 + T9)

## 一、需要补充的实验数据/结果

### 已就绪 ✅
- F4 训练曲线已生成（含 std band + best marker）
- T9 计算效率表（runtime per instance）

### 需补充 ⚠️

| 补充项 | 具体内容 | 优先级 |
|--------|---------|--------|
| **L_v3 完整训练曲线** | 当前 L_v3 仅到 ~650K/2M 步。完成 2M 步训练后需更新 F4 | **P1** |
| **Per-decision latency** | 当前 T9 只有 total episode runtime。需增加 per-decision latency (ms)。修改 `evaluate_all.py` 在每个决策点前后加 `time.perf_counter()` | **P1** |
| **GPU/CPU 配置声明** | T9 需表注说明硬件环境（GPU 型号、CPU、RAM），否则 runtime 数据不可复现 | **P1** |

## 二、展示方式

### Table 9: Computational Efficiency
**展示格式：表格（必需）**

建议扩展结构：

```
| Scale | Method       | Inference (ms/decision) | Episode (s) | Training (h) | Hardware |
|-------|-------------|------------------------|-------------|-------------|----------|
| S     | RL-APC      | ~x                     | xx          | x.x         | GPU      |
| S     | Greedy-FR   | ~x                     | xx          | —            | CPU      |
| S     | ALNS-FR     | —                      | xx          | —            | CPU      |
| ...   | ...         | ...                    | ...         | ...          | ...      |
```

核心论点：RL 的 inference 是 ms 级（单次前向传播），ALNS 的 per-instance solving time 是秒级。这在在线部署场景下有实质差异。

### Figure 4 (F4): Training Convergence Curves
**展示格式：图（必需）**

当前版本已有。建议：
- 4 条线（S/M/L/XL），用不同线型 + 颜色
- 如有 multi-seed，加半透明 confidence band
- 标注 best checkpoint 的位置（用小圆点 + 箭头）
- 如果各 scale 的 reward 量级差异很大，用 4 个小面板（subplots）而非单图
- X 轴统一到 timesteps（千步单位），方便跨 scale 比较收敛速度

## 三、文字描述策略

**本节目标：** 建立工程可落地性，论证 RL-APC 满足实时部署要求。篇幅控制在 ~300 词。

### 段落结构（建议 3 段，~300 词）

**第1段：Inference latency — 核心论点（~100词）**

> "For online deployment, the decision policy must produce actions within the physical execution cycle of the AMR fleet (typically several seconds per task handoff). Table 9 reports that RL-APC inference requires approximately X ms per decision, orders of magnitude below the execution time horizon. By contrast, ALNS requires X seconds per instance, precluding its use in real-time scheduling."
> 把 ms 和 seconds 的对比讲清楚。

**第2段：Training cost（~100词）**

> 引出 F4，报告各 scale 的训练时间和收敛特征。
> "Training is a one-time offline investment. S- and M-scale agents converge within X and Y hours respectively on a single [GPU model]. L-scale training exhibits slower convergence and higher variance (Figure 4), consistent with the increased state-action space complexity."
> 对 L 的收敛困难诚实报告，与 5.3 中 L-scale 的弱表现呼应。

**第3段：过渡（~40词）**

> "The computational profile confirms that RL-APC is suitable for online warehouse scheduling. The final section examines the robustness of these results."

---

# 5.7 Robustness and Sensitivity (Q6)

## 一、需要补充的实验数据/结果

### 已就绪 ✅
- T5 Wilcoxon 检验（Bonferroni 校正）
- T10 L-scale v1/v2/v3 改善趋势
- L_v3 训练仍在进行（将提供更完整的数据点）

### 需补充 ⚠️⚠️

| 补充项 | 具体内容 | 优先级 |
|--------|---------|--------|
| **至少 1 个 ablation 实验** | 移除 partial charging discretization（只保留 full recharge）。修改 `charge_level_ratios=[1.0]`，重训 M-scale agent，评估 30 instances。这直接量化 core contribution 的价值 | **P0** |
| **真正的 hyperparameter sensitivity** | 当前 T10 的 v1→v3 是 "debugging history" 而非 systematic sensitivity。需要在 M-scale 上测试 entropy coefficient {0.01, 0.03, 0.05, 0.10} 或 network size {[64,64], [128,64], [256,128], [512,256]}，每个变体训 + 评估 | **P1** |
| **Cross-scale transfer 测试** | S-scale 训练的 agent 在 M-scale 上的表现如何？如果 zero-shot transfer 完全失败，这也是有意义的 negative result | **P2** |

### Ablation 实验详情（P0 必做）

这是论文能否通过审稿的关键实验。EJOR 几乎一定会要求 ablation。

**最小可行 ablation（2 个变体足够）：**

| 变体 | 修改 | 验证的贡献 |
|------|------|-----------|
| RL-APC−PC | `charge_level_ratios=[1.0]`（只能 full recharge，不能选择 partial charging 档位） | Contribution: Learnable partial charging |
| RL-APC−FM | 关闭 `_compute_feasibility_mask()`（保留最小安全网防 crash） | Contribution: Safety shielding |

每个变体在 M-scale 上训练（用相同超参数、相同 training budget），评估 30 instances。

**预期结果：**
- RL-APC−PC 的成本应高于 Full（如果 partial charging 确实有用）
- RL-APC−FM 的成本可能高或低（masking 可能帮助也可能限制探索）

**即使结果 "不好"（如 −PC 和 Full 差别很小），也是有价值的发现**——说明在当前成本参数下 partial charging 的边际价值有限，这是一个诚实且有洞察力的结论。

## 二、展示方式

### Table (新增): Ablation Results
**展示格式：表格（必需）**

```
| Variant           | Description                    | M Cost  | M Comp | M Rej | Δ vs Full |
|-------------------|-------------------------------|---------|--------|-------|-----------|
| RL-APC (Full)     | Complete model                | 286,665 | 18.1   | 0.0   | —         |
| RL-APC−PC         | No partial charging           | xxx     | xx.x   | x.x   | +x.x%    |
| RL-APC−FM         | No feasibility masking        | xxx     | xx.x   | x.x   | +x.x%    |
| Random (= −RL)    | Uniform random rule selection | xxx     | xx.x   | x.x   | +x.x%    |
```

注意：Random baseline 在 5.3 中已有数据，可直接复用作为 "−RL" 变体。

### Table 10: L-Scale Version Comparison
**展示格式：表格（已有，调整呈现）**

当前 T10 展示 L v1/v2/v3。建议将其定位为 "training configuration sensitivity" 而非 "debugging history"。调整列名：

```
| Configuration | Net Arch | Steps | VecNorm | L Cost | Gap vs Best Rule |
```

### Table (可选): Hyperparameter Sensitivity
**展示格式：表格（如果做了实验）**

```
| Entropy Coef | M Cost | M Comp | Training Time | Notes          |
|-------------|--------|--------|---------------|----------------|
| 0.01        | xxx    | xx.x   | x.x h         | Low exploration|
| 0.03        | xxx    | xx.x   | x.x h         | Baseline       |
| 0.05        | xxx    | xx.x   | x.x h         | High exploration|
| 0.10        | xxx    | xx.x   | x.x h         | Very high      |
```

## 三、文字描述策略

**本节目标：** 通过 ablation 量化各组件贡献，通过 sensitivity 展示鲁棒性。

### 段落结构（建议 4–5 段，~400 词）

**第1段：Ablation 设计说明（~60词）**

> "To isolate the contribution of each architectural component, we evaluate two ablation variants on M-scale instances: RL-APC−PC removes partial-charging discretisation (all charging events use full recharge), and RL-APC−FM disables feasibility masking. Both variants are retrained from scratch with identical hyperparameters."

**第2段：Ablation 结果（~100词）**

> 引出 ablation table。按 Δ 值讨论。
> 如果 −PC 有显著成本增加："Removing partial charging increases total cost by X.X%, confirming that the SOC target discretisation provides a meaningful degree of freedom."
> 如果 −PC 差别不大："The marginal impact of partial charging (Δ = X.X%) suggests that under the current cost parameters, the distinction between full and partial recharge has limited influence. This finding motivates future work with more differentiated charging cost structures."
> **两种结果都有叙事路径**，不要害怕负面结果。

**第3段：Feasibility masking 贡献（~80词）**

> 讨论 −FM 的结果。masking 的价值不仅在成本，更在安全性：
> "Beyond cost impact, feasibility masking eliminates battery-infeasible decisions during evaluation: the full model records zero energy depletion events across all 30 instances, while RL-APC−FM triggers X emergency fallback interventions."

**第4段：L-scale sensitivity / hyperparameter sensitivity（~80词）**

> 简要报告 T10 或 hyperparameter sensitivity 结果。
> 如果做了 entropy coefficient 实验："Table X shows that performance is robust across entropy coefficients in [0.01, 0.05], with degradation observed only at 0.10 where excessive exploration prevents convergence within the training budget."

**第5段：小结 + 通向 Conclusion（~40词）**

> "The ablation analysis confirms that [component X] is the primary performance driver. These findings, together with the computational efficiency results, support the practical viability of the proposed framework."

---

# 全文表/图总汇（修订版）

| 编号 | 类型 | 名称 | 所在节 | 状态 | 优先级 |
|------|------|------|--------|------|--------|
| T3 | 表 | Individual Rule Performance (11 rows × 4 scales) | 5.2 | ✅ 需合并等价规则 | P0 |
| T4+T6 | 表 | RL vs Fixed Rules: Cost + Service Quality (合并) | 5.3 | ⚠️ 需补 Best Rule 行 + cost/task | P0 |
| T5 | 表 | Wilcoxon Tests (可嵌入 T4) | 5.3 | ✅ | P0 |
| T8 | 表 | Cost Decomposition | 5.3 | ⚠️ 需修复分项≠总和 | P0 |
| T7 | 表 | Online-Offline Gap | 5.5 | ✅ (L 缺 ALNS) | P1 |
| T9 | 表 | Computational Efficiency | 5.6 | ⚠️ 需补 per-decision latency | P1 |
| T_abl | 表 | Ablation Results (新增) | 5.7 | ❌ 需跑实验 | **P0** |
| T10 | 表 | L-scale Sensitivity | 5.7 | ✅ 调整呈现 | P1 |
| F1 | 图 | Rule Performance Bar Chart | 5.2 | ✅ | P0 |
| F2 | 图 | Cost Distribution Boxplots | 5.3 | ✅ 加 Best Rule | P1 |
| F3 | 图 | Rule Selection Heatmap | 5.4 | ✅ 合并等价规则 | P0 |
| F4 | 图 | Training Convergence Curves | 5.6 | ⚠️ 等 L_v3 完训更新 | P1 |

**总计：8 张表 + 4 张图 = 12 个展示元素**

---

# 优先级执行清单

## Blocking (提交前必须完成)

1. ☐ **T8 成本分解修复** — 查清分项合计 ≠ Total 的原因，修复或加 Other 列
2. ☐ **T4+T6 合并 + 补全** — 加入 Best Rule 行 + cost-per-task 列
3. ☐ **Ablation 实验** — 至少 RL-APC−PC (no partial charging) on M-scale
4. ☐ **T3 合并等价规则** — 11 行 + 表注说明

## High Priority (强烈推荐)

5. ☐ Per-decision latency 数据 (T9)
6. ☐ L_v3 完整训练曲线更新 (F4)
7. ☐ Rejection penalty sensitivity (M-scale, 不需重训)
8. ☐ Gap 公式定义 + ALNS 定位声明 (T7 表注)

## Nice-to-have

9. ☐ Event-type breakdown 可视化 (5.4)
10. ☐ 2-rule ablation (5.4)
11. ☐ Entropy coefficient sensitivity (5.7)
12. ☐ L-scale ALNS 离线解 (T7)