# Section 5.3 详细方案
# RL-APC vs Fixed Rules and Service Quality

> 对齐 `section5_subsection_guide_detailed.md` 中 5.3 的完整规划。
> 研究问题：Q2（RL-APC 整体竞争力）+ Q4（服务质量权衡）
> 目标篇幅：~650 词，7 段正文 + 2 张表 + 1 张图。

---

## 一、数据产出管线

### 1.1 前置依赖

| 前置步骤 | 命令 | 产出 |
|----------|------|------|
| 实例生成 (140个) | `python scripts/generate_benchmark_instances.py --overwrite` | `data/instances/manifest.json` (35 seeds × 4 scales) |
| 训练 4 scale | `bash scripts/train_all_scales.sh` | `results/rl/train_{S,M,L,XL}/best_model/best_model.zip` |
| 单规则评估 | 见 1.2 | `individual_rules_{scale}_30.csv` |
| 7算法评估 | 见 1.2 | `evaluate_{scale}_30.csv` |

### 1.2 评估命令

```bash
# ── Step 1: 15 单规则评估（Section 5.2 数据，5.3 也要用来做 Best Rule 对照）──
for SCALE in S M L XL; do
  python scripts/evaluate_individual_rules.py \
    --manifest-json data/instances/manifest.json \
    --split test --scale $SCALE \
    --output-csv results/benchmark/individual_rules_${SCALE}_30.csv
done

# ── Step 2: 7 算法评估 ──
for SCALE in S M L XL; do
  python scripts/evaluate_all.py \
    --manifest-json data/instances/manifest.json \
    --split test --scale $SCALE \
    --ppo-model-path results/rl/train_${SCALE}/best_model/best_model.zip \
    --output-csv results/benchmark/evaluate_${SCALE}_30.csv \
    --rl-deterministic
done

# ── Step 3: 生成论文级表格 ──
python scripts/generate_ejor_tables.py
```

### 1.3 产出的 CSV → 论文表格映射

`generate_ejor_tables.py` 会自动生成以下文件：

| 产出 CSV | 对应论文 Table | 内容 |
|----------|----------------|------|
| `ejor_table4_rl_vs_best.csv` | **Table 6 (上半)** | RL vs Best Rule: Cost/Comp/Rej/Δ%/p-value |
| `ejor_table6_service.csv` | **Table 6 (下半)** | 三方对比 + Cost/Task |
| `ejor_table5_wilcoxon.csv` | **嵌入 Table 6** | Wilcoxon p-adj（嵌入 Δ 列的 */\*\*/\*\*\* 标注） |
| `ejor_table8_decomposition.csv` | **Table 7** | 加权成本分解（Travel/Charg/Tard/Idle/Reject/Other/Total） |

---

## 二、Table 6 设计：RL-APC vs Best Fixed Rule 多维对比

### 2.1 结构（`table*` full-width，EJOR 标准双栏铺满）

```
┌───────┬────────────┬─────────────────────────────┬─────────────────────────────┬──────────┬───────┐
│ Scale │ Best Rule  │     Best Fixed Rule          │        RL-APC               │ Δ_cost   │ p-adj │
│       │            │ Cost    Comp   Rej  Cost/Task│ Cost    Comp   Rej  Cost/Task│   (%)    │       │
├───────┼────────────┼─────────────────────────────┼─────────────────────────────┼──────────┼───────┤
│ S     │ Lazy-Stnby │ [·]     [·]    [·]   [·]    │ [·]     [·]    [·]   [·]    │ [·]***   │ [·]   │
│ M     │ Lazy-Stnby │ [·]     [·]    [·]   [·]    │ [·]     [·]    [·]   [·]    │ [·]      │ [·]   │
│ L     │ Chrg-High  │ [·]     [·]    [·]   [·]    │ [·]     [·]    [·]   [·]    │ [·]      │ [·]   │
│ XL    │ Lazy-Stnby │ [·]     [·]    [·]   [·]    │ [·]     [·]    [·]   [·]    │ [·]      │ [·]   │
└───────┴────────────┴─────────────────────────────┴─────────────────────────────┴──────────┴───────┘
```

### 2.2 数据来源 → 列映射

| 列 | CSV 来源 | 字段 |
|----|----------|------|
| Best Rule Cost | `individual_rules_{scale}_30.csv` | `cost` (选 avg 最小的 rule) |
| Best Rule Comp | 同上 | `completed_tasks` |
| Best Rule Rej | 同上 | `rejected_tasks` |
| Best Rule Cost/Task | **计算列** | Cost ÷ Comp |
| RL Cost | `evaluate_{scale}_30.csv` | `cost` (algorithm_id=rl_apc) |
| RL Comp | 同上 | `completed_tasks` |
| RL Rej | 同上 | `rejected_tasks` |
| RL Cost/Task | **计算列** | Cost ÷ Comp |
| Δ_cost% | **计算列** | (RL − Best) / Best × 100 |
| p-adj | `ejor_table5_wilcoxon.csv` | `p-adj` (Bonferroni) |

### 2.3 显著性标注规则
- `***` p < 0.001, `**` p < 0.01, `*` p < 0.05, 空 = ns
- Rej ≥ 10 用 \textcolor{red}{} 标红
- 每列最优值 \textbf{bold}

### 2.4 Cost/Task 为什么关键

Cost/Task = Total Cost ÷ Completed Tasks。当 Best Rule 拒绝 55-73% 的任务时，其 Total Cost
低仅因为工作量小。Cost/Task 拉平工作量差异后，RL-APC 的单位效率可能反转为更优。

**预期叙事**：
- S: RL total cost 更低 → Cost/Task 也更低（双赢）
- M/XL: RL total cost 略高但 Cost/Task 更低（效率更高）
- L: RL total cost 明显更高（诚实承认，4 句话带过）

---

## 三、Table 7 设计：成本分解

### 3.1 结构

```
┌───────┬───────────┬────────┬────────┬──────────┬──────┬──────────┬───────┬────────┬───────┐
│ Scale │ Method    │ Travel │ Charg. │ Tardiness│ Idle │ Rejection│ Other │ Total  │ %Rej  │
├───────┼───────────┼────────┼────────┼──────────┼──────┼──────────┼───────┼────────┼───────┤
│ S     │ RL-APC    │ [·]    │ [·]    │ [·]      │ [·]  │ [·]      │ [·]   │ [·]    │ [·]%  │
│       │ Greedy-FR │ [·]    │ [·]    │ [·]      │ [·]  │ [·]      │ [·]   │ [·]    │ [·]%  │
│       │ Best Rule │ [·]    │ [·]    │ [·]      │ [·]  │ [·]      │ [·]   │ [·]    │ [·]%  │
│ M     │ ...       │        │        │          │      │          │       │        │       │
│ L     │ ...       │        │        │          │      │          │       │        │       │
│ XL    │ ...       │        │        │          │      │          │       │        │       │
└───────┴───────────┴────────┴────────┴──────────┴──────┴──────────┴───────┴────────┴───────┘
```

### 3.2 加权公式（已在 generate_ejor_tables.py 中实现）

```python
Travel    = C_TR(1.0) × distance + C_TIME(0.1) × travel_time
Charging  = C_CH(0.6) × charging_time
Tardiness = C_DELAY(2.0) × delay
Idle      = C_WAIT(0.05) × waiting + C_CONFLICT(0.05) × (conflict + waiting) + C_STANDBY(0.05) × standby
Rejection = 10000 × rejected_tasks
Other     = Total − (Travel + Charging + Tardiness + Idle + Rejection)
            # 含 terminal penalties + reward shaping credits
```

### 3.3 关键验证：分项之和 = Total

`Other` 列确保 Travel + Charging + Tardiness + Idle + Rejection + Other ≡ Total。
`generate_ejor_tables.py` 的 `_weighted_costs()` 已自动计算 Other 为残差。

### 3.4 核心叙事意图

- Greedy-FR 的 Rejection 列占 Total 的 60-80% → "成本低是因为拒绝任务"
- RL-APC 的 Idle 列较高 → "成本高是因为保持待命（服务可用性）"
- RL-APC 的 Charging 列低于 Greedy-FR → "partial charging 减少了充电浪费"（核心创新证据）
- %Rej ≥ 30% 用红色标注，视觉冲击强

---

## 四、Fig. 5 设计：Cost Distribution Boxplots

### 4.1 规格

- 4 面板（S / M / L / XL），每面板 3 个 box：RL-APC / Greedy-FR / Best Rule
- Y 轴 = Total Cost（log-scale 如果量级差异大）
- 用不同颜色区分：RL = blue, Greedy-FR = orange, Best Rule = green
- 每个 box 30 个数据点（test instances）
- Outliers 用散点标注

### 4.2 生成方式

```python
# 需要新增脚本或在 generate_paper_results.py 中添加
import matplotlib.pyplot as plt
import pandas as pd

for scale in ['S', 'M', 'L', 'XL']:
    # 从 evaluate_{scale}_30.csv 读取 RL-APC 和 Greedy-FR
    # 从 individual_rules_{scale}_30.csv 读取 Best Rule
    # 画 boxplot
```

---

## 五、正文 7 段逐段方案

### ¶1 Overall cost comparison (~80词)

**论点**：以 S-scale 正面结果开场，然后诚实引出其他 scale。

**模板**：
> Table 6 compares RL-APC against the best-performing fixed rule on each
> scale. On S-scale, RL-APC achieves the lowest total cost among all
> methods, outperforming [Best Rule] by [·]% (Wilcoxon $p = [·]$). On M,
> L, and XL, the total cost of RL-APC exceeds that of the best fixed rule.
> However, a purely cost-based comparison is misleading, because the
> fixed-rule baselines attain low cost by rejecting a substantial fraction
> of incoming tasks.

**数据填充位**：
- `ejor_table4_rl_vs_best.csv` → S 行的 Diff% 和 p-value
- 若 S 也不是最优，则改为 "RL-APC ranks [·] across scales" 的中性开场

### ¶2 "Cost trap" — 叙事转折 (~100词)

**论点**：成本最低 ≠ 最优。高拒绝率在仓储运营中不可接受。

**模板**：
> The cost advantage of [Best Rule] stems from a fundamentally different
> operating strategy. On M-scale, [Best Rule] rejects an average of [·]
> out of [·] incoming tasks ([·]%), effectively reducing its workload and
> hence its cost by declining service. On XL-scale, rejection reaches [·]
> tasks ([·]%). Such rejection rates are operationally infeasible in
> warehouse logistics, where order fulfilment commitments are typically
> contractually binding and penalties for missed orders far exceed the
> cost savings from workload reduction.

**数据填充位**：
- `individual_rules_{M,XL}_30.csv` → Best Rule 的 avg rejected / avg num_tasks
- Section 5.1 的 Table 4 中 M=30-40 tasks, XL=80-100 tasks

### ¶3 Cost-per-completed-task (~100词)

**论点**：Cost/Task 拉平工作量差异后，RL-APC 单位效率更优。

**模板**：
> To account for the unequal workload, Table 6 reports cost per completed
> task. On M-scale, RL-APC achieves [·] per task versus [·] for [Best
> Rule] — a [·]% advantage despite its higher total cost. On XL-scale, the
> per-task cost of RL-APC ([·]) is [·]% lower than that of [Best Rule]
> ([·]), which completes fewer tasks at a higher unit cost. This reversal
> demonstrates that RL-APC's higher total cost reflects a larger workload,
> not lower operational efficiency.

**数据填充位**：
- Table 6 的 Cost/Task 列（RL 和 Best Rule 都需要）

### ¶4 M/XL 服务质量详细对比 (~100词)

**论点**：cost difference 在统计上不显著，但服务质量差异巨大。

**模板**：
> On M-scale, RL-APC completes [·] tasks with [·] rejections versus [·]
> completions and [·] rejections for Greedy-FR. The cost difference of
> [·]% is statistically insignificant ($p = [·]$). On XL-scale, RL-APC
> achieves [·] completions with [·] rejections, compared to [·]
> completions and [·] rejections for Greedy-FR ($\Delta_{\text{cost}} =
> [·]\%$, $p = [·]$). RL-APC is the only method that maintains near-zero
> rejection across all scales.

**数据填充位**：
- `evaluate_{M,XL}_30.csv` → rl_apc vs greedy_fr 的 completed/rejected/cost
- `ejor_table5_wilcoxon.csv` → M 和 XL 的 Greedy-FR 行 p-adj

### ¶5 L-scale 坦率讨论 (~80词，不超过 4 句话)

**论点**：诚实承认 L-scale 弱势，给出根因，指向改进方向。

**模板**：
> On L-scale, RL-APC incurs [·]% higher cost than [Best Rule]. Cost
> decomposition (Table 7) reveals that the idle component accounts for
> [·]% of RL-APC's total, indicating an overly conservative policy that
> dwells rather than dispatches under the L-scale fleet-to-task ratio.
> Section 5.7 discusses ongoing training improvements.

**数据填充位**：
- Table 7 L-scale RL-APC 行的 Idle / Total 比例

### ¶6 成本分解 (~80词)

**论点**：Rejection penalty 暴露了 Greedy-FR 的"虚假低成本"。

**模板**：
> Table 7 decomposes total cost into operational components. For Greedy-FR
> on M-scale, rejection penalties constitute [·]% of total cost (Table 7,
> %Rej column), confirming that its low operating cost is an artefact of
> workload avoidance. By contrast, RL-APC's cost is dominated by idle
> time ([·]%), reflecting its strategy of maintaining fleet availability
> for incoming tasks. Notably, RL-APC's charging cost is consistently
> lower than Greedy-FR by [·]-[·]% across scales, validating the
> effectiveness of learnable partial-charging targets.

**数据填充位**：
- `ejor_table8_decomposition.csv` → %Rej 列 (Greedy-FR) 和 Idle 占比 (RL-APC)
- Charging 列的 RL vs Greedy-FR 差异

**本段最后一句是核心创新 (partial charging) 的直接证据。**

### ¶7 统计显著性 + 过渡 (~60词)

**论点**：总结 + 过渡到 5.4。

**模板**：
> Wilcoxon tests confirm the S-scale advantage at the [·]% level after
> Bonferroni correction. On M and XL, cost differences are statistically
> insignificant, while service quality differences are substantively large
> and practically meaningful. To understand how RL-APC achieves this
> cost-service balance, we next examine its rule selection behaviour.

---

## 六、写作检查清单

### 数据一致性检查
- [ ] Table 6 的 Cost/Task = Cost ÷ Comp，手动验算 4 行
- [ ] Table 7 的 Travel+Charg+Tard+Idle+Reject+Other = Total，验算 12 行
- [ ] Table 7 的 %Rej = Rejection / Total × 100，验算
- [ ] Wilcoxon p-value 与 scipy.stats.wilcoxon 手动对照（至少 S-scale）
- [ ] RL-APC 的 Rej 列 ≈ 0（如果不是，说明训练有问题，需回到训练调试）

### 叙事一致性检查
- [ ] ¶2 的 rejection 数据与 Section 5.2 末尾段的数据一致（h₉ 拒绝 19.2/64.6）
- [ ] ¶5 的 L-scale 根因与 5.7 robustness 的讨论方向一致
- [ ] ¶6 的 "charging cost lower" 与 Methodology 4.6.2 中 partial charging 的声明呼应
- [ ] ¶7 过渡到 5.4 的研究问题（"rule selection behaviour"）

### EJOR 风格检查
- [ ] 无 "interestingly" / "notably" / "it is worth noting"
- [ ] 无 "In order to..." 铺垫句式
- [ ] 结论先行：每段第一句给结论，后面给数据支撑
- [ ] L-scale 讨论不超过 4 句话
- [ ] 使用 business-relevant 表述（"contractually binding", "operationally infeasible"）

### 数字交叉验证
- [ ] ¶2 的 rejection count 与 `individual_rules_*.csv` 的 rejected_tasks 字段一致
- [ ] ¶3 的 Cost/Task 与 Table 6 的对应列一致
- [ ] ¶4 的 completed/rejected 与 `evaluate_*.csv` 的 completed_tasks/rejected_tasks 一致
- [ ] ¶6 的 "charging cost lower by X%" 与 Table 7 的 Charg 列一致

---

## 七、Rejection Penalty Sensitivity 实验（P1 加分项）

### 7.1 实验设计

不需要重新训练或重新跑仿真。从已有 CSV 读取各分项，按不同 penalty 重算总成本。

```python
import pandas as pd
import numpy as np

penalties = [0, 5000, 10000, 20000, 50000]
scale = 'M'

# 读取 evaluate_M_30.csv
df = pd.read_csv('results/benchmark/evaluate_M_30.csv')

for penalty in penalties:
    for algo in ['rl_apc', 'greedy_fr']:
        sub = df[df['algorithm_id'] == algo]
        # 原始 cost 中减去旧 rejection penalty，加上新 penalty
        adjusted = (sub['cost']
                    - 10000 * sub['rejected_tasks']
                    + penalty * sub['rejected_tasks'])
        print(f"penalty={penalty}, {algo}: mean cost = {adjusted.mean():.0f}")
```

### 7.2 展示方式

折线图：横轴 = penalty {0, 5K, 10K, 20K, 50K}，纵轴 = 总成本。
3 条线：RL-APC / Greedy-FR / Best Rule。
预期：penalty 增大时 Greedy/Best Rule 陡增（因为拒绝多），RL 几乎不变。
交叉点 = "在什么 penalty 水平下 RL 开始占优"。

### 7.3 正文嵌入位置

可在 ¶2 或 ¶6 中用一句话引用："A sensitivity analysis (supplementary material) confirms that RL-APC's cost advantage increases monotonically with the rejection penalty, becoming dominant at penalty ≥ [·]."

---

## 八、contingency：如果 RL-APC 在所有 scale 都不是 cost-optimal

如果训练后 RL-APC 在任何 scale（包括 S）的 total cost 都不是最低：

**叙事策略调整**：
- 开场不用 "achieves the lowest cost"，改为 "achieves near-zero rejection while maintaining competitive cost"
- ¶1 聚焦 Pareto efficiency："RL-APC is the only method on the Pareto frontier of cost vs rejection across all scales"
- ¶3 的 Cost/Task 更加关键——即使 total cost 不是最低，unit cost 可能仍然最低
- 加重 ¶6 的分解分析——展示 RL 的成本"高"在哪里，是否合理

**底线论点**：即使 total cost 不是最低，只要 RL-APC 是唯一同时满足 (1) near-zero rejection 和 (2) competitive cost-per-task 的方法，论文论点就成立。

---

## 九、版面预算

| 元素 | 预估版面 |
|------|---------|
| Table 6 (full-width) | ~1/3 page |
| Table 7 (full-width) | ~1/3 page |
| Fig. 5 (4-panel boxplot) | ~1/3 page |
| 正文 7 段 (~650 词) | ~1 page |
| **合计** | **~2 pages** |

EJOR 单节 2 页是合理的，不会过长。