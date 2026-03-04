# Q2: RL-APC 的自适应选择有效吗？

> **结论: 部分有效。S-scale 大胜 (-34.4%)，M/L/XL 均输给最优单规则。但 M/XL 的成本差距小且服务质量远优（详见 Q4）。**

## RL-APC vs 每个 Scale 的最优单规则

| Scale | Best Rule | Best Rule Cost | RL-APC Cost | Delta | 判定 |
|-------|-----------|---------------|-------------|-------|------|
| S | Standby-Lazy | 60,731 | 39,846 | **-34.4%** | **RL 大胜** |
| M | Standby-Lazy | 224,316 | 286,665 | +27.8% | RL 输 (成本) |
| L | Charge-High/Opp | 336,386 | 662,267 | +96.9% | RL 大输 |
| XL | Standby-Lazy | 696,148 | 814,344 | +17.0% | RL 输 (成本) |

## RL-APC vs Greedy-FR (=Charge-Opp) 对比

| Scale | Greedy-FR Cost | RL-APC Cost | Delta | 判定 |
|-------|---------------|-------------|-------|------|
| S | 94,772 | 39,846 | **-58.0%** | **RL 大胜** |
| M | 281,349 | 286,665 | +1.9% (ns) | 持平 |
| L | 336,386 | 662,267 | +96.9% | RL 大输 |
| XL | 809,830 | 814,344 | +0.6% (ns) | 持平 |

> 注: L-scale Greedy-FR = Charge-Opp = L-scale 最优单规则

## RL-APC vs 全部 15 规则

### S-Scale: RL 击败所有 15 条规则 ✓

RL-APC (39,846) 低于所有单规则 (最优 Standby-Lazy = 60,731)。
- 对比最优规则: **-34.4%**
- 对比 Greedy-FR (Charge-Opp): **-58.0%**
- **在 S-scale，自适应选择的价值得到充分验证**

### M-Scale: RL 成本不如最优单规则 ✗

RL-APC (286,665) 高于 Standby-Lazy (224,316)，但：
- 成本: RL 输 +27.8%
- vs Greedy-FR: 仅 +1.9% (统计不显著 p=0.271)
- **服务质量**: RL 完成 18.1 / 0 拒绝 vs Greedy 11.5 / 21.8 拒绝 → **RL 服务质量远优**

### L-Scale: RL 成本远差于最优单规则 ✗✗

RL-APC (662,267) 远高于 Charge-High (336,386)：
- 成本: RL 输 +96.9%
- 最优单规则恰好是 Greedy-FR (=Charge-Opp)
- **L-scale 是 RL-APC 的明确弱点**

### XL-Scale: RL 成本略高于最优单规则 ✗

RL-APC (814,344) 高于 Standby-Lazy (696,148)：
- vs 最优单规则: +17.0%
- vs Greedy-FR: 仅 +0.6% (统计不显著 p=0.655)
- **服务质量**: RL 拒绝 1.5 vs Greedy 57.7 → **RL 服务质量远优**
- Greedy-FR (#10, 790,373) 远非最优单规则 (#1 Standby-Lazy = 696,148)

## 关键发现

### 1. 纯成本对比: RL 仅在 S 胜出

如果仅看成本，RL-APC 只在 S-scale 优于最优单规则。这是实验的"坦诚面"。

### 2. 但 Greedy-FR 也不是最优单规则

| Scale | Greedy-FR Rank | Best Rule | Gap to Best |
|-------|---------------|-----------|-------------|
| S | #3 | Standby-Lazy | +38.2% |
| M | #3 | Standby-Lazy | +18.7% |
| L | **#1** | Charge-High/Opp | 0% |
| XL | #10 | Standby-Lazy | +13.5% |

- Greedy-FR 只在 L-scale 恰好是最优（运气好）
- 在其他 3 个 scale，Greedy-FR 都不是最优

### 3. "后见之明"的最优单规则不具有实用性

选择最优单规则需要：
- 事先知道 scale → 不同 scale 最优规则不同
- 事先知道 instance → 同 scale 内也有实例间差异
- **在真实在线场景中，不可能事先选对最优单规则**

### 4. RL vs Greedy-FR 的对比更公平

两者都是在线方法，无需事先知道最优规则：
- S: RL 大胜 (-58.0%)
- M: 持平 (+1.9%) + 服务质量远优
- L: RL 大输 (+96.9%)
- XL: 持平 (+0.6%) + 服务质量远优

## 论文叙事建议

**不能说**: "RL-APC 在成本上优于所有 baseline"（仅 S 成立 vs best rule）

**可以说**:
1. "RL-APC 是唯一在 S-scale 超越所有 15 条单规则的方法 (-34.4%)"
2. "与同为在线方法的 Greedy-FR 相比，RL-APC 在 S 大胜 (-58%)，M/XL 持平，L 需改进"
3. "在 M/XL，RL-APC 以可接受的成本 (+1.9%/+0.6%) 换取了极大的服务质量提升（见 Q4）"
4. "没有任何单规则能在所有 scale 上都最优，而 RL-APC 提供了无需人工选择的自适应机制"
5. "L-scale 的 failure case 被诚实讨论并提供了改进路径（见 Q6）"

## 对应表格

- T3: 15 规则 × 4 scale 全表 (Section 5.2)
- T4: RL-APC vs best-per-scale (Section 5.3)
- F1: 柱状图 (Section 5.2)
