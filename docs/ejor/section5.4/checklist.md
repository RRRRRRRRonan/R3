# Section 5.4 Writing Checklist

> Section: Policy Interpretability — Rule Selection Behaviour (Q3)
> Goal: "RL-APC 是怎么赢的？"

## Data Consistency Checks

- [x] Two-rule sum: S=76.7%, M=94.5%, L=86.5%, XL=84.1% — verified from rule_freq_*.csv
- [x] Dispatch rules all < 0.5% per scale — verified (max: EDD on XL = 0.5%)
- [x] TASK_ARRIVAL → Accept-Value dominant: S=47.2%, M=79.7%, L=59.7%, XL=73.5%
- [x] DEADLOCK_RISK → Charge-Opp dominant: S=52.0%, M=86.0%, L=58.8%, XL=76.0%
- [x] ROBOT_IDLE: M Standby-Lazy=71.4%, L=66.2%, S Charge-Opp=45.4%, XL=59.0%

## Narrative Consistency Checks

- [x] ¶1 "two rules dominate" — supported by heatmap data
- [x] ¶2 S/XL vs M/L pattern — matches fleet-to-task ratio argument
- [x] ¶3 Accept-Value on task arrival — explains 5.3's near-zero rejection
- [x] ¶3 Charge-Opp on deadlock — explains 5.3's low charging cost
- [ ] ¶4 transition to 5.5 "online-offline gap" — needs cross-check when 5.5 written

## Cross-Reference with Section 5.3

- [x] 5.3 charging cost -23% to -93% → 5.4 Charge-Opp is core strategy
- [x] 5.3 near-zero rejection → 5.4 Accept-Value used on task arrival
- [x] 5.3 lowest Oper on 3/4 scales → 5.4 Standby-Lazy = zero operational cost when idle

## EJOR Style Checks

- [ ] No "interestingly" / "notably" / "it is worth noting"
- [ ] Conclusion-first: each paragraph leads with its claim
- [ ] Do NOT analyse each rule individually (focus on 3 findings)
- [ ] Proactively address "only 2 rules used" concern
- [ ] ≤ 400 words total

## Files in docs/ejor/section5.4/

| File | Description |
|------|-------------|
| `outline.md` | Writing outline with full data |
| `data_filled.md` | All paragraphs with data filled in |
| `checklist.md` | This file |
| `event_breakdown_summary.csv` | Event-type × scale summary table |

## Figures

| Figure | File | Status |
|--------|------|--------|
| F3: Rule Selection Heatmap | `results/paper/fig_rule_selection_heatmap.png` | ✅ Ready |
| Event Breakdown (optional) | — | ❌ Not yet generated (P1) |
