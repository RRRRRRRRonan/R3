# Section 5.3 Writing Checklist (Updated 2026-03-06 — Option B)

> Option B: Total = Oper + Reject + Terminal (no reward shaping in total)

## Data Consistency Checks

- [x] Table 6 Cost/Task = Cost / Comp — verified 4 rows (S:13162/14.5=908, M:48745/18.1=2694, L:101616/17.5=5807, XL:130077/22.2=5859)
- [x] Table 7 Travel+Charg+Tard+Idle = Oper, Oper+Reject+Terminal = Total — verified 11 rows
- [x] Table 7 Rej% = Rejection/Total×100 — verified
- [x] Wilcoxon p-adj matches scipy output — all 4 scales p < 0.001 (Bonferroni ×4)
- [x] Table 4 p-raw == Table 5 p-raw for all scales

## Option B Key Results

| Scale | RL-APC | Best Rule | Best Cost | Diff% | p-value | Sig |
|-------|--------|-----------|-----------|-------|---------|-----|
| S | 13,162 | Standby-Lazy | 45,043 | -70.8% | 1.86e-08 | *** |
| M | 48,745 | Standby-Lazy | 202,882 | -76.0% | 1.86e-09 | *** |
| L | 101,616 | Charge-High | 223,112 | -54.5% | 7.99e-06 | *** |
| XL | 130,077 | Standby-Lazy | 667,971 | -80.5% | 1.86e-09 | *** |

## Narrative Consistency Checks

- [x] ¶1 RL wins ALL 4 scales (-54.5% to -80.5%), all p < 0.001
- [x] ¶2 rejection data (Standby-Lazy: M=19.2, XL=64.6) — from ejor_table4_rl_vs_best.csv
- [x] ¶3 Cost/Task: S:908/3633, M:2694/14702, L:5807/7339, XL:5859/35530
- [ ] ¶5 L-scale root cause consistent with 5.7 robustness discussion — needs cross-check when 5.7 written
- [x] ¶6 charging cost lower — verified: S:-93%, M:-41%, L:-79%, XL:-23%
- [x] ¶7 transition to 5.4 "rule selection behaviour" — matches Q3 content

## EJOR Style Checks

- [ ] No "interestingly" / "notably" / "it is worth noting"
- [ ] No "In order to..." filler
- [ ] Conclusion-first: each paragraph leads with its claim
- [ ] L-scale discussion ≤ 3 sentences
- [ ] Business-relevant framing ("contractually binding", "operationally infeasible")

## Numbers Cross-Validation

- [x] ¶2 rejection count matches individual_rules CSVs (Standby-Lazy M=19.2, XL=64.6)
- [x] ¶3 Cost/Task matches Table 6 (S:908/3633, M:2694/14702, L:5807/7339, XL:5859/35530)
- [x] ¶4 completed/rejected matches evaluate CSVs (M: RL 18.1/0, GR 11.5/21.8; XL: RL 22.2/1.5, GR 20.0/57.7)
- [x] ¶6 charging difference matches Table 7 (M: RL 199 vs GR 336 = -41%; XL: RL 392 vs GR 509 = -23%)

## Files in docs/ejor/section5.3/

| File | Description |
|------|-------------|
| `outline.md` | Writing outline with Option B data |
| `data_filled.md` | All paragraphs with Option B data filled in |
| `checklist.md` | This file |
| `table7_template.md` | Table 7 LaTeX template (Option B: Oper+Reject+Terminal) |
| `table7_cost_decomposition.png` | Table 7 image (Option B) |
| `table7_cost_decomposition.pdf` | Table 7 PDF version |
| `fig_cost_boxplots.png` | Fig. 5: 4-panel boxplot (Option B clean cost) |
| `fig_cost_boxplots.pdf` | Fig. 5 PDF version |
