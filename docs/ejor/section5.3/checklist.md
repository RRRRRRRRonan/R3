# Section 5.3 Writing Checklist (Verified 2026-03-05)

## Data Consistency Checks

- [x] Table 6 Cost/Task = Cost / Comp — verified 4 rows (S:39846/14.5=2748, M:286665/18.1=15867, L:662267/17.5=37916, XL:814344/22.2=36737)
- [x] Table 7 Travel+Charg+Tard+Idle+Reject+Terminal+Shaping = Total — verified 11 rows (max rounding error = 2)
- [x] Table 7 Rej% = Rejection/Total×100 — verified
- [x] Wilcoxon p-adj matches scipy output — S vs Standby-Lazy: p-raw=5.38e-03, p-adj=3.23e-02 (Bonferroni ×6)
- [x] Table 4 p-raw == Table 5 p-raw for all scales (verified after seed-pairing fix)

## Narrative Consistency Checks

- [x] ¶2 rejection data (Standby-Lazy: M=19.2, XL=64.6) — from ejor_table4_rl_vs_best.csv
- [ ] ¶5 L-scale root cause consistent with 5.7 robustness discussion — needs cross-check when 5.7 written
- [x] ¶6 "charging cost lower" — verified: RL charging < Greedy on all 4 scales (S:-93%, M:-41%, L:-79%, XL:-23%)
- [x] ¶7 transition to 5.4 "rule selection behaviour" — matches Q3 content in 03_Q3_rule_selection.md

## EJOR Style Checks

- [ ] No "interestingly" / "notably" / "it is worth noting"
- [ ] No "In order to..." filler
- [ ] Conclusion-first: each paragraph leads with its claim
- [ ] L-scale discussion ≤ 4 sentences
- [ ] Business-relevant framing ("contractually binding", "operationally infeasible")

## Numbers Cross-Validation

- [x] ¶2 rejection count matches individual_rules CSVs (Standby-Lazy M=19.2, XL=64.6)
- [x] ¶3 Cost/Task matches Table 6 (S:2748/4911, M:15867/16294, XL:36737/37029)
- [x] ¶4 completed/rejected matches evaluate CSVs (M: RL 18.1/0, GR 11.5/21.8; XL: RL 22.2/1.5, GR 20.0/57.7)
- [x] ¶6 charging difference matches Table 7 (M: RL 199 vs GR 336 = -41%; XL: RL 392 vs GR 509 = -23%)

## Files in docs/ejor/section5.3/

| File | Description |
|------|-------------|
| `outline.md` | Original writing outline (from section5.3.md) |
| `data_filled.md` | All paragraphs with data filled in, narrative corrections noted |
| `checklist.md` | This file |
| `ejor_table4_rl_vs_best.csv` | Table 6 upper: RL vs Best Rule |
| `ejor_table5_wilcoxon.csv` | Wilcoxon p-adj values |
| `ejor_table6_service.csv` | Table 6 lower: 3-way service quality + Cost/Task |
| `ejor_table8_decomposition.csv` | Table 7: cost decomposition with Terminal + Shaping |
| `fig_cost_boxplots.png` | Fig. 5: 4-panel boxplot (RL vs Greedy vs top-3 rules) |
| `fig_cost_boxplots.pdf` | Fig. 5 PDF version |
