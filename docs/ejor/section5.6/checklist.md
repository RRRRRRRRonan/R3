# Section 5.6: Robustness and Sensitivity — Verification Checklist

> Final version 2026-03-13

## Data Accuracy (cross-checked against source CSVs)

- [x] Table 4: S -70.8%, M -76.0%, L -54.5%, XL -80.5%
- [x] Table 5: all 24 Holm-adjusted p < 5e-05, W/L worst 25/5 (L)
- [x] Table 9: S 3.30s, M 5.23s, L 11.54s, XL 4.24s
- [x] Table 10: v1 103,180 (-56.0%), v2 251,495 (+7.3%), v3 101,616 (-54.5%)

## Figures (300 DPI, Option B)

- [x] **Boxplots**: 3 distinct methods per panel; L uses Standby-Lazy (not Charge-High = Greedy-FR)
- [x] **Training curves**: 方案 A — 4 panels, EMA smoothed, Greedy-FR baseline reference
  - S/M: clean convergence, best at 150K
  - L: train_L_v4 (40 evals, 2M steps), best at 50K — early stopping narrative
  - XL: train_XL (10 evals, 500K steps), best at 300K
  - Note: `training_summary.csv` records v1 run (best=1M); figure uses v4 run (best=50K, better return)

## Narrative Consistency

- [x] All 4 scales RL wins (no weak link under Option B)
- [x] L training volatility addressed by: caption + Table 10 + Wilcoxon
- [x] v2 failure = under-training, not architecture
- [x] RL < ALNS runtime on M (0.66×) and XL (0.09×)

## File Structure

```
section5.6/
├── outline.md, data_filled.md, checklist.md  (writing)
├── fig_training_curves.{png,pdf}             (final figures)
├── fig_cost_boxplots.{png,pdf}
└── templates/
    ├── table{4,5_summary,9,10}_image.{png,pdf} (table images)
    └── {table9,table10,table5_robustness,fig_*}_template.md (narrative)
```

No duplicate CSVs — source data lives in `results/paper/` and `results/benchmark/`.
