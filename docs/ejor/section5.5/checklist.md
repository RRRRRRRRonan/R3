# Section 5.5 Writing Checklist

> Section: Online-Offline Performance Gap (Q5)
> Cost metric: Option B clean cost for online methods; raw cost for ALNS

## Data Consistency Checks

- [x] S RL-APC = 13,162 — matches Table 6 ✅
- [x] M RL-APC = 48,745 — matches Table 6 ✅
- [x] XL RL-APC = 130,077 — matches Table 6 ✅
- [x] S Greedy-FR = 64,725 — matches Table 8 ✅
- [x] M Greedy-FR = 233,564 — matches Table 8 ✅
- [x] XL Greedy-FR = 606,680 — matches Table 8 ✅
- [x] S ALNS-PR = 5,282 — verified from evaluate_S_30.csv ✅
- [x] M ALNS-PR = 53,600 — verified from evaluate_M_synced_30.csv ✅
- [x] XL ALNS-PR = 170,514 — verified from evaluate_XL_synced_30.csv ✅
- [x] Gap calculation: S RL = (13162-5282)/5282 = +149.2% ✅
- [x] Gap calculation: M RL = (48745-53600)/53600 = -9.1% ✅
- [x] Gap calculation: XL RL = (130077-170514)/170514 = -23.7% ✅
- [x] ALNS cost/task: S=308, M=1542, XL=1900 — verified ✅
- [x] RL cost/task matches Table 6: S=908, M=2694, XL=5859 ✅
- [x] Gap reduction on S: (1125-149)/1125 = 86.8% ≈ 87% ✅

## Narrative Consistency Checks

- [x] ¶1 S-scale gap consistent with 5.3 Option B results
- [x] ¶2 M/XL reversal consistent with 5.3 Table 8 terminal dominance
- [x] ¶2 "selective acceptance" consistent with 5.4 Accept-Value finding
- [ ] ¶3 transition to 5.6 — needs cross-check when 5.6 written
- [x] L-scale marked "—" (no ALNS data for v3/v4 instances)

## EJOR Style Checks

- [ ] No "interestingly" / "notably" / "it is worth noting"
- [ ] Conclusion-first: each paragraph leads with its claim
- [ ] ≤ 300 words total
- [ ] L-scale not discussed (just "—" in table)
- [ ] ALNS described as "heuristic", not "optimal" or "lower bound"

## Files in docs/ejor/section5.5/

| File | Description |
|------|-------------|
| `outline.md` | Writing outline with full data and narrative plan |
| `data_filled.md` | All paragraphs with data filled in |
| `checklist.md` | This file |
| `ejor_table7_offline_gap.csv` | Table 7 data (Option B, from results/paper/) |
| `alns_detail.csv` | ALNS-FR vs ALNS-PR per-scale detail |
| `per_task_efficiency.csv` | ALNS vs RL vs Greedy cost/task comparison |

## Figures

None required — Section 5.5 uses only Table 7 + text.
