# Figure: Cost Distribution Boxplots — Template

> Updated 2026-03-13, Option B clean cost
> L-scale green box = Standby-Lazy (not Charge-High, since Charge-High = Greedy-FR on L)

## Figure Description

2×2 panel plot showing Option B total cost distribution across 30 test instances.
Three methods per panel:
- **RL-APC** (red): learned adaptive policy
- **Greedy-FR** (blue): greedy with first-removal
- **Green box** (per scale):
  - S/M/XL: Standby-Lazy (best fixed rule)
  - L: Standby-Lazy (2nd-best; Charge-High = Greedy-FR, already shown in blue)

Diamond = mean; line = median; box = IQR; whiskers = 1.5×IQR.

## Data Summary

| Scale | RL-APC | Greedy-FR | Green Box | Green Box Name |
|-------|--------|-----------|-----------|----------------|
| S | 13,162 | 64,725 | 45,043 | Standby-Lazy |
| M | 48,745 | 233,564 | 202,882 | Standby-Lazy |
| L | 101,616 | 223,112 | 394,979 | Standby-Lazy |
| XL | 130,077 | 606,680 | 667,971 | Standby-Lazy |

## Recommended Caption

```
Total cost distribution (Option B) across 30 test instances. RL-APC (red)
is compared against Greedy-FR (blue) and Standby-Lazy (green). On L-scale,
Charge-High equals Greedy-FR and is shown in blue; Standby-Lazy serves as
the additional reference. Diamonds indicate means; boxes span the IQR.
RL-APC achieves both lower mean cost and tighter variance on all four
scales, with non-overlapping interquartile ranges.
```

## Key Visual Takeaways

- **All 4 panels**: RL-APC red box at bottom, well separated from baselines
- **Tight IQR**: RL-APC variance is consistently narrow → reliable deployment
- **Non-overlapping IQRs**: S, M, XL have no overlap between RL and any baseline
- **L-scale**: RL-APC (100K) clearly below Greedy-FR (223K) and Standby-Lazy (395K)
