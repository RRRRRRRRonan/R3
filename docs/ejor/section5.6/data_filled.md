# Section 5.6: Robustness and Sensitivity — Data-Filled Draft

> Updated 2026-03-13, Option B clean cost (Total = Oper + Reject + Terminal)

---

## P1: Cross-Scale Statistical Significance

Table 4 summarises the comparison between RL-APC and the best-performing
dispatching rule on each scale. RL-APC reduces total cost by 70.8% on S
(p = 1.86 × 10⁻⁸), 76.0% on M (p = 1.86 × 10⁻⁹), 54.5% on L
(p = 7.99 × 10⁻⁶), and 80.5% on XL (p = 1.86 × 10⁻⁹), all statistically
significant at the 0.1% level after Holm correction. Table 5 extends the
analysis to all six baselines per scale (24 comparisons total); every adjusted
p-value remains below 5 × 10⁻⁵, confirming that the improvement is not
an artefact of a favourable baseline choice. Instance-level win/loss ratios
range from 25/5 (L vs Charge-High) to 30/0 (M and XL vs several baselines),
indicating consistent superiority rather than a few large outliers pulling
the mean.

---

## P2: Training Stability and Multi-Seed Replication

Figure~\ref{fig:training-curves} plots the learning curves for all four
scales. On S and M, the best evaluation return is reached within 15%
of the training budget, and the policy remains stable thereafter with
decreasing variance. The early convergence on M (best checkpoint at
approximately 150K of 1M steps) reflects the moderate state-action
space of this scale; continued training explores but does not improve
upon the best policy, so the held-out validation checkpoint is used
rather than the final weights. L and XL require a larger fraction
of the budget (35--40%) to converge, consistent with the higher
combinatorial complexity of these instances, but both ultimately
stabilise above the Greedy-FR baseline. The best checkpoint is selected
via held-out validation and deployed for testing.
Figure~\ref{fig:cost-boxplots} shows the distribution of total cost
(Option B) across 30 test instances; on all four scales, RL-APC's
interquartile range is tight and well separated from the baselines,
indicating both low variance and consistent dominance. Multi-seed
experiments on S-scale (seeds 42, 43, 44) confirm that different random
initialisations converge to comparable policies.

---

## P3: Hyper-Parameter Sensitivity (L-Scale)

Table 10 reports the effect of architecture and terminal-penalty choices on
L-scale performance. Configuration v1 ([256, 128] network, penalty = 3,000
per unfinished task) achieves a cost of 103,180, beating the Greedy-FR
baseline (234,316) by 56.0%. Configuration v2 ([512, 256], penalty = 2,000)
with an identical training budget yields 251,495 (+7.3% vs Greedy) — the
only setting where RL-APC fails to outperform the baseline. However,
configuration v3, which uses the same architecture and penalty as v2 but
extends training, recovers to 101,616 (−54.5%), demonstrating that the
v2 failure is attributable to under-training rather than a fundamental
architecture mismatch. Two key observations emerge: (i) the terminal
penalty is the most influential hyper-parameter, as it directly shapes
the incentive for task acceptance versus rejection; (ii) given sufficient
training, both network architectures deliver strong results.

---

## P4: Computational Cost

Table 9 reports wall-clock runtimes for a single simulation episode.
RL-APC requires 3.30 s (S), 5.23 s (M), 11.54 s (L), and 4.24 s (XL),
compared with 0.29–1.59 s for Greedy-FR — a factor of 3–11× overhead.
However, RL remains faster than the offline ALNS heuristic on M
(5.23 s vs 7.87 s for ALNS-PR) and substantially faster on XL
(4.24 s vs 45.93 s). The RL runtime includes full simulation overhead
(event processing, state updates); the neural-network inference itself
executes in sub-millisecond time per decision point, making RL-APC
suitable for real-time dispatching even on the largest instances tested.
