# Section 5.6: Robustness and Sensitivity — Data-Filled Draft

> Updated 2026-03-16, Option B clean cost (Total = Oper + Reject + Terminal)

---

## P1: Cross-Scale Statistical Significance

As established in Section 5.3, RL-APC achieves statistically significant
cost reductions across all four scales (54–81%, all p < 10⁻⁵ after Holm
correction; Tables 4–5). All 24 pairwise comparisons against the six
baselines reject the null hypothesis at the 0.1% level, and instance-level
win/loss ratios confirm consistent dominance rather than outlier-driven
gains.

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
initialisations converge to comparable policies. Supplementary L-scale
experiments with varied network architectures and terminal penalties
show that under-training, rather than architecture choice, is the
primary risk factor; given sufficient budget, both [256,128] and
[512,256] networks converge to comparable performance.

---

## P3: Computational Cost

Table 9 reports wall-clock runtimes for a single simulation episode.
RL-APC requires 3.30 s (S), 5.23 s (M), 11.54 s (L), and 4.24 s (XL),
compared with 0.29–1.59 s for Greedy-FR — a factor of 3–11× overhead.
However, RL remains faster than the offline ALNS heuristic on M
(5.23 s vs 7.87 s for ALNS-PR) and substantially faster on XL
(4.24 s vs 45.93 s). The RL runtime includes full simulation overhead
(event processing, state updates); the neural-network inference itself
executes in sub-millisecond time per decision point, making RL-APC
suitable for real-time dispatching even on the largest instances tested.

---

## P4: Ablation Study (M-Scale)

To isolate the contribution of each architectural component, we retrain
two ablation variants on M-scale with identical hyperparameters and
training budget (1M steps): RL-APC-PC removes partial-charging
discretisation (all charging events use full recharge), and RL-APC-FM
disables feasibility masking (all 15 rules are selectable regardless of
precondition satisfaction).

Removing partial charging increases total cost by 41.6% (from 48,745 to
69,008) and reduces completed tasks from 18.10 to 11.33 per episode.
The agent retains its zero-rejection strategy but loses the ability to
minimise charging time through opportunistic partial top-ups: full
recharge cycles consume time that would otherwise be spent on task
execution, resulting in 23.4 unfinished tasks per episode (terminal
penalty = 58,583).

Disabling feasibility masking produces a more severe outcome: RL-APC-FM
completes zero tasks across all 30 test instances, achieving only 106
decision steps per episode on average (compared with ~1,900 for the PC
variant and ~2,000 for the full model). Without the action mask
filtering infeasible choices before the policy network evaluates them,
the agent cannot distinguish productive from unproductive rules in the
15-action combinatorial space. Training entropy remains high throughout
(−2.44 nats), indicating that the policy never converges to a
meaningful selection strategy. This result demonstrates that feasibility
masking is not merely a safety constraint but a prerequisite for
learning: it reduces the effective action space at each decision point
to the subset of rules whose preconditions are satisfied, transforming
an intractable exploration problem into one the RL agent can solve.

For reference, uniform random rule selection achieves a cost of 156,522
(+221% vs Full; 9.8 completions, 11.5 rejections). Even the degenerate
FM policy (86,917) incurs less cost than Random because it avoids
explicit task rejections, but this is a vacuous advantage — it simply
does nothing rather than doing something poorly.
