# Table: Ablation Results (M-Scale, 30 Test Instances)

> RL-APC-PC and RL-APC-FM rows are from real training + evaluation.
> Source models: `results/rl/train_M_ablation_PC/`, `results/rl/train_M_ablation_FM/`
> Source CSVs: `results/benchmark/evaluate_M_ablation_pc_30.csv`, `evaluate_M_ablation_fm_30.csv`

## Data

| Variant | Description | M Cost | Completed | Rejected | Delta vs Full |
|---------|-------------|--------|-----------|----------|---------------|
| RL-APC (Full) | Complete model | 48,745 | 18.10 | 0.00 | -- |
| RL-APC-PC | No partial charging | 69,008 | 11.33 | 0.00 | +41.6% |
| RL-APC-FM | No feasibility masking | 86,917 | 0.00 | 0.00 | +78.3% |
| Random | Uniform random rule selection | 156,522 | 9.80 | 11.50 | +221.1% |

## Data Provenance

**RL-APC (Full)** — Real data from Section 5.3 (Table 4).

**RL-APC-PC** — Real data from `results/benchmark/evaluate_M_ablation_pc_30.csv`:
- Model: `results/rl/train_M_ablation_PC/best_model/` (best checkpoint at 650K of 1M steps)
- Training: `--no-partial-charging` → `charge_level_ratios=(1.0,)`, all other hyperparameters identical to Full
- 30/30 instances OK, all terminated at `max_time`
- Cost breakdown: Oper 10,424.5 + Reject 0.0 + Terminal 58,583.3 = **69,007.8** (std 7,968.3)
- Completed 11.33 tasks (vs 18.10 for Full) — removing partial charging wastes time on full-recharge cycles
- Zero rejections: task acceptance strategy preserved, but fewer tasks completed in time
- Steps per episode: mean 1,879 (normal range, agent is functional)

**RL-APC-FM** — Real data from `results/benchmark/evaluate_M_ablation_fm_30.csv`:
- Model: `results/rl/train_M_ablation_FM/best_model/` (best checkpoint at 500K of 1M steps)
- Training: `--no-feasibility-mask` → `disable_feasibility_mask=True`, all other hyperparameters identical to Full
- 30/30 instances OK; 26 terminated at `max_time`, 4 at `no_progress`
- Cost breakdown: Oper 0.0 + Reject 0.0 + Terminal 86,916.7 = **86,916.7** (std 7,242.2)
- **Zero completions, zero rejections** — agent is completely non-functional
- Steps per episode: mean 106 (vs 1,879 for PC, vs ~2,000+ for Full) — policy collapses
- Training entropy remained high (-2.44) throughout, indicating the policy never converged
- Interpretation: without feasibility masking, the RL agent cannot distinguish feasible from
  infeasible actions in a 15-rule action space. The resulting policy is degenerate — it selects
  rules whose preconditions are not met, producing no-op actions that fail to advance the
  simulation. Feasibility masking is not an optimisation but a **prerequisite** for learning.

**Random** — Real data from `results/benchmark/evaluate_M_synced_30.csv`:
- Cost breakdown: Oper 7,855 + Rejection 115,000 + Terminal 33,667 = **156,522** (std 26,599)
- Completed 9.80, Rejected 11.50
- Random is cheaper than fixed rules (Standby-Lazy 202,882) because stochastic rule selection
  inadvertently rejects fewer tasks (11.5 vs 19–27), incurring lower rejection penalty.

## Key Narrative Points

1. **Feasibility masking is a prerequisite, not just an optimisation** (+78.3%).
   Without it, the RL agent cannot learn a functional policy at all — zero task
   completions across 30 test instances. The action mask transforms the 15-rule
   combinatorial space into a tractable decision problem by eliminating infeasible
   choices before the policy network sees them.

2. **Partial charging contributes substantially** (+41.6%).
   Removing SOC-target discretisation forces full-recharge cycles, reducing
   completed tasks from 18.10 to 11.33. The agent retains its acceptance strategy
   (zero rejections) but loses the ability to minimise charging time through
   opportunistic partial top-ups.

3. **Both ablation variants still beat Random** (156,522).
   RL-APC-PC (69,008) is 56% cheaper than Random; RL-APC-FM (86,917) is 44%
   cheaper. Even the degenerate FM policy incurs less cost than random rule
   selection because it at least avoids explicit task rejections.

4. **RL-APC-FM does NOT beat fixed rules** — unlike the Full model.
   RL-APC-FM (86,917) is cheaper than Standby-Lazy (202,882) only because
   it completes zero tasks (all cost is terminal penalty), while Standby-Lazy
   accepts and rejects many tasks (high rejection penalty). This is a degenerate
   comparison, not a meaningful advantage.

5. **Random baseline confirms RL contribution** (+221.1%). The gap between
   Random (156,522) and Full (48,745) quantifies the value of learned
   decision-making with both components intact.

## Recommended Paragraph (P_ablation)

```
To isolate the contribution of each architectural component, we evaluate
two ablation variants on M-scale instances: RL-APC-PC removes partial-
charging discretisation (all charging events use full recharge), and
RL-APC-FM disables feasibility masking (all 15 rules are selectable
regardless of precondition satisfaction). Both variants are retrained
from scratch with identical hyperparameters and training budget (1M
steps, seed 42).

Removing partial charging increases total cost by 41.6% (from 48,745 to
69,008) and reduces completed tasks from 18.10 to 11.33 per episode.
The agent retains its zero-rejection strategy but loses the ability to
minimise charging time through opportunistic partial top-ups, resulting
in substantial terminal penalty from unfinished tasks.

The more dramatic result concerns feasibility masking: disabling it
causes complete policy collapse. RL-APC-FM completes zero tasks across
all 30 test instances, with the agent averaging only 106 decision steps
per episode (compared to ~1,900 for the PC variant). Without the action
mask filtering infeasible choices, the policy network cannot distinguish
productive from unproductive actions in the 15-rule combinatorial space,
and training entropy remains high throughout (−2.44 at convergence).
This demonstrates that feasibility masking is not merely a safety
constraint but a prerequisite for learning: it transforms an intractable
decision problem into one the RL agent can solve.
```
