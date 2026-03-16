# Table 9: Computational Efficiency — LaTeX Template

> Updated 2026-03-13
> Runtime = wall-clock seconds for one simulation episode (30-instance average)
> RL-APC includes simulation overhead; actual NN inference is sub-millisecond per decision

## Data Verification

| Scale | RL-APC | Greedy-FR | Greedy-PR | Random | ALNS-FR | ALNS-PR | Source |
|-------|--------|-----------|-----------|--------|---------|---------|--------|
| S | 3.30 | 0.29 | 0.29 | 0.49 | 0.80 | 0.98 | evaluate_S_30.csv |
| M | 5.23 | 0.58 | 0.58 | 1.44 | 2.16 | 7.87 | evaluate_M_synced_30.csv |
| L | 11.54 | 1.59 | 1.56 | 1.25 | — | — | evaluate_L_v3_30.csv |
| XL | 4.24 | 1.03 | 1.02 | 2.58 | 9.42 | 45.93 | evaluate_XL_synced_30.csv |

Cross-checks:
- RL-APC > Greedy-FR on all scales ✅ (NN inference + masked action sampling overhead)
- ALNS-PR > ALNS-FR on all scales ✅ (post-removal adds local-search passes)
- L no ALNS data ✅ (v3/v4 instances not evaluated with ALNS)
- XL RL (4.24s) < XL ALNS-PR (45.93s) ✅ — key deployment argument

## LaTeX Code

```latex
\begin{table}[htbp]
\centering
\caption{Average wall-clock runtime per instance (seconds, 30 test instances).
  RL-APC runtime includes full simulation overhead; the neural-network
  inference itself executes in sub-millisecond time per decision point.
  ALNS is an offline heuristic with access to complete future task information.}
\label{tab:runtime}
\begin{tabular}{l rr rr}
\toprule
Scale & RL-APC & Greedy-FR & ALNS-FR & ALNS-PR \\
\midrule
S  & 3.30  & \textbf{0.29} & 0.80   & 0.98  \\
M  & 5.23  & \textbf{0.58} & 2.16   & 7.87  \\
L  & 11.54 & \textbf{1.59} & ---    & ---   \\
XL & 4.24  & \textbf{1.03} & 9.42   & 45.93 \\
\bottomrule
\end{tabular}

\smallskip
\footnotesize
\textit{Notes:}
Greedy-FR and Greedy-PR runtimes are identical (same rule set, different
post-removal strategy adds negligible overhead) and are shown as a single
column. Random included in full data but omitted here for brevity.
L-scale ALNS unavailable for the current instance set.
\end{table}
```

## Key Narrative Points

### RL-APC is real-time capable
- 3–12 seconds per episode for 15–100 tasks
- Per-decision latency: sub-millisecond (single forward pass through [256,128] or [512,256] MLP)
- Acceptable for real-time warehouse dispatching (decisions every ~seconds)

### RL-APC vs ALNS tradeoff
| Scale | RL / ALNS-PR | RL Cost | ALNS Cost | Verdict |
|-------|-------------|---------|-----------|---------|
| S | 3.4× slower | 13,162 | 5,282 | ALNS faster + cheaper |
| M | 0.66× (faster) | 48,745 | 53,600 | RL faster + cheaper |
| XL | 0.09× (11× faster) | 130,077 | 170,514 | RL faster + cheaper |

On M and XL, RL-APC is both **faster** and **cheaper** than the best offline heuristic.

### Why RL-APC is slower than Greedy
- Greedy: single rule lookup per decision (O(1))
- RL-APC: feature extraction → NN forward pass → action masking → sampling
- Overhead is ~3–11× but remains well within real-time constraints
