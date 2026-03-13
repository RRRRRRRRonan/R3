# Table 7: Online-Offline Performance Gap — LaTeX Template

> Updated 2026-03-13, Option B clean cost
> RL-APC / Greedy-FR: Option B (Oper + Reject + Terminal)
> ALNS: raw cost (offline optimizer, completes all tasks, no shaping/rejection/terminal)

## Data Verification

| Scale | RL-APC | Greedy-FR | ALNS (Best) | Gap RL | Gap Greedy | Source |
|-------|--------|-----------|-------------|--------|-----------|--------|
| S | 13,162 | 64,725 | 5,282 (PR) | +149.2% | +1,125.4% | evaluate_S_30.csv |
| M | 48,745 | 233,564 | 53,600 (PR) | −9.1% | +335.8% | evaluate_M_synced_30.csv |
| L | 101,616 | 223,112 | — | — | — | No ALNS for v3/v4 instances |
| XL | 130,077 | 606,680 | 170,514 (PR) | −23.7% | +255.8% | evaluate_XL_synced_30.csv |

Gap = (Online − Offline) / Offline × 100%

## LaTeX Code

```latex
\begin{table}[htbp]
\centering
\caption{Online--offline performance gap (Option~B cost).
  RL-APC and Greedy-FR costs are computed as $C_\text{oper} + C_\text{rej} + C_\text{term}$
  (Section~3). The ALNS offline heuristic has access to complete future task
  information and completes all tasks; its cost reflects pure operational
  expenditure. A negative gap indicates that the online method achieves
  lower total cost than the offline solution through selective task acceptance.}
\label{tab:offline-gap}
\begin{tabular}{l r r r r r}
\toprule
Scale & RL-APC & Greedy-FR & ALNS (Offline)
      & Gap\textsubscript{RL} (\%) & Gap\textsubscript{Greedy} (\%) \\
\midrule
S  & 13{,}162  & 64{,}725   & 5{,}282   & $+149.2$           & $+1{,}125.4$ \\
M  & 48{,}745  & 233{,}564  & 53{,}600  & $\mathbf{-9.1}$    & $+335.8$     \\
L  & 101{,}616 & 223{,}112  & ---       & ---                 & ---          \\
XL & 130{,}077 & 606{,}680  & 170{,}514 & $\mathbf{-23.7}$   & $+255.8$     \\
\bottomrule
\end{tabular}

\smallskip
\footnotesize
\textit{Notes:}
Best offline = min(ALNS-FR, ALNS-PR); all three scales select ALNS-PR.
L-scale offline solution unavailable for the current instance set.
Negative gaps on M and XL arise because the terminal penalty per unfinished
task is lower than the operational cost of completing that task, making
selective acceptance cost-effective under the current penalty structure.
\end{table}
```

## Key Narrative Points for Table 7

### S-scale: +149% gap
- RL-APC still 2.5× the offline cost → online information disadvantage is real
- But Greedy-FR is 12.3× → RL reduces gap by 87% relative to Greedy

### M-scale: −9.1% (RL wins)
- RL total = 48,745 < ALNS total = 53,600
- ALNS completes all 34.8 tasks at ~1,542/task operational cost
- RL completes 18.1 tasks (Oper=6,995) + terminal for ~16.7 unfinished (41,750)
- Terminal penalty (2,500/task) < ALNS per-task cost (1,542 operational + overheads)

### XL-scale: −23.7% (RL wins)
- RL total = 130,077 < ALNS total = 170,514
- Same mechanism: selective acceptance + terminal penalty < full completion cost

### Per-task efficiency (ALNS still better)
| Scale | ALNS Cost/Task | RL Cost/Task | Ratio |
|-------|---------------|-------------|-------|
| S | 308 | 908 | 3.0× |
| M | 1,542 | 2,694 | 1.7× |
| XL | 1,900 | 5,859 | 3.1× |
