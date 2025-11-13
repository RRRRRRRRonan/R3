@echo off
REM Week 2: CURRENT Epsilon Strategy - MEDIUM Scenario
setlocal enabledelayedexpansion

set SEEDS=2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
set STRATEGY=current
set SCENARIO=medium
set OUTPUT_DIR=results\week2\epsilon_experiments

for %%s in (%SEEDS%) do (
    set output_file=%OUTPUT_DIR%\epsilon_%STRATEGY%_%SCENARIO%_seed%%s.json
    echo Running %SCENARIO% scenario with %STRATEGY% epsilon, seed %%s...
    python scripts\week2\run_experiment.py --scenario %SCENARIO% --epsilon_strategy %STRATEGY% --seed %%s --output !output_file!
)

echo.
echo Completed all %SCENARIO% experiments for %STRATEGY% epsilon strategy!
