@echo off
REM Week 5: Run OLD (ROI-aware baseline) reward on MEDIUM scale
REM Run all 10 seeds sequentially

echo ========================================
echo Week 5: Medium Scale - OLD Reward
echo ========================================

for %%s in (2025 2026 2027 2028 2029 2030 2031 2032 2033 2034) do (
    echo.
    echo Running seed %%s...
    python scripts/week5/run_reward_experiment.py --scale medium --reward old --seed %%s
    if errorlevel 1 (
        echo ERROR: Seed %%s failed!
        exit /b 1
    )
)

echo.
echo ========================================
echo All MEDIUM-OLD experiments completed!
echo ========================================
