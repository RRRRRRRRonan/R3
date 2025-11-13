@echo off
REM Week 5: Run NEW (scale-aware) reward on LARGE scale
REM Run all 10 seeds sequentially

echo ========================================
echo Week 5: Large Scale - NEW Reward
echo ========================================

for %%s in (2025 2026 2027 2028 2029 2030 2031 2032 2033 2034) do (
    echo.
    echo Running seed %%s...
    python scripts/week5/run_reward_experiment.py --scale large --reward new --seed %%s
    if errorlevel 1 (
        echo ERROR: Seed %%s failed!
        exit /b 1
    )
)

echo.
echo ========================================
echo All LARGE-NEW experiments completed!
echo ========================================
