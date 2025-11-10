@echo off
REM Week 1: Remaining Q-table Initialization Experiments (Windows version)
REM
REM This script runs the remaining 3 initialization strategies:
REM   - uniform: Uniform positive bias (all Q-values = 50.0)
REM   - action_specific: Higher bias for matheuristic operators
REM   - state_specific: Higher bias for stuck states
REM
REM Note: baseline (zero initialization) already completed - 30 experiments
REM
REM Expected runtime: ~30-60 hours total (3 strategies × 3 scales × 10 seeds = 90 runs)
REM Expected output: 90 JSON files

setlocal enabledelayedexpansion

REM Configuration
set SEEDS=2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
set SCENARIOS=small medium large
set STRATEGIES=uniform action_specific state_specific
set OUTPUT_DIR=results\week1\init_experiments

REM Setup Python path
set PYTHONPATH=%CD%\src;%PYTHONPATH%

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Print header
echo ======================================================================
echo Week 1: Remaining Q-table Initialization Experiments
echo ======================================================================
echo Start time: %date% %time%
echo Output directory: %OUTPUT_DIR%
echo Strategies: %STRATEGIES%
echo Note: Skipping 'zero' strategy (already completed as baseline)
echo.

REM Run experiments
set total=0
set success=0
set failed=0

for %%t in (%STRATEGIES%) do (
    echo ======================================================================
    echo Strategy: %%t
    echo ======================================================================

    for %%s in (%SCENARIOS%) do (
        echo ----------------------------------------------------------------------
        echo Running %%s scenario with %%t initialization...
        echo ----------------------------------------------------------------------

        for %%d in (%SEEDS%) do (
            set /a total+=1
            set output_file=%OUTPUT_DIR%\init_%%t_%%s_seed%%d.json

            REM Show progress without verbose output
            python scripts\week1\run_experiment.py --scenario %%s --init_strategy %%t --seed %%d --output !output_file!

            if !errorlevel! equ 0 (
                set /a success+=1
            ) else (
                set /a failed+=1
                echo   [ERROR] Experiment failed!
            )
        )
        echo.
    )
)

REM Summary
echo ======================================================================
echo Remaining Initialization Experiments Complete
echo ======================================================================
echo End time: %date% %time%
echo Total runs: %total%
echo Successful: %success%
echo Failed: %failed%
echo.
echo Results saved in: %OUTPUT_DIR%
echo.
echo Combined with 30 baseline experiments, you now have:
echo   - Total experiments: 120 (30 baseline + 90 new)
echo   - All 4 strategies tested on 3 scales with 10 seeds each
echo.
echo Next step: Run analysis script
echo   python scripts\week1\analyze_init_strategies.py

pause
