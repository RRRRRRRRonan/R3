@echo off
REM Week 1, Day 4-7: Q-table Initialization Experiments (Windows version)
REM
REM This script tests 4 different initialization strategies:
REM   - zero: Current baseline (all Q-values = 0.0)
REM   - uniform: Uniform positive bias (all Q-values = 50.0)
REM   - action_specific: Higher bias for matheuristic operators
REM   - state_specific: Higher bias for stuck states
REM
REM Expected runtime: ~2 hours (4 strategies × 3 scales × 10 seeds = 120 runs)
REM Expected output: 120 JSON files

setlocal enabledelayedexpansion

REM Configuration
set SEEDS=2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
set SCENARIOS=small medium large
set STRATEGIES=zero uniform action_specific state_specific
set OUTPUT_DIR=results\week1\init_experiments

REM Setup Python path
set PYTHONPATH=%CD%\src;%PYTHONPATH%

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Print header
echo ======================================================================
echo Week 1: Q-table Initialization Experiments
echo ======================================================================
echo Start time: %date% %time%
echo Output directory: %OUTPUT_DIR%
echo Strategies: %STRATEGIES%
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

            echo [!total!] %%t/%%s/seed%%d...

            python scripts\week1\run_experiment.py --scenario %%s --init_strategy %%t --seed %%d --output !output_file! 2>nul

            if !errorlevel! equ 0 (
                set /a success+=1
                echo Done
            ) else (
                set /a failed+=1
                echo FAILED
            )
        )
        echo.
    )
)

REM Summary
echo ======================================================================
echo Initialization Experiments Complete
echo ======================================================================
echo End time: %date% %time%
echo Total runs: %total%
echo Successful: %success%
echo Failed: %failed%
echo.
echo Results saved in: %OUTPUT_DIR%
echo.
echo Next step: Run analysis script
echo   python scripts\week1\analyze_init_strategies.py

pause
