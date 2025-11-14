@echo off
REM Week 1: Remaining Experiments - FAST MODE (Windows version)
REM
REM This version uses --disable_matheuristic_adaptation for faster execution
REM at the cost of potentially lower solution quality.
REM
REM Expected runtime: ~10-30 hours (much faster than standard mode)

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
echo Week 1: Remaining Experiments - FAST MODE
echo ======================================================================
echo Start time: %date% %time%
echo Output directory: %OUTPUT_DIR%
echo Mode: Fast (matheuristic adaptation disabled)
echo Note: Solution quality may be lower, but comparisons are still valid
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
        echo Running %%s scenario with %%t initialization (FAST)...
        echo ----------------------------------------------------------------------

        for %%d in (%SEEDS%) do (
            set /a total+=1
            set output_file=%OUTPUT_DIR%\init_%%t_%%s_seed%%d.json

            REM Fast mode: disable matheuristic adaptation
            python scripts\week1\run_experiment.py --scenario %%s --init_strategy %%t --seed %%d --output !output_file! --disable_matheuristic_adaptation

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
echo Fast Mode Experiments Complete
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
