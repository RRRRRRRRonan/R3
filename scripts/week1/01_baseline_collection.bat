@echo off
REM Week 1, Day 1-3: Baseline Collection (Windows version)
REM
REM This script collects baseline performance data for the current Q-learning
REM implementation (zero initialization) across 10 seeds and 3 scales.
REM
REM Expected runtime: ~30 minutes (10 seeds × 3 scales)
REM Expected output: 30 JSON files

setlocal enabledelayedexpansion

REM Configuration
set SEEDS=2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
set SCENARIOS=small medium large
set INIT_STRATEGY=zero
set OUTPUT_DIR=results\week1\baseline

REM Setup Python path
set PYTHONPATH=%CD%\src;%PYTHONPATH%

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Print header
echo ======================================================================
echo Week 1: Baseline Collection (Zero Initialization)
echo ======================================================================
echo Start time: %date% %time%
echo Output directory: %OUTPUT_DIR%
echo.

REM Run experiments
set total=0
set success=0
set failed=0

for %%s in (%SCENARIOS%) do (
    echo ----------------------------------------------------------------------
    echo Running %%s scenario...
    echo ----------------------------------------------------------------------

    for %%d in (%SEEDS%) do (
        set /a total+=1
        set output_file=%OUTPUT_DIR%\baseline_%%s_seed%%d.json

        echo [!total!] %%s seed %%d...

        python scripts\week1\run_experiment.py --scenario %%s --init_strategy %INIT_STRATEGY% --seed %%d --output !output_file! 2>nul

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

REM Summary
echo ======================================================================
echo Baseline Collection Complete
echo ======================================================================
echo End time: %date% %time%
echo Total runs: %total%
echo Successful: %success%
echo Failed: %failed%
echo.
echo Results saved in: %OUTPUT_DIR%
echo.
echo Next step: Run analysis script
echo   python scripts\week1\analyze_baseline.py

pause
