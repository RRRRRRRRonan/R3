@echo off
REM Week 1: ACTION_SPECIFIC Strategy Experiments
REM Run this in a separate PowerShell window for parallel execution

setlocal enabledelayedexpansion

set SEEDS=2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
set SCENARIOS=small medium large
set STRATEGY=action_specific
set OUTPUT_DIR=results\week1\init_experiments

set PYTHONPATH=%CD%\src;%PYTHONPATH%

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ======================================================================
echo Week 1: ACTION_SPECIFIC Initialization Strategy
echo ======================================================================
echo Start time: %date% %time%
echo Strategy: %STRATEGY%
echo Window: This window is dedicated to ACTION_SPECIFIC experiments
echo ======================================================================
echo.

set total=0
set success=0
set failed=0

for %%s in (%SCENARIOS%) do (
    echo ----------------------------------------------------------------------
    echo Running %%s scenario with %STRATEGY% initialization...
    echo ----------------------------------------------------------------------

    for %%d in (%SEEDS%) do (
        set /a total+=1
        set output_file=%OUTPUT_DIR%\init_%STRATEGY%_%%s_seed%%d.json

        echo [!total!/30] %STRATEGY%/%%s/seed%%d...

        python scripts\week1\run_experiment.py --scenario %%s --init_strategy %STRATEGY% --seed %%d --output !output_file!

        if !errorlevel! equ 0 (
            set /a success+=1
            echo   Done
        ) else (
            set /a failed+=1
            echo   FAILED
        )
    )
    echo.
)

echo ======================================================================
echo ACTION_SPECIFIC Strategy Complete
echo ======================================================================
echo End time: %date% %time%
echo Total: %total% / Successful: %success% / Failed: %failed%
echo ======================================================================

pause
