@echo off
REM ACTION_SPECIFIC (Fixed) - SMALL Scenario Only
REM This script re-runs action_specific experiments with the bug fix

setlocal enabledelayedexpansion

set SEEDS=2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
set STRATEGY=action_specific
set SCENARIO=small
set OUTPUT_DIR=results\week1\init_experiments

set PYTHONPATH=%CD%\src;%PYTHONPATH%
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ======================================================================
echo ACTION_SPECIFIC (FIXED) - SMALL Scenario
echo ======================================================================
echo Start: %date% %time%
echo Bug fix: 'lp' operator now correctly identified as matheuristic
echo Expected: LP repair should get initial Q-value of 100.0
echo.

set total=0
set success=0

for %%d in (%SEEDS%) do (
    set /a total+=1
    set output_file=%OUTPUT_DIR%\init_%STRATEGY%_%SCENARIO%_seed%%d.json
    echo [!total!/10] %STRATEGY%/%SCENARIO%/seed%%d...
    python scripts\week1\run_experiment.py --scenario %SCENARIO% --init_strategy %STRATEGY% --seed %%d --output !output_file!
    if !errorlevel! equ 0 (
        set /a success+=1
        echo   Done
    ) else (
        echo   FAILED
    )
)

echo.
echo ======================================================================
echo Complete: %success%/10 successful
echo End: %date% %time%
echo ======================================================================
pause
