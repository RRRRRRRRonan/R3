@echo off
REM UNIFORM Strategy - LARGE Scenario Only
setlocal enabledelayedexpansion

set SEEDS=2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
set STRATEGY=uniform
set SCENARIO=large
set OUTPUT_DIR=results\week1\init_experiments

set PYTHONPATH=%CD%\src;%PYTHONPATH%
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ======================================================================
echo UNIFORM - LARGE Scenario
echo ======================================================================
echo Start: %date% %time%
echo.

set total=0
set success=0

for %%d in (%SEEDS%) do (
    set /a total+=1
    set output_file=%OUTPUT_DIR%\init_%STRATEGY%_%SCENARIO%_seed%%d.json
    echo [!total!/10] %STRATEGY%/%SCENARIO%/seed%%d...
    python scripts\week1\run_experiment.py --scenario %SCENARIO% --init_strategy %STRATEGY% --seed %%d --output !output_file!
    if !errorlevel! equ 0 (set /a success+=1)
)

echo.
echo Complete: %success%/10
pause
