@echo off
REM Week 1, Day 1-3: Baseline Collection (Windows version with error display)

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
echo Python path: %PYTHONPATH%
echo.

REM Test single run first
echo Testing single experiment...
python scripts\week1\run_experiment.py --scenario small --init_strategy zero --seed 2025 --output test_run.json --verbose
if !errorlevel! neq 0 (
    echo.
    echo ERROR: Test run failed! Please check the error above.
    echo.
    pause
    exit /b 1
)
echo Test run successful!
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

        REM Run without error suppression to see what's wrong
        python scripts\week1\run_experiment.py --scenario %%s --init_strategy %INIT_STRATEGY% --seed %%d --output !output_file!

        if !errorlevel! equ 0 (
            set /a success+=1
            echo   Done
        ) else (
            set /a failed+=1
            echo   FAILED - See error above
            echo.
            echo Press any key to continue or Ctrl+C to stop...
            pause >nul
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
if %success% gtr 0 (
    echo Next step: Run analysis script
    echo   python scripts\week1\analyze_baseline.py
)

pause
