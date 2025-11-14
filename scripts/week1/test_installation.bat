@echo off
REM Week 1 Installation Test for Windows

echo Setting up Python path...
set PYTHONPATH=%CD%\src;%PYTHONPATH%

echo Running installation verification...
python scripts\week1\test_installation.py

pause
