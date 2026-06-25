@echo off
REM ============================================================================
REM  WHISKER Labeler - double-click launcher
REM
REM  Opens the labeler using its own built-in workspace (stored inside this
REM  folder under \workspace), so your projects, datasets, and annotations are
REM  always here and load automatically. Create projects/datasets from inside
REM  the app with the "New Project / Dataset..." button.
REM
REM  On another machine, only ENV_PY below may need editing (the python.exe of
REM  the `whisker-labeler` conda env).
REM ============================================================================

set "ENV_PY=%USERPROFILE%\AppData\Local\miniconda3\envs\whisker-labeler\python.exe"

if not exist "%ENV_PY%" (
    echo [ERROR] Could not find the whisker-labeler environment python at:
    echo         %ENV_PY%
    echo.
    echo Create it first with:  conda env create -f environment.yaml
    echo Then install once with: conda activate whisker-labeler ^&^& pip install -e .
    echo.
    pause
    exit /b 1
)

echo Launching WHISKER Labeler...
"%ENV_PY%" -m whisker.main

if errorlevel 1 (
    echo.
    echo [WHISKER Labeler exited with an error - see messages above.]
    pause
)
