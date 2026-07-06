@echo off
REM ============================================================================
REM  WHISKER Labeler - double-click launcher
REM
REM  Launches the WHISKER Labeler GUI. On first run it opens a WHISKER workspace
REM  folder (defaulting to this folder, or the last one you used); use
REM  File > Open Workspace... inside the app to switch at any time. Create
REM  projects and datasets from the app's File menu, or import them from a full
REM  WHISKER workspace.
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
