@echo off
REM ============================================================
REM  SENTINEL-3D Instant 1-Click Launcher (Windows)
REM  DRDO Defense Tactical Perception Platform
REM ============================================================

title SENTINEL-3D Tactical Perception Engine
cls

echo ============================================================
echo   SENTINEL-3D - Defense Tactical Perception Engine (DRDO)
echo   Adaptive Foveated Multi-Resolution LiDAR Perception
echo ============================================================
echo.

cd /d "%~dp0"

REM 1. Identify Python binary with priority for virtual environments
set "PYTHON_BIN="

if exist "..\\.venv\\Scripts\\python.exe" (
    set "PYTHON_BIN=..\\.venv\\Scripts\\python.exe"
) else if exist ".venv\\Scripts\\python.exe" (
    set "PYTHON_BIN=.venv\\Scripts\\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BIN=python"
    ) else (
        where py >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_BIN=py -3"
        )
    )
)

if "%PYTHON_BIN%"=="" (
    echo [ERROR] Python not detected in PATH or .venv!
    echo Please install Python 3.10+ or configure your virtual environment.
    echo.
    pause
    exit /b 1
)

echo [*] Initializing tactical pipeline via: %PYTHON_BIN%
echo.

%PYTHON_BIN% launch.py %*

if errorlevel 1 (
    echo.
    echo [!] SENTINEL-3D process stopped with error code %errorlevel%.
    pause
)
