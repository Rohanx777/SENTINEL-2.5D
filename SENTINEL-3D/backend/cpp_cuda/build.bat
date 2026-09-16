@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   KAVACH 2.5D -- C++/CUDA Build Script (Windows)
echo ============================================================

:: 1. Check for CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] CMake not found in PATH!
    echo Please install CMake from: https://cmake.org/download/
    exit /b 1
)

:: 2. Check for CUDA
if not defined CUDA_PATH (
    echo [ERROR] CUDA_PATH is not set!
    echo Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)
echo [CUDA] Found CUDA at: %CUDA_PATH%

:: 3. Check for LibTorch
set LIBTORCH_PATH=C:\libtorch
if not exist "%LIBTORCH_PATH%" (
    echo.
    echo [WARN] LibTorch not found at %LIBTORCH_PATH%
    echo Please download LibTorch (Release, CUDA 12.1) from:
    echo   https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.1.0%%2Bcu121.zip
    echo and extract it to C:\libtorch
    echo.
    set /p LIBTORCH_CUSTOM="Or enter custom LibTorch path: "
    if defined LIBTORCH_CUSTOM (
        set LIBTORCH_PATH=%LIBTORCH_CUSTOM%
    ) else (
        exit /b 1
    )
)
echo [LibTorch] Using LibTorch at: %LIBTORCH_PATH%

:: 4. Create build directory
if not exist "build" mkdir build
cd build

:: 5. Configure with CMake
echo.
echo [1/2] Configuring with CMake...
cmake .. -DCMAKE_PREFIX_PATH="%LIBTORCH_PATH%" -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed!
    exit /b 1
)

:: 6. Build
echo.
echo [2/2] Building C++/CUDA backend...
cmake --build . --config Release --parallel
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo ============================================================
echo   Build Successful!
echo   Executable: build\Release\kaavach_bridge.exe
echo ============================================================
echo.
echo To run:
echo   cd build\Release
echo   kaavach_bridge.exe
echo.

pause
