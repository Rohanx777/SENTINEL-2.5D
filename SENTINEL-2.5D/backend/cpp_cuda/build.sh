#!/bin/bash
set -e

echo "============================================================"
echo "  KAVACH 2.5D -- C++/CUDA Build Script (Linux)"
echo "============================================================"

# 1. Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "[ERROR] CMake not found!"
    echo "Install with: sudo apt install cmake"
    exit 1
fi

# 2. Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] NVCC (CUDA compiler) not found!"
    echo "Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
echo "[CUDA] Found nvcc: $(which nvcc)"

# 3. Check for LibTorch
LIBTORCH_PATH="/opt/libtorch"
if [ ! -d "$LIBTORCH_PATH" ]; then
    echo ""
    echo "[WARN] LibTorch not found at $LIBTORCH_PATH"
    echo "Please download LibTorch (cxx11 ABI, CUDA 12.1) from:"
    echo "  https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
    echo "and extract to /opt/libtorch"
    echo ""
    read -p "Or enter custom LibTorch path: " LIBTORCH_CUSTOM
    if [ -n "$LIBTORCH_CUSTOM" ]; then
        LIBTORCH_PATH="$LIBTORCH_CUSTOM"
    else
        exit 1
    fi
fi
echo "[LibTorch] Using LibTorch at: $LIBTORCH_PATH"

# 4. Create build directory
mkdir -p build
cd build

# 5. Configure with CMake
echo ""
echo "[1/2] Configuring with CMake..."
cmake .. -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" -DCMAKE_BUILD_TYPE=Release

# 6. Build
echo ""
echo "[2/2] Building C++/CUDA backend..."
make -j$(nproc)

echo ""
echo "============================================================"
echo "  Build Successful!"
echo "  Executable: build/kaavach_bridge"
echo "============================================================"
echo ""
echo "To run:"
echo "  ./build/kaavach_bridge"
echo ""
