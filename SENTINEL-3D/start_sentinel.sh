#!/bin/bash
# ============================================================
#  SENTINEL-3D Unified Launcher (Linux/macOS)
#  Starts C++/CUDA backend + Next.js frontend in one command
# ============================================================

echo ""
echo "============================================================"
echo "  SENTINEL-3D - Defense Tactical Perception Engine"
echo "  Launching Full Stack..."
echo "============================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if model weights exist
if [ ! -f "backend/cpp_cuda/weights/model_load.pt" ]; then
    echo "[!] TorchScript model not found. Exporting from PyTorch checkpoint..."
    python scripts/export_model_to_torchscript.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Model export failed. Ensure weights/model_load.pth exists."
        exit 1
    fi
fi

# Check if C++ backend is built
if [ ! -f "backend/cpp_cuda/build/sentinel_bridge" ]; then
    echo "[!] C++ backend not built. Building now..."
    cd backend/cpp_cuda
    chmod +x build.sh
    ./build.sh
    cd ../..
    if [ $? -ne 0 ]; then
        echo "[ERROR] Build failed. Check CMake and CUDA installation."
        exit 1
    fi
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "[!] Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    if [ $? -ne 0 ]; then
        echo "[ERROR] npm install failed."
        exit 1
    fi
fi

echo ""
echo "[OK] Pre-flight checks passed."
echo ""
echo "Starting services..."
echo "  - Backend (C++/CUDA): ws://localhost:8000/ws/stream_map"
echo "  - Frontend (Next.js): http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping SENTINEL-3D services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "All services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend in background
cd backend/cpp_cuda/build
./sentinel_bridge &
BACKEND_PID=$!
cd ../../..

# Wait 3 seconds for backend to initialize
sleep 3

# Start frontend in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================================"
echo "  SENTINEL-3D is now running!"
echo ""
echo "  Dashboard:  http://localhost:3000"
echo "  WebSocket:  ws://localhost:8000/ws/stream_map"
echo ""
echo "  Press Ctrl+C to stop all services."
echo "============================================================"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
