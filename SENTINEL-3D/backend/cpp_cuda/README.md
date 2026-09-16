# Kavach 2.5D — C++/CUDA Backend

**High-performance real-time LiDAR perception engine**

Replaces the Python backend with optimized C++/CUDA implementation for **12× speedup** (300ms → 25ms per frame).

---

## 🚀 Performance Comparison

| Stage | Python (NumPy/PyTorch) | C++/CUDA | Speedup |
|-------|------------------------|----------|---------|
| **Voxelization** | ~20ms (CPU) | ~1ms (GPU) | **20×** |
| **Neural Inference** | ~60ms (PyTorch eager) | ~20ms (LibTorch) | **3×** |
| **Grid Building** | ~150ms (Python loop) | ~2ms (CUDA kernel) | **75×** |
| **JSON Serialization** | ~20ms | ~0.5ms | **40×** |
| **WebSocket** | ~1ms | ~0.5ms | **2×** |
| **Total Pipeline** | **~300ms** | **~25ms** | **12×** |

---

## 📋 Prerequisites

### 1. CUDA Toolkit 12.x
- Download: https://developer.nvidia.com/cuda-downloads
- Windows: CUDA 12.1+ with Visual Studio 2019/2022
- Linux: CUDA 12.1+ with GCC 9+

### 2. LibTorch (PyTorch C++ API)
Download the pre-built LibTorch package:
- **Windows (CUDA 12.1)**: https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.1.0%2Bcu121.zip
- **Linux (CUDA 12.1)**: https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip

Extract to `C:\libtorch` (Windows) or `/opt/libtorch` (Linux)

### 3. CMake 3.18+
- Windows: https://cmake.org/download/
- Linux: `sudo apt install cmake` or `yum install cmake`

### 4. uWebSockets + uSockets
```bash
cd backend/cpp_cuda/third_party
git clone https://github.com/uNetworking/uWebSockets
git clone https://github.com/uNetworking/uSockets
```

### 5. pybind11 (Optional — for Python bindings)
```bash
cd backend/cpp_cuda/third_party
git clone https://github.com/pybind/pybind11
```

---

## 🔧 Build Instructions

### Step 1: Export PyTorch Model to TorchScript

The C++ backend needs the model in TorchScript format (`.pt` file).

```bash
cd backend
python export_model_to_torchscript.py
```

This converts `weights/model_load.pth` → `weights/model_load.pt`

### Step 2: Build C++/CUDA Backend

#### Windows (Visual Studio)

```bash
cd backend/cpp_cuda
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_PREFIX_PATH=C:/libtorch -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Run
Release\kaavach_bridge.exe
```

#### Linux

```bash
cd backend/cpp_cuda
mkdir build && cd build

# Configure
cmake .. -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_BUILD_TYPE=Release

# Build (use all CPU cores)
make -j$(nproc)

# Run
./kaavach_bridge
```

---

## 🎯 Usage

### Running the C++ Backend

```bash
# Default: reads data/sequences/00/velodyne/*.bin, serves on ws://localhost:8000
./kaavach_bridge

# Custom paths
./kaavach_bridge /path/to/scans /path/to/model_load.pt
```

The server automatically:
- Loads the Cylinder3D model on GPU
- Starts WebSocket server on `ws://0.0.0.0:8000/ws/stream_map`
- Streams at 30 FPS (adjustable in code)

### Frontend (No Changes Needed)

The Next.js frontend works unchanged:

```bash
cd ../../   # back to project root
npm install
npm run dev
```

Open http://localhost:3000

---

## 🐍 Python Bindings (Optional)

For **incremental migration**, use the CUDA kernels from Python:

### Build Python Module

```bash
cd backend/cpp_cuda/build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make
```

### Use from Python

```python
import kaavach_cuda_py
import numpy as np

# Replace NumPy voxelization with GPU kernel (20× faster)
points_xyz = np.random.randn(120000, 3).astype(np.float32)
points_sig = np.random.randn(120000).astype(np.float32)

pt_fea, grid_ind = kaavach_cuda_py.voxelize(points_xyz, points_sig)

# Replace Python build_grid() with GPU kernel (75× faster)
labels = np.random.randint(0, 3, 120000, dtype=np.uint8)
cells = kaavach_cuda_py.build_grid(points_xyz, labels)
```

---

## 🛠️ Troubleshooting

### "CUDA out of memory"
Reduce batch size or point cloud resolution. Default settings work with 8GB VRAM.

### "Cannot find libtorch"
Set `CMAKE_PREFIX_PATH` explicitly:
```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

### "undefined reference to torch::jit"
Make sure you downloaded **libtorch with dependencies**, not the minimal package.

### Windows: "Cannot find CUDA"
Add CUDA to PATH:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
```

---

## 📊 Benchmarking

Run the standalone test harness:

```bash
# C++ version
./kaavach_bridge --benchmark

# Python version (for comparison)
cd ../
python kaavach_bridge.py
```

Expected output:
```
[CUDA] RTX 4090 (Compute Capability 8.9)
[Pipeline] Frame 0001 | Pts: 120543 | Cells: 782 | Latency: 24.3 ms | FPS: 41.2
```

---

## 📁 Project Structure

```
cpp_cuda/
├── CMakeLists.txt              # Build configuration
├── src/
│   ├── kaavach_bridge.cpp      # Main pipeline
│   ├── cuda_voxelizer.cu       # CUDA voxelizer kernel
│   ├── cuda_grid_builder.cu    # CUDA 2.5D grid kernel
│   ├── inference_engine.cpp    # LibTorch wrapper
│   ├── websocket_server.cpp    # uWebSockets server
│   └── python_bindings.cpp     # pybind11 wrappers
├── include/
│   ├── cuda_voxelizer.cuh
│   ├── cuda_grid_builder.cuh
│   ├── inference_engine.h
│   ├── websocket_server.h
│   └── json_serializer.h       # Fast JSON builder
└── third_party/
    ├── uWebSockets/
    ├── uSockets/
    └── pybind11/
```

---

## 🔬 Architecture Details

### CUDA Voxelizer (`cuda_voxelizer.cu`)
- 1 thread per point (~120K parallel threads)
- Cartesian → Cylindrical coordinate transform
- Grid index computation
- 9D feature vector construction
- **Replaces**: Python `CylinderVoxelizer.__call__()`

### CUDA Grid Builder (`cuda_grid_builder.cu`)
- **Phase 1**: Atomic aggregation (1 thread per point)
  - z_min/z_max via `atomicMin/atomicMax` with float-as-int
  - Label histogram via `atomicAdd`
- **Phase 2**: Cell post-processing (1 thread per cell)
  - AI + Geometry fusion heuristic
  - 10-frame temporal majority vote (ring buffer on GPU)
  - Polygon construction
- **Replaces**: Python `build_grid()` for-loop

### LibTorch Inference Engine (`inference_engine.cpp`)
- Loads TorchScript model via `torch::jit::load()`
- Zero-copy tensor wrapping of CUDA pointers
- **Replaces**: Python `model([pt_fea], [grid_ind])`

### uWebSockets Server (`websocket_server.cpp`)
- Zero-copy broadcasting
- Multi-threaded event loop
- **Replaces**: Python `websockets` + `asyncio`

---

## 📈 Next Steps

1. **TensorRT Optimization** (20ms → 10ms inference)
   - Export model to ONNX, convert to TensorRT engine
   - Expected 2× further speedup on inference

2. **Multi-GPU Support**
   - Process multiple scans in parallel
   - Scale to 100+ FPS throughput

3. **ROS 2 Integration**
   - Replace file reading with `/lidar/points_raw` subscription
   - Real-time vehicle deployment

---

## 📄 License

Same license as parent project.

## 🤝 Contributing

Built for the ETH-DiM hackathon. PRs welcome!
