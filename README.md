<div align="center">

# 🛡️ SENTINEL-3D

### Strategic Environment Navigation & Threat Intelligence Neural Engine for LiDAR – 3D

**GPU-Accelerated Real-Time Semantic Segmentation of 3D LiDAR Point Clouds**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)]()

---

**25ms latency** · **40 FPS throughput** · **120K+ points/frame** · **99.8% bandwidth compression**

*A defense-grade LiDAR perception system that processes raw Velodyne point clouds through GPU-accelerated CUDA kernels, performs semantic segmentation via Cylinder3D neural inference, and streams compressed tactical data to a real-time WebGL dashboard — all in under 25 milliseconds.*

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Performance](#-performance-benchmarks) · [Dashboard](#-tactical-dashboard) · [Deployment](#-deployment)

</div>

---

## 🎯 What is SENTINEL-3D?

SENTINEL-3D is a high-performance perception pipeline built for autonomous vehicles and defense surveillance. It takes raw 3D LiDAR scans (120,000+ points per frame) and delivers classified, compressed tactical maps to a real-time browser dashboard at 40 FPS.

The system compresses Cylinder3D's 20-class SemanticKITTI output into 3 tactical classes:

| Class | Color | Description | Examples |
|-------|-------|-------------|----------|
| 🟢 **Terrain** | Emerald | Safe traversable zones | Road, sidewalk, vegetation |
| 🔴 **Static Obstacle** | Red | Fixed structures | Buildings, fences, poles |
| 🟡 **Dynamic Threat** | Amber | Moving entities | Vehicles, pedestrians, cyclists |

---

## 🏗️ Architecture

```
                        SENTINEL-3D PERCEPTION PIPELINE
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  RAW LiDAR (.bin)     GPU PROCESSING           TACTICAL OUTPUT  │
  │  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
  │  │ Velodyne Scan │───▶│ CUDA Voxelizer│───▶│ 2.5D Compressed  │  │
  │  │ 120K points   │1ms│ Cylindrical   │    │ Grid (~800 cells)│  │
  │  │ (x, y, z, i) │    └───────────────┘    └──────────────────┘  │
  │  └──────────────┘           │                      │            │
  │                             │ 20ms                 │            │
  │                    ┌────────▼────────┐             │            │
  │                    │   Cylinder3D    │             │ 0.5ms      │
  │                    │   Neural Net    │             │            │
  │                    │   (LibTorch)    │    ┌────────▼─────────┐  │
  │                    └────────┬────────┘    │    WebSocket     │  │
  │                             │ 2ms         │    Streaming     │  │
  │                    ┌────────▼────────┐    │    (JSON)        │  │
  │                    │ CUDA Grid       │───▶│                  │  │
  │                    │ Builder +       │    └────────┬─────────┘  │
  │                    │ Temporal Vote   │             │            │
  │                    └─────────────────┘             │            │
  │                                          ┌────────▼─────────┐  │
  │                                          │ Tactical WebGL   │  │
  │                                          │ Dashboard        │  │
  │                                          │ (Next.js+Deck.gl)│  │
  │                                          └──────────────────┘  │
  └─────────────────────────────────────────────────────────────────┘
                    Total: ~25ms end-to-end
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **GPU Kernels** | CUDA 12.1 | Cylindrical voxelization, grid compression |
| **Neural Inference** | LibTorch / PyTorch | Cylinder3D sparse convolution (spconv 2.x) |
| **Backend Server** | C++17 / uWebSockets | Sub-ms WebSocket streaming |
| **Frontend** | Next.js 16 + TypeScript | Tactical dashboard framework |
| **3D Rendering** | Deck.gl (WebGL) | Real-time point cloud & grid visualization |
| **Styling** | Tailwind CSS | Military-grade dark UI |
| **Charts** | Recharts | Telemetry & performance visualization |

---

## 📊 Performance Benchmarks

Measured on **NVIDIA RTX 4090** with **SemanticKITTI Sequence 00** (120,543 avg points/frame):

| Metric | C++/CUDA | Python Baseline | Speedup |
|--------|----------|-----------------|---------|
| **Total Latency** | 24.3 ms | 297.4 ms | **12.2×** |
| **Throughput** | 41.2 FPS | 3.4 FPS | **12.1×** |
| **Bandwidth Saved** | 99.8% | 99.5% | — |
| **GPU Utilization** | 78% | 45% | +73% |

### Latency Breakdown (C++/CUDA Pipeline)

```
Voxelization ██░░░░░░░░░░░░░░░░░░   1.1 ms  ( 5%)
Inference    ████████████████░░░░  19.8 ms  (81%)
Grid Build   ██░░░░░░░░░░░░░░░░░░   2.1 ms  ( 9%)
Serialize    █░░░░░░░░░░░░░░░░░░░   0.6 ms  ( 2%)
WebSocket    █░░░░░░░░░░░░░░░░░░░   0.7 ms  ( 3%)
─────────────────────────────────────────────────
Total                              24.3 ms (100%)
```

---

## 🖥️ Tactical Dashboard

The frontend is a real-time WebGL tactical dashboard with:

- **Foveated Polar Grid** — Adaptive resolution that increases near the sensor (where detail matters most)
- **3D Point Cloud Overlay** — Full-resolution classified point cloud rendering
- **Split View Comparison** — Side-by-side 3D vs 2.5D analysis
- **Motion-Compensated World Frame** — Ego-motion tracking via KITTI odometry
- **Real-Time Telemetry** — FPS, latency, GPU usage, compression metrics
- **Threat Ticker** — Live feed of detected dynamic obstacles with coordinates
- **Playback Speed Control** — 1×, 2×, 4×, 8× turbo modes

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| **GPU** | NVIDIA RTX 3060+ / Tesla T4+ (8GB+ VRAM) |
| **CUDA Toolkit** | 12.1+ |
| **Python** | 3.10+ |
| **Node.js** | 18+ |
| **OS** | Windows 10/11 or Ubuntu 20.04+ |

### 1-Click Launch (Windows)

```cmd
git clone https://github.com/Rohanx777/SENTINEL-3D.git
cd SENTINEL-3D
start.bat
```

This single command:
- ✅ Auto-detects the best backend (C++/CUDA → Python → Mock)
- ✅ Installs frontend dependencies if missing
- ✅ Starts the backend perception engine (port 8005)
- ✅ Starts the Next.js tactical dashboard (port 3000)
- ✅ Opens the browser automatically
- ✅ Handles graceful multi-process shutdown with Ctrl+C

### Manual Setup

```bash
# Clone
git clone https://github.com/Rohanx777/SENTINEL-3D.git
cd SENTINEL-3D

# Backend: Build C++/CUDA pipeline (one-time)
cd SENTINEL-3D/backend/cpp_cuda
./build.sh          # Linux
build.bat           # Windows

# Frontend: Install dependencies (one-time)
cd ../../frontend
npm install

# Launch everything
cd ..
python launch.py
```

**Dashboard** → [http://localhost:3000](http://localhost:3000)
**WebSocket** → `ws://localhost:8005/ws/stream_map`

---

## 📂 Project Structure

```
SENTINEL-3D/
├── start.bat                           # 1-click Windows launcher
│
├── SENTINEL-3D/
│   ├── launch.py                       # Universal cross-platform launcher
│   ├── sentinel.bat                    # Windows batch orchestrator
│   ├── start_sentinel.sh              # Linux/macOS shell launcher
│   │
│   ├── backend/
│   │   ├── cpp_cuda/                   # ⚡ High-performance C++/CUDA backend
│   │   │   ├── src/
│   │   │   │   ├── sentinel_bridge.cpp # Main pipeline orchestrator (25ms)
│   │   │   │   ├── cuda_voxelizer.cu   # GPU cylindrical voxelization (~1ms)
│   │   │   │   ├── cuda_grid_builder.cu# GPU 2.5D grid compression (~2ms)
│   │   │   │   ├── inference_engine.cpp# LibTorch neural inference (~20ms)
│   │   │   │   └── websocket_server.cpp# Low-latency WS streaming
│   │   │   ├── include/                # CUDA/C++ headers
│   │   │   └── CMakeLists.txt          # CMake build config
│   │   │
│   │   ├── python_legacy/              # 🐍 Python reference implementation
│   │   │   ├── sentinel_bridge.py      # Original server (300ms baseline)
│   │   │   ├── cylinder3d_net.py       # Neural network definition
│   │   │   └── dl_inference.py         # ROS 2 inference node
│   │   │
│   │   └── config/
│   │       └── sensor_config.yaml      # Sensor, grid & model configuration
│   │
│   ├── frontend/                       # 🌐 Tactical WebGL dashboard
│   │   ├── app/
│   │   │   ├── page.tsx                # Main dashboard layout
│   │   │   ├── components/
│   │   │   │   ├── DeckCanvas2D.tsx    # Deck.gl 3D/2.5D viewport
│   │   │   │   ├── TelemetryPanel.tsx  # Real-time metrics display
│   │   │   │   ├── ThreatTicker.tsx    # Dynamic threat alert feed
│   │   │   │   ├── ComparativeStudyModal.tsx
│   │   │   │   └── FoveatedStratifiedModal.tsx
│   │   │   └── hooks/
│   │   │       └── useLidarStream.ts   # WebSocket data hook
│   │   ├── package.json
│   │   └── next.config.ts
│   │
│   ├── scripts/                        # 🔧 Utilities
│   │   ├── download_semantickitti_seq00.py
│   │   ├── export_model_to_torchscript.py
│   │   ├── benchmark_comparison.py
│   │   └── mock_server.py             # Synthetic data server (no GPU needed)
│   │
│   └── docs/                           # 📖 Documentation
│       ├── ARCHITECTURE.md
│       └── DEPLOYMENT.md
│
└── .gitignore
```

---

## 🔧 Backend Modes

The launcher auto-selects the best available backend:

| Priority | Mode | Latency | FPS | Requirements |
|----------|------|---------|-----|-------------|
| 1️⃣ | **C++/CUDA** | ~25ms | 40 | Built binary + CUDA GPU |
| 2️⃣ | **Python** | ~300ms | 3 | PyTorch + model weights + dataset |
| 3️⃣ | **Mock Server** | ~50ms | 20 | Python only (synthetic data) |

The mock server generates realistic synthetic LiDAR data — perfect for frontend development and demos without a GPU.

---

## 🔬 Key Innovations

### Adaptive Foveated Multi-Resolution Grid
Unlike uniform grids, SENTINEL-3D uses a polar foveated grid with higher angular resolution near the sensor — matching how real perception systems prioritize nearby obstacles.

### 2.5D Volumetric Compression
Raw point clouds (120K points × 4 floats = 2MB/frame) are compressed into ~800 2D grid cells with height metadata (~20KB JSON). A **99.8% bandwidth reduction** that enables real-time WebSocket streaming.

### 10-Frame Temporal Majority Voting
A sliding window buffer takes the statistical mode of class predictions over the last 10 frames, eliminating neural network prediction flicker and producing stable tactical classifications.

### Motion-Compensated World Frame
Vehicle ego-motion is tracked via KITTI odometry poses, enabling world-fixed coordinate visualization that removes sensor motion artifacts.

---

## 🛡️ Security & Deployment

- **Zero external dependencies** — all processing runs on-device
- **Localhost-only by default** — WebSocket isolated to 127.0.0.1
- **No cloud telemetry** — no data leaves the machine
- **Docker-ready** — NVIDIA Container Toolkit supported

See [Deployment Guide](SENTINEL-3D/docs/DEPLOYMENT.md) for production configuration, network setup, and Docker instructions.

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](SENTINEL-3D/docs/ARCHITECTURE.md) | Full data flow & tech stack breakdown |
| [Deployment](SENTINEL-3D/docs/DEPLOYMENT.md) | Production setup, Docker, network config |
| [Quick Start](SENTINEL-3D/QUICKSTART.md) | Step-by-step launch instructions |

---

## 🙏 Acknowledgments

- **[Cylinder3D](https://github.com/xinge008/Cylinder3D)** — Zhu et al., CVPR 2021 — Sparse convolutional 3D segmentation
- **[SemanticKITTI](http://www.semantic-kitti.org/)** — KITTI Vision Benchmark Suite
- **[Deck.gl](https://deck.gl/)** — Uber's WebGL visualization framework
- **[spconv](https://github.com/traveller59/spconv)** — Sparse convolution library

---

<div align="center">

**Built with ⚡ CUDA, 🧠 PyTorch, and 🌐 Next.js**

*SENTINEL-3D — See everything. Miss nothing.*

</div>
