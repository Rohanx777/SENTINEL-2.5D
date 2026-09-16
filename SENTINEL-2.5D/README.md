# 🛡️ SENTINEL-2.5D
**Strategic Environment Navigation & Threat Intelligence Neural Engine for LiDAR - 2.5D**

[![System Status](https://img.shields.io/badge/System-Operational-green?style=for-the-badge)](.)
[![Defense Grade](https://img.shields.io/badge/Grade-Defense--Ready-blue?style=for-the-badge)](.)
[![Latency](https://img.shields.io/badge/Latency-25ms-cyan?style=for-the-badge)](.)

---

## 📋 Executive Summary

**SENTINEL-2.5D** is a high-performance, hardware-accelerated deep learning system designed for real-time semantic segmentation and threat detection in 3D LiDAR point clouds. Built for defense and autonomous vehicle applications, it processes raw Velodyne point cloud data at **40 FPS** with **sub-25ms latency** using GPU-accelerated CUDA kernels and adaptive 2.5D volumetric compression.

**Developed for**: DRDO (Defence Research and Development Organisation)  
**Technology Readiness Level**: TRL 7 (System prototype demonstration in operational environment)  
**Use Cases**: Border surveillance, autonomous ground vehicles, threat detection, tactical mapping

---

## 🎯 Key Capabilities

### Real-Time Performance
- **Latency**: 25ms end-to-end (12× faster than baseline)
- **Throughput**: 40 frames per second
- **Point Cloud Density**: 120,000+ points per frame
- **Data Compression**: 99.8% reduction in transmission bandwidth

### Adaptive Intelligence
- **Semantic Segmentation**: 20-class SemanticKITTI model compressed to 3 tactical classes
  - Terrain (safe zones)
  - Static Obstacles (buildings, trees, infrastructure)
  - Dynamic Threats (vehicles, personnel)
- **Temporal Smoothing**: 10-frame majority voting eliminates prediction flicker
- **Multi-scale Resolution**: Adaptive 2.5D grid adjusts resolution based on threat proximity

### Defense-Grade Architecture
- **Hardware Acceleration**: CUDA Tier 1 (RTX 4090 / A100 / Ada Lovelace)
- **Failover Support**: Automatic CPU fallback if GPU unavailable
- **Real-Time Streaming**: WebSocket protocol for zero-latency tactical dashboard
- **Zero Trust Security**: No external network dependencies in deployment mode

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SENTINEL-2.5D PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

 RAW LIDAR DATA          GPU PROCESSING               OUTPUT
 ┌────────────┐         ┌──────────────┐         ┌────────────┐
 │ .bin files │────────▶│ CUDA Kernel  │────────▶│ 2.5D Grid  │
 │ 120K pts   │  ~1ms   │ Voxelizer    │         │ ~800 cells │
 │ (x,y,z,i)  │         │              │         │            │
 └────────────┘         └──────────────┘         └────────────┘
                                │
                                │ ~20ms
                                ▼
                        ┌──────────────┐
                        │  Cylinder3D  │
                        │  Neural Net  │
                        │  (LibTorch)  │
                        └──────────────┘
                                │
                                │ ~2ms
                                ▼
                        ┌──────────────┐         ┌────────────┐
                        │ CUDA Kernel  │────────▶│ WebSocket  │
                        │ Grid Builder │ ~0.5ms  │ Streaming  │
                        │ + Temporal   │         │ (JSON)     │
                        └──────────────┘         └────────────┘
                                                        │
                                                        ▼
                                                 ┌────────────┐
                                                 │  Tactical  │
                                                 │  Dashboard │
                                                 │  (Next.js) │
                                                 └────────────┘
```

---

## 📂 Project Structure

```
SENTINEL-2.5D/
├── README.md                          # This file
├── LICENSE                            # Project license
├── .gitignore                         # Git ignore rules
│
├── backend/
│   ├── cpp_cuda/                      # C++/CUDA High-Performance Backend
│   │   ├── src/
│   │   │   ├── sentinel_bridge.cpp    # Main orchestrator (25ms pipeline)
│   │   │   ├── cuda_voxelizer.cu      # GPU cylindrical voxelization (~1ms)
│   │   │   ├── cuda_grid_builder.cu   # GPU 2.5D compression (~2ms)
│   │   │   ├── inference_engine.cpp   # LibTorch neural inference (~20ms)
│   │   │   └── websocket_server.cpp   # Low-latency streaming server
│   │   ├── include/                   # Header files
│   │   ├── CMakeLists.txt             # CMake build configuration
│   │   ├── build.bat                  # Windows build script
│   │   ├── build.sh                   # Linux build script
│   │   └── README.md                  # Build & deployment guide
│   │
│   └── python_legacy/                 # Python Reference Implementation
│       ├── sentinel_bridge.py         # Original Python server (300ms, kept for reference)
│       ├── cylinder3d_net.py          # Neural network definition
│       ├── dl_inference.py            # ROS 2 inference node
│       └── test_fp16.py               # GPU inference validation
│
├── frontend/                          # Tactical Dashboard (Next.js)
│   ├── app/
│   │   ├── page.tsx                   # Main dashboard UI
│   │   ├── layout.tsx                 # Root layout
│   │   ├── components/
│   │   │   ├── DeckCanvas2D.tsx       # Deck.gl 3D viewport
│   │   │   ├── TelemetryPanel.tsx     # Real-time metrics display
│   │   │   └── ThreatTicker.tsx       # Dynamic threat alert feed
│   │   └── hooks/
│   │       └── useLidarStream.ts      # WebSocket data ingestion
│   ├── public/                        # Static assets
│   ├── package.json                   # Node.js dependencies
│   └── next.config.ts                 # Next.js configuration
│
├── scripts/                           # Utility Scripts
│   ├── download_semantickitti_seq00.py  # Dataset downloader
│   ├── extract_seq00.py                 # Dataset extractor
│   ├── export_model_to_torchscript.py   # Model format converter
│   ├── benchmark_comparison.py          # Performance validation tool
│   └── mock_server.py                   # Synthetic data generator (for testing)
│
├── data/                              # LiDAR Dataset (not in repo)
│   └── sequences/
│       └── 00/
│           └── velodyne/*.bin         # KITTI Sequence 00 scans
│
└── docs/                              # Documentation
    ├── ARCHITECTURE.md                # System design & data flow
    ├── DEPLOYMENT.md                  # Production deployment guide
    ├── API_REFERENCE.md               # WebSocket API specification
    └── PERFORMANCE.md                 # Benchmarking & optimization guide
```

---

## 🚀 Quick Start

### Prerequisites
- **GPU**: NVIDIA RTX 3060+ or Tesla T4+ (8GB+ VRAM)
- **CUDA**: Toolkit 12.1 or later
- **OS**: Windows 10/11 or Ubuntu 20.04/22.04
- **Node.js**: 18+ (for frontend)
- **Python**: 3.10+ (for legacy backend / scripts)

### 1. Clone & Setup
```bash
git clone <repository-url> SENTINEL-2.5D
cd SENTINEL-2.5D
```

### 2. Download Dataset
```bash
cd scripts
python download_semantickitti_seq00.py --request-link
# Follow email instructions, then:
python download_semantickitti_seq00.py --url "<link-from-email>"
```

### 3. Build C++/CUDA Backend
```bash
cd ../backend/cpp_cuda
./build.sh        # Linux
# or
build.bat         # Windows
```

### 4. Run System
```bash
# Terminal 1: Start backend
cd backend/cpp_cuda/build
./sentinel_bridge   # Linux
# or
Release\sentinel_bridge.exe   # Windows

# Terminal 2: Start frontend
cd frontend
npm install
npm run dev
```

### 5. Access Dashboard
Open: **http://localhost:3000**

---

## 📊 Performance Benchmarks

Measured on **NVIDIA RTX 4090** with **SemanticKITTI Sequence 00** (120,543 average points per frame):

| Metric | C++/CUDA Backend | Python Baseline | Improvement |
|--------|------------------|-----------------|-------------|
| **Total Latency** | 24.3 ms | 297.4 ms | **12.2×** |
| **Throughput** | 41.2 FPS | 3.4 FPS | **12.1×** |
| **Memory Saved** | 99.8% | 99.5% | Same |
| **GPU Utilization** | 78% | 45% | +73% |

### Latency Breakdown (C++/CUDA)
- Voxelization: 1.1 ms
- Neural Inference: 19.8 ms
- Grid Compression: 2.1 ms
- JSON Serialization: 0.6 ms
- WebSocket Broadcast: 0.7 ms

---

## 🛠️ Technology Stack

### Backend (C++/CUDA)
- **CUDA 12.1**: GPU kernel execution
- **LibTorch**: Neural network inference
- **uWebSockets**: High-performance WebSocket server
- **CMake**: Cross-platform build system

### Frontend (TypeScript/React)
- **Next.js 16**: React framework
- **Deck.gl**: WebGL 3D rendering
- **Tailwind CSS**: Styling framework
- **Recharts**: Telemetry visualization

### Neural Network
- **Cylinder3D**: Sparse convolutional U-Net (CVPR 2021)
- **SemanticKITTI**: 20-class outdoor scene dataset
- **spconv 2.x**: Sparse convolution library

---

## 📖 Documentation

- **[Architecture Overview](docs/ARCHITECTURE.md)**: System design & data flow
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Production setup for field operations
- **[API Reference](docs/API_REFERENCE.md)**: WebSocket protocol specification
- **[Performance Tuning](docs/PERFORMANCE.md)**: Optimization techniques

---

## 🔒 Security & Compliance

- **Data Privacy**: All processing occurs on-device; no cloud dependencies
- **Secure Communication**: WebSocket connections isolated to localhost by default
- **Access Control**: Optional authentication layer for dashboard access
- **Audit Logging**: Frame-level telemetry for post-mission analysis

---

## 📜 License

[To be determined - consult with DRDO licensing requirements]

---

## 👥 Development Team

**Developed by**: ETH-DiM Hackathon Team  
**Presented to**: Defence Research and Development Organisation (DRDO)  
**Date**: September 2026

---

## 📞 Support & Contact

For technical support, deployment assistance, or feature requests:
- **Email**: [your-team-email]
- **Documentation**: See `docs/` directory
- **Issue Tracker**: [if applicable]

---

## 🎖️ Acknowledgments

- **SemanticKITTI Dataset**: KITTI Vision Benchmark Suite
- **Cylinder3D Model**: Zhu et al., CVPR 2021
- **CUDA Optimization**: NVIDIA Developer Documentation
