# 🛡️ Kavach 2.5D: Tactical LiDAR Perception

**Kavach 2.5D** is a real-time, hardware-accelerated deep learning pipeline and tactical dashboard for semantic segmentation of 3D LiDAR point clouds. 

It takes raw, heavy 3D Velodyne point cloud data, passes it through a heavy-duty **spconv-based Cylinder3D neural network** on the GPU, and compresses it down into an ultra-fast **2.5D Cartesian Volumetric Grid**. The result is a lightweight, radar-like tactical UI built with Next.js and Deck.gl that streams at 5-10 FPS without melting the browser.

![Kavach Dashboard](https://img.shields.io/badge/System-Active-cyan?style=for-the-badge) ![Status](https://img.shields.io/badge/Engine-CUDA_Tier_1-green?style=for-the-badge)

---

## 🎯 Architecture Overview

The system is split into two decoupled architectures communicating over high-speed WebSockets.

### 1. The AI Inference Bridge (Python / PyTorch)
Located in `backend/`, the Python bridge handles the heavy lifting:
- **Cylinder3D AI Model:** Utilizes Sparse Convolutions (`spconv v2`) and a U-Net architecture to semantically segment 3D point clouds (classifying points as road, building, vehicle, pedestrian, etc.).
- **Voxel-to-Grid Projection:** It maps the segmented 3D points (X, Y, Z) into a rigid 2.5D Cartesian grid (`80m x 40m`).
- **Temporal Hysteresis Smoothing (10-Frame):** A rolling `deque` buffer acts as a geometric memory layer. By establishing a majority-vote over the last 10 frames, it completely stabilizes flickering edge-cases in the ML predictions, locking moving vehicles (dynamic) and buildings (static) to solid outputs.
- **WebSocket Emitter:** Streams highly compressed JSON polygon vertices rather than millions of raw points. Achieves ~99.8% memory savings over the wire.

### 2. The Tactical Dashboard (Next.js / DeckGL)
Located in the root directory, the frontend renders the data:
- **Deck.gl PolygonLayer:** The 3D grid data stream is injected into an Orbit-view canvas. Roads render as flat green slabs, while static objects (red) and dynamic threats (amber) are elevated proportionally to their true `Z-height` physics.
- **Live Telemetry:** Tracks rendering FPS, AI inference latency (in milliseconds), raw point count reduction, and overall network efficiency.
- **Threat Ticker:** A persistent HUD log that actively tracks and drops alerts when a dynamic object gets too close to the origin (the sensor car).

---

## 🚀 Setup & Execution

### Prerequisites
- Node.js 18+ (for frontend)
- Python 3.10+ (for backend)
- Nvidia GPU with CUDA support (highly recommended for PyTorch inference)

### 1. Download Dataset & Weights
Currently calibrated for **SemanticKITTI Sequence 00**. 
Navigate to the `backend` folder and run the provided scripts to fetch the dataset and exact trained PyTorch weights.
```bash
cd backend
# Refer to download_semantickitti_seq00.py instructions to get the dataset
```

### 2. Start the AI Server
Start the inference bridge on local port `8000`.
```bash
cd backend
python kaavach_bridge.py
```
*Note: The first launch takes a moment to convert `spconv v1` weights to `v2`, load them into VRAM, and cache the 4,500+ LIDAR frames.*

### 3. Start the Next.js Frontend
In a new terminal window:
```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend will automatically hook into `ws://localhost:8000/ws/stream_map`, stream the Cartesian grid, and render the map.

---

## 🛠️ Tech Stack
- **AI / Deep Learning:** PyTorch, Spconv v2.x, Cylinder3D Architecture
- **Backend Data processing:** NumPy, Math, AsyncIO, WebSockets (~5 FPS lock)
- **Frontend / UI:** Next.js (App Router), React, Tailwind CSS, Lucide-React
- **WebGL Rendering:** Deck.gl (`@deck.gl/react`, `@deck.gl/layers`), Math.js 
