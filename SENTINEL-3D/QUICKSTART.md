# 🚀 SENTINEL-3D Quick Start Guide

**One-Command Launch for the Entire System**

---

## ⚡ Fastest Way to Launch (Recommended)

### Option 1: Python Universal Launcher (All Platforms)

```bash
python launch.py
```

**What it does:**
- ✅ Auto-detects best backend mode (C++/CUDA → Python → Mock)
- ✅ Checks and installs frontend dependencies
- ✅ Starts backend on `ws://localhost:8000`
- ✅ Starts frontend on `http://localhost:3000`
- ✅ **Opens browser automatically**
- ✅ Graceful shutdown with `Ctrl+C`

---

### Option 2: Platform-Specific Launchers

**Windows:**
```cmd
start_sentinel.bat
```

**Linux/macOS:**
```bash
./start_sentinel.sh
```

**From Frontend Directory:**
```bash
cd frontend
npm run sentinel
```

---

## 📋 What Happens During Launch?

1. **Pre-flight Checks:**
   - Verifies TorchScript model exists (`backend/cpp_cuda/weights/model_load.pt`)
   - Checks if C++ backend is compiled
   - Installs frontend dependencies if needed

2. **Backend Launch** (Port 8000):
   - **Priority 1**: C++/CUDA `sentinel_bridge` (25ms latency, 40 FPS)
   - **Priority 2**: Python `sentinel_bridge.py` (300ms latency, 3 FPS)
   - **Priority 3**: Mock server (synthetic data for frontend testing)

3. **Frontend Launch** (Port 3000):
   - Next.js development server with Deck.gl WebGL rendering
   - WebSocket connects to backend automatically

4. **Browser Opens**:
   - Dashboard accessible at `http://localhost:3000`

---

## 🛠️ Manual Step-by-Step (If Needed)

### First-Time Setup

1. **Build C++ Backend** (one-time):
   ```bash
   cd backend/cpp_cuda
   ./build.sh       # Linux/macOS
   build.bat        # Windows
   ```

2. **Export Model** (one-time):
   ```bash
   python scripts/export_model_to_torchscript.py
   ```

3. **Install Frontend**:
   ```bash
   cd frontend
   npm install
   ```

### Running Manually (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd backend/cpp_cuda/build
./sentinel_bridge          # Linux
.\Release\sentinel_bridge.exe   # Windows
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Open browser: `http://localhost:3000`

---

## 🎯 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | `http://localhost:3000` | Tactical WebGL interface |
| **WebSocket** | `ws://localhost:8000/ws/stream_map` | Real-time LiDAR stream |
| **Backend Logs** | Terminal output | Latency, FPS, telemetry |

---

## 🔧 Troubleshooting

### Issue: "Model not found"
**Fix:** Run model export:
```bash
python scripts/export_model_to_torchscript.py
```

### Issue: "sentinel_bridge not found"
**Fix:** Build the C++ backend:
```bash
cd backend/cpp_cuda
./build.sh  # or build.bat on Windows
```

### Issue: "Cannot find module 'next'"
**Fix:** Install frontend dependencies:
```bash
cd frontend
npm install
```

### Issue: Browser doesn't open automatically
**Fix:** Manually navigate to `http://localhost:3000`

---

## 🎮 Command Reference

| Command | Description |
|---------|-------------|
| `python launch.py` | **Universal launcher** (auto-detects environment) |
| `start_sentinel.bat` | Windows batch launcher |
| `./start_sentinel.sh` | Linux/macOS shell launcher |
| `cd frontend && npm run sentinel` | Launch from npm scripts |
| `Ctrl+C` | Stop all services gracefully |

---

## 📊 Expected Performance

| Backend Mode | Latency | FPS | Use Case |
|--------------|---------|-----|----------|
| **C++/CUDA** | ~25ms | 40 | Production, DRDO demo |
| **Python** | ~300ms | 3 | Development, debugging |
| **Mock** | ~50ms | 20 | Frontend testing (no GPU) |

---

## 🛡️ For DRDO Presentation

1. Ensure you're using **C++/CUDA backend** for optimal performance
2. Run `python launch.py` at least once before the demo to verify setup
3. The launcher automatically opens the dashboard — ready to present in seconds

---

**Last Updated:** September 2026  
**System:** SENTINEL-3D v1.0.0  
**Status:** Production-Ready
