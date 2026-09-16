#!/usr/bin/env python3
"""
launch.py — SENTINEL-2.5D Universal Launcher (Cross-Platform)

Automatically manages:
1. Environment pre-flight checks (C++/CUDA vs Python vs Mock)
2. Starts the Backend Perception Engine (Port 8000)
3. Starts the Tactical Frontend Dashboard (Port 3000)
4. Opens browser automatically to http://localhost:3000
5. Graceful multi-process shutdown on Ctrl+C
"""

import os
import sys
import time
import signal
import shutil
import subprocess
import webbrowser
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_CPP_DIR = ROOT_DIR / "backend" / "cpp_cuda"
BACKEND_PY_DIR = ROOT_DIR / "backend" / "python_legacy"
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Locate Python binary with priority for virtual environments
possible_venvs = [
    ROOT_DIR.parent / ".venv" / "Scripts" / "python.exe",
    ROOT_DIR / ".venv" / "Scripts" / "python.exe",
    ROOT_DIR.parent / ".venv" / "bin" / "python",
    ROOT_DIR / ".venv" / "bin" / "python",
]
VENV_PYTHON = next((p for p in possible_venvs if p.exists()), Path(sys.executable))

processes = []

def cleanup(sig=None, frame=None):
    print("\n\n[~] Shutting down SENTINEL-2.5D services...")
    for proc in processes:
        if proc.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                else:
                    proc.terminate()
            except Exception:
                pass
    print("[OK] All services stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def free_port_if_needed(port: int):
    """Ensure port is available before starting services."""
    if sys.platform == "win32":
        try:
            res = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True,
                capture_output=True,
                text=True
            )
            for line in res.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in parts:
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != os.getpid():
                        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        except Exception:
            pass

def check_frontend_deps():
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("[!] Installing frontend dependencies (npm install)...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.run([npm_cmd, "install"], cwd=FRONTEND_DIR, check=True)

def find_backend_mode():
    """Determine best backend mode available."""
    # Check for compiled C++ executable
    cpp_exe_release = BACKEND_CPP_DIR / "build" / "Release" / ("sentinel_bridge.exe" if sys.platform == "win32" else "sentinel_bridge")
    cpp_exe_direct = BACKEND_CPP_DIR / "build" / ("sentinel_bridge.exe" if sys.platform == "win32" else "sentinel_bridge")

    if cpp_exe_release.exists():
        return ("cpp", cpp_exe_release)
    if cpp_exe_direct.exists():
        return ("cpp", cpp_exe_direct)

    # Check for weights/scans for Python mode
    py_bridge = BACKEND_PY_DIR / "sentinel_bridge.py"
    weights_pth = ROOT_DIR / "weights" / "model_load.pth"
    scans_dir = ROOT_DIR / "data" / "sequences" / "00" / "velodyne"
    if py_bridge.exists() and weights_pth.exists() and scans_dir.exists():
        return ("python", py_bridge)

    # Fallback to high-fidelity synthetic mock server (guaranteed to run)
    mock_server = SCRIPTS_DIR / "mock_server.py"
    return ("mock", mock_server)

def main():
    print("=" * 65)
    print("  [+] SENTINEL-2.5D - Defense Tactical Perception System")
    print("  Strategic Environment Navigation & Threat Intelligence")
    print("=" * 65)

    # Free ports 8005 and 3000 if hung instances exist
    free_port_if_needed(8005)
    free_port_if_needed(3000)

    mode, backend_target = find_backend_mode()
    print(f"\n[1/3] Backend Engine Selected: [{mode.upper()}] -> {backend_target.name}")

    print("[2/3] Checking Frontend Environment...")
    check_frontend_deps()

    print("[3/3] Starting Tactical Services...")

    # Launch backend
    backend_env = os.environ.copy()
    backend_env["PYTHONPATH"] = str(ROOT_DIR / "backend" / "python_legacy")
    if mode == "cpp":
        print("  -> Starting C++/CUDA Perception Pipeline (Sub-30ms)...")
        backend_proc = subprocess.Popen([str(backend_target)], cwd=backend_target.parent, env=backend_env)
    elif mode == "python":
        print("  -> Starting Python Reference Pipeline...")
        backend_proc = subprocess.Popen([str(VENV_PYTHON), str(backend_target)], cwd=BACKEND_PY_DIR, env=backend_env)
    else:
        print("  -> Starting Synthetic LiDAR Mock Server (ws://localhost:8005)...")
        backend_proc = subprocess.Popen([str(VENV_PYTHON), str(backend_target)], cwd=SCRIPTS_DIR, env=backend_env)
    processes.append(backend_proc)

    time.sleep(1)

    # Launch frontend
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    print("  -> Starting Tactical WebGL Dashboard on http://localhost:3000 ...")
    frontend_proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=FRONTEND_DIR, env=backend_env)
    processes.append(frontend_proc)

    time.sleep(2)

    print("\n" + "=" * 65)
    print("  [OK] SENTINEL-2.5D IS ACTIVE AND OPERATIONAL")
    print("  * WebGL Dashboard:  http://localhost:3000")
    print("  * Stream Protocol:  ws://localhost:8005/ws/stream_map")
    print("  * Press Ctrl+C in this terminal to terminate all processes.")
    print("=" * 65 + "\n")

    # Auto open browser
    try:
        webbrowser.open("http://localhost:3000")
    except Exception:
        pass

    # Wait for processes
    try:
        while True:
            if backend_proc.poll() is not None and frontend_proc.poll() is not None:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
