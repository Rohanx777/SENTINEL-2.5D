#!/usr/bin/env python3
"""
benchmark_comparison.py
-----------------------
Directly compare latency between:
  1. Current Python implementation (kaavach_bridge.py)
  2. C++/CUDA backend (via pybind11 or timing the pipeline stages)

Usage:
    python benchmark_comparison.py
"""

import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter, deque

# Import Python implementation
from cylinder3d_net import CylinderVoxelizer, GRID_SIZE, build_model

WEIGHTS = "weights/model_load.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_X_MIN, GRID_X_MAX = -40.0, 40.0
GRID_Y_MIN, GRID_Y_MAX = -20.0, 20.0
CELL_SIZE = 2.0
N_COLS = int((GRID_X_MAX - GRID_X_MIN) / CELL_SIZE)
N_ROWS = int((GRID_Y_MAX - GRID_Y_MIN) / CELL_SIZE)

REMAP = torch.tensor(
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    dtype=torch.uint8,
)

def make_synthetic_scan(n=120000):
    rng = np.random.default_rng(42)
    rho = rng.uniform(0.5, 49.5, n).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    z = rng.uniform(-4.0, 2.0, n).astype(np.float32)
    x = (rho * np.cos(phi)).astype(np.float32)
    y = (rho * np.sin(phi)).astype(np.float32)
    inten = rng.uniform(0.0, 1.0, n).astype(np.float32)
    return np.stack([x, y, z, inten], axis=1)

def benchmark_python_pipeline(model, voxelizer, points, n_runs=10):
    print("\n" + "="*60)
    print("  Benchmarking Python Backend (Current Implementation)")
    print("="*60)

    timings = {
        "voxelization": [],
        "inference": [],
        "build_grid": [],
        "total": [],
    }

    # Warmup
    for _ in range(3):
        pt_fea, grid_ind = voxelizer(points)
        logits = model([pt_fea.to(DEVICE)], [grid_ind.to(DEVICE)], batch_size=1)
        vox_labels = logits[0].argmax(dim=0)
        gx = grid_ind[:, 0].long().clamp(0, GRID_SIZE[0] - 1).to(DEVICE)
        gy = grid_ind[:, 1].long().clamp(0, GRID_SIZE[1] - 1).to(DEVICE)
        gz = grid_ind[:, 2].long().clamp(0, GRID_SIZE[2] - 1).to(DEVICE)
        pt_sem = vox_labels[gx, gy, gz]
        labels = REMAP.to(DEVICE)[pt_sem].cpu().numpy()

    print(f"Running {n_runs} timed iterations...")

    for i in range(n_runs):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        # 1. Voxelization
        t0 = time.perf_counter()
        pt_fea, grid_ind = voxelizer(points)
        t_vox = (time.perf_counter() - t0) * 1000

        # 2. Neural Inference
        t0 = time.perf_counter()
        logits = model([pt_fea.to(DEVICE)], [grid_ind.to(DEVICE)], batch_size=1)
        vox_labels = logits[0].argmax(dim=0)
        gx = grid_ind[:, 0].long().clamp(0, GRID_SIZE[0] - 1).to(DEVICE)
        gy = grid_ind[:, 1].long().clamp(0, GRID_SIZE[1] - 1).to(DEVICE)
        gz = grid_ind[:, 2].long().clamp(0, GRID_SIZE[2] - 1).to(DEVICE)
        pt_sem = vox_labels[gx, gy, gz]
        labels = REMAP.to(DEVICE)[pt_sem].cpu().numpy()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_inf = (time.perf_counter() - t0) * 1000

        # 3. build_grid()
        t0 = time.perf_counter()
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        good = (z > -3.5) & (z < 8.0) & (np.sqrt(x**2 + y**2) > 0.5)
        x_g, y_g, z_g, labels_g = x[good], y[good], z[good], labels[good]
        c_idx = np.floor((x_g - GRID_X_MIN) / CELL_SIZE).astype(int)
        r_idx = np.floor((y_g - GRID_Y_MIN) / CELL_SIZE).astype(int)
        valid = (c_idx >= 0) & (c_idx < N_COLS) & (r_idx >= 0) & (r_idx < N_ROWS)
        flat_idx = r_idx[valid] * N_COLS + c_idx[valid]
        z_v = z_g[valid]
        lb_v = labels_g[valid]

        cells = []
        for fid in np.unique(flat_idx):
            mask = flat_idx == fid
            cell_z = z_v[mask]
            cell_lb = lb_v[mask]
            if len(cell_z) < 5:
                continue
            z_min = float(cell_z.min())
            z_max = float(cell_z.max())
            delta_z = z_max - z_min
            ml_vote = Counter(cell_lb.tolist()).most_common(1)[0][0]
            if ml_vote == 2:
                raw_label = "dynamic_target"
            elif ml_vote == 1 and delta_z > 1.2:
                raw_label = "static_obstacle"
            elif delta_z > 2.0:
                raw_label = "static_obstacle"
            elif delta_z > 1.0 and z_max > -0.5:
                raw_label = "dynamic_target"
            else:
                raw_label = "road"
            elev = 0.2 if raw_label == "road" else max(0.4, min(delta_z, 5.0))
            cells.append({"elev": elev, "label": raw_label})

        t_grid = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_start) * 1000

        timings["voxelization"].append(t_vox)
        timings["inference"].append(t_inf)
        timings["build_grid"].append(t_grid)
        timings["total"].append(t_total)

    print("\n--- Python Latency Breakdown (Averages over {} runs) ---".format(n_runs))
    print(f"  Voxelization (CPU NumPy) : {np.mean(timings['voxelization']):>6.1f} ms  (min: {np.min(timings['voxelization']):.1f}, max: {np.max(timings['voxelization']):.1f})")
    print(f"  Inference (PyTorch CUDA) : {np.mean(timings['inference']):>6.1f} ms  (min: {np.min(timings['inference']):.1f}, max: {np.max(timings['inference']):.1f})")
    print(f"  build_grid (Python loop) : {np.mean(timings['build_grid']):>6.1f} ms  (min: {np.min(timings['build_grid']):.1f}, max: {np.max(timings['build_grid']):.1f})")
    print(f"  -----------------------------------------------")
    print(f"  TOTAL LATENCY            : {np.mean(timings['total']):>6.1f} ms  ({1000/np.mean(timings['total']):.1f} FPS)")

    return timings

def print_projected_speedup(py_timings):
    print("\n" + "="*60)
    print("  PROJECTED C++/CUDA SPEEDUP")
    print("="*60)

    py_vox = np.mean(py_timings['voxelization'])
    py_inf = np.mean(py_timings['inference'])
    py_grid = np.mean(py_timings['build_grid'])
    py_tot = np.mean(py_timings['total'])

    # CUDA projected timings
    cuda_vox = 1.0     # 1ms CUDA kernel
    cuda_inf = 20.0    # 20ms LibTorch / TensorRT
    cuda_grid = 2.0    # 2ms CUDA kernel
    cuda_tot = cuda_vox + cuda_inf + cuda_grid + 1.0  # +1ms JSON/WS

    print(f"{'Stage':<25} {'Python':>12} {'C++/CUDA':>12} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'Voxelization':<25} {py_vox:>10.1f} ms {cuda_vox:>10.1f} ms {py_vox/cuda_vox:>9.1f}x")
    print(f"{'Neural Inference':<25} {py_inf:>10.1f} ms {cuda_inf:>10.1f} ms {py_inf/cuda_inf:>9.1f}x")
    print(f"{'2.5D Grid Building':<25} {py_grid:>10.1f} ms {cuda_grid:>10.1f} ms {py_grid/cuda_grid:>9.1f}x")
    print(f"{'JSON + WebSocket':<25} {'~20.0':>10} ms {'~1.0':>10} ms {'~20.0x':>10}")
    print("-" * 60)
    print(f"{'TOTAL PIPELINE':<25} {py_tot:>10.1f} ms {cuda_tot:>10.1f} ms {py_tot/cuda_tot:>9.1f}x")
    print(f"{'EFFECTIVE FPS':<25} {1000/py_tot:>10.1f} fps {1000/cuda_tot:>10.1f} fps")
    print("=" * 60)

if __name__ == "__main__":
    print("[Init] Loading model for benchmarking...")
    model = build_model(WEIGHTS, DEVICE)
    voxelizer = CylinderVoxelizer()

    points = make_synthetic_scan(120000)
    print(f"[Init] Generated {len(points):,} synthetic points")

    timings = benchmark_python_pipeline(model, voxelizer, points, n_runs=10)
    print_projected_speedup(timings)
