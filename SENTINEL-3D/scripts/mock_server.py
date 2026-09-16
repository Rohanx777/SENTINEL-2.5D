#!/usr/bin/env python3
"""
mock_server.py — SENTINEL-3D Multi-Resolution Foveated Perception Server
=======================================================================
Streams real-time LiDAR perception data with:
  1. Adaptive Foveated Multi-Resolution Grid (0-10m high-res, 10-30m med-res, 30-100m coarse-res).
  2. Micro-Terrain & Negative Obstacle Analysis (Potholes, Curbs, Slopes).
  3. Dynamic Threat Kinematics & Time-to-Collision (TTC) Tracking.
  4. Distance-Stratified Performance Benchmarks & Tactical Radio Link Simulator.
  5. Motion-Compensated World vs Sensor Coordinates.

Streams to: ws://0.0.0.0:8005/ws/stream_map
"""

import asyncio
import json
import math
import random
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "python_legacy"))

from config_loader import SentinelConfig
from coordinate_transformer import (
    CoordinateTransformer,
    OdometryPose,
    PoseManager,
    make_linear_cartesian_box,
)
from comparative_analyzer import ComparativeMetrics, sample_point_cloud_3d
from foveated_grid_engine import FoveatedGridEngine, ThreatKinematicTracker

try:
    import websockets
except ImportError:
    print("[ERROR] websockets not installed. Run: pip install websockets")
    sys.exit(1)


config = SentinelConfig()

HOST = "0.0.0.0"
PORT = 8005
ROUTE = "/ws/stream_map"
PLAYBACK_SPEED = 2.0
FRAME_STRIDE = 2
TARGET_FPS = config.target_fps
FRAME_INTERVAL = 1.0 / TARGET_FPS

CLIENTS = set()

# Initialize Pose Manager, Foveated Engine, and Threat Tracker
poses_file = Path("data/sequences/00/poses.txt")
pose_manager = PoseManager(poses_file if poses_file.exists() else None)
foveated_engine = FoveatedGridEngine(history_len=10)
threat_tracker = ThreatKinematicTracker()

# ─────────────────────────────────────────────────────────────────────────────
# Ground-Truth World Environment (Static Buildings, Trees, Potholes)
# ─────────────────────────────────────────────────────────────────────────────
def generate_world_environment():
    static_buildings = []
    for x_center in range(-100, 600, 35):
        y_north = random.uniform(22.0, 38.0)
        static_buildings.append({
            "type": "building",
            "x0": x_center - 10.0, "x1": x_center + 10.0,
            "y0": y_north, "y1": y_north + 18.0,
            "z_min": -0.2, "z_max": random.uniform(4.0, 8.0),
            "label": "static_obstacle",
        })
        y_south = -random.uniform(22.0, 38.0)
        static_buildings.append({
            "type": "building",
            "x0": x_center - 10.0, "x1": x_center + 10.0,
            "y0": y_south - 18.0, "y1": y_south,
            "z_min": -0.2, "z_max": random.uniform(4.0, 7.0),
            "label": "static_obstacle",
        })

    # Fixed trees
    static_trees = []
    for x_pos in range(-80, 580, 16):
        static_trees.append({"x": x_pos, "y": 13.5, "radius": 2.2, "z_max": 3.8})
        static_trees.append({"x": x_pos, "y": -13.5, "radius": 2.2, "z_max": 3.8})

    # Negative obstacles (potholes / ditches / trench hazards in world coordinates)
    world_potholes = [
        {"x": 14.0, "y": 1.2, "radius": 1.4, "depth": 0.35},
        {"x": 38.0, "y": -1.8, "radius": 1.6, "depth": 0.42},
        {"x": 75.0, "y": 0.8, "radius": 1.5, "depth": 0.38},
        {"x": 120.0, "y": -1.2, "radius": 2.2, "depth": 0.55},
        {"x": 180.0, "y": 1.5, "radius": 1.8, "depth": 0.48},
        {"x": 260.0, "y": -0.8, "radius": 2.5, "depth": 0.60},
        {"x": 350.0, "y": 0.5, "radius": 2.0, "depth": 0.50},
        {"x": 450.0, "y": -1.5, "radius": 2.2, "depth": 0.52},
    ]

    return static_buildings, static_trees, world_potholes


WORLD_BUILDINGS, WORLD_TREES, WORLD_POTHOLES = generate_world_environment()


def generate_foveated_synthetic_frame(frame_idx: int):
    """Generate high-fidelity foveated LiDAR frame with micro-hazards & kinematics."""
    pose = pose_manager.get_pose(frame_idx, dt=FRAME_INTERVAL)
    ego_x = pose.x
    ego_y = pose.y
    ego_speed_mps = 10.0 * (PLAYBACK_SPEED / 2.0)

    # 1. Synthesize 3D Points based on World Map & Diverse Dynamic Entities
    t = frame_idx * 0.05 * (PLAYBACK_SPEED / 2.0)

    # Moving Threats (in world coordinate system)
    # A. Forward Lead Armored Convoy
    d1_world_x = ego_x + 22.0 + 8.0 * math.sin(t * 0.3)
    d1_world_y = ego_y + 2.0 + 0.5 * math.sin(t * 0.6)
    d1_vx, d1_vy = -1.5 * (PLAYBACK_SPEED / 2.0), 0.2

    # B. High-Speed Intercepting Hostile Vehicle (Approaching ego vehicle)
    d2_world_x = ego_x + 65.0 - (frame_idx * 1.4 * (PLAYBACK_SPEED / 2.0)) % 140.0
    d2_world_y = ego_y - 2.2
    d2_vx, d2_vy = -14.0 * (PLAYBACK_SPEED / 2.0), 0.0

    # C. Dismounted Recon Patrol Squad (Lateral crossing)
    d3_world_x = ego_x + 15.0 + 2.0 * math.sin(t * 0.5)
    d3_world_y = ego_y + 6.0 - (t * 2.2) % 14.0
    d3_vx, d3_vy = 0.0, -2.2 * (PLAYBACK_SPEED / 2.0)

    # D. Tactical Recon Drone / UAV (Overhead hovering / tracking)
    d4_world_x = ego_x + 12.0 + 10.0 * math.cos(t * 0.4)
    d4_world_y = ego_y - 5.5 + 5.0 * math.sin(t * 0.4)
    d4_vx, d4_vy = -4.0 * math.sin(t * 0.4), 2.0 * math.cos(t * 0.4)

    # E. Flanking Light Tactical Patrol (Overtaking on lateral side)
    d5_world_x = ego_x - 30.0 + (frame_idx * 1.8 * (PLAYBACK_SPEED / 2.0)) % 130.0
    d5_world_y = ego_y + 6.5
    d5_vx, d5_vy = 8.0 * (PLAYBACK_SPEED / 2.0), 0.0

    # F. Perimeter Barricade Checkpoint Threat
    d6_world_x = ego_x + 45.0 - (frame_idx * 0.5 * (PLAYBACK_SPEED / 2.0)) % 90.0
    d6_world_y = ego_y + 1.2
    d6_vx, d6_vy = -5.0 * (PLAYBACK_SPEED / 2.0), 0.1

    dynamic_entities = [
        {"x": d1_world_x, "y": d1_world_y, "vx": d1_vx, "vy": d1_vy, "type": "ARMORED_LEAD_CONVOY", "radius": 2.8, "z_min": -0.2, "z_max": 2.2, "label_id": 2},
        {"x": d2_world_x, "y": d2_world_y, "vx": d2_vx, "vy": d2_vy, "type": "INTERCEPTING_HOSTILE_VEHICLE", "radius": 2.2, "z_min": -0.2, "z_max": 1.8, "label_id": 2},
        {"x": d3_world_x, "y": d3_world_y, "vx": d3_vx, "vy": d3_vy, "type": "DISMOUNTED_SCOUT_SQUAD", "radius": 1.2, "z_min": -0.2, "z_max": 1.7, "label_id": 2},
        {"x": d4_world_x, "y": d4_world_y, "vx": d4_vx, "vy": d4_vy, "type": "TACTICAL_DRONE_UAV", "radius": 1.0, "z_min": 1.5, "z_max": 3.2, "label_id": 2},
        {"x": d5_world_x, "y": d5_world_y, "vx": d5_vx, "vy": d5_vy, "type": "FLANKING_TACTICAL_PATROL", "radius": 2.4, "z_min": -0.2, "z_max": 2.0, "label_id": 2},
        {"x": d6_world_x, "y": d6_world_y, "vx": d6_vx, "vy": d6_vy, "type": "HOSTILE_TECHNICAL_TRUCK", "radius": 2.5, "z_min": -0.2, "z_max": 2.1, "label_id": 2},
    ]

    # Sensor frame coordinates of dynamic entities
    pose_inv = pose.inverse()
    r_inv = pose_inv.rotation[:2, :2]
    t_inv = pose_inv.translation[:2]

    raw_threats = []
    for ent in dynamic_entities:
        p_w = np.array([ent["x"], ent["y"]])
        p_s = r_inv @ p_w + t_inv
        dist = math.sqrt(p_s[0]**2 + p_s[1]**2)
        if dist <= 100.0:
            raw_threats.append({
                "type": ent["type"],
                "distance_m": round(dist, 1),
                "coordinates": [round(float(p_s[0]), 1), round(float(p_s[1]), 1)],
                "world_coordinates": [round(ent["x"], 1), round(ent["y"], 1)],
                "vx": ent["vx"],
                "vy": ent["vy"],
            })

    # Track kinematics & calculate TTC
    enhanced_threats = threat_tracker.update_threats(raw_threats, ego_speed_mps=ego_speed_mps, dt=FRAME_INTERVAL)

    # 2. Synthesize Dense 3D Point Cloud (~120,000 pts)
    points_list = []
    labels_list = []

    # A. Safety Core (0.5 - 10m): High Density (~50,000 pts)
    n_core = 50000
    r_core = np.sqrt(np.random.uniform(0.5**2, 10.0**2, n_core))
    phi_core = np.random.uniform(-math.pi, math.pi, n_core)
    xs_core = r_core * np.cos(phi_core)
    ys_core = r_core * np.sin(phi_core)

    # Transform to world coords to check potholes/curbs
    p_s_core = np.stack([xs_core, ys_core], axis=1)
    p_w_core = (pose.rotation[:2, :2] @ p_s_core.T).T + pose.translation[:2]

    # Ground base Z around -1.6m
    zs_core = np.random.normal(-1.6, 0.03, n_core)
    lbl_core = np.zeros(n_core, dtype=np.uint8)

    # Carve out negative obstacles (potholes) in world space
    for pot in WORLD_POTHOLES:
        d_pot = np.sqrt((p_w_core[:, 0] - pot["x"])**2 + (p_w_core[:, 1] - pot["y"])**2)
        pot_mask = d_pot < pot["radius"]
        zs_core[pot_mask] -= pot["depth"]  # Drop down below road plane

    # Curbs along road edge (y ~ +/- 4.5m)
    curb_mask = (np.abs(ys_core) > 4.2) & (np.abs(ys_core) < 4.8)
    zs_core[curb_mask] += np.random.uniform(0.12, 0.18, np.sum(curb_mask))

    # B. Tactical Corridor (10 - 30m): Medium Density (~45,000 pts)
    n_mid = 45000
    r_mid = np.sqrt(np.random.uniform(10.0**2, 30.0**2, n_mid))
    phi_mid = np.random.uniform(-math.pi, math.pi, n_mid)
    xs_mid = r_mid * np.cos(phi_mid)
    ys_mid = r_mid * np.sin(phi_mid)
    zs_mid = np.random.normal(-1.6, 0.06, n_mid)
    lbl_mid = np.zeros(n_mid, dtype=np.uint8)

    p_s_mid = np.stack([xs_mid, ys_mid], axis=1)
    p_w_mid = (pose.rotation[:2, :2] @ p_s_mid.T).T + pose.translation[:2]

    # Add static obstacle heights (trees, buildings)
    for bldg in WORLD_BUILDINGS:
        mask_b = (p_w_mid[:, 0] >= bldg["x0"]) & (p_w_mid[:, 0] <= bldg["x1"]) & (p_w_mid[:, 1] >= bldg["y0"]) & (p_w_mid[:, 1] <= bldg["y1"])
        zs_mid[mask_b] = np.random.uniform(0.0, bldg["z_max"], np.sum(mask_b))
        lbl_mid[mask_b] = 1

    # Add dynamic vehicles in mid-zone
    for ent in dynamic_entities:
        d_ent = np.sqrt((p_w_mid[:, 0] - ent["x"])**2 + (p_w_mid[:, 1] - ent["y"])**2)
        mask_d = d_ent < ent["radius"]
        zs_mid[mask_d] = np.random.uniform(-1.4, ent["z_max"], np.sum(mask_d))
        lbl_mid[mask_d] = 2

    # C. Perimeter Surveillance (30 - 100m): Coarse Density (~25,000 pts)
    n_far = 25000
    r_far = np.sqrt(np.random.uniform(30.0**2, 100.0**2, n_far))
    phi_far = np.random.uniform(-math.pi, math.pi, n_far)
    xs_far = r_far * np.cos(phi_far)
    ys_far = r_far * np.sin(phi_far)
    zs_far = np.random.normal(-1.6, 0.12, n_far)
    lbl_far = np.zeros(n_far, dtype=np.uint8)

    p_s_far = np.stack([xs_far, ys_far], axis=1)
    p_w_far = (pose.rotation[:2, :2] @ p_s_far.T).T + pose.translation[:2]
    for bldg in WORLD_BUILDINGS:
        mask_bf = (p_w_far[:, 0] >= bldg["x0"]) & (p_w_far[:, 0] <= bldg["x1"]) & (p_w_far[:, 1] >= bldg["y0"]) & (p_w_far[:, 1] <= bldg["y1"])
        zs_far[mask_bf] = np.random.uniform(0.0, bldg["z_max"], np.sum(mask_bf))
        lbl_far[mask_bf] = 1

    # Combine all points into (N, 4)
    all_x = np.concatenate([xs_core, xs_mid, xs_far])
    all_y = np.concatenate([ys_core, ys_mid, ys_far])
    all_z = np.concatenate([zs_core, zs_mid, zs_far])
    all_i = np.random.uniform(0.2, 0.9, len(all_x))
    all_lbl = np.concatenate([lbl_core, lbl_mid, lbl_far])

    raw_points_np = np.stack([all_x, all_y, all_z, all_i], axis=1).astype(np.float32)

    # 3. Project into Foveated Multi-Resolution 2.5D Grid Engine
    foveated_cells, stratified_metrics = foveated_engine.project_point_cloud(raw_points_np, all_lbl, pose)

    # 4. Generate Linear Cartesian fallback grid
    linear_cells = []
    x_min, x_max, y_min, y_max = config.cartesian_bounds
    cell_size = config.cartesian_cell_size
    n_cols = int((x_max - x_min) / cell_size)
    n_rows = int((y_max - y_min) / cell_size)

    for r_i in range(n_rows):
        y0_s = y_min + r_i * cell_size
        y1_s = y0_s + cell_size
        cy_s = (y0_s + y1_s) / 2.0
        for c_i in range(n_cols):
            x0_s = x_min + c_i * cell_size
            x1_s = x0_s + cell_size
            cx_s = (x0_s + x1_s) / 2.0
            r_c = math.sqrt(cx_s**2 + cy_s**2)
            if r_c > 45.0:
                continue

            p_s_vec = np.array([cx_s, cy_s])
            p_w_vec = pose.rotation[:2, :2] @ p_s_vec + pose.translation[:2]
            cx_w, cy_w = float(p_w_vec[0]), float(p_w_vec[1])

            is_dynamic = any(math.sqrt((cx_s - th["coordinates"][0])**2 + (cy_s - th["coordinates"][1])**2) < 3.0 for th in raw_threats)
            is_static = any(b["x0"] <= cx_w <= b["x1"] and b["y0"] <= cy_w <= b["y1"] for b in WORLD_BUILDINGS)
            label = "dynamic_target" if is_dynamic else ("static_obstacle" if is_static else "road")
            elev = 2.0 if is_dynamic else (4.0 if is_static else 0.2)

            box_s = make_linear_cartesian_box(x0_s, y0_s, x1_s, y1_s)
            box_w = CoordinateTransformer.transform_cell_polygon(box_s, pose, to_world=True)
            linear_cells.append({
                "polygon": box_s,
                "world_polygon": box_w,
                "elev": elev,
                "delta_z": elev,
                "label": label,
                "cx": round(cx_s, 2), "cy": round(cy_s, 2),
                "world_cx": round(cx_w, 2), "world_cy": round(cy_w, 2),
                "r": round(r_c, 2),
                "azimuth_deg": round((math.degrees(math.atan2(cy_s, cx_s)) + 360.0) % 360.0, 1),
            })

    # 5. Sample 3D Points for direct 3D point cloud rendering
    pointcloud_sample_3d = sample_point_cloud_3d(raw_points_np, all_lbl, max_sample_points=2500)

    # 6. Compute 3D vs 2.5D Comparative Study Metrics
    comp_study = ComparativeMetrics.compute(
        points_3d=raw_points_np,
        cells_25d=foveated_cells,
        inference_time_ms=21.8,
        compression_time_ms=1.9,
    )

    ego_state = {
        "pose": {
            "x": round(ego_x, 2),
            "y": round(ego_y, 2),
            "z": round(pose.z, 2),
            "yaw_deg": round(pose.yaw_deg, 2),
        },
        "speed_kmh": round(ego_speed_mps * 3.6, 1),
        "frame_id": frame_idx,
        "playback_speed": PLAYBACK_SPEED,
    }

    return foveated_cells, linear_cells, pointcloud_sample_3d, ego_state, comp_study, enhanced_threats, stratified_metrics


async def handle_client_commands(ws):
    global PLAYBACK_SPEED, FRAME_STRIDE, TARGET_FPS, FRAME_INTERVAL
    try:
        async for message in ws:
            try:
                data = json.loads(message)
                action = data.get("action")
                if action == "set_speed":
                    speed = float(data.get("speed", 2.0))
                    PLAYBACK_SPEED = max(0.5, min(16.0, speed))
                    FRAME_STRIDE = max(1, int(round(PLAYBACK_SPEED)))
                    print(f"[SpeedControl] Speed updated: {PLAYBACK_SPEED}x (stride={FRAME_STRIDE})")
                elif action == "set_stride":
                    stride = int(data.get("stride", 2))
                    FRAME_STRIDE = max(1, min(12, stride))
                elif action == "set_fps":
                    fps = int(data.get("fps", 30))
                    TARGET_FPS = max(10, min(60, fps))
                    FRAME_INTERVAL = 1.0 / TARGET_FPS
            except Exception as e:
                print(f"[WARN] Client command error: {e}")
    except websockets.ConnectionClosed:
        pass


async def stream_handler(ws):
    CLIENTS.add(ws)
    print(f"[+] Client Connected: {ws.remote_address}")
    cmd_task = asyncio.create_task(handle_client_commands(ws))
    try:
        await ws.wait_closed()
    finally:
        cmd_task.cancel()
        CLIENTS.discard(ws)
        print(f"[-] Client Disconnected: {ws.remote_address}")


async def broadcast_loop():
    global PLAYBACK_SPEED, FRAME_STRIDE, TARGET_FPS, FRAME_INTERVAL
    tick = 0
    raw_points_total = 120000

    while True:
        t0 = time.perf_counter()
        tick += FRAME_STRIDE

        foveated_cells, linear_cells, pointcloud_3d, ego_state, comp_study, threats, stratified_metrics = generate_foveated_synthetic_frame(tick)
        latency_ms = round(random.uniform(21.5, 24.8), 1)

        payload = {
            "header": {
                "frame_id": tick,
                "timestamp": time.time(),
                "active_engine": "CUDA_TIER_1",
                "foveated_zones_active": 3,
                "representation_modes": ["foveated_polar", "linear_grid", "pointcloud_3d", "comparison"],
            },
            "telemetry": {
                "fps": TARGET_FPS,
                "latency_ms": latency_ms,
                "raw_points_count": raw_points_total,
                "compressed_cells_count": len(foveated_cells),
                "memory_saved_percent": comp_study["summary"]["compression_ratio_pct"],
                "speed_kmh": ego_state["speed_kmh"],
                "playback_speed": PLAYBACK_SPEED,
            },
            "ego_vehicle": ego_state,
            "grid_data": foveated_cells,
            "foveated_grid": foveated_cells,
            "polar_grid": foveated_cells,
            "linear_grid": linear_cells,
            "pointcloud_3d": pointcloud_3d,
            "comparative_study": comp_study,
            "stratified_metrics": stratified_metrics,
            "threats": threats,
        }

        msg = json.dumps(payload)
        if CLIENTS:
            await asyncio.gather(
                *[ws.send(msg) for ws in list(CLIENTS)],
                return_exceptions=True,
            )

        elapsed = time.perf_counter() - t0
        target_delay = (1.0 / max(10, TARGET_FPS)) / max(1.0, PLAYBACK_SPEED / 2.0)
        await asyncio.sleep(max(0.001, target_delay - elapsed))


async def main():
    print("=" * 68)
    print("  SENTINEL-3D Adaptive Foveated Perception Server (DRDO Edition)")
    print(f"  WebSocket: ws://{HOST}:{PORT}{ROUTE}")
    print(f"  Target FPS: {TARGET_FPS} | Sub-25ms Real-Time Inference")
    print("  Capabilities:")
    print("    [+] Foveated Non-Uniform Grid (Zone 0: 0-10m, Zone 1: 10-30m, Zone 2: 30-100m)")
    print("    [+] Micro-Terrain Analyzer: Negative Obstacles, Potholes & Curbs")
    print("    [+] Threat Kinematics Tracker & Time-to-Collision (TTC) Engine")
    print("    [+] Distance-Stratified Benchmarks & Tactical Radio Link Simulator")
    print("    [+] SE(3) Motion-Compensated World & Sensor Frames")
    print("=" * 68)

    async with websockets.serve(stream_handler, HOST, PORT):
        print(f"[OK] Server streaming on port {PORT}...")
        await broadcast_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[~] Server stopped.")
