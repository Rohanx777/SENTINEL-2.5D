#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dl_inference.py
---------------
ROS 2 Humble node — real-time LiDAR semantic segmentation via Cylinder3D.

Topic I/O
---------
  SUB  /lidar/points_raw         sensor_msgs/PointCloud2
  PUB  /lidar/points_classified  sensor_msgs/PointCloud2
         Fields: x (float32), y (float32), z (float32),
                 intensity (float32), label (uint8)
         Label values:
           0 → Terrain
           1 → Static Obstacle
           2 → Dynamic Entity

Standalone test (no ROS 2 required)
------------------------------------
  python dl_inference.py --test data/sequences/00/velodyne/000000.bin
  python dl_inference.py --test path/to/scan.bin --weights weights/model_load.pth
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Cylinder3D model + voxeliser (companion file) ──
from cylinder3d_net import (  # type: ignore[import]  # pyrefly: ignore
    CylinderVoxelizer,
    GRID_SIZE,
    build_model,
)

# ---------------------------------------------------------------------------
# SemanticKITTI 20-class → 3-class remapping
# ---------------------------------------------------------------------------
# fmt: off
#   SemanticKITTI learning index  →  target class
#   0  unlabeled          → 0 Terrain         (safe default)
#   1  car                → 2 Dynamic Entity
#   2  bicycle            → 2 Dynamic Entity
#   3  motorcycle         → 2 Dynamic Entity
#   4  truck              → 2 Dynamic Entity
#   5  other-vehicle      → 2 Dynamic Entity
#   6  person             → 2 Dynamic Entity
#   7  bicyclist          → 2 Dynamic Entity
#   8  motorcyclist       → 2 Dynamic Entity
#   9  road               → 0 Terrain
#   10 parking            → 0 Terrain
#   11 sidewalk           → 0 Terrain
#   12 other-ground       → 0 Terrain
#   13 building           → 1 Static Obstacle
#   14 fence              → 1 Static Obstacle
#   15 vegetation         → 0 Terrain
#   16 trunk              → 1 Static Obstacle
#   17 terrain            → 0 Terrain
#   18 pole               → 1 Static Obstacle
#   19 traffic-sign       → 1 Static Obstacle
_REMAP_TABLE = torch.tensor(
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    dtype=torch.uint8,
)
TARGET_NAMES = {0: "Terrain", 1: "Static Obstacle", 2: "Dynamic Entity"}
# fmt: on


# ---------------------------------------------------------------------------
# PointCloud2 helpers (pure Python, no ROS 2 Python extras needed)
# ---------------------------------------------------------------------------

# PointCloud2 field data-type codes → struct format char + byte size
_PC2_DTYPE = {
    1: ("b", 1), 2: ("B", 1),
    3: ("h", 2), 4: ("H", 2),
    5: ("i", 4), 6: ("I", 4),
    7: ("f", 4), 8: ("d", 8),
}


def _pc2_to_numpy(msg) -> np.ndarray:
    """
    Convert a sensor_msgs/PointCloud2 message to a (N, 4) float32 numpy
    array containing [x, y, z, intensity]. Handles both dense and organised
    clouds and arbitrary field offsets.
    """
    field_map  = {f.name: f for f in msg.fields}
    n_points   = msg.width * msg.height
    point_step = msg.point_step
    raw        = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    result = np.zeros((n_points, 4), dtype=np.float32)
    names  = ["x", "y", "z", "intensity"]

    for col, name in enumerate(names):
        if name not in field_map:
            continue
        f        = field_map[name]
        fmt, sz  = _PC2_DTYPE.get(f.datatype, ("f", 4))
        col_raw  = np.lib.stride_tricks.as_strided(
            raw[f.offset:],
            shape=(n_points, sz),
            strides=(point_step, 1),
        ).copy()
        result[:, col] = np.frombuffer(col_raw.tobytes(), dtype=np.dtype(fmt)).astype(np.float32)

    return result


def _numpy_to_pc2(header, points: np.ndarray, labels: np.ndarray):
    """
    Pack (N, 4) [x, y, z, intensity] + (N,) uint8 labels into a
    sensor_msgs/PointCloud2 with an added `label` field.
    """
    from sensor_msgs.msg import PointCloud2, PointField  # type: ignore[import]
    import array as arr

    n = len(points)

    fields = [
        PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="label",     offset=16, datatype=PointField.UINT8,   count=1),
    ]
    point_step = 17
    row_step   = point_step * n

    buf = bytearray(row_step)
    pts_f32 = points.astype(np.float32)
    lbl_u8  = labels.astype(np.uint8)

    for i in range(n):
        base = i * point_step
        struct.pack_into("4f", buf, base,
                         pts_f32[i, 0], pts_f32[i, 1],
                         pts_f32[i, 2], pts_f32[i, 3])
        struct.pack_into("B", buf, base + 16, int(lbl_u8[i]))

    msg             = PointCloud2()
    msg.header      = header
    msg.height      = 1
    msg.width       = n
    msg.fields      = fields
    msg.is_bigendian = False
    msg.point_step  = point_step
    msg.row_step    = row_step
    msg.is_dense    = True
    msg.data        = arr.array("B", buf)
    return msg


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------
class Inferencer:
    def __init__(self, checkpoint: str, device: torch.device):
        self.device    = device
        self.voxelizer = CylinderVoxelizer()
        
        # -- SpConv 1.x to 2.x Weight Patching --
        # Intercept and patch the checkpoint file dynamically so cylinder3d_net.py loads it seamlessly
        ckpt_path = Path(checkpoint)
        patched_ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_spconv2{ckpt_path.suffix}")
        
        if not patched_ckpt_path.exists():
            print(f"[INFO] Patching SpConv 1.x weights for SpConv 2.x compatibility on {device}...")
            raw_ckpt = torch.load(str(ckpt_path), map_location="cpu")
            state_dict = raw_ckpt.get("state_dict", raw_ckpt)
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if v.dim() == 5:
                    # SpConv 1.x: (D, H, W, Cin, Cout) -> SpConv 2.x: (Cout, D, H, W, Cin)
                    new_state_dict[k] = v.permute(4, 0, 1, 2, 3).contiguous()
                else:
                    new_state_dict[k] = v
                    
            if "state_dict" in raw_ckpt:
                raw_ckpt["state_dict"] = new_state_dict
            else:
                raw_ckpt = new_state_dict
                
            torch.save(raw_ckpt, str(patched_ckpt_path))
            print(f"[INFO] Saved compatible weights to: {patched_ckpt_path}")
            
        self.model = build_model(str(patched_ckpt_path), device)
        self.remap = _REMAP_TABLE.to(device)

    @torch.inference_mode()
    def __call__(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.array([], dtype=np.uint8)

        pt_fea, grid_ind = self.voxelizer(points)
        pt_fea   = pt_fea.to(self.device)
        grid_ind = grid_ind.to(self.device)

        logits = self.model([pt_fea], [grid_ind], batch_size=1)
        vox_labels = logits[0].argmax(dim=0)

        gx = grid_ind[:, 0].long().clamp(0, GRID_SIZE[0] - 1)
        gy = grid_ind[:, 1].long().clamp(0, GRID_SIZE[1] - 1)
        gz = grid_ind[:, 2].long().clamp(0, GRID_SIZE[2] - 1)
        pt_sem = vox_labels[gx, gy, gz]

        pt_target = self.remap[pt_sem]
        return pt_target.cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------
class LidarInferenceNode:
    def __init__(self, inferencer: Inferencer):
        import rclpy  # type: ignore[import]
        from rclpy.node import Node  # type: ignore[import]
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # type: ignore[import]
        from sensor_msgs.msg import PointCloud2  # type: ignore[import]

        class _Node(Node):
            def __init__(inner_self):
                super().__init__("dl_inference")
                inner_self._inf = inferencer

                qos = QoSProfile(
                    reliability = ReliabilityPolicy.BEST_EFFORT,
                    history     = HistoryPolicy.KEEP_LAST,
                    depth       = 1,
                )

                inner_self._pub = inner_self.create_publisher(
                    PointCloud2, "/lidar/points_classified", qos
                )
                inner_self._sub = inner_self.create_subscription(
                    PointCloud2, "/lidar/points_raw",
                    inner_self._callback, qos
                )
                inner_self.get_logger().info(
                    "dl_inference ready — "
                    "sub: /lidar/points_raw  "
                    "pub: /lidar/points_classified"
                )

            def _callback(inner_self, msg: PointCloud2):
                t0 = time.perf_counter()

                points = _pc2_to_numpy(msg)            # (N, 4)
                labels = inner_self._inf(points)       # (N,) uint8

                out_msg = _numpy_to_pc2(msg.header, points, labels)
                inner_self._pub.publish(out_msg)

                dt_ms = (time.perf_counter() - t0) * 1000
                dist  = {k: int((labels == k).sum()) for k in range(3)}
                inner_self.get_logger().info(
                    f"Inferred {len(points)} pts in {dt_ms:.1f} ms | "
                    f"Terrain={dist[0]}  Static={dist[1]}  Dynamic={dist[2]}"
                )

        self._node_cls = _Node

    def spin(self):
        import rclpy  # type: ignore[import]
        rclpy.init()
        node = self._node_cls()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


# ---------------------------------------------------------------------------
# Standalone test mode (no ROS 2 needed)
# ---------------------------------------------------------------------------
def _run_test(bin_path: str, weights: str, dev: torch.device):
    print(f"[TEST] Device: {dev}")

    inf = Inferencer(weights, dev)

    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    print(f"[TEST] Loaded {len(pts):,} points from {bin_path}")

    # warm-up
    _ = inf(pts)

    t0     = time.perf_counter()
    labels = inf(pts)
    dt_ms  = (time.perf_counter() - t0) * 1000

    print(f"[TEST] Inference time : {dt_ms:.1f} ms")
    print("[TEST] Label distribution:")
    for cls_id, cls_name in TARGET_NAMES.items():
        count = int((labels == cls_id).sum())
        pct   = 100.0 * count / max(len(labels), 1)
        print(f"       {cls_id} ({cls_name:<17}): {count:>7,} pts  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cylinder3D LiDAR Inference Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test", "-t", metavar="BIN",
        help="Standalone mode: run inference on a KITTI .bin scan file (no ROS 2 needed)"
    )
    parser.add_argument(
        "--weights", "-w", default="weights/model_load.pth",
        help="Path to Cylinder3D checkpoint (default: weights/model_load.pth)"
    )
    args = parser.parse_args()

    weights = str(Path(args.weights).resolve())
    if not Path(weights).exists():
        print(f"[ERROR] Checkpoint not found: {weights}")
        sys.exit(1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Standalone test ───────────────────────────────────────────────────
    if args.test:
        _run_test(args.test, weights, dev)
        return

    # ── ROS 2 node ───────────────────────────────────────────────────────
    try:
        import rclpy  # type: ignore[import]  # noqa: F401
    except ImportError:
        print("[ERROR] rclpy not found.")
        print("        ROS 2 Humble must be installed and sourced:")
        print("          source /opt/ros/humble/setup.bash")
        print()
        print("        For offline testing without ROS 2:")
        print("          python dl_inference.py --test path/to/scan.bin")
        sys.exit(1)

    inferencer = Inferencer(weights, dev)
    LidarInferenceNode(inferencer).spin()


if __name__ == "__main__":
    main()