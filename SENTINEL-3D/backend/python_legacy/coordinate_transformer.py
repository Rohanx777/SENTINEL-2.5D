"""
coordinate_transformer.py — Coordinate System & Motion Compensation Engine
Handles transformations between:
  1. Sensor Polar / Cylindrical (r, azimuth_deg, z)
  2. Sensor Cartesian (x_sensor, y_sensor, z_sensor)
  3. World Cartesian (x_world, y_world, z_world) using 6-DOF odometry / pose matrices

Solves the critical issue where static objects appear locked or incorrectly displaced
during ego-vehicle motion by computing verified world-frame transforms and relative motion.
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np


class OdometryPose:
    """Represents a 6-DOF vehicle pose in SE(3) [Rotation (3x3), Translation (3,)]."""

    def __init__(self, transform_matrix: Optional[np.ndarray] = None):
        if transform_matrix is None:
            self.matrix = np.eye(4, dtype=np.float64)
        else:
            if transform_matrix.shape == (3, 4):
                self.matrix = np.eye(4, dtype=np.float64)
                self.matrix[:3, :] = transform_matrix
            elif transform_matrix.shape == (4, 4):
                self.matrix = transform_matrix.astype(np.float64)
            else:
                raise ValueError(f"Invalid pose matrix shape: {transform_matrix.shape}")

    @classmethod
    def from_xyz_yaw(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw_rad: float = 0.0) -> "OdometryPose":
        """Create SE(3) pose matrix from x, y, z and yaw rotation."""
        mat = np.eye(4, dtype=np.float64)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        mat[0, 0] = c
        mat[0, 1] = -s
        mat[1, 0] = s
        mat[1, 1] = c
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z
        return cls(mat)

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    @property
    def x(self) -> float:
        return float(self.matrix[0, 3])

    @property
    def y(self) -> float:
        return float(self.matrix[1, 3])

    @property
    def z(self) -> float:
        return float(self.matrix[2, 3])

    @property
    def yaw_deg(self) -> float:
        """Extract yaw angle in degrees from rotation matrix."""
        yaw_rad = math.atan2(self.matrix[1, 0], self.matrix[0, 0])
        return math.degrees(yaw_rad)

    def inverse(self) -> "OdometryPose":
        """Compute inverse SE(3) transform (world -> sensor)."""
        r_inv = self.rotation.T
        t_inv = -r_inv @ self.translation
        inv_mat = np.eye(4, dtype=np.float64)
        inv_mat[:3, :3] = r_inv
        inv_mat[:3, 3] = t_inv
        return OdometryPose(inv_mat)


class PoseManager:
    """Loads and manages actual KITTI odometry poses or generates realistic kinematic trajectories."""

    def __init__(self, poses_file: Optional[Union[str, Path]] = None):
        self.poses: List[OdometryPose] = []
        self._is_synthetic = False

        if poses_file is not None and Path(poses_file).exists():
            self._load_kitti_poses(Path(poses_file))
        else:
            self._is_synthetic = True

    def _load_kitti_poses(self, path: Path) -> None:
        """Load KITTI format poses.txt (each row is 12 float numbers for 3x4 matrix)."""
        try:
            raw_data = np.loadtxt(str(path))
            for row in raw_data:
                mat = row.reshape(3, 4)
                self.poses.append(OdometryPose(mat))
            print(f"[PoseManager] Loaded {len(self.poses)} ground-truth poses from {path}")
        except Exception as e:
            print(f"[PoseManager] Error loading poses from {path}: {e}. Falling back to kinematic model.")
            self._is_synthetic = True

    def get_pose(self, frame_idx: int, dt: float = 0.1) -> OdometryPose:
        """Get the vehicle pose at a given frame."""
        if not self._is_synthetic and len(self.poses) > 0:
            return self.poses[frame_idx % len(self.poses)]

        # Kinematic Bicycle / Ackermann Motion Model for realistic trajectory
        # Vehicle drives forward, navigates turns and curves
        t = frame_idx * dt
        # Velocity ~ 10 m/s (~36 km/h)
        speed = 10.0
        # Curved path with gentle turns
        x = speed * t * 0.8
        y = 6.0 * math.sin(t * 0.15)
        yaw = 0.15 * math.cos(t * 0.15)

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        mat = np.array([
            [cos_y, -sin_y, 0.0, x],
            [sin_y,  cos_y, 0.0, y],
            [0.0,    0.0,   1.0, 0.0],
            [0.0,    0.0,   0.0, 1.0],
        ], dtype=np.float64)

        return OdometryPose(mat)


class CoordinateTransformer:
    """
    Transforms coordinates between:
      - Sensor-frame Polar (r, azimuth, z)
      - Sensor-frame Cartesian (x, y, z)
      - World-frame Cartesian (X, Y, Z)
    """

    @staticmethod
    def polar_to_cartesian_sensor(r: float, azimuth_deg: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Convert sensor-centric polar coordinate to sensor Cartesian."""
        az_rad = math.radians(azimuth_deg)
        x = r * math.cos(az_rad)
        y = r * math.sin(az_rad)
        return float(x), float(y), float(z)

    @staticmethod
    def cartesian_to_polar_sensor(x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Convert sensor Cartesian to sensor polar (r, azimuth_deg, z)."""
        r = math.sqrt(x * x + y * y)
        az_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        return float(r), float(az_deg), float(z)

    @staticmethod
    def sensor_to_world_points(points: np.ndarray, pose: OdometryPose) -> np.ndarray:
        """
        Transform (N, 3) or (N, 4) sensor-frame points to world-frame.
        P_world = R * P_sensor + T
        """
        if len(points) == 0:
            return points

        pts_xyz = points[:, :3]
        r = pose.rotation
        t = pose.translation

        world_xyz = (r @ pts_xyz.T).T + t

        if points.shape[1] > 3:
            # Preserve intensity/labels/etc.
            result = np.zeros_like(points)
            result[:, :3] = world_xyz
            result[:, 3:] = points[:, 3:]
            return result
        return world_xyz

    @staticmethod
    def world_to_sensor_points(world_points: np.ndarray, pose: OdometryPose) -> np.ndarray:
        """
        Transform (N, 3) or (N, 4) world-frame points back to sensor frame.
        P_sensor = R^T * (P_world - T)
        """
        if len(world_points) == 0:
            return world_points

        pts_xyz = world_points[:, :3]
        pose_inv = pose.inverse()
        r_inv = pose_inv.rotation
        t_inv = pose_inv.translation

        sensor_xyz = (r_inv @ pts_xyz.T).T + t_inv

        if world_points.shape[1] > 3:
            result = np.zeros_like(world_points)
            result[:, :3] = sensor_xyz
            result[:, 3:] = world_points[:, 3:]
            return result
        return sensor_xyz

    @staticmethod
    def transform_cell_polygon(
        polygon: List[List[float]],
        pose: OdometryPose,
        to_world: bool = True
    ) -> List[List[float]]:
        """Transform 2D polygon vertices between sensor and world coordinate frames."""
        r = pose.rotation[:2, :2] if to_world else pose.inverse().rotation[:2, :2]
        t = pose.translation[:2] if to_world else pose.inverse().translation[:2]

        transformed = []
        for pt in polygon:
            p_vec = np.array([pt[0], pt[1]], dtype=np.float64)
            p_new = r @ p_vec + t
            transformed.append([round(float(p_new[0]), 3), round(float(p_new[1]), 3)])
        return transformed


def make_linear_cartesian_box(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
    """Return standard rectangular polygon for linear Cartesian grid cell."""
    return [
        [round(x0, 3), round(y0, 3)],
        [round(x1, 3), round(y0, 3)],
        [round(x1, 3), round(y1, 3)],
        [round(x0, 3), round(y1, 3)],
    ]


def make_polar_annular_wedge(r0: float, r1: float, phi0: float, phi1: float) -> List[List[float]]:
    """Return polygonal annular sector wedge for polar grid cell."""
    phi_mid = 0.5 * (phi0 + phi1)
    return [
        [round(r0 * math.cos(phi0), 3), round(r0 * math.sin(phi0), 3)],
        [round(r1 * math.cos(phi0), 3), round(r1 * math.sin(phi0), 3)],
        [round(r1 * math.cos(phi_mid), 3), round(r1 * math.sin(phi_mid), 3)],
        [round(r1 * math.cos(phi1), 3), round(r1 * math.sin(phi1), 3)],
        [round(r0 * math.cos(phi1), 3), round(r0 * math.sin(phi1), 3)],
    ]
