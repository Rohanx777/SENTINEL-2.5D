"""
config_loader.py — Dynamic Configuration Loader for SENTINEL-2.5D
Loads sensor parameters, grid configurations, and calibration data
from YAML config files, eliminating all hardcoded values.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Try to import yaml, if not available use a lightweight fallback parser
try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class SentinelConfig:
    """Manages all configuration parameters for the SENTINEL-2.5D system."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to backend/config/sensor_config.yaml
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.config_path = root_dir / "backend" / "config" / "sensor_config.yaml"
        else:
            self.config_path = Path(config_path)

        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file or initialize with defaults."""
        if self.config_path.exists() and HAS_YAML:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f)
                return
            except Exception as e:
                print(f"[WARN] Failed to load config from {self.config_path}: {e}")

        # Fallback default configuration if YAML not found or failed to parse
        self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Provides default configuration matching standard Velodyne HDL-64E."""
        return {
            "sensor": {
                "name": "Velodyne HDL-64E",
                "frequency_hz": 10,
                "range": {
                    "min_meters": 2.0,
                    "max_meters": 120.0,
                    "operational_max": 45.0,
                },
                "filtering": {
                    "z_min": -3.5,
                    "z_max": 8.0,
                    "min_distance_from_origin": 0.5,
                },
            },
            "grid": {
                "polar": {
                    "enabled": True,
                    "num_rings": 24,
                    "num_sectors": 64,
                    "r_min": 2.0,
                    "r_max": 45.0,
                },
                "cartesian": {
                    "enabled": True,
                    "x_min": -40.0,
                    "x_max": 40.0,
                    "y_min": -20.0,
                    "y_max": 20.0,
                    "cell_size": 2.0,
                },
                "voxel": {
                    "size": [480, 360, 32],
                    "range": {
                        "rho": [0.0, 50.0],
                        "phi": [-3.14159, 3.14159],
                        "z": [-4.0, 2.0],
                    },
                },
            },
            "coordinate_systems": {
                "sensor_frame": {"name": "sensor", "origin": [0.0, 0.0, 0.0]},
                "world_frame": {"name": "world", "requires_odometry": True},
                "default": "sensor",
            },
            "segmentation": {
                "class_mapping": {
                    "terrain": {
                        "label_id": 0,
                        "color": [34, 197, 94, 190],
                        "semantic_kitti_classes": [0, 9, 10, 11, 12, 15, 17],
                    },
                    "static_obstacle": {
                        "label_id": 1,
                        "color": [239, 68, 68, 230],
                        "semantic_kitti_classes": [13, 14, 16, 18, 19],
                    },
                    "dynamic_target": {
                        "label_id": 2,
                        "color": [251, 191, 36, 240],
                        "semantic_kitti_classes": [1, 2, 3, 4, 5, 6, 7, 8],
                    },
                },
                "temporal_smoothing": {
                    "enabled": True,
                    "history_frames": 10,
                    "method": "majority_vote",
                },
            },
            "geometry_heuristics": {
                "height_thresholds": {
                    "road_max_delta_z": 1.0,
                    "vehicle_min_delta_z": 1.0,
                    "vehicle_max_delta_z": 3.5,
                    "building_min_delta_z": 2.0,
                },
                "elevation_mapping": {
                    "road_elevation": 0.2,
                    "min_elevation": 0.12,
                    "max_elevation": 5.0,
                },
            },
            "performance": {
                "target_fps": 30,
                "inference_device": "cuda",
                "half_precision": False,
            },
            "streaming": {
                "host": "0.0.0.0",
                "port": 8005,
                "route": "/ws/stream_map",
            },
        }

    # ── Accessor Methods ─────────────────────────────────────────────────────────

    @property
    def sensor_range_min(self) -> float:
        return float(self._config.get("sensor", {}).get("range", {}).get("min_meters", 2.0))

    @property
    def sensor_range_max(self) -> float:
        return float(self._config.get("sensor", {}).get("range", {}).get("operational_max", 45.0))

    @property
    def z_filter_min(self) -> float:
        return float(self._config.get("sensor", {}).get("filtering", {}).get("z_min", -3.5))

    @property
    def z_filter_max(self) -> float:
        return float(self._config.get("sensor", {}).get("filtering", {}).get("z_max", 8.0))

    @property
    def polar_num_rings(self) -> int:
        return int(self._config.get("grid", {}).get("polar", {}).get("num_rings", 24))

    @property
    def polar_num_sectors(self) -> int:
        return int(self._config.get("grid", {}).get("polar", {}).get("num_sectors", 64))

    @property
    def cartesian_bounds(self) -> tuple[float, float, float, float]:
        cfg = self._config.get("grid", {}).get("cartesian", {})
        return (
            float(cfg.get("x_min", -40.0)),
            float(cfg.get("x_max", 40.0)),
            float(cfg.get("y_min", -20.0)),
            float(cfg.get("y_max", 20.0)),
        )

    @property
    def cartesian_cell_size(self) -> float:
        return float(self._config.get("grid", {}).get("cartesian", {}).get("cell_size", 2.0))

    @property
    def target_fps(self) -> int:
        return int(self._config.get("performance", {}).get("target_fps", 30))

    @property
    def temporal_history_len(self) -> int:
        return int(
            self._config.get("segmentation", {})
            .get("temporal_smoothing", {})
            .get("history_frames", 10)
        )

    def get_remap_table(self) -> np.ndarray:
        """Build 20-class to 3-class remapping table from config."""
        remap = np.zeros(20, dtype=np.uint8)
        classes = self._config.get("segmentation", {}).get("class_mapping", {})
        for class_name, data in classes.items():
            target_id = data.get("label_id", 0)
            for src_class in data.get("semantic_kitti_classes", []):
                if 0 <= src_class < 20:
                    remap[src_class] = target_id
        return remap

    def get_color_map(self) -> Dict[str, List[int]]:
        """Return label-to-color mapping for visualization."""
        color_map = {}
        classes = self._config.get("segmentation", {}).get("class_mapping", {})
        for name, data in classes.items():
            color_map[name] = data.get("color", [100, 116, 139, 140])
        return color_map

    def get_raw_dict(self) -> Dict[str, Any]:
        return self._config


# Global singleton instance
config = SentinelConfig()
