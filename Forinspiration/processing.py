"""
Video processing pipeline for skeleton visualization.
Handles video upload, MediaPipe pose detection, rudder angle detection, and project management.
"""

import os
import json
import uuid
import shutil
import threading
import concurrent.futures
import warnings
import math
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_OPT_AVAILABLE = True
except Exception:
    SCIPY_OPT_AVAILABLE = False

# Rayplane and skeleton metrics imports
from rayplane import (
    default_camera_pose_and_rotation,
    load_fisheye_undistorted_intrinsics,
    intersect_world_z_plane,
    ray_from_norm_landmark_undistorted,
    get_landmark_norm,
    place_skeleton_on_boat,
)
from skeleton_metrics import compute_center_of_mass

# MediaPipe imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Rudder detection imports (PilotNet NN)
from rudder_nn import PilotNetDetector
from skeleton_filter import SkeletonPlacementKalman, normalize_skeleton_filter_params

PROJECTS_DIR = Path(__file__).parent / "projects"

# MediaPipe model paths
MODEL_PATHS = {
    "lite": Path(__file__).parent / "pose_landmarker_lite.task",
    "full": Path(__file__).parent / "pose_landmarker_full.task",
    "heavy": Path(__file__).parent / "pose_landmarker_heavy.task",
}

# Constants for metrics computation
GRAVITY = 9.81

# Filter thresholds for plausibility checks
FILTER_THRESHOLDS = {
    'trunk_angle': {'min': 0, 'max': 180},
    'moment_x': {'min': -1500, 'max': 100},
    'moment_y': {'min': -1300, 'max': 1300},
    'com_x': {'min': -3, 'max': 0},
    'com_y': {'min': -2, 'max': 2},
    'com_z': {'min': -0.5, 'max': 0.5},
}

# Hull mesh transform used in the Three.js viewer:
# scale 0.01, then rotate with [[0,0,-1],[0,1,0],[1,0,0]], then translate.
HULL_VIEWER_SCALE = 0.01
HULL_VIEWER_ROT = np.array([
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)
HULL_VIEWER_TRANSLATION = np.array([-2.974, 0.0, 0.0], dtype=np.float64)

# Contact-based placement parameters
# Contact constraints should be driven by legs/feet, not hips:
# using hips as contact anchors can introduce lateral centerline bias.
CONTACT_BODY_INDICES = (25, 26, 27, 28, 29, 30, 31, 32)
CONTACT_SNAP_INDICES = (27, 28, 29, 30, 31, 32)
CONTACT_WEIGHTS = {
    27: 2.0, 28: 2.0, 29: 2.0, 30: 2.0, 31: 1.8, 32: 1.8,  # feet/ankles
    25: 1.3, 26: 1.3,  # knees
    23: 1.0, 24: 1.0,  # hips
}
CONTACT_SNAP_MAX_DIST_M = 0.20
CONTACT_USE_OPTIMIZER_DEFAULT = False
CONTACT_PARAMS_DEFAULT = {
    "contact_weight": 1.0,
    "penetration_weight": 1.0,
    "snap_weight": 1.0,
    "snap_max_dist_m": CONTACT_SNAP_MAX_DIST_M,
    "use_optimizer": CONTACT_USE_OPTIMIZER_DEFAULT,
}

# Dynamic lower-plane bias for seated posture:
# z_low_eff = z_plane_lm28 - drop_per_score * sitting_score, clamped.
SEATED_LOWER_PLANE_DROP_PER_SCORE_M = 0.14
SEATED_LOWER_PLANE_MAX_DROP_M = 0.18

# Seated fore-aft (x) stabilizer:
# when posture is seated and not strongly hiking, damp frame-to-frame global x
# translation to reduce ray/plane induced fore-aft jitter.
SEATED_X_STAB_ENABLED_DEFAULT = True
SEATED_X_STAB_SIT_START = 0.30
SEATED_X_STAB_SIT_FULL = 0.80
SEATED_X_STAB_ALPHA_MIN = 0.32
SEATED_X_STAB_ALPHA_MAX = 0.82
SEATED_X_STAB_STEP_CAP_SEATED_M = 0.08
SEATED_X_STAB_STEP_CAP_DEFAULT_M = 0.16
SEATED_X_STAB_HIKE_RELAX = 0.80
SEATED_X_STAB_MAX_SHIFT_M = 0.30
SEATED_X_STAB_FORWARD_RELEASE_GAIN = 1.0
SEATED_X_STAB_FORWARD_RELEASE_MAX_M = 0.45
SEATED_X_STAB_DEFAULTS = {
    "enabled": SEATED_X_STAB_ENABLED_DEFAULT,
    "sit_start": SEATED_X_STAB_SIT_START,
    "sit_full": SEATED_X_STAB_SIT_FULL,
    "alpha_min": SEATED_X_STAB_ALPHA_MIN,
    "alpha_max": SEATED_X_STAB_ALPHA_MAX,
    "step_cap_seated_m": SEATED_X_STAB_STEP_CAP_SEATED_M,
    "step_cap_default_m": SEATED_X_STAB_STEP_CAP_DEFAULT_M,
    "hike_relax": SEATED_X_STAB_HIKE_RELAX,
    "max_shift_m": SEATED_X_STAB_MAX_SHIFT_M,
    "forward_release_gain": SEATED_X_STAB_FORWARD_RELEASE_GAIN,
    "forward_release_max_m": SEATED_X_STAB_FORWARD_RELEASE_MAX_M,
}

# Lateral y stabilizer:
# when hiking/layback is pronounced, damp frame-to-frame global y translation
# to reduce side-to-side jitter from ray/plane placement noise.
LATERAL_Y_STAB_ENABLED_DEFAULT = True
LATERAL_Y_STAB_HIKE_START = 0.20
LATERAL_Y_STAB_HIKE_FULL = 0.70
LATERAL_Y_STAB_ALPHA_MIN = 0.24
LATERAL_Y_STAB_ALPHA_MAX = 0.78
LATERAL_Y_STAB_STEP_CAP_HIKING_M = 0.05
LATERAL_Y_STAB_STEP_CAP_DEFAULT_M = 0.12
LATERAL_Y_STAB_MAX_SHIFT_M = 0.22
LATERAL_Y_STAB_SIT_RELAX = 0.45
LATERAL_Y_STAB_LOCK_GAIN = 1.0
LATERAL_Y_STAB_LOCK_MAX_DIST_M = 0.35
LATERAL_Y_STAB_DEFAULTS = {
    "enabled": LATERAL_Y_STAB_ENABLED_DEFAULT,
    "hike_start": LATERAL_Y_STAB_HIKE_START,
    "hike_full": LATERAL_Y_STAB_HIKE_FULL,
    "alpha_min": LATERAL_Y_STAB_ALPHA_MIN,
    "alpha_max": LATERAL_Y_STAB_ALPHA_MAX,
    "step_cap_hiking_m": LATERAL_Y_STAB_STEP_CAP_HIKING_M,
    "step_cap_default_m": LATERAL_Y_STAB_STEP_CAP_DEFAULT_M,
    "max_shift_m": LATERAL_Y_STAB_MAX_SHIFT_M,
    "sit_relax": LATERAL_Y_STAB_SIT_RELAX,
    "lock_gain": LATERAL_Y_STAB_LOCK_GAIN,
    "lock_max_dist_m": LATERAL_Y_STAB_LOCK_MAX_DIST_M,
}

# Auto camera-PnP recalibration (YOLO keypoints -> solvePnP)
AUTO_CAMERA_PNP_INTERVAL_FRAMES = 10000
AUTO_CAMERA_PNP_AVG_FRAMES = 5
AUTO_CAMERA_PNP_MIN_VALID_FRAMES = 5
AUTO_CAMERA_PNP_MIN_KPT_CONF = 0.8
AUTO_CAMERA_PNP_MIN_PAIRS = 6
AUTO_CAMERA_PNP_MAX_REPROJ_ERR_PX = 30.0
AUTO_CAMERA_PNP_ROLL_OFFSET_DEG = 90.0
AUTO_CAMERA_PNP_BOUNDS = {
    "pitch_min_deg": 10.0,
    "pitch_max_deg": 23.0,
    "yaw_abs_max_deg": 10.0,
    "roll_abs_max_deg": 5.0,
    "x_min_m": -3.4,
    "x_max_m": -3.1,
    "y_abs_max_m": 0.1,
    "z_min_m": 0.5,
    "z_max_m": 0.8,
}
AUTO_CAMERA_PNP_MODEL_PATH = Path(__file__).parent / "best.pt"
AUTO_CAMERA_PNP_KEYPOINT_LABELS = [
    "frontdeck",
    "porttop",
    "portmid",
    "portlow",
    "starboardtop",
    "starboardmid",
    "starboardlow",
    "portback",
    "starboardback",
]
AUTO_CAMERA_PNP_OBJECT_POINTS = {
    "frontdeck": [-0.531, 0.000, 0.001],
    "porttop": [-1.174, -0.005, -0.308],
    "portmid": [-1.182, -0.060, -0.294],
    "portlow": [-1.189, -0.176, -0.295],
    "portback": [-2.169, -0.007, -0.284],
    "starboardtop": [-1.165, -0.005, 0.334],
    "starboardmid": [-1.162, -0.060, 0.312],
    "starboardlow": [-1.177, -0.185, 0.313],
    "starboardback": [-2.169, -0.012, 0.306],
}
AUTO_CAMERA_BASE_R_WC = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)

_auto_camera_pnp_model = None
_auto_camera_pnp_model_lock = threading.Lock()


def normalize_contact_params(params: Optional[dict]) -> dict:
    """Clamp/normalize contact fitting params."""
    out = dict(CONTACT_PARAMS_DEFAULT)
    if not isinstance(params, dict):
        return out
    try:
        out["contact_weight"] = float(np.clip(float(params.get("contact_weight", out["contact_weight"])), 0.0, 10.0))
    except Exception:
        pass
    try:
        out["penetration_weight"] = float(np.clip(float(params.get("penetration_weight", out["penetration_weight"])), 0.0, 2.0))
    except Exception:
        pass
    try:
        out["snap_weight"] = float(np.clip(float(params.get("snap_weight", out["snap_weight"])), 0.0, 2.0))
    except Exception:
        pass
    try:
        out["snap_max_dist_m"] = float(np.clip(float(params.get("snap_max_dist_m", out["snap_max_dist_m"])), 0.0, 1.0))
    except Exception:
        pass
    out["use_optimizer"] = bool(params.get("use_optimizer", out["use_optimizer"]))
    return out


def normalize_seated_x_stabilizer_params(params: Optional[dict]) -> dict:
    """Clamp/normalize seated fore-aft x stabilizer params."""
    out = dict(SEATED_X_STAB_DEFAULTS)
    if not isinstance(params, dict):
        return out

    out["enabled"] = bool(params.get("enabled", out["enabled"]))
    try:
        out["sit_start"] = float(np.clip(float(params.get("sit_start", out["sit_start"])), 0.0, 1.0))
    except Exception:
        pass
    try:
        out["sit_full"] = float(np.clip(float(params.get("sit_full", out["sit_full"])), 0.0, 1.0))
    except Exception:
        pass
    if out["sit_full"] < out["sit_start"] + 0.01:
        out["sit_full"] = min(1.0, out["sit_start"] + 0.01)

    try:
        out["alpha_min"] = float(np.clip(float(params.get("alpha_min", out["alpha_min"])), 0.01, 0.99))
    except Exception:
        pass
    try:
        out["alpha_max"] = float(np.clip(float(params.get("alpha_max", out["alpha_max"])), 0.01, 0.99))
    except Exception:
        pass
    if out["alpha_max"] < out["alpha_min"]:
        out["alpha_max"] = out["alpha_min"]

    try:
        out["step_cap_seated_m"] = float(
            np.clip(float(params.get("step_cap_seated_m", out["step_cap_seated_m"])), 1e-4, 1.0)
        )
    except Exception:
        pass
    try:
        out["step_cap_default_m"] = float(
            np.clip(float(params.get("step_cap_default_m", out["step_cap_default_m"])), 1e-4, 2.0)
        )
    except Exception:
        pass
    if out["step_cap_default_m"] < out["step_cap_seated_m"]:
        out["step_cap_default_m"] = out["step_cap_seated_m"]

    try:
        out["hike_relax"] = float(np.clip(float(params.get("hike_relax", out["hike_relax"])), 0.0, 1.0))
    except Exception:
        pass
    try:
        out["max_shift_m"] = float(np.clip(float(params.get("max_shift_m", out["max_shift_m"])), 0.01, 1.0))
    except Exception:
        pass
    try:
        out["forward_release_gain"] = float(
            np.clip(float(params.get("forward_release_gain", out["forward_release_gain"])), 0.0, 3.0)
        )
    except Exception:
        pass
    try:
        out["forward_release_max_m"] = float(
            np.clip(float(params.get("forward_release_max_m", out["forward_release_max_m"])), 0.0, 1.5)
        )
    except Exception:
        pass

    return out


def normalize_lateral_y_stabilizer_params(params: Optional[dict]) -> dict:
    """Clamp/normalize lateral y stabilizer params."""
    out = dict(LATERAL_Y_STAB_DEFAULTS)
    if not isinstance(params, dict):
        return out

    out["enabled"] = bool(params.get("enabled", out["enabled"]))
    try:
        out["hike_start"] = float(np.clip(float(params.get("hike_start", out["hike_start"])), 0.0, 1.0))
    except Exception:
        pass
    try:
        out["hike_full"] = float(np.clip(float(params.get("hike_full", out["hike_full"])), 0.0, 1.0))
    except Exception:
        pass
    if out["hike_full"] < out["hike_start"] + 0.01:
        out["hike_full"] = min(1.0, out["hike_start"] + 0.01)

    try:
        out["alpha_min"] = float(np.clip(float(params.get("alpha_min", out["alpha_min"])), 0.01, 0.99))
    except Exception:
        pass
    try:
        out["alpha_max"] = float(np.clip(float(params.get("alpha_max", out["alpha_max"])), 0.01, 0.99))
    except Exception:
        pass
    if out["alpha_max"] < out["alpha_min"]:
        out["alpha_max"] = out["alpha_min"]

    try:
        out["step_cap_hiking_m"] = float(
            np.clip(float(params.get("step_cap_hiking_m", out["step_cap_hiking_m"])), 1e-4, 1.0)
        )
    except Exception:
        pass
    try:
        out["step_cap_default_m"] = float(
            np.clip(float(params.get("step_cap_default_m", out["step_cap_default_m"])), 1e-4, 2.0)
        )
    except Exception:
        pass
    if out["step_cap_default_m"] < out["step_cap_hiking_m"]:
        out["step_cap_default_m"] = out["step_cap_hiking_m"]

    try:
        out["max_shift_m"] = float(np.clip(float(params.get("max_shift_m", out["max_shift_m"])), 0.01, 1.0))
    except Exception:
        pass
    try:
        out["sit_relax"] = float(np.clip(float(params.get("sit_relax", out["sit_relax"])), 0.0, 1.0))
    except Exception:
        pass
    try:
        out["lock_gain"] = float(np.clip(float(params.get("lock_gain", out["lock_gain"])), 0.0, 3.0))
    except Exception:
        pass
    try:
        out["lock_max_dist_m"] = float(
            np.clip(float(params.get("lock_max_dist_m", out["lock_max_dist_m"])), 0.0, 1.5)
        )
    except Exception:
        pass
    return out


def _find_hull_stl_path() -> Optional[Path]:
    candidates = [
        Path(__file__).parent / "Hull.stl",
        Path("Hull.stl"),
        Path(__file__).parent.parent / "Hull.stl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def _load_hull_mesh_boat_frame():
    """Load hull STL and transform it into the boat/world frame used in viewer/metrics."""
    if not TRIMESH_AVAILABLE:
        return None

    stl_path = _find_hull_stl_path()
    if stl_path is None:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = trimesh.load(str(stl_path), force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                return None
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        verts = (verts @ HULL_VIEWER_ROT.T) * HULL_VIEWER_SCALE + HULL_VIEWER_TRANSLATION
        mesh_boat = trimesh.Trimesh(
            vertices=verts,
            faces=np.asarray(mesh.faces, dtype=np.int64),
            process=False,
        )
        return mesh_boat
    except Exception:
        return None


def _safe_closest_point(mesh, points: np.ndarray):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            closest, dist, _ = trimesh.proximity.closest_point(mesh, points)
        return np.asarray(closest, dtype=np.float64), np.asarray(dist, dtype=np.float64)
    except Exception:
        return None, None


def _safe_contains(mesh, points: np.ndarray) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inside = mesh.contains(points)
        inside = np.asarray(inside, dtype=bool)
        if inside.shape[0] != points.shape[0]:
            return np.zeros(points.shape[0], dtype=bool)
        return inside
    except Exception:
        # If contains fails (e.g., non-watertight mesh, missing acceleration backend),
        # continue without hard inside/outside classification.
        return np.zeros(points.shape[0], dtype=bool)


def _extract_camera_frame_landmarks(row) -> Dict[int, np.ndarray]:
    """Extract MediaPipe world landmarks from row (camera frame coordinates)."""
    out: Dict[int, np.ndarray] = {}
    for i in range(33):
        x = row.get(f"lm{i}_world_x")
        y = row.get(f"lm{i}_world_y")
        z = row.get(f"lm{i}_world_z")
        if x is None or y is None or z is None:
            continue
        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            continue
        out[i] = np.array([float(x), float(y), float(z)], dtype=np.float64)
    return out


def _guard_vertical_foot_outliers(
    placed: Dict[int, np.ndarray],
    z_plane_lm28: float,
    snap_max_dist: float,
) -> Dict[int, np.ndarray]:
    """Clamp implausibly high foot placement from nearest-surface sidewall snaps."""
    if not placed:
        return placed

    foot_z: List[float] = []
    for idx in CONTACT_SNAP_INDICES:
        p = placed.get(int(idx))
        if p is None:
            continue
        z = float(p[2])
        if np.isfinite(z):
            foot_z.append(z)
    if not foot_z:
        return placed

    foot_min_z = float(np.min(np.asarray(foot_z, dtype=np.float64)))
    max_foot_lift = max(0.08, 0.5 * float(snap_max_dist))
    lift_cap = float(z_plane_lm28) + max_foot_lift
    if foot_min_z <= lift_cap:
        return placed

    target_min_z = float(z_plane_lm28) + min(0.02, 0.5 * max_foot_lift)
    dz = float(np.clip(target_min_z - foot_min_z, -0.40, 0.0))
    if abs(dz) < 1e-9:
        return placed

    adjusted: Dict[int, np.ndarray] = {}
    for idx, p in placed.items():
        q = np.asarray(p, dtype=np.float64).copy()
        q[2] += dz
        adjusted[int(idx)] = q
    return adjusted


def _guard_hip_height_from_foot_lock(
    placed: Dict[int, np.ndarray],
    z_plane_lm24: float,
    z_plane_lm28: float,
    sitting_score: float = 0.0,
) -> Dict[int, np.ndarray]:
    """Reduce floating-hip artifacts caused by strict lower-limb floor locking.

    Allows a small downward shift (with limited foot penetration tolerance) so
    hip height stays near configured deck reference instead of drifting high.
    """
    if not placed:
        return placed

    hip_z: List[float] = []
    for idx in (23, 24):
        p = placed.get(int(idx))
        if p is None:
            continue
        z = float(np.asarray(p, dtype=np.float64).reshape(3)[2])
        if np.isfinite(z):
            hip_z.append(z)
    if not hip_z:
        return placed

    low_z: List[float] = []
    for idx in (27, 28, 29, 30, 31, 32, 25, 26):
        p = placed.get(int(idx))
        if p is None:
            continue
        z = float(np.asarray(p, dtype=np.float64).reshape(3)[2])
        if np.isfinite(z):
            low_z.append(z)
    if not low_z:
        return placed

    hip_mid_z = float(np.mean(np.asarray(hip_z, dtype=np.float64)))
    low_min_z = float(np.min(np.asarray(low_z, dtype=np.float64)))

    sit = float(np.clip(sitting_score, 0.0, 1.0))

    # For seated posture (high knee flexion), tighten hip-height cap.
    hip_target_z = float(z_plane_lm24) + (0.03 - 0.06 * sit)
    hip_max_z = hip_target_z + (0.12 - 0.05 * sit)
    if hip_mid_z <= hip_max_z:
        return placed

    desired_drop = float(hip_mid_z - hip_max_z)

    # Permit deeper sub-plane feet for seated posture; this avoids forcing
    # unrealistically high hips when knees are deeply flexed.
    min_allowed_foot_z = float(z_plane_lm28) - (0.05 + 0.16 * sit)
    max_drop_by_feet = float(low_min_z - min_allowed_foot_z)
    if max_drop_by_feet <= 1e-6:
        return placed

    dz = -float(np.clip(min(desired_drop, max_drop_by_feet), 0.0, 0.30 + 0.20 * sit))
    if abs(dz) < 1e-9:
        return placed

    adjusted: Dict[int, np.ndarray] = {}
    for idx, p in placed.items():
        if p is None:
            adjusted[int(idx)] = p
            continue
        q = np.asarray(p, dtype=np.float64).reshape(3).copy()
        q[2] += dz
        adjusted[int(idx)] = q
    return adjusted


def _apply_seated_pelvis_height_constraint(
    placed: Dict[int, np.ndarray],
    z_plane_lm24: float,
    sitting_score: float = 0.0,
) -> Dict[int, np.ndarray]:
    """Constrain seated pelvis height from knee + torso geometry.

    This is a pure kinematic guard (no hull-distance pull): when posture is
    strongly seated and hips drift too high relative to knees/torso, apply a
    bounded downward global z shift.
    """
    if not placed:
        return placed

    req = (11, 12, 23, 24, 25, 26, 27, 28)
    if not all(i in placed and placed[i] is not None for i in req):
        return placed

    sh_mid = 0.5 * (np.asarray(placed[11], dtype=np.float64) + np.asarray(placed[12], dtype=np.float64))
    hip_l = np.asarray(placed[23], dtype=np.float64)
    hip_r = np.asarray(placed[24], dtype=np.float64)
    hip_mid = 0.5 * (hip_l + hip_r)
    knee_mid = 0.5 * (np.asarray(placed[25], dtype=np.float64) + np.asarray(placed[26], dtype=np.float64))
    ankle_mid = 0.5 * (np.asarray(placed[27], dtype=np.float64) + np.asarray(placed[28], dtype=np.float64))

    if not np.all(np.isfinite(sh_mid)) or not np.all(np.isfinite(hip_mid)) or not np.all(np.isfinite(knee_mid)) or not np.all(np.isfinite(ankle_mid)):
        return placed

    sit_raw = float(np.clip(sitting_score, 0.0, 1.0))
    knee_raise = float(knee_mid[2] - float(z_plane_lm24))
    shoulder_knee_span = float(sh_mid[2] - knee_mid[2])
    knee_fold = float(knee_mid[2] - ankle_mid[2])

    # Geometric seated cues from lower-body fold + shoulder/knee relation.
    sit_geom = float(np.clip((knee_raise - 0.05) / 0.22, 0.0, 1.0)) * float(np.clip((shoulder_knee_span - 0.14) / 0.34, 0.0, 1.0))
    sit_geom = max(sit_geom, float(np.clip((knee_fold - 0.10) / 0.25, 0.0, 1.0)))
    sit = max(sit_raw, sit_geom)
    if sit < 0.35:
        return placed

    # Relax the constraint for pronounced layback/hiking posture.
    trunk = sh_mid - hip_mid
    trunk_z = float(abs(trunk[2]))
    lean_back = 0.0
    if trunk_z > 1e-6:
        lean_ratio = float((-trunk[0]) / trunk_z)
        lean_back = float(np.clip((lean_ratio - 0.12) / 0.55, 0.0, 1.0))
    lean_relax = float(1.0 - 0.45 * lean_back)

    shoulder_knee_span = max(0.08, shoulder_knee_span)
    alpha = float(np.clip(0.26 - 0.14 * sit, 0.12, 0.26))
    hip_cap_geom = float(knee_mid[2] + alpha * shoulder_knee_span)
    hip_cap_plane = float(z_plane_lm24) + (0.15 - 0.06 * sit)
    hip_cap = min(hip_cap_geom, hip_cap_plane + 0.03)

    # Allow some bilateral asymmetry without triggering aggressive correction.
    hip_asym = abs(float(hip_l[2] - hip_r[2]))
    hip_cap += min(0.05, 0.45 * hip_asym)

    hip_excess = float(hip_mid[2] - hip_cap)
    if hip_excess <= 1e-6:
        return placed

    max_drop = float(np.clip((0.09 + 0.17 * sit) * lean_relax, 0.07, 0.20))
    dz = -float(np.clip(hip_excess, 0.0, max_drop))
    if abs(dz) < 1e-9:
        return placed

    adjusted: Dict[int, np.ndarray] = {}
    for idx, p in placed.items():
        if p is None:
            adjusted[int(idx)] = p
            continue
        q = np.asarray(p, dtype=np.float64).reshape(3).copy()
        q[2] += dz
        adjusted[int(idx)] = q
    return adjusted


class _SeatedForeAftStabilizer:
    """Global fore-aft x stabilizer for seated posture."""

    def __init__(
        self,
        sit_start: float = SEATED_X_STAB_SIT_START,
        sit_full: float = SEATED_X_STAB_SIT_FULL,
        alpha_min: float = SEATED_X_STAB_ALPHA_MIN,
        alpha_max: float = SEATED_X_STAB_ALPHA_MAX,
        step_cap_seated_m: float = SEATED_X_STAB_STEP_CAP_SEATED_M,
        step_cap_default_m: float = SEATED_X_STAB_STEP_CAP_DEFAULT_M,
        hike_relax: float = SEATED_X_STAB_HIKE_RELAX,
        max_shift_m: float = SEATED_X_STAB_MAX_SHIFT_M,
        forward_release_gain: float = SEATED_X_STAB_FORWARD_RELEASE_GAIN,
        forward_release_max_m: float = SEATED_X_STAB_FORWARD_RELEASE_MAX_M,
    ):
        self.sit_start = float(sit_start)
        self.sit_full = float(max(sit_start + 1e-6, sit_full))
        self.alpha_min = float(np.clip(alpha_min, 0.01, 0.99))
        self.alpha_max = float(np.clip(alpha_max, self.alpha_min, 0.99))
        self.step_cap_seated_m = float(max(1e-4, step_cap_seated_m))
        self.step_cap_default_m = float(max(self.step_cap_seated_m, step_cap_default_m))
        self.hike_relax = float(np.clip(hike_relax, 0.0, 1.0))
        self.max_shift_m = float(max(1e-4, max_shift_m))
        self.forward_release_gain = float(max(0.0, forward_release_gain))
        self.forward_release_max_m = float(max(0.0, forward_release_max_m))
        self._prev_hip_mid_x: Optional[float] = None
        self.last_debug: Dict[str, Optional[float]] = {
            "sitting_score": None,
            "lean_back": None,
            "seat_weight": None,
            "stab_weight": None,
            "alpha": None,
            "step_cap": None,
            "shift_x": None,
            "max_shift_aft": None,
            "raw_forward_gap": None,
            "raw_forward_pull": None,
        }

    def reset(self) -> None:
        self._prev_hip_mid_x = None
        self.last_debug = {
            "sitting_score": None,
            "lean_back": None,
            "seat_weight": None,
            "stab_weight": None,
            "alpha": None,
            "step_cap": None,
            "shift_x": None,
            "max_shift_aft": None,
            "raw_forward_gap": None,
            "raw_forward_pull": None,
        }

    @staticmethod
    def _lean_back_score(placed: Dict[int, np.ndarray]) -> float:
        req = (11, 12, 23, 24)
        if not all(i in placed and placed[i] is not None for i in req):
            return 0.0
        sh_mid = 0.5 * (np.asarray(placed[11], dtype=np.float64) + np.asarray(placed[12], dtype=np.float64))
        hip_mid = 0.5 * (np.asarray(placed[23], dtype=np.float64) + np.asarray(placed[24], dtype=np.float64))
        if not np.all(np.isfinite(sh_mid)) or not np.all(np.isfinite(hip_mid)):
            return 0.0
        trunk = sh_mid - hip_mid
        trunk_z = float(abs(trunk[2]))
        if trunk_z < 1e-6:
            return 0.0
        lean_ratio = float((-trunk[0]) / trunk_z)
        return float(np.clip((lean_ratio - 0.12) / 0.55, 0.0, 1.0))

    def apply(
        self,
        placed: Optional[Dict[int, np.ndarray]],
        sitting_score: float,
        raw_reference: Optional[Dict[int, np.ndarray]] = None,
    ) -> Optional[Dict[int, np.ndarray]]:
        if not placed:
            self._prev_hip_mid_x = None
            self.last_debug = {
                "sitting_score": None,
                "lean_back": None,
                "seat_weight": None,
                "stab_weight": None,
                "alpha": None,
                "step_cap": None,
                "shift_x": None,
                "max_shift_aft": None,
                "raw_forward_gap": None,
                "raw_forward_pull": None,
            }
            return placed
        if 23 not in placed or 24 not in placed or placed[23] is None or placed[24] is None:
            self._prev_hip_mid_x = None
            self.last_debug = {
                "sitting_score": None,
                "lean_back": None,
                "seat_weight": None,
                "stab_weight": None,
                "alpha": None,
                "step_cap": None,
                "shift_x": None,
                "max_shift_aft": None,
                "raw_forward_gap": None,
                "raw_forward_pull": None,
            }
            return placed

        hip_l = np.asarray(placed[23], dtype=np.float64).reshape(3)
        hip_r = np.asarray(placed[24], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(hip_l)) or not np.all(np.isfinite(hip_r)):
            self._prev_hip_mid_x = None
            self.last_debug = {
                "sitting_score": None,
                "lean_back": None,
                "seat_weight": None,
                "stab_weight": None,
                "alpha": None,
                "step_cap": None,
                "shift_x": None,
                "max_shift_aft": None,
                "raw_forward_gap": None,
                "raw_forward_pull": None,
            }
            return placed

        x_meas = float(0.5 * (hip_l[0] + hip_r[0]))
        sit = float(np.clip(sitting_score, 0.0, 1.0))
        seat_w = float(np.clip((sit - self.sit_start) / (self.sit_full - self.sit_start), 0.0, 1.0))
        lean_back = self._lean_back_score(placed)
        non_hike = float(np.clip(1.0 - self.hike_relax * lean_back, 0.0, 1.0))
        stab_w = float(np.clip(seat_w * non_hike, 0.0, 1.0))

        if stab_w <= 1e-6:
            self._prev_hip_mid_x = x_meas
            self.last_debug = {
                "sitting_score": float(sit),
                "lean_back": float(lean_back),
                "seat_weight": float(seat_w),
                "stab_weight": float(stab_w),
                "alpha": None,
                "step_cap": None,
                "shift_x": 0.0,
                "max_shift_aft": None,
                "raw_forward_gap": None,
                "raw_forward_pull": None,
            }
            return placed

        x_input = x_meas
        raw_forward_gap = 0.0
        raw_forward_pull = 0.0
        if (
            raw_reference is not None
            and 23 in raw_reference and 24 in raw_reference
            and raw_reference[23] is not None and raw_reference[24] is not None
        ):
            raw_l = np.asarray(raw_reference[23], dtype=np.float64).reshape(3)
            raw_r = np.asarray(raw_reference[24], dtype=np.float64).reshape(3)
            if np.all(np.isfinite(raw_l)) and np.all(np.isfinite(raw_r)):
                x_raw = float(0.5 * (raw_l[0] + raw_r[0]))
                raw_forward_gap = float(np.clip(
                    x_raw - x_meas,
                    0.0,
                    self.forward_release_max_m,
                ))
                raw_forward_pull = float(self.forward_release_gain * stab_w * raw_forward_gap)
                x_input = x_meas + raw_forward_pull

        step_cap = float(np.clip(
            self.step_cap_default_m - (self.step_cap_default_m - self.step_cap_seated_m) * stab_w,
            self.step_cap_seated_m,
            self.step_cap_default_m,
        ))
        alpha = float(np.clip(
            self.alpha_max - (self.alpha_max - self.alpha_min) * stab_w,
            self.alpha_min,
            self.alpha_max,
        ))

        if self._prev_hip_mid_x is None or not np.isfinite(self._prev_hip_mid_x):
            x_filtered = x_input
        else:
            x_capped = float(self._prev_hip_mid_x + np.clip(x_input - self._prev_hip_mid_x, -step_cap, step_cap))
            x_filtered = float(alpha * x_capped + (1.0 - alpha) * self._prev_hip_mid_x)

        max_shift_aft = float(np.clip(self.max_shift_m * (1.0 - 0.65 * stab_w), 0.06, self.max_shift_m))
        dx = float(np.clip(x_filtered - x_meas, -max_shift_aft, self.max_shift_m))
        self._prev_hip_mid_x = float(x_meas + dx)
        self.last_debug = {
            "sitting_score": float(sit),
            "lean_back": float(lean_back),
            "seat_weight": float(seat_w),
            "stab_weight": float(stab_w),
            "alpha": float(alpha),
            "step_cap": float(step_cap),
            "shift_x": float(dx),
            "max_shift_aft": float(max_shift_aft),
            "raw_forward_gap": float(raw_forward_gap),
            "raw_forward_pull": float(raw_forward_pull),
        }
        if abs(dx) < 1e-9:
            return placed

        adjusted: Dict[int, np.ndarray] = {}
        for idx, p in placed.items():
            if p is None:
                adjusted[int(idx)] = p
                continue
            q = np.asarray(p, dtype=np.float64).reshape(3).copy()
            q[0] += dx
            adjusted[int(idx)] = q
        return adjusted


class _LateralYStabilizer:
    """Global lateral y stabilizer weighted by hiking posture."""

    def __init__(
        self,
        hike_start: float = LATERAL_Y_STAB_HIKE_START,
        hike_full: float = LATERAL_Y_STAB_HIKE_FULL,
        alpha_min: float = LATERAL_Y_STAB_ALPHA_MIN,
        alpha_max: float = LATERAL_Y_STAB_ALPHA_MAX,
        step_cap_hiking_m: float = LATERAL_Y_STAB_STEP_CAP_HIKING_M,
        step_cap_default_m: float = LATERAL_Y_STAB_STEP_CAP_DEFAULT_M,
        max_shift_m: float = LATERAL_Y_STAB_MAX_SHIFT_M,
        sit_relax: float = LATERAL_Y_STAB_SIT_RELAX,
        lock_gain: float = LATERAL_Y_STAB_LOCK_GAIN,
        lock_max_dist_m: float = LATERAL_Y_STAB_LOCK_MAX_DIST_M,
    ):
        self.hike_start = float(hike_start)
        self.hike_full = float(max(hike_start + 1e-6, hike_full))
        self.alpha_min = float(np.clip(alpha_min, 0.01, 0.99))
        self.alpha_max = float(np.clip(alpha_max, self.alpha_min, 0.99))
        self.step_cap_hiking_m = float(max(1e-4, step_cap_hiking_m))
        self.step_cap_default_m = float(max(self.step_cap_hiking_m, step_cap_default_m))
        self.max_shift_m = float(max(1e-4, max_shift_m))
        self.sit_relax = float(np.clip(sit_relax, 0.0, 1.0))
        self.lock_gain = float(max(0.0, lock_gain))
        self.lock_max_dist_m = float(max(0.0, lock_max_dist_m))
        self._prev_hip_mid_y: Optional[float] = None
        self.last_debug: Dict[str, Optional[float]] = {
            "sitting_score": None,
            "lean_back": None,
            "hike_weight": None,
            "stab_weight": None,
            "alpha": None,
            "step_cap": None,
            "shift_y": None,
            "raw_lock_gap": None,
            "raw_lock_pull": None,
        }

    def reset(self) -> None:
        self._prev_hip_mid_y = None
        self.last_debug = {
            "sitting_score": None,
            "lean_back": None,
            "hike_weight": None,
            "stab_weight": None,
            "alpha": None,
            "step_cap": None,
            "shift_y": None,
            "raw_lock_gap": None,
            "raw_lock_pull": None,
        }

    @staticmethod
    def _lean_back_score(placed: Dict[int, np.ndarray]) -> float:
        req = (11, 12, 23, 24)
        if not all(i in placed and placed[i] is not None for i in req):
            return 0.0
        sh_mid = 0.5 * (np.asarray(placed[11], dtype=np.float64) + np.asarray(placed[12], dtype=np.float64))
        hip_mid = 0.5 * (np.asarray(placed[23], dtype=np.float64) + np.asarray(placed[24], dtype=np.float64))
        if not np.all(np.isfinite(sh_mid)) or not np.all(np.isfinite(hip_mid)):
            return 0.0
        trunk = sh_mid - hip_mid
        trunk_z = float(abs(trunk[2]))
        if trunk_z < 1e-6:
            return 0.0
        lean_ratio = float((-trunk[0]) / trunk_z)
        return float(np.clip((lean_ratio - 0.12) / 0.55, 0.0, 1.0))

    def apply(
        self,
        placed: Optional[Dict[int, np.ndarray]],
        sitting_score: float,
        raw_reference: Optional[Dict[int, np.ndarray]] = None,
    ) -> Optional[Dict[int, np.ndarray]]:
        _ = raw_reference
        if not placed:
            self.reset()
            return placed
        if 23 not in placed or 24 not in placed or placed[23] is None or placed[24] is None:
            self.reset()
            return placed

        hip_l = np.asarray(placed[23], dtype=np.float64).reshape(3)
        hip_r = np.asarray(placed[24], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(hip_l)) or not np.all(np.isfinite(hip_r)):
            self.reset()
            return placed

        y_meas = float(0.5 * (hip_l[1] + hip_r[1]))
        sit = float(np.clip(sitting_score, 0.0, 1.0))
        lean_back = self._lean_back_score(placed)
        hike_w = float(np.clip((lean_back - self.hike_start) / (self.hike_full - self.hike_start), 0.0, 1.0))
        non_seated = float(np.clip(1.0 - self.sit_relax * sit, 0.0, 1.0))
        stab_w = float(np.clip(hike_w * non_seated, 0.0, 1.0))

        if stab_w <= 1e-6:
            self._prev_hip_mid_y = y_meas
            self.last_debug = {
                "sitting_score": float(sit),
                "lean_back": float(lean_back),
                "hike_weight": float(hike_w),
                "stab_weight": float(stab_w),
                "alpha": None,
                "step_cap": None,
                "shift_y": 0.0,
                "raw_lock_gap": 0.0,
                "raw_lock_pull": 0.0,
            }
            return placed

        y_input = y_meas
        raw_lock_gap = 0.0
        raw_lock_pull = 0.0
        if (
            raw_reference is not None
            and 23 in raw_reference and 24 in raw_reference
            and raw_reference[23] is not None and raw_reference[24] is not None
        ):
            raw_l = np.asarray(raw_reference[23], dtype=np.float64).reshape(3)
            raw_r = np.asarray(raw_reference[24], dtype=np.float64).reshape(3)
            if np.all(np.isfinite(raw_l)) and np.all(np.isfinite(raw_r)):
                y_raw = float(0.5 * (raw_l[1] + raw_r[1]))
                raw_lock_gap = float(np.clip(
                    y_raw - y_meas,
                    -self.lock_max_dist_m,
                    self.lock_max_dist_m,
                ))
                raw_lock_pull = float(self.lock_gain * stab_w * raw_lock_gap)
                y_input = y_meas + raw_lock_pull

        step_cap = float(np.clip(
            self.step_cap_default_m - (self.step_cap_default_m - self.step_cap_hiking_m) * stab_w,
            self.step_cap_hiking_m,
            self.step_cap_default_m,
        ))
        alpha = float(np.clip(
            self.alpha_max - (self.alpha_max - self.alpha_min) * stab_w,
            self.alpha_min,
            self.alpha_max,
        ))

        if self._prev_hip_mid_y is None or not np.isfinite(self._prev_hip_mid_y):
            y_filtered = y_input
        else:
            y_capped = float(self._prev_hip_mid_y + np.clip(y_input - self._prev_hip_mid_y, -step_cap, step_cap))
            y_filtered = float(alpha * y_capped + (1.0 - alpha) * self._prev_hip_mid_y)

        dy = float(np.clip(y_filtered - y_meas, -self.max_shift_m, self.max_shift_m))
        self._prev_hip_mid_y = float(y_meas + dy)
        self.last_debug = {
            "sitting_score": float(sit),
            "lean_back": float(lean_back),
            "hike_weight": float(hike_w),
            "stab_weight": float(stab_w),
            "alpha": float(alpha),
            "step_cap": float(step_cap),
            "shift_y": float(dy),
            "raw_lock_gap": float(raw_lock_gap),
            "raw_lock_pull": float(raw_lock_pull),
        }
        if abs(dy) < 1e-9:
            return placed

        adjusted: Dict[int, np.ndarray] = {}
        for idx, p in placed.items():
            if p is None:
                adjusted[int(idx)] = p
                continue
            q = np.asarray(p, dtype=np.float64).reshape(3).copy()
            q[1] += dy
            adjusted[int(idx)] = q
        return adjusted


def _row_world_point(row, idx: int) -> Optional[np.ndarray]:
    x = row.get(f"lm{idx}_world_x")
    y = row.get(f"lm{idx}_world_y")
    z = row.get(f"lm{idx}_world_z")
    if x is None or y is None or z is None:
        return None
    if pd.isna(x) or pd.isna(y) or pd.isna(z):
        return None
    p = np.array([float(x), float(y), float(z)], dtype=np.float64)
    if not np.all(np.isfinite(p)):
        return None
    return p


def _estimate_sitting_score_from_row(row) -> float:
    """Estimate seated posture confidence from bilateral knee flexion.

    Returns [0,1], where 1 means strongly seated (deep knee bend).
    """
    def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
        v1 = np.asarray(a, dtype=np.float64).reshape(3) - np.asarray(b, dtype=np.float64).reshape(3)
        v2 = np.asarray(c, dtype=np.float64).reshape(3) - np.asarray(b, dtype=np.float64).reshape(3)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-8 or n2 < 1e-8:
            return None
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    knee_scores: List[float] = []
    for hip_idx, knee_idx, ankle_idx, foot_idx in ((23, 25, 27, 31), (24, 26, 28, 32)):
        hip = _row_world_point(row, hip_idx)
        knee = _row_world_point(row, knee_idx)
        ankle = _row_world_point(row, ankle_idx)
        if ankle is None:
            ankle = _row_world_point(row, foot_idx)
        if hip is None or knee is None or ankle is None:
            continue
        ang = _joint_angle_deg(hip, knee, ankle)
        if ang is None or not np.isfinite(ang):
            continue
        # ~170+ = straight leg (standing), ~90-110 = seated/bent.
        score = float(np.clip((145.0 - ang) / 55.0, 0.0, 1.0))
        knee_scores.append(score)

    if not knee_scores:
        return 0.0
    return float(np.clip(np.mean(np.asarray(knee_scores, dtype=np.float64)), 0.0, 1.0))


def _estimate_sitting_score_from_placed(placed: Optional[Dict[int, np.ndarray]]) -> float:
    """Estimate seated posture confidence from placed 3D skeleton."""
    if not placed:
        return 0.0

    def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
        v1 = np.asarray(a, dtype=np.float64).reshape(3) - np.asarray(b, dtype=np.float64).reshape(3)
        v2 = np.asarray(c, dtype=np.float64).reshape(3) - np.asarray(b, dtype=np.float64).reshape(3)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-8 or n2 < 1e-8:
            return None
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    knee_scores: List[float] = []
    for hip_idx, knee_idx, ankle_idx, foot_idx in ((23, 25, 27, 31), (24, 26, 28, 32)):
        hip = placed.get(int(hip_idx))
        knee = placed.get(int(knee_idx))
        ankle = placed.get(int(ankle_idx))
        if ankle is None:
            ankle = placed.get(int(foot_idx))
        if hip is None or knee is None or ankle is None:
            continue
        ang = _joint_angle_deg(hip, knee, ankle)
        if ang is None or not np.isfinite(ang):
            continue
        score = float(np.clip((145.0 - ang) / 55.0, 0.0, 1.0))
        knee_scores.append(score)

    if not knee_scores:
        return 0.0
    return float(np.clip(np.mean(np.asarray(knee_scores, dtype=np.float64)), 0.0, 1.0))


def create_seated_x_stabilizer(params: Optional[dict] = None) -> Optional[_SeatedForeAftStabilizer]:
    """Create seated fore-aft x stabilizer from config params."""
    cfg = normalize_seated_x_stabilizer_params(params)
    if not cfg.get("enabled", True):
        return None
    return _SeatedForeAftStabilizer(
        sit_start=float(cfg["sit_start"]),
        sit_full=float(cfg["sit_full"]),
        alpha_min=float(cfg["alpha_min"]),
        alpha_max=float(cfg["alpha_max"]),
        step_cap_seated_m=float(cfg["step_cap_seated_m"]),
        step_cap_default_m=float(cfg["step_cap_default_m"]),
        hike_relax=float(cfg["hike_relax"]),
        max_shift_m=float(cfg["max_shift_m"]),
        forward_release_gain=float(cfg["forward_release_gain"]),
        forward_release_max_m=float(cfg["forward_release_max_m"]),
    )


def create_lateral_y_stabilizer(params: Optional[dict] = None) -> Optional[_LateralYStabilizer]:
    """Create lateral y stabilizer from config params."""
    cfg = normalize_lateral_y_stabilizer_params(params)
    if not cfg.get("enabled", True):
        return None
    return _LateralYStabilizer(
        hike_start=float(cfg["hike_start"]),
        hike_full=float(cfg["hike_full"]),
        alpha_min=float(cfg["alpha_min"]),
        alpha_max=float(cfg["alpha_max"]),
        step_cap_hiking_m=float(cfg["step_cap_hiking_m"]),
        step_cap_default_m=float(cfg["step_cap_default_m"]),
        max_shift_m=float(cfg["max_shift_m"]),
        sit_relax=float(cfg["sit_relax"]),
        lock_gain=float(cfg["lock_gain"]),
        lock_max_dist_m=float(cfg["lock_max_dist_m"]),
    )


def _effective_lower_plane_z(z_plane_lm28: float, sitting_score: float) -> float:
    """Lower-limb target plane adjusted by seated posture confidence."""
    base = float(z_plane_lm28)
    sit = float(np.clip(sitting_score, 0.0, 1.0))
    z_eff = base - float(SEATED_LOWER_PLANE_DROP_PER_SCORE_M) * sit
    z_min = base - float(SEATED_LOWER_PLANE_MAX_DROP_M)
    return float(np.clip(z_eff, z_min, base))


def _raycast_intersection_for_landmark(
    row,
    idx: int,
    z_plane: float,
    K_undist,
    W,
    H,
    camera_pos: np.ndarray,
    R_wc: np.ndarray,
) -> Optional[np.ndarray]:
    lm_norm = get_landmark_norm(row, idx)
    if lm_norm is None:
        return None
    try:
        ray = ray_from_norm_landmark_undistorted(float(lm_norm[0]), float(lm_norm[1]), W, H, K_undist)
        p = intersect_world_z_plane(ray, R_wc=R_wc, t_wc=camera_pos, Z0=float(z_plane))
    except Exception:
        return None
    if p is None:
        return None
    p = np.asarray(p, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p)):
        return None
    return p


def _compute_placed_skeleton_symmetric_raycast_rigid(
    row,
    K_undist,
    W,
    H,
    camera_pos: np.ndarray,
    R_wc: np.ndarray,
    z_plane_lm24: float,
    z_plane_lm28: float,
    lower_landmark: str = "ankle",
) -> Optional[dict]:
    """Symmetric raycast-based rigid placement using both sides equally.

    This avoids side-dependent anchor logic and avoids hull sidewall snap artifacts.
    """
    if K_undist is None or W is None or H is None:
        return None

    if lower_landmark == "knee":
        left_low_idx, right_low_idx = 25, 26
    else:
        left_low_idx, right_low_idx = 27, 28
    sitting_score = _estimate_sitting_score_from_row(row)
    z_low_eff = _effective_lower_plane_z(float(z_plane_lm28), float(sitting_score))

    # Build symmetric source/target pairs for hips and lower limbs.
    pair_specs = (
        (23, float(z_plane_lm24)),
        (24, float(z_plane_lm24)),
        (left_low_idx, float(z_low_eff)),
        (right_low_idx, float(z_low_eff)),
    )
    src_pts: Dict[int, np.ndarray] = {}
    tgt_pts: Dict[int, np.ndarray] = {}
    for idx, z_plane in pair_specs:
        src_p = _row_world_point(row, int(idx))
        tgt_p = _raycast_intersection_for_landmark(
            row=row,
            idx=int(idx),
            z_plane=z_plane,
            K_undist=K_undist,
            W=W,
            H=H,
            camera_pos=np.asarray(camera_pos, dtype=np.float64).reshape(3),
            R_wc=np.asarray(R_wc, dtype=np.float64).reshape(3, 3),
        )
        if src_p is not None and tgt_p is not None:
            src_pts[int(idx)] = src_p
            tgt_pts[int(idx)] = tgt_p

    hip_ids = [i for i in (23, 24) if i in src_pts and i in tgt_pts]
    low_ids = [i for i in (left_low_idx, right_low_idx) if i in src_pts and i in tgt_pts]
    if not hip_ids or not low_ids:
        return None

    src_anchor = np.mean(np.asarray([src_pts[i] for i in hip_ids], dtype=np.float64), axis=0)
    tgt_anchor = np.mean(np.asarray([tgt_pts[i] for i in hip_ids], dtype=np.float64), axis=0)
    src_dir = np.mean(np.asarray([src_pts[i] for i in low_ids], dtype=np.float64), axis=0)
    tgt_dir = np.mean(np.asarray([tgt_pts[i] for i in low_ids], dtype=np.float64), axis=0)

    if np.linalg.norm(src_dir - src_anchor) < 1e-8 or np.linalg.norm(tgt_dir - tgt_anchor) < 1e-8:
        return None

    # Secondary roll cue from bilateral hips when available, else bilateral lower points.
    src_roll = None
    tgt_roll = None
    roll_idx = None
    if all(i in src_pts and i in tgt_pts for i in (23, 24)):
        src_roll = np.asarray(src_pts[24], dtype=np.float64)
        tgt_roll = np.asarray(tgt_pts[24], dtype=np.float64)
        roll_idx = 24
    elif all(i in src_pts and i in tgt_pts for i in (left_low_idx, right_low_idx)):
        src_roll = np.asarray(src_pts[right_low_idx], dtype=np.float64)
        tgt_roll = np.asarray(tgt_pts[right_low_idx], dtype=np.float64)
        roll_idx = int(right_low_idx)

    placed = place_skeleton_on_boat(
        row=row,
        anchor_idx=24 if 24 in src_pts else 23,
        dir_idx=int(right_low_idx if right_low_idx in src_pts else left_low_idx),
        anchor_boat=tgt_anchor,
        dir_boat=tgt_dir,
        use_scale=False,
        roll_idx=roll_idx,
        roll_boat=tgt_roll,
        src_anchor=src_anchor,
        src_dir=src_dir,
        src_roll=src_roll,
    )
    if placed is None:
        return None

    # Translation refinement: align constrained landmarks to their ray-plane targets.
    diffs = []
    for idx, tgt_p in tgt_pts.items():
        p = placed.get(int(idx))
        if p is None:
            continue
        p = np.asarray(p, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            continue
        diffs.append(np.asarray(tgt_p, dtype=np.float64).reshape(3) - p)
    if diffs:
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        delta = np.median(diffs_arr, axis=0)
        delta = np.asarray(delta, dtype=np.float64).reshape(3)

        # Keep fore-aft placement stable against hip-plane tuning:
        # derive x from lower-limb constraints when available.
        diffs_low_x: List[float] = []
        for idx, tgt_p in tgt_pts.items():
            if int(idx) not in (left_low_idx, right_low_idx, 29, 30, 31, 32):
                continue
            p = placed.get(int(idx))
            if p is None:
                continue
            p = np.asarray(p, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(p)):
                continue
            d = np.asarray(tgt_p, dtype=np.float64).reshape(3) - p
            if np.all(np.isfinite(d)):
                diffs_low_x.append(float(d[0]))
        if diffs_low_x:
            delta[0] = float(np.median(np.asarray(diffs_low_x, dtype=np.float64)))

        delta[0] = float(np.clip(delta[0], -2.0, 2.0))
        delta[1] = float(np.clip(delta[1], -2.0, 2.0))
        delta[2] = float(np.clip(delta[2], -0.8, 0.8))
        for idx, p in list(placed.items()):
            if p is None:
                continue
            q = np.asarray(p, dtype=np.float64).reshape(3) + delta
            placed[int(idx)] = q

    # Guard against floating lower body (common when side rays are noisy).
    lower_z = []
    for idx in (left_low_idx, right_low_idx, 31, 32, 29, 30):
        p = placed.get(int(idx))
        if p is None:
            continue
        z = float(np.asarray(p, dtype=np.float64).reshape(3)[2])
        if np.isfinite(z):
            lower_z.append(z)
    if lower_z:
        min_low_z = float(np.min(np.asarray(lower_z, dtype=np.float64)))
        if min_low_z > float(z_low_eff) + 0.06:
            dz = float(z_low_eff) + 0.02 - min_low_z
            dz = float(np.clip(dz, -0.6, 0.0))
            for idx, p in list(placed.items()):
                if p is None:
                    continue
                q = np.asarray(p, dtype=np.float64).reshape(3).copy()
                q[2] += dz
                placed[int(idx)] = q

    placed = _guard_hip_height_from_foot_lock(
        placed=placed,
        z_plane_lm24=float(z_plane_lm24),
        z_plane_lm28=float(z_low_eff),
        sitting_score=float(sitting_score),
    )
    placed = _apply_seated_pelvis_height_constraint(
        placed=placed,
        z_plane_lm24=float(z_plane_lm24),
        sitting_score=float(sitting_score),
    )

    # Final lateral correction from hip midpoint if both hips are constrained.
    if all(i in placed and placed[i] is not None for i in (23, 24)) and all(i in tgt_pts for i in (23, 24)):
        hip_mid_tgt = 0.5 * (np.asarray(tgt_pts[23], dtype=np.float64) + np.asarray(tgt_pts[24], dtype=np.float64))
        hip_mid_cur = 0.5 * (np.asarray(placed[23], dtype=np.float64) + np.asarray(placed[24], dtype=np.float64))
        dy = float(np.clip(hip_mid_tgt[1] - hip_mid_cur[1], -0.30, 0.30))
        if abs(dy) > 1e-9:
            for idx, p in list(placed.items()):
                if p is None:
                    continue
                q = np.asarray(p, dtype=np.float64).reshape(3).copy()
                q[1] += dy
                placed[int(idx)] = q

    return placed


def _compute_placed_skeleton_symmetric_raycast(
    row,
    K_undist,
    W,
    H,
    camera_pos: np.ndarray,
    R_wc: np.ndarray,
    z_plane_lm24: float,
    z_plane_lm28: float,
    lower_landmark: str = "ankle",
) -> Optional[dict]:
    """Symmetric raycast placement that preserves MediaPipe orientation.

    Unlike rigid reorientation, this keeps the raw MediaPipe body orientation in
    boat frame and solves only a global translation from symmetric ray-plane
    constraints. This matches the behavior users rely on for non-hiking postures.
    """
    if K_undist is None or W is None or H is None:
        return None

    lm_cam = _extract_camera_frame_landmarks(row)
    if not lm_cam:
        return None

    ids = np.array(sorted(lm_cam.keys()), dtype=np.int32)
    cam_pts = np.array([lm_cam[int(i)] for i in ids], dtype=np.float64)
    world_raw = (np.asarray(R_wc, dtype=np.float64).reshape(3, 3) @ cam_pts.T).T + np.asarray(camera_pos, dtype=np.float64).reshape(1, 3)
    idx_to_row = {int(idx): k for k, idx in enumerate(ids)}

    if lower_landmark == "knee":
        left_low_idx, right_low_idx = 25, 26
    else:
        left_low_idx, right_low_idx = 27, 28
    sitting_score = _estimate_sitting_score_from_row(row)
    z_low_eff = _effective_lower_plane_z(float(z_plane_lm28), float(sitting_score))

    # Ray constraints used to estimate a translation only.
    constraints = [
        (23, float(z_plane_lm24), 2.4, "hip"),
        (24, float(z_plane_lm24), 2.4, "hip"),
        (left_low_idx, float(z_low_eff), 1.9, "low"),
        (right_low_idx, float(z_low_eff), 1.9, "low"),
        (29, float(z_low_eff), 1.0, "low"),
        (30, float(z_low_eff), 1.0, "low"),
        (31, float(z_low_eff), 0.9, "low"),
        (32, float(z_low_eff), 0.9, "low"),
    ]

    diffs_all: List[tuple[np.ndarray, float]] = []
    diffs_hip: List[tuple[np.ndarray, float]] = []
    diffs_low: List[tuple[np.ndarray, float]] = []
    diffs_low_x_ref: List[tuple[float, float]] = []
    hip_targets: Dict[int, np.ndarray] = {}
    low_targets: Dict[int, np.ndarray] = {}

    for idx, z_plane, w, group in constraints:
        r = idx_to_row.get(int(idx))
        if r is None:
            continue
        tgt = _raycast_intersection_for_landmark(
            row=row,
            idx=int(idx),
            z_plane=float(z_plane),
            K_undist=K_undist,
            W=W,
            H=H,
            camera_pos=np.asarray(camera_pos, dtype=np.float64).reshape(3),
            R_wc=np.asarray(R_wc, dtype=np.float64).reshape(3, 3),
        )
        if tgt is None:
            continue
        src = np.asarray(world_raw[r], dtype=np.float64).reshape(3)
        d = np.asarray(tgt, dtype=np.float64).reshape(3) - src
        if not np.all(np.isfinite(d)):
            continue
        ww = float(max(w, 1e-6))
        diffs_all.append((d, ww))
        if group == "hip":
            diffs_hip.append((d, ww))
            hip_targets[int(idx)] = np.asarray(tgt, dtype=np.float64).reshape(3)
        else:
            diffs_low.append((d, ww))
            low_targets[int(idx)] = np.asarray(tgt, dtype=np.float64).reshape(3)
            # Keep lower-plane seated drop for vertical placement, but derive an
            # x-reference from the nominal lower plane to reduce seated x drift.
            if abs(float(z_low_eff) - float(z_plane_lm28)) > 1e-6:
                tgt_ref = _raycast_intersection_for_landmark(
                    row=row,
                    idx=int(idx),
                    z_plane=float(z_plane_lm28),
                    K_undist=K_undist,
                    W=W,
                    H=H,
                    camera_pos=np.asarray(camera_pos, dtype=np.float64).reshape(3),
                    R_wc=np.asarray(R_wc, dtype=np.float64).reshape(3, 3),
                )
                if tgt_ref is not None:
                    d_ref = np.asarray(tgt_ref, dtype=np.float64).reshape(3) - src
                    if np.all(np.isfinite(d_ref)):
                        diffs_low_x_ref.append((float(d_ref[0]), ww))

    if not diffs_all:
        return None

    def _wavg(pairs: List[tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
        if not pairs:
            return None
        arr = np.asarray([p[0] for p in pairs], dtype=np.float64)
        w = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None
        if not np.all(np.isfinite(arr)) or not np.all(np.isfinite(w)):
            return None
        if float(np.sum(w)) < 1e-9:
            w = np.ones_like(w)
        return np.average(arr, axis=0, weights=w)

    def _wavg_scalar(pairs: List[tuple[float, float]]) -> Optional[float]:
        if not pairs:
            return None
        vals = np.asarray([p[0] for p in pairs], dtype=np.float64)
        w = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if vals.ndim != 1 or not np.all(np.isfinite(vals)) or not np.all(np.isfinite(w)):
            return None
        if float(np.sum(w)) < 1e-9:
            w = np.ones_like(w)
        return float(np.average(vals, weights=w))

    def _lean_back_score() -> float:
        """Score pronounced layback posture from shoulder/hip geometry in boat frame."""
        req = (11, 12, 23, 24)
        if not all(int(i) in idx_to_row for i in req):
            return 0.0
        sh_l = np.asarray(world_raw[idx_to_row[11]], dtype=np.float64).reshape(3)
        sh_r = np.asarray(world_raw[idx_to_row[12]], dtype=np.float64).reshape(3)
        hip_l = np.asarray(world_raw[idx_to_row[23]], dtype=np.float64).reshape(3)
        hip_r = np.asarray(world_raw[idx_to_row[24]], dtype=np.float64).reshape(3)
        sh_mid = 0.5 * (sh_l + sh_r)
        hip_mid = 0.5 * (hip_l + hip_r)
        trunk = sh_mid - hip_mid
        trunk_z = float(abs(trunk[2]))
        if trunk_z < 1e-6:
            return 0.0
        # boat x is fore-aft: shoulders moving aft of hips => layback.
        lean_ratio = float((-trunk[0]) / trunk_z)
        return float(np.clip((lean_ratio - 0.12) / 0.55, 0.0, 1.0))

    d_all = _wavg(diffs_all)
    d_hip = _wavg(diffs_hip)
    d_low = _wavg(diffs_low)
    d_low_x_ref = _wavg_scalar(diffs_low_x_ref)
    if d_all is None:
        return None

    lean_back_score = _lean_back_score()
    hiking_score = 0.0

    if d_hip is not None and d_low is not None:
        delta = np.zeros(3, dtype=np.float64)
        # Lateral placement should be driven by hips to avoid side inversion.
        delta[1] = 0.80 * d_hip[1] + 0.20 * d_all[1]
        # Seated posture: rely more on lower-limb constraints for z to avoid high hips.
        low_z_w = float(np.clip(0.70 + 0.18 * sitting_score, 0.70, 0.90))
        hip_z_w = float(1.0 - low_z_w)
        delta[2] = low_z_w * d_low[2] + hip_z_w * d_hip[2]
        hip_sink_score = float(np.clip((float(d_hip[2] - d_low[2]) - 0.04) / 0.26, 0.0, 1.0))
        d_low_x = float(d_low[0])
        if d_low_x_ref is not None:
            seated_non_hike = float(np.clip(float(sitting_score) * (1.0 - lean_back_score), 0.0, 1.0))
            low_drop = float(np.clip(float(z_plane_lm28) - float(z_low_eff), 0.0, 0.5))
            drop_w = float(np.clip(low_drop / 0.16, 0.0, 1.0))
            x_ref_w = float(np.clip(seated_non_hike * drop_w, 0.0, 1.0))
            d_low_x = float((1.0 - x_ref_w) * d_low_x + x_ref_w * float(d_low_x_ref))
        # In seated/non-hiking frames, cap how far aft lower-limb x can pull
        # relative to hip x to reduce hip-height-induced aft placement.
        seated_non_hike_for_gap = float(np.clip(float(sitting_score) * (1.0 - lean_back_score), 0.0, 1.0))
        aft_gap = float(d_hip[0] - d_low_x)
        max_aft_gap = float(np.clip(0.26 - 0.18 * seated_non_hike_for_gap, 0.08, 0.26))
        if aft_gap > max_aft_gap:
            d_low_x = float(d_hip[0] - max_aft_gap)
        x_disagreement_score = float(np.clip((abs(float(d_hip[0] - d_low_x)) - 0.03) / 0.25, 0.0, 1.0))
        # Fore-aft translation should usually follow lower-limb constraints to
        # stabilize x against hip-plane tuning, but seated non-hiking frames can
        # over-shift aft when hip/lower cues disagree. In that case, reduce the
        # lower-limb dominance and blend in hip/all-body cues.
        low_support = float(np.clip(len(diffs_low) / 3.0, 0.0, 1.0))
        hiking_pose_score = float(np.clip(0.70 * lean_back_score + 0.30 * hip_sink_score, 0.0, 1.0))
        seated_non_hike = float(np.clip(float(sitting_score) * (1.0 - hiking_pose_score), 0.0, 1.0))
        seated_disagreement = float(seated_non_hike * x_disagreement_score)
        low_x_w = float(np.clip(
            0.72 + 0.16 * low_support - 0.22 * seated_non_hike - 0.24 * seated_disagreement,
            0.38,
            0.90,
        ))
        base_mix_dx = 0.60 * float(d_all[0]) + 0.40 * float(d_hip[0])
        base_dx = low_x_w * d_low_x + (1.0 - low_x_w) * base_mix_dx
        hiking_score = float(np.clip(
            0.55 * lean_back_score + 0.30 * hip_sink_score + 0.15 * x_disagreement_score,
            0.0,
            1.0,
        ))
        dx_blend = 0.75 * hiking_score * low_support
        delta[0] = (1.0 - dx_blend) * base_dx + dx_blend * d_low_x
    elif d_hip is not None:
        delta = np.asarray(d_hip, dtype=np.float64).reshape(3)
    elif d_low is not None:
        delta = np.asarray(d_low, dtype=np.float64).reshape(3)
    else:
        delta = np.asarray(d_all, dtype=np.float64).reshape(3)

    delta[0] = float(np.clip(delta[0], -2.0, 2.0))
    delta[1] = float(np.clip(delta[1], -2.0, 2.0))
    delta[2] = float(np.clip(delta[2], -0.8, 0.8))

    pts = world_raw + delta.reshape(1, 3)
    placed: Dict[int, np.ndarray] = {int(ids[k]): np.asarray(pts[k], dtype=np.float64) for k in range(len(ids))}

    # Global lateral re-balance from both hips and ankles.
    # This avoids hard hip side-lock and prevents leg stretching artifacts.
    dy_vals: List[float] = []
    dy_wts: List[float] = []

    if all(i in hip_targets for i in (23, 24)) and all(i in placed and placed[i] is not None for i in (23, 24)):
        hip_tgt_mid = 0.5 * (hip_targets[23] + hip_targets[24])
        hip_cur_mid = 0.5 * (np.asarray(placed[23], dtype=np.float64) + np.asarray(placed[24], dtype=np.float64))
        dy_vals.append(float(hip_tgt_mid[1] - hip_cur_mid[1]))
        dy_wts.append(0.35)

    if all(i in low_targets for i in (27, 28)) and all(i in placed and placed[i] is not None for i in (27, 28)):
        ankle_tgt_mid = 0.5 * (low_targets[27] + low_targets[28])
        ankle_cur_mid = 0.5 * (np.asarray(placed[27], dtype=np.float64) + np.asarray(placed[28], dtype=np.float64))
        dy_vals.append(float(ankle_tgt_mid[1] - ankle_cur_mid[1]))
        dy_wts.append(0.65)

    if dy_vals:
        dy = float(np.average(np.asarray(dy_vals, dtype=np.float64), weights=np.asarray(dy_wts, dtype=np.float64)))
        dy = float(np.clip(dy, -0.30, 0.30))
        if abs(dy) > 1e-9:
            for idx, p in list(placed.items()):
                q = np.asarray(p, dtype=np.float64).reshape(3).copy()
                q[1] += dy
                placed[int(idx)] = q

    # Lower-body height guard.
    lower_z = []
    for idx in (left_low_idx, right_low_idx, 29, 30, 31, 32):
        p = placed.get(int(idx))
        if p is None:
            continue
        z = float(np.asarray(p, dtype=np.float64).reshape(3)[2])
        if np.isfinite(z):
            lower_z.append(z)
    if lower_z:
        min_low_z = float(np.min(np.asarray(lower_z, dtype=np.float64)))
        max_low_z = float(np.max(np.asarray(lower_z, dtype=np.float64)))
        if min_low_z > float(z_low_eff) + 0.05:
            dz = float(z_low_eff) + 0.02 - min_low_z
            dz = float(np.clip(dz, -0.6, 0.0))
            for idx, p in list(placed.items()):
                q = np.asarray(p, dtype=np.float64).reshape(3).copy()
                q[2] += dz
                placed[int(idx)] = q
        elif max_low_z < float(z_low_eff) - 0.45:
            dz = float(z_low_eff) - 0.25 - max_low_z
            dz = float(np.clip(dz, 0.0, 0.25))
            for idx, p in list(placed.items()):
                q = np.asarray(p, dtype=np.float64).reshape(3).copy()
                q[2] += dz
                placed[int(idx)] = q

    placed = _guard_hip_height_from_foot_lock(
        placed=placed,
        z_plane_lm24=float(z_plane_lm24),
        z_plane_lm28=float(z_low_eff),
        sitting_score=float(sitting_score),
    )
    placed = _apply_seated_pelvis_height_constraint(
        placed=placed,
        z_plane_lm24=float(z_plane_lm24),
        sitting_score=float(sitting_score),
    )

    # Keep MediaPipe orientation by default. Only use rigid rescue when
    # both head and shoulders are strongly below the hip midpoint, which is
    # a much stronger inversion signal than the previous broad heuristics.
    need_rigid_rescue = False
    if all(k in placed and placed[k] is not None for k in (0, 11, 12, 23, 24)):
        sh_mid = 0.5 * (np.asarray(placed[11], dtype=np.float64) + np.asarray(placed[12], dtype=np.float64))
        hip_mid = 0.5 * (np.asarray(placed[23], dtype=np.float64) + np.asarray(placed[24], dtype=np.float64))
        head = np.asarray(placed[0], dtype=np.float64)
        shoulder_delta = float(sh_mid[2] - hip_mid[2])
        head_delta = float(head[2] - hip_mid[2])
        if shoulder_delta < -0.20 and head_delta < -0.20:
            need_rigid_rescue = True

    if need_rigid_rescue:
        placed_rigid = _compute_placed_skeleton_symmetric_raycast_rigid(
            row=row,
            K_undist=K_undist,
            W=W,
            H=H,
            camera_pos=np.asarray(camera_pos, dtype=np.float64).reshape(3),
            R_wc=np.asarray(R_wc, dtype=np.float64).reshape(3, 3),
            z_plane_lm24=float(z_plane_lm24),
            z_plane_lm28=float(z_low_eff),
            lower_landmark=lower_landmark,
        )
        if placed_rigid is not None:
            return placed_rigid

    return placed


def _compute_placed_skeleton_contact(
    row,
    camera_pos: np.ndarray,
    R_wc: np.ndarray,
    z_plane_lm24: float,
    z_plane_lm28: float,
    contact_params: Optional[dict] = None,
    K_undist=None,
    W=None,
    H=None,
    lower_landmark: str = "ankle",
) -> Optional[dict]:
    """Place skeleton by fitting transformed landmarks to hull contact/non-penetration."""
    if not TRIMESH_AVAILABLE:
        return None

    mesh = _load_hull_mesh_boat_frame()
    if mesh is None:
        return None

    lm_cam = _extract_camera_frame_landmarks(row)
    if not lm_cam:
        return None
    sitting_score = _estimate_sitting_score_from_row(row)
    z_low_eff = _effective_lower_plane_z(float(z_plane_lm28), float(sitting_score))

    ids = np.array(sorted(lm_cam.keys()), dtype=np.int32)
    cam_pts = np.array([lm_cam[int(i)] for i in ids], dtype=np.float64)

    # Convert camera-frame landmarks to world/boat frame using solved camera extrinsics.
    world_raw = (R_wc @ cam_pts.T).T + np.asarray(camera_pos, dtype=np.float64).reshape(1, 3)

    # Absolute depth/translation is ambiguous in MediaPipe world space.
    # Initialize a global translation from ray-plane anchors to place the body
    # in the boat/world frame before contact fitting.
    if K_undist is not None and W is not None and H is not None:
        try:
            anchors = raycast_anchors(
                row=row,
                K_undist=K_undist,
                W=W,
                H=H,
                camera_pos=camera_pos,
                R_wc=R_wc,
                z_plane_lm24=z_plane_lm24,
                z_plane_lm28=z_low_eff,
                lower_landmark=lower_landmark,
            )
        except Exception:
            anchors = None
        if anchors is not None:
            idx_to_row = {int(idx): k for k, idx in enumerate(ids)}
            anchor_shift = None
            dir_shift = None
            src_anchor = anchors.get("src_anchor")
            if src_anchor is not None and anchors.get("anchor_int") is not None:
                src_anchor_w = (R_wc @ np.asarray(src_anchor, dtype=np.float64).reshape(3)) + np.asarray(camera_pos, dtype=np.float64).reshape(3)
                anchor_shift = np.asarray(anchors["anchor_int"], dtype=np.float64) - src_anchor_w
            src_dir = anchors.get("src_dir")
            if src_dir is not None and anchors.get("dir_int") is not None:
                src_dir_w = (R_wc @ np.asarray(src_dir, dtype=np.float64).reshape(3)) + np.asarray(camera_pos, dtype=np.float64).reshape(3)
                dir_shift = np.asarray(anchors["dir_int"], dtype=np.float64) - src_dir_w
            if anchor_shift is None and dir_shift is None:
                # Backward-compatible fallback for legacy anchor definitions.
                a_idx = int(anchors["anchor_idx"])
                d_idx = int(anchors["dir_idx"])
                if a_idx in idx_to_row and anchors.get("anchor_int") is not None:
                    anchor_shift = np.asarray(anchors["anchor_int"], dtype=np.float64) - world_raw[idx_to_row[a_idx]]
                if d_idx in idx_to_row and anchors.get("dir_int") is not None:
                    dir_shift = np.asarray(anchors["dir_int"], dtype=np.float64) - world_raw[idx_to_row[d_idx]]

            if anchor_shift is not None and dir_shift is not None:
                shift = 0.5 * (anchor_shift + dir_shift)
                # Lateral placement should follow hip anchor to avoid side bias.
                shift[1] = anchor_shift[1]
            elif anchor_shift is not None:
                shift = anchor_shift
            elif dir_shift is not None:
                shift = dir_shift
            else:
                shift = None

            if shift is not None:
                world_raw = world_raw + np.asarray(shift, dtype=np.float64).reshape(1, 3)

    contact_mask = np.isin(ids, np.array(CONTACT_BODY_INDICES, dtype=np.int32))
    if not np.any(contact_mask):
        return {int(ids[k]): world_raw[k] for k in range(len(ids))}

    cp_cfg = normalize_contact_params(contact_params)
    contact_weight_scalar = float(cp_cfg["contact_weight"])
    penetration_weight = float(cp_cfg["penetration_weight"])
    snap_weight = float(cp_cfg["snap_weight"])
    snap_max_dist = float(cp_cfg["snap_max_dist_m"])
    use_optimizer = bool(cp_cfg["use_optimizer"])

    cp0, d0 = _safe_closest_point(mesh, world_raw[contact_mask])
    if cp0 is None or d0 is None:
        return None

    w_contact = np.array([CONTACT_WEIGHTS.get(int(i), 1.0) for i in ids[contact_mask]], dtype=np.float64)
    w_contact = w_contact * max(contact_weight_scalar, 0.0)
    if float(np.sum(w_contact)) < 1e-9:
        w_contact = np.ones_like(w_contact)
    delta0 = np.average(cp0 - world_raw[contact_mask], axis=0, weights=w_contact)
    # Preserve lateral placement from ray initialization to avoid side-dependent
    # inward drift from nearest-surface projections.
    delta0[1] = 0.0

    best_delta = delta0.copy()
    if use_optimizer and SCIPY_OPT_AVAILABLE:
        idx_map = {int(idx): k for k, idx in enumerate(ids)}

        def objective(delta_vec):
            delta_vec = np.asarray(delta_vec, dtype=np.float64).reshape(3)
            pts = world_raw + delta_vec.reshape(1, 3)
            cp, dist = _safe_closest_point(mesh, pts)
            if cp is None or dist is None or np.any(~np.isfinite(dist)):
                return 1e9

            # Contact: pull lower-body joints onto surface.
            d_contact = dist[contact_mask]
            contact_cost = float(np.average(d_contact * d_contact, weights=w_contact))

            # Non-penetration: penalize points classified as inside mesh.
            inside = _safe_contains(mesh, pts)
            non_pen_cost = 0.0
            if np.any(inside):
                non_pen_cost = float(np.mean((dist[inside] + 1e-4) ** 2))

            # Weak vertical priors for hips/ankles to preserve plausible placement.
            z_cost = 0.0
            for lm_idx, z_target, w in (
                (23, z_plane_lm24, 0.70),
                (24, z_plane_lm24, 0.70),
                (27, z_low_eff, 0.35),
                (28, z_low_eff, 0.35),
            ):
                k = idx_map.get(lm_idx)
                if k is not None:
                    dz = float(pts[k, 2] - z_target)
                    z_cost += w * dz * dz

            reg_cost = float(np.sum((delta_vec - delta0) ** 2))
            lateral_lock_cost = float(delta_vec[1] * delta_vec[1])
            return 8.0 * contact_cost + 35.0 * non_pen_cost + 0.42 * z_cost + 0.02 * reg_cost + 2.0 * lateral_lock_cost

        try:
            res = minimize(
                objective,
                x0=delta0,
                method="Powell",
                options={"maxiter": 120, "xtol": 1e-3, "ftol": 1e-4, "disp": False},
            )
            if res is not None and np.all(np.isfinite(res.x)):
                best_delta = np.asarray(res.x, dtype=np.float64).reshape(3)
        except Exception:
            pass

    pts = world_raw + best_delta.reshape(1, 3)
    cp, dist = _safe_closest_point(mesh, pts)
    if cp is None or dist is None:
        return None
    inside = _safe_contains(mesh, pts)

    placed: Dict[int, np.ndarray] = {}
    snap_set = set(CONTACT_SNAP_INDICES)
    for k, idx in enumerate(ids):
        p = pts[k]
        if inside[k]:
            corr = np.clip(penetration_weight, 0.0, 2.0) * (cp[k] - p)
            if int(idx) in (23, 24):
                corr = corr.copy()
                corr[1] = 0.0
            p = p + corr
        elif int(idx) in snap_set and dist[k] < snap_max_dist:
            corr = np.clip(snap_weight, 0.0, 2.0) * (cp[k] - p)
            if int(idx) in (23, 24):
                corr = corr.copy()
                corr[1] = 0.0
            p = p + corr
        placed[int(idx)] = np.asarray(p, dtype=np.float64)
    placed = _guard_vertical_foot_outliers(
        placed=placed,
        z_plane_lm28=float(z_low_eff),
        snap_max_dist=float(snap_max_dist),
    )
    placed = _guard_hip_height_from_foot_lock(
        placed=placed,
        z_plane_lm24=float(z_plane_lm24),
        z_plane_lm28=float(z_low_eff),
        sitting_score=float(sitting_score),
    )
    placed = _apply_seated_pelvis_height_constraint(
        placed=placed,
        z_plane_lm24=float(z_plane_lm24),
        sitting_score=float(sitting_score),
    )
    return placed


def _parse_camera_rotation_matrix(value: Any) -> Optional[np.ndarray]:
    """Parse a 3x3 camera->world rotation matrix from config payload."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    try:
        R = np.asarray(value, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    if not np.all(np.isfinite(R)):
        return None
    return R


def get_camera_pose_from_config(config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Resolve camera position and camera->world rotation from project config."""
    cam_pos_cfg = config.get("camera_position", [-3.374, 0.0, 0.5])
    camera_pos = np.array(cam_pos_cfg, dtype=np.float64)

    R_cfg = _parse_camera_rotation_matrix(config.get("camera_R_wc"))
    if R_cfg is not None:
        return camera_pos, R_cfg

    _, R_wc = default_camera_pose_and_rotation(
        pitch=config.get("camera_pitch_deg", 8.0),
        yaw=config.get("camera_yaw_deg", 0.0),
        roll=config.get("camera_roll_deg", 0.0),
    )
    return camera_pos, R_wc


def _rotation_matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    """Extract XYZ Euler angles (rx, ry, rz) in degrees from R = Rz * Ry * Rx."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    ry = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
        rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        rx = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        rz = 0.0
    return np.degrees([rx, ry, rz])


def _camera_pose_angles_from_rwc(R_wc: np.ndarray) -> Dict[str, float]:
    """Return camera pose angles with app convention (pitch positive = looking down)."""
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    R_rel = AUTO_CAMERA_BASE_R_WC.T @ R_wc
    rx, ry, rz = _rotation_matrix_to_euler_xyz_deg(R_rel)
    return {
        "pitch_deg": float(-rx),
        "yaw_deg": float(ry),
        "roll_deg": float(rz),
    }


def _wrap_deg(angle_deg: float) -> float:
    """Wrap degrees to [-180, 180)."""
    a = float(angle_deg)
    return ((a + 180.0) % 360.0) - 180.0


def _average_auto_pnp_solutions(solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Robustly aggregate multiple frame-level auto-PnP solutions.

    Uses median statistics and MAD-based outlier rejection.
    """
    if not solutions:
        return {"success": False, "reason": "No successful frame-level PnP solutions to aggregate"}

    cam_list = []
    pitch_list = []
    yaw_list = []
    roll_list = []
    err_list = []
    inlier_list = []
    pair_list = []
    mapping_list: List[str] = []
    solver_list: List[str] = []

    for s in solutions:
        cam = np.asarray(s.get("camera_pos"), dtype=np.float64).reshape(3)
        ang = s.get("angles", {})
        p = float(ang.get("pitch_deg", np.nan))
        y = _wrap_deg(float(ang.get("yaw_deg", np.nan)))
        r = _wrap_deg(-float(ang.get("roll_deg", np.nan)))
        e = float(s.get("err_px", np.nan))
        if not (np.all(np.isfinite(cam)) and np.isfinite(p) and np.isfinite(y) and np.isfinite(r)):
            continue
        cam_list.append(cam)
        pitch_list.append(p)
        yaw_list.append(y)
        roll_list.append(r)
        err_list.append(e if np.isfinite(e) else np.nan)
        inlier_list.append(int(s.get("num_inliers", 0)))
        pair_list.append(int(s.get("num_pairs", 0)))
        mapping_list.append(str(s.get("mapping", "unknown")))
        solver_list.append(str(s.get("solve_method", "unknown")))

    n = len(cam_list)
    if n == 0:
        return {"success": False, "reason": "No finite frame-level camera solutions to aggregate"}

    cam = np.asarray(cam_list, dtype=np.float64)  # N x 3
    pitch = np.asarray(pitch_list, dtype=np.float64)
    yaw = np.asarray(yaw_list, dtype=np.float64)
    roll = np.asarray(roll_list, dtype=np.float64)
    err = np.asarray(err_list, dtype=np.float64)

    keep = np.ones(n, dtype=bool)
    if n >= 3:
        cam_med = np.median(cam, axis=0)
        cam_mad = np.median(np.abs(cam - cam_med[None, :]), axis=0)
        cam_sigma = 1.4826 * cam_mad
        cam_floor = np.array([0.20, 0.08, 0.08], dtype=np.float64)  # m
        cam_thr = np.maximum(3.0 * cam_sigma, cam_floor)
        keep_cam = np.all(np.abs(cam - cam_med[None, :]) <= cam_thr[None, :], axis=1)

        pitch_med = float(np.median(pitch))
        pitch_mad = float(np.median(np.abs(pitch - pitch_med)))
        pitch_thr = max(3.0 * 1.4826 * pitch_mad, 2.0)  # deg
        keep_pitch = np.abs(pitch - pitch_med) <= pitch_thr

        yaw_med = float(np.median(yaw))
        yaw_diff = np.array([abs(_wrap_deg(v - yaw_med)) for v in yaw], dtype=np.float64)
        yaw_mad = float(np.median(yaw_diff))
        yaw_thr = max(3.0 * 1.4826 * yaw_mad, 2.0)  # deg
        keep_yaw = yaw_diff <= yaw_thr

        roll_med = float(np.median(roll))
        roll_diff = np.array([abs(_wrap_deg(v - roll_med)) for v in roll], dtype=np.float64)
        roll_mad = float(np.median(roll_diff))
        roll_thr = max(3.0 * 1.4826 * roll_mad, 2.0)  # deg
        keep_roll = roll_diff <= roll_thr

        keep_err = np.ones(n, dtype=bool)
        finite_err = np.isfinite(err)
        if np.any(finite_err):
            e = err[finite_err]
            err_med = float(np.median(e))
            err_mad = float(np.median(np.abs(e - err_med)))
            err_thr = err_med + max(3.0 * 1.4826 * err_mad, 5.0)  # px
            keep_err = ~finite_err | (err <= err_thr)

        keep = keep_cam & keep_pitch & keep_yaw & keep_roll & keep_err
        min_keep = min(n, max(3, int(np.ceil(0.6 * n))))
        if int(np.count_nonzero(keep)) < min_keep:
            err_rank = np.where(np.isfinite(err), err, 1e9)
            order = np.argsort(err_rank)
            keep = np.zeros(n, dtype=bool)
            keep[order[:min_keep]] = True

    cam_sel = cam[keep]
    pitch_sel = pitch[keep]
    yaw_sel = yaw[keep]
    roll_sel = roll[keep]
    err_sel = err[keep]
    inlier_sel = np.asarray(inlier_list, dtype=np.float64)[keep]
    pair_sel = np.asarray(pair_list, dtype=np.float64)[keep]
    mapping_sel = [mapping_list[i] for i in np.where(keep)[0].tolist()]

    camera_pos = np.median(cam_sel, axis=0)
    pitch_med = float(np.median(pitch_sel))
    yaw_med = _wrap_deg(float(np.median(yaw_sel)))
    roll_med = _wrap_deg(float(np.median(roll_sel)))

    _, R_wc = default_camera_pose_and_rotation(
        pitch=pitch_med if np.isfinite(pitch_med) else 0.0,
        yaw=yaw_med if np.isfinite(yaw_med) else 0.0,
        roll=roll_med if np.isfinite(roll_med) else 0.0,
    )
    angles = _camera_pose_angles_from_rwc(R_wc)
    angles["yaw_deg"] = _wrap_deg(angles["yaw_deg"])
    angles["roll_deg"] = _wrap_deg(angles["roll_deg"])

    finite_err_sel = err_sel[np.isfinite(err_sel)]
    err_px = float(np.median(finite_err_sel)) if finite_err_sel.size else float("nan")
    med_inliers = float(np.median(inlier_sel)) if inlier_sel.size else 0.0
    med_pairs = float(np.median(pair_sel)) if pair_sel.size else 0.0
    mapping_counts: Dict[str, int] = {}
    for m in mapping_sel:
        mapping_counts[m] = mapping_counts.get(m, 0) + 1
    mapping = max(mapping_counts.items(), key=lambda kv: kv[1])[0] if mapping_counts else "unknown"
    solver_sel = [solver_list[i] for i in np.where(keep)[0].tolist()]
    solver_counts: Dict[str, int] = {}
    for s in solver_sel:
        solver_counts[s] = solver_counts.get(s, 0) + 1
    solve_method = max(solver_counts.items(), key=lambda kv: kv[1])[0] if solver_counts else "unknown"

    return {
        "success": True,
        "camera_pos": np.asarray(camera_pos, dtype=np.float64).reshape(3),
        "R_wc": np.asarray(R_wc, dtype=np.float64).reshape(3, 3),
        "angles": angles,
        "err_px": err_px,
        "num_inliers": int(round(med_inliers)),
        "num_pairs": int(round(med_pairs)),
        "mapping": mapping,
        "solve_method": solve_method,
        "num_frame_solutions": int(n),
        "num_used_after_filter": int(np.count_nonzero(keep)),
        "num_rejected_outliers": int(n - np.count_nonzero(keep)),
        "aggregation": "median_mad_trim",
    }


def _create_auto_camera_pnp_model_instance():
    """Create one YOLO model instance for auto camera-PnP."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed; auto camera PnP is unavailable.") from exc
    if not AUTO_CAMERA_PNP_MODEL_PATH.exists():
        raise RuntimeError(f"Auto camera PnP model missing: {AUTO_CAMERA_PNP_MODEL_PATH}")
    return YOLO(str(AUTO_CAMERA_PNP_MODEL_PATH))


def _get_auto_camera_pnp_model():
    """Lazy-load shared YOLO model used for auto camera pose updates."""
    global _auto_camera_pnp_model
    with _auto_camera_pnp_model_lock:
        if _auto_camera_pnp_model is None:
            _auto_camera_pnp_model = _create_auto_camera_pnp_model_instance()
        return _auto_camera_pnp_model


def _bind_pnp_keypoints_geometry(points_xy: np.ndarray) -> Dict[str, int]:
    """Bind detected keypoints to semantic labels using geometric structure."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = len(AUTO_CAMERA_PNP_KEYPOINT_LABELS)
    if pts.shape[0] < n:
        raise ValueError(f"Expected {n} keypoints, got {pts.shape[0]}")
    pts = pts[:n]
    if not np.all(np.isfinite(pts)):
        raise ValueError("Detected keypoints contain non-finite values")

    front_idx = int(np.argmin(pts[:, 1]))  # highest in image
    front_pt = pts[front_idx]

    remaining = [i for i in range(n) if i != front_idx]
    rem_pts = pts[remaining]
    x_order = np.argsort(rem_pts[:, 0])
    left_side = [remaining[i] for i in x_order[:4]]
    right_side = [remaining[i] for i in x_order[4:]]

    def split_side(side_indices: List[int]) -> Dict[str, int]:
        side_pts = pts[side_indices]
        dist = np.linalg.norm(side_pts - front_pt[None, :], axis=1)
        back_local = int(np.argmax(dist))
        back_idx = int(side_indices[back_local])
        rail_idx = [idx for idx in side_indices if idx != back_idx]
        rail_pts = pts[rail_idx]
        rail_order = [rail_idx[i] for i in np.argsort(rail_pts[:, 1])]
        return {
            "top": int(rail_order[0]),
            "mid": int(rail_order[1]),
            "low": int(rail_order[2]),
            "back": int(back_idx),
        }

    left = split_side(left_side)
    right = split_side(right_side)
    return {
        "frontdeck": front_idx,
        "porttop": left["top"],
        "portmid": left["mid"],
        "portlow": left["low"],
        "starboardtop": right["top"],
        "starboardmid": right["mid"],
        "starboardlow": right["low"],
        "portback": left["back"],
        "starboardback": right["back"],
    }


def _swap_port_starboard_mapping(label_to_index: Dict[str, int]) -> Dict[str, int]:
    """Return a mapping variant with port/starboard labels swapped."""
    swapped = dict(label_to_index)
    for pair in (
        ("porttop", "starboardtop"),
        ("portmid", "starboardmid"),
        ("portlow", "starboardlow"),
        ("portback", "starboardback"),
    ):
        a, b = pair
        swapped[a], swapped[b] = swapped[b], swapped[a]
    return swapped


def _build_pnp_correspondences(
    raw_pts: np.ndarray,
    raw_conf: np.ndarray,
    label_to_index: Dict[str, int],
    min_kpt_conf: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build filtered 3D/2D correspondences from keypoints."""
    obj_pts = []
    img_pts = []
    for label in AUTO_CAMERA_PNP_KEYPOINT_LABELS:
        idx = int(label_to_index[label])
        uv = np.asarray(raw_pts[idx], dtype=np.float64).reshape(2)
        conf = float(raw_conf[idx]) if np.isfinite(raw_conf[idx]) else np.nan
        if not np.all(np.isfinite(uv)):
            continue
        if np.isfinite(conf) and conf < min_kpt_conf:
            continue
        obj_pts.append(np.asarray(AUTO_CAMERA_PNP_OBJECT_POINTS[label], dtype=np.float64))
        img_pts.append(uv)

    if not obj_pts:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), 0
    return (
        np.asarray(obj_pts, dtype=np.float64),
        np.asarray(img_pts, dtype=np.float64),
        len(obj_pts),
    )


def _solve_pnp_pose(
    obj: np.ndarray,
    img: np.ndarray,
    K_undist: np.ndarray,
) -> Dict[str, Any]:
    """Solve PnP with robust candidate selection and diagnostics.

    Tries multiple solvers and scores each candidate using:
    - full-point residual quality (median error),
    - inlier ratio,
    - soft plausibility penalties in corrected pose space.
    """
    if obj.shape[0] < 4:
        return {"ok": False, "error": "Need at least 4 correspondences"}

    K = np.asarray(K_undist, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((4, 1), dtype=np.float64)

    def _range_violation(v: float, lo: float, hi: float) -> float:
        val = float(v)
        span = max(1e-6, abs(hi - lo))
        if val < lo:
            return (lo - val) / span
        if val > hi:
            return (val - hi) / span
        return 0.0

    def _abs_violation(v: float, lim: float) -> float:
        val = abs(float(v))
        return max(val - lim, 0.0) / max(1e-6, lim)

    def _corrected_pose_from_rt(rvec_local: np.ndarray, tvec_local: np.ndarray):
        # Raw camera pose from OpenCV solve.
        R_cw_local, _ = cv2.Rodrigues(rvec_local)
        R_wc_raw_local = R_cw_local.T
        camera_pos_raw_local = (-R_wc_raw_local @ tvec_local).reshape(3)
        raw_angles_local = _camera_pose_angles_from_rwc(R_wc_raw_local)

        # Apply user-requested correction for operational pose outputs.
        _, R_wc_corr_local = default_camera_pose_and_rotation(
            pitch=raw_angles_local["pitch_deg"],
            yaw=raw_angles_local["yaw_deg"],
            roll=(raw_angles_local["roll_deg"] + AUTO_CAMERA_PNP_ROLL_OFFSET_DEG),
        )
        camera_pos_corr_local = np.array(
            [camera_pos_raw_local[0], camera_pos_raw_local[2], camera_pos_raw_local[1]],
            dtype=np.float64,
        )
        angles_corr_local = _camera_pose_angles_from_rwc(R_wc_corr_local)
        angles_corr_local["yaw_deg"] = _wrap_deg(angles_corr_local["yaw_deg"])
        angles_corr_local["roll_deg"] = _wrap_deg(angles_corr_local["roll_deg"])
        return R_wc_raw_local, camera_pos_raw_local, R_wc_corr_local, camera_pos_corr_local, angles_corr_local

    def _build_candidate(method: str, rvec_local: np.ndarray, tvec_local: np.ndarray, inlier_idx_local: np.ndarray):
        inlier_idx_local = np.asarray(inlier_idx_local, dtype=np.int32).reshape(-1)
        inlier_idx_local = inlier_idx_local[(inlier_idx_local >= 0) & (inlier_idx_local < len(obj))]
        if len(inlier_idx_local) == 0:
            inlier_idx_local = np.arange(len(obj), dtype=np.int32)

        (
            R_wc_raw_local,
            camera_pos_raw_local,
            _R_wc_corr_local,
            camera_pos_corr_local,
            angles_corr_local,
        ) = _corrected_pose_from_rt(rvec_local, tvec_local)

        proj_local, _ = cv2.projectPoints(obj, rvec_local, tvec_local, K, dist)
        proj_local = proj_local.reshape(-1, 2)
        residuals_local = np.linalg.norm(proj_local - img, axis=1)

        mean_all = float(np.mean(residuals_local)) if len(residuals_local) else float("inf")
        median_all = float(np.median(residuals_local)) if len(residuals_local) else float("inf")
        max_all = float(np.max(residuals_local)) if len(residuals_local) else float("inf")
        inlier_err = float(np.mean(residuals_local[inlier_idx_local])) if len(inlier_idx_local) else mean_all

        b = AUTO_CAMERA_PNP_BOUNDS
        x, y, z = [float(v) for v in camera_pos_corr_local.reshape(3)]
        pitch = float(angles_corr_local.get("pitch_deg", np.nan))
        yaw = float(angles_corr_local.get("yaw_deg", np.nan))
        roll = float(angles_corr_local.get("roll_deg", np.nan))

        # Soft plausibility term in corrected output space.
        violation = 0.0
        violation += _range_violation(pitch, float(b["pitch_min_deg"]), float(b["pitch_max_deg"]))
        violation += _abs_violation(yaw, float(b["yaw_abs_max_deg"]))
        violation += _abs_violation(roll, float(b["roll_abs_max_deg"]))
        violation += _range_violation(x, float(b["x_min_m"]), float(b["x_max_m"]))
        violation += _abs_violation(y, float(b["y_abs_max_m"]))
        violation += _range_violation(z, float(b["z_min_m"]), float(b["z_max_m"]))
        violation += max(inlier_err - AUTO_CAMERA_PNP_MAX_REPROJ_ERR_PX, 0.0) / max(1e-6, AUTO_CAMERA_PNP_MAX_REPROJ_ERR_PX)

        inlier_ratio = float(len(inlier_idx_local)) / float(len(obj))
        low_inlier_penalty = max(0.0, 0.70 - inlier_ratio) * 25.0
        # Lower score is better.
        score = median_all + 25.0 * violation + low_inlier_penalty

        return {
            "method": method,
            "rvec": rvec_local,
            "tvec": tvec_local,
            "R_wc_raw": R_wc_raw_local,
            "camera_pos_raw": camera_pos_raw_local,
            "inlier_idx": inlier_idx_local,
            "num_inliers": int(len(inlier_idx_local)),
            "num_pairs": int(len(obj)),
            "mean_error_all": mean_all,
            "median_error_all": median_all,
            "max_error_all": max_all,
            "inlier_error": inlier_err,
            "score": float(score),
        }

    candidates: List[Dict[str, Any]] = []

    # Candidate A: RANSAC + LM on inliers.
    if len(obj) >= 6:
        try:
            ok_r, rvec_r, tvec_r, inliers_r = cv2.solvePnPRansac(
                obj,
                img,
                K,
                dist,
                iterationsCount=400,
                reprojectionError=3.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except Exception:
            ok_r = False
            inliers_r = None
        if ok_r:
            inlier_idx_r = (
                inliers_r.reshape(-1).astype(np.int32)
                if inliers_r is not None and len(inliers_r) > 0
                else np.arange(len(obj), dtype=np.int32)
            )
            if len(inlier_idx_r) >= 4:
                try:
                    rvec_r, tvec_r = cv2.solvePnPRefineLM(obj[inlier_idx_r], img[inlier_idx_r], K, dist, rvec_r, tvec_r)
                except Exception:
                    pass
            candidates.append(_build_candidate("ransac_epnp", rvec_r, tvec_r, inlier_idx_r))

    # Candidate B: Iterative all-point fit (+LM refine).
    try:
        ok_i, rvec_i, tvec_i = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        ok_i = False
    if ok_i:
        try:
            rvec_i, tvec_i = cv2.solvePnPRefineLM(obj, img, K, dist, rvec_i, tvec_i)
        except Exception:
            pass
        candidates.append(
            _build_candidate("iterative_all", rvec_i, tvec_i, np.arange(len(obj), dtype=np.int32))
        )

    if not candidates:
        return {"ok": False, "error": "solvePnP failed (no valid solver candidate)"}

    best = min(candidates, key=lambda c: c["score"])
    return {
        "ok": True,
        "R_wc": best["R_wc_raw"],
        "camera_pos": best["camera_pos_raw"],
        "num_pairs": int(best["num_pairs"]),
        "num_inliers": int(best["num_inliers"]),
        "mean_error_px": float(best["mean_error_all"]),
        "median_error_px": float(best["median_error_all"]),
        "max_error_px": float(best["max_error_all"]),
        "inlier_error_px": float(best["inlier_error"]),
        "solve_method": best["method"],
        "solve_score": float(best["score"]),
        "candidate_diagnostics": [
            {
                "method": str(c["method"]),
                "score": float(c["score"]),
                "num_inliers": int(c["num_inliers"]),
                "num_pairs": int(c["num_pairs"]),
                "mean_reprojection_error_px": float(c["mean_error_all"]),
                "median_reprojection_error_px": float(c["median_error_all"]),
                "max_reprojection_error_px": float(c["max_error_all"]),
                "inlier_reprojection_error_px": float(c["inlier_error"]),
            }
            for c in candidates
        ],
    }


def _validate_auto_camera_pose(
    camera_pos: np.ndarray,
    angles_deg: Dict[str, float],
    reproj_error_px: float,
) -> Tuple[bool, List[str]]:
    """Validate camera pose against expected bounds."""
    b = AUTO_CAMERA_PNP_BOUNDS
    x, y, z = [float(v) for v in np.asarray(camera_pos, dtype=np.float64).reshape(3)]
    pitch = float(angles_deg.get("pitch_deg", np.nan))
    yaw = _wrap_deg(float(angles_deg.get("yaw_deg", np.nan)))
    roll = _wrap_deg(float(angles_deg.get("roll_deg", np.nan)))
    err = float(reproj_error_px) if np.isfinite(reproj_error_px) else np.nan

    issues: List[str] = []
    if not (b["pitch_min_deg"] <= pitch <= b["pitch_max_deg"]):
        issues.append(f"pitch={pitch:.2f}deg not in [{b['pitch_min_deg']}, {b['pitch_max_deg']}]")
    if not (abs(yaw) <= b["yaw_abs_max_deg"]):
        issues.append(f"yaw={yaw:.2f}deg exceeds +/-{b['yaw_abs_max_deg']}")
    if not (abs(roll) <= b["roll_abs_max_deg"]):
        issues.append(f"roll={roll:.2f}deg exceeds +/-{b['roll_abs_max_deg']}")
    if not (b["x_min_m"] <= x <= b["x_max_m"]):
        issues.append(f"x={x:.3f}m not in [{b['x_min_m']}, {b['x_max_m']}]")
    if not (abs(y) <= b["y_abs_max_m"]):
        issues.append(f"y={y:.3f}m exceeds +/-{b['y_abs_max_m']}")
    if not (b["z_min_m"] <= z <= b["z_max_m"]):
        issues.append(f"z={z:.3f}m not in [{b['z_min_m']}, {b['z_max_m']}]")
    if not np.isfinite(err) or err > AUTO_CAMERA_PNP_MAX_REPROJ_ERR_PX:
        issues.append(f"reproj_err={err:.2f}px exceeds {AUTO_CAMERA_PNP_MAX_REPROJ_ERR_PX}px")
    return len(issues) == 0, issues


def _attempt_auto_camera_pnp(
    frame_bgr: np.ndarray,
    K_undist: np.ndarray,
    model_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run YOLO keypoint detection + PnP + correction/validation for one frame."""
    model = model_override if model_override is not None else _get_auto_camera_pnp_model()
    min_conf = float(AUTO_CAMERA_PNP_MIN_KPT_CONF)

    results = model.predict(
        source=frame_bgr,
        verbose=False,
        conf=0.05,
        iou=0.6,
        max_det=4,
    )
    if not results:
        return {"success": False, "reason": "YOLO returned no detections"}

    result = results[0]
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return {"success": False, "reason": "No keypoints detected"}

    xy_all = result.keypoints.xy.detach().cpu().numpy()
    if xy_all.ndim != 3 or xy_all.shape[2] != 2:
        return {"success": False, "reason": "Unexpected YOLO keypoint tensor shape"}

    conf_tensor = result.keypoints.conf
    if conf_tensor is not None:
        conf_all = conf_tensor.detach().cpu().numpy()
    else:
        conf_all = np.ones((xy_all.shape[0], xy_all.shape[1]), dtype=np.float64)

    n_labels = len(AUTO_CAMERA_PNP_KEYPOINT_LABELS)
    if xy_all.shape[1] < n_labels:
        return {"success": False, "reason": f"Model returned {xy_all.shape[1]} keypoints, need {n_labels}"}

    box_conf = np.ones((xy_all.shape[0],), dtype=np.float64)
    if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) == xy_all.shape[0]:
        box_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
    mean_kpt_conf = np.mean(np.nan_to_num(conf_all[:, :n_labels], nan=0.0), axis=1)
    det_scores = box_conf * mean_kpt_conf
    det_idx = int(np.argmax(det_scores))

    raw_pts = xy_all[det_idx, :n_labels, :].astype(np.float64)
    raw_conf = conf_all[det_idx, :n_labels].astype(np.float64)

    try:
        mapping_primary = _bind_pnp_keypoints_geometry(raw_pts)
    except Exception as exc:
        return {"success": False, "reason": f"Keypoint binding failed: {exc}"}

    mappings = [
        ("geometry", mapping_primary),
        ("geometry_swapped_port_starboard", _swap_port_starboard_mapping(mapping_primary)),
    ]

    attempted = []
    valid_candidates = []
    for map_name, label_to_index in mappings:
        obj, img, valid_pts = _build_pnp_correspondences(raw_pts, raw_conf, label_to_index, min_conf)
        if valid_pts < AUTO_CAMERA_PNP_MIN_PAIRS:
            attempted.append({
                "mapping": map_name,
                "valid_points": int(valid_pts),
                "reason": f"Only {valid_pts} points >= confidence threshold",
            })
            continue

        solved = _solve_pnp_pose(obj, img, K_undist)
        if not solved.get("ok"):
            attempted.append({
                "mapping": map_name,
                "valid_points": int(valid_pts),
                "reason": solved.get("error", "solvePnP failed"),
            })
            continue

        # User-specific correction:
        # 1) roll uses +90 deg offset: (raw_roll + offset)
        # 2) y/z position outputs are swapped
        raw_angles = _camera_pose_angles_from_rwc(solved["R_wc"])
        _, R_wc_corr = default_camera_pose_and_rotation(
            pitch=raw_angles["pitch_deg"],
            yaw=raw_angles["yaw_deg"],
            roll=(raw_angles["roll_deg"] + AUTO_CAMERA_PNP_ROLL_OFFSET_DEG),
        )
        camera_pos_raw = np.asarray(solved["camera_pos"], dtype=np.float64).reshape(3)
        camera_pos_corr = np.array(
            [camera_pos_raw[0], camera_pos_raw[2], camera_pos_raw[1]],
            dtype=np.float64,
        )
        angles_corr = _camera_pose_angles_from_rwc(R_wc_corr)
        angles_corr["yaw_deg"] = _wrap_deg(angles_corr["yaw_deg"])
        angles_corr["roll_deg"] = _wrap_deg(angles_corr["roll_deg"])
        if np.isfinite(solved.get("median_error_px", np.nan)):
            err_px = float(solved["median_error_px"])
        elif np.isfinite(solved.get("inlier_error_px", np.nan)):
            err_px = float(solved["inlier_error_px"])
        else:
            err_px = float(solved.get("mean_error_px", np.nan))

        is_valid, issues = _validate_auto_camera_pose(camera_pos_corr, angles_corr, err_px)
        candidate = {
            "mapping": map_name,
            "valid_points": int(valid_pts),
            "camera_pos": camera_pos_corr,
            "R_wc": R_wc_corr,
            "angles": angles_corr,
            "err_px": float(err_px),
            "num_inliers": int(solved["num_inliers"]),
            "num_pairs": int(solved["num_pairs"]),
            "solve_method": str(solved.get("solve_method", "unknown")),
            "is_valid": bool(is_valid),
            "issues": issues,
        }
        attempted.append({
            "mapping": map_name,
            "valid_points": int(valid_pts),
            "reason": (
                f"ok ({solved.get('solve_method', 'unknown')})"
                if is_valid
                else f"{solved.get('solve_method', 'unknown')}: " + "; ".join(issues)
            ),
        })
        if is_valid:
            valid_candidates.append(candidate)

    if valid_candidates:
        best = min(valid_candidates, key=lambda c: c["err_px"])
        return {
            "success": True,
            "camera_pos": best["camera_pos"],
            "R_wc": best["R_wc"],
            "angles": best["angles"],
            "err_px": best["err_px"],
            "num_inliers": best["num_inliers"],
            "num_pairs": best["num_pairs"],
            "mapping": best["mapping"],
            "solve_method": best.get("solve_method", "unknown"),
            "det_idx": det_idx,
            "det_score": float(det_scores[det_idx]),
        }

    if attempted:
        first_reason = attempted[0].get("reason", "No valid solution")
        if len(attempted) > 1:
            first_reason = " | ".join(f"{a['mapping']}: {a['reason']}" for a in attempted)
        return {"success": False, "reason": first_reason}
    return {"success": False, "reason": "No PnP candidates attempted"}


def raycast_anchors(row, K_undist, W, H, camera_pos, R_wc,
                    z_plane_lm24, z_plane_lm28, lower_landmark="ankle"):
    """Compute raw raycast intersection points for skeleton placement.

    Returns a dict with anchor_int, dir_int, roll_int (numpy arrays or None),
    anchor_idx, dir_idx, roll_idx (ints), and use_left_side (bool).
    Returns None if raycasting fails entirely.
    """
    def _ray_intersection(norm_xyz, z_plane):
        if norm_xyz is None:
            return None
        ray = ray_from_norm_landmark_undistorted(norm_xyz[0], norm_xyz[1], W, H, K_undist)
        return intersect_world_z_plane(ray, R_wc=R_wc, t_wc=camera_pos, Z0=z_plane)

    def _world_point(idx):
        x = row.get(f"lm{idx}_world_x")
        y = row.get(f"lm{idx}_world_y")
        z = row.get(f"lm{idx}_world_z")
        if x is None or y is None or z is None or pd.isna(x) or pd.isna(y) or pd.isna(z):
            return None
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    lm23_norm = get_landmark_norm(row, 23)  # Left hip
    lm24_norm = get_landmark_norm(row, 24)  # Right hip
    if lm23_norm is None and lm24_norm is None:
        return None

    if lower_landmark == "knee":
        left_dir_idx, right_dir_idx = 25, 26
    else:
        left_dir_idx, right_dir_idx = 27, 28

    lm_left_norm = get_landmark_norm(row, left_dir_idx)
    lm_right_norm = get_landmark_norm(row, right_dir_idx)
    if lm_left_norm is None and lm_right_norm is None:
        return None

    hip23_int = _ray_intersection(lm23_norm, z_plane_lm24)
    hip24_int = _ray_intersection(lm24_norm, z_plane_lm24)
    if hip23_int is None and hip24_int is None:
        return None

    left_dir_int = _ray_intersection(lm_left_norm, z_plane_lm28)
    right_dir_int = _ray_intersection(lm_right_norm, z_plane_lm28)
    if left_dir_int is None and right_dir_int is None:
        return None

    hip_ints = [p for p in (hip23_int, hip24_int) if p is not None]
    dir_ints = [p for p in (left_dir_int, right_dir_int) if p is not None]
    anchor_int = np.mean(np.asarray(hip_ints, dtype=np.float64), axis=0)
    dir_int = np.mean(np.asarray(dir_ints, dtype=np.float64), axis=0)

    # Symmetric target roll reference: right hip when available, else left hip.
    roll_int = hip24_int if hip24_int is not None else hip23_int

    # Symmetric source references in MediaPipe world (camera) coordinates.
    src_hip23 = _world_point(23)
    src_hip24 = _world_point(24)
    src_anchor_pts = [p for p in (src_hip23, src_hip24) if p is not None]
    if not src_anchor_pts:
        return None
    src_anchor = np.mean(np.asarray(src_anchor_pts, dtype=np.float64), axis=0)

    src_left_dir = _world_point(left_dir_idx)
    src_right_dir = _world_point(right_dir_idx)
    src_dir_pts = [p for p in (src_left_dir, src_right_dir) if p is not None]
    if not src_dir_pts:
        return None
    src_dir = np.mean(np.asarray(src_dir_pts, dtype=np.float64), axis=0)

    src_roll = src_hip24 if src_hip24 is not None else src_hip23

    # Keep legacy IDs for compatibility with old call sites/diagnostics.
    anchor_idx = 24 if lm24_norm is not None else 23
    dir_idx = right_dir_idx if lm_right_norm is not None else left_dir_idx
    roll_idx = 24 if lm24_norm is not None else 23

    # Side label kept for diagnostics only.
    use_left_side = bool(anchor_int[1] < 0.0)

    return {
        "anchor_int": anchor_int,
        "dir_int": dir_int,
        "roll_int": roll_int,
        "anchor_idx": anchor_idx,
        "dir_idx": dir_idx,
        "roll_idx": roll_idx,
        "use_left_side": use_left_side,
        "src_anchor": src_anchor,
        "src_dir": src_dir,
        "src_roll": src_roll,
    }


def _compute_placed_skeleton_raycast(
    row,
    K_undist,
    W,
    H,
    camera_pos,
    R_wc,
    z_plane_lm24,
    z_plane_lm28,
    lower_landmark="ankle",
):
    """Legacy raycast-based skeleton placement."""
    sitting_score = _estimate_sitting_score_from_row(row)
    z_low_eff = _effective_lower_plane_z(float(z_plane_lm28), float(sitting_score))

    anchors = raycast_anchors(
        row,
        K_undist,
        W,
        H,
        camera_pos,
        R_wc,
        z_plane_lm24,
        z_low_eff,
        lower_landmark,
    )
    if anchors is None:
        return None

    anchor_int = anchors["anchor_int"]
    dir_int = anchors["dir_int"]
    roll_int = anchors["roll_int"]

    placed = place_skeleton_on_boat(
        row,
        anchor_idx=anchors["anchor_idx"],
        dir_idx=anchors["dir_idx"],
        anchor_boat=anchor_int,
        dir_boat=dir_int,
        use_scale=False,
        roll_idx=anchors["roll_idx"],
        roll_boat=roll_int,
        src_anchor=anchors.get("src_anchor"),
        src_dir=anchors.get("src_dir"),
        src_roll=anchors.get("src_roll"),
    )
    if placed is None:
        return None
    placed = _guard_hip_height_from_foot_lock(
        placed=placed,
        z_plane_lm24=float(z_plane_lm24),
        z_plane_lm28=float(z_low_eff),
        sitting_score=float(sitting_score),
    )
    placed = _apply_seated_pelvis_height_constraint(
        placed=placed,
        z_plane_lm24=float(z_plane_lm24),
        sitting_score=float(sitting_score),
    )
    return placed


def compute_placed_skeleton(
    row,
    K_undist,
    W,
    H,
    camera_pos,
    R_wc,
    z_plane_lm24,
    z_plane_lm28,
    lower_landmark="ankle",
    contact_params: Optional[dict] = None,
):
    """Compute skeleton placement in boat/world frame.

    Primary mode uses a symmetric raycast translation fit that preserves
    MediaPipe body orientation. If that fails, falls back to legacy raycast.
    """
    placed_sym = _compute_placed_skeleton_symmetric_raycast(
        row=row,
        K_undist=K_undist,
        W=W,
        H=H,
        camera_pos=np.asarray(camera_pos, dtype=np.float64).reshape(3),
        R_wc=np.asarray(R_wc, dtype=np.float64).reshape(3, 3),
        z_plane_lm24=float(z_plane_lm24),
        z_plane_lm28=float(z_plane_lm28),
        lower_landmark=lower_landmark,
    )
    if placed_sym is not None:
        return placed_sym

    placed_legacy = _compute_placed_skeleton_raycast(
        row=row,
        K_undist=K_undist,
        W=W,
        H=H,
        camera_pos=camera_pos,
        R_wc=R_wc,
        z_plane_lm24=z_plane_lm24,
        z_plane_lm28=z_plane_lm28,
        lower_landmark=lower_landmark,
    )
    return placed_legacy


def is_plausible_com(com):
    """Check if COM position is plausible."""
    if com is None:
        return False
    x, y, z = com[0], com[1], com[2]
    if x < FILTER_THRESHOLDS['com_x']['min'] or x > FILTER_THRESHOLDS['com_x']['max']:
        return False
    if y < FILTER_THRESHOLDS['com_y']['min'] or y > FILTER_THRESHOLDS['com_y']['max']:
        return False
    if z < FILTER_THRESHOLDS['com_z']['min'] or z > FILTER_THRESHOLDS['com_z']['max']:
        return False
    return True


def compute_trunk_angle_midpoints(placed):
    """Compute trunk angle from placed skeleton using midpoints."""
    if placed is None:
        return None
    
    if not all(k in placed and placed[k] is not None for k in (11, 12, 23, 24)):
        return None
    
    shoulder_mid = 0.5 * (np.array(placed[11]) + np.array(placed[12]))
    hip_mid = 0.5 * (np.array(placed[23]) + np.array(placed[24]))
    
    v = shoulder_mid - hip_mid
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    
    v_unit = v / n
    z_unit = np.array([0.0, 0.0, 1.0])
    
    cos_theta = float(np.clip(np.dot(v_unit, z_unit), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    return angle_deg


def compute_metrics_for_skeleton(placed, boat_com, athlete_mass):
    """Compute all metrics for a placed skeleton.
    
    Returns dict with trunk_angle, com_x, com_y, com_z, moment_pitch, moment_roll.
    """
    result = {
        'trunk_angle': None,
        'com_x': None,
        'com_y': None,
        'com_z': None,
        'moment_pitch': None,
        'moment_roll': None,
    }
    
    if placed is None:
        return result
    
    # Compute trunk angle
    result['trunk_angle'] = compute_trunk_angle_midpoints(placed)
    
    # Compute center of mass
    try:
        com = compute_center_of_mass(placed)
    except Exception:
        com = None
    if com is not None and is_plausible_com(com):
        result['com_x'] = float(com[0])
        result['com_y'] = float(com[1])
        result['com_z'] = float(com[2])
        
        # Compute moments
        dx = float(com[0] - boat_com)
        dy = float(com[1])
        result['moment_pitch'] = athlete_mass * GRAVITY * dx
        result['moment_roll'] = athlete_mass * GRAVITY * dy
    
    return result


def load_undistorted_intrinsics(npz_path: str, img_width: int, img_height: int):
    """Load fisheye calibration and return undistorted intrinsics K_new."""
    cal = np.load(npz_path)
    K = cal["K"]
    D = cal["D"]
    img_size = (img_width, img_height)
    
    R = np.eye(3)
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_size, R, balance=0.0, new_size=img_size
    )
    
    return K_new

# Processing state tracking
processing_jobs: Dict[str, Dict[str, Any]] = {}


def ensure_projects_dir():
    """Create projects directory if it doesn't exist."""
    PROJECTS_DIR.mkdir(exist_ok=True)


def get_project_path(project_id: str) -> Path:
    """Get the path to a project directory."""
    return PROJECTS_DIR / project_id


def list_projects() -> List[Dict[str, Any]]:
    """List all existing projects."""
    ensure_projects_dir()
    projects = []
    
    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        
        config_path = project_dir / "config.json"
        if not config_path.exists():
            continue
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Use frame_count from config (saved after processing),
            # fall back to fast line count instead of full CSV parse
            frame_count = config.get("frame_count", 0)
            if frame_count == 0:
                pose_csv = project_dir / "pose.csv"
                if pose_csv.exists():
                    with open(pose_csv, "r") as csv_f:
                        frame_count = max(0, sum(1 for _ in csv_f) - 1)  # subtract header
            
            projects.append({
                "id": project_dir.name,
                "name": config.get("name", project_dir.name),
                "created": config.get("created", ""),
                "frame_count": frame_count,
                "state": config.get("state", "unknown"),
            })
        except Exception:
            continue
    
    # Sort by creation date, newest first
    projects.sort(key=lambda p: p.get("created", ""), reverse=True)
    return projects


def get_project_config(project_id: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a project."""
    config_path = get_project_path(project_id) / "config.json"
    if not config_path.exists():
        return None
    
    with open(config_path) as f:
        return json.load(f)


def delete_project(project_id: str) -> bool:
    """Delete a project and all its files."""
    project_path = get_project_path(project_id)
    if not project_path.exists():
        return False
    
    shutil.rmtree(project_path)
    return True


def create_project(
    name: str,
    video_path: str,
    calibration_path: Optional[str] = None,
    athlete_weight: float = 75.0,
    ankle_height: float = 0.01,
    hip_height: float = 0.10,
    camera_x: float = -3.374,
    camera_y: float = 0.0,
    camera_z: float = 0.5,
    camera_pitch_deg: float = 8.0,
    camera_yaw_deg: float = 0.0,
    camera_roll_deg: float = 0.0,
    camera_R_wc: Optional[List[List[float]]] = None,
    boat_com: float = -1.114,
    pose_model: str = "full",
    lower_landmark: str = "ankle",
    # Rudder detection parameters
    rudder_enabled: bool = False,
    # Skeleton placement smoothing
    skeleton_filter: Optional[dict] = None,
    # Seated fore-aft x stabilizer
    seated_x_stabilizer: Optional[dict] = None,
    # Hiking/lateral y stabilizer
    lateral_y_stabilizer: Optional[dict] = None,
    # Contact-fitting placement tuning
    contact_params: Optional[dict] = None,
    # Processing instances (<=1 disables segment parallel mode)
    mediapipe_workers: Optional[int] = None,
) -> str:
    """Create a new project and start processing."""
    ensure_projects_dir()
    
    project_id = str(uuid.uuid4())[:8] + "_" + name.replace(" ", "_")[:20]
    project_path = get_project_path(project_id)
    project_path.mkdir(exist_ok=True)
    
    # Copy video to project directory
    video_dest = project_path / "video.mp4"
    shutil.copy2(video_path, video_dest)
    
    # Copy calibration if provided
    calib_dest = None
    if calibration_path and os.path.exists(calibration_path):
        calib_dest = project_path / "calibration.npz"
        shutil.copy2(calibration_path, calib_dest)
    
    # Prepare rudder config
    rudder_config = {
        "enabled": rudder_enabled,
    }
    
    # Save configuration
    mediapipe_workers_cfg = None
    if mediapipe_workers is not None:
        try:
            mediapipe_workers_cfg = int(mediapipe_workers)
        except Exception:
            mediapipe_workers_cfg = None

    config = {
        "name": name,
        "created": datetime.now().isoformat(),
        "state": "pending",
        "video_file": "video.mp4",
        "calibration_file": "calibration.npz" if calib_dest else None,
        "athlete_weight": athlete_weight,
        "ankle_height": ankle_height,
        "hip_height": hip_height,
        "camera_position": [camera_x, camera_y, camera_z],
        "camera_pitch_deg": camera_pitch_deg,
        "camera_yaw_deg": camera_yaw_deg,
        "camera_roll_deg": camera_roll_deg,
        "camera_R_wc": camera_R_wc,
        "boat_com": boat_com,
        "pose_model": pose_model,
        "lower_landmark": lower_landmark,
        "skeleton_filter": normalize_skeleton_filter_params(skeleton_filter),
        "seated_x_stabilizer": normalize_seated_x_stabilizer_params(seated_x_stabilizer),
        "lateral_y_stabilizer": normalize_lateral_y_stabilizer_params(lateral_y_stabilizer),
        "contact_params": normalize_contact_params(contact_params),
        "rudder": rudder_config,
        "mediapipe_workers": mediapipe_workers_cfg,
    }
    
    with open(project_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize processing state
    processing_jobs[project_id] = {
        "state": "pending",
        "progress": 0,
        "status": "Queued for processing",
        "log_lines": [],
        "error": None,
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=process_video, args=(project_id,))
    thread.daemon = True
    thread.start()
    
    return project_id


def get_processing_status(project_id: str) -> Dict[str, Any]:
    """Get the current processing status for a project."""
    if project_id in processing_jobs:
        job = processing_jobs[project_id]
        # Get new log lines and clear them
        new_lines = job.get("log_lines", [])
        job["log_lines"] = []
        return {
            "state": job.get("state", "unknown"),
            "progress": job.get("progress", 0),
            "status": job.get("status", "Unknown"),
            "error": job.get("error"),
            "log_lines": new_lines,
        }
    
    # Check config file
    config = get_project_config(project_id)
    if config:
        return {
            "state": config.get("state", "unknown"),
            "progress": 100 if config.get("state") == "completed" else 0,
            "status": "Completed" if config.get("state") == "completed" else "Unknown",
            "error": None,
            "log_lines": [],
        }
    
    return {
        "state": "not_found",
        "progress": 0,
        "status": "Project not found",
        "error": "Project does not exist",
        "log_lines": [],
    }


def add_log(project_id: str, message: str, log_type: str = ""):
    """Add a log line to the processing job."""
    if project_id in processing_jobs:
        processing_jobs[project_id]["log_lines"].append({
            "message": message,
            "type": log_type,
        })


def update_progress(project_id: str, progress: int, status: str):
    """Update the progress of a processing job."""
    if project_id in processing_jobs:
        processing_jobs[project_id]["progress"] = progress
        processing_jobs[project_id]["status"] = status


def _resolve_mediapipe_worker_count(config: Dict[str, Any], total_frames: int) -> int:
    """Resolve segment instance count for parallel processing."""
    cpu_count = max(1, int(os.cpu_count() or 1))
    max_workers = max(1, min(8, cpu_count))

    raw = config.get("mediapipe_workers", None)
    if raw is None:
        # Conservative auto-default: only enable parallel mode on longer clips.
        if total_frames >= 240 and cpu_count >= 4:
            return min(2, max_workers)
        return 1

    try:
        workers = int(raw)
    except Exception:
        return 1

    if workers <= 1:
        return 1
    return int(np.clip(workers, 1, max_workers))


def _split_frame_ranges(total_frames: int, segments: int) -> List[Tuple[int, int]]:
    """Split [0,total_frames) into contiguous ranges."""
    n = int(max(0, total_frames))
    if n <= 0:
        return []
    seg = int(max(1, segments))
    seg = min(seg, n)
    base = n // seg
    rem = n % seg
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(seg):
        extra = 1 if i < rem else 0
        end = start + base + extra
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


def _pose_result_to_packet(result: Any) -> Optional[Dict[str, np.ndarray]]:
    """Convert MediaPipe pose result to fixed-shape numpy arrays for thread-safe handoff."""
    if result is None or not result.pose_world_landmarks or not result.pose_landmarks:
        return None
    world_lm = result.pose_world_landmarks[0]
    norm_lm = result.pose_landmarks[0]
    if len(world_lm) < 33 or len(norm_lm) < 33:
        return None

    world = np.empty((33, 3), dtype=np.float32)
    norm = np.empty((33, 4), dtype=np.float32)  # x, y, z, visibility
    for i in range(33):
        w = world_lm[i]
        n = norm_lm[i]
        world[i, 0] = float(w.x)
        world[i, 1] = float(w.y)
        world[i, 2] = float(w.z)
        norm[i, 0] = float(n.x)
        norm[i, 1] = float(n.y)
        norm[i, 2] = float(n.z)
        norm[i, 3] = float(getattr(n, "visibility", np.nan))
    return {"world": world, "norm": norm}


def _append_pose_packet_to_csv_row(row: List[Any], pose_packet: Optional[Dict[str, np.ndarray]]) -> None:
    """Append MediaPipe world/norm landmark values to a CSV row buffer."""
    if pose_packet is None:
        row.extend([np.nan] * (33 * 7))
        return

    world = np.asarray(pose_packet["world"], dtype=np.float32)
    norm = np.asarray(pose_packet["norm"], dtype=np.float32)
    for i in range(33):
        row.extend([
            float(world[i, 0]),
            float(world[i, 1]),
            float(world[i, 2]),
            float(norm[i, 0]),
            float(norm[i, 1]),
            float(norm[i, 2]),
            float(norm[i, 3]),
        ])


def _pose_packet_to_row_dict(
    frame_idx: int,
    timestamp_ms: int,
    pose_packet: Optional[Dict[str, np.ndarray]],
) -> Optional[Dict[str, Any]]:
    """Build compute_placed_skeleton input dict from pose packet."""
    if pose_packet is None:
        return None
    world = np.asarray(pose_packet["world"], dtype=np.float32)
    norm = np.asarray(pose_packet["norm"], dtype=np.float32)
    if world.shape != (33, 3) or norm.shape != (33, 4):
        return None

    row_dict: Dict[str, Any] = {"frame_idx": int(frame_idx), "timestamp_ms": int(timestamp_ms)}
    for i in range(33):
        row_dict[f"lm{i}_world_x"] = float(world[i, 0])
        row_dict[f"lm{i}_world_y"] = float(world[i, 1])
        row_dict[f"lm{i}_world_z"] = float(world[i, 2])
        row_dict[f"lm{i}_norm_x"] = float(norm[i, 0])
        row_dict[f"lm{i}_norm_y"] = float(norm[i, 1])
        row_dict[f"lm{i}_norm_z"] = float(norm[i, 2])
        row_dict[f"lm{i}_visibility"] = float(norm[i, 3])
    return row_dict


def _extract_landmark_confidence_from_row(
    row_like: Optional[Dict[str, Any]],
    landmark_count: int = 33,
) -> Optional[Dict[int, float]]:
    """Extract per-landmark [0,1] confidence from a row-like object."""
    if row_like is None:
        return None
    out: Dict[int, float] = {}
    for i in range(int(max(1, landmark_count))):
        key = f"lm{i}_visibility"
        try:
            raw = row_like.get(key)  # dict or pandas Series
        except Exception:
            continue
        try:
            v = float(raw)
        except Exception:
            continue
        if np.isfinite(v):
            out[i] = float(np.clip(v, 0.0, 1.0))
    return out or None


def _build_threaded_pose_detector(model_path: Path):
    """Create a thread-local IMAGE-mode detector for parallel frame inference."""
    detector_lock = threading.Lock()
    detectors_by_tid: Dict[int, Any] = {}

    def _get_detector():
        tid = int(threading.get_ident())
        with detector_lock:
            existing = detectors_by_tid.get(tid)
        if existing is not None:
            return existing

        local_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_segmentation_masks=False,
        )
        created = vision.PoseLandmarker.create_from_options(local_options)

        with detector_lock:
            existing = detectors_by_tid.get(tid)
            if existing is not None:
                try:
                    created.close()
                except Exception:
                    pass
                return existing
            detectors_by_tid[tid] = created
            return created

    def detect_frame_rgb(frame_rgb: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        detector = _get_detector()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return _pose_result_to_packet(detector.detect(mp_image))

    def close_all_detectors() -> None:
        with detector_lock:
            all_detectors = list(detectors_by_tid.values())
            detectors_by_tid.clear()
        for detector in all_detectors:
            try:
                detector.close()
            except Exception:
                pass

    return detect_frame_rgb, close_all_detectors


def process_video(project_id: str):
    """Process video and extract pose data."""
    project_path = get_project_path(project_id)
    config_path = project_path / "config.json"
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Update state
        config["state"] = "processing"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        if project_id in processing_jobs:
            processing_jobs[project_id]["state"] = "processing"
        
        video_path = project_path / config["video_file"]
        output_csv = project_path / "pose.csv"
        
        add_log(project_id, "Loading video...")
        update_progress(project_id, 5, "Loading video")
        
        # Check if MediaPipe is available
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Install with: pip install mediapipe")
        
        # Get model path based on config
        pose_model = config.get("pose_model", "full")
        if pose_model not in MODEL_PATHS:
            pose_model = "full"
        model_path = MODEL_PATHS[pose_model]
        
        # Check if model exists
        if not model_path.exists():
            raise RuntimeError(f"MediaPipe model not found at {model_path}. Download pose_landmarker_{pose_model}.task from MediaPipe.")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        add_log(project_id, f"Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        
        # Load calibration if available
        undistort_maps = None
        K_undist = None
        calib_file = config.get("calibration_file")
        if calib_file:
            calib_path = project_path / calib_file
            if calib_path.exists():
                add_log(project_id, "Loading camera calibration...")
                try:
                    undistort_maps = load_calibration(str(calib_path), width, height)
                    K_undist = load_undistorted_intrinsics(str(calib_path), width, height)
                    add_log(project_id, "Calibration loaded successfully", "success")
                except Exception as e:
                    add_log(project_id, f"Warning: Failed to load calibration: {e}", "error")
        
        # Set up camera parameters for skeleton placement
        camera_pos, R_wc = get_camera_pose_from_config(config)
        z_hip = config.get("hip_height", 0.10)
        z_ankle = config.get("ankle_height", 0.01)
        boat_com = config.get("boat_com", -1.114)
        athlete_mass = config.get("athlete_weight", 75.0)
        lower_landmark = config.get("lower_landmark", "ankle")

        add_log(project_id, "Skeleton placement: orientation-preserving symmetric raycast fitter enabled", "success")
        
        # Initialize MediaPipe
        add_log(project_id, f"Initializing MediaPipe Pose Landmarker ({pose_model} model)...")
        update_progress(project_id, 10, "Initializing MediaPipe")

        mediapipe_workers = _resolve_mediapipe_worker_count(config, total_frames)
        segment_instances = int(max(1, mediapipe_workers))
        segment_parallel_enabled = bool(segment_instances > 1 and total_frames > 0)

        landmarker = None
        pose_executor = None
        threaded_pose_detect = None
        threaded_pose_close = None
        parallel_pose_enabled = False

        if segment_parallel_enabled:
            add_log(
                project_id,
                f"Segment parallel mode enabled: instances={segment_instances} "
                f"(video split into {segment_instances} contiguous frame ranges).",
                "success",
            )
        else:
            parallel_pose_enabled = bool(mediapipe_workers > 1)
            if parallel_pose_enabled:
                try:
                    threaded_pose_detect, threaded_pose_close = _build_threaded_pose_detector(model_path)
                    pose_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=mediapipe_workers,
                        thread_name_prefix="mp_pose",
                    )
                    add_log(
                        project_id,
                        f"MediaPipe parallel inference enabled: workers={mediapipe_workers}, "
                        "running_mode=IMAGE (ordered output preserved).",
                        "success",
                    )
                except Exception as e:
                    parallel_pose_enabled = False
                    pose_executor = None
                    threaded_pose_detect = None
                    threaded_pose_close = None
                    add_log(
                        project_id,
                        f"Parallel MediaPipe init failed, falling back to single-thread VIDEO mode: {e}",
                        "error",
                    )

            if not parallel_pose_enabled:
                base_options = python.BaseOptions(model_asset_path=str(model_path))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    output_segmentation_masks=False,
                )
                landmarker = vision.PoseLandmarker.create_from_options(options)
                add_log(project_id, "MediaPipe single-thread VIDEO mode enabled", "info")

        add_log(project_id, "MediaPipe initialized", "success")
        
        contact_params = normalize_contact_params(config.get("contact_params"))

        # Skeleton-placement Kalman smoothing (post-placement, full skeleton).
        skeleton_filter_cfg = normalize_skeleton_filter_params(config.get("skeleton_filter"))
        if "skeleton_filter" not in config and "kalman_enabled" in config:
            # Backward compatibility with older projects.
            skeleton_filter_cfg["enabled"] = bool(config.get("kalman_enabled", True))

        skeleton_smoother = None
        if skeleton_filter_cfg.get("enabled", True):
            skeleton_smoother = SkeletonPlacementKalman(
                fps=fps,
                process_noise_acc=float(skeleton_filter_cfg["process_noise_acc"]),
                measurement_noise=float(skeleton_filter_cfg["measurement_noise"]),
                use_landmark_confidence=bool(skeleton_filter_cfg["use_landmark_confidence"]),
                min_landmark_confidence=float(skeleton_filter_cfg["min_landmark_confidence"]),
                confidence_floor=float(skeleton_filter_cfg["confidence_floor"]),
                confidence_power=float(skeleton_filter_cfg["confidence_power"]),
                max_confidence_noise_scale=float(skeleton_filter_cfg["max_confidence_noise_scale"]),
                gate_sigma=float(skeleton_filter_cfg["gate_sigma"]),
                max_consecutive_misses=int(skeleton_filter_cfg["max_consecutive_misses"]),
                initial_velocity_std=float(skeleton_filter_cfg["initial_velocity_std"]),
                velocity_decay=float(skeleton_filter_cfg["velocity_decay"]),
                max_speed=float(skeleton_filter_cfg["max_speed"]),
                max_measurement_jump=float(skeleton_filter_cfg["max_measurement_jump"]),
                reacquire_frames=int(skeleton_filter_cfg["reacquire_frames"]),
                reacquire_max_jump=float(skeleton_filter_cfg["reacquire_max_jump"]),
            )
            add_log(
                project_id,
                "Skeleton Kalman smoothing enabled: "
                f"proc={skeleton_filter_cfg['process_noise_acc']}, "
                f"meas={skeleton_filter_cfg['measurement_noise']}, "
                f"conf={skeleton_filter_cfg['use_landmark_confidence']} "
                f"(min={skeleton_filter_cfg['min_landmark_confidence']}, "
                f"floor={skeleton_filter_cfg['confidence_floor']}, "
                f"pow={skeleton_filter_cfg['confidence_power']}, "
                f"max_scale={skeleton_filter_cfg['max_confidence_noise_scale']}), "
                f"gate={skeleton_filter_cfg['gate_sigma']}, "
                f"max_miss={skeleton_filter_cfg['max_consecutive_misses']}, "
                f"init_vel={skeleton_filter_cfg['initial_velocity_std']}, "
                f"vel_decay={skeleton_filter_cfg['velocity_decay']}, "
                f"max_speed={skeleton_filter_cfg['max_speed']}, "
                f"jump_cap={skeleton_filter_cfg['max_measurement_jump']}, "
                f"reacq={skeleton_filter_cfg['reacquire_frames']}/{skeleton_filter_cfg['reacquire_max_jump']}",
                "success",
            )
        else:
            add_log(project_id, "Skeleton Kalman smoothing disabled", "info")

        seated_x_stabilizer_cfg = normalize_seated_x_stabilizer_params(config.get("seated_x_stabilizer"))
        seated_x_stabilizer = create_seated_x_stabilizer(seated_x_stabilizer_cfg)
        if seated_x_stabilizer is not None:
            add_log(
                project_id,
                "Seated fore-aft x stabilizer enabled "
                f"(sit={seated_x_stabilizer_cfg['sit_start']:.2f}-{seated_x_stabilizer_cfg['sit_full']:.2f}, "
                f"alpha={seated_x_stabilizer_cfg['alpha_min']:.2f}-{seated_x_stabilizer_cfg['alpha_max']:.2f}, "
                f"step_cap={seated_x_stabilizer_cfg['step_cap_seated_m']:.3f}-{seated_x_stabilizer_cfg['step_cap_default_m']:.3f} m/frame, "
                f"hike_relax={seated_x_stabilizer_cfg['hike_relax']:.2f}, max_shift={seated_x_stabilizer_cfg['max_shift_m']:.3f} m)",
                "info",
            )
        else:
            add_log(project_id, "Seated fore-aft x stabilizer disabled", "info")

        lateral_y_stabilizer_cfg = normalize_lateral_y_stabilizer_params(config.get("lateral_y_stabilizer"))
        lateral_y_stabilizer = create_lateral_y_stabilizer(lateral_y_stabilizer_cfg)
        if lateral_y_stabilizer is not None:
            add_log(
                project_id,
                "Hiking lateral y stabilizer enabled "
                f"(hike={lateral_y_stabilizer_cfg['hike_start']:.2f}-{lateral_y_stabilizer_cfg['hike_full']:.2f}, "
                f"alpha={lateral_y_stabilizer_cfg['alpha_min']:.2f}-{lateral_y_stabilizer_cfg['alpha_max']:.2f}, "
                f"step_cap={lateral_y_stabilizer_cfg['step_cap_hiking_m']:.3f}-{lateral_y_stabilizer_cfg['step_cap_default_m']:.3f} m/frame, "
                f"max_shift={lateral_y_stabilizer_cfg['max_shift_m']:.3f} m, "
                f"sit_relax={lateral_y_stabilizer_cfg['sit_relax']:.2f})",
                "info",
            )
        else:
            add_log(project_id, "Hiking lateral y stabilizer disabled", "info")

        # Auto camera PnP recalibration setup
        auto_pnp_cfg = config.get("auto_camera_pnp", {})
        if isinstance(auto_pnp_cfg, bool):
            auto_pnp_enabled = bool(auto_pnp_cfg)
            auto_pnp_interval = AUTO_CAMERA_PNP_INTERVAL_FRAMES
            auto_pnp_avg_frames = AUTO_CAMERA_PNP_AVG_FRAMES
            auto_pnp_min_valid_frames = AUTO_CAMERA_PNP_MIN_VALID_FRAMES
        elif isinstance(auto_pnp_cfg, dict):
            auto_pnp_enabled = bool(auto_pnp_cfg.get("enabled", True))
            try:
                auto_pnp_interval = int(auto_pnp_cfg.get("interval_frames", AUTO_CAMERA_PNP_INTERVAL_FRAMES))
            except Exception:
                auto_pnp_interval = AUTO_CAMERA_PNP_INTERVAL_FRAMES
            try:
                auto_pnp_avg_frames = int(auto_pnp_cfg.get("avg_frames", AUTO_CAMERA_PNP_AVG_FRAMES))
            except Exception:
                auto_pnp_avg_frames = AUTO_CAMERA_PNP_AVG_FRAMES
            try:
                auto_pnp_min_valid_frames = int(auto_pnp_cfg.get("min_valid_frames", auto_pnp_avg_frames))
            except Exception:
                auto_pnp_min_valid_frames = auto_pnp_avg_frames
        else:
            auto_pnp_enabled = True
            auto_pnp_interval = AUTO_CAMERA_PNP_INTERVAL_FRAMES
            auto_pnp_avg_frames = AUTO_CAMERA_PNP_AVG_FRAMES
            auto_pnp_min_valid_frames = AUTO_CAMERA_PNP_MIN_VALID_FRAMES
        auto_pnp_interval = max(1, auto_pnp_interval)
        auto_pnp_avg_frames = max(1, auto_pnp_avg_frames)
        auto_pnp_min_valid_frames = max(1, int(auto_pnp_min_valid_frames))
        auto_pnp_target_successes = max(auto_pnp_avg_frames, auto_pnp_min_valid_frames)

        auto_pnp_active = False
        auto_pnp_retry_active = False
        auto_pnp_retry_count = 0
        next_auto_pnp_frame = 0
        auto_pnp_collect_active = False
        auto_pnp_collect_start_frame = -1
        auto_pnp_collect_successes: List[Dict[str, Any]] = []
        auto_pnp_collect_fail_reasons: List[str] = []
        auto_pnp_collect_fail_count = 0
        if auto_pnp_enabled and K_undist is not None:
            try:
                if segment_parallel_enabled:
                    if not AUTO_CAMERA_PNP_MODEL_PATH.exists():
                        raise RuntimeError(f"Auto camera PnP model missing: {AUTO_CAMERA_PNP_MODEL_PATH}")
                    try:
                        from ultralytics import YOLO as _YoloImportCheck  # noqa: F401
                    except ImportError as exc:
                        raise RuntimeError(
                            "Ultralytics is not installed; auto camera PnP is unavailable."
                        ) from exc
                else:
                    _get_auto_camera_pnp_model()
                auto_pnp_active = True
                b = AUTO_CAMERA_PNP_BOUNDS
                add_log(
                    project_id,
                    f"Auto camera PnP enabled: interval={auto_pnp_interval} frames, "
                    f"target={auto_pnp_target_successes} successful frame solves (non-consecutive), "
                    f"robust median+outlier filtering enabled, retries from next frame until valid pose.",
                    "success",
                )
                add_log(
                    project_id,
                    "Auto camera PnP constraints: "
                    f"pitch {b['pitch_min_deg']}-{b['pitch_max_deg']} deg, "
                    f"|yaw|<={b['yaw_abs_max_deg']} deg, |roll|<={b['roll_abs_max_deg']} deg, "
                    f"x in [{b['x_min_m']},{b['x_max_m']}], |y|<={b['y_abs_max_m']}, "
                    f"z in [{b['z_min_m']},{b['z_max_m']}], reproj<={AUTO_CAMERA_PNP_MAX_REPROJ_ERR_PX} px "
                    f"(kpt_conf>={AUTO_CAMERA_PNP_MIN_KPT_CONF}, with y/z swap and roll correction=(raw+{AUTO_CAMERA_PNP_ROLL_OFFSET_DEG})).",
                    "info",
                )
            except Exception as e:
                add_log(project_id, f"Auto camera PnP disabled: {e}", "error")
        elif auto_pnp_enabled and K_undist is None:
            add_log(project_id, "Auto camera PnP disabled: calibration/intrinsics unavailable.", "info")

        # Rudder detector setup.
        rudder_detector = None
        rudder_config = config.get("rudder", {})
        rudder_enabled = bool(rudder_config.get("enabled", False))
        if rudder_enabled:
            if segment_parallel_enabled:
                add_log(
                    project_id,
                    f"PilotNet rudder detector enabled: one instance per segment worker ({segment_instances} total).",
                    "success",
                )
            else:
                add_log(project_id, "Initializing dual-model PilotNet rudder detector...")
                rudder_detector = PilotNetDetector(fps=fps)
                add_log(project_id, "PilotNet rudder detector initialized (2 models + Kalman filter)", "success")
        
        # Process frames
        add_log(project_id, "Processing frames...")
        
        # Create CSV header
        header = ["frame_idx", "timestamp_ms"]
        for i in range(33):
            header.extend([
                f"lm{i}_world_x", f"lm{i}_world_y", f"lm{i}_world_z",
                f"lm{i}_norm_x", f"lm{i}_norm_y", f"lm{i}_norm_z",
                f"lm{i}_visibility"
            ])
        
        # Add rudder angle column if enabled
        if rudder_enabled:
            header.append("rudder_angle")
        
        # Add computed skeleton positions (placed in boat frame)
        for i in range(33):
            header.extend([f"skel{i}_x", f"skel{i}_y", f"skel{i}_z"])
        
        # Add computed metrics columns
        header.extend([
            "trunk_angle", "com_x", "com_y", "com_z", 
            "moment_pitch", "moment_roll"
        ])
        
        rows: List[List[Any]] = []
        frame_idx = 0
        written_frames = 0
        last_progress_update = 0

        pending_pose_futures: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
        ready_pose_packets: Dict[int, tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]] = {}
        next_emit_frame_idx = 0
        max_inflight = max(8, int(mediapipe_workers * 4))

        def _append_output_row(
            frame_idx_local: int,
            timestamp_ms_local: int,
            pose_packet_local: Optional[Dict[str, np.ndarray]],
            rudder_angle_local: float,
            camera_pos_local: np.ndarray,
            R_wc_local: np.ndarray,
        ) -> None:
            nonlocal written_frames, last_progress_update

            row = [int(frame_idx_local), int(timestamp_ms_local)]
            _append_pose_packet_to_csv_row(row, pose_packet_local)

            if rudder_detector is not None:
                row.append(float(rudder_angle_local) if np.isfinite(rudder_angle_local) else np.nan)

            placed = None
            if K_undist is not None:
                row_dict = _pose_packet_to_row_dict(frame_idx_local, timestamp_ms_local, pose_packet_local)
                if row_dict is not None:
                    sitting_score = _estimate_sitting_score_from_row(row_dict)
                    raw_placed = compute_placed_skeleton(
                        row_dict,
                        K_undist=K_undist,
                        W=width,
                        H=height,
                        camera_pos=np.asarray(camera_pos_local, dtype=np.float64).reshape(3),
                        R_wc=np.asarray(R_wc_local, dtype=np.float64).reshape(3, 3),
                        z_plane_lm24=z_hip,
                        z_plane_lm28=z_ankle,
                        lower_landmark=lower_landmark,
                        contact_params=contact_params,
                    )
                    lm_conf = _extract_landmark_confidence_from_row(row_dict)
                    if skeleton_smoother is not None:
                        placed = skeleton_smoother.smooth(raw_placed, landmark_confidence=lm_conf)
                    else:
                        placed = raw_placed
                    if seated_x_stabilizer is not None and placed is not None:
                        placed = seated_x_stabilizer.apply(
                            placed,
                            sitting_score=float(sitting_score),
                            raw_reference=raw_placed,
                        )
                    if lateral_y_stabilizer is not None and placed is not None:
                        placed = lateral_y_stabilizer.apply(
                            placed,
                            sitting_score=float(sitting_score),
                            raw_reference=raw_placed,
                        )

            for i in range(33):
                if placed is not None and i in placed and placed[i] is not None:
                    row.extend([placed[i][0], placed[i][1], placed[i][2]])
                else:
                    row.extend([np.nan, np.nan, np.nan])

            metrics = compute_metrics_for_skeleton(placed, boat_com, athlete_mass)
            row.extend([
                metrics['trunk_angle'] if metrics['trunk_angle'] is not None else np.nan,
                metrics['com_x'] if metrics['com_x'] is not None else np.nan,
                metrics['com_y'] if metrics['com_y'] is not None else np.nan,
                metrics['com_z'] if metrics['com_z'] is not None else np.nan,
                metrics['moment_pitch'] if metrics['moment_pitch'] is not None else np.nan,
                metrics['moment_roll'] if metrics['moment_roll'] is not None else np.nan,
            ])

            rows.append(row)
            written_frames += 1

            if total_frames > 0:
                progress = int(10 + (written_frames / total_frames) * 85)
                if progress > last_progress_update + 1:
                    update_progress(project_id, progress, f"Processing frame {written_frames}/{total_frames}")
                    last_progress_update = progress
                    if (written_frames % 100) == 0:
                        add_log(project_id, f"Processed {written_frames}/{total_frames} frames")

        def _drain_ready_pose_packets() -> None:
            nonlocal next_emit_frame_idx
            while next_emit_frame_idx in ready_pose_packets:
                meta, pose_packet_local = ready_pose_packets.pop(next_emit_frame_idx)
                _append_output_row(
                    frame_idx_local=int(meta["frame_idx"]),
                    timestamp_ms_local=int(meta["timestamp_ms"]),
                    pose_packet_local=pose_packet_local,
                    rudder_angle_local=float(meta["rudder_angle"]),
                    camera_pos_local=np.asarray(meta["camera_pos"], dtype=np.float64).reshape(3),
                    R_wc_local=np.asarray(meta["R_wc"], dtype=np.float64).reshape(3, 3),
                )
                next_emit_frame_idx += 1

        def _collect_pose_futures(wait_for_one: bool) -> None:
            if not pending_pose_futures:
                return
            done, _ = concurrent.futures.wait(
                list(pending_pose_futures.keys()),
                timeout=None if wait_for_one else 0.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                meta = pending_pose_futures.pop(fut)
                pose_packet_local: Optional[Dict[str, np.ndarray]]
                try:
                    pose_packet_local = fut.result()
                except Exception as exc:
                    add_log(
                        project_id,
                        f"MediaPipe worker failed on frame {meta.get('frame_idx', '?')}: {exc}",
                        "error",
                    )
                    pose_packet_local = None
                ready_pose_packets[int(meta["frame_idx"])] = (meta, pose_packet_local)
            _drain_ready_pose_packets()

        segment_rows_ready = False
        if segment_parallel_enabled:
            cap.release()
            frame_ranges = _split_frame_ranges(total_frames, segment_instances)
            add_log(
                project_id,
                f"Starting {len(frame_ranges)} segment workers "
                "(MediaPipe + PnP + placement per segment).",
                "info",
            )

            progress_lock = threading.Lock()
            progress_written = 0
            progress_last_update = 0
            progress_last_log = 0

            def _segment_update_progress(increment: int) -> None:
                nonlocal progress_written, progress_last_update, progress_last_log
                if increment <= 0:
                    return
                with progress_lock:
                    progress_written += int(increment)
                    if total_frames > 0:
                        progress = int(10 + (progress_written / total_frames) * 85)
                        if progress > progress_last_update + 1:
                            update_progress(
                                project_id,
                                progress,
                                f"Processing frame {progress_written}/{total_frames}",
                            )
                            progress_last_update = progress
                        if (progress_written // 100) > progress_last_log:
                            progress_last_log = progress_written // 100
                            add_log(project_id, f"Processed {progress_written}/{total_frames} frames")

            def _make_segment_smoother() -> Optional[SkeletonPlacementKalman]:
                if not skeleton_filter_cfg.get("enabled", True):
                    return None
                return SkeletonPlacementKalman(
                    fps=fps,
                    process_noise_acc=float(skeleton_filter_cfg["process_noise_acc"]),
                    measurement_noise=float(skeleton_filter_cfg["measurement_noise"]),
                    use_landmark_confidence=bool(skeleton_filter_cfg["use_landmark_confidence"]),
                    min_landmark_confidence=float(skeleton_filter_cfg["min_landmark_confidence"]),
                    confidence_floor=float(skeleton_filter_cfg["confidence_floor"]),
                    confidence_power=float(skeleton_filter_cfg["confidence_power"]),
                    max_confidence_noise_scale=float(skeleton_filter_cfg["max_confidence_noise_scale"]),
                    gate_sigma=float(skeleton_filter_cfg["gate_sigma"]),
                    max_consecutive_misses=int(skeleton_filter_cfg["max_consecutive_misses"]),
                    initial_velocity_std=float(skeleton_filter_cfg["initial_velocity_std"]),
                    velocity_decay=float(skeleton_filter_cfg["velocity_decay"]),
                    max_speed=float(skeleton_filter_cfg["max_speed"]),
                    max_measurement_jump=float(skeleton_filter_cfg["max_measurement_jump"]),
                    reacquire_frames=int(skeleton_filter_cfg["reacquire_frames"]),
                    reacquire_max_jump=float(skeleton_filter_cfg["reacquire_max_jump"]),
                )

            def _make_segment_x_stabilizer() -> Optional[_SeatedForeAftStabilizer]:
                return create_seated_x_stabilizer(seated_x_stabilizer_cfg)

            def _make_segment_y_stabilizer() -> Optional[_LateralYStabilizer]:
                return create_lateral_y_stabilizer(lateral_y_stabilizer_cfg)

            def _process_segment(seg_idx: int, seg_start: int, seg_end: int) -> Dict[str, Any]:
                seg_rows: List[List[Any]] = []
                seg_label = f"[segment {seg_idx + 1}/{len(frame_ranges)}]"
                seg_cap = None
                seg_landmarker = None
                try:
                    seg_cap = cv2.VideoCapture(str(video_path))
                    if not seg_cap.isOpened():
                        raise RuntimeError(f"{seg_label} cannot open video")

                    if seg_start > 0:
                        seg_cap.set(cv2.CAP_PROP_POS_FRAMES, float(seg_start))
                        pos_after_seek = int(seg_cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                        while pos_after_seek < seg_start:
                            ok_skip, _ = seg_cap.read()
                            if not ok_skip:
                                break
                            pos_after_seek += 1

                    seg_options = vision.PoseLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=str(model_path)),
                        running_mode=vision.RunningMode.VIDEO,
                        output_segmentation_masks=False,
                    )
                    seg_landmarker = vision.PoseLandmarker.create_from_options(seg_options)
                    seg_rudder = PilotNetDetector(fps=fps) if rudder_enabled else None
                    seg_smoother = _make_segment_smoother()
                    seg_x_stabilizer = _make_segment_x_stabilizer()
                    seg_y_stabilizer = _make_segment_y_stabilizer()
                    seg_camera_pos = np.asarray(camera_pos, dtype=np.float64).reshape(3).copy()
                    seg_R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3).copy()

                    seg_next_pnp_frame = seg_start
                    seg_pnp_retry = 0
                    seg_pnp_model = _create_auto_camera_pnp_model_instance() if auto_pnp_active else None

                    for fidx in range(seg_start, seg_end):
                        ok_frame, frame_bgr = seg_cap.read()
                        if not ok_frame:
                            break

                        if undistort_maps is not None:
                            frame_bgr = cv2.remap(
                                frame_bgr,
                                undistort_maps[0],
                                undistort_maps[1],
                                cv2.INTER_LINEAR,
                            )

                        if auto_pnp_active and fidx >= seg_next_pnp_frame and K_undist is not None:
                            attempt = _attempt_auto_camera_pnp(
                                frame_bgr,
                                K_undist,
                                model_override=seg_pnp_model,
                            )
                            if attempt.get("success"):
                                seg_camera_pos = np.asarray(attempt["camera_pos"], dtype=np.float64).reshape(3)
                                seg_R_wc = np.asarray(attempt["R_wc"], dtype=np.float64).reshape(3, 3)
                                seg_next_pnp_frame = int(fidx + auto_pnp_interval)
                                seg_pnp_retry = 0
                            else:
                                seg_pnp_retry += 1
                                seg_next_pnp_frame = int(fidx + 1)
                                if seg_pnp_retry == 1 or (seg_pnp_retry % 25) == 0:
                                    add_log(
                                        project_id,
                                        f"{seg_label} Auto camera PnP retry #{seg_pnp_retry}: "
                                        f"{attempt.get('reason', 'unknown failure')}",
                                        "info",
                                    )

                        rudder_angle = np.nan
                        if seg_rudder is not None:
                            rudder_result = seg_rudder.detect(frame_bgr)
                            angle_2d = rudder_result.get("angle_2d")
                            if angle_2d is not None:
                                rudder_angle = float(angle_2d)

                        timestamp_ms = int((fidx / fps) * 1000)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                        pose_packet = _pose_result_to_packet(seg_landmarker.detect_for_video(mp_image, timestamp_ms))

                        row = [int(fidx), int(timestamp_ms)]
                        _append_pose_packet_to_csv_row(row, pose_packet)
                        if rudder_enabled:
                            row.append(float(rudder_angle) if np.isfinite(rudder_angle) else np.nan)

                        placed = None
                        if K_undist is not None:
                            row_dict = _pose_packet_to_row_dict(fidx, timestamp_ms, pose_packet)
                            if row_dict is not None:
                                sitting_score = _estimate_sitting_score_from_row(row_dict)
                                raw_placed = compute_placed_skeleton(
                                    row_dict,
                                    K_undist=K_undist,
                                    W=width,
                                    H=height,
                                    camera_pos=np.asarray(seg_camera_pos, dtype=np.float64).reshape(3),
                                    R_wc=np.asarray(seg_R_wc, dtype=np.float64).reshape(3, 3),
                                    z_plane_lm24=z_hip,
                                    z_plane_lm28=z_ankle,
                                    lower_landmark=lower_landmark,
                                    contact_params=contact_params,
                                )
                                lm_conf = _extract_landmark_confidence_from_row(row_dict)
                                placed = (
                                    seg_smoother.smooth(raw_placed, landmark_confidence=lm_conf)
                                    if seg_smoother is not None
                                    else raw_placed
                                )
                                if seg_x_stabilizer is not None and placed is not None:
                                    placed = seg_x_stabilizer.apply(
                                        placed,
                                        sitting_score=float(sitting_score),
                                        raw_reference=raw_placed,
                                    )
                                if seg_y_stabilizer is not None and placed is not None:
                                    placed = seg_y_stabilizer.apply(
                                        placed,
                                        sitting_score=float(sitting_score),
                                        raw_reference=raw_placed,
                                    )

                        for i in range(33):
                            if placed is not None and i in placed and placed[i] is not None:
                                row.extend([placed[i][0], placed[i][1], placed[i][2]])
                            else:
                                row.extend([np.nan, np.nan, np.nan])

                        metrics = compute_metrics_for_skeleton(placed, boat_com, athlete_mass)
                        row.extend([
                            metrics["trunk_angle"] if metrics["trunk_angle"] is not None else np.nan,
                            metrics["com_x"] if metrics["com_x"] is not None else np.nan,
                            metrics["com_y"] if metrics["com_y"] is not None else np.nan,
                            metrics["com_z"] if metrics["com_z"] is not None else np.nan,
                            metrics["moment_pitch"] if metrics["moment_pitch"] is not None else np.nan,
                            metrics["moment_roll"] if metrics["moment_roll"] is not None else np.nan,
                        ])
                        seg_rows.append(row)
                        _segment_update_progress(1)

                    return {
                        "rows": seg_rows,
                        "end_frame": int(seg_start + len(seg_rows)),
                        "camera_pos": np.asarray(seg_camera_pos, dtype=np.float64).reshape(3),
                        "R_wc": np.asarray(seg_R_wc, dtype=np.float64).reshape(3, 3),
                    }
                finally:
                    if seg_cap is not None:
                        seg_cap.release()
                    if seg_landmarker is not None:
                        seg_landmarker.close()

            segment_results: Dict[int, Dict[str, Any]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(frame_ranges),
                thread_name_prefix="segment_proc",
            ) as seg_executor:
                future_to_meta: Dict[concurrent.futures.Future, Tuple[int, int, int]] = {}
                for seg_idx, (seg_start, seg_end) in enumerate(frame_ranges):
                    fut = seg_executor.submit(_process_segment, seg_idx, int(seg_start), int(seg_end))
                    future_to_meta[fut] = (seg_idx, int(seg_start), int(seg_end))

                for fut in concurrent.futures.as_completed(future_to_meta):
                    seg_idx, seg_start, seg_end = future_to_meta[fut]
                    seg_result = fut.result()
                    segment_results[seg_idx] = seg_result
                    seg_rows_done = int(len(seg_result.get("rows", [])))
                    add_log(
                        project_id,
                        f"[segment {seg_idx + 1}/{len(frame_ranges)}] completed {seg_rows_done} frames "
                        f"({seg_start}-{max(seg_start, seg_start + seg_rows_done - 1)})",
                        "success",
                    )

            for seg_idx in range(len(frame_ranges)):
                rows.extend(segment_results[seg_idx].get("rows", []))

            if rows:
                rows.sort(key=lambda r: int(r[0]))

            if segment_results:
                last_seg = max(segment_results.values(), key=lambda item: int(item.get("end_frame", 0)))
                camera_pos = np.asarray(last_seg.get("camera_pos", camera_pos), dtype=np.float64).reshape(3)
                R_wc = np.asarray(last_seg.get("R_wc", R_wc), dtype=np.float64).reshape(3, 3)
                config["camera_position"] = [float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])]
                config["camera_R_wc"] = np.asarray(R_wc, dtype=np.float64).tolist()
                angles = _camera_pose_angles_from_rwc(R_wc)
                config["camera_pitch_deg"] = float(angles.get("pitch_deg", config.get("camera_pitch_deg", 0.0)))
                config["camera_yaw_deg"] = float(angles.get("yaw_deg", config.get("camera_yaw_deg", 0.0)))
                config["camera_roll_deg"] = float(angles.get("roll_deg", config.get("camera_roll_deg", 0.0)))
            segment_rows_ready = True

        try:
            while not segment_rows_ready:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Undistort if calibration available
                if undistort_maps is not None:
                    frame_bgr = cv2.remap(
                        frame_bgr,
                        undistort_maps[0],
                        undistort_maps[1],
                        cv2.INTER_LINEAR,
                    )

                # Periodically auto-solve camera pose from YOLO keypoints.
                # Collects successful solves across frames (non-consecutive), then
                # robustly aggregates them via median + outlier filtering.
                if auto_pnp_active:
                    if (not auto_pnp_collect_active) and (auto_pnp_retry_active or (frame_idx >= next_auto_pnp_frame)):
                        auto_pnp_collect_active = True
                        auto_pnp_collect_start_frame = frame_idx
                        auto_pnp_collect_successes = []
                        auto_pnp_collect_fail_reasons = []
                        auto_pnp_collect_fail_count = 0

                    if auto_pnp_collect_active:
                        attempt = _attempt_auto_camera_pnp(frame_bgr, K_undist)
                        if attempt.get("success"):
                            auto_pnp_collect_successes.append(attempt)
                        else:
                            auto_pnp_collect_fail_count += 1
                            auto_pnp_collect_fail_reasons.append(str(attempt.get("reason", "unknown failure")))
                            # Avoid excessive log noise while still showing continued retries.
                            if auto_pnp_collect_fail_count == 1 or (auto_pnp_collect_fail_count % 25) == 0:
                                add_log(
                                    project_id,
                                    f"Auto camera PnP collecting: {len(auto_pnp_collect_successes)}/{auto_pnp_target_successes} "
                                    f"valid solves so far; latest fail: {auto_pnp_collect_fail_reasons[-1]}",
                                    "info",
                                )

                        if len(auto_pnp_collect_successes) >= auto_pnp_target_successes:
                            frame_a = auto_pnp_collect_start_frame
                            frame_b = frame_idx
                            averaged = None
                            batch_valid = False
                            batch_reason = ""
                            averaged = _average_auto_pnp_solutions(auto_pnp_collect_successes)
                            if not averaged.get("success"):
                                batch_reason = str(averaged.get("reason", "robust aggregation failed"))
                            else:
                                is_valid, issues = _validate_auto_camera_pose(
                                    averaged["camera_pos"],
                                    averaged["angles"],
                                    averaged["err_px"],
                                )
                                if not is_valid:
                                    batch_reason = "; ".join(issues)
                                else:
                                    batch_valid = True

                            if batch_valid and averaged is not None:
                                camera_pos = np.asarray(averaged["camera_pos"], dtype=np.float64).reshape(3)
                                R_wc = np.asarray(averaged["R_wc"], dtype=np.float64).reshape(3, 3)
                                angles = averaged.get("angles", {})
                                err_px = float(averaged.get("err_px", np.nan))
                                n_in = int(averaged.get("num_inliers", 0))
                                n_all = int(averaged.get("num_pairs", 0))
                                mapping = str(averaged.get("mapping", "unknown"))
                                solve_method = str(averaged.get("solve_method", "unknown"))
                                total_n = int(averaged.get("num_frame_solutions", len(auto_pnp_collect_successes)))
                                used_n = int(averaged.get("num_used_after_filter", total_n))
                                rej_n = int(averaged.get("num_rejected_outliers", max(0, total_n - used_n)))
                                pitch_deg = float(angles.get("pitch_deg", np.nan))
                                yaw_deg = float(angles.get("yaw_deg", np.nan))
                                roll_deg = float(angles.get("roll_deg", np.nan))

                                # Persist latest accepted camera pose in config output.
                                config["camera_position"] = [float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])]
                                config["camera_R_wc"] = np.asarray(R_wc, dtype=np.float64).tolist()
                                if np.isfinite(pitch_deg):
                                    config["camera_pitch_deg"] = pitch_deg
                                if np.isfinite(yaw_deg):
                                    config["camera_yaw_deg"] = yaw_deg
                                if np.isfinite(roll_deg):
                                    config["camera_roll_deg"] = roll_deg

                                add_log(
                                    project_id,
                                    f"Auto camera PnP success @ frames {frame_a}-{frame_b} "
                                    f"(median over {used_n}/{total_n} accepted, rejected={rej_n}, failed={auto_pnp_collect_fail_count}): "
                                    f"x={camera_pos[0]:.3f}, y={camera_pos[1]:.3f}, z={camera_pos[2]:.3f}, "
                                    f"pitch={pitch_deg:.2f}, yaw={yaw_deg:.2f}, roll={roll_deg:.2f}, "
                                    f"inliers={n_in}/{n_all}, err={err_px:.2f}px, mapping={mapping}, solver={solve_method}",
                                    "success",
                                )

                                auto_pnp_retry_active = False
                                auto_pnp_retry_count = 0
                                next_auto_pnp_frame = frame_idx + auto_pnp_interval
                            else:
                                auto_pnp_retry_active = True
                                auto_pnp_retry_count += 1
                                if auto_pnp_retry_count == 1 or (auto_pnp_retry_count % 10) == 0:
                                    add_log(
                                        project_id,
                                        f"Auto camera PnP failed @ frames {frame_a}-{frame_b} "
                                        f"(retry cycle #{auto_pnp_retry_count} from next frame): {batch_reason}",
                                        "error",
                                    )

                            auto_pnp_collect_active = False
                            auto_pnp_collect_start_frame = -1
                            auto_pnp_collect_successes = []
                            auto_pnp_collect_fail_reasons = []
                            auto_pnp_collect_fail_count = 0

                rudder_angle = np.nan
                if rudder_detector is not None:
                    rudder_result = rudder_detector.detect(frame_bgr)
                    angle_2d = rudder_result.get('angle_2d')
                    if angle_2d is not None:
                        rudder_angle = float(angle_2d)

                timestamp_ms = int((frame_idx / fps) * 1000)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                camera_pos_frame = np.asarray(camera_pos, dtype=np.float64).reshape(3).copy()
                R_wc_frame = np.asarray(R_wc, dtype=np.float64).reshape(3, 3).copy()

                if parallel_pose_enabled:
                    assert pose_executor is not None
                    assert threaded_pose_detect is not None
                    fut = pose_executor.submit(threaded_pose_detect, frame_rgb)
                    pending_pose_futures[fut] = {
                        "frame_idx": int(frame_idx),
                        "timestamp_ms": int(timestamp_ms),
                        "rudder_angle": float(rudder_angle),
                        "camera_pos": camera_pos_frame,
                        "R_wc": R_wc_frame,
                    }

                    if len(pending_pose_futures) >= max_inflight:
                        _collect_pose_futures(wait_for_one=True)
                    else:
                        _collect_pose_futures(wait_for_one=False)
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    pose_packet = _pose_result_to_packet(result)
                    _append_output_row(
                        frame_idx_local=frame_idx,
                        timestamp_ms_local=timestamp_ms,
                        pose_packet_local=pose_packet,
                        rudder_angle_local=float(rudder_angle),
                        camera_pos_local=camera_pos_frame,
                        R_wc_local=R_wc_frame,
                    )

                frame_idx += 1

            if parallel_pose_enabled:
                while pending_pose_futures:
                    _collect_pose_futures(wait_for_one=True)
                _drain_ready_pose_packets()
        finally:
            cap.release()
            if pose_executor is not None:
                pose_executor.shutdown(wait=True)
            if threaded_pose_close is not None:
                threaded_pose_close()
            if landmarker is not None:
                landmarker.close()
        
        # Save CSV
        add_log(project_id, "Saving pose data...")
        update_progress(project_id, 96, "Saving pose data")
        
        df = pd.DataFrame(rows, columns=header)
        df.to_csv(output_csv, index=False)
        
        add_log(project_id, f"Saved {len(df)} frames to pose.csv", "success")
        
        # Update config
        config["state"] = "completed"
        config["frame_count"] = len(df)
        config["fps"] = fps
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        update_progress(project_id, 100, "Complete")
        if project_id in processing_jobs:
            processing_jobs[project_id]["state"] = "completed"
        
        add_log(project_id, "Processing complete!", "success")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        add_log(project_id, f"Error: {error_msg}", "error")
        add_log(project_id, traceback.format_exc(), "error")
        
        if project_id in processing_jobs:
            processing_jobs[project_id]["state"] = "error"
            processing_jobs[project_id]["error"] = error_msg
        
        # Update config
        try:
            with open(config_path) as f:
                config = json.load(f)
            config["state"] = "error"
            config["error"] = error_msg
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass


def load_calibration(npz_path: str, img_width: int, img_height: int):
    """Load fisheye calibration and create undistort maps."""
    cal = np.load(npz_path)
    K = cal["K"]
    D = cal["D"]
    img_size = (img_width, img_height)
    
    # Estimate new camera matrix
    R = np.eye(3)
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_size, R, balance=0.0, new_size=img_size
    )
    
    # Create undistort maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, K_new, img_size, cv2.CV_16SC2
    )
    
    return (map1, map2)
