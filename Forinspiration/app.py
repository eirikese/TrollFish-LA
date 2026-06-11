import os
import json
import uuid
import shutil
import tempfile
import time
import math
import io
import base64
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response
from flask_compress import Compress
from werkzeug.utils import secure_filename

from skeletontocsv import process_video_to_csv
from rayplane import (
    default_camera_pose_and_rotation,
    load_fisheye_undistorted_intrinsics,
    intersect_world_z_plane,
    ray_from_norm_landmark_undistorted,
    get_landmark_norm,
    place_skeleton_on_boat,
)
from skeleton_metrics import compute_center_of_mass
from skeleton_filter import (
    SkeletonPlacementKalman,
    normalize_skeleton_filter_params,
)
from rudder_nn import PilotNetDetector
from processing import compute_placed_skeleton
import processing

# Graph data filtering thresholds
FILTER_THRESHOLDS = {
    'trunk_angle': {'min': 0, 'max': 180},  # degrees
    'moment_x': {'min': -1500, 'max': 100},  # Nm
    'moment_y': {'min': -1300, 'max': 1300},  # Nm
    'com_x': {'min': -3, 'max': 0},  # meters
    'com_y': {'min': -2, 'max': 2},  # meters
    'com_z': {'min': -0.5, 'max': 0.5},  # meters
}

# Rudder low pass filter coefficient (0-1, higher = smoother but more lag)
RUDDER_LPF_ALPHA = 0.0



app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024 * 1024  # 25GB  max upload

# Enable gzip compression for faster data transfer
Compress(app)
app.config['COMPRESS_MIMETYPES'] = ['application/json', 'text/html', 'text/css', 'text/javascript', 'application/javascript']
app.config['COMPRESS_LEVEL'] = 6
app.config['COMPRESS_MIN_SIZE'] = 500

# --- Configuration ---
BOAT_STL_PATH = "Hull.stl"
POSE_CSV_PATH = "pose.csv"
FISHEYE_CALIB_NPZ = "gopro_fisheye_calib.npz"
Z_PLANE_LM24 = 0.10
Z_PLANE_LM28 = 0.01
VIDEO_FILE = "GX010235.MP4"
PNP_YOLO_MODEL_PATH = Path(__file__).parent / "best.pt"
PNP_YOLO_MIN_CONF = 0.8
PNP_YOLO_KEYPOINT_LABELS = [
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
PNP_YOLO_OBJECT_POINTS = {
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

# MediaPipe connections for skeleton
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

BOAT_COM = -1.114
ATHLETE_MASS_KG = 75.0
GRAVITY = 9.81

# --- Global Data ---
df = None
camera_pos = None
R_wc = None
K_undist = None
W, H = None, None
boat_stl_path = None
placed_cache = {}
fps_estimate = 30.0
# --- Project-specific caches ---
project_data_cache = {}  # {project_id: {df, K, W, H, camera_pos, R_wc, config, ...}}
project_placed_cache = {}  # {project_id: {frame_idx: placed_skeleton}}
project_metrics_cache = {}  # {project_id: metrics_response}
project_bulk_cache = {}  # {"project_id|pose=0/1": bulk_response}
project_chunk_cache = {}  # {"project_id|start=..|count=..|pose=0/1": chunk_response}
skeleton_tuning_cache = {}  # {cache_id: cached raw placement snippet for tuning}
hull_side_profile_cache = {}  # {"stl_path|mtime_ns": payload}
HULL_SIDE_PROFILE_CACHE_VERSION = "v2"
gps_sync_mp4_cache = {}  # {"path|mtime_ns": [samples]}


def parse_camera_rotation_matrix(value):
    """Parse a 3x3 camera->world rotation matrix from incoming payload."""
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


BASE_R_WC = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def _rotation_matrix_to_euler_xyz_deg(R):
    """Extract XYZ Euler angles (rx, ry, rz) in degrees from R = Rz*Ry*Rx."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    ry = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
        rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        # Gimbal lock fallback
        rx = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        rz = 0.0
    return np.degrees([rx, ry, rz])


def camera_pose_angles_from_rwc(R_wc):
    """Return human-readable camera pose angles from camera->world rotation."""
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    R_rel = BASE_R_WC.T @ R_wc
    rx, ry, rz = _rotation_matrix_to_euler_xyz_deg(R_rel)
    # Preserve app convention: pitch positive means looking down.
    return {
        "pitch_deg": float(-rx),
        "yaw_deg": float(ry),
        "roll_deg": float(rz),
    }


def safe_unlink(path, retries=5, delay_s=0.05):
    """Best-effort unlink for Windows, where file handles can linger briefly."""
    if not path:
        return
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.unlink(path)
            return
        except PermissionError:
            if i == retries - 1:
                return
            time.sleep(delay_s)
        except Exception:
            return


def apply_low_pass_filter(values, alpha=RUDDER_LPF_ALPHA):
    """Apply exponential moving average (low pass filter) to a list of values.
    
    Args:
        values: List of values (can contain None)
        alpha: Smoothing factor (0-1). Higher = smoother but more lag.
               Output = alpha * previous + (1 - alpha) * current
    
    Returns:
        List of filtered values (None values are replaced with last valid value)
    """
    if not values:
        return values
    
    filtered = []
    last_valid = None
    
    for val in values:
        if val is None:
            # Hold the last valid value instead of outputting None
            if last_valid is not None:
                filtered.append(last_valid)
            else:
                filtered.append(None)
        elif last_valid is None:
            # First valid value, no filtering
            last_valid = val
            filtered.append(val)
        else:
            # Apply low pass filter: new = alpha * old + (1 - alpha) * new
            smoothed = alpha * last_valid + (1 - alpha) * val
            last_valid = smoothed
            filtered.append(round(smoothed, 2))
    
    return filtered


def is_plausible_com(com):
    """Check if COM position is plausible."""
    if com is None:
        return False
    x, y, z = com[0], com[1], com[2]
    # Check against filter thresholds
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
    
    # Need shoulders (11,12) and hips (23,24)
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


def get_project_data(project_id):
    """Load and cache project data (CSV, config)."""
    global project_data_cache
    
    if project_id in project_data_cache:
        return project_data_cache[project_id]
    
    project_path = processing.get_project_path(project_id)
    config = processing.get_project_config(project_id)
    
    if not config:
        return None
    
    pose_csv = project_path / "pose.csv"
    if not pose_csv.exists():
        return None
    
    project_df = pd.read_csv(pose_csv)
    
    project_camera_pos, project_R_wc = processing.get_camera_pose_from_config(config)
    
    data = {
        "df": project_df,
        "camera_pos": project_camera_pos,
        "R_wc": project_R_wc,
        "config": config,
        "boat_com": config.get("boat_com", -1.114),
        "athlete_mass": config.get("athlete_weight", 75.0),
    }
    
    project_data_cache[project_id] = data
    return data


def invalidate_project_caches(project_id):
    """Clear cached project blobs that depend on config/derived metrics."""
    global project_data_cache, project_metrics_cache
    project_data_cache.pop(project_id, None)
    project_metrics_cache.pop(project_id, None)


def _update_project_config(project_id, updates=None, remove_keys=None):
    """Update a project's config.json in-place and return the new dict."""
    updates = updates or {}
    remove_keys = remove_keys or ()
    project_path = processing.get_project_path(project_id)
    config_path = project_path / "config.json"
    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for k in remove_keys:
        config.pop(k, None)
    for k, v in updates.items():
        config[k] = v

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def _resolve_project_video_path(project_path, config):
    """Resolve the most likely project video file path."""
    candidates = []
    configured_name = str((config or {}).get("video_file", "") or "").strip()
    if configured_name:
        candidates.append(Path(configured_name).name)
    candidates.append("video.mp4")

    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        p = Path(project_path) / name
        if p.exists() and p.is_file():
            return p

    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.mkv", "*.MKV", "*.avi", "*.AVI"):
        matches = sorted(Path(project_path).glob(ext))
        if matches:
            return matches[0]
    return None


def get_project_placed(project_id, frame_idx, row, data):
    """Get pre-computed placed skeleton from CSV data for a project frame."""
    global project_placed_cache
    
    if project_id not in project_placed_cache:
        project_placed_cache[project_id] = {}
    
    cache = project_placed_cache[project_id]
    if frame_idx in cache:
        return cache[frame_idx]
    
    # Extract pre-computed skeleton positions from CSV columns
    placed = {}
    has_valid_skeleton = False
    
    for i in range(33):
        x_col = f"skel{i}_x"
        y_col = f"skel{i}_y"
        z_col = f"skel{i}_z"
        
        if x_col in row and y_col in row and z_col in row:
            x = row.get(x_col)
            y = row.get(y_col)
            z = row.get(z_col)
            
            if pd.notna(x) and pd.notna(y) and pd.notna(z):
                placed[i] = np.array([float(x), float(y), float(z)])
                has_valid_skeleton = True
            else:
                placed[i] = None
        else:
            placed[i] = None
    
    if not has_valid_skeleton:
        placed = None
    
    cache[frame_idx] = placed
    return placed


REPORT_HIKING_TRUNK_DEG = 38.0
REPORT_HIGH_ROLL_LOAD_NM = 350.0


def _parse_int_query_arg(raw_value):
    if raw_value in (None, ""):
        return None
    try:
        return int(raw_value)
    except Exception:
        return None


def _numeric_column(frame_df, column_name):
    if column_name not in frame_df.columns:
        return np.full(len(frame_df), np.nan, dtype=np.float64)
    return pd.to_numeric(frame_df[column_name], errors="coerce").to_numpy(dtype=np.float64)


def _series_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out = {
        "count": int(arr.size),
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "p05": None,
        "p25": None,
        "p75": None,
        "p95": None,
        "abs_mean": None,
        "abs_p95": None,
        "peak_abs": None,
    }
    if arr.size == 0:
        return out
    abs_arr = np.abs(arr)
    out.update({
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "abs_mean": float(np.mean(abs_arr)),
        "abs_p95": float(np.percentile(abs_arr, 95)),
        "peak_abs": float(np.max(abs_arr)),
    })
    return out


def _compute_time_axis_seconds(frame_df, fallback_fps):
    n = len(frame_df)
    fps = max(float(fallback_fps), 1.0)
    default_dt = 1.0 / fps
    if n <= 0:
        return np.zeros(0, dtype=np.float64), 0.0, fps, default_dt

    if "timestamp_ms" in frame_df.columns:
        ts = pd.to_numeric(frame_df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(ts)
        if np.count_nonzero(finite_mask) >= 2:
            idx = np.arange(n, dtype=np.float64)
            t = np.full(n, np.nan, dtype=np.float64)
            first_ts = float(ts[finite_mask][0])
            t[finite_mask] = (ts[finite_mask] - first_ts) / 1000.0
            if np.any(~finite_mask):
                t = np.interp(idx, idx[finite_mask], t[finite_mask])
            diffs = np.diff(t)
            diffs = diffs[diffs > 1e-6]
            dt = float(np.median(diffs)) if diffs.size > 0 else default_dt
            duration_s = float(max(0.0, t[-1] - t[0])) if n >= 2 else 0.0
            fps_eff = float(1.0 / dt) if dt > 1e-9 else fps
            return t, duration_s, fps_eff, dt

    t = np.arange(n, dtype=np.float64) * default_dt
    duration_s = float((n - 1) * default_dt) if n >= 2 else 0.0
    return t, duration_s, fps, default_dt


def _build_report_plot_image_base64(time_s, trunk, moment_pitch, moment_roll, com_x, com_y):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time_s = np.asarray(time_s, dtype=np.float64)
        trunk = np.asarray(trunk, dtype=np.float64)
        moment_pitch = np.asarray(moment_pitch, dtype=np.float64)
        moment_roll = np.asarray(moment_roll, dtype=np.float64)
        com_x = np.asarray(com_x, dtype=np.float64)
        com_y = np.asarray(com_y, dtype=np.float64)

        fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.0), sharex=True)
        fig.patch.set_facecolor("white")

        ax0, ax1, ax2 = axes
        for ax in axes:
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
            ax.set_facecolor("#f8fafc")

        ax0.plot(time_s, trunk, color="#1f77b4", linewidth=1.6)
        ax0.axhline(REPORT_HIKING_TRUNK_DEG, color="#6b7280", linestyle="--", linewidth=1.0)
        ax0.set_ylabel("Trunk (deg)")
        ax0.set_title("Session Timeline")

        ax1.plot(time_s, moment_pitch, color="#0ea5e9", linewidth=1.2, label="Pitch moment")
        ax1.plot(time_s, moment_roll, color="#10b981", linewidth=1.2, label="Roll moment")
        ax1.axhline(REPORT_HIGH_ROLL_LOAD_NM, color="#64748b", linestyle="--", linewidth=0.9)
        ax1.axhline(-REPORT_HIGH_ROLL_LOAD_NM, color="#64748b", linestyle="--", linewidth=0.9)
        ax1.set_ylabel("Moment (Nm)")
        ax1.legend(loc="upper right", fontsize=8)

        ax2.plot(time_s, com_x, color="#8b5cf6", linewidth=1.2, label="COM X")
        ax2.plot(time_s, com_y, color="#f59e0b", linewidth=1.2, label="COM Y")
        ax2.set_ylabel("COM (m)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        return None


def _build_report_insights(report):
    insights = []
    quality = report.get("quality", {})
    stats = report.get("stats", {})
    derived = report.get("derived", {})

    metric_cov = quality.get("frame_metric_coverage_pct")
    if metric_cov is not None and metric_cov < 70.0:
        insights.append("Data quality: less than 70% of frames have valid computed metrics in this range. Re-check camera and landmark visibility for stronger analytics.")

    hiking_pct = derived.get("hiking_pct")
    if hiking_pct is not None:
        if hiking_pct < 12.0:
            insights.append("Load profile: very little time above hiking trunk threshold. If conditions allowed, increase controlled hiking exposure for stronger righting moment.")
        elif hiking_pct > 55.0:
            insights.append("Load profile: sustained high hiking time. Ensure pacing and recovery strategy so technique quality does not degrade late in the session.")

    trunk_std = (stats.get("trunk_angle") or {}).get("std")
    if trunk_std is not None and trunk_std > 14.0:
        insights.append("Posture consistency: trunk angle variability is high. Focus on smoother transitions between sit and hike phases to reduce energy leakage.")

    roll_peak = (stats.get("moment_roll") or {}).get("peak_abs")
    if roll_peak is not None and roll_peak < 320.0:
        insights.append("Roll leverage: peak roll moment is relatively low. Review body projection and timing relative to gusts to convert posture into more righting load.")

    lateral_span = derived.get("lateral_span_m")
    if lateral_span is not None and lateral_span > 0.48:
        insights.append("Lateral stability: COM side-to-side range is large. Target steadier edge control to keep power transfer more continuous.")

    fore_aft_span = derived.get("fore_aft_span_m")
    if fore_aft_span is not None and fore_aft_span > 0.85:
        insights.append("Fore-aft trim control: COM travel along the boat is wide. Tighten trim movement windows to reduce pitch disruptions.")

    if not insights:
        insights.append("Session looked balanced in this range: no dominant risk flags from trunk, moment, or COM distributions.")

    return insights[:6]


def build_project_report(project_id, start_frame=None, end_frame=None, include_chart=True):
    data = get_project_data(project_id)
    if not data:
        return None, "Project not found"

    project_df = data["df"]
    total_frames = int(len(project_df))
    if total_frames <= 0:
        return None, "Project has no frames"

    start = int(0 if start_frame is None else max(0, min(int(start_frame), total_frames - 1)))
    end = int(total_frames - 1 if end_frame is None else max(0, min(int(end_frame), total_frames - 1)))
    if end < start:
        end = start

    section_df = project_df.iloc[start:end + 1]
    frame_count = int(len(section_df))
    config = data.get("config", {}) or {}

    configured_fps = float(config.get("fps", 30.0) or 30.0)
    time_s, duration_s, fps_eff, dt_s = _compute_time_axis_seconds(section_df, configured_fps)

    trunk_angle = _numeric_column(section_df, "trunk_angle")
    moment_pitch = _numeric_column(section_df, "moment_pitch")
    moment_roll = _numeric_column(section_df, "moment_roll")
    com_x = _numeric_column(section_df, "com_x")
    com_y = _numeric_column(section_df, "com_y")
    com_z = _numeric_column(section_df, "com_z")
    rudder_angle = _numeric_column(section_df, "rudder_angle")
    boom_angle = _numeric_column(section_df, "boom_angle")

    metric_valid = (
        np.isfinite(trunk_angle) |
        np.isfinite(moment_pitch) |
        np.isfinite(moment_roll) |
        np.isfinite(com_x) |
        np.isfinite(com_y) |
        np.isfinite(com_z)
    )
    frame_metric_coverage_pct = float(100.0 * np.mean(metric_valid)) if frame_count > 0 else 0.0

    if "skel23_x" in section_df.columns and "skel24_x" in section_df.columns:
        hip_l_x = _numeric_column(section_df, "skel23_x")
        hip_r_x = _numeric_column(section_df, "skel24_x")
        skeleton_valid = np.isfinite(hip_l_x) & np.isfinite(hip_r_x)
        skeleton_coverage_pct = float(100.0 * np.mean(skeleton_valid)) if frame_count > 0 else 0.0
    else:
        skeleton_coverage_pct = frame_metric_coverage_pct

    hiking_mask = np.isfinite(trunk_angle) & (trunk_angle >= REPORT_HIKING_TRUNK_DEG)
    high_roll_mask = np.isfinite(moment_roll) & (np.abs(moment_roll) >= REPORT_HIGH_ROLL_LOAD_NM)
    hiking_time_s = float(np.sum(hiking_mask) * dt_s)
    high_roll_time_s = float(np.sum(high_roll_mask) * dt_s)

    duration_for_ratio = max(duration_s, dt_s)
    hiking_pct = float(100.0 * hiking_time_s / duration_for_ratio) if duration_for_ratio > 1e-9 else 0.0
    high_roll_pct = float(100.0 * high_roll_time_s / duration_for_ratio) if duration_for_ratio > 1e-9 else 0.0

    finite_roll = np.where(np.isfinite(moment_roll), np.abs(moment_roll), 0.0)
    roll_work_proxy_kNms = float(np.sum(finite_roll) * dt_s / 1000.0)

    com_x_stats = _series_stats(com_x)
    com_y_stats = _series_stats(com_y)

    fore_aft_span = None
    if com_x_stats.get("p95") is not None and com_x_stats.get("p05") is not None:
        fore_aft_span = float(com_x_stats["p95"] - com_x_stats["p05"])
    lateral_span = None
    if com_y_stats.get("p95") is not None and com_y_stats.get("p05") is not None:
        lateral_span = float(com_y_stats["p95"] - com_y_stats["p05"])

    report = {
        "project_id": project_id,
        "project_name": str(config.get("name", project_id)),
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "range": {
            "start_frame": int(start),
            "end_frame": int(end),
            "frame_count": int(frame_count),
        },
        "duration_s": float(duration_s),
        "fps_effective": float(fps_eff),
        "athlete": {
            "weight_kg": float(config.get("athlete_weight", 75.0)),
            "boat_com_x_m": float(config.get("boat_com", -1.114)),
            "hip_plane_m": float(config.get("hip_height", 0.10)),
            "ankle_plane_m": float(config.get("ankle_height", 0.01)),
        },
        "settings": {
            "pose_model": config.get("pose_model"),
            "lower_landmark": config.get("lower_landmark"),
            "skeleton_filter": config.get("skeleton_filter"),
            "seated_x_stabilizer": config.get("seated_x_stabilizer"),
            "lateral_y_stabilizer": config.get("lateral_y_stabilizer"),
            "contact_params": config.get("contact_params"),
        },
        "quality": {
            "frame_metric_coverage_pct": frame_metric_coverage_pct,
            "skeleton_coverage_pct": skeleton_coverage_pct,
        },
        "stats": {
            "trunk_angle": _series_stats(trunk_angle),
            "moment_pitch": _series_stats(moment_pitch),
            "moment_roll": _series_stats(moment_roll),
            "com_x": com_x_stats,
            "com_y": com_y_stats,
            "com_z": _series_stats(com_z),
            "rudder_angle": _series_stats(rudder_angle),
            "boom_angle": _series_stats(boom_angle),
        },
        "derived": {
            "hiking_threshold_deg": float(REPORT_HIKING_TRUNK_DEG),
            "high_roll_threshold_nm": float(REPORT_HIGH_ROLL_LOAD_NM),
            "hiking_time_s": hiking_time_s,
            "hiking_pct": hiking_pct,
            "high_roll_load_time_s": high_roll_time_s,
            "high_roll_load_pct": high_roll_pct,
            "roll_work_proxy_kNms": roll_work_proxy_kNms,
            "fore_aft_span_m": fore_aft_span,
            "lateral_span_m": lateral_span,
        },
    }
    report["insights"] = _build_report_insights(report)

    if include_chart:
        report["plot_image_base64"] = _build_report_plot_image_base64(
            time_s=time_s,
            trunk=trunk_angle,
            moment_pitch=moment_pitch,
            moment_roll=moment_roll,
            com_x=com_x,
            com_y=com_y,
        )

    return report, None


def _nullable_float_list(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(v) if np.isfinite(v) else None for v in arr]


def build_project_report_timeseries(project_id, start_frame=None, end_frame=None):
    data = get_project_data(project_id)
    if not data:
        return None, "Project not found"

    project_df = data["df"]
    total_frames = int(len(project_df))
    if total_frames <= 0:
        return None, "Project has no frames"

    start = int(0 if start_frame is None else max(0, min(int(start_frame), total_frames - 1)))
    end = int(total_frames - 1 if end_frame is None else max(0, min(int(end_frame), total_frames - 1)))
    if end < start:
        end = start

    section_df = project_df.iloc[start:end + 1]
    frame_count = int(len(section_df))
    config = data.get("config", {}) or {}
    configured_fps = float(config.get("fps", 30.0) or 30.0)

    frame_idx = np.arange(start, end + 1, dtype=np.int32)
    timestamps = _numeric_column(section_df, "timestamp_ms")
    time_s, duration_s, fps_eff, dt_s = _compute_time_axis_seconds(section_df, configured_fps)

    trunk_angle = _numeric_column(section_df, "trunk_angle")
    moment_pitch = _numeric_column(section_df, "moment_pitch")
    moment_roll = _numeric_column(section_df, "moment_roll")
    com_x = _numeric_column(section_df, "com_x")
    com_y = _numeric_column(section_df, "com_y")
    com_z = _numeric_column(section_df, "com_z")
    rudder_angle = _numeric_column(section_df, "rudder_angle")
    boom_angle = _numeric_column(section_df, "boom_angle")
    boat_heel = np.full(frame_count, np.nan, dtype=np.float64)
    boat_trim = np.full(frame_count, np.nan, dtype=np.float64)
    boat_attitude_meta = {
        "available": False,
        "file": None,
        "offset_s": 0.0,
        "points": 0,
        "synced_points": 0,
        "columns": {},
        "reason": "Unavailable",
    }
    try:
        boat_heel_full, boat_trim_full, boat_attitude_meta = _gps_sync_boat_attitude_for_project(
            project_id=project_id,
            config=config,
            num_frames=total_frames,
            fps=configured_fps,
        )
        boat_heel = np.asarray(boat_heel_full[start:end + 1], dtype=np.float64)
        boat_trim = np.asarray(boat_trim_full[start:end + 1], dtype=np.float64)
    except Exception as e:
        boat_attitude_meta = {
            "available": False,
            "file": None,
            "offset_s": 0.0,
            "points": 0,
            "synced_points": 0,
            "columns": {},
            "reason": str(e),
        }

    side_hint = np.zeros(frame_count, dtype=np.int8)  # -1=starboard, +1=port, 0=center/unknown
    finite_cy = np.isfinite(com_y)
    side_hint[finite_cy & (com_y > 0.02)] = 1
    side_hint[finite_cy & (com_y < -0.02)] = -1
    if np.any(side_hint == 0):
        finite_roll = np.isfinite(moment_roll)
        side_hint[(side_hint == 0) & finite_roll & (moment_roll > 0.0)] = 1
        side_hint[(side_hint == 0) & finite_roll & (moment_roll < 0.0)] = -1

    skel_x = _numeric_column(section_df, "skel23_x")
    skel_z = _numeric_column(section_df, "skel23_z")
    skel_valid = np.isfinite(skel_x) & np.isfinite(skel_z)

    payload = {
        "project_id": project_id,
        "project_name": str(config.get("name", project_id)),
        "start_frame": int(start),
        "end_frame": int(end),
        "frame_count": int(frame_count),
        "duration_s": float(duration_s),
        "fps": float(configured_fps),
        "fps_effective": float(fps_eff),
        "dt_s": float(dt_s),
        "athlete": {
            "weight_kg": float(config.get("athlete_weight", 75.0)),
        },
        "frame_idx": frame_idx.tolist(),
        "timestamp_ms": _nullable_float_list(timestamps),
        "time_s": _nullable_float_list(time_s),
        "trunk_angle": _nullable_float_list(trunk_angle),
        "moment_pitch": _nullable_float_list(moment_pitch),
        "moment_roll": _nullable_float_list(moment_roll),
        "com_x": _nullable_float_list(com_x),
        "com_y": _nullable_float_list(com_y),
        "com_z": _nullable_float_list(com_z),
        "rudder_angle": _nullable_float_list(rudder_angle),
        "boom_angle": _nullable_float_list(boom_angle),
        "boat_heel": _nullable_float_list(boat_heel),
        "boat_trim": _nullable_float_list(boat_trim),
        "boat_attitude": boat_attitude_meta,
        "side_hint": side_hint.astype(int).tolist(),
        "skeleton_valid": skel_valid.astype(bool).tolist(),
        "defaults": {
            "hike_low_deg": 25.0,
            "hike_mid_deg": 38.0,
            "hike_high_deg": 52.0,
            "side_deadband_m": 0.03,
            "iqr_factor": 1.5,
            "min_segment_s": 1.8,
            "bridge_gap_s": 0.45,
            "roll_high_nm": 300.0,
            "roll_extreme_nm": 450.0,
            "sustain_s": 8.0,
            "rolling_window_s": 20.0,
            "segment_bin_s": 2.0,
            "target_trunk_low_deg": 42.0,
            "target_trunk_high_deg": 58.0,
            "acwr_acute_s": 60.0,
            "acwr_chronic_s": 300.0,
            "lag_max_s": 6.0,
            "auto_max_s": 12.0,
        },
    }
    return payload, None


def _default_hull_side_profile_payload():
    profile = [
        {"s": -2.24, "z": 0.07},
        {"s": -1.82, "z": 0.08},
        {"s": -1.20, "z": 0.08},
        {"s": -0.72, "z": 0.07},
        {"s": -0.54, "z": 0.04},
        {"s": -0.70, "z": -0.05},
        {"s": -1.04, "z": -0.23},
        {"s": -1.56, "z": -0.33},
        {"s": -2.05, "z": -0.30},
        {"s": -2.24, "z": -0.20},
    ]
    return {
        "source": "fallback",
        "flip_x": True,
        "profile_xz": profile,
        "bounds": {
            "minS": -2.35,
            "maxS": -0.35,
            "minZ": -0.42,
            "maxZ": 0.92,
        },
    }


def _smooth_1d(values, window=9):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 2 or window <= 1:
        return arr
    w = int(max(1, min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)))
    if w <= 1:
        return arr
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return arr
    pad = w // 2
    kernel = np.full(w, 1.0 / w, dtype=np.float64)
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _build_hull_side_profile_payload():
    global hull_side_profile_cache

    fallback = _default_hull_side_profile_payload()
    stl_path = None
    try:
        if hasattr(processing, "_find_hull_stl_path"):
            stl_path = processing._find_hull_stl_path()
    except Exception:
        stl_path = None

    cache_key = None
    if stl_path is not None and Path(stl_path).exists():
        try:
            p = Path(stl_path)
            cache_key = f"{HULL_SIDE_PROFILE_CACHE_VERSION}|{p.resolve()}|{p.stat().st_mtime_ns}"
            if cache_key in hull_side_profile_cache:
                return hull_side_profile_cache[cache_key]
        except Exception:
            cache_key = None

    try:
        mesh = processing._load_hull_mesh_boat_frame() if hasattr(processing, "_load_hull_mesh_boat_frame") else None
        if mesh is None or getattr(mesh, "vertices", None) is None:
            return fallback

        verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(verts).all(axis=1)
        verts = verts[finite]
        if verts.shape[0] < 200:
            return fallback

        x = verts[:, 0]  # fore-aft
        y_up = verts[:, 1]  # vertical in transformed hull coordinates

        x_lo, x_hi = np.percentile(x, [1.0, 99.0])
        keep = (x >= x_lo) & (x <= x_hi) & np.isfinite(y_up)
        x = x[keep]
        y_up = y_up[keep]
        if x.size < 200:
            return fallback

        bins = 220
        edges = np.linspace(float(x_lo), float(x_hi), bins + 1, dtype=np.float64)
        idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
        x_mid = 0.5 * (edges[:-1] + edges[1:])

        top = np.full(bins, np.nan, dtype=np.float64)
        bot = np.full(bins, np.nan, dtype=np.float64)
        for bi in range(bins):
            m = idx == bi
            if np.count_nonzero(m) < 8:
                continue
            yv = y_up[m]
            top[bi] = float(np.percentile(yv, 98.0))
            bot[bi] = float(np.percentile(yv, 2.0))

        valid = np.isfinite(top) & np.isfinite(bot)
        if np.count_nonzero(valid) < 30:
            return fallback

        xv = x_mid[valid]
        topv = _smooth_1d(top[valid], window=9)
        botv = _smooth_1d(bot[valid], window=9)

        outline = np.concatenate(
            [
                np.column_stack([xv, topv]),
                np.column_stack([xv[::-1], botv[::-1]]),
            ],
            axis=0,
        )
        profile = [
            # Keep key name "z" for client compatibility; value here is side-view vertical from STL.
            {"s": float(p[0]), "z": float(p[1])}
            for p in outline
            if np.isfinite(p[0]) and np.isfinite(p[1])
        ]
        if len(profile) < 20:
            return fallback

        bounds = {
            "minS": float(np.min(xv) - 0.08),
            "maxS": float(np.max(xv) + 0.08),
            "minZ": float(np.min(botv) - 0.10),
            "maxZ": float(np.max(topv) + 0.85),
        }
        payload = {
            "source": "stl",
            "flip_x": True,
            "profile_xz": profile,
            "bounds": bounds,
        }
    except Exception:
        payload = fallback

    if cache_key:
        hull_side_profile_cache = {cache_key: payload}
    return payload


def _gps_sync_parse_float(value):
    try:
        x = float(value)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def _gps_sync_parse_epoch_from_text(text):
    s = str(text or "").strip()
    if not s:
        raise ValueError("Empty timestamp")

    if s.endswith("Z"):
        s_iso = s[:-1] + "+00:00"
    else:
        s_iso = s
    try:
        dt = datetime.fromisoformat(s_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            continue

    raise ValueError(f"Could not parse timestamp: {s}")


def _gps_sync_find_exiftool():
    exe = shutil.which("exiftool")
    if exe:
        return exe
    fallback = r"C:\Program Files\exiftool-13.51_64\exiftool.exe"
    if os.path.exists(fallback):
        return fallback
    return None


def _gps_sync_extract_gopro_gps_from_mp4(video_path):
    """Extract MP4 GPS samples via ExifTool as [{t,lat,lon,gpsdt,ts}, ...]."""
    global gps_sync_mp4_cache

    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")

    exiftool = _gps_sync_find_exiftool()
    if not exiftool:
        raise RuntimeError(
            "ExifTool not found. Install ExifTool and ensure `exiftool` is on PATH."
        )

    try:
        cache_key = f"{p.resolve()}|{p.stat().st_mtime_ns}"
    except Exception:
        cache_key = str(p)
    cached = gps_sync_mp4_cache.get(cache_key)
    if isinstance(cached, list) and cached:
        return cached

    fmt = "$SampleTime,$GPSLatitude,$GPSLongitude,$GPSDateTime"
    cmd = [exiftool, "-ee", "-n", "-p", fmt, str(p)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(stderr or "ExifTool failed while extracting MP4 GPS.")

    def parse_gps_datetime_to_epoch(gpsdt):
        s = str(gpsdt or "").strip()
        if not s:
            return np.nan
        candidates = [s]
        if s.endswith("Z"):
            candidates.insert(0, s[:-1])
        for candidate in candidates:
            for fmt_local in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(candidate, fmt_local)
                    # Treat GoPro GPSDateTime as UTC.
                    return float(dt.replace(tzinfo=timezone.utc).timestamp())
                except Exception:
                    continue
        return np.nan

    points = []
    for line in (proc.stdout or "").splitlines():
        parts = str(line).strip().split(",", 3)
        if len(parts) < 3:
            continue
        t = _gps_sync_parse_float(parts[0])
        lat = _gps_sync_parse_float(parts[1])
        lon = _gps_sync_parse_float(parts[2])
        if t is None or lat is None or lon is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        gpsdt = parts[3].strip() if len(parts) >= 4 else ""
        ts = parse_gps_datetime_to_epoch(gpsdt)
        if not np.isfinite(ts):
            continue
        points.append(
            {
                "t": float(t),
                "lat": float(lat),
                "lon": float(lon),
                "gpsdt": gpsdt,
                "ts": float(ts),
            }
        )

    points.sort(key=lambda d: d["t"])
    dedup = []
    last_t = None
    for pt in points:
        if last_t is None or abs(pt["t"] - last_t) > 1e-9:
            dedup.append(pt)
            last_t = pt["t"]

    if cache_key:
        gps_sync_mp4_cache = {cache_key: dedup}
    return dedup


def _gps_sync_pick_column(cols_map, names):
    for n in names:
        if n in cols_map:
            return cols_map[n]
    return None


def _gps_sync_parse_vakaros_csv(csv_path):
    """Parse Vakaros CSV into {'track': [...], 'columns': {...}}."""
    df_local = pd.read_csv(csv_path)
    if df_local is None or df_local.empty:
        return {"track": [], "columns": {}}

    cols_map = {str(c).strip().lower(): c for c in df_local.columns}
    c_ts = _gps_sync_pick_column(cols_map, ("timestamp", "time", "datetime", "date_time"))
    c_lat = _gps_sync_pick_column(cols_map, ("latitude", "lat"))
    c_lon = _gps_sync_pick_column(cols_map, ("longitude", "lon", "lng", "long"))
    if not c_ts or not c_lat or not c_lon:
        raise ValueError("Vakaros CSV must include timestamp, latitude, longitude columns.")

    c_heel = _gps_sync_pick_column(cols_map, ("heel", "roll", "roll_deg"))
    c_trim = _gps_sync_pick_column(cols_map, ("trim", "pitch", "pitch_deg"))
    c_sog = _gps_sync_pick_column(cols_map, ("sog_kts", "sog", "speed_kts"))
    c_cog = _gps_sync_pick_column(cols_map, ("cog", "course", "course_over_ground"))
    c_hdg = _gps_sync_pick_column(cols_map, ("hdg_true", "heading", "hdg"))

    track = []
    for _, row in df_local.iterrows():
        ts_raw = row.get(c_ts, None)
        try:
            ts = _gps_sync_parse_epoch_from_text(ts_raw)
        except Exception:
            continue
        lat = _gps_sync_parse_float(row.get(c_lat, None))
        lon = _gps_sync_parse_float(row.get(c_lon, None))
        if lat is None or lon is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue

        item = {"ts": float(ts), "lat": float(lat), "lon": float(lon)}
        if c_heel:
            item["heel"] = _gps_sync_parse_float(row.get(c_heel, None))
        if c_trim:
            item["trim"] = _gps_sync_parse_float(row.get(c_trim, None))
        if c_sog:
            item["sog_kts"] = _gps_sync_parse_float(row.get(c_sog, None))
        if c_cog:
            item["cog"] = _gps_sync_parse_float(row.get(c_cog, None))
        if c_hdg:
            item["hdg_true"] = _gps_sync_parse_float(row.get(c_hdg, None))
        track.append(item)

    track.sort(key=lambda d: d["ts"])
    dedup = []
    last_ts = None
    for pt in track:
        ts = pt["ts"]
        if last_ts is None or abs(ts - last_ts) > 1e-9:
            dedup.append(pt)
            last_ts = ts
        else:
            dedup[-1] = pt

    return {
        "track": dedup,
        "columns": {
            "heel": bool(c_heel),
            "trim": bool(c_trim),
            "sog_kts": bool(c_sog),
            "cog": bool(c_cog),
            "hdg_true": bool(c_hdg),
        },
    }


def _gps_sync_parse_pose_csv(pose_csv_path, fallback_fps=30.0):
    """Parse pose CSV and normalize to seconds in video-time."""
    df_local = pd.read_csv(pose_csv_path)
    if df_local is None or df_local.empty:
        return []

    cols_map = {str(c).strip().lower(): c for c in df_local.columns}
    c_frame = _gps_sync_pick_column(cols_map, ("frame_idx", "frame", "index"))
    c_ts = _gps_sync_pick_column(cols_map, ("timestamp_ms", "timestamp", "time_ms", "time"))
    c_trunk = _gps_sync_pick_column(cols_map, ("trunk_angle", "trunk_deg", "trunk"))
    c_mr = _gps_sync_pick_column(cols_map, ("moment_roll", "moment_y", "roll_moment"))
    c_mp = _gps_sync_pick_column(cols_map, ("moment_pitch", "moment_x", "pitch_moment"))

    nrows = int(len(df_local))
    if nrows <= 0:
        return []

    if c_frame and c_frame in df_local.columns:
        frame_arr = pd.to_numeric(df_local[c_frame], errors="coerce").to_numpy(dtype=np.float64)
    else:
        frame_arr = np.arange(nrows, dtype=np.float64)
    if frame_arr.size != nrows:
        frame_arr = np.arange(nrows, dtype=np.float64)
    invalid_frame = ~np.isfinite(frame_arr)
    frame_arr[invalid_frame] = np.arange(nrows, dtype=np.float64)[invalid_frame]

    trunk_arr = (
        pd.to_numeric(df_local[c_trunk], errors="coerce").to_numpy(dtype=np.float64)
        if c_trunk and c_trunk in df_local.columns
        else np.full(nrows, np.nan, dtype=np.float64)
    )
    mr_arr = (
        pd.to_numeric(df_local[c_mr], errors="coerce").to_numpy(dtype=np.float64)
        if c_mr and c_mr in df_local.columns
        else np.full(nrows, np.nan, dtype=np.float64)
    )
    mp_arr = (
        pd.to_numeric(df_local[c_mp], errors="coerce").to_numpy(dtype=np.float64)
        if c_mp and c_mp in df_local.columns
        else np.full(nrows, np.nan, dtype=np.float64)
    )

    if c_ts and c_ts in df_local.columns:
        ts_raw = pd.to_numeric(df_local[c_ts], errors="coerce").to_numpy(dtype=np.float64)
    else:
        ts_raw = np.full(nrows, np.nan, dtype=np.float64)

    valid_ts = ts_raw[np.isfinite(ts_raw)]
    t_arr = np.full(nrows, np.nan, dtype=np.float64)
    if valid_ts.size <= 0:
        fps_fallback = float(max(1.0, fallback_fps))
        t_arr = frame_arr / fps_fallback
    else:
        ts_max = float(np.nanmax(valid_ts))
        ts_sorted = np.sort(valid_ts)
        ts_d = np.diff(ts_sorted)
        ts_d = ts_d[ts_d > 1e-9]
        med_delta = float(np.median(ts_d)) if ts_d.size > 0 else np.nan

        if ts_max > 1e11:
            mode = "epoch_ms"
            base = float(np.nanmin(valid_ts))
            t_arr[np.isfinite(ts_raw)] = (ts_raw[np.isfinite(ts_raw)] - base) / 1000.0
        elif ts_max > 1e9:
            mode = "epoch_s"
            base = float(np.nanmin(valid_ts))
            t_arr[np.isfinite(ts_raw)] = (ts_raw[np.isfinite(ts_raw)] - base)
        elif np.isfinite(med_delta) and med_delta > 5.0:
            mode = "relative_ms"
            t_arr[np.isfinite(ts_raw)] = ts_raw[np.isfinite(ts_raw)] / 1000.0
        else:
            mode = "relative_s"
            t_arr[np.isfinite(ts_raw)] = ts_raw[np.isfinite(ts_raw)]

        # For any missing timestamps, backfill by frame/fps.
        if np.any(~np.isfinite(t_arr)):
            fps_fallback = float(max(1.0, fallback_fps))
            t_arr[~np.isfinite(t_arr)] = frame_arr[~np.isfinite(t_arr)] / fps_fallback

    out = []
    for i in range(nrows):
        t_val = t_arr[i]
        if not np.isfinite(t_val):
            continue
        out.append(
            {
                "frame": int(round(frame_arr[i])) if np.isfinite(frame_arr[i]) else int(i),
                "t": float(t_val),
                "trunk_angle": float(trunk_arr[i]) if np.isfinite(trunk_arr[i]) else None,
                "moment_roll": float(mr_arr[i]) if np.isfinite(mr_arr[i]) else None,
                "moment_pitch": float(mp_arr[i]) if np.isfinite(mp_arr[i]) else None,
            }
        )

    out.sort(key=lambda d: d["t"])
    return out


def _gps_sync_downsample_by_time(points, time_key="t", min_dt=0.1):
    if not points:
        return []
    out = []
    last_t = None
    for p in points:
        t = _gps_sync_parse_float(p.get(time_key))
        if t is None:
            continue
        if last_t is None or (t - last_t) >= float(min_dt):
            out.append(p)
            last_t = t
    if out and points[-1] is not out[-1]:
        t_src = _gps_sync_parse_float(points[-1].get(time_key))
        t_out = _gps_sync_parse_float(out[-1].get(time_key))
        if t_src is not None and t_out is not None and abs(t_src - t_out) > 1e-9:
            out.append(points[-1])
    return out


def _gps_sync_sync_pose_to_mp4(pose_rows, mp4_track):
    if not pose_rows or not mp4_track:
        return []

    mp4_t = np.asarray([_gps_sync_parse_float(p.get("t")) for p in mp4_track], dtype=np.float64)
    mp4_lat = np.asarray([_gps_sync_parse_float(p.get("lat")) for p in mp4_track], dtype=np.float64)
    mp4_lon = np.asarray([_gps_sync_parse_float(p.get("lon")) for p in mp4_track], dtype=np.float64)
    mp4_ts = np.asarray([_gps_sync_parse_float(p.get("ts")) for p in mp4_track], dtype=np.float64)
    ok = np.isfinite(mp4_t) & np.isfinite(mp4_lat) & np.isfinite(mp4_lon) & np.isfinite(mp4_ts)
    mp4_t = mp4_t[ok]
    mp4_lat = mp4_lat[ok]
    mp4_lon = mp4_lon[ok]
    mp4_ts = mp4_ts[ok]
    if mp4_t.size < 2:
        return []

    pose_t = np.asarray([_gps_sync_parse_float(p.get("t")) for p in pose_rows], dtype=np.float64)
    valid = np.isfinite(pose_t)
    if not np.any(valid):
        return []

    pose_t_valid = pose_t[valid]
    lat_i = np.interp(pose_t_valid, mp4_t, mp4_lat)
    lon_i = np.interp(pose_t_valid, mp4_t, mp4_lon)
    ts_i = np.interp(pose_t_valid, mp4_t, mp4_ts)

    out = []
    idx_valid = np.where(valid)[0]
    for j, i in enumerate(idx_valid.tolist()):
        r = pose_rows[i]
        out.append(
            {
                "frame": int(r.get("frame", i)),
                "t": float(pose_t_valid[j]),
                "ts": float(ts_i[j]),
                "lat": float(lat_i[j]),
                "lon": float(lon_i[j]),
                "trunk_angle": r.get("trunk_angle", None),
                "moment_roll": r.get("moment_roll", None),
                "moment_pitch": r.get("moment_pitch", None),
            }
        )
    return out


def _gps_sync_detect_pose_segments(
    pose_sync_rows,
    trunk_threshold_deg=20.0,
    min_duration_s=2.0,
    bridge_gap_s=0.35,
):
    if not pose_sync_rows:
        return []

    segments = []
    start_idx = None
    last_true_idx = None
    gap_start_idx = None
    seg_id = 1
    rows = pose_sync_rows

    def finalize_segment(s_idx, e_idx):
        nonlocal seg_id
        if s_idx is None or e_idx is None or e_idx < s_idx:
            return
        t0 = _gps_sync_parse_float(rows[s_idx].get("t"))
        t1 = _gps_sync_parse_float(rows[e_idx].get("t"))
        if t0 is None or t1 is None:
            return
        dur = t1 - t0
        if dur < float(min_duration_s):
            return

        sub = rows[s_idx : e_idx + 1]
        trunk_vals = [float(x["trunk_angle"]) for x in sub if _gps_sync_parse_float(x.get("trunk_angle")) is not None]
        if not trunk_vals:
            return
        mr_vals = [float(x["moment_roll"]) for x in sub if _gps_sync_parse_float(x.get("moment_roll")) is not None]
        side = "Unknown"
        if mr_vals:
            side = "Port" if (sum(mr_vals) / len(mr_vals)) >= 0.0 else "Starboard"

        path_src = [
            {
                "lat": float(x["lat"]),
                "lon": float(x["lon"]),
                "t": float(x["t"]),
                "frame": int(x.get("frame", 0)),
            }
            for x in sub
            if _gps_sync_parse_float(x.get("lat")) is not None and _gps_sync_parse_float(x.get("lon")) is not None
        ]
        path = _gps_sync_downsample_by_time(path_src, time_key="t", min_dt=0.2)
        if len(path) < 2:
            return

        segments.append(
            {
                "id": int(seg_id),
                "start_t": float(t0),
                "end_t": float(t1),
                "duration_s": float(dur),
                "start_frame": int(rows[s_idx].get("frame", 0)),
                "end_frame": int(rows[e_idx].get("frame", 0)),
                "mean_trunk_deg": float(sum(trunk_vals) / len(trunk_vals)),
                "peak_trunk_deg": float(max(trunk_vals)),
                "side": side,
                "path": path,
            }
        )
        seg_id += 1

    for i, r in enumerate(rows):
        trunk = _gps_sync_parse_float(r.get("trunk_angle"))
        is_hiking = trunk is not None and trunk >= float(trunk_threshold_deg)
        if is_hiking:
            if start_idx is None:
                start_idx = i
            last_true_idx = i
            gap_start_idx = None
            continue

        if start_idx is None:
            continue
        if gap_start_idx is None:
            gap_start_idx = i
            continue

        t_gap = _gps_sync_parse_float(rows[gap_start_idx].get("t"))
        t_now = _gps_sync_parse_float(rows[i].get("t"))
        if t_gap is None or t_now is None:
            continue
        if (t_now - t_gap) > float(bridge_gap_s):
            if last_true_idx is not None:
                finalize_segment(start_idx, last_true_idx)
            start_idx = None
            last_true_idx = None
            gap_start_idx = None

    if start_idx is not None and last_true_idx is not None:
        finalize_segment(start_idx, last_true_idx)

    return segments


def _gps_sync_haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = np.radians(float(lat1))
    p2 = np.radians(float(lat2))
    dp = np.radians(float(lat2) - float(lat1))
    dl = np.radians(float(lon2) - float(lon1))
    a = np.sin(dp * 0.5) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl * 0.5) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(max(1e-12, 1.0 - a)))
    return float(r * c)


def _gps_sync_distance_stats(mp4_track, vak_track):
    if not mp4_track or not vak_track:
        return {}
    mp4_sample = _gps_sync_downsample_by_time(mp4_track, time_key="t", min_dt=0.2)
    if not mp4_sample:
        return {}

    vak_ts = np.asarray([_gps_sync_parse_float(v.get("ts")) for v in vak_track], dtype=np.float64)
    vak_lat = np.asarray([_gps_sync_parse_float(v.get("lat")) for v in vak_track], dtype=np.float64)
    vak_lon = np.asarray([_gps_sync_parse_float(v.get("lon")) for v in vak_track], dtype=np.float64)
    ok = np.isfinite(vak_ts) & np.isfinite(vak_lat) & np.isfinite(vak_lon)
    vak_ts = vak_ts[ok]
    vak_lat = vak_lat[ok]
    vak_lon = vak_lon[ok]
    if vak_ts.size < 2:
        return {}

    dists = []
    for p in mp4_sample:
        ts = _gps_sync_parse_float(p.get("ts"))
        lat = _gps_sync_parse_float(p.get("lat"))
        lon = _gps_sync_parse_float(p.get("lon"))
        if ts is None or lat is None or lon is None:
            continue
        if ts < vak_ts[0] or ts > vak_ts[-1]:
            continue
        vlat = float(np.interp(ts, vak_ts, vak_lat))
        vlon = float(np.interp(ts, vak_ts, vak_lon))
        dists.append(_gps_sync_haversine_m(lat, lon, vlat, vlon))

    if not dists:
        return {}
    dists = sorted(float(x) for x in dists if np.isfinite(x))
    if not dists:
        return {}
    n = len(dists)
    p95 = dists[int(round(0.95 * (n - 1)))]
    return {
        "count": int(n),
        "mean_m": float(sum(dists) / n),
        "p95_m": float(p95),
        "max_m": float(dists[-1]),
    }


def _gps_sync_shift_track_time(track, time_key="ts", shift_s=0.0):
    """Return shallow-copied track with `time_key` shifted by `-shift_s`."""
    shift = float(_gps_sync_parse_float(shift_s) or 0.0)
    if abs(shift) <= 1e-12 or not track:
        return list(track or [])
    out = []
    for p in track:
        q = dict(p)
        t = _gps_sync_parse_float(q.get(time_key))
        if t is not None:
            q[time_key] = float(t - shift)
        out.append(q)
    return out


def _gps_sync_interp_series(src_t, src_v, query_t):
    """Interpolate src_v(src_t) onto query_t with NaN outside bounds."""
    out = np.full(query_t.shape, np.nan, dtype=np.float64)
    if src_t.size < 2 or src_v.size < 2 or query_t.size <= 0:
        return out

    m = np.isfinite(src_t) & np.isfinite(src_v)
    if np.count_nonzero(m) < 2:
        return out

    t = np.asarray(src_t[m], dtype=np.float64)
    v = np.asarray(src_v[m], dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    if t.size < 2:
        return out

    # Keep last sample when duplicate timestamps exist.
    unique_t = [float(t[0])]
    unique_v = [float(v[0])]
    for i in range(1, t.size):
        if abs(float(t[i]) - unique_t[-1]) <= 1e-9:
            unique_t[-1] = float(t[i])
            unique_v[-1] = float(v[i])
        else:
            unique_t.append(float(t[i]))
            unique_v.append(float(v[i]))
    if len(unique_t) < 2:
        return out

    t = np.asarray(unique_t, dtype=np.float64)
    v = np.asarray(unique_v, dtype=np.float64)
    qmask = np.isfinite(query_t) & (query_t >= t[0]) & (query_t <= t[-1])
    if np.any(qmask):
        out[qmask] = np.interp(query_t[qmask], t, v)
    return out


def _gps_sync_boat_attitude_for_project(project_id, config, num_frames, fps):
    """Sync Vakaros heel/trim onto per-frame video time for viewer metrics."""
    heel = np.full(int(max(0, num_frames)), np.nan, dtype=np.float64)
    trim = np.full(int(max(0, num_frames)), np.nan, dtype=np.float64)
    meta = {
        "available": False,
        "file": None,
        "offset_s": float(_gps_sync_parse_float((config or {}).get("vakaros_offset_s")) or 0.0),
        "points": 0,
        "synced_points": 0,
        "columns": {},
        "reason": None,
    }
    if num_frames <= 0:
        meta["reason"] = "No frames"
        return heel, trim, meta

    vak_file = str((config or {}).get("vakaros_file", "") or "").strip()
    if not vak_file:
        meta["reason"] = "No Vakaros file configured"
        return heel, trim, meta

    project_path = processing.get_project_path(project_id)
    vak_path = project_path / Path(vak_file).name
    if not vak_path.exists():
        meta["file"] = Path(vak_file).name
        meta["reason"] = "Configured Vakaros file missing"
        return heel, trim, meta
    meta["file"] = vak_path.name

    video_path = _resolve_project_video_path(project_path, config or {})
    if video_path is None:
        meta["reason"] = "Project video missing"
        return heel, trim, meta

    try:
        mp4_track = _gps_sync_extract_gopro_gps_from_mp4(video_path)
        vak_data = _gps_sync_parse_vakaros_csv(str(vak_path))
    except Exception as e:
        meta["reason"] = str(e)
        return heel, trim, meta

    vak_track = vak_data.get("track", []) or []
    meta["columns"] = vak_data.get("columns", {}) or {}
    meta["points"] = int(len(vak_track))
    if len(mp4_track) < 2:
        meta["reason"] = "No usable MP4 GPS"
        return heel, trim, meta
    if len(vak_track) < 2:
        meta["reason"] = "No usable Vakaros rows"
        return heel, trim, meta

    mp4_t = np.asarray([_gps_sync_parse_float(p.get("t")) for p in mp4_track], dtype=np.float64)
    mp4_ts = np.asarray([_gps_sync_parse_float(p.get("ts")) for p in mp4_track], dtype=np.float64)
    m_mp4 = np.isfinite(mp4_t) & np.isfinite(mp4_ts)
    if np.count_nonzero(m_mp4) < 2:
        meta["reason"] = "Insufficient MP4 timestamps"
        return heel, trim, meta
    mp4_t = mp4_t[m_mp4]
    mp4_ts = mp4_ts[m_mp4]
    order = np.argsort(mp4_t)
    mp4_t = mp4_t[order]
    mp4_ts = mp4_ts[order]

    fps_eff = float(_gps_sync_parse_float(fps) or 30.0)
    fps_eff = max(1.0, fps_eff)
    frame_t = np.arange(int(num_frames), dtype=np.float64) / fps_eff

    frame_ts = np.full(frame_t.shape, np.nan, dtype=np.float64)
    in_video_range = (frame_t >= mp4_t[0]) & (frame_t <= mp4_t[-1])
    if np.any(in_video_range):
        frame_ts[in_video_range] = np.interp(frame_t[in_video_range], mp4_t, mp4_ts)
    query_ts = frame_ts + float(meta["offset_s"])

    vak_ts = np.asarray([_gps_sync_parse_float(v.get("ts")) for v in vak_track], dtype=np.float64)
    vak_heel = np.asarray([_gps_sync_parse_float(v.get("heel")) for v in vak_track], dtype=np.float64)
    vak_trim = np.asarray([_gps_sync_parse_float(v.get("trim")) for v in vak_track], dtype=np.float64)

    heel = _gps_sync_interp_series(vak_ts, vak_heel, query_ts)
    trim = _gps_sync_interp_series(vak_ts, vak_trim, query_ts)
    valid = np.isfinite(heel) | np.isfinite(trim)
    meta["synced_points"] = int(np.count_nonzero(valid))
    meta["available"] = bool(meta["synced_points"] > 0)
    if not meta["available"]:
        meta["reason"] = "No synced heel/trim overlap"
    return heel, trim, meta


@app.route('/')
def index():
    """Projects list page."""
    return render_template('projects.html')


@app.route('/static/boat.stl')
def serve_boat_stl():
    """Serve the boat STL file."""
    from pathlib import Path
    import glob
    
    # Find Hull.stl file
    stl_files = glob.glob('Hull.stl')
    if stl_files:
        return send_from_directory('.', stl_files[0])
    
    # Look in parent directory
    parent_files = glob.glob('../Hull.stl')
    if parent_files:
        return send_from_directory('..', Path(parent_files[0]).name)
    
    return "STL file not found", 404


@app.route('/static/rudder.stl')
def serve_rudder_stl():
    """Serve the rudder STL file."""
    from pathlib import Path
    import glob
    
    # Find Rudder.stl file
    stl_files = glob.glob('Rudder.stl')
    if stl_files:
        return send_from_directory('.', stl_files[0])
    
    # Look in parent directory
    parent_files = glob.glob('../Rudder.stl')
    if parent_files:
        return send_from_directory('..', Path(parent_files[0]).name)
    
    return "Rudder STL file not found", 404


@app.route('/static/boom_mast.stl')
def serve_boom_mast_stl():
    """Serve the boom and mast STL file."""
    from pathlib import Path
    import glob
    
    # Find Boom&Mast.stl file
    stl_files = glob.glob('Boom&Mast.stl')
    if stl_files:
        return send_from_directory('.', stl_files[0])
    
    # Also try without the ampersand
    stl_files = glob.glob('Boom*.stl')
    if stl_files:
        return send_from_directory('.', stl_files[0])
    
    # Look in parent directory
    parent_files = glob.glob('../Boom&Mast.stl')
    if parent_files:
        return send_from_directory('..', Path(parent_files[0]).name)
    
    return "Boom & Mast STL file not found", 404


@app.route('/video/direct')
def serve_direct_video():
    """Serve the original video file directly."""
    video_path = Path(__file__).parent / VIDEO_FILE
    if not video_path.exists():
        return "Video file not found", 404
    return send_from_directory(str(video_path.parent), video_path.name)


def init_data():
    global df, camera_pos, R_wc, K_undist, W, H, fps_estimate

    if df is not None:
        return

    csv_path = Path(__file__).parent / POSE_CSV_PATH
    if not csv_path.exists():
        df = None
        return

    df = pd.read_csv(csv_path)

    camera_pos, R_wc = default_camera_pose_and_rotation()

    calib_path = Path(__file__).parent / FISHEYE_CALIB_NPZ
    if calib_path.exists():
        try:
            K_undist, (W, H) = load_fisheye_undistorted_intrinsics(str(calib_path), balance=0.0)
        except Exception:
            K_undist, W, H = None, None, None
    else:
        K_undist, W, H = None, None, None

    fps_estimate = 30.0
    if "timestamp_ms" in df.columns:
        ts = df["timestamp_ms"].dropna().values
        if len(ts) >= 2:
            deltas = np.diff(ts)
            deltas = deltas[deltas > 0]
            if len(deltas) > 0:
                median_dt = float(np.median(deltas))
                if median_dt > 0:
                    fps_estimate = 1000.0 / median_dt


@app.route('/api/frames/metadata')
def frames_metadata():
    init_data()
    if df is None:
        return jsonify({"error": "pose.csv not found"}), 404

    timestamps = None
    if "timestamp_ms" in df.columns:
        timestamps = [
            (float(v) if pd.notna(v) else None)
            for v in df["timestamp_ms"].tolist()
        ]

    return jsonify({
        "count": int(len(df)),
        "fps": float(fps_estimate),
        "timestamps": timestamps,
    })


@app.route('/api/frames/<int:frame_idx>')
def get_frame(frame_idx: int):
    init_data()
    if df is None:
        return jsonify({"error": "pose.csv not found"}), 404

    if frame_idx < 0 or frame_idx >= len(df):
        return jsonify({"error": "frame out of range"}), 404

    row = df.iloc[frame_idx]

    if frame_idx in placed_cache:
        placed = placed_cache[frame_idx]
    else:
        placed = None
        if K_undist is not None and W is not None and H is not None:
            placed = compute_placed_skeleton(
                row,
                K_undist=K_undist,
                W=W,
                H=H,
                camera_pos=camera_pos,
                R_wc=R_wc,
                z_plane_lm24=Z_PLANE_LM24,
                z_plane_lm28=Z_PLANE_LM28,
            )
        placed_cache[frame_idx] = placed

    skeleton = None
    if placed is not None:
        skeleton = {int(k): [float(v[0]), float(v[1]), float(v[2])] for k, v in placed.items() if v is not None}

    pose_norm = extract_pose_norm(row)

    timestamp = None
    if "timestamp_ms" in row and pd.notna(row["timestamp_ms"]):
        timestamp = float(row["timestamp_ms"])

    return jsonify({
        "frame_idx": int(frame_idx),
        "timestamp": timestamp,
        "skeleton": skeleton,
        "pose_norm": pose_norm,
    })


def filter_value(value, key):
    """Filter unreasonable values based on thresholds."""
    if value is None or not np.isfinite(value):
        return None
    thresholds = FILTER_THRESHOLDS.get(key, {})
    min_val = thresholds.get('min', float('-inf'))
    max_val = thresholds.get('max', float('inf'))
    if value < min_val or value > max_val:
        return None
    return float(value)


def compute_trunk_angle(skeleton):
    """Compute trunk angle from skeleton dict."""
    if skeleton is None:
        return None
    
    # Convert skeleton dict to numpy arrays
    skel_np = {}
    for k, v in skeleton.items():
        if v is not None:
            skel_np[int(k)] = np.array(v)
    
    # Need shoulders (11,12) and hips (23,24)
    if not all(k in skel_np for k in (11, 12, 23, 24)):
        return None
    
    shoulder_mid = 0.5 * (skel_np[11] + skel_np[12])
    hip_mid = 0.5 * (skel_np[23] + skel_np[24])
    
    v = shoulder_mid - hip_mid
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    
    v_unit = v / n
    z_unit = np.array([0.0, 0.0, 1.0])
    
    cos_theta = float(np.clip(np.dot(v_unit, z_unit), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    return angle_deg


def compute_moments(skeleton):
    """Compute moments (Pitch, Roll) from skeleton dict."""
    if skeleton is None:
        return None, None
    
    skel_np = {}
    for k, v in skeleton.items():
        if v is not None:
            skel_np[int(k)] = np.array(v)
    
    com = compute_center_of_mass(skel_np)
    if com is None:
        return None, None
    
    dx = float(com[0] - BOAT_COM)
    dy = float(com[1])
    moment_pitch = ATHLETE_MASS_KG * GRAVITY * dx
    moment_roll = ATHLETE_MASS_KG * GRAVITY * dy
    
    return moment_pitch, moment_roll


def compute_com_position(skeleton):
    """Compute center of mass position from skeleton dict."""
    if skeleton is None:
        return None, None, None
    
    skel_np = {}
    for k, v in skeleton.items():
        if v is not None:
            skel_np[int(k)] = np.array(v)
    
    com = compute_center_of_mass(skel_np)
    if com is None:
        return None, None, None
    
    return float(com[0]), float(com[1]), float(com[2])


def extract_pose_norm(row, landmark_count=33):
    """Extract MediaPipe normalized pose landmarks from a CSV row.

    Returns a list of length landmark_count with [x, y] or None per landmark.
    """
    pose = [None] * landmark_count
    for i in range(landmark_count):
        x = row.get(f"lm{i}_norm_x")
        y = row.get(f"lm{i}_norm_y")
        if pd.notna(x) and pd.notna(y):
            pose[i] = [float(x), float(y)]
    return pose


def extract_landmark_confidence(row, landmark_count=33):
    """Extract per-landmark visibility confidence in [0,1] from a row-like object."""
    conf = {}
    for i in range(int(max(1, landmark_count))):
        key = f"lm{i}_visibility"
        try:
            raw = row.get(key)
        except Exception:
            continue
        try:
            v = float(raw)
        except Exception:
            continue
        if np.isfinite(v):
            conf[i] = float(np.clip(v, 0.0, 1.0))
    return conf or None


@app.route('/api/graph-data')
def get_graph_data():
    """Get all computed metrics for graphs."""
    init_data()
    if df is None:
        return jsonify({"error": "pose.csv not found"}), 404
    
    # Precompute all skeletons with placement smoothing if not cached
    needs_recompute = any(i not in placed_cache for i in range(len(df)))
    
    if needs_recompute:
        filt_cfg = normalize_skeleton_filter_params({})
        smoother = SkeletonPlacementKalman(
            fps=fps_estimate,
            process_noise_acc=float(filt_cfg["process_noise_acc"]),
            measurement_noise=float(filt_cfg["measurement_noise"]),
            use_landmark_confidence=bool(filt_cfg["use_landmark_confidence"]),
            min_landmark_confidence=float(filt_cfg["min_landmark_confidence"]),
            confidence_floor=float(filt_cfg["confidence_floor"]),
            confidence_power=float(filt_cfg["confidence_power"]),
            max_confidence_noise_scale=float(filt_cfg["max_confidence_noise_scale"]),
            gate_sigma=float(filt_cfg["gate_sigma"]),
            max_consecutive_misses=int(filt_cfg["max_consecutive_misses"]),
            initial_velocity_std=float(filt_cfg["initial_velocity_std"]),
            velocity_decay=float(filt_cfg["velocity_decay"]),
            max_speed=float(filt_cfg["max_speed"]),
            max_measurement_jump=float(filt_cfg["max_measurement_jump"]),
            reacquire_frames=int(filt_cfg["reacquire_frames"]),
            reacquire_max_jump=float(filt_cfg["reacquire_max_jump"]),
        )
        for i in range(len(df)):
            row = df.iloc[i]
            placed = None
            if K_undist is not None and W is not None and H is not None:
                raw_placed = compute_placed_skeleton(
                    row,
                    K_undist=K_undist,
                    W=W,
                    H=H,
                    camera_pos=camera_pos,
                    R_wc=R_wc,
                    z_plane_lm24=Z_PLANE_LM24,
                    z_plane_lm28=Z_PLANE_LM28,
                )
                lm_conf = extract_landmark_confidence(row)
                placed = smoother.smooth(raw_placed, landmark_confidence=lm_conf)
            placed_cache[i] = placed
    
    # Compute all metrics
    timestamps_list = []
    trunk_angles = []
    moment_pitch_list = []
    moment_roll_list = []
    com_x_list = []
    com_y_list = []
    com_z_list = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        placed = placed_cache.get(i)
        
        # Timestamp
        ts = None
        if "timestamp_ms" in row and pd.notna(row["timestamp_ms"]):
            ts = float(row["timestamp_ms"])
        timestamps_list.append(ts)
        
        # Trunk angle
        trunk = compute_trunk_angle(placed)
        trunk_angles.append(filter_value(trunk, 'trunk_angle'))
        
        # Moments
        mp, mr = compute_moments(placed)
        moment_pitch_list.append(filter_value(mp, 'moment_x')) # Reuse threshold key
        moment_roll_list.append(filter_value(mr, 'moment_y')) # Reuse threshold key
        
        # COM position
        cx, cy, cz = compute_com_position(placed)
        com_x_list.append(filter_value(cx, 'com_x'))
        com_y_list.append(filter_value(cy, 'com_y'))
        com_z_list.append(filter_value(cz, 'com_z'))
    
    return jsonify({
        "count": len(df),
        "timestamps": timestamps_list,
        "trunk_angle": trunk_angles,
        "moment_pitch": moment_pitch_list,
        "moment_roll": moment_roll_list,
        "com_x": com_x_list,
        "com_y": com_y_list,
        "com_z": com_z_list,
    })


# =============================================================================
# Project Management Routes
# =============================================================================

@app.route('/setup')
def setup_page():
    """Setup page for creating new projects."""
    return render_template('setup.html')


@app.route('/viewer')
def viewer_page():
    """Viewer page - can load a specific project."""
    return render_template('viewer.html')


@app.route('/gps-sync')
def gps_sync_page():
    """In-app GPS/pose synchronization viewer."""
    initial_project_id = str(request.args.get("project", "") or "").strip()
    return render_template("gps_sync.html", initial_project_id=initial_project_id)


@app.route('/api/projects')
def list_projects():
    """List all projects."""
    projects = processing.list_projects()
    return jsonify({"projects": projects})


@app.route('/api/projects/<project_id>/vakaros/status')
def project_vakaros_status(project_id):
    """Return configured Vakaros sync status for a project."""
    config = processing.get_project_config(project_id)
    if not config:
        return jsonify({"error": "Project not found"}), 404

    project_path = processing.get_project_path(project_id)
    vak_file = str(config.get("vakaros_file", "") or "").strip()
    offset_s = float(_gps_sync_parse_float(config.get("vakaros_offset_s")) or 0.0)
    payload = {
        "configured": bool(vak_file),
        "available": False,
        "file": Path(vak_file).name if vak_file else None,
        "offset_s": offset_s,
        "columns": {},
        "points": 0,
        "error": None,
    }
    if not vak_file:
        return jsonify(payload)

    vak_path = project_path / Path(vak_file).name
    if not vak_path.exists():
        payload["error"] = "Configured Vakaros file not found on disk."
        return jsonify(payload)

    try:
        vak_data = _gps_sync_parse_vakaros_csv(str(vak_path))
        payload["columns"] = vak_data.get("columns", {}) or {}
        payload["points"] = int(len(vak_data.get("track", []) or []))
        payload["available"] = payload["points"] >= 2
        if not payload["available"]:
            payload["error"] = "No usable Vakaros rows found."
    except Exception as e:
        payload["error"] = str(e)
    return jsonify(payload)


@app.route('/api/projects/<project_id>/vakaros/upload', methods=['POST'])
def project_vakaros_upload(project_id):
    """Upload and attach a Vakaros CSV to a project for viewer boat-attitude sync."""
    config = processing.get_project_config(project_id)
    if not config:
        return jsonify({"error": "Project not found"}), 404

    csv_file = request.files.get("vakaros_csv")
    if csv_file is None or not csv_file.filename:
        return jsonify({"error": "vakaros_csv file is required"}), 400

    offset_s = float(_gps_sync_parse_float(request.form.get("offset_s")) or 0.0)
    offset_s = float(np.clip(offset_s, -600.0, 600.0))

    project_path = processing.get_project_path(project_id)
    if not project_path.exists():
        return jsonify({"error": "Project directory not found"}), 404

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            csv_file.save(tmp.name)
            tmp_path = tmp.name

        vak_data = _gps_sync_parse_vakaros_csv(tmp_path)
        vak_track = vak_data.get("track", []) or []
        if len(vak_track) < 2:
            return jsonify({"error": "No usable Vakaros rows found in uploaded CSV."}), 400

        dest_name = "vakaros.csv"
        dest_path = project_path / dest_name
        shutil.copy2(tmp_path, dest_path)

        updated = _update_project_config(
            project_id,
            updates={
                "vakaros_file": dest_name,
                "vakaros_offset_s": float(offset_s),
            },
        )
        if not updated:
            return jsonify({"error": "Could not update project config"}), 500

        invalidate_project_caches(project_id)
        return jsonify(
            {
                "success": True,
                "file": dest_name,
                "offset_s": float(offset_s),
                "points": int(len(vak_track)),
                "columns": vak_data.get("columns", {}) or {},
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        safe_unlink(tmp_path)


@app.route('/api/projects/<project_id>/vakaros/offset', methods=['POST'])
def project_vakaros_offset(project_id):
    """Update Vakaros time offset (seconds) for an already attached CSV."""
    config = processing.get_project_config(project_id)
    if not config:
        return jsonify({"error": "Project not found"}), 404

    vak_file = str(config.get("vakaros_file", "") or "").strip()
    if not vak_file:
        return jsonify({"error": "No Vakaros CSV attached to this project."}), 400

    offset_s = float(_gps_sync_parse_float(request.form.get("offset_s")) or 0.0)
    offset_s = float(np.clip(offset_s, -600.0, 600.0))
    updated = _update_project_config(project_id, updates={"vakaros_offset_s": float(offset_s)})
    if not updated:
        return jsonify({"error": "Could not update project config"}), 500

    invalidate_project_caches(project_id)
    return jsonify({"success": True, "offset_s": float(offset_s)})


@app.route('/api/projects/<project_id>/vakaros', methods=['DELETE'])
def project_vakaros_clear(project_id):
    """Detach Vakaros CSV from project config (and delete local copy if present)."""
    config = processing.get_project_config(project_id)
    if not config:
        return jsonify({"error": "Project not found"}), 404

    vak_file = str(config.get("vakaros_file", "") or "").strip()
    if vak_file:
        project_path = processing.get_project_path(project_id)
        vak_path = project_path / Path(vak_file).name
        safe_unlink(str(vak_path))

    updated = _update_project_config(
        project_id,
        remove_keys=("vakaros_file", "vakaros_offset_s"),
    )
    if not updated:
        return jsonify({"error": "Could not update project config"}), 500

    invalidate_project_caches(project_id)
    return jsonify({"success": True})


@app.route('/api/projects/<project_id>/gps-sync')
def project_gps_sync_data(project_id):
    """Build synchronized MP4 GPS + configured Vakaros + pose payload for viewer map."""
    config = processing.get_project_config(project_id)
    if not config:
        return jsonify({"error": "Project not found"}), 404

    project_path = processing.get_project_path(project_id)
    if not project_path.exists():
        return jsonify({"error": "Project directory not found"}), 404

    video_path = _resolve_project_video_path(project_path, config)
    if video_path is None:
        return jsonify({"error": "Project video not found"}), 404

    vak_file_cfg = str(config.get("vakaros_file", "") or "").strip()
    if not vak_file_cfg:
        return jsonify({"error": "No Vakaros CSV attached to project. Upload one in Viewer first."}), 400
    vak_path = project_path / Path(vak_file_cfg).name
    if not vak_path.exists():
        return jsonify({"error": f"Configured Vakaros CSV not found: {vak_path.name}"}), 400

    trunk_threshold_deg = float(request.args.get("trunk_threshold_deg", 20.0) or 20.0)
    trunk_threshold_deg = float(np.clip(trunk_threshold_deg, 0.0, 180.0))
    min_segment_s = float(request.args.get("min_segment_s", 2.0) or 2.0)
    min_segment_s = float(np.clip(min_segment_s, 0.1, 120.0))
    bridge_gap_s = float(request.args.get("bridge_gap_s", 0.35) or 0.35)
    bridge_gap_s = float(np.clip(bridge_gap_s, 0.0, 10.0))
    vak_offset_s = float(_gps_sync_parse_float(config.get("vakaros_offset_s")) or 0.0)

    try:
        mp4_track = _gps_sync_extract_gopro_gps_from_mp4(video_path)
        if len(mp4_track) < 2:
            return jsonify({"error": "No usable GPS samples found in project MP4."}), 400

        vak_data = _gps_sync_parse_vakaros_csv(str(vak_path))
        vak_track = vak_data.get("track", [])
        if len(vak_track) < 2:
            return jsonify({"error": "No usable track rows found in configured Vakaros CSV."}), 400

        pose_rows = []
        pose_path = project_path / "pose.csv"
        if pose_path.exists():
            fps_cfg = float(_gps_sync_parse_float(config.get("fps")) or 30.0)
            pose_rows = _gps_sync_parse_pose_csv(pose_path, fallback_fps=fps_cfg)

        pose_sync = _gps_sync_sync_pose_to_mp4(pose_rows, mp4_track)
        segments = _gps_sync_detect_pose_segments(
            pose_sync,
            trunk_threshold_deg=trunk_threshold_deg,
            min_duration_s=min_segment_s,
            bridge_gap_s=bridge_gap_s,
        )

        mp4_view = _gps_sync_downsample_by_time(mp4_track, time_key="t", min_dt=0.1)
        pose_view = _gps_sync_downsample_by_time(pose_sync, time_key="t", min_dt=(1.0 / 15.0))
        vak_view = _gps_sync_downsample_by_time(vak_track, time_key="ts", min_dt=0.1)
        vak_for_stats = _gps_sync_shift_track_time(vak_track, time_key="ts", shift_s=vak_offset_s)
        sync_stats = _gps_sync_distance_stats(mp4_track, vak_for_stats)

        return jsonify(
            {
                "project_id": project_id,
                "video_url": url_for("serve_project_video", project_id=project_id),
                "mp4_track": mp4_view,
                "vakaros": {
                    "track": vak_view,
                    "columns": vak_data.get("columns", {}),
                },
                "pose_track": pose_view,
                "segments": segments,
                "meta": {
                    "video_file": video_path.name,
                    "vakaros_file": vak_path.name,
                    "pose_file": ("pose.csv" if pose_path.exists() else None),
                    "mp4_points": int(len(mp4_track)),
                    "vakaros_points": int(len(vak_track)),
                    "pose_points": int(len(pose_view)),
                    "segments": int(len(segments)),
                    "trunk_threshold_deg": trunk_threshold_deg,
                    "min_segment_s": min_segment_s,
                    "bridge_gap_s": bridge_gap_s,
                    "vakaros_offset_s": float(vak_offset_s),
                    "sync_distance_stats": sync_stats,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/gps-sync/data', methods=['POST'])
def gps_sync_data():
    """Build synchronized MP4 GPS + Vakaros + pose payload for the web GPS view."""
    project_id = str(request.form.get("project_id", "") or "").strip()
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    config = processing.get_project_config(project_id)
    if not config:
        return jsonify({"error": "Project not found"}), 404

    project_path = processing.get_project_path(project_id)
    if not project_path.exists():
        return jsonify({"error": "Project directory not found"}), 404

    video_path = _resolve_project_video_path(project_path, config)
    if video_path is None:
        return jsonify({"error": "Project video not found"}), 404

    vak_file = request.files.get("vakaros_csv")
    if vak_file is None or not vak_file.filename:
        return jsonify({"error": "Vakaros CSV file is required"}), 400

    trunk_threshold_deg = float(request.form.get("trunk_threshold_deg", 20.0) or 20.0)
    trunk_threshold_deg = float(np.clip(trunk_threshold_deg, 0.0, 180.0))
    min_segment_s = float(request.form.get("min_segment_s", 2.0) or 2.0)
    min_segment_s = float(np.clip(min_segment_s, 0.1, 120.0))
    bridge_gap_s = float(request.form.get("bridge_gap_s", 0.35) or 0.35)
    bridge_gap_s = float(np.clip(bridge_gap_s, 0.0, 10.0))

    temp_paths = []
    vak_temp_path = None
    pose_temp_path = None
    try:
        # Save uploaded Vakaros CSV to a temporary file for robust parsing.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            vak_file.save(tmp.name)
            vak_temp_path = tmp.name
            temp_paths.append(vak_temp_path)

        pose_upload = request.files.get("pose_csv")
        pose_path = project_path / "pose.csv"
        if pose_upload is not None and pose_upload.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                pose_upload.save(tmp.name)
                pose_temp_path = tmp.name
                temp_paths.append(pose_temp_path)
            pose_path = Path(pose_temp_path)

        fps_cfg = float(config.get("fps", 30.0) or 30.0)
        mp4_track = _gps_sync_extract_gopro_gps_from_mp4(video_path)
        if len(mp4_track) < 2:
            return jsonify({"error": "No usable GPS samples found in project MP4."}), 400

        vak_data = _gps_sync_parse_vakaros_csv(vak_temp_path)
        vak_track = vak_data.get("track", [])
        if len(vak_track) < 2:
            return jsonify({"error": "No usable track rows found in Vakaros CSV."}), 400

        pose_rows = []
        if pose_path.exists():
            pose_rows = _gps_sync_parse_pose_csv(pose_path, fallback_fps=fps_cfg)

        pose_sync = _gps_sync_sync_pose_to_mp4(pose_rows, mp4_track)
        segments = _gps_sync_detect_pose_segments(
            pose_sync,
            trunk_threshold_deg=trunk_threshold_deg,
            min_duration_s=min_segment_s,
            bridge_gap_s=bridge_gap_s,
        )

        mp4_view = _gps_sync_downsample_by_time(mp4_track, time_key="t", min_dt=0.1)
        pose_view = _gps_sync_downsample_by_time(pose_sync, time_key="t", min_dt=(1.0 / 15.0))
        vak_view = _gps_sync_downsample_by_time(vak_track, time_key="ts", min_dt=0.1)
        sync_stats = _gps_sync_distance_stats(mp4_track, vak_track)

        return jsonify(
            {
                "project_id": project_id,
                "video_url": url_for("serve_project_video", project_id=project_id),
                "mp4_track": mp4_view,
                "vakaros": {
                    "track": vak_view,
                    "columns": vak_data.get("columns", {}),
                },
                "pose_track": pose_view,
                "segments": segments,
                "meta": {
                    "video_file": video_path.name,
                    "vakaros_file": secure_filename(vak_file.filename or "uploaded.csv"),
                    "pose_file": (
                        secure_filename(pose_upload.filename)
                        if pose_upload is not None and pose_upload.filename
                        else ("pose.csv" if (project_path / "pose.csv").exists() else None)
                    ),
                    "mp4_points": int(len(mp4_track)),
                    "vakaros_points": int(len(vak_track)),
                    "pose_points": int(len(pose_view)),
                    "segments": int(len(segments)),
                    "trunk_threshold_deg": trunk_threshold_deg,
                    "min_segment_s": min_segment_s,
                    "bridge_gap_s": bridge_gap_s,
                    "sync_distance_stats": sync_stats,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in temp_paths:
            safe_unlink(p)


@app.route('/api/projects/create', methods=['POST'])
def create_project():
    """Create a new project and start processing."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if not video_file.filename:
        return jsonify({"error": "No video file selected"}), 400
    
    project_name = request.form.get('project_name', 'untitled')
    
    # Save video to temp file
    video_ext = Path(video_file.filename).suffix or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
        video_file.save(tmp.name)
        temp_video_path = tmp.name
    
    # Save calibration if provided
    temp_calib_path = None
    if 'calibration' in request.files:
        calib_file = request.files['calibration']
        if calib_file.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
                calib_file.save(tmp.name)
                temp_calib_path = tmp.name
    
    try:
        # Parse rudder configuration
        rudder_enabled = request.form.get('rudder_enabled', 'false').lower() == 'true'

        # Parse contact-fitting tuning (JSON string)
        contact_params = None
        contact_json = request.form.get('contact_params')
        if contact_json:
            try:
                contact_params = json.loads(contact_json)
            except (json.JSONDecodeError, TypeError):
                pass

        skeleton_filter = None
        skeleton_filter_json = request.form.get('skeleton_filter')
        if skeleton_filter_json:
            try:
                skeleton_filter = json.loads(skeleton_filter_json)
            except (json.JSONDecodeError, TypeError):
                skeleton_filter = None

        seated_x_stabilizer = None
        seated_x_stab_json = request.form.get('seated_x_stabilizer')
        if seated_x_stab_json:
            try:
                seated_x_stabilizer = json.loads(seated_x_stab_json)
            except (json.JSONDecodeError, TypeError):
                seated_x_stabilizer = None

        lateral_y_stabilizer = None
        lateral_y_stab_json = request.form.get('lateral_y_stabilizer')
        if lateral_y_stab_json:
            try:
                lateral_y_stabilizer = json.loads(lateral_y_stab_json)
            except (json.JSONDecodeError, TypeError):
                lateral_y_stabilizer = None

        camera_R_wc = None
        camera_rwc_json = request.form.get('camera_R_wc')
        if camera_rwc_json:
            try:
                parsed = json.loads(camera_rwc_json)
                parsed_R = parse_camera_rotation_matrix(parsed)
                if parsed_R is not None:
                    camera_R_wc = parsed_R.tolist()
            except (json.JSONDecodeError, TypeError):
                pass

        mediapipe_workers = None
        mediapipe_workers_raw = request.form.get('mediapipe_workers')
        if mediapipe_workers_raw not in (None, ""):
            try:
                mediapipe_workers = int(mediapipe_workers_raw)
            except Exception:
                mediapipe_workers = None
        
        project_id = processing.create_project(
            name=project_name,
            video_path=temp_video_path,
            calibration_path=temp_calib_path,
            athlete_weight=float(request.form.get('athlete_weight', 75.0)),
            ankle_height=float(request.form.get('ankle_height', 0.01)),
            hip_height=float(request.form.get('hip_height', 0.10)),
            camera_x=float(request.form.get('camera_x', -3.374)),
            camera_y=float(request.form.get('camera_y', 0.0)),
            camera_z=float(request.form.get('camera_z', 0.5)),
            boat_com=float(request.form.get('boat_com', -1.114)),
            pose_model=request.form.get('pose_model', 'full'),
            lower_landmark=request.form.get('lower_landmark', 'ankle'),
            rudder_enabled=rudder_enabled,
            camera_pitch_deg=float(request.form.get('camera_pitch_deg', 8.0)),
            camera_yaw_deg=float(request.form.get('camera_yaw_deg', 0.0)),
            camera_roll_deg=float(request.form.get('camera_roll_deg', 0.0)),
            camera_R_wc=camera_R_wc,
            skeleton_filter=skeleton_filter,
            seated_x_stabilizer=seated_x_stabilizer,
            lateral_y_stabilizer=lateral_y_stabilizer,
            contact_params=contact_params,
            mediapipe_workers=mediapipe_workers,
        )
        
        return jsonify({"project_id": project_id})
        
    finally:
        # Clean up temp files
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        if temp_calib_path and os.path.exists(temp_calib_path):
            os.unlink(temp_calib_path)


@app.route('/api/projects/<project_id>')
def get_project(project_id):
    """Get project configuration."""
    config = processing.get_project_config(project_id)
    if not config:
        return jsonify({"error": "Project not found"}), 404
    return jsonify(config)


@app.route('/api/projects/<project_id>/camera-view')
def get_project_camera_view(project_id):
    """Get camera view parameters (position and rotation matrix) for 3D viewer."""
    data = get_project_data(project_id)
    if not data:
        return jsonify({"error": "Project not found"}), 404
    
    camera_pos = data["camera_pos"]
    R_wc = data["R_wc"]
    pose_angles = camera_pose_angles_from_rwc(R_wc)
    
    R_z = np.array([
    [np.cos(np.pi/2),  np.sin(np.pi/2), 0],
    [-np.sin(np.pi/2), np.cos(np.pi/2), 0],
    [0, 0, 1],
    ])
    R_wc = R_z @ R_wc
    
    
    # Return camera position and rotation matrix as a flat list (row-major)
    return jsonify({
        "camera_position": [float(camera_pos[1]), -float(camera_pos[0])+1.5, float(camera_pos[2])+0.2],
        "R_wc": R_wc.flatten().tolist(),  # 3x3 matrix as 9 element list (row-major)
        "camera_pose_deg": pose_angles,
    })


@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project."""
    if processing.delete_project(project_id):
        return jsonify({"success": True})
    return jsonify({"error": "Project not found"}), 404


@app.route('/api/projects/<project_id>/status')
def get_project_status(project_id):
    """Get processing status for a project."""
    status = processing.get_processing_status(project_id)
    return jsonify(status)


# =============================================================================
# Rudder Detection Preview/Tuning API
# =============================================================================

# Temporary storage for uploaded preview video
preview_video_cache = {}

# Cached PilotNetDetector for preview (avoid reloading ONNX models on every frame)
_preview_detector = None
_pnp_yolo_model = None

def _get_preview_detector():
    global _preview_detector
    if _preview_detector is None:
        _preview_detector = PilotNetDetector()
    return _preview_detector


def _read_cached_preview_frame(temp_video_id, frame_number, *, require_calibration=False):
    """Read a frame from the cached preview video and optionally apply cached undistortion."""
    if not temp_video_id:
        raise ValueError("temp_video_id is required")
    if temp_video_id not in preview_video_cache:
        raise ValueError("Unknown temp_video_id. Preview a frame first.")

    video_path = preview_video_cache[temp_video_id]
    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found. Please re-upload the video.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video contains no frames")

    frame_idx = max(0, min(int(frame_number), total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Cannot read frame")

    calib_key = f"skel_calib_{temp_video_id}"
    cached = preview_video_cache.get(calib_key)
    if cached and "maps" in cached:
        frame = cv2.remap(frame, cached["maps"][0], cached["maps"][1], cv2.INTER_LINEAR)
    elif require_calibration:
        raise RuntimeError("No calibration cached for this video. Load a skeleton frame with calibration first.")

    return frame, frame_idx, total_frames, frame_width, frame_height


def _get_pnp_yolo_model():
    """Lazy-load the YOLO pose model used for PnP keypoint auto-detection."""
    global _pnp_yolo_model
    if _pnp_yolo_model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics") from exc

        if not PNP_YOLO_MODEL_PATH.exists():
            raise RuntimeError(f"YOLO model file not found: {PNP_YOLO_MODEL_PATH.name}")
        _pnp_yolo_model = YOLO(str(PNP_YOLO_MODEL_PATH))
    return _pnp_yolo_model


def _bind_pnp_keypoints_geometry(points_xy):
    """Bind 9 detected keypoints to semantic labels using geometric rules.

    Rules:
    - frontdeck = highest point in image (smallest y)
    - remaining points are split left/right in image (left side treated as port)
    - per side: farthest from frontdeck = back; remaining sorted top/mid/low by image y
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = len(PNP_YOLO_KEYPOINT_LABELS)
    if pts.shape[0] < n:
        raise ValueError(f"Expected at least {n} keypoints, got {pts.shape[0]}")
    pts = pts[:n]
    if not np.all(np.isfinite(pts)):
        raise ValueError("Keypoint coordinates contain non-finite values")

    front_idx = int(np.argmin(pts[:, 1]))
    front_pt = pts[front_idx]

    remaining = [i for i in range(n) if i != front_idx]
    rem_pts = pts[remaining]
    x_order = np.argsort(rem_pts[:, 0])
    left_side = [remaining[i] for i in x_order[:4]]
    right_side = [remaining[i] for i in x_order[4:]]

    def split_side(side_indices):
        side_pts = pts[side_indices]
        dist = np.linalg.norm(side_pts - front_pt[None, :], axis=1)
        back_local = int(np.argmax(dist))
        back_idx = side_indices[back_local]
        rail_idx = [idx for idx in side_indices if idx != back_idx]
        rail_pts = pts[rail_idx]
        rail_order = [rail_idx[i] for i in np.argsort(rail_pts[:, 1])]
        return {
            "top": rail_order[0],
            "mid": rail_order[1],
            "low": rail_order[2],
            "back": back_idx,
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


def _encode_pnp_overlay_image(frame_bgr, overlay_points):
    """Draw labeled keypoints on frame and return base64-encoded JPEG."""
    import base64

    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    base = max(1, int(round(min(h, w) / 700.0)))
    radius = 3 + base
    font_scale = 0.38 + 0.08 * base
    text_thickness = 1 + (base // 2)

    for item in overlay_points:
        xy = item.get("xy")
        if not xy or len(xy) != 2:
            continue
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        ix = int(round(x))
        iy = int(round(y))
        conf = item.get("confidence")
        usable = bool(item.get("usable", False))
        color = (70, 220, 70) if usable else (0, 180, 255)

        cv2.circle(vis, (ix, iy), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (ix, iy), radius, color, -1, cv2.LINE_AA)

        label = str(item.get("label", "pt"))
        if conf is not None and np.isfinite(conf):
            text = f"{label} {float(conf):.2f}"
        else:
            text = label

        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        tx = min(max(ix + radius + 5, 2), max(w - text_size[0] - 2, 2))
        ty = min(max(iy - radius - 5, text_size[1] + 2), max(h - 2, text_size[1] + 2))

        bg_tl = (tx - 2, ty - text_size[1] - 2)
        bg_br = (tx + text_size[0] + 2, ty + baseline + 2)
        cv2.rectangle(vis, bg_tl, bg_br, (0, 0, 0), -1)
        cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)

    vis_small = cv2.resize(vis, None, fx=0.5, fy=0.5)
    ok, buffer = cv2.imencode(".jpg", vis_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


@app.route('/api/camera/pnp/autodetect', methods=['POST'])
def camera_pnp_autodetect():
    """Auto-detect PnP image points with YOLOv8 pose model and bind to hull points."""
    data = request.get_json(force=True, silent=True) or {}
    temp_video_id = data.get("temp_video_id")
    frame_number = data.get("frame_number", 0)

    try:
        frame, frame_idx, total_frames, frame_width, frame_height = _read_cached_preview_frame(
            temp_video_id,
            frame_number,
            require_calibration=True,
        )
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to load frame: {exc}"}), 500

    try:
        min_kpt_conf = float(data.get("min_kpt_conf", PNP_YOLO_MIN_CONF))
    except (TypeError, ValueError):
        min_kpt_conf = PNP_YOLO_MIN_CONF
    min_kpt_conf = max(0.0, min(1.0, min_kpt_conf))

    try:
        model = _get_pnp_yolo_model()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    try:
        results = model.predict(
            source=frame,
            verbose=False,
            conf=0.05,
            iou=0.6,
            max_det=4,
        )
    except Exception as exc:
        return jsonify({"error": f"YOLO inference failed: {exc}"}), 500

    if not results:
        return jsonify({"error": "YOLO returned no results"}), 400

    result = results[0]
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return jsonify({"error": "No YOLO keypoints detected in this frame"}), 404

    xy_all = result.keypoints.xy.detach().cpu().numpy()
    if xy_all.ndim != 3 or xy_all.shape[2] != 2:
        return jsonify({"error": "Unexpected YOLO keypoint output shape"}), 500

    conf_tensor = result.keypoints.conf
    if conf_tensor is not None:
        conf_all = conf_tensor.detach().cpu().numpy()
    else:
        conf_all = np.ones((xy_all.shape[0], xy_all.shape[1]), dtype=np.float64)

    n_labels = len(PNP_YOLO_KEYPOINT_LABELS)
    if xy_all.shape[1] < n_labels:
        return jsonify({
            "error": f"Model returned {xy_all.shape[1]} keypoints, expected {n_labels}",
        }), 500

    box_conf = np.ones((xy_all.shape[0],), dtype=np.float64)
    if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) == xy_all.shape[0]:
        box_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
    mean_kpt_conf = np.mean(np.nan_to_num(conf_all[:, :n_labels], nan=0.0), axis=1)
    det_scores = box_conf * mean_kpt_conf
    det_idx = int(np.argmax(det_scores))

    raw_pts = xy_all[det_idx, :n_labels, :].astype(np.float64)
    raw_conf = conf_all[det_idx, :n_labels].astype(np.float64)

    try:
        label_to_index = _bind_pnp_keypoints_geometry(raw_pts)
        binding_method = "geometry_frontdeck_top_left_is_port"
    except Exception:
        # Fallback to raw model index order if geometric binding fails.
        label_to_index = {label: i for i, label in enumerate(PNP_YOLO_KEYPOINT_LABELS)}
        binding_method = "model_index_fallback"

    pairs = []
    overlay_points = []
    ready_pairs = 0
    for i, label in enumerate(PNP_YOLO_KEYPOINT_LABELS, start=1):
        src_idx = int(label_to_index[label])
        uv = raw_pts[src_idx]
        conf_val = raw_conf[src_idx]
        conf_out = float(conf_val) if np.isfinite(conf_val) else None
        image_point = None
        if np.all(np.isfinite(uv)) and (conf_out is None or conf_out >= min_kpt_conf):
            image_point = [float(uv[0]), float(uv[1])]
            ready_pairs += 1

        if np.all(np.isfinite(uv)):
            overlay_points.append({
                "label": label,
                "xy": [float(uv[0]), float(uv[1])],
                "confidence": conf_out,
                "usable": image_point is not None,
            })

        pairs.append({
            "id": i,
            "label": label,
            "source_index": src_idx,
            "confidence": conf_out,
            "image_point": image_point,
            "object_point": PNP_YOLO_OBJECT_POINTS[label],
        })

    annotated_image = None
    try:
        annotated_image = _encode_pnp_overlay_image(frame, overlay_points)
    except Exception:
        annotated_image = None

    if ready_pairs < 4:
        return jsonify({
            "error": f"Detected only {ready_pairs} usable keypoints (threshold={min_kpt_conf:.2f}); need at least 4.",
            "pairs": pairs,
            "binding_method": binding_method,
            "annotated_image": annotated_image,
        }), 400

    return jsonify({
        "success": True,
        "temp_video_id": temp_video_id,
        "frame_number": int(frame_idx),
        "total_frames": int(total_frames),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "detected_objects": int(xy_all.shape[0]),
        "selected_detection_index": int(det_idx),
        "selected_detection_score": float(det_scores[det_idx]),
        "binding_method": binding_method,
        "min_kpt_conf": float(min_kpt_conf),
        "ready_pairs": int(ready_pairs),
        "pairs": pairs,
        "annotated_image": annotated_image,
    })


@app.route('/api/camera/pnp/solve', methods=['POST'])
def camera_pnp_solve():
    """Solve camera pose from 2D-3D correspondences with OpenCV PnP."""
    data = request.get_json(force=True, silent=True) or {}

    temp_video_id = data.get('temp_video_id')
    if not temp_video_id:
        return jsonify({"error": "temp_video_id is required"}), 400

    calib_key = f"skel_calib_{temp_video_id}"
    if calib_key not in preview_video_cache:
        return jsonify({"error": "No calibration cached for this video. Load a skeleton frame with calibration first."}), 400

    cached = preview_video_cache[calib_key]
    K_undist = cached.get('K_undist')
    if K_undist is None:
        return jsonify({"error": "Undistorted intrinsics not available in cache"}), 400

    pairs = data.get('pairs', [])
    if not isinstance(pairs, list):
        return jsonify({"error": "pairs must be a list"}), 400

    object_points = []
    image_points = []
    pair_ids = []
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            continue
        obj = pair.get('object_point')
        img = pair.get('image_point')
        if obj is None or img is None:
            continue
        try:
            obj_pt = np.asarray(obj, dtype=np.float64).reshape(3)
            img_pt = np.asarray(img, dtype=np.float64).reshape(2)
        except Exception:
            continue
        if not np.all(np.isfinite(obj_pt)) or not np.all(np.isfinite(img_pt)):
            continue
        object_points.append(obj_pt)
        image_points.append(img_pt)
        pair_ids.append(pair.get('id', i))

    if len(object_points) < 4:
        return jsonify({
            "error": f"At least 4 valid point pairs are required (got {len(object_points)} valid of {len(pairs)} submitted)"
        }), 400

    obj = np.asarray(object_points, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64)
    K = np.asarray(K_undist, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((4, 1), dtype=np.float64)

    def _wrap_deg(deg):
        val = float(deg)
        return ((val + 180.0) % 360.0) - 180.0

    def _correct_pose_from_rt(rvec_local, tvec_local):
        R_cw_local, _ = cv2.Rodrigues(rvec_local)
        R_wc_raw_local = R_cw_local.T
        camera_pos_raw_local = (-R_wc_raw_local @ tvec_local).reshape(3)
        angles_raw_local = camera_pose_angles_from_rwc(R_wc_raw_local)
        _, R_wc_corr_local = default_camera_pose_and_rotation(
            pitch=angles_raw_local["pitch_deg"],
            yaw=angles_raw_local["yaw_deg"],
            roll=-(angles_raw_local["roll_deg"] + 90.0),
        )
        camera_pos_corr_local = np.array(
            [camera_pos_raw_local[0], camera_pos_raw_local[2], camera_pos_raw_local[1]],
            dtype=np.float64,
        )
        angles_corr_local = camera_pose_angles_from_rwc(R_wc_corr_local)
        angles_corr_local["yaw_deg"] = _wrap_deg(angles_corr_local["yaw_deg"])
        angles_corr_local["roll_deg"] = _wrap_deg(angles_corr_local["roll_deg"])
        return R_wc_corr_local, camera_pos_corr_local, angles_corr_local

    def _range_violation(val, lo, hi):
        v = float(val)
        if v < lo:
            return (lo - v) / max(1e-6, abs(hi - lo))
        if v > hi:
            return (v - hi) / max(1e-6, abs(hi - lo))
        return 0.0

    def _abs_violation(val, lim):
        return max(abs(float(val)) - lim, 0.0) / max(1e-6, lim)

    def _candidate_from_solution(method, rvec_local, tvec_local, inlier_idx_local):
        R_wc_local, camera_pos_local, angles_local = _correct_pose_from_rt(rvec_local, tvec_local)
        proj_local, _ = cv2.projectPoints(obj, rvec_local, tvec_local, K, dist)
        proj_local = proj_local.reshape(-1, 2)
        residuals_local = np.linalg.norm(proj_local - img, axis=1)
        mean_all = float(np.mean(residuals_local)) if len(residuals_local) else float("inf")
        med_all = float(np.median(residuals_local)) if len(residuals_local) else float("inf")
        max_all = float(np.max(residuals_local)) if len(residuals_local) else float("inf")
        inlier_idx_local = np.asarray(inlier_idx_local, dtype=np.int32).reshape(-1)
        inlier_idx_local = inlier_idx_local[(inlier_idx_local >= 0) & (inlier_idx_local < len(obj))]
        if len(inlier_idx_local) == 0:
            inlier_idx_local = np.arange(len(obj), dtype=np.int32)
        inlier_err = float(np.mean(residuals_local[inlier_idx_local])) if len(inlier_idx_local) else mean_all

        x, y, z = [float(v) for v in camera_pos_local.reshape(3)]
        pitch = float(angles_local.get("pitch_deg", float("nan")))
        yaw = float(angles_local.get("yaw_deg", float("nan")))
        roll = float(angles_local.get("roll_deg", float("nan")))

        # Soft plausibility score (lower is better), tuned to reject unrealistic local minima.
        violation = 0.0
        violation += _range_violation(pitch, 10.0, 23.0)
        violation += _abs_violation(yaw, 5.0)
        violation += _abs_violation(roll, 5.0)
        violation += _range_violation(x, -3.4, -3.1)
        violation += _abs_violation(y, 0.1)
        violation += _range_violation(z, 0.5, 0.8)
        violation += max(inlier_err - 30.0, 0.0) / 30.0

        inlier_ratio = float(len(inlier_idx_local)) / float(len(obj))
        low_inlier_penalty = max(0.0, 0.70 - inlier_ratio) * 25.0
        # Blend geometric fit with plausibility. Median residual uses all correspondences.
        score = med_all + 25.0 * violation + low_inlier_penalty

        return {
            "method": method,
            "rvec": rvec_local,
            "tvec": tvec_local,
            "R_wc": R_wc_local,
            "camera_pos": camera_pos_local,
            "angles": angles_local,
            "proj": proj_local,
            "residuals": residuals_local,
            "inlier_idx": inlier_idx_local,
            "num_inliers": int(len(inlier_idx_local)),
            "num_pairs": int(len(obj)),
            "mean_error_all": mean_all,
            "median_error_all": med_all,
            "max_error_all": max_all,
            "inlier_error": inlier_err,
            "score": float(score),
        }

    candidates = []

    # Candidate A: RANSAC + optional LM refine on inliers.
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
            candidates.append(_candidate_from_solution("ransac_epnp", rvec_r, tvec_r, inlier_idx_r))

    # Candidate B: iterative solve on all points.
    try:
        ok_i, rvec_i, tvec_i = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        ok_i = False
    if ok_i:
        try:
            rvec_i, tvec_i = cv2.solvePnPRefineLM(obj, img, K, dist, rvec_i, tvec_i)
        except Exception:
            pass
        candidates.append(_candidate_from_solution("iterative_all", rvec_i, tvec_i, np.arange(len(obj), dtype=np.int32)))

    if not candidates:
        return jsonify({"error": "PnP solve failed (no valid solver candidate)"}), 400

    best = min(candidates, key=lambda c: c["score"])
    R_wc = best["R_wc"]
    camera_pos = best["camera_pos"]
    angles = best["angles"]
    proj = best["proj"]
    residuals = best["residuals"]
    inlier_idx = best["inlier_idx"]
    mean_error = best["mean_error_all"]
    inlier_error = best["inlier_error"]

    pair_results = []
    inlier_set = set(int(i) for i in inlier_idx.tolist())
    for i in range(len(obj)):
        pair_results.append({
            "id": pair_ids[i],
            "object_point": obj[i].tolist(),
            "image_point": img[i].tolist(),
            "projected_point": proj[i].tolist(),
            "reprojection_error_px": float(residuals[i]),
            "is_inlier": i in inlier_set,
        })

    return jsonify({
        "success": True,
        "num_pairs": len(obj),
        "num_inliers": int(len(inlier_idx)),
        "solve_method": best["method"],
        "solve_score": float(best["score"]),
        "camera_position": camera_pos.tolist(),
        "R_wc": R_wc.tolist(),
        "camera_pose_deg": angles,
        "mean_reprojection_error_px": mean_error,
        "inlier_reprojection_error_px": inlier_error,
        "median_reprojection_error_px": float(best["median_error_all"]),
        "max_reprojection_error_px": float(best["max_error_all"]),
        "candidate_diagnostics": [
            {
                "method": c["method"],
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
        "pairs": pair_results,
    })


@app.route('/api/rudder/preview', methods=['POST'])
def rudder_preview():
    """Preview PilotNet rudder angle prediction on a single frame.
    
    Accepts either:
    - A video file with frame_number to extract a specific frame
    - An existing temp_video_id from a previous upload
    
    Returns base64-encoded images of:
    - Full frame with both ROI rectangles
    - ROI crops from both models
    - Fused prediction overlay
    """
    import base64
    
    video_path = None
    
    # Check if we have a temp video ID
    temp_video_id = request.form.get('temp_video_id')
    if temp_video_id and temp_video_id in preview_video_cache:
        video_path = preview_video_cache[temp_video_id]
    elif 'video' in request.files:
        video_file = request.files['video']
        if video_file.filename:
            # Save to temp file and cache
            video_ext = Path(video_file.filename).suffix or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
                video_file.save(tmp.name)
                temp_video_id = str(uuid.uuid4())[:8]
                preview_video_cache[temp_video_id] = tmp.name
                video_path = tmp.name
    else:
        return jsonify({"error": "No video provided"}), 400
    
    if video_path is None or not os.path.exists(video_path):
        return jsonify({"error": "Video file not found. Please re-upload the video."}), 404
    
    # Extract frame
    frame_number = int(request.form.get('frame_number', 0))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_number, total_frames - 1))
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return jsonify({"error": "Cannot read frame"}), 400
    
    # Apply undistortion from cache if available
    calib_cache_key = f"rudder_calib_{temp_video_id}"
    if calib_cache_key in preview_video_cache:
        cached = preview_video_cache[calib_cache_key]
        frame = cv2.remap(frame, cached['maps'][0], cached['maps'][1], cv2.INTER_LINEAR)
    elif 'calibration' in request.files:
        calib_file = request.files['calibration']
        if calib_file.filename:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.npz')
            tmp_path = tmp.name
            tmp.close()
            calib_file.save(tmp_path)
            try:
                with np.load(tmp_path) as cal:
                    K = cal["K"].copy()
                    D = cal["D"].copy()
                img_size = (frame_width, frame_height)
                R = np.eye(3)
                K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, img_size, R, balance=0.0, new_size=img_size
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, R, K_new, img_size, cv2.CV_16SC2
                )
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                preview_video_cache[calib_cache_key] = {
                    'maps': (map1, map2),
                }
            except Exception:
                pass
            finally:
                safe_unlink(tmp_path)
    
    # Run PilotNet prediction (dual model, cached detector)
    try:
        detector = _get_preview_detector()
        result = detector.detect(frame, return_debug=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"PilotNet prediction failed: {e}"}), 500
    
    # Encode images as base64
    def encode_image(img):
        if img is None:
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    # Draw both ROI rectangles on full frame
    frame_with_roi = frame.copy()
    # Model A ROI (green)
    tA, bA, lA, rA = detector._model_a.roi_crop
    cv2.rectangle(frame_with_roi, (lA, tA), (rA, bA), (0, 255, 0), 2)
    cv2.putText(frame_with_roi, "A", (lA + 5, tA + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    # Model B ROI (cyan)
    tB, bB, lB, rB = detector._model_b.roi_crop
    cv2.rectangle(frame_with_roi, (lB, tB), (rB, bB), (255, 255, 0), 2)
    cv2.putText(frame_with_roi, "B", (lB + 5, tB + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    # Fused angle label
    if result.get("success"):
        label = f"Fused: {result['angle_deg']:.1f} deg  (A:{result['angle_a']:.1f}  B:{result['angle_b']:.1f})"
        cv2.putText(frame_with_roi, label, (lA + 5, tA - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Scale down full frame for preview
    scale = 0.25
    frame_small = cv2.resize(frame_with_roi, None, fx=scale, fy=scale)
    
    response = {
        "temp_video_id": temp_video_id,
        "total_frames": total_frames,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_number": frame_number,
        "angle_deg": result.get('angle_deg'),
        "angle_a": result.get('angle_a'),
        "angle_b": result.get('angle_b'),
        "angle_2d": result.get('angle_2d'),
        "success": result.get('success', False),
        "images": {
            "full_frame": encode_image(frame_small),
            "roi_a": encode_image(result.get('debug_roi_a')),
            "roi_b": encode_image(result.get('debug_roi_b')),
            "overlay": encode_image(result.get('debug_overlay')),
        }
    }
    
    return jsonify(response)


@app.route('/api/rudder/preview/cleanup', methods=['POST'])
def rudder_preview_cleanup():
    """Clean up temporary preview video files."""
    temp_video_id = request.json.get('temp_video_id') if request.is_json else request.form.get('temp_video_id')
    
    if temp_video_id and temp_video_id in preview_video_cache:
        video_path = preview_video_cache.pop(temp_video_id)
        preview_video_cache.pop(f"skel_calib_{temp_video_id}", None)
        stale_cache_ids = [cid for cid, entry in skeleton_tuning_cache.items() if entry.get("temp_video_id") == temp_video_id]
        for cid in stale_cache_ids:
            skeleton_tuning_cache.pop(cid, None)
        if os.path.exists(video_path):
            os.unlink(video_path)
        return jsonify({"success": True})
    
    return jsonify({"error": "Not found"}), 404


@app.route('/api/rudder/tuning', methods=['POST'])
def rudder_tuning():
    """Process a snippet of frames with custom Kalman parameters.

    Accepts JSON:
        temp_video_id : str
        start_frame   : int
        end_frame     : int
        process_noise : float  (default 5.0)
        measurement_noise : float  (default 2.0)
        gate_sigma    : float  (default 5.0)

    Returns per‑frame angle arrays for charting.
    """
    data = request.get_json(force=True)
    temp_video_id = data.get('temp_video_id')
    if not temp_video_id or temp_video_id not in preview_video_cache:
        return jsonify({"error": "No cached video – preview a frame first"}), 400

    video_path = preview_video_cache[temp_video_id]
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    start_frame = int(data.get('start_frame', 0))
    end_frame = int(data.get('end_frame', start_frame + 150))
    process_noise = float(data.get('process_noise', 5.0))
    measurement_noise = float(data.get('measurement_noise', 2.0))
    gate_sigma = float(data.get('gate_sigma', 5.0))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = min(end_frame, total - 1)
    start_frame = max(0, min(start_frame, end_frame))

    # Undistortion maps (reuse from preview cache if available)
    map1 = map2 = None
    calib_key = f"rudder_calib_{temp_video_id}"
    if calib_key in preview_video_cache:
        cached = preview_video_cache[calib_key]
        map1, map2 = cached['maps']

    # Fresh detector with custom Kalman parameters
    from rudder_nn import PilotNetDetector as _PND
    detector = _PND(
        fps=fps,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        gate_sigma=gate_sigma,
    )

    frames_list = []
    angle_a_list = []
    angle_b_list = []
    fused_list = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for f in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        result = detector.detect(frame)
        frames_list.append(f)
        angle_a_list.append(None if not np.isfinite(result.get('angle_a', float('nan'))) else round(result['angle_a'], 2))
        angle_b_list.append(None if not np.isfinite(result.get('angle_b', float('nan'))) else round(result['angle_b'], 2))
        fused_list.append(None if not result.get('success') else round(result['angle_deg'], 2))

    cap.release()

    return jsonify({
        "frames": frames_list,
        "angle_a": angle_a_list,
        "angle_b": angle_b_list,
        "fused": fused_list,
        "total_frames": total,
        "fps": fps,
        "params": {
            "process_noise": process_noise,
            "measurement_noise": measurement_noise,
            "gate_sigma": gate_sigma,
        }
    })


def _serialize_skeleton_3d(placed):
    if placed is None:
        return None
    out = {}
    for k, v in placed.items():
        if v is None:
            continue
        try:
            a = np.asarray(v, dtype=np.float64).reshape(3)
        except Exception:
            continue
        if not np.all(np.isfinite(a)):
            continue
        out[int(k)] = [float(a[0]), float(a[1]), float(a[2])]
    return out or None


def _serialize_landmark_confidence(row_like, landmark_count=33):
    """Serialize landmark visibility confidences as a fixed-size list."""
    out = [None] * int(max(1, landmark_count))
    if row_like is None:
        return out
    for i in range(len(out)):
        key = f"lm{i}_visibility"
        try:
            raw = row_like.get(key)
        except Exception:
            continue
        try:
            v = float(raw)
        except Exception:
            continue
        if np.isfinite(v):
            out[i] = float(np.clip(v, 0.0, 1.0))
    return out


def _deserialize_landmark_confidence(serialized_conf):
    if serialized_conf is None:
        return None
    out = {}
    try:
        items = list(serialized_conf)
    except Exception:
        return None
    for i, raw in enumerate(items):
        if raw is None:
            continue
        try:
            v = float(raw)
        except Exception:
            continue
        if np.isfinite(v):
            out[int(i)] = float(np.clip(v, 0.0, 1.0))
    return out or None


def _deserialize_skeleton_3d(placed):
    if not isinstance(placed, dict):
        return None
    out = {}
    for k, v in placed.items():
        try:
            idx = int(k)
            arr = np.asarray(v, dtype=np.float64).reshape(3)
        except Exception:
            continue
        if not np.all(np.isfinite(arr)):
            continue
        out[idx] = arr
    return out or None


def _build_row_dict_from_landmarks(world_lm, norm_lm):
    row = {}
    for i in range(33):
        row[f"lm{i}_world_x"] = world_lm[i].x
        row[f"lm{i}_world_y"] = world_lm[i].y
        row[f"lm{i}_world_z"] = world_lm[i].z
        row[f"lm{i}_norm_x"] = norm_lm[i].x
        row[f"lm{i}_norm_y"] = norm_lm[i].y
        row[f"lm{i}_norm_z"] = norm_lm[i].z
        row[f"lm{i}_visibility"] = norm_lm[i].visibility
    return row


def _compute_skeleton_series(serialized_skeletons):
    pos_x = []
    pos_y = []
    pos_z = []
    com_x = []
    com_y = []
    com_z = []
    trunk_angle = []

    for ser in serialized_skeletons:
        skel = _deserialize_skeleton_3d(ser)
        if skel is None:
            pos_x.append(None)
            pos_y.append(None)
            pos_z.append(None)
            com_x.append(None)
            com_y.append(None)
            com_z.append(None)
            trunk_angle.append(None)
            continue

        hips = []
        if skel.get(23) is not None:
            hips.append(np.asarray(skel[23], dtype=np.float64))
        if skel.get(24) is not None:
            hips.append(np.asarray(skel[24], dtype=np.float64))
        if hips:
            hip_mid = np.mean(np.asarray(hips, dtype=np.float64), axis=0)
            pos_x.append(float(hip_mid[0]))
            pos_y.append(float(hip_mid[1]))
            pos_z.append(float(hip_mid[2]))
        else:
            pos_x.append(None)
            pos_y.append(None)
            pos_z.append(None)

        skel_np = {int(k): np.asarray(v, dtype=np.float64) for k, v in skel.items() if v is not None}
        com = compute_center_of_mass(skel_np)
        if com is not None and np.all(np.isfinite(com)):
            com_x.append(float(com[0]))
            com_y.append(float(com[1]))
            com_z.append(float(com[2]))
        else:
            com_x.append(None)
            com_y.append(None)
            com_z.append(None)

        tr = compute_trunk_angle_midpoints(skel)
        trunk_angle.append(float(tr) if tr is not None and np.isfinite(tr) else None)

    return {
        "pos_x": pos_x,
        "pos_y": pos_y,
        "pos_z": pos_z,
        "com_x": com_x,
        "com_y": com_y,
        "com_z": com_z,
        "trunk_angle": trunk_angle,
    }


def _apply_seated_x_stabilizer_sequence(serialized_skeletons, params, raw_reference_skeletons=None):
    """Apply seated fore-aft x stabilizer to serialized skeleton sequence."""
    cfg = processing.normalize_seated_x_stabilizer_params(params)
    stabilizer = processing.create_seated_x_stabilizer(cfg)
    if stabilizer is None:
        return list(serialized_skeletons), cfg, {
            "sitting_score": [None] * len(serialized_skeletons),
            "lean_back": [None] * len(serialized_skeletons),
            "seat_weight": [None] * len(serialized_skeletons),
            "stab_weight": [None] * len(serialized_skeletons),
            "alpha": [None] * len(serialized_skeletons),
            "step_cap": [None] * len(serialized_skeletons),
            "shift_x": [None] * len(serialized_skeletons),
            "max_shift_aft": [None] * len(serialized_skeletons),
            "raw_forward_gap": [None] * len(serialized_skeletons),
            "raw_forward_pull": [None] * len(serialized_skeletons),
        }

    if not isinstance(raw_reference_skeletons, list):
        raw_reference_skeletons = [None] * len(serialized_skeletons)

    out = []
    dbg_sit = []
    dbg_lean = []
    dbg_seat_w = []
    dbg_stab_w = []
    dbg_alpha = []
    dbg_step_cap = []
    dbg_shift = []
    dbg_max_shift_aft = []
    dbg_raw_gap = []
    dbg_raw_pull = []

    for idx, ser in enumerate(serialized_skeletons):
        skel = _deserialize_skeleton_3d(ser)
        raw_ser = raw_reference_skeletons[idx] if idx < len(raw_reference_skeletons) else None
        raw_ref = _deserialize_skeleton_3d(raw_ser)
        if skel is None:
            out.append(None)
            stabilizer.reset()
            dbg_sit.append(None)
            dbg_lean.append(None)
            dbg_seat_w.append(None)
            dbg_stab_w.append(None)
            dbg_alpha.append(None)
            dbg_step_cap.append(None)
            dbg_shift.append(None)
            dbg_max_shift_aft.append(None)
            dbg_raw_gap.append(None)
            dbg_raw_pull.append(None)
            continue

        sit = float(processing._estimate_sitting_score_from_placed(skel))
        stabilized = stabilizer.apply(skel, sitting_score=sit, raw_reference=raw_ref)
        out.append(_serialize_skeleton_3d(stabilized))

        dbg = dict(getattr(stabilizer, "last_debug", {}) or {})
        dbg_sit.append(dbg.get("sitting_score"))
        dbg_lean.append(dbg.get("lean_back"))
        dbg_seat_w.append(dbg.get("seat_weight"))
        dbg_stab_w.append(dbg.get("stab_weight"))
        dbg_alpha.append(dbg.get("alpha"))
        dbg_step_cap.append(dbg.get("step_cap"))
        dbg_shift.append(dbg.get("shift_x"))
        dbg_max_shift_aft.append(dbg.get("max_shift_aft"))
        dbg_raw_gap.append(dbg.get("raw_forward_gap"))
        dbg_raw_pull.append(dbg.get("raw_forward_pull"))

    return out, cfg, {
        "sitting_score": dbg_sit,
        "lean_back": dbg_lean,
        "seat_weight": dbg_seat_w,
        "stab_weight": dbg_stab_w,
        "alpha": dbg_alpha,
        "step_cap": dbg_step_cap,
        "shift_x": dbg_shift,
        "max_shift_aft": dbg_max_shift_aft,
        "raw_forward_gap": dbg_raw_gap,
        "raw_forward_pull": dbg_raw_pull,
    }


def _apply_lateral_y_stabilizer_sequence(serialized_skeletons, params, raw_reference_skeletons=None):
    """Apply hiking-weighted lateral y stabilizer to serialized skeleton sequence."""
    cfg = processing.normalize_lateral_y_stabilizer_params(params)
    stabilizer = processing.create_lateral_y_stabilizer(cfg)
    if stabilizer is None:
        return list(serialized_skeletons), cfg, {
            "sitting_score": [None] * len(serialized_skeletons),
            "lean_back": [None] * len(serialized_skeletons),
            "hike_weight": [None] * len(serialized_skeletons),
            "stab_weight": [None] * len(serialized_skeletons),
            "alpha": [None] * len(serialized_skeletons),
            "step_cap": [None] * len(serialized_skeletons),
            "shift_y": [None] * len(serialized_skeletons),
            "raw_lock_gap": [None] * len(serialized_skeletons),
            "raw_lock_pull": [None] * len(serialized_skeletons),
        }

    if not isinstance(raw_reference_skeletons, list):
        raw_reference_skeletons = [None] * len(serialized_skeletons)

    out = []
    dbg_sit = []
    dbg_lean = []
    dbg_hike = []
    dbg_stab_w = []
    dbg_alpha = []
    dbg_step_cap = []
    dbg_shift = []
    dbg_raw_gap = []
    dbg_raw_pull = []

    for idx, ser in enumerate(serialized_skeletons):
        skel = _deserialize_skeleton_3d(ser)
        raw_ser = raw_reference_skeletons[idx] if idx < len(raw_reference_skeletons) else None
        raw_ref = _deserialize_skeleton_3d(raw_ser)
        if skel is None:
            out.append(None)
            stabilizer.reset()
            dbg_sit.append(None)
            dbg_lean.append(None)
            dbg_hike.append(None)
            dbg_stab_w.append(None)
            dbg_alpha.append(None)
            dbg_step_cap.append(None)
            dbg_shift.append(None)
            dbg_raw_gap.append(None)
            dbg_raw_pull.append(None)
            continue

        sit = float(processing._estimate_sitting_score_from_placed(skel))
        stabilized = stabilizer.apply(skel, sitting_score=sit, raw_reference=raw_ref)
        out.append(_serialize_skeleton_3d(stabilized))

        dbg = dict(getattr(stabilizer, "last_debug", {}) or {})
        dbg_sit.append(dbg.get("sitting_score"))
        dbg_lean.append(dbg.get("lean_back"))
        dbg_hike.append(dbg.get("hike_weight"))
        dbg_stab_w.append(dbg.get("stab_weight"))
        dbg_alpha.append(dbg.get("alpha"))
        dbg_step_cap.append(dbg.get("step_cap"))
        dbg_shift.append(dbg.get("shift_y"))
        dbg_raw_gap.append(dbg.get("raw_lock_gap"))
        dbg_raw_pull.append(dbg.get("raw_lock_pull"))

    return out, cfg, {
        "sitting_score": dbg_sit,
        "lean_back": dbg_lean,
        "hike_weight": dbg_hike,
        "stab_weight": dbg_stab_w,
        "alpha": dbg_alpha,
        "step_cap": dbg_step_cap,
        "shift_y": dbg_shift,
        "raw_lock_gap": dbg_raw_gap,
        "raw_lock_pull": dbg_raw_pull,
    }


def _trim_skeleton_tuning_cache(max_items=6):
    if len(skeleton_tuning_cache) <= max_items:
        return
    oldest = sorted(
        skeleton_tuning_cache.items(),
        key=lambda kv: float(kv[1].get("created_at", 0.0)),
    )
    for cache_id, _entry in oldest[: max(0, len(skeleton_tuning_cache) - max_items)]:
        skeleton_tuning_cache.pop(cache_id, None)


@app.route('/api/skeleton/kalman/cache', methods=['POST'])
def skeleton_kalman_cache():
    """Cache a raw placement snippet (no smoothing) for fast Kalman retuning."""
    data = request.get_json(force=True, silent=True) or {}

    temp_video_id = data.get("temp_video_id")
    if not temp_video_id or temp_video_id not in preview_video_cache:
        return jsonify({"error": "No cached video – preview a frame first"}), 400

    video_path = preview_video_cache[temp_video_id]
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    calib_key = f"skel_calib_{temp_video_id}"
    cached_calib = preview_video_cache.get(calib_key)
    if not cached_calib or "K_undist" not in cached_calib or "maps" not in cached_calib:
        return jsonify({"error": "No calibration cached – preview a skeleton frame first"}), 400

    K_undist = cached_calib["K_undist"]
    map1, map2 = cached_calib["maps"]

    try:
        start_frame = int(data.get("start_frame", 0))
    except Exception:
        start_frame = 0
    try:
        end_frame = int(data.get("end_frame", start_frame + 150))
    except Exception:
        end_frame = start_frame + 150

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        return jsonify({"error": "Video has no frames"}), 400

    end_frame = max(0, min(end_frame, total - 1))
    start_frame = max(0, min(start_frame, end_frame))

    hip_height = float(data.get("hip_height", 0.10))
    ankle_height = float(data.get("ankle_height", 0.01))
    lower_landmark = str(data.get("lower_landmark", "ankle"))
    pose_model = str(data.get("pose_model", "full"))
    if pose_model not in {"lite", "full", "heavy"}:
        pose_model = "full"

    contact_params = data.get("contact_params")
    if not isinstance(contact_params, dict):
        contact_params = None

    camera_x = float(data.get("camera_x", -3.194))
    camera_y = float(data.get("camera_y", 0.0))
    camera_z = float(data.get("camera_z", 0.585))
    camera_pitch = float(data.get("camera_pitch", 8.0))
    camera_yaw = float(data.get("camera_yaw", 0.0))
    camera_roll = float(data.get("camera_roll", 0.0))
    camera_R_wc = parse_camera_rotation_matrix(data.get("camera_R_wc"))
    camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float64)
    if camera_R_wc is not None:
        R_wc = camera_R_wc
    else:
        _, R_wc = default_camera_pose_and_rotation(
            pitch=camera_pitch,
            yaw=camera_yaw,
            roll=camera_roll,
        )

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        cap.release()
        return jsonify({"error": "MediaPipe not installed"}), 500

    model_path = Path(__file__).parent / f"pose_landmarker_{pose_model}.task"
    if not model_path.exists():
        cap.release()
        return jsonify({"error": f"Model file not found: {model_path.name}"}), 500

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    frames = []
    raw_skeletons = []
    raw_confidences = []
    pose_detected_count = 0
    placement_count = 0

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for f in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int((f / max(fps, 1.0)) * 1000.0)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            frames.append(int(f))
            if not result.pose_landmarks or not result.pose_world_landmarks:
                raw_skeletons.append(None)
                raw_confidences.append(None)
                continue

            pose_detected_count += 1
            row_dict = _build_row_dict_from_landmarks(result.pose_world_landmarks[0], result.pose_landmarks[0])
            raw_confidences.append(_serialize_landmark_confidence(row_dict))
            placed = compute_placed_skeleton(
                row_dict,
                K_undist=K_undist,
                W=fw,
                H=fh,
                camera_pos=camera_pos,
                R_wc=R_wc,
                z_plane_lm24=hip_height,
                z_plane_lm28=ankle_height,
                lower_landmark=lower_landmark,
                contact_params=contact_params,
            )
            ser = _serialize_skeleton_3d(placed)
            if ser is not None:
                placement_count += 1
            raw_skeletons.append(ser)
    finally:
        cap.release()
        landmarker.close()

    cache_id = str(uuid.uuid4())[:10]
    skeleton_tuning_cache[cache_id] = {
        "created_at": time.time(),
        "temp_video_id": temp_video_id,
        "fps": float(fps),
        "frames": frames,
        "raw_skeletons": raw_skeletons,
        "raw_confidences": raw_confidences,
    }
    _trim_skeleton_tuning_cache()

    return jsonify({
        "success": True,
        "cache_id": cache_id,
        "frames": frames,
        "fps": float(fps),
        "total_frames": int(total),
        "pose_detected_count": int(pose_detected_count),
        "placement_count": int(placement_count),
        "raw": _compute_skeleton_series(raw_skeletons),
    })


@app.route('/api/skeleton/kalman/apply', methods=['POST'])
def skeleton_kalman_apply():
    """Apply smoothing params to a cached raw snippet."""
    data = request.get_json(force=True, silent=True) or {}
    cache_id = str(data.get("cache_id", "")).strip()
    if not cache_id or cache_id not in skeleton_tuning_cache:
        return jsonify({"error": "Invalid cache_id. Run cache first."}), 400

    cache = skeleton_tuning_cache[cache_id]
    params = data.get("skeleton_filter")
    if not isinstance(params, dict):
        params = {}
    for k in (
        "enabled",
        "process_noise_acc",
        "measurement_noise",
        "use_landmark_confidence",
        "min_landmark_confidence",
        "confidence_floor",
        "confidence_power",
        "max_confidence_noise_scale",
        "gate_sigma",
        "max_consecutive_misses",
        "initial_velocity_std",
        "velocity_decay",
        "max_speed",
        "max_measurement_jump",
        "reacquire_frames",
        "reacquire_max_jump",
    ):
        if k in data and k not in params:
            params[k] = data[k]
    cfg = normalize_skeleton_filter_params(params)
    seated_x_params = data.get("seated_x_stabilizer")
    if not isinstance(seated_x_params, dict):
        seated_x_params = {}
    lateral_y_params = data.get("lateral_y_stabilizer")
    if not isinstance(lateral_y_params, dict):
        lateral_y_params = {}

    raw_skeletons = list(cache.get("raw_skeletons", []))
    raw_confidences = list(cache.get("raw_confidences", []))
    fps = float(cache.get("fps", 30.0))

    if cfg.get("enabled", True):
        smoother = SkeletonPlacementKalman(
            fps=fps,
            process_noise_acc=float(cfg["process_noise_acc"]),
            measurement_noise=float(cfg["measurement_noise"]),
            use_landmark_confidence=bool(cfg["use_landmark_confidence"]),
            min_landmark_confidence=float(cfg["min_landmark_confidence"]),
            confidence_floor=float(cfg["confidence_floor"]),
            confidence_power=float(cfg["confidence_power"]),
            max_confidence_noise_scale=float(cfg["max_confidence_noise_scale"]),
            gate_sigma=float(cfg["gate_sigma"]),
            max_consecutive_misses=int(cfg["max_consecutive_misses"]),
            initial_velocity_std=float(cfg["initial_velocity_std"]),
            velocity_decay=float(cfg["velocity_decay"]),
            max_speed=float(cfg["max_speed"]),
            max_measurement_jump=float(cfg["max_measurement_jump"]),
            reacquire_frames=int(cfg["reacquire_frames"]),
            reacquire_max_jump=float(cfg["reacquire_max_jump"]),
        )
        filtered_skeletons = []
        for i, raw_ser in enumerate(raw_skeletons):
            raw = _deserialize_skeleton_3d(raw_ser)
            conf = _deserialize_landmark_confidence(raw_confidences[i]) if i < len(raw_confidences) else None
            filtered = smoother.smooth(raw, landmark_confidence=conf)
            filtered_skeletons.append(_serialize_skeleton_3d(filtered))
    else:
        filtered_skeletons = list(raw_skeletons)

    stabilized_skeletons, seated_x_cfg, seated_x_series = _apply_seated_x_stabilizer_sequence(
        filtered_skeletons,
        seated_x_params,
        raw_reference_skeletons=raw_skeletons,
    )
    stabilized_skeletons, lateral_y_cfg, lateral_y_series = _apply_lateral_y_stabilizer_sequence(
        stabilized_skeletons,
        lateral_y_params,
        raw_reference_skeletons=raw_skeletons,
    )

    return jsonify({
        "success": True,
        "cache_id": cache_id,
        "frames": list(cache.get("frames", [])),
        "fps": float(fps),
        "params": cfg,
        "raw_skeletons": raw_skeletons,
        "filtered_skeletons": stabilized_skeletons,
        "raw": _compute_skeleton_series(raw_skeletons),
        "filtered": _compute_skeleton_series(stabilized_skeletons),
        "x_stabilizer": {
            "params": seated_x_cfg,
            "series": seated_x_series,
        },
        "y_stabilizer": {
            "params": lateral_y_cfg,
            "series": lateral_y_series,
        },
    })


@app.route('/api/raycast/tuning', methods=['POST'])
def raycast_tuning():
    """Legacy endpoint retained for compatibility."""
    return jsonify({
        "error": "Deprecated endpoint. Use /api/skeleton/kalman/cache and /api/skeleton/kalman/apply."
    }), 410


@app.route('/api/skeleton/frame', methods=['POST'])
def skeleton_frame():
    """Return a raw video frame (no MediaPipe) for quick preview while scrubbing."""
    import base64

    temp_video_id = request.form.get('temp_video_id')
    if temp_video_id and temp_video_id in preview_video_cache:
        video_path = preview_video_cache[temp_video_id]
    elif 'video' in request.files:
        video_file = request.files['video']
        if video_file.filename:
            video_ext = Path(video_file.filename).suffix or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
                video_file.save(tmp.name)
                temp_video_id = str(uuid.uuid4())[:8]
                preview_video_cache[temp_video_id] = tmp.name
                video_path = tmp.name
    else:
        return jsonify({"error": "No video provided"}), 400

    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    frame_number = int(request.form.get('frame_number', 0))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_number, total_frames - 1))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return jsonify({"error": "Cannot read frame"}), 400

    # Apply undistortion if calibration cached
    calib_cache_key = f"skel_calib_{temp_video_id}"
    if calib_cache_key in preview_video_cache:
        cached = preview_video_cache[calib_cache_key]
        frame = cv2.remap(frame, cached['maps'][0], cached['maps'][1], cv2.INTER_LINEAR)
    elif 'calibration' in request.files:
        calib_file = request.files['calibration']
        if calib_file.filename:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.npz')
            tmp_path = tmp.name
            tmp.close()
            calib_file.save(tmp_path)
            try:
                with np.load(tmp_path) as cal:
                    K = cal["K"].copy()
                    D = cal["D"].copy()
                img_size = (frame_width, frame_height)
                R = np.eye(3)
                K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, img_size, R, balance=0.0, new_size=img_size
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, R, K_new, img_size, cv2.CV_16SC2
                )
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                preview_video_cache[calib_cache_key] = {
                    'K_undist': K_new,
                    'maps': (map1, map2),
                }
            except Exception:
                pass
            finally:
                safe_unlink(tmp_path)

    frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
    _, buffer = cv2.imencode('.jpg', frame_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "temp_video_id": temp_video_id,
        "total_frames": total_frames,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_number": frame_number,
        "image": img_b64,
    })


@app.route('/api/skeleton/preview', methods=['POST'])
def skeleton_preview():
    """Preview skeleton placement on a single frame with given camera/height params.

    Runs MediaPipe on the frame, places skeleton in boat frame, and returns
    the skeleton drawn on the frame plus 3D joint positions.
    """
    import base64
    
    # Check if we have a temp video ID (reuse the same cache as rudder)
    temp_video_id = request.form.get('temp_video_id')
    if temp_video_id and temp_video_id in preview_video_cache:
        video_path = preview_video_cache[temp_video_id]
    elif 'video' in request.files:
        video_file = request.files['video']
        if video_file.filename:
            video_ext = Path(video_file.filename).suffix or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
                video_file.save(tmp.name)
                temp_video_id = str(uuid.uuid4())[:8]
                preview_video_cache[temp_video_id] = tmp.name
                video_path = tmp.name
    else:
        return jsonify({"error": "No video provided"}), 400
    
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    # Extract frame
    frame_number = int(request.form.get('frame_number', 0))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_number, total_frames - 1))
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return jsonify({"error": "Cannot read frame"}), 400
    
    # Apply undistortion if calibration provided or cached
    K_undist = None
    undistort_maps = None
    
    # Check for cached calibration from previous request
    calib_cache_key = f"skel_calib_{temp_video_id}"
    if calib_cache_key in preview_video_cache:
        cached = preview_video_cache[calib_cache_key]
        K_undist = cached['K_undist']
        undistort_maps = cached['maps']
        frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)
    elif 'calibration' in request.files:
        calib_file = request.files['calibration']
        if calib_file.filename:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.npz')
            tmp_path = tmp.name
            tmp.close()
            calib_file.save(tmp_path)
            try:
                with np.load(tmp_path) as cal:
                    K = cal["K"].copy()
                    D = cal["D"].copy()
                img_size = (frame_width, frame_height)
                R = np.eye(3)
                K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, img_size, R, balance=0.0, new_size=img_size
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, R, K_new, img_size, cv2.CV_16SC2
                )
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                K_undist = K_new
                undistort_maps = (map1, map2)
                # Cache for subsequent requests
                preview_video_cache[calib_cache_key] = {
                    'K_undist': K_undist,
                    'maps': undistort_maps,
                }
            except Exception:
                pass
            finally:
                safe_unlink(tmp_path)
    
    # Parse parameters
    hip_height = float(request.form.get('hip_height', 0.10))
    ankle_height = float(request.form.get('ankle_height', 0.01))
    camera_x = float(request.form.get('camera_x', -3.194))
    camera_y = float(request.form.get('camera_y', 0.0))
    camera_z = float(request.form.get('camera_z', 0.585))
    camera_pitch = float(request.form.get('camera_pitch', 8.0))
    camera_yaw = float(request.form.get('camera_yaw', 0.0))
    camera_roll = float(request.form.get('camera_roll', 0.0))
    camera_R_wc = parse_camera_rotation_matrix(request.form.get('camera_R_wc'))
    pose_model = request.form.get('pose_model', 'full')
    lower_landmark = request.form.get('lower_landmark', 'ankle')
    contact_params = None
    contact_json = request.form.get('contact_params')
    if contact_json:
        try:
            contact_params = json.loads(contact_json)
        except (json.JSONDecodeError, TypeError):
            contact_params = None
    
    # Set up camera
    camera_pos_vec = np.array([camera_x, camera_y, camera_z], dtype=np.float64)
    if camera_R_wc is not None:
        R_wc = camera_R_wc
    else:
        _, R_wc = default_camera_pose_and_rotation(
            pitch=camera_pitch,
            yaw=camera_yaw,
            roll=camera_roll,
        )
    
    # Run MediaPipe
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return jsonify({"error": "MediaPipe not installed"}), 500
    
    if pose_model not in {"lite", "full", "heavy"}:
        pose_model = "full"
    model_path = Path(__file__).parent / f"pose_landmarker_{pose_model}.task"
    if not model_path.exists():
        return jsonify({"error": f"Model file not found: {model_path.name}"}), 500
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    result = landmarker.detect(mp_image)
    landmarker.close()
    
    if not result.pose_landmarks or not result.pose_world_landmarks:
        # No pose detected — draw frame without skeleton
        frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        _, buffer = cv2.imencode('.jpg', frame_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            "temp_video_id": temp_video_id,
            "total_frames": total_frames,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "frame_number": frame_number,
            "pose_detected": False,
            "image": img_b64,
            "skeleton_3d": None,
        })
    
    # Build row dict for rayplane functions
    world_lm = result.pose_world_landmarks[0]
    norm_lm = result.pose_landmarks[0]
    row_dict = {}
    pose_norm = [None] * 33
    for i in range(33):
        row_dict[f"lm{i}_world_x"] = world_lm[i].x
        row_dict[f"lm{i}_world_y"] = world_lm[i].y
        row_dict[f"lm{i}_world_z"] = world_lm[i].z
        row_dict[f"lm{i}_norm_x"] = norm_lm[i].x
        row_dict[f"lm{i}_norm_y"] = norm_lm[i].y
        row_dict[f"lm{i}_norm_z"] = norm_lm[i].z
        row_dict[f"lm{i}_visibility"] = norm_lm[i].visibility
        pose_norm[i] = [norm_lm[i].x, norm_lm[i].y]
    
    # Compute skeleton placement if calibration available
    placed = None
    skeleton_3d = None
    if K_undist is not None:
        placed = compute_placed_skeleton(
            row_dict,
            K_undist=K_undist,
            W=frame_width,
            H=frame_height,
            camera_pos=camera_pos_vec,
            R_wc=R_wc,
            z_plane_lm24=hip_height,
            z_plane_lm28=ankle_height,
            lower_landmark=lower_landmark,
            contact_params=contact_params,
        )
        if placed is not None:
            skeleton_3d = {}
            for k, v in placed.items():
                if v is not None:
                    skeleton_3d[int(k)] = [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]
    
    # Draw skeleton on frame
    SKEL_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32),
    ]
    
    vis_frame = frame.copy()
    
    # Draw 2D pose (green = MediaPipe raw)
    for a, b in SKEL_CONNECTIONS:
        pa = pose_norm[a]
        pb = pose_norm[b]
        if pa and pb:
            x1 = int(pa[0] * frame_width)
            y1 = int(pa[1] * frame_height)
            x2 = int(pb[0] * frame_width)
            y2 = int(pb[1] * frame_height)
            cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    
    for i, p in enumerate(pose_norm):
        if p:
            x = int(p[0] * frame_width)
            y = int(p[1] * frame_height)
            # Highlight hip (24) and ankle (28) landmarks
            if i == 24 or i == 23:
                cv2.circle(vis_frame, (x, y), 8, (0, 0, 255), -1)  # Red for hips
                cv2.putText(vis_frame, f"Hip {i}", (x + 10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            elif i == 28 or i == 27:
                cv2.circle(vis_frame, (x, y), 8, (255, 128, 0), -1)  # Orange for ankles
                cv2.putText(vis_frame, f"Ankle {i}", (x + 10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(vis_frame, (x, y), 4, (0, 255, 0), -1)
    
    # Draw height labels
    info_y = 30
    cv2.putText(vis_frame, f"Hip Z-plane: {hip_height:.3f}m", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_frame, f"Ankle Z-plane: {ankle_height:.3f}m", (10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2, cv2.LINE_AA)
    
    if placed is not None:
        cv2.putText(vis_frame, "3D Placement: OK", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        # Show hip/ankle 3D coords
        if 24 in placed and placed[24] is not None:
            p = placed[24]
            cv2.putText(vis_frame, f"Hip24 3D: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})",
                        (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1, cv2.LINE_AA)
        if 28 in placed and placed[28] is not None:
            p = placed[28]
            cv2.putText(vis_frame, f"Ankle28 3D: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})",
                        (10, info_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1, cv2.LINE_AA)
    elif K_undist is not None:
        cv2.putText(vis_frame, "3D Placement: FAILED", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis_frame, "3D Placement: No calibration", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    
    # Scale down for transfer
    frame_small = cv2.resize(vis_frame, None, fx=0.5, fy=0.5)
    _, buffer = cv2.imencode('.jpg', frame_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "temp_video_id": temp_video_id,
        "total_frames": total_frames,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_number": frame_number,
        "pose_detected": True,
        "image": img_b64,
        "skeleton_3d": skeleton_3d,
        "has_calibration": K_undist is not None,
    })


@app.route('/api/projects/<project_id>/video')
def serve_project_video(project_id):
    """Serve video file for a project."""
    project_path = processing.get_project_path(project_id)
    if not project_path.exists():
        return "Project not found", 404

    config = processing.get_project_config(project_id) or {}
    video_path = _resolve_project_video_path(project_path, config)
    if video_path is None:
        return "Video not found", 404
    return send_from_directory(str(project_path), video_path.name)


@app.route('/api/projects/<project_id>/frames/metadata')
def project_frames_metadata(project_id):
    """Get frame metadata for a project."""
    data = get_project_data(project_id)
    if not data:
        return jsonify({"error": "Project not found"}), 404
    
    project_df = data["df"]
    config = data["config"]
    
    timestamps = None
    if "timestamp_ms" in project_df.columns:
        timestamps = [
            (float(v) if pd.notna(v) else None)
            for v in project_df["timestamp_ms"].tolist()
        ]
    
    return jsonify({
        "count": int(len(project_df)),
        "fps": float(config.get("fps", 30.0)),
        "timestamps": timestamps,
        "boat_com": float(config.get("boat_com", -1.114)),
    })


@app.route('/api/projects/<project_id>/frames/<int:frame_idx>')
def project_frame(project_id, frame_idx):
    """Get a single frame for a project."""
    data = get_project_data(project_id)
    if not data:
        return jsonify({"error": "Project not found"}), 404
    
    project_df = data["df"]
    
    if frame_idx < 0 or frame_idx >= len(project_df):
        return jsonify({"error": "Frame index out of range"}), 404
    
    row = project_df.iloc[frame_idx]
    
    timestamp = None
    if "timestamp_ms" in row and pd.notna(row["timestamp_ms"]):
        timestamp = float(row["timestamp_ms"])
    
    placed = get_project_placed(project_id, frame_idx, row, data)
    
    skeleton = None
    if placed is not None:
        skeleton = {int(k): [float(v[0]), float(v[1]), float(v[2])] for k, v in placed.items() if v is not None}

    pose_norm = extract_pose_norm(row)
    
    return jsonify({
        "frame_idx": int(frame_idx),
        "timestamp": timestamp,
        "skeleton": skeleton,
        "pose_norm": pose_norm,
    })


@app.route('/api/projects/<project_id>/frames/bulk')
def project_frames_bulk(project_id):
    """Get all frames for a project."""
    global project_bulk_cache

    include_pose_raw = request.args.get("include_pose", "1").strip().lower()
    include_pose = include_pose_raw not in ("0", "false", "no")
    cache_key = f"{project_id}|pose={int(include_pose)}"
    
    # Return cached response if available
    if cache_key in project_bulk_cache:
        return jsonify(project_bulk_cache[cache_key])
    
    data = get_project_data(project_id)
    if not data:
        return jsonify({"error": "Project not found"}), 404
    
    project_df = data["df"]
    num_frames = len(project_df)
    cols = project_df.columns
    
    # --- Vectorized column extraction (avoids slow per-row iloc) ---
    ts_arr = project_df["timestamp_ms"].values if "timestamp_ms" in cols else np.full(num_frames, np.nan)
    
    # Pre-extract skeleton columns as numpy arrays
    skel_arrays = {}  # {landmark_idx: (x_arr, y_arr, z_arr)}
    for i in range(33):
        xc, yc, zc = f"skel{i}_x", f"skel{i}_y", f"skel{i}_z"
        if xc in cols and yc in cols and zc in cols:
            skel_arrays[i] = (project_df[xc].values, project_df[yc].values, project_df[zc].values)
    
    # Pre-extract pose norm columns as numpy arrays (optional for large projects)
    norm_arrays = {}  # {landmark_idx: (x_arr, y_arr)}
    if include_pose:
        for i in range(33):
            xc, yc = f"lm{i}_norm_x", f"lm{i}_norm_y"
            if xc in cols and yc in cols:
                norm_arrays[i] = (project_df[xc].values, project_df[yc].values)
    
    # Build frames from arrays (no iloc, no per-row Series creation)
    frames = [None] * num_frames
    for idx in range(num_frames):
        timestamp = float(ts_arr[idx]) if not np.isnan(ts_arr[idx]) else None
        
        # Build skeleton dict from pre-extracted arrays
        skeleton = None
        for i, (xa, ya, za) in skel_arrays.items():
            x, y, z = xa[idx], ya[idx], za[idx]
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                if skeleton is None:
                    skeleton = {}
                skeleton[i] = [round(float(x), 4), round(float(y), 4), round(float(z), 4)]
        
        # Build pose_norm from pre-extracted arrays (optional)
        pose_norm = None
        if include_pose:
            pose_norm = [None] * 33
            for i, (nx, ny) in norm_arrays.items():
                vx, vy = nx[idx], ny[idx]
                if not (np.isnan(vx) or np.isnan(vy)):
                    pose_norm[i] = [float(vx), float(vy)]
        
        frames[idx] = {
            "frame_idx": idx,
            "timestamp": timestamp,
            "skeleton": skeleton,
            "pose_norm": pose_norm,
        }
    
    response = {"count": num_frames, "frames": frames}
    project_bulk_cache[cache_key] = response
    return jsonify(response)


@app.route('/api/projects/<project_id>/frames/chunk')
def project_frames_chunk(project_id):
    """Get a contiguous chunk of frames for a project."""
    global project_chunk_cache

    include_pose_raw = request.args.get("include_pose", "1").strip().lower()
    include_pose = include_pose_raw not in ("0", "false", "no")

    try:
        stride = int(request.args.get("stride", 1))
    except (TypeError, ValueError):
        stride = 1
    stride = max(1, min(stride, 20))

    try:
        start = int(request.args.get("start", 0))
    except (TypeError, ValueError):
        start = 0

    try:
        count = int(request.args.get("count", 600))
    except (TypeError, ValueError):
        count = 600

    count = max(1, min(count, 5000))

    data = get_project_data(project_id)
    if not data:
        return jsonify({"error": "Project not found"}), 404

    project_df = data["df"]
    total_count = len(project_df)
    if total_count <= 0:
        return jsonify({
            "total_count": 0,
            "start": 0,
            "count": 0,
            "include_pose": bool(include_pose),
            "frames": [],
        })

    start = max(0, min(start, total_count - 1))
    end = min(total_count, start + count)
    idx_range = list(range(start, end, stride))
    row_count = len(idx_range)
    span_count = max(0, end - start)

    cache_key = f"{project_id}|start={start}|count={span_count}|stride={stride}|pose={int(include_pose)}"
    if cache_key in project_chunk_cache:
        return jsonify(project_chunk_cache[cache_key])

    cols = project_df.columns

    ts_arr = project_df["timestamp_ms"].values if "timestamp_ms" in cols else np.full(total_count, np.nan)

    skel_arrays = {}
    for i in range(33):
        xc, yc, zc = f"skel{i}_x", f"skel{i}_y", f"skel{i}_z"
        if xc in cols and yc in cols and zc in cols:
            skel_arrays[i] = (project_df[xc].values, project_df[yc].values, project_df[zc].values)

    norm_arrays = {}
    if include_pose:
        for i in range(33):
            xc, yc = f"lm{i}_norm_x", f"lm{i}_norm_y"
            if xc in cols and yc in cols:
                norm_arrays[i] = (project_df[xc].values, project_df[yc].values)

    frames = [None] * row_count
    for out_idx, idx in enumerate(idx_range):
        timestamp = float(ts_arr[idx]) if not np.isnan(ts_arr[idx]) else None

        skeleton = None
        for i, (xa, ya, za) in skel_arrays.items():
            x, y, z = xa[idx], ya[idx], za[idx]
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                if skeleton is None:
                    skeleton = {}
                skeleton[i] = [round(float(x), 4), round(float(y), 4), round(float(z), 4)]

        pose_norm = None
        if include_pose:
            pose_norm = [None] * 33
            for i, (nx, ny) in norm_arrays.items():
                vx, vy = nx[idx], ny[idx]
                if not (np.isnan(vx) or np.isnan(vy)):
                    pose_norm[i] = [round(float(vx), 4), round(float(vy), 4)]

        frame_payload = {
            "frame_idx": idx,
            "timestamp": timestamp,
            "skeleton": skeleton,
        }
        if include_pose:
            frame_payload["pose_norm"] = pose_norm
        frames[out_idx] = frame_payload

    response = {
        "total_count": total_count,
        "start": start,
        "count": row_count,
        "span_count": span_count,
        "stride": stride,
        "include_pose": bool(include_pose),
        "frames": frames,
    }
    project_chunk_cache[cache_key] = response
    if len(project_chunk_cache) > 256:
        # Keep chunk cache bounded to avoid unbounded memory growth.
        project_chunk_cache.pop(next(iter(project_chunk_cache)))
    return jsonify(response)


@app.route('/api/projects/<project_id>/report/summary')
def project_report_summary(project_id):
    """Build a session report summary for the full project or a frame range."""
    start_frame = _parse_int_query_arg(request.args.get("start_frame"))
    end_frame = _parse_int_query_arg(request.args.get("end_frame"))
    report, err = build_project_report(
        project_id,
        start_frame=start_frame,
        end_frame=end_frame,
        include_chart=False,
    )
    if err:
        status_code = 404 if "not found" in str(err).lower() else 400
        return jsonify({"error": err}), status_code
    return jsonify(report)


@app.route('/projects/<project_id>/report')
def project_report_page(project_id):
    """Interactive, data-heavy technique report view."""
    data = get_project_data(project_id)
    if not data:
        return "Project not found", 404

    total_frames = int(len(data["df"]))
    if total_frames <= 0:
        return "Project has no frames", 400

    start_frame = _parse_int_query_arg(request.args.get("start_frame"))
    end_frame = _parse_int_query_arg(request.args.get("end_frame"))
    start = int(0 if start_frame is None else max(0, min(int(start_frame), total_frames - 1)))
    end = int(total_frames - 1 if end_frame is None else max(0, min(int(end_frame), total_frames - 1)))
    if end < start:
        end = start

    project_name = str((data.get("config", {}) or {}).get("name", project_id))
    return render_template(
        "technique_report.html",
        project_id=project_id,
        project_name=project_name,
        start_frame=start,
        end_frame=end,
    )


@app.route('/api/projects/<project_id>/report/data')
def project_report_data(project_id):
    """Detailed frame-wise data for interactive technique analysis report."""
    start_frame = _parse_int_query_arg(request.args.get("start_frame"))
    end_frame = _parse_int_query_arg(request.args.get("end_frame"))
    payload, err = build_project_report_timeseries(
        project_id,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if err:
        status_code = 404 if "not found" in str(err).lower() else 400
        return jsonify({"error": err}), status_code
    return jsonify(payload)


@app.route('/api/projects/<project_id>/hull/side-profile')
def project_hull_side_profile(project_id):
    """2D hull side-profile extracted from STL for report side-view overlay."""
    data = get_project_data(project_id)
    if not data:
        return jsonify({"error": "Project not found"}), 404
    return jsonify(_build_hull_side_profile_payload())


@app.route('/api/projects/<project_id>/report/download')
def project_report_download(project_id):
    """Download a high-quality HTML session report for the selected project."""
    start_frame = _parse_int_query_arg(request.args.get("start_frame"))
    end_frame = _parse_int_query_arg(request.args.get("end_frame"))
    report, err = build_project_report(
        project_id,
        start_frame=start_frame,
        end_frame=end_frame,
        include_chart=True,
    )
    if err:
        status_code = 404 if "not found" in str(err).lower() else 400
        return jsonify({"error": err}), status_code

    safe_project_name = secure_filename(report.get("project_name", project_id)) or project_id
    r0 = report["range"]["start_frame"]
    r1 = report["range"]["end_frame"]
    filename = f"{safe_project_name}_session_report_f{r0}-{r1}.html"

    html = render_template("session_report.html", report=report)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(html, mimetype="text/html", headers=headers)


@app.route('/api/projects/<project_id>/metrics')
def project_metrics(project_id):
    """Get metrics for a project."""
    global project_metrics_cache
    
    # Return cached response if available
    if project_id in project_metrics_cache:
        return jsonify(project_metrics_cache[project_id])
    
    data = get_project_data(project_id)
    if not data:
        return jsonify({"error": "Project not found"}), 404
    
    project_df = data["df"]
    num_frames = len(project_df)
    cols = project_df.columns
    
    # --- Vectorized column extraction (avoids slow per-row iloc) ---
    def col_values(name):
        return project_df[name].values if name in cols else np.full(num_frames, np.nan)
    
    ts_arr = col_values("timestamp_ms")
    rudder_arr = col_values("rudder_angle")
    boom_arr = col_values("boom_angle")
    trunk_arr = col_values("trunk_angle")
    cx_arr = col_values("com_x")
    cy_arr = col_values("com_y")
    cz_arr = col_values("com_z")
    mp_arr = col_values("moment_pitch")
    mr_arr = col_values("moment_roll")
    
    # Convert rudder/boom to Python lists with None for NaN, then filter
    rudder_list = [float(v) if np.isfinite(v) else None for v in rudder_arr]
    boom_list = [float(v) if np.isfinite(v) else None for v in boom_arr]
    filtered_rudder = apply_low_pass_filter(rudder_list)
    filtered_boom = apply_low_pass_filter(boom_list)

    config = data.get("config", {}) or {}
    fps_cfg = float(_gps_sync_parse_float(config.get("fps")) or 30.0)
    boat_heel_arr, boat_trim_arr, boat_meta = _gps_sync_boat_attitude_for_project(
        project_id=project_id,
        config=config,
        num_frames=num_frames,
        fps=fps_cfg,
    )
    
    # Build metrics from arrays in a single pass (no iloc)
    metrics = [None] * num_frames
    for i in range(num_frames):
        ts = float(ts_arr[i]) if np.isfinite(ts_arr[i]) else None
        ta = float(trunk_arr[i]) if np.isfinite(trunk_arr[i]) else None
        mp_val = float(mp_arr[i]) if np.isfinite(mp_arr[i]) else None
        mr_val = float(mr_arr[i]) if np.isfinite(mr_arr[i]) else None
        boat_heel = float(boat_heel_arr[i]) if np.isfinite(boat_heel_arr[i]) else None
        boat_trim = float(boat_trim_arr[i]) if np.isfinite(boat_trim_arr[i]) else None
        
        com = None
        if np.isfinite(cx_arr[i]) and np.isfinite(cy_arr[i]) and np.isfinite(cz_arr[i]):
            com = [float(cx_arr[i]), float(cy_arr[i]), float(cz_arr[i])]
        
        metrics[i] = {
            "frame_idx": i,
            "timestamp": ts,
            "trunk_angle": ta,
            "com": com,
            "moment_pitch": mp_val,
            "moment_roll": mr_val,
            "rudder_angle": filtered_rudder[i],
            "boom_angle": filtered_boom[i],
            "boat_heel": boat_heel,
            "boat_trim": boat_trim,
        }
    
    response = {
        "count": num_frames,
        "metrics": metrics,
        "boat_attitude": boat_meta,
    }
    project_metrics_cache[project_id] = response
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
