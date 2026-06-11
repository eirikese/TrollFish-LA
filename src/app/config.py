from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path("workspace_data").resolve()
PROJECTS_ROOT = DATA_ROOT / "projects"
TMP_UPLOAD_ROOT = DATA_ROOT / "tmp" / "uploads"
APP_DB_PATH = DATA_ROOT / "app.sqlite"

UPLOAD_BUFFER_SIZE = 8 * 1024 * 1024
MAP_MAX_POINTS_PER_TRACK = 2500
MATCH_SAMPLE_POINTS = 320

EXIFTOOL_FALLBACKS = [
    r"C:\Program Files\ExifTool\exiftool.exe",
    r"C:\Program Files\exiftool-13.51_64\exiftool.exe",
]

# Disabled by default for faster ingest on slower machines.
# Set TROLLFISH_ENABLE_PROXY_TRANSCODE=1 to enable ffmpeg proxy generation.
ENABLE_VIDEO_PROXY_TRANSCODE = os.getenv("TROLLFISH_ENABLE_PROXY_TRANSCODE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


ENABLE_SKELETON_RAYCAST_PROCESSING = os.getenv(
    "TROLLFISH_ENABLE_SKELETON_RAYCAST", "1"
).strip().lower() in {"1", "true", "yes", "on"}
POSE_MODEL_PATH = os.getenv("TROLLFISH_POSE_MODEL", "").strip() or None
SKELETON_TARGET_FPS = _env_float("TROLLFISH_SKELETON_TARGET_FPS", 8.0)
SKELETON_LOWER_PLANE_Z = _env_float("TROLLFISH_SKELETON_LOWER_PLANE_Z", 0.0)
SKELETON_HIP_PLANE_Z = _env_float("TROLLFISH_SKELETON_HIP_PLANE_Z", 0.06)
