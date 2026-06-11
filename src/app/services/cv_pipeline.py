from __future__ import annotations

import base64
import bisect
import concurrent.futures
import csv
import sys
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.app.config import PROJECTS_ROOT
from src.app.db import MetadataStore
from src.app.services.assets import (
    resolve_auto_pnp_model_path,
    resolve_default_calibration_path,
    resolve_gopro13_calibration_path,
    resolve_pose_model_path,
)

try:
    from src.app.services.pose_ref import processing_core as proc
    _PROC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pylint: disable=broad-except
    proc = None  # type: ignore[assignment]
    _PROC_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

_POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class CvArtifactPaths:
    pose_csv: Path
    skeleton_jsonl: Path
    metrics_csv: Path
    summary_json: Path
    autopnp_json: Path


@dataclass(slots=True)
class PoseSeriesCacheEntry:
    signature: tuple[int, int]
    video_s: list[float]
    rows: list[dict[str, Any]]


def get_cv_artifact_paths(project_id: str, file_id: str) -> CvArtifactPaths:
    base = PROJECTS_ROOT / project_id / "derived" / "cv" / file_id
    base.parent.mkdir(parents=True, exist_ok=True)
    return CvArtifactPaths(
        pose_csv=base.with_suffix(".pose.csv"),
        skeleton_jsonl=base.with_suffix(".skeleton.jsonl"),
        metrics_csv=base.with_suffix(".metrics.csv"),
        summary_json=base.with_suffix(".summary.json"),
        autopnp_json=base.with_suffix(".autopnp_history.json"),
    )


def _configure_reference_paths() -> None:
    if proc is None:
        return
    for model_name in ("lite", "full", "heavy"):
        model_path = resolve_pose_model_path(model_name)
        if model_path is not None:
            proc.MODEL_PATHS[model_name] = model_path
    pnp_model = resolve_auto_pnp_model_path()
    if pnp_model is not None:
        proc.AUTO_CAMERA_PNP_MODEL_PATH = pnp_model


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def build_default_cv_config() -> dict[str, Any]:
    _configure_reference_paths()
    default_calib = resolve_default_calibration_path()
    if proc is None:
        return {
            "pose_model": "full",
            "calibration_path": str(default_calib) if default_calib else None,
            "camera_position": [-3.194, 0.02, 0.585],
            "camera_pitch_deg": 14.7,
            "camera_yaw_deg": 0.0,
            "camera_roll_deg": 0.0,
            "camera_R_wc": None,
            "lower_plane_z": 0.0,
            "hip_plane_z": 0.08,
            "lower_landmark": "ankle",
            "athlete_weight": 75.0,
            "boat_com": -1.114,
            "mediapipe_workers": 2,
            "skeleton_filter": {},
            "contact_params": {},
            "seated_x_stabilizer": {},
            "lateral_y_stabilizer": {},
            "auto_camera_pnp": {
                "enabled": True,
                "interval_frames": 10000,
                "avg_frames": 5,
                "min_valid_frames": 5,
            },
        }
    return {
        "pose_model": "full",
        "calibration_path": str(default_calib) if default_calib else None,
        "camera_position": [-3.194, 0.02, 0.585],
        "camera_pitch_deg": 14.7,
        "camera_yaw_deg": 0.0,
        "camera_roll_deg": 0.0,
        "camera_R_wc": None,
        "lower_plane_z": 0.0,
        "hip_plane_z": 0.08,
        "lower_landmark": "ankle",
        "athlete_weight": 75.0,
        "boat_com": -1.114,
        "mediapipe_workers": 2,
        "skeleton_filter": proc.normalize_skeleton_filter_params(None),
        "contact_params": proc.normalize_contact_params(None),
        "seated_x_stabilizer": proc.normalize_seated_x_stabilizer_params(None),
        "lateral_y_stabilizer": proc.normalize_lateral_y_stabilizer_params(None),
        "auto_camera_pnp": {
            "enabled": True,
            "interval_frames": int(proc.AUTO_CAMERA_PNP_INTERVAL_FRAMES),
            "avg_frames": int(proc.AUTO_CAMERA_PNP_AVG_FRAMES),
            "min_valid_frames": int(proc.AUTO_CAMERA_PNP_MIN_VALID_FRAMES),
        },
    }


def _resolve_calibration_path(project_id: str, config: dict[str, Any]) -> Path | None:
    configured = config.get("calibration_path")
    candidates: list[Path] = []
    if isinstance(configured, str) and configured.strip():
        candidates.append(Path(configured.strip()))
    project_root = PROJECTS_ROOT / project_id
    candidates.extend([project_root / "calibration.npz", project_root / "gopro_fisheye_calib.npz"])
    default_path = resolve_default_calibration_path()
    if default_path is not None:
        candidates.append(default_path)
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else (project_root / candidate)
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def _looks_like_gopro13(value: Any) -> bool:
    text = str(value or "").lower()
    return bool(re.search(r"\bhero\s*13\b|\bhero13\b|\bgopro\s*13\b", text))


def _apply_video_specific_calibration(config: dict[str, Any], file_row: dict[str, Any]) -> dict[str, Any]:
    camera_text = " ".join(
        str(file_row.get(key) or "")
        for key in ("device_make", "device_model", "filename")
    )
    if not _looks_like_gopro13(camera_text):
        return config
    gopro13_calib = resolve_gopro13_calibration_path()
    if gopro13_calib is None:
        return config
    out = dict(config)
    out["calibration_path"] = str(gopro13_calib)
    out["camera_model"] = "gopro13"
    return out


def _resolve_model(config: dict[str, Any]) -> Path | None:
    if proc is None:
        return None
    _configure_reference_paths()
    pose_model = str(config.get("pose_model") or "full").strip().lower()
    if pose_model not in {"lite", "full", "heavy"}:
        pose_model = "full"
    found = resolve_pose_model_path(pose_model)
    if found is not None:
        return found
    alt = proc.MODEL_PATHS.get(pose_model)
    return Path(alt) if alt and Path(alt).exists() else None


def _merge_config(current: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in updates.items():
        if value is not None:
            merged[key] = value
    merged["mediapipe_workers"] = _clamp_int(
        merged.get("mediapipe_workers"),
        default=2,
        minimum=1,
        maximum=16,
    )
    if proc is None:
        auto = merged.get("auto_camera_pnp")
        if not isinstance(auto, dict):
            auto = {}
        merged["auto_camera_pnp"] = {
            "enabled": bool(auto.get("enabled", True)),
            "interval_frames": _clamp_int(auto.get("interval_frames"), 10000, 1, 1_000_000),
            "avg_frames": _clamp_int(auto.get("avg_frames"), 5, 1, 200),
            "min_valid_frames": _clamp_int(auto.get("min_valid_frames"), 5, 1, 200),
        }
        return merged
    merged["skeleton_filter"] = proc.normalize_skeleton_filter_params(merged.get("skeleton_filter"))
    merged["contact_params"] = proc.normalize_contact_params(merged.get("contact_params"))
    merged["seated_x_stabilizer"] = proc.normalize_seated_x_stabilizer_params(merged.get("seated_x_stabilizer"))
    merged["lateral_y_stabilizer"] = proc.normalize_lateral_y_stabilizer_params(merged.get("lateral_y_stabilizer"))
    auto = merged.get("auto_camera_pnp")
    if not isinstance(auto, dict):
        auto = {}
    merged["auto_camera_pnp"] = {
        "enabled": bool(auto.get("enabled", True)),
        "interval_frames": _clamp_int(
            auto.get("interval_frames"),
            int(proc.AUTO_CAMERA_PNP_INTERVAL_FRAMES),
            1,
            1_000_000,
        ),
        "avg_frames": _clamp_int(
            auto.get("avg_frames"),
            int(proc.AUTO_CAMERA_PNP_AVG_FRAMES),
            1,
            200,
        ),
        "min_valid_frames": _clamp_int(
            auto.get("min_valid_frames"),
            int(proc.AUTO_CAMERA_PNP_MIN_VALID_FRAMES),
            1,
            200,
        ),
    }
    return merged


def _frame_ts_ms(frame_idx: int, fps: float) -> int:
    if fps <= 1e-6:
        fps = 30.0
    return int(round((float(frame_idx) / float(fps)) * 1000.0))


def _get_video_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_idx = int(frame_index)
        if target_idx < 0:
            target_idx = 0
        if total_frames > 0:
            target_idx = min(target_idx, total_frames - 1)
        if target_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {target_idx} from video: {video_path}")
        return frame_bgr, fps
    finally:
        cap.release()


def _draw_pose_overlay(frame_bgr: np.ndarray, pose_packet: dict[str, np.ndarray] | None) -> np.ndarray:
    out = frame_bgr.copy()
    if pose_packet is None:
        return out
    norm = pose_packet.get("norm")
    if norm is None:
        return out
    norm_arr = np.asarray(norm, dtype=np.float32)
    if norm_arr.shape != (33, 4):
        return out

    h, w = out.shape[:2]
    points: list[tuple[int, int] | None] = [None] * 33
    for idx in range(33):
        x = float(norm_arr[idx, 0]) * float(w)
        y = float(norm_arr[idx, 1]) * float(h)
        vis = float(norm_arr[idx, 3]) if np.isfinite(norm_arr[idx, 3]) else 0.0
        if not np.isfinite(x) or not np.isfinite(y) or vis < 0.15:
            continue
        points[idx] = (int(round(x)), int(round(y)))

    for a, b in _POSE_CONNECTIONS:
        pa = points[a]
        pb = points[b]
        if pa is None or pb is None:
            continue
        cv2.line(out, pa, pb, (24, 196, 132), 2, cv2.LINE_AA)

    for point in points:
        if point is None:
            continue
        cv2.circle(out, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, point, 5, (24, 196, 132), 1, cv2.LINE_AA)

    return out


def _pose_row_header() -> list[str]:
    header = ["frame_idx", "timestamp_ms"]
    for i in range(33):
        header.extend(
            [f"lm{i}_world_x", f"lm{i}_world_y", f"lm{i}_world_z", f"lm{i}_norm_x", f"lm{i}_norm_y", f"lm{i}_norm_z", f"lm{i}_visibility"]
        )
    for i in range(33):
        header.extend([f"skel{i}_x", f"skel{i}_y", f"skel{i}_z"])
    header.extend(["trunk_angle", "com_x", "com_y", "com_z", "moment_pitch", "moment_roll"])
    return header


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        parsed = int(float(value))
        return parsed
    except (TypeError, ValueError):
        return int(default)


def _is_finite(value: Any) -> bool:
    return bool(np.isfinite(_safe_float(value)))


def _reference_runtime_root() -> Path:
    return PROJECTS_ROOT / "_reference_runtime"


def _artifact_base_from_pose_csv(pose_csv_path: Path) -> Path:
    """Return the shared artifact base path for a `<file_id>.pose.csv` file."""
    base = pose_csv_path.with_suffix("")
    if base.suffix == ".pose":
        base = base.with_suffix("")
    return base




def _finite_xy_arrays(x_values: list[float], y_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return x[mask], y[mask]



def _write_metrics_and_skeleton_from_pose_csv(
    pose_csv_path: Path,
    metrics_csv_path: Path,
    skeleton_jsonl_path: Path,
    fps: float,
) -> dict[str, int]:
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    frames_with_pose = 0
    frames_with_skeleton = 0
    frames_with_com = 0

    # Some pose CSV cells can contain large JSON blobs; raise the limit to avoid field size errors.
    try:
        csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    except OverflowError:
        csv.field_size_limit(10 * 1024 * 1024)  # 10 MB fallback

    with (
        pose_csv_path.open("r", encoding="utf-8", newline="") as pose_file,
        metrics_csv_path.open("w", encoding="utf-8", newline="") as metrics_file,
        skeleton_jsonl_path.open("w", encoding="utf-8") as skeleton_file,
    ):
        reader = csv.DictReader(pose_file)
        writer = csv.writer(metrics_file)
        writer.writerow(
            [
                "frame_idx",
                "timestamp_ms",
                "video_s",
                "has_pose",
                "has_skeleton",
                "trunk_angle",
                "com_x",
                "com_y",
                "com_z",
                "moment_pitch",
                "moment_roll",
            ]
        )

        for row in reader:
            frame_idx = _safe_int(row.get("frame_idx"), default=frame_count)
            timestamp_ms = _safe_int(row.get("timestamp_ms"), default=max(0, int((frame_idx / max(1e-6, float(fps))) * 1000.0)))
            video_s = float(frame_idx) / max(1e-6, float(fps))
            has_pose = int(_is_finite(row.get("lm0_world_x")))
            has_skeleton = int(_is_finite(row.get("skel0_x")))

            trunk_angle = _safe_float(row.get("trunk_angle"))
            com_x = _safe_float(row.get("com_x"))
            com_y = _safe_float(row.get("com_y"))
            com_z = _safe_float(row.get("com_z"))
            moment_pitch = _safe_float(row.get("moment_pitch"))
            moment_roll = _safe_float(row.get("moment_roll"))

            writer.writerow(
                [
                    frame_idx,
                    timestamp_ms,
                    video_s,
                    has_pose,
                    has_skeleton,
                    trunk_angle,
                    com_x,
                    com_y,
                    com_z,
                    moment_pitch,
                    moment_roll,
                ]
            )

            frame_count += 1
            if has_pose:
                frames_with_pose += 1
            if has_skeleton:
                frames_with_skeleton += 1
                if np.isfinite(com_x):
                    frames_with_com += 1

            if has_skeleton:
                landmarks: list[list[float] | None] = []
                for idx in range(33):
                    x = _safe_float(row.get(f"skel{idx}_x"))
                    y = _safe_float(row.get(f"skel{idx}_y"))
                    z = _safe_float(row.get(f"skel{idx}_z"))
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        landmarks.append([float(x), float(y), float(z)])
                    else:
                        landmarks.append(None)
                metrics = {
                    "trunk_angle": float(trunk_angle) if np.isfinite(trunk_angle) else None,
                    "com_x": float(com_x) if np.isfinite(com_x) else None,
                    "com_y": float(com_y) if np.isfinite(com_y) else None,
                    "com_z": float(com_z) if np.isfinite(com_z) else None,
                    "moment_pitch": float(moment_pitch) if np.isfinite(moment_pitch) else None,
                    "moment_roll": float(moment_roll) if np.isfinite(moment_roll) else None,
                }
                skeleton_file.write(
                    json.dumps(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "video_s": float(video_s),
                            "landmarks": landmarks,
                            "metrics": metrics,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    return {
        "frame_count": int(frame_count),
        "frames_with_pose": int(frames_with_pose),
        "frames_with_skeleton": int(frames_with_skeleton),
        "frames_with_com": int(frames_with_com),
    }


def _merge_jsonl_file(existing_path: Path, new_path: Path, output_path: Path) -> None:
    """Merge two skeleton JSONL files, deduplicating by frame_idx and sorting by video_s."""
    rows_by_frame: dict[int, str] = {}

    def _load(path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    frame_idx = int(obj.get("frame_idx", -1))
                    rows_by_frame[frame_idx] = line
                except Exception:
                    continue

    _load(existing_path)
    _load(new_path)

    sorted_lines = [rows_by_frame[k] for k in sorted(rows_by_frame)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for line in sorted_lines:
            fh.write(line + "\n")


def _merge_csv_file(existing_path: Path, new_path: Path, output_path: Path, key_col: str = "frame_idx") -> None:
    """Merge two CSV files, deduplicating by key_col and sorting by it."""
    import csv as _csv_mod
    rows_by_key: dict[Any, dict] = {}
    header: list[str] | None = None

    def _load(path: Path) -> None:
        nonlocal header
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = _csv_mod.DictReader(fh)
            if header is None and reader.fieldnames:
                header = list(reader.fieldnames)
            for row in reader:
                try:
                    key = int(float(row[key_col]))
                    rows_by_key[key] = row
                except Exception:
                    continue

    _load(existing_path)
    _load(new_path)

    if header is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv_mod.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for k in sorted(rows_by_key):
            writer.writerow(rows_by_key[k])


def _merge_range_artifacts(artifact_paths: CvArtifactPaths, tmp_paths: CvArtifactPaths) -> None:
    """Merge range-run artifacts (tmp_paths) into permanent artifact files (artifact_paths)."""
    _merge_jsonl_file(artifact_paths.skeleton_jsonl, tmp_paths.skeleton_jsonl, artifact_paths.skeleton_jsonl)
    _merge_csv_file(artifact_paths.metrics_csv, tmp_paths.metrics_csv, artifact_paths.metrics_csv, key_col="frame_idx")
    _merge_csv_file(artifact_paths.pose_csv, tmp_paths.pose_csv, artifact_paths.pose_csv, key_col="frame_idx")


def _build_reference_config(
    project_name: str,
    video_path: Path,
    calibration_path: Path | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    auto_cfg = config.get("auto_camera_pnp") if isinstance(config.get("auto_camera_pnp"), dict) else {}
    return {
        "name": project_name,
        "created": now_utc_iso(),
        "state": "pending",
        "video_file": str(video_path.resolve()),
        "calibration_file": str(calibration_path.resolve()) if calibration_path is not None else None,
        "athlete_weight": float(config.get("athlete_weight", 75.0)),
        "ankle_height": float(config.get("lower_plane_z", 0.0)),
        "hip_height": float(config.get("hip_plane_z", 0.08)),
        "camera_position": [float(x) for x in list(config.get("camera_position") or [-3.194, 0.02, 0.585])[:3]],
        "camera_pitch_deg": float(config.get("camera_pitch_deg", 14.7)),
        "camera_yaw_deg": float(config.get("camera_yaw_deg", 0.0)),
        "camera_roll_deg": float(config.get("camera_roll_deg", 0.0)),
        "camera_R_wc": config.get("camera_R_wc"),
        "boat_com": float(config.get("boat_com", -1.114)),
        "pose_model": str(config.get("pose_model") or "full"),
        "lower_landmark": str(config.get("lower_landmark") or "ankle"),
        "skeleton_filter": proc.normalize_skeleton_filter_params(config.get("skeleton_filter")),
        "seated_x_stabilizer": proc.normalize_seated_x_stabilizer_params(config.get("seated_x_stabilizer")),
        "lateral_y_stabilizer": proc.normalize_lateral_y_stabilizer_params(config.get("lateral_y_stabilizer")),
        "contact_params": proc.normalize_contact_params(config.get("contact_params")),
        "auto_camera_pnp": {
            "enabled": bool(auto_cfg.get("enabled", True)),
            "interval_frames": _clamp_int(auto_cfg.get("interval_frames"), int(proc.AUTO_CAMERA_PNP_INTERVAL_FRAMES), 1, 1_000_000),
            "avg_frames": _clamp_int(auto_cfg.get("avg_frames"), int(proc.AUTO_CAMERA_PNP_AVG_FRAMES), 1, 200),
            "min_valid_frames": _clamp_int(auto_cfg.get("min_valid_frames"), int(proc.AUTO_CAMERA_PNP_MIN_VALID_FRAMES), 1, 200),
        },
        "rudder": {"enabled": False},
        "mediapipe_workers": _clamp_int(config.get("mediapipe_workers"), default=2, minimum=1, maximum=16),
        "start_frame": int(config.get("start_frame") or 0),
        "end_frame": int(config.get("end_frame") or 0),  # 0 means full video
    }


def run_exact_video_pipeline(
    project_id: str,
    video_path: Path,
    config: dict[str, Any],
    output_paths: CvArtifactPaths,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if proc is None:
        return {"status": "failed", "reason": f"Reference CV stack is unavailable: {_PROC_IMPORT_ERROR}"}
    model_path = _resolve_model(config)
    if model_path is None:
        return {"status": "skipped", "reason": "pose_model_missing"}

    calibration_path = _resolve_calibration_path(project_id, config)
    probe = cv2.VideoCapture(str(video_path))
    if not probe.isOpened():
        return {"status": "failed", "reason": "video_open_failed"}
    fps = float(probe.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.release()

    runtime_root = _reference_runtime_root()
    runtime_root.mkdir(parents=True, exist_ok=True)
    proc.PROJECTS_DIR = runtime_root
    proc.ensure_projects_dir()

    runtime_project_id = f"{project_id}_{uuid.uuid4().hex[:12]}"
    runtime_project_path = proc.get_project_path(runtime_project_id)
    runtime_project_path.mkdir(parents=True, exist_ok=True)
    runtime_config_path = runtime_project_path / "config.json"
    runtime_pose_csv = runtime_project_path / "pose.csv"

    runtime_config = _build_reference_config(
        project_name=f"trollfish_{runtime_project_id}",
        video_path=video_path,
        calibration_path=calibration_path,
        config=config,
    )
    with runtime_config_path.open("w", encoding="utf-8") as handle:
        json.dump(runtime_config, handle, indent=2)

    proc.processing_jobs[runtime_project_id] = {
        "state": "pending",
        "progress": 0,
        "status": "Queued",
        "log_lines": [],
        "error": None,
    }

    def run_reference() -> None:
        proc.process_video(runtime_project_id)

    worker = threading.Thread(target=run_reference, name=f"pose-ref-{runtime_project_id}", daemon=True)
    worker.start()

    last_progress = -1.0
    try:
        while worker.is_alive():
            if progress_cb is not None:
                job = proc.processing_jobs.get(runtime_project_id, {})
                progress = float(job.get("progress") or 0.0)
                state = str(job.get("state") or "processing")
                log_lines: list = list(job.get("log_lines") or [])
                # Build rich message: last meaningful log line + status
                last_log = ""
                for entry in reversed(log_lines):
                    msg = str(entry.get("message") or "").strip() if isinstance(entry, dict) else str(entry).strip()
                    if msg:
                        last_log = msg
                        break
                base_status = str(job.get("status") or state)
                message = last_log if last_log else base_status
                if progress != last_progress:
                    last_progress = progress
                    progress_cb(min(0.96, max(0.0, progress / 100.0)), message)
            worker.join(timeout=0.25)

        worker.join()
        if not runtime_pose_csv.exists():
            cfg_state = {}
            if runtime_config_path.exists():
                with runtime_config_path.open("r", encoding="utf-8") as handle:
                    cfg_state = json.load(handle)
            reason = str(cfg_state.get("error") or proc.processing_jobs.get(runtime_project_id, {}).get("error") or "reference_pose_csv_missing")
            summary = {"status": "failed", "reason": reason, "video_path": str(video_path)}
            with output_paths.summary_json.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, separators=(",", ":"))
            return summary

        output_paths.pose_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime_pose_csv, output_paths.pose_csv)
        counts = _write_metrics_and_skeleton_from_pose_csv(
            pose_csv_path=output_paths.pose_csv,
            metrics_csv_path=output_paths.metrics_csv,
            skeleton_jsonl_path=output_paths.skeleton_jsonl,
            fps=fps,
        )

        cfg_after = runtime_config
        if runtime_config_path.exists():
            with runtime_config_path.open("r", encoding="utf-8") as handle:
                cfg_after = json.load(handle)
        auto_pnp_enabled = bool((cfg_after.get("auto_camera_pnp") or {}).get("enabled", True))
        auto_pnp_bootstrap_applied = bool(cfg_after.get("auto_pnp_bootstrap_applied", False))
        auto_pnp_bootstrap_target_successes = int(cfg_after.get("auto_pnp_bootstrap_target_successes", 0) or 0)
        auto_pnp_updates = int(cfg_after.get("auto_pnp_updates", 0) or 0)
        autopnp_payload = {
            "camera_position": cfg_after.get("camera_position"),
            "camera_R_wc": cfg_after.get("camera_R_wc"),
            "camera_pitch_deg": cfg_after.get("camera_pitch_deg"),
            "camera_yaw_deg": cfg_after.get("camera_yaw_deg"),
            "camera_roll_deg": cfg_after.get("camera_roll_deg"),
            "auto_pnp_enabled": auto_pnp_enabled,
            "auto_pnp_bootstrap_applied": auto_pnp_bootstrap_applied,
            "auto_pnp_bootstrap_target_successes": auto_pnp_bootstrap_target_successes,
            "auto_pnp_updates": auto_pnp_updates,
            "auto_pnp_interval_frames": int(cfg_after.get("auto_pnp_interval_frames", 0) or 0),
            "auto_pnp_avg_frames": int(cfg_after.get("auto_pnp_avg_frames", 0) or 0),
            "auto_pnp_min_valid_frames": int(cfg_after.get("auto_pnp_min_valid_frames", 0) or 0),
            "state": cfg_after.get("state"),
            "error": cfg_after.get("error"),
        }
        output_paths.autopnp_json.parent.mkdir(parents=True, exist_ok=True)
        with output_paths.autopnp_json.open("w", encoding="utf-8") as handle:
            json.dump(autopnp_payload, handle, separators=(",", ":"))

        summary = {
            "status": "ok",
            "video_path": str(video_path),
            "model_path": str(model_path),
            "calibration_path": str(calibration_path) if calibration_path is not None else None,
            "fps": float(fps),
            "total_frames": int(total_frames),
            "sampled_frames": int(counts["frame_count"]),
            "frames_with_pose": int(counts["frames_with_pose"]),
            "frames_with_skeleton": int(counts["frames_with_skeleton"]),
            "frames_with_com": int(counts["frames_with_com"]),
            "auto_pnp_enabled": auto_pnp_enabled,
            "auto_pnp_bootstrap_applied": auto_pnp_bootstrap_applied,
            "auto_pnp_bootstrap_target_successes": auto_pnp_bootstrap_target_successes,
            "auto_pnp_updates": auto_pnp_updates,
            "pose_csv_path": str(output_paths.pose_csv),
            "metrics_csv_path": str(output_paths.metrics_csv),
            "skeleton_jsonl_path": str(output_paths.skeleton_jsonl),
            "autopnp_json_path": str(output_paths.autopnp_json),
            "lower_plane_z": float(config.get("lower_plane_z", 0.0)),
            "hip_plane_z": float(config.get("hip_plane_z", 0.08)),
            "reference_pipeline": True,

        }
        with output_paths.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, separators=(",", ":"))
        if progress_cb is not None:
            progress_cb(1.0, "Completed")
        return summary
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Reference CV pipeline failed: %s", video_path)
        summary = {"status": "failed", "reason": str(exc), "video_path": str(video_path)}
        with output_paths.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, separators=(",", ":"))
        return summary
    finally:
        proc.processing_jobs.pop(runtime_project_id, None)
        for _ in range(5):
            try:
                if runtime_project_path.exists():
                    shutil.rmtree(runtime_project_path, ignore_errors=False)
                break
            except OSError:
                time.sleep(0.15)


class CvService:
    def __init__(self, store: MetadataStore) -> None:
        self.store = store
        self._batch_progress_lock = threading.Lock()
        self._pose_cache_lock = threading.Lock()
        self._pose_series_cache: dict[str, PoseSeriesCacheEntry] = {}
        self._pose_cache_order: list[str] = []
        self._pose_cache_limit = 6
        _configure_reference_paths()

    def get_project_config(self, project_id: str) -> dict[str, Any]:
        row = self.store.get_project_cv_config(project_id)
        if row is None:
            cfg = build_default_cv_config()
            self.store.upsert_project_cv_config(project_id, cfg)
            return cfg
        return _merge_config(build_default_cv_config(), row.get("config") or {})

    def update_project_config(self, project_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        merged = _merge_config(self.get_project_config(project_id), updates)
        self.store.upsert_project_cv_config(project_id, merged)
        return merged

    def get_video_status(self, project_id: str, file_id: str) -> dict[str, Any] | None:
        return self.store.get_video_cv_run(project_id, file_id)

    def _resolve_parallel_video_workers(self, config: dict[str, Any], total_videos: int) -> int:
        cpu = max(1, int(os.cpu_count() or 1))
        requested = config.get("mediapipe_workers")
        if requested is None:
            desired = min(2, total_videos)
        else:
            desired = _clamp_int(requested, default=2, minimum=1, maximum=64)
        return max(1, min(desired, total_videos, cpu))

    def _find_nearest_processed_skeleton(self, skeleton_jsonl_path: Path, target_video_s: float) -> dict[str, Any] | None:
        target = max(0.0, float(target_video_s))
        entry = self._load_pose_series_cache_entry(skeleton_jsonl_path)
        if not entry.rows:
            return None
        idx = bisect.bisect_left(entry.video_s, target)
        if idx <= 0:
            return entry.rows[0]
        if idx >= len(entry.video_s):
            return entry.rows[-1]
        prev_idx = idx - 1
        if abs(entry.video_s[idx] - target) < abs(entry.video_s[prev_idx] - target):
            return entry.rows[idx]
        return entry.rows[prev_idx]

    def _extract_skeleton_dict(self, raw_landmarks: Any) -> dict[int, list[float]]:
        skeleton: dict[int, list[float]] = {}
        if not isinstance(raw_landmarks, list):
            return skeleton
        for idx, item in enumerate(raw_landmarks):
            if not isinstance(item, list) or len(item) < 3:
                continue
            try:
                x = float(item[0])
                y = float(item[1])
                z = float(item[2])
            except (TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                skeleton[idx] = [x, y, z]
        return skeleton

    def _extract_metrics_dict(self, raw_metrics: Any) -> dict[str, float | None]:
        metrics: dict[str, float | None] = {}
        if not isinstance(raw_metrics, dict):
            return metrics
        for key, raw_value in raw_metrics.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                metrics[str(key)] = None
                continue
            metrics[str(key)] = value if np.isfinite(value) else None
        return metrics

    def _skeleton_file_signature(self, path: Path) -> tuple[int, int]:
        stat = path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)

    def _touch_pose_cache_key(self, key: str) -> None:
        if key in self._pose_cache_order:
            self._pose_cache_order.remove(key)
        self._pose_cache_order.append(key)

    def _load_pose_series_cache_entry(self, skeleton_jsonl_path: Path) -> PoseSeriesCacheEntry:
        resolved = skeleton_jsonl_path.resolve()
        key = str(resolved)
        signature = self._skeleton_file_signature(resolved)

        with self._pose_cache_lock:
            existing = self._pose_series_cache.get(key)
            if existing is not None and existing.signature == signature:
                self._touch_pose_cache_key(key)
                return existing

        rows: list[dict[str, Any]] = []
        video_s_values: list[float] = []
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError:
                    continue
                raw_video_s = raw.get("video_s")
                try:
                    row_video_s = float(raw_video_s)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(row_video_s):
                    continue
                rows.append(
                    {
                        "frame_idx": int(raw.get("frame_idx") or -1),
                        "timestamp_ms": int(raw.get("timestamp_ms") or -1),
                        "video_s": row_video_s,
                        "landmarks": raw.get("landmarks"),
                        "metrics": raw.get("metrics"),
                    }
                )
                video_s_values.append(row_video_s)

        entry = PoseSeriesCacheEntry(signature=signature, video_s=video_s_values, rows=rows)
        with self._pose_cache_lock:
            self._pose_series_cache[key] = entry
            self._touch_pose_cache_key(key)
            while len(self._pose_cache_order) > self._pose_cache_limit:
                stale_key = self._pose_cache_order.pop(0)
                self._pose_series_cache.pop(stale_key, None)
        return entry

    def get_processed_pose_at_time(self, project_id: str, file_id: str, video_s: float) -> dict[str, Any]:
        file_row = self.store.get_file(file_id)
        if file_row is None or str(file_row.get("project_id")) != project_id:
            raise RuntimeError("Video file not found")

        run = self.store.get_video_cv_run(project_id, file_id)
        if run is None or str(run.get("status")) != "completed":
            raise RuntimeError("No completed skeleton run found for this video")

        skeleton_path_raw = run.get("skeleton_jsonl_path")
        if not skeleton_path_raw:
            raise RuntimeError("Skeleton output is missing for this video")
        skeleton_path = Path(str(skeleton_path_raw))
        if not skeleton_path.exists():
            raise RuntimeError("Skeleton output file is missing on disk")

        nearest = self._find_nearest_processed_skeleton(skeleton_path, float(video_s))
        if nearest is None:
            return {
                "project_id": project_id,
                "file_id": file_id,
                "video_s": float(video_s),
                "frame_index": -1,
                "timestamp_ms": -1,
                "has_skeleton": False,
                "skeleton_3d": None,
                "metrics": {},
            }

        skeleton = self._extract_skeleton_dict(nearest.get("landmarks"))
        metrics = self._extract_metrics_dict(nearest.get("metrics"))

        frame_index = int(nearest.get("frame_idx") or -1)
        timestamp_ms = int(nearest.get("timestamp_ms") or -1)
        row_video_s = float(nearest.get("video_s") or video_s)
        return {
            "project_id": project_id,
            "file_id": file_id,
            "video_s": row_video_s,
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "has_skeleton": bool(skeleton),
            "skeleton_3d": skeleton if skeleton else None,
            "metrics": metrics,
        }

    def get_processed_pose_range(
        self,
        project_id: str,
        file_id: str,
        frame_start: int,
        frame_end: int,
        frame_step: int = 1,
        max_samples: int = 1600,
    ) -> dict[str, Any]:
        file_row = self.store.get_file(file_id)
        if file_row is None or str(file_row.get("project_id")) != project_id:
            raise RuntimeError("Video file not found")

        run = self.store.get_video_cv_run(project_id, file_id)
        if run is None or str(run.get("status")) != "completed":
            raise RuntimeError("No completed skeleton run found for this video")

        skeleton_path_raw = run.get("skeleton_jsonl_path")
        if not skeleton_path_raw:
            raise RuntimeError("Skeleton output is missing for this video")
        skeleton_path = Path(str(skeleton_path_raw))
        if not skeleton_path.exists():
            raise RuntimeError("Skeleton output file is missing on disk")

        entry = self._load_pose_series_cache_entry(skeleton_path)
        frame_start = max(0, int(frame_start))
        frame_end = max(frame_start, int(frame_end))
        frame_step = max(1, int(frame_step))
        max_samples = max(20, min(int(max_samples), 4000))
        requested_span = max(1, frame_end - frame_start + 1)
        effective_step = max(frame_step, int(math.ceil(requested_span / max_samples)))

        samples: list[dict[str, Any]] = []
        for raw in entry.rows:
            frame_idx = int(raw.get("frame_idx") or -1)
            if frame_idx < frame_start or frame_idx > frame_end:
                continue
            if (frame_idx - frame_start) % effective_step != 0:
                continue

            row_video_s = float(raw.get("video_s") or 0.0)
            if not np.isfinite(row_video_s):
                row_video_s = 0.0
            skeleton = self._extract_skeleton_dict(raw.get("landmarks"))
            metrics = self._extract_metrics_dict(raw.get("metrics"))
            samples.append(
                {
                    "frame_index": frame_idx,
                    "video_s": row_video_s,
                    "has_skeleton": bool(skeleton),
                    "skeleton_3d": skeleton if skeleton else None,
                    "metrics": metrics,
                }
            )
            if len(samples) >= max_samples:
                break

        return {
            "project_id": project_id,
            "file_id": file_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_step": frame_step,
            "effective_step": effective_step,
            "max_samples": max_samples,
            "sample_count": len(samples),
            "samples": samples,
        }

    def process_skeleton_batch(
        self,
        project_id: str,
        video_file_ids: list[str] | None,
        force: bool,
        progress_cb: Callable[[float, str], None] | None = None,
        mediapipe_workers: int | None = None,
        video_ranges: dict[str, dict[str, float | None]] | None = None,
        description: str | None = None,
    ) -> dict[str, int]:
        files = [f for f in self.store.list_files(project_id) if str(f.get("kind")) == "video"]
        if video_file_ids:
            selected = set(video_file_ids)
            files = [f for f in files if str(f.get("id")) in selected]
        if not files:
            raise RuntimeError("No video files selected for skeleton processing.")

        config = self.get_project_config(project_id)
        total = len(files)
        processed = 0
        skipped = 0
        failed = 0

        progress_by_file: dict[str, float] = {}
        run_files: list[dict[str, Any]] = []
        for file_row in files:
            file_id = str(file_row["id"])
            artifact_paths = get_cv_artifact_paths(project_id, file_id)
            existing = self.store.get_video_cv_run(project_id, file_id)
            if (
                not force
                and existing is not None
                and str(existing.get("status")) == "completed"
                and artifact_paths.summary_json.exists()
            ):
                # Don't skip if the new request covers more than the previous run.
                # A partial previous run (start_frame/end_frame set) must not block
                # a subsequent full-video run (no range specified for this file).
                new_is_full_run = (video_ranges or {}).get(file_id) is None
                if new_is_full_run:
                    try:
                        snap_raw = existing.get("config_snapshot_json") or "{}"
                        snap = json.loads(snap_raw) if isinstance(snap_raw, str) else (snap_raw or {})
                        prev_sf = int(snap.get("start_frame") or 0)
                        prev_ef = int(snap.get("end_frame") or 0)
                        existing_was_partial = prev_sf > 0 or prev_ef > 0
                    except Exception:
                        existing_was_partial = False
                    if existing_was_partial:
                        # Previous run was partial — let the full-feed run through.
                        progress_by_file[file_id] = 0.0
                        run_files.append(file_row)
                        continue
                skipped += 1
                progress_by_file[file_id] = 1.0
                continue
            progress_by_file[file_id] = 0.0
            run_files.append(file_row)

        def emit_overall(message: str) -> None:
            if progress_cb is None:
                return
            with self._batch_progress_lock:
                overall = sum(progress_by_file.values()) / float(max(1, total))
            prefix = f"[{description}] " if description else ""
            progress_cb(float(overall), f"{prefix}{message}")

        if skipped:
            emit_overall(f"Skipped {skipped}/{total} already completed videos")
        if not run_files:
            emit_overall("No videos needed processing")
            return {"processed": 0, "skipped": skipped, "failed": 0, "total": total}

        worker_count = self._resolve_parallel_video_workers(config, len(run_files))

        def process_one(file_row: dict[str, Any]) -> bool:
            file_id = str(file_row["id"])
            filename = str(file_row.get("filename") or file_id)
            artifact_paths = get_cv_artifact_paths(project_id, file_id)
            camera_file_row = dict(file_row)
            track_row = self.store.get_track_by_file_id(file_id)
            if track_row and track_row.get("meta_json"):
                try:
                    track_meta = json.loads(str(track_row.get("meta_json") or "{}"))
                except json.JSONDecodeError:
                    track_meta = {}
                for key in ("device_make", "device_model"):
                    if track_meta.get(key) and not camera_file_row.get(key):
                        camera_file_row[key] = track_meta[key]
            cfg_snapshot = _apply_video_specific_calibration(dict(config), camera_file_row)
            # Override mediapipe_workers if caller specified one
            if mediapipe_workers is not None:
                cfg_snapshot["mediapipe_workers"] = int(mediapipe_workers)
            # Apply per-video time trimming: convert seconds → frames using video fps
            vrange = (video_ranges or {}).get(file_id)
            is_range_run = bool(vrange)
            if vrange:
                probe = cv2.VideoCapture(str(file_row["path"]))
                vfps = float(probe.get(cv2.CAP_PROP_FPS) or 30.0)
                vtotal = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                probe.release()
                start_s = vrange.get("start_sec")
                end_s = vrange.get("end_sec")
                if start_s is not None:
                    cfg_snapshot["start_frame"] = max(0, int(round(float(start_s) * vfps)))
                if end_s is not None:
                    cfg_snapshot["end_frame"] = min(vtotal, int(round(float(end_s) * vfps)))

            # For range runs write to a temp location so we can merge with existing
            if is_range_run:
                tmp_base = PROJECTS_ROOT / project_id / "derived" / "cv" / f"{file_id}_range_tmp"
                run_paths = CvArtifactPaths(
                    pose_csv=tmp_base.with_suffix(".pose.csv"),
                    skeleton_jsonl=tmp_base.with_suffix(".skeleton.jsonl"),
                    metrics_csv=tmp_base.with_suffix(".metrics.csv"),
                    summary_json=tmp_base.with_suffix(".summary.json"),
                    autopnp_json=tmp_base.with_suffix(".autopnp_history.json"),
                )
            else:
                run_paths = artifact_paths

            try:
                self.store.upsert_video_cv_run(
                    project_id=project_id,
                    file_id=file_id,
                    status="running",
                    progress=0.0,
                    message="Starting skeleton processing",
                    error=None,
                    config_snapshot=cfg_snapshot,
                    started_at=now_utc_iso(),
                    finished_at=None,
                )

                def on_video_progress(progress: float, message: str) -> None:
                    clamped = max(0.0, min(1.0, float(progress)))
                    with self._batch_progress_lock:
                        progress_by_file[file_id] = clamped
                    self.store.upsert_video_cv_run(
                        project_id=project_id,
                        file_id=file_id,
                        status="running",
                        progress=clamped,
                        message=message,
                        error=None,
                        config_snapshot=cfg_snapshot,
                    )
                    emit_overall(f"{filename}: {message}")

                summary = run_exact_video_pipeline(
                    project_id,
                    Path(str(file_row["path"])),
                    cfg_snapshot,
                    run_paths,
                    progress_cb=on_video_progress,
                )
                ok = str(summary.get("status")) == "ok"
                with self._batch_progress_lock:
                    progress_by_file[file_id] = 1.0

                if ok:
                    if is_range_run:
                        # Merge range results into permanent artifacts instead of overwriting
                        try:
                            _merge_range_artifacts(artifact_paths, run_paths)
                        except Exception as merge_exc:  # pylint: disable=broad-except
                            logger.exception("Failed to merge range artifacts for %s: %s", file_id, merge_exc)
                        finally:
                            # Clean up temp files
                            for tmp_path in (run_paths.pose_csv, run_paths.skeleton_jsonl,
                                             run_paths.metrics_csv, run_paths.summary_json,
                                             run_paths.autopnp_json):
                                try:
                                    tmp_path.unlink(missing_ok=True)
                                except OSError:
                                    pass
                        # Invalidate the in-memory skeleton cache so the merged file is re-read
                        with self._pose_cache_lock:
                            cache_key = str(artifact_paths.skeleton_jsonl.resolve())
                            self._pose_series_cache.pop(cache_key, None)
                            if cache_key in self._pose_cache_order:
                                self._pose_cache_order.remove(cache_key)

                    self.store.upsert_video_cv_run(
                        project_id=project_id,
                        file_id=file_id,
                        status="completed",
                        progress=1.0,
                        message="Completed",
                        error=None,
                        config_snapshot=cfg_snapshot,
                        summary_json_path=str(artifact_paths.summary_json),
                        metrics_csv_path=str(artifact_paths.metrics_csv),
                        skeleton_jsonl_path=str(artifact_paths.skeleton_jsonl),
                        pose_csv_path=str(artifact_paths.pose_csv),
                        autopnp_json_path=str(artifact_paths.autopnp_json),
                        finished_at=now_utc_iso(),
                    )
                    emit_overall(f"{filename}: completed")
                    return True

                # Clean up temp files on failure too
                if is_range_run:
                    for tmp_path in (run_paths.pose_csv, run_paths.skeleton_jsonl,
                                     run_paths.metrics_csv, run_paths.summary_json,
                                     run_paths.autopnp_json):
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except OSError:
                            pass

                self.store.upsert_video_cv_run(
                    project_id=project_id,
                    file_id=file_id,
                    status="failed",
                    progress=1.0,
                    message="Failed",
                    error=str(summary.get("reason") or "Unknown error"),
                    config_snapshot=cfg_snapshot,
                    finished_at=now_utc_iso(),
                    summary_json_path=str(artifact_paths.summary_json) if artifact_paths.summary_json.exists() else None,
                    metrics_csv_path=str(artifact_paths.metrics_csv) if artifact_paths.metrics_csv.exists() else None,
                    skeleton_jsonl_path=str(artifact_paths.skeleton_jsonl) if artifact_paths.skeleton_jsonl.exists() else None,
                    pose_csv_path=str(artifact_paths.pose_csv) if artifact_paths.pose_csv.exists() else None,
                    autopnp_json_path=str(artifact_paths.autopnp_json) if artifact_paths.autopnp_json.exists() else None,
                )
                emit_overall(f"{filename}: failed")
                return False
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Skeleton processing crashed for %s", file_row.get("path"))
                with self._batch_progress_lock:
                    progress_by_file[file_id] = 1.0
                self.store.upsert_video_cv_run(
                    project_id=project_id,
                    file_id=file_id,
                    status="failed",
                    progress=1.0,
                    message="Failed",
                    error=str(exc),
                    config_snapshot=cfg_snapshot,
                    finished_at=now_utc_iso(),
                    summary_json_path=str(artifact_paths.summary_json) if artifact_paths.summary_json.exists() else None,
                    metrics_csv_path=str(artifact_paths.metrics_csv) if artifact_paths.metrics_csv.exists() else None,
                    skeleton_jsonl_path=str(artifact_paths.skeleton_jsonl) if artifact_paths.skeleton_jsonl.exists() else None,
                    pose_csv_path=str(artifact_paths.pose_csv) if artifact_paths.pose_csv.exists() else None,
                    autopnp_json_path=str(artifact_paths.autopnp_json) if artifact_paths.autopnp_json.exists() else None,
                )
                emit_overall(f"{filename}: failed")
                return False

        emit_overall(f"Running skeleton batch with {worker_count} parallel video workers")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="cv-video",
        ) as executor:
            futures = [executor.submit(process_one, file_row) for file_row in run_files]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    processed += 1
                else:
                    failed += 1

        emit_overall("Skeleton batch finished")
        return {"processed": processed, "skipped": skipped, "failed": failed, "total": total}

    def preview_frame(self, project_id: str, file_id: str, frame_index: int) -> dict[str, Any]:
        if proc is None:
            raise RuntimeError(f"Reference CV stack is unavailable: {_PROC_IMPORT_ERROR}")
        file_row = self.store.get_file(file_id)
        if file_row is None or str(file_row.get("project_id")) != project_id:
            raise RuntimeError("Video file not found")
        video_path = Path(str(file_row["path"]))
        frame_bgr, fps = _get_video_frame(video_path, frame_index)
        ts_ms = _frame_ts_ms(frame_index, fps)
        config = self.get_project_config(project_id)
        model_path = _resolve_model(config)
        if model_path is None:
            raise RuntimeError("Pose model missing")
        h, w = frame_bgr.shape[:2]
        K_undist = None
        calib_path = _resolve_calibration_path(project_id, config)
        if calib_path is not None:
            K_undist = proc.load_undistorted_intrinsics(str(calib_path), w, h)
        cam_pos, R_wc = proc.get_camera_pose_from_config(config)
        cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
        R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
        filt = proc.normalize_skeleton_filter_params(config.get("skeleton_filter"))
        smoother = None
        if filt.get("enabled", True):
            smoother = proc.SkeletonPlacementKalman(
                fps=max(1.0, fps),
                process_noise_acc=float(filt["process_noise_acc"]),
                measurement_noise=float(filt["measurement_noise"]),
                use_landmark_confidence=bool(filt["use_landmark_confidence"]),
                min_landmark_confidence=float(filt["min_landmark_confidence"]),
                confidence_floor=float(filt["confidence_floor"]),
                confidence_power=float(filt["confidence_power"]),
                max_confidence_noise_scale=float(filt["max_confidence_noise_scale"]),
                gate_sigma=float(filt["gate_sigma"]),
                max_consecutive_misses=int(filt["max_consecutive_misses"]),
                initial_velocity_std=float(filt["initial_velocity_std"]),
                velocity_decay=float(filt["velocity_decay"]),
                max_speed=float(filt["max_speed"]),
                max_measurement_jump=float(filt["max_measurement_jump"]),
                reacquire_frames=int(filt["reacquire_frames"]),
                reacquire_max_jump=float(filt["reacquire_max_jump"]),
            )
        landmarker = None
        try:
            landmarker = proc.vision.PoseLandmarker.create_from_options(
                proc.vision.PoseLandmarkerOptions(
                    base_options=proc.python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=proc.vision.RunningMode.VIDEO,
                )
            )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(proc.mp.Image(image_format=proc.mp.ImageFormat.SRGB, data=frame_rgb), ts_ms)
            pose_packet = proc._pose_result_to_packet(result)
            row_dict = proc._pose_packet_to_row_dict(frame_index, ts_ms, pose_packet)
            placed = None
            metrics = {"trunk_angle": None, "com_x": None, "com_y": None, "com_z": None, "moment_pitch": None, "moment_roll": None}
            if row_dict is not None and K_undist is not None:
                raw = proc.compute_placed_skeleton(row=row_dict, K_undist=K_undist, W=w, H=h, camera_pos=cam_pos, R_wc=R_wc, z_plane_lm24=float(config.get("hip_plane_z", 0.08)), z_plane_lm28=float(config.get("lower_plane_z", 0.0)), lower_landmark=str(config.get("lower_landmark", "ankle")), contact_params=config.get("contact_params"))
                lm_conf = proc._extract_landmark_confidence_from_row(row_dict)
                placed = smoother.smooth(raw, landmark_confidence=lm_conf) if smoother is not None else raw
                if placed is not None:
                    metrics = proc.compute_metrics_for_skeleton(placed, float(config.get("boat_com", -1.114)), float(config.get("athlete_weight", 75.0)))
            overlay = _draw_pose_overlay(frame_bgr, pose_packet)
            ok, jpeg = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ok:
                raise RuntimeError("Failed to encode preview image")
            return {
                "file_id": file_id,
                "frame_index": int(frame_index),
                "timestamp_ms": int(ts_ms),
                "image_b64_jpeg": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                "has_pose": bool(pose_packet is not None),
                "has_skeleton": bool(placed is not None),
                "skeleton_3d": ({int(k): [float(v[0]), float(v[1]), float(v[2])] for k, v in placed.items() if v is not None} if placed is not None else None),
                "metrics": {k: (float(v) if v is not None and np.isfinite(v) else None) for k, v in metrics.items()},
                "camera_position": [float(v) for v in cam_pos.tolist()],
                "camera_R_wc": [[float(x) for x in row] for row in R_wc.tolist()],
            }
        finally:
            if landmarker is not None:
                landmarker.close()

    def autodetect_pnp_pairs(self, project_id: str, file_id: str, frame_index: int) -> dict[str, Any]:
        if proc is None:
            raise RuntimeError(f"Reference CV stack is unavailable: {_PROC_IMPORT_ERROR}")
        file_row = self.store.get_file(file_id)
        if file_row is None or str(file_row.get("project_id")) != project_id:
            raise RuntimeError("Video file not found")
        frame_bgr, _fps = _get_video_frame(Path(str(file_row["path"])), frame_index)
        model = proc._get_auto_camera_pnp_model()
        results = model.predict(source=frame_bgr, verbose=False, conf=0.05, iou=0.6, max_det=4)
        if not results:
            raise RuntimeError("YOLO returned no detections")
        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            raise RuntimeError("No keypoints detected")
        xy = result.keypoints.xy.detach().cpu().numpy()[0]
        conf = result.keypoints.conf.detach().cpu().numpy()[0] if result.keypoints.conf is not None else np.ones((xy.shape[0],), dtype=np.float64)
        n = len(proc.AUTO_CAMERA_PNP_KEYPOINT_LABELS)
        pts = xy[:n].astype(np.float64)
        cf = conf[:n].astype(np.float64)
        mapping = proc._bind_pnp_keypoints_geometry(pts)
        pairs: list[dict[str, Any]] = []
        pid = 1
        for label in proc.AUTO_CAMERA_PNP_KEYPOINT_LABELS:
            idx = int(mapping[label])
            if np.isfinite(cf[idx]) and float(cf[idx]) < float(proc.AUTO_CAMERA_PNP_MIN_KPT_CONF):
                continue
            obj = np.asarray(proc.AUTO_CAMERA_PNP_OBJECT_POINTS[label], dtype=np.float64).reshape(3)
            pairs.append({"id": int(pid), "image_point": [float(pts[idx][0]), float(pts[idx][1])], "object_point": [float(obj[0]), float(obj[1]), float(obj[2])]})
            pid += 1
        return {"frame_index": int(frame_index), "pairs": pairs}

    def solve_pnp(self, project_id: str, file_id: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        if proc is None:
            raise RuntimeError(f"Reference CV stack is unavailable: {_PROC_IMPORT_ERROR}")
        file_row = self.store.get_file(file_id)
        if file_row is None or str(file_row.get("project_id")) != project_id:
            raise RuntimeError("Video file not found")
        config = self.get_project_config(project_id)
        calib_path = _resolve_calibration_path(project_id, config)
        if calib_path is None:
            raise RuntimeError("No calibration file available for PnP")
        cap = cv2.VideoCapture(str(file_row["path"]))
        if not cap.isOpened():
            raise RuntimeError("Unable to open video")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid video dimensions")
        K_undist = proc.load_undistorted_intrinsics(str(calib_path), width, height)
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        for pair in pairs:
            obj = pair.get("object_point")
            img = pair.get("image_point")
            if obj is None or img is None:
                continue
            o = np.asarray(obj, dtype=np.float64).reshape(3)
            i = np.asarray(img, dtype=np.float64).reshape(2)
            if np.all(np.isfinite(o)) and np.all(np.isfinite(i)):
                obj_pts.append(o)
                img_pts.append(i)
        if len(obj_pts) < 4:
            raise RuntimeError(f"At least 4 valid pairs required (got {len(obj_pts)})")
        solved = proc._solve_pnp_pose(np.asarray(obj_pts), np.asarray(img_pts), K_undist)
        if not solved.get("ok"):
            raise RuntimeError(str(solved.get("error") or "solvePnP failed"))
        R_wc_raw = np.asarray(solved["R_wc"], dtype=np.float64).reshape(3, 3)
        cam_pos_raw = np.asarray(solved["camera_pos"], dtype=np.float64).reshape(3)
        raw_angles = proc._camera_pose_angles_from_rwc(R_wc_raw)
        # Roll correction: negate and add +90 offset (matches negated auto-PnP in processing_core)
        _, R_wc_corr = proc.default_camera_pose_and_rotation(pitch=raw_angles["pitch_deg"], yaw=raw_angles["yaw_deg"], roll=-(raw_angles["roll_deg"] + 90.0))
        cam_pos_corr = np.array([cam_pos_raw[0], cam_pos_raw[2], cam_pos_raw[1]], dtype=np.float64)
        angles_corr = proc._camera_pose_angles_from_rwc(R_wc_corr)
        angles_corr["yaw_deg"] = float(proc._wrap_deg(angles_corr["yaw_deg"]))
        angles_corr["roll_deg"] = float(proc._wrap_deg(angles_corr["roll_deg"]))
        return {
            "camera_position": [float(v) for v in cam_pos_corr.tolist()],
            "camera_R_wc": [[float(x) for x in row] for row in np.asarray(R_wc_corr, dtype=np.float64).tolist()],
            "camera_pose_deg": {"pitch_deg": float(angles_corr["pitch_deg"]), "yaw_deg": float(angles_corr["yaw_deg"]), "roll_deg": float(angles_corr["roll_deg"])},
            "mean_reprojection_error_px": float(solved.get("mean_error_px") or 0.0),
            "inlier_reprojection_error_px": float(solved.get("inlier_error_px") or 0.0),
            "num_pairs": int(solved.get("num_pairs") or len(obj_pts)),
            "num_inliers": int(solved.get("num_inliers") or 0),
            "solve_method": str(solved.get("solve_method") or "unknown"),
        }
