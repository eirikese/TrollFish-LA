#!/usr/bin/env python3
"""
Run YOLO pose + PnP on a dual-fisheye .insv video and export a filtered boom-vector
training dataset.

For each raw pose frame, writes one audit CSV row with:
- camera position (x, y, z)
- camera pose (yaw, pitch, roll)

Then it downsamples the successful poses to a 20 fps boom-label CSV. The boom
label is derived from camera orientation, not camera position. Camera position is
kept only for sanity filtering and debugging.

Typical usage:
    python src/predict_insv_pose_30fps_rot180.py --insv insvfiles/VID_20260211_234708_00_006.insv --rotate-180 --device 0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    from auto_pnp_from_insv import (
        CORRESPONDENCE_SETS,
        DEFAULT_CORRESPONDENCE_PROFILE,
        DEFAULT_KPT_INDEX_MAP,
        KEYPOINT_LABELS,
        OBJECT_POINTS,
        attempt_pose_on_equirect_frame,
        choose_best_lens_result,
        get_correspondence_profile,
        inspect_video,
        parse_kpt_index_map,
        prepare_split_pair,
        select_best_detection,
        solve_pose_from_equirect_points,
    )
    from fisheye_to_equirect import (
        DEFAULT_CALIB_A,
        DEFAULT_CALIB_B,
        EquirectProjector,
        ROOT_DIR,
        load_fisheye_calibration,
    )
else:
    from .auto_pnp_from_insv import (
        CORRESPONDENCE_SETS,
        DEFAULT_CORRESPONDENCE_PROFILE,
        DEFAULT_KPT_INDEX_MAP,
        KEYPOINT_LABELS,
        OBJECT_POINTS,
        attempt_pose_on_equirect_frame,
        choose_best_lens_result,
        get_correspondence_profile,
        inspect_video,
        parse_kpt_index_map,
        prepare_split_pair,
        select_best_detection,
        solve_pose_from_equirect_points,
    )
    from .fisheye_to_equirect import (
        DEFAULT_CALIB_A,
        DEFAULT_CALIB_B,
        EquirectProjector,
        ROOT_DIR,
        load_fisheye_calibration,
    )


DEFAULT_OUTPUT_ROOT = ROOT_DIR / "autopnp_runs"
DEFAULT_MODEL = ROOT_DIR / "training_runs" / "360pnp_pose_v7_lightaug-2" / "weights" / "best.pt"
FALLBACK_MODEL = ROOT_DIR / "yolov8n-pose.pt"
DEFAULT_MAST_MOUNT_POINT = (0.0, 0.55, 0.0)


def latest_default_checkpoint() -> Path | None:
    runs_root = ROOT_DIR / "training_runs"
    candidates = list(runs_root.glob("360pnp_pose_v7_lightaug-2*/weights/best.pt"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return existing[0]


def default_model_path() -> Path:
    latest = latest_default_checkpoint()
    if latest is not None:
        return latest
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    return FALLBACK_MODEL


def parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(value).replace(";", ",").replace(" ", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three coordinates, e.g. 0,0.55,0")
    try:
        xyz = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Coordinates must be numeric, e.g. 0,0.55,0") from exc
    if not all(math.isfinite(v) for v in xyz):
        raise argparse.ArgumentTypeError("Coordinates must be finite numbers")
    return xyz  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict camera pose for each frame in a dual-fisheye .insv video using "
            "YOLO keypoints + PnP, then export a filtered boom-vector training CSV."
        )
    )
    parser.add_argument("--insv", type=Path, required=True, help="Input dual-fisheye .insv video")
    rotation_group = parser.add_mutually_exclusive_group()
    rotation_group.add_argument(
        "--rotate-180",
        dest="rotate_180",
        action="store_true",
        help="Rotate each split lens frame by 180 degrees before projection/inference.",
    )
    rotation_group.add_argument(
        "--no-rotate-180",
        dest="rotate_180",
        action="store_false",
        help="Use split lens frames without the 180-degree pre-rotation.",
    )
    parser.set_defaults(rotate_180=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Trained YOLO pose model (.pt). Defaults to training_runs/360pnp_pose_v7_lightaug-2/weights/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to autopnp_runs/<insv-stem>_rot180_boom_dataset or _boom_dataset.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Optional folder for cached split lens videos. Defaults to <output-dir>/split_lenses",
    )
    parser.add_argument("--calib-a", type=Path, default=DEFAULT_CALIB_A, help="Fisheye calibration for lens A")
    parser.add_argument("--calib-b", type=Path, default=DEFAULT_CALIB_B, help="Fisheye calibration for lens B")
    parser.add_argument("--equirect-width", type=int, default=2880, help="Equirectangular output width")
    parser.add_argument("--equirect-height", type=int, default=1440, help="Equirectangular output height")
    parser.add_argument("--model-imgsz", type=int, default=640, help="Square YOLO inference size")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on source frames to read, useful for quick smoke tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='Ultralytics device string, e.g. "0", "cpu", or empty for auto',
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference in Ultralytics. This is usually only useful on CUDA GPUs.",
    )
    parser.add_argument(
        "--inference-batch-frames",
        type=int,
        default=4,
        help=(
            "Number of sampled source frames to project before one YOLO call. "
            "In best/both lens mode the actual YOLO image batch is twice this value. "
            "Raise for better GPU throughput; lower if VRAM/RAM is tight."
        ),
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help=(
            "Disable CPU/GPU overlap. By default the next equirect batch is decoded/projected "
            "in a background thread while YOLO/PnP runs on the current batch."
        ),
    )
    parser.add_argument(
        "--project-workers",
        type=int,
        default=1,
        help=(
            "CPU workers for equirect projection inside each sampled frame. Use 2 to project lens A and B "
            "in parallel when CPU cores are available."
        ),
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=0,
        help="Set OpenCV internal thread count. 0 keeps OpenCV default; try 2-4 with --project-workers 2.",
    )
    parser.add_argument("--det-conf", type=float, default=0.05, help="YOLO confidence threshold")
    parser.add_argument("--det-iou", type=float, default=0.3, help="YOLO NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=4, help="Maximum detections per lens frame")
    parser.add_argument("--min-kpt-conf", type=float, default=0.4, help="Minimum keypoint confidence")
    parser.add_argument("--min-pairs", type=int, default=4, help="Minimum PnP correspondences")
    parser.add_argument(
        "--max-reproj-error-px",
        type=float,
        default=50.0,
        help="Maximum mean reprojection error for accepting a pose",
    )
    parser.add_argument(
        "--inlier-threshold-px",
        type=float,
        default=24.0,
        help="Per-point reprojection threshold for inlier counting",
    )
    parser.add_argument("--fov-a-deg", type=float, default=None, help="Optional fisheye FOV override for lens A")
    parser.add_argument("--fov-b-deg", type=float, default=None, help="Optional fisheye FOV override for lens B")
    parser.add_argument("--yaw-a-deg", type=float, default=0.0, help="Yaw offset for lens A projection")
    parser.add_argument("--pitch-a-deg", type=float, default=0.0, help="Pitch offset for lens A projection")
    parser.add_argument("--roll-a-deg", type=float, default=-90.0, help="Roll offset for lens A projection")
    parser.add_argument("--yaw-b-deg", type=float, default=0.0, help="Yaw offset for lens B projection")
    parser.add_argument("--pitch-b-deg", type=float, default=0.0, help="Pitch offset for lens B projection")
    parser.add_argument("--roll-b-deg", type=float, default=90.0, help="Roll offset for lens B projection")
    parser.add_argument(
        "--lens-mode",
        type=str,
        choices=("best", "both", "a", "b"),
        default="best",
        help=(
            "Lens selection mode: best (evaluate A and B then pick best), both (alias of best), "
            "a only, or b only."
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=60.0,
        help="Raw pose inference FPS before filtering. Use 60 for the default 60 fps -> 20 fps dataset path.",
    )
    parser.add_argument(
        "--dataset-fps",
        type=float,
        default=20.0,
        help="Filtered boom-vector label FPS written to boom_angle_dataset.csv.",
    )
    parser.add_argument(
        "--mast-mount",
        type=parse_xyz,
        default=DEFAULT_MAST_MOUNT_POINT,
        help="Mast boom mount point in hull/world coordinates as x,y,z meters. Default: 0,0.55,0.",
    )
    parser.add_argument(
        "--max-boom-speed-mps",
        type=float,
        default=4.0,
        help=(
            "Drop raw pose samples whose camera-position speed exceeds this value before window averaging. "
            "Use <= 0 to disable speed filtering."
        ),
    )
    parser.add_argument(
        "--min-boom-length-m",
        type=float,
        default=0.1,
        help="Broad sanity gate: drop raw pose samples with mast-to-camera boom length below this. Use <= 0 to disable.",
    )
    parser.add_argument(
        "--max-boom-length-m",
        type=float,
        default=1.5,
        help="Broad sanity gate: drop raw pose samples with mast-to-camera boom length above this. Use <= 0 to disable.",
    )
    parser.add_argument(
        "--min-camera-y-m",
        type=float,
        default=0.0,
        help="Drop raw pose samples with camera_y_m below this before dataset averaging. Use -inf to disable.",
    )
    parser.add_argument(
        "--max-camera-y-m",
        type=float,
        default=1.5,
        help="Drop raw pose samples with camera_y_m above this before dataset averaging. Use inf to disable.",
    )
    parser.add_argument(
        "--max-frame-jump-m",
        type=float,
        default=0.30,
        help=(
            "Drop isolated one-frame position spikes when both neighboring samples are farther than this. "
            "Use <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--jump-gap-sec",
        type=float,
        default=0.08,
        help="Only compare neighboring pose samples for jump filtering when their time gap is at most this.",
    )
    parser.add_argument(
        "--max-angle-jump-deg",
        type=float,
        default=35.0,
        help=(
            "Drop isolated one-frame boom-direction angle spikes when both neighboring directions differ by more than this. "
            "Use <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--speed-gap-sec",
        type=float,
        default=0.25,
        help="Only compute movement speed across consecutive successful poses if their time gap is at most this.",
    )
    parser.add_argument(
        "--outlier-mad-threshold",
        type=float,
        default=3.5,
        help="Median-absolute-deviation threshold for removing position outliers inside each dataset window.",
    )
    parser.add_argument(
        "--max-window-spread-m",
        type=float,
        default=0.35,
        help="Drop a 20 fps output window if kept positions still spread farther than this. Use <= 0 to disable.",
    )
    parser.add_argument(
        "--min-window-samples",
        type=int,
        default=2,
        help="Minimum kept raw pose samples required to write one filtered boom-label row.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=120,
        help="Print progress every N frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--csv-flush-every",
        type=int,
        default=120,
        help="Flush CSV to disk every N written rows. Use 0 to flush every row.",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=0,
        help="Write one labeled debug image every N written output rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--debug-max-images",
        type=int,
        default=60,
        help="Maximum number of debug images to save. Use <= 0 for no limit.",
    )
    parser.add_argument(
        "--debug-image-dir",
        type=Path,
        default=None,
        help="Folder for labeled debug images. Defaults to <output-dir>/debug_predictions.",
    )
    parser.add_argument(
        "--kpt-index-map",
        type=str,
        default="default",
        help=(
            "Comma-separated permutation where each entry gives the model index for the corresponding "
            "AutoPnP keypoint slot (canonical AutoPnP index -> model output index). "
            "Use 'default' for profile defaults (hull9 uses 6,7,8,3,4,5,0,1,2; front6 uses identity)."
        ),
    )
    parser.add_argument(
        "--correspondence-profile",
        type=str,
        choices=tuple(sorted(CORRESPONDENCE_SETS.keys())),
        default=DEFAULT_CORRESPONDENCE_PROFILE,
        help="3D correspondence profile used for PnP mapping.",
    )
    parser.add_argument(
        "--kpt-map-mode",
        type=str,
        choices=("fixed", "auto"),
        default="auto",
        help="Use a fixed keypoint map or evaluate multiple map hypotheses per frame.",
    )
    parser.add_argument(
        "--auto-kpt-maps",
        type=str,
        default="0,1,2,3,4,5,6,7,8;6,7,8,3,4,5,0,1,2",
        help=(
            "Semicolon-separated list of map hypotheses for --kpt-map-mode auto. "
            "Each map uses canonical-slot -> model-index entries."
        ),
    )
    return parser.parse_args()


def build_default_output_dir(insv_path: Path, rotate_180: bool) -> Path:
    suffix = "rot180_boom_dataset" if rotate_180 else "boom_dataset"
    return DEFAULT_OUTPUT_ROOT / f"{insv_path.stem}_{suffix}"


CSV_FIELDNAMES = [
    "frame_index",
    "timestamp_sec",
    "source_fps",
    "target_fps",
    "success",
    "selected_lens",
    "camera_x_m",
    "camera_y_m",
    "camera_z_m",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "mean_reprojection_error_px",
    "median_reprojection_error_px",
    "num_inliers",
    "num_pairs",
    "det_score",
    "mean_kpt_conf",
    "box_conf",
    "det_idx",
    "boom_length_m",
    "detected_object_count",
    "evaluated_candidate_count",
    "successful_candidate_count",
    "candidate_rank_rule",
    "det_class_name",
    "correspondence_profile_used",
    "kpt_index_map_used",
    "reason",
    "lens_a_success",
    "lens_a_reason",
    "lens_a_mean_reprojection_error_px",
    "lens_a_det_class_name",
    "lens_a_correspondence_profile_used",
    "lens_a_kpt_index_map_used",
    "lens_b_success",
    "lens_b_reason",
    "lens_b_mean_reprojection_error_px",
    "lens_b_det_class_name",
    "lens_b_correspondence_profile_used",
    "lens_b_kpt_index_map_used",
]


BOOM_CSV_FIELDNAMES = [
    "sample_index",
    "timestamp_sec",
    "window_start_sec",
    "window_end_sec",
    "source_fps",
    "pose_fps",
    "dataset_fps",
    "success",
    "frame_index_center",
    "frame_index_first",
    "frame_index_last",
    "raw_pose_count",
    "position_filtered_count",
    "length_filtered_count",
    "jump_filtered_count",
    "angle_jump_filtered_count",
    "speed_filtered_count",
    "outlier_filtered_count",
    "used_pose_count",
    "dropped_position_count",
    "dropped_length_count",
    "dropped_jump_count",
    "dropped_angle_jump_count",
    "dropped_speed_count",
    "dropped_outlier_count",
    "window_position_spread_m",
    "camera_x_m",
    "camera_y_m",
    "camera_z_m",
    "mount_x_m",
    "mount_y_m",
    "mount_z_m",
    "boom_label_source",
    "boom_vector_x_m",
    "boom_vector_y_m",
    "boom_vector_z_m",
    "boom_unit_x",
    "boom_unit_y",
    "boom_unit_z",
    "boom_length_m",
    "boom_azimuth_deg",
    "boom_elevation_deg",
    "mean_reprojection_error_px",
    "median_reprojection_error_px",
    "mean_keypoint_conf",
    "mean_det_score",
    "mean_speed_mps",
    "max_speed_mps",
    "selected_lens_counts",
    "frame_indices",
    "reason",
]


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else math.nan


def finite_median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(finite)) if finite else math.nan


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64).reshape(3) / norm


def boom_angles_from_vector(boom_vector: np.ndarray) -> tuple[float, float]:
    vec = np.asarray(boom_vector, dtype=np.float64).reshape(3)
    horizontal = math.hypot(float(vec[0]), float(vec[2]))
    azimuth_deg = math.degrees(math.atan2(float(vec[2]), float(vec[0])))
    elevation_deg = math.degrees(math.atan2(float(vec[1]), horizontal))
    azimuth_deg = ((azimuth_deg + 180.0) % 360.0) - 180.0
    return float(azimuth_deg), float(elevation_deg)


def boom_direction_from_camera_pose(R_wc: np.ndarray, selected_lens: Any) -> np.ndarray:
    R = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    direction = np.asarray([R[0, 2], R[1, 2], R[2, 2]], dtype=np.float64)
    if str(selected_lens or "").strip().upper() == "A":
        direction = -direction
    return normalize_vector(direction)


def angle_between_unit_vectors_deg(a: np.ndarray, b: np.ndarray) -> float:
    va = normalize_vector(np.asarray(a, dtype=np.float64).reshape(3))
    vb = normalize_vector(np.asarray(b, dtype=np.float64).reshape(3))
    if float(np.linalg.norm(va)) <= 1e-12 or float(np.linalg.norm(vb)) <= 1e-12:
        return math.nan
    dot = float(np.dot(va, vb))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def robust_position_inlier_mask(positions: np.ndarray, mad_threshold: float) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    count = int(positions.shape[0])
    if count < 3 or float(mad_threshold) <= 0.0:
        return np.ones((count,), dtype=bool)

    center = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - center.reshape(1, 3), axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    if mad <= 1e-12:
        return distances <= (median_distance + 1e-6)

    robust_sigma = 1.4826 * mad
    return distances <= (median_distance + float(mad_threshold) * robust_sigma)


def quality_weights(samples: list[dict[str, Any]]) -> np.ndarray:
    weights: list[float] = []
    for sample in samples:
        reproj = safe_float(sample.get("mean_reprojection_error_px"), default=math.nan)
        conf = safe_float(sample.get("mean_kpt_conf"), default=math.nan)
        reproj_term = max(reproj, 1.0) if math.isfinite(reproj) else 25.0
        conf_term = min(max(conf, 0.05), 1.0) if math.isfinite(conf) else 0.5
        weights.append(conf_term / reproj_term)

    arr = np.asarray(weights, dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)) or float(np.nansum(arr)) <= 1e-12:
        return np.ones((len(samples),), dtype=np.float64) / max(1, len(samples))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(arr))
    if total <= 1e-12:
        return np.ones((len(samples),), dtype=np.float64) / max(1, len(samples))
    return arr / total


def camera_position_bounds_ok(position: np.ndarray, args: argparse.Namespace) -> tuple[bool, str]:
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(pos)):
        return False, "camera position is not finite"

    min_y = float(getattr(args, "min_camera_y_m", -math.inf))
    max_y = float(getattr(args, "max_camera_y_m", math.inf))
    y = float(pos[1])
    if math.isfinite(min_y) and y < min_y:
        return False, f"camera_y_m {y:.3f} < {min_y:.3f}"
    if math.isfinite(max_y) and y > max_y:
        return False, f"camera_y_m {y:.3f} > {max_y:.3f}"
    return True, ""


def annotate_sample_filters(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    mast_mount: np.ndarray,
    min_boom_length_m: float,
    max_boom_length_m: float,
) -> None:
    for sample in samples:
        position = np.asarray(sample["position_m"], dtype=np.float64).reshape(3)
        position_ok, position_reason = camera_position_bounds_ok(position, args)
        sample["position_bounds_ok"] = bool(position_ok)
        sample["position_bounds_reason"] = position_reason

        boom_length = float(np.linalg.norm(position - mast_mount.reshape(3))) if position_ok else math.nan
        length_ok = bool(position_ok)
        length_reason = ""
        if length_ok and float(min_boom_length_m) > 0.0 and boom_length < float(min_boom_length_m):
            length_ok = False
            length_reason = f"boom length {boom_length:.3f}m < {float(min_boom_length_m):.3f}m"
        if length_ok and float(max_boom_length_m) > 0.0 and boom_length > float(max_boom_length_m):
            length_ok = False
            length_reason = f"boom length {boom_length:.3f}m > {float(max_boom_length_m):.3f}m"

        sample["boom_speed_mps"] = math.nan
        sample["boom_length_raw_m"] = boom_length
        sample["boom_length_ok"] = bool(length_ok)
        sample["boom_length_reason"] = length_reason
        sample["temporal_jump_ok"] = bool(length_ok)
        sample["temporal_jump_reason"] = ""
        direction = normalize_vector(np.asarray(sample.get("boom_direction_unit", np.zeros(3)), dtype=np.float64).reshape(3))
        sample["boom_direction_unit"] = direction
        sample["temporal_angle_ok"] = bool(length_ok and float(np.linalg.norm(direction)) > 1e-12)
        sample["temporal_angle_reason"] = "" if sample["temporal_angle_ok"] else "boom direction is not finite"

    max_frame_jump_m = float(getattr(args, "max_frame_jump_m", 0.0))
    jump_gap_sec = float(getattr(args, "jump_gap_sec", 0.0))
    if max_frame_jump_m > 0.0:
        valid_indices = [idx for idx, sample in enumerate(samples) if bool(sample.get("boom_length_ok", False))]
        for pos_in_valid, sample_idx in enumerate(valid_indices):
            sample = samples[sample_idx]
            timestamp = safe_float(sample.get("timestamp_sec"))
            position = np.asarray(sample["position_m"], dtype=np.float64).reshape(3)

            prev_distance = math.nan
            next_distance = math.nan
            if pos_in_valid > 0:
                prev = samples[valid_indices[pos_in_valid - 1]]
                dt_prev = timestamp - safe_float(prev.get("timestamp_sec"))
                if dt_prev > 0.0 and (jump_gap_sec <= 0.0 or dt_prev <= jump_gap_sec):
                    prev_distance = float(np.linalg.norm(position - np.asarray(prev["position_m"], dtype=np.float64).reshape(3)))

            if pos_in_valid + 1 < len(valid_indices):
                nxt = samples[valid_indices[pos_in_valid + 1]]
                dt_next = safe_float(nxt.get("timestamp_sec")) - timestamp
                if dt_next > 0.0 and (jump_gap_sec <= 0.0 or dt_next <= jump_gap_sec):
                    next_distance = float(np.linalg.norm(position - np.asarray(nxt["position_m"], dtype=np.float64).reshape(3)))

            has_prev = math.isfinite(prev_distance)
            has_next = math.isfinite(next_distance)
            if has_prev and has_next and prev_distance > max_frame_jump_m and next_distance > max_frame_jump_m:
                sample["temporal_jump_ok"] = False
                sample["temporal_jump_reason"] = (
                    f"isolated jump prev={prev_distance:.3f}m next={next_distance:.3f}m > {max_frame_jump_m:.3f}m"
                )

    max_angle_jump_deg = float(getattr(args, "max_angle_jump_deg", 0.0))
    if max_angle_jump_deg > 0.0:
        valid_indices = [
            idx for idx, sample in enumerate(samples)
            if bool(sample.get("boom_length_ok", False)) and bool(sample.get("temporal_jump_ok", False))
        ]
        for pos_in_valid, sample_idx in enumerate(valid_indices):
            sample = samples[sample_idx]
            timestamp = safe_float(sample.get("timestamp_sec"))
            direction = np.asarray(sample["boom_direction_unit"], dtype=np.float64).reshape(3)

            prev_angle = math.nan
            next_angle = math.nan
            if pos_in_valid > 0:
                prev = samples[valid_indices[pos_in_valid - 1]]
                dt_prev = timestamp - safe_float(prev.get("timestamp_sec"))
                if dt_prev > 0.0 and (jump_gap_sec <= 0.0 or dt_prev <= jump_gap_sec):
                    prev_angle = angle_between_unit_vectors_deg(direction, prev["boom_direction_unit"])

            if pos_in_valid + 1 < len(valid_indices):
                nxt = samples[valid_indices[pos_in_valid + 1]]
                dt_next = safe_float(nxt.get("timestamp_sec")) - timestamp
                if dt_next > 0.0 and (jump_gap_sec <= 0.0 or dt_next <= jump_gap_sec):
                    next_angle = angle_between_unit_vectors_deg(direction, nxt["boom_direction_unit"])

            has_prev = math.isfinite(prev_angle)
            has_next = math.isfinite(next_angle)
            if has_prev and has_next and prev_angle > max_angle_jump_deg and next_angle > max_angle_jump_deg:
                sample["temporal_angle_ok"] = False
                sample["temporal_angle_reason"] = (
                    f"isolated angle jump prev={prev_angle:.1f}deg next={next_angle:.1f}deg > {max_angle_jump_deg:.1f}deg"
                )

    previous: dict[str, Any] | None = None
    max_gap_sec = float(getattr(args, "speed_gap_sec", 0.0))
    for sample in samples:
        if not (
            bool(sample.get("boom_length_ok", False))
            and bool(sample.get("temporal_jump_ok", False))
            and bool(sample.get("temporal_angle_ok", False))
        ):
            continue
        position = np.asarray(sample["position_m"], dtype=np.float64).reshape(3)
        if previous is not None:
            dt = safe_float(sample.get("timestamp_sec")) - safe_float(previous.get("timestamp_sec"))
            if dt > 1e-9 and (float(max_gap_sec) <= 0.0 or dt <= float(max_gap_sec)):
                p0 = np.asarray(previous["position_m"], dtype=np.float64).reshape(3)
                sample["boom_speed_mps"] = float(np.linalg.norm(position - p0) / dt)
        previous = sample


def lens_counts_text(samples: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for sample in samples:
        lens = str(sample.get("selected_lens", "") or "-").strip().upper() or "-"
        counts[lens] = counts.get(lens, 0) + 1
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def write_boom_dataset_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOOM_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_boom_dataset_rows(
    samples: list[dict[str, Any]],
    source_fps: float,
    pose_fps: float,
    dataset_fps: float,
    mast_mount: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not samples:
        return [], {
            "raw_success_samples": 0,
            "windows_total": 0,
            "windows_written": 0,
            "windows_dropped_min_samples": 0,
            "windows_dropped_spread": 0,
            "samples_dropped_by_position_bounds": 0,
            "samples_dropped_by_length": 0,
            "samples_dropped_by_jump": 0,
            "samples_dropped_by_angle_jump": 0,
            "samples_dropped_by_speed": 0,
            "samples_dropped_as_outliers": 0,
        }

    max_speed = float(args.max_boom_speed_mps)
    min_boom_length_m = float(args.min_boom_length_m)
    max_boom_length_m = float(args.max_boom_length_m)
    min_window_samples = int(args.min_window_samples)
    max_window_spread_m = float(args.max_window_spread_m)

    annotate_sample_filters(
        samples=samples,
        args=args,
        mast_mount=mast_mount,
        min_boom_length_m=min_boom_length_m,
        max_boom_length_m=max_boom_length_m,
    )
    window_by_index: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        timestamp_sec = safe_float(sample.get("timestamp_sec"))
        if not math.isfinite(timestamp_sec):
            continue
        window_index = int(math.floor(timestamp_sec * float(dataset_fps) + 1e-9))
        window_by_index.setdefault(window_index, []).append(sample)

    rows: list[dict[str, Any]] = []
    stats = {
        "raw_success_samples": int(len(samples)),
        "windows_total": int(len(window_by_index)),
        "windows_written": 0,
        "windows_dropped_min_samples": 0,
        "windows_dropped_spread": 0,
        "samples_dropped_by_position_bounds": 0,
        "samples_dropped_by_length": 0,
        "samples_dropped_by_jump": 0,
        "samples_dropped_by_angle_jump": 0,
        "samples_dropped_by_speed": 0,
        "samples_dropped_as_outliers": 0,
    }

    sample_index = 0

    for window_index in sorted(window_by_index):
        window_samples = window_by_index[window_index]
        position_kept: list[dict[str, Any]] = []
        for sample in window_samples:
            if not bool(sample.get("position_bounds_ok", True)):
                stats["samples_dropped_by_position_bounds"] += 1
                continue
            position_kept.append(sample)

        if len(position_kept) < min_window_samples:
            stats["windows_dropped_min_samples"] += 1
            continue

        length_kept: list[dict[str, Any]] = []
        for sample in position_kept:
            if not bool(sample.get("boom_length_ok", True)):
                stats["samples_dropped_by_length"] += 1
                continue
            length_kept.append(sample)

        if len(length_kept) < min_window_samples:
            stats["windows_dropped_min_samples"] += 1
            continue

        jump_kept: list[dict[str, Any]] = []
        for sample in length_kept:
            if not bool(sample.get("temporal_jump_ok", True)):
                stats["samples_dropped_by_jump"] += 1
                continue
            jump_kept.append(sample)

        if len(jump_kept) < min_window_samples:
            stats["windows_dropped_min_samples"] += 1
            continue

        angle_kept: list[dict[str, Any]] = []
        for sample in jump_kept:
            if not bool(sample.get("temporal_angle_ok", True)):
                stats["samples_dropped_by_angle_jump"] += 1
                continue
            angle_kept.append(sample)

        if len(angle_kept) < min_window_samples:
            stats["windows_dropped_min_samples"] += 1
            continue

        speed_kept: list[dict[str, Any]] = []
        for sample in angle_kept:
            speed = safe_float(sample.get("boom_speed_mps"), default=math.nan)
            if max_speed > 0.0 and math.isfinite(speed) and speed > max_speed:
                stats["samples_dropped_by_speed"] += 1
                continue
            speed_kept.append(sample)

        if len(speed_kept) < min_window_samples:
            stats["windows_dropped_min_samples"] += 1
            continue

        speed_positions = np.asarray([sample["position_m"] for sample in speed_kept], dtype=np.float64).reshape(-1, 3)
        outlier_mask = robust_position_inlier_mask(speed_positions, float(args.outlier_mad_threshold))
        outlier_kept = [sample for sample, keep in zip(speed_kept, outlier_mask.tolist(), strict=False) if bool(keep)]
        dropped_outliers = int(len(speed_kept) - len(outlier_kept))
        stats["samples_dropped_as_outliers"] += dropped_outliers

        if len(outlier_kept) < min_window_samples:
            stats["windows_dropped_min_samples"] += 1
            continue

        positions = np.asarray([sample["position_m"] for sample in outlier_kept], dtype=np.float64).reshape(-1, 3)
        directions = np.asarray([sample["boom_direction_unit"] for sample in outlier_kept], dtype=np.float64).reshape(-1, 3)
        weights = quality_weights(outlier_kept)
        camera_position = np.average(positions, axis=0, weights=weights)
        distances_from_mean = np.linalg.norm(positions - camera_position.reshape(1, 3), axis=1)
        window_spread = float(np.max(distances_from_mean)) if distances_from_mean.size else 0.0
        if max_window_spread_m > 0.0 and window_spread > max_window_spread_m:
            stats["windows_dropped_spread"] += 1
            continue

        boom_unit = normalize_vector(np.average(directions, axis=0, weights=weights))
        if float(np.linalg.norm(boom_unit)) <= 1e-12:
            stats["windows_dropped_min_samples"] += 1
            continue
        boom_vector = boom_unit.copy()
        boom_length = float(np.linalg.norm(boom_vector))
        boom_azimuth_deg, boom_elevation_deg = boom_angles_from_vector(boom_vector)
        timestamps = [safe_float(sample.get("timestamp_sec")) for sample in outlier_kept]
        frame_indices = [int(sample["frame_index"]) for sample in outlier_kept]
        speeds = [safe_float(sample.get("boom_speed_mps")) for sample in outlier_kept]
        mean_reproj_values = [safe_float(sample.get("mean_reprojection_error_px")) for sample in outlier_kept]
        median_reproj_values = [safe_float(sample.get("median_reprojection_error_px")) for sample in outlier_kept]
        conf_values = [safe_float(sample.get("mean_kpt_conf")) for sample in outlier_kept]
        det_values = [safe_float(sample.get("det_score")) for sample in outlier_kept]

        window_start_sec = float(window_index) / float(dataset_fps)
        window_end_sec = float(window_index + 1) / float(dataset_fps)
        row = {
            "sample_index": int(sample_index),
            "timestamp_sec": float((window_start_sec + window_end_sec) * 0.5),
            "window_start_sec": float(window_start_sec),
            "window_end_sec": float(window_end_sec),
            "source_fps": float(source_fps),
            "pose_fps": float(pose_fps),
            "dataset_fps": float(dataset_fps),
            "success": True,
            "frame_index_center": int(round(float(np.mean(frame_indices)))),
            "frame_index_first": int(min(frame_indices)),
            "frame_index_last": int(max(frame_indices)),
            "raw_pose_count": int(len(window_samples)),
            "position_filtered_count": int(len(position_kept)),
            "length_filtered_count": int(len(length_kept)),
            "jump_filtered_count": int(len(jump_kept)),
            "angle_jump_filtered_count": int(len(angle_kept)),
            "speed_filtered_count": int(len(speed_kept)),
            "outlier_filtered_count": int(len(outlier_kept)),
            "used_pose_count": int(len(outlier_kept)),
            "dropped_position_count": int(len(window_samples) - len(position_kept)),
            "dropped_length_count": int(len(position_kept) - len(length_kept)),
            "dropped_jump_count": int(len(length_kept) - len(jump_kept)),
            "dropped_angle_jump_count": int(len(jump_kept) - len(angle_kept)),
            "dropped_speed_count": int(len(angle_kept) - len(speed_kept)),
            "dropped_outlier_count": int(dropped_outliers),
            "window_position_spread_m": float(window_spread),
            "camera_x_m": float(camera_position[0]),
            "camera_y_m": float(camera_position[1]),
            "camera_z_m": float(camera_position[2]),
            "mount_x_m": float(mast_mount[0]),
            "mount_y_m": float(mast_mount[1]),
            "mount_z_m": float(mast_mount[2]),
            "boom_label_source": "camera_orientation",
            "boom_vector_x_m": float(boom_vector[0]),
            "boom_vector_y_m": float(boom_vector[1]),
            "boom_vector_z_m": float(boom_vector[2]),
            "boom_unit_x": float(boom_unit[0]),
            "boom_unit_y": float(boom_unit[1]),
            "boom_unit_z": float(boom_unit[2]),
            "boom_length_m": float(boom_length),
            "boom_azimuth_deg": float(boom_azimuth_deg),
            "boom_elevation_deg": float(boom_elevation_deg),
            "mean_reprojection_error_px": finite_mean(mean_reproj_values),
            "median_reprojection_error_px": finite_median(median_reproj_values),
            "mean_keypoint_conf": finite_mean(conf_values),
            "mean_det_score": finite_mean(det_values),
            "mean_speed_mps": finite_mean(speeds),
            "max_speed_mps": max([speed for speed in speeds if math.isfinite(speed)], default=math.nan),
            "selected_lens_counts": lens_counts_text(outlier_kept),
            "frame_indices": ";".join(str(frame_index) for frame_index in frame_indices),
            "reason": "ok",
        }
        rows.append(row)
        sample_index += 1

    stats["windows_written"] = int(len(rows))
    return rows, stats


def canonicalize_kpt_map(map_str: str, expected_keypoints: int) -> str:
    arr = parse_kpt_index_map(map_str, expected_keypoints)
    return ",".join(str(int(v)) for v in arr.tolist())


def build_kpt_map_candidates(args: argparse.Namespace, expected_keypoints: int) -> list[str]:
    if args.kpt_map_mode == "fixed":
        return [canonicalize_kpt_map(args.kpt_index_map, expected_keypoints)]

    raw_parts = [part.strip() for part in str(args.auto_kpt_maps).split(";") if part.strip()]
    if not raw_parts:
        raise ValueError("--auto-kpt-maps must include at least one map when --kpt-map-mode auto")

    normalized: list[str] = []
    for part in raw_parts:
        if str(part).strip().lower() in ("default", "auto"):
            normalized.append("default")
        else:
            normalized.append(canonicalize_kpt_map(part, expected_keypoints))

    unique: list[str] = []
    for item in normalized:
        if item not in unique:
            unique.append(item)
    if "default" not in unique:
        unique.insert(0, "default")
    return unique


def choose_best_map_result(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in candidates if item.get("success")]
    if successes:
        successes.sort(
            key=lambda item: (
                float(item.get("mean_reprojection_error_px", float("inf"))),
                -int(item.get("num_inliers", 0)),
                -int(item.get("num_pairs", 0)),
                -float(item.get("mean_kpt_conf", 0.0)),
                -float(item.get("det_score", 0.0)),
            )
        )
        return dict(successes[0])

    failed = [dict(item) for item in candidates]
    failed.sort(
        key=lambda item: (
            -float(item.get("mean_kpt_conf", 0.0)),
            -float(item.get("det_score", 0.0)),
        )
    )
    return failed[0] if failed else {"success": False, "lens": "", "reason": "No map candidates were evaluated"}


def get_class_name_from_yolo_result(yolo_result: Any, class_id: int) -> str:
    names = getattr(yolo_result, "names", None)
    if isinstance(names, dict):
        return str(names.get(int(class_id), ""))
    if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
        return str(names[int(class_id)])
    return ""


def extract_detection_candidates(yolo_result: Any) -> tuple[list[dict[str, Any]], str]:
    if yolo_result.keypoints is None or yolo_result.keypoints.xy is None or len(yolo_result.keypoints.xy) == 0:
        return [], "No keypoints detected"

    xy_all = yolo_result.keypoints.xy.detach().cpu().numpy()
    if xy_all.ndim != 3 or xy_all.shape[2] != 2:
        return [], "Unexpected YOLO keypoint tensor shape"

    conf_tensor = yolo_result.keypoints.conf
    if conf_tensor is not None:
        conf_all = conf_tensor.detach().cpu().numpy().astype(np.float64)
    else:
        conf_all = np.ones((xy_all.shape[0], xy_all.shape[1]), dtype=np.float64)

    box_conf = np.ones((xy_all.shape[0],), dtype=np.float64)
    if yolo_result.boxes is not None and yolo_result.boxes.conf is not None and len(yolo_result.boxes.conf) == xy_all.shape[0]:
        box_conf = yolo_result.boxes.conf.detach().cpu().numpy().astype(np.float64)

    class_ids = np.full((xy_all.shape[0],), -1, dtype=np.int64)
    if yolo_result.boxes is not None and yolo_result.boxes.cls is not None and len(yolo_result.boxes.cls) == xy_all.shape[0]:
        class_ids = yolo_result.boxes.cls.detach().cpu().numpy().astype(np.int64)

    detections: list[dict[str, Any]] = []
    for det_idx in range(int(xy_all.shape[0])):
        conf = conf_all[det_idx].astype(np.float64)
        mean_conf = float(np.mean(np.nan_to_num(conf, nan=0.0))) if conf.size else 0.0
        class_id = int(class_ids[det_idx])
        detections.append(
            {
                "det_idx": int(det_idx),
                "points": xy_all[det_idx].astype(np.float64),
                "conf": conf,
                "box_conf": float(box_conf[det_idx]),
                "class_id": class_id,
                "class_name": get_class_name_from_yolo_result(yolo_result, class_id),
                "det_score_base": float(box_conf[det_idx]) * mean_conf,
                "mean_kpt_conf_base": mean_conf,
            }
        )

    detections.sort(
        key=lambda item: (
            -float(item.get("det_score_base", 0.0)),
            -float(item.get("mean_kpt_conf_base", 0.0)),
            int(item.get("det_idx", 0)),
        )
    )
    return detections, ""


def compatible_kpt_map_candidates(map_candidates: list[str], expected_keypoints: int) -> list[tuple[str, np.ndarray]]:
    compatible: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for map_candidate in map_candidates:
        try:
            arr = parse_kpt_index_map(map_candidate, expected_keypoints)
        except ValueError:
            continue
        canonical = ",".join(str(int(v)) for v in arr.tolist())
        if canonical in seen:
            continue
        seen.add(canonical)
        label = "default" if str(map_candidate).strip().lower() in ("", "default", "auto") else canonical
        compatible.append((label, arr))

    if not compatible:
        arr = parse_kpt_index_map("default", expected_keypoints)
        compatible.append(("default", arr))
    return compatible


def boom_length_for_position(camera_position: np.ndarray, args: argparse.Namespace) -> float:
    mast_mount = np.asarray(getattr(args, "mast_mount", DEFAULT_MAST_MOUNT_POINT), dtype=np.float64).reshape(3)
    return float(np.linalg.norm(np.asarray(camera_position, dtype=np.float64).reshape(3) - mast_mount))


def boom_length_is_plausible(boom_length_m: float, args: argparse.Namespace) -> tuple[bool, str]:
    min_length = float(getattr(args, "min_boom_length_m", 0.0))
    max_length = float(getattr(args, "max_boom_length_m", 0.0))
    if min_length > 0.0 and float(boom_length_m) < min_length:
        return False, f"boom length {boom_length_m:.2f}m < {min_length:.2f}m"
    if max_length > 0.0 and float(boom_length_m) > max_length:
        return False, f"boom length {boom_length_m:.2f}m > {max_length:.2f}m"
    return True, ""


def normalized_detection_class_name(class_name: Any) -> str:
    text = str(class_name or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def profiles_for_detection_class(class_name: Any, available_profiles: list[str]) -> list[str]:
    normalized = normalized_detection_class_name(class_name)
    if "front" in normalized:
        return [profile for profile in available_profiles if profile == "front6"]
    if "back" in normalized:
        return [profile for profile in available_profiles if profile == "hull9"]
    return list(available_profiles)


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int, int, float, float, int]:
    return (
        float(candidate.get("mean_reprojection_error_px", math.inf)),
        -int(candidate.get("num_inliers", 0)),
        -int(candidate.get("num_pairs", 0)),
        -float(candidate.get("mean_kpt_conf", 0.0)),
        -float(candidate.get("det_score", 0.0)),
        int(candidate.get("det_idx", 0)),
    )


def choose_best_pose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [dict(candidate) for candidate in candidates if candidate.get("success")]
    if successes:
        successes.sort(key=candidate_sort_key)
        best = dict(successes[0])
        best["reason"] = "ok"
        best["successful_candidate_count"] = int(len(successes))
        return best

    failed = [dict(candidate) for candidate in candidates]
    failed.sort(
        key=lambda item: (
            -float(item.get("mean_kpt_conf", item.get("mean_kpt_conf_base", 0.0))),
            -float(item.get("det_score", item.get("det_score_base", 0.0))),
            int(item.get("det_idx", 0)),
        )
    )
    return failed[0] if failed else {"success": False, "lens": "", "reason": "No pose candidates were evaluated"}


def choose_best_lens_pose_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [dict(result) for result in results if result.get("success")]
    if successes:
        successes.sort(key=candidate_sort_key)
        chosen = dict(successes[0])
        chosen["reason"] = "ok"
        return chosen

    reason = " | ".join(f"{result.get('lens', '?')}: {result.get('reason', 'failed')}" for result in results)
    return {"success": False, "lens": "", "reason": reason}


def attempt_pose_with_map_hypotheses(
    model: Any,
    equirect_frame: np.ndarray,
    lens_name: str,
    args: argparse.Namespace,
    map_candidates: list[str],
) -> dict[str, Any]:
    candidate_results: list[dict[str, Any]] = []
    for map_candidate in map_candidates:
        local_args = argparse.Namespace(**vars(args))
        local_args.kpt_index_map = map_candidate
        result = attempt_pose_on_equirect_frame(model, equirect_frame, lens_name, local_args)
        result_with_map = dict(result)
        result_with_map["kpt_index_map_used"] = map_candidate
        candidate_results.append(result_with_map)

    return choose_best_map_result(candidate_results)


def attempt_pose_from_yolo_result_for_map(
    yolo_result: Any,
    frame_shape: tuple[int, int],
    lens_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])

    profile_attempt_order = [profile for profile in ("hull9", "front6") if profile in CORRESPONDENCE_SETS]
    if not profile_attempt_order:
        return {"success": False, "lens": lens_name, "reason": "No correspondence profiles are configured"}

    max_expected_keypoints = max(len(get_correspondence_profile(profile)[0]) for profile in profile_attempt_order)
    base_selection = select_best_detection(yolo_result, expected_keypoints=max_expected_keypoints)
    if not base_selection["ok"]:
        return {"success": False, "lens": lens_name, "reason": str(base_selection["reason"])}

    profile_failure_reasons: list[str] = []
    for active_profile in profile_attempt_order:
        labels, object_points = get_correspondence_profile(active_profile)
        expected_keypoints = len(labels)

        try:
            kpt_index_map = parse_kpt_index_map(
                getattr(args, "kpt_index_map", DEFAULT_KPT_INDEX_MAP),
                expected_keypoints,
            )
        except ValueError as exc:
            profile_failure_reasons.append(f"{active_profile}: keypoint index map incompatible ({exc})")
            continue

        selection = select_best_detection(yolo_result, expected_keypoints=expected_keypoints)
        if not selection["ok"]:
            profile_failure_reasons.append(f"{active_profile}: {selection['reason']}")
            continue

        mapped_points = selection["points"][kpt_index_map]
        mapped_conf = selection["conf"][kpt_index_map]
        equirect_points = np.asarray(mapped_points, dtype=np.float64).reshape(-1, 2)
        valid_mask = np.isfinite(mapped_conf) & (mapped_conf >= float(args.min_kpt_conf))

        debug_payload = {
            "all_keypoints_equirect_xy": equirect_points.tolist(),
            "all_keypoint_conf": mapped_conf.astype(np.float64).tolist(),
            "valid_keypoint_mask": valid_mask.astype(bool).tolist(),
            "keypoint_labels": list(labels),
            "correspondence_profile": str(active_profile),
            "keypoint_index_map": [int(v) for v in kpt_index_map.tolist()],
            "det_class_id": int(selection.get("class_id", -1)),
            "det_class_name": str(selection.get("class_name", "")),
            "used_keypoint_labels": [
                labels[idx] for idx, keep in enumerate(valid_mask.tolist()) if bool(keep)
            ],
            "profile_attempt_order": list(profile_attempt_order),
        }

        filtered_object_points = object_points[valid_mask]
        filtered_equirect_points = equirect_points[valid_mask]
        if len(filtered_object_points) < int(args.min_pairs):
            profile_failure_reasons.append(
                f"{active_profile}: Only {len(filtered_object_points)} keypoints >= confidence threshold"
            )
            continue

        solved = solve_pose_from_equirect_points(
            object_points=filtered_object_points,
            equirect_points=filtered_equirect_points,
            eq_width=frame_w,
            eq_height=frame_h,
            inlier_threshold_px=float(args.inlier_threshold_px),
        )
        if not solved["ok"]:
            profile_failure_reasons.append(f"{active_profile}: {solved['reason']}")
            continue

        if float(solved["mean_reprojection_error_px"]) > float(args.max_reproj_error_px):
            profile_failure_reasons.append(
                f"{active_profile}: Mean reprojection error {solved['mean_reprojection_error_px']:.2f}px "
                f"> {float(args.max_reproj_error_px):.2f}px"
            )
            continue

        return {
            "success": True,
            "lens": lens_name,
            "camera_position": solved["camera_position"],
            "R_wc": solved["R_wc"],
            "rvec": solved["rvec"],
            "tvec": solved["tvec"],
            "camera_pose_deg": solved["camera_pose_deg"],
            "num_pairs": int(solved["num_pairs"]),
            "num_inliers": int(solved["num_inliers"]),
            "mean_reprojection_error_px": float(solved["mean_reprojection_error_px"]),
            "median_reprojection_error_px": float(solved["median_reprojection_error_px"]),
            "det_score": float(selection["det_score"]),
            "mean_kpt_conf": float(selection["mean_kpt_conf"]),
            "per_point_errors_px": solved["per_point_errors_px"],
            "projected_points": solved["projected_points"],
            **debug_payload,
        }

    final_reason = " ; ".join(profile_failure_reasons) if profile_failure_reasons else "All correspondence profiles failed"
    return {
        "success": False,
        "lens": lens_name,
        "reason": final_reason,
        "det_class_id": int(base_selection.get("class_id", -1)),
        "det_class_name": str(base_selection.get("class_name", "")),
        "profile_attempt_order": list(profile_attempt_order),
    }


def attempt_pose_from_yolo_result_with_map_hypotheses(
    yolo_result: Any,
    frame_shape: tuple[int, int],
    lens_name: str,
    args: argparse.Namespace,
    map_candidates: list[str],
) -> dict[str, Any]:
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])

    profile_attempt_order = [profile for profile in ("hull9", "front6") if profile in CORRESPONDENCE_SETS]
    if not profile_attempt_order:
        return {"success": False, "lens": lens_name, "reason": "No correspondence profiles are configured"}

    detections, detection_reason = extract_detection_candidates(yolo_result)
    if not detections:
        return {
            "success": False,
            "lens": lens_name,
            "reason": detection_reason or "No YOLO detections",
            "detected_object_count": 0,
            "evaluated_candidate_count": 0,
            "successful_candidate_count": 0,
        }

    success_candidates: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    evaluated_candidate_count = 0

    def count_failure(reason: str) -> None:
        failure_counts[reason] = failure_counts.get(reason, 0) + 1

    for detection in detections:
        detection_profiles = profiles_for_detection_class(detection.get("class_name", ""), profile_attempt_order)
        if not detection_profiles:
            count_failure(f"{detection.get('class_name', '') or 'class'}: no compatible correspondence profile")
            continue

        for active_profile in detection_profiles:
            labels, object_points = get_correspondence_profile(active_profile)
            expected_keypoints = len(labels)
            raw_points = np.asarray(detection["points"], dtype=np.float64)
            raw_conf = np.asarray(detection["conf"], dtype=np.float64)
            if raw_points.shape[0] < expected_keypoints or raw_conf.shape[0] < expected_keypoints:
                count_failure(f"{active_profile}: model returned too few keypoints")
                continue

            profile_points = raw_points[:expected_keypoints, :]
            profile_conf = raw_conf[:expected_keypoints]
            for map_label, kpt_index_map in compatible_kpt_map_candidates(map_candidates, expected_keypoints):
                evaluated_candidate_count += 1
                mapped_points = profile_points[kpt_index_map]
                mapped_conf = profile_conf[kpt_index_map]
                equirect_points = np.asarray(mapped_points, dtype=np.float64).reshape(-1, 2)
                valid_mask = np.isfinite(mapped_conf) & (mapped_conf >= float(args.min_kpt_conf))
                mean_kpt_conf = float(np.mean(np.nan_to_num(mapped_conf, nan=0.0))) if mapped_conf.size else 0.0
                det_score = float(detection["box_conf"]) * mean_kpt_conf

                debug_payload = {
                    "all_keypoints_equirect_xy": equirect_points.tolist(),
                    "all_keypoint_conf": mapped_conf.astype(np.float64).tolist(),
                    "valid_keypoint_mask": valid_mask.astype(bool).tolist(),
                    "keypoint_labels": list(labels),
                    "correspondence_profile": str(active_profile),
                    "keypoint_index_map": [int(v) for v in kpt_index_map.tolist()],
                    "kpt_index_map_used": str(map_label),
                    "det_idx": int(detection["det_idx"]),
                    "det_class_id": int(detection.get("class_id", -1)),
                    "det_class_name": str(detection.get("class_name", "")),
                    "used_keypoint_labels": [
                        labels[idx] for idx, keep in enumerate(valid_mask.tolist()) if bool(keep)
                    ],
                    "profile_attempt_order": list(profile_attempt_order),
                    "profiles_considered_for_detection": list(detection_profiles),
                }

                filtered_object_points = object_points[valid_mask]
                filtered_equirect_points = equirect_points[valid_mask]
                if len(filtered_object_points) < int(args.min_pairs):
                    count_failure(
                        f"{active_profile}: only {len(filtered_object_points)} keypoints >= confidence threshold"
                    )
                    continue

                solved = solve_pose_from_equirect_points(
                    object_points=filtered_object_points,
                    equirect_points=filtered_equirect_points,
                    eq_width=frame_w,
                    eq_height=frame_h,
                    inlier_threshold_px=float(args.inlier_threshold_px),
                )
                if not solved["ok"]:
                    count_failure(f"{active_profile}: {solved['reason']}")
                    continue

                mean_reproj = float(solved["mean_reprojection_error_px"])
                if mean_reproj > float(args.max_reproj_error_px):
                    count_failure(f"{active_profile}: reprojection error above threshold")
                    continue

                position_ok, position_reason = camera_position_bounds_ok(solved["camera_position"], args)
                if not position_ok:
                    count_failure(f"{active_profile}: {position_reason}")
                    continue

                boom_length_m = boom_length_for_position(solved["camera_position"], args)
                boom_ok, boom_reason = boom_length_is_plausible(boom_length_m, args)
                if not boom_ok:
                    count_failure(f"{active_profile}: {boom_reason}")
                    continue

                success_candidates.append(
                    {
                        "success": True,
                        "lens": lens_name,
                        "camera_position": solved["camera_position"],
                        "R_wc": solved["R_wc"],
                        "rvec": solved["rvec"],
                        "tvec": solved["tvec"],
                        "camera_pose_deg": solved["camera_pose_deg"],
                        "num_pairs": int(solved["num_pairs"]),
                        "num_inliers": int(solved["num_inliers"]),
                        "mean_reprojection_error_px": mean_reproj,
                        "median_reprojection_error_px": float(solved["median_reprojection_error_px"]),
                        "det_score": float(det_score),
                        "mean_kpt_conf": float(mean_kpt_conf),
                        "box_conf": float(detection["box_conf"]),
                        "boom_length_m": float(boom_length_m),
                        "per_point_errors_px": solved["per_point_errors_px"],
                        "projected_points": solved["projected_points"],
                        **debug_payload,
                    }
                )

    if success_candidates:
        chosen = choose_best_pose_candidate(success_candidates)
        chosen["detected_object_count"] = int(len(detections))
        chosen["evaluated_candidate_count"] = int(evaluated_candidate_count)
        chosen["candidate_rank_rule"] = "mean_reprojection_error, inliers, pairs, keypoint_conf, det_score"
        return chosen

    top_failures = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    failure_text = "; ".join(f"{reason} ({count})" for reason, count in top_failures)
    return {
        "success": False,
        "lens": lens_name,
        "reason": (
            f"No physically plausible PnP candidates among {evaluated_candidate_count} evaluated"
            + (f": {failure_text}" if failure_text else "")
        ),
        "detected_object_count": int(len(detections)),
        "evaluated_candidate_count": int(evaluated_candidate_count),
        "successful_candidate_count": 0,
        "det_class_id": int(detections[0].get("class_id", -1)),
        "det_class_name": str(detections[0].get("class_name", "")),
        "profile_attempt_order": list(profile_attempt_order),
    }


def predict_lens_pose_results(
    model: Any,
    equirect_by_lens: dict[str, np.ndarray],
    args: argparse.Namespace,
    map_candidates: list[str],
    lens_mode_effective: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return predict_lens_pose_results_batch(
        model=model,
        equirect_batch=[equirect_by_lens],
        args=args,
        map_candidates=map_candidates,
        lens_mode_effective=lens_mode_effective,
        timing_sec=None,
    )[0]


def skipped_lens_result(lens_name: str) -> dict[str, Any]:
    return {
        "success": False,
        "lens": lens_name,
        "reason": "Skipped by --lens-mode",
        "kpt_index_map_used": "",
    }


def predict_lens_pose_results_batch(
    model: Any,
    equirect_batch: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
    map_candidates: list[str],
    lens_mode_effective: str,
    timing_sec: dict[str, float] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_item = [
        {
            "A": skipped_lens_result("A"),
            "B": skipped_lens_result("B"),
        }
        for _ in equirect_batch
    ]

    sources: list[np.ndarray] = []
    source_meta: list[tuple[int, str]] = []
    for item_index, equirect_by_lens in enumerate(equirect_batch):
        active_lenses = [
            lens for lens in ("A", "B")
            if lens in equirect_by_lens and lens_mode_effective in ("best", lens.lower())
        ]
        for lens in active_lenses:
            sources.append(equirect_by_lens[lens])
            source_meta.append((item_index, lens))

    if not sources:
        return [(item["A"], item["B"]) for item in by_item]

    predict_kwargs: dict[str, Any] = {
        "source": sources,
        "imgsz": int(args.model_imgsz),
        "device": args.device,
        "verbose": False,
        "conf": float(args.det_conf),
        "iou": float(args.det_iou),
        "max_det": int(args.max_det),
        "batch": int(len(sources)),
    }
    if bool(getattr(args, "half", False)):
        predict_kwargs["half"] = True

    t0 = time.perf_counter()
    yolo_results = model.predict(**predict_kwargs)
    if timing_sec is not None:
        timing_sec["yolo_infer"] = timing_sec.get("yolo_infer", 0.0) + (time.perf_counter() - t0)

    t1 = time.perf_counter()
    for (item_index, lens_name), yolo_result in zip(source_meta, yolo_results, strict=False):
        frame = equirect_batch[item_index][lens_name]
        by_item[item_index][lens_name] = attempt_pose_from_yolo_result_with_map_hypotheses(
            yolo_result=yolo_result,
            frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
            lens_name=lens_name,
            args=args,
            map_candidates=map_candidates,
        )
    if timing_sec is not None:
        timing_sec["pnp_postprocess"] = timing_sec.get("pnp_postprocess", 0.0) + (time.perf_counter() - t1)

    return [(item["A"], item["B"]) for item in by_item]


def _to_int_point(point_xy: Any) -> tuple[int, int] | None:
    if point_xy is None:
        return None
    try:
        x = float(point_xy[0])
        y = float(point_xy[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return int(round(x)), int(round(y))


def draw_lens_debug_overlay(
    frame: np.ndarray,
    lens_name: str,
    result: dict[str, Any],
    is_selected: bool,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    border_color = (0, 255, 0) if is_selected else (160, 160, 160)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border_color, 3)

    success = bool(result.get("success", False))
    reason = str(result.get("reason", ""))
    cv2.putText(
        canvas,
        f"Lens {lens_name} | {'OK' if success else 'FAIL'}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (30, 245, 30) if success else (40, 40, 255),
        2,
        cv2.LINE_AA,
    )
    if reason:
        cv2.putText(
            canvas,
            reason[:120],
            (16, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

    det_class = str(result.get("det_class_name", ""))
    profile_used = str(result.get("correspondence_profile", ""))
    if det_class or profile_used:
        cv2.putText(
            canvas,
            f"class={det_class or '-'} profile={profile_used or '-'}",
            (16, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

    points = result.get("all_keypoints_equirect_xy") or []
    conf = result.get("all_keypoint_conf") or []
    valid_mask = result.get("valid_keypoint_mask") or []
    labels = result.get("keypoint_labels") or KEYPOINT_LABELS

    for idx, point_xy in enumerate(points):
        p = _to_int_point(point_xy)
        if p is None:
            continue
        is_valid = bool(valid_mask[idx]) if idx < len(valid_mask) else False
        confidence = float(conf[idx]) if idx < len(conf) else float("nan")
        color = (0, 220, 0) if is_valid else (0, 0, 255)

        cv2.circle(canvas, p, 7, color, -1, lineType=cv2.LINE_AA)
        label_name = str(labels[idx]) if idx < len(labels) else f"kp{idx}"
        cv2.putText(
            canvas,
            f"{label_name}:{confidence:.2f}",
            (p[0] + 9, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas


def maybe_write_debug_image(
    debug_dir: Path,
    frame_index: int,
    timestamp_sec: float,
    selected_lens: str,
    eq_a: np.ndarray,
    eq_b: np.ndarray,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
) -> Path:
    overlay_a = draw_lens_debug_overlay(eq_a, "A", result_a, selected_lens == "A")
    overlay_b = draw_lens_debug_overlay(eq_b, "B", result_b, selected_lens == "B")

    # Keep debug images compact while preserving enough detail for labels.
    target_width = 1280
    out_h = max(1, int(round(float(target_width) * float(overlay_a.shape[0]) / float(max(overlay_a.shape[1], 1)))))
    overlay_a = cv2.resize(overlay_a, (target_width, out_h), interpolation=cv2.INTER_AREA)
    overlay_b = cv2.resize(overlay_b, (target_width, out_h), interpolation=cv2.INTER_AREA)

    combined = cv2.vconcat([overlay_a, overlay_b])
    cv2.putText(
        combined,
        f"frame={frame_index}  t={timestamp_sec:.3f}s  selected_lens={selected_lens or '-'}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"frame_{frame_index:07d}_t{timestamp_sec:010.3f}_lens{selected_lens or 'none'}.jpg"
    if not cv2.imwrite(str(out_path), combined):
        raise RuntimeError(f"Failed to write debug image: {out_path}")
    return out_path


def main() -> int:
    args = parse_args()
    insv_path = args.insv.resolve()
    model_path = args.model.resolve()

    if not insv_path.exists():
        raise FileNotFoundError(f".insv file not found: {insv_path}")
    if insv_path.suffix.lower() != ".insv":
        raise ValueError(f"Expected an .insv file, got: {insv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if args.target_fps <= 0.0:
        raise ValueError("--target-fps must be positive.")
    if args.dataset_fps <= 0.0:
        raise ValueError("--dataset-fps must be positive.")
    if args.max_frames is not None and int(args.max_frames) <= 0:
        raise ValueError("--max-frames must be positive when provided.")
    if int(args.inference_batch_frames) <= 0:
        raise ValueError("--inference-batch-frames must be positive.")
    if int(args.project_workers) <= 0:
        raise ValueError("--project-workers must be positive.")
    if int(args.opencv_threads) < 0:
        raise ValueError("--opencv-threads must be >= 0.")
    if int(args.opencv_threads) > 0:
        cv2.setNumThreads(int(args.opencv_threads))
    if args.min_window_samples <= 0:
        raise ValueError("--min-window-samples must be positive.")
    if args.min_boom_length_m > 0.0 and args.max_boom_length_m > 0.0 and args.min_boom_length_m > args.max_boom_length_m:
        raise ValueError("--min-boom-length-m cannot be greater than --max-boom-length-m.")
    if float(args.min_camera_y_m) > float(args.max_camera_y_m):
        raise ValueError("--min-camera-y-m cannot be greater than --max-camera-y-m.")
    if args.jump_gap_sec < 0.0:
        raise ValueError("--jump-gap-sec must be >= 0.")
    if args.speed_gap_sec < 0.0:
        raise ValueError("--speed-gap-sec must be >= 0.")
    if args.debug_every < 0:
        raise ValueError("--debug-every must be >= 0.")
    mast_mount = np.asarray(args.mast_mount, dtype=np.float64).reshape(3)
    profile_labels, profile_object_points = get_correspondence_profile(args.correspondence_profile)
    map_candidates = build_kpt_map_candidates(args, len(profile_labels))
    kpt_index_map = parse_kpt_index_map(map_candidates[0], len(profile_labels))

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else build_default_output_dir(insv_path, bool(args.rotate_180))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_root = args.split_dir.resolve() if args.split_dir is not None else output_dir / "split_lenses"
    debug_image_dir = args.debug_image_dir.resolve() if args.debug_image_dir is not None else output_dir / "debug_predictions"

    lens_a_path, lens_b_path = prepare_split_pair(insv_path, split_root)
    source_size_a, fps_a, frame_count_a = inspect_video(lens_a_path)
    source_size_b, fps_b, frame_count_b = inspect_video(lens_b_path)
    if source_size_a != source_size_b:
        raise RuntimeError(f"Lens resolutions do not match: A={source_size_a}, B={source_size_b}")

    fps = min(float(fps_a), float(fps_b))
    if fps <= 0.0:
        raise RuntimeError("Invalid source FPS reported by input videos.")

    effective_target_fps = float(min(args.target_fps, fps))
    if args.target_fps > fps + 1e-6:
        print(
            f"[WARN] target-fps={args.target_fps:.3f} is above source fps={fps:.3f}. "
            f"Using source fps ({fps:.3f})."
        )
    effective_dataset_fps = float(min(args.dataset_fps, effective_target_fps))
    if args.dataset_fps > effective_target_fps + 1e-6:
        print(
            f"[WARN] dataset-fps={args.dataset_fps:.3f} is above pose fps={effective_target_fps:.3f}. "
            f"Using pose fps ({effective_target_fps:.3f})."
        )

    total_frames = min(int(frame_count_a), int(frame_count_b))
    csv_path = output_dir / "camera_pose_per_frame.csv"
    boom_csv_path = output_dir / "boom_angle_dataset.csv"
    summary_path = output_dir / "summary_boom_dataset.json"

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install it with: pip install ultralytics") from exc

    model = YOLO(str(model_path))
    calib_a = load_fisheye_calibration(args.calib_a)
    calib_b = load_fisheye_calibration(args.calib_b)
    projector_a = EquirectProjector(
        calibration=calib_a,
        source_size=source_size_a,
        output_size=(int(args.equirect_width), int(args.equirect_height)),
        yaw_deg=float(args.yaw_a_deg),
        pitch_deg=float(args.pitch_a_deg),
        roll_deg=float(args.roll_a_deg),
        fov_deg=args.fov_a_deg,
    )
    projector_b = EquirectProjector(
        calibration=calib_b,
        source_size=source_size_b,
        output_size=(int(args.equirect_width), int(args.equirect_height)),
        yaw_deg=float(args.yaw_b_deg),
        pitch_deg=float(args.pitch_b_deg),
        roll_deg=float(args.roll_b_deg),
        fov_deg=args.fov_b_deg,
    )

    cap_a = cv2.VideoCapture(str(lens_a_path))
    cap_b = cv2.VideoCapture(str(lens_b_path))
    if not cap_a.isOpened():
        raise RuntimeError(f"Could not open lens A video: {lens_a_path}")
    if not cap_b.isOpened():
        cap_a.release()
        raise RuntimeError(f"Could not open lens B video: {lens_b_path}")

    successes = 0
    written_rows = 0
    debug_images_saved = 0
    processed_source_frames = 0
    pose_samples: list[dict[str, Any]] = []
    source_frame_index = -1
    frame_interval_sec = 1.0 / max(effective_target_fps, 1e-9)
    next_process_ts = 0.0

    print(f"Running per-frame pose prediction on: {insv_path.name}")
    print(f"Model: {model_path}")
    if model_path.resolve() == FALLBACK_MODEL.resolve():
        print(
            "[WARN] Using fallback model yolov8n-pose.pt. This is a generic human keypoint model and "
            "is not suitable for 360pnp hull keypoints. Pass --model to your trained best.pt."
        )
    if bool(args.rotate_180):
        print("Applying 180-degree pre-rotation to both split lens frames before projection/inference")
    else:
        print("Using split lens frames without 180-degree pre-rotation")
    lens_mode_effective = "best" if args.lens_mode in ("best", "both") else args.lens_mode
    print(f"Lens mode: {args.lens_mode} (effective: {lens_mode_effective})")
    yolo_images_per_call = int(args.inference_batch_frames) * (2 if lens_mode_effective == "best" else 1)
    print(
        f"Inference batch: {int(args.inference_batch_frames)} sampled frame(s) per YOLO call "
        f"(up to {yolo_images_per_call} equirect image(s) in {lens_mode_effective} mode)"
    )
    print(
        f"Pipeline prefetch: {'off' if bool(args.no_prefetch) else 'on'}  |  "
        f"project_workers={int(args.project_workers)}  |  "
        f"opencv_threads={cv2.getNumThreads()}"
    )
    print(f"Correspondence profile: {args.correspondence_profile}")
    if args.kpt_map_mode == "fixed":
        print(f"Keypoint index map mode: fixed ({map_candidates[0]})")
    else:
        print(f"Keypoint index map mode: auto ({' | '.join(map_candidates)})")
    print(
        f"Frames: {total_frames}, source_fps={fps:.3f}, "
        f"pose_fps={effective_target_fps:.3f}, dataset_fps={effective_dataset_fps:.3f}"
    )
    print(
        "Boom labels use camera orientation direction "
        "(camera forward, lens A inverted for consistent boom direction). "
        f"Camera position is still sanity-filtered against mast_mount=({mast_mount[0]:.3f}, {mast_mount[1]:.3f}, {mast_mount[2]:.3f})"
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_handle = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDNAMES)
    csv_writer.writeheader()
    start_time = time.perf_counter()
    timing_sec: dict[str, float] = {
        "video_grab": 0.0,
        "video_retrieve": 0.0,
        "rotate_project": 0.0,
        "yolo_infer": 0.0,
        "pnp_postprocess": 0.0,
        "csv_debug": 0.0,
    }

    def flush_prediction_batch(batch_items: list[dict[str, Any]]) -> None:
        nonlocal successes, written_rows, debug_images_saved
        if not batch_items:
            return

        equirect_batch = [
            item["equirect_by_lens"]
            for item in batch_items
        ]
        batch_results = predict_lens_pose_results_batch(
            model=model,
            equirect_batch=equirect_batch,
            args=args,
            map_candidates=map_candidates,
            lens_mode_effective=lens_mode_effective,
            timing_sec=timing_sec,
        )

        t_write = time.perf_counter()
        for item, (result_a, result_b) in zip(batch_items, batch_results, strict=False):
            source_frame_index_item = int(item["frame_index"])
            timestamp_sec = float(item["timestamp_sec"])

            if lens_mode_effective == "best":
                chosen = choose_best_lens_pose_result([result_a, result_b])
            elif lens_mode_effective == "a":
                chosen = dict(result_a)
                if chosen.get("success", False):
                    chosen["reason"] = "ok"
            else:
                chosen = dict(result_b)
                if chosen.get("success", False):
                    chosen["reason"] = "ok"

            row: dict[str, object] = {
                "frame_index": int(source_frame_index_item),
                "timestamp_sec": float(timestamp_sec),
                "source_fps": float(fps),
                "target_fps": float(effective_target_fps),
                "success": bool(chosen.get("success", False)),
                "selected_lens": str(chosen.get("lens", "")),
                "camera_x_m": "",
                "camera_y_m": "",
                "camera_z_m": "",
                "yaw_deg": "",
                "pitch_deg": "",
                "roll_deg": "",
                "mean_reprojection_error_px": "",
                "median_reprojection_error_px": "",
                "num_inliers": "",
                "num_pairs": "",
                "det_score": "",
                "mean_kpt_conf": "",
                "box_conf": chosen.get("box_conf", ""),
                "det_idx": chosen.get("det_idx", ""),
                "boom_length_m": chosen.get("boom_length_m", ""),
                "detected_object_count": chosen.get("detected_object_count", ""),
                "evaluated_candidate_count": chosen.get("evaluated_candidate_count", ""),
                "successful_candidate_count": chosen.get("successful_candidate_count", ""),
                "candidate_rank_rule": str(chosen.get("candidate_rank_rule", "")),
                "det_class_name": str(chosen.get("det_class_name", "")),
                "correspondence_profile_used": str(chosen.get("correspondence_profile", "")),
                "kpt_index_map_used": str(chosen.get("kpt_index_map_used", "")),
                "reason": str(chosen.get("reason", "")),
                "lens_a_success": bool(result_a.get("success", False)),
                "lens_a_reason": str(result_a.get("reason", "")),
                "lens_a_mean_reprojection_error_px": result_a.get("mean_reprojection_error_px", ""),
                "lens_a_det_class_name": str(result_a.get("det_class_name", "")),
                "lens_a_correspondence_profile_used": str(result_a.get("correspondence_profile", "")),
                "lens_a_kpt_index_map_used": str(result_a.get("kpt_index_map_used", "")),
                "lens_b_success": bool(result_b.get("success", False)),
                "lens_b_reason": str(result_b.get("reason", "")),
                "lens_b_mean_reprojection_error_px": result_b.get("mean_reprojection_error_px", ""),
                "lens_b_det_class_name": str(result_b.get("det_class_name", "")),
                "lens_b_correspondence_profile_used": str(result_b.get("correspondence_profile", "")),
                "lens_b_kpt_index_map_used": str(result_b.get("kpt_index_map_used", "")),
            }

            if chosen.get("success", False):
                pose = chosen["camera_pose_deg"]
                pos = np.asarray(chosen["camera_position"], dtype=np.float64).reshape(3)
                boom_direction = boom_direction_from_camera_pose(
                    np.asarray(chosen["R_wc"], dtype=np.float64).reshape(3, 3),
                    chosen.get("lens", ""),
                )
                row.update(
                    {
                        "camera_x_m": float(pos[0]),
                        "camera_y_m": float(pos[1]),
                        "camera_z_m": float(pos[2]),
                        "yaw_deg": float(pose["yaw_deg"]),
                        "pitch_deg": float(pose["pitch_deg"]),
                        "roll_deg": float(pose["roll_deg"]),
                        "mean_reprojection_error_px": float(chosen["mean_reprojection_error_px"]),
                        "median_reprojection_error_px": float(chosen["median_reprojection_error_px"]),
                        "num_inliers": int(chosen["num_inliers"]),
                        "num_pairs": int(chosen["num_pairs"]),
                        "det_score": float(chosen["det_score"]),
                        "mean_kpt_conf": float(chosen["mean_kpt_conf"]),
                        "box_conf": float(chosen.get("box_conf", math.nan)),
                        "det_idx": int(chosen.get("det_idx", -1)),
                        "boom_length_m": float(chosen.get("boom_length_m", math.nan)),
                        "detected_object_count": int(chosen.get("detected_object_count", 0)),
                        "evaluated_candidate_count": int(chosen.get("evaluated_candidate_count", 0)),
                        "successful_candidate_count": int(chosen.get("successful_candidate_count", 0)),
                        "candidate_rank_rule": str(chosen.get("candidate_rank_rule", "")),
                        "kpt_index_map_used": str(chosen.get("kpt_index_map_used", "")),
                    }
                )
                successes += 1
                pose_samples.append(
                    {
                        "frame_index": int(source_frame_index_item),
                        "timestamp_sec": float(timestamp_sec),
                        "position_m": pos.astype(np.float64, copy=True),
                        "boom_direction_unit": boom_direction.astype(np.float64, copy=True),
                        "selected_lens": str(chosen.get("lens", "")),
                        "mean_reprojection_error_px": float(chosen["mean_reprojection_error_px"]),
                        "median_reprojection_error_px": float(chosen["median_reprojection_error_px"]),
                        "num_inliers": int(chosen["num_inliers"]),
                        "num_pairs": int(chosen["num_pairs"]),
                        "det_score": float(chosen["det_score"]),
                        "mean_kpt_conf": float(chosen["mean_kpt_conf"]),
                    }
                )

            csv_writer.writerow(row)
            written_rows += 1

            should_save_debug = (
                args.debug_every > 0
                and written_rows % int(args.debug_every) == 0
                and (args.debug_max_images <= 0 or debug_images_saved < int(args.debug_max_images))
            )
            if should_save_debug:
                equirect_by_lens = item["equirect_by_lens"]
                if "A" in equirect_by_lens and "B" in equirect_by_lens:
                    saved_path = maybe_write_debug_image(
                        debug_dir=debug_image_dir,
                        frame_index=int(source_frame_index_item),
                        timestamp_sec=float(timestamp_sec),
                        selected_lens=str(chosen.get("lens", "")),
                        eq_a=equirect_by_lens["A"],
                        eq_b=equirect_by_lens["B"],
                        result_a=result_a,
                        result_b=result_b,
                    )
                    debug_images_saved += 1
                    print(f"  debug image saved: {saved_path}")
                else:
                    print("  debug image skipped because only one lens was projected")

            if args.csv_flush_every <= 0 or written_rows % int(args.csv_flush_every) == 0:
                csv_handle.flush()

            if args.progress_every > 0 and (source_frame_index_item + 1) % int(args.progress_every) == 0:
                elapsed = max(1e-6, time.perf_counter() - start_time)
                top_timing = sorted(timing_sec.items(), key=lambda item: -item[1])[:3]
                timing_text = " ".join(
                    f"{key}={100.0 * value / elapsed:.0f}%"
                    for key, value in top_timing
                    if value > 0.0
                )
                pct = 100.0 * float(source_frame_index_item + 1) / float(total_frames)
                print(
                    f"  frame={source_frame_index_item + 1}/{total_frames} ({pct:.1f}%) "
                    f"rows={written_rows} successes={successes} "
                    f"rate={written_rows / elapsed:.2f} rows/s"
                    + (f"  timing: {timing_text}" if timing_text else "")
                )
        timing_sec["csv_debug"] = timing_sec.get("csv_debug", 0.0) + (time.perf_counter() - t_write)

    def rotate_and_project(projector: EquirectProjector, frame: np.ndarray) -> np.ndarray:
        frame_work = cv2.rotate(frame, cv2.ROTATE_180) if bool(args.rotate_180) else frame
        return projector.apply(frame_work)

    project_executor: concurrent.futures.ThreadPoolExecutor | None = None
    if int(args.project_workers) > 1:
        project_executor = concurrent.futures.ThreadPoolExecutor(max_workers=int(args.project_workers))

    def project_lenses(frame_a: np.ndarray, frame_b: np.ndarray) -> dict[str, np.ndarray]:
        equirect_by_lens: dict[str, np.ndarray] = {}
        needs_lens_a = lens_mode_effective in ("best", "a") or int(args.debug_every) > 0
        needs_lens_b = lens_mode_effective in ("best", "b") or int(args.debug_every) > 0

        if project_executor is not None and needs_lens_a and needs_lens_b:
            future_a = project_executor.submit(rotate_and_project, projector_a, frame_a)
            future_b = project_executor.submit(rotate_and_project, projector_b, frame_b)
            equirect_by_lens["A"] = future_a.result()
            equirect_by_lens["B"] = future_b.result()
            return equirect_by_lens

        if needs_lens_a:
            equirect_by_lens["A"] = rotate_and_project(projector_a, frame_a)
        if needs_lens_b:
            equirect_by_lens["B"] = rotate_and_project(projector_b, frame_b)
        return equirect_by_lens

    def read_prediction_batch() -> list[dict[str, Any]]:
        nonlocal source_frame_index, processed_source_frames, next_process_ts
        prediction_batch: list[dict[str, Any]] = []
        while len(prediction_batch) < int(args.inference_batch_frames):
            t_grab = time.perf_counter()
            ok_a = cap_a.grab()
            ok_b = cap_b.grab()
            timing_sec["video_grab"] += time.perf_counter() - t_grab
            if not ok_a or not ok_b:
                break

            source_frame_index += 1
            processed_source_frames += 1
            if args.max_frames is not None and processed_source_frames > int(args.max_frames):
                processed_source_frames -= 1
                break
            timestamp_sec = float(source_frame_index) / max(fps, 1e-9)

            # Downsample by timestamp so sources like 59.94 fps can hit the requested pose FPS cleanly.
            if timestamp_sec + 1e-9 < next_process_ts:
                continue
            next_process_ts += frame_interval_sec
            if next_process_ts < timestamp_sec:
                # Recover from drift in long runs without processing bursts.
                steps_ahead = math.floor((timestamp_sec - next_process_ts) / frame_interval_sec)
                if steps_ahead > 0:
                    next_process_ts += (steps_ahead + 1) * frame_interval_sec

            t_retrieve = time.perf_counter()
            ok_a, frame_a = cap_a.retrieve()
            ok_b, frame_b = cap_b.retrieve()
            timing_sec["video_retrieve"] += time.perf_counter() - t_retrieve
            if not ok_a or frame_a is None:
                raise RuntimeError(f"Could not decode lens A frame {source_frame_index}")
            if not ok_b or frame_b is None:
                raise RuntimeError(f"Could not decode lens B frame {source_frame_index}")

            t_project = time.perf_counter()
            equirect_by_lens = project_lenses(frame_a, frame_b)
            timing_sec["rotate_project"] += time.perf_counter() - t_project

            prediction_batch.append(
                {
                    "frame_index": int(source_frame_index),
                    "timestamp_sec": float(timestamp_sec),
                    "equirect_by_lens": equirect_by_lens,
                }
            )
        return prediction_batch

    try:
        if bool(args.no_prefetch):
            while True:
                prediction_batch = read_prediction_batch()
                if not prediction_batch:
                    break
                flush_prediction_batch(prediction_batch)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                future = prefetch_executor.submit(read_prediction_batch)
                while True:
                    prediction_batch = future.result()
                    if not prediction_batch:
                        break
                    future = prefetch_executor.submit(read_prediction_batch)
                    flush_prediction_batch(prediction_batch)
    finally:
        if project_executor is not None:
            project_executor.shutdown(wait=True)
        cap_a.release()
        cap_b.release()
        csv_handle.flush()
        csv_handle.close()

    prediction_elapsed_sec = max(1e-6, time.perf_counter() - start_time)
    timed_total_sec = float(sum(timing_sec.values()))
    timing_summary = {
        key: {
            "seconds": float(value),
            "percent_of_elapsed": float(100.0 * value / prediction_elapsed_sec),
            "percent_of_timed": float(100.0 * value / timed_total_sec) if timed_total_sec > 1e-9 else 0.0,
        }
        for key, value in timing_sec.items()
    }
    boom_rows, boom_stats = build_boom_dataset_rows(
        samples=pose_samples,
        source_fps=float(fps),
        pose_fps=float(effective_target_fps),
        dataset_fps=float(effective_dataset_fps),
        mast_mount=mast_mount,
        args=args,
    )
    write_boom_dataset_csv(boom_csv_path, boom_rows)

    summary = {
        "insv": str(insv_path),
        "model": str(model_path),
        "lens_a_video": str(lens_a_path),
        "lens_b_video": str(lens_b_path),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "boom_csv_path": str(boom_csv_path),
        "source_fps": float(fps),
        "target_fps": float(effective_target_fps),
        "pose_fps": float(effective_target_fps),
        "dataset_fps": float(effective_dataset_fps),
        "total_source_frames": int(total_frames),
        "processed_source_frames": int(processed_source_frames),
        "processed_rows": int(written_rows),
        "successful_frames": int(successes),
        "success_rate": float(successes) / float(written_rows) if written_rows else 0.0,
        "prediction_elapsed_sec": float(prediction_elapsed_sec),
        "prediction_rows_per_sec": float(written_rows) / float(prediction_elapsed_sec) if written_rows else 0.0,
        "timing": timing_summary,
        "pre_rotation_deg": 180 if bool(args.rotate_180) else 0,
        "equirect_width": int(args.equirect_width),
        "equirect_height": int(args.equirect_height),
        "model_imgsz": int(args.model_imgsz),
        "half_inference": bool(args.half),
        "inference_batch_frames": int(args.inference_batch_frames),
        "max_yolo_images_per_call": int(yolo_images_per_call),
        "pipeline_prefetch": not bool(args.no_prefetch),
        "project_workers": int(args.project_workers),
        "opencv_threads": int(cv2.getNumThreads()),
        "video_decode_strategy": "grab skipped frames, retrieve sampled frames",
        "debug_image_dir": str(debug_image_dir),
        "debug_every": int(args.debug_every),
        "debug_max_images": int(args.debug_max_images),
        "debug_images_saved": int(debug_images_saved),
        "lens_mode": str(args.lens_mode),
        "lens_mode_effective": str(lens_mode_effective),
        "kpt_map_mode": str(args.kpt_map_mode),
        "keypoint_index_maps_considered": list(map_candidates),
        "keypoint_index_map": [int(v) for v in kpt_index_map.tolist()],
        "correspondence_profile": str(args.correspondence_profile),
        "class_to_profile_map": {"front": "front6", "back": "hull9"},
        "keypoint_labels": list(profile_labels),
        "object_points_3d_m": np.asarray(profile_object_points, dtype=np.float64).tolist(),
        "mast_mount_point_m": [float(v) for v in mast_mount.tolist()],
        "boom_label_source": "camera_orientation",
        "boom_label_note": (
            "boom_vector and boom_unit in boom_angle_dataset.csv are averaged camera-forward directions; "
            "lens A directions are inverted before averaging. camera_x/y/z are retained for filtering/debugging."
        ),
        "boom_filtering": {
            "max_boom_speed_mps": float(args.max_boom_speed_mps),
            "min_boom_length_m": float(args.min_boom_length_m),
            "max_boom_length_m": float(args.max_boom_length_m),
            "min_camera_y_m": float(args.min_camera_y_m),
            "max_camera_y_m": float(args.max_camera_y_m),
            "max_frame_jump_m": float(args.max_frame_jump_m),
            "jump_gap_sec": float(args.jump_gap_sec),
            "max_angle_jump_deg": float(args.max_angle_jump_deg),
            "speed_gap_sec": float(args.speed_gap_sec),
            "outlier_mad_threshold": float(args.outlier_mad_threshold),
            "max_window_spread_m": float(args.max_window_spread_m),
            "min_window_samples": int(args.min_window_samples),
        },
        "boom_dataset": boom_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved raw pose CSV: {csv_path}")
    print(f"Saved boom dataset CSV: {boom_csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Successful poses: {successes}/{written_rows}")
    print(f"Boom dataset rows: {len(boom_rows)}")
    print(f"Prediction speed: {written_rows / prediction_elapsed_sec if written_rows else 0.0:.2f} rows/s")
    if written_rows:
        print("Timing breakdown:")
        for key, payload in sorted(timing_summary.items(), key=lambda item: -item[1]["seconds"]):
            print(
                f"  {key}: {payload['seconds']:.2f}s "
                f"({payload['percent_of_elapsed']:.1f}% elapsed)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
