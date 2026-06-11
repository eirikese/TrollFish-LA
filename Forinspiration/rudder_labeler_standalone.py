"""
Standalone rudder labeling tool.

What it does:
1. Lets you pick multiple video files through a file-dialog GUI.
2. Runs AutoPnP (YOLO keypoints + solvePnP) per video to estimate camera pose.
3. Samples random frames and runs a single PilotNet model (no Kalman filtering).
4. Draws a rudder line hinged at a fixed point, corrected using the estimated camera pose.
5. Lets you approve or rotate the line with keyboard controls.
6. Saves frame images and angle labels to a dataset folder.

This is a standalone script and does not require running the Flask app.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tkinter import Tk, filedialog, messagebox

    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


# ---------------------------------------------------------------------------
# AutoPnP constants (ported from your processing/app logic)
# ---------------------------------------------------------------------------
PNP_KEYPOINT_LABELS = [
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

PNP_OBJECT_POINTS = {
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

BASE_R_WC = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

PNP_BOUNDS = {
    "pitch_min_deg": 10.0,
    "pitch_max_deg": 23.0,
    "yaw_abs_max_deg": 10.0,
    "roll_abs_max_deg": 7.0,
    "x_min_m": -3.8,
    "x_max_m": -2.8,
    "y_abs_max_m": 0.5,
    "z_min_m": 0.2,
    "z_max_m": 1.5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wrap_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _rot_x(deg: float) -> np.ndarray:
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=np.float64,
    )


def _rot_y(deg: float) -> np.ndarray:
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ],
        dtype=np.float64,
    )


def _rot_z(deg: float) -> np.ndarray:
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def default_camera_pose_and_rotation(
    pitch: float = 8.0,
    yaw: float = 0.0,
    roll: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return default camera position and camera->world rotation."""
    camera_pos = np.array([-3.194, 0.0, 0.585], dtype=np.float64)
    R_wc = BASE_R_WC.copy()
    R_rel = _rot_z(roll) @ _rot_y(yaw) @ _rot_x(-pitch)
    R_wc = R_wc @ R_rel
    return camera_pos, R_wc


def _rotation_matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
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


def camera_pose_angles_from_rwc(R_wc: np.ndarray) -> Dict[str, float]:
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    R_rel = BASE_R_WC.T @ R_wc
    rx, ry, rz = _rotation_matrix_to_euler_xyz_deg(R_rel)
    return {
        "pitch_deg": float(-rx),
        "yaw_deg": float(ry),
        "roll_deg": float(rz),
    }


def ray_from_pixel(u: float, v: float, K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x_n = (float(u) - cx) / fx
    y_n = (float(v) - cy) / fy
    d_c = np.array([x_n, y_n, 1.0], dtype=np.float64)
    n = float(np.linalg.norm(d_c))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return d_c / n


def intersect_world_z_plane(
    u: float,
    v: float,
    K: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    z_plane: float,
) -> Optional[np.ndarray]:
    d_c = ray_from_pixel(u, v, K)
    d_w = np.asarray(R_wc, dtype=np.float64).reshape(3, 3) @ d_c
    o_w = np.asarray(t_wc, dtype=np.float64).reshape(3)
    dz = float(d_w[2])
    if abs(dz) < 1e-9:
        return None
    t = (float(z_plane) - float(o_w[2])) / dz
    if t <= 0.0:
        return None
    return o_w + t * d_w


def project_world_to_pixel(
    P_w: np.ndarray,
    K: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
) -> Optional[Tuple[float, float]]:
    P_w = np.asarray(P_w, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3)
    R_cw = R_wc.T
    p_c = R_cw @ (P_w - t_wc)
    z = float(p_c[2])
    if z <= 1e-8:
        return None
    u = float(K[0, 0] * (p_c[0] / z) + K[0, 2])
    v = float(K[1, 1] * (p_c[1] / z) + K[1, 2])
    return u, v


def choose_files_gui(
    prompt: str,
    filetypes: List[Tuple[str, str]],
    multiple: bool = False,
) -> List[Path]:
    if not TK_AVAILABLE:
        return []
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.update()
    paths: List[str]
    if multiple:
        paths = list(filedialog.askopenfilenames(title=prompt, filetypes=filetypes))
    else:
        p = filedialog.askopenfilename(title=prompt, filetypes=filetypes)
        paths = [p] if p else []
    root.attributes("-topmost", False)
    root.destroy()
    return [Path(p) for p in paths if p]


def choose_dir_gui(prompt: str) -> Optional[Path]:
    if not TK_AVAILABLE:
        return None
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.update()
    p = filedialog.askdirectory(title=prompt)
    root.attributes("-topmost", False)
    root.destroy()
    if not p:
        return None
    return Path(p)


def show_gui_error(title: str, message: str) -> None:
    # Disabled by default because message boxes can appear behind windows
    # or block indefinitely in some shell/display setups.
    if str(os.environ.get("RUDDER_LABELER_GUI_ERRORS", "0")).strip() != "1":
        return
    if not TK_AVAILABLE:
        return
    try:
        root = Tk()
        root.withdraw()
        root.update()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        return


def describe_pose(
    camera_pos: Optional[np.ndarray],
    angles: Optional[Dict[str, float]],
    err_px: Optional[float],
) -> str:
    if camera_pos is None or angles is None:
        return "pose=missing"
    x, y, z = [float(v) for v in np.asarray(camera_pos, dtype=np.float64).reshape(3)]
    pitch = float(angles.get("pitch_deg", float("nan")))
    yaw = float(angles.get("yaw_deg", float("nan")))
    roll = float(angles.get("roll_deg", float("nan")))
    if err_px is None or not np.isfinite(err_px):
        return (
            f"pose: x={x:.3f} y={y:.3f} z={z:.3f} | "
            f"pitch={pitch:.2f} yaw={yaw:.2f} roll={roll:.2f}"
        )
    return (
        f"pose: x={x:.3f} y={y:.3f} z={z:.3f} | pitch={pitch:.2f} "
        f"yaw={yaw:.2f} roll={roll:.2f} | err={float(err_px):.2f}px"
    )


# ---------------------------------------------------------------------------
# PilotNet (single model, no Kalman)
# ---------------------------------------------------------------------------
class SinglePilotNetDetector:
    """Single-model PilotNet detector.

    Supports:
    - ONNX weights (`.onnx`) via onnxruntime (default/recommended).
    - PyTorch checkpoint (`.pth` / `.pt`) via lazy torch import.
    """

    def __init__(
        self,
        weights_path: Path,
        meta_path: Optional[Path] = None,
        roi_override: Optional[Tuple[int, int, int, int]] = None,
        device: str = "auto",
    ):
        if not weights_path.exists():
            raise FileNotFoundError(f"PilotNet weights not found: {weights_path}")

        meta: Dict[str, Any] = {}
        if meta_path is not None and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        # Reasonable defaults from your existing metadata.
        self.angle_mean = float(meta.get("angle_mean", 93.75167499999999))
        self.angle_std = float(meta.get("angle_std", 16.375090119290753))
        self.angle_std = self.angle_std if abs(self.angle_std) >= 1e-8 else 1.0
        self.angle_domain = str(meta.get("angle_domain", "corrected_no_runtime_yaw"))
        self.runtime_yaw_correction = bool(
            meta.get(
                "runtime_yaw_correction",
                self.angle_domain == "uncorrected_runtime_yaw_subtract",
            )
        )
        input_size_meta = meta.get("input_size", [66, 200])
        if isinstance(input_size_meta, (list, tuple)) and len(input_size_meta) == 2:
            self.input_h = int(input_size_meta[0])
            self.input_w = int(input_size_meta[1])
        else:
            self.input_h, self.input_w = 66, 200

        roi_meta = meta.get("roi_crop", [880, 1080, 200, 1685])
        if roi_override is not None:
            self.roi_crop = tuple(int(v) for v in roi_override)
        elif isinstance(roi_meta, (list, tuple)) and len(roi_meta) == 4:
            self.roi_crop = tuple(int(v) for v in roi_meta)
        else:
            self.roi_crop = (880, 1080, 200, 1685)

        ext = weights_path.suffix.lower()
        self.backend = ""
        self._ort_sess = None
        self._ort_input_name = None
        self._torch_model = None
        self._torch_device = None

        if ext == ".onnx":
            self._init_onnx(weights_path)
        elif ext in {".pth", ".pt"}:
            self._init_torch(weights_path, device=device)
        else:
            raise ValueError(
                f"Unsupported PilotNet weights extension: {weights_path.suffix}. "
                "Use .onnx (recommended) or .pth/.pt."
            )

    def _init_onnx(self, weights_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for ONNX PilotNet. Install with: pip install onnxruntime"
            ) from exc

        self._ort_sess = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])
        self._ort_input_name = self._ort_sess.get_inputs()[0].name
        self.backend = "onnx"

    def _init_torch(self, weights_path: Path, device: str = "auto") -> None:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for .pth/.pt PilotNet weights. "
                "If torch import hangs on your machine, use .onnx weights instead."
            ) from exc

        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)

        class _PilotNet(nn.Module):
            def __init__(self, dropout_rate: float = 0.0):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 24, kernel_size=5, stride=2),
                    nn.ELU(),
                    nn.Conv2d(24, 36, kernel_size=5, stride=2),
                    nn.ELU(),
                    nn.Conv2d(36, 48, kernel_size=5, stride=2),
                    nn.ELU(),
                    nn.Conv2d(48, 64, kernel_size=3),
                    nn.ELU(),
                    nn.Conv2d(64, 64, kernel_size=3),
                    nn.ELU(),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64 * 1 * 18, 100),
                    nn.ELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(100, 50),
                    nn.ELU(),
                    nn.Linear(50, 10),
                    nn.ELU(),
                    nn.Linear(10, 1),
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x.squeeze(-1)

        checkpoint = torch.load(str(weights_path), map_location=torch_device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.angle_mean = float(checkpoint.get("angle_mean", self.angle_mean))
            std_val = float(checkpoint.get("angle_std", self.angle_std))
            self.angle_std = std_val if abs(std_val) >= 1e-8 else self.angle_std
        else:
            state_dict = checkpoint

        model = _PilotNet(dropout_rate=0.0).to(torch_device)
        model.load_state_dict(state_dict)
        model.eval()

        self._torch_model = model
        self._torch_device = torch_device
        self.backend = "torch"

    def predict_angle(self, frame_bgr: np.ndarray) -> float:
        top, bottom, left, right = self.roi_crop
        h, w = frame_bgr.shape[:2]
        top = int(np.clip(top, 0, h))
        bottom = int(np.clip(bottom, 0, h))
        left = int(np.clip(left, 0, w))
        right = int(np.clip(right, 0, w))
        crop = frame_bgr[top:bottom, left:right]
        if crop.size == 0:
            return float("nan")

        resized = cv2.resize(
            crop,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_AREA,
        )
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        x = yuv.astype(np.float32) / 127.5 - 1.0
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

        if self.backend == "onnx":
            assert self._ort_sess is not None and self._ort_input_name is not None
            raw = float(self._ort_sess.run(None, {self._ort_input_name: x})[0].item())
        elif self.backend == "torch":
            assert self._torch_model is not None and self._torch_device is not None
            import torch

            with torch.no_grad():
                xt = torch.from_numpy(x).to(self._torch_device)
                raw = float(self._torch_model(xt).item())
        else:
            raise RuntimeError("PilotNet detector backend is not initialized.")
        return raw * self.angle_std + self.angle_mean

    def to_corrected_angle(self, model_angle_deg: float, camera_yaw_deg: float = 0.0) -> float:
        """Convert raw model output into corrected labeling angle domain."""
        a = float(model_angle_deg)
        if not np.isfinite(a):
            return float("nan")
        if self.runtime_yaw_correction:
            y = float(camera_yaw_deg)
            if np.isfinite(y):
                a = a - y
        return float(np.clip(a, 0.0, 180.0))


# ---------------------------------------------------------------------------
# Camera intrinsics / undistortion
# ---------------------------------------------------------------------------
@dataclass
class IntrinsicsSetup:
    K_undist: np.ndarray
    map1: Optional[np.ndarray] = None
    map2: Optional[np.ndarray] = None
    used_calibration: bool = False
    notes: str = ""


def build_intrinsics_for_video(
    width: int,
    height: int,
    calib_path: Optional[Path],
) -> IntrinsicsSetup:
    if calib_path is not None and calib_path.exists():
        try:
            with np.load(str(calib_path)) as cal:
                K = np.asarray(cal["K"], dtype=np.float64).reshape(3, 3)
                D = np.asarray(cal["D"], dtype=np.float64).reshape(-1, 1)
                img_size_arr = np.asarray(cal["img_size"]).reshape(-1)
            if len(img_size_arr) >= 2:
                calib_w = int(img_size_arr[0])
                calib_h = int(img_size_arr[1])
            else:
                calib_w, calib_h = width, height

            if (calib_w, calib_h) == (width, height):
                R = np.eye(3, dtype=np.float64)
                K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K,
                    D,
                    (calib_w, calib_h),
                    R,
                    balance=0.0,
                    new_size=(width, height),
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K,
                    D,
                    R,
                    K_new,
                    (width, height),
                    cv2.CV_16SC2,
                )
                return IntrinsicsSetup(
                    K_undist=np.asarray(K_new, dtype=np.float64).reshape(3, 3),
                    map1=map1,
                    map2=map2,
                    used_calibration=True,
                    notes=f"fisheye undistortion active ({calib_w}x{calib_h})",
                )

            # Resolution mismatch fallback: scale K and skip remap.
            sx = float(width) / float(calib_w)
            sy = float(height) / float(calib_h)
            K_scaled = K.copy()
            K_scaled[0, 0] *= sx
            K_scaled[1, 1] *= sy
            K_scaled[0, 2] *= sx
            K_scaled[1, 2] *= sy
            return IntrinsicsSetup(
                K_undist=K_scaled,
                map1=None,
                map2=None,
                used_calibration=True,
                notes=(
                    f"calibration size mismatch ({calib_w}x{calib_h} -> "
                    f"{width}x{height}); using scaled intrinsics only"
                ),
            )
        except Exception as exc:
            print(f"[WARN] Failed to load calibration {calib_path}: {exc}")

    # Fallback pinhole approximation.
    f_guess = 0.90 * max(float(width), float(height))
    K_approx = np.array(
        [
            [f_guess, 0.0, width / 2.0],
            [0.0, f_guess, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return IntrinsicsSetup(
        K_undist=K_approx,
        map1=None,
        map2=None,
        used_calibration=False,
        notes="no calibration provided; using approximate pinhole intrinsics",
    )


def maybe_undistort(frame_bgr: np.ndarray, intr: IntrinsicsSetup) -> np.ndarray:
    if intr.map1 is None or intr.map2 is None:
        return frame_bgr
    return cv2.remap(frame_bgr, intr.map1, intr.map2, interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# AutoPnP estimator
# ---------------------------------------------------------------------------
class AutoPnPEstimator:
    def __init__(self, model_path: Path, min_kpt_conf: float = 0.8):
        if not model_path.exists():
            raise FileNotFoundError(f"AutoPnP model not found: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            install_cmd = f'"{sys.executable}" -m pip install ultralytics'
            raise RuntimeError(
                "Ultralytics is required for AutoPnP.\n"
                f"Active interpreter: {sys.executable}\n"
                f"Install with: {install_cmd}"
            ) from exc
        self.model = YOLO(str(model_path))
        self.min_kpt_conf = float(np.clip(min_kpt_conf, 0.0, 1.0))
        self.roll_offset_deg = 90.0

    @staticmethod
    def _bind_keypoints_geometry(points_xy: np.ndarray) -> Dict[str, int]:
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        n = len(PNP_KEYPOINT_LABELS)
        if pts.shape[0] < n:
            raise ValueError(f"Expected {n} keypoints, got {pts.shape[0]}")
        pts = pts[:n]
        if not np.all(np.isfinite(pts)):
            raise ValueError("Detected keypoints contain non-finite values")

        front_idx = int(np.argmin(pts[:, 1]))
        front_pt = pts[front_idx]

        remaining = [i for i in range(n) if i != front_idx]
        rem_pts = pts[remaining]
        x_order = np.argsort(rem_pts[:, 0])
        left_side = [remaining[i] for i in x_order[:4]]
        right_side = [remaining[i] for i in x_order[4:]]

        def split_side(indices: List[int]) -> Dict[str, int]:
            side_pts = pts[indices]
            dist = np.linalg.norm(side_pts - front_pt[None, :], axis=1)
            back_local = int(np.argmax(dist))
            back_idx = int(indices[back_local])
            rail_idx = [idx for idx in indices if idx != back_idx]
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

    @staticmethod
    def _swap_port_starboard(label_to_index: Dict[str, int]) -> Dict[str, int]:
        swapped = dict(label_to_index)
        for a, b in (
            ("porttop", "starboardtop"),
            ("portmid", "starboardmid"),
            ("portlow", "starboardlow"),
            ("portback", "starboardback"),
        ):
            swapped[a], swapped[b] = swapped[b], swapped[a]
        return swapped

    def _build_correspondences(
        self,
        raw_pts: np.ndarray,
        raw_conf: np.ndarray,
        label_to_index: Dict[str, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        obj_pts: List[np.ndarray] = []
        img_pts: List[np.ndarray] = []
        for label in PNP_KEYPOINT_LABELS:
            idx = int(label_to_index[label])
            uv = np.asarray(raw_pts[idx], dtype=np.float64).reshape(2)
            conf = float(raw_conf[idx]) if np.isfinite(raw_conf[idx]) else np.nan
            if not np.all(np.isfinite(uv)):
                continue
            if np.isfinite(conf) and conf < self.min_kpt_conf:
                continue
            obj_pts.append(np.asarray(PNP_OBJECT_POINTS[label], dtype=np.float64).reshape(3))
            img_pts.append(uv)
        if not obj_pts:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        return np.asarray(obj_pts, dtype=np.float64), np.asarray(img_pts, dtype=np.float64)

    @staticmethod
    def _range_violation(v: float, lo: float, hi: float) -> float:
        vv = float(v)
        span = max(1e-6, abs(hi - lo))
        if vv < lo:
            return (lo - vv) / span
        if vv > hi:
            return (vv - hi) / span
        return 0.0

    @staticmethod
    def _abs_violation(v: float, lim: float) -> float:
        vv = abs(float(v))
        return max(vv - lim, 0.0) / max(1e-6, lim)

    @staticmethod
    def _validate_pose(
        camera_pos: np.ndarray,
        angles_deg: Dict[str, float],
        reproj_error_px: float,
    ) -> Tuple[bool, List[str]]:
        b = PNP_BOUNDS
        x, y, z = [float(v) for v in np.asarray(camera_pos, dtype=np.float64).reshape(3)]
        pitch = float(angles_deg.get("pitch_deg", np.nan))
        yaw = wrap_deg(float(angles_deg.get("yaw_deg", np.nan)))
        roll = wrap_deg(float(angles_deg.get("roll_deg", np.nan)))
        err = float(reproj_error_px) if np.isfinite(reproj_error_px) else np.nan

        issues: List[str] = []
        if not (b["pitch_min_deg"] <= pitch <= b["pitch_max_deg"]):
            issues.append(f"pitch={pitch:.2f}")
        if abs(yaw) > b["yaw_abs_max_deg"]:
            issues.append(f"yaw={yaw:.2f}")
        if abs(roll) > b["roll_abs_max_deg"]:
            issues.append(f"roll={roll:.2f}")
        if not (b["x_min_m"] <= x <= b["x_max_m"]):
            issues.append(f"x={x:.3f}")
        if abs(y) > b["y_abs_max_m"]:
            issues.append(f"y={y:.3f}")
        if not (b["z_min_m"] <= z <= b["z_max_m"]):
            issues.append(f"z={z:.3f}")
        if not np.isfinite(err) or err > 60.0:
            issues.append(f"err={err:.2f}px")
        return (len(issues) == 0), issues

    def _solve_pose(self, obj: np.ndarray, img: np.ndarray, K_undist: np.ndarray) -> Dict[str, Any]:
        if obj.shape[0] < 4:
            return {"ok": False, "error": "Need at least 4 correspondences"}

        K = np.asarray(K_undist, dtype=np.float64).reshape(3, 3)
        dist = np.zeros((4, 1), dtype=np.float64)
        candidates: List[Dict[str, Any]] = []

        def corrected_pose_from_rt(rvec_local: np.ndarray, tvec_local: np.ndarray):
            R_cw_local, _ = cv2.Rodrigues(rvec_local)
            R_wc_raw_local = R_cw_local.T
            camera_pos_raw_local = (-R_wc_raw_local @ tvec_local).reshape(3)
            raw_angles = camera_pose_angles_from_rwc(R_wc_raw_local)

            _, R_wc_corr_local = default_camera_pose_and_rotation(
                pitch=raw_angles["pitch_deg"],
                yaw=raw_angles["yaw_deg"],
                roll=(raw_angles["roll_deg"] + self.roll_offset_deg),
            )
            camera_pos_corr_local = np.array(
                [camera_pos_raw_local[0], camera_pos_raw_local[2], camera_pos_raw_local[1]],
                dtype=np.float64,
            )
            angles_corr = camera_pose_angles_from_rwc(R_wc_corr_local)
            angles_corr["yaw_deg"] = wrap_deg(angles_corr["yaw_deg"])
            angles_corr["roll_deg"] = wrap_deg(angles_corr["roll_deg"])
            return R_wc_raw_local, camera_pos_raw_local, camera_pos_corr_local, angles_corr

        def build_candidate(method: str, rvec_local: np.ndarray, tvec_local: np.ndarray, inlier_idx_local: np.ndarray):
            inlier_idx_local = np.asarray(inlier_idx_local, dtype=np.int32).reshape(-1)
            inlier_idx_local = inlier_idx_local[(inlier_idx_local >= 0) & (inlier_idx_local < len(obj))]
            if len(inlier_idx_local) == 0:
                inlier_idx_local = np.arange(len(obj), dtype=np.int32)

            R_wc_raw, camera_pos_raw, camera_pos_corr, angles_corr = corrected_pose_from_rt(rvec_local, tvec_local)
            proj_local, _ = cv2.projectPoints(obj, rvec_local, tvec_local, K, dist)
            proj_local = proj_local.reshape(-1, 2)
            residuals = np.linalg.norm(proj_local - img, axis=1)
            mean_all = float(np.mean(residuals)) if len(residuals) else float("inf")
            med_all = float(np.median(residuals)) if len(residuals) else float("inf")
            inlier_err = float(np.mean(residuals[inlier_idx_local])) if len(inlier_idx_local) else mean_all

            b = PNP_BOUNDS
            x, y, z = [float(v) for v in camera_pos_corr.reshape(3)]
            pitch = float(angles_corr.get("pitch_deg", np.nan))
            yaw = float(angles_corr.get("yaw_deg", np.nan))
            roll = float(angles_corr.get("roll_deg", np.nan))

            violation = 0.0
            violation += self._range_violation(pitch, float(b["pitch_min_deg"]), float(b["pitch_max_deg"]))
            violation += self._abs_violation(yaw, float(b["yaw_abs_max_deg"]))
            violation += self._abs_violation(roll, float(b["roll_abs_max_deg"]))
            violation += self._range_violation(x, float(b["x_min_m"]), float(b["x_max_m"]))
            violation += self._abs_violation(y, float(b["y_abs_max_m"]))
            violation += self._range_violation(z, float(b["z_min_m"]), float(b["z_max_m"]))
            violation += max(inlier_err - 30.0, 0.0) / 30.0
            inlier_ratio = float(len(inlier_idx_local)) / float(len(obj))
            low_inlier_penalty = max(0.0, 0.70 - inlier_ratio) * 25.0
            score = med_all + 25.0 * violation + low_inlier_penalty

            return {
                "method": method,
                "score": float(score),
                "rvec": rvec_local,
                "tvec": tvec_local,
                "R_wc_raw": R_wc_raw,
                "camera_pos_raw": camera_pos_raw,
                "mean_error_px": mean_all,
                "median_error_px": med_all,
                "inlier_error_px": inlier_err,
                "num_inliers": int(len(inlier_idx_local)),
                "num_pairs": int(len(obj)),
            }

        # Candidate A: RANSAC + LM
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
                        rvec_r, tvec_r = cv2.solvePnPRefineLM(
                            obj[inlier_idx_r],
                            img[inlier_idx_r],
                            K,
                            dist,
                            rvec_r,
                            tvec_r,
                        )
                    except Exception:
                        pass
                candidates.append(build_candidate("ransac_epnp", rvec_r, tvec_r, inlier_idx_r))

        # Candidate B: iterative solve all points
        try:
            ok_i, rvec_i, tvec_i = cv2.solvePnP(
                obj,
                img,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            ok_i = False
        if ok_i:
            try:
                rvec_i, tvec_i = cv2.solvePnPRefineLM(obj, img, K, dist, rvec_i, tvec_i)
            except Exception:
                pass
            candidates.append(
                build_candidate(
                    "iterative_all",
                    rvec_i,
                    tvec_i,
                    np.arange(len(obj), dtype=np.int32),
                )
            )

        if not candidates:
            return {"ok": False, "error": "solvePnP failed"}

        best = min(candidates, key=lambda c: c["score"])
        return {
            "ok": True,
            "R_wc": best["R_wc_raw"],
            "camera_pos": best["camera_pos_raw"],
            "num_pairs": best["num_pairs"],
            "num_inliers": best["num_inliers"],
            "mean_error_px": best["mean_error_px"],
            "median_error_px": best["median_error_px"],
            "inlier_error_px": best["inlier_error_px"],
            "solve_method": best["method"],
            "solve_score": best["score"],
        }

    def attempt(self, frame_bgr: np.ndarray, K_undist: np.ndarray) -> Dict[str, Any]:
        try:
            results = self.model.predict(
                source=frame_bgr,
                verbose=False,
                conf=0.05,
                iou=0.6,
                max_det=4,
            )
        except Exception as exc:
            return {"success": False, "reason": f"YOLO inference failed: {exc}"}
        if not results:
            return {"success": False, "reason": "YOLO returned no detections"}

        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            return {"success": False, "reason": "No keypoints detected"}

        xy_all = result.keypoints.xy.detach().cpu().numpy()
        if xy_all.ndim != 3 or xy_all.shape[2] != 2:
            return {"success": False, "reason": "Unexpected keypoint output shape"}

        conf_tensor = result.keypoints.conf
        if conf_tensor is not None:
            conf_all = conf_tensor.detach().cpu().numpy()
        else:
            conf_all = np.ones((xy_all.shape[0], xy_all.shape[1]), dtype=np.float64)

        n_labels = len(PNP_KEYPOINT_LABELS)
        if xy_all.shape[1] < n_labels:
            return {
                "success": False,
                "reason": f"Model returned {xy_all.shape[1]} keypoints, need {n_labels}",
            }

        box_conf = np.ones((xy_all.shape[0],), dtype=np.float64)
        if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) == xy_all.shape[0]:
            box_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
        mean_kpt_conf = np.mean(np.nan_to_num(conf_all[:, :n_labels], nan=0.0), axis=1)
        det_scores = box_conf * mean_kpt_conf
        det_idx = int(np.argmax(det_scores))

        raw_pts = xy_all[det_idx, :n_labels, :].astype(np.float64)
        raw_conf = conf_all[det_idx, :n_labels].astype(np.float64)

        try:
            map_primary = self._bind_keypoints_geometry(raw_pts)
            mappings = [
                ("geometry", map_primary),
                (
                    "geometry_swapped_port_starboard",
                    self._swap_port_starboard(map_primary),
                ),
            ]
        except Exception as exc:
            return {"success": False, "reason": f"Keypoint binding failed: {exc}"}

        attempted: List[str] = []
        candidates: List[Dict[str, Any]] = []
        for map_name, mapping in mappings:
            obj, img = self._build_correspondences(raw_pts, raw_conf, mapping)
            if len(obj) < 6:
                attempted.append(f"{map_name}: only {len(obj)} valid points")
                continue

            solved = self._solve_pose(obj, img, K_undist)
            if not solved.get("ok"):
                attempted.append(f"{map_name}: {solved.get('error', 'solve failed')}")
                continue

            raw_angles = camera_pose_angles_from_rwc(solved["R_wc"])
            _, R_wc_corr = default_camera_pose_and_rotation(
                pitch=raw_angles["pitch_deg"],
                yaw=raw_angles["yaw_deg"],
                roll=(raw_angles["roll_deg"] + self.roll_offset_deg),
            )
            camera_pos_raw = np.asarray(solved["camera_pos"], dtype=np.float64).reshape(3)
            camera_pos_corr = np.array(
                [camera_pos_raw[0], camera_pos_raw[2], camera_pos_raw[1]],
                dtype=np.float64,
            )
            angles_corr = camera_pose_angles_from_rwc(R_wc_corr)
            angles_corr["yaw_deg"] = wrap_deg(angles_corr["yaw_deg"])
            angles_corr["roll_deg"] = wrap_deg(angles_corr["roll_deg"])

            if np.isfinite(solved.get("median_error_px", np.nan)):
                err_px = float(solved["median_error_px"])
            elif np.isfinite(solved.get("inlier_error_px", np.nan)):
                err_px = float(solved["inlier_error_px"])
            else:
                err_px = float(solved.get("mean_error_px", np.nan))

            is_valid, issues = self._validate_pose(camera_pos_corr, angles_corr, err_px)
            candidates.append(
                {
                    "mapping": map_name,
                    "camera_pos": camera_pos_corr,
                    "R_wc": R_wc_corr,
                    "angles": angles_corr,
                    "err_px": err_px,
                    "num_inliers": int(solved["num_inliers"]),
                    "num_pairs": int(solved["num_pairs"]),
                    "solve_method": str(solved.get("solve_method", "unknown")),
                    "is_valid": bool(is_valid),
                    "issues": issues,
                }
            )
            attempted.append(
                f"{map_name}: "
                + (
                    f"ok ({solved.get('solve_method', 'unknown')})"
                    if is_valid
                    else f"pose out-of-bounds ({', '.join(issues)})"
                )
            )

        if not candidates:
            reason = " | ".join(attempted) if attempted else "No candidates"
            return {"success": False, "reason": reason}

        valid_candidates = [c for c in candidates if c["is_valid"]]
        best = min(valid_candidates, key=lambda c: c["err_px"]) if valid_candidates else min(candidates, key=lambda c: c["err_px"])
        return {
            "success": True,
            "camera_pos": np.asarray(best["camera_pos"], dtype=np.float64).reshape(3),
            "R_wc": np.asarray(best["R_wc"], dtype=np.float64).reshape(3, 3),
            "angles": best["angles"],
            "err_px": float(best["err_px"]),
            "num_inliers": int(best["num_inliers"]),
            "num_pairs": int(best["num_pairs"]),
            "mapping": str(best["mapping"]),
            "solve_method": str(best.get("solve_method", "unknown")),
            "is_valid": bool(best["is_valid"]),
            "reason": " | ".join(attempted),
        }


def average_pnp_solutions(solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not solutions:
        return {"success": False, "reason": "No successful frame-level solutions"}

    cam_list = []
    pitch_list = []
    yaw_list = []
    roll_list = []
    err_list = []
    inlier_list = []
    pair_list = []
    for s in solutions:
        cam = np.asarray(s.get("camera_pos"), dtype=np.float64).reshape(3)
        ang = s.get("angles", {})
        p = float(ang.get("pitch_deg", np.nan))
        y = wrap_deg(float(ang.get("yaw_deg", np.nan)))
        r = wrap_deg(-float(ang.get("roll_deg", np.nan)))
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

    n = len(cam_list)
    if n == 0:
        return {"success": False, "reason": "No finite solutions after filtering"}

    cam = np.asarray(cam_list, dtype=np.float64)
    pitch = np.asarray(pitch_list, dtype=np.float64)
    yaw = np.asarray(yaw_list, dtype=np.float64)
    roll = np.asarray(roll_list, dtype=np.float64)
    err = np.asarray(err_list, dtype=np.float64)

    keep = np.ones(n, dtype=bool)
    if n >= 3:
        cam_med = np.median(cam, axis=0)
        cam_mad = np.median(np.abs(cam - cam_med[None, :]), axis=0)
        cam_sigma = 1.4826 * cam_mad
        cam_floor = np.array([0.20, 0.10, 0.10], dtype=np.float64)
        cam_thr = np.maximum(3.0 * cam_sigma, cam_floor)
        keep_cam = np.all(np.abs(cam - cam_med[None, :]) <= cam_thr[None, :], axis=1)

        pitch_med = float(np.median(pitch))
        pitch_mad = float(np.median(np.abs(pitch - pitch_med)))
        keep_pitch = np.abs(pitch - pitch_med) <= max(3.0 * 1.4826 * pitch_mad, 2.5)

        yaw_med = float(np.median(yaw))
        yaw_diff = np.array([abs(wrap_deg(v - yaw_med)) for v in yaw], dtype=np.float64)
        yaw_mad = float(np.median(yaw_diff))
        keep_yaw = yaw_diff <= max(3.0 * 1.4826 * yaw_mad, 2.5)

        roll_med = float(np.median(roll))
        roll_diff = np.array([abs(wrap_deg(v - roll_med)) for v in roll], dtype=np.float64)
        roll_mad = float(np.median(roll_diff))
        keep_roll = roll_diff <= max(3.0 * 1.4826 * roll_mad, 2.5)

        keep_err = np.ones(n, dtype=bool)
        finite_err = np.isfinite(err)
        if np.any(finite_err):
            e = err[finite_err]
            err_med = float(np.median(e))
            err_mad = float(np.median(np.abs(e - err_med)))
            keep_err = (~finite_err) | (err <= err_med + max(3.0 * 1.4826 * err_mad, 8.0))

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

    camera_pos = np.median(cam_sel, axis=0)
    pitch_med = float(np.median(pitch_sel))
    yaw_med = wrap_deg(float(np.median(yaw_sel)))
    roll_med = wrap_deg(float(np.median(roll_sel)))

    _, R_wc = default_camera_pose_and_rotation(
        pitch=pitch_med if np.isfinite(pitch_med) else 0.0,
        yaw=yaw_med if np.isfinite(yaw_med) else 0.0,
        roll=roll_med if np.isfinite(roll_med) else 0.0,
    )
    angles = camera_pose_angles_from_rwc(R_wc)
    angles["yaw_deg"] = wrap_deg(angles["yaw_deg"])
    angles["roll_deg"] = wrap_deg(angles["roll_deg"])

    finite_err = err_sel[np.isfinite(err_sel)]
    err_px = float(np.median(finite_err)) if len(finite_err) else float("nan")
    return {
        "success": True,
        "camera_pos": np.asarray(camera_pos, dtype=np.float64).reshape(3),
        "R_wc": np.asarray(R_wc, dtype=np.float64).reshape(3, 3),
        "angles": angles,
        "err_px": err_px,
        "num_inliers": int(round(float(np.median(inlier_sel)))) if len(inlier_sel) else 0,
        "num_pairs": int(round(float(np.median(pair_sel)))) if len(pair_sel) else 0,
        "num_frame_solutions": int(n),
        "num_used_after_filter": int(np.count_nonzero(keep)),
        "num_rejected_outliers": int(n - np.count_nonzero(keep)),
    }


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class VideoState:
    path: Path
    total_frames: int
    width: int
    height: int
    fps: float
    intrinsics: IntrinsicsSetup
    camera_pos: Optional[np.ndarray] = None
    R_wc: Optional[np.ndarray] = None
    angles: Optional[Dict[str, float]] = None
    pnp_error_px: Optional[float] = None
    hinge_px: Optional[Tuple[int, int]] = None
    used_label_frames: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Labeling session
# ---------------------------------------------------------------------------
class LabelingSession:
    LEFT_KEYS = {81, 2424832, ord("a"), ord("A")}
    RIGHT_KEYS = {83, 2555904, ord("d"), ord("D")}
    UP_KEYS = {82, 2490368, ord("w"), ord("W")}
    DOWN_KEYS = {84, 2621440, ord("s"), ord("S")}

    def __init__(
        self,
        videos: List[VideoState],
        detector: SinglePilotNetDetector,
        output_dir: Path,
        plane_z: float = 0.0,
        line_length_m: float = 0.55,
        angle_step_deg: float = 1.0,
        display_scale: float = 0.8,
        rng_seed: int = 123,
        initial_hinge_px: Optional[Tuple[int, int]] = None,
        high_angle_only: bool = False,
        high_angle_threshold_deg: float = 105.0,
    ):
        self.videos = videos
        self.detector = detector
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.csv_path = output_dir / "labels.csv"
        self.plane_z = float(plane_z)
        self.line_length_m = float(line_length_m)
        self.angle_step_deg = max(0.01, float(angle_step_deg))
        self.half_step_deg = 0.5
        self.display_scale = float(np.clip(display_scale, 0.2, 1.5))
        self.rng = random.Random(int(rng_seed))
        self.initial_hinge_px = (
            (int(initial_hinge_px[0]), int(initial_hinge_px[1]))
            if initial_hinge_px is not None
            else None
        )
        self.high_angle_only = bool(high_angle_only)
        # High-angle mode keeps samples outside the center band:
        # angle <= (180 - threshold) OR angle >= threshold.
        # With threshold=105 deg, the band is [75, 105].
        self.high_angle_threshold_deg = float(np.clip(float(high_angle_threshold_deg), 90.0, 180.0))

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()
        self.next_index = self._find_next_index()

        self.window_main = "Rudder Labeler"
        self.window_roi = "ROI Zoom"
        self.current: Optional[Dict[str, Any]] = None
        self.current_angle_deg: float = float("nan")
        self.await_hinge_click = False
        self.saved_count = 0
        self.skipped_count = 0
        self.start_time = time.time()

    def _high_angle_bounds(self) -> Tuple[float, float]:
        high = float(np.clip(self.high_angle_threshold_deg, 90.0, 180.0))
        low = float(np.clip(180.0 - high, 0.0, 90.0))
        return low, high

    def _is_high_angle(self, angle_deg: float) -> bool:
        low, high = self._high_angle_bounds()
        a = float(angle_deg)
        return a <= low or a >= high

    def _default_hinge_for_video(self, video: VideoState) -> Tuple[int, int]:
        _ = video
        if self.initial_hinge_px is not None:
            return self.initial_hinge_px
        t, b, l, r = self.detector.roi_crop
        hx = int(round((l + r) * 0.5))
        hy = int(round(b - 1))
        return hx, hy

    @staticmethod
    def _set_video_hinge(video: VideoState, hx: int, hy: int) -> None:
        video.hinge_px = (int(hx), int(hy))

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "filename",
                    "angle_deg",
                    "video_path",
                    "frame_index",
                    "model_pred_angle_deg",
                    "camera_x",
                    "camera_y",
                    "camera_z",
                    "camera_pitch_deg",
                    "camera_yaw_deg",
                    "camera_roll_deg",
                    "camera_reproj_err_px",
                    "hinge_u",
                    "hinge_v",
                    "roi_top",
                    "roi_bottom",
                    "roi_left",
                    "roi_right",
                    "plane_z",
                    "line_length_m",
                    "saved_at_unix_s",
                ]
            )

    def _find_next_index(self) -> int:
        max_idx = 0
        for p in self.images_dir.glob("img_*.jpg"):
            stem = p.stem
            parts = stem.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
        return max_idx + 1

    def _read_frame(self, video: VideoState, frame_idx: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(video.path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return maybe_undistort(frame, video.intrinsics)

    def _pick_random_sample(self) -> Optional[Dict[str, Any]]:
        if not self.videos:
            return None

        for _ in range(1200):
            video = self.rng.choice(self.videos)
            if video.total_frames <= 0:
                continue
            low = max(0, int(0.05 * video.total_frames))
            high = min(video.total_frames - 1, max(low, int(0.95 * video.total_frames)))
            frame_idx = self.rng.randint(low, high) if high > low else low

            # Prefer unseen frames.
            if len(video.used_label_frames) < max(1, int(0.95 * video.total_frames)):
                for _ in range(20):
                    if frame_idx not in video.used_label_frames:
                        break
                    frame_idx = self.rng.randint(low, high) if high > low else frame_idx

            frame = self._read_frame(video, frame_idx)
            if frame is None:
                continue

            pred_model_raw = self.detector.predict_angle(frame)
            if not np.isfinite(pred_model_raw):
                continue
            # Convert model output to corrected labeling convention.
            # If model is trained in uncorrected/yaw-dependent domain, subtract yaw here.
            camera_yaw_deg = -self._yaw_deg_for_video(video)
            pred_angle = self.detector.to_corrected_angle(
                model_angle_deg=float(pred_model_raw),
                camera_yaw_deg=float(camera_yaw_deg),
            )
            if self.high_angle_only and not self._is_high_angle(pred_angle):
                continue

            if video.hinge_px is None:
                d = self._default_hinge_for_video(video)
                self._set_video_hinge(video, int(d[0]), int(d[1]))

            return {
                "video": video,
                "frame": frame,
                "frame_idx": int(frame_idx),
                "pred_angle": float(pred_angle),
            }
        return None

    @staticmethod
    def _angle_to_image_dir(angle_deg: float) -> np.ndarray:
        # User convention in image coordinates:
        # 0 deg = right, 90 deg = up, 180 deg = left.
        # Image y-axis points downward, so "up" is negative y.
        a = math.radians(float(angle_deg))
        return np.array([math.cos(a), -math.sin(a)], dtype=np.float64)

    @staticmethod
    def _yaw_deg_for_video(video: VideoState) -> float:
        if video.angles is None:
            return 0.0
        y = float(video.angles.get("yaw_deg", 0.0))
        if not np.isfinite(y):
            return 0.0
        # Use negated yaw sign for visual correction.
        return -y

    def _visual_angle_with_yaw(self, video: VideoState, angle_deg: float) -> float:
        # Apply yaw correction in the visualizer so camera yaw is visible in the overlay.
        # Positive camera yaw rotates the visual line in the opposite image direction.
        return float(angle_deg) - self._yaw_deg_for_video(video)

    def _line_endpoints_px(
        self,
        video: VideoState,
        angle_deg: float,
        hinge_px: Tuple[int, int],
        length_scale: float = 1.0,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        hx, hy = int(hinge_px[0]), int(hinge_px[1])
        p0 = (hx, hy)
        vis_angle_deg = self._visual_angle_with_yaw(video, angle_deg)
        length_scale = max(0.1, float(length_scale))

        if video.camera_pos is None or video.R_wc is None:
            # Fallback to image-plane line if no pose.
            d_img = self._angle_to_image_dir(vis_angle_deg)
            length_px = max(40, int(0.20 * min(video.width, video.height) * length_scale))
            ex = int(round(hx + length_px * float(d_img[0])))
            ey = int(round(hy + length_px * float(d_img[1])))
            return p0, (ex, ey)

        hinge_world = intersect_world_z_plane(
            u=float(hx),
            v=float(hy),
            K=video.intrinsics.K_undist,
            R_wc=video.R_wc,
            t_wc=video.camera_pos,
            z_plane=self.plane_z,
        )
        if hinge_world is None:
            return None

        # Camera-corrected direction:
        # 1) Apply yaw correction to the visual angle.
        # 2) Build direction in image space.
        # 3) Convert local image direction to world-plane direction via ray-plane intersections.
        d_img = self._angle_to_image_dir(vis_angle_deg)
        probe_px = 120.0
        u_probe = float(hx) + float(d_img[0]) * probe_px
        v_probe = float(hy) + float(d_img[1]) * probe_px
        probe_world = intersect_world_z_plane(
            u=u_probe,
            v=v_probe,
            K=video.intrinsics.K_undist,
            R_wc=video.R_wc,
            t_wc=video.camera_pos,
            z_plane=self.plane_z,
        )
        if probe_world is None:
            # Rare fallback: keep image-space direction directly.
            ex = int(round(hx + 160.0 * float(d_img[0])))
            ey = int(round(hy + 160.0 * float(d_img[1])))
            return p0, (ex, ey)

        d_world = np.asarray(probe_world, dtype=np.float64).reshape(3) - np.asarray(hinge_world, dtype=np.float64).reshape(3)
        d_world[2] = 0.0
        n = float(np.linalg.norm(d_world))
        if n < 1e-9:
            ex = int(round(hx + 160.0 * float(d_img[0])))
            ey = int(round(hy + 160.0 * float(d_img[1])))
            return p0, (ex, ey)
        d_world /= n

        tip_world = np.asarray(hinge_world, dtype=np.float64).reshape(3) + d_world * float(self.line_length_m) * length_scale

        tip_uv = project_world_to_pixel(
            P_w=tip_world,
            K=video.intrinsics.K_undist,
            R_wc=video.R_wc,
            t_wc=video.camera_pos,
        )
        if tip_uv is None:
            return None

        ex = int(round(float(tip_uv[0])))
        ey = int(round(float(tip_uv[1])))
        return p0, (ex, ey)

    def _draw_overlay(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.current is not None
        video = self.current["video"]
        frame = self.current["frame"]
        pred_angle = float(self.current["pred_angle"])
        frame_idx = int(self.current["frame_idx"])
        yaw_deg = self._yaw_deg_for_video(video)
        pred_angle_vis = self._visual_angle_with_yaw(video, pred_angle)
        label_angle_vis = self._visual_angle_with_yaw(video, self.current_angle_deg)

        vis = frame.copy()
        t, b, l, r = self.detector.roi_crop
        cv2.rectangle(vis, (l, t), (r, b), (0, 255, 255), 2)
        cv2.putText(
            vis,
            "ROI",
            (l + 5, max(20, t - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        hinge = video.hinge_px
        if hinge is not None:
            cv2.circle(vis, hinge, 6, (255, 180, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, hinge, 11, (0, 0, 0), 2, cv2.LINE_AA)

        pred_line = self._line_endpoints_px(video, pred_angle, hinge, length_scale=1.0) if hinge is not None else None
        adj_line = self._line_endpoints_px(video, self.current_angle_deg, hinge, length_scale=2.2) if hinge is not None else None

        if pred_line is not None:
            cv2.line(vis, pred_line[0], pred_line[1], (0, 220, 255), 2, cv2.LINE_AA)
        if adj_line is not None:
            cv2.line(vis, adj_line[0], adj_line[1], (80, 255, 80), 3, cv2.LINE_AA)

        lines = [
            f"Video: {video.path.name}",
            f"Frame: {frame_idx}/{max(video.total_frames - 1, 0)}",
            f"Model angle: {pred_angle:.2f} deg",
            f"Label angle: {self.current_angle_deg:.2f} deg",
            (
                f"Sampling: {'HIGH-ANGLE ONLY' if self.high_angle_only else 'all frames'} "
                f"(|angle-90| high: <= {self._high_angle_bounds()[0]:.1f} or >= {self._high_angle_bounds()[1]:.1f} deg, H=toggle)"
            ),
            (
                f"Yaw correction: yaw={yaw_deg:+.2f} deg | "
                f"model_vis={pred_angle_vis:.2f} label_vis={label_angle_vis:.2f}"
            ),
            (
                f"Hinge px: ({int(hinge[0])}, {int(hinge[1])})"
                if hinge is not None
                else "Hinge px: (unset)"
            ),
            describe_pose(video.camera_pos, video.angles, video.pnp_error_px),
            f"Saved: {self.saved_count}  Skipped: {self.skipped_count}",
            "Keys: LEFT/RIGHT or A/D=rotate  ,/.=0.5deg  UP/DOWN or W/S=coarse  ENTER/SPACE=save",
            "      I/J/K/L=hinge nudge (up/left/down/right)  Y/U/B/O=coarse nudge",
            "      C=click hinge  N=next/skip  H=toggle high-angle sampling  R=reset  Q/ESC=quit",
            "      (hinge is per-video)",
        ]
        if self.await_hinge_click:
            lines.append("HINGE MODE: click a point in the image.")

        y = 28
        for txt in lines:
            cv2.putText(vis, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
            y += 26

        # ROI zoom window
        h, w = vis.shape[:2]
        t2 = int(np.clip(t, 0, h))
        b2 = int(np.clip(b, 0, h))
        l2 = int(np.clip(l, 0, w))
        r2 = int(np.clip(r, 0, w))
        roi = vis[t2:b2, l2:r2]
        if roi.size == 0:
            roi = np.zeros((320, 640, 3), dtype=np.uint8)
        roi_zoom = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        vis_disp = (
            cv2.resize(vis, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)
            if self.display_scale != 1.0
            else vis
        )
        return vis_disp, roi_zoom

    def _save_current_label(self) -> None:
        assert self.current is not None
        video: VideoState = self.current["video"]
        frame = self.current["frame"]
        frame_idx = int(self.current["frame_idx"])
        pred_angle = float(self.current["pred_angle"])
        label_angle = float(self.current_angle_deg)
        hinge = video.hinge_px if video.hinge_px is not None else (-1, -1)

        filename = f"img_{self.next_index:06d}.jpg"
        img_path = self.images_dir / filename
        ok = cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not ok:
            print(f"[WARN] Failed to save image: {img_path}")
            return

        cam = (
            np.asarray(video.camera_pos, dtype=np.float64).reshape(3)
            if video.camera_pos is not None
            else np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        )
        ang = video.angles or {}
        t, b, l, r = self.detector.roi_crop

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    filename,
                    f"{label_angle:.6f}",
                    str(video.path),
                    frame_idx,
                    f"{pred_angle:.6f}",
                    f"{float(cam[0]):.6f}",
                    f"{float(cam[1]):.6f}",
                    f"{float(cam[2]):.6f}",
                    f"{float(ang.get('pitch_deg', np.nan)):.6f}",
                    f"{float(ang.get('yaw_deg', np.nan)):.6f}",
                    f"{float(ang.get('roll_deg', np.nan)):.6f}",
                    f"{float(video.pnp_error_px if video.pnp_error_px is not None else np.nan):.6f}",
                    int(hinge[0]),
                    int(hinge[1]),
                    int(t),
                    int(b),
                    int(l),
                    int(r),
                    f"{self.plane_z:.6f}",
                    f"{self.line_length_m:.6f}",
                    f"{time.time():.6f}",
                ]
            )

        self.next_index += 1
        self.saved_count += 1
        video.used_label_frames.add(frame_idx)
        print(
            f"[SAVE] {filename} | angle={label_angle:.2f} deg | "
            f"video={video.path.name} frame={frame_idx}"
        )

    def _mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        userdata: Any,
    ) -> None:
        _ = flags
        _ = userdata
        if event != cv2.EVENT_LBUTTONDOWN or not self.await_hinge_click or self.current is None:
            return
        video: VideoState = self.current["video"]
        scale = self.display_scale if self.display_scale > 0 else 1.0
        hx = int(round(x / scale))
        hy = int(round(y / scale))
        self._set_video_hinge(video, hx, hy)
        self.await_hinge_click = False
        print(f"[HINGE] {video.path.name}: ({hx}, {hy})")

    def run(self) -> None:
        cv2.namedWindow(self.window_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_roi, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_main, self._mouse_callback)

        print("\nInteractive labeling started.")
        print(
            "Controls: LEFT/RIGHT or A/D rotate, ,/. half-degree, UP/DOWN or W/S coarse, "
            "ENTER/SPACE save, I/J/K/L hinge, Y/U/B/O coarse hinge, C click hinge, "
            "N skip, H toggle high-angle sampling, R reset, Q quit."
        )
        low, high = self._high_angle_bounds()
        print(
            f"Sampling mode: {'HIGH-ANGLE ONLY' if self.high_angle_only else 'all frames'} "
            f"(high angles are <= {low:.1f} or >= {high:.1f} deg)."
        )

        while True:
            self.current = self._pick_random_sample()
            if self.current is None:
                if self.high_angle_only:
                    low, high = self._high_angle_bounds()
                    print(
                        "[WARN] Could not find a sample outside high-angle band "
                        f"(<= {low:.1f} or >= {high:.1f} deg). "
                        "Disabling high-angle-only mode.",
                        flush=True,
                    )
                    self.high_angle_only = False
                    continue
                print("[ERROR] Could not sample a valid frame. Exiting session.")
                break

            self.current_angle_deg = float(np.clip(self.current["pred_angle"], 0.0, 180.0))

            while True:
                vis, roi_zoom = self._draw_overlay()
                cv2.imshow(self.window_main, vis)
                cv2.imshow(self.window_roi, roi_zoom)

                key = cv2.waitKeyEx(20)
                if key < 0:
                    continue

                if key in self.LEFT_KEYS:
                    self.current_angle_deg = float(
                        np.clip(self.current_angle_deg + self.angle_step_deg, 0.0, 180.0)
                    )
                elif key in self.RIGHT_KEYS:
                    self.current_angle_deg = float(
                        np.clip(self.current_angle_deg - self.angle_step_deg, 0.0, 180.0)
                    )
                elif key in (ord(","), ord("<")):
                    self.current_angle_deg = float(
                        np.clip(self.current_angle_deg + self.half_step_deg, 0.0, 180.0)
                    )
                elif key in (ord("."), ord(">")):
                    self.current_angle_deg = float(
                        np.clip(self.current_angle_deg - self.half_step_deg, 0.0, 180.0)
                    )
                elif key in self.DOWN_KEYS:
                    self.current_angle_deg = float(
                        np.clip(self.current_angle_deg + 5.0 * self.angle_step_deg, 0.0, 180.0)
                    )
                elif key in self.UP_KEYS:
                    self.current_angle_deg = float(
                        np.clip(self.current_angle_deg - 5.0 * self.angle_step_deg, 0.0, 180.0)
                    )
                elif key in (ord("j"), ord("J"), ord("l"), ord("L"), ord("i"), ord("I"), ord("k"), ord("K"),
                             ord("y"), ord("Y"), ord("u"), ord("U"), ord("b"), ord("B"), ord("o"), ord("O")):
                    video = self.current["video"]
                    if video.hinge_px is None:
                        video.hinge_px = self._default_hinge_for_video(video)
                    hx, hy = int(video.hinge_px[0]), int(video.hinge_px[1])
                    # Fine nudge: 1 px. Coarse nudge: 10 px.
                    if key in (ord("j"), ord("J")):
                        hx -= 1
                    elif key in (ord("l"), ord("L")):
                        hx += 1
                    elif key in (ord("i"), ord("I")):
                        hy -= 1
                    elif key in (ord("k"), ord("K")):
                        hy += 1
                    elif key in (ord("u"), ord("U")):
                        hx -= 10
                    elif key in (ord("o"), ord("O")):
                        hx += 10
                    elif key in (ord("y"), ord("Y")):
                        hy -= 10
                    elif key in (ord("b"), ord("B")):
                        hy += 10
                    self._set_video_hinge(video, hx, hy)
                elif key in (13, 10, 32):  # enter, return, space
                    self._save_current_label()
                    break
                elif key in (ord("n"), ord("N")):
                    self.skipped_count += 1
                    print(
                        f"[SKIP] video={self.current['video'].path.name} "
                        f"frame={self.current['frame_idx']}"
                    )
                    break
                elif key in (ord("h"), ord("H")):
                    self.high_angle_only = not self.high_angle_only
                    mode = "ON" if self.high_angle_only else "OFF"
                    low, high = self._high_angle_bounds()
                    print(
                        f"[INFO] High-angle-only sampling {mode} "
                        f"(<= {low:.1f} or >= {high:.1f} deg).",
                        flush=True,
                    )
                    # Immediately resample based on the new mode.
                    break
                elif key in (ord("r"), ord("R")):
                    self.current_angle_deg = float(np.clip(self.current["pred_angle"], 0.0, 180.0))
                elif key in (ord("c"), ord("C")):
                    self.await_hinge_click = True
                    print("[INFO] Click hinge point in main window.")
                elif key in (27, ord("q"), ord("Q")):
                    elapsed = time.time() - self.start_time
                    print(
                        f"[DONE] Exiting. saved={self.saved_count}, skipped={self.skipped_count}, "
                        f"elapsed={elapsed:.1f}s"
                    )
                    cv2.destroyAllWindows()
                    return

            self.await_hinge_click = False

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Video / pose setup
# ---------------------------------------------------------------------------
def get_video_metadata(path: Path) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if total <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return total, width, height, fps if np.isfinite(fps) and fps > 0 else 30.0


def read_frame(path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def estimate_camera_pose_for_video(
    video: VideoState,
    pnp: AutoPnPEstimator,
    sample_frames: int,
    min_valid_frames: int,
    rng: random.Random,
) -> Dict[str, Any]:
    total = int(video.total_frames)
    low = max(0, int(0.05 * total))
    high = min(total - 1, max(low, int(0.95 * total)))
    if high < low:
        low, high = 0, max(0, total - 1)

    max_trials = max(sample_frames * 10, sample_frames + 20)
    tried = set()
    successes: List[Dict[str, Any]] = []
    failures = 0

    for trial_i in range(max_trials):
        frame_idx = rng.randint(low, high) if high > low else low
        if frame_idx in tried and len(tried) < (high - low + 1):
            continue
        tried.add(frame_idx)

        if (trial_i % 5) == 0:
            print(
                f"    collecting PnP solves: tried={len(tried)}/{max_trials} "
                f"valid={len(successes)}/{sample_frames} fails={failures}",
                flush=True,
            )

        frame = read_frame(video.path, frame_idx)
        if frame is None:
            failures += 1
            continue
        frame_u = maybe_undistort(frame, video.intrinsics)
        attempt = pnp.attempt(frame_u, video.intrinsics.K_undist)
        if attempt.get("success"):
            attempt["frame_idx"] = int(frame_idx)
            successes.append(attempt)
            if len(successes) >= sample_frames:
                break
        else:
            failures += 1

    print(
        f"    finished PnP collection: valid={len(successes)}, fails={failures}, tried={len(tried)}",
        flush=True,
    )

    if len(successes) < max(1, int(min_valid_frames)):
        reason = "insufficient valid PnP frame solutions"
        if successes:
            reason += f" ({len(successes)} valid, need {min_valid_frames})"
        return {
            "success": False,
            "reason": reason,
            "num_successes": len(successes),
            "num_failures": failures,
        }

    averaged = average_pnp_solutions(successes)
    if not averaged.get("success"):
        return {
            "success": False,
            "reason": str(averaged.get("reason", "failed averaging")),
            "num_successes": len(successes),
            "num_failures": failures,
        }

    return {
        "success": True,
        "camera_pos": averaged["camera_pos"],
        "R_wc": averaged["R_wc"],
        "angles": averaged["angles"],
        "err_px": averaged["err_px"],
        "num_successes": len(successes),
        "num_failures": failures,
        "num_used_after_filter": int(averaged.get("num_used_after_filter", len(successes))),
        "num_rejected_outliers": int(averaged.get("num_rejected_outliers", 0)),
    }


def parse_roi_arg(roi_values: Optional[List[int]]) -> Optional[Tuple[int, int, int, int]]:
    if roi_values is None:
        return None
    if len(roi_values) != 4:
        raise ValueError("--roi must contain exactly 4 integers: top bottom left right")
    return tuple(int(v) for v in roi_values)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_onnx = root / "best_pilotnet.onnx"
    default_pth = root / "best_pilotnet.pth"
    default_weights = default_onnx if default_onnx.exists() else default_pth
    default_pnp_candidates = [root / "best.pt", root.parent / "best.pt"]
    default_pnp_model = next(
        (p for p in default_pnp_candidates if p.exists()),
        default_pnp_candidates[0],
    )
    parser = argparse.ArgumentParser(description="Standalone GUI tool for generating rudder angle labels.")
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Input video paths. If omitted, GUI file picker is used.",
    )
    parser.add_argument("--calibration", type=str, default="", help="Optional fisheye calibration .npz")
    parser.add_argument("--output-dir", type=str, default="", help="Dataset output folder. If omitted, GUI folder picker is used.")
    parser.add_argument(
        "--pilotnet-weights",
        type=str,
        default=str(default_weights),
        help="PilotNet weights file (.onnx recommended, or .pth/.pt).",
    )
    parser.add_argument("--pilotnet-meta", type=str, default=str(root / "best_pilotnet_meta.json"))
    parser.add_argument(
        "--pnp-model",
        type=str,
        default=str(default_pnp_model),
        help="AutoPnP YOLO model (.pt). Defaults to best.pt in script dir, then parent dir.",
    )
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))
    parser.add_argument("--pnp-samples", type=int, default=7, help="Target valid frame-level PnP solves per video")
    parser.add_argument("--pnp-min-valid", type=int, default=4, help="Minimum valid frame-level solves required per video")
    parser.add_argument("--pnp-min-kpt-conf", type=float, default=0.8)
    parser.add_argument("--line-length-m", type=float, default=0.55)
    parser.add_argument("--plane-z", type=float, default=0.0)
    parser.add_argument("--hinge-u", type=int, default=None, help="Initial hinge x pixel (can be outside frame).")
    parser.add_argument("--hinge-v", type=int, default=None, help="Initial hinge y pixel (can be outside frame).")
    parser.add_argument("--angle-step", type=float, default=1.0)
    parser.add_argument("--display-scale", type=float, default=0.8)
    parser.add_argument(
        "--high-angle-threshold",
        type=float,
        default=105.0,
        help=(
            "High-angle upper threshold in degrees (used by H toggle). "
            "Sampling keeps angle <= (180-threshold) or >= threshold. "
            "Example: 105 -> keep <=75 or >=105."
        ),
    )
    parser.add_argument(
        "--high-angle-only",
        action="store_true",
        help="Start session with high-angle-only frame sampling enabled.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for PilotNet inference")
    parser.add_argument(
        "--no-gui-dialogs",
        action="store_true",
        help="Do not open file/folder dialogs; require CLI paths or defaults.",
    )
    parser.add_argument(
        "--allow-default-pose",
        action="store_true",
        help="If AutoPnP fails/unavailable, continue labeling with default camera pose.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> Tuple[List[Path], Optional[Path], Path]:
    # Videos
    if args.videos:
        videos = [Path(v) for v in args.videos]
    else:
        if args.no_gui_dialogs:
            raise RuntimeError(
                "No videos provided. Use --videos <file1> <file2> ... when --no-gui-dialogs is set."
            )
        print(
            "[INFO] Opening video picker dialog. If you do not see it, Alt-Tab to find it, "
            "or rerun with --videos <file1> <file2> ...",
            flush=True,
        )
        videos = choose_files_gui(
            "Select video files",
            [("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
            multiple=True,
        )
    videos = [p.resolve() for p in videos if p.exists()]
    if not videos:
        raise RuntimeError("No videos selected.")

    # Calibration (optional)
    calib_path: Optional[Path] = None
    if args.calibration:
        p = Path(args.calibration).resolve()
        if p.exists():
            calib_path = p
        else:
            print(f"[WARN] Calibration file not found: {p}. Continuing without calibration.")
    else:
        if not args.no_gui_dialogs:
            print(
                "[INFO] Opening optional calibration picker dialog (you can Cancel).",
                flush=True,
            )
            picked = choose_files_gui(
                "Select calibration .npz (optional)",
                [("NPZ files", "*.npz"), ("All files", "*.*")],
                multiple=False,
            )
            if picked:
                calib_path = picked[0].resolve()

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        if args.no_gui_dialogs:
            output_dir = Path.cwd() / f"rudder_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            print(
                "[INFO] Opening output folder picker dialog. If canceled, a folder is auto-created.",
                flush=True,
            )
            picked_dir = choose_dir_gui("Select output dataset folder")
            output_dir = (
                picked_dir.resolve()
                if picked_dir is not None
                else Path.cwd() / f"rudder_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    return videos, calib_path, output_dir


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))

    print(
        "[INFO] Starting rudder labeler... (this can open file dialogs before main logs appear)",
        flush=True,
    )

    if (args.hinge_u is None) ^ (args.hinge_v is None):
        print("[ERROR] You must provide both --hinge-u and --hinge-v together.", flush=True)
        return 2

    try:
        roi_override = parse_roi_arg(args.roi)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        show_gui_error("Argument error", str(exc))
        return 2

    try:
        video_paths, calib_path, output_dir = resolve_inputs(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        show_gui_error("Input selection error", str(exc))
        return 2

    print("\nStandalone rudder labeler", flush=True)
    print(f"- Videos: {len(video_paths)}", flush=True)
    print(f"- Calibration: {calib_path if calib_path else 'none'}", flush=True)
    print(f"- Output dir: {output_dir}", flush=True)
    print(f"- Python: {sys.executable}", flush=True)
    pnp_model_path = Path(args.pnp_model).expanduser().resolve()
    print(f"- AutoPnP model: {pnp_model_path}", flush=True)

    # Load single-model PilotNet detector
    weights_path = Path(args.pilotnet_weights).resolve()
    meta_path = Path(args.pilotnet_meta).resolve() if args.pilotnet_meta else None
    try:
        detector = SinglePilotNetDetector(
            weights_path=weights_path,
            meta_path=meta_path,
            roi_override=roi_override,
            device="cpu" if args.cpu else "auto",
        )
    except Exception as exc:
        msg = f"Failed to load PilotNet model: {exc}"
        print(f"[ERROR] {msg}", flush=True)
        show_gui_error("PilotNet load error", msg)
        return 2
    print(
        f"- PilotNet loaded ({detector.backend}) | "
        f"roi={detector.roi_crop} input={detector.input_h}x{detector.input_w} "
        f"mean={detector.angle_mean:.4f} std={detector.angle_std:.4f} "
        f"domain={detector.angle_domain} yaw_runtime={detector.runtime_yaw_correction} | "
        f"weights={weights_path}",
        flush=True,
    )
    if meta_path is not None:
        print(f"- PilotNet metadata: {meta_path}", flush=True)

    # Build video states
    video_states: List[VideoState] = []
    for p in video_paths:
        try:
            total, width, height, fps = get_video_metadata(p)
            intr = build_intrinsics_for_video(width, height, calib_path)
            vs = VideoState(
                path=p,
                total_frames=total,
                width=width,
                height=height,
                fps=fps,
                intrinsics=intr,
            )
            video_states.append(vs)
            print(
                f"[VIDEO] {p.name}: {width}x{height}, frames={total}, fps={fps:.2f}, intrinsics={intr.notes}",
                flush=True,
            )
        except Exception as exc:
            print(f"[WARN] Skipping video {p}: {exc}", flush=True)

    if not video_states:
        print("[ERROR] No usable videos.", flush=True)
        show_gui_error("Video error", "No usable videos were found.")
        return 2

    # AutoPnP estimation
    pnp_estimator: Optional[AutoPnPEstimator] = None
    force_default_pose = False
    try:
        pnp_estimator = AutoPnPEstimator(
            model_path=pnp_model_path,
            min_kpt_conf=float(args.pnp_min_kpt_conf),
        )
    except FileNotFoundError as exc:
        if args.allow_default_pose:
            force_default_pose = True
            print(
                f"[WARN] AutoPnP unavailable ({exc}). Continuing with default camera pose because --allow-default-pose is set.",
                flush=True,
            )
        else:
            script_root = Path(__file__).resolve().parent
            msg = (
                f"Failed to initialize AutoPnP: {exc}\n\n"
                f"Expected model path: {pnp_model_path}\n"
                "Set --pnp-model <path-to-best.pt>, or place best.pt in:\n"
                f"- {script_root}\n"
                f"- {script_root.parent}\n"
                "Or run with --allow-default-pose to continue without AutoPnP."
            )
            print(f"[ERROR] {msg}", flush=True)
            show_gui_error("AutoPnP setup error", msg)
            return 2
    except Exception as exc:
        if args.allow_default_pose:
            force_default_pose = True
            print(
                f"[WARN] AutoPnP unavailable ({exc}). Continuing with default camera pose because --allow-default-pose is set.",
                flush=True,
            )
        else:
            install_cmd = f'"{sys.executable}" -m pip install ultralytics'
            msg = (
                f"Failed to initialize AutoPnP: {exc}\n\n"
                f"Active interpreter: {sys.executable}\n"
                f"Install dependency: {install_cmd}\n"
                "Or run with --allow-default-pose to continue without AutoPnP."
            )
            print(f"[ERROR] {msg}", flush=True)
            show_gui_error("AutoPnP setup error", msg)
            return 2

    print("\nEstimating camera pose for each video (AutoPnP)...", flush=True)
    for vs in video_states:
        print(f"[PnP] {vs.path.name} ...", flush=True)
        if force_default_pose:
            cam, R_wc = default_camera_pose_and_rotation()
            vs.camera_pos = cam
            vs.R_wc = R_wc
            vs.angles = camera_pose_angles_from_rwc(R_wc)
            vs.pnp_error_px = float("nan")
            print("  [WARN] Using default camera pose.", flush=True)
        else:
            assert pnp_estimator is not None
            result = estimate_camera_pose_for_video(
                video=vs,
                pnp=pnp_estimator,
                sample_frames=max(1, int(args.pnp_samples)),
                min_valid_frames=max(1, int(args.pnp_min_valid)),
                rng=rng,
            )
            if result.get("success"):
                vs.camera_pos = np.asarray(result["camera_pos"], dtype=np.float64).reshape(3)
                vs.R_wc = np.asarray(result["R_wc"], dtype=np.float64).reshape(3, 3)
                vs.angles = dict(result["angles"])
                vs.pnp_error_px = float(result.get("err_px", np.nan))
                print(
                    f"  [OK] {describe_pose(vs.camera_pos, vs.angles, vs.pnp_error_px)} | "
                    f"valid={result.get('num_successes', 0)} used={result.get('num_used_after_filter', 0)} "
                    f"rejected={result.get('num_rejected_outliers', 0)}",
                    flush=True,
                )
            elif args.allow_default_pose:
                cam, R_wc = default_camera_pose_and_rotation()
                vs.camera_pos = cam
                vs.R_wc = R_wc
                vs.angles = camera_pose_angles_from_rwc(R_wc)
                vs.pnp_error_px = float("nan")
                print(
                    "  [WARN] AutoPnP failed for this video, using default camera pose "
                    f"({result.get('reason', 'unknown')})",
                    flush=True,
                )
            else:
                print(
                    f"  [WARN] PnP failed: {result.get('reason', 'unknown')} "
                    f"(valid={result.get('num_successes', 0)}, fails={result.get('num_failures', 0)})",
                    flush=True,
                )

    usable = [v for v in video_states if v.camera_pos is not None and v.R_wc is not None]
    if not usable:
        msg = (
            "No videos produced usable camera poses.\n"
            "Try lowering PnP thresholds, use better calibration, or run with --allow-default-pose."
        )
        print(f"[ERROR] {msg}", flush=True)
        show_gui_error("AutoPnP failed", msg)
        return 2

    print(
        f"\nPnP complete: {len(usable)}/{len(video_states)} videos ready for labeling.",
        flush=True,
    )
    print(f"Dataset output: {output_dir}", flush=True)

    initial_hinge_px = None
    if args.hinge_u is not None and args.hinge_v is not None:
        initial_hinge_px = (int(args.hinge_u), int(args.hinge_v))
        print(
            f"[INFO] Initial hinge override: ({initial_hinge_px[0]}, {initial_hinge_px[1]})",
            flush=True,
        )

    session = LabelingSession(
        videos=usable,
        detector=detector,
        output_dir=output_dir,
        plane_z=float(args.plane_z),
        line_length_m=float(args.line_length_m),
        angle_step_deg=float(args.angle_step),
        display_scale=float(args.display_scale),
        rng_seed=int(args.seed),
        initial_hinge_px=initial_hinge_px,
        high_angle_only=bool(args.high_angle_only),
        high_angle_threshold_deg=float(args.high_angle_threshold),
    )
    session.run()

    print("\nSession summary")
    print(f"- Saved labels: {session.saved_count}")
    print(f"- Skipped frames: {session.skipped_count}")
    print(f"- Labels CSV: {session.csv_path}")
    print(f"- Images dir: {session.images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
