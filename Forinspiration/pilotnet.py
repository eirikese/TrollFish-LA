#!/usr/bin/env python3
"""
Small PilotNet trainer/exporter for the GoPro boom-angle dataset.

The important choice in this file is that angles are never trained as plain
degrees. Boom azimuth wraps at -180/180, so -179 deg and +179 deg are only
2 deg apart physically but 358 deg apart as scalar numbers. The model therefore
learns a circular target:

    target = [sin(angle_rad), cos(angle_rad)]

Validation MAE is computed with circular distance in degrees, and ONNX export
wraps the model so the exported output is directly an angle in degrees.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "gopro_boom_training_dataset"
FISHEYE_CALIB_NPZ = str(ROOT_DIR / "gopro_fisheye_calib1080.npz")

# PilotNet canonical input is height x width.
INPUT_SIZE = (66, 200)

# Default for this dataset. A data-dir/roi_config.json overrides this by default.
ROI_CROP = (250, 350, 750, 1170)  # top, bottom, left, right

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{value:.0f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def estimate_tensor_cache_bytes(num_images: int, input_size: tuple[int, int] = INPUT_SIZE) -> int:
    channels = 3
    bytes_per_float32 = 4
    return int(num_images) * channels * int(input_size[0]) * int(input_size[1]) * bytes_per_float32


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------


def wrap_degrees_np(values: np.ndarray | float) -> np.ndarray:
    """Wrap degrees into [-180, 180)."""
    arr = np.asarray(values, dtype=np.float32)
    return ((arr + 180.0) % 360.0) - 180.0


def wrap_degrees_tensor(values: torch.Tensor) -> torch.Tensor:
    """Wrap degrees into [-180, 180)."""
    return torch.remainder(values + 180.0, 360.0) - 180.0


def angle_to_unit_np(angle_deg: np.ndarray) -> np.ndarray:
    """Encode degrees as [sin(theta), cos(theta)]."""
    radians = np.deg2rad(wrap_degrees_np(angle_deg)).astype(np.float32)
    return np.stack((np.sin(radians), np.cos(radians)), axis=-1).astype(np.float32)


def unit_to_angle_deg_tensor(unit_or_raw: torch.Tensor) -> torch.Tensor:
    """Decode model output [sin, cos] into degrees in [-180, 180)."""
    return torch.atan2(unit_or_raw[..., 0], unit_or_raw[..., 1]) * (180.0 / math.pi)


def circular_abs_error_deg_np(pred_deg: np.ndarray, target_deg: np.ndarray) -> np.ndarray:
    return np.abs(wrap_degrees_np(np.asarray(pred_deg) - np.asarray(target_deg)))


def circular_mean_deg_np(angle_deg: np.ndarray) -> float:
    radians = np.deg2rad(wrap_degrees_np(angle_deg))
    s = float(np.mean(np.sin(radians)))
    c = float(np.mean(np.cos(radians)))
    return float(math.degrees(math.atan2(s, c)))


def angle_bins(angle_deg: np.ndarray, bin_width: int) -> np.ndarray:
    if bin_width <= 0 or 360 % int(bin_width) != 0:
        raise ValueError("--balance-bin-deg must be a positive divisor of 360")
    count = 360 // int(bin_width)
    bins = np.floor((wrap_degrees_np(angle_deg) + 180.0) / float(bin_width)).astype(np.int64)
    return np.clip(bins, 0, count - 1)


# ---------------------------------------------------------------------------
# ROI and preprocessing
# ---------------------------------------------------------------------------


def validate_roi(roi: Iterable[int]) -> tuple[int, int, int, int]:
    top, bottom, left, right = [int(v) for v in roi]
    if bottom <= top or right <= left:
        raise ValueError(f"Invalid ROI {tuple(roi)}; expected top,bottom,left,right with bottom>top and right>left")
    return top, bottom, left, right


def load_roi_from_config(path: str | Path) -> tuple[int, int, int, int]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "roi" not in data:
        raise ValueError(f"ROI config missing key 'roi': {path}")
    return validate_roi(data["roi"])


def resolve_roi(args: argparse.Namespace, data_dir: Path) -> tuple[int, int, int, int]:
    if args.roi is not None:
        return validate_roi(args.roi)
    roi_config = Path(args.roi_config) if args.roi_config else data_dir / "roi_config.json"
    if roi_config.exists():
        roi = load_roi_from_config(roi_config)
        print(f"Loaded ROI from {roi_config}: top={roi[0]} bottom={roi[1]} left={roi[2]} right={roi[3]}")
        return roi
    return validate_roi(ROI_CROP)


def clamp_roi_to_frame(roi: tuple[int, int, int, int], frame_bgr: np.ndarray) -> tuple[int, int, int, int]:
    top, bottom, left, right = roi
    height, width = frame_bgr.shape[:2]
    top = max(0, min(height - 1, int(top)))
    bottom = max(top + 1, min(height, int(bottom)))
    left = max(0, min(width - 1, int(left)))
    right = max(left + 1, min(width, int(right)))
    return top, bottom, left, right


def preprocess_bgr(
    frame_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    input_size: tuple[int, int] = INPUT_SIZE,
) -> torch.Tensor:
    """BGR frame -> ROI crop -> resize -> YUV -> float32 NCHW in [-1, 1]."""
    top, bottom, left, right = clamp_roi_to_frame(roi, frame_bgr)
    crop = frame_bgr[top:bottom, left:right]
    crop = cv2.resize(crop, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
    crop = crop.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.transpose(crop, (2, 0, 1))).contiguous()


def preprocess_image(
    img_path: str | Path,
    roi: tuple[int, int, int, int],
    input_size: tuple[int, int] = INPUT_SIZE,
) -> torch.Tensor:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    return preprocess_bgr(img, roi=roi, input_size=input_size)


def load_fisheye_undistort_maps(npz_path: str, balance: float = 0.0):
    """
    Kept for compatibility with pseudo-label tools. This trainer does not use
    undistortion for the current dataset because dataset_config.json says the
    images are stored as distorted_raw_gopro.
    """
    cal = np.load(npz_path)
    K, D = cal["K"], cal["D"]
    img_size = tuple(int(v) for v in cal["img_size"])
    width, height = int(img_size[0]), int(img_size[1])
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (width, height), np.eye(3), balance=float(balance), new_size=(width, height)
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (width, height), cv2.CV_16SC2
    )
    return map1, map2, (width, height)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BoomAngleDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        roi: tuple[int, int, int, int],
        input_size: tuple[int, int] = INPUT_SIZE,
        augment: bool = False,
        cache: str = "none",
        cache_device: torch.device | None = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir)
        self.roi = validate_roi(roi)
        self.input_size = tuple(int(v) for v in input_size)
        self.augment = bool(augment)
        self.cache = str(cache).lower()

        self.filenames = self.df["filename"].astype(str).to_numpy()
        self.angles_deg = self.df["__angle_deg"].to_numpy(dtype=np.float32)
        self.targets = torch.from_numpy(angle_to_unit_np(self.angles_deg))
        self.images: torch.Tensor | None = None

        if self.cache not in {"none", "ram", "gpu"}:
            raise ValueError("--cache must be one of: none, ram, gpu")

        if self.cache != "none":
            if self.cache == "gpu":
                if cache_device is None or cache_device.type != "cuda":
                    raise RuntimeError("--cache gpu requires --device cuda or an available CUDA device")
                tensor_device = cache_device
                where = f"GPU ({tensor_device})"
            else:
                tensor_device = torch.device("cpu")
                where = "RAM"

            estimate = estimate_tensor_cache_bytes(len(self.filenames), self.input_size)
            print(f"Caching {len(self.filenames)} preprocessed images in {where} ({format_bytes(estimate)})...")
            self.images = torch.empty(
                (len(self.filenames), 3, self.input_size[0], self.input_size[1]),
                dtype=torch.float32,
                device=tensor_device,
            )
            for i in range(len(self.filenames)):
                self.images[i] = self._load_image(i).to(tensor_device, non_blocking=False)

    def __len__(self) -> int:
        return len(self.filenames)

    def _image_path(self, idx: int) -> Path:
        name = Path(str(self.filenames[idx]))
        return name if name.is_absolute() else self.data_dir / name

    def _load_image(self, idx: int) -> torch.Tensor:
        return preprocess_image(self._image_path(idx), roi=self.roi, input_size=self.input_size)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return img

        # Photometric augmentation only. Geometric flips/rotations would change
        # the angle label, so they are intentionally not used.
        img = img.clone()
        device = img.device
        contrast = float(torch.empty((), device=img.device).uniform_(0.90, 1.10))
        brightness = float(torch.empty((), device=img.device).uniform_(-0.08, 0.08))
        img.mul_(contrast)
        img[0].add_(brightness)
        if torch.rand((), device=device) < 0.5:
            img.add_(torch.randn_like(img) * 0.01)
        return img.clamp_(-1.0, 1.0)

    def __getitem__(self, idx: int):
        img = self.images[idx] if self.images is not None else self._load_image(idx)
        return self._augment(img), self.targets[idx], torch.tensor(self.angles_deg[idx], dtype=torch.float32)


def load_labels(
    data_dir: Path,
    labels_name: str,
    target_col: str,
    include_videos: list[str] | None = None,
    exclude_videos: list[str] | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    labels_path = Path(labels_name)
    if not labels_path.is_absolute():
        labels_path = data_dir / labels_name
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    df = pd.read_csv(labels_path)
    if "filename" not in df.columns:
        raise ValueError(f"{labels_path} is missing required column 'filename'")
    if target_col not in df.columns:
        raise ValueError(f"{labels_path} is missing target column '{target_col}'")

    if include_videos and "gopro_video" in df.columns:
        wanted = {v.lower() for v in include_videos}
        df = df[df["gopro_video"].astype(str).str.lower().isin(wanted)].copy()
    if exclude_videos and "gopro_video" in df.columns:
        blocked = {v.lower() for v in exclude_videos}
        df = df[~df["gopro_video"].astype(str).str.lower().isin(blocked)].copy()
    if max_samples is not None:
        df = df.head(int(max_samples)).copy()

    raw = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)
    finite = np.isfinite(raw)
    if not finite.all():
        dropped = int((~finite).sum())
        print(f"Dropping {dropped} rows with non-finite '{target_col}'")
        df = df.loc[finite].copy()
        raw = raw[finite]
    if len(df) == 0:
        raise ValueError("No rows left after label filtering")

    wrapped = wrap_degrees_np(raw)
    changed = int(np.sum(np.abs(wrapped - raw) > 1e-4))
    if changed:
        print(f"Wrapped {changed} target values into [-180, 180).")
    df["__angle_deg"] = wrapped.astype(np.float32)
    return df.reset_index(drop=True)


def check_image_files(df: pd.DataFrame, data_dir: Path, limit: int | None = None) -> None:
    filenames = df["filename"].astype(str).tolist()
    if limit is not None:
        filenames = filenames[: int(limit)]
    missing = []
    for name in filenames:
        path = Path(name)
        path = path if path.is_absolute() else data_dir / path
        if not path.exists():
            missing.append(str(path))
            if len(missing) >= 10:
                break
    if missing:
        raise FileNotFoundError("Missing image files:\n" + "\n".join(missing))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PilotNet(nn.Module):
    """
    NVIDIA-style PilotNet adapted for circular angle regression.

    Input:  B x 3 x 66 x 200 preprocessed tensor
    Output: B x output_dim. For this script output_dim=2 => [sin, cos].
    """

    def __init__(self, dropout_rate: float = 0.2, output_dim: int = 2):
        super().__init__()
        self.output_dim = int(output_dim)
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ELU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(inplace=True),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(100, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


def build_model(
    arch: str = "pilotnet",
    dropout_rate: float = 0.2,
    output_dim: int = 2,
    **_: object,
) -> nn.Module:
    arch = str(arch).lower()
    if arch != "pilotnet":
        raise ValueError("This simplified script only builds arch='pilotnet'")
    return PilotNet(dropout_rate=dropout_rate, output_dim=output_dim)


def infer_arch_from_state_dict(state_dict: dict[str, torch.Tensor], fallback: str = "pilotnet") -> str:
    if any(key.startswith("classifier.") for key in state_dict.keys()):
        return "pilotnet"
    return fallback


def infer_output_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ("classifier.9.weight", "classifier.8.weight", "head.4.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0])
    return 2


def normalize_model_kwargs(
    arch: str,
    model_kwargs: dict,
    fallback_dropout: float = 0.2,
    fallback_width_mult: float = 1.0,
    fallback_coordconv: bool = True,
    fallback_output_dim: int = 2,
) -> dict:
    """Compatibility helper for older scripts that import pilotnet.py."""
    _ = arch, fallback_width_mult, fallback_coordconv
    return {
        "dropout_rate": float(model_kwargs.get("dropout_rate", fallback_dropout)),
        "output_dim": int(model_kwargs.get("output_dim", fallback_output_dim)),
    }


class UnitVectorLoss(nn.Module):
    """Circular loss for raw [sin, cos] predictions."""

    def __init__(self, norm_weight: float = 0.05):
        super().__init__()
        self.norm_weight = float(norm_weight)

    def forward(self, pred_raw: torch.Tensor, target_unit: torch.Tensor) -> torch.Tensor:
        pred_unit = F.normalize(pred_raw, dim=-1, eps=1e-6)
        dot = torch.sum(pred_unit * target_unit, dim=-1).clamp(-1.0, 1.0)
        direction_loss = (1.0 - dot).mean()
        norm_loss = (pred_raw.norm(dim=-1) - 1.0).pow(2).mean()
        return direction_loss + self.norm_weight * norm_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Splitting, balancing, and audit output
# ---------------------------------------------------------------------------


def split_dataframe(
    df: pd.DataFrame,
    val_split: float,
    split_mode: str,
    bin_width: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < float(val_split) < 1.0:
        raise ValueError("--val-split must be between 0 and 1")
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(df))
    split_mode = str(split_mode).lower()

    if split_mode == "temporal":
        n_val = max(1, int(round(len(df) * float(val_split))))
        train_idx = indices[:-n_val]
        val_idx = indices[-n_val:]
    elif split_mode == "video":
        if "gopro_video" not in df.columns:
            raise ValueError("--split-mode video requires a gopro_video column")
        videos = df["gopro_video"].astype(str).drop_duplicates().tolist()
        if len(videos) < 2:
            raise ValueError("--split-mode video needs at least two distinct videos")
        target_val = int(round(len(df) * float(val_split)))
        val_videos: list[str] = []
        val_count = 0
        for video in reversed(videos):
            val_videos.append(video)
            val_count += int((df["gopro_video"].astype(str) == video).sum())
            if val_count >= target_val:
                break
        val_mask = df["gopro_video"].astype(str).isin(val_videos).to_numpy()
        train_idx = indices[~val_mask]
        val_idx = indices[val_mask]
        print(f"Validation videos: {', '.join(reversed(val_videos))}")
    elif split_mode == "stratified":
        bins = angle_bins(df["__angle_deg"].to_numpy(dtype=np.float32), bin_width)
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for bin_id in np.unique(bins):
            bin_idx = indices[bins == bin_id]
            rng.shuffle(bin_idx)
            if len(bin_idx) <= 1:
                train_parts.append(bin_idx)
                continue
            n_val = int(round(len(bin_idx) * float(val_split)))
            n_val = max(1, min(len(bin_idx) - 1, n_val))
            val_parts.append(bin_idx[:n_val])
            train_parts.append(bin_idx[n_val:])
        train_idx = np.concatenate(train_parts)
        val_idx = np.concatenate(val_parts)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
    else:
        raise ValueError("--split-mode must be stratified, temporal, or video")

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split produced an empty train or validation set")
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def make_balanced_sampler(
    train_df: pd.DataFrame,
    bin_width: int,
    balance_strength: float,
    max_weight_ratio: float,
) -> WeightedRandomSampler:
    bins = angle_bins(train_df["__angle_deg"].to_numpy(dtype=np.float32), bin_width)
    counts = np.bincount(bins, minlength=360 // int(bin_width)).astype(np.float64)
    strength = max(0.0, min(1.0, float(balance_strength)))
    per_sample = counts[bins] ** (-strength)
    if max_weight_ratio > 0:
        min_weight = float(np.min(per_sample))
        per_sample = np.minimum(per_sample, min_weight * float(max_weight_ratio))
    per_sample = per_sample / np.mean(per_sample)
    return WeightedRandomSampler(
        weights=torch.as_tensor(per_sample, dtype=torch.double),
        num_samples=len(per_sample),
        replacement=True,
    )


def make_balanced_subset(
    train_df: pd.DataFrame,
    bin_width: int,
    max_samples_per_bin: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Downsample only the overrepresented angle bins.

    This keeps all rare bins and caps the big clusters, producing a physically
    smaller and easier-to-audit training set. If max_samples_per_bin <= 0, choose
    a soft automatic cap from the 75th percentile of non-empty bin counts.
    """
    rng = np.random.default_rng(int(seed))
    bins = angle_bins(train_df["__angle_deg"].to_numpy(dtype=np.float32), bin_width)
    counts = np.bincount(bins, minlength=360 // int(bin_width)).astype(np.int64)
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        raise ValueError("Cannot balance an empty training set")

    if int(max_samples_per_bin) > 0:
        cap = int(max_samples_per_bin)
        cap_source = "manual"
    else:
        cap = int(math.ceil(float(np.percentile(nonzero, 75))))
        cap = max(1, cap)
        cap_source = "auto_p75_nonempty_bin_count"

    kept_parts: list[np.ndarray] = []
    all_indices = np.arange(len(train_df))
    for bin_id in np.unique(bins):
        bin_idx = all_indices[bins == bin_id]
        if len(bin_idx) > cap:
            bin_idx = rng.choice(bin_idx, size=cap, replace=False)
        kept_parts.append(np.asarray(bin_idx, dtype=np.int64))

    kept = np.concatenate(kept_parts)
    rng.shuffle(kept)
    balanced_df = train_df.iloc[kept].reset_index(drop=True)
    stats = {
        "enabled": True,
        "bin_width_deg": int(bin_width),
        "max_samples_per_bin": int(cap),
        "cap_source": cap_source,
        "rows_before": int(len(train_df)),
        "rows_after": int(len(balanced_df)),
        "rows_removed": int(len(train_df) - len(balanced_df)),
    }
    return balanced_df, stats


def histogram_counts(angle_deg: np.ndarray, bin_width: int) -> np.ndarray:
    bins = angle_bins(angle_deg, bin_width)
    return np.bincount(bins, minlength=360 // int(bin_width))


def print_angle_audit(df: pd.DataFrame, bin_width: int, title: str) -> dict:
    angles = df["__angle_deg"].to_numpy(dtype=np.float32)
    counts = histogram_counts(angles, bin_width)
    nonzero = counts[counts > 0]
    imbalance_ratio = float(nonzero.max() / max(1, nonzero.min())) if len(nonzero) else 0.0

    print(f"\n{title}")
    print(f"Rows: {len(df)}")
    print(
        "Angle deg: "
        f"min={float(np.min(angles)):.2f} max={float(np.max(angles)):.2f} "
        f"mean={float(np.mean(angles)):.2f} circular_mean={circular_mean_deg_np(angles):.2f} "
        f"std={float(np.std(angles)):.2f}"
    )
    print(f"Non-empty {bin_width}-deg bins: {int(np.sum(counts > 0))}/{len(counts)}")
    print(f"Max/min non-empty bin ratio: {imbalance_ratio:.1f}x")

    top = np.argsort(-counts)[: min(12, len(counts))]
    print("Most common angle bins:")
    for bin_id in top:
        if counts[bin_id] == 0:
            continue
        start = -180 + bin_id * bin_width
        end = start + bin_width
        pct = 100.0 * float(counts[bin_id]) / float(len(df))
        print(f"  [{start:4d},{end:4d}) deg: {int(counts[bin_id]):6d}  {pct:5.1f}%")

    return {
        "rows": int(len(df)),
        "min_deg": float(np.min(angles)),
        "max_deg": float(np.max(angles)),
        "mean_deg": float(np.mean(angles)),
        "circular_mean_deg": circular_mean_deg_np(angles),
        "std_deg": float(np.std(angles)),
        "bin_width_deg": int(bin_width),
        "non_empty_bins": int(np.sum(counts > 0)),
        "bin_count": int(len(counts)),
        "max_min_nonempty_bin_ratio": imbalance_ratio,
        "histogram": [int(v) for v in counts.tolist()],
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def resolve_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device_arg not in {"cuda", "cpu"}:
        raise ValueError("--device must be auto, cuda, or cpu")
    return torch.device(device_arg)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for images, targets, _angles_deg in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        batch = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch
        total_count += batch
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    pred_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []

    for images, targets, angles_deg in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images)
        loss = criterion(preds, targets)

        pred_deg = unit_to_angle_deg_tensor(preds.float()).detach().cpu().numpy()
        target_deg = angles_deg.numpy()
        pred_all.append(pred_deg.astype(np.float32))
        target_all.append(target_deg.astype(np.float32))

        batch = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch
        total_count += batch

    pred = np.concatenate(pred_all)
    target = np.concatenate(target_all)
    errors = circular_abs_error_deg_np(pred, target)
    return total_loss / max(1, total_count), float(np.mean(errors)), float(np.median(errors))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    roi: tuple[int, int, int, int],
    train_stats: dict,
    val_stats: dict,
    best_epoch: int,
    best_val_mae: float,
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": "pilotnet",
            "model_kwargs": {
                "dropout_rate": float(args.dropout),
                "output_dim": 2,
            },
            "output_dim": 2,
            "target_mode": "sincos",
            "angle_encoding": "target=[sin(deg), cos(deg)] with circular MAE",
            "angle_mean": float(train_stats["mean_deg"]),
            "angle_std": float(train_stats["std_deg"]),
            "target_col": str(args.target),
            "roi": list(roi),
            "roi_config": str(args.roi_config or (Path(args.data_dir) / "roi_config.json")),
            "input_size": list(INPUT_SIZE),
            "preprocess": "BGR image -> raw GoPro ROI -> resize -> YUV -> [-1, 1] -> NCHW float32",
            "image_geometry": "distorted_raw_gopro",
            "best_epoch": int(best_epoch),
            "best_val_mae_deg": float(best_val_mae),
            "train_angle_stats": train_stats,
            "val_angle_stats": val_stats,
            "balance_mode": str(args.balance_mode),
            "balanced_sampler": bool(args.balance_mode in {"sampler", "both"}),
            "balanced_subset": getattr(args, "_balanced_subset_stats", {"enabled": False}),
            "balance_bin_deg": int(args.balance_bin_deg),
            "balance_strength": float(args.balance_strength),
            "max_sample_weight_ratio": float(args.max_sample_weight_ratio),
            "max_samples_per_bin": int(args.max_samples_per_bin),
            "cache": str(args.cache),
            "split_mode": str(args.split_mode),
            "history": history,
        },
        str(path),
    )


def train(args: argparse.Namespace) -> Path:
    global DEVICE
    DEVICE = resolve_device(args.device)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.cache == "gpu" and DEVICE.type != "cuda":
        raise RuntimeError("--cache gpu requires CUDA. Use --device cuda or choose --cache ram/none.")
    if args.cache == "gpu" and int(args.num_workers) != 0:
        print("--cache gpu keeps tensors on the main CUDA device; forcing --num-workers 0")
        args.num_workers = 0

    data_dir = Path(args.data_dir)
    roi = resolve_roi(args, data_dir)
    weights_path = Path(args.weights) if args.weights else data_dir / "best_pilotnet_boom.pth"

    df = load_labels(
        data_dir=data_dir,
        labels_name=args.labels,
        target_col=args.target,
        include_videos=args.include_videos,
        exclude_videos=args.exclude_videos,
        max_samples=args.max_samples,
    )
    if args.check_files:
        check_image_files(df, data_dir)

    all_stats = print_angle_audit(df, args.balance_bin_deg, "Full Dataset Angle Audit")
    if all_stats["max_min_nonempty_bin_ratio"] > 10:
        print("Balance warning: this dataset is strongly unbalanced. Use --balance-mode sampler, subset, or both.")
    if all_stats["non_empty_bins"] < all_stats["bin_count"] // 2:
        print("Coverage warning: many angle bins are empty. Balancing cannot invent missing angles.")

    if args.audit_only:
        return weights_path

    train_df, val_df = split_dataframe(
        df,
        val_split=args.val_split,
        split_mode=args.split_mode,
        bin_width=args.balance_bin_deg,
        seed=args.seed,
    )

    if args.balance_mode in {"subset", "both"}:
        train_df, subset_stats = make_balanced_subset(
            train_df,
            bin_width=args.balance_bin_deg,
            max_samples_per_bin=args.max_samples_per_bin,
            seed=args.seed,
        )
        args._balanced_subset_stats = subset_stats
        print(
            "\nBalanced training subset: "
            f"{subset_stats['rows_before']} -> {subset_stats['rows_after']} rows, "
            f"cap={subset_stats['max_samples_per_bin']} samples/{args.balance_bin_deg}deg bin "
            f"({subset_stats['cap_source']})"
        )
    else:
        args._balanced_subset_stats = {"enabled": False}

    train_stats = print_angle_audit(train_df, args.balance_bin_deg, "Train Split Angle Audit")
    val_stats = print_angle_audit(val_df, args.balance_bin_deg, "Validation Split Angle Audit")

    train_ds = BoomAngleDataset(
        train_df,
        data_dir=data_dir,
        roi=roi,
        input_size=INPUT_SIZE,
        augment=args.augment,
        cache=args.cache,
        cache_device=DEVICE,
    )
    val_ds = BoomAngleDataset(
        val_df,
        data_dir=data_dir,
        roi=roi,
        input_size=INPUT_SIZE,
        augment=False,
        cache=args.cache,
        cache_device=DEVICE,
    )

    sampler = (
        make_balanced_sampler(
            train_df,
            args.balance_bin_deg,
            balance_strength=args.balance_strength,
            max_weight_ratio=args.max_sample_weight_ratio,
        )
        if args.balance_mode in {"sampler", "both"}
        else None
    )
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": DEVICE.type == "cuda" and args.cache != "gpu",
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=sampler is None,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = PilotNet(dropout_rate=float(args.dropout), output_dim=2).to(DEVICE)
    criterion = UnitVectorLoss(norm_weight=float(args.norm_weight))
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print("\nTraining")
    print(f"Device: {DEVICE}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Image cache: {args.cache}")
    print(f"Num workers: {loader_kwargs['num_workers']}")
    print(f"Balance mode: {args.balance_mode}")
    print(
        f"Balanced sampler: {bool(args.balance_mode in {'sampler', 'both'})} "
        f"(strength={float(args.balance_strength):.2f}, max_weight_ratio={float(args.max_sample_weight_ratio):.1f})"
    )
    print(f"Target: {args.target} encoded as [sin, cos]")
    print(f"ROI: top={roi[0]} bottom={roi[1]} left={roi[2]} right={roi[3]}")
    print(f"Weights: {weights_path}")

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_mae, val_median = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_mae)

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_mae_deg": float(val_mae),
            "val_median_abs_error_deg": float(val_median),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        improved = val_mae < best_val_mae - float(args.min_delta)
        if improved:
            best_val_mae = float(val_mae)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            save_checkpoint(
                weights_path,
                model=model,
                args=args,
                roi=roi,
                train_stats=train_stats,
                val_stats=val_stats,
                best_epoch=best_epoch,
                best_val_mae=best_val_mae,
                history=history,
            )
        else:
            epochs_without_improvement += 1

        print_every = max(1, int(args.print_every))
        if epoch == 1 or epoch % print_every == 0 or improved:
            marker = " best" if improved else ""
            print(
                f"Epoch {epoch:03d}/{int(args.epochs):03d} "
                f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"val_mae={val_mae:.2f}deg val_med={val_median:.2f}deg "
                f"lr={optimizer.param_groups[0]['lr']:.2e}{marker}"
            )

        if int(args.patience) > 0 and epochs_without_improvement >= int(args.patience):
            print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
            break

    history_path = weights_path.with_suffix(".training_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nBest validation MAE: {best_val_mae:.2f} deg at epoch {best_epoch}")
    print(f"Saved checkpoint: {weights_path}")
    print(f"Saved history: {history_path}")
    return weights_path


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class AngleDegreesWrapper(nn.Module):
    """Wrap a checkpoint so ONNX outputs degrees instead of raw [sin, cos]."""

    def __init__(self, model: nn.Module, target_mode: str, angle_mean: float, angle_std: float):
        super().__init__()
        self.model = model
        self.target_mode = str(target_mode)
        self.register_buffer("angle_mean", torch.tensor(float(angle_mean), dtype=torch.float32))
        self.register_buffer("angle_std", torch.tensor(float(angle_std), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if self.target_mode == "sincos":
            return unit_to_angle_deg_tensor(y.float())
        return y.float().reshape(-1) * self.angle_std + self.angle_mean


def load_checkpoint_model(weights_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {weights_path}")

    output_dim = int(checkpoint.get("output_dim", infer_output_dim_from_state_dict(state_dict))) if isinstance(checkpoint, dict) else infer_output_dim_from_state_dict(state_dict)
    kwargs = checkpoint.get("model_kwargs", {}) if isinstance(checkpoint, dict) else {}
    dropout = float(kwargs.get("dropout_rate", 0.0))
    model = PilotNet(dropout_rate=dropout, output_dim=output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    metadata = dict(checkpoint) if isinstance(checkpoint, dict) else {}
    metadata.pop("model_state_dict", None)
    metadata.setdefault("arch", "pilotnet")
    metadata.setdefault("output_dim", output_dim)
    metadata.setdefault("target_mode", "sincos" if output_dim == 2 else "scalar")
    metadata.setdefault("input_size", list(INPUT_SIZE))
    metadata.setdefault("roi", list(ROI_CROP))
    metadata.setdefault("target_col", "boom_angle")
    metadata.setdefault("angle_mean", 0.0)
    metadata.setdefault("angle_std", 1.0)
    return model, metadata


def add_onnx_metadata(onnx_path: Path, metadata: dict) -> None:
    try:
        import onnx  # type: ignore
    except Exception:
        return
    model_proto = onnx.load(str(onnx_path))
    model_proto.metadata_props.clear()
    for key, value in metadata.items():
        prop = model_proto.metadata_props.add()
        prop.key = str(key)
        prop.value = json.dumps(value) if isinstance(value, (dict, list, tuple, bool, int, float)) else str(value)
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(onnx_path))


def export_onnx(args: argparse.Namespace, weights_path: Path | None = None) -> Path:
    global DEVICE
    DEVICE = resolve_device(args.device)
    weights = Path(weights_path or args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model, metadata = load_checkpoint_model(weights, DEVICE)
    input_size = tuple(int(v) for v in metadata.get("input_size", INPUT_SIZE))
    if len(input_size) != 2:
        raise ValueError(f"Invalid checkpoint input_size: {input_size}")

    wrapped = AngleDegreesWrapper(
        model=model,
        target_mode=str(metadata.get("target_mode", "sincos")),
        angle_mean=float(metadata.get("angle_mean", 0.0)),
        angle_std=float(metadata.get("angle_std", 1.0)),
    ).to(DEVICE)
    wrapped.eval()

    onnx_path = Path(args.onnx_out) if args.onnx_out else weights.with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 3, input_size[0], input_size[1]), dtype=torch.float32, device=DEVICE)

    dynamic_axes = None if args.onnx_static_batch else {"input": {0: "batch"}, "angle_deg": {0: "batch"}}
    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["angle_deg"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.onnx_opset),
        do_constant_folding=True,
    )

    export_meta = dict(metadata)
    export_meta.update(
        {
            "weights": str(weights.resolve()),
            "onnx_path": str(onnx_path.resolve()),
            "onnx_opset": int(args.onnx_opset),
            "onnx_input_name": "input",
            "onnx_input_shape": ["batch", 3, input_size[0], input_size[1]],
            "onnx_input_dtype": "float32",
            "onnx_output_name": "angle_deg",
            "onnx_output_units": "degrees",
            "onnx_dynamic_batch": not bool(args.onnx_static_batch),
            "onnx_output_range": "[-180, 180)",
        }
    )
    metadata_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    metadata_path.write_text(json.dumps(export_meta, indent=2), encoding="utf-8")
    add_onnx_metadata(onnx_path, export_meta)

    print(f"Exported ONNX: {onnx_path}")
    print(f"Saved ONNX metadata: {metadata_path}")
    print(f"ONNX input: float32 NCHW, batch x 3 x {input_size[0]} x {input_size[1]}")
    print("ONNX output: angle_deg in degrees, decoded from [sin, cos]")
    return onnx_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/export a simplified PilotNet boom-angle model")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--labels", type=str, default="labels.csv")
    parser.add_argument("--target", type=str, default="boom_angle")
    parser.add_argument("--weights", type=str, default="")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--norm-weight", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["stratified", "temporal", "video"], default="stratified")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--balance-mode", choices=["none", "sampler", "subset", "both"], default="sampler",
                        help="none: use rows as-is; sampler: reweight batches; subset: downsample large angle bins; both: subset then sample.")
    parser.add_argument("--max-samples-per-bin", type=int, default=0,
                        help="For --balance-mode subset/both. 0 chooses an automatic cap from the train histogram.")
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=None,
                        help="Deprecated alias: --no-balanced-sampler maps to --balance-mode none.")
    parser.add_argument("--balance-bin-deg", type=int, default=10)
    parser.add_argument("--balance-strength", type=float, default=0.7,
                        help="0 disables reweighting; 1 makes non-empty angle bins nearly equal.")
    parser.add_argument("--max-sample-weight-ratio", type=float, default=50.0,
                        help="Cap rare-sample weight relative to common-sample weight. Use <=0 for no cap.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", choices=["none", "ram", "gpu"], default="none",
                        help="Cache preprocessed image tensors. gpu is fastest but requires VRAM and num-workers=0.")
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=None,
                        help="Deprecated alias: --cache-images maps to --cache ram.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-videos", nargs="*", default=None)
    parser.add_argument("--exclude-videos", nargs="*", default=None)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--audit-only", action="store_true")

    default_workers = 0 if os.name == "nt" else max(1, min(4, os.cpu_count() or 1))
    parser.add_argument("--num-workers", type=int, default=default_workers)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--print-every", type=int, default=1)

    parser.add_argument("--roi", type=int, nargs=4, default=None, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))
    parser.add_argument("--roi-config", type=str, default="")

    parser.add_argument("--export-onnx", action="store_true", help="Export --weights to ONNX and exit")
    parser.add_argument("--train-and-export", action="store_true", help="Train first, then export the best checkpoint")
    parser.add_argument("--onnx-out", type=str, default="")
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument("--onnx-static-batch", action="store_true")
    args = parser.parse_args()
    if args.balanced_sampler is False:
        args.balance_mode = "none"
    elif args.balanced_sampler is True and args.balance_mode == "none":
        args.balance_mode = "sampler"
    if args.cache_images is True and args.cache == "none":
        args.cache = "ram"
    elif args.cache_images is False:
        args.cache = "none"
    return args


def main() -> None:
    args = parse_args()
    if not args.weights:
        args.weights = str(Path(args.data_dir) / "best_pilotnet_boom.pth")

    if args.export_onnx:
        export_onnx(args)
        return

    weights_path = train(args)
    if args.train_and_export and not args.audit_only:
        export_onnx(args, weights_path=weights_path)


if __name__ == "__main__":
    main()
