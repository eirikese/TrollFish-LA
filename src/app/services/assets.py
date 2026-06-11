from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_REF_ROOT = Path(r"C:\Users\larso\webinterfacepose")


def _candidate_paths(filename: str, env_var: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if env_var:
        raw = os.getenv(env_var, "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            WORKSPACE_ROOT / "assets" / filename,
            WORKSPACE_ROOT / filename,
            WORKSPACE_ROOT / "Forinspiration" / filename,
            DEFAULT_EXTERNAL_REF_ROOT / filename,
        ]
    )
    return candidates


def resolve_asset_path(filename: str, env_var: str | None = None) -> Path | None:
    for candidate in _candidate_paths(filename, env_var=env_var):
        path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def resolve_pose_model_path(model_name: str = "full") -> Path | None:
    model_name = (model_name or "full").strip().lower()
    if model_name not in {"lite", "full", "heavy"}:
        model_name = "full"
    return resolve_asset_path(
        f"pose_landmarker_{model_name}.task",
        env_var="TROLLFISH_POSE_MODEL",
    )


def resolve_auto_pnp_model_path() -> Path | None:
    return resolve_asset_path("best.pt", env_var="TROLLFISH_AUTO_PNP_MODEL")


def resolve_hull_stl_path() -> Path | None:
    return resolve_asset_path("Hull.stl", env_var="TROLLFISH_HULL_STL")


def resolve_default_calibration_path() -> Path | None:
    return resolve_asset_path("gopro_fisheye_calib.npz", env_var="TROLLFISH_GOPRO_CALIB_NPZ")


def resolve_gopro13_calibration_path() -> Path | None:
    return resolve_asset_path("GOPRO13CALIB.npz", env_var="TROLLFISH_GOPRO13_CALIB_NPZ")
