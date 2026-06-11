from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from src.app.config import (
    DATA_ROOT,
    ENABLE_SKELETON_RAYCAST_PROCESSING,
    ENABLE_VIDEO_PROXY_TRANSCODE,
    MAP_MAX_POINTS_PER_TRACK,
    POSE_MODEL_PATH,
    PROJECTS_ROOT,
    SKELETON_HIP_PLANE_Z,
    SKELETON_LOWER_PLANE_Z,
    SKELETON_TARGET_FPS,
    TMP_UPLOAD_ROOT,
)
from src.app.db import MetadataStore
from src.app.services.assets import resolve_pose_model_path as resolve_pose_model_asset
from src.app.services.matcher import match_video_tracks_to_csv
from src.app.services.parsers import ParsedTrack, parse_csv_track, parse_gopro_video_track

logger = logging.getLogger(__name__)


def get_skeleton_artifact_paths(project_id: str, file_id: str, root: Path) -> dict[str, Path]:
    base = root / project_id / "derived" / "cv" / file_id
    return {
        "jsonl": base.with_suffix(".skeleton.jsonl"),
        "metrics_csv": base.with_suffix(".metrics.csv"),
        "summary_json": base.with_suffix(".summary.json"),
        "pose_csv": base.with_suffix(".pose.csv"),
        "autopnp_json": base.with_suffix(".autopnp_history.json"),

    }


def resolve_pose_model_path(raw_value: str | None) -> Path | None:
    raw = (raw_value or "").strip()
    if raw:
        candidate = Path(raw)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
        if raw.lower() in {"lite", "full", "heavy"}:
            return resolve_pose_model_asset(raw.lower())
    return resolve_pose_model_asset("full")


def run_skeleton_raycast_processing(
    *,
    video_path: Path,
    jsonl_path: Path,
    metrics_csv_path: Path,
    summary_path: Path,
    model_path: Path | None,
    target_fps: float,
    lower_plane_z: float,
    hip_plane_z: float,
) -> dict[str, object]:
    # Kept as an explicit no-op to avoid reintroducing the removed legacy hook.
    summary = {
        "status": "disabled",
        "reason": "legacy_skeleton_hook_disabled",
        "video_path": str(video_path),
        "jsonl_path": str(jsonl_path),
        "metrics_csv_path": str(metrics_csv_path),
        "model_path": str(model_path) if model_path is not None else None,
        "target_fps": float(target_fps),
        "lower_plane_z": float(lower_plane_z),
        "hip_plane_z": float(hip_plane_z),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, separators=(",", ":"))
    return summary


def ensure_project_dirs(project_id: str) -> dict[str, Path]:
    project_root = PROJECTS_ROOT / project_id
    paths = {
        "project_root": project_root,
        "raw_videos": project_root / "raw" / "videos",
        "raw_csv": project_root / "raw" / "csv",
        "derived_tracks": project_root / "derived" / "tracks",
        "derived_matches": project_root / "derived" / "alignment",
        "derived_proxies": project_root / "derived" / "proxies",
        "derived_cv": project_root / "derived" / "cv",
        "derived_exports": project_root / "derived" / "exports",
        "tmp_uploads": TMP_UPLOAD_ROOT,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return paths


def classify_file_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".mp4", ".mov"}:
        return "video"
    if ext == ".csv":
        return "csv"
    return "other"


def _downsample_for_map(
    points: list[dict[str, float]],
    max_points: int = MAP_MAX_POINTS_PER_TRACK,
) -> list[dict[str, float]]:
    if len(points) <= max_points:
        return points
    stride = max(1, len(points) // max_points)
    sampled = points[::stride]
    if sampled[-1] is not points[-1]:
        sampled.append(points[-1])
    return sampled


def sanitize_filename(filename: str) -> str:
    base = os.path.basename(filename)
    sanitized = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in base).strip()
    if not sanitized:
        sanitized = "uploaded_file"
    return sanitized


def get_video_proxy_path(project_id: str, file_id: str) -> Path:
    return PROJECTS_ROOT / project_id / "derived" / "proxies" / f"{file_id}.mp4"


def get_video_skeleton_paths(project_id: str, file_id: str) -> dict[str, Path]:
    return get_skeleton_artifact_paths(project_id, file_id, PROJECTS_ROOT)


def save_track_points(path: Path, points: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(points, handle, separators=(",", ":"))


def load_track_points(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    points: list[dict[str, float]] = []
    for item in raw:
        try:
            parsed = {
                "ts": float(item["ts"]),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
            }
            video_s = item.get("video_s")
            if video_s is not None:
                parsed["video_s"] = float(video_s)
            for opt_key in ("sog", "heel", "trim", "cog", "hdg"):
                val = item.get(opt_key)
                if val is not None:
                    parsed[opt_key] = float(val)
            points.append(parsed)
        except (KeyError, TypeError, ValueError):
            continue
    return points


_INSTRUMENT_KEYS = ("heel", "trim", "sog", "cog", "hdg")


def merge_csv_instruments_into_video_tracks(
    video_tracks: dict[str, list[dict]],
    store: "MetadataStore",
    project_id: str,
) -> dict[str, list[dict]]:
    """Interpolate instrument columns from matched CSV tracks onto video tracks.

    For each video, look up its best-matched CSV (rank=1). For every video
    track point, find the two nearest CSV track points by timestamp and
    linearly interpolate the instrument columns (heel, trim, sog, cog, hdg).

    Mutates and returns *video_tracks* for convenience.
    """
    import bisect

    matches = store.list_matches(project_id)
    best_csv_by_video: dict[str, str] = {}
    for m in matches:
        if m["rank"] == 1 and m["video_file_id"] not in best_csv_by_video:
            best_csv_by_video[m["video_file_id"]] = m["csv_file_id"]

    if not best_csv_by_video:
        return video_tracks

    # Load CSV tracks that are needed
    csv_points_cache: dict[str, list[dict]] = {}
    for csv_fid in set(best_csv_by_video.values()):
        track_row = store.get_track_by_file_id(csv_fid)
        if track_row and track_row.get("points_path"):
            pp = Path(str(track_row["points_path"]))
            if pp.exists():
                csv_points_cache[csv_fid] = load_track_points(pp)

    for video_fid, video_pts in video_tracks.items():
        csv_fid = best_csv_by_video.get(video_fid)
        if not csv_fid:
            continue
        csv_pts = csv_points_cache.get(csv_fid)
        if not csv_pts:
            continue

        # Build sorted timestamp array for bisect
        csv_times = [p["ts"] for p in csv_pts]
        n_csv = len(csv_times)
        if n_csv == 0:
            continue

        merged_count = 0
        for vp in video_pts:
            ts = vp.get("ts")
            if ts is None:
                continue

            # Find bracketing CSV points
            idx = bisect.bisect_right(csv_times, ts)
            lo_i = max(0, idx - 1)
            hi_i = min(idx, n_csv - 1)

            lo = csv_pts[lo_i]
            hi = csv_pts[hi_i]

            # Skip if too far from CSV coverage (>10 s gap)
            if ts < csv_times[0] - 10 or ts > csv_times[-1] + 10:
                continue

            if lo_i == hi_i or abs(hi["ts"] - lo["ts"]) < 1e-9:
                # Exact or single-point: copy directly
                for key in _INSTRUMENT_KEYS:
                    val = lo.get(key)
                    if val is not None and key not in vp:
                        vp[key] = val
                        merged_count += 1
            else:
                # Linear interpolation
                frac = (ts - lo["ts"]) / (hi["ts"] - lo["ts"])
                frac = max(0.0, min(1.0, frac))
                for key in _INSTRUMENT_KEYS:
                    v_lo = lo.get(key)
                    v_hi = hi.get(key)
                    if v_lo is not None and v_hi is not None and key not in vp:
                        vp[key] = v_lo + frac * (v_hi - v_lo)
                        merged_count += 1
                    elif v_lo is not None and key not in vp:
                        vp[key] = v_lo
                        merged_count += 1

        if merged_count > 0:
            logger.info(
                "Merged %d instrument values from CSV %s onto video %s",
                merged_count, csv_fid, video_fid,
            )

    return video_tracks


class PipelineService:
    def __init__(self, store: MetadataStore) -> None:
        self.store = store
        self.enable_video_proxy = ENABLE_VIDEO_PROXY_TRANSCODE
        self.ffmpeg_bin = shutil.which("ffmpeg") if self.enable_video_proxy else None
        self.enable_skeleton = ENABLE_SKELETON_RAYCAST_PROCESSING
        self.pose_model_path = resolve_pose_model_path(POSE_MODEL_PATH)

    def process_project(
        self,
        project_id: str,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> dict[str, int]:
        files = self.store.list_files(project_id)
        if not files:
            raise RuntimeError("Project has no files to process.")

        processable = [f for f in files if f["kind"] in {"video", "csv"}]
        total = max(1, len(processable))
        processed = 0
        failed = 0

        # Skip files that are already processed and have a valid track
        need_parse: list[dict] = []
        existing_tracks = {str(t["file_id"]): t for t in self.store.list_tracks(project_id)}
        for f in processable:
            fid = str(f["id"])
            status = str(f.get("status") or "")
            if (
                status == "processed"
                and fid in existing_tracks
                and int(existing_tracks[fid].get("point_count") or 0) >= 3
            ):
                processed += 1
                if progress_cb:
                    progress_cb(processed / total * 0.75, f"Skipped {processed}/{total} (already processed)")
                continue
            need_parse.append(f)

        # Parse remaining files in parallel (ExifTool / CSV parsing is I/O-bound)
        cpu_count = max(1, os.cpu_count() or 1)
        parse_workers = min(len(need_parse), cpu_count, 4)

        def _parse_one(file_row: dict) -> tuple[dict, ParsedTrack | None, str | None]:
            """Parse a single file, returning (file_row, parsed_track | None, error | None)."""
            kind = str(file_row["kind"])
            source_path = Path(str(file_row["path"]))
            try:
                parsed = self._parse_file(kind, source_path)
                if len(parsed.points) < 3:
                    return file_row, None, "Not enough GPS points parsed from file."
                return file_row, parsed, None
            except Exception as exc:  # pylint: disable=broad-except
                return file_row, None, str(exc)

        if need_parse:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=parse_workers, thread_name_prefix="parse"
            ) as pool:
                futures = {pool.submit(_parse_one, f): f for f in need_parse}
                for future in concurrent.futures.as_completed(futures):
                    file_row, parsed, error = future.result()
                    file_id = str(file_row["id"])
                    processed += 1
                    if error is not None:
                        failed += 1
                        logger.error("Failed to process %s: %s", file_row.get("path"), error)
                        self.store.update_file_status(file_id, "failed", error)
                    else:
                        assert parsed is not None
                        try:
                            self.store.update_file_status(file_id, "processing", None)
                            self._persist_track(project_id, file_row, parsed)
                            if str(file_row["kind"]) == "video" and self.enable_video_proxy:
                                self._ensure_video_proxy(project_id, file_row)
                            self.store.update_file_status(file_id, "processed", None)
                        except Exception as exc:  # pylint: disable=broad-except
                            failed += 1
                            processed -= 1  # wasn't truly processed
                            logger.exception("Failed to persist %s", file_row.get("path"))
                            self.store.update_file_status(file_id, "failed", str(exc))
                    if progress_cb:
                        progress_cb(processed / total * 0.75, f"Parsed {processed}/{total} files")

        if progress_cb:
            progress_cb(0.80, "Computing video/csv matches")
        matches = self._compute_matches(project_id)
        if progress_cb:
            progress_cb(1.0, "Completed matching and map tracks")
        return {"processed": processed, "failed": failed, "matches": len(matches)}

    def recompute_matches(self, project_id: str) -> int:
        matches = self._compute_matches(project_id)
        return len(matches)

    def _parse_file(self, kind: str, path: Path) -> ParsedTrack:
        if kind == "video":
            return parse_gopro_video_track(path)
        if kind == "csv":
            return parse_csv_track(path)
        raise RuntimeError(f"Unsupported parse kind: {kind}")

    def _persist_track(
        self,
        project_id: str,
        file_row: dict[str, str],
        parsed: ParsedTrack,
    ) -> None:
        track_path = PROJECTS_ROOT / project_id / "derived" / "tracks" / f"{file_row['id']}.json"
        save_track_points(track_path, parsed.points)
        ts_start = parsed.points[0]["ts"] if parsed.points else None
        ts_end = parsed.points[-1]["ts"] if parsed.points else None
        self.store.upsert_track(
            file_id=str(file_row["id"]),
            project_id=project_id,
            kind=str(file_row["kind"]),
            point_count=len(parsed.points),
            ts_start=ts_start,
            ts_end=ts_end,
            points_path=str(track_path),
            meta=parsed.metadata,
        )

    def _process_video_skeleton(self, project_id: str, file_row: dict[str, str]) -> dict[str, object]:
        if not self.enable_skeleton:
            return {"status": "disabled", "reason": "skeleton_processing_disabled"}

        file_id = str(file_row["id"])
        source_path = Path(str(file_row["path"]))
        paths = get_video_skeleton_paths(project_id, file_id)
        model_path = self.pose_model_path or resolve_pose_model_path(POSE_MODEL_PATH)
        if model_path is not None and self.pose_model_path is None:
            self.pose_model_path = model_path
        summary = run_skeleton_raycast_processing(
            video_path=source_path,
            jsonl_path=paths["jsonl"],
            metrics_csv_path=paths["metrics_csv"],
            summary_path=paths["summary_json"],
            model_path=model_path,
            target_fps=SKELETON_TARGET_FPS,
            lower_plane_z=SKELETON_LOWER_PLANE_Z,
            hip_plane_z=SKELETON_HIP_PLANE_Z,
        )
        status = str(summary.get("status") or "")
        if status in {"failed", "skipped"}:
            logger.warning(
                "Skeleton processing %s for %s: %s",
                status,
                source_path,
                summary.get("reason"),
            )
        return summary

    def _ensure_video_proxy(self, project_id: str, file_row: dict[str, str]) -> None:
        if self.ffmpeg_bin is None:
            return

        file_id = str(file_row["id"])
        source_path = Path(str(file_row["path"]))
        if not source_path.exists():
            return

        proxy_path = get_video_proxy_path(project_id, file_id)
        proxy_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if (
                proxy_path.exists()
                and proxy_path.stat().st_size > 0
                and proxy_path.stat().st_mtime >= source_path.stat().st_mtime
            ):
                return
        except OSError:
            pass

        temp_path = proxy_path.with_suffix(".tmp.mp4")
        temp_path.unlink(missing_ok=True)

        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-vf",
            "scale=min(1280\\,iw):-2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ac",
            "2",
            str(temp_path),
        ]

        run = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if run.returncode != 0:
            temp_path.unlink(missing_ok=True)
            err = (run.stderr or "").strip()
            logger.warning("Proxy generation failed for %s: %s", source_path, err[:400])
            return
        temp_path.replace(proxy_path)

    def _compute_matches(self, project_id: str) -> list[dict[str, float | int | str]]:
        file_map = {file_row["id"]: file_row for file_row in self.store.list_files(project_id)}
        video_tracks = []
        csv_tracks = []
        for track_row in self.store.list_tracks(project_id):
            file_id = track_row["file_id"]
            file_row = file_map.get(file_id)
            if file_row is None:
                continue
            points = load_track_points(Path(track_row["points_path"]))
            if not points:
                continue
            payload = {
                "file_id": file_id,
                "filename": file_row["filename"],
                "points": points,
            }
            if track_row["kind"] == "video":
                video_tracks.append(payload)
            elif track_row["kind"] == "csv":
                csv_tracks.append(payload)

        matches = match_video_tracks_to_csv(video_tracks, csv_tracks, max_rank_per_video=3)
        self.store.replace_matches(project_id, matches)

        out_path = PROJECTS_ROOT / project_id / "derived" / "alignment" / "matches.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(matches, handle, indent=2)
        return matches

    def get_map_payload(self, project_id: str) -> dict[str, object]:
        files = self.store.list_files(project_id)
        tracks = self.store.list_tracks(project_id)
        matches = self.store.list_matches(project_id)
        files_by_id = {item["id"]: item for item in files}

        best_match_by_video: dict[str, str] = {}
        for match in matches:
            if match["rank"] == 1 and match["video_file_id"] not in best_match_by_video:
                best_match_by_video[match["video_file_id"]] = match["csv_file_id"]

        videos: list[dict[str, object]] = []
        csvs: list[dict[str, object]] = []
        for track in tracks:
            file_id = track["file_id"]
            file_row = files_by_id.get(file_id)
            if file_row is None:
                continue
            points = load_track_points(Path(track["points_path"]))
            payload = {
                "file_id": file_id,
                "filename": file_row["filename"],
                "point_count": int(track["point_count"]),
                "ts_start": track["ts_start"],
                "ts_end": track["ts_end"],
                "points": _downsample_for_map(points),
                "best_match_csv_id": best_match_by_video.get(file_id),
            }
            if track["kind"] == "video":
                videos.append(payload)
            elif track["kind"] == "csv":
                csvs.append(payload)

        return {
            "project_id": project_id,
            "videos": videos,
            "csvs": csvs,
            "matches": matches,
            "files": files,
        }
