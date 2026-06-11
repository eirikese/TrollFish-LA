from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from src.app.db import MetadataStore
from src.app.services.cv_pipeline import CvService
from src.app.services.pipeline import PipelineService

logger = logging.getLogger(__name__)


class _JobCancelled(Exception):
    """Raised inside progress callbacks when the job has been cancelled."""


class JobWorker:
    def __init__(
        self,
        store: MetadataStore,
        pipeline: PipelineService,
        cv_service: CvService,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self.cv_service = cv_service
        self.poll_interval_s = poll_interval_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cancel_current_job = threading.Event()
        self._current_job_id: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="local-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            job = self.store.fetch_next_pending_job()
            if job is None:
                time.sleep(self.poll_interval_s)
                continue
            self._execute_job(job)

    def _execute_job(self, job_row: dict[str, Any]) -> None:
        job_id = str(job_row["id"])
        job_type = str(job_row["type"])
        self._current_job_id = job_id
        self._cancel_current_job.clear()
        try:
            payload = self._parse_payload(job_row)

            def _progress_cb(progress: float, message: str) -> None:
                if self._cancel_current_job.is_set():
                    raise _JobCancelled()
                self.store.update_job_progress(job_id, progress, message)

            if job_type == "process_project":
                project_id = str(payload["project_id"])
                self.pipeline.process_project(
                    project_id=project_id,
                    progress_cb=_progress_cb,
                )
                self.store.complete_job(job_id, "Project processed")
            elif job_type == "process_skeleton_batch":
                project_id = str(payload["project_id"])
                video_file_ids = payload.get("video_file_ids")
                if not isinstance(video_file_ids, list):
                    video_file_ids = None
                force = bool(payload.get("force", False))
                mediapipe_workers_val = payload.get("mediapipe_workers")
                video_ranges_val = payload.get("video_ranges") or {}
                description_val = payload.get("description") or None
                self.cv_service.process_skeleton_batch(
                    project_id=project_id,
                    video_file_ids=video_file_ids,
                    force=force,
                    progress_cb=_progress_cb,
                    mediapipe_workers=int(mediapipe_workers_val) if mediapipe_workers_val is not None else None,
                    video_ranges=video_ranges_val if isinstance(video_ranges_val, dict) else None,
                    description=description_val,
                )
                self.store.complete_job(job_id, "Skeleton batch processed")
            elif job_type == "generate_report":
                project_id = str(payload["project_id"])
                self._generate_report(project_id, _progress_cb)
                self.store.complete_job(job_id, "PDF report generated")
            else:
                raise RuntimeError(f"Unknown job type: {job_type}")
        except _JobCancelled:
            logger.info("Job cancelled: %s", job_id)
            # DB status already set to 'cancelled' by the cancel endpoint
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Worker job failed: %s", job_id)
            self.store.fail_job(job_id, str(exc))
        finally:
            self._current_job_id = None

    def request_cancel(self, job_id: str) -> bool:
        """Signal the worker to cancel the given job. Returns True if it was the current job."""
        if self._current_job_id == job_id:
            self._cancel_current_job.set()
            return True
        return False

    def _generate_report(self, project_id: str, progress_cb) -> None:
        """Generate a PDF report with progress updates."""
        from pathlib import Path
        from src.app.config import PROJECTS_ROOT
        from src.app.services.report import build_report_data
        from src.app.services.report_pdf import generate_pdf_report
        from src.app.services.pipeline import load_track_points, merge_csv_instruments_into_video_tracks

        progress_cb(0.05, "Loading segments…")

        # Load segments
        seg_path = PROJECTS_ROOT / project_id / "segments.json"
        if not seg_path.exists():
            raise RuntimeError("No segments.json found")
        segments = json.loads(seg_path.read_text(encoding="utf-8"))
        if not segments:
            raise RuntimeError("No segments defined")
        segment_ids = [s["id"] for s in segments]

        progress_cb(0.10, "Loading project data…")

        # Load athletes + file meta
        ath_path = PROJECTS_ROOT / project_id / "athletes.json"
        athletes = json.loads(ath_path.read_text(encoding="utf-8")) if ath_path.exists() else []
        fm_path = PROJECTS_ROOT / project_id / "file_metadata.json"
        file_meta = json.loads(fm_path.read_text(encoding="utf-8")) if fm_path.exists() else {}

        progress_cb(0.15, "Loading track points…")

        # Track points
        all_files = self.store.list_files(project_id)
        video_files = [f for f in all_files if str(f.get("kind")) == "video"]
        all_track_points: dict[str, list[dict]] = {}
        for vf in video_files:
            fid = str(vf["id"])
            track_row = self.store.get_track_by_file_id(fid)
            if track_row and track_row.get("points_path"):
                pp = Path(str(track_row["points_path"]))
                if pp.exists():
                    all_track_points[fid] = load_track_points(pp)

        progress_cb(0.20, "Merging instrument data…")
        merge_csv_instruments_into_video_tracks(all_track_points, self.store, project_id)

        progress_cb(0.25, "Building segment mapping…")

        # Build synthetic file meta with per-video splits
        synthetic_meta: dict[str, dict] = {}
        for fid, meta in file_meta.items():
            synthetic_meta[fid] = {**meta, "splits": []}

        for seg in segments:
            ts_start = seg.get("tsStart", 0)
            ts_end = seg.get("tsEnd", 0)
            for fid, pts in all_track_points.items():
                if not pts:
                    continue
                tss = [p.get("ts") for p in pts if p.get("ts") is not None]
                if not tss:
                    continue
                fmin, fmax = min(tss), max(tss)
                if ts_start > fmax or ts_end < fmin:
                    continue

                def _ts2vs(target, _pts=pts):
                    bi = 0
                    for i, p in enumerate(_pts):
                        if p.get("ts") is not None and p["ts"] <= target:
                            bi = i
                    lo = _pts[bi]
                    hi_i = min(bi + 1, len(_pts) - 1)
                    hi = _pts[hi_i]
                    if lo.get("video_s") is None:
                        return None
                    if hi_i == bi or hi.get("ts") is None or hi.get("video_s") is None:
                        return lo["video_s"]
                    if hi["ts"] == lo["ts"]:
                        return lo["video_s"]
                    fr = (target - lo["ts"]) / (hi["ts"] - lo["ts"])
                    return lo["video_s"] + fr * (hi["video_s"] - lo["video_s"])

                vs0 = _ts2vs(max(ts_start, fmin))
                vs1 = _ts2vs(min(ts_end, fmax))
                if vs0 is None or vs1 is None:
                    continue
                if fid not in synthetic_meta:
                    synthetic_meta[fid] = {"splits": []}
                synthetic_meta[fid].setdefault("splits", []).append({
                    "id": seg["id"],
                    "name": seg.get("name", "Segment"),
                    "start_s": vs0,
                    "end_s": vs1,
                })

        involved = {fid for fid, m in synthetic_meta.items() if m.get("splits")}
        skel_paths: dict[str, Path] = {}
        for fid in involved:
            row = self.cv_service.get_video_status(project_id, fid)
            if row and row.get("skeleton_jsonl_path"):
                p = Path(str(row["skeleton_jsonl_path"]))
                if p.exists():
                    skel_paths[fid] = p

        cv_cfg = self.cv_service.get_project_config(project_id)

        progress_cb(0.35, "Building report data…")

        report_data = build_report_data(
            project_id=project_id,
            split_ids=segment_ids,
            file_meta=synthetic_meta,
            athletes=athletes,
            track_points_by_file={fid: all_track_points[fid] for fid in involved if fid in all_track_points},
            skeleton_jsonl_paths=skel_paths,
            cv_config=cv_cfg,
        )

        progress_cb(0.55, "Generating PDF…")

        export_dir = PROJECTS_ROOT / project_id / "derived" / "exports"
        pdf_path = generate_pdf_report(
            report_data, project_id, export_dir,
            full_track_points=all_track_points,
            file_meta=file_meta,
            athletes=athletes,
        )
        progress_cb(1.0, "PDF report complete")
        logger.info("Generated PDF report: %s", pdf_path)

    @staticmethod
    def _parse_payload(job_row: dict[str, Any]) -> dict[str, Any]:
        payload_json = str(job_row.get("payload_json") or "{}")
        try:
            return json.loads(payload_json)
        except json.JSONDecodeError:
            return {}
