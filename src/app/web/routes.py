from __future__ import annotations

import json
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Annotated, AsyncIterator

import anyio
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from src.app.config import PROJECTS_ROOT, UPLOAD_BUFFER_SIZE
from src.app.db import MetadataStore
from src.app.models import (
    CreateProjectRequest,
    JobResponse,
    MapDataResponse,
    PnPApplyRequest,
    PnPAutodetectRequest,
    PnPAutodetectResponse,
    PnPPair,
    PnPSolveRequest,
    PnPSolveResponse,
    ProjectCvConfig,
    ProjectCvConfigUpdate,
    ProjectResponse,
    SkeletonBatchRequest,
    SkeletonPreviewRequest,
    SkeletonPreviewResponse,
    UploadFileResult,
    UploadResponse,
    VideoCvPoseResponse,
    VideoCvStatus,
    VideoRangeRequest,
)
from src.app.services.assets import resolve_hull_stl_path
from src.app.services.cv_pipeline import CvService
from src.app.services.pipeline import (
    PipelineService,
    classify_file_kind,
    ensure_project_dirs,
    get_video_proxy_path,
    get_video_skeleton_paths,
    load_track_points,
    merge_csv_instruments_into_video_tracks,
    sanitize_filename,
)
from src.app.services.report import build_report_data
from src.app.worker import JobWorker


def get_store(request: Request) -> MetadataStore:
    return request.app.state.store


def get_pipeline(request: Request) -> PipelineService:
    return request.app.state.pipeline


def get_cv_service(request: Request) -> CvService:
    return request.app.state.cv_service


def get_worker(request: Request) -> JobWorker:
    return request.app.state.worker


router = APIRouter()
api = APIRouter(prefix="/api")
STREAM_CHUNK_SIZE = 1024 * 1024


def _guess_media_type(path: Path) -> str:
    _EXTRA_TYPES = {".mjs": "text/javascript", ".mts": "video/mp2t", ".wasm": "application/wasm", ".task": "application/octet-stream"}
    suffix = path.suffix.lower()
    if suffix in _EXTRA_TYPES:
        return _EXTRA_TYPES[suffix]
    guessed, _encoding = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _parse_http_range(range_header: str | None, file_size: int) -> tuple[int, int, bool]:
    if file_size <= 0:
        return 0, -1, False
    if not range_header:
        return 0, file_size - 1, False

    value = range_header.strip().lower()
    if not value.startswith("bytes="):
        return 0, file_size - 1, False

    first_range = value[6:].split(",", 1)[0].strip()
    if "-" not in first_range:
        raise ValueError("Invalid range format")

    start_raw, end_raw = first_range.split("-", 1)
    if not start_raw:
        suffix = int(end_raw)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        start = max(file_size - suffix, 0)
        end = file_size - 1
        return start, end, True

    start = int(start_raw)
    end = file_size - 1 if not end_raw else int(end_raw)
    if start < 0 or end < 0:
        raise ValueError("Invalid range values")
    if start >= file_size:
        raise ValueError("Range start beyond file size")
    end = min(end, file_size - 1)
    if end < start:
        raise ValueError("Range end before start")
    return start, end, True


async def _stream_file_chunks(path: Path, start: int, end: int) -> AsyncIterator[bytes]:
    if end < start:
        return
    async with await anyio.open_file(path, "rb") as handle:
        await handle.seek(start)
        remaining =  end - start + 1
        while remaining > 0:
            chunk = await handle.read(min(STREAM_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


async def _stream_file_response(path: Path, request: Request, media_type: str | None = None) -> Response:
    file_size = path.stat().st_size
    if file_size <= 0:
        return Response(
            content=b"",
            media_type=media_type or _guess_media_type(path),
            headers={"Accept-Ranges": "bytes", "Content-Length": "0"},
        )

    range_header = request.headers.get("range")
    try:
        start, end, is_partial = _parse_http_range(range_header, file_size)
    except ValueError:
        return Response(
            status_code=416,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes */{file_size}",
            },
        )

    content_length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Cache-Control": "private, max-age=3600",
    }
    status_code = 206 if is_partial else 200
    if is_partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

    return StreamingResponse(
        _stream_file_chunks(path, start, end),
        status_code=status_code,
        media_type=media_type or _guess_media_type(path),
        headers=headers,
    )


@router.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    here = Path(__file__).resolve().parent
    return FileResponse(here / "static" / "index.html")


@router.get("/report", response_class=FileResponse)
async def report_page() -> FileResponse:
    here = Path(__file__).resolve().parent
    return FileResponse(here / "static" / "report.html")


@router.get("/setup", response_class=FileResponse)
async def setup_page() -> FileResponse:
    here = Path(__file__).resolve().parent
    return FileResponse(here / "static" / "setup.html")


@router.get("/manifest.webmanifest")
async def manifest() -> FileResponse:
    here = Path(__file__).resolve().parent
    return FileResponse(
        here / "static" / "manifest.webmanifest",
        media_type="application/manifest+json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/sw.js")
async def service_worker() -> FileResponse:
    here = Path(__file__).resolve().parent
    return FileResponse(
        here / "static" / "sw.js",
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache",
            "Service-Worker-Allowed": "/",
        },
    )


@router.get("/static/{asset_path:path}")
async def static_asset(asset_path: str) -> FileResponse:
    static_root = (Path(__file__).resolve().parent / "static").resolve()
    target = (static_root / asset_path).resolve()
    if static_root not in target.parents and target != static_root:
        raise HTTPException(status_code=404, detail="Asset not found")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target, media_type=_guess_media_type(target))


@api.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@api.post("/projects", response_model=ProjectResponse)
async def create_project(
    payload: CreateProjectRequest,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> ProjectResponse:
    project = store.create_project(payload.name.strip())
    ensure_project_dirs(project["id"])
    return ProjectResponse(**project)


@api.get("/projects", response_model=list[ProjectResponse])
async def list_projects(
    store: Annotated[MetadataStore, Depends(get_store)],
) -> list[ProjectResponse]:
    return [ProjectResponse(**row) for row in store.list_projects()]


@api.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> dict[str, object]:
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if store.has_active_jobs(project_id):
        raise HTTPException(
            status_code=409,
            detail="Project has active jobs. Wait for completion before deleting.",
        )

    deleted_counts = store.delete_project(project_id)
    project_path = PROJECTS_ROOT / project_id
    if project_path.exists():
        shutil.rmtree(project_path, ignore_errors=True)

    return {
        "deleted": bool(deleted_counts.get("projects", 0)),
        "project_id": project_id,
        "counts": deleted_counts,
    }


@api.post("/projects/{project_id}/uploads", response_model=UploadResponse)
async def upload_files(
    project_id: str,
    files: Annotated[list[UploadFile], File(description="CSV/MP4 files")],
    store: Annotated[MetadataStore, Depends(get_store)],
    enqueue_process: bool = Query(default=True),
) -> UploadResponse:
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    dirs = ensure_project_dirs(project_id)
    uploaded_results: list[UploadFileResult] = []
    for upload in files:
        original_name = sanitize_filename(upload.filename or "file")
        kind = classify_file_kind(original_name)
        if kind not in {"video", "csv"}:
            await upload.close()
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for '{original_name}'. Allowed: .mp4/.mov/.csv",
            )
        if kind == "video":
            target_dir = dirs["raw_videos"]
        elif kind == "csv":
            target_dir = dirs["raw_csv"]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        target_name = f"{uuid.uuid4().hex}_{original_name}"
        target_path = target_dir / target_name
        size_bytes = await _save_upload_stream(upload, target_path)
        row = store.insert_file(
            project_id=project_id,
            filename=original_name,
            kind=kind,
            path=str(target_path),
            size_bytes=size_bytes,
            status="uploaded",
        )
        uploaded_results.append(
            UploadFileResult(
                id=str(row["id"]),
                filename=str(row["filename"]),
                kind=str(row["kind"]),
                size_bytes=int(row["size_bytes"]),
                status=str(row["status"]),
            )
        )

    job_id: str | None = None
    if enqueue_process:
        job_row = store.create_job(
            project_id=project_id,
            job_type="process_project",
            payload={"project_id": project_id, "trigger": "upload"},
            priority=100,
        )
        job_id = str(job_row["id"])
    return UploadResponse(
        project_id=project_id,
        files=uploaded_results,
        job_id=job_id,
    )


@api.post("/projects/{project_id}/process", response_model=JobResponse)
async def enqueue_project_processing(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> JobResponse:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    job_row = store.create_job(
        project_id=project_id,
        job_type="process_project",
        payload={"project_id": project_id, "trigger": "manual"},
        priority=100,
    )
    return JobResponse(
        id=str(job_row["id"]),
        project_id=str(job_row["project_id"]),
        type=str(job_row["type"]),
        status=str(job_row["status"]),
        progress=float(job_row["progress"]),
        message=job_row.get("message"),
        error=job_row.get("error"),
        created_at=str(job_row["created_at"]),
        started_at=job_row.get("started_at"),
        finished_at=job_row.get("finished_at"),
    )


@api.get("/projects/{project_id}/jobs", response_model=list[JobResponse])
async def list_project_jobs(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> list[JobResponse]:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    rows = store.list_jobs(project_id, limit=100)
    return [
        JobResponse(
            id=str(row["id"]),
            project_id=str(row["project_id"]),
            type=str(row["type"]),
            status=str(row["status"]),
            progress=float(row["progress"]),
            message=row.get("message"),
            error=row.get("error"),
            created_at=str(row["created_at"]),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
        )
        for row in rows
    ]


@api.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> JobResponse:
    row = store.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        type=str(row["type"]),
        status=str(row["status"]),
        progress=float(row["progress"]),
        message=row.get("message"),
        error=row.get("error"),
        created_at=str(row["created_at"]),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )


@api.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    worker: Annotated[JobWorker, Depends(get_worker)],
) -> dict:
    row = store.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    cancelled = store.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail="Job cannot be cancelled (already completed or failed)")
    # Signal the worker thread to stop if this is the currently running job
    worker.request_cancel(job_id)
    return {"ok": True, "job_id": job_id}


@api.get("/projects/{project_id}/cv-config", response_model=ProjectCvConfig)
async def get_project_cv_config(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> ProjectCvConfig:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    cfg = cv_service.get_project_config(project_id)
    return ProjectCvConfig(**cfg)


@api.put("/projects/{project_id}/cv-config", response_model=ProjectCvConfig)
async def update_project_cv_config(
    project_id: str,
    payload: ProjectCvConfigUpdate,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> ProjectCvConfig:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    cfg = cv_service.update_project_config(project_id, payload.model_dump(exclude_unset=True))
    return ProjectCvConfig(**cfg)


@api.post("/projects/{project_id}/jobs/skeleton", response_model=JobResponse)
async def enqueue_skeleton_batch_job(
    project_id: str,
    payload: SkeletonBatchRequest,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> JobResponse:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    job_row = store.create_job(
        project_id=project_id,
        job_type="process_skeleton_batch",
        payload={
            "project_id": project_id,
            "video_file_ids": payload.video_file_ids,
            "force": bool(payload.force),
            "mediapipe_workers": payload.mediapipe_workers,
            "video_ranges": {
                fid: {"start_sec": r.start_sec, "end_sec": r.end_sec}
                for fid, r in (payload.video_ranges or {}).items()
            },
            "description": payload.description,
        },
        priority=95,
    )
    return JobResponse(
        id=str(job_row["id"]),
        project_id=str(job_row["project_id"]),
        type=str(job_row["type"]),
        status=str(job_row["status"]),
        progress=float(job_row["progress"]),
        message=job_row.get("message"),
        error=job_row.get("error"),
        created_at=str(job_row["created_at"]),
        started_at=job_row.get("started_at"),
        finished_at=job_row.get("finished_at"),
    )


@api.get("/projects/{project_id}/cv/statuses")
async def get_all_cv_statuses(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> dict[str, object]:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    files = [f for f in store.list_files(project_id) if str(f.get("kind")) == "video"]
    statuses: dict[str, object] = {}
    for f in files:
        fid = str(f["id"])
        row = cv_service.get_video_status(project_id, fid)
        if row is not None:
            statuses[fid] = {
                "status": str(row.get("status") or "unknown"),
                "progress": float(row.get("progress") or 0.0),
                "message": row.get("message"),
                "error": row.get("error"),
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"),
            }
        else:
            statuses[fid] = {"status": "none", "progress": 0.0}
    return statuses


@api.get("/projects/{project_id}/videos/{file_id}/cv/status", response_model=VideoCvStatus)
async def get_video_cv_status(
    project_id: str,
    file_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> VideoCvStatus:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    row = cv_service.get_video_status(project_id, file_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No CV run found for this video")
    return VideoCvStatus(
        project_id=project_id,
        file_id=file_id,
        status=str(row.get("status") or "unknown"),
        progress=float(row.get("progress") or 0.0),
        message=row.get("message"),
        error=row.get("error"),
        summary_json_path=row.get("summary_json_path"),
        metrics_csv_path=row.get("metrics_csv_path"),
        skeleton_jsonl_path=row.get("skeleton_jsonl_path"),
        pose_csv_path=row.get("pose_csv_path"),
        autopnp_json_path=row.get("autopnp_json_path"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )


@api.get("/projects/{project_id}/videos/{file_id}/cv/skeleton-coverage")
async def get_skeleton_coverage(
    project_id: str,
    file_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> dict[str, object]:
    """Return list of [start_s, end_s] processed intervals from skeleton JSONL."""
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    row = cv_service.get_video_status(project_id, file_id)
    if row is None or not row.get("skeleton_jsonl_path"):
        return {"intervals": []}
    path = Path(str(row["skeleton_jsonl_path"]))
    if not path.exists():
        return {"intervals": []}
    GAP_S = 0.5
    intervals: list[list[float]] = []
    seg_start: float | None = None
    last_s: float | None = None
    content = await anyio.to_thread.run_sync(lambda: path.read_text(encoding="utf-8"))
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line).get("video_s")
        except Exception:
            continue
        if t is None:
            continue
        t = float(t)
        if seg_start is None:
            seg_start = t
            last_s = t
        elif t - last_s > GAP_S:  # type: ignore[operator]
            intervals.append([seg_start, last_s])
            seg_start = t
        last_s = t
    if last_s is not None:
        intervals.append([seg_start, last_s])  # type: ignore[arg-type]
    return {"intervals": intervals}


@api.get("/projects/{project_id}/videos/{file_id}/cv/skeleton")
async def stream_skeleton_frames(
    project_id: str,
    file_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> FileResponse:
    """Serve skeleton JSONL file for 3-D visualisation."""
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    row = cv_service.get_video_status(project_id, file_id)
    if row is None or not row.get("skeleton_jsonl_path"):
        raise HTTPException(status_code=404, detail="No skeleton data found for this video")
    path = Path(str(row["skeleton_jsonl_path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Skeleton file is missing from disk")
    return FileResponse(path, media_type="application/x-ndjson")


@api.get("/projects/{project_id}/videos/{file_id}/cv/pose-at", response_model=VideoCvPoseResponse)
async def get_video_cv_pose_at_time(
    project_id: str,
    file_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
    video_s: float = Query(default=0.0, ge=0.0),
) -> VideoCvPoseResponse:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        out = cv_service.get_processed_pose_at_time(project_id, file_id, float(video_s))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VideoCvPoseResponse(**out)


@api.post("/projects/{project_id}/videos/{file_id}/skeleton/preview", response_model=SkeletonPreviewResponse)
async def preview_video_skeleton(
    project_id: str,
    file_id: str,
    payload: SkeletonPreviewRequest,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> SkeletonPreviewResponse:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        out = cv_service.preview_frame(project_id, file_id, int(payload.frame_index))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SkeletonPreviewResponse(**out)


@api.post("/projects/{project_id}/videos/{file_id}/camera/pnp/autodetect", response_model=PnPAutodetectResponse)
async def autodetect_video_pnp(
    project_id: str,
    file_id: str,
    payload: PnPAutodetectRequest,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> PnPAutodetectResponse:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        out = cv_service.autodetect_pnp_pairs(project_id, file_id, int(payload.frame_index))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PnPAutodetectResponse(
        frame_index=int(out.get("frame_index") or 0),
        pairs=[PnPPair(**pair) for pair in out.get("pairs") or []],
    )


@api.post("/projects/{project_id}/videos/{file_id}/camera/pnp/solve", response_model=PnPSolveResponse)
async def solve_video_pnp(
    project_id: str,
    file_id: str,
    payload: PnPSolveRequest,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> PnPSolveResponse:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        out = cv_service.solve_pnp(
            project_id,
            file_id,
            [pair.model_dump() for pair in payload.pairs],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PnPSolveResponse(**out)


@api.post("/projects/{project_id}/camera/pnp/apply", response_model=ProjectCvConfig)
async def apply_project_pnp_pose(
    project_id: str,
    payload: PnPApplyRequest,
    store: Annotated[MetadataStore, Depends(get_store)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> ProjectCvConfig:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    cfg = cv_service.update_project_config(
        project_id,
        {
            "camera_position": payload.camera_position,
            "camera_R_wc": payload.camera_R_wc,
            "camera_pitch_deg": payload.camera_pose_deg.get("pitch_deg"),
            "camera_yaw_deg": payload.camera_pose_deg.get("yaw_deg"),
            "camera_roll_deg": payload.camera_pose_deg.get("roll_deg"),
        },
    )
    return ProjectCvConfig(**cfg)


@api.get("/assets/hull.stl")
async def get_hull_asset() -> FileResponse:
    path = resolve_hull_stl_path()
    if path is None:
        raise HTTPException(status_code=404, detail="Hull STL asset was not found")
    return FileResponse(path, media_type="model/stl")


@api.get("/projects/{project_id}/map-data", response_model=MapDataResponse)
async def get_map_data(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    pipeline: Annotated[PipelineService, Depends(get_pipeline)],
) -> MapDataResponse:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    payload = pipeline.get_map_payload(project_id)
    return MapDataResponse(**payload)


@api.get("/projects/{project_id}/files/{file_id}")
async def get_raw_file(
    project_id: str,
    file_id: str,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> Response:
    project_path = PROJECTS_ROOT / project_id
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    row = store.get_file(file_id)
    if row is None or row["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="File not found")
    path = Path(str(row["path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored file path is missing")
    return await _stream_file_response(path, request, media_type=_guess_media_type(path))


@api.get("/projects/{project_id}/videos/{file_id}/playback")
async def get_video_playback_file(
    project_id: str,
    file_id: str,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> Response:
    project_path = PROJECTS_ROOT / project_id
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    row = store.get_file(file_id)
    if row is None or row["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="File not found")
    if row["kind"] != "video":
        raise HTTPException(status_code=400, detail="Requested file is not a video")

    raw_path = Path(str(row["path"]))
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Stored file path is missing")

    proxy_path = get_video_proxy_path(project_id, file_id)
    use_proxy = False
    try:
        use_proxy = proxy_path.exists() and proxy_path.stat().st_size > 1024
    except OSError:
        use_proxy = False
    playback_path = proxy_path if use_proxy else raw_path
    return await _stream_file_response(
        playback_path,
        request,
        media_type=_guess_media_type(playback_path),
    )


@api.get("/projects/{project_id}/videos/{file_id}/track")
async def get_video_track(
    project_id: str,
    file_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> dict[str, object]:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    file_row = store.get_file(file_id)
    if file_row is None or file_row["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Video file not found")
    if file_row["kind"] != "video":
        raise HTTPException(status_code=400, detail="Requested file is not a video")

    track_row = store.get_track_by_file_id(file_id)
    if track_row is None:
        raise HTTPException(status_code=404, detail="No processed track available for this video")

    points_path = Path(str(track_row["points_path"]))
    if not points_path.exists():
        raise HTTPException(status_code=404, detail="Stored track file is missing")

    return {
        "project_id": project_id,
        "file_id": file_id,
        "filename": file_row["filename"],
        "point_count": int(track_row["point_count"]),
        "ts_start": track_row["ts_start"],
        "ts_end": track_row["ts_end"],
        "points": load_track_points(points_path),
    }


@api.delete("/projects/{project_id}/segments/{file_id}")
async def delete_track_segment(
    project_id: str,
    file_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
    pipeline: Annotated[PipelineService, Depends(get_pipeline)],
) -> dict[str, object]:
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if store.has_active_jobs(project_id):
        raise HTTPException(
            status_code=409,
            detail="Project has active jobs. Wait for completion before deleting segments.",
        )

    file_row = store.get_file(file_id)
    if file_row is None or file_row["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Segment not found")
    if file_row["kind"] not in {"video", "csv"}:
        raise HTTPException(status_code=400, detail="Only video/csv tracks can be deleted")

    track_row = store.get_track_by_file_id(file_id)
    raw_path = Path(str(file_row["path"]))
    track_path = Path(str(track_row["points_path"])) if track_row else None
    proxy_path = get_video_proxy_path(project_id, file_id) if file_row["kind"] == "video" else None
    skeleton_paths = get_video_skeleton_paths(project_id, file_id) if file_row["kind"] == "video" else {}

    deleted_counts = store.delete_segment(project_id, file_id)

    for path in (
        raw_path,
        track_path,
        proxy_path,
        skeleton_paths.get("jsonl"),
        skeleton_paths.get("metrics_csv"),
        skeleton_paths.get("summary_json"),
        skeleton_paths.get("pose_csv"),
        skeleton_paths.get("autopnp_json"),

    ):
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Keep DB state authoritative even if stale files fail to remove.
            pass

    match_count = pipeline.recompute_matches(project_id)

    # Clean up file_metadata.json entry for this file
    try:
        meta = _load_file_meta(project_id)
        if file_id in meta:
            del meta[file_id]
            _save_file_meta(project_id, meta)
    except Exception:
        pass  # non-fatal

    return {
        "deleted": bool(deleted_counts.get("files")),
        "project_id": project_id,
        "file_id": file_id,
        "counts": deleted_counts,
        "match_count": match_count,
    }


# ── Segments (project-level, absolute timestamps) ─────────────────────────

def _segments_path(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id / "segments.json"


def _load_segments(project_id: str) -> list[dict]:
    p = _segments_path(project_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_segments(project_id: str, segments: list[dict]) -> None:
    p = _segments_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(segments, indent=2), encoding="utf-8")


@api.get("/projects/{project_id}/segments")
async def get_segments(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> list[dict]:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _load_segments(project_id)


@api.put("/projects/{project_id}/segments")
async def put_segments(
    project_id: str,
    payload: list[dict],
    store: Annotated[MetadataStore, Depends(get_store)],
) -> list[dict]:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    _save_segments(project_id, payload)
    return payload


# ── Athletes ──────────────────────────────────────────────────────────────

def _athletes_path(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id / "athletes.json"


def _file_meta_path(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id / "file_metadata.json"


def _load_athletes(project_id: str) -> list[dict]:
    p = _athletes_path(project_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_athletes(project_id: str, athletes: list[dict]) -> None:
    p = _athletes_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(athletes, indent=2), encoding="utf-8")


def _load_file_meta(project_id: str) -> dict:
    p = _file_meta_path(project_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_file_meta(project_id: str, meta: dict) -> None:
    p = _file_meta_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


@api.get("/projects/{project_id}/athletes")
async def get_athletes(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> list[dict]:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _load_athletes(project_id)


@api.put("/projects/{project_id}/athletes")
async def save_athletes(
    project_id: str,
    payload: list[dict],
    store: Annotated[MetadataStore, Depends(get_store)],
) -> list[dict]:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    # Validate and normalise
    sanitised: list[dict] = []
    for a in payload:
        sanitised.append({
            "id": str(a.get("id") or uuid.uuid4()),
            "name": str(a.get("name") or "Athlete")[:100],
            "weight": float(a.get("weight") or 75.0),
        })
    _save_athletes(project_id, sanitised)
    return sanitised


@api.get("/projects/{project_id}/file-meta")
async def get_file_meta(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> dict:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _load_file_meta(project_id)


@api.put("/projects/{project_id}/files/{file_id}/meta")
async def update_file_meta(
    project_id: str,
    file_id: str,
    payload: dict,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> dict:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    row = store.get_file(file_id)
    if row is None or str(row.get("project_id")) != project_id:
        raise HTTPException(status_code=404, detail="File not found")
    meta = _load_file_meta(project_id)
    existing = meta.get(file_id, {})
    for k, v in payload.items():
        if k in {"athlete_id", "label", "notes", "splits", "manual_athlete"}:
            existing[k] = v
    meta[file_id] = existing

    # If this is a CSV and athlete_id changed, propagate to matched videos
    # (but skip videos where the user has manually assigned an athlete)
    if "athlete_id" in payload and str(row.get("kind")) == "csv":
        matches = store.list_matches(project_id)
        for m in matches:
            if str(m.get("csv_file_id")) == file_id and m.get("rank") == 1:
                vid_id = str(m["video_file_id"])
                vid_meta = meta.get(vid_id, {})
                if vid_meta.get("manual_athlete"):
                    continue  # respect manual override
                vid_meta["athlete_id"] = payload["athlete_id"]
                meta[vid_id] = vid_meta

    _save_file_meta(project_id, meta)
    return meta


@api.get("/projects/{project_id}/files")
async def list_project_files(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> list[dict]:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    files = store.list_files(project_id)
    meta = _load_file_meta(project_id)
    for f in files:
        f["meta"] = meta.get(str(f["id"]), {})
    return files


async def _save_upload_stream(upload: UploadFile, destination: Path) -> int:
    await anyio.Path(destination.parent).mkdir(parents=True, exist_ok=True)
    size_bytes = 0
    async with await anyio.open_file(destination, "wb") as handle:
        while True:
            chunk = await upload.read(UPLOAD_BUFFER_SIZE)
            if not chunk:
                break
            await handle.write(chunk)
            size_bytes += len(chunk)
    await upload.close()
    return size_bytes


# ── Report ────────────────────────────────────────────────────────────────


@api.post("/projects/{project_id}/report")
async def generate_report(
    project_id: str,
    payload: dict,
    store: Annotated[MetadataStore, Depends(get_store)],
    pipeline: Annotated[PipelineService, Depends(get_pipeline)],
    cv_service: Annotated[CvService, Depends(get_cv_service)],
) -> dict:
    """Generate a segment analysis report.

    Body: ``{"segment_ids": ["uuid", ...]}``

    Segments are now stored at project level with absolute epoch timestamps
    (tsStart, tsEnd). We convert them to per-video start_s/end_s for the
    report builder using track-point timestamp mapping.
    """
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    segment_ids = payload.get("segment_ids") or payload.get("split_ids") or []
    if not segment_ids:
        raise HTTPException(status_code=400, detail="No segments selected")

    # Load project-level segments
    proj_segments = _load_segments(project_id)
    selected_segs = [s for s in proj_segments if s.get("id") in segment_ids]
    if not selected_segs:
        raise HTTPException(status_code=400, detail="No matching segments found")

    file_meta = _load_file_meta(project_id)
    athletes = _load_athletes(project_id)

    # Load ALL video tracks to find overlapping files for each segment
    all_files = store.list_files(project_id)
    video_files = [f for f in all_files if str(f.get("kind")) == "video"]

    all_track_points: dict[str, list[dict]] = {}
    for vf in video_files:
        fid = str(vf["id"])
        track_row = store.get_track_by_file_id(fid)
        if track_row and track_row.get("points_path"):
            pp = Path(str(track_row["points_path"]))
            if pp.exists():
                all_track_points[fid] = load_track_points(pp)

    # Merge instrument data (heel, trim, etc.) from matched CSV tracks
    merge_csv_instruments_into_video_tracks(all_track_points, store, project_id)

    # Convert project-level segments to per-file splits format for report builder
    # For each segment, find which video files overlap and compute video_s ranges
    synthetic_file_meta = {}
    for fid, meta in file_meta.items():
        synthetic_file_meta[fid] = {**meta, "splits": []}

    for seg in selected_segs:
        ts_start = seg.get("tsStart", 0)
        ts_end = seg.get("tsEnd", 0)
        for fid, pts in all_track_points.items():
            if not pts:
                continue
            tss = [p.get("ts") for p in pts if p.get("ts") is not None]
            if not tss:
                continue
            file_ts_min, file_ts_max = min(tss), max(tss)
            # Check overlap
            if ts_start > file_ts_max or ts_end < file_ts_min:
                continue
            # Convert absolute ts to video_s using interpolation
            def _ts_to_video_s(target_ts):
                best_idx = 0
                for i, p in enumerate(pts):
                    if p.get("ts") is not None and p["ts"] <= target_ts:
                        best_idx = i
                lo = pts[best_idx]
                hi_idx = min(best_idx + 1, len(pts) - 1)
                hi = pts[hi_idx]
                if lo.get("video_s") is None:
                    return None
                if hi_idx == best_idx or hi.get("ts") is None or hi.get("video_s") is None:
                    return lo["video_s"]
                if hi["ts"] == lo["ts"]:
                    return lo["video_s"]
                frac = (target_ts - lo["ts"]) / (hi["ts"] - lo["ts"])
                return lo["video_s"] + frac * (hi["video_s"] - lo["video_s"])

            vs_start = _ts_to_video_s(max(ts_start, file_ts_min))
            vs_end = _ts_to_video_s(min(ts_end, file_ts_max))
            if vs_start is None or vs_end is None:
                continue
            if fid not in synthetic_file_meta:
                synthetic_file_meta[fid] = {"splits": []}
            synthetic_file_meta[fid].setdefault("splits", []).append({
                "id": seg["id"],
                "name": seg.get("name", "Segment"),
                "start_s": vs_start,
                "end_s": vs_end,
            })

    involved_file_ids = {fid for fid, m in synthetic_file_meta.items() if m.get("splits")}

    # Skeleton JSONL paths per file
    skel_paths: dict[str, Path] = {}
    for fid in involved_file_ids:
        row = cv_service.get_video_status(project_id, fid)
        if row and row.get("skeleton_jsonl_path"):
            p = Path(str(row["skeleton_jsonl_path"]))
            if p.exists():
                skel_paths[fid] = p

    cv_cfg = cv_service.get_project_config(project_id)

    import anyio

    report = await anyio.to_thread.run_sync(
        lambda: build_report_data(
            project_id=project_id,
            split_ids=[s["id"] for s in selected_segs],
            file_meta=synthetic_file_meta,
            athletes=athletes,
            track_points_by_file={fid: all_track_points[fid] for fid in involved_file_ids if fid in all_track_points},
            skeleton_jsonl_paths=skel_paths,
            cv_config=cv_cfg,
        )
    )
    return report


# ── PDF Report ────────────────────────────────────────────────────────────


@api.post("/projects/{project_id}/jobs/report", response_model=JobResponse)
async def enqueue_report_job(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> JobResponse:
    """Queue a PDF report generation job for ALL segments."""
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Use all segments
    all_segs = _load_segments(project_id)
    segment_ids = [s["id"] for s in all_segs]
    if not segment_ids:
        raise HTTPException(status_code=400, detail="No segments available")

    job_row = store.create_job(
        project_id=project_id,
        job_type="generate_report",
        payload={
            "project_id": project_id,
            "segment_ids": segment_ids,
        },
        priority=90,
    )
    return JobResponse(
        id=str(job_row["id"]),
        project_id=str(job_row["project_id"]),
        type=str(job_row["type"]),
        status=str(job_row["status"]),
        progress=float(job_row["progress"]),
        message=job_row.get("message"),
        error=job_row.get("error"),
        created_at=str(job_row["created_at"]),
        started_at=job_row.get("started_at"),
        finished_at=job_row.get("finished_at"),
    )


@api.get("/projects/{project_id}/report/pdf/{filename}")
async def download_pdf(
    project_id: str,
    filename: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> FileResponse:
    """Download a previously generated PDF report."""
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Sanitise filename to prevent path traversal
    safe_name = Path(filename).name
    pdf_path = PROJECTS_ROOT / project_id / "derived" / "exports" / safe_name
    if not pdf_path.exists() or not pdf_path.suffix == ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=safe_name,
    )


@api.get("/projects/{project_id}/report/pdfs")
async def list_pdfs(
    project_id: str,
    store: Annotated[MetadataStore, Depends(get_store)],
) -> list[dict]:
    """List all generated PDF reports for a project."""
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    exports_dir = PROJECTS_ROOT / project_id / "derived" / "exports"
    if not exports_dir.exists():
        return []

    pdfs = sorted(exports_dir.glob("report_*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for p in pdfs:
        stat = p.stat()
        result.append({
            "filename": p.name,
            "download_url": f"/api/projects/{project_id}/report/pdf/{p.name}",
            "size_bytes": stat.st_size,
            "created": stat.st_mtime,
        })
    return result
