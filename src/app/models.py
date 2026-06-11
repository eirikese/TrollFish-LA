from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ProjectResponse(BaseModel):
    id: str
    name: str
    created_at: str


class UploadFileResult(BaseModel):
    id: str
    filename: str
    kind: str
    size_bytes: int
    status: str


class UploadResponse(BaseModel):
    project_id: str
    files: list[UploadFileResult]
    job_id: str | None = None


class JobResponse(BaseModel):
    id: str
    project_id: str
    type: str
    status: str
    progress: float
    message: str | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class TrackPoint(BaseModel):
    ts: float
    lat: float
    lon: float
    video_s: float | None = None


class MatchResult(BaseModel):
    video_file_id: str
    csv_file_id: str
    rank: int
    score: float
    median_distance_m: float
    p90_distance_m: float
    coverage: float
    offset_seconds: float
    sample_count: int


class MapTrack(BaseModel):
    file_id: str
    filename: str
    point_count: int
    ts_start: float | None = None
    ts_end: float | None = None
    points: list[TrackPoint]
    best_match_csv_id: str | None = None


class MapDataResponse(BaseModel):
    project_id: str
    videos: list[MapTrack]
    csvs: list[MapTrack]
    matches: list[MatchResult]
    files: list[dict[str, Any]]


class AutoPnPConfig(BaseModel):
    enabled: bool = True
    interval_frames: int = Field(default=10000, ge=1, le=200000)
    avg_frames: int = Field(default=5, ge=1, le=1000)
    min_valid_frames: int = Field(default=5, ge=1, le=1000)


class ProjectCvConfig(BaseModel):
    pose_model: str = Field(default="full")
    calibration_path: str | None = None
    camera_position: list[float] = Field(default_factory=lambda: [-3.194, 0.02, 0.585])
    camera_pitch_deg: float = 14.7
    camera_yaw_deg: float = 0.0
    camera_roll_deg: float = 0.0
    camera_R_wc: list[list[float]] | None = None
    lower_plane_z: float = 0.0
    hip_plane_z: float = 0.05
    lower_landmark: str = "ankle"
    athlete_weight: float = 75.0
    boat_com: float = -1.114
    mediapipe_workers: int | None = None
    skeleton_filter: dict[str, Any] = Field(default_factory=dict)
    contact_params: dict[str, Any] = Field(default_factory=dict)
    seated_x_stabilizer: dict[str, Any] = Field(default_factory=dict)
    lateral_y_stabilizer: dict[str, Any] = Field(default_factory=dict)
    auto_camera_pnp: AutoPnPConfig = Field(default_factory=AutoPnPConfig)


class ProjectCvConfigUpdate(BaseModel):
    pose_model: str | None = None
    calibration_path: str | None = None
    camera_position: list[float] | None = None
    camera_pitch_deg: float | None = None
    camera_yaw_deg: float | None = None
    camera_roll_deg: float | None = None
    camera_R_wc: list[list[float]] | None = None
    lower_plane_z: float | None = None
    hip_plane_z: float | None = None
    lower_landmark: str | None = None
    athlete_weight: float | None = None
    boat_com: float | None = None
    mediapipe_workers: int | None = None
    skeleton_filter: dict[str, Any] | None = None
    contact_params: dict[str, Any] | None = None
    seated_x_stabilizer: dict[str, Any] | None = None
    lateral_y_stabilizer: dict[str, Any] | None = None
    auto_camera_pnp: AutoPnPConfig | None = None


class VideoRangeRequest(BaseModel):
    """Optional time range (seconds) to process for a single video file."""
    start_sec: float | None = None  # None = from start
    end_sec: float | None = None    # None = to end


class SkeletonBatchRequest(BaseModel):
    video_file_ids: list[str] | None = None
    force: bool = False
    mediapipe_workers: int | None = None  # per-video internal worker threads
    video_ranges: dict[str, VideoRangeRequest] | None = None  # file_id -> time range
    description: str | None = None  # e.g. segment name, shown in queue


class VideoCvStatus(BaseModel):
    project_id: str
    file_id: str
    status: str
    progress: float
    message: str | None = None
    error: str | None = None
    summary_json_path: str | None = None
    metrics_csv_path: str | None = None
    skeleton_jsonl_path: str | None = None
    pose_csv_path: str | None = None
    autopnp_json_path: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class VideoCvPoseResponse(BaseModel):
    project_id: str
    file_id: str
    video_s: float
    frame_index: int
    timestamp_ms: int
    has_skeleton: bool
    skeleton_3d: dict[int, list[float]] | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)


class SkeletonPreviewRequest(BaseModel):
    frame_index: int = Field(default=0, ge=0)


class SkeletonPreviewResponse(BaseModel):
    file_id: str
    frame_index: int
    timestamp_ms: int
    image_b64_jpeg: str
    has_pose: bool
    has_skeleton: bool
    skeleton_3d: dict[int, list[float]] | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    camera_position: list[float]
    camera_R_wc: list[list[float]]


class PnPPair(BaseModel):
    id: int
    image_point: list[float]
    object_point: list[float]


class PnPSolveRequest(BaseModel):
    pairs: list[PnPPair]


class PnPSolveResponse(BaseModel):
    camera_position: list[float]
    camera_R_wc: list[list[float]]
    camera_pose_deg: dict[str, float]
    mean_reprojection_error_px: float
    inlier_reprojection_error_px: float
    num_pairs: int
    num_inliers: int
    solve_method: str


class PnPAutodetectRequest(BaseModel):
    frame_index: int = Field(default=0, ge=0)


class PnPAutodetectResponse(BaseModel):
    frame_index: int
    pairs: list[PnPPair]


class PnPApplyRequest(BaseModel):
    camera_position: list[float]
    camera_R_wc: list[list[float]]
    camera_pose_deg: dict[str, float] = Field(default_factory=dict)
