from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA temp_store=MEMORY;

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id);
                CREATE INDEX IF NOT EXISTS idx_files_project_kind ON files(project_id, kind);

                CREATE TABLE IF NOT EXISTS tracks (
                    id TEXT PRIMARY KEY,
                    file_id TEXT UNIQUE NOT NULL,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    point_count INTEGER NOT NULL,
                    ts_start REAL,
                    ts_end REAL,
                    points_path TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(file_id) REFERENCES files(id),
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE INDEX IF NOT EXISTS idx_tracks_project_kind ON tracks(project_id, kind);

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_priority ON jobs(status, priority DESC, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS matches (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    video_file_id TEXT NOT NULL,
                    csv_file_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    score REAL NOT NULL,
                    median_distance_m REAL NOT NULL,
                    p90_distance_m REAL NOT NULL,
                    coverage REAL NOT NULL,
                    offset_seconds REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE INDEX IF NOT EXISTS idx_matches_project_video ON matches(project_id, video_file_id);
                CREATE INDEX IF NOT EXISTS idx_matches_project_csv ON matches(project_id, csv_file_id);

                CREATE TABLE IF NOT EXISTS project_cv_config (
                    project_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );

                CREATE TABLE IF NOT EXISTS video_cv_runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT,
                    error TEXT,
                    config_snapshot_json TEXT NOT NULL,
                    summary_json_path TEXT,
                    metrics_csv_path TEXT,
                    skeleton_jsonl_path TEXT,
                    pose_csv_path TEXT,
                    autopnp_json_path TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    UNIQUE(project_id, file_id),
                    FOREIGN KEY(project_id) REFERENCES projects(id),
                    FOREIGN KEY(file_id) REFERENCES files(id)
                );
                CREATE INDEX IF NOT EXISTS idx_video_cv_runs_project_status ON video_cv_runs(project_id, status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_video_cv_runs_status_created ON video_cv_runs(status, created_at DESC);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def create_project(self, name: str) -> dict[str, Any]:
        project_id = str(uuid.uuid4())
        created_at = now_utc_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
                (project_id, name, created_at),
            )
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        result = self._row_dict(row)
        if result is None:
            raise RuntimeError("Failed to create project.")
        return result

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._row_dict(row)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [self._row_dict(r) for r in rows if r is not None]

    def has_active_jobs(self, project_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1) AS count_active
                FROM jobs
                WHERE project_id = ? AND status IN ('pending', 'running')
                """,
                (project_id,),
            ).fetchone()
        if row is None:
            return False
        return int(row["count_active"]) > 0

    def delete_project(self, project_id: str) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            deleted_cv_runs = conn.execute(
                "DELETE FROM video_cv_runs WHERE project_id = ?",
                (project_id,),
            ).rowcount
            deleted_cv_configs = conn.execute(
                "DELETE FROM project_cv_config WHERE project_id = ?",
                (project_id,),
            ).rowcount
            deleted_matches = conn.execute(
                "DELETE FROM matches WHERE project_id = ?",
                (project_id,),
            ).rowcount
            deleted_tracks = conn.execute(
                "DELETE FROM tracks WHERE project_id = ?",
                (project_id,),
            ).rowcount
            deleted_jobs = conn.execute(
                "DELETE FROM jobs WHERE project_id = ?",
                (project_id,),
            ).rowcount
            deleted_files = conn.execute(
                "DELETE FROM files WHERE project_id = ?",
                (project_id,),
            ).rowcount
            deleted_projects = conn.execute(
                "DELETE FROM projects WHERE id = ?",
                (project_id,),
            ).rowcount
        return {
            "projects": int(deleted_projects),
            "files": int(deleted_files),
            "tracks": int(deleted_tracks),
            "jobs": int(deleted_jobs),
            "matches": int(deleted_matches),
            "cv_runs": int(deleted_cv_runs),
            "cv_configs": int(deleted_cv_configs),
        }

    def insert_file(
        self,
        project_id: str,
        filename: str,
        kind: str,
        path: str,
        size_bytes: int,
        status: str = "uploaded",
        error: str | None = None,
    ) -> dict[str, Any]:
        file_id = str(uuid.uuid4())
        created_at = now_utc_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (id, project_id, filename, kind, path, size_bytes, status, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, project_id, filename, kind, path, size_bytes, status, error, created_at),
            )
            row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        result = self._row_dict(row)
        if result is None:
            raise RuntimeError("Failed to insert file metadata.")
        return result

    def list_files(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE project_id = ? ORDER BY created_at ASC",
                (project_id,),
            ).fetchall()
        return [self._row_dict(r) for r in rows if r is not None]

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return self._row_dict(row)

    def update_file_status(self, file_id: str, status: str, error: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE files SET status = ?, error = ? WHERE id = ?",
                (status, error, file_id),
            )

    def upsert_track(
        self,
        file_id: str,
        project_id: str,
        kind: str,
        point_count: int,
        ts_start: float | None,
        ts_end: float | None,
        points_path: str,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        now = now_utc_iso()
        track_id = str(uuid.uuid4())
        meta_json = json.dumps(meta, separators=(",", ":"))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tracks (
                    id, file_id, project_id, kind, point_count, ts_start, ts_end, points_path, meta_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    kind = excluded.kind,
                    point_count = excluded.point_count,
                    ts_start = excluded.ts_start,
                    ts_end = excluded.ts_end,
                    points_path = excluded.points_path,
                    meta_json = excluded.meta_json,
                    updated_at = excluded.updated_at
                """,
                (
                    track_id,
                    file_id,
                    project_id,
                    kind,
                    point_count,
                    ts_start,
                    ts_end,
                    points_path,
                    meta_json,
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM tracks WHERE file_id = ?", (file_id,)).fetchone()
        result = self._row_dict(row)
        if result is None:
            raise RuntimeError("Failed to upsert track.")
        return result

    def list_tracks(self, project_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM tracks WHERE project_id = ? AND kind = ? ORDER BY created_at ASC",
                    (project_id, kind),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tracks WHERE project_id = ? ORDER BY created_at ASC",
                    (project_id,),
                ).fetchall()
        return [self._row_dict(r) for r in rows if r is not None]

    def get_track_by_file_id(self, file_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tracks WHERE file_id = ?",
                (file_id,),
            ).fetchone()
        return self._row_dict(row)

    def delete_segment(self, project_id: str, file_id: str) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            deleted_cv_runs = conn.execute(
                "DELETE FROM video_cv_runs WHERE project_id = ? AND file_id = ?",
                (project_id, file_id),
            ).rowcount
            deleted_matches = conn.execute(
                """
                DELETE FROM matches
                WHERE project_id = ? AND (video_file_id = ? OR csv_file_id = ?)
                """,
                (project_id, file_id, file_id),
            ).rowcount
            deleted_tracks = conn.execute(
                "DELETE FROM tracks WHERE project_id = ? AND file_id = ?",
                (project_id, file_id),
            ).rowcount
            deleted_files = conn.execute(
                "DELETE FROM files WHERE project_id = ? AND id = ?",
                (project_id, file_id),
            ).rowcount
        return {
            "files": int(deleted_files),
            "tracks": int(deleted_tracks),
            "matches": int(deleted_matches),
            "cv_runs": int(deleted_cv_runs),
        }

    def create_job(
        self,
        project_id: str,
        job_type: str,
        payload: dict[str, Any],
        priority: int = 100,
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        created_at = now_utc_iso()
        payload_json = json.dumps(payload, separators=(",", ":"))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, project_id, type, payload_json, status, priority, progress, message, error, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, 0.0, NULL, NULL, ?)
                """,
                (job_id, project_id, job_type, payload_json, priority, created_at),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        result = self._row_dict(row)
        if result is None:
            raise RuntimeError("Failed to create job.")
        return result

    def fetch_next_pending_job(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            started_at = now_utc_iso()
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?, progress = 0.0, message = 'Started'
                WHERE id = ?
                """,
                (started_at, row["id"]),
            )
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return self._row_dict(updated)

    def update_job_progress(self, job_id: str, progress: float, message: str | None = None) -> None:
        progress = max(0.0, min(1.0, float(progress)))
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ?, message = ? WHERE id = ?",
                (progress, message, job_id),
            )

    def complete_job(self, job_id: str, message: str | None = None) -> None:
        finished_at = now_utc_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed', progress = 1.0, message = ?, finished_at = ?, error = NULL
                WHERE id = ?
                """,
                (message, finished_at, job_id),
            )

    def fail_job(self, job_id: str, error: str) -> None:
        finished_at = now_utc_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', message = 'Failed', error = ?, finished_at = ?
                WHERE id = ?
                """,
                (error, finished_at, job_id),
            )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job. Returns True if the job was cancelled."""
        finished_at = now_utc_iso()
        with self._lock, self._connect() as conn:
            result = conn.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', message = 'Cancelled', finished_at = ?
                WHERE id = ? AND status IN ('pending', 'running')
                """,
                (finished_at, job_id),
            )
            return result.rowcount > 0

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_dict(row)

    def list_jobs(self, project_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
        return [self._row_dict(r) for r in rows if r is not None]

    def replace_matches(self, project_id: str, matches: list[dict[str, Any]]) -> None:
        created_at = now_utc_iso()
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM matches WHERE project_id = ?", (project_id,))
            if not matches:
                return
            conn.executemany(
                """
                INSERT INTO matches (
                    id, project_id, video_file_id, csv_file_id, rank, score, median_distance_m,
                    p90_distance_m, coverage, offset_seconds, sample_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(uuid.uuid4()),
                        project_id,
                        item["video_file_id"],
                        item["csv_file_id"],
                        item["rank"],
                        item["score"],
                        item["median_distance_m"],
                        item["p90_distance_m"],
                        item["coverage"],
                        item["offset_seconds"],
                        item["sample_count"],
                        created_at,
                    )
                    for item in matches
                ],
            )

    def list_matches(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM matches
                WHERE project_id = ?
                ORDER BY video_file_id ASC, rank ASC
                """,
                (project_id,),
            ).fetchall()
        return [self._row_dict(r) for r in rows if r is not None]

    def get_project_cv_config(self, project_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM project_cv_config WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        parsed = self._row_dict(row)
        if parsed is None:
            return None
        raw = parsed.get("config_json")
        try:
            parsed["config"] = json.loads(str(raw)) if raw else {}
        except json.JSONDecodeError:
            parsed["config"] = {}
        return parsed

    def upsert_project_cv_config(self, project_id: str, config: dict[str, Any]) -> dict[str, Any]:
        updated_at = now_utc_iso()
        payload = json.dumps(config, separators=(",", ":"))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO project_cv_config (project_id, config_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    config_json = excluded.config_json,
                    updated_at = excluded.updated_at
                """,
                (project_id, payload, updated_at),
            )
            row = conn.execute(
                "SELECT * FROM project_cv_config WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        parsed = self._row_dict(row)
        if parsed is None:
            raise RuntimeError("Failed to upsert project CV config.")
        try:
            parsed["config"] = json.loads(str(parsed.get("config_json") or "{}"))
        except json.JSONDecodeError:
            parsed["config"] = {}
        return parsed

    def upsert_video_cv_run(
        self,
        project_id: str,
        file_id: str,
        status: str,
        progress: float,
        message: str | None,
        error: str | None,
        config_snapshot: dict[str, Any],
        summary_json_path: str | None = None,
        metrics_csv_path: str | None = None,
        skeleton_jsonl_path: str | None = None,
        pose_csv_path: str | None = None,
        autopnp_json_path: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> dict[str, Any]:
        now = now_utc_iso()
        run_id = str(uuid.uuid4())
        snapshot_json = json.dumps(config_snapshot, separators=(",", ":"))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO video_cv_runs (
                    id, project_id, file_id, status, progress, message, error, config_snapshot_json,
                    summary_json_path, metrics_csv_path, skeleton_jsonl_path, pose_csv_path, autopnp_json_path,
                    created_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, file_id) DO UPDATE SET
                    status = excluded.status,
                    progress = excluded.progress,
                    message = excluded.message,
                    error = excluded.error,
                    config_snapshot_json = excluded.config_snapshot_json,
                    summary_json_path = COALESCE(excluded.summary_json_path, video_cv_runs.summary_json_path),
                    metrics_csv_path = COALESCE(excluded.metrics_csv_path, video_cv_runs.metrics_csv_path),
                    skeleton_jsonl_path = COALESCE(excluded.skeleton_jsonl_path, video_cv_runs.skeleton_jsonl_path),
                    pose_csv_path = COALESCE(excluded.pose_csv_path, video_cv_runs.pose_csv_path),
                    autopnp_json_path = COALESCE(excluded.autopnp_json_path, video_cv_runs.autopnp_json_path),
                    started_at = COALESCE(excluded.started_at, video_cv_runs.started_at),
                    finished_at = COALESCE(excluded.finished_at, video_cv_runs.finished_at)
                """,
                (
                    run_id,
                    project_id,
                    file_id,
                    status,
                    max(0.0, min(1.0, float(progress))),
                    message,
                    error,
                    snapshot_json,
                    summary_json_path,
                    metrics_csv_path,
                    skeleton_jsonl_path,
                    pose_csv_path,
                    autopnp_json_path,
                    now,
                    started_at,
                    finished_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM video_cv_runs WHERE project_id = ? AND file_id = ?",
                (project_id, file_id),
            ).fetchone()
        parsed = self._row_dict(row)
        if parsed is None:
            raise RuntimeError("Failed to upsert video CV run.")
        return parsed

    def get_video_cv_run(self, project_id: str, file_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM video_cv_runs WHERE project_id = ? AND file_id = ?",
                (project_id, file_id),
            ).fetchone()
        return self._row_dict(row)

    def list_video_cv_runs(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM video_cv_runs
                WHERE project_id = ?
                ORDER BY created_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [self._row_dict(r) for r in rows if r is not None]

    def parse_job_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        payload_raw = row.get("payload_json") if row else None
        if not payload_raw:
            return {}
        try:
            return json.loads(payload_raw)
        except json.JSONDecodeError:
            return {}
