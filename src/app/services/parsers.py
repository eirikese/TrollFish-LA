from __future__ import annotations

import bisect
import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.app.config import EXIFTOOL_FALLBACKS


@dataclass(slots=True)
class ParsedTrack:
    points: list[dict[str, float]]
    metadata: dict[str, Any]


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalize_iso_timezone(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return s
    if s.endswith("Z"):
        return s[:-1] + "+00:00"
    # Convert +0000 -> +00:00 and -0230 -> -02:30
    if len(s) >= 5 and (s[-5] == "+" or s[-5] == "-") and s[-3] != ":":
        if s[-4:].isdigit():
            return f"{s[:-5]}{s[-5:-2]}:{s[-2:]}"
    return s


def parse_timestamp(value: Any) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None

    s = _normalize_iso_timezone(raw)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except ValueError:
        pass

    time_formats = (
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y:%m:%d %H:%M:%S",
    )
    for fmt in time_formats:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _pick_column(columns_lower: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns_lower:
            return columns_lower[candidate]
    return None


# ---------------------------------------------------------------------------
# GPS outlier filter – ported from gopro_mp4_gps_dual_video_map.py
# ---------------------------------------------------------------------------
_MAX_PLAUSIBLE_SPEED_MS = 25.0  # ~50 knots; well above any dinghy/foiler


def _haversine_fast(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Fast equirectangular distance in metres (good enough for <1 km gaps)."""
    dlat = (lat2 - lat1) * 111_320.0
    dlon = (lon2 - lon1) * 111_320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.sqrt(dlat * dlat + dlon * dlon)


def _filter_gps_outliers(
    pts: list[dict[str, float]],
    max_speed: float = _MAX_PLAUSIBLE_SPEED_MS,
    max_passes: int = 3,
) -> list[dict[str, float]]:
    """
    Remove GPS points that imply travel faster than *max_speed* m/s.

    Multi-pass interior check: if both the A→B and B→C legs exceed
    *max_speed*, point B is a bad-fix and is dropped.  Edge points are
    dropped if the single leg to their only neighbour exceeds the limit.
    Repeats up to *max_passes* times because removing one outlier can
    expose the next.
    """
    if len(pts) < 3:
        return pts

    for _ in range(max_passes):
        n = len(pts)
        if n < 3:
            break
        keep = [True] * n

        # Interior points: both-neighbour speed check
        for i in range(1, n - 1):
            dt_ab = max(0.1, abs(pts[i]["ts"] - pts[i - 1]["ts"]))
            dt_bc = max(0.1, abs(pts[i + 1]["ts"] - pts[i]["ts"]))
            d_ab = _haversine_fast(pts[i - 1]["lat"], pts[i - 1]["lon"], pts[i]["lat"], pts[i]["lon"])
            d_bc = _haversine_fast(pts[i]["lat"], pts[i]["lon"], pts[i + 1]["lat"], pts[i + 1]["lon"])
            if (d_ab / dt_ab) > max_speed and (d_bc / dt_bc) > max_speed:
                keep[i] = False

        # Edge: first point
        dt = max(0.1, abs(pts[1]["ts"] - pts[0]["ts"]))
        d = _haversine_fast(pts[0]["lat"], pts[0]["lon"], pts[1]["lat"], pts[1]["lon"])
        if d / dt > max_speed:
            keep[0] = False

        # Edge: last point
        dt = max(0.1, abs(pts[-1]["ts"] - pts[-2]["ts"]))
        d = _haversine_fast(pts[-2]["lat"], pts[-2]["lon"], pts[-1]["lat"], pts[-1]["lon"])
        if d / dt > max_speed:
            keep[-1] = False

        new_pts = [p for p, k in zip(pts, keep) if k]
        if len(new_pts) == len(pts):
            break
        pts = new_pts

    return pts


def _ms_to_epoch(ms_value: float, anchors: list[tuple[float, float]]) -> float | None:
    if not anchors:
        return None
    if len(anchors) == 1:
        a_ms, a_epoch = anchors[0]
        return a_epoch + (ms_value - a_ms) / 1000.0

    anchor_ms = [a[0] for a in anchors]
    idx = bisect.bisect_right(anchor_ms, ms_value)
    if idx <= 0:
        m0, t0 = anchors[0]
        m1, t1 = anchors[1]
    elif idx >= len(anchors):
        m0, t0 = anchors[-2]
        m1, t1 = anchors[-1]
    else:
        m0, t0 = anchors[idx - 1]
        m1, t1 = anchors[idx]

    if abs(m1 - m0) < 1e-9:
        return t0 + (ms_value - m0) / 1000.0
    slope = (t1 - t0) / (m1 - m0)
    return t0 + slope * (ms_value - m0)


def _parse_ts_vakaros(s: str) -> float | None:
    """Parse Vakaros-native ISO timestamps (format A)."""
    raw = (s or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(raw, fmt).timestamp()
        except ValueError:
            pass
    return parse_timestamp(raw)


def _parse_ts_sensor(s: str) -> float | None:
    """
    Parse sensor-logger timestamps (format B).

    Tries pure-numeric epoch-ms first, then a range of ISO variants
    including the bare 'Z' suffix that :func:`parse_timestamp` misses.
    """
    raw = (s or "").strip()
    if not raw:
        return None
    # Pure numeric → treat as epoch milliseconds
    try:
        return float(raw) / 1000.0
    except ValueError:
        pass
    # ISO variants with explicit Z
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%SZ",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC).timestamp()
        except ValueError:
            pass
    return parse_timestamp(raw)


def parse_csv_track(csv_path: Path) -> ParsedTrack:
    """
    Parse a GPS CSV into a ``ParsedTrack``.

    Supported formats (case-insensitive column names):

    * **Format A – Vakaros-native**: ``timestamp``, ``latitude``, ``longitude``
      (optional ``sog_mps`` / ``sog`` for speed-over-ground).
    * **Format B – Sensor/IMU logger**: ``lat``, ``lon`` plus either
      ``iso_time`` (ISO-8601 / epoch-ms) or ``timestamp_ms`` (raw ms).
    * **Generic**: any combination of recognised lat/lon/time columns.

    GPS outliers that imply physically impossible speeds are removed with
    the same multi-pass algorithm used in the GoPro dual-video viewer.
    """
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return ParsedTrack(points=[], metadata={"source": str(csv_path), "reason": "empty_csv"})

    columns_lower = {name.strip().lower(): name for name in fieldnames if name}

    def _has(*names: str) -> bool:
        return all(n in columns_lower for n in names)

    # ------------------------------------------------------------------
    # Detect SOG column (optional, carried through to output points)
    # ------------------------------------------------------------------
    sog_col: str | None = None
    for sog_name in ("sog_mps", "sog", "sog_kts"):
        if sog_name in columns_lower:
            sog_col = columns_lower[sog_name]
            break

    # ------------------------------------------------------------------
    # Detect instrument columns (heel, trim — optional)
    # ------------------------------------------------------------------
    heel_col: str | None = None
    for heel_name in ("heel", "heel_deg", "roll_deg"):
        if heel_name in columns_lower:
            heel_col = columns_lower[heel_name]
            break
    trim_col: str | None = None
    for trim_name in ("trim", "trim_deg", "pitch", "pitch_deg"):
        if trim_name in columns_lower:
            trim_col = columns_lower[trim_name]
            break
    cog_col: str | None = None
    if "cog" in columns_lower:
        cog_col = columns_lower["cog"]
    hdg_col: str | None = None
    for hdg_name in ("hdg_true", "hdg", "heading", "heading_deg", "mag_hdg"):
        if hdg_name in columns_lower:
            hdg_col = columns_lower[hdg_name]
            break

    # ------------------------------------------------------------------
    # Detect format and choose column names + timestamp parser
    # ------------------------------------------------------------------
    detected_format: str

    if _has("timestamp", "latitude", "longitude"):
        # Format A: Vakaros-native
        lat_col: str | None = columns_lower["latitude"]
        lon_col: str | None = columns_lower["longitude"]
        ts_col: str | None = columns_lower["timestamp"]
        ts_ms_col: str | None = None
        parse_fn = _parse_ts_vakaros
        detected_format = "vakaros_native"

    elif _has("lat", "lon") and (_has("iso_time") or _has("timestamp_ms")):
        # Format B: Sensor / IMU logger
        lat_col = columns_lower["lat"]
        lon_col = columns_lower["lon"]
        ts_col = columns_lower.get("iso_time") or columns_lower.get("timestamp_ms")
        ts_ms_col = None
        parse_fn = _parse_ts_sensor
        detected_format = "sensor_logger"

    else:
        # Generic fallback: pick best available columns
        lat_col = _pick_column(columns_lower, ["latitude", "lat", "gps_lat", "gpslatitude"])
        lon_col = _pick_column(columns_lower, ["longitude", "lon", "lng", "gps_lon", "gpslongitude"])
        ts_col = _pick_column(
            columns_lower,
            ["timestamp", "datetime", "time", "date_time", "iso_time", "gps_time", "utc_time"],
        )
        ts_ms_col = _pick_column(columns_lower, ["timestamp_ms", "time_ms", "elapsed_ms", "ms"])
        parse_fn = parse_timestamp  # type: ignore[assignment]
        detected_format = "generic"

    if not lat_col or not lon_col:
        return ParsedTrack(
            points=[],
            metadata={
                "source": str(csv_path),
                "reason": "missing_lat_lon",
                "columns": fieldnames,
                "detected_format": detected_format,
            },
        )

    # ------------------------------------------------------------------
    # Generic-mode: build ms→epoch anchors for timestamp_ms interpolation
    # ------------------------------------------------------------------
    anchors: list[tuple[float, float]] = []
    if detected_format == "generic" and ts_ms_col:
        for row in rows:
            ms_value = _parse_float(row.get(ts_ms_col))
            if ms_value is None:
                continue
            epoch = parse_timestamp(row.get(ts_col)) if ts_col else None
            if epoch is None:
                epoch = parse_timestamp(row.get("iso_time"))
            if epoch is not None:
                anchors.append((ms_value, epoch))
        anchors.sort(key=lambda item: item[0])
        deduped_anchors: list[tuple[float, float]] = []
        last_ms: float | None = None
        for item in anchors:
            if last_ms is None or abs(item[0] - last_ms) > 1e-9:
                deduped_anchors.append(item)
                last_ms = item[0]
        anchors = deduped_anchors

    # ------------------------------------------------------------------
    # Parse rows → points
    # ------------------------------------------------------------------
    points: list[dict[str, float]] = []
    for row in rows:
        lat = _parse_float(row.get(lat_col))
        lon = _parse_float(row.get(lon_col))
        if lat is None or lon is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue

        ts: float | None
        if detected_format == "generic":
            ts = parse_timestamp(row.get(ts_col)) if ts_col else None
            if ts is None:
                ts = parse_timestamp(row.get("iso_time"))
            if ts is None and ts_ms_col:
                ms_value = _parse_float(row.get(ts_ms_col))
                if ms_value is not None:
                    ts = _ms_to_epoch(ms_value, anchors)
        else:
            ts = parse_fn(row.get(ts_col) or "")  # type: ignore[arg-type]

        if ts is None:
            continue

        pt: dict[str, float] = {"ts": ts, "lat": lat, "lon": lon}
        if sog_col is not None:
            sog = _parse_float(row.get(sog_col))
            if sog is not None:
                pt["sog"] = sog
        if heel_col is not None:
            heel = _parse_float(row.get(heel_col))
            if heel is not None:
                pt["heel"] = heel
        if trim_col is not None:
            trim = _parse_float(row.get(trim_col))
            if trim is not None:
                pt["trim"] = trim
        if cog_col is not None:
            cog = _parse_float(row.get(cog_col))
            if cog is not None:
                pt["cog"] = cog
        if hdg_col is not None:
            hdg = _parse_float(row.get(hdg_col))
            if hdg is not None:
                pt["hdg"] = hdg

        points.append(pt)

    points.sort(key=lambda p: p["ts"])
    deduped: list[dict[str, float]] = []
    last_ts: float | None = None
    for point in points:
        if last_ts is None or abs(point["ts"] - last_ts) > 1e-9:
            deduped.append(point)
            last_ts = point["ts"]
    points = deduped

    # Remove physically impossible GPS jumps (same algorithm as the GoPro dual-video viewer)
    n_before = len(points)
    points = _filter_gps_outliers(points)
    n_removed = n_before - len(points)

    metadata: dict[str, Any] = {
        "source": str(csv_path),
        "detected_format": detected_format,
        "columns": fieldnames,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "ts_col": ts_col,
        "ts_ms_col": ts_ms_col,
        "sog_col": sog_col,
        "heel_col": heel_col,
        "trim_col": trim_col,
        "cog_col": cog_col,
        "hdg_col": hdg_col,
        "anchor_count": len(anchors),
        "row_count": len(rows),
        "point_count": len(points),
        "outliers_removed": n_removed,
    }
    return ParsedTrack(points=points, metadata=metadata)


def _find_exiftool() -> str:
    found = shutil.which("exiftool")
    if found:
        return found
    for fallback in EXIFTOOL_FALLBACKS:
        path = Path(fallback)
        if path.exists():
            return str(path)
    raise RuntimeError(
        "ExifTool was not found. Install ExifTool and ensure `exiftool` is available in PATH."
    )


def _parse_gopro_gps_datetime(value: str) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        core = raw[:-1]
        for fmt in ("%Y:%m:%d %H:%M:%S.%f", "%Y:%m:%d %H:%M:%S"):
            try:
                dt = datetime.strptime(core, fmt).replace(tzinfo=UTC)
                return dt.timestamp()
            except ValueError:
                continue
    return parse_timestamp(raw)


def parse_gopro_video_track(video_path: Path) -> ParsedTrack:
    exiftool = _find_exiftool()
    command = [
        exiftool,
        "-ee",
        "-n",
        "-p",
        "$SampleTime,$GPSLatitude,$GPSLongitude,$GPSDateTime",
        str(video_path),
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    points: list[dict[str, float]] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",", 3)
        if len(parts) < 4:
            continue
        video_s = _parse_float(parts[0])
        lat = _parse_float(parts[1])
        lon = _parse_float(parts[2])
        ts = _parse_gopro_gps_datetime(parts[3])
        if lat is None or lon is None or ts is None or video_s is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        points.append({"ts": ts, "lat": lat, "lon": lon, "video_s": video_s})

    stderr_text = ""
    if process.stderr is not None:
        stderr_text = process.stderr.read().strip()
    return_code = process.wait()
    if return_code != 0:
        message = stderr_text if stderr_text else "ExifTool failed while parsing video GPS."
        raise RuntimeError(message)

    points.sort(key=lambda p: p["video_s"])
    deduped: list[dict[str, float]] = []
    last_video_s: float | None = None
    for point in points:
        if last_video_s is None or abs(point["video_s"] - last_video_s) > 1e-9:
            deduped.append(point)
            last_video_s = point["video_s"]
    points = deduped

    # Remove physically impossible GPS jumps (same algorithm as the GoPro dual-video viewer)
    n_before = len(points)
    points = _filter_gps_outliers(points)
    n_removed = n_before - len(points)

    metadata: dict[str, Any] = {
        "source": str(video_path),
        "point_count": len(points),
        "has_video_s": True,
        "outliers_removed": n_removed,
        "stderr": stderr_text[:1000] if stderr_text else None,
    }
    return ParsedTrack(points=points, metadata=metadata)
