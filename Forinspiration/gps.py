"""
GoPro GPS + Vakaros + Pose viewer generator.

Generates one standalone HTML viewer that syncs:
- GoPro MP4 GPS metadata (SampleTime + GPSDateTime),
- Vakaros telemetry CSV (timestamp, lat, lon, heel, trim),
- Pose CSV (timestamp_ms/frame_idx, trunk angle, moments).
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
import shutil
import subprocess
import tkinter as tk
from datetime import datetime, timezone
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _i(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _parse_epoch(s: str) -> float:
    t = (s or "").strip()
    if not t:
        raise ValueError("empty timestamp")
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(t, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            pass
    raise ValueError(f"Could not parse timestamp: {s}")


def _interp_numeric(points: List[dict], times: List[float], x: float, keys: tuple[str, ...]) -> Optional[dict]:
    if not points or not times:
        return None
    n = len(points)
    if n == 1:
        p = points[0]
        return {k: _f(p.get(k)) for k in keys}
    if x <= times[0]:
        p = points[0]
        return {k: _f(p.get(k)) for k in keys}
    if x >= times[-1]:
        p = points[-1]
        return {k: _f(p.get(k)) for k in keys}

    i0 = bisect.bisect_right(times, x) - 1
    i0 = max(0, min(i0, n - 2))
    i1 = i0 + 1
    t0 = times[i0]
    t1 = times[i1]
    a = 0.0 if abs(t1 - t0) < 1e-12 else (x - t0) / (t1 - t0)
    p0 = points[i0]
    p1 = points[i1]

    out: Dict[str, Optional[float]] = {}
    for k in keys:
        v0 = _f(p0.get(k))
        v1 = _f(p1.get(k))
        if v0 is not None and v1 is not None:
            out[k] = (1.0 - a) * v0 + a * v1
        elif v0 is not None:
            out[k] = v0
        elif v1 is not None:
            out[k] = v1
        else:
            out[k] = None
    return out


def _downsample(points: List[dict], t_key: str = "t", min_dt: float = 0.1) -> List[dict]:
    if not points:
        return []
    out: List[dict] = []
    last_t = None
    for p in points:
        t = _f(p.get(t_key))
        if t is None:
            continue
        if last_t is None or t - last_t >= min_dt:
            out.append(p)
            last_t = t
    if out and points[-1] is not out[-1]:
        out.append(points[-1])
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda * 0.5) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))
    return r * c


def extract_gopro_gps_from_mp4(mp4_path: str, exiftool_exe: Optional[str] = None) -> List[dict]:
    """Extract GoPro GPS samples with ExifTool."""

    def find_exiftool() -> str:
        if exiftool_exe and os.path.exists(exiftool_exe):
            return exiftool_exe
        e = shutil.which("exiftool")
        if e:
            return e
        fallback = r"C:\Program Files\exiftool-13.51_64\exiftool.exe"
        if os.path.exists(fallback):
            return fallback
        raise RuntimeError("ExifTool not found. Install ExifTool and ensure `exiftool` is on PATH.")

    def gpsdt_to_epoch(gpsdt: str) -> float:
        s = (gpsdt or "").strip()
        if not s:
            return float("nan")
        if s.endswith("Z"):
            core = s[:-1]
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(core, fmt).replace(tzinfo=timezone.utc)
                    return dt.timestamp()
                except Exception:
                    pass
        return float("nan")

    exif = find_exiftool()
    fmt = "$SampleTime,$GPSLatitude,$GPSLongitude,$GPSDateTime"
    cmd = [exif, "-ee", "-n", "-p", fmt, mp4_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "ExifTool failed.")

    out: List[dict] = []
    for line in p.stdout.splitlines():
        parts = line.strip().split(",", 3)
        if len(parts) < 3:
            continue
        t = _f(parts[0])
        lat = _f(parts[1])
        lon = _f(parts[2])
        if t is None or lat is None or lon is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        gpsdt = parts[3].strip() if len(parts) >= 4 else ""
        ts = gpsdt_to_epoch(gpsdt)
        if not math.isfinite(ts):
            continue
        out.append({"t": t, "lat": lat, "lon": lon, "gpsdt": gpsdt, "ts": ts})

    out.sort(key=lambda d: d["t"])
    dedup: List[dict] = []
    last_t = None
    for pnt in out:
        if last_t is None or abs(pnt["t"] - last_t) > 1e-9:
            dedup.append(pnt)
            last_t = pnt["t"]
    return dedup


def parse_vakaros_csv(csv_path: str) -> dict:
    """Parse Vakaros CSV with timestamp, latitude, longitude and optional heel/trim."""
    if not csv_path:
        return {"track": [], "columns": {}}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"track": [], "columns": {}}

    cols = {str(c).strip().lower(): c for c in rows[0].keys() if c is not None}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_ts = pick("timestamp", "time", "datetime")
    c_lat = pick("latitude", "lat")
    c_lon = pick("longitude", "lon", "lng")
    if not c_ts or not c_lat or not c_lon:
        raise RuntimeError("Vakaros CSV must include timestamp, latitude, longitude columns.")

    c_heel = pick("heel", "roll", "roll_deg")
    c_trim = pick("trim", "pitch", "pitch_deg")
    c_sog = pick("sog_kts", "sog")
    c_cog = pick("cog")

    track: List[dict] = []
    for r in rows:
        try:
            ts = _parse_epoch(str(r.get(c_ts, "")))
        except Exception:
            continue
        lat = _f(r.get(c_lat))
        lon = _f(r.get(c_lon))
        if lat is None or lon is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        row = {"ts": ts, "lat": lat, "lon": lon}
        if c_heel:
            row["heel"] = _f(r.get(c_heel))
        if c_trim:
            row["trim"] = _f(r.get(c_trim))
        if c_sog:
            row["sog_kts"] = _f(r.get(c_sog))
        if c_cog:
            row["cog"] = _f(r.get(c_cog))
        track.append(row)

    track.sort(key=lambda d: d["ts"])
    return {
        "track": track,
        "columns": {
            "heel": bool(c_heel),
            "trim": bool(c_trim),
            "sog_kts": bool(c_sog),
            "cog": bool(c_cog),
        },
    }


def parse_pose_csv(pose_csv_path: str) -> List[dict]:
    """Parse pose CSV and return rows normalized to video seconds."""
    if not pose_csv_path:
        return []

    with open(pose_csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    cols = {str(c).strip().lower(): c for c in rows[0].keys() if c is not None}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_frame = pick("frame_idx", "frame", "index")
    c_ts = pick("timestamp_ms", "timestamp", "time_ms", "time")
    c_trunk = pick("trunk_angle", "trunk_deg", "trunk")
    c_mp = pick("moment_pitch", "moment_x")
    c_mr = pick("moment_roll", "moment_y")

    raw: List[dict] = []
    for idx, r in enumerate(rows):
        raw.append(
            {
                "frame": _i(r.get(c_frame)) if c_frame else idx,
                "ts_raw": _f(r.get(c_ts)) if c_ts else None,
                "trunk_angle": _f(r.get(c_trunk)) if c_trunk else None,
                "moment_pitch": _f(r.get(c_mp)) if c_mp else None,
                "moment_roll": _f(r.get(c_mr)) if c_mr else None,
            }
        )

    ts_vals = [x["ts_raw"] for x in raw if x["ts_raw"] is not None]
    if not ts_vals:
        return []

    ts_max = max(ts_vals)
    mode = "relative_s"
    if ts_max > 1e11:
        mode = "epoch_ms"
    elif ts_max > 1e9:
        mode = "epoch_s"
    else:
        deltas = [ts_vals[i] - ts_vals[i - 1] for i in range(1, len(ts_vals)) if ts_vals[i] > ts_vals[i - 1]]
        if deltas:
            deltas.sort()
            if deltas[len(deltas) // 2] > 5.0:
                mode = "relative_ms"

    base = min(ts_vals) if mode.startswith("epoch_") else 0.0
    out: List[dict] = []
    for row in raw:
        ts = row["ts_raw"]
        if ts is None:
            continue
        if mode == "relative_ms":
            t = ts / 1000.0
        elif mode == "relative_s":
            t = ts
        elif mode == "epoch_ms":
            t = (ts - base) / 1000.0
        else:
            t = ts - base
        if not math.isfinite(t):
            continue
        out.append(
            {
                "frame": int(row["frame"] if row["frame"] is not None else 0),
                "t": float(t),
                "trunk_angle": row["trunk_angle"],
                "moment_pitch": row["moment_pitch"],
                "moment_roll": row["moment_roll"],
            }
        )

    out.sort(key=lambda d: d["t"])
    return out


def sync_pose_to_mp4(pose_rows: List[dict], mp4_track: List[dict]) -> List[dict]:
    if not pose_rows or not mp4_track:
        return []
    mp4_times = [float(p["t"]) for p in mp4_track]
    out: List[dict] = []
    for r in pose_rows:
        t = _f(r.get("t"))
        if t is None:
            continue
        inter = _interp_numeric(mp4_track, mp4_times, t, ("lat", "lon", "ts"))
        if not inter:
            continue
        lat = inter.get("lat")
        lon = inter.get("lon")
        ts = inter.get("ts")
        if lat is None or lon is None or ts is None:
            continue
        out.append(
            {
                "frame": int(r.get("frame", 0)),
                "t": float(t),
                "ts": float(ts),
                "lat": float(lat),
                "lon": float(lon),
                "trunk_angle": r.get("trunk_angle"),
                "moment_pitch": r.get("moment_pitch"),
                "moment_roll": r.get("moment_roll"),
            }
        )
    return out


def detect_pose_segments(pose_sync: List[dict], trunk_threshold_deg: float = 20.0, min_duration_s: float = 2.0) -> List[dict]:
    if not pose_sync:
        return []

    segments: List[dict] = []
    start = None
    sid = 1
    for i, r in enumerate(pose_sync):
        trunk = _f(r.get("trunk_angle"))
        hiking = trunk is not None and trunk >= trunk_threshold_deg
        if hiking and start is None:
            start = i
        if (not hiking or i == len(pose_sync) - 1) and start is not None:
            end = i if hiking and i == len(pose_sync) - 1 else i - 1
            if end >= start:
                t0 = _f(pose_sync[start].get("t"))
                t1 = _f(pose_sync[end].get("t"))
                if t0 is not None and t1 is not None and (t1 - t0) >= min_duration_s:
                    sub = pose_sync[start : end + 1]
                    trunk_vals = [float(x["trunk_angle"]) for x in sub if _f(x.get("trunk_angle")) is not None]
                    mr_vals = [float(x["moment_roll"]) for x in sub if _f(x.get("moment_roll")) is not None]
                    side = "Unknown"
                    if mr_vals:
                        side = "Port" if (sum(mr_vals) / len(mr_vals)) >= 0.0 else "Starboard"
                    path = _downsample(
                        [
                            {"lat": x["lat"], "lon": x["lon"], "t": x["t"], "frame": x["frame"]}
                            for x in sub
                            if _f(x.get("lat")) is not None and _f(x.get("lon")) is not None
                        ],
                        "t",
                        0.2,
                    )
                    if len(path) >= 2:
                        segments.append(
                            {
                                "id": sid,
                                "start_t": float(t0),
                                "end_t": float(t1),
                                "duration_s": float(t1 - t0),
                                "start_frame": int(pose_sync[start]["frame"]),
                                "end_frame": int(pose_sync[end]["frame"]),
                                "mean_trunk_deg": float(sum(trunk_vals) / len(trunk_vals)) if trunk_vals else None,
                                "peak_trunk_deg": float(max(trunk_vals)) if trunk_vals else None,
                                "side": side,
                                "path": path,
                            }
                        )
                        sid += 1
            start = None
    return segments


def sync_distance_stats(mp4_track: List[dict], vak_track: List[dict]) -> dict:
    if not mp4_track or not vak_track:
        return {}
    vak_times = [float(v["ts"]) for v in vak_track]
    sample = _downsample(mp4_track, "t", 0.2)
    dists = []
    for p in sample:
        ts = _f(p.get("ts"))
        if ts is None:
            continue
        iv = _interp_numeric(vak_track, vak_times, ts, ("lat", "lon"))
        if not iv:
            continue
        lat = _f(iv.get("lat"))
        lon = _f(iv.get("lon"))
        if lat is None or lon is None:
            continue
        dists.append(_haversine_m(float(p["lat"]), float(p["lon"]), lat, lon))
    if not dists:
        return {}
    dists.sort()
    n = len(dists)
    return {
        "count": n,
        "mean_m": float(sum(dists) / n),
        "p95_m": float(dists[int(round(0.95 * (n - 1)))]),
        "max_m": float(dists[-1]),
    }


HTML_TEMPLATE_PARTS: List[str] = []

HTML_TEMPLATE_PARTS.append(
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>GoPro GPS Pose Sync Viewer</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body { height:100%; margin:0; background:#0b1220; color:#e6eef8; font-family:Segoe UI, Arial, sans-serif; }
    * { box-sizing:border-box; }
    .wrap { height:100%; display:grid; grid-template-columns:420px 1fr; gap:10px; padding:10px; }
    .panel { background:#111b2e; border:1px solid #243954; border-radius:10px; min-height:0; }
    .left { display:grid; grid-template-rows:auto auto auto 1fr minmax(120px,34%); gap:8px; padding:10px; overflow:hidden; }
    .right { display:grid; grid-template-rows:62% 38%; gap:10px; min-height:0; }
    .box { border:1px solid #263f60; border-radius:8px; padding:8px; background:#0f1828; }
    .row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    .space { justify-content:space-between; }
    .small { font-size:12px; color:#9fb2ca; }
    .mono { font-family:ui-monospace,Consolas,monospace; }
    .pill { border:1px solid #2f4e73; border-radius:6px; padding:1px 6px; background:#0a1524; }
    .seglist { overflow:auto; min-height:0; }
    .seg { width:100%; text-align:left; margin-bottom:6px; padding:7px; border-radius:7px; border:1px solid #2b4564; background:#12243a; color:#e6eef8; }
    .seg.active { border-color:#f4c066; background:#1a3452; }
    button { background:#13253c; color:#e6eef8; border:1px solid #2b4564; border-radius:8px; padding:7px 10px; cursor:pointer; }
    button:hover { border-color:#4a729f; }
    input[type="range"] { width:100%; accent-color:#58a8ff; }
    input[type="file"] { width:100%; }
    video { width:100%; border-radius:8px; background:#000; }
    .map-panel, .plot-panel { position:relative; overflow:hidden; }
    #map, #attChart { width:100%; height:100%; }
    .overlay { position:absolute; top:10px; right:10px; z-index:800; background:rgba(7,13,24,0.82); border:1px solid #35597f; border-radius:9px; padding:9px; font-size:12px; line-height:1.35; max-width:78%; pointer-events:none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel left">
      <div class="box">
        <div><b>GoPro GPS + Pose Sync Viewer</b></div>
        <div class="small">MP4 GPS, Vakaros, and pose CSV are synchronized by timestamps.</div>
      </div>

      <div class="box">
        <div class="row"><b>Video</b></div>
        <input id="videoFile" type="file" accept="video/mp4"/>
        <div class="row small mono" style="margin-top:6px;">
          <span>MP4 <span id="nMp4" class="pill">0</span></span>
          <span>Vakaros <span id="nVak" class="pill">0</span></span>
          <span>Pose <span id="nPose" class="pill">0</span></span>
          <span>Segments <span id="nSeg" class="pill">0</span></span>
        </div>
        <div id="syncLine" class="small mono"></div>
      </div>

      <div class="box">
        <div class="row">
          <button id="playPause">Play</button>
          <button id="prevSeg">Prev Seg</button>
          <button id="nextSeg">Next Seg</button>
        </div>
        <div style="margin-top:8px;">
          <input id="scrub" type="range" min="0" max="0" step="0.01" value="0"/>
          <div class="row space small mono" style="margin-top:6px;">
            <span><span id="curT">0.00</span> / <span id="durT">0.00</span> s</span>
            <span>Offset <span id="offLbl">0.0</span> s</span>
          </div>
          <input id="vakOffset" type="range" min="-60" max="60" step="0.1" value="0"/>
        </div>
      </div>

      <div class="box seglist">
        <div class="row space" style="margin-bottom:8px;"><b>Pose Segments</b><span class="small">click to jump</span></div>
        <div id="segList"></div>
      </div>

      <div class="box"><video id="vid" controls playsinline></video></div>
    </div>

    <div class="right">
      <div class="panel map-panel">
        <div id="map"></div>
        <div class="overlay mono">
          <div>t: <b id="ovT">0.00</b> s</div>
          <div>MP4 time: <b id="ovMp4Time">-</b></div>
          <div>MP4 lat/lon: <b id="ovMp4Pos">-</b></div>
          <div style="margin-top:4px;">Vakaros time: <b id="ovVakTime">-</b></div>
          <div>Vakaros lat/lon: <b id="ovVakPos">-</b></div>
          <div>Heel: <b id="ovHeel">-</b> deg | Trim: <b id="ovTrim">-</b> deg</div>
          <div style="margin-top:4px;">Pose frame: <b id="ovFrame">-</b> | trunk: <b id="ovTrunk">-</b> deg</div>
          <div>Mom roll/pitch: <b id="ovMom">-</b></div>
        </div>
      </div>
      <div class="panel plot-panel"><div id="attChart"></div></div>
    </div>
  </div>

<script>
const mp4Track = __MP4_JSON__;
const vak = __VAKAROS_JSON__;
const poseTrack = __POSE_JSON__;
const segments = __SEGMENTS_JSON__;
const meta = __META_JSON__;

const vakTrack = (vak && Array.isArray(vak.track)) ? vak.track : [];
const mp4Times = mp4Track.map(p => Number(p.t));
const vakTimes = vakTrack.map(p => Number(p.ts));
const poseTimes = poseTrack.map(p => Number(p.t));

function n(v, d=2) { return Number.isFinite(v) ? v.toFixed(d) : "-"; }
function epochStr(ts) { return Number.isFinite(ts) ? new Date(ts * 1000).toISOString().replace("T"," ").replace(".000Z","Z") : "-"; }
function gpsdtStr(s) {
  if (!s || typeof s !== "string") return "-";
  const z = s.endsWith("Z") ? "Z" : "";
  const c = z ? s.slice(0, -1) : s;
  const parts = c.split(" ");
  if (parts.length !== 2) return s;
  return parts[0].replaceAll(":", "-") + " " + parts[1] + z;
}
function idxLE(arr, x) {
  let lo = 0, hi = arr.length - 1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] <= x) lo = m + 1; else hi = m - 1;
  }
  return Math.max(0, Math.min(arr.length - 1, hi));
}
function interp(points, times, x, keys) {
  if (!points.length || !times.length) return null;
  if (points.length === 1) {
    const p = points[0], out = {_i: 0};
    for (const k of keys) { const v = Number(p[k]); out[k] = Number.isFinite(v) ? v : null; }
    return out;
  }
  if (x <= times[0]) { const p = points[0], out = {_i:0}; for (const k of keys) { const v = Number(p[k]); out[k] = Number.isFinite(v) ? v : null; } return out; }
  if (x >= times[times.length - 1]) { const p = points[points.length - 1], out = {_i: points.length - 1}; for (const k of keys) { const v = Number(p[k]); out[k] = Number.isFinite(v) ? v : null; } return out; }
  const i0 = idxLE(times, x), i1 = Math.min(i0 + 1, points.length - 1);
  const t0 = times[i0], t1 = times[i1];
  const a = Math.abs(t1 - t0) < 1e-12 ? 0 : (x - t0) / (t1 - t0);
  const p0 = points[i0], p1 = points[i1], out = {_i: i0};
  for (const k of keys) {
    const v0 = Number(p0[k]), v1 = Number(p1[k]);
    if (Number.isFinite(v0) && Number.isFinite(v1)) out[k] = v0 + (v1 - v0) * a;
    else if (Number.isFinite(v0)) out[k] = v0;
    else if (Number.isFinite(v1)) out[k] = v1;
    else out[k] = null;
  }
  return out;
}
"""
)


HTML_TEMPLATE_PARTS.append(
    r"""
document.getElementById("nMp4").textContent = String(mp4Track.length);
document.getElementById("nVak").textContent = String(vakTrack.length);
document.getElementById("nPose").textContent = String(poseTrack.length);
document.getElementById("nSeg").textContent = String(segments.length);

if (meta && meta.sync_distance_stats && Number.isFinite(meta.sync_distance_stats.mean_m)) {
  const s = meta.sync_distance_stats;
  document.getElementById("syncLine").textContent = `MP4↔Vakaros mean ${s.mean_m.toFixed(1)}m, p95 ${s.p95_m.toFixed(1)}m`;
} else {
  document.getElementById("syncLine").textContent = "No MP4↔Vakaros distance summary available.";
}

function sampleMp4(t) {
  if (!mp4Track.length) return null;
  const o = interp(mp4Track, mp4Times, t, ["lat", "lon", "ts"]);
  if (!o) return null;
  const i = Math.max(0, Math.min(mp4Track.length - 1, o._i || 0));
  return {lat:o.lat, lon:o.lon, ts:o.ts, gpsdt: mp4Track[i].gpsdt || ""};
}
function sampleVak(ts) {
  if (!vakTrack.length || !Number.isFinite(ts)) return null;
  return interp(vakTrack, vakTimes, ts, ["lat", "lon", "heel", "trim", "sog_kts", "cog"]);
}
function samplePose(t) {
  if (!poseTrack.length) return null;
  return interp(poseTrack, poseTimes, t, ["lat", "lon", "frame", "trunk_angle", "moment_roll", "moment_pitch"]);
}

function segColor(side) {
  const s = String(side || "").toLowerCase();
  if (s === "port") return "#7dc3ff";
  if (s === "starboard") return "#ffca72";
  return "#c9d6e8";
}

const map = L.map("map", { zoomControl: true, preferCanvas: true });
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const vakLatLng = vakTrack.map(p => [Number(p.lat), Number(p.lon)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
const mp4LatLng = mp4Track.map(p => [Number(p.lat), Number(p.lon)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
const poseLatLng = poseTrack.map(p => [Number(p.lat), Number(p.lon)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));

const vakLine = vakLatLng.length ? L.polyline(vakLatLng, { color: "#8f9db1", weight: 3, opacity: 0.8 }).addTo(map) : null;
const mp4Line = mp4LatLng.length ? L.polyline(mp4LatLng, { color: "#58a8ff", weight: 3.5, opacity: 0.9 }).addTo(map) : null;
const poseLine = poseLatLng.length ? L.polyline(poseLatLng, { color: "#f4be5e", weight: 3, opacity: 0.82, dashArray: "6,5" }).addTo(map) : null;

const vakDot = vakLatLng.length ? L.circleMarker(vakLatLng[0], { radius: 6, color: "#80f0c4", fillColor: "#80f0c4", fillOpacity: 0.95, weight: 1 }).addTo(map) : null;
const mp4Dot = mp4LatLng.length ? L.circleMarker(mp4LatLng[0], { radius: 6, color: "#58a8ff", fillColor: "#58a8ff", fillOpacity: 0.95, weight: 1 }).addTo(map) : null;
const poseDot = poseLatLng.length ? L.circleMarker(poseLatLng[0], { radius: 6, color: "#f4be5e", fillColor: "#f4be5e", fillOpacity: 0.95, weight: 1 }).addTo(map) : null;

const segLayers = new Map();
for (const s of segments) {
  const pts = (Array.isArray(s.path) ? s.path : []).map(p => [Number(p.lat), Number(p.lon)]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (pts.length < 2) continue;
  const ly = L.polyline(pts, { color: segColor(s.side), weight: 5, opacity: 0.8 }).addTo(map);
  ly.on("click", () => {
    if (Number.isFinite(s.start_t)) {
      vid.currentTime = s.start_t;
      updateAll(s.start_t);
    }
    setActiveSeg(s.id, true);
  });
  segLayers.set(s.id, ly);
}

{
  const refs = [];
  if (vakLine) refs.push(vakLine);
  if (mp4Line) refs.push(mp4Line);
  if (poseLine) refs.push(poseLine);
  if (refs.length) {
    let b = refs[0].getBounds();
    for (let k = 1; k < refs.length; k++) b = b.extend(refs[k].getBounds());
    map.fitBounds(b.pad(0.15));
  } else {
    map.setView([0, 0], 2);
  }
}

const segList = document.getElementById("segList");
const segButtons = new Map();
let activeSeg = null;

function setActiveSeg(segId, fitBounds=false) {
  activeSeg = segId;
  for (const s of segments) {
    const ly = segLayers.get(s.id);
    const btn = segButtons.get(s.id);
    const on = (s.id === segId);
    if (ly) {
      ly.setStyle({ weight: on ? 7 : 5, opacity: on ? 1.0 : 0.8 });
      if (on && fitBounds) map.fitBounds(ly.getBounds().pad(0.35));
    }
    if (btn) {
      if (on) btn.classList.add("active"); else btn.classList.remove("active");
    }
  }
}

function segAtTime(t) {
  for (const s of segments) {
    if (Number.isFinite(s.start_t) && Number.isFinite(s.end_t) && t >= s.start_t && t <= s.end_t) return s;
  }
  return null;
}

if (!segments.length) {
  segList.innerHTML = '<div class="small">No pose segments found.</div>';
} else {
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    const b = document.createElement("button");
    b.className = "seg";
    b.innerHTML = `<div class="row space"><b>#${i + 1}</b><span>${s.side || "Unknown"}</span></div>
      <div class="small mono">${n(s.start_t,2)}s - ${n(s.end_t,2)}s (${n(s.duration_s,2)}s)</div>
      <div class="small mono">frames ${s.start_frame} - ${s.end_frame}</div>
      <div class="small mono">mean trunk ${n(Number(s.mean_trunk_deg),1)} deg</div>`;
    b.addEventListener("click", () => {
      if (Number.isFinite(s.start_t)) {
        vid.currentTime = s.start_t;
        updateAll(s.start_t);
      }
      setActiveSeg(s.id, true);
    });
    segList.appendChild(b);
    segButtons.set(s.id, b);
  }
}
"""
)

HTML_TEMPLATE_PARTS.append(
    r"""
const vid = document.getElementById("vid");
const videoFile = document.getElementById("videoFile");
const playPause = document.getElementById("playPause");
const prevSeg = document.getElementById("prevSeg");
const nextSeg = document.getElementById("nextSeg");
const scrub = document.getElementById("scrub");
const curT = document.getElementById("curT");
const durT = document.getElementById("durT");
const off = document.getElementById("vakOffset");
const offLbl = document.getElementById("offLbl");

const ovT = document.getElementById("ovT");
const ovMp4Time = document.getElementById("ovMp4Time");
const ovMp4Pos = document.getElementById("ovMp4Pos");
const ovVakTime = document.getElementById("ovVakTime");
const ovVakPos = document.getElementById("ovVakPos");
const ovHeel = document.getElementById("ovHeel");
const ovTrim = document.getElementById("ovTrim");
const ovFrame = document.getElementById("ovFrame");
const ovTrunk = document.getElementById("ovTrunk");
const ovMom = document.getElementById("ovMom");

let scrubbing = false;
let vakOffsetSec = 0.0;
let chartReady = false;

videoFile.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  vid.src = URL.createObjectURL(file);
  vid.load();
});
playPause.addEventListener("click", () => { if (vid.paused) vid.play(); else vid.pause(); });
vid.addEventListener("play", () => { playPause.textContent = "Pause"; });
vid.addEventListener("pause", () => { playPause.textContent = "Play"; });
vid.addEventListener("loadedmetadata", () => {
  const d = Number.isFinite(vid.duration) ? vid.duration : 0;
  scrub.max = d.toFixed(2);
  durT.textContent = d.toFixed(2);
});

scrub.addEventListener("input", () => {
  scrubbing = true;
  const t = parseFloat(scrub.value || "0");
  curT.textContent = t.toFixed(2);
  updateAll(t);
});
scrub.addEventListener("change", () => {
  vid.currentTime = parseFloat(scrub.value || "0");
  scrubbing = false;
});

off.addEventListener("input", () => {
  vakOffsetSec = parseFloat(off.value || "0");
  if (!Number.isFinite(vakOffsetSec)) vakOffsetSec = 0;
  offLbl.textContent = vakOffsetSec.toFixed(1);
  buildPlot(vakOffsetSec);
  updateAll(vid.currentTime || 0);
});

function jumpSeg(dir) {
  if (!segments.length) return;
  const t = Number.isFinite(vid.currentTime) ? vid.currentTime : 0;
  let target = null;
  if (dir < 0) {
    for (let i = segments.length - 1; i >= 0; i--) {
      if (segments[i].start_t < t - 1e-6) { target = segments[i]; break; }
    }
    if (!target) target = segments[0];
  } else {
    for (let i = 0; i < segments.length; i++) {
      if (segments[i].start_t > t + 1e-6) { target = segments[i]; break; }
    }
    if (!target) target = segments[segments.length - 1];
  }
  if (!target) return;
  vid.currentTime = target.start_t;
  updateAll(target.start_t);
  setActiveSeg(target.id, true);
}
prevSeg.addEventListener("click", () => jumpSeg(-1));
nextSeg.addEventListener("click", () => jumpSeg(1));

function attitudeSeries(offsetSec) {
  const t = [], heel = [], trim = [];
  let lastT = -1e9;
  for (const p of mp4Track) {
    const tv = Number(p.t), ts = Number(p.ts);
    if (!Number.isFinite(tv) || !Number.isFinite(ts)) continue;
    if (tv - lastT < 0.08) continue;
    lastT = tv;
    const v = sampleVak(ts + offsetSec);
    t.push(tv);
    heel.push(v && Number.isFinite(v.heel) ? v.heel : null);
    trim.push(v && Number.isFinite(v.trim) ? v.trim : null);
  }
  return {t, heel, trim};
}

function updateCursor(t) {
  if (!chartReady) return;
  Plotly.relayout("attChart", {"shapes[0].x0": t, "shapes[0].x1": t});
}

function buildPlot(offsetSec) {
  const as = attitudeSeries(offsetSec);
  const tx = [], ty = [];
  for (const p of poseTrack) {
    const t = Number(p.t), tr = Number(p.trunk_angle);
    if (!Number.isFinite(t)) continue;
    tx.push(t);
    ty.push(Number.isFinite(tr) ? tr : null);
  }
  const traces = [
    {x: as.t, y: as.heel, name: "Heel/Roll (deg)", mode: "lines", line: {color:"#58a8ff", width:1.8}},
    {x: as.t, y: as.trim, name: "Trim/Pitch (deg)", mode: "lines", line: {color:"#42ce90", width:1.8}},
    {x: tx, y: ty, name: "Trunk (deg)", mode: "lines", yaxis: "y2", line: {color:"#f4be5e", width:1.3}},
  ];
  const t0 = Number.isFinite(vid.currentTime) ? vid.currentTime : 0;
  const layout = {
    margin: {l:50, r:50, t:24, b:40},
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color:"#d9e6f6"},
    hovermode: "x unified",
    legend: {orientation:"h", y:1.14, x:0.01},
    xaxis: {title: "Video time (s)", gridcolor:"#27415f"},
    yaxis: {title: "Boat heel/trim (deg)", gridcolor:"#27415f"},
    yaxis2: {title: "Trunk (deg)", overlaying: "y", side: "right", showgrid: false},
    shapes: [{type:"line", x0:t0, x1:t0, y0:0, y1:1, xref:"x", yref:"paper", line:{color:"#f8f9fb", width:1, dash:"dot"}}],
  };
  Plotly.react("attChart", traces, layout, {responsive:true, displaylogo:false});
  chartReady = true;
}

function updateAll(t) {
  ovT.textContent = n(t, 2);
  const m = sampleMp4(t);
  if (m) {
    if (mp4Dot && Number.isFinite(m.lat) && Number.isFinite(m.lon)) mp4Dot.setLatLng([m.lat, m.lon]);
    ovMp4Time.textContent = gpsdtStr(m.gpsdt);
    ovMp4Pos.textContent = `${n(m.lat, 6)}, ${n(m.lon, 6)}`;
  } else {
    ovMp4Time.textContent = "-";
    ovMp4Pos.textContent = "-";
  }

  const v = (m && Number.isFinite(m.ts)) ? sampleVak(m.ts + vakOffsetSec) : null;
  if (v) {
    if (vakDot && Number.isFinite(v.lat) && Number.isFinite(v.lon)) vakDot.setLatLng([v.lat, v.lon]);
    ovVakTime.textContent = epochStr((m ? m.ts : NaN) + vakOffsetSec);
    ovVakPos.textContent = `${n(v.lat, 6)}, ${n(v.lon, 6)}`;
    ovHeel.textContent = n(v.heel, 2);
    ovTrim.textContent = n(v.trim, 2);
  } else {
    ovVakTime.textContent = "-"; ovVakPos.textContent = "-"; ovHeel.textContent = "-"; ovTrim.textContent = "-";
  }

  const p = samplePose(t);
  if (p) {
    if (poseDot && Number.isFinite(p.lat) && Number.isFinite(p.lon)) poseDot.setLatLng([p.lat, p.lon]);
    ovFrame.textContent = Number.isFinite(p.frame) ? String(Math.round(p.frame)) : "-";
    ovTrunk.textContent = n(p.trunk_angle, 2);
    ovMom.textContent = `${n(p.moment_roll, 0)} / ${n(p.moment_pitch, 0)} Nm`;
  } else {
    ovFrame.textContent = "-"; ovTrunk.textContent = "-"; ovMom.textContent = "-";
  }

  const s = segAtTime(t);
  setActiveSeg(s ? s.id : null, false);
  updateCursor(t);
}

vid.addEventListener("timeupdate", () => {
  const t = Number.isFinite(vid.currentTime) ? vid.currentTime : 0;
  if (!scrubbing) scrub.value = t.toFixed(2);
  curT.textContent = t.toFixed(2);
  updateAll(t);
});
vid.addEventListener("seeked", () => updateAll(vid.currentTime || 0));

buildPlot(0);
updateAll(0);
</script>
</body>
</html>
"""
)

HTML_TEMPLATE = "".join(HTML_TEMPLATE_PARTS)


def pick_file(title: str, filetypes) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path or ""


def main():
    mp4 = pick_file("Select GoPro MP4", [("MP4 files", "*.mp4"), ("All files", "*.*")])
    if not mp4:
        return

    vak_csv = pick_file("Select Vakaros CSV (optional)", [("CSV files", "*.csv"), ("All files", "*.*")])
    pose_csv = pick_file("Select Pose CSV (optional)", [("CSV files", "*.csv"), ("All files", "*.*")])

    try:
        mp4_track = extract_gopro_gps_from_mp4(mp4)
        if len(mp4_track) < 2:
            raise RuntimeError("No usable GPS samples found in MP4 (GPS disabled or no lock).")

        vak = parse_vakaros_csv(vak_csv) if vak_csv else {"track": [], "columns": {}}
        pose_rows = parse_pose_csv(pose_csv) if pose_csv else []
        pose_sync = sync_pose_to_mp4(pose_rows, mp4_track)
        pose_segments = detect_pose_segments(pose_sync, trunk_threshold_deg=20.0, min_duration_s=2.0)

        pose_view = _downsample(pose_sync, "t", 1.0 / 15.0)
        stats = sync_distance_stats(mp4_track, vak.get("track", []))
        meta = {
            "video_file": os.path.basename(mp4),
            "vakaros_file": os.path.basename(vak_csv) if vak_csv else None,
            "pose_file": os.path.basename(pose_csv) if pose_csv else None,
            "sync_distance_stats": stats,
            "mp4_points": len(mp4_track),
            "vakaros_points": len(vak.get("track", [])),
            "pose_points": len(pose_view),
            "segments": len(pose_segments),
        }

        out_dir = os.path.dirname(mp4)
        base = os.path.splitext(os.path.basename(mp4))[0]
        out_html = os.path.join(out_dir, f"{base}_gps_pose_sync_viewer.html")

        html = HTML_TEMPLATE.replace("__MP4_JSON__", json.dumps(mp4_track, separators=(",", ":")))
        html = html.replace("__VAKAROS_JSON__", json.dumps(vak, separators=(",", ":")))
        html = html.replace("__POSE_JSON__", json.dumps(pose_view, separators=(",", ":")))
        html = html.replace("__SEGMENTS_JSON__", json.dumps(pose_segments, separators=(",", ":")))
        html = html.replace("__META_JSON__", json.dumps(meta, separators=(",", ":")))

        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)

        messagebox.showinfo(
            "Done",
            (
                f"Created:\n{out_html}\n\n"
                f"MP4 points: {len(mp4_track)}\n"
                f"Vakaros points: {len(vak.get('track', []))}\n"
                f"Pose synced points: {len(pose_view)}\n"
                f"Pose segments: {len(pose_segments)}\n\n"
                "Open the HTML in your browser and choose the MP4 inside the page."
            ),
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    main()
