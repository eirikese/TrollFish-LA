"""
Dual GoPro MP4 + Vakaros GPS viewer generator  (multi-video per athlete)

Each athlete / dataset can have *multiple consecutive GoPro clips*.
Videos are synchronised via embedded GPS epoch timestamps.

  Dataset 1  =  RED  shades
  Dataset 2  =  BLUE shades

When the master timeline is outside a video's GPS range the corresponding
pane shows a black screen.

Install / setup (Windows)
1) Install ExifTool  →  https://sourceforge.net/projects/exiftool/
2) Python: no extra pip packages required (standard library only).
   (The HTML uses Leaflet via CDN.)
"""

import csv
import json
import os
import shutil
import subprocess
import time as _time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import filedialog, messagebox
from datetime import datetime, timezone

# Thinning interval: keep ≤1 GPS point per this many seconds of video time.
# Lower = denser track / more accurate sync.  Higher = faster processing & smaller HTML.
GPS_THIN_INTERVAL_S = 60

# ------------------------------------------------------------------
# GPS extraction from GoPro MP4
# ------------------------------------------------------------------
def extract_gopro_gps_from_mp4(mp4_path: str, exiftool_exe: str | None = None):
    """
    Extract GoPro GPS samples from an MP4 using ExifTool.

    Returns: list of dicts, sorted by SampleTime:
      {
        "t":   float   # SampleTime in seconds (HTML5 video currentTime)
        "lat": float
        "lon": float
        "gpsdt": str   # GPSDateTime string from ExifTool
        "ts":  float   # epoch seconds (UTC)
      }
    """
    def find_exiftool():
        if exiftool_exe and os.path.exists(exiftool_exe):
            return exiftool_exe
        exe = shutil.which("exiftool")
        if exe:
            return exe
        fallback = r"C:\Program Files\exiftool-13.51_64\exiftool.exe"
        if os.path.exists(fallback):
            return fallback
        raise RuntimeError("ExifTool not found. Ensure exiftool is on PATH or set exiftool_exe.")

    def gpsdt_to_epoch(gpsdt: str) -> float:
        s = (gpsdt or "").strip()
        if not s or s in ("-", "undef", "UNDEF"):
            return float("nan")
        z = s.endswith("Z")
        core = s[:-1] if z else s
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
            try:
                dt = datetime.strptime(core, fmt).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                pass
        return float("nan")

    exif = find_exiftool()
    fmt = "$SampleTime,$GPSLatitude,$GPSLongitude,$GPSDateTime"
    cmd = [exif, "-ee", "-n", "-p", fmt, mp4_path]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "ExifTool failed.")

    pts = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = line.split(",", 3)
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
            lat = float(parts[1])
            lon = float(parts[2])
        except ValueError:
            continue
        gpsdt = parts[3].strip() if len(parts) == 4 else ""
        ts = gpsdt_to_epoch(gpsdt)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if not (ts == ts):  # NaN check
            continue
        pts.append({"t": t, "lat": lat, "lon": lon, "gpsdt": gpsdt, "ts": ts})

    pts.sort(key=lambda d: d["t"])
    cleaned = []
    last_t = None
    for d in pts:
        if last_t is None or d["t"] != last_t:
            cleaned.append(d)
            last_t = d["t"]

    # --- Outlier removal: drop points that imply impossible speed ---
    n_before = len(cleaned)
    cleaned = _filter_gps_outliers(cleaned)
    n_removed = n_before - len(cleaned)
    if n_removed:
        print(f"    GPS outlier filter: removed {n_removed}/{n_before} bad-fix points "
              f"from {os.path.basename(mp4_path)}")

    return cleaned


# Maximum plausible speed (m/s) for a sailing dinghy / small keelboat.
# Anything above this between consecutive GPS samples is treated as a
# bad-fix outlier.  25 m/s ≈ 50 knots – well above any dinghy/foiler.
_MAX_PLAUSIBLE_SPEED_MS = 25.0


def _haversine_m(lat1, lon1, lat2, lon2):
    """Quick equirectangular distance in metres (good enough for <1 km)."""
    import math
    dlat = (lat2 - lat1) * 111320
    dlon = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat * dlat + dlon * dlon)


def _filter_gps_outliers(pts: list[dict],
                         max_speed: float = _MAX_PLAUSIBLE_SPEED_MS,
                         max_passes: int = 3) -> list[dict]:
    """
    Remove GPS outlier points that imply travel faster than *max_speed* m/s.

    Strategy (multi-pass):
      For each consecutive triple (A, B, C), if A→B speed > max_speed
      AND B→C speed > max_speed, then B is almost certainly an outlier
      (bad fix) – remove it.  Single-sided spikes (only one neighbour
      is fast) could be a legitimate sharp turn; keep those.

      Also drops the very first / very last point if the speed to/from
      its only neighbour exceeds the limit (start/end glitches).

      Repeat up to *max_passes* times because removing one bad point
      can reveal the next one as a new spike.

    Also filters out (0,0) and duplicate-timestamp points.
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
            dt_ab = abs(pts[i]["ts"] - pts[i - 1]["ts"])
            dt_bc = abs(pts[i + 1]["ts"] - pts[i]["ts"])
            if dt_ab < 0.1:
                dt_ab = 0.1
            if dt_bc < 0.1:
                dt_bc = 0.1
            d_ab = _haversine_m(pts[i - 1]["lat"], pts[i - 1]["lon"],
                                pts[i]["lat"], pts[i]["lon"])
            d_bc = _haversine_m(pts[i]["lat"], pts[i]["lon"],
                                pts[i + 1]["lat"], pts[i + 1]["lon"])
            spd_ab = d_ab / dt_ab
            spd_bc = d_bc / dt_bc
            if spd_ab > max_speed and spd_bc > max_speed:
                keep[i] = False

        # Edge: first point
        if n >= 2:
            dt = abs(pts[1]["ts"] - pts[0]["ts"])
            if dt < 0.1:
                dt = 0.1
            d = _haversine_m(pts[0]["lat"], pts[0]["lon"],
                             pts[1]["lat"], pts[1]["lon"])
            if d / dt > max_speed:
                keep[0] = False

        # Edge: last point
        if n >= 2:
            dt = abs(pts[-1]["ts"] - pts[-2]["ts"])
            if dt < 0.1:
                dt = 0.1
            d = _haversine_m(pts[-2]["lat"], pts[-2]["lon"],
                             pts[-1]["lat"], pts[-1]["lon"])
            if d / dt > max_speed:
                keep[-1] = False

        new_pts = [p for p, k in zip(pts, keep) if k]
        removed = len(pts) - len(new_pts)
        if removed == 0:
            break
        pts = new_pts

    return pts


def thin_gps_track(pts: list[dict], interval_s: float = GPS_THIN_INTERVAL_S) -> list[dict]:
    """
    Thin a sorted GPS point list to ≤1 point per *interval_s* seconds of
    video time (the "t" field).  Always keeps the first and last point so
    that epoch min/max stay accurate.
    """
    if len(pts) <= 2 or interval_s <= 0:
        return pts
    out = [pts[0]]
    last_t = pts[0]["t"]
    for p in pts[1:-1]:
        if p["t"] - last_t >= interval_s:
            out.append(p)
            last_t = p["t"]
    out.append(pts[-1])
    return out


# ------------------------------------------------------------------
# Vakaros CSV parsing
# ------------------------------------------------------------------
def parse_vakaros_csv(csv_path: str):
    """
    Reads a GPS CSV file.

    Supported column layouts (case-insensitive):
      A) Vakaros-native:  timestamp, latitude, longitude
      B) Sensor logger:   iso_time (or timestamp_ms), lat, lon

    Returns: {"track":[{"lat":...,"lon":...,"ts": epoch_seconds, "sog": float|None}, ...]}
    """
    if not csv_path:
        return {"track": []}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"track": []}

    cols = {c.strip().lower(): c for c in rows[0].keys()}

    def has(*names):
        return all(n in cols for n in names)

    # --- Detect SOG column (optional) ---
    c_sog = None
    for sog_name in ("sog_mps", "sog"):
        if sog_name in cols:
            c_sog = cols[sog_name]
            break

    # --- Detect format ---
    if has("timestamp", "latitude", "longitude"):
        # Format A: Vakaros-native
        c_ts  = cols["timestamp"]
        c_lat = cols["latitude"]
        c_lon = cols["longitude"]
        fmt_a = True
    elif has("lat", "lon") and (has("iso_time") or has("timestamp_ms")):
        # Format B: Sensor/IMU logger
        c_lat = cols["lat"]
        c_lon = cols["lon"]
        c_ts  = cols.get("iso_time") or cols.get("timestamp_ms")
        fmt_a = False
    else:
        raise RuntimeError(
            f"CSV columns not recognised. Found: {list(rows[0].keys())}\n"
            "Expected either {timestamp, latitude, longitude} "
            "or {iso_time / timestamp_ms, lat, lon}."
        )

    def parse_ts_a(s: str) -> float:
        """Vakaros ISO timestamps."""
        s = s.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(s, fmt).timestamp()
            except ValueError:
                pass
        raise ValueError(s)

    def parse_ts_b(s: str) -> float:
        """Sensor logger: iso_time (2026-02-11T14:06:30Z) or epoch-ms."""
        s = s.strip()
        if not s:
            raise ValueError("empty")
        # Pure numeric → epoch milliseconds
        try:
            ms = float(s)
            return ms / 1000.0
        except ValueError:
            pass
        # ISO variants
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%SZ",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                pass
        raise ValueError(s)

    parse_fn = parse_ts_a if fmt_a else parse_ts_b
    total = len(rows)
    track = []
    for idx, r in enumerate(rows):
        if idx % 5000 == 0 and total > 5000:
            print(f"  Vakaros CSV: {idx}/{total} rows ({100*idx//total}%)")
        try:
            ts  = parse_fn(r[c_ts])
            lat = float(r[c_lat])
            lon = float(r[c_lon])
        except Exception:
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        pt = {"lat": lat, "lon": lon, "ts": ts}
        if c_sog is not None:
            try:
                pt["sog"] = float(r[c_sog])
            except (ValueError, KeyError):
                pt["sog"] = None
        track.append(pt)

    print(f"  Vakaros CSV: {total}/{total} rows (100%) — {len(track)} valid points")
    track.sort(key=lambda d: d["ts"])
    return {"track": track}


# ------------------------------------------------------------------
# Build segment list from multiple MP4s + optional Vakaros CSV
# ------------------------------------------------------------------
def build_dataset(mp4_paths: list[str], vak_csv: str):
    """
    Returns a JSON-serialisable dict:
      {
        "segments": [
            {"name": "GX010001.MP4", "gps": [...], "epochMin": ..., "epochMax": ...},
            ...
        ],
        "vakaros": {"track": [...]}
      }
    Segments are sorted by epochMin.
    """
    segments = []
    n = len(mp4_paths)
    for i, mp4 in enumerate(mp4_paths, 1):
        name = os.path.basename(mp4)
        print(f"  [{i}/{n}] Extracting GPS from {name} ...")
        track = extract_gopro_gps_from_mp4(mp4)
        if len(track) < 2:
            print(f"           ⚠ Skipped (only {len(track)} GPS point(s))")
            continue
        epochs = [p["ts"] for p in track]
        segments.append({
            "name": name,
            "gps": track,
            "epochMin": min(epochs),
            "epochMax": max(epochs),
        })
        print(f"           ✓ {len(track)} GPS points  "
              f"({datetime.fromtimestamp(min(epochs), tz=timezone.utc).strftime('%H:%M:%S')} – "
              f"{datetime.fromtimestamp(max(epochs), tz=timezone.utc).strftime('%H:%M:%S')} UTC)")
    segments.sort(key=lambda s: s["epochMin"])
    if vak_csv:
        print(f"  Parsing Vakaros CSV: {os.path.basename(vak_csv)}")
    vak = parse_vakaros_csv(vak_csv) if vak_csv else {"track": []}
    return {"segments": segments, "vakaros": vak}


def _extract_one(mp4_path: str):
    """Worker for parallel extraction.  Returns (path, basename, track, elapsed)."""
    name = os.path.basename(mp4_path)
    t0 = _time.perf_counter()
    track = extract_gopro_gps_from_mp4(mp4_path)
    dt = _time.perf_counter() - t0
    return (mp4_path, name, track, dt)


def build_datasets_parallel(
    mp4s_1: list[str], vak_csv_1: str,
    mp4s_2: list[str], vak_csv_2: str,
):
    """
    Build both datasets with all MP4 GPS extractions running in parallel.
    Returns (ds1, ds2) dicts.
    """
    all_jobs = [(1, p) for p in mp4s_1] + [(2, p) for p in mp4s_2]
    total = len(all_jobs)
    results_1: dict[str, tuple] = {}   # path → (name, track, dt)
    results_2: dict[str, tuple] = {}
    done = 0

    print(f"\n  Extracting GPS from {total} MP4{'s' if total > 1 else ''} in parallel ...")
    t_all = _time.perf_counter()

    with ThreadPoolExecutor(max_workers=min(total, os.cpu_count() or 4)) as pool:
        futures = {}
        for ds_idx, path in all_jobs:
            f = pool.submit(_extract_one, path)
            futures[f] = ds_idx

        for f in as_completed(futures):
            ds_idx = futures[f]
            path, name, track, dt = f.result()
            done += 1
            status = f"✓ {len(track)} pts" if len(track) >= 2 else f"⚠ {len(track)} pts"
            colour = "RED" if ds_idx == 1 else "BLUE"
            print(f"  [{done}/{total}] {colour}  {name}  {status}  [{dt:.1f}s]")
            bucket = results_1 if ds_idx == 1 else results_2
            bucket[path] = (name, track, dt)

    dt_all = _time.perf_counter() - t_all
    print(f"  All extractions done in {dt_all:.1f}s\n")

    def assemble(mp4_paths, results, vak_csv, label):
        segments = []
        for mp4 in mp4_paths:
            name, track, dt = results[mp4]
            if len(track) < 2:
                continue
            raw_n = len(track)
            track = thin_gps_track(track)
            epochs = [p["ts"] for p in track]
            segments.append({
                "name": name, "gps": track,
                "epochMin": min(epochs), "epochMax": max(epochs),
            })
            print(f"  {label}  {name}: {raw_n} → {len(track)} pts  "
                  f"({datetime.fromtimestamp(min(epochs), tz=timezone.utc).strftime('%H:%M:%S')} – "
                  f"{datetime.fromtimestamp(max(epochs), tz=timezone.utc).strftime('%H:%M:%S')} UTC)")
        segments.sort(key=lambda s: s["epochMin"])
        if vak_csv:
            print(f"  {label}  Parsing Vakaros CSV: {os.path.basename(vak_csv)}")
        vak = parse_vakaros_csv(vak_csv) if vak_csv else {"track": []}
        return {"segments": segments, "vakaros": vak}

    print("=== Athlete 1 (RED) ===")
    ds1 = assemble(mp4s_1, results_1, vak_csv_1, "RED ")
    print("\n=== Athlete 2 (BLUE) ===")
    ds2 = assemble(mp4s_2, results_2, vak_csv_2, "BLUE")
    return ds1, ds2


def auto_build_datasets(mp4_paths: list[str], csv_paths: list[str]):
    """
    Two-phase approach:
      Phase 1 – FAST: read file-level GPS (no -ee) for camera→CSV matching.
      Phase 2 – Full -ee extraction in parallel, only after assignment.
    Returns (ds1, ds2) dicts.
    """
    import math
    import bisect

    # ---- Phase 1: fast GPS for matching ----
    _SNIPPET_PTS   = 30   # collect up to this many GPS points per clip
    _SNIPPET_SECS  = 30   # stop reading after this many wall-clock seconds
    _SNIPPET_THIN  = 5    # keep ≤1 point per N seconds of GPS time

    def _find_exiftool() -> str | None:
        exif = shutil.which("exiftool")
        if exif:
            return exif
        fallback = r"C:\Program Files\exiftool-13.51_64\exiftool.exe"
        return fallback if os.path.exists(fallback) else None

    def _parse_ts(raw: str) -> float | None:
        """Parse ExifTool timestamp string → epoch float or None."""
        if not raw or raw in ("-", "undef", ""):
            return None
        s = str(raw).strip()
        core = s[:-1] if s.endswith("Z") else s
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(core, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
        return None

    def _valid(lat, lon):
        return -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0)

    def _quick_gps(mp4_path: str) -> dict | None:
        """
        Fast GPS snippet per clip for matching.
        Uses -ee embedded stream, reads lines until we have _SNIPPET_PTS
        valid points (thinned to 1 per _SNIPPET_THIN secs GPS-time)
        or _SNIPPET_SECS wall-clock time elapses, then kills ExifTool.
        Returns dict with 'track' (list of {lat, lon, ts}) for matching,
        plus first-point lat/lon/ts for display.
        """
        name = os.path.basename(mp4_path)
        exif = _find_exiftool()
        if not exif:
            return None

        try:
            proc = subprocess.Popen(
                [exif, "-ee", "-n", "-p",
                 "$GPSLatitude,$GPSLongitude,$GPSDateTime",
                 mp4_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except Exception:
            return None

        points: list[dict] = []
        last_ts = -999.0
        deadline = _time.perf_counter() + _SNIPPET_SECS

        try:
            for line in proc.stdout:
                if _time.perf_counter() > deadline or len(points) >= _SNIPPET_PTS:
                    break
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                except ValueError:
                    continue
                if not _valid(lat, lon):
                    continue
                ts = _parse_ts(parts[2])
                if ts is None:
                    continue
                # thin: skip if too close in time to previous kept point
                if ts - last_ts < _SNIPPET_THIN:
                    continue
                last_ts = ts
                points.append({"lat": lat, "lon": lon, "ts": ts})
        finally:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait()

        if not points:
            return None

        # Filter outliers from snippet too
        points = _filter_gps_outliers(points)
        if not points:
            return None

        return {
            "lat": points[0]["lat"],
            "lon": points[0]["lon"],
            "ts": points[0]["ts"],
            "track": points,
            "name": name,
            "path": mp4_path,
        }

    print(f"\n  Phase 1: Quick GPS scan ({len(mp4_paths)} files) ...")
    t0 = _time.perf_counter()
    quick: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(mp4_paths), os.cpu_count() or 4)) as pool:
        futs = {pool.submit(_quick_gps, p): p for p in mp4_paths}
        for f in as_completed(futs):
            res = f.result()
            if res:
                quick.append(res)
                print(f"    ✓ {res['name']}  ({res['lat']:.5f}, {res['lon']:.5f})  "
                      f"{len(res['track'])} pts")
            else:
                print(f"    ✗ {os.path.basename(futs[f])} — no GPS found")
    dt1 = _time.perf_counter() - t0
    print(f"  Phase 1 done in {dt1:.1f}s — {len(quick)}/{len(mp4_paths)} clips with GPS\n")

    if len(quick) < 1:
        raise RuntimeError("No clips with GPS found.")

    # --- Parse CSVs ---
    csv_data: list[dict] = []
    for csv_path in (csv_paths or []):
        print(f"  Parsing CSV: {os.path.basename(csv_path)}")
        vak = parse_vakaros_csv(csv_path)
        if not vak["track"]:
            continue
        epochs = [p["ts"] for p in vak["track"]]
        csv_data.append({
            "path": csv_path,
            "name": os.path.basename(csv_path),
            "vak": vak,
            "epochs": epochs,
            "track": vak["track"],
        })

    # --- Group by camera ID ---
    def camera_id(name: str) -> str:
        stem = os.path.splitext(name)[0]
        return stem[-4:] if len(stem) >= 4 else stem

    cam_groups: dict[str, list[dict]] = {}
    for q in quick:
        cid = camera_id(q["name"])
        cam_groups.setdefault(cid, []).append(q)

    print(f"  Detected {len(cam_groups)} camera group(s):")
    for cid, grp in cam_groups.items():
        names = ", ".join(q["name"] for q in grp)
        print(f"    Camera '{cid}': {len(grp)} clip(s) — {names}")

    # --- Match cameras to CSVs using GPS snippets + nearest-in-time CSV point ---
    def cam_dist_to_csv(cam_grp: list[dict], csv_entry: dict) -> float:
        """Compute mean spatial distance between all snippet points in a
        camera group and the closest-in-time CSV points."""
        csv_epochs = csv_entry["epochs"]
        csv_track = csv_entry["track"]
        n = len(csv_epochs)
        total_d = 0.0
        count = 0
        for q in cam_grp:
            for pt in q.get("track", []):
                ts = pt["ts"]
                idx = bisect.bisect_left(csv_epochs, ts)
                best_i = min(max(idx, 0), n - 1)
                if 0 < idx < n:
                    if abs(csv_epochs[idx - 1] - ts) < abs(csv_epochs[idx] - ts):
                        best_i = idx - 1
                elif idx >= n:
                    best_i = n - 1
                if abs(csv_epochs[best_i] - ts) > 300:
                    continue
                cp = csv_track[best_i]
                dlat = (pt["lat"] - cp["lat"]) * 111320
                dlon = (pt["lon"] - cp["lon"]) * 111320 * math.cos(math.radians(pt["lat"]))
                total_d += math.sqrt(dlat * dlat + dlon * dlon)
                count += 1
        return (total_d / count) if count > 0 else float("inf")

    cam_ids = list(cam_groups.keys())
    n_cams = len(cam_ids)

    # --- Single-athlete mode: 1 camera group or only 1 CSV ---
    single_athlete = (n_cams < 2)
    if n_cams < 2 and len(csv_data) >= 2:
        # Only 1 camera but 2 CSVs — match to the closest CSV
        dists = [cam_dist_to_csv(cam_groups[cam_ids[0]], ce) for ce in csv_data]
        best_csv = min(range(len(dists)), key=lambda i: dists[i])
        print(f"  Single camera '{cam_ids[0]}' — matched to {csv_data[best_csv]['name']} "
              f"(dist {dists[best_csv]:.0f}m)")
        g1_ids = cam_ids[:]
        g2_ids = []
        vak1_data = csv_data[best_csv]["vak"]
        vak2_data = {"track": []}
    elif n_cams < 2:
        # 1 camera, 0-1 CSVs → single athlete
        print(f"  Single camera group '{cam_ids[0]}' → single-athlete mode")
        g1_ids = cam_ids[:]
        g2_ids = []
        vak1_data = csv_data[0]["vak"] if csv_data else {"track": []}
        vak2_data = {"track": []}
    elif len(csv_data) >= 2:
        n_total_pts = sum(len(pt) for grp in cam_groups.values() for q in grp for pt in [q.get("track", [])])
        print(f"\n  Matching cameras to CSVs using {n_total_pts} GPS snippet points ...\n")
        cam_dists: dict[str, list[float]] = {}
        for cid in cam_ids:
            n_pts = sum(len(q.get("track", [])) for q in cam_groups[cid])
            dists = [cam_dist_to_csv(cam_groups[cid], ce) for ce in csv_data]
            cam_dists[cid] = dists
            dist_str = "  ".join(f"{csv_data[i]['name']}={dists[i]:.0f}m" for i in range(len(csv_data)))
            print(f"    Camera '{cid}' ({n_pts} pts): {dist_str}")

        # Optimal partition
        best_cost, best_mask = float("inf"), 0
        for mask in range(1, (1 << n_cams) - 1):
            cost = sum(
                cam_dists[cam_ids[bit]][0 if (mask >> bit) & 1 else 1] * len(cam_groups[cam_ids[bit]])
                for bit in range(n_cams)
            )
            if cost < best_cost:
                best_cost, best_mask = cost, mask

        g1_ids = [cam_ids[b] for b in range(n_cams) if (best_mask >> b) & 1]
        g2_ids = [cam_ids[b] for b in range(n_cams) if not ((best_mask >> b) & 1)]
        vak1_data = csv_data[0]["vak"]
        vak2_data = csv_data[1]["vak"]

        print(f"\n  Best assignment:")
        for cid in g1_ids:
            print(f"    Camera '{cid}' → RED  ({csv_data[0]['name']})")
        for cid in g2_ids:
            print(f"    Camera '{cid}' → BLUE ({csv_data[1]['name']})")

    elif len(csv_data) == 1:
        scored = sorted(
            ((cam_dist_to_csv(cam_groups[cid], csv_data[0]), cid) for cid in cam_ids),
            key=lambda x: x[0],
        )
        mid = len(scored) // 2 or 1
        g1_ids = [cid for _, cid in scored[:mid]]
        g2_ids = [cid for _, cid in scored[mid:]]
        vak1_data = csv_data[0]["vak"]
        vak2_data = {"track": []}
    else:
        # No CSVs — split by mean GPS position from snippet tracks
        cam_cen = {}
        for cid, grp in cam_groups.items():
            all_pts = [pt for q in grp for pt in q.get("track", [])]
            if not all_pts:
                all_pts = [{"lat": q["lat"], "lon": q["lon"]} for q in grp]
            cam_cen[cid] = (
                sum(p["lat"] for p in all_pts) / len(all_pts),
                sum(p["lon"] for p in all_pts) / len(all_pts),
            )
        max_d, si, sj = -1.0, 0, 1
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                d = (cam_cen[cam_ids[i]][0] - cam_cen[cam_ids[j]][0]) ** 2 + \
                    (cam_cen[cam_ids[i]][1] - cam_cen[cam_ids[j]][1]) ** 2
                if d > max_d:
                    max_d, si, sj = d, i, j
        s1, s2 = cam_cen[cam_ids[si]], cam_cen[cam_ids[sj]]
        g1_ids, g2_ids = [], []
        for cid in cam_ids:
            cc = cam_cen[cid]
            if (cc[0]-s1[0])**2 + (cc[1]-s1[1])**2 <= (cc[0]-s2[0])**2 + (cc[1]-s2[1])**2:
                g1_ids.append(cid)
            else:
                g2_ids.append(cid)
        vak1_data = {"track": []}
        vak2_data = {"track": []}

    if not g1_ids and not g2_ids:
        raise RuntimeError("No clips with GPS could be assigned.")
    if not g1_ids:
        # Swap so ds1 always has content
        g1_ids, g2_ids = g2_ids, g1_ids
        vak1_data, vak2_data = vak2_data, vak1_data

    # Build path lists per group
    g1_paths = [q["path"] for cid in g1_ids for q in cam_groups[cid]]
    g2_paths = [q["path"] for cid in g2_ids for q in cam_groups[cid]]

    print(f"\n  RED  clips: {', '.join(os.path.basename(p) for p in g1_paths)}")
    if g2_paths:
        print(f"  BLUE clips: {', '.join(os.path.basename(p) for p in g2_paths)}")
    else:
        print("  BLUE clips: (none — single-athlete mode)")

    # ---- Phase 2: Full GPS extraction (parallel, only now) ----
    all_mp4 = g1_paths + g2_paths
    total = len(all_mp4)
    print(f"\n  Phase 2: Full GPS extraction from {total} MP4s in parallel ...")
    t2 = _time.perf_counter()
    results: dict[str, tuple] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=min(total, os.cpu_count() or 4)) as pool:
        futures = {pool.submit(_extract_one, p): p for p in all_mp4}
        for f in as_completed(futures):
            path, name, track, dt = f.result()
            done += 1
            status = f"✓ {len(track)} pts" if len(track) >= 2 else f"⚠ {len(track)} pts"
            print(f"  [{done}/{total}] {name}  {status}  [{dt:.1f}s]")
            results[path] = (name, track, dt)

    dt2 = _time.perf_counter() - t2
    print(f"  Phase 2 done in {dt2:.1f}s\n")

    # Assemble datasets
    def assemble(paths, vak_data, label):
        segments = []
        for mp4 in paths:
            name, track, dt = results[mp4]
            if len(track) < 2:
                continue
            raw_n = len(track)
            track = thin_gps_track(track)
            epochs = [p["ts"] for p in track]
            segments.append({
                "name": name, "gps": track,
                "epochMin": min(epochs), "epochMax": max(epochs),
            })
            print(f"  {label}  {name}: {raw_n} → {len(track)} pts  "
                  f"({datetime.fromtimestamp(min(epochs), tz=timezone.utc).strftime('%H:%M:%S')} – "
                  f"{datetime.fromtimestamp(max(epochs), tz=timezone.utc).strftime('%H:%M:%S')} UTC)")
        segments.sort(key=lambda s: s["epochMin"])
        return {"segments": segments, "vakaros": vak_data}

    print("=== Athlete 1 (RED) ===")
    ds1 = assemble(g1_paths, vak1_data, "RED ")
    if g2_paths:
        print("\n=== Athlete 2 (BLUE) ===")
        ds2 = assemble(g2_paths, vak2_data, "BLUE")
    else:
        print("\n=== Single-athlete mode — no Athlete 2 ===")
        ds2 = {"segments": [], "vakaros": vak2_data}
    return ds1, ds2


# ------------------------------------------------------------------
# HTML template  – multi-video per athlete + map
# ------------------------------------------------------------------
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Dual Athlete Multi-Video + Vakaros Viewer</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    *,*::before,*::after{box-sizing:border-box;}
    html,body{height:100%;margin:0;background:#0b0f14;color:#e6edf3;
      font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;}

    .wrap{height:100%;display:grid;grid-template-columns:1fr 1fr;
      grid-template-rows:1fr auto auto;gap:10px;padding:10px;}

    .panel{background:#111826;border:1px solid #223047;border-radius:10px;}
    .nopad{padding:0;overflow:hidden;}

    .left{display:flex;flex-direction:column;gap:6px;padding:10px;}

    .bottom-bar{grid-column:1/-1;background:#111826;border:1px solid #223047;
      border-radius:10px;padding:10px 14px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;}

    .file-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
    .file-row label{font-weight:bold;white-space:nowrap;}
    .file-row input[type="file"]{min-width:0;}

    .fstat{font-size:11px;line-height:1.3;}
    .fstat .ok{color:#66bb6a;} .fstat .miss{color:#ff7043;}

    .red-label{color:#ef5350;} .blue-label{color:#42a5f5;}

    .ctrl-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
    input[type="range"]{flex:1;min-width:80px;}

    button{background:#0b1220;color:#e6edf3;border:1px solid #223047;
      border-radius:10px;padding:8px 14px;cursor:pointer;white-space:nowrap;}
    button:hover{border-color:#35507a;}

    .stats-row{display:flex;gap:12px;flex-wrap:wrap;}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}
    .small{font-size:12px;}
    code{background:#0b1220;padding:2px 6px;border-radius:6px;}

    .vid-stack{flex:1;min-height:0;display:flex;flex-direction:column;gap:8px;}
    .vid-box{flex:1;min-height:250px;position:relative;border-radius:10px;overflow:hidden;display:flex;}
    .red-frame{border:3px solid #e53935;} .blue-frame{border:3px solid #1e88e5;}
    .vid-box video{position:absolute;top:0;left:0;width:100%;height:100%;
      object-fit:contain;background:#000;display:none;}

    .blk{position:absolute;top:0;left:0;width:100%;height:100%;background:#000;
      display:flex;align-items:center;justify-content:center;
      color:#555;font-size:13px;z-index:5;pointer-events:none;}

    #mapW{position:relative;width:100%;height:100%;}
    #map{width:100%;height:100%;}

    .arrow-icon{background:none !important;border:none !important;}
    .arrow-svg{transform-origin:center center;}

    .overlay{position:absolute;top:10px;right:10px;z-index:1000;
      background:rgba(10,16,26,0.88);border:1px solid rgba(80,120,170,0.35);
      border-radius:10px;padding:10px 14px;font-size:12px;line-height:1.45;
      backdrop-filter:blur(4px);max-width:70%;pointer-events:none;}
    .overlay hr{border:none;border-top:1px solid #223047;margin:6px 0;}

    .sog-bar{grid-column:1/-1;background:#111826;border:1px solid #223047;
      border-radius:10px;padding:6px 12px;display:flex;flex-direction:column;gap:2px;}
    .sog-bar .sog-label{font-size:10px;opacity:0.6;}
    #sogCanvas{width:100%;height:54px;border-radius:4px;cursor:crosshair;display:block;}
  </style>
</head>
<body>
  <div class="wrap">
    <!-- ===== LEFT PANEL (VIDEOS + CONTROLS) ===== -->
    <div class="left panel">
      <!-- transport controls -->
      <div class="ctrl-row">
        <button id="bk10">-10 s</button>
        <button id="ppBtn">&#9654; Play</button>
        <button id="fw10">+10 s</button>
        <input id="scrub" type="range" min="0" max="0" step="0.1" value="0">
        <span class="mono small" id="tDisp">0:00 / 0:00</span>
      </div>

      <div class="stats-row mono small">
        <span style="color:#ef5350;">Clips&#x2081;: <code id="nc1">0</code></span>
        <span style="color:#ef9a9a;">Vak&#x2081;: <code id="nk1">0</code></span>
        <span style="color:#42a5f5;">Clips&#x2082;: <code id="nc2">0</code></span>
        <span style="color:#90caf9;">Vak&#x2082;: <code id="nk2">0</code></span>
      </div>

      <!-- dual video panes -->
      <div class="vid-stack">
        <div class="vid-box red-frame" id="vc1">
          <div class="blk" id="blk1"><span>No data at this time</span></div>
        </div>
        <div class="vid-box blue-frame" id="vc2">
          <div class="blk" id="blk2"><span>No data at this time</span></div>
        </div>
      </div>
    </div>

    <!-- ===== RIGHT PANEL (MAP) ===== -->
    <div class="panel nopad">
      <div id="mapW">
        <div id="map"></div>
        <div class="overlay mono">
          <div>UTC: <code id="oUTC">-</code></div>
          <hr>
          <div style="color:#ef5350;font-weight:bold;">Athlete 1 (Red)</div>
          <div>Clip: <code id="o1clip">-</code></div>
          <div>GPS: <code id="o1gps">-</code></div>
          <div>Vakaros: <code id="o1vak">-</code></div>
          <hr>
          <div style="color:#42a5f5;font-weight:bold;">Athlete 2 (Blue)</div>
          <div>Clip: <code id="o2clip">-</code></div>
          <div>GPS: <code id="o2gps">-</code></div>
          <div>Vakaros: <code id="o2vak">-</code></div>
          <hr>
          <div class="small" style="opacity:0.7;">
            Solid lines = Vakaros tracks.<br>
            Dashed lines = GoPro GPS tracks.
          </div>
        </div>
      </div>
    </div>

    <!-- ===== BOTTOM: SOG timeseries ===== -->
    <div class="sog-bar" id="sogBar" style="display:none;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span class="sog-label mono" style="flex:1;">SOG (m/s) — <span style="color:#ef5350;">Red</span> / <span style="color:#42a5f5;">Blue</span>  — click or drag to scrub</span>
        <button id="sogZoomIn" title="Zoom in" style="padding:2px 8px;">+</button>
        <button id="sogZoomOut" title="Zoom out" style="padding:2px 8px;">&minus;</button>
        <button id="sogZoomReset" title="Reset zoom" style="padding:2px 8px;font-size:10px;">1:1</button>
      </div>
      <canvas id="sogCanvas" height="54"></canvas>
    </div>

    <!-- ===== BOTTOM: File imports ===== -->
    <div class="bottom-bar">
      <div class="file-row">
        <label>Import all videos:</label>
        <input id="vfAll" type="file" accept="video/mp4" multiple>
        <label style="margin-left:18px;cursor:pointer;user-select:none;">
          <input type="checkbox" id="hideGoPro"> Hide GoPro GPS tracks
        </label>
        <label style="margin-left:18px;cursor:pointer;user-select:none;">
          <input type="checkbox" id="fadeTracks"> Fade tracks (±60 s window)
        </label>
      </div>
      <div class="fstat mono" id="fsAll" style="flex-basis:100%;"></div>
    </div>
  </div>

<script>
// ======================== Embedded data ========================
const ds1 = __DS1_JSON__;   // {segments:[{name,gps,epochMin,epochMax},...], vakaros:{track:[...]}}
const ds2 = __DS2_JSON__;

const vakTrack1 = (ds1.vakaros && ds1.vakaros.track) || [];
const vakTrack2 = (ds2.vakaros && ds2.vakaros.track) || [];
const vakEpochs1 = vakTrack1.map(p => p.ts);
const vakEpochs2 = vakTrack2.map(p => p.ts);

// Stats
const singleMode = ds2.segments.length === 0;
document.getElementById('nc1').textContent = ds1.segments.length;
document.getElementById('nk1').textContent = vakTrack1.length;
document.getElementById('nc2').textContent = ds2.segments.length;
document.getElementById('nk2').textContent = vakTrack2.length;

// Hide blue UI elements in single-athlete mode
if (singleMode) {
  document.getElementById('vc2').style.display = 'none';
  // Grow red video box
  document.getElementById('vc1').style.flex = '1';
  // Hide blue stats
  document.querySelectorAll('.stats-row span').forEach(el => {
    if (el.style.color === 'rgb(66, 165, 245)' || el.style.color === 'rgb(144, 202, 249)')
      el.style.display = 'none';
  });
  // Hide blue overlay section
  const overlayHrs = document.querySelectorAll('.overlay hr');
  const o2header = document.querySelector('.overlay div[style*="42a5f5"]');
  if (o2header) {
    // Hide from second <hr> onward (blue section)
    let hide = false;
    o2header.parentElement.childNodes.forEach(n => {
      if (n === overlayHrs[1]) hide = true;
      if (hide && n.style !== undefined) n.style.display = 'none';
      if (hide && n.nodeType === 1) n.style.display = 'none';
    });
  }
}

// Pre-compute per-segment lookup arrays
function initSegments(ds) {
  ds.segments.forEach(seg => {
    seg.epochs = seg.gps.map(p => p.ts);
    seg.vTimes = seg.gps.map(p => p.t);
    seg.el = null;          // <video> element (created below)
    seg.loaded = false;
    seg.pSeek = 0;          // programmatic seek counter
  });
}
initSegments(ds1);
initSegments(ds2);

// Collect all GPS points per dataset for polylines
function allGps(ds) {
  const pts = [];
  ds.segments.forEach(s => s.gps.forEach(p => pts.push(p)));
  pts.sort((a,b) => a.ts - b.ts);
  return pts;
}
const allGps1 = allGps(ds1);
const allGps2 = allGps(ds2);

// Global epoch range (union of all tracks)
const allEp = [].concat(
  allGps1.map(p=>p.ts), allGps2.map(p=>p.ts),
  vakEpochs1, vakEpochs2
);
const epochMin = allEp.length ? allEp.reduce((a,b)=>Math.min(a,b),Infinity)  : 0;
const epochMax = allEp.length ? allEp.reduce((a,b)=>Math.max(a,b),-Infinity) : 1;
const totalDur = epochMax - epochMin;

// ======================== Helpers ========================
function findIdxLE(arr, x) {
  let lo = 0, hi = arr.length - 1;
  while (lo <= hi) { const m = (lo+hi)>>1; if (arr[m]<=x) lo=m+1; else hi=m-1; }
  return Math.max(0, Math.min(arr.length-1, hi));
}

function epochToVTime(epochs, vTimes, ep) {
  if (!epochs.length) return NaN;
  if (ep <= epochs[0]) return vTimes[0];
  if (ep >= epochs[epochs.length-1]) return vTimes[vTimes.length-1];
  const i = findIdxLE(epochs, ep);
  if (i >= epochs.length-1) return vTimes[i];
  const f = (ep - epochs[i])/(epochs[i+1]-epochs[i]);
  return vTimes[i] + f*(vTimes[i+1]-vTimes[i]);
}

function vTimeToEpoch(epochs, vTimes, vt) {
  if (!vTimes.length) return NaN;
  if (vt <= vTimes[0]) return epochs[0];
  if (vt >= vTimes[vTimes.length-1]) return epochs[epochs.length-1];
  const i = findIdxLE(vTimes, vt);
  if (i >= vTimes.length-1) return epochs[i];
  const f = (vt - vTimes[i])/(vTimes[i+1]-vTimes[i]);
  return epochs[i] + f*(epochs[i+1]-epochs[i]);
}

function inRange(epochs, ep) {
  return epochs.length >= 2 && ep >= epochs[0] && ep <= epochs[epochs.length-1];
}

function interpPos(track, epochs, ep) {
  if (!track.length || !inRange(epochs, ep)) return null;
  const i = findIdxLE(epochs, ep);
  if (i >= track.length-1) return {lat: track[i].lat, lon: track[i].lon};
  const f = (ep-epochs[i])/(epochs[i+1]-epochs[i]);
  return {
    lat: track[i].lat + f*(track[i+1].lat - track[i].lat),
    lon: track[i].lon + f*(track[i+1].lon - track[i].lon)
  };
}

// Interpolate across all GPS in a dataset (all segments combined)
function interpDsGps(allPts, ep) {
  if (!allPts.length) return null;
  const eps = allPts.map(p=>p.ts);
  return interpPos(allPts, eps, ep);
}

function fmtElapsed(s) {
  s = Math.max(0, Math.floor(s));
  const m = Math.floor(s/60), h = Math.floor(m/60);
  if (h) return h+':'+String(m%60).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
  return m+':'+String(s%60).padStart(2,'0');
}
function fmtEpochUTC(ts) {
  if (!isFinite(ts)) return '-';
  return new Date(ts*1000).toISOString().replace('T',' ').replace('.000Z','Z');
}
function fmtLL(pos) {
  return pos ? pos.lat.toFixed(6)+', '+pos.lon.toFixed(6) : '-';
}

// ======================== Map ========================
const map = L.map('map', {zoomControl:true});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  maxZoom:19, attribution:'&copy; OpenStreetMap contributors'
}).addTo(map);

// Polylines: Vakaros = solid bright, GoPro = dashed lighter
const vakLine1 = vakTrack1.length
  ? L.polyline(vakTrack1.map(p=>[p.lat,p.lon]),{color:'#e53935',weight:3,opacity:0.85}).addTo(map) : null;
const mp4Line1 = allGps1.length
  ? L.polyline(allGps1.map(p=>[p.lat,p.lon]),{color:'#ef9a9a',weight:3,opacity:0.55,dashArray:'6 4'}).addTo(map) : null;
const vakLine2 = vakTrack2.length
  ? L.polyline(vakTrack2.map(p=>[p.lat,p.lon]),{color:'#1e88e5',weight:3,opacity:0.85}).addTo(map) : null;
const mp4Line2 = allGps2.length
  ? L.polyline(allGps2.map(p=>[p.lat,p.lon]),{color:'#90caf9',weight:3,opacity:0.55,dashArray:'6 4'}).addTo(map) : null;

// ======================== Fade-track support ========================
// Grey background lines (full track, shown only when fade mode is on)
const greyVak1 = vakTrack1.length
  ? L.polyline(vakTrack1.map(p=>[p.lat,p.lon]),{color:'#888',weight:2,opacity:0.25}).addTo(map) : null;
const greyMp41 = allGps1.length
  ? L.polyline(allGps1.map(p=>[p.lat,p.lon]),{color:'#888',weight:2,opacity:0.25,dashArray:'4 3'}).addTo(map) : null;
const greyVak2 = vakTrack2.length
  ? L.polyline(vakTrack2.map(p=>[p.lat,p.lon]),{color:'#888',weight:2,opacity:0.25}).addTo(map) : null;
const greyMp42 = allGps2.length
  ? L.polyline(allGps2.map(p=>[p.lat,p.lon]),{color:'#888',weight:2,opacity:0.25,dashArray:'4 3'}).addTo(map) : null;
// Hide grey lines initially
[greyVak1,greyMp41,greyVak2,greyMp42].forEach(l=>{if(l) map.removeLayer(l);});

// Fade polylines: 5-segment smooth fade per colored track
// We create them once (empty) and update coordinates each frame
const FADE_WIN = 60;   // seconds each side
const FADE_OUTER = 15; // outermost fade zone (ultra-faint)
const FADE_INNER = 15; // inner fade zone (faint)
// Total structure: [ultra-fade-in: 15s][fade-in: 15s][solid: 60s][fade-out: 15s][ultra-fade-out: 15s]
function makeFadeSet(color, weight, dashArr) {
  const opts = {weight, interactive:false};
  if (dashArr) opts.dashArray = dashArr;
  return {
    ultraFadeIn:  L.polyline([],Object.assign({},opts,{color,opacity:0.15})),
    fadeIn:       L.polyline([],Object.assign({},opts,{color,opacity:0.35})),
    solid:        L.polyline([],Object.assign({},opts,{color,opacity:0.85})),
    fadeOut:      L.polyline([],Object.assign({},opts,{color,opacity:0.35})),
    ultraFadeOut: L.polyline([],Object.assign({},opts,{color,opacity:0.15})),
  };
}
const fadeVak1 = makeFadeSet('#e53935', 3);
const fadeMp41 = makeFadeSet('#ef9a9a', 3, '6 4');
const fadeVak2 = makeFadeSet('#1e88e5', 3);
const fadeMp42 = makeFadeSet('#90caf9', 3, '6 4');
const allFadeSets = [fadeVak1, fadeMp41, fadeVak2, fadeMp42];

let fadeMode = false;

// Fast time-window slicer: returns 5-segment coordinate arrays for smooth fade
// Uses binary search on pre-sorted epochs
function sliceTrack(pts, epochs, ep) {
  const tLo = ep - FADE_WIN, tHi = ep + FADE_WIN;
  const tUltraFadeEnd = ep - FADE_WIN + FADE_OUTER;
  const tFadeInEnd = ep - FADE_WIN + FADE_OUTER + FADE_INNER;
  const tFadeOutStart = ep + FADE_WIN - FADE_OUTER - FADE_INNER;
  const tUltraFadeStart = ep + FADE_WIN - FADE_OUTER;
  // Binary search for window bounds
  let lo = 0, hi = epochs.length;
  while (lo < hi) { const m = (lo+hi)>>1; epochs[m] < tLo ? lo=m+1 : hi=m; }
  const iStart = lo;
  lo = iStart; hi = epochs.length;
  while (lo < hi) { const m = (lo+hi)>>1; epochs[m] <= tHi ? lo=m+1 : hi=m; }
  const iEnd = lo; // exclusive
  if (iStart >= iEnd) return {ultraFadeIn:[],fadeIn:[],solid:[],fadeOut:[],ultraFadeOut:[]};

  const ultraFadeInPts = [], fadeInPts = [], solidPts = [], fadeOutPts = [], ultraFadeOutPts = [];
  for (let i = iStart; i < iEnd; i++) {
    const t = epochs[i], ll = [pts[i].lat, pts[i].lon];
    if (t < tUltraFadeEnd) {
      ultraFadeInPts.push(ll);
    } else if (t < tFadeInEnd) {
      fadeInPts.push(ll);
    } else if (t <= tFadeOutStart) {
      solidPts.push(ll);
    } else if (t <= tUltraFadeStart) {
      fadeOutPts.push(ll);
    } else {
      ultraFadeOutPts.push(ll);
    }
  }
  // Overlap: share boundary points for visual continuity
  if (ultraFadeInPts.length && fadeInPts.length) fadeInPts.unshift(ultraFadeInPts[ultraFadeInPts.length-1]);
  if (fadeInPts.length && solidPts.length) solidPts.unshift(fadeInPts[fadeInPts.length-1]);
  if (solidPts.length && fadeOutPts.length) fadeOutPts.unshift(solidPts[solidPts.length-1]);
  if (fadeOutPts.length && ultraFadeOutPts.length) ultraFadeOutPts.unshift(fadeOutPts[fadeOutPts.length-1]);
  return {ultraFadeIn: ultraFadeInPts, fadeIn: fadeInPts, solid: solidPts, fadeOut: fadeOutPts, ultraFadeOut: ultraFadeOutPts};
}

// Pre-compute epoch arrays for fast binary search
const vakEpochArr1 = vakTrack1.map(p=>p.ts);
const mp4EpochArr1 = allGps1.map(p=>p.ts);
const vakEpochArr2 = vakTrack2.map(p=>p.ts);
const mp4EpochArr2 = allGps2.map(p=>p.ts);

function updateFadeTracks(ep) {
  const pairs = [
    [fadeVak1, vakTrack1, vakEpochArr1],
    [fadeMp41, allGps1,   mp4EpochArr1],
    [fadeVak2, vakTrack2, vakEpochArr2],
    [fadeMp42, allGps2,   mp4EpochArr2],
  ];
  for (const [fs, pts, epochs] of pairs) {
    if (!pts.length) continue;
    const s = sliceTrack(pts, epochs, ep);
    fs.ultraFadeIn.setLatLngs(s.ultraFadeIn);
    fs.fadeIn.setLatLngs(s.fadeIn);
    fs.solid.setLatLngs(s.solid);
    fs.fadeOut.setLatLngs(s.fadeOut);
    fs.ultraFadeOut.setLatLngs(s.ultraFadeOut);
  }
}

function enableFade() {
  // Hide normal colored lines
  [vakLine1,mp4Line1,vakLine2,mp4Line2].forEach(l=>{if(l) map.removeLayer(l);});
  // Show grey bg lines
  [greyVak1,greyMp41,greyVak2,greyMp42].forEach(l=>{if(l) map.addLayer(l);});
  // Add fade polylines (5 segments per track)
  allFadeSets.forEach(fs=>{
    map.addLayer(fs.ultraFadeIn); map.addLayer(fs.fadeIn); map.addLayer(fs.solid);
    map.addLayer(fs.fadeOut); map.addLayer(fs.ultraFadeOut);
  });
  // Respect hide-gopro state
  if (goProGpsHidden) {
    [greyMp41,greyMp42].forEach(l=>{if(l) map.removeLayer(l);});
    [fadeMp41,fadeMp42].forEach(fs=>{
      map.removeLayer(fs.ultraFadeIn); map.removeLayer(fs.fadeIn); map.removeLayer(fs.solid);
      map.removeLayer(fs.fadeOut); map.removeLayer(fs.ultraFadeOut);
    });
  }
}
function disableFade() {
  // Remove fade polylines (5 segments per track)
  allFadeSets.forEach(fs=>{
    map.removeLayer(fs.ultraFadeIn); map.removeLayer(fs.fadeIn); map.removeLayer(fs.solid);
    map.removeLayer(fs.fadeOut); map.removeLayer(fs.ultraFadeOut);
  });
  // Hide grey bg lines
  [greyVak1,greyMp41,greyVak2,greyMp42].forEach(l=>{if(l) map.removeLayer(l);});
  // Restore normal colored lines
  [vakLine1,mp4Line1,vakLine2,mp4Line2].forEach(l=>{if(l) map.addLayer(l);});
  // Respect hide-gopro state
  if (goProGpsHidden) {
    [mp4Line1,mp4Line2].forEach(l=>{if(l) map.removeLayer(l);});
  }
}

// Arrow icon factory — the SVG has a class we rotate, not the marker container
function makeArrowIcon(fill, stroke) {
  const svg = `<svg class="arrow-svg" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="-11 -11 22 22" style="display:block;">
    <polygon points="0,-10 6,7 0,3 -6,7" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/></svg>`;
  return L.divIcon({
    html: svg, className: 'arrow-icon', iconSize:[22,22], iconAnchor:[11,11]
  });
}

// Vakaros = arrow markers
const vakArrow1 = vakTrack1.length
  ? L.marker([vakTrack1[0].lat,vakTrack1[0].lon],{icon:makeArrowIcon('#e53935','#b71c1c'),zIndexOffset:400}).addTo(map) : null;
const vakArrow2 = vakTrack2.length
  ? L.marker([vakTrack2[0].lat,vakTrack2[0].lon],{icon:makeArrowIcon('#1e88e5','#0d47a1'),zIndexOffset:400}).addTo(map) : null;

// GoPro = round circle dots
const mp4Dot1 = allGps1.length
  ? L.circleMarker([allGps1[0].lat,allGps1[0].lon],{radius:6,color:'#e53935',fillColor:'#ef9a9a',fillOpacity:0.9,weight:2}).addTo(map) : null;
const mp4Dot2 = allGps2.length
  ? L.circleMarker([allGps2[0].lat,allGps2[0].lon],{radius:6,color:'#1e88e5',fillColor:'#90caf9',fillOpacity:0.9,weight:2}).addTo(map) : null;

// Heading computation from consecutive points
function computeHeading(pts, ep) {
  if (!pts || pts.length < 2) return null;
  let lo=0, hi=pts.length-1;
  while(lo<hi){const m=(lo+hi)>>1; pts[m].ts<ep?lo=m+1:hi=m;}
  const i = Math.max(0, Math.min(lo, pts.length-2));
  const a=pts[i], b=pts[Math.min(i+1,pts.length-1)];
  const dlat=b.lat-a.lat, dlon=b.lon-a.lon;
  if(Math.abs(dlat)<1e-8 && Math.abs(dlon)<1e-8) return null;
  return (Math.atan2(dlon,dlat)*180/Math.PI + 360) % 360;
}
function setArrowHeading(marker, heading) {
  if (!marker) return;
  // Rotate only the inner SVG, never touch the marker container's transform
  const el = marker.getElement && marker.getElement();
  if (!el) return;
  const svg = el.querySelector('.arrow-svg');
  if (svg) svg.style.transform = `rotate(${heading}deg)`;
}

// Fit bounds
const bnds = [mp4Line1,vakLine1,mp4Line2,vakLine2].filter(Boolean).map(l=>l.getBounds());
if (bnds.length) {
  let b = bnds[0]; for (let i=1;i<bnds.length;i++) b=b.extend(bnds[i]);
  map.fitBounds(b.pad(0.15));
} else { map.setView([0,0],2); }

// ======================== Clickable tracks ========================
function findNearestEpoch(pts, latlng) {
  let best = -1, bestD = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const d = Math.pow(pts[i].lat - latlng.lat, 2) + Math.pow(pts[i].lon - latlng.lng, 2);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best >= 0 ? pts[best].ts : null;
}
function onTrackClick(pts, e) {
  const ep = findNearestEpoch(pts, e.latlng);
  if (ep !== null) {
    if (playing) pauseAll();
    masterEpoch = Math.max(epochMin, Math.min(epochMax, ep));
    syncUI();
  }
}
if (mp4Line1) mp4Line1.on('click', e => onTrackClick(allGps1, e));
if (vakLine1) vakLine1.on('click', e => onTrackClick(vakTrack1, e));
if (mp4Line2) mp4Line2.on('click', e => onTrackClick(allGps2, e));
if (vakLine2) vakLine2.on('click', e => onTrackClick(vakTrack2, e));

// ======================== Create <video> elements ========================
function createVidEls(ds, container, blk) {
  ds.segments.forEach((seg, i) => {
    const v = document.createElement('video');
    v.setAttribute('playsinline','');
    v.preload = 'auto';
    container.insertBefore(v, blk);
    seg.el = v;

    // User-initiated seek → update master clock
    v.addEventListener('seeked', () => {
      if (seg.pSeek > 0) { seg.pSeek--; return; }
      if (v.readyState >= 1) {
        masterEpoch = vTimeToEpoch(seg.epochs, seg.vTimes, v.currentTime);
        masterEpoch = Math.max(epochMin, Math.min(epochMax, masterEpoch));
        syncUI();
      }
    });
  });
}

const vc1 = document.getElementById('vc1');
const vc2 = document.getElementById('vc2');
const blk1 = document.getElementById('blk1');
const blk2 = document.getElementById('blk2');
createVidEls(ds1, vc1, blk1);
createVidEls(ds2, vc2, blk2);

// ======================== File input (unified) ========================
function loadAllFiles(ev) {
  const files = Array.from(ev.target.files || []);
  const nameMap = {};
  files.forEach(f => { nameMap[f.name.toLowerCase()] = f; });
  [ds1, ds2].forEach(ds => {
    ds.segments.forEach(seg => {
      const key = seg.name.toLowerCase();
      if (nameMap[key]) {
        seg.el.src = URL.createObjectURL(nameMap[key]);
        seg.el.load();
        seg.loaded = true;
      }
    });
  });
  updateAllFileStatus();
}

function updateAllFileStatus() {
  const parts = [];
  function add(ds, color, label) {
    ds.segments.forEach(seg => {
      const tag = seg.loaded ? 'ok' : 'miss';
      const icon = seg.loaded ? '&#10003;' : '&#10007;';
      parts.push('<span class="'+tag+'">'+icon+' '+seg.name+' <span style="color:'+color+';">('+label+')</span></span>');
    });
  }
  add(ds1, '#ef5350', 'Red');
  add(ds2, '#42a5f5', 'Blue');
  document.getElementById('fsAll').innerHTML = parts.join(' &nbsp; ');
}

document.getElementById('vfAll').addEventListener('change', loadAllFiles);

// ======================== Hide GoPro GPS toggle ========================
let goProGpsHidden = false;
document.getElementById('hideGoPro').addEventListener('change', function() {
  goProGpsHidden = this.checked;
  if (fadeMode) {
    // In fade mode, control grey bg + fade polylines for GoPro
    [greyMp41,greyMp42].forEach(l=>{
      if(!l) return;
      if(goProGpsHidden) map.removeLayer(l); else map.addLayer(l);
    });
    [fadeMp41,fadeMp42].forEach(fs=>{
      if(goProGpsHidden){
        map.removeLayer(fs.ultraFadeIn); map.removeLayer(fs.fadeIn); map.removeLayer(fs.solid);
        map.removeLayer(fs.fadeOut); map.removeLayer(fs.ultraFadeOut);
      } else {
        map.addLayer(fs.ultraFadeIn); map.addLayer(fs.fadeIn); map.addLayer(fs.solid);
        map.addLayer(fs.fadeOut); map.addLayer(fs.ultraFadeOut);
      }
    });
  } else {
    [mp4Line1, mp4Line2].forEach(l => {
      if (!l) return;
      if (goProGpsHidden) map.removeLayer(l); else map.addLayer(l);
    });
  }
  [mp4Dot1, mp4Dot2].forEach(d => {
    if (!d) return;
    if (goProGpsHidden) d.setStyle({opacity:0,fillOpacity:0});
    else d.setStyle({opacity:1,fillOpacity:0.9});
  });
});

// ======================== Fade tracks toggle ========================
document.getElementById('fadeTracks').addEventListener('change', function() {
  fadeMode = this.checked;
  if (fadeMode) enableFade(); else disableFade();
});

updateAllFileStatus();

// ======================== Master clock ========================
let masterEpoch = epochMin;
let playing = false;
let lastTs = null;

const ppBtn  = document.getElementById('ppBtn');
const scrub  = document.getElementById('scrub');
const tDisp  = document.getElementById('tDisp');

scrub.max = totalDur.toFixed(1);
tDisp.textContent = '0:00 / ' + fmtElapsed(totalDur);

function playAll()  { playing=true;  lastTs=null; ppBtn.innerHTML='&#9646;&#9646; Pause'; }
function pauseAll() {
  playing=false; lastTs=null;
  ppBtn.innerHTML='&#9654; Play';
  ds1.segments.forEach(s => { if(s.el && !s.el.paused) s.el.pause(); });
  ds2.segments.forEach(s => { if(s.el && !s.el.paused) s.el.pause(); });
}
ppBtn.addEventListener('click', () => { playing ? pauseAll() : playAll(); });

document.getElementById('bk10').addEventListener('click', () => {
  if (playing) pauseAll();
  masterEpoch = Math.max(epochMin, masterEpoch - 10);
  syncUI();
});
document.getElementById('fw10').addEventListener('click', () => {
  if (playing) pauseAll();
  masterEpoch = Math.min(epochMax, masterEpoch + 10);
  syncUI();
});

// Spacebar play/pause
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
    e.preventDefault();
    playing ? pauseAll() : playAll();
  }
});

let scrubbing = false;
scrub.addEventListener('input', () => {
  scrubbing = true;
  if (playing) pauseAll();
  masterEpoch = epochMin + parseFloat(scrub.value || '0');
  syncUI();
});
scrub.addEventListener('change', () => { scrubbing = false; });

// ======================== Sync logic ========================
// Find which segment covers masterEpoch; returns index or -1
function findActiveSeg(ds) {
  for (let i = 0; i < ds.segments.length; i++) {
    if (masterEpoch >= ds.segments[i].epochMin && masterEpoch <= ds.segments[i].epochMax)
      return i;
  }
  return -1;
}

function syncDataset(ds, blkEl) {
  const ai = findActiveSeg(ds);

  ds.segments.forEach((seg, i) => {
    if (i === ai) {
      seg.el.style.display = 'block';
      // Sync this video
      if (seg.loaded && seg.el.readyState >= 1) {
        const target = epochToVTime(seg.epochs, seg.vTimes, masterEpoch);
        const drift = Math.abs(seg.el.currentTime - target);
        if (playing) {
          if (seg.el.paused) seg.el.play().catch(()=>{});
          if (drift > 0.5) { seg.pSeek++; seg.el.currentTime = target; }
        } else {
          if (!seg.el.paused) seg.el.pause();
          if (drift > 0.05) { seg.pSeek++; seg.el.currentTime = target; }
        }
      }
    } else {
      seg.el.style.display = 'none';
      if (!seg.el.paused) seg.el.pause();
    }
  });

  blkEl.style.display = (ai >= 0) ? 'none' : 'flex';
  return ai;
}

function updateArrow(marker, allPts, ep, isVak, track, eps) {
  if (!marker) return;
  const pos = isVak ? interpPos(track, eps, ep) : interpDsGps(allPts, ep);
  const pts = isVak ? track : allPts;
  if (pos) {
    marker.setLatLng([pos.lat,pos.lon]);
    marker.setOpacity(1);
    const h = computeHeading(pts, ep);
    if (h !== null) setArrowHeading(marker, h);
  } else {
    marker.setOpacity(0);
  }
}
function updateDot(dot, allPts, ep) {
  if (!dot) return;
  if (typeof goProGpsHidden !== 'undefined' && goProGpsHidden) {
    dot.setStyle({opacity:0,fillOpacity:0}); return;
  }
  const pos = interpDsGps(allPts, ep);
  if (pos) { dot.setLatLng([pos.lat,pos.lon]); dot.setStyle({opacity:1,fillOpacity:0.9}); }
  else     { dot.setStyle({opacity:0,fillOpacity:0}); }
}

// Overlay refs
const oUTC   = document.getElementById('oUTC');
const o1clip = document.getElementById('o1clip');
const o1gps  = document.getElementById('o1gps');
const o1vak  = document.getElementById('o1vak');
const o2clip = document.getElementById('o2clip');
const o2gps  = document.getElementById('o2gps');
const o2vak  = document.getElementById('o2vak');

function syncUI() {
  if (!scrubbing) scrub.value = (masterEpoch - epochMin).toFixed(1);
  tDisp.textContent = fmtElapsed(masterEpoch - epochMin) + ' / ' + fmtElapsed(totalDur);

  const a1 = syncDataset(ds1, blk1);
  const a2 = syncDataset(ds2, blk2);

  // Map markers
  updateDot(mp4Dot1, allGps1, masterEpoch);
  updateArrow(vakArrow1, null, masterEpoch, true, vakTrack1, vakEpochs1);
  updateDot(mp4Dot2, allGps2, masterEpoch);
  updateArrow(vakArrow2, null, masterEpoch, true, vakTrack2, vakEpochs2);

  // Fade-track update
  if (fadeMode) updateFadeTracks(masterEpoch);

  // Overlay
  oUTC.textContent = fmtEpochUTC(masterEpoch);

  if (a1 >= 0) {
    const seg = ds1.segments[a1];
    o1clip.textContent = seg.name + ' @ ' + epochToVTime(seg.epochs,seg.vTimes,masterEpoch).toFixed(1) + 's';
    o1gps.textContent  = fmtLL(interpDsGps(allGps1, masterEpoch));
  } else { o1clip.textContent = '-'; o1gps.textContent = '-'; }
  o1vak.textContent = fmtLL(interpPos(vakTrack1, vakEpochs1, masterEpoch));

  if (a2 >= 0) {
    const seg = ds2.segments[a2];
    o2clip.textContent = seg.name + ' @ ' + epochToVTime(seg.epochs,seg.vTimes,masterEpoch).toFixed(1) + 's';
    o2gps.textContent  = fmtLL(interpDsGps(allGps2, masterEpoch));
  } else { o2clip.textContent = '-'; o2gps.textContent = '-'; }
  o2vak.textContent = fmtLL(interpPos(vakTrack2, vakEpochs2, masterEpoch));
}

// ======================== Animation loop ========================
function tick(ts) {
  if (playing) {
    if (lastTs !== null) {
      masterEpoch += (ts - lastTs) / 1000;
      if (masterEpoch >= epochMax) { masterEpoch = epochMax; pauseAll(); }
    }
    lastTs = ts;
  }
  syncUI();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
syncUI();

// ======================== SOG timeseries chart ========================
(function() {
  function buildSog(track) {
    const pts = [];
    for (const p of track) {
      if (p.sog !== undefined && p.sog !== null && isFinite(p.sog))
        pts.push({ts: p.ts, sog: p.sog});
    }
    pts.sort((a,b) => a.ts - b.ts);
    return pts;
  }

  const sog1 = buildSog(vakTrack1);
  const sog2 = buildSog(vakTrack2);
  if (sog1.length === 0 && sog2.length === 0) return;

  const sogBar = document.getElementById('sogBar');
  sogBar.style.display = '';

  const canvas = document.getElementById('sogCanvas');
  const ctx = canvas.getContext('2d');

  let sogMax = 0;
  for (const p of sog1) sogMax = Math.max(sogMax, p.sog);
  for (const p of sog2) sogMax = Math.max(sogMax, p.sog);
  if (sogMax <= 0) sogMax = 1;
  sogMax *= 1.1;

  // Zoom state
  let sogZoom = 1;
  let sogViewLo = epochMin;
  let sogViewHi = epochMax;

  function updateView() {
    const span = totalDur / sogZoom;
    let center = masterEpoch;
    let lo = center - span / 2;
    let hi = center + span / 2;
    if (lo < epochMin) { lo = epochMin; hi = lo + span; }
    if (hi > epochMax) { hi = epochMax; lo = hi - span; }
    sogViewLo = Math.max(lo, epochMin);
    sogViewHi = Math.min(hi, epochMax);
  }

  const offCanvas = document.createElement('canvas');
  const offCtx = offCanvas.getContext('2d');

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    renderStatic();
    drawCursor();
  }

  function renderStatic() {
    updateView();
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    const dpr = window.devicePixelRatio || 1;
    offCanvas.width = canvas.width;
    offCanvas.height = canvas.height;
    offCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const vSpan = sogViewHi - sogViewLo;

    offCtx.fillStyle = '#0b1220';
    offCtx.fillRect(0, 0, W, H);

    offCtx.strokeStyle = 'rgba(255,255,255,0.06)';
    offCtx.lineWidth = 1;
    for (let i = 1; i <= 2; i++) {
      const y = H - (i / 3) * H;
      offCtx.beginPath(); offCtx.moveTo(0, y); offCtx.lineTo(W, y); offCtx.stroke();
    }

    offCtx.fillStyle = 'rgba(255,255,255,0.3)';
    offCtx.font = '9px monospace';
    offCtx.textBaseline = 'bottom';
    offCtx.fillText((sogMax / 1.1).toFixed(1) + ' m/s', 3, 12);
    if (sogZoom > 1) {
      offCtx.textBaseline = 'top';
      offCtx.fillText(sogZoom + 'x', W - 30, 3);
    }

    function drawLine(pts, color) {
      if (pts.length < 2) return;
      offCtx.strokeStyle = color;
      offCtx.lineWidth = 1;
      offCtx.globalAlpha = 0.8;
      offCtx.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const x = ((pts[i].ts - sogViewLo) / vSpan) * W;
        const y = H - (pts[i].sog / sogMax) * H;
        if (i === 0) offCtx.moveTo(x, y); else offCtx.lineTo(x, y);
      }
      offCtx.stroke();
      offCtx.globalAlpha = 1;
    }
    drawLine(sog1, '#e53935');
    drawLine(sog2, '#42a5f5');
  }

  function drawCursor() {
    const W = canvas.getBoundingClientRect().width;
    const H = canvas.getBoundingClientRect().height;
    const vSpan = sogViewHi - sogViewLo;
    ctx.clearRect(0, 0, W, H);
    const dpr = window.devicePixelRatio || 1;
    ctx.save(); ctx.setTransform(1,0,0,1,0,0);
    ctx.drawImage(offCanvas, 0, 0);
    ctx.restore();

    const cx = ((masterEpoch - sogViewLo) / vSpan) * W;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  window.addEventListener('resize', resizeCanvas);
  setTimeout(resizeCanvas, 100);

  const origSyncUI = syncUI;
  syncUI = function() {
    origSyncUI();
    if (sogZoom > 1) renderStatic();
    drawCursor();
  };

  // Zoom buttons
  document.getElementById('sogZoomIn').addEventListener('click', () => {
    sogZoom = Math.min(sogZoom * 2, 64);
    renderStatic(); drawCursor();
  });
  document.getElementById('sogZoomOut').addEventListener('click', () => {
    sogZoom = Math.max(sogZoom / 2, 1);
    renderStatic(); drawCursor();
  });
  document.getElementById('sogZoomReset').addEventListener('click', () => {
    sogZoom = 1;
    renderStatic(); drawCursor();
  });

  // Drag / click to scrub
  let sogDragging = false;
  function sogScrub(e) {
    const rect = canvas.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const vSpan = sogViewHi - sogViewLo;
    masterEpoch = sogViewLo + frac * vSpan;
    masterEpoch = Math.max(epochMin, Math.min(epochMax, masterEpoch));
    if (playing) pauseAll();
    syncUI();
  }
  canvas.addEventListener('mousedown', e => { sogDragging = true; sogScrub(e); });
  window.addEventListener('mousemove', e => { if (sogDragging) sogScrub(e); });
  window.addEventListener('mouseup', () => { sogDragging = false; });

  canvas.addEventListener('touchstart', e => {
    sogDragging = true;
    sogScrub(e.touches[0]);
    e.preventDefault();
  }, {passive:false});
  canvas.addEventListener('touchmove', e => {
    if (sogDragging) { sogScrub(e.touches[0]); e.preventDefault(); }
  }, {passive:false});
  canvas.addEventListener('touchend', () => { sogDragging = false; });
})();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# File picker helpers
# ------------------------------------------------------------------
def pick_file(title: str, filetypes):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path or ""


def pick_files(title: str, filetypes):
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    root.destroy()
    return list(paths) if paths else []


# ------------------------------------------------------------------
# Main – pick all MP4s + CSVs at once, auto-cluster, generate HTML
# ------------------------------------------------------------------
def main():
    all_mp4s = pick_files(
        "Select ALL GoPro MP4s (both athletes)",
        [("MP4 files", "*.mp4"), ("All files", "*.*")],
    )
    if not all_mp4s:
        return

    all_csvs = pick_files(
        "Select Vakaros / sensor CSV files (0–2, Cancel to skip)",
        [("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not all_csvs:
        all_csvs = []

    try:
        t_start = _time.perf_counter()

        ds1, ds2 = auto_build_datasets(all_mp4s, all_csvs)

        if not ds1["segments"]:
            raise RuntimeError("No usable GPS found in any clip.")

        total_clips = len(ds1["segments"]) + len(ds2["segments"])
        total_gps = sum(len(s["gps"]) for s in ds1["segments"]) + \
                    sum(len(s["gps"]) for s in ds2["segments"])

        # Output next to first MP4
        out_dir = os.path.dirname(all_mp4s[0])
        out_html = os.path.join(out_dir, "dual_athlete_viewer.html")

        html = HTML_TEMPLATE
        html = html.replace("__DS1_JSON__", json.dumps(ds1))
        html = html.replace("__DS2_JSON__", json.dumps(ds2))

        print(f"\nWriting HTML ({len(html)//1024} KB) ...")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        dt_total = _time.perf_counter() - t_start
        print(f"Done → {out_html}  [{dt_total:.1f}s total]")

        names1 = ", ".join(s["name"] for s in ds1["segments"])
        names2 = ", ".join(s["name"] for s in ds2["segments"]) if ds2["segments"] else "(none)"

        mode = "Dual-athlete" if ds2["segments"] else "Single-athlete"
        messagebox.showinfo(
            "Done",
            f"Created:\n{out_html}\n\n"
            f"Mode: {mode}\n"
            f"Clips: {total_clips}  |  GPS points: {total_gps}\n\n"
            f"Red  ({len(ds1['segments'])} clips): {names1}\n"
            f"Blue ({len(ds2['segments'])} clips): {names2}\n\n"
            "Open in a browser, then use the file input to load all MP4s at once.\n"
            "Clips are auto-matched by filename.",
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    main()
