"""Report generation service.

Computes per-segment statistics from skeleton JSONL, metrics CSV, and GPS
track data.  Returns a rich JSON payload consumed by the client-side report
renderer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.app.config import PROJECTS_ROOT

# ── helpers ───────────────────────────────────────────────────────────────


def _load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iqr_filter(values: np.ndarray) -> np.ndarray:
    """Remove outliers outside 1.5×IQR of Q1–Q3."""
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return values[(values >= lo) & (values <= hi)]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _compute_sog_from_track(points: list[dict], start_s: float, end_s: float):
    """Return arrays of (video_s, sog_knots, lat, lon) within [start_s, end_s].

    Speed over ground is calculated from consecutive GPS fixes using haversine.
    """
    seg_pts = [p for p in points if p.get("video_s") is not None and start_s <= p["video_s"] <= end_s]
    seg_pts.sort(key=lambda p: p["video_s"])

    times: list[float] = []
    sogs: list[float] = []
    lats: list[float] = []
    lons: list[float] = []

    for i in range(1, len(seg_pts)):
        p0, p1 = seg_pts[i - 1], seg_pts[i]
        dt = p1["video_s"] - p0["video_s"]
        if dt <= 0:
            continue
        dist = _haversine(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
        speed_ms = dist / dt
        speed_kn = speed_ms * 1.94384  # m/s → knots

        t_mid = (p0["video_s"] + p1["video_s"]) / 2
        lat_mid = (p0["lat"] + p1["lat"]) / 2
        lon_mid = (p0["lon"] + p1["lon"]) / 2

        times.append(t_mid)
        sogs.append(speed_kn)
        lats.append(lat_mid)
        lons.append(lon_mid)

    return times, sogs, lats, lons


def _density_to_color(d: float):
    """Blue→cyan→green→yellow→red gradient returning (r, g, b, a) in [0,255]."""
    if d < 0.25:
        t = d / 0.25
        r, g, b = 0, int(225 * t), 255
    elif d < 0.5:
        t = (d - 0.25) / 0.25
        r, g, b = 0, 255, int(255 * (1 - t))
    elif d < 0.75:
        t = (d - 0.5) / 0.25
        r, g, b = int(255 * t), 255, 0
    else:
        t = (d - 0.75) / 0.25
        r, g, b = 255, int(255 * (1 - t)), 0
    a = 0.8 + d * 0.2
    return r, g, b, a


def _generate_density_grid(
    points_xy: np.ndarray,
    grid_size_x: float = 5.0,
    grid_size_y: float = 3.0,
    grid_center_x: float = -1.5,
    grid_center_y: float = 0.0,
    resolution: int = 120,
    sigma_cells: float = 2.5,
) -> dict:
    """Produce a density image (base64 PNG) from 2-D point cloud.

    Returns dict with keys: image_b64, grid meta, point_count.
    """
    import base64
    import io

    cells_x = resolution
    cells_y = max(1, round(resolution * (grid_size_y / grid_size_x)))
    cell_w = grid_size_x / cells_x
    cell_h = grid_size_y / cells_y

    density = np.zeros((cells_y, cells_x), dtype=np.float32)
    sigma = sigma_cells
    kern_r = int(math.ceil(sigma * 3))

    origin_x = grid_center_x - grid_size_x / 2
    origin_y = grid_center_y - grid_size_y / 2

    for px, py in points_xy:
        gx = (px - origin_x) / cell_w
        gy = (py - origin_y) / cell_h
        cx0 = max(0, int(gx) - kern_r)
        cx1 = min(cells_x - 1, int(gx) + kern_r)
        cy0 = max(0, int(gy) - kern_r)
        cy1 = min(cells_y - 1, int(gy) + kern_r)
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                dsq = (gx - cx) ** 2 + (gy - cy) ** 2
                density[cy, cx] += math.exp(-dsq / (2 * sigma * sigma))

    mx = density.max()
    if mx > 0:
        density /= mx

    # Render to RGBA image (flip Y so row-0 = bottom)
    rgba = np.zeros((cells_y, cells_x, 4), dtype=np.uint8)
    for y in range(cells_y):
        for x in range(cells_x):
            d = density[y, x]
            r, g, b, a = _density_to_color(d)
            fy = cells_y - 1 - y
            rgba[fy, x] = [r, g, b, int(a * 255) if d > 0.01 else 0]

    # Encode as PNG
    try:
        from PIL import Image

        img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Minimal fallback: raw base64 of pixel data (client can decode)
        b64 = base64.b64encode(rgba.tobytes()).decode()

    return {
        "image_b64": b64,
        "width": cells_x,
        "height": cells_y,
        "grid_size_x": grid_size_x,
        "grid_size_y": grid_size_y,
        "grid_center_x": grid_center_x,
        "grid_center_y": grid_center_y,
        "point_count": len(points_xy),
    }


# ── COM helpers ──────────────────────────────────────────────────────────


_COM_SEGMENTS = [
    # (idx_a, idx_b, mass_fraction, com_fraction)  — mirrors skeleton_metrics.py
    # Head — shoulder-mid → head-top-mid
    ((11, 12), (7, 8), 0.081, 1.000),
    # Trunk — shoulder-mid → hip-mid
    ((11, 12), (23, 24), 0.497, 0.500),
    # Left upper arm
    (11, 13, 0.028, 0.436),
    (13, 15, 0.016, 0.430),
    (15, 17, 0.006, 0.506),
    # Right upper arm
    (12, 14, 0.028, 0.436),
    (14, 16, 0.016, 0.430),
    (16, 18, 0.006, 0.506),
    # Left leg
    (23, 25, 0.100, 0.433),
    (25, 27, 0.0465, 0.433),
    (27, 31, 0.0145, 0.500),
    # Right leg
    (24, 26, 0.100, 0.433),
    (26, 28, 0.0465, 0.433),
    (28, 32, 0.0145, 0.500),
]


def _midpoint(lm, a, b):
    pa, pb = lm[a] if a < len(lm) and lm[a] else None, lm[b] if b < len(lm) and lm[b] else None
    if pa and pb:
        return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2]
    return pa or pb


def _compute_com_from_landmarks(lm):
    """Segmental COM from 33 landmarks. Returns [x, y, z] or None."""

    def _pt(idx):
        if isinstance(idx, tuple):
            return _midpoint(lm, idx[0], idx[1])
        if idx < len(lm) and lm[idx]:
            return lm[idx]
        return None

    total_mass = 0.0
    cx = cy = cz = 0.0
    for seg in _COM_SEGMENTS:
        p1 = _pt(seg[0])
        p2 = _pt(seg[1])
        mf = seg[2]
        cf = seg[3]
        if p1 is None or p2 is None:
            continue
        sx = p1[0] + cf * (p2[0] - p1[0])
        sy = p1[1] + cf * (p2[1] - p1[1])
        sz = p1[2] + cf * (p2[2] - p1[2])
        cx += mf * sx
        cy += mf * sy
        cz += mf * sz
        total_mass += mf
    if total_mass < 0.35:
        return None
    return [cx / total_mass, cy / total_mass, cz / total_mass]


# ── main report builder ─────────────────────────────────────────────────


def build_report_data(
    project_id: str,
    split_ids: list[str],
    file_meta: dict,
    athletes: list[dict],
    track_points_by_file: dict[str, list[dict]],
    skeleton_jsonl_paths: dict[str, Path],
    cv_config: dict | None = None,
) -> dict:
    """Build the full report payload.

    Parameters
    ----------
    project_id : str
    split_ids : list[str]
        The IDs of splits the user selected.
    file_meta : dict
        ``{file_id: {athlete_id, splits: [{id, name, start_s, end_s}], ...}}``
    athletes : list[dict]
        ``[{id, name, weight}, ...]``
    track_points_by_file : dict
        ``{file_id: [{ts, lat, lon, video_s}, ...]}``
    skeleton_jsonl_paths : dict
        ``{file_id: Path}``
    cv_config : dict | None
        Project CV config (for athlete_weight, boat_com).

    Returns
    -------
    dict
        Full report payload ready for JSON serialisation.
    """
    athlete_map = {a["id"]: a for a in athletes}
    athlete_weight = float((cv_config or {}).get("athlete_weight", 75.0))
    boat_com_x = float((cv_config or {}).get("boat_com", 0.0))

    # ── Resolve splits ────────────────────────────────────────────────
    segments: list[dict] = []
    for fid, meta in file_meta.items():
        for sp in meta.get("splits") or []:
            if sp["id"] not in split_ids:
                continue
            ath_id = meta.get("athlete_id")
            ath = athlete_map.get(ath_id) if ath_id else None
            segments.append(
                {
                    "split_id": sp["id"],
                    "name": sp["name"],
                    "start_s": sp["start_s"],
                    "end_s": sp["end_s"],
                    "file_id": fid,
                    "athlete_id": ath_id,
                    "athlete_name": ath["name"] if ath else "Unassigned",
                    "athlete_weight": ath["weight"] if ath else athlete_weight,
                }
            )

    if not segments:
        return {"project_id": project_id, "athletes": [], "segments": []}

    # ── Load skeleton data per file ───────────────────────────────────
    skel_by_file: dict[str, list[dict]] = {}
    for fid, path in skeleton_jsonl_paths.items():
        if not path.exists():
            continue
        frames: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                frames.append(obj)
            except Exception:
                continue
        frames.sort(key=lambda f: f.get("video_s", 0))
        skel_by_file[fid] = frames

    # ── Process each segment ──────────────────────────────────────────
    seg_results: list[dict] = []
    for seg in segments:
        fid = seg["file_id"]
        s0, s1 = seg["start_s"], seg["end_s"]
        weight = seg["athlete_weight"]

        # --- Skeleton metrics within [s0, s1] ---
        skel_frames = skel_by_file.get(fid, [])
        seg_skel = [f for f in skel_frames if f.get("video_s") is not None and s0 <= f["video_s"] <= s1]

        trunk_angles: list[float] = []
        moments_pitch: list[float] = []
        moments_roll: list[float] = []
        com_xs: list[float] = []
        com_ys: list[float] = []
        com_zs: list[float] = []
        all_keypoints_xy: list[tuple[float, float]] = []
        all_com_xy: list[tuple[float, float]] = []
        trunk_angle_time: list[dict] = []  # {video_s, value}
        moment_time: list[dict] = []

        for fr in seg_skel:
            m = fr.get("metrics") or {}
            vs = fr["video_s"]

            ta = m.get("trunk_angle")
            if ta is not None and 0 <= ta <= 180:
                trunk_angles.append(ta)
                trunk_angle_time.append({"t": vs, "v": ta})

            mp = m.get("moment_pitch")
            mr = m.get("moment_roll")
            if mp is not None:
                moments_pitch.append(mp)
            if mr is not None:
                moments_roll.append(mr)
                moment_time.append({"t": vs, "v": mr})

            cx, cy, cz = m.get("com_x"), m.get("com_y"), m.get("com_z")
            if cx is not None:
                com_xs.append(cx)
            if cy is not None:
                com_ys.append(cy)
            if cz is not None:
                com_zs.append(cz)

            # Keypoint positions (x=fore/aft, y=port/stbd in boat coords)
            lm = fr.get("landmarks") or []
            for pt in lm:
                if pt and len(pt) == 3 and all(math.isfinite(v) for v in pt):
                    all_keypoints_xy.append((pt[0], pt[1]))

            # COM for heatmap
            com = _compute_com_from_landmarks(lm) if lm else None
            if com is None and cx is not None and cy is not None:
                com = [cx, cy, cz or 0]
            if com is not None:
                all_com_xy.append((com[0], com[1]))

        # --- GPS / SOG ---
        track_pts = track_points_by_file.get(fid, [])
        sog_times, sog_vals, sog_lats, sog_lons = _compute_sog_from_track(track_pts, s0, s1)

        # --- Heel from instrument CSV ---
        heel_vals: list[float] = []
        heel_time: list[dict] = []  # {t: video_s, v: heel_deg}
        seg_track_pts = [p for p in track_pts if p.get("video_s") is not None and s0 <= p["video_s"] <= s1]
        for p in seg_track_pts:
            h = p.get("heel")
            if h is not None:
                heel_vals.append(float(h))
                heel_time.append({"t": p["video_s"], "v": float(h)})

        # GPS path for this segment
        gps_path = [
            {"lat": lat, "lon": lon, "sog": sog}
            for lat, lon, sog in zip(sog_lats, sog_lons, sog_vals)
        ]

        # --- Statistics ---
        def _stats(arr_raw, iqr_filter=False):
            if not arr_raw:
                return {"avg": None, "max": None, "min": None, "count": 0}
            a = np.array(arr_raw, dtype=np.float64)
            if iqr_filter:
                a = _iqr_filter(a)
            if len(a) == 0:
                return {"avg": None, "max": None, "min": None, "count": 0}
            return {
                "avg": float(np.mean(a)),
                "max": float(np.max(a)),
                "min": float(np.min(a)),
                "std": float(np.std(a)),
                "count": int(len(a)),
            }

        sog_stats = _stats(sog_vals)
        abs_moments_roll = [np.abs(v) for v in moments_roll if v is not None]
        moment_roll_stats = _stats(abs_moments_roll, iqr_filter=True)
        moment_pitch_stats = _stats(moments_pitch, iqr_filter=True)
        trunk_stats = _stats(trunk_angles)
        heel_stats = _stats(heel_vals)

        # --- Hiking histogram (trunk angle bins) ---
        hiking_hist = _build_hiking_histogram(trunk_angles)

        # --- Cumulative side load balance ---
        side_load_cumulative = _build_cumulative_side_load(seg_skel, weight)

        # --- Heatmaps (base64 PNG) ---
        kp_heatmap = None
        com_heatmap = None
        if all_keypoints_xy:
            kp_heatmap = _generate_density_grid(
                np.array(all_keypoints_xy),
                resolution=120,
                sigma_cells=2.5,
            )
        if all_com_xy:
            com_heatmap = _generate_density_grid(
                np.array(all_com_xy),
                resolution=120,
                sigma_cells=3.0,
            )

        # --- Trunk angle vs SOG scatter ---
        trunk_vs_sog = _build_trunk_vs_sog_scatter(trunk_angle_time, sog_times, sog_vals)

        # --- Side-switch analysis (com_y sign changes) ---
        side_switch_data = _build_side_switch_analysis(
            seg_skel, trunk_angle_time, moment_time
        )

        # --- Subsample time-series for payload ---
        trunk_timeline = _subsample_time_series(trunk_angle_time, 600)
        moment_timeline = _subsample_time_series(moment_time, 600)
        sog_timeline = _build_sog_timeline(sog_times, sog_vals, 600)
        heel_timeline = _subsample_time_series(heel_time, 600)

        seg_results.append(
            {
                "split_id": seg["split_id"],
                "name": seg["name"],
                "start_s": s0,
                "end_s": s1,
                "file_id": fid,
                "athlete_id": seg["athlete_id"],
                "athlete_name": seg["athlete_name"],
                "duration_s": s1 - s0,
                "sog": sog_stats,
                "moment_roll": moment_roll_stats,
                "moment_pitch": moment_pitch_stats,
                "trunk_angle": trunk_stats,
                "heel": heel_stats,
                "gps_path": gps_path,
                "hiking_histogram": hiking_hist,
                "side_load_cumulative": side_load_cumulative,
                "keypoint_heatmap": kp_heatmap,
                "com_heatmap": com_heatmap,
                "kp_xy": all_keypoints_xy,
                "com_xy": all_com_xy,
                "trunk_vs_sog": trunk_vs_sog,
                "trunk_angle_timeline": trunk_timeline,
                "moment_timeline": moment_timeline,
                "sog_timeline": sog_timeline,
                "heel_timeline": heel_timeline,
                "side_switches": side_switch_data,
            }
        )

    # ── Group by athlete and find golds ──────────────────────────────
    athlete_ids_ordered = []
    for s in seg_results:
        aid = s["athlete_id"] or "__unassigned__"
        if aid not in athlete_ids_ordered:
            athlete_ids_ordered.append(aid)

    athletes_out: list[dict] = []
    for aid in athlete_ids_ordered:
        ath = athlete_map.get(aid)
        a_segs = [s for s in seg_results if (s["athlete_id"] or "__unassigned__") == aid]
        athletes_out.append(
            {
                "athlete_id": aid,
                "name": ath["name"] if ath else "Unassigned",
                "weight": ath["weight"] if ath else athlete_weight,
                "segment_ids": [s["split_id"] for s in a_segs],
            }
        )

    # Gold marks: best segment per metric (across ALL segments).
    # Categories: max_sog, avg_sog, max_trunk, avg_trunk,
    #             max_moment_roll (abs), avg_moment_roll (abs)
    golds = _compute_golds(seg_results)

    return {
        "project_id": project_id,
        "athletes": athletes_out,
        "segments": seg_results,
        "golds": golds,
    }


# ── histogram / scatter builders ─────────────────────────────────────────


def _build_hiking_histogram(trunk_angles: list[float], bin_width: float = 5.0) -> dict:
    """Build histogram of trunk angles for hiking distribution."""
    if not trunk_angles:
        return {"bins": [], "counts": [], "bin_width": bin_width}

    a = np.array(trunk_angles)
    lo = float(np.floor(a.min() / bin_width) * bin_width)
    hi = float(np.ceil(a.max() / bin_width) * bin_width)
    bins = np.arange(lo, hi + bin_width, bin_width)
    counts, edges = np.histogram(a, bins=bins)
    bin_centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    return {
        "bins": bin_centers,
        "counts": counts.tolist(),
        "bin_width": bin_width,
    }


def _build_trunk_vs_sog_scatter(
    trunk_time: list[dict],
    sog_times: list[float],
    sog_vals: list[float],
) -> dict:
    """Match trunk angle samples to nearest SOG sample and return scatter +
    linear fit."""
    if not trunk_time or not sog_times:
        return {"points": [], "fit_slope": None, "fit_intercept": None}

    sog_t = np.array(sog_times)
    sog_v = np.array(sog_vals)

    scatter_x: list[float] = []  # SOG
    scatter_y: list[float] = []  # trunk angle

    for rec in trunk_time:
        t = rec["t"]
        # Find nearest SOG sample
        idx = int(np.argmin(np.abs(sog_t - t)))
        if abs(sog_t[idx] - t) > 2.0:
            continue  # too far apart
        scatter_x.append(float(sog_v[idx]))
        scatter_y.append(rec["v"])

    if len(scatter_x) < 3:
        return {"points": list(zip(scatter_x, scatter_y)), "fit_slope": None, "fit_intercept": None}

    # Linear regression
    x_arr = np.array(scatter_x)
    y_arr = np.array(scatter_y)
    try:
        coeffs = np.polyfit(x_arr, y_arr, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
    except Exception:
        slope, intercept = None, None

    # Subsample for payload size (max ~500 points)
    pts = list(zip(scatter_x, scatter_y))
    if len(pts) > 500:
        step = len(pts) / 500
        pts = [pts[int(i * step)] for i in range(500)]

    return {"points": pts, "fit_slope": slope, "fit_intercept": intercept}


def _build_cumulative_side_load(seg_skel: list[dict], weight: float) -> dict:
    """Compute cumulative side load balance over the segment.

    Side load is proportional to moment_roll (= weight * g * com_y).
    We accumulate the signed rolling moment over time, giving a running
    total of how much load the athlete has applied to port vs starboard.
    """
    times: list[float] = []
    cumulative: list[float] = []
    running = 0.0

    prev_t: float | None = None
    for fr in seg_skel:
        vs = fr.get("video_s")
        m = fr.get("metrics") or {}
        mr = m.get("moment_roll")
        if vs is None or mr is None:
            continue
        dt = (vs - prev_t) if prev_t is not None else 0.0
        if dt < 0:
            dt = 0
        # Impulse = moment × dt  (Nm·s)
        running += mr * dt
        times.append(vs)
        cumulative.append(running)
        prev_t = vs

    # Subsample for payload (max 600 points)
    if len(times) > 600:
        step = len(times) / 600
        times = [times[int(i * step)] for i in range(600)]
        cumulative = [cumulative[int(i * step)] for i in range(600)]

    return {"times": times, "values": cumulative}


def _subsample_time_series(data: list[dict], max_points: int = 600) -> list[dict]:
    """Subsample a list of {t, v} dicts to at most *max_points*."""
    if len(data) <= max_points:
        return data
    step = len(data) / max_points
    return [data[int(i * step)] for i in range(max_points)]


def _build_sog_timeline(
    sog_times: list[float], sog_vals: list[float], max_points: int = 600
) -> list[dict]:
    """Return [{t, v}] for SOG time-series."""
    pts = [{"t": t, "v": v} for t, v in zip(sog_times, sog_vals)]
    return _subsample_time_series(pts, max_points)


def _build_side_switch_analysis(
    seg_skel: list[dict],
    trunk_angle_time: list[dict],
    moment_time: list[dict],
) -> dict:
    """Detect side switches (com_y sign changes) and compute per-side metrics.

    Port = com_y > 0, Starboard = com_y < 0 (boat coordinate convention).
    """
    # Collect com_y values with timestamps
    com_y_series: list[tuple[float, float]] = []  # (video_s, com_y)
    for fr in seg_skel:
        vs = fr.get("video_s")
        m = fr.get("metrics") or {}
        cy = m.get("com_y")
        if vs is not None and cy is not None:
            com_y_series.append((vs, cy))

    if not com_y_series:
        return {
            "count": 0,
            "port_fraction": 0.5,
            "starboard_fraction": 0.5,
            "port_trunk_avg": None,
            "stbd_trunk_avg": None,
            "port_moment_avg": None,
            "stbd_moment_avg": None,
        }

    # Count sign changes (ignoring zero crossings where cy==0)
    switch_count = 0
    prev_sign: int | None = None
    for _, cy in com_y_series:
        if cy == 0:
            continue
        s = 1 if cy > 0 else -1
        if prev_sign is not None and s != prev_sign:
            switch_count += 1
        prev_sign = s

    # Time spent on each side
    port_count = sum(1 for _, cy in com_y_series if cy > 0)
    stbd_count = sum(1 for _, cy in com_y_series if cy < 0)
    total = port_count + stbd_count
    port_frac = port_count / total if total > 0 else 0.5
    stbd_frac = stbd_count / total if total > 0 else 0.5

    # Build a quick lookup: video_s → side ('port' / 'stbd')
    # Use chronologically nearest com_y for each trunk/moment sample
    com_y_arr_t = np.array([t for t, _ in com_y_series])
    com_y_arr_v = np.array([v for _, v in com_y_series])

    def _side_at(t: float) -> str | None:
        idx = int(np.argmin(np.abs(com_y_arr_t - t)))
        if abs(com_y_arr_t[idx] - t) > 1.0:
            return None
        return "port" if com_y_arr_v[idx] > 0 else "stbd"

    # Per-side trunk angle stats
    port_ta: list[float] = []
    stbd_ta: list[float] = []
    for rec in trunk_angle_time:
        side = _side_at(rec["t"])
        if side == "port":
            port_ta.append(rec["v"])
        elif side == "stbd":
            stbd_ta.append(rec["v"])

    # Per-side moment stats
    port_mr: list[float] = []
    stbd_mr: list[float] = []
    for rec in moment_time:
        side = _side_at(rec["t"])
        if side == "port":
            port_mr.append(rec["v"])
        elif side == "stbd":
            stbd_mr.append(rec["v"])

    return {
        "count": switch_count,
        "port_fraction": round(port_frac, 3),
        "starboard_fraction": round(stbd_frac, 3),
        "port_trunk_avg": float(np.mean(port_ta)) if port_ta else None,
        "stbd_trunk_avg": float(np.mean(stbd_ta)) if stbd_ta else None,
        "port_moment_avg": float(np.mean(np.abs(port_mr))) if port_mr else None,
        "stbd_moment_avg": float(np.mean(np.abs(stbd_mr))) if stbd_mr else None,
    }


def _compute_golds(seg_results: list[dict]) -> dict:
    """Find the best athlete in each category *within each segment*.

    Return ``{category: {split_id: athlete_name}}``.
    Gold is only awarded when at least 2 athletes have data in the same segment.
    """
    from collections import defaultdict

    categories = {
        "max_sog": lambda s: s["sog"]["max"],
        "avg_sog": lambda s: s["sog"]["avg"],
        "max_trunk_angle": lambda s: s["trunk_angle"]["max"],
        "avg_trunk_angle": lambda s: s["trunk_angle"]["avg"],
        "max_moment_roll": lambda s: abs(s["moment_roll"]["max"]) if s["moment_roll"]["max"] else None,
        "avg_moment_roll": lambda s: abs(s["moment_roll"]["avg"]) if s["moment_roll"]["avg"] else None,
        "max_heel": lambda s: s["heel"]["max"] if s["heel"]["max"] else None,
        "avg_heel": lambda s: s["heel"]["avg"] if s["heel"]["avg"] else None,
    }

    # Group results by segment
    by_split: dict[str, list[dict]] = defaultdict(list)
    for s in seg_results:
        by_split[s["split_id"]].append(s)

    golds: dict[str, dict[str, str]] = {}
    for cat, fn in categories.items():
        golds[cat] = {}
        for split_id, segs in by_split.items():
            best_val = None
            best_name: str | None = None
            candidates_with_data = 0
            for s in segs:
                try:
                    v = fn(s)
                except Exception:
                    v = None
                if v is None:
                    continue
                candidates_with_data += 1
                if best_val is None or v > best_val:
                    best_val = v
                    best_name = s["athlete_name"]
            # Only award gold when there are multiple athletes to compare
            if candidates_with_data >= 2 and best_name is not None:
                golds[cat][split_id] = best_name
    return golds
