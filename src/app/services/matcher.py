from __future__ import annotations

import bisect
import math
from statistics import median
from typing import Any

from src.app.config import MATCH_SAMPLE_POINTS


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))
    return radius * c


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


def _sample_track(points: list[dict[str, float]], max_samples: int = MATCH_SAMPLE_POINTS) -> list[dict[str, float]]:
    if len(points) <= max_samples:
        return points
    stride = max(1, len(points) // max_samples)
    sampled = points[::stride]
    if sampled[-1] is not points[-1]:
        sampled.append(points[-1])
    return sampled


def _interpolate(
    points: list[dict[str, float]], times: list[float], target_ts: float, max_gap_s: float = 5.0
) -> dict[str, float] | None:
    if not points:
        return None
    if target_ts < times[0] or target_ts > times[-1]:
        return None
    if len(points) == 1:
        return points[0]

    idx = bisect.bisect_right(times, target_ts)
    i0 = max(0, idx - 1)
    i1 = min(len(points) - 1, idx)
    p0 = points[i0]
    p1 = points[i1]
    t0 = p0["ts"]
    t1 = p1["ts"]
    if abs(t1 - t0) < 1e-9:
        return p0
    if (t1 - t0) > max_gap_s:
        return None
    alpha = (target_ts - t0) / (t1 - t0)
    lat = (1.0 - alpha) * p0["lat"] + alpha * p1["lat"]
    lon = (1.0 - alpha) * p0["lon"] + alpha * p1["lon"]
    return {"lat": lat, "lon": lon, "ts": target_ts}


def _score_for_offset(
    video_track: list[dict[str, float]],
    csv_track: list[dict[str, float]],
    offset_seconds: float,
) -> dict[str, float] | None:
    video_sample = _sample_track(video_track)
    csv_times = [p["ts"] for p in csv_track]
    distances: list[float] = []
    for point in video_sample:
        interp = _interpolate(csv_track, csv_times, point["ts"] + offset_seconds)
        if interp is None:
            continue
        distances.append(
            haversine_m(point["lat"], point["lon"], float(interp["lat"]), float(interp["lon"]))
        )

    if len(distances) < 10:
        return None
    med = float(median(distances))
    p90 = float(_percentile(distances, 0.90))
    coverage = len(distances) / max(1, len(video_sample))
    score = med + (1.0 - coverage) * 80.0 + max(0.0, p90 - 75.0) * 0.15
    return {
        "score": score,
        "median_distance_m": med,
        "p90_distance_m": p90,
        "coverage": coverage,
        "sample_count": float(len(distances)),
        "offset_seconds": float(offset_seconds),
    }


def _best_offset(video_track: list[dict[str, float]], csv_track: list[dict[str, float]]) -> dict[str, float] | None:
    best: dict[str, float] | None = None

    for coarse_offset in range(-300, 301, 5):
        result = _score_for_offset(video_track, csv_track, float(coarse_offset))
        if result is None:
            continue
        if best is None or result["score"] < best["score"]:
            best = result

    if best is None:
        return None

    center = int(round(best["offset_seconds"]))
    for fine_offset in range(center - 6, center + 7):
        result = _score_for_offset(video_track, csv_track, float(fine_offset))
        if result is None:
            continue
        if result["score"] < best["score"]:
            best = result
    return best


def match_video_tracks_to_csv(
    video_tracks: list[dict[str, Any]],
    csv_tracks: list[dict[str, Any]],
    max_rank_per_video: int = 3,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for video in video_tracks:
        candidates: list[dict[str, Any]] = []
        video_points = video.get("points") or []
        if len(video_points) < 12:
            continue
        for csv_item in csv_tracks:
            csv_points = csv_item.get("points") or []
            if len(csv_points) < 12:
                continue
            best = _best_offset(video_points, csv_points)
            if best is None:
                continue
            candidates.append(
                {
                    "video_file_id": video["file_id"],
                    "csv_file_id": csv_item["file_id"],
                    "score": float(best["score"]),
                    "median_distance_m": float(best["median_distance_m"]),
                    "p90_distance_m": float(best["p90_distance_m"]),
                    "coverage": float(best["coverage"]),
                    "offset_seconds": float(best["offset_seconds"]),
                    "sample_count": int(best["sample_count"]),
                }
            )
        candidates.sort(key=lambda item: item["score"])
        for rank, candidate in enumerate(candidates[:max_rank_per_video], start=1):
            candidate["rank"] = rank
            results.append(candidate)
    return results
