from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

from .boat_simulation import (
    DEFAULT_DEAD_ZONE_ANGLE_RAD,
    ILCA7_POLAR_SPEEDS_MS,
    ILCA7_TWA_DEG,
    ILCA7_TWS_KTS,
    KNOT_TO_MS,
    angle_difference,
    mirror_angle_to_half_circle,
    wrap_phase,
)


@dataclass
class WindTrack:
    schema_name: str
    timestamps: list[str]
    latitude: np.ndarray
    longitude: np.ndarray
    speed_ms: np.ndarray
    course_rad: np.ndarray
    heading_rad: np.ndarray
    heel_deg: np.ndarray
    trim_deg: np.ndarray
    elapsed_seconds: np.ndarray
    dt_seconds: np.ndarray
    turn_rate_deg_s: np.ndarray
    acceleration_ms2: np.ndarray

    @property
    def sample_count(self) -> int:
        return len(self.timestamps)


@dataclass
class AthletePolarPoint:
    twa_deg: float
    athlete_speed_kts: float
    reference_speed_kts: float
    performance_pct: float
    sample_count: int


@dataclass
class AthletePolar:
    source_name: str
    estimated_true_wind_kts: float
    statistic_label: str
    points: list[AthletePolarPoint]

    @property
    def point_count(self) -> int:
        return len(self.points)

    def rows(self) -> list[dict[str, float | int | str]]:
        return [
            {
                "source_name": self.source_name,
                "estimated_true_wind_kts": round(self.estimated_true_wind_kts, 2),
                "statistic": self.statistic_label,
                "twa_deg": round(point.twa_deg, 1),
                "athlete_speed_kts": round(point.athlete_speed_kts, 3),
                "reference_speed_kts": round(point.reference_speed_kts, 3),
                "performance_pct": "" if not np.isfinite(point.performance_pct) else round(point.performance_pct, 1),
                "sample_count": int(point.sample_count),
            }
            for point in self.points
        ]


@dataclass
class LocalWindEstimate:
    timestamp: str
    elapsed_seconds: float
    latitude: float
    longitude: float
    true_wind_direction_deg: float
    true_wind_speed_kts: float
    fit_score: float
    sample_count: int
    inlier_count: int


@dataclass
class WindTimeSeries:
    source_name: str
    window_seconds: float
    step_seconds: float
    points: list[LocalWindEstimate]

    @property
    def point_count(self) -> int:
        return len(self.points)

    def rows(self) -> list[dict[str, float | int | str]]:
        return [
            {
                "source_name": self.source_name,
                "window_seconds": round(self.window_seconds, 1),
                "step_seconds": round(self.step_seconds, 1),
                "timestamp": point.timestamp,
                "elapsed_seconds": round(point.elapsed_seconds, 3),
                "latitude": "" if not np.isfinite(point.latitude) else round(point.latitude, 7),
                "longitude": "" if not np.isfinite(point.longitude) else round(point.longitude, 7),
                "true_wind_direction_deg": round(point.true_wind_direction_deg, 2),
                "true_wind_speed_kts": round(point.true_wind_speed_kts, 2),
                "fit_score": round(point.fit_score, 6),
                "sample_count": point.sample_count,
                "inlier_count": point.inlier_count,
            }
            for point in self.points
        ]


@dataclass
class WindEstimateResult:
    true_wind_direction_rad: float
    true_wind_speed_ms: float
    score: float
    inlier_mask: np.ndarray = field(repr=False)
    tow_outlier_mask: np.ndarray = field(repr=False)
    low_confidence_mask: np.ndarray = field(repr=False)
    predicted_speed_ms: np.ndarray = field(repr=False)
    residual_speed_ms: np.ndarray = field(repr=False)
    twa_deg: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    timestamps: list[str] = field(repr=False)
    observed_speed_ms: np.ndarray = field(repr=False)
    heading_rad: np.ndarray = field(repr=False)
    heel_deg: np.ndarray = field(repr=False)
    trim_deg: np.ndarray = field(repr=False)

    @property
    def true_wind_direction_deg(self) -> float:
        return float(np.rad2deg(self.true_wind_direction_rad) % 360.0)

    @property
    def true_wind_speed_kts(self) -> float:
        return float(self.true_wind_speed_ms / KNOT_TO_MS)

    @property
    def sample_count(self) -> int:
        return int(self.observed_speed_ms.size)

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def tow_outlier_count(self) -> int:
        return int(np.count_nonzero(self.tow_outlier_mask))

    @property
    def low_confidence_count(self) -> int:
        return int(np.count_nonzero(self.low_confidence_mask))

    @property
    def mean_absolute_error_kts(self) -> float:
        if not np.any(self.inlier_mask):
            return float("nan")
        return float(np.mean(np.abs(self.residual_speed_ms[self.inlier_mask])) / KNOT_TO_MS)

    @property
    def median_absolute_error_kts(self) -> float:
        if not np.any(self.inlier_mask):
            return float("nan")
        return float(np.median(np.abs(self.residual_speed_ms[self.inlier_mask])) / KNOT_TO_MS)

    def summary_dict(self) -> dict[str, float | int]:
        return {
            "true_wind_direction_deg": round(self.true_wind_direction_deg, 2),
            "true_wind_speed_kts": round(self.true_wind_speed_kts, 2),
            "fit_score": round(float(self.score), 6),
            "sample_count": self.sample_count,
            "inlier_count": self.inlier_count,
            "tow_outlier_count": self.tow_outlier_count,
            "low_confidence_count": self.low_confidence_count,
            "mean_absolute_error_kts": round(self.mean_absolute_error_kts, 3),
            "median_absolute_error_kts": round(self.median_absolute_error_kts, 3),
        }


def load_track_csv(path: str | Path) -> WindTrack:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} does not contain any data rows")

    columns = set(rows[0].keys())
    if {"timestamp", "latitude", "longitude", "sog_kts", "cog", "hdg_true", "heel", "trim"}.issubset(columns):
        return _load_vakaros_rows(rows)
    if {"timestamp_ms", "sog_mps", "mag_hdg", "roll_deg", "pitch_deg"}.issubset(columns):
        return _load_blue_rows(rows)
    raise ValueError(f"Unsupported CSV schema in {path}. Columns: {sorted(columns)}")


def load_vakaros_csv(path: str | Path) -> WindTrack:
    return load_track_csv(path)


def _load_vakaros_rows(rows: list[dict[str, str]]) -> WindTrack:
    timestamps = [str(row["timestamp"]) for row in rows]
    datetimes = [_parse_timestamp(value) for value in timestamps]
    latitude = np.array([_parse_required_float(row, "latitude") for row in rows], dtype=float)
    longitude = np.array([_parse_required_float(row, "longitude") for row in rows], dtype=float)
    speed_ms = np.array([_parse_required_float(row, "sog_kts") * KNOT_TO_MS for row in rows], dtype=float)
    course_rad = np.deg2rad(np.array([_parse_required_float(row, "cog") for row in rows], dtype=float))
    heading_rad = np.deg2rad(np.array([_parse_required_float(row, "hdg_true") for row in rows], dtype=float))
    heel_deg = np.array([_parse_required_float(row, "heel") for row in rows], dtype=float)
    trim_deg = np.array([_parse_required_float(row, "trim") for row in rows], dtype=float)

    return _build_track(
        schema_name="vakaros",
        timestamps=timestamps,
        datetimes=datetimes,
        latitude=latitude,
        longitude=longitude,
        speed_ms=speed_ms,
        course_rad=course_rad,
        heading_rad=heading_rad,
        heel_deg=heel_deg,
        trim_deg=trim_deg,
    )


def _load_blue_rows(rows: list[dict[str, str]]) -> WindTrack:
    timestamp_ms = np.array([_parse_required_float(row, "timestamp_ms") for row in rows], dtype=float)
    base_row_index = next(
        (index for index, row in enumerate(rows) if str(row.get("iso_time", "")).strip()),
        None,
    )
    if base_row_index is None:
        raise ValueError("BLUE-format CSV is missing an absolute iso_time anchor")

    base_datetime = _parse_timestamp(str(rows[base_row_index]["iso_time"]).strip())
    base_timestamp_ms = float(timestamp_ms[base_row_index])
    datetimes = [base_datetime + timedelta(milliseconds=float(value - base_timestamp_ms)) for value in timestamp_ms]
    timestamps = [_format_timestamp(value) for value in datetimes]
    elapsed_seconds = (timestamp_ms - timestamp_ms[0]) / 1000.0

    latitude = _parse_optional_float_column(rows, "lat")
    longitude = _parse_optional_float_column(rows, "lon")
    latitude, longitude = _sanitize_gps_positions(latitude, longitude, elapsed_seconds)
    speed_ms = _sanitize_speed_column(_parse_optional_float_column(rows, "sog_mps"))

    heading_deg = _parse_optional_float_column(rows, "heading_deg")
    heading_deg = _forward_fill_numeric(heading_deg)
    magnetic_heading_deg = _wrap_degrees(np.nan_to_num(_parse_optional_float_column(rows, "mag_hdg"), nan=0.0))
    heading_deg = np.where(np.isnan(heading_deg), magnetic_heading_deg, heading_deg)
    heading_deg = _wrap_degrees(_forward_fill_numeric(heading_deg, fill_value=0.0))
    heading_rad = np.deg2rad(heading_deg)

    course_deg = _course_from_positions(latitude, longitude)
    course_deg = np.where(np.isfinite(course_deg), course_deg, heading_deg)
    course_deg = _wrap_degrees(_forward_fill_numeric(course_deg, fill_value=float(heading_deg[0])))
    course_rad = np.deg2rad(course_deg)

    heel_deg = np.nan_to_num(_parse_optional_float_column(rows, "roll_deg"), nan=0.0)
    trim_deg = np.nan_to_num(_parse_optional_float_column(rows, "pitch_deg"), nan=0.0)

    return _build_track(
        schema_name="blue_sensor",
        timestamps=timestamps,
        datetimes=datetimes,
        latitude=latitude,
        longitude=longitude,
        speed_ms=speed_ms,
        course_rad=course_rad,
        heading_rad=heading_rad,
        heel_deg=heel_deg,
        trim_deg=trim_deg,
    )


def _build_track(
    schema_name: str,
    timestamps: list[str],
    datetimes: list[datetime],
    latitude: np.ndarray,
    longitude: np.ndarray,
    speed_ms: np.ndarray,
    course_rad: np.ndarray,
    heading_rad: np.ndarray,
    heel_deg: np.ndarray,
    trim_deg: np.ndarray,
) -> WindTrack:
    elapsed_seconds = np.array([(value - datetimes[0]).total_seconds() for value in datetimes], dtype=float)
    dt_seconds = np.diff(elapsed_seconds, prepend=elapsed_seconds[0])
    if dt_seconds.size > 1:
        dt_seconds[0] = float(np.median(dt_seconds[1:]))
    elif dt_seconds.size == 1:
        dt_seconds[0] = 0.5
    dt_seconds = np.clip(dt_seconds, 1e-3, None)

    heading_delta_deg = np.array(
        [0.0, *[np.rad2deg(angle_difference(heading_rad[index], heading_rad[index - 1])) for index in range(1, len(timestamps))]],
        dtype=float,
    )
    speed_delta_ms = np.diff(speed_ms, prepend=speed_ms[0])
    turn_rate_deg_s = np.abs(heading_delta_deg) / dt_seconds
    acceleration_ms2 = np.abs(speed_delta_ms) / dt_seconds

    return WindTrack(
        schema_name=schema_name,
        timestamps=timestamps,
        latitude=latitude,
        longitude=longitude,
        speed_ms=speed_ms,
        course_rad=course_rad,
        heading_rad=heading_rad,
        heel_deg=heel_deg,
        trim_deg=trim_deg,
        elapsed_seconds=elapsed_seconds,
        dt_seconds=dt_seconds,
        turn_rate_deg_s=turn_rate_deg_s,
        acceleration_ms2=acceleration_ms2,
    )


def estimate_true_wind(
    track: WindTrack,
    coarse_direction_step_deg: float = 4.0,
    coarse_speed_step_kts: float = 0.5,
    refine_direction_half_span_deg: float = 6.0,
    refine_direction_step_deg: float = 0.25,
    refine_speed_half_span_kts: float = 2.0,
    refine_speed_step_kts: float = 0.1,
    min_true_wind_speed_kts: float = 4.0,
    max_true_wind_speed_kts: float = 20.0,
    coarse_direction_center_deg: float | None = None,
    coarse_direction_half_span_deg: float | None = None,
    prior_direction_deg: float | None = None,
    prior_direction_weight: float = 0.0,
    prior_speed_kts: float | None = None,
    prior_speed_weight: float = 0.0,
) -> WindEstimateResult:
    if track.sample_count == 0:
        raise ValueError("Cannot estimate wind from an empty track")

    base_mask = np.isfinite(track.speed_ms) & np.isfinite(track.heading_rad) & (track.speed_ms >= 0.45 * KNOT_TO_MS)
    base_weights = _base_sample_weights(track) * base_mask.astype(float)
    if not np.any(base_weights > 0.0):
        raise ValueError("No usable sailing samples were found in the track")

    if coarse_direction_center_deg is None or coarse_direction_half_span_deg is None:
        coarse_direction_grid_deg = np.arange(0.0, 360.0, coarse_direction_step_deg, dtype=float)
    else:
        coarse_direction_grid_deg = _wrapped_grid(
            float(coarse_direction_center_deg),
            half_span_deg=float(coarse_direction_half_span_deg),
            step_deg=coarse_direction_step_deg,
        )
    coarse_speed_grid_kts = _inclusive_grid(min_true_wind_speed_kts, max_true_wind_speed_kts, coarse_speed_step_kts)
    coarse_search = _search_true_wind(
        track,
        coarse_direction_grid_deg,
        coarse_speed_grid_kts,
        base_weights,
        prior_direction_deg=prior_direction_deg,
        prior_direction_weight=prior_direction_weight,
        prior_speed_kts=prior_speed_kts,
        prior_speed_weight=prior_speed_weight,
    )

    first_twa_deg = coarse_search["twa_deg"]
    first_prediction_ms = coarse_search["prediction_ms"]
    first_tow_mask = _detect_tow_outliers(
        track,
        first_prediction_ms,
        first_twa_deg,
        float(coarse_search["speed_kts"] * KNOT_TO_MS),
        base_mask,
    )

    refined_weights = base_weights * (~first_tow_mask).astype(float)
    debiased_coarse_search = _search_true_wind(
        track,
        coarse_direction_grid_deg,
        coarse_speed_grid_kts,
        refined_weights,
        prior_direction_deg=prior_direction_deg,
        prior_direction_weight=prior_direction_weight,
        prior_speed_kts=prior_speed_kts,
        prior_speed_weight=prior_speed_weight,
    )
    direction_grid_deg = _wrapped_grid(
        debiased_coarse_search["direction_deg"],
        half_span_deg=refine_direction_half_span_deg,
        step_deg=refine_direction_step_deg,
    )
    min_refined_speed_kts = max(min_true_wind_speed_kts, debiased_coarse_search["speed_kts"] - refine_speed_half_span_kts)
    max_refined_speed_kts = min(max_true_wind_speed_kts, debiased_coarse_search["speed_kts"] + refine_speed_half_span_kts)
    speed_grid_kts = _inclusive_grid(min_refined_speed_kts, max_refined_speed_kts, refine_speed_step_kts)
    refined_search = _search_true_wind(
        track,
        direction_grid_deg,
        speed_grid_kts,
        refined_weights,
        prior_direction_deg=prior_direction_deg,
        prior_direction_weight=prior_direction_weight,
        prior_speed_kts=prior_speed_kts,
        prior_speed_weight=prior_speed_weight,
    )

    predicted_speed_ms = refined_search["prediction_ms"]
    twa_deg = refined_search["twa_deg"]
    tow_outlier_mask = _detect_tow_outliers(
        track,
        predicted_speed_ms,
        twa_deg,
        float(refined_search["speed_kts"] * KNOT_TO_MS),
        base_mask,
    )
    low_confidence_mask = _detect_low_confidence_samples(track, twa_deg, predicted_speed_ms, tow_outlier_mask, base_mask)
    inlier_mask = base_mask & ~tow_outlier_mask & ~low_confidence_mask
    residual_speed_ms = track.speed_ms - predicted_speed_ms

    return WindEstimateResult(
        true_wind_direction_rad=float(np.deg2rad(refined_search["direction_deg"])),
        true_wind_speed_ms=float(refined_search["speed_kts"] * KNOT_TO_MS),
        score=float(refined_search["score"]),
        inlier_mask=inlier_mask,
        tow_outlier_mask=tow_outlier_mask,
        low_confidence_mask=low_confidence_mask,
        predicted_speed_ms=predicted_speed_ms,
        residual_speed_ms=residual_speed_ms,
        twa_deg=twa_deg,
        weights=refined_weights,
        timestamps=track.timestamps,
        observed_speed_ms=track.speed_ms,
        heading_rad=track.heading_rad,
        heel_deg=track.heel_deg,
        trim_deg=track.trim_deg,
    )


def estimate_true_wind_from_csv(path: str | Path, **kwargs: float) -> WindEstimateResult:
    return estimate_true_wind(load_track_csv(path), **kwargs)


def estimate_true_wind_series(
    track: WindTrack,
    source_name: str = "session",
    window_seconds: float = 300.0,
    step_seconds: float = 180.0,
    min_window_samples: int = 180,
    min_inlier_count: int = 40,
    max_direction_step_deg: float = 18.0,
) -> WindTimeSeries:
    if track.sample_count == 0:
        return WindTimeSeries(source_name=source_name, window_seconds=window_seconds, step_seconds=step_seconds, points=[])
    if window_seconds <= 0.0 or step_seconds <= 0.0:
        raise ValueError("window_seconds and step_seconds must be positive")

    global_result = estimate_true_wind(track)
    total_duration = float(track.elapsed_seconds[-1] - track.elapsed_seconds[0])
    if total_duration <= window_seconds:
        centers = np.array([0.5 * (track.elapsed_seconds[0] + track.elapsed_seconds[-1])], dtype=float)
    else:
        start_center = float(track.elapsed_seconds[0] + window_seconds * 0.5)
        end_center = float(track.elapsed_seconds[-1] - window_seconds * 0.5)
        centers = np.arange(start_center, end_center + 1e-6, step_seconds, dtype=float)

    points: list[LocalWindEstimate] = []
    direction_anchor_deg = float(global_result.true_wind_direction_deg)
    speed_anchor_kts = float(global_result.true_wind_speed_kts)

    for center_seconds in centers:
        window_mask = np.abs(track.elapsed_seconds - center_seconds) <= 0.5 * window_seconds
        window_indices = np.flatnonzero(window_mask)
        if window_indices.size < int(min_window_samples):
            continue

        window_track = _subset_track(track, window_indices)
        local_result = estimate_true_wind(
            window_track,
            coarse_direction_step_deg=6.0,
            coarse_speed_step_kts=0.5,
            refine_direction_half_span_deg=max(4.0, 0.5 * max_direction_step_deg),
            refine_direction_step_deg=0.5,
            refine_speed_half_span_kts=1.5,
            refine_speed_step_kts=0.1,
            min_true_wind_speed_kts=max(2.0, speed_anchor_kts - 4.0),
            max_true_wind_speed_kts=min(24.0, speed_anchor_kts + 4.0),
            coarse_direction_center_deg=direction_anchor_deg,
            coarse_direction_half_span_deg=max_direction_step_deg,
            prior_direction_deg=direction_anchor_deg,
            prior_direction_weight=0.00035,
            prior_speed_kts=speed_anchor_kts,
            prior_speed_weight=0.012,
        )
        if local_result.inlier_count < int(min_inlier_count):
            continue

        direction_anchor_deg = float(local_result.true_wind_direction_deg)
        speed_anchor_kts = float(local_result.true_wind_speed_kts)
        representative_index = int(window_indices[np.argmin(np.abs(track.elapsed_seconds[window_indices] - center_seconds))])
        latitude, longitude = _window_representative_position(track, window_indices)
        points.append(
            LocalWindEstimate(
                timestamp=track.timestamps[representative_index],
                elapsed_seconds=float(track.elapsed_seconds[representative_index]),
                latitude=latitude,
                longitude=longitude,
                true_wind_direction_deg=float(local_result.true_wind_direction_deg),
                true_wind_speed_kts=float(local_result.true_wind_speed_kts),
                fit_score=float(local_result.score),
                sample_count=int(window_indices.size),
                inlier_count=int(local_result.inlier_count),
            )
        )

    return WindTimeSeries(
        source_name=source_name,
        window_seconds=float(window_seconds),
        step_seconds=float(step_seconds),
        points=_smooth_wind_time_series(points),
    )


def estimate_true_wind_series_from_csv(path: str | Path, **kwargs: float) -> WindTimeSeries:
    csv_path = Path(path)
    return estimate_true_wind_series(load_track_csv(csv_path), source_name=csv_path.name, **kwargs)


def generate_athlete_specific_polar(
    track: WindTrack,
    result: WindEstimateResult,
    source_name: str = "session",
    twa_centers_deg: np.ndarray | None = None,
    statistic_quantile: float = 0.8,
    min_samples_per_bin: int = 12,
) -> AthletePolar:
    if not 0.0 < float(statistic_quantile) < 1.0:
        raise ValueError("statistic_quantile must be between 0 and 1")

    centers = (
        np.array([35.0, 45.0, 60.0, 75.0, 90.0, 110.0, 120.0, 135.0, 150.0, 160.0, 170.0, 180.0], dtype=float)
        if twa_centers_deg is None
        else np.array(twa_centers_deg, dtype=float)
    )
    edges = _angle_bin_edges_from_centers(centers)
    no_go_twa_deg = float(np.rad2deg(DEFAULT_DEAD_ZONE_ANGLE_RAD))
    athlete_points: list[AthletePolarPoint] = [
        AthletePolarPoint(
            twa_deg=0.0,
            athlete_speed_kts=0.0,
            reference_speed_kts=0.0,
            performance_pct=float("nan"),
            sample_count=0,
        ),
        AthletePolarPoint(
            twa_deg=no_go_twa_deg,
            athlete_speed_kts=0.0,
            reference_speed_kts=0.0,
            performance_pct=float("nan"),
            sample_count=0,
        ),
    ]

    for index, center in enumerate(centers):
        lower_edge = edges[index]
        upper_edge = edges[index + 1]
        if index == len(centers) - 1:
            angle_mask = (result.twa_deg >= lower_edge) & (result.twa_deg <= upper_edge)
        else:
            angle_mask = (result.twa_deg >= lower_edge) & (result.twa_deg < upper_edge)
        sample_mask = result.inlier_mask & angle_mask
        sample_count = int(np.count_nonzero(sample_mask))
        if sample_count < int(min_samples_per_bin):
            continue

        athlete_speed_ms = float(np.quantile(track.speed_ms[sample_mask], statistic_quantile))
        reference_speed_ms = _reference_speed_ms_at_twa(result.true_wind_speed_ms, float(center))
        performance_pct = 0.0 if reference_speed_ms <= 1e-9 else athlete_speed_ms / reference_speed_ms * 100.0
        athlete_points.append(
            AthletePolarPoint(
                twa_deg=float(center),
                athlete_speed_kts=float(athlete_speed_ms / KNOT_TO_MS),
                reference_speed_kts=float(reference_speed_ms / KNOT_TO_MS),
                performance_pct=float(performance_pct),
                sample_count=sample_count,
            )
        )

    athlete_points.sort(key=lambda point: point.twa_deg)

    return AthletePolar(
        source_name=source_name,
        estimated_true_wind_kts=float(result.true_wind_speed_kts),
        statistic_label=f"P{int(round(float(statistic_quantile) * 100.0))} inlier speed",
        points=athlete_points,
    )


def save_athlete_specific_polar_csv(athlete_polar: AthletePolar, output_path: str | Path) -> Path:
    output = Path(output_path)
    fieldnames = [
        "source_name",
        "estimated_true_wind_kts",
        "statistic",
        "twa_deg",
        "athlete_speed_kts",
        "reference_speed_kts",
        "performance_pct",
        "sample_count",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in athlete_polar.rows():
            writer.writerow(row)
    return output


def save_wind_time_series_csv(wind_series: WindTimeSeries, output_path: str | Path) -> Path:
    output = Path(output_path)
    fieldnames = [
        "source_name",
        "window_seconds",
        "step_seconds",
        "timestamp",
        "elapsed_seconds",
        "latitude",
        "longitude",
        "true_wind_direction_deg",
        "true_wind_speed_kts",
        "fit_score",
        "sample_count",
        "inlier_count",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in wind_series.rows():
            writer.writerow(row)
    return output


def build_wind_map_html(
    track: WindTrack,
    result: WindEstimateResult,
    athlete_polar: AthletePolar | None = None,
    wind_series: WindTimeSeries | None = None,
    title: str = "Wind Estimate Map",
) -> str:
    valid_track_indices = np.flatnonzero(np.isfinite(track.latitude) & np.isfinite(track.longitude))
    sample_indices = valid_track_indices[_downsample_indices(len(valid_track_indices), max_points=4000)] if valid_track_indices.size else np.array([], dtype=int)
    track_points = [
        [round(float(track.latitude[index]), 7), round(float(track.longitude[index]), 7)]
        for index in sample_indices
    ]
    outlier_indices = np.flatnonzero(result.tow_outlier_mask)
    outlier_indices = outlier_indices[np.isfinite(track.latitude[outlier_indices]) & np.isfinite(track.longitude[outlier_indices])]
    outlier_sample = outlier_indices[_downsample_indices(len(outlier_indices), max_points=800)] if outlier_indices.size else np.array([], dtype=int)
    tow_outliers = [
        {
            "lat": round(float(track.latitude[index]), 7),
            "lon": round(float(track.longitude[index]), 7),
            "timestamp": result.timestamps[index],
            "speed_kts": round(float(result.observed_speed_ms[index] / KNOT_TO_MS), 2),
            "predicted_kts": round(float(result.predicted_speed_ms[index] / KNOT_TO_MS), 2),
            "heading_deg": round(float(np.rad2deg(result.heading_rad[index]) % 360.0), 1),
        }
        for index in outlier_sample
    ]
    ranked_outlier_indices = outlier_indices[np.argsort(result.residual_speed_ms[outlier_indices])[::-1]][:5]
    top_outliers = [
        {
            "timestamp": result.timestamps[index],
            "speed_kts": round(float(result.observed_speed_ms[index] / KNOT_TO_MS), 2),
            "predicted_kts": round(float(result.predicted_speed_ms[index] / KNOT_TO_MS), 2),
            "heading_deg": round(float(np.rad2deg(result.heading_rad[index]) % 360.0), 1),
        }
        for index in ranked_outlier_indices
    ]

    summary = result.summary_dict()
    blowing_to_deg = round((summary["true_wind_direction_deg"] + 180.0) % 360.0, 2)
    athlete_polar_rows = [] if athlete_polar is None else [
        {
            "twaDeg": round(point.twa_deg, 1),
            "athleteSpeedKts": round(point.athlete_speed_kts, 2),
            "referenceSpeedKts": round(point.reference_speed_kts, 2),
            "performancePct": None if not np.isfinite(point.performance_pct) else round(point.performance_pct, 1),
            "sampleCount": int(point.sample_count),
        }
        for point in athlete_polar.points
    ]
    local_wind_rows = [] if wind_series is None else [
        {
            "timestamp": point.timestamp,
            "lat": round(point.latitude, 7),
            "lon": round(point.longitude, 7),
            "windFromDeg": round(point.true_wind_direction_deg, 2),
            "windToDeg": round((point.true_wind_direction_deg + 180.0) % 360.0, 2),
            "windSpeedKts": round(point.true_wind_speed_kts, 2),
            "fitScore": round(point.fit_score, 6),
            "sampleCount": int(point.sample_count),
            "inlierCount": int(point.inlier_count),
        }
        for point in wind_series.points
        if np.isfinite(point.latitude) and np.isfinite(point.longitude)
    ]
    payload = {
        "title": title,
        "summary": summary,
        "trackPoints": track_points,
        "towOutliers": tow_outliers,
        "topOutliers": top_outliers,
        "startPoint": track_points[0] if track_points else None,
        "finishPoint": track_points[-1] if track_points else None,
        "windFromDeg": summary["true_wind_direction_deg"],
        "windToDeg": blowing_to_deg,
        "trackSchema": track.schema_name,
        "athletePolar": athlete_polar_rows,
        "athletePolarLabel": None if athlete_polar is None else athlete_polar.statistic_label,
        "athletePolarTwsKts": None if athlete_polar is None else round(athlete_polar.estimated_true_wind_kts, 2),
        "localWind": local_wind_rows,
        "windSeriesWindowSeconds": None if wind_series is None else round(wind_series.window_seconds, 1),
        "windSeriesStepSeconds": None if wind_series is None else round(wind_series.step_seconds, 1),
    }
    data_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="preconnect" href="https://unpkg.com">
  <link rel="preconnect" href="https://tile.openstreetmap.org">
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  >
  <style>
    :root {{
      --ink: #17313f;
      --sea: #0f5b73;
      --foam: #f5efe2;
      --sand: #dcc9a3;
      --wind: #d95d39;
      --mist: #e8f1f2;
      --shadow: rgba(23, 49, 63, 0.18);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(217, 93, 57, 0.16), transparent 32%),
        linear-gradient(135deg, #f8f4eb 0%, #eef4f2 55%, #e5efe7 100%);
    }}

    .layout {{
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
      min-height: 100vh;
      gap: 18px;
      padding: 18px;
    }}

    .panel {{
      border-radius: 24px;
      padding: 22px 20px;
      background: rgba(255, 250, 241, 0.92);
      box-shadow: 0 18px 45px var(--shadow);
      backdrop-filter: blur(10px);
    }}

    .sidebar {{
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}

    .eyebrow {{
      margin: 0 0 6px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.75rem;
      color: rgba(23, 49, 63, 0.65);
    }}

    h1, h2 {{
      margin: 0;
      font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
      font-weight: 700;
      line-height: 1.05;
    }}

    h1 {{
      font-size: clamp(1.8rem, 3vw, 2.6rem);
    }}

    h2 {{
      font-size: 1rem;
      margin-bottom: 10px;
      letter-spacing: 0.03em;
    }}

    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}

    .metric {{
      border-radius: 18px;
      padding: 14px 12px;
      background: linear-gradient(160deg, rgba(15, 91, 115, 0.12), rgba(255, 255, 255, 0.85));
      border: 1px solid rgba(15, 91, 115, 0.1);
    }}

    .metric-label {{
      display: block;
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(23, 49, 63, 0.66);
      margin-bottom: 6px;
    }}

    .metric-value {{
      font-size: 1.25rem;
      font-weight: 700;
    }}

    .metric-note {{
      font-size: 0.86rem;
      color: rgba(23, 49, 63, 0.78);
      margin-top: 2px;
    }}

    .legend-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      font-size: 0.92rem;
    }}

    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
    }}

    .outlier-list {{
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 10px;
      font-size: 0.92rem;
    }}

    .polar-card {{
      display: grid;
      gap: 12px;
    }}

    .polar-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 0.88rem;
      color: rgba(23, 49, 63, 0.8);
    }}

    .polar-line-key {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}

    .polar-line-key::before {{
      content: "";
      width: 22px;
      height: 3px;
      border-radius: 999px;
      background: currentColor;
    }}

    .polar-line-key.reference {{ color: #0f5b73; }}
    .polar-line-key.athlete {{ color: #d95d39; }}

    .polar-svg {{
      width: 100%;
      height: auto;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(15, 91, 115, 0.08), rgba(255, 255, 255, 0.86));
      border: 1px solid rgba(15, 91, 115, 0.12);
    }}

    .polar-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}

    .polar-table th,
    .polar-table td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid rgba(23, 49, 63, 0.08);
    }}

    .polar-table th {{
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(23, 49, 63, 0.64);
    }}

    .map-shell {{
      position: relative;
      overflow: hidden;
      border-radius: 28px;
      box-shadow: 0 22px 55px var(--shadow);
      min-height: 78vh;
      background: linear-gradient(180deg, rgba(15, 91, 115, 0.18), rgba(232, 241, 242, 0.9));
    }}

    #map {{
      position: absolute;
      inset: 0;
    }}

    .map-caption {{
      position: absolute;
      left: 18px;
      right: 18px;
      bottom: 18px;
      z-index: 600;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-end;
      pointer-events: none;
    }}

    .caption-card {{
      max-width: min(420px, 100%);
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(255, 250, 241, 0.92);
      box-shadow: 0 12px 32px rgba(23, 49, 63, 0.18);
      backdrop-filter: blur(10px);
    }}

    .caption-title {{
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: rgba(23, 49, 63, 0.68);
      margin-bottom: 4px;
    }}

    .caption-strong {{
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 4px;
    }}

    .caption-copy {{
      font-size: 0.92rem;
      line-height: 1.4;
      color: rgba(23, 49, 63, 0.84);
    }}

    .wind-control {{
      width: 182px;
      padding: 14px;
      border-radius: 18px;
      background: rgba(255, 250, 241, 0.94);
      box-shadow: 0 12px 36px rgba(23, 49, 63, 0.22);
      backdrop-filter: blur(10px);
      color: var(--ink);
    }}

    .wind-control h3 {{
      margin: 0 0 10px;
      font-size: 0.88rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    .compass {{
      position: relative;
      width: 132px;
      height: 132px;
      margin: 0 auto 10px;
      border-radius: 999px;
      border: 2px solid rgba(23, 49, 63, 0.15);
      background:
        radial-gradient(circle at center, rgba(255, 255, 255, 0.95) 0 38%, transparent 39%),
        conic-gradient(from 0deg, rgba(15, 91, 115, 0.04), rgba(15, 91, 115, 0.12), rgba(15, 91, 115, 0.04));
    }}

    .compass::before,
    .compass::after {{
      content: "";
      position: absolute;
      inset: 50%;
      background: rgba(23, 49, 63, 0.18);
      transform: translate(-50%, -50%);
    }}

    .compass::before {{
      width: 2px;
      height: 100px;
    }}

    .compass::after {{
      width: 100px;
      height: 2px;
    }}

    .cardinal {{
      position: absolute;
      font-size: 0.76rem;
      font-weight: 700;
      color: rgba(23, 49, 63, 0.72);
    }}

    .cardinal.n {{ top: 6px; left: 50%; transform: translateX(-50%); }}
    .cardinal.e {{ right: 9px; top: 50%; transform: translateY(-50%); }}
    .cardinal.s {{ bottom: 6px; left: 50%; transform: translateX(-50%); }}
    .cardinal.w {{ left: 9px; top: 50%; transform: translateY(-50%); }}

    .wind-arrow {{
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      transform: rotate(calc(var(--wind-to-deg) * 1deg));
      transition: transform 240ms ease;
    }}

    .wind-arrow svg {{
      overflow: visible;
      filter: drop-shadow(0 4px 8px rgba(217, 93, 57, 0.22));
    }}

    .wind-copy {{
      text-align: center;
      font-size: 0.92rem;
      line-height: 1.35;
    }}

    .wind-copy strong {{
      display: block;
      font-size: 1.08rem;
      margin-bottom: 2px;
    }}

    .local-wind-marker {{
      width: 22px;
      height: 22px;
      border-radius: 999px;
      background: rgba(255, 250, 241, 0.92);
      border: 2px solid rgba(23, 49, 63, 0.18);
      box-shadow: 0 8px 18px rgba(23, 49, 63, 0.2);
      display: grid;
      place-items: center;
    }}

    .local-wind-glyph {{
      width: 0;
      height: 0;
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      border-bottom: 12px solid #d95d39;
      transform: rotate(calc(var(--wind-to-deg) * 1deg));
      transform-origin: 50% 85%;
    }}

    .leaflet-popup-content {{
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      line-height: 1.4;
    }}

    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}

      .map-shell {{
        min-height: 62vh;
      }}

      .map-caption {{
        flex-direction: column;
        align-items: stretch;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <section class="panel">
        <p class="eyebrow">Wind Fit</p>
        <h1>{title}</h1>
        <div class="summary-grid">
          <div class="metric">
            <span class="metric-label">Wind From</span>
            <div class="metric-value">{summary["true_wind_direction_deg"]:.2f} deg</div>
            <div class="metric-note">True bearing, same frame as the boat heading column used in the CSV</div>
          </div>
          <div class="metric">
            <span class="metric-label">Wind Speed</span>
            <div class="metric-value">{summary["true_wind_speed_kts"]:.2f} kts</div>
            <div class="metric-note">Best-fit constant true wind</div>
          </div>
          <div class="metric">
            <span class="metric-label">Inliers</span>
            <div class="metric-value">{summary["inlier_count"]}</div>
            <div class="metric-note">Samples used in the final fit</div>
          </div>
          <div class="metric">
            <span class="metric-label">Tow Outliers</span>
            <div class="metric-value">{summary["tow_outlier_count"]}</div>
            <div class="metric-note">Rows that exceed the polar by too much</div>
          </div>
        </div>
        <div class="legend-row"><span class="legend-swatch" style="background:#0f5b73"></span> Sailing track</div>
        <div class="legend-row"><span class="legend-swatch" style="background:#d95d39"></span> Likely tow / impossible overspeed points</div>
      </section>

      <section class="panel">
        <h2>How The Degree Is Calculated</h2>
        <div class="caption-copy">
          Each candidate wind direction is treated as a true compass bearing that the wind is coming from.
          For every CSV row we compute <code>TWA = wind_from - heading</code>, wrap it to the circle,
          compare the observed boat speed to the ILCA 7 polar at that true-wind angle and speed, then pick the
          direction and wind speed that minimize the weighted error after rejecting obvious overspeed outliers.
        </div>
        <div class="metric-note" id="local-wind-note"></div>
      </section>

      <section class="panel">
        <h2>Largest Overspeed Rows</h2>
        <ol class="outlier-list" id="top-outliers"></ol>
      </section>

      <section class="panel">
        <h2>Athlete Specific Polar</h2>
        <div class="caption-copy">
          Generated from the fitted wind and the inlier sailing samples in this session.
          The athlete curve uses the best representative inlier pace in each true-wind-angle bin.
        </div>
        <div class="polar-card">
          <div class="polar-legend">
            <span class="polar-line-key reference">Reference polar</span>
            <span class="polar-line-key athlete">Athlete session polar</span>
          </div>
          <svg id="polar-chart" class="polar-svg" viewBox="0 0 420 420" aria-label="Athlete polar chart"></svg>
          <div class="metric-note" id="athlete-polar-note"></div>
          <table class="polar-table">
            <thead>
              <tr>
                <th>TWA</th>
                <th>Athlete</th>
                <th>Reference</th>
                <th>Perf</th>
                <th>Samples</th>
              </tr>
            </thead>
            <tbody id="athlete-polar-body"></tbody>
          </table>
        </div>
      </section>
    </aside>

    <section class="map-shell">
      <div id="map"></div>
      <div class="map-caption">
        <div class="caption-card">
          <div class="caption-title">Interpretation</div>
          <div class="caption-strong">Wind from {summary["true_wind_direction_deg"]:.2f} deg, blowing toward {blowing_to_deg:.2f} deg</div>
          <div class="caption-copy">
            The orange arrow points where the air is flowing to. The label still uses the sailing convention:
            the fitted number is the direction the wind is coming from.
          </div>
        </div>
      </div>
    </section>
  </div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const payload = {data_json};

    const map = L.map("map", {{
      zoomControl: false,
      preferCanvas: true
    }});
    L.control.zoom({{ position: "bottomright" }}).addTo(map);
    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);

    const track = payload.trackPoints.map((point) => [point[0], point[1]]);
    if (track.length > 0) {{
      const trackLine = L.polyline(track, {{
        color: "#0f5b73",
        weight: 4,
        opacity: 0.92
      }}).addTo(map);
      map.fitBounds(trackLine.getBounds().pad(0.12));

      if (payload.startPoint) {{
        L.circleMarker(payload.startPoint, {{
          radius: 6,
          weight: 2,
          color: "#17313f",
          fillColor: "#f5efe2",
          fillOpacity: 1
        }}).bindTooltip("Start").addTo(map);
      }}

      if (payload.finishPoint) {{
        L.circleMarker(payload.finishPoint, {{
          radius: 6,
          weight: 2,
          color: "#17313f",
          fillColor: "#dcc9a3",
          fillOpacity: 1
        }}).bindTooltip("Finish").addTo(map);
      }}
    }} else {{
      map.setView([0, 0], 2);
    }}

    payload.towOutliers.forEach((row) => {{
      L.circleMarker([row.lat, row.lon], {{
        radius: 4,
        color: "#d95d39",
        weight: 1,
        fillColor: "#d95d39",
        fillOpacity: 0.7
      }})
        .bindPopup(
          `<strong>${{row.timestamp}}</strong><br>` +
          `Speed: ${{row.speed_kts.toFixed(2)}} kts<br>` +
          `Polar fit: ${{row.predicted_kts.toFixed(2)}} kts<br>` +
          `Heading: ${{row.heading_deg.toFixed(1)}} deg`
        )
        .addTo(map);
    }});

    payload.localWind.forEach((row) => {{
      const icon = L.divIcon({{
        className: '',
        html: `<div class="local-wind-marker"><div class="local-wind-glyph" style="--wind-to-deg:${{row.windToDeg}}"></div></div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11]
      }});
      L.marker([row.lat, row.lon], {{ icon }})
        .bindPopup(
          `<strong>${{row.timestamp}}</strong><br>` +
          `Local wind from: ${{row.windFromDeg.toFixed(1)}} deg<br>` +
          `Local wind speed: ${{row.windSpeedKts.toFixed(2)}} kts<br>` +
          `Inliers: ${{row.inlierCount}} / ${{row.sampleCount}}`
        )
        .addTo(map);
    }});

    const windControl = L.control({{ position: "topright" }});
    windControl.onAdd = function () {{
      const container = L.DomUtil.create("div", "wind-control");
      container.innerHTML = `
        <h3>Likely Wind</h3>
        <div class="compass" style="--wind-to-deg:${{payload.windToDeg}};">
          <span class="cardinal n">N</span>
          <span class="cardinal e">E</span>
          <span class="cardinal s">S</span>
          <span class="cardinal w">W</span>
          <div class="wind-arrow" aria-hidden="true">
            <svg width="18" height="92" viewBox="0 0 18 92" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9 4 L17 20 H11 V78 H7 V20 H1 Z" fill="#d95d39"/>
              <circle cx="9" cy="80" r="5.5" fill="#17313f" fill-opacity="0.92"/>
            </svg>
          </div>
        </div>
        <div class="wind-copy">
          <strong>Wind from ${{payload.windFromDeg.toFixed(2)}} deg</strong>
          Flowing toward ${{payload.windToDeg.toFixed(2)}} deg
        </div>
      `;
      return container;
    }};
    windControl.addTo(map);

    const topOutliers = document.getElementById("top-outliers");
    if (payload.topOutliers.length === 0) {{
      topOutliers.innerHTML = "<li>No tow-like overspeed rows were flagged.</li>";
    }} else {{
      payload.topOutliers.forEach((row) => {{
        const item = document.createElement("li");
        item.textContent = `${{row.timestamp}} | speed ${{row.speed_kts.toFixed(2)}} kts vs polar ${{row.predicted_kts.toFixed(2)}} kts | heading ${{row.heading_deg.toFixed(1)}} deg`;
        topOutliers.appendChild(item);
      }});
    }}

    const localWindNote = document.getElementById("local-wind-note");
    if (payload.localWind.length > 0) {{
      localWindNote.textContent = `Local wind arrows are estimated every ${{payload.windSeriesStepSeconds.toFixed(0)}} s using ${{payload.windSeriesWindowSeconds.toFixed(0)}} s windows.`;
    }} else {{
      localWindNote.textContent = 'This report only shows the session-wide wind estimate.';
    }}

    const athletePolarBody = document.getElementById("athlete-polar-body");
    const athletePolarNote = document.getElementById("athlete-polar-note");
    if (payload.athletePolar.length === 0) {{
      athletePolarBody.innerHTML = '<tr><td colspan="5">Not enough clean samples to build an athlete-specific polar from this file.</td></tr>';
      athletePolarNote.textContent = 'No athlete-polar curve was generated.';
    }} else {{
      athletePolarNote.textContent = `${{payload.athletePolarLabel}} at estimated TWS ${{payload.athletePolarTwsKts.toFixed(2)}} kts`;
      payload.athletePolar.forEach((row) => {{
        const tableRow = document.createElement("tr");
        const performanceText = row.performancePct === null ? 'impossible angle' : `${{row.performancePct.toFixed(0)}}%`;
        tableRow.innerHTML =
          `<td>${{row.twaDeg.toFixed(0)}} deg</td>` +
          `<td>${{row.athleteSpeedKts.toFixed(2)}} kts</td>` +
          `<td>${{row.referenceSpeedKts.toFixed(2)}} kts</td>` +
          `<td>${{performanceText}}</td>` +
          `<td>${{row.sampleCount}}</td>`;
        athletePolarBody.appendChild(tableRow);
      }});
    }}

    const polarSvg = document.getElementById("polar-chart");
    const drawPolarChart = () => {{
      const rows = payload.athletePolar;
      const width = 420;
      const height = 420;
      const centerX = width / 2;
      const centerY = height / 2;
      const outerRadius = 168;
      const maxSpeed = Math.max(
        1,
        ...rows.flatMap((row) => [row.athleteSpeedKts, row.referenceSpeedKts])
      );
      const speedCeiling = Math.ceil(maxSpeed * 1.15);
      const radius = (speed) => (speed / speedCeiling) * outerRadius;
      const polarPoint = (twaDeg, speedKts, tackSign) => {{
        const angleRad = ((tackSign * twaDeg) - 90) * Math.PI / 180;
        const r = radius(speedKts);
        return {{
          x: centerX + r * Math.cos(angleRad),
          y: centerY + r * Math.sin(angleRad)
        }};
      }};

      const buildMirroredPath = (speedField) => {{
        const port = [...rows].slice().reverse().map((row) => polarPoint(row.twaDeg, row[speedField], -1));
        const starboard = rows.map((row) => polarPoint(row.twaDeg, row[speedField], 1));
        const points = [...port, ...starboard];
        return points.map((point, index) => `${{index === 0 ? 'M' : 'L'}}${{point.x.toFixed(2)}},${{point.y.toFixed(2)}}`).join(" ") + " Z";
      }};

      const radialTicks = [0.25, 0.5, 0.75, 1.0].map((fraction) => Number((speedCeiling * fraction).toFixed(1)));
      const spokeAngles = [0, 30, 60, 90, 120, 150, 180];
      const gridLines = radialTicks.map((tick) =>
        `<circle cx="${{centerX}}" cy="${{centerY}}" r="${{radius(tick)}}" fill="none" stroke="rgba(23,49,63,0.08)" />`
      ).join("") + spokeAngles.flatMap((angle) => {{
        const starboard = polarPoint(angle, speedCeiling, 1);
        const port = polarPoint(angle, speedCeiling, -1);
        return [
          `<line x1="${{centerX}}" y1="${{centerY}}" x2="${{starboard.x}}" y2="${{starboard.y}}" stroke="rgba(23,49,63,0.08)" />`,
          `<line x1="${{centerX}}" y1="${{centerY}}" x2="${{port.x}}" y2="${{port.y}}" stroke="rgba(23,49,63,0.08)" />`
        ];
      }}).join("");

      const refPath = buildMirroredPath("referenceSpeedKts");
      const athletePath = buildMirroredPath("athleteSpeedKts");

      const athletePoints = rows.flatMap((row) => {{
        const starboard = polarPoint(row.twaDeg, row.athleteSpeedKts, 1);
        const port = polarPoint(row.twaDeg, row.athleteSpeedKts, -1);
        return [
          `<circle cx="${{starboard.x.toFixed(2)}}" cy="${{starboard.y.toFixed(2)}}" r="3.4" fill="#d95d39" />`,
          `<circle cx="${{port.x.toFixed(2)}}" cy="${{port.y.toFixed(2)}}" r="3.4" fill="#d95d39" />`
        ];
      }}).join("");

      const angleLabels = spokeAngles.flatMap((angle) => {{
        const starboard = polarPoint(angle, speedCeiling + speedCeiling * 0.08, 1);
        const port = polarPoint(angle, speedCeiling + speedCeiling * 0.08, -1);
        return [
          `<text x="${{starboard.x.toFixed(2)}}" y="${{starboard.y.toFixed(2)}}" text-anchor="middle" font-size="11" fill="rgba(23,49,63,0.72)">${{angle}}</text>`,
          angle === 0 || angle === 180 ? "" : `<text x="${{port.x.toFixed(2)}}" y="${{port.y.toFixed(2)}}" text-anchor="middle" font-size="11" fill="rgba(23,49,63,0.72)">${{angle}}</text>`
        ];
      }}).join("");

      const radialLabels = radialTicks.map((tick) =>
        `<text x="${{centerX + 6}}" y="${{(centerY - radius(tick)).toFixed(2) - 4}}" font-size="11" fill="rgba(23,49,63,0.72)">${{tick}}</text>`
      ).join("");

      polarSvg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="18" ry="18" fill="transparent" />
        ${{gridLines}}
        <circle cx="${{centerX}}" cy="${{centerY}}" r="${{outerRadius}}" fill="none" stroke="rgba(23,49,63,0.18)" />
        <path d="${{refPath}}" fill="rgba(15,91,115,0.08)" stroke="#0f5b73" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
        <path d="${{athletePath}}" fill="rgba(217,93,57,0.10)" stroke="#d95d39" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
        ${{athletePoints}}
        ${{angleLabels}}
        ${{radialLabels}}
        <text x="${{centerX}}" y="20" text-anchor="middle" font-size="12" fill="rgba(23,49,63,0.78)">0 deg TWA / into the wind</text>
        <text x="${{centerX}}" y="${{height - 10}}" text-anchor="middle" font-size="12" fill="rgba(23,49,63,0.78)">180 deg TWA / dead downwind</text>
        <text x="${{centerX + outerRadius + 14}}" y="${{centerY + 4}}" font-size="11" fill="rgba(23,49,63,0.78)">Starboard</text>
        <text x="${{centerX - outerRadius - 14}}" y="${{centerY + 4}}" text-anchor="end" font-size="11" fill="rgba(23,49,63,0.78)">Port</text>
      `;
    }};

    if (payload.athletePolar.length > 0) {{
      drawPolarChart();
    }} else {{
      polarSvg.innerHTML = '<text x="210" y="210" text-anchor="middle" font-size="14" fill="rgba(23,49,63,0.7)">Not enough athlete-polar points to draw a curve.</text>';
    }}
  </script>
</body>
</html>
"""


def save_wind_map_html(
    track: WindTrack,
    result: WindEstimateResult,
    athlete_polar: AthletePolar | None,
    wind_series: WindTimeSeries | None,
    output_path: str | Path,
    title: str = "Wind Estimate Map",
) -> Path:
    output = Path(output_path)
    output.write_text(
        build_wind_map_html(track, result, athlete_polar=athlete_polar, wind_series=wind_series, title=title),
        encoding="utf-8",
    )
    return output


def build_wind_map_from_csv(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    **kwargs: float,
) -> tuple[Path, WindEstimateResult, AthletePolar, WindTimeSeries]:
    csv_file = Path(csv_path)
    track = load_track_csv(csv_file)
    result = estimate_true_wind(track, **kwargs)
    athlete_polar = generate_athlete_specific_polar(track, result, source_name=csv_file.name)
    wind_series = estimate_true_wind_series(track, source_name=csv_file.name)
    html_title = title or f"Wind Estimate for {csv_file.name}"
    output = Path(output_path) if output_path is not None else csv_file.with_name(f"{csv_file.stem} wind map.html")
    save_wind_map_html(track, result, athlete_polar, wind_series, output, title=html_title)
    return output, result, athlete_polar, wind_series


def format_estimate_report(result: WindEstimateResult, top_outliers: int = 5) -> str:
    summary = result.summary_dict()
    lines = [
        f"Estimated true wind direction: {summary['true_wind_direction_deg']:.2f} deg",
        f"Estimated true wind speed: {summary['true_wind_speed_kts']:.2f} kts",
        f"Fit score: {summary['fit_score']}",
        (
            f"Samples: {summary['sample_count']} total, {summary['inlier_count']} inliers, "
            f"{summary['tow_outlier_count']} likely towing/overspeed outliers, "
            f"{summary['low_confidence_count']} low-confidence samples"
        ),
        (
            f"Inlier speed error: mean {summary['mean_absolute_error_kts']:.3f} kts, "
            f"median {summary['median_absolute_error_kts']:.3f} kts"
        ),
    ]

    outlier_indices = np.flatnonzero(result.tow_outlier_mask)
    if outlier_indices.size:
        lines.append("Largest likely towing/overspeed rows:")
        ranked_indices = outlier_indices[np.argsort(result.residual_speed_ms[outlier_indices])[::-1]][:top_outliers]
        for index in ranked_indices:
            lines.append(
                "  "
                + (
                    f"{result.timestamps[index]} speed={result.observed_speed_ms[index] / KNOT_TO_MS:.2f}kts "
                    f"pred={result.predicted_speed_ms[index] / KNOT_TO_MS:.2f}kts "
                    f"heading={np.rad2deg(result.heading_rad[index]) % 360.0:.1f}deg "
                    f"heel={result.heel_deg[index]:.1f} trim={result.trim_deg[index]:.1f}"
                )
            )
    return "\n".join(lines)


def _parse_timestamp(value: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value!r}")


def _format_timestamp(value: datetime) -> str:
    formatted = value.isoformat()
    return formatted.replace("+00:00", "Z")


def _subset_track(track: WindTrack, indices: np.ndarray) -> WindTrack:
    subset_indices = np.array(indices, dtype=int)
    elapsed_seconds = track.elapsed_seconds[subset_indices] - track.elapsed_seconds[subset_indices[0]]
    dt_seconds = np.array(track.dt_seconds[subset_indices], dtype=float, copy=True)
    if dt_seconds.size > 1:
        dt_seconds[0] = float(np.median(dt_seconds[1:]))
    elif dt_seconds.size == 1:
        dt_seconds[0] = max(float(dt_seconds[0]), 1e-3)
    dt_seconds = np.clip(dt_seconds, 1e-3, None)
    return WindTrack(
        schema_name=track.schema_name,
        timestamps=[track.timestamps[index] for index in subset_indices.tolist()],
        latitude=track.latitude[subset_indices].copy(),
        longitude=track.longitude[subset_indices].copy(),
        speed_ms=track.speed_ms[subset_indices].copy(),
        course_rad=track.course_rad[subset_indices].copy(),
        heading_rad=track.heading_rad[subset_indices].copy(),
        heel_deg=track.heel_deg[subset_indices].copy(),
        trim_deg=track.trim_deg[subset_indices].copy(),
        elapsed_seconds=elapsed_seconds,
        dt_seconds=dt_seconds,
        turn_rate_deg_s=track.turn_rate_deg_s[subset_indices].copy(),
        acceleration_ms2=track.acceleration_ms2[subset_indices].copy(),
    )


def _window_representative_position(track: WindTrack, indices: np.ndarray) -> tuple[float, float]:
    latitudes = track.latitude[indices]
    longitudes = track.longitude[indices]
    valid = np.isfinite(latitudes) & np.isfinite(longitudes)
    if not np.any(valid):
        return float("nan"), float("nan")
    return float(np.median(latitudes[valid])), float(np.median(longitudes[valid]))


def _smooth_wind_time_series(points: list[LocalWindEstimate], span: int = 3) -> list[LocalWindEstimate]:
    if len(points) <= 2 or span <= 1:
        return points
    half_span = span // 2
    smoothed_points: list[LocalWindEstimate] = []
    for index, point in enumerate(points):
        start = max(0, index - half_span)
        stop = min(len(points), index + half_span + 1)
        window = points[start:stop]
        directions_rad = np.deg2rad([item.true_wind_direction_deg for item in window])
        mean_sin = float(np.mean(np.sin(directions_rad)))
        mean_cos = float(np.mean(np.cos(directions_rad)))
        smoothed_direction_deg = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0)
        smoothed_speed_kts = float(np.median([item.true_wind_speed_kts for item in window]))
        smoothed_points.append(
            LocalWindEstimate(
                timestamp=point.timestamp,
                elapsed_seconds=point.elapsed_seconds,
                latitude=point.latitude,
                longitude=point.longitude,
                true_wind_direction_deg=smoothed_direction_deg,
                true_wind_speed_kts=smoothed_speed_kts,
                fit_score=point.fit_score,
                sample_count=point.sample_count,
                inlier_count=point.inlier_count,
            )
        )
    return smoothed_points


def _parse_required_float(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing required numeric value for {key!r}")
    return float(value)


def _parse_optional_float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw_value = str(row.get(key, "")).strip()
        values.append(float(raw_value) if raw_value else float("nan"))
    return np.array(values, dtype=float)


def _forward_fill_numeric(values: np.ndarray, fill_value: float | None = None) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    valid = np.flatnonzero(np.isfinite(result))
    if valid.size == 0:
        if fill_value is None:
            return result
        return np.full(result.shape, float(fill_value), dtype=float)

    first_index = int(valid[0])
    if fill_value is None:
        result[:first_index] = result[first_index]
    else:
        result[:first_index] = float(fill_value)

    for index in range(first_index + 1, result.size):
        if not np.isfinite(result[index]):
            result[index] = result[index - 1]
    return result


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(values, dtype=float), 360.0)


def _course_from_positions(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    course_deg = np.full(latitude.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(latitude) & np.isfinite(longitude))
    if valid_indices.size < 2:
        return course_deg

    for previous_index, current_index in zip(valid_indices[:-1], valid_indices[1:]):
        bearing_deg = _bearing_between_points(
            float(latitude[previous_index]),
            float(longitude[previous_index]),
            float(latitude[current_index]),
            float(longitude[current_index]),
        )
        course_deg[previous_index:current_index + 1] = bearing_deg
    return course_deg


def _sanitize_speed_column(speed_ms: np.ndarray, max_reasonable_speed_ms: float = 15.0) -> np.ndarray:
    sanitized = np.array(speed_ms, dtype=float, copy=True)
    invalid_mask = ~np.isfinite(sanitized) | (sanitized < 0.0) | (sanitized > max_reasonable_speed_ms)
    sanitized[invalid_mask] = 0.0
    return sanitized


def _sanitize_gps_positions(
    latitude: np.ndarray,
    longitude: np.ndarray,
    elapsed_seconds: np.ndarray,
    max_implied_speed_ms: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    sanitized_lat = np.array(latitude, dtype=float, copy=True)
    sanitized_lon = np.array(longitude, dtype=float, copy=True)
    bounds_mask = (
        np.isfinite(sanitized_lat)
        & np.isfinite(sanitized_lon)
        & (sanitized_lat >= -90.0)
        & (sanitized_lat <= 90.0)
        & (sanitized_lon >= -180.0)
        & (sanitized_lon <= 180.0)
    )
    sanitized_lat[~bounds_mask] = np.nan
    sanitized_lon[~bounds_mask] = np.nan

    changed = True
    while changed:
        changed = False
        valid_indices = np.flatnonzero(np.isfinite(sanitized_lat) & np.isfinite(sanitized_lon))
        if valid_indices.size < 3:
            break
        for idx in range(1, valid_indices.size - 1):
            prev_index = int(valid_indices[idx - 1])
            current_index = int(valid_indices[idx])
            next_index = int(valid_indices[idx + 1])
            prev_dt = float(elapsed_seconds[current_index] - elapsed_seconds[prev_index])
            next_dt = float(elapsed_seconds[next_index] - elapsed_seconds[current_index])
            bridge_dt = float(elapsed_seconds[next_index] - elapsed_seconds[prev_index])
            if prev_dt <= 0.0 or next_dt <= 0.0 or bridge_dt <= 0.0:
                continue

            prev_speed = _distance_between_points_m(
                sanitized_lat[prev_index],
                sanitized_lon[prev_index],
                sanitized_lat[current_index],
                sanitized_lon[current_index],
            ) / prev_dt
            next_speed = _distance_between_points_m(
                sanitized_lat[current_index],
                sanitized_lon[current_index],
                sanitized_lat[next_index],
                sanitized_lon[next_index],
            ) / next_dt
            bridge_speed = _distance_between_points_m(
                sanitized_lat[prev_index],
                sanitized_lon[prev_index],
                sanitized_lat[next_index],
                sanitized_lon[next_index],
            ) / bridge_dt

            if prev_speed > max_implied_speed_ms and next_speed > max_implied_speed_ms and bridge_speed <= max_implied_speed_ms:
                sanitized_lat[current_index] = np.nan
                sanitized_lon[current_index] = np.nan
                changed = True
                break

    return sanitized_lat, sanitized_lon


def _bearing_between_points(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    lat1_rad = np.deg2rad(lat1_deg)
    lat2_rad = np.deg2rad(lat2_deg)
    delta_lon_rad = np.deg2rad(lon2_deg - lon1_deg)
    x = np.sin(delta_lon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon_rad)
    return float(np.rad2deg(np.arctan2(x, y)) % 360.0)


def _distance_between_points_m(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    earth_radius_m = 6371000.0
    lat1_rad = np.deg2rad(lat1_deg)
    lat2_rad = np.deg2rad(lat2_deg)
    delta_lat_rad = np.deg2rad(lat2_deg - lat1_deg)
    delta_lon_rad = np.deg2rad(lon2_deg - lon1_deg)
    a = (
        np.sin(delta_lat_rad / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon_rad / 2.0) ** 2
    )
    return float(2.0 * earth_radius_m * np.arctan2(np.sqrt(a), np.sqrt(max(0.0, 1.0 - a))))


def _angle_bin_edges_from_centers(centers_deg: np.ndarray) -> np.ndarray:
    centers = np.array(centers_deg, dtype=float)
    if centers.size == 0:
        raise ValueError("At least one angle center is required")
    if centers.size == 1:
        return np.array([max(0.0, centers[0] - 5.0), min(180.0, centers[0] + 5.0)], dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    lower_edge = max(0.0, centers[0] - 0.5 * (centers[1] - centers[0]))
    upper_edge = min(180.0, centers[-1] + 0.5 * (centers[-1] - centers[-2]))
    return np.concatenate([[lower_edge], mids, [upper_edge]]).astype(float)


def _reference_speed_ms_at_twa(true_wind_speed_ms: float, twa_deg: float) -> float:
    speed_by_tws = _interpolate_speed_by_angle(np.array([float(twa_deg)], dtype=float))
    low_index, high_index, alpha = _prepare_tws_interpolation(np.array([float(true_wind_speed_ms) / KNOT_TO_MS], dtype=float))
    interpolated = _interpolate_speed_grid(speed_by_tws, low_index, high_index, alpha)
    return float(interpolated[0, 0])


def _inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Grid step must be positive")
    count = int(np.floor((stop - start) / step + 0.5))
    return start + np.arange(count + 1, dtype=float) * step


def _wrapped_grid(center_deg: float, half_span_deg: float, step_deg: float) -> np.ndarray:
    start = center_deg - half_span_deg
    stop = center_deg + half_span_deg
    values = _inclusive_grid(start, stop, step_deg)
    return np.unique(np.mod(values, 360.0))


def _downsample_indices(count: int, max_points: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    if count <= max_points:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, max_points, dtype=int))


def _base_sample_weights(track: WindTrack) -> np.ndarray:
    speed_weight = np.clip((track.speed_ms / KNOT_TO_MS - 0.8) / 2.5, 0.0, 1.0)
    heel_weight = np.clip((np.abs(track.heel_deg) - 2.0) / 8.0, 0.0, 1.0)
    steady_turn_weight = np.clip(1.0 - track.turn_rate_deg_s / 18.0, 0.0, 1.0)
    steady_speed_weight = np.clip(1.0 - track.acceleration_ms2 / 1.2, 0.0, 1.0)
    heading_course_delta = np.abs(np.rad2deg([angle_difference(course, heading) for course, heading in zip(track.course_rad, track.heading_rad)]))
    leeway_weight = np.clip(1.0 - heading_course_delta / 45.0, 0.0, 1.0)
    dt_weight = np.clip(1.0 - np.maximum(track.dt_seconds - 1.5, 0.0) / 4.0, 0.0, 1.0)
    weights = 0.1 + 0.4 * speed_weight + 0.2 * heel_weight + 0.15 * steady_turn_weight + 0.1 * steady_speed_weight + 0.05 * leeway_weight
    return weights * dt_weight


def _search_true_wind(
    track: WindTrack,
    direction_grid_deg: np.ndarray,
    speed_grid_kts: np.ndarray,
    sample_weights: np.ndarray,
    prior_direction_deg: float | None = None,
    prior_direction_weight: float = 0.0,
    prior_speed_kts: float | None = None,
    prior_speed_weight: float = 0.0,
) -> dict[str, float | np.ndarray]:
    best_search: dict[str, float | np.ndarray] | None = None
    tws_low_index, tws_high_index, tws_alpha = _prepare_tws_interpolation(speed_grid_kts)

    for direction_deg in direction_grid_deg:
        true_wind_angle = wrap_phase(np.deg2rad(direction_deg) - track.heading_rad)
        twa_deg = np.rad2deg(mirror_angle_to_half_circle(true_wind_angle))
        twa_reliability = _twa_reliability(twa_deg)
        combined_weights = sample_weights * twa_reliability
        weight_sum = float(np.sum(combined_weights))
        if weight_sum <= 1e-9:
            continue

        speed_by_tws = _interpolate_speed_by_angle(twa_deg)
        predicted_grid_ms = _interpolate_speed_grid(speed_by_tws, tws_low_index, tws_high_index, tws_alpha)
        residual_grid_ms = track.speed_ms[None, :] - predicted_grid_ms
        score_grid = _weighted_loss_grid(residual_grid_ms, predicted_grid_ms, combined_weights)
        if prior_direction_deg is not None and prior_direction_weight > 0.0:
            direction_delta_deg = abs(float(angle_difference(np.deg2rad(direction_deg), np.deg2rad(prior_direction_deg))))
            score_grid = score_grid + prior_direction_weight * np.rad2deg(direction_delta_deg) ** 2
        if prior_speed_kts is not None and prior_speed_weight > 0.0:
            score_grid = score_grid + prior_speed_weight * (speed_grid_kts - float(prior_speed_kts)) ** 2
        best_index = int(np.argmin(score_grid))
        best_score = float(score_grid[best_index])

        if best_search is None or best_score < float(best_search["score"]):
            best_search = {
                "score": best_score,
                "direction_deg": float(direction_deg),
                "speed_kts": float(speed_grid_kts[best_index]),
                "prediction_ms": predicted_grid_ms[best_index],
                "twa_deg": twa_deg,
            }

    if best_search is None:
        raise ValueError("The track did not contain enough reliable sailing samples to estimate the wind")
    return best_search


def _prepare_tws_interpolation(speed_grid_kts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_speed_grid_kts = np.clip(speed_grid_kts, ILCA7_TWS_KTS[0], ILCA7_TWS_KTS[-1])
    high_index = np.searchsorted(ILCA7_TWS_KTS, clipped_speed_grid_kts, side="right")
    high_index = np.clip(high_index, 1, len(ILCA7_TWS_KTS) - 1)
    low_index = high_index - 1
    span = ILCA7_TWS_KTS[high_index] - ILCA7_TWS_KTS[low_index]
    alpha = (clipped_speed_grid_kts - ILCA7_TWS_KTS[low_index]) / span
    return low_index, high_index, alpha


def _interpolate_speed_by_angle(twa_deg: np.ndarray) -> np.ndarray:
    return np.array([np.interp(twa_deg, ILCA7_TWA_DEG, row) for row in ILCA7_POLAR_SPEEDS_MS], dtype=float)


def _interpolate_speed_grid(
    speed_by_tws: np.ndarray,
    low_index: np.ndarray,
    high_index: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    low_speed = speed_by_tws[low_index]
    high_speed = speed_by_tws[high_index]
    return (1.0 - alpha[:, None]) * low_speed + alpha[:, None] * high_speed


def _twa_reliability(twa_deg: np.ndarray) -> np.ndarray:
    upwind_reliability = np.clip((twa_deg - 32.0) / 18.0, 0.0, 1.0)
    downwind_reliability = np.clip((180.0 - twa_deg) / 12.0, 0.0, 1.0)
    return upwind_reliability * downwind_reliability


def _weighted_loss_grid(residual_grid_ms: np.ndarray, predicted_grid_ms: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
    abs_residual_grid_ms = np.abs(residual_grid_ms)
    huber_delta_ms = 0.35
    huber_loss = np.where(
        abs_residual_grid_ms <= huber_delta_ms,
        0.5 * abs_residual_grid_ms**2,
        huber_delta_ms * (abs_residual_grid_ms - 0.5 * huber_delta_ms),
    )
    overspeed_margin_ms = 0.45 + 0.08 * predicted_grid_ms
    overspeed_penalty = np.square(np.clip(residual_grid_ms - overspeed_margin_ms, 0.0, None))
    weighted_loss = (huber_loss + 1.75 * overspeed_penalty) * sample_weights[None, :]
    return np.sum(weighted_loss, axis=1) / np.sum(sample_weights)


def _max_polar_speed_ms(true_wind_speed_ms: float) -> float:
    candidate_twa_deg = np.linspace(35.0, 180.0, 146, dtype=float)
    speed_by_tws = _interpolate_speed_by_angle(candidate_twa_deg)
    low_index, high_index, alpha = _prepare_tws_interpolation(np.array([true_wind_speed_ms / KNOT_TO_MS], dtype=float))
    predicted_speed = _interpolate_speed_grid(speed_by_tws, low_index, high_index, alpha)[0]
    return float(np.max(predicted_speed))


def _detect_tow_outliers(
    track: WindTrack,
    predicted_speed_ms: np.ndarray,
    twa_deg: np.ndarray,
    true_wind_speed_ms: float,
    base_mask: np.ndarray,
) -> np.ndarray:
    residual_speed_ms = track.speed_ms - predicted_speed_ms
    max_polar_speed_ms = _max_polar_speed_ms(true_wind_speed_ms)
    trusted_angle_mask = twa_deg >= 45.0
    hard_margin_ms = 0.75 * KNOT_TO_MS
    soft_margin_ms = 1.0 * KNOT_TO_MS + 0.1 * predicted_speed_ms
    hard_overspeed_mask = track.speed_ms > max_polar_speed_ms + hard_margin_ms
    soft_overspeed_mask = trusted_angle_mask & (residual_speed_ms > soft_margin_ms)
    return base_mask & (hard_overspeed_mask | soft_overspeed_mask)


def _detect_low_confidence_samples(
    track: WindTrack,
    twa_deg: np.ndarray,
    predicted_speed_ms: np.ndarray,
    tow_outlier_mask: np.ndarray,
    base_mask: np.ndarray,
) -> np.ndarray:
    residual_speed_ms = track.speed_ms - predicted_speed_ms
    twa_confidence_mask = (twa_deg >= 40.0) & (twa_deg <= 172.0)
    underspeed_margin_ms = 0.8 * KNOT_TO_MS + 0.18 * predicted_speed_ms
    maneuver_mask = (track.turn_rate_deg_s > 16.0) | (track.acceleration_ms2 > 0.9)
    underspeed_mask = residual_speed_ms < -underspeed_margin_ms
    return base_mask & ~tow_outlier_mask & (~twa_confidence_mask | maneuver_mask | underspeed_mask)


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate true wind direction and speed from supported sailing-track CSV exports.")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file to analyze")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    parser.add_argument("--html-output", type=Path, help="Write an interactive HTML map report to this path")
    parser.add_argument("--athlete-polar-csv", type=Path, help="Write the derived athlete-specific polar to this CSV path")
    parser.add_argument("--wind-series-csv", type=Path, help="Write local sliding-window wind estimates to this CSV path")
    parser.add_argument("--window-seconds", type=float, default=300.0, help="Window size for local wind estimates")
    parser.add_argument("--step-seconds", type=float, default=180.0, help="Step size between local wind estimates")
    parser.add_argument("--open-browser", action="store_true", help="Open the HTML map report in the default browser")
    args = parser.parse_args(list(argv) if argv is not None else None)

    track = load_track_csv(args.csv_path)
    result = estimate_true_wind(track)
    athlete_polar = generate_athlete_specific_polar(track, result, source_name=args.csv_path.name)
    wind_series: WindTimeSeries | None = None
    if args.html_output is not None or args.open_browser or args.wind_series_csv is not None:
        wind_series = estimate_true_wind_series(
            track,
            source_name=args.csv_path.name,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
        )
    html_output = args.html_output
    if args.open_browser and html_output is None:
        html_output = args.csv_path.with_name(f"{args.csv_path.stem} wind map.html")
    if html_output is not None:
        saved_path = save_wind_map_html(
            track,
            result,
            athlete_polar,
            wind_series,
            html_output,
            title=f"Wind Estimate for {args.csv_path.name}",
        )
        print(f"HTML map saved to {saved_path}")
        if args.open_browser:
            webbrowser.open(saved_path.resolve().as_uri())
    if args.athlete_polar_csv is not None:
        athlete_path = save_athlete_specific_polar_csv(athlete_polar, args.athlete_polar_csv)
        print(f"Athlete polar CSV saved to {athlete_path}")
    if args.wind_series_csv is not None:
        if wind_series is None:
            wind_series = estimate_true_wind_series(
                track,
                source_name=args.csv_path.name,
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
            )
        wind_series_path = save_wind_time_series_csv(wind_series, args.wind_series_csv)
        print(f"Wind series CSV saved to {wind_series_path}")
    if args.json:
        print(json.dumps(result.summary_dict(), indent=2))
    else:
        print(format_estimate_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
