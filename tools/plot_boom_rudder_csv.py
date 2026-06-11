#!/usr/bin/env python3
"""Plot TrollFish boom/rudder time-series CSV exports.

Creates:
  - smoothed histogram for rudder angle
  - smoothed histogram for boom/mast angle
  - separate timelines
  - combined fixed-scale timeline (-40..40 deg by default)
  - stats as CSV and JSON

Example:
  python tools/plot_boom_rudder_csv.py "%USERPROFILE%\\Downloads\\Line_2026-05-24_boom_rudder_timeseries.csv" -o "%USERPROFILE%\\Pictures\\TrollFishPlots"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover - fallback for minimal installs
    gaussian_kde = None


RUDDER_CANDIDATES = ("rudder_angle_deg", "rudder_deg", "rudder", "rudder_angle")
MAST_CANDIDATES = (
    "mast_angle_deg",
    "mast_deg",
    "mast_angle",
    "boom_angle_deg",
    "boom_deg",
    "boom_angle",
    "boom",
)
TIME_CANDIDATES = ("time_s", "time_vs_anchor_s", "t", "seconds")
ABS_TIME_CANDIDATES = ("abs_ts_s", "timestamp_s", "epoch_s")

# Plot text scaling. Increase this value if you want even larger text.
PLOT_TEXT_SCALE = 2.0


def apply_plot_text_scale(scale: float = PLOT_TEXT_SCALE) -> None:
    """Scale all Matplotlib text used in the generated plots."""
    plt.rcParams.update({
        "font.size": 10 * scale,
        "axes.titlesize": 12 * scale,
        "axes.labelsize": 10 * scale,
        "xtick.labelsize": 10 * scale,
        "ytick.labelsize": 10 * scale,
        "legend.fontsize": 10 * scale,
        "figure.titlesize": 12 * scale,
    })


apply_plot_text_scale()


def slugify(value: object, fallback: str = "segment") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:100] or fallback


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    by_lower = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        hit = by_lower.get(candidate.lower())
        if hit:
            return hit
    if required:
        raise ValueError(f"Missing required column. Tried: {', '.join(candidates)}")
    return None


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def clean_xy(df: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "time_s": numeric_series(df, time_col),
        "value": numeric_series(df, value_col),
    }).dropna()
    out = out[np.isfinite(out["time_s"]) & np.isfinite(out["value"])]
    return out.sort_values("time_s")


def describe_values(values: pd.Series, name: str) -> dict[str, float | int | str | None]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return {
            "metric": name,
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "abs_mean": None,
            "rms": None,
        }

    arr = vals.to_numpy(dtype=float)
    percentiles = np.percentile(arr, [1, 5, 10, 25, 75, 90, 95, 99])
    return {
        "metric": name,
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p25": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "p99": float(percentiles[7]),
        "abs_mean": float(np.mean(np.abs(arr))),
        "rms": float(math.sqrt(np.mean(arr * arr))),
    }


def save_stats(stats: list[dict], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / f"{stem}_stats.csv", index=False)
    with (out_dir / f"{stem}_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def plot_smoothed_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    color: str,
    out_path: Path,
    bins: int = 60,
) -> None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    vals = vals[np.isfinite(vals)].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    if vals.size == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(vals, bins=bins, density=True, alpha=0.22, color=color, edgecolor="none", label="Histogram")
        if vals.size >= 3 and gaussian_kde is not None and np.std(vals) > 1e-9:
            pad = max(2.0, (float(vals.max()) - float(vals.min())) * 0.08)
            xs = np.linspace(float(vals.min()) - pad, float(vals.max()) + pad, 500)
            kde = gaussian_kde(vals)
            ax.plot(xs, kde(xs), color=color, linewidth=2.4, label="Smoothed density")
        else:
            counts, edges = np.histogram(vals, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            kernel = np.array([1, 2, 3, 2, 1], dtype=float)
            kernel /= kernel.sum()
            smooth = np.convolve(counts, kernel, mode="same")
            ax.plot(centers, smooth, color=color, linewidth=2.4, label="Smoothed density")
        ax.axvline(np.mean(vals), color="#222222", linestyle="--", linewidth=1.1, label="Mean")
        ax.axvline(np.median(vals), color="#666666", linestyle=":", linewidth=1.4, label="Median")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_timeline(
    time_s: pd.Series,
    value: pd.Series,
    title: str,
    ylabel: str,
    color: str,
    out_path: Path,
) -> None:
    data = pd.DataFrame({"time_s": time_s, "value": value}).dropna().sort_values("time_s")
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    if data.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(data["time_s"], data["value"], color=color, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined_fixed(
    time_s: pd.Series,
    rudder: pd.Series,
    mast: pd.Series,
    mast_label: str,
    out_path: Path,
    y_min: float,
    y_max: float,
) -> None:
    data = pd.DataFrame({"time_s": time_s, "rudder": rudder, "mast": mast}).sort_values("time_s")
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=150)
    if data.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(data["time_s"], data["rudder"], color="#1f77b4", linewidth=1.25, label="Rudder angle")
        ax.plot(data["time_s"], data["mast"], color="#2ca02c", linewidth=1.25, label=mast_label)
    ax.set_title(f"Rudder and {mast_label} timeline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="#222222", linewidth=0.8, alpha=0.55)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def segment_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "segment_id" not in df.columns:
        return [("all_data", df)]

    groups: list[tuple[str, pd.DataFrame]] = []
    for idx, (segment_id, group) in enumerate(df.groupby("segment_id", dropna=False), start=1):
        name = None
        if "segment_name" in group.columns and group["segment_name"].notna().any():
            name = str(group["segment_name"].dropna().iloc[0])
        label = slugify(name or segment_id or f"segment_{idx}", fallback=f"segment_{idx}")
        groups.append((label, group.copy()))
    return groups or [("all_data", df)]


def process_group(
    group: pd.DataFrame,
    label: str,
    out_dir: Path,
    time_col: str,
    rudder_col: str,
    mast_col: str,
    mast_label: str,
    fixed_ylim: tuple[float, float],
    bins: int,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    time = numeric_series(group, time_col)
    rudder = numeric_series(group, rudder_col)
    mast = numeric_series(group, mast_col)

    plot_smoothed_histogram(
        rudder,
        f"{label} - rudder angle distribution",
        "Rudder angle (deg)",
        "#1f77b4",
        out_dir / f"{label}_rudder_histogram.png",
        bins=bins,
    )
    plot_smoothed_histogram(
        mast,
        f"{label} - {mast_label} distribution",
        f"{mast_label} (deg)",
        "#2ca02c",
        out_dir / f"{label}_{slugify(mast_label.lower())}_histogram.png",
        bins=bins,
    )
    plot_timeline(
        time,
        rudder,
        f"{label} - rudder timeline",
        "Rudder angle (deg)",
        "#1f77b4",
        out_dir / f"{label}_rudder_timeline.png",
    )
    plot_timeline(
        time,
        mast,
        f"{label} - {mast_label} timeline",
        f"{mast_label} (deg)",
        "#2ca02c",
        out_dir / f"{label}_{slugify(mast_label.lower())}_timeline.png",
    )
    plot_combined_fixed(
        time,
        rudder,
        mast,
        mast_label,
        out_dir / f"{label}_rudder_{slugify(mast_label.lower())}_fixed_scale.png",
        fixed_ylim[0],
        fixed_ylim[1],
    )

    stats = []
    for metric_name, values in (("rudder_angle_deg", rudder), (mast_label, mast)):
        row = describe_values(values, metric_name)
        row["segment"] = label
        stats.append(row)
    save_stats(stats, out_dir, label)
    return stats


def process_csv(csv_path: Path, output_root: Path, fixed_ylim: tuple[float, float], bins: int) -> Path:
    df = pd.read_csv(csv_path)
    time_col = find_column(df, TIME_CANDIDATES)
    rudder_col = find_column(df, RUDDER_CANDIDATES)
    mast_col = find_column(df, MAST_CANDIDATES)
    assert time_col is not None and rudder_col is not None and mast_col is not None

    mast_label = "Mast angle" if "mast" in mast_col.lower() else "Boom angle"
    file_out_dir = output_root / slugify(csv_path.stem, fallback="plots")
    file_out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    for label, group in segment_groups(df):
        group_out = file_out_dir / label
        all_stats.extend(
            process_group(
                group,
                label,
                group_out,
                time_col,
                rudder_col,
                mast_col,
                mast_label,
                fixed_ylim,
                bins,
            )
        )

    aggregate_stats = []
    for metric_name, values in (
        ("rudder_angle_deg", numeric_series(df, rudder_col)),
        (mast_label, numeric_series(df, mast_col)),
    ):
        row = describe_values(values, metric_name)
        row["segment"] = "all_segments"
        aggregate_stats.append(row)

    save_stats([*aggregate_stats, *all_stats], file_out_dir, "all_segments")
    return file_out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create plots and stats from TrollFish boom/rudder CSV exports.")
    parser.add_argument("csv", nargs="+", type=Path, help="One or more boom/rudder time-series CSV files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path.cwd() / "plot_outputs",
        help="Folder where plot folders should be written. Default: ./plot_outputs",
    )
    parser.add_argument("--ylim", nargs=2, type=float, default=(-40.0, 40.0), metavar=("MIN", "MAX"), help="Fixed y-axis range for combined comparison plot.")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bin count. Default: 60")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_ylim = (min(args.ylim), max(args.ylim))

    written = []
    for csv_path in args.csv:
        csv_path = csv_path.expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        written.append(process_csv(csv_path, args.output_dir, fixed_ylim, args.bins))

    print("Wrote plot outputs:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
