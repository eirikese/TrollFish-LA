"""Server-side PDF report generator.

Produces a print-friendly A4 report from the output of
``build_report_data()``.  Uses **fpdf2** for layout and **matplotlib** for
charts / maps.

Output path: ``<project>/derived/exports/report_<id>.pdf``
"""

from __future__ import annotations

# Force non-interactive matplotlib backend BEFORE any pyplot import
import matplotlib
matplotlib.use("Agg")

import io
import logging
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Colour palette (matches the app.js frontend) ─────────────────────────

PALETTE = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#00bcd4",
    "#ff6b6b",
    "#4ecdc4",
]

_LIGHT_BG = "#ffffff"
_DARK_TEXT = "#222222"
_GRID_COLOR = "#dddddd"
_TABLE_HEADER_BG = (41, 128, 185)  # blue-ish
_TABLE_HEADER_FG = (255, 255, 255)
_TABLE_ALT_BG = (240, 245, 250)
_GOLD_COLOR = (255, 200, 0)  # Gold fill for the dot marker


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _athlete_color(idx: int) -> str:
    return PALETTE[idx % len(PALETTE)]


def _safe_latin1(text: str) -> str:
    """Replace characters outside latin-1 with ASCII equivalents."""
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ── matplotlib figure helpers ─────────────────────────────────────────────

def _fig_to_png_bytes(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


# Fixed axis limits for histograms — enables cross-session comparison
_HISTOGRAM_XLIMITS: dict[str, tuple[float, float]] = {
    "Trunk Angle": (0, 120),
    "Rolling Moment": (0, 800),
    "Heel Angle": (-45, 45),
    "Speed Over Ground": (0, 18),
}


def _make_smooth_histogram(
    data_by_athlete: dict[str, list[float]],
    athlete_colors: dict[str, str],
    xlabel: str,
    ylabel: str = "Density",
    title: str = "",
    unit: str = "",
) -> bytes:
    """KDE-smoothed histogram with one curve per athlete + mean vertical lines.

    Uses fixed x-axis limits (when defined) so plots are comparable across sessions.
    Returns PNG bytes.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Determine fixed x-limits if available
    fixed_xlim = _HISTOGRAM_XLIMITS.get(xlabel)

    for ath_name, values in data_by_athlete.items():
        arr = np.array(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            continue
        color = athlete_colors.get(ath_name, "#888888")
        mean_val = float(np.mean(arr))
        try:
            kde = gaussian_kde(arr, bw_method="scott")
            if fixed_xlim:
                xs = np.linspace(fixed_xlim[0], fixed_xlim[1], 300)
            else:
                lo, hi = float(arr.min()), float(arr.max())
                margin = (hi - lo) * 0.15 or 1.0
                xs = np.linspace(lo - margin, hi + margin, 300)
            ys = kde(xs)
            ax.fill_between(xs, ys, alpha=0.25, color=color)
            ax.plot(xs, ys, color=color, linewidth=1.5, label=ath_name)
        except Exception:
            ax.hist(arr, bins=20, density=True, alpha=0.4, color=color, label=ath_name)

        # Mean vertical line
        ax.axvline(mean_val, color=color, linewidth=1.2, linestyle="--", alpha=0.8)
        # Small label at top
        ylims = ax.get_ylim()
        ax.text(mean_val, ylims[1] * 0.92, f"{mean_val:.1f}",
                color=color, fontsize=6.5, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7, lw=0.5))

    if fixed_xlim:
        ax.set_xlim(*fixed_xlim)

    ax.set_xlabel(f"{xlabel}{(' (' + unit + ')') if unit else ''}", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, color=_GRID_COLOR)
    if len(data_by_athlete) > 1:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ── Map helpers ───────────────────────────────────────────────────────────

def _to_mercator(lat: float, lon: float) -> tuple[float, float]:
    x = lon * 20037508.34 / 180.0
    y_rad = math.radians(lat)
    y = math.log(math.tan(math.pi / 4 + y_rad / 2)) * 20037508.34 / math.pi
    return x, y


def _make_map_image(
    gps_paths: dict[str, list[dict]],
    athlete_colors: dict[str, str],
    width_inches: float = 6.5,
    height_inches: float = 4.0,
) -> bytes | None:
    """Render a contextily map with GPS tracks overlaid.

    ``gps_paths``: ``{label: [{lat, lon, sog?}, ...]}``
    Returns PNG bytes or None on failure.
    """
    import matplotlib.pyplot as plt

    all_lats: list[float] = []
    all_lons: list[float] = []
    for pts in gps_paths.values():
        for p in pts:
            if p.get("lat") is not None and p.get("lon") is not None:
                all_lats.append(p["lat"])
                all_lons.append(p["lon"])

    if len(all_lats) < 2:
        return None

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    fig.patch.set_facecolor("white")

    for label, pts in gps_paths.items():
        if len(pts) < 2:
            continue
        color = athlete_colors.get(label, "#888888")
        xs, ys = [], []
        for p in pts:
            mx, my = _to_mercator(p["lat"], p["lon"])
            xs.append(mx)
            ys.append(my)
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.85, label=label)

        # Direction arrows along the track
        n_pts = len(xs)
        if n_pts >= 6:
            n_arrows = min(5, n_pts // 4)
            step = n_pts // (n_arrows + 1)
            for ai in range(1, n_arrows + 1):
                idx = ai * step
                if idx + 1 < n_pts:
                    ax.annotate(
                        "",
                        xy=(xs[idx + 1], ys[idx + 1]),
                        xytext=(xs[idx], ys[idx]),
                        arrowprops=dict(
                            arrowstyle="->", color=color,
                            lw=1.5, mutation_scale=10,
                        ),
                        zorder=3,
                    )

    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom="auto")
    except Exception as exc:
        logger.warning("contextily tile fetch failed: %s", exc)
        ax.set_facecolor("#f0f0f0")

    ax.set_axis_off()
    if len(gps_paths) > 1:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    return _fig_to_png_bytes(fig)


def _make_summary_map_image(
    full_track_points: dict[str, list[dict]],
    file_meta: dict[str, dict],
    athletes: list[dict],
    segments: list[dict],
    athlete_color_map: dict[str, str],
    width_inches: float = 7.0,
    height_inches: float = 4.5,
) -> bytes | None:
    """Summary map showing the full sailing session (thin gray) with
    segment portions highlighted in bold athlete colours.

    Parameters
    ----------
    full_track_points : {file_id: [{lat, lon, ts, video_s, ...}, ...]}
        ALL GPS track points for every video in the project.
    file_meta : {file_id: {athlete_id, ...}}
    athletes : [{id, name}, ...]
    segments : report_data["segments"]
    athlete_color_map : {name: hex_color}
    """
    import matplotlib.pyplot as plt

    if not full_track_points:
        return None

    all_lats: list[float] = []
    all_lons: list[float] = []

    for pts in full_track_points.values():
        for p in pts:
            if p.get("lat") is not None and p.get("lon") is not None:
                all_lats.append(p["lat"])
                all_lons.append(p["lon"])

    if len(all_lats) < 2:
        return None

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    fig.patch.set_facecolor("white")

    # 1) Draw full session tracks in thin gray
    for fid, pts in full_track_points.items():
        if len(pts) < 2:
            continue
        xs, ys = [], []
        for p in pts:
            if p.get("lat") is not None and p.get("lon") is not None:
                mx, my = _to_mercator(p["lat"], p["lon"])
                xs.append(mx)
                ys.append(my)
        if xs:
            ax.plot(xs, ys, color="#bbbbbb", linewidth=0.8, alpha=0.6, zorder=1)

    # 2) Overlay segment GPS portions in bold athlete colours + text labels
    seen_athletes: dict[str, Any] = {}  # for legend (athlete-only)
    for seg in segments:
        gps_path = seg.get("gps_path", [])
        if len(gps_path) < 2:
            continue
        ath_name = seg.get("athlete_name", "?")
        seg_name = seg.get("name", "")
        color = athlete_color_map.get(ath_name, "#888888")
        xs, ys = [], []
        for p in gps_path:
            if p.get("lat") is not None and p.get("lon") is not None:
                mx, my = _to_mercator(p["lat"], p["lon"])
                xs.append(mx)
                ys.append(my)
        if xs:
            line, = ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9,
                            zorder=2)
            if ath_name not in seen_athletes:
                seen_athletes[ath_name] = line

            # Text label at segment midpoint
            mid = len(xs) // 2
            ax.annotate(
                seg_name[:18],
                xy=(xs[mid], ys[mid]),
                fontsize=5.5, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.8, lw=0),
                ha="center", va="center", zorder=3,
            )

    # Add map tiles
    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom="auto")
    except Exception as exc:
        logger.warning("contextily tile fetch failed: %s", exc)
        ax.set_facecolor("#f0f0f0")

    ax.set_axis_off()

    # Legend shows only athlete names (not individual segments)
    if seen_athletes:
        from matplotlib.patches import Patch
        patches = [
            Patch(facecolor=athlete_color_map.get(n, "#888"), alpha=0.9, label=n)
            for n in seen_athletes
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper left", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    return _fig_to_png_bytes(fig)


# ── Hull STL detail loader ────────────────────────────────────────────────

_hull_detail_cache: list[dict] | None = None
_hull_detail_loaded: bool = False


def _load_hull_detail() -> list[dict]:
    """Load Hull.stl cross-sections at multiple depths for detailed outline.

    Returns list of ``{"depth": float, "paths": [[(x, y), ...], ...]}``
    where *x* = fore/aft (metres), *y* = port/stbd (metres).
    Shallower slices (near deck) show cockpit / structure detail.
    """
    global _hull_detail_cache, _hull_detail_loaded
    if _hull_detail_loaded:
        return _hull_detail_cache or []

    _hull_detail_loaded = True
    sections: list[dict] = []
    try:
        from src.app.services.assets import resolve_hull_stl_path
        stl_path = resolve_hull_stl_path()
        if stl_path is None:
            logger.warning("Hull.stl not found")
            return []

        import trimesh
        mesh = trimesh.load(str(stl_path))

        # Slice at several Y-depth levels (STL Y = depth, 0 = top, negative = deeper)
        for y_level in [-0.5, -1, -2, -5, -10]:
            try:
                section = mesh.section(
                    plane_origin=[0, y_level, 0],
                    plane_normal=[0, 1, 0],
                )
                if section is None:
                    continue
                sv = np.array(section.vertices)
                paths: list[list[tuple[float, float]]] = []
                for entity in section.entities:
                    pts_3d = sv[entity.points]
                    boat_x = pts_3d[:, 2] / 100.0  # STL Z → fore/aft (m)
                    boat_y = pts_3d[:, 0] / 100.0  # STL X → port/stbd (m)
                    path = list(zip(boat_x.tolist(), boat_y.tolist()))
                    if len(path) >= 2:
                        paths.append(path)
                if paths:
                    sections.append({"depth": y_level, "paths": paths})
            except Exception:
                continue

        logger.info(
            "Hull detail loaded: %d depth levels, %d total paths",
            len(sections),
            sum(len(s["paths"]) for s in sections),
        )
        _hull_detail_cache = sections
        return sections
    except Exception:
        logger.exception("Failed to load hull detail")
        return []


def _draw_hull_on_ax(ax, hull_detail: list[dict] | None = None) -> None:
    """Draw detailed hull cross-sections on a matplotlib Axes.

    Shallower slices are drawn bolder (deck-level structure is most useful
    for understanding crew positions).
    """
    if hull_detail is None:
        hull_detail = _load_hull_detail()

    if not hull_detail:
        # Fallback simple boat shape
        bx = [-4.2, -3.5, -1.5, 0.0, 0.5, 0.0, -1.5, -3.5, -4.2]
        by = [0, 0.6, 0.73, 0.5, 0, -0.5, -0.73, -0.6, 0]
        ax.plot(bx, by, color="#888888", linewidth=1.2, alpha=0.7)
        return

    for section in hull_detail:
        depth = section["depth"]
        if depth >= -1:
            lw, alpha, color = 1.4, 0.85, "#444444"
        elif depth >= -3:
            lw, alpha, color = 0.9, 0.55, "#777777"
        else:
            lw, alpha, color = 0.5, 0.35, "#aaaaaa"
        for path in section["paths"]:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                    solid_capstyle="round")

    ax.axhline(0, color="#cccccc", linewidth=0.4, alpha=0.4)


# ── Heatmap rendering ────────────────────────────────────────────────────

def _make_heatmap_image(
    data_by_athlete: dict[str, list[tuple[float, float]]],
    athlete_colors: dict[str, str],
    title: str = "",
    grid_size_x: float = 5.0,
    grid_size_y: float = 3.0,
    grid_center_x: float = -2.0,
    grid_center_y: float = 0.0,
    resolution: int = 100,
    per_athlete: bool = False,
    show_mean_pos: bool = False,
) -> bytes | None:
    """Render heatmaps of positional density on the hull outline.

    When *per_athlete* is True and there are multiple athletes, each athlete
    gets a separate subplot with a gradient in their colour for clarity.
    Otherwise all athletes are overlaid on one shared plot.
    Returns PNG bytes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    athletes_with_data = {n: pts for n, pts in data_by_athlete.items()
                          if len(pts) >= 3}
    if not athletes_with_data:
        return None

    half_x = grid_size_x / 2
    half_y = grid_size_y / 2
    x_lo = grid_center_x - half_x
    x_hi = grid_center_x + half_x
    y_lo = grid_center_y - half_y
    y_hi = grid_center_y + half_y

    hull_detail = _load_hull_detail()
    n_ath = len(athletes_with_data)

    if per_athlete and n_ath >= 1:
        # ── Per-athlete side-by-side subplots ─────────────────────────
        fig_w = min(4.2 * n_ath, 14)
        fig, axes = plt.subplots(1, n_ath, figsize=(fig_w, 3.2))
        if n_ath == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")

        for ax_i, (ath_name, points) in enumerate(athletes_with_data.items()):
            ax = axes[ax_i]
            ax.set_facecolor("white")
            _draw_hull_on_ax(ax, hull_detail)

            color_hex = athlete_colors.get(ath_name, "#888888")
            pts_arr = np.array(points, dtype=np.float64)
            mask = (
                (pts_arr[:, 0] >= x_lo) & (pts_arr[:, 0] <= x_hi)
                & (pts_arr[:, 1] >= y_lo) & (pts_arr[:, 1] <= y_hi)
            )
            pts_arr = pts_arr[mask]

            if len(pts_arr) >= 3:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(pts_arr.T, bw_method=0.3)
                    xi = np.linspace(x_lo, x_hi, resolution)
                    yi = np.linspace(y_lo, y_hi, resolution)
                    Xi, Yi = np.meshgrid(xi, yi)
                    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                    mx = Zi.max()
                    if mx > 0:
                        Zi /= mx

                    # Gradient: transparent white → athlete colour
                    cmap = LinearSegmentedColormap.from_list(
                        f"ath_{ax_i}",
                        [(1, 1, 1, 0), to_rgba(color_hex, alpha=0.92)],
                        N=256,
                    )
                    ax.contourf(Xi, Yi, Zi,
                                levels=np.linspace(0.05, 1.0, 10),
                                cmap=cmap, antialiased=True)
                    ax.contour(Xi, Yi, Zi,
                               levels=[0.2, 0.5, 0.8],
                               colors=[color_hex], linewidths=0.6, alpha=0.4)
                except Exception:
                    ax.scatter(pts_arr[:, 0], pts_arr[:, 1],
                               s=3, alpha=0.4, color=color_hex)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(ath_name, fontsize=9, color=color_hex, fontweight="bold")
            if ax_i == 0:
                ax.set_ylabel("Port / Stbd (m)", fontsize=7)
            ax.set_xlabel("Fore / Aft (m)", fontsize=7)
            ax.tick_params(labelsize=6)

            # Mean position annotation (for COM plots)
            if show_mean_pos and len(pts_arr) >= 3:
                mean_x = float(np.mean(pts_arr[:, 0]))
                mean_y = float(np.mean(pts_arr[:, 1]))
                ax.text(
                    x_lo + 0.12, y_lo + 0.28,
                    f"Fwd = {mean_x:.2f}m\nSide = {mean_y:.2f}m",
                    fontsize=6, color=color_hex, fontweight="bold",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color_hex, alpha=0.8, lw=0.5),
                )

        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()
        return _fig_to_png_bytes(fig)

    else:
        # ── Combined overlay plot ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        _draw_hull_on_ax(ax, hull_detail)

        for ath_i, (ath_name, points) in enumerate(athletes_with_data.items()):
            color_hex = athlete_colors.get(ath_name, "#888888")
            pts_arr = np.array(points, dtype=np.float64)
            mask = (
                (pts_arr[:, 0] >= x_lo) & (pts_arr[:, 0] <= x_hi)
                & (pts_arr[:, 1] >= y_lo) & (pts_arr[:, 1] <= y_hi)
            )
            pts_arr = pts_arr[mask]
            if len(pts_arr) < 3:
                continue
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(pts_arr.T, bw_method=0.3)
                xi = np.linspace(x_lo, x_hi, resolution)
                yi = np.linspace(y_lo, y_hi, resolution)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                mx = Zi.max()
                if mx > 0:
                    Zi /= mx

                # Gradient: transparent white → athlete colour
                cmap = LinearSegmentedColormap.from_list(
                    f"overlay_{ath_i}",
                    [(1, 1, 1, 0), to_rgba(color_hex, alpha=0.92)],
                    N=256,
                )
                ax.contourf(Xi, Yi, Zi,
                            levels=np.linspace(0.05, 1.0, 10),
                            cmap=cmap, antialiased=True)
                ax.contour(Xi, Yi, Zi,
                            levels=[0.2, 0.5, 0.8],
                            colors=[color_hex], linewidths=0.6, alpha=0.4)
            except Exception:
                ax.scatter(pts_arr[:, 0], pts_arr[:, 1],
                            s=3, alpha=0.4, color=color_hex)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Fore / Aft (m)", fontsize=8)
        ax.set_ylabel("Port / Stbd (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold")
        if len(athletes_with_data) > 1:
            from matplotlib.patches import Patch
            patches = [
                Patch(facecolor=athlete_colors.get(n, "#888"), alpha=0.7, label=n)
                for n in athletes_with_data
            ]
            ax.legend(handles=patches, fontsize=7, loc="upper right")

        # Mean position annotations (for COM overlay plots)
        if show_mean_pos:
            y_offset = 0.28
            for ath_name_m, points_m in athletes_with_data.items():
                pts_m = np.array(points_m, dtype=np.float64)
                mask_m = (
                    (pts_m[:, 0] >= x_lo) & (pts_m[:, 0] <= x_hi)
                    & (pts_m[:, 1] >= y_lo) & (pts_m[:, 1] <= y_hi)
                )
                pts_m = pts_m[mask_m]
                if len(pts_m) >= 3:
                    mean_x = float(np.mean(pts_m[:, 0]))
                    mean_y = float(np.mean(pts_m[:, 1]))
                    c_hex = athlete_colors.get(ath_name_m, "#888")
                    ax.text(
                        x_lo + 0.12, y_lo + y_offset,
                        f"{ath_name_m}: Fwd={mean_x:.2f}m  Side={mean_y:.2f}m",
                        fontsize=5.5, color=c_hex, fontweight="bold",
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=c_hex, alpha=0.8, lw=0.5),
                    )
                    y_offset += 0.35

        fig.tight_layout()
        return _fig_to_png_bytes(fig)


# ── Stats formatting ──────────────────────────────────────────────────────

def _format_stat(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def _format_stat_pm(avg: float | None, std: float | None, decimals: int = 1) -> str:
    """Format as 'avg +/- std'."""
    if avg is None:
        return "-"
    avg_s = f"{avg:.{decimals}f}"
    if std is not None and std > 0:
        std_s = f"{std:.{decimals}f}"
        return f"{avg_s} +/- {std_s}"
    return avg_s


# ── Main PDF builder ─────────────────────────────────────────────────────


def generate_pdf_report(
    report_data: dict,
    project_id: str,
    output_dir: Path,
    *,
    full_track_points: dict[str, list[dict]] | None = None,
    file_meta: dict[str, dict] | None = None,
    athletes: list[dict] | None = None,
) -> Path:
    """Generate a full PDF report and return the output path.

    Parameters
    ----------
    report_data : dict
        The payload from ``build_report_data()``.
    project_id : str
        Project UUID (for naming).
    output_dir : Path
        Directory to write the PDF to (e.g. ``<project>/derived/exports``).
    full_track_points : dict, optional
        ALL GPS track points for every video file {file_id: [point, ...]}.
        Used for the summary map to show the full session.
    file_meta : dict, optional
        File metadata mapping {file_id: {athlete_id, ...}}.
    athletes : list, optional
        Athletes list [{id, name}, ...].

    Returns
    -------
    Path
        Absolute path to the generated PDF file.
    """
    from fpdf import FPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    report_id = uuid.uuid4().hex[:12]
    pdf_path = output_dir / f"report_{report_id}.pdf"

    rpt_athletes: list[dict] = report_data.get("athletes", [])
    segments: list[dict] = report_data.get("segments", [])
    golds: dict = report_data.get("golds", {})

    if not segments:
        logger.warning("No segments in report data - skipping PDF generation")
        return pdf_path

    # Build colour maps
    athlete_color_map: dict[str, str] = {}
    for i, ath in enumerate(rpt_athletes):
        athlete_color_map[ath["name"]] = _athlete_color(i)

    # Group segments by split_id for multi-athlete overlay
    from collections import defaultdict
    segs_by_split: dict[str, list[dict]] = defaultdict(list)
    for s in segments:
        segs_by_split[s["split_id"]].append(s)

    # Ordered unique split_ids
    split_ids_ordered: list[str] = []
    for s in segments:
        if s["split_id"] not in split_ids_ordered:
            split_ids_ordered.append(s["split_id"])

    # Unique athlete names (ordered)
    athlete_names_ordered: list[str] = []
    for ath in rpt_athletes:
        if ath["name"] not in athlete_names_ordered:
            athlete_names_ordered.append(ath["name"])

    # ── Initialise fpdf2 ─────────────────────────────────────────────
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(34, 34, 34)
    pdf.cell(0, 14, "TrollFish Sailing Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)

    from datetime import datetime
    pdf.cell(0, 6, _safe_latin1(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"{len(split_ids_ordered)} segment(s)  |  "
        f"{len(athlete_names_ordered)} athlete(s)"
    ), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # ── Summary map (full session + segments highlighted) ─────────────
    if full_track_points:
        map_png = _make_summary_map_image(
            full_track_points, file_meta or {}, athletes or [],
            segments, athlete_color_map,
            width_inches=7.0, height_inches=4.5,
        )
    else:
        # Fallback: just segment GPS
        all_gps: dict[str, list[dict]] = {}
        for s in segments:
            name = s.get("athlete_name", "?")
            path = s.get("gps_path", [])
            if path:
                key = f"{name} - {s['name']}"
                all_gps[key] = path
                athlete_color_map.setdefault(key, athlete_color_map.get(name, "#888888"))
        map_png = _make_map_image(all_gps, athlete_color_map,
                                  width_inches=7.0, height_inches=4.5)

    if map_png:
        _embed_image(pdf, map_png, max_w=180, align_center=True)
    pdf.ln(3)

    # ── Summary stats table ───────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(34, 34, 34)
    pdf.cell(0, 8, "Summary Statistics", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    _draw_summary_table(pdf, segments, segs_by_split, split_ids_ordered,
                        athlete_names_ordered, golds, athlete_color_map)
    pdf.ln(6)

    # ── Per-segment sections ──────────────────────────────────────────
    for split_id in split_ids_ordered:
        seg_group = segs_by_split[split_id]
        seg_name = seg_group[0]["name"]

        pdf.add_page()

        # Section header
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(34, 34, 34)
        pdf.cell(0, 10, _safe_latin1(f"Segment: {seg_name}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        # Duration & athletes info
        dur = seg_group[0].get("duration_s", 0)
        ath_names = [s["athlete_name"] for s in seg_group]
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5,
                 _safe_latin1(f"Duration: {dur:.0f}s  |  Athletes: {', '.join(ath_names)}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # ── Segment map ──────────────────────────────────────────────
        seg_gps: dict[str, list[dict]] = {}
        for s in seg_group:
            path = s.get("gps_path", [])
            if path:
                seg_gps[s["athlete_name"]] = path
        seg_map_png = _make_map_image(seg_gps, athlete_color_map,
                                      width_inches=6.0, height_inches=3.5)
        if seg_map_png:
            _embed_image(pdf, seg_map_png, max_w=165, align_center=True)
            pdf.ln(2)

        # ── Segment stats table (athletes as columns) ────────────────
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(34, 34, 34)
        pdf.cell(0, 7, "Segment Statistics", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        _draw_segment_stats_table(pdf, seg_group, golds, athlete_color_map)
        pdf.ln(3)

        # ── Histograms (2x2 grid) ────────────────────────────────────
        hist_charts = _build_segment_histograms(seg_group, athlete_color_map)
        if hist_charts:
            _embed_histogram_grid(pdf, hist_charts)

        # ── Heatmaps (rendered from raw XY data with hull outline) ───
        _draw_segment_heatmaps(pdf, seg_group, athlete_color_map)

    # ── Save ──────────────────────────────────────────────────────────
    pdf.output(str(pdf_path))
    logger.info("PDF report saved: %s", pdf_path)
    return pdf_path


# ── Internal drawing helpers ──────────────────────────────────────────────


def _embed_image(
    pdf,
    png_bytes: bytes,
    max_w: float = 180,
    align_center: bool = False,
) -> None:
    """Write PNG bytes into a temp file and embed in the PDF."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp.write(png_bytes)
        tmp.close()
        if align_center:
            from PIL import Image as PILImage
            with PILImage.open(tmp.name) as img:
                img_w, img_h = img.size
            aspect = img_h / img_w
            disp_w = min(max_w, pdf.epw)
            disp_h = disp_w * aspect
            x = (pdf.w - disp_w) / 2
            if pdf.get_y() + disp_h > pdf.h - pdf.b_margin:
                pdf.add_page()
            pdf.image(tmp.name, x=x, w=disp_w)
        else:
            pdf.image(tmp.name, w=max_w)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _draw_gold_dot(pdf, x: float, y: float, r: float = 1.2) -> None:
    """Draw a small filled gold circle at (x, y), preserving current colours."""
    # Save colour values (0-1 floats) so we can properly re-emit via setters
    prev_fill = tuple(int(c * 255) for c in pdf.fill_color.colors)
    prev_draw = tuple(int(c * 255) for c in pdf.draw_color.colors)
    pdf.set_fill_color(*_GOLD_COLOR)
    pdf.set_draw_color(*_GOLD_COLOR)
    pdf.ellipse(x - r, y - r, 2 * r, 2 * r, style="FD")
    # Restore via setters to re-emit PDF color commands to the stream
    pdf.set_fill_color(*prev_fill)
    pdf.set_draw_color(*prev_draw)


def _is_gold(golds: dict, gold_key: str | None, split_id: str, athlete_name: str) -> bool:
    """Check if a specific athlete won gold for a category within a segment.

    Gold format: {category: {split_id: athlete_name}}
    """
    if gold_key is None:
        return False
    g = golds.get(gold_key)
    if not isinstance(g, dict):
        return False
    return g.get(split_id) == athlete_name


def _draw_summary_table(
    pdf,
    segments: list[dict],
    segs_by_split: dict[str, list[dict]],
    split_ids_ordered: list[str],
    athlete_names_ordered: list[str],
    golds: dict,
    athlete_color_map: dict[str, str],
) -> None:
    """Draw ONE unified summary statistics table.

    Layout:
        Row 0 (super-header): Metric | Segment1 (spanning athletes) | Segment2 (spanning) | ...
        Row 1 (sub-header)  :        | Ath1 | Ath2                  | Ath1 | Ath2          | ...
        Data rows           : label  | val  | val                   | val  | val           | ...
    """
    n_ath = len(athlete_names_ordered)
    n_seg = len(split_ids_ordered)
    if n_ath == 0 or n_seg == 0:
        return

    # Column widths
    metric_col_w = 36
    n_data_cols = n_seg * n_ath
    avail = pdf.epw - metric_col_w
    col_w = min(avail / max(n_data_cols, 1), 40)
    total_w = metric_col_w + col_w * n_data_cols

    # Scale down if wider than page
    if total_w > pdf.epw:
        scale = pdf.epw / total_w
        metric_col_w *= scale
        col_w *= scale
        total_w = pdf.epw

    row_h = 6.5
    header_h = 6.0

    metric_defs = [
        ("Avg SOG (kts)",       "sog",          "avg", 1, "avg_sog"),
        ("Max SOG (kts)",       "sog",          "max", 1, "max_sog"),
        ("Avg Heel (deg)",      "heel",         "avg", 1, "avg_heel"),
        ("Max Heel (deg)",      "heel",         "max", 1, "max_heel"),
        ("Avg Roll Moment (Nm)","moment_roll",  "avg", 0, "avg_moment_roll"),
        ("Max Roll Moment (Nm)","moment_roll",  "max", 0, "max_moment_roll"),
        ("Avg Trunk Angle (deg)", "trunk_angle","avg", 1, "avg_trunk_angle"),
        ("Max Trunk Angle (deg)", "trunk_angle","max", 1, "max_trunk_angle"),
    ]

    # Check page space
    needed = header_h * 2 + row_h * len(metric_defs)
    if pdf.get_y() + needed > pdf.h - pdf.b_margin:
        pdf.add_page()

    # ── Row 0: Segment super-headers ──────────────────────────────────
    y0 = pdf.get_y()
    # Metric column header (spans both rows)
    pdf.set_fill_color(*_TABLE_HEADER_BG)
    pdf.set_text_color(*_TABLE_HEADER_FG)
    pdf.set_font("Helvetica", "B", 7)
    pdf.rect(pdf.l_margin, y0, metric_col_w, header_h * 2, style="F")
    pdf.set_xy(pdf.l_margin, y0 + header_h * 0.5)
    pdf.cell(metric_col_w, header_h, "Metric", align="C")

    # Segment name headers (each spans n_ath columns)
    seg_names = [segs_by_split[sid][0]["name"] for sid in split_ids_ordered]
    for si, (sid, sname) in enumerate(zip(split_ids_ordered, seg_names)):
        x = pdf.l_margin + metric_col_w + si * n_ath * col_w
        w = n_ath * col_w
        pdf.set_fill_color(60, 60, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 6.5)
        pdf.rect(x, y0, w, header_h, style="F")
        pdf.set_xy(x, y0)
        pdf.cell(w, header_h, _safe_latin1(sname[:20]), align="C")

    # ── Row 1: Athlete sub-headers (tinted with athlete colour) ────────
    y1 = y0 + header_h
    pdf.set_font("Helvetica", "B", 6.5)
    for si in range(n_seg):
        for ai, aname in enumerate(athlete_names_ordered):
            x = pdf.l_margin + metric_col_w + (si * n_ath + ai) * col_w
            # Blend athlete colour into the header background
            ar, ag, ab = _hex_to_rgb(athlete_color_map.get(aname, "#888888"))
            tr = int(ar * 0.35 + _TABLE_HEADER_BG[0] * 0.65)
            tg = int(ag * 0.35 + _TABLE_HEADER_BG[1] * 0.65)
            tb = int(ab * 0.35 + _TABLE_HEADER_BG[2] * 0.65)
            pdf.set_fill_color(tr, tg, tb)
            pdf.rect(x, y1, col_w, header_h, style="F")
            pdf.set_xy(x, y1)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(col_w, header_h, _safe_latin1(aname[:12]), align="C")
    pdf.set_y(y1 + header_h)

    # ── Build lookup: (split_id, athlete_name) -> segment data ────────
    seg_lookup: dict[tuple[str, str], dict] = {}
    for s in segments:
        seg_lookup[(s["split_id"], s["athlete_name"])] = s

    # ── Data rows ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 6.5)
    for mi, (label, stat_key, sub_key, decimals, gold_key) in enumerate(metric_defs):
        if mi % 2 == 0:
            pdf.set_fill_color(*_TABLE_ALT_BG)
        else:
            pdf.set_fill_color(255, 255, 255)

        y = pdf.get_y()
        if y + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            y = pdf.get_y()

        fill = mi % 2 == 0

        # Metric label (neutral dark text)
        pdf.set_text_color(34, 34, 34)
        pdf.set_xy(pdf.l_margin, y)
        pdf.cell(metric_col_w, row_h, _safe_latin1(label), fill=fill)

        # Data cells — text coloured to match athlete
        for si, split_id in enumerate(split_ids_ordered):
            for ai, aname in enumerate(athlete_names_ordered):
                x = pdf.l_margin + metric_col_w + (si * n_ath + ai) * col_w
                pdf.set_xy(x, y)
                seg_data = seg_lookup.get((split_id, aname))
                if seg_data:
                    stats = seg_data.get(stat_key, {})
                    val = stats.get(sub_key)
                    std = stats.get("std")
                    if sub_key == "max":
                        cell_text = _format_stat(val, decimals)
                    else:
                        cell_text = _format_stat_pm(val, std, decimals)
                else:
                    cell_text = "-"

                # Athlete-coloured text (darkened 30 % for readability)
                ar, ag, ab = _hex_to_rgb(athlete_color_map.get(aname, "#888888"))
                pdf.set_text_color(max(int(ar * 0.7), 0),
                                   max(int(ag * 0.7), 0),
                                   max(int(ab * 0.7), 0))
                pdf.cell(col_w, row_h, _safe_latin1(cell_text), align="C", fill=fill)

                # Gold dot
                if seg_data and _is_gold(golds, gold_key, split_id, aname):
                    dot_x = x + col_w - 2.0
                    dot_y = y + row_h / 2
                    _draw_gold_dot(pdf, dot_x, dot_y, r=0.9)

        pdf.set_y(y + row_h)


def _draw_segment_stats_table(
    pdf,
    seg_group: list[dict],
    golds: dict,
    athlete_color_map: dict[str, str],
) -> None:
    """Per-segment detail stats table with athletes as COLUMNS.

    Rows are metrics, columns are athletes.
    """
    ath_names = []
    for s in seg_group:
        if s["athlete_name"] not in ath_names:
            ath_names.append(s["athlete_name"])

    n_ath = len(ath_names)
    if n_ath == 0:
        return

    metric_col_w = 36
    avail = pdf.epw - metric_col_w
    ath_col_w = min(avail / max(n_ath, 1), 50)
    total_w = metric_col_w + ath_col_w * n_ath

    row_h = 6.5

    split_id = seg_group[0]["split_id"]

    metric_defs = [
        ("Avg SOG (kts)",       "sog",          "avg", 1, "avg_sog"),
        ("Max SOG (kts)",       "sog",          "max", 1, "max_sog"),
        ("Avg Heel (deg)",      "heel",         "avg", 1, "avg_heel"),
        ("Max Heel (deg)",      "heel",         "max", 1, "max_heel"),
        ("Avg Roll Moment (Nm)","moment_roll",  "avg", 0, "avg_moment_roll"),
        ("Max Roll Moment (Nm)","moment_roll",  "max", 0, "max_moment_roll"),
        ("Avg Trunk Angle (deg)", "trunk_angle","avg", 1, "avg_trunk_angle"),
        ("Max Trunk Angle (deg)", "trunk_angle","max", 1, "max_trunk_angle"),
        ("Duration (s)",        None,           None,  0, None),
    ]

    # Header row — athlete columns tinted with their colour
    y0 = pdf.get_y()
    pdf.set_font("Helvetica", "B", 7.5)

    # Metric column header
    pdf.set_fill_color(*_TABLE_HEADER_BG)
    pdf.set_text_color(*_TABLE_HEADER_FG)
    pdf.rect(pdf.l_margin, y0, metric_col_w, row_h, style="F")
    pdf.set_xy(pdf.l_margin, y0)
    pdf.cell(metric_col_w, row_h, "Metric", align="L")

    for ai, aname in enumerate(ath_names):
        x = pdf.l_margin + metric_col_w + ai * ath_col_w
        color_hex = athlete_color_map.get(aname, "#888888")
        ar, ag, ab = _hex_to_rgb(color_hex)
        # Blend 35 % athlete colour into the header background
        tr = int(ar * 0.35 + _TABLE_HEADER_BG[0] * 0.65)
        tg = int(ag * 0.35 + _TABLE_HEADER_BG[1] * 0.65)
        tb = int(ab * 0.35 + _TABLE_HEADER_BG[2] * 0.65)
        pdf.set_fill_color(tr, tg, tb)
        pdf.rect(x, y0, ath_col_w, row_h, style="F")
        pdf.set_xy(x, y0)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(ath_col_w, row_h, _safe_latin1(aname[:15]), align="C")
    pdf.set_y(y0 + row_h)

    # Build lookup
    seg_by_ath: dict[str, dict] = {}
    for s in seg_group:
        seg_by_ath[s["athlete_name"]] = s

    # Data rows — text coloured to match athlete
    pdf.set_font("Helvetica", "", 7.5)
    for mi, (label, stat_key, sub_key, decimals, gold_key) in enumerate(metric_defs):
        if mi % 2 == 0:
            pdf.set_fill_color(*_TABLE_ALT_BG)
        else:
            pdf.set_fill_color(255, 255, 255)

        y = pdf.get_y()
        if y + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            y = pdf.get_y()

        fill = mi % 2 == 0

        # Metric label (neutral dark text)
        pdf.set_text_color(34, 34, 34)
        pdf.set_xy(pdf.l_margin, y)
        pdf.cell(metric_col_w, row_h, _safe_latin1(label), fill=fill)

        # Athlete values
        for ai, aname in enumerate(ath_names):
            x = pdf.l_margin + metric_col_w + ai * ath_col_w
            pdf.set_xy(x, y)

            seg_data = seg_by_ath.get(aname)
            if seg_data is None:
                cell_text = "-"
            elif stat_key is None:
                # Duration row
                cell_text = f"{seg_data.get('duration_s', 0):.0f}"
            else:
                stats = seg_data.get(stat_key, {})
                val = stats.get(sub_key)
                std = stats.get("std")
                if sub_key == "max":
                    cell_text = _format_stat(val, decimals)
                else:
                    cell_text = _format_stat_pm(val, std, decimals)

            # Athlete-coloured text (darkened 30 % for readability)
            ar, ag, ab = _hex_to_rgb(athlete_color_map.get(aname, "#888888"))
            pdf.set_text_color(max(int(ar * 0.7), 0),
                               max(int(ag * 0.7), 0),
                               max(int(ab * 0.7), 0))
            pdf.cell(ath_col_w, row_h, _safe_latin1(cell_text), align="C", fill=fill)

            # Gold dot
            if seg_data and _is_gold(golds, gold_key, split_id, aname):
                dot_x = x + ath_col_w - 2.5
                dot_y = y + row_h / 2
                _draw_gold_dot(pdf, dot_x, dot_y, r=1.0)

        pdf.set_y(y + row_h)


def _build_segment_histograms(
    seg_group: list[dict],
    athlete_color_map: dict[str, str],
) -> list[bytes]:
    """Build 4 histogram PNGs for a segment group: trunk angle, roll moment, heel, SOG."""
    charts: list[bytes] = []

    def _collect_timeline(key: str, val_key: str = "v") -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for s in seg_group:
            name = s["athlete_name"]
            tl = s.get(key, [])
            vals = [rec[val_key] for rec in tl
                    if isinstance(rec, dict) and rec.get(val_key) is not None]
            if vals:
                out[name] = vals
        return out

    # 1. Trunk angle
    ta_data = _collect_timeline("trunk_angle_timeline")
    if ta_data:
        charts.append(_make_smooth_histogram(
            ta_data, athlete_color_map,
            xlabel="Trunk Angle", unit="deg", title="Trunk Angle Distribution",
        ))

    # 2. Rolling moment
    mr_data = _collect_timeline("moment_timeline")
    mr_abs: dict[str, list[float]] = {}
    for name, vals in mr_data.items():
        mr_abs[name] = [abs(v) for v in vals]
    if mr_abs:
        charts.append(_make_smooth_histogram(
            mr_abs, athlete_color_map,
            xlabel="Rolling Moment", unit="Nm", title="Rolling Moment Distribution",
        ))

    # 3. Heel
    heel_data = _collect_timeline("heel_timeline")
    if heel_data:
        charts.append(_make_smooth_histogram(
            heel_data, athlete_color_map,
            xlabel="Heel Angle", unit="deg", title="Heel Angle Distribution",
        ))

    # 4. SOG
    sog_data = _collect_timeline("sog_timeline", val_key="v")
    if sog_data:
        charts.append(_make_smooth_histogram(
            sog_data, athlete_color_map,
            xlabel="Speed Over Ground", unit="kts", title="SOG Distribution",
        ))

    return charts


def _embed_histogram_grid(pdf, charts: list[bytes]) -> None:
    """Place up to 4 histogram PNGs in a 2x2 grid."""
    if not charts:
        return

    cell_w = pdf.epw / 2 - 2
    y_row_start = pdf.get_y()

    for i, png_data in enumerate(charts):
        col = i % 2
        if col == 0:
            if pdf.get_y() + 60 > pdf.h - pdf.b_margin:
                pdf.add_page()
            y_row_start = pdf.get_y()

        x = pdf.l_margin + col * (cell_w + 4)
        y = y_row_start

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            tmp.write(png_data)
            tmp.close()
            pdf.image(tmp.name, x=x, y=y, w=cell_w)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

        if col == 1:
            from PIL import Image as PILImage
            with PILImage.open(io.BytesIO(png_data)) as img:
                aspect = img.height / img.width
            row_h = cell_w * aspect
            pdf.set_y(y_row_start + row_h + 3)

    # If odd number of charts, advance past the last row
    if len(charts) % 2 == 1:
        from PIL import Image as PILImage
        with PILImage.open(io.BytesIO(charts[-1])) as img:
            aspect = img.height / img.width
        row_h = cell_w * aspect
        pdf.set_y(y_row_start + row_h + 3)


def _draw_segment_heatmaps(
    pdf,
    seg_group: list[dict],
    athlete_color_map: dict[str, str],
) -> None:
    """Draw keypoint density and COM heatmaps using raw XY data with hull outline."""

    # Collect raw XY coordinates from the report data
    kp_by_athlete: dict[str, list[tuple[float, float]]] = {}
    com_by_athlete: dict[str, list[tuple[float, float]]] = {}

    for s in seg_group:
        name = s["athlete_name"]

        kp_xy = s.get("kp_xy", [])
        if kp_xy:
            kp_by_athlete[name] = [(p[0], p[1]) for p in kp_xy]

        com_xy = s.get("com_xy", [])
        if com_xy:
            com_by_athlete[name] = [(p[0], p[1]) for p in com_xy]

    if not kp_by_athlete and not com_by_athlete:
        # Fallback: try base64 compositing if no raw XY
        _draw_segment_heatmaps_fallback(pdf, seg_group, athlete_color_map)
        return

    if pdf.get_y() + 55 > pdf.h - pdf.b_margin:
        pdf.add_page()

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(34, 34, 34)
    pdf.cell(0, 7, "Position Heatmaps", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Generate per-athlete individual plots AND combined overlay for both KP and COM
    kp_indiv_png = None
    kp_overlay_png = None
    com_indiv_png = None
    com_overlay_png = None

    if kp_by_athlete:
        kp_indiv_png = _make_heatmap_image(
            kp_by_athlete, athlete_color_map,
            title="Keypoint Density \u2013 Individual",
            per_athlete=True,
        )
        if len(kp_by_athlete) >= 2:
            kp_overlay_png = _make_heatmap_image(
                kp_by_athlete, athlete_color_map,
                title="Keypoint Density \u2013 Overlay Comparison",
                per_athlete=False,
            )

    if com_by_athlete:
        com_indiv_png = _make_heatmap_image(
            com_by_athlete, athlete_color_map,
            title="COM Density \u2013 Individual",
            per_athlete=True,
            show_mean_pos=True,
        )
        if len(com_by_athlete) >= 2:
            com_overlay_png = _make_heatmap_image(
                com_by_athlete, athlete_color_map,
                title="COM Density \u2013 Overlay Comparison",
                per_athlete=False,
                show_mean_pos=True,
            )

    has_any = kp_indiv_png or kp_overlay_png or com_indiv_png or com_overlay_png
    if not has_any:
        return

    y0 = pdf.get_y()
    if y0 + 55 > pdf.h - pdf.b_margin:
        pdf.add_page()

    # ── Keypoint density section ──
    if kp_indiv_png:
        _embed_image(pdf, kp_indiv_png, max_w=pdf.epw, align_center=True)
        pdf.ln(2)
    if kp_overlay_png:
        if pdf.get_y() + 50 > pdf.h - pdf.b_margin:
            pdf.add_page()
        overlay_w = min(pdf.epw * 0.55, 100)
        _embed_image(pdf, kp_overlay_png, max_w=overlay_w, align_center=True)
        pdf.ln(3)

    # ── COM density section ──
    if com_indiv_png:
        if pdf.get_y() + 50 > pdf.h - pdf.b_margin:
            pdf.add_page()
        _embed_image(pdf, com_indiv_png, max_w=pdf.epw, align_center=True)
        pdf.ln(2)
    if com_overlay_png:
        if pdf.get_y() + 50 > pdf.h - pdf.b_margin:
            pdf.add_page()
        overlay_w = min(pdf.epw * 0.55, 100)
        _embed_image(pdf, com_overlay_png, max_w=overlay_w, align_center=True)
        pdf.ln(2)


def _draw_segment_heatmaps_fallback(
    pdf,
    seg_group: list[dict],
    athlete_color_map: dict[str, str],
) -> None:
    """Fallback: composite pre-rendered base64 heatmap PNGs when raw XY is unavailable."""
    import base64

    kp_pngs: list[tuple[str, bytes]] = []
    com_pngs: list[tuple[str, bytes]] = []

    for s in seg_group:
        name = s["athlete_name"]
        kp_hm = s.get("keypoint_heatmap")
        if kp_hm and kp_hm.get("image_b64"):
            try:
                kp_pngs.append((name, base64.b64decode(kp_hm["image_b64"])))
            except Exception:
                pass
        com_hm = s.get("com_heatmap")
        if com_hm and com_hm.get("image_b64"):
            try:
                com_pngs.append((name, base64.b64decode(com_hm["image_b64"])))
            except Exception:
                pass

    if not kp_pngs and not com_pngs:
        return

    if pdf.get_y() + 55 > pdf.h - pdf.b_margin:
        pdf.add_page()

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(34, 34, 34)
    pdf.cell(0, 7, "Position Heatmaps", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    cell_w = pdf.epw / 2 - 4
    y0 = pdf.get_y()

    if y0 + 55 > pdf.h - pdf.b_margin:
        pdf.add_page()
        y0 = pdf.get_y()

    if kp_pngs:
        composite = _composite_heatmap_pngs(kp_pngs)
        if composite:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.set_xy(pdf.l_margin, y0)
            pdf.cell(cell_w, 5, "Keypoint Density", align="C")
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                tmp.write(composite)
                tmp.close()
                pdf.image(tmp.name, x=pdf.l_margin, y=y0 + 5, w=cell_w)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    if com_pngs:
        composite = _composite_heatmap_pngs(com_pngs)
        if composite:
            x_right = pdf.l_margin + cell_w + 8
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.set_xy(x_right, y0)
            pdf.cell(cell_w, 5, "COM Density", align="C")
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                tmp.write(composite)
                tmp.close()
                pdf.image(tmp.name, x=x_right, y=y0 + 5, w=cell_w)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    pdf.set_y(y0 + cell_w * 0.6 + 10)


def _composite_heatmap_pngs(pairs: list[tuple[str, bytes]]) -> bytes | None:
    """Alpha-composite multiple RGBA heatmap PNGs together."""
    from PIL import Image as PILImage

    base = None
    for name, png_bytes in pairs:
        try:
            img = PILImage.open(io.BytesIO(png_bytes)).convert("RGBA")
            if base is None:
                base = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
            else:
                if img.size != base.size:
                    img = img.resize(base.size, PILImage.LANCZOS)
            base = PILImage.alpha_composite(base, img)
        except Exception:
            continue

    if base is None:
        return None

    buf = io.BytesIO()
    base.save(buf, format="PNG")
    return buf.getvalue()
