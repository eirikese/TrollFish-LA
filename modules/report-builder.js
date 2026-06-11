/**
 * TrollFish — report-builder.js
 * Builds report data from skeleton + GPS data in IndexedDB / OPFS.
 * Direct port of Python report.py (~785 lines → ~500 lines JS).
 *
 * Main entry:  buildReportData(projectId, segmentIds) → reportData
 */

import * as DB from './db.js';
import { loadSkeletonFrames, loadMetrics } from './pose-engine.js';
import { detectProjectManeuvers } from './maneuvers.js';
import { computeSegmentVmg } from './vmg.js';
import { buildAthletePolarFromPoints } from './wind-estimation.js';

// ── Constants ─────────────────────────────────────────────────────────

/**
 * Segmental centre-of-mass table (de Leva model, 14 segments).
 * [idx_a, idx_b, mass_fraction, com_fraction]
 * idx can be int (single landmark) or [int, int] (midpoint of two landmarks).
 */
const _COM_SEGMENTS = [
  [[11,12], [7,8],   0.081, 1.000],   // Head+Neck
  [[11,12], [23,24], 0.497, 0.500],   // Trunk
  [11, 13, 0.028, 0.436],             // L upper arm
  [13, 15, 0.016, 0.430],             // L forearm
  [15, 17, 0.006, 0.506],             // L hand
  [12, 14, 0.028, 0.436],             // R upper arm
  [14, 16, 0.016, 0.430],             // R forearm
  [16, 18, 0.006, 0.506],             // R hand
  [23, 25, 0.100, 0.433],             // L thigh
  [25, 27, 0.0465, 0.433],            // L shank
  [27, 31, 0.0145, 0.500],            // L foot
  [24, 26, 0.100, 0.433],             // R thigh
  [26, 28, 0.0465, 0.433],            // R shank
  [28, 32, 0.0145, 0.500],            // R foot
];

// ── Math helpers ──────────────────────────────────────────────────────

function iqrFilter(arr) {
  if (arr.length < 4) return arr;
  const sorted = [...arr].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1;
  const lo = q1 - 1.5 * iqr;
  const hi = q3 + 1.5 * iqr;
  return arr.filter(v => v >= lo && v <= hi);
}

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function stats(arr, doIqr = false) {
  if (!arr || arr.length === 0) return { avg: null, max: null, min: null, std: null, count: 0 };
  const vals = doIqr ? iqrFilter(arr) : arr;
  if (vals.length === 0) return { avg: null, max: null, min: null, std: null, count: 0 };
  const n = vals.length;
  const sum = vals.reduce((a, b) => a + b, 0);
  const avg = sum / n;
  const mx = Math.max(...vals);
  const mn = Math.min(...vals);
  const variance = vals.reduce((a, v) => a + (v - avg) ** 2, 0) / n;
  return { avg, max: mx, min: mn, std: Math.sqrt(variance), count: n };
}

/**
 * Convert absolute epoch timestamp to video_s using GPS track points.
 */
function absTs2VideoSec(trackPoints, absTs) {
  if (!trackPoints?.length) return null;
  let loIdx = 0;
  for (let i = 0; i < trackPoints.length; i++) {
    if (trackPoints[i].ts != null && trackPoints[i].ts <= absTs) loIdx = i;
    else break;
  }
  const lo = trackPoints[loIdx];
  const hiIdx = Math.min(loIdx + 1, trackPoints.length - 1);
  const hi = trackPoints[hiIdx];
  if (lo.video_s == null) return null;
  if (hiIdx === loIdx || hi.ts == null || hi.video_s == null || hi.ts === lo.ts) return lo.video_s;
  const frac = (absTs - lo.ts) / (hi.ts - lo.ts);
  return lo.video_s + frac * (hi.video_s - lo.video_s);
}

function videoSec2AbsTs(trackPoints, videoSec) {
  const target = Number(videoSec);
  if (!trackPoints?.length || !Number.isFinite(target)) return null;
  let loIdx = 0;
  for (let i = 0; i < trackPoints.length; i++) {
    if (trackPoints[i].video_s != null && trackPoints[i].video_s <= target) loIdx = i;
    else break;
  }
  const lo = trackPoints[loIdx];
  const hiIdx = Math.min(loIdx + 1, trackPoints.length - 1);
  const hi = trackPoints[hiIdx];
  if (lo?.ts == null) return null;
  if (hiIdx === loIdx || hi?.video_s == null || hi?.ts == null || hi.video_s === lo.video_s) return Number(lo.ts);
  const frac = (target - lo.video_s) / (hi.video_s - lo.video_s);
  return Number(lo.ts) + frac * (Number(hi.ts) - Number(lo.ts));
}

function absTs2SegmentSec(absTs, segStartTs) {
  const abs = normalizeEpochSeconds(absTs);
  const start = normalizeEpochSeconds(segStartTs);
  if (!Number.isFinite(abs) || !Number.isFinite(start)) return null;
  return abs - start;
}

function videoSec2SegmentSec(trackPoints, videoSec, segStartTs) {
  const absTs = videoSec2AbsTs(trackPoints, videoSec);
  return absTs2SegmentSec(absTs, segStartTs);
}

/**
 * Compute SOG (knots) from consecutive GPS fixes within a video_s range.
 * Returns { times:[], sogs:[], lats:[], lons:[] }.
 */
function computeSogFromTrack(points, startS, endS) {
  const pts = points.filter(p => p.video_s != null && p.video_s >= startS && p.video_s <= endS);
  if (pts.length < 2) return { times: [], sogs: [], lats: [], lons: [] };
  const times = [], sogs = [], lats = [], lons = [];
  for (let i = 1; i < pts.length; i++) {
    const dt = pts[i].video_s - pts[i - 1].video_s;
    if (dt < 0.01) continue;
    const dist = haversineM(pts[i - 1].lat, pts[i - 1].lon, pts[i].lat, pts[i].lon);
    const speed = (dist / dt) * 1.94384; // m/s → knots
    if (speed > 50) continue; // implausible
    const tmid = (pts[i].video_s + pts[i - 1].video_s) / 2;
    times.push(tmid);
    sogs.push(speed);
    lats.push((pts[i].lat + pts[i - 1].lat) / 2);
    lons.push((pts[i].lon + pts[i - 1].lon) / 2);
  }
  return { times, sogs, lats, lons };
}

function subsampleTimeSeries(data, maxPoints = 600) {
  if (!Number.isFinite(Number(maxPoints)) || Number(maxPoints) <= 0) return data;
  if (data.length <= maxPoints) return data;
  const step = (data.length - 1) / (maxPoints - 1);
  const result = [];
  for (let i = 0; i < maxPoints; i++) {
    result.push(data[Math.round(i * step)]);
  }
  return result;
}

function yieldReportWork() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

const _INSTRUMENT_KEYS = ['heel', 'trim', 'sog', 'cog', 'hdg'];

function _bisectRight(sortedVals, x) {
  let lo = 0;
  let hi = sortedVals.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedVals[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function _lowerBound(sortedVals, x) {
  let lo = 0;
  let hi = sortedVals.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedVals[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function _upperBound(sortedVals, x) {
  let lo = 0;
  let hi = sortedVals.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedVals[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function _isSortedNumeric(values) {
  for (let i = 1; i < values.length; i++) {
    if (values[i] < values[i - 1]) return false;
  }
  return true;
}

function buildTrackIndex(points) {
  const rows = Array.isArray(points) ? points : [];
  const tsVals = rows.map(p => Number(p?.ts));
  const videoVals = rows.map(p => Number(p?.video_s));
  return {
    rows,
    tsVals,
    videoVals,
    tsSorted: tsVals.every(Number.isFinite) && _isSortedNumeric(tsVals),
    videoSorted: videoVals.every(Number.isFinite) && _isSortedNumeric(videoVals),
  };
}

function buildMetricIndex(rows) {
  const data = Array.isArray(rows) ? rows : [];
  const tsVals = data.map(row => Number(row?.ts));
  return {
    rows: data,
    tsVals,
    tsSorted: tsVals.every(Number.isFinite) && _isSortedNumeric(tsVals),
  };
}

function sliceTrackByVideoRange(index, startS, endS) {
  const start = Number(startS);
  const end = Number(endS);
  const rows = index?.rows || [];
  if (!rows.length || !Number.isFinite(start) || !Number.isFinite(end)) return [];
  if (!index.videoSorted) {
    return rows.filter(p => p.video_s != null && p.video_s >= start && p.video_s <= end);
  }
  const lo = _lowerBound(index.videoVals, start);
  const hi = _upperBound(index.videoVals, end);
  return rows.slice(lo, hi);
}

function sliceMetricsByVideoRange(index, startS, endS) {
  const start = Number(startS);
  const end = Number(endS);
  const rows = index?.rows || [];
  if (!rows.length || !Number.isFinite(start) || !Number.isFinite(end)) return [];
  if (!index.tsSorted) return rows.filter(m => m.ts >= start && m.ts <= end);
  const lo = _lowerBound(index.tsVals, start);
  const hi = _upperBound(index.tsVals, end);
  return rows.slice(lo, hi);
}

function absTs2VideoSecIndexed(index, absTs) {
  const target = Number(absTs);
  if (!index?.rows?.length || !Number.isFinite(target)) return null;
  if (!index.tsSorted) return absTs2VideoSec(index.rows, target);
  const loIdx = Math.max(0, _upperBound(index.tsVals, target) - 1);
  const lo = index.rows[loIdx];
  const hiIdx = Math.min(loIdx + 1, index.rows.length - 1);
  const hi = index.rows[hiIdx];
  if (lo?.video_s == null) return null;
  if (hiIdx === loIdx || hi?.ts == null || hi?.video_s == null || hi.ts === lo.ts) return lo.video_s;
  const frac = (target - lo.ts) / (hi.ts - lo.ts);
  return lo.video_s + frac * (hi.video_s - lo.video_s);
}

function videoSec2AbsTsIndexed(index, videoSec) {
  const target = Number(videoSec);
  if (!index?.rows?.length || !Number.isFinite(target)) return null;
  if (!index.videoSorted) return videoSec2AbsTs(index.rows, target);
  const loIdx = Math.max(0, _upperBound(index.videoVals, target) - 1);
  const lo = index.rows[loIdx];
  const hiIdx = Math.min(loIdx + 1, index.rows.length - 1);
  const hi = index.rows[hiIdx];
  if (lo?.ts == null) return null;
  if (hiIdx === loIdx || hi?.video_s == null || hi?.ts == null || hi.video_s === lo.video_s) return Number(lo.ts);
  const frac = (target - lo.video_s) / (hi.video_s - lo.video_s);
  return Number(lo.ts) + frac * (Number(hi.ts) - Number(lo.ts));
}

function videoSec2SegmentSecIndexed(index, videoSec, segStartTs) {
  const absTs = videoSec2AbsTsIndexed(index, videoSec);
  return absTs2SegmentSec(absTs, segStartTs);
}

function computeSogFromTrackIndexed(index, startS, endS) {
  const pts = sliceTrackByVideoRange(index, startS, endS);
  if (pts.length < 2) return { times: [], sogs: [], lats: [], lons: [] };
  const times = [], sogs = [], lats = [], lons = [];
  for (let i = 1; i < pts.length; i++) {
    const dt = pts[i].video_s - pts[i - 1].video_s;
    if (dt < 0.01) continue;
    const dist = haversineM(pts[i - 1].lat, pts[i - 1].lon, pts[i].lat, pts[i].lon);
    const speed = (dist / dt) * 1.94384;
    if (speed > 50) continue;
    const tmid = (pts[i].video_s + pts[i - 1].video_s) / 2;
    times.push(tmid);
    sogs.push(speed);
    lats.push((pts[i].lat + pts[i - 1].lat) / 2);
    lons.push((pts[i].lon + pts[i - 1].lon) / 2);
  }
  return { times, sogs, lats, lons };
}

/**
 * Mirror of Python merge_csv_instruments_into_video_tracks():
 * interpolate instrument columns from each video's best-matched CSV track.
 */
function mergeCsvInstrumentsIntoVideoTracks(videoTracksByFile, csvTracksByFile, matches) {
  const bestCsvByVideo = {};
  for (const m of (matches || [])) {
    if (Number(m?.rank) === 1 && m.video_file_id && m.csv_file_id && !(m.video_file_id in bestCsvByVideo)) {
      bestCsvByVideo[m.video_file_id] = m.csv_file_id;
    }
  }

  for (const [videoFid, videoPts] of Object.entries(videoTracksByFile || {})) {
    const csvFid = bestCsvByVideo[videoFid];
    if (!csvFid) continue;
    const csvPts = (csvTracksByFile || {})[csvFid];
    if (!Array.isArray(videoPts) || !Array.isArray(csvPts) || csvPts.length === 0) continue;

    // Ensure same ordering/indexing as csvPts for bisect interpolation.
    const indexed = csvPts
      .map(p => ({ ts: Number(p?.ts), p }))
      .filter(rec => Number.isFinite(rec.ts))
      .sort((a, b) => a.ts - b.ts);
    if (indexed.length === 0) continue;
    const sortedTimes = indexed.map(rec => rec.ts);
    const sortedPts = indexed.map(rec => rec.p);

    for (const vp of videoPts) {
      const ts = Number(vp?.ts);
      if (!Number.isFinite(ts)) continue;
      if (ts < sortedTimes[0] - 10 || ts > sortedTimes[sortedTimes.length - 1] + 10) continue;

      const idx = _bisectRight(sortedTimes, ts);
      const loIdx = Math.max(0, idx - 1);
      const hiIdx = Math.min(idx, sortedTimes.length - 1);
      const lo = sortedPts[loIdx];
      const hi = sortedPts[hiIdx];
      const loTs = Number(lo?.ts);
      const hiTs = Number(hi?.ts);

      if (loIdx === hiIdx || !Number.isFinite(loTs) || !Number.isFinite(hiTs) || Math.abs(hiTs - loTs) < 1e-9) {
        for (const key of _INSTRUMENT_KEYS) {
          const v = Number(lo?.[key]);
          if (Number.isFinite(v) && vp[key] == null) vp[key] = v;
        }
        continue;
      }

      const frac = Math.max(0, Math.min(1, (ts - loTs) / (hiTs - loTs)));
      for (const key of _INSTRUMENT_KEYS) {
        const vlo = Number(lo?.[key]);
        const vhi = Number(hi?.[key]);
        if (vp[key] != null) continue;
        if (Number.isFinite(vlo) && Number.isFinite(vhi)) {
          vp[key] = vlo + frac * (vhi - vlo);
        } else if (Number.isFinite(vlo)) {
          vp[key] = vlo;
        }
      }
    }
  }

  return videoTracksByFile;
}

// ── Histogram & scatter builders ──────────────────────────────────────

function buildHikingHistogram(trunkAngles, binWidth = 5.0) {
  if (trunkAngles.length === 0) return { bins: [], counts: [], bin_width: binWidth };
  const mn = Math.floor(Math.min(...trunkAngles) / binWidth) * binWidth;
  const mx = Math.ceil(Math.max(...trunkAngles) / binWidth) * binWidth;
  const bins = [];
  const counts = [];
  for (let edge = mn; edge < mx; edge += binWidth) {
    const center = edge + binWidth / 2;
    const count = trunkAngles.filter(v => v >= edge && v < edge + binWidth).length;
    bins.push(center);
    counts.push(count);
  }
  return { bins, counts, bin_width: binWidth };
}

function buildTrunkVsSogScatter(trunkTimePairs, sogTimes, sogVals) {
  if (!trunkTimePairs.length || !sogTimes.length) {
    return { points: [], fit_slope: null, fit_intercept: null };
  }
  // For each trunk angle sample, find nearest SOG within 2.0s
  const points = [];
  for (const { t, v } of trunkTimePairs) {
    let bestDist = Infinity, bestSog = null;
    for (let i = 0; i < sogTimes.length; i++) {
      const d = Math.abs(sogTimes[i] - t);
      if (d < bestDist) { bestDist = d; bestSog = sogVals[i]; }
    }
    if (bestDist <= 2.0 && bestSog != null) {
      points.push([bestSog, v]); // [sog, trunkAngle]
    }
  }
  if (points.length < 3) return { points, fit_slope: null, fit_intercept: null };

  // Simple linear regression: y = slope * x + intercept
  const n = points.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const [x, y] of points) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
  const denom = n * sxx - sx * sx;
  const slope = denom !== 0 ? (n * sxy - sx * sy) / denom : 0;
  const intercept = (sy - slope * sx) / n;

  // Subsample points for display
  const subsampled = points.length > 500
    ? subsampleTimeSeries(points, 500)
    : points;

  return { points: subsampled, fit_slope: slope, fit_intercept: intercept };
}

function buildCumulativeSideLoad(metricsInRange, weight) {
  if (!metricsInRange.length) return { times: [], values: [] };
  const times = [], values = [];
  let cumulative = 0;
  let lastTs = null;
  for (const m of metricsInRange) {
    if (m.roll_moment == null) continue;
    const dt = lastTs != null ? m.ts - lastTs : 0;
    cumulative += m.roll_moment * dt;
    times.push(m.ts);
    values.push(cumulative);
    lastTs = m.ts;
  }
  return { times: subsampleTimeSeries(times, 600), values: subsampleTimeSeries(values, 600) };
}

function buildSideSwitchAnalysis(metricsInRange, trunkAngleTime, momentTime) {
  // Detect com_y sign changes (port < 0, starboard > 0).
  const comYSamples = metricsInRange.filter(m => m.com_y != null);
  if (comYSamples.length < 3) {
    return { count: 0, port_fraction: null, starboard_fraction: null,
      port_trunk_avg: null, stbd_trunk_avg: null, port_moment_avg: null, stbd_moment_avg: null };
  }
  let switchCount = 0;
  let lastSign = comYSamples[0].com_y < 0 ? 1 : -1;
  for (let i = 1; i < comYSamples.length; i++) {
    const sign = comYSamples[i].com_y < 0 ? 1 : -1;
    if (sign !== lastSign) { switchCount++; lastSign = sign; }
  }

  // Time fractions
  let portTime = 0, stbdTime = 0;
  for (let i = 1; i < comYSamples.length; i++) {
    const dt = comYSamples[i].ts - comYSamples[i - 1].ts;
    if (comYSamples[i].com_y < 0) portTime += dt;
    else stbdTime += dt;
  }
  const total = portTime + stbdTime || 1;

  // Per-side averages
  const portTrunk = [], stbdTrunk = [];
  const portMoment = [], stbdMoment = [];
  for (const { t, v } of trunkAngleTime) {
    const nearest = findNearestComY(comYSamples, t, 1.0);
    if (nearest == null) continue;
    if (nearest < 0) portTrunk.push(v); else stbdTrunk.push(v);
  }
  for (const { t, v } of momentTime) {
    const nearest = findNearestComY(comYSamples, t, 1.0);
    if (nearest == null) continue;
    if (nearest < 0) portMoment.push(Math.abs(v)); else stbdMoment.push(Math.abs(v));
  }

  return {
    count: switchCount,
    port_fraction: portTime / total,
    starboard_fraction: stbdTime / total,
    port_trunk_avg: portTrunk.length ? portTrunk.reduce((a, b) => a + b, 0) / portTrunk.length : null,
    stbd_trunk_avg: stbdTrunk.length ? stbdTrunk.reduce((a, b) => a + b, 0) / stbdTrunk.length : null,
    port_moment_avg: portMoment.length ? portMoment.reduce((a, b) => a + b, 0) / portMoment.length : null,
    stbd_moment_avg: stbdMoment.length ? stbdMoment.reduce((a, b) => a + b, 0) / stbdMoment.length : null,
  };
}

function findNearestComY(comYSamples, t, maxDist) {
  let best = null, bestD = Infinity;
  for (const m of comYSamples) {
    const d = Math.abs(m.ts - t);
    if (d < bestD) { bestD = d; best = m.com_y; }
  }
  return bestD <= maxDist ? best : null;
}

function median(arr) {
  if (!arr || arr.length === 0) return null;
  const s = [...arr].sort((a, b) => a - b);
  const n = s.length;
  const m = Math.floor(n / 2);
  return n % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function wrapDeg180(deg) {
  const a = Number(deg);
  if (!Number.isFinite(a)) return null;
  return ((a + 180) % 360 + 360) % 360 - 180;
}

function normalizeHeadingDeg(deg) {
  const a = Number(deg);
  if (!Number.isFinite(a)) return null;
  return ((a % 360) + 360) % 360;
}

function angularDiffDeg(a, b) {
  const da = normalizeHeadingDeg(a);
  const db = normalizeHeadingDeg(b);
  if (da == null || db == null) return null;
  return wrapDeg180(da - db);
}

function circularMeanDeg(vals) {
  if (!Array.isArray(vals) || vals.length === 0) return null;
  let sx = 0, sy = 0, n = 0;
  for (const v of vals) {
    const a = normalizeHeadingDeg(v);
    if (a == null) continue;
    const r = a * Math.PI / 180;
    sx += Math.cos(r);
    sy += Math.sin(r);
    n++;
  }
  if (n === 0) return null;
  const ang = Math.atan2(sy / n, sx / n) * 180 / Math.PI;
  return normalizeHeadingDeg(ang);
}

function cleanTimeline(points, valueFn = p => p?.v) {
  if (!Array.isArray(points)) return [];
  return points
    .map(p => {
      const t = Number(p?.t);
      const vRaw = valueFn(p);
      const v = Number(vRaw);
      if (vRaw == null || !Number.isFinite(t) || !Number.isFinite(v)) return null;
      return { t, v };
    })
    .filter(Boolean)
    .sort((a, b) => a.t - b.t);
}

function extractWindowSeries(series, t0, t1, refT, maxPoints = 260) {
  const win = (series || [])
    .filter(p => p.t >= t0 && p.t <= t1)
    .map(p => ({ t: p.t - refT, v: p.v }));
  return subsampleTimeSeries(win, maxPoints);
}

function unwrapHeadingSeries(series) {
  if (!Array.isArray(series) || series.length === 0) return [];
  const out = [];
  let prevRaw = normalizeHeadingDeg(series[0].v);
  if (prevRaw == null) return [];
  let acc = prevRaw;
  out.push({ t: series[0].t, v: acc });
  for (let i = 1; i < series.length; i++) {
    const raw = normalizeHeadingDeg(series[i].v);
    if (raw == null) continue;
    const d = angularDiffDeg(raw, prevRaw);
    if (d == null) continue;
    acc += d;
    prevRaw = raw;
    out.push({ t: series[i].t, v: acc });
  }
  return out;
}

function meanLinearWindow(series, a, b) {
  const vals = (series || []).filter(p => p.t >= a && p.t <= b).map(p => p.v);
  if (!vals.length) return null;
  return vals.reduce((sum, v) => sum + v, 0) / vals.length;
}

function nearestSeriesValue(series, tRef, maxDist = 1.5) {
  let bestVal = null;
  let bestDt = Infinity;
  for (const p of (series || [])) {
    const dt = Math.abs(p.t - tRef);
    if (dt < bestDt) {
      bestDt = dt;
      bestVal = p.v;
    }
  }
  return bestDt <= maxDist ? bestVal : null;
}

function linearWindowOrNearest(series, winStart, winEnd, fallbackT, fallbackMaxDist = 1.5) {
  const winMean = meanLinearWindow(series, winStart, winEnd);
  if (Number.isFinite(winMean)) return winMean;
  return nearestSeriesValue(series, fallbackT, fallbackMaxDist);
}

function findZeroCrossingTime(series, t0, t1) {
  const pts = (series || []).filter(p => p.t >= t0 && p.t <= t1);
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1];
    const b = pts[i];
    if (!Number.isFinite(a.v) || !Number.isFinite(b.v)) continue;
    if (a.v === 0) return a.t;
    if (b.v === 0) return b.t;
    if ((a.v < 0 && b.v > 0) || (a.v > 0 && b.v < 0)) {
      const frac = (0 - a.v) / (b.v - a.v);
      return a.t + frac * (b.t - a.t);
    }
  }
  return null;
}

function findHeadingChangeStart(headingSeriesUnwrapped, comCrossT, comStableT, preHeading, signedHeadingDelta) {
  if (!Number.isFinite(comCrossT) || !Number.isFinite(preHeading) || !Number.isFinite(signedHeadingDelta)) return null;
  const dir = Math.sign(signedHeadingDelta);
  const absDelta = Math.abs(signedHeadingDelta);
  if (!dir || absDelta < 1.0) return null;

  const onsetThreshold = Math.max(4.0, Math.min(10.0, absDelta * 0.18));
  const holdThreshold = Math.max(2.0, onsetThreshold * 0.65);
  const holdSec = 0.25;
  const searchStart = comCrossT - 3.0;
  const searchEnd = Math.min(comStableT, comCrossT + 1.0);
  const pts = (headingSeriesUnwrapped || []).filter(p => p.t >= searchStart && p.t <= searchEnd);

  for (let i = 0; i < pts.length; i++) {
    const startPt = pts[i];
    const directedDisp = dir * (startPt.v - preHeading);
    if (directedDisp < onsetThreshold) continue;

    let minDirectedDisp = directedDisp;
    let covered = false;
    for (let j = i; j < pts.length; j++) {
      const directed = dir * (pts[j].v - preHeading);
      minDirectedDisp = Math.min(minDirectedDisp, directed);
      if (pts[j].t - startPt.t >= holdSec) {
        covered = true;
        break;
      }
    }
    if (covered && minDirectedDisp >= holdThreshold) {
      return startPt.t;
    }
  }

  return null;
}

function buildTackAnalysis(comYTime, headingTime, rudderTime, sogTime, trunkTime, boomTime = []) {
  const comSeriesRaw = cleanTimeline(comYTime);
  const headingSeries = cleanTimeline(headingTime, p => normalizeHeadingDeg(p?.v));
  const rudderSeries = cleanTimeline(rudderTime);
  const boomSeries = cleanTimeline(boomTime);
  const sogSeries = cleanTimeline(sogTime);
  const trunkSeries = cleanTimeline(trunkTime);
  const headingSeriesUnwrapped = unwrapHeadingSeries(headingSeries);

  if (comSeriesRaw.length < 6 || headingSeries.length < 4) {
    return { count: 0, tacks: [] };
  }

  // Smooth COM Y lightly before side classification.
  const comSeries = [];
  let ema = null;
  const alpha = 0.28;
  for (const p of comSeriesRaw) {
    if (!Number.isFinite(ema)) ema = p.v;
    else ema = alpha * ema + (1.0 - alpha) * p.v;
    comSeries.push({ t: p.t, v: ema });
  }

  const deadband = 0.06;
  const minStableSideSec = 0.25;
  const minHeadingDeltaDeg = 20.0;
  const signOf = (y) => (y > deadband ? 1 : (y < -deadband ? -1 : 0));

  const runs = [];
  for (const p of comSeries) {
    const s = signOf(p.v);
    if (!runs.length || runs[runs.length - 1].sign !== s) {
      runs.push({ sign: s, start: p.t, end: p.t });
    } else {
      runs[runs.length - 1].end = p.t;
    }
  }

  const headingMeanWindow = (a, b) => {
    const vals = headingSeries.filter(p => p.t >= a && p.t <= b).map(p => p.v);
    return circularMeanDeg(vals);
  };
  const headingWindowOrNearest = (winStart, winEnd, fallbackT, fallbackMaxDist = 1.5) => {
    const winMean = headingMeanWindow(winStart, winEnd);
    if (Number.isFinite(winMean)) return winMean;
    return nearestSeriesValue(headingSeries, fallbackT, fallbackMaxDist);
  };

  const tacks = [];
  let i = 0;
  while (i < runs.length - 1) {
    const prev = runs[i];
    if (Math.abs(prev.sign) !== 1) { i++; continue; }

    let j = i + 1;
    const neutral = (j < runs.length && runs[j].sign === 0) ? runs[j] : null;
    if (neutral) j++;
    if (j >= runs.length) break;

    const next = runs[j];
    if (next.sign !== -prev.sign) { i++; continue; }

    const prevDur = prev.end - prev.start;
    const nextDur = next.end - next.start;
    if (prevDur < minStableSideSec || nextDur < minStableSideSec) { i++; continue; }

    const crossSearchStart = Math.max(prev.start, prev.end - 0.8);
    const crossSearchEnd = Math.min(next.end, next.start + 0.8);
    const comCrossT = findZeroCrossingTime(comSeries, crossSearchStart, crossSearchEnd)
      ?? (neutral ? (neutral.start + neutral.end) / 2 : (prev.end + next.start) / 2);
    const comStabilizeSec = Math.min(0.8, Math.max(0.35, nextDur * 0.35));
    const comStableT = Math.min(next.end, next.start + comStabilizeSec);
    if (!(comStableT > comCrossT)) { i++; continue; }

    const preHeading = headingWindowOrNearest(
      comCrossT - 1.2,
      comCrossT - 0.05,
      comCrossT - 0.35,
      1.8
    );
    const postHeading = headingWindowOrNearest(
      comStableT + 0.05,
      comStableT + 1.2,
      comStableT + 0.35,
      1.8
    );
    const headingDelta = (preHeading == null || postHeading == null)
      ? null
      : Math.abs(angularDiffDeg(postHeading, preHeading));
    if (!(Number.isFinite(headingDelta) && headingDelta >= minHeadingDeltaDeg)) { i++; continue; }

    const preHeadingUnwrapped = linearWindowOrNearest(
      headingSeriesUnwrapped,
      comCrossT - 1.2,
      comCrossT - 0.05,
      comCrossT - 0.35,
      1.8
    );
    const postHeadingUnwrapped = linearWindowOrNearest(
      headingSeriesUnwrapped,
      comStableT + 0.05,
      comStableT + 1.2,
      comStableT + 0.35,
      1.8
    );
    const signedHeadingDelta = (preHeadingUnwrapped == null || postHeadingUnwrapped == null)
      ? null
      : (postHeadingUnwrapped - preHeadingUnwrapped);
    const headingStartT = findHeadingChangeStart(
      headingSeriesUnwrapped,
      comCrossT,
      comStableT,
      preHeadingUnwrapped,
      signedHeadingDelta
    );
    const tackStart = Number.isFinite(headingStartT) ? headingStartT : comCrossT;
    const tackEnd = comStableT;
    if (!(tackEnd > tackStart)) { i++; continue; }

    const windowBeforeSec = 5.0;
    const windowAfterSec = 5.0;
    const winStart = tackStart - windowBeforeSec;
    const winEnd = tackEnd + windowAfterSec;
    const comWin = extractWindowSeries(comSeries, winStart, winEnd, tackStart);
    const rudderWin = extractWindowSeries(rudderSeries, winStart, winEnd, tackStart);
    const boomWin = extractWindowSeries(boomSeries, winStart, winEnd, tackStart);
    const headingWinRaw = extractWindowSeries(headingSeries, winStart, winEnd, tackStart);
    const headingUnwrapped = unwrapHeadingSeries(headingWinRaw);
    const headingRef = headingUnwrapped.find(p => p.t >= 0)?.v ?? headingUnwrapped[0]?.v ?? 0;
    const headingWin = headingUnwrapped.map(p => ({ t: p.t, v: p.v - headingRef }));
    const sogWin = extractWindowSeries(sogSeries, winStart, winEnd, tackStart);
    const trunkWin = extractWindowSeries(trunkSeries, winStart, winEnd, tackStart);

    tacks.push({
      start_t: tackStart,
      end_t: tackEnd,
      duration_s: tackEnd - tackStart,
      side_from: prev.sign < 0 ? 'port' : 'starboard',
      side_to: next.sign < 0 ? 'port' : 'starboard',
      heading_delta_deg: headingDelta,
      com_y: comWin,
      trunk: trunkWin,
      rudder: rudderWin,
      boom: boomWin,
      heading: headingWin,
      sog: sogWin,
    });

    i = j;
  }

  return { count: tacks.length, tacks };
}

function buildSegmentManeuverAnalysis(projectManeuvers, athleteId, segStartSec, segEndSec, acc) {
  const maneuvers = (Array.isArray(projectManeuvers) ? projectManeuvers : [])
    .filter(move => String(move?.athlete_id || '') === String(athleteId || ''))
    .filter(move => {
      const anchorTs = Number(move?.anchor_ts);
      return Number.isFinite(anchorTs) && anchorTs >= Number(segStartSec) && anchorTs <= Number(segEndSec);
    })
    .sort((a, b) => Number(a?.anchor_ts || 0) - Number(b?.anchor_ts || 0));

  if (!maneuvers.length) return { count: 0, moves: [] };

  const headingSeries = cleanTimeline(acc?.heading_time, p => normalizeHeadingDeg(p?.v));
  const rudderSeries = cleanTimeline(acc?.rudder_time);
  const boomSeries = cleanTimeline(acc?.boom_time);
  const sogSeries = cleanTimeline((acc?.sog_times || []).map((t, idx) => ({ t, v: acc?.sog_vals?.[idx] })));
  const trunkSeries = cleanTimeline(acc?.trunk_angle_time);
  const comSeries = cleanTimeline(acc?.com_y_time);

  const moves = maneuvers.map(move => {
    const startRel = Number(move.start_ts) - Number(segStartSec);
    const endRel = Number(move.end_ts) - Number(segStartSec);
    const durationS = Number.isFinite(endRel - startRel) ? (endRel - startRel) : Number(move?.duration_s);
    const winStart = startRel - 5.0;
    const winEnd = endRel + 5.0;
    const headingWinRaw = extractWindowSeries(headingSeries, winStart, winEnd, startRel);
    const headingUnwrapped = unwrapHeadingSeries(headingWinRaw);
    const headingRef = headingUnwrapped.find(point => point.t >= 0)?.v ?? headingUnwrapped[0]?.v ?? 0;
    const headingWin = headingUnwrapped.map(point => ({ t: point.t, v: point.v - headingRef }));
    return {
      ...move,
      start_t: startRel,
      end_t: endRel,
      duration_s: durationS,
      com_y: extractWindowSeries(comSeries, winStart, winEnd, startRel),
      trunk: extractWindowSeries(trunkSeries, winStart, winEnd, startRel),
      rudder: extractWindowSeries(rudderSeries, winStart, winEnd, startRel),
      boom: extractWindowSeries(boomSeries, winStart, winEnd, startRel),
      heading: headingWin,
      sog: extractWindowSeries(sogSeries, winStart, winEnd, startRel),
    };
  });

  return { count: moves.length, moves };
}

function sortTimelineByTInPlace(points) {
  if (!Array.isArray(points)) return points;
  points.sort((a, b) => Number(a?.t || 0) - Number(b?.t || 0));
  return points;
}

function sortRowsByTsInPlace(rows) {
  if (!Array.isArray(rows)) return rows;
  rows.sort((a, b) => Number(a?.ts || 0) - Number(b?.ts || 0));
  return rows;
}

function normalizeEpochSeconds(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  const a = Math.abs(n);
  if (a > 1e14) return n / 1e6; // microseconds
  if (a > 1e11) return n / 1e3; // milliseconds
  return n;
}

function buildAthleteCsvTrackLookup(csvTrackPointsByFile, fileMeta = {}) {
  const byAthlete = new Map();

  for (const [fileId, points] of Object.entries(csvTrackPointsByFile || {})) {
    const athleteIdRaw = fileMeta?.[fileId]?.athlete_id;
    const athleteId = athleteIdRaw == null ? '' : String(athleteIdRaw);
    if (!athleteId || !Array.isArray(points) || points.length === 0) continue;

    if (!byAthlete.has(athleteId)) byAthlete.set(athleteId, []);
    const bucket = byAthlete.get(athleteId);
    for (const point of points) {
      const ts = normalizeEpochSeconds(point?.ts);
      const lat = Number(point?.lat);
      const lon = Number(point?.lon);
      if (!Number.isFinite(ts) || !Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      bucket.push({ ...point, ts });
    }
  }

  for (const [athleteId, points] of byAthlete.entries()) {
    points.sort((a, b) => Number(a?.ts || 0) - Number(b?.ts || 0));
    const deduped = [];
    let lastTs = Number.NaN;
    for (const point of points) {
      const ts = Number(point?.ts);
      if (deduped.length && Math.abs(ts - lastTs) <= 1e-6) {
        deduped[deduped.length - 1] = point;
      } else {
        deduped.push(point);
      }
      lastTs = ts;
    }
    byAthlete.set(athleteId, deduped);
  }

  return byAthlete;
}

function sliceTrackPointsByTimeRange(points, startSec, endSec) {
  if (!Array.isArray(points) || points.length === 0) return [];
  const start = normalizeEpochSeconds(startSec);
  const end = normalizeEpochSeconds(endSec);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return [];
  return points.filter(point => {
    const ts = normalizeEpochSeconds(point?.ts);
    return Number.isFinite(ts) && ts >= start && ts <= end;
  });
}

/**
 * Normalize report XY orientation for report outputs
 * (x = fore/aft, y = port/stbd with port < 0 in this project setup).
 *
 * The placement pipeline (rayplane.js) produces the same Y-axis convention
 * as the current project pipeline — negative Y = port, positive Y = starboard.
 * The 3D hull viewer already renders raw skeleton data correctly without
 * any Y-flip, confirming the stored data has the right sign.
 *
 * Only apply a Y-flip when explicitly requested via config (report_flip_y).
 * The previous auto-detection heuristic (left-hip vs right-hip Y comparison)
 * was inverted: with a stern-mounted camera, the person's left hip
 * (MediaPipe lm23) naturally maps to starboard (-Y), making the median
 * negative in the NORMAL case.  The heuristic interpreted that as "mirrored"
 * and flipped, which caused the heatmap to appear on the wrong side.
 */
function normalizeReportAxes(acc, cvConfig = {}) {
  const cfgFlipY = cvConfig?.report_flip_y;
  let flipY = false;

  if (typeof cfgFlipY === 'boolean') {
    flipY = cfgFlipY;
  }
  // No auto-detection — matches Python behavior which never flips Y.

  if (!flipY) return;

  if (Array.isArray(acc.all_keypoints_xy)) {
    acc.all_keypoints_xy = acc.all_keypoints_xy.map(([x, y]) => [x, -y]);
  }
  if (Array.isArray(acc.all_com_xy)) {
    acc.all_com_xy = acc.all_com_xy.map(([x, y]) => [x, -y]);
  }
  if (Array.isArray(acc.com_ys)) {
    acc.com_ys = acc.com_ys.map(v => (v == null ? v : -v));
  }
  if (Array.isArray(acc.metrics_in_range)) {
    for (const m of acc.metrics_in_range) {
      if (m && Number.isFinite(m.com_y)) m.com_y = -m.com_y;
      if (m && Number.isFinite(m.roll_moment)) m.roll_moment = -m.roll_moment;
    }
  }
}


// ── Density grid / heatmap ────────────────────────────────────────────

/**
 * Generate a density heatmap image from 2D points.
 * Returns { image_b64, width, height, grid_size_x, grid_size_y, ... }.
 */
export function generateDensityGrid(pointsXY, opts = {}) {
  const {
    grid_size_x = 5.0, grid_size_y = 3.0,
    grid_center_x = -1.5, grid_center_y = 0.0,
    resolution = 120, sigma_cells = 2.5,
  } = opts;

  if (pointsXY.length === 0) return null;

  const cellsX = resolution;
  const cellsY = Math.round(resolution * (grid_size_y / grid_size_x));
  const cellW = grid_size_x / cellsX;
  const cellH = grid_size_y / cellsY;
  const originX = grid_center_x - grid_size_x / 2;
  const originY = grid_center_y - grid_size_y / 2;

  // Accumulate density
  const grid = new Float64Array(cellsX * cellsY);
  const radius = Math.ceil(sigma_cells * 3);
  const sigma2 = sigma_cells * sigma_cells;

  for (const [px, py] of pointsXY) {
    const gx = (px - originX) / cellW;
    const gy = (py - originY) / cellH;
    const ix0 = Math.max(0, Math.floor(gx) - radius);
    const ix1 = Math.min(cellsX - 1, Math.floor(gx) + radius);
    const iy0 = Math.max(0, Math.floor(gy) - radius);
    const iy1 = Math.min(cellsY - 1, Math.floor(gy) + radius);
    for (let iy = iy0; iy <= iy1; iy++) {
      for (let ix = ix0; ix <= ix1; ix++) {
        const d2 = (gx - ix) ** 2 + (gy - iy) ** 2;
        grid[iy * cellsX + ix] += Math.exp(-d2 / (2 * sigma2));
      }
    }
  }

  // Normalize
  let maxVal = 0;
  for (let i = 0; i < grid.length; i++) if (grid[i] > maxVal) maxVal = grid[i];
  if (maxVal > 0) for (let i = 0; i < grid.length; i++) grid[i] /= maxVal;

  // Render to Canvas
  const canvas = document.createElement('canvas');
  canvas.width = cellsX;
  canvas.height = cellsY;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(cellsX, cellsY);

  for (let iy = 0; iy < cellsY; iy++) {
    for (let ix = 0; ix < cellsX; ix++) {
      const d = grid[(cellsY - 1 - iy) * cellsX + ix]; // flip Y: row-0 = bottom
      const rgba = densityToColor(d);
      const idx = (iy * cellsX + ix) * 4;
      imgData.data[idx] = rgba[0];
      imgData.data[idx + 1] = rgba[1];
      imgData.data[idx + 2] = rgba[2];
      imgData.data[idx + 3] = rgba[3];
    }
  }
  ctx.putImageData(imgData, 0, 0);

  return {
    image_b64: canvas.toDataURL('image/png'),
    width: cellsX,
    height: cellsY,
    grid_size_x, grid_size_y,
    grid_center_x, grid_center_y,
    point_count: pointsXY.length,
  };
}

function densityToColor(d) {
  if (d < 0.01) return [0, 0, 0, 0];
  let r, g, b;
  if (d < 0.25) {
    const t = d / 0.25;
    r = 0; g = Math.round(225 * t); b = 255;
  } else if (d < 0.5) {
    const t = (d - 0.25) / 0.25;
    r = 0; g = 255; b = Math.round(255 * (1 - t));
  } else if (d < 0.75) {
    const t = (d - 0.5) / 0.25;
    r = Math.round(255 * t); g = 255; b = 0;
  } else {
    const t = (d - 0.75) / 0.25;
    r = 255; g = Math.round(255 * (1 - t)); b = 0;
  }
  const a = Math.round((0.8 + d * 0.2) * 255);
  return [r, g, b, a];
}


// ── Golds (best-athlete per metric per segment) ──────────────────────

const GOLD_CATEGORIES = [
  ['max_sog',          'sog',         'max', true],
  ['avg_sog',          'sog',         'avg', true],
  ['best_upwind_vmg',  'vmg_upwind',  'max', true],
  ['avg_upwind_vmg',   'vmg_upwind',  'avg', true],
  ['best_downwind_vmg','vmg_downwind','max', true],
  ['avg_downwind_vmg', 'vmg_downwind','avg', true],
  ['max_trunk_angle',  'trunk_angle', 'max', true],
  ['avg_trunk_angle',  'trunk_angle', 'avg', true],
  ['max_moment_roll',  'moment_roll', 'max', true],
  ['avg_moment_roll',  'moment_roll', 'avg', true],
  ['max_heel',         'heel',        'max', true],
  ['avg_heel',         'heel',        'avg', true],
];

function computeGolds(segResults) {
  const golds = {};
  for (const [cat, metricKey, statKey, higherBetter] of GOLD_CATEGORIES) {
    golds[cat] = {};
    // Group segments by split_id
    const bySegment = {};
    for (const seg of segResults) {
      if (!bySegment[seg.split_id]) bySegment[seg.split_id] = [];
      bySegment[seg.split_id].push(seg);
    }
    for (const [splitId, segs] of Object.entries(bySegment)) {
      const withData = segs.filter(s => s[metricKey]?.[statKey] != null);
      if (withData.length < 2) continue;
      let best = withData[0];
      for (const s of withData) {
        const val = Math.abs(s[metricKey][statKey]);
        const bestVal = Math.abs(best[metricKey][statKey]);
        if (higherBetter ? val > bestVal : val < bestVal) best = s;
      }
      golds[cat][splitId] = best.athlete_name;
    }
  }
  return golds;
}

function buildVmgStatBlock(modeSummary) {
  return {
    avg: Number.isFinite(Number(modeSummary?.avg)) ? Number(modeSummary.avg) : null,
    max: Number.isFinite(Number(modeSummary?.best_stable)) ? Number(modeSummary.best_stable) : null,
    min: Number.isFinite(Number(modeSummary?.min)) ? Number(modeSummary.min) : null,
    std: Number.isFinite(Number(modeSummary?.std)) ? Number(modeSummary.std) : null,
    count: Number.isFinite(Number(modeSummary?.count)) ? Number(modeSummary.count) : 0,
    coverage_s: Number.isFinite(Number(modeSummary?.coverage_s)) ? Number(modeSummary.coverage_s) : 0,
    eligible: !!modeSummary?.eligible,
  };
}

function applySegmentVmgComparison(segResults) {
  if (!Array.isArray(segResults) || !segResults.length) return;
  const upwindCoverage = segResults.reduce((sum, seg) => sum + (Number(seg?.vmg_summary?.upwind?.coverage_s) || 0), 0);
  const downwindCoverage = segResults.reduce((sum, seg) => sum + (Number(seg?.vmg_summary?.downwind?.coverage_s) || 0), 0);
  let compareMode = upwindCoverage >= downwindCoverage ? 'upwind' : 'downwind';

  let candidates = segResults
    .map(seg => ({
      seg,
      value: Number(seg?.vmg_summary?.[compareMode]?.avg),
      eligible: !!seg?.vmg_summary?.[compareMode]?.eligible,
    }))
    .filter(row => row.eligible && Number.isFinite(row.value));
  if (!candidates.length) {
    const fallbackMode = compareMode === 'upwind' ? 'downwind' : 'upwind';
    const fallbackCandidates = segResults
      .map(seg => ({
        seg,
        value: Number(seg?.vmg_summary?.[fallbackMode]?.avg),
        eligible: !!seg?.vmg_summary?.[fallbackMode]?.eligible,
      }))
      .filter(row => row.eligible && Number.isFinite(row.value));
    if (fallbackCandidates.length) {
      compareMode = fallbackMode;
      candidates = fallbackCandidates;
    }
  }

  let leader = null;
  for (const row of candidates) {
    if (!leader || row.value > leader.value) leader = row;
  }

  for (const seg of segResults) {
    const athleteValue = Number(seg?.vmg_summary?.[compareMode]?.avg);
    seg.vmg_comparison = {
      mode: leader ? compareMode : null,
      leader_athlete_name: leader?.seg?.athlete_name || null,
      leader_value_kts: Number.isFinite(leader?.value) ? leader.value : null,
      athlete_value_kts: Number.isFinite(athleteValue) ? athleteValue : null,
      gap_to_leader_kts: leader && Number.isFinite(athleteValue)
        ? Math.max(0, leader.value - athleteValue)
        : null,
      eligible: !!leader && !!seg?.vmg_summary?.[compareMode]?.eligible,
    };
  }
}


// ── Midpoint / COM from landmarks ─────────────────────────────────────

function midpointLm(lm, a, b) {
  const pa = Array.isArray(a) ? midpointLm(lm, a[0], a[1]) : lm[String(a)];
  const pb = Array.isArray(b) ? midpointLm(lm, b[0], b[1]) : lm[String(b)];
  if (!pa && !pb) return null;
  if (!pa) return pb;
  if (!pb) return pa;
  return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2];
}

function computeComFromLandmarks(lm) {
  let comX = 0, comY = 0, comZ = 0, totalMass = 0;
  for (const [a, b, massFrac, comFrac] of _COM_SEGMENTS) {
    const pA = Array.isArray(a) ? midpointLm(lm, a[0], a[1]) : lm[String(a)];
    const pB = Array.isArray(b) ? midpointLm(lm, b[0], b[1]) : lm[String(b)];
    if (!pA || !pB) continue;
    const segCom = [
      pA[0] + comFrac * (pB[0] - pA[0]),
      pA[1] + comFrac * (pB[1] - pA[1]),
      pA[2] + comFrac * (pB[2] - pA[2]),
    ];
    comX += segCom[0] * massFrac;
    comY += segCom[1] * massFrac;
    comZ += segCom[2] * massFrac;
    totalMass += massFrac;
  }
  if (totalMass < 0.35) return null;
  return [comX / totalMass, comY / totalMass, comZ / totalMass];
}


// ── Main entry point ──────────────────────────────────────────────────

/**
 * Build report data for selected segments.
 *
 * @param {string} projectId
 * @param {string[]} segmentIds — which segments to include
 * @param {(msg: string, pct: number) => void} [onProgress]
 * @param {{ includeDensityImages?: boolean, includeLegacyVisuals?: boolean, polarPlots?: boolean, timelineMaxPoints?: number|null, maxHeatmapPoints?: { keypoints?: number, com?: number }, wind?: { session?: object, byCsvId?: Record<string, object> } }} [opts]
 * @returns {object} — report data payload
 */
export async function buildReportData(projectId, segmentIds, onProgress = null, opts = {}) {
  const report = (msg, pct) => { if (onProgress) onProgress(msg, pct); };
  const includeDensityImages = opts?.includeDensityImages !== false;
  const includeLegacyVisuals = opts?.includeLegacyVisuals !== false;
  const includePolarPlots = !!opts?.polarPlots;
  const includeManeuverAnalysis = !!opts?.maneuverAnalysis;
  const timelineMaxPoints = Object.prototype.hasOwnProperty.call(opts || {}, 'timelineMaxPoints')
    ? opts.timelineMaxPoints
    : 600;
  const maxHeatmapKeypoints = Number.isFinite(Number(opts?.maxHeatmapPoints?.keypoints))
    ? Math.max(300, Math.round(Number(opts.maxHeatmapPoints.keypoints)))
    : null;
  const maxHeatmapComPoints = Number.isFinite(Number(opts?.maxHeatmapPoints?.com))
    ? Math.max(120, Math.round(Number(opts.maxHeatmapPoints.com)))
    : null;
  const sessionWind = opts?.wind?.session && Number.isFinite(Number(opts.wind.session.directionDeg))
    ? opts.wind.session
    : null;
  const windByCsvId = Object.fromEntries(
    Object.entries(opts?.wind?.byCsvId || {}).map(([csvId, row]) => [String(csvId), row?.localSeries || null]),
  );
  report('Loading project data...', 0);

  // 1. Gather project data
  const [allSegments, athletes, fileMeta, cvConfigRec, files, tracks, matches] = await Promise.all([
    DB.getSegments(projectId),
    DB.getAthletes(projectId),
    DB.getFileMeta(projectId),
    DB.getCvConfig(projectId),
    DB.listFiles(projectId),
    DB.listTracks(projectId),
    DB.listMatches(projectId),
  ]);

  const cvConfig = cvConfigRec?.config || {};
  const athleteWeight = cvConfig.athlete_weight || 75.0;
  const boatCom = cvConfig.boat_com || 0.0;

  const segments = allSegments.filter(s => segmentIds.includes(s.id));
  if (segments.length === 0) throw new Error('No segments selected');

  // 2. Build track points lookups and merge CSV instrument channels onto videos
  const trackPointsByFile = {};
  const trackKindByFile = {};
  const trackBatchSize = 6;
  for (let trackIdx = 0; trackIdx < tracks.length; trackIdx += trackBatchSize) {
    const batch = tracks.slice(trackIdx, trackIdx + trackBatchSize);
    const batchRows = await Promise.all(batch.map(async (track) => ({
      track,
      pts: await DB.getTrackPoints(track.id),
    })));
    for (const row of batchRows) {
      trackPointsByFile[row.track.file_id] = row.pts;
      trackKindByFile[row.track.file_id] = row.track.kind;
    }
    const loadedCount = Math.min(trackIdx + batch.length, tracks.length);
    report('Loading track data...', 0.04 + (loadedCount / Math.max(1, tracks.length)) * 0.08);
    await yieldReportWork();
  }

  const videoTrackPointsByFile = {};
  const csvTrackPointsByFile = {};
  for (const [fid, pts] of Object.entries(trackPointsByFile)) {
    const kind = trackKindByFile[fid];
    if (kind === 'video') videoTrackPointsByFile[fid] = pts;
    else if (kind === 'csv') csvTrackPointsByFile[fid] = pts;
  }
  mergeCsvInstrumentsIntoVideoTracks(videoTrackPointsByFile, csvTrackPointsByFile, matches);
  const trackIndexByFile = {};
  for (const [fid, pts] of Object.entries(videoTrackPointsByFile)) {
    trackIndexByFile[fid] = buildTrackIndex(pts);
  }
  report('Preparing report inputs...', 0.14);
  await yieldReportWork();

  const projectManeuvers = includeManeuverAnalysis
    ? await detectProjectManeuvers(projectId, {
        windContext: opts?.wind || {},
      })
    : [];
  if (includeManeuverAnalysis) {
    report('Preparing maneuver summaries...', 0.18);
    await yieldReportWork();
  }

  const bestCsvByVideo = {};
  for (const match of (matches || [])) {
    if (Number(match?.rank) !== 1 || !match?.video_file_id || !match?.csv_file_id) continue;
    if (!(match.video_file_id in bestCsvByVideo)) bestCsvByVideo[match.video_file_id] = match.csv_file_id;
  }

  const athleteCsvTrackPoints = includePolarPlots
    ? buildAthleteCsvTrackLookup(csvTrackPointsByFile, fileMeta)
    : new Map();
  const sessionPolarCache = new Map();
  const segmentPolarCache = new Map();
  const getAthleteCsvPoints = (athleteId) => athleteCsvTrackPoints.get(String(athleteId ?? '')) || [];
  const getSessionPolar = async (athleteId, athleteName) => {
    const key = String(athleteId ?? '');
    if (!includePolarPlots || !key || key === 'unassigned') return null;
    if (sessionPolarCache.has(key)) return sessionPolarCache.get(key);

    const task = (async () => {
      const points = getAthleteCsvPoints(key);
      if (points.length < 80) return null;
      try {
        return await buildAthletePolarFromPoints(points, {
          sourceName: athleteName || key,
          minTrackSamples: 80,
          minSamplesPerBin: 12,
          statisticQuantile: 0.8,
          yieldBudgetMs: 10,
        });
      } catch (err) {
        console.warn(`[report-builder] session polar failed for athlete ${athleteName || key}:`, err);
        return null;
      }
    })();

    sessionPolarCache.set(key, task);
    return task;
  };
  const getSegmentPolar = async (athleteId, athleteName, segmentName, segStartSec, segEndSec, sessionPolar) => {
    const key = `${String(athleteId ?? '')}:${Number(segStartSec)}:${Number(segEndSec)}`;
    if (!includePolarPlots || !athleteId || athleteId === 'unassigned') return null;
    if (segmentPolarCache.has(key)) return segmentPolarCache.get(key);

    const task = (async () => {
      const segmentPoints = sliceTrackPointsByTimeRange(getAthleteCsvPoints(athleteId), segStartSec, segEndSec);
      if (segmentPoints.length < 45) return null;

      const priorDirectionDeg = Number(sessionPolar?.wind?.directionDeg);
      const priorSpeedKts = Number(sessionPolar?.wind?.speedKts);
      const windEstimateOptions = {};
      if (Number.isFinite(priorDirectionDeg)) {
        windEstimateOptions.coarseDirectionCenterDeg = priorDirectionDeg;
        windEstimateOptions.coarseDirectionHalfSpanDeg = 35;
        windEstimateOptions.priorDirectionDeg = priorDirectionDeg;
        windEstimateOptions.priorDirectionWeight = 0.00035;
      }
      if (Number.isFinite(priorSpeedKts)) {
        windEstimateOptions.minTrueWindSpeedKts = Math.max(2, priorSpeedKts - 5);
        windEstimateOptions.maxTrueWindSpeedKts = Math.min(24, priorSpeedKts + 5);
        windEstimateOptions.priorSpeedKts = priorSpeedKts;
        windEstimateOptions.priorSpeedWeight = 0.012;
      }

      try {
        return await buildAthletePolarFromPoints(segmentPoints, {
          sourceName: `${athleteName || athleteId} - ${segmentName}`,
          minTrackSamples: 45,
          minSamplesPerBin: 6,
          statisticQuantile: 0.8,
          yieldBudgetMs: 10,
          windEstimateOptions,
        });
      } catch (err) {
        console.warn(`[report-builder] segment polar failed for athlete ${athleteName || athleteId} in ${segmentName}:`, err);
        return null;
      }
    })();

    segmentPolarCache.set(key, task);
    return task;
  };

  // 3. Identify video files
  const videoFiles = files.filter(f => f.kind === 'video' && !f.playback_only);
  const metricsByFile = new Map();
  const loadMetricsCached = async (fileId) => {
    if (!metricsByFile.has(fileId)) {
      metricsByFile.set(fileId, loadMetrics(projectId, fileId).then(buildMetricIndex));
    }
    return metricsByFile.get(fileId);
  };

  // 3b. Full-session GPS tracks for Python-style summary map
  const fullTrackPoints = {};
  let gpsSessionStart = Infinity;
  let gpsSessionEnd = -Infinity;
  for (let videoIdx = 0; videoIdx < videoFiles.length; videoIdx++) {
    const vf = videoFiles[videoIdx];
    const pts = (videoTrackPointsByFile[vf.id] || trackPointsByFile[vf.id] || [])
      .filter(p => Number.isFinite(Number(p?.lat)) && Number.isFinite(Number(p?.lon)))
      .map(p => ({
        lat: Number(p.lat),
        lon: Number(p.lon),
        ts: normalizeEpochSeconds(p?.ts),
      }));

    for (const p of pts) {
      const ts = Number(p?.ts);
      if (!Number.isFinite(ts)) continue;
      if (ts < gpsSessionStart) gpsSessionStart = ts;
      if (ts > gpsSessionEnd) gpsSessionEnd = ts;
    }

    if (pts.length >= 2) {
      // Keep payload bounded while preserving track shape.
      fullTrackPoints[vf.id] = subsampleTimeSeries(pts, 3000);
    }
    if ((videoIdx + 1) % 4 === 0 || videoIdx === videoFiles.length - 1) {
      report('Preparing GPS tracks...', 0.14 + ((videoIdx + 1) / Math.max(1, videoFiles.length)) * 0.06);
      await yieldReportWork();
    }
  }

  // 4. Process each segment
  const segResults = [];
  let segIdx = 0;
  const segmentPctStart = 0.20;
  const segmentPctSpan = 0.72;

  for (const seg of segments) {
    segIdx++;
    const segStartSec = normalizeEpochSeconds(seg.tsStart);
    const segEndSec = normalizeEpochSeconds(seg.tsEnd);
    const segmentDuration = Math.max(0, Number(segEndSec) - Number(segStartSec));
    const pctBase = segmentPctStart + ((segIdx - 1) / Math.max(1, segments.length)) * segmentPctSpan;
    const pctStep = segmentPctSpan / Math.max(1, segments.length);
    const reportSegmentProgress = (msg, frac) => {
      report(msg, pctBase + pctStep * Math.max(0, Math.min(1, frac)));
    };
    reportSegmentProgress(`Analyzing segment: ${seg.name}...`, 0);
    await yieldReportWork();

    // Find video files that overlap this segment's time range
    const overlapping = [];
    for (const vf of videoFiles) {
      const meta = fileMeta[vf.id] || {};
      const trackPts = videoTrackPointsByFile[vf.id] || trackPointsByFile[vf.id];
      const trackIndex = trackIndexByFile[vf.id] || buildTrackIndex(trackPts);
      if (!trackPts?.length) continue;
      const trackStart = trackIndex.tsSorted ? trackIndex.tsVals[0] : trackPts[0].ts;
      const trackEnd = trackIndex.tsSorted ? trackIndex.tsVals[trackIndex.tsVals.length - 1] : trackPts[trackPts.length - 1].ts;
      // Check overlap
      if (seg.tsEnd < trackStart || seg.tsStart > trackEnd) continue;
      overlapping.push({ file: vf, meta, trackPts, trackIndex });
    }
    overlapping.sort((a, b) => {
      const aStart = Math.max(Number(seg.tsStart) || 0, Number(a?.trackPts?.[0]?.ts) || 0);
      const bStart = Math.max(Number(seg.tsStart) || 0, Number(b?.trackPts?.[0]?.ts) || 0);
      return aStart - bStart;
    });
    reportSegmentProgress(`Analyzing segment: ${seg.name}...`, 0.08);

    // For each overlapping file, collect skeleton + metrics + GPS data
    const perAthlete = {}; // athleteId → accumulated data

    for (let overlapIdx = 0; overlapIdx < overlapping.length; overlapIdx++) {
      const { file, meta, trackPts, trackIndex } = overlapping[overlapIdx];
      // Resolve athlete: direct assignment → derive from matched CSV
      let athId = meta.athlete_id || null;
      if (!athId) {
        // Find best-match CSV for this video and use its athlete_id
        const bestMatch = (matches || []).find(m => m.video_file_id === file.id && Number(m.rank) === 1);
        if (bestMatch?.csv_file_id) {
          athId = (fileMeta[bestMatch.csv_file_id] || {}).athlete_id || null;
        }
      }
      if (!athId) athId = 'unassigned';
      const ath = athletes.find(a => a.id === athId);
      const athName = ath?.name || 'Unassigned';
      const athColor = ath?.color || null;

      if (!perAthlete[athId]) {
        perAthlete[athId] = {
          athlete_id: athId, athlete_name: athName,
          athlete_color: athColor,
          trunk_angles: [], moments_pitch: [], moments_roll: [],
          com_xs: [], com_ys: [], com_zs: [],
          trunk_angle_time: [], moment_time: [], pitch_moment_time: [],
          all_keypoints_xy: [], all_com_xy: [],
          lr_hip_y_diffs: [],
          sog_times: [], sog_vals: [], sog_lats: [], sog_lons: [],
          heel_vals: [], heel_time: [],
          rudder_vals: [], rudder_time: [],
          boom_vals: [], boom_time: [],
          heading_vals: [], heading_time: [],
          com_y_time: [],
          gps_path: [],
          metrics_in_range: [],
          vmg_samples: [],
        };
      }
      const acc = perAthlete[athId];
      const bestCsvId = bestCsvByVideo[file.id] ? String(bestCsvByVideo[file.id]) : null;

      // Convert segment epoch range to video_s range
      const vsStart = absTs2VideoSecIndexed(trackIndex, seg.tsStart);
      const vsEnd = absTs2VideoSecIndexed(trackIndex, seg.tsEnd);
      if (vsStart == null || vsEnd == null) continue;
      const normalizeSegmentTime = (t) => {
        const n = Number(t);
        if (!Number.isFinite(n)) return null;
        if (n < -0.35 || n > segmentDuration + 0.35) return null;
        return Math.max(0, Math.min(segmentDuration, n));
      };
      const toSegmentT = (videoSec) => normalizeSegmentTime(videoSec2SegmentSecIndexed(trackIndex, videoSec, segStartSec));

      // Load skeleton + metrics from OPFS
      const [skelFrames, metricsIndex] = await Promise.all([
        loadSkeletonFrames(projectId, file.id, vsStart, vsEnd),
        loadMetricsCached(file.id),
      ]);
      const metricsWindow = sliceMetricsByVideoRange(metricsIndex, vsStart, vsEnd);
      const metricsWindowNormalized = metricsWindow
        .map(m => {
          const t = toSegmentT(m.ts);
          return Number.isFinite(t) ? { raw: m, t } : null;
        })
        .filter(Boolean);
      const metricsInRange = metricsWindowNormalized.filter(entry => entry.raw?.detected);

      acc.metrics_in_range.push(...metricsInRange.map(({ raw, t }) => ({ ...raw, ts: t })));

      for (const { raw: m, t } of metricsWindowNormalized) {
        const rudder = Number(m.rudder_angle);
        if (Number.isFinite(rudder) && rudder >= -90 && rudder <= 90) {
          acc.rudder_vals.push(rudder);
          acc.rudder_time.push({ t, v: rudder });
        }
        const boom = Number(m.boom_angle);
        if (Number.isFinite(boom) && boom >= -360 && boom <= 360) {
          acc.boom_vals.push(boom);
          acc.boom_time.push({ t, v: boom });
        }
      }

      // Collect metrics
      for (const { raw: m, t } of metricsInRange) {
        if (m.trunk_angle != null && m.trunk_angle >= 0 && m.trunk_angle <= 180) {
          acc.trunk_angles.push(m.trunk_angle);
          acc.trunk_angle_time.push({ t, v: m.trunk_angle });
        }
        if (m.pitch_moment != null) {
          acc.moments_pitch.push(m.pitch_moment);
          acc.pitch_moment_time.push({ t, v: m.pitch_moment });
        }
        if (m.roll_moment != null) {
          acc.moments_roll.push(Math.abs(m.roll_moment));
          acc.moment_time.push({ t, v: m.roll_moment });
        }
        if (m.com_x != null) {
          acc.com_xs.push(m.com_x);
          acc.com_ys.push(m.com_y);
          acc.com_zs.push(m.com_z);
          acc.all_com_xy.push([m.com_x, m.com_y]);
        }
        if (Number.isFinite(Number(m.com_y))) {
          acc.com_y_time.push({ t, v: Number(m.com_y) });
        }
      }

      // Collect keypoint positions from skeleton frames
      for (const frame of skelFrames) {
        if (!frame.skeleton) continue;
        for (const [idx, pt] of Object.entries(frame.skeleton)) {
          if (pt && pt.length >= 2) {
            acc.all_keypoints_xy.push([pt[0], pt[1]]);
          }
        }
        const lh = frame.skeleton?.['23'];
        const rh = frame.skeleton?.['24'];
        if (
          Array.isArray(lh) && lh.length >= 2 &&
          Array.isArray(rh) && rh.length >= 2 &&
          Number.isFinite(lh[1]) && Number.isFinite(rh[1])
        ) {
          acc.lr_hip_y_diffs.push(lh[1] - rh[1]);
        }
      }

      // SOG from GPS track
      const sog = computeSogFromTrackIndexed(trackIndex, vsStart, vsEnd);
      for (let i = 0; i < sog.times.length; i++) {
        const t = toSegmentT(sog.times[i]);
        if (!Number.isFinite(t)) continue;
        acc.sog_times.push(t);
        acc.sog_vals.push(sog.sogs[i]);
        acc.sog_lats.push(sog.lats[i]);
        acc.sog_lons.push(sog.lons[i]);
      }

      // Heel from track points (if present)
      const rangeTrackPts = sliceTrackByVideoRange(trackIndex, vsStart, vsEnd);
      const heelPts = rangeTrackPts.filter(p => p.heel != null);
      for (const p of heelPts) {
        const t = toSegmentT(p.video_s);
        if (!Number.isFinite(t)) continue;
        acc.heel_vals.push(p.heel);
        acc.heel_time.push({ t, v: p.heel });
      }

      const headingPts = rangeTrackPts.filter(p => p.hdg != null || p.cog != null);
      for (const p of headingPts) {
        const t = toSegmentT(p.video_s);
        if (!Number.isFinite(t)) continue;
        const vRaw = p.hdg != null ? Number(p.hdg) : Number(p.cog);
        const v = normalizeHeadingDeg(vRaw);
        if (v == null) continue;
        acc.heading_vals.push(v);
        acc.heading_time.push({ t, v });
      }

      // GPS path for map
      const gpsPts = rangeTrackPts;
      for (const p of gpsPts) {
        const t = toSegmentT(p.video_s);
        if (!Number.isFinite(t)) continue;
        const absTs = normalizeEpochSeconds(p?.ts);
        if (Number.isFinite(absTs) && (p.cog != null || p.hdg != null)) {
          const motionDirDeg = p.cog != null ? Number(p.cog) : Number(p.hdg);
          const sogKts = Number.isFinite(Number(p?.sog))
            ? Number(p.sog)
            : findNearestSog(acc.sog_times, acc.sog_vals, t);
          acc.vmg_samples.push({
            t,
            absTs,
            sogKts,
            motionDirDeg,
            localWindKey: bestCsvId,
          });
        }
        const sogAtPoint = acc.sog_vals.length > 0
          ? findNearestSog(acc.sog_times, acc.sog_vals, t) : null;
        acc.gps_path.push({ lat: p.lat, lon: p.lon, sog: sogAtPoint });
      }

      reportSegmentProgress(
        `Analyzing segment: ${seg.name} (${overlapIdx + 1}/${Math.max(1, overlapping.length)} videos)`,
        0.12 + 0.68 * ((overlapIdx + 1) / Math.max(1, overlapping.length))
      );
      await yieldReportWork();
    }

    // Build per-athlete results for this segment
    const athleteResults = [];
    const athleteEntries = Object.entries(perAthlete);

    for (let athleteIdx = 0; athleteIdx < athleteEntries.length; athleteIdx++) {
      const [athId, acc] = athleteEntries[athleteIdx];
      sortRowsByTsInPlace(acc.metrics_in_range);
      sortTimelineByTInPlace(acc.trunk_angle_time);
      sortTimelineByTInPlace(acc.moment_time);
      sortTimelineByTInPlace(acc.pitch_moment_time);
      sortTimelineByTInPlace(acc.heel_time);
      sortTimelineByTInPlace(acc.rudder_time);
      sortTimelineByTInPlace(acc.boom_time);
      sortTimelineByTInPlace(acc.heading_time);
      sortTimelineByTInPlace(acc.com_y_time);
      normalizeReportAxes(acc, cvConfig);
      const vmg = computeSegmentVmg(acc.vmg_samples, {
        sessionWind,
        localWindSeriesByKey: windByCsvId,
      });
      const sessionPolar = includePolarPlots
        ? await getSessionPolar(athId, acc.athlete_name)
        : null;
      const segmentPolar = includePolarPlots
        ? await getSegmentPolar(athId, acc.athlete_name, seg.name, segStartSec, segEndSec, sessionPolar)
        : null;
      const legacyTackAnalysis = buildTackAnalysis(
        acc.com_y_time,
        acc.heading_time,
        acc.rudder_time,
        acc.sog_times.map((t, idx) => ({ t, v: acc.sog_vals[idx] })),
        acc.trunk_angle_time,
        acc.boom_time
      );
      const maneuverAnalysis = includeManeuverAnalysis
        ? buildSegmentManeuverAnalysis(projectManeuvers, athId, segStartSec, segEndSec, acc)
        : null;
      const maneuverTacks = maneuverAnalysis
        ? maneuverAnalysis.moves.filter(move => String(move?.type || '') === 'tack')
        : legacyTackAnalysis.tacks;

      const segResult = {
        split_id: seg.id,
        name: seg.name,
        start_s: seg.tsStart,
        end_s: seg.tsEnd,
        file_id: overlapping.map(o => o.file.id),
        athlete_id: athId,
        athlete_name: acc.athlete_name,
        athlete_color: acc.athlete_color || null,
        color: acc.athlete_color || null,
        duration_s: seg.tsEnd - seg.tsStart,

        sog: stats(acc.sog_vals),
        moment_roll: stats(acc.moments_roll.map(Math.abs), true),
        moment_pitch: stats(acc.moments_pitch, true),
        trunk_angle: stats(acc.trunk_angles),
        heel: stats(acc.heel_vals),
        rudder: stats(acc.rudder_vals),
        boom: stats(acc.boom_vals),
        vmg_summary: vmg.summary,
        vmg_upwind: buildVmgStatBlock(vmg.summary?.upwind),
        vmg_downwind: buildVmgStatBlock(vmg.summary?.downwind),
        vmg_comparison: null,
        polar_session: sessionPolar,
        polar_segment: segmentPolar,

        gps_path: subsampleTimeSeries(acc.gps_path, 500),
        kp_xy: maxHeatmapKeypoints ? subsampleTimeSeries(acc.all_keypoints_xy, maxHeatmapKeypoints) : acc.all_keypoints_xy,
        com_xy: maxHeatmapComPoints ? subsampleTimeSeries(acc.all_com_xy, maxHeatmapComPoints) : acc.all_com_xy,

        trunk_angle_timeline: subsampleTimeSeries(acc.trunk_angle_time, timelineMaxPoints),
        moment_timeline: subsampleTimeSeries(acc.moment_time, timelineMaxPoints),
        pitch_moment_timeline: subsampleTimeSeries(acc.pitch_moment_time, timelineMaxPoints),
        sog_timeline: subsampleTimeSeries(
          acc.sog_times
            .map((t, i) => ({ t, v: acc.sog_vals[i] }))
            .sort((a, b) => a.t - b.t),
          timelineMaxPoints
        ),
        heel_timeline: subsampleTimeSeries(acc.heel_time, timelineMaxPoints),
        rudder_timeline: subsampleTimeSeries(acc.rudder_time, timelineMaxPoints),
        boom_timeline: subsampleTimeSeries(acc.boom_time, timelineMaxPoints),
        heading_timeline: subsampleTimeSeries(acc.heading_time, timelineMaxPoints),
        com_y_timeline: subsampleTimeSeries(acc.com_y_time, timelineMaxPoints),
        vmg_timeline: subsampleTimeSeries(vmg.timeline, timelineMaxPoints),
        vmg_local_timeline: subsampleTimeSeries(vmg.localTimeline, timelineMaxPoints),
        vmg_mode_timeline: subsampleTimeSeries(vmg.modeTimeline, 240),
        maneuver_analysis: maneuverAnalysis,
        tack_analysis: {
          count: maneuverTacks.length,
          tacks: maneuverTacks,
        },
      };
      if (includeDensityImages) {
        segResult.keypoint_heatmap = generateDensityGrid(acc.all_keypoints_xy);
        segResult.com_heatmap = generateDensityGrid(acc.all_com_xy, {
          grid_size_x: 5.0, grid_size_y: 3.0,
          grid_center_x: -2.0, grid_center_y: 0.0,
        });
      }
      if (includeLegacyVisuals) {
        segResult.hiking_histogram = buildHikingHistogram(acc.trunk_angles);
        segResult.side_load_cumulative = buildCumulativeSideLoad(acc.metrics_in_range, athleteWeight);
        segResult.trunk_vs_sog = buildTrunkVsSogScatter(
          acc.trunk_angle_time, acc.sog_times, acc.sog_vals
        );
        segResult.side_switches = buildSideSwitchAnalysis(
          acc.metrics_in_range, acc.trunk_angle_time, acc.moment_time
        );
      }
      athleteResults.push(segResult);
      if ((athleteIdx + 1) % 2 === 0 || athleteIdx === athleteEntries.length - 1) {
        reportSegmentProgress(
          `Summarizing segment: ${seg.name}...`,
          0.84 + 0.14 * ((athleteIdx + 1) / Math.max(1, athleteEntries.length))
        );
        await yieldReportWork();
      }
    }

    applySegmentVmgComparison(athleteResults);
    segResults.push(...athleteResults);
    reportSegmentProgress(`Segment ${seg.name} done`, 1);
    await yieldReportWork();
  }

  // 5. Build athlete summary
  report('Building athlete summary...', 0.94);
  await yieldReportWork();
  const athleteSummary = athletes.map(a => ({
    id: a.id,
    athlete_id: a.id,
    name: a.name,
    color: a.color || null,
    weight: a.weight,
    segment_ids: segResults.filter(s => s.athlete_id === a.id).map(s => s.split_id),
  }));

  // 6. Compute golds
  report('Computing best-of metrics...', 0.97);
  await yieldReportWork();
  const golds = computeGolds(segResults);

  report('Report data ready', 1);

  return {
    project_id: projectId,
    athletes: athleteSummary,
    segments: segResults,
    golds,
    full_track_points: fullTrackPoints,
    session_utc_start_s: Number.isFinite(gpsSessionStart) ? gpsSessionStart : null,
    session_utc_end_s: Number.isFinite(gpsSessionEnd) ? gpsSessionEnd : null,
  };
}

function findNearestSog(sogTimes, sogVals, t) {
  if (!Array.isArray(sogTimes) || !Array.isArray(sogVals) || sogTimes.length === 0 || sogVals.length === 0) return null;
  let lo = 0;
  let hi = Math.min(sogTimes.length, sogVals.length);
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(sogTimes[mid]) <= t) lo = mid + 1;
    else hi = mid;
  }

  let bestIdx = Math.max(0, Math.min(lo, sogVals.length - 1));
  let bestDist = Math.abs(Number(sogTimes[bestIdx]) - t);
  if (bestIdx > 0) {
    const prevDist = Math.abs(Number(sogTimes[bestIdx - 1]) - t);
    if (prevDist <= bestDist) {
      bestIdx -= 1;
      bestDist = prevDist;
    }
  }
  if (bestIdx + 1 < Math.min(sogTimes.length, sogVals.length)) {
    const nextDist = Math.abs(Number(sogTimes[bestIdx + 1]) - t);
    if (nextDist < bestDist) {
      bestIdx += 1;
      bestDist = nextDist;
    }
  }

  return bestDist <= 2.0 ? sogVals[bestIdx] : null;
}
