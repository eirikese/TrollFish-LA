const VMG_MODE_UPWIND = 'upwind';
const VMG_MODE_DOWNWIND = 'downwind';
const VMG_MODE_REACH = 'reach';

const VMG_UPWIND_MAX_TWA_DEG = 75;
const VMG_DOWNWIND_MIN_TWA_DEG = 115;
const VMG_MIN_SOG_KTS = 0.65;
const VMG_MAX_DT_SEC = 6.0;
const VMG_MAX_TURN_RATE_DEG_S = 22.0;
const VMG_MAX_ACCEL_KTS_S = 2.6;
const VMG_MIN_MODE_COVERAGE_SEC = 8.0;
const VMG_MIN_MODE_SAMPLES = 4;
const VMG_STABLE_WINDOW_SEC = 8.0;
const VMG_MODE_SMOOTH_HALF_SAMPLES = 2;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function wrapDegrees(value) {
  let out = Number(value) % 360;
  if (out < 0) out += 360;
  return out;
}

function angleDifferenceDeg(a, b) {
  let diff = wrapDegrees(a) - wrapDegrees(b);
  if (diff > 180) diff -= 360;
  else if (diff < -180) diff += 360;
  return diff;
}

function mirrorAngleToHalfCircleDeg(angleDeg) {
  const wrapped = Math.abs(angleDifferenceDeg(angleDeg, 0));
  return wrapped > 180 ? 360 - wrapped : wrapped;
}

function finiteOrNull(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function median(values) {
  const nums = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!nums.length) return null;
  const mid = Math.floor(nums.length / 2);
  return nums.length % 2 === 1 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
}

function classifyModeFromTwa(twaDeg) {
  const twa = Number(twaDeg);
  if (!Number.isFinite(twa)) return VMG_MODE_REACH;
  if (twa <= VMG_UPWIND_MAX_TWA_DEG) return VMG_MODE_UPWIND;
  if (twa >= VMG_DOWNWIND_MIN_TWA_DEG) return VMG_MODE_DOWNWIND;
  return VMG_MODE_REACH;
}

function resolveLocalWindPoint(localSeries, absTs) {
  const points = Array.isArray(localSeries?.points) ? localSeries.points : [];
  const targetTs = Number(absTs);
  if (!points.length || !Number.isFinite(targetTs)) return null;

  let lo = 0;
  let hi = points.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(points[mid]?.ts) < targetTs) lo = mid + 1;
    else hi = mid;
  }

  let best = null;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (const idx of [lo - 1, lo, lo + 1]) {
    const point = points[idx];
    const ts = Number(point?.ts);
    if (!Number.isFinite(ts)) continue;
    const delta = Math.abs(ts - targetTs);
    if (delta < bestDelta) {
      best = point;
      bestDelta = delta;
    }
  }
  if (!best) return null;

  const maxGap = Math.max(
    (Number(localSeries?.stepSeconds) || 0) * 1.5,
    (Number(localSeries?.windowSeconds) || 0) * 0.65,
    90,
  );
  if (bestDelta > maxGap) return null;
  if (!Number.isFinite(Number(best?.directionDeg)) || !Number.isFinite(Number(best?.speedKts))) return null;
  return best;
}

function preprocessSamples(samples) {
  const rows = (Array.isArray(samples) ? samples : [])
    .map(sample => ({
      t: finiteOrNull(sample?.t),
      absTs: finiteOrNull(sample?.absTs),
      sogKts: finiteOrNull(sample?.sogKts),
      motionDirDeg: finiteOrNull(sample?.motionDirDeg),
      localWindKey: sample?.localWindKey != null ? String(sample.localWindKey) : null,
    }))
    .filter(sample => sample.t != null && sample.absTs != null && sample.motionDirDeg != null)
    .sort((a, b) => (a.absTs - b.absTs) || (a.t - b.t));

  const deduped = [];
  for (const sample of rows) {
    const prev = deduped[deduped.length - 1];
    if (
      prev
      && Math.abs(prev.absTs - sample.absTs) <= 0.20
      && Math.abs(prev.t - sample.t) <= 0.20
      && prev.localWindKey === sample.localWindKey
    ) {
      if (prev.sogKts == null && sample.sogKts != null) prev.sogKts = sample.sogKts;
      continue;
    }
    deduped.push(sample);
  }

  for (let i = 0; i < deduped.length; i++) {
    const prev = deduped[Math.max(0, i - 1)];
    const next = deduped[Math.min(deduped.length - 1, i + 1)];
    const dtPrev = i > 0 ? Math.max(1e-3, deduped[i].absTs - prev.absTs) : null;
    const dtNext = i + 1 < deduped.length ? Math.max(1e-3, next.absTs - deduped[i].absTs) : null;
    deduped[i].dtSec = finiteOrNull(dtPrev ?? dtNext) ?? 1.0;

    if (i > 0 && dtPrev != null) {
      deduped[i].turnRateDegS = Math.abs(angleDifferenceDeg(deduped[i].motionDirDeg, prev.motionDirDeg)) / dtPrev;
      if (deduped[i].sogKts != null && prev.sogKts != null) {
        deduped[i].accelKtsS = Math.abs(deduped[i].sogKts - prev.sogKts) / dtPrev;
      } else {
        deduped[i].accelKtsS = null;
      }
    } else {
      deduped[i].turnRateDegS = 0;
      deduped[i].accelKtsS = 0;
    }
  }
  return deduped;
}

function isMotionSampleUsable(sample) {
  if (sample?.sogKts == null || sample.motionDirDeg == null) return false;
  if (sample.sogKts < VMG_MIN_SOG_KTS) return false;
  if (!Number.isFinite(sample.dtSec) || sample.dtSec <= 0 || sample.dtSec > VMG_MAX_DT_SEC) return false;
  if (Number.isFinite(sample.turnRateDegS) && sample.turnRateDegS > VMG_MAX_TURN_RATE_DEG_S) return false;
  if (Number.isFinite(sample.accelKtsS) && sample.accelKtsS > VMG_MAX_ACCEL_KTS_S) return false;
  return true;
}

function computeRawVmg(sample, windDirectionDeg) {
  if (!isMotionSampleUsable(sample) || !Number.isFinite(Number(windDirectionDeg))) {
    return { twaDeg: null, projectionKts: null };
  }
  const twaDeg = mirrorAngleToHalfCircleDeg(angleDifferenceDeg(sample.motionDirDeg, windDirectionDeg));
  const projectionKts = sample.sogKts * Math.cos((twaDeg * Math.PI) / 180);
  return { twaDeg, projectionKts };
}

function smoothModeTwa(rows, key) {
  const out = new Array(rows.length).fill(null);
  for (let i = 0; i < rows.length; i++) {
    const window = [];
    for (let j = Math.max(0, i - VMG_MODE_SMOOTH_HALF_SAMPLES); j <= Math.min(rows.length - 1, i + VMG_MODE_SMOOTH_HALF_SAMPLES); j++) {
      const value = Number(rows[j]?.[key]);
      if (Number.isFinite(value)) window.push(value);
    }
    out[i] = median(window);
  }
  return out;
}

function buildModeTimeline(points, modeKey = 'mode') {
  const out = [];
  let lastMode = null;
  for (const point of points) {
    const mode = String(point?.[modeKey] || VMG_MODE_REACH);
    const t = Number(point?.t);
    if (!Number.isFinite(t)) continue;
    if (mode === lastMode) continue;
    out.push({ t, v: mode });
    lastMode = mode;
  }
  return out;
}

function summarizeMode(points, mode) {
  const rows = points.filter(point => point.mode === mode && Number.isFinite(point.v) && point.v > 0);
  const coverageSec = rows.reduce((sum, point) => sum + Math.max(0, Number(point.weightSec) || 0), 0);
  if (coverageSec < VMG_MIN_MODE_COVERAGE_SEC || rows.length < VMG_MIN_MODE_SAMPLES) {
    return {
      avg: null,
      max: null,
      min: null,
      std: null,
      count: rows.length,
      coverage_s: coverageSec,
      best_stable: null,
      peak: rows.length ? Math.max(...rows.map(point => point.v)) : null,
      eligible: false,
    };
  }

  let sum = 0;
  let weightSum = 0;
  for (const row of rows) {
    const weight = Math.max(0.05, Number(row.weightSec) || 0);
    sum += row.v * weight;
    weightSum += weight;
  }
  const avg = weightSum > 0 ? sum / weightSum : null;

  let variance = 0;
  if (avg != null && weightSum > 0) {
    for (const row of rows) {
      const weight = Math.max(0.05, Number(row.weightSec) || 0);
      variance += ((row.v - avg) ** 2) * weight;
    }
    variance /= weightSum;
  }

  let bestStable = null;
  for (let i = 0; i < rows.length; i++) {
    const centerT = rows[i].t;
    let localSum = 0;
    let localWeight = 0;
    for (let j = 0; j < rows.length; j++) {
      if (Math.abs(rows[j].t - centerT) > (VMG_STABLE_WINDOW_SEC / 2)) continue;
      const weight = Math.max(0.05, Number(rows[j].weightSec) || 0);
      localSum += rows[j].v * weight;
      localWeight += weight;
    }
    if (localWeight <= 0) continue;
    const rollingAvg = localSum / localWeight;
    if (bestStable == null || rollingAvg > bestStable) bestStable = rollingAvg;
  }

  return {
    avg,
    max: bestStable,
    min: Math.min(...rows.map(point => point.v)),
    std: Number.isFinite(variance) ? Math.sqrt(variance) : null,
    count: rows.length,
    coverage_s: coverageSec,
    best_stable: bestStable,
    peak: Math.max(...rows.map(point => point.v)),
    eligible: true,
  };
}

export function computeSegmentVmg(samples, options = {}) {
  const processed = preprocessSamples(samples);
  const sessionWind = options?.sessionWind || null;
  const localWindSeriesByKey = options?.localWindSeriesByKey || {};
  const sessionWindValid = Number.isFinite(Number(sessionWind?.directionDeg)) && Number.isFinite(Number(sessionWind?.speedKts));

  if (!processed.length) {
    return {
      timeline: [],
      localTimeline: [],
      modeTimeline: [],
      summary: {
        dominant_mode: null,
        scored_coverage_s: 0,
        upwind: summarizeMode([], VMG_MODE_UPWIND),
        downwind: summarizeMode([], VMG_MODE_DOWNWIND),
      },
    };
  }

  const sessionRaw = new Array(processed.length).fill(null);
  const localRaw = new Array(processed.length).fill(null);
  for (let i = 0; i < processed.length; i++) {
    sessionRaw[i] = sessionWindValid ? computeRawVmg(processed[i], sessionWind.directionDeg) : { twaDeg: null, projectionKts: null };
    const localKey = processed[i].localWindKey;
    const localSeries = localKey ? localWindSeriesByKey[localKey] : null;
    const localWindPoint = resolveLocalWindPoint(localSeries, processed[i].absTs);
    localRaw[i] = localWindPoint
      ? computeRawVmg(processed[i], localWindPoint.directionDeg)
      : { twaDeg: null, projectionKts: null };
  }

  const smoothedSessionTwa = smoothModeTwa(sessionRaw, 'twaDeg');
  const smoothedLocalTwa = smoothModeTwa(localRaw, 'twaDeg');

  const timeline = [];
  const localTimeline = [];
  for (let i = 0; i < processed.length; i++) {
    const sample = processed[i];
    const sessionMode = classifyModeFromTwa(smoothedSessionTwa[i]);
    const localMode = classifyModeFromTwa(smoothedLocalTwa[i]);
    const sessionProjection = sessionRaw[i]?.projectionKts;
    const localProjection = localRaw[i]?.projectionKts;
    const sessionVmg = sessionMode === VMG_MODE_UPWIND
      ? sessionProjection
      : (sessionMode === VMG_MODE_DOWNWIND && Number.isFinite(sessionProjection) ? -sessionProjection : null);
    const localVmg = localMode === VMG_MODE_UPWIND
      ? localProjection
      : (localMode === VMG_MODE_DOWNWIND && Number.isFinite(localProjection) ? -localProjection : null);
    const weightSec = clamp(Number(sample.dtSec) || 1, 0.25, 3.0);

    timeline.push({
      t: sample.t,
      absTs: sample.absTs,
      v: Number.isFinite(sessionVmg) && sessionVmg > 0 ? sessionVmg : null,
      mode: sessionMode,
      twa: Number.isFinite(smoothedSessionTwa[i]) ? smoothedSessionTwa[i] : null,
      weightSec,
    });
    localTimeline.push({
      t: sample.t,
      absTs: sample.absTs,
      v: Number.isFinite(localVmg) && localVmg > 0 ? localVmg : null,
      mode: sessionMode !== VMG_MODE_REACH ? sessionMode : localMode,
      twa: Number.isFinite(smoothedLocalTwa[i]) ? smoothedLocalTwa[i] : null,
      weightSec,
    });
  }

  const upwind = summarizeMode(timeline, VMG_MODE_UPWIND);
  const downwind = summarizeMode(timeline, VMG_MODE_DOWNWIND);
  const dominantMode = (upwind.coverage_s > downwind.coverage_s && upwind.eligible)
    ? VMG_MODE_UPWIND
    : ((downwind.eligible || downwind.coverage_s > upwind.coverage_s) ? VMG_MODE_DOWNWIND : (upwind.eligible ? VMG_MODE_UPWIND : null));

  return {
    timeline,
    localTimeline,
    modeTimeline: buildModeTimeline(timeline),
    summary: {
      dominant_mode: dominantMode,
      scored_coverage_s: (Number(upwind.coverage_s) || 0) + (Number(downwind.coverage_s) || 0),
      upwind,
      downwind,
    },
  };
}

export const VMG_MODES = Object.freeze({
  UPWIND: VMG_MODE_UPWIND,
  DOWNWIND: VMG_MODE_DOWNWIND,
  REACH: VMG_MODE_REACH,
});
