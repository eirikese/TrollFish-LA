import { db, getAthletes, getCvConfig, getFileMeta, getManeuverAnalysis, getTrackPoints, listFiles, listMatches, listTracks, saveManeuverAnalysis } from './db.js';
import { loadMetrics, loadSkeletonFrames } from './pose-engine.js';
import { computeCenterOfMass } from './skeleton-metrics.js';

export const MANEUVER_TYPE_TACK = 'tack';
export const MANEUVER_TYPE_JIBE = 'jibe';
export const MANEUVER_MODE_UPWIND = 'upwind';
export const MANEUVER_MODE_DOWNWIND = 'downwind';
export const MANEUVER_MODE_REACH = 'reach';
export const DEFAULT_MANEUVER_DETECTION_SETTINGS = Object.freeze({
  minHeadingDeltaDeg: 30,
  minStableSideSec: 1.5,
  statsWindowSec: 3,
});

const MANEUVER_UPWIND_MAX_TWA_DEG = 100;
const MANEUVER_DOWNWIND_MIN_TWA_DEG = 130;
const MANEUVER_SIGN_DEADBAND_DEG = 3;
const MANEUVER_MAX_GAP_SEC = 8;
const MANEUVER_MIN_SOG_KTS = 0.5;
const MANEUVER_BOUNDARY_LOOKBACK_SEC = 12;
const MANEUVER_RECOVERY_LOOKAHEAD_SEC = 20;
const MANEUVER_SHORT_NEUTRAL_RUN_MAX_SEC = 1.15;
const MANEUVER_SHORT_REVERSAL_RUN_MAX_SEC = 1.0;
const MANEUVER_MIN_STABLE_AXIS_DEG = 6;
const MANEUVER_SAME_SIGN_MODE_WOBBLE_MAX_SEC = 1.25;
const MANEUVER_POSE_WINDOW_PAD_SEC = 3.0;
const MANEUVER_TRANSIENT_REVERSAL_WINDOW_SEC = 22;
const MANEUVER_TRANSIENT_REVERSAL_SCORE_EPSILON = 1.5;
const MANEUVER_MIN_JIBE_SUPPORT_SEC = 2.8;
const MANEUVER_MIN_DURATION_SEC = 1.2;
const MANEUVER_MIN_DETECTION_STEP_SEC = 0.12;
export const MANEUVER_ANALYSIS_SCHEMA_VERSION = 3;
const EARTH_RADIUS_M = 6371000;

function finiteOrNull(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
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
  const diff = Math.abs(angleDifferenceDeg(angleDeg, 0));
  return diff > 180 ? 360 - diff : diff;
}

function median(values) {
  const nums = (Array.isArray(values) ? values : [])
    .map(v => Number(v))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!nums.length) return null;
  const mid = Math.floor(nums.length / 2);
  return nums.length % 2 === 1 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
}

function circularMeanDeg(values) {
  const nums = (Array.isArray(values) ? values : []).map(v => Number(v)).filter(Number.isFinite);
  if (!nums.length) return null;
  let x = 0;
  let y = 0;
  for (const value of nums) {
    const rad = wrapDegrees(value) * Math.PI / 180;
    x += Math.cos(rad);
    y += Math.sin(rad);
  }
  if (Math.abs(x) < 1e-9 && Math.abs(y) < 1e-9) return null;
  return wrapDegrees(Math.atan2(y, x) * 180 / Math.PI);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeManeuverId(trackFileId, type, anchorTs) {
  const baseTrackId = String(trackFileId || 'track');
  const baseType = String(type || 'move');
  const anchorKey = Number.isFinite(Number(anchorTs)) ? Math.round(Number(anchorTs) * 10) : Date.now();
  return `${baseTrackId}:${baseType}:${anchorKey}`;
}

function normalizeSideLabel(sign) {
  return Number(sign) < 0 ? 'port' : 'starboard';
}

function classifyMode(absTwaDeg) {
  const twa = Math.abs(Number(absTwaDeg));
  if (!Number.isFinite(twa)) return MANEUVER_MODE_REACH;
  if (twa <= MANEUVER_UPWIND_MAX_TWA_DEG) return MANEUVER_MODE_UPWIND;
  if (twa >= MANEUVER_DOWNWIND_MIN_TWA_DEG) return MANEUVER_MODE_DOWNWIND;
  return MANEUVER_MODE_REACH;
}

function normalizeHeadingDeg(value) {
  const num = Number(value);
  return Number.isFinite(num) ? wrapDegrees(num) : null;
}

export function normalizeManeuverDetectionSettings(settings = {}) {
  return {
    minHeadingDeltaDeg: clamp(Math.round(Number(settings?.minHeadingDeltaDeg) || DEFAULT_MANEUVER_DETECTION_SETTINGS.minHeadingDeltaDeg), 10, 120),
    minStableSideSec: clamp(Number(settings?.minStableSideSec) || DEFAULT_MANEUVER_DETECTION_SETTINGS.minStableSideSec, 0.5, 8),
    statsWindowSec: clamp(Number(settings?.statsWindowSec) || DEFAULT_MANEUVER_DETECTION_SETTINGS.statsWindowSec, 1, 10),
  };
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

  const directionDeg = finiteOrNull(best?.directionDeg);
  const speedKts = finiteOrNull(best?.speedKts);
  if (!Number.isFinite(directionDeg) || !Number.isFinite(speedKts)) return null;
  return {
    ts: Number(best.ts),
    directionDeg,
    speedKts,
    source: 'local',
  };
}

function resolveWindSample(trackFileId, absTs, windContext = {}) {
  const byCsvId = windContext?.byCsvId || {};
  const localSeries = byCsvId?.[String(trackFileId)]?.localSeries || byCsvId?.[String(trackFileId)] || null;
  const localWind = resolveLocalWindPoint(localSeries, absTs);
  if (localWind) {
    return {
      directionDeg: localWind.directionDeg,
      speedKts: localWind.speedKts,
      source: 'local',
    };
  }
  const session = windContext?.session || null;
  if (Number.isFinite(Number(session?.directionDeg)) && Number.isFinite(Number(session?.speedKts))) {
    return {
      directionDeg: Number(session.directionDeg),
      speedKts: Number(session.speedKts),
      source: 'session',
    };
  }
  return null;
}

function buildTrackPointSamples(points, { trackFileId = null, windContext = {}, headingFallbackToMotion = true } = {}) {
  const sourcePoints = Array.isArray(points) ? points : [];
  const rows = [];
  for (const point of sourcePoints) {
    const absTs = finiteOrNull(point?.ts);
    const motionDirDeg = normalizeHeadingDeg(point?.cog ?? point?.hdg);
    const headingDeg = normalizeHeadingDeg(point?.hdg ?? (headingFallbackToMotion ? point?.cog : null));
    const sogKts = finiteOrNull(point?.sog);
    if (!Number.isFinite(absTs) || !Number.isFinite(motionDirDeg)) continue;
    const wind = resolveWindSample(trackFileId, absTs, windContext);
    if (!wind) continue;
    // Prefer heading for TWA side-crossing; COG can be unstable at low speed.
    const twaReferenceDeg = Number.isFinite(headingDeg) ? headingDeg : motionDirDeg;
    const signedTwaDeg = angleDifferenceDeg(twaReferenceDeg, wind.directionDeg);
    const absTwaDeg = mirrorAngleToHalfCircleDeg(signedTwaDeg);
    const mode = classifyMode(absTwaDeg);
    const crossAxisDeg = mode === MANEUVER_MODE_DOWNWIND
      ? Math.sign(signedTwaDeg || 1) * Math.max(0, 180 - absTwaDeg)
      : signedTwaDeg;
    const absCrossAxisDeg = Math.abs(crossAxisDeg);
    const vmgAbsKts = Number.isFinite(sogKts)
      ? Math.abs(sogKts * Math.cos((absTwaDeg * Math.PI) / 180))
      : null;
    rows.push({
      absTs,
      relT: absTs,
      lat: finiteOrNull(point?.lat),
      lon: finiteOrNull(point?.lon),
      sogKts,
      headingDeg,
      motionDirDeg,
      signedTwaDeg,
      absTwaDeg,
      mode,
      crossAxisDeg,
      absCrossAxisDeg,
      vmgAbsKts,
      heelDeg: finiteOrNull(point?.heel),
      pitchDeg: finiteOrNull(point?.trim),
      rudderDeg: finiteOrNull(point?.rudder_angle),
      boomDeg: finiteOrNull(point?.boom_angle),
      trunkDeg: null,
      comY: null,
      windDirectionDeg: wind.directionDeg,
      windSpeedKts: wind.speedKts,
      windSource: wind.source,
    });
  }
  rows.sort((a, b) => a.absTs - b.absTs);
  return rows;
}

function decimateDetectionSamples(samples, minStepSec = MANEUVER_MIN_DETECTION_STEP_SEC) {
  const rows = Array.isArray(samples) ? samples : [];
  if (rows.length <= 2) return rows.slice();
  const step = Math.max(0.02, Number(minStepSec) || MANEUVER_MIN_DETECTION_STEP_SEC);
  const out = [rows[0]];
  let lastTs = Number(rows[0]?.absTs);
  for (let i = 1; i < rows.length - 1; i++) {
    const sample = rows[i];
    const ts = Number(sample?.absTs);
    if (!Number.isFinite(ts)) continue;
    if ((ts - lastTs) >= step) {
      out.push(sample);
      lastTs = ts;
    }
  }
  out.push(rows[rows.length - 1]);
  return out;
}

function getWindowSamples(samples, startTs, endTs) {
  const start = Number(startTs);
  const end = Number(endTs);
  if (!Array.isArray(samples) || !(end > start)) return [];
  return samples.filter(sample => Number(sample?.absTs) >= start && Number(sample?.absTs) <= end);
}

function getWindowSeriesValues(samples, startTs, endTs, field) {
  return getWindowSamples(samples, startTs, endTs)
    .map(sample => Number(sample?.[field]))
    .filter(Number.isFinite);
}

function computeArithmeticMean(values) {
  const nums = (Array.isArray(values) ? values : []).map(v => Number(v)).filter(Number.isFinite);
  if (!nums.length) return null;
  return nums.reduce((sum, value) => sum + value, 0) / nums.length;
}

function computeCircularMeanFromField(samples, startTs, endTs, field) {
  const values = getWindowSamples(samples, startTs, endTs)
    .map(sample => Number(sample?.[field]))
    .filter(Number.isFinite);
  return circularMeanDeg(values);
}

function findNearestSample(samples, targetTs, maxGapSec = 4) {
  const target = Number(targetTs);
  if (!Array.isArray(samples) || !samples.length || !Number.isFinite(target)) return null;
  let best = null;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (const sample of samples) {
    const ts = Number(sample?.absTs);
    if (!Number.isFinite(ts)) continue;
    const delta = Math.abs(ts - target);
    if (delta < bestDelta) {
      best = sample;
      bestDelta = delta;
    }
  }
  if (!best || bestDelta > maxGapSec) return null;
  return best;
}

function findNearestValue(samples, targetTs, field, maxGapSec = 4) {
  const sample = findNearestSample(samples, targetTs, maxGapSec);
  const value = Number(sample?.[field]);
  return Number.isFinite(value) ? value : null;
}

function interpolateValueAtTs(a, b, field, targetTs) {
  const target = Number(targetTs);
  const tA = Number(a?.absTs);
  const tB = Number(b?.absTs);
  const vA = Number(a?.[field]);
  const vB = Number(b?.[field]);
  if (!Number.isFinite(target) || !Number.isFinite(tA) || !Number.isFinite(tB)) return null;
  if (!Number.isFinite(vA) || !Number.isFinite(vB)) return null;
  if (Math.abs(tB - tA) < 1e-9) return vA;
  const frac = clamp((target - tA) / (tB - tA), 0, 1);
  return vA + frac * (vB - vA);
}

function zeroCrossingBetween(a, b, field = 'crossAxisDeg') {
  const tA = Number(a?.absTs);
  const tB = Number(b?.absTs);
  const vA = Number(a?.[field]);
  const vB = Number(b?.[field]);
  if (!Number.isFinite(tA) || !Number.isFinite(tB) || !Number.isFinite(vA) || !Number.isFinite(vB)) return null;
  if (!(tB > tA) || (tB - tA) > MANEUVER_MAX_GAP_SEC) return null;
  if (vA === 0) return tA;
  if (vB === 0) return tB;
  if ((vA < 0 && vB < 0) || (vA > 0 && vB > 0)) return null;
  const denom = vB - vA;
  if (Math.abs(denom) < 1e-9) return null;
  const frac = clamp((0 - vA) / denom, 0, 1);
  return tA + frac * (tB - tA);
}

function buildDetectionRuns(samples, deadband = MANEUVER_SIGN_DEADBAND_DEG) {
  const runs = [];
  const source = Array.isArray(samples) ? samples : [];
  for (const sample of source) {
    const absTs = Number(sample?.absTs);
    const mode = String(sample?.mode || MANEUVER_MODE_REACH);
    const crossAxisDeg = Number(sample?.crossAxisDeg);
    if (!Number.isFinite(absTs) || !Number.isFinite(crossAxisDeg)) continue;
    const sign = crossAxisDeg > deadband ? 1 : (crossAxisDeg < -deadband ? -1 : 0);
    const last = runs[runs.length - 1];
    if (last && last.sign === sign && last.mode === mode) {
      last.endTs = absTs;
      last.samples.push(sample);
    } else {
      runs.push({
        sign,
        mode,
        startTs: absTs,
        endTs: absTs,
        samples: [sample],
      });
    }
  }
  return runs;
}

function cloneDetectionRun(run) {
  return {
    sign: Number(run?.sign) || 0,
    mode: String(run?.mode || MANEUVER_MODE_REACH),
    startTs: finiteOrNull(run?.startTs),
    endTs: finiteOrNull(run?.endTs),
    samples: Array.isArray(run?.samples) ? run.samples.slice() : [],
  };
}

function mergeDetectionRunTriplet(runs, centerIdx) {
  const prev = runs[centerIdx - 1];
  const cur = runs[centerIdx];
  const next = runs[centerIdx + 1];
  if (!prev || !cur || !next) return runs;
  const merged = cloneDetectionRun(prev);
  merged.endTs = finiteOrNull(next.endTs) ?? merged.endTs;
  merged.samples = [...(prev.samples || []), ...(cur.samples || []), ...(next.samples || [])]
    .filter(Boolean)
    .sort((a, b) => Number(a?.absTs) - Number(b?.absTs));
  const out = runs.slice();
  out.splice(centerIdx - 1, 3, merged);
  return out;
}

function simplifyDetectionRuns(runs, settings) {
  let out = (Array.isArray(runs) ? runs : []).map(cloneDetectionRun);
  if (out.length < 3) return out;
  const neutralMaxSec = Math.max(MANEUVER_SHORT_NEUTRAL_RUN_MAX_SEC, Number(settings?.minStableSideSec || 0) * 0.75);
  const reversalMaxSec = Math.max(MANEUVER_SHORT_REVERSAL_RUN_MAX_SEC, Number(settings?.minStableSideSec || 0) * 0.7);

  let changed = true;
  while (changed && out.length >= 3) {
    changed = false;
    for (let idx = 1; idx < out.length - 1; idx++) {
      const prev = out[idx - 1];
      const cur = out[idx];
      const next = out[idx + 1];
      const curDur = Math.max(0, Number(cur?.endTs) - Number(cur?.startTs));
      const sameOuterMode = String(prev?.mode || '') === String(next?.mode || '');
      const sameOuterSign = Number(prev?.sign) === Number(next?.sign);
      if (!sameOuterMode || !sameOuterSign) continue;

      if (Number(cur?.sign) === 0 && curDur <= neutralMaxSec) {
        out = mergeDetectionRunTriplet(out, idx);
        changed = true;
        break;
      }
      if (
        Math.abs(Number(cur?.sign)) === 1
        && curDur <= reversalMaxSec
        && String(cur?.mode || '') === String(prev?.mode || '')
      ) {
        out = mergeDetectionRunTriplet(out, idx);
        changed = true;
        break;
      }
    }
  }
  return out;
}

function findStableSampleNearAnchor(runSamples, direction = 'before', stableMedianAbs) {
  const samples = Array.isArray(runSamples) ? runSamples : [];
  if (!samples.length) return null;
  const threshold = Math.max(6, Number(stableMedianAbs) * 0.82 || 0);
  if (direction === 'before') {
    for (let i = samples.length - 1; i >= 0; i--) {
      const value = Math.abs(Number(samples[i]?.crossAxisDeg));
      if (Number.isFinite(value) && value >= threshold) return samples[i];
    }
    return samples[samples.length - 1];
  }
  for (const sample of samples) {
    const value = Math.abs(Number(sample?.crossAxisDeg));
    if (Number.isFinite(value) && value >= threshold) return sample;
  }
  return samples[0];
}

function extendSameSignSupportEndTs(runs, startIdx, targetMode) {
  const source = Array.isArray(runs) ? runs : [];
  const base = source[startIdx];
  if (!base) return null;
  const targetSign = Number(base?.sign) || 0;
  const mode = String(targetMode || base?.mode || '');
  let endTs = finiteOrNull(base?.endTs);
  let wobbleSec = 0;
  for (let idx = startIdx + 1; idx < source.length; idx++) {
    const run = source[idx];
    if (!run || Number(run?.sign) !== targetSign) break;
    const runStart = finiteOrNull(run?.startTs);
    const runEnd = finiteOrNull(run?.endTs);
    const runDur = Math.max(0, Number(runEnd) - Number(runStart));
    if (String(run?.mode || '') !== mode) {
      wobbleSec += runDur;
      if (wobbleSec > MANEUVER_SAME_SIGN_MODE_WOBBLE_MAX_SEC) break;
    } else {
      wobbleSec = 0;
    }
    if (Number.isFinite(runEnd)) endTs = Math.max(Number(endTs) || runEnd, runEnd);
  }
  return endTs;
}

function runDurationSec(run) {
  return Math.max(0, Number(run?.endTs) - Number(run?.startTs));
}

function getTransitionSupportSec(runs, startIdx, targetMode) {
  const startTs = finiteOrNull(runs?.[startIdx]?.startTs);
  const endTs = extendSameSignSupportEndTs(runs, startIdx, targetMode) ?? runs?.[startIdx]?.endTs;
  if (!Number.isFinite(startTs) || !Number.isFinite(Number(endTs))) return 0;
  return Math.max(0, Number(endTs) - startTs);
}

function findRunWindowCrossingTs(samples, startTs, endTs, fromSign, toSign, { prefer = 'first' } = {}) {
  const rows = getWindowSamples(samples, startTs, endTs)
    .filter(sample => Number.isFinite(Number(sample?.crossAxisDeg)));
  if (rows.length < 2) return null;

  const candidates = [];
  for (let idx = 1; idx < rows.length; idx++) {
    const prev = rows[idx - 1];
    const next = rows[idx];
    const crossingTs = zeroCrossingBetween(prev, next, 'crossAxisDeg');
    if (!Number.isFinite(crossingTs)) continue;

    const prevSign = Number(prev?.crossAxisDeg) === 0 ? 0 : Math.sign(Number(prev?.crossAxisDeg));
    const nextSign = Number(next?.crossAxisDeg) === 0 ? 0 : Math.sign(Number(next?.crossAxisDeg));
    const matchesFrom = !Number.isFinite(Number(fromSign)) || Number(fromSign) === 0 || prevSign === 0 || prevSign === Number(fromSign);
    const matchesTo = !Number.isFinite(Number(toSign)) || Number(toSign) === 0 || nextSign === 0 || nextSign === Number(toSign);
    if (matchesFrom && matchesTo) candidates.push(crossingTs);
  }

  if (!candidates.length) return null;
  return prefer === 'last'
    ? candidates[candidates.length - 1]
    : candidates[0];
}

function findDetectionTransition(runs, startIdx, settings) {
  const source = Array.isArray(runs) ? runs : [];
  const prev = source[startIdx];
  if (!prev || Math.abs(Number(prev?.sign)) !== 1) return null;

  const mode = String(prev?.mode || '');
  if (mode !== MANEUVER_MODE_UPWIND && mode !== MANEUVER_MODE_DOWNWIND) return null;

  const minStableSec = Math.max(0.5, Number(settings?.minStableSideSec) || DEFAULT_MANEUVER_DETECTION_SETTINGS.minStableSideSec);
  const isDownwind = mode === MANEUVER_MODE_DOWNWIND;
  const maxBridgeSec = isDownwind
    ? Math.max(24, minStableSec * 16)
    : Math.max(6, minStableSec * 4.5);
  const stopOnReturnSec = isDownwind
    ? Math.max(4.5, minStableSec * 2.4)
    : Math.max(2.4, minStableSec * 1.7);
  const candidateMinSupportSec = isDownwind
    ? Math.max(MANEUVER_MIN_JIBE_SUPPORT_SEC, minStableSec * 1.6)
    : minStableSec;

  let best = null;
  let bestSupportSec = -Infinity;
  let sawOppositeCandidate = false;

  for (let idx = startIdx + 1; idx < source.length; idx++) {
    const run = source[idx];
    if (!run) continue;

    const spanFromPrevEndSec = Math.max(0, Number(run?.endTs) - Number(prev?.endTs));
    if (spanFromPrevEndSec > maxBridgeSec) break;

    const runSign = Number(run?.sign) || 0;
    const runMode = String(run?.mode || '');
    const runDurSec = runDurationSec(run);

    if (
      idx > startIdx + 1
      && runSign === Number(prev?.sign)
      && runMode === mode
      && runDurSec >= stopOnReturnSec
      && !sawOppositeCandidate
    ) {
      break;
    }

    if (
      runSign === -Number(prev?.sign)
      && runMode === mode
      && runDurSec >= minStableSec
    ) {
      const supportSec = getTransitionSupportSec(source, idx, mode);
      if (supportSec < candidateMinSupportSec) {
        sawOppositeCandidate = true;
        continue;
      }
      const currentBest = best ? source[best.nextIdx] : null;
      const bestStartTs = Number(currentBest?.startTs);
      const runStartTs = Number(run?.startTs);
      const preferCurrent = !best
        || supportSec > (bestSupportSec + 0.35)
        || (isDownwind && supportSec >= (bestSupportSec - 0.5) && runStartTs > (bestStartTs + 0.75))
        || (!isDownwind && supportSec >= (bestSupportSec - 0.2) && runStartTs < (bestStartTs - 0.35));

      if (preferCurrent) {
        best = {
          nextIdx: idx,
          bridgeStartTs: Number(prev?.endTs),
          bridgeEndTs: runStartTs,
        };
        bestSupportSec = supportSec;
      }
      sawOppositeCandidate = true;
      continue;
    }

    if (
      idx > startIdx + 1
      && runSign === Number(prev?.sign)
      && runMode === mode
      && runDurSec >= stopOnReturnSec
      && sawOppositeCandidate
    ) {
      break;
    }
  }

  return best;
}

function computeWindowMetricSummary(samples, startTs, endTs, maneuverContext = {}) {
  const rows = getWindowSamples(samples, startTs, endTs);
  const avgSogKts = computeArithmeticMean(rows.map(sample => sample.sogKts));
  const avgVmgKts = computeArithmeticMean(rows.map(sample => sample.vmgAbsKts));
  const avgSignedTwaDeg = computeArithmeticMean(rows.map(sample => sample.signedTwaDeg));
  const avgAbsTwaDeg = computeArithmeticMean(rows.map(sample => sample.absTwaDeg));
  const avgHeelDeg = computeArithmeticMean(rows.map(sample => sample.heelDeg));
  const avgRudderDeg = computeArithmeticMean(rows.map(sample => sample.rudderDeg));
  const avgBoomDeg = computeArithmeticMean(rows.map(sample => sample.boomDeg));
  const avgTrunkDeg = computeArithmeticMean(rows.map(sample => sample.trunkDeg));
  const headingDeg = circularMeanDeg(rows.map(sample => sample.headingDeg));
  const entrySpeedKts = findNearestValue(samples, startTs, 'sogKts');
  const exitSpeedKts = findNearestValue(samples, endTs, 'sogKts');
  const minSpeedKts = rows.length
    ? Math.min(...rows.map(sample => Number(sample?.sogKts)).filter(Number.isFinite))
    : null;

  let speedRecoveryTimeS = null;
  if (Number.isFinite(entrySpeedKts)) {
    const recoveryRows = getWindowSamples(samples, endTs, endTs + MANEUVER_RECOVERY_LOOKAHEAD_SEC);
    const threshold = Math.max(0, entrySpeedKts * 0.98);
    const recovered = recoveryRows.find(sample => Number(sample?.sogKts) >= threshold);
    if (recovered) speedRecoveryTimeS = Math.max(0, Number(recovered.absTs) - Number(endTs));
  }

  return {
    windowStartTs: Number.isFinite(Number(startTs)) ? Number(startTs) : null,
    windowEndTs: Number.isFinite(Number(endTs)) ? Number(endTs) : null,
    durationS: Number.isFinite(Number(endTs - startTs)) ? Number(endTs - startTs) : null,
    sampleCount: rows.length,
    avgSogKts,
    avgVmgKts,
    avgSignedTwaDeg,
    avgAbsTwaDeg,
    headingDeg,
    avgHeelDeg,
    avgRudderDeg,
    avgBoomDeg,
    avgTrunkDeg,
    headingDeltaDeg: finiteOrNull(maneuverContext?.headingDeltaDeg),
    entrySpeedKts,
    minSpeedKts,
    exitSpeedKts,
    speedRecoveryTimeS,
  };
}

function toRelativeSeries(samples, startTs, endTs, field, relBaseTs) {
  const rows = getWindowSamples(samples, startTs, endTs);
  const base = Number(relBaseTs);
  return rows
    .map(sample => ({
      t: Number(sample?.absTs) - base,
      v: Number(sample?.[field]),
    }))
    .filter(sample => Number.isFinite(sample.t) && Number.isFinite(sample.v));
}

function smoothTimelineSeriesMedian(points, windowSize = 5) {
  const rows = Array.isArray(points) ? points : [];
  if (rows.length < 3) return rows.slice();
  let size = Math.max(3, Math.round(Number(windowSize)) || 3);
  if (size % 2 === 0) size += 1;
  const half = Math.floor(size / 2);
  return rows.map((point, idx) => {
    const start = Math.max(0, idx - half);
    const end = Math.min(rows.length, idx + half + 1);
    const vals = rows.slice(start, end)
      .map(p => Number(p?.v))
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    if (!vals.length) return point;
    const median = vals[Math.floor(vals.length / 2)];
    return { t: Number(point?.t), v: median };
  });
}

function latLonToLocalMeters(lat, lon, anchorLat, anchorLon) {
  const latRad = Number(anchorLat) * Math.PI / 180;
  const dLat = (Number(lat) - Number(anchorLat)) * Math.PI / 180;
  const dLon = (Number(lon) - Number(anchorLon)) * Math.PI / 180;
  return {
    x: dLon * Math.cos(latRad) * EARTH_RADIUS_M,
    y: dLat * EARTH_RADIUS_M,
  };
}

function rotatePoint(x, y, degrees) {
  const rad = Number(degrees) * Math.PI / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return {
    x: x * cos - y * sin,
    y: x * sin + y * cos,
  };
}

function computeDeepestPoint(samples, startTs, endTs, anchorTs, windDirectionDeg) {
  const rows = getWindowSamples(samples, startTs, endTs)
    .filter(sample => Number.isFinite(sample?.lat) && Number.isFinite(sample?.lon));
  const anchor = findNearestSample(rows, anchorTs, Math.max(6, endTs - startTs));
  if (!anchor || !Number.isFinite(Number(anchor?.lat)) || !Number.isFinite(Number(anchor?.lon))) {
    return {
      deepestTs: finiteOrNull(anchorTs),
      overlayPoints: [],
      anchorPoint: null,
      deepestPoint: null,
    };
  }
  const anchorLat = Number(anchor.lat);
  const anchorLon = Number(anchor.lon);
  const rotateBy = -(Number(windDirectionDeg) || 0);
  const overlayPoints = rows.map(sample => {
    const local = latLonToLocalMeters(sample.lat, sample.lon, anchorLat, anchorLon);
    const rotated = rotatePoint(local.x, local.y, rotateBy);
    return {
      absTs: Number(sample.absTs),
      x: rotated.x,
      y: rotated.y,
      lat: Number(sample.lat),
      lon: Number(sample.lon),
    };
  });
  let deepestPoint = overlayPoints[0] || null;
  for (const point of overlayPoints) {
    if (!deepestPoint || Math.abs(point.x) > Math.abs(deepestPoint.x)) deepestPoint = point;
  }
  return {
    deepestTs: deepestPoint?.absTs ?? finiteOrNull(anchorTs),
    overlayPoints,
    anchorPoint: {
      absTs: Number(anchor.absTs),
      x: 0,
      y: 0,
      lat: anchorLat,
      lon: anchorLon,
    },
    deepestPoint,
  };
}

function getManeuverCommitmentScore(move) {
  if (!move) return 0;
  const entrySpeed = finiteOrNull(move?.preStats?.entrySpeedKts ?? move?.duringStats?.entrySpeedKts);
  const minSpeed = finiteOrNull(move?.duringStats?.minSpeedKts);
  const speedDrop = Number.isFinite(entrySpeed) && Number.isFinite(minSpeed)
    ? Math.max(0, entrySpeed - minSpeed)
    : 0;
  const headingDelta = finiteOrNull(move?.heading_delta_deg ?? move?.duringStats?.headingDeltaDeg) || 0;
  const durationS = finiteOrNull(move?.duration_s ?? move?.duringStats?.durationS) || 0;
  const duringAbsTwa = finiteOrNull(move?.duringStats?.avgAbsTwaDeg);
  const postAbsTwa = finiteOrNull(move?.postStats?.avgAbsTwaDeg);
  const depthBonus = String(move?.type || '') === MANEUVER_TYPE_JIBE
    ? Math.max(0, 180 - (duringAbsTwa || 180)) * 0.08
    : Math.max(0, MANEUVER_UPWIND_MAX_TWA_DEG - (postAbsTwa || MANEUVER_UPWIND_MAX_TWA_DEG)) * 0.05;
  return (speedDrop * 6.4) + (headingDelta * 0.28) + (durationS * 1.45) + depthBonus;
}

function isTransientOppositeReversal(a, b) {
  if (!a || !b) return false;
  if (String(a?.type || '') !== String(b?.type || '')) return false;
  if (String(a?.side_from || '') !== String(b?.side_to || '')) return false;
  if (String(a?.side_to || '') !== String(b?.side_from || '')) return false;
  const anchorGapSec = Number(b?.anchor_ts) - Number(a?.anchor_ts);
  const maxGapSec = String(a?.type || '') === MANEUVER_TYPE_JIBE
    ? MANEUVER_TRANSIENT_REVERSAL_WINDOW_SEC
    : Math.min(12, MANEUVER_TRANSIENT_REVERSAL_WINDOW_SEC);
  if (!Number.isFinite(anchorGapSec) || anchorGapSec <= 0 || anchorGapSec > maxGapSec) return false;
  return true;
}

function collapseTransientOppositeMoves(moves = []) {
  const ordered = (Array.isArray(moves) ? moves : [])
    .slice()
    .sort((a, b) => (Number(a?.anchor_ts) || 0) - (Number(b?.anchor_ts) || 0));
  if (ordered.length < 2) return ordered;

  const kept = [];
  for (const move of ordered) {
    const previous = kept[kept.length - 1];
    if (!previous || !isTransientOppositeReversal(previous, move)) {
      kept.push(move);
      continue;
    }

    const previousScore = getManeuverCommitmentScore(previous);
    const currentScore = getManeuverCommitmentScore(move);
    const preferLaterJibe = String(move?.type || '') === MANEUVER_TYPE_JIBE;
    const chooseCurrent = preferLaterJibe
      ? currentScore >= (previousScore - 4.5)
      : (currentScore > (previousScore + MANEUVER_TRANSIENT_REVERSAL_SCORE_EPSILON)
        || Math.abs(currentScore - previousScore) <= MANEUVER_TRANSIENT_REVERSAL_SCORE_EPSILON);
    if (chooseCurrent) {
      kept[kept.length - 1] = move;
    }
  }
  return kept;
}

function isTransientSameDirectionDuplicate(a, b) {
  if (!a || !b) return false;
  if (String(a?.type || '') !== String(b?.type || '')) return false;
  if (String(a?.side_from || '') !== String(b?.side_from || '')) return false;
  if (String(a?.side_to || '') !== String(b?.side_to || '')) return false;
  const anchorGapSec = Number(b?.anchor_ts) - Number(a?.anchor_ts);
  const maxGapSec = String(a?.type || '') === MANEUVER_TYPE_JIBE ? 34 : 18;
  if (!Number.isFinite(anchorGapSec) || anchorGapSec <= 0 || anchorGapSec > maxGapSec) return false;
  return true;
}

function collapseTransientSameDirectionMoves(moves = []) {
  const ordered = (Array.isArray(moves) ? moves : [])
    .slice()
    .sort((a, b) => (Number(a?.anchor_ts) || 0) - (Number(b?.anchor_ts) || 0));
  if (ordered.length < 2) return ordered;

  const kept = [];
  for (const move of ordered) {
    const previous = kept[kept.length - 1];
    if (!previous || !isTransientSameDirectionDuplicate(previous, move)) {
      kept.push(move);
      continue;
    }

    const previousScore = getManeuverCommitmentScore(previous);
    const currentScore = getManeuverCommitmentScore(move);
    const previousDurationS = finiteOrNull(previous?.duration_s ?? previous?.duringStats?.durationS) || 0;
    const currentDurationS = finiteOrNull(move?.duration_s ?? move?.duringStats?.durationS) || 0;
    const preferLater = String(move?.type || '') === MANEUVER_TYPE_JIBE;
    const chooseCurrent = preferLater
      ? currentScore >= (previousScore - 2.8) || currentDurationS > (previousDurationS + 1.2)
      : currentScore > (previousScore + 1.0) || currentDurationS > (previousDurationS + 0.8);

    if (chooseCurrent) {
      kept[kept.length - 1] = move;
    }
  }
  return kept;
}

function buildDetectionSignalSamples(samples) {
  const rows = Array.isArray(samples) ? samples : [];
  return rows.map((sample, idx) => {
    const neighborhood = rows.slice(Math.max(0, idx - 2), Math.min(rows.length, idx + 3));
    const smoothAbsTwaDeg = median(neighborhood.map(row => row?.absTwaDeg));
    const smoothCrossAxisDeg = median(neighborhood.map(row => row?.crossAxisDeg));
    const smoothMotionDirDeg = circularMeanDeg(neighborhood.map(row => row?.motionDirDeg));
    const smoothHeadingDeg = circularMeanDeg(neighborhood.map(row => row?.headingDeg));
    const absTwaDeg = Number.isFinite(Number(smoothAbsTwaDeg)) ? Number(smoothAbsTwaDeg) : Number(sample?.absTwaDeg);
    return {
      ...sample,
      absTwaDeg,
      crossAxisDeg: Number.isFinite(Number(smoothCrossAxisDeg)) ? Number(smoothCrossAxisDeg) : Number(sample?.crossAxisDeg),
      motionDirDeg: Number.isFinite(Number(smoothMotionDirDeg)) ? Number(smoothMotionDirDeg) : Number(sample?.motionDirDeg),
      headingDeg: Number.isFinite(Number(smoothHeadingDeg)) ? Number(smoothHeadingDeg) : Number(sample?.headingDeg),
      mode: classifyMode(absTwaDeg),
    };
  });
}

function summarizeWindowState(samples, startTs, endTs, deadband = MANEUVER_SIGN_DEADBAND_DEG) {
  const rows = getWindowSamples(samples, startTs, endTs);
  const pos = rows.filter(sample => Number(sample?.crossAxisDeg) > deadband).length;
  const neg = rows.filter(sample => Number(sample?.crossAxisDeg) < -deadband).length;
  const nonZeroCount = pos + neg;
  const sign = !nonZeroCount ? 0 : (pos >= neg ? 1 : -1);
  const purity = nonZeroCount ? Math.max(pos, neg) / nonZeroCount : 0;
  const maxAbsTwa = rows.reduce((best, sample) => {
    const value = Number(sample?.absTwaDeg);
    return Number.isFinite(value) ? Math.max(best, value) : best;
  }, Number.NEGATIVE_INFINITY);
  const firstTs = Number(rows[0]?.absTs);
  const lastTs = Number(rows[rows.length - 1]?.absTs);
  return {
    rows,
    sampleCount: rows.length,
    sign,
    purity,
    nonZeroCount,
    supportSec: Number.isFinite(firstTs) && Number.isFinite(lastTs) ? Math.max(0, lastTs - firstTs) : 0,
    medianAbsTwaDeg: median(rows.map(sample => sample?.absTwaDeg)),
    avgAbsTwaDeg: computeArithmeticMean(rows.map(sample => sample?.absTwaDeg)),
    medianAbsAxisDeg: median(rows.map(sample => Math.abs(Number(sample?.crossAxisDeg)))),
    courseDeg: circularMeanDeg(rows.map(sample => sample?.motionDirDeg)),
    headingDeg: circularMeanDeg(rows.map(sample => sample?.headingDeg)),
    mode: classifyMode(median(rows.map(sample => sample?.absTwaDeg))),
    maxAbsTwaDeg: Number.isFinite(maxAbsTwa) ? maxAbsTwa : null,
  };
}

function findStableOppositeRun(runs, startIdx, settings) {
  const source = Array.isArray(runs) ? runs : [];
  const prev = source[startIdx];
  if (!prev || Math.abs(Number(prev?.sign)) !== 1) return null;
  const mode = String(prev?.mode || '');
  if (mode !== MANEUVER_MODE_UPWIND && mode !== MANEUVER_MODE_DOWNWIND) return null;

  const isUpwind = mode === MANEUVER_MODE_UPWIND;
  const minStableSec = Math.max(0.5, Number(settings?.minStableSideSec) || DEFAULT_MANEUVER_DETECTION_SETTINGS.minStableSideSec);
  const oppositeMinSec = isUpwind ? Math.max(0.8, minStableSec * 0.65) : Math.max(2.2, minStableSec * 1.5);
  const bridgeMaxSec = isUpwind ? Math.max(9, minStableSec * 5.2) : Math.max(26, minStableSec * 13);
  const returnBreakSec = isUpwind ? Math.max(1.3, minStableSec * 0.85) : Math.max(3.2, minStableSec * 2.1);

  for (let idx = startIdx + 1; idx < source.length; idx++) {
    const run = source[idx];
    if (!run) continue;
    if ((Number(run?.endTs) - Number(prev?.endTs)) > bridgeMaxSec) break;

    const runDurSec = runDurationSec(run);
    const runSign = Number(run?.sign) || 0;
    const runMode = String(run?.mode || '');
    if (runSign === 0 || runMode === MANEUVER_MODE_REACH) continue;

    if (runMode === mode && runSign === -Number(prev?.sign) && runDurSec >= oppositeMinSec) {
      return {
        nextIdx: idx,
        bridgeStartTs: Number(prev?.endTs),
        bridgeEndTs: Number(run?.startTs),
      };
    }

    if (runMode === mode && runSign === Number(prev?.sign) && runDurSec >= returnBreakSec) break;
  }
  return null;
}

export function detectManeuversFromSamples(rawSamples, opts = {}) {
  const settings = normalizeManeuverDetectionSettings(opts?.settings);
  const baseSamples = (Array.isArray(rawSamples) ? rawSamples : [])
    .map(sample => ({
      ...sample,
      absTs: finiteOrNull(sample?.absTs),
      relT: finiteOrNull(sample?.relT),
      sogKts: finiteOrNull(sample?.sogKts),
      headingDeg: normalizeHeadingDeg(sample?.headingDeg),
      motionDirDeg: normalizeHeadingDeg(sample?.motionDirDeg),
      signedTwaDeg: finiteOrNull(sample?.signedTwaDeg),
      absTwaDeg: finiteOrNull(sample?.absTwaDeg),
      crossAxisDeg: finiteOrNull(sample?.crossAxisDeg),
      vmgAbsKts: finiteOrNull(sample?.vmgAbsKts),
      windDirectionDeg: finiteOrNull(sample?.windDirectionDeg),
      windSpeedKts: finiteOrNull(sample?.windSpeedKts),
      heelDeg: finiteOrNull(sample?.heelDeg),
      rudderDeg: finiteOrNull(sample?.rudderDeg),
      boomDeg: finiteOrNull(sample?.boomDeg),
      trunkDeg: finiteOrNull(sample?.trunkDeg),
      comY: finiteOrNull(sample?.comY),
    }))
    .filter(sample => Number.isFinite(sample.absTs) && Number(sample?.sogKts) >= MANEUVER_MIN_SOG_KTS)
    .sort((a, b) => sampleSort(a, b));

  const samples = decimateDetectionSamples(baseSamples);

  if (samples.length < 6) return { count: 0, moves: [] };

  const signalSamples = buildDetectionSignalSamples(samples);
  const moves = [];

  const STABLE_SEC = Math.max(0.5, Number(settings.minStableSideSec) || 1.5);
  const MIN_HEADING_DELTA = Math.max(18, Number(settings.minHeadingDeltaDeg) * 0.5);

  const sides = signalSamples.map(s => {
    const a = Number(s.absTwaDeg);
    const twa = Number(s.signedTwaDeg);
    if (a > 6 && a < 174) {
      return twa > 0 ? 1 : (twa < 0 ? -1 : 0);
    }
    return 0;
  });

  const stablePeriods = [];
  let currentSide = 0;
  let currentStartIdx = 0;
  let inStable = false;

  for (let i = 0; i < signalSamples.length; i++) {
    const side = sides[i];
    if (side === 0) continue;

    if (side !== currentSide) {
      if (inStable && currentSide !== 0) {
        stablePeriods.push({
          side: currentSide,
          startIdx: currentStartIdx,
          endIdx: i - 1
        });
      }
      currentSide = side;
      currentStartIdx = i;
      inStable = false;
    } else if (!inStable && (signalSamples[i].absTs - signalSamples[currentStartIdx].absTs >= STABLE_SEC)) {
      inStable = true;
    }
  }
  if (inStable && currentSide !== 0) {
    stablePeriods.push({ side: currentSide, startIdx: currentStartIdx, endIdx: signalSamples.length - 1 });
  }

  let lastMoveEndTs = -Infinity;

  for (let p = 0; p < stablePeriods.length - 1; p++) {
    const prev = stablePeriods[p];
    let next = null;
    let nextIdx = p + 1;
    while (nextIdx < stablePeriods.length) {
      if (stablePeriods[nextIdx].side === -prev.side) {
        next = stablePeriods[nextIdx];
        break;
      }
      nextIdx++;
    }
    
    if (!next) break;

    const gapSec = signalSamples[next.startIdx].absTs - signalSamples[prev.endIdx].absTs;
    if (gapSec > MANEUVER_MAX_GAP_SEC * 2) continue;

    const prevTs = signalSamples[prev.endIdx].absTs;
    const nextTs = signalSamples[next.startIdx].absTs;
    
    let anchorTs = (prevTs + nextTs) / 2;
    for (let j = prev.endIdx; j <= next.startIdx; j++) {
      if (Math.abs(signalSamples[j].signedTwaDeg) < 5 || Math.abs(signalSamples[j].absTwaDeg) > 175) {
        anchorTs = signalSamples[j].absTs;
        break;
      }
    }

    if (anchorTs < lastMoveEndTs - 0.4) continue;

    const preWindowSamples = getWindowSamples(signalSamples, prevTs - 6, prevTs);
    const postWindowSamples = getWindowSamples(signalSamples, nextTs, nextTs + 6);
    
    if (!preWindowSamples.length || !postWindowSamples.length) continue;

    const preHeading = circularMeanDeg(preWindowSamples.map(x => x.headingDeg));
    const postHeading = circularMeanDeg(postWindowSamples.map(x => x.headingDeg));
    const preCourse = circularMeanDeg(preWindowSamples.map(x => x.motionDirDeg));
    const postCourse = circularMeanDeg(postWindowSamples.map(x => x.motionDirDeg));

    const courseDeltaDeg = (preCourse == null || postCourse == null) ? null : Math.abs(angleDifferenceDeg(postCourse, preCourse));
    const headingDeltaRawDeg = (preHeading == null || postHeading == null) ? null : Math.abs(angleDifferenceDeg(postHeading, preHeading));
    const headingDeltaDeg = Number.isFinite(headingDeltaRawDeg) ? headingDeltaRawDeg : courseDeltaDeg;
    
    if (headingDeltaDeg < MIN_HEADING_DELTA) continue;

    const crossWindow = getWindowSamples(signalSamples, anchorTs - 1.5, anchorTs + 1.5);
    const minAbsTwa = crossWindow.length ? Math.min(...crossWindow.map(x => x.absTwaDeg)) : 90;
    const maxAbsTwa = crossWindow.length ? Math.max(...crossWindow.map(x => x.absTwaDeg)) : 90;
    
    let type;
    if (minAbsTwa < 80) type = MANEUVER_TYPE_TACK;
    else if (maxAbsTwa > 100) type = MANEUVER_TYPE_JIBE;
    else type = (preWindowSamples[0].absTwaDeg < 90) ? MANEUVER_TYPE_TACK : MANEUVER_TYPE_JIBE;

    const windAtAnchor = resolveInterpolatedWind(samples, anchorTs);
    let startTs = prevTs;
    let endTs = nextTs;
    const rawDurationSec = endTs - startTs;
    if (rawDurationSec < MANEUVER_MIN_DURATION_SEC) {
      const expandSec = (MANEUVER_MIN_DURATION_SEC - rawDurationSec) / 2;
      startTs = Math.max(signalSamples[0].absTs, startTs - expandSec);
      endTs = Math.min(signalSamples[signalSamples.length - 1].absTs, endTs + expandSec);
    }
    if (!(endTs > startTs)) continue;
    const deep = computeDeepestPoint(samples, startTs, endTs, anchorTs, windAtAnchor?.directionDeg ?? 0);
    const preStats = computeWindowMetricSummary(samples, Math.max(samples[0].absTs, startTs - settings.statsWindowSec), startTs, { headingDeltaDeg });
    const duringStats = computeWindowMetricSummary(samples, startTs, endTs, { headingDeltaDeg });
    const postStats = computeWindowMetricSummary(samples, endTs, Math.min(samples[samples.length - 1].absTs, endTs + settings.statsWindowSec), { headingDeltaDeg });

    const move = {
      id: opts?.trackFileId ? normalizeManeuverId(opts.trackFileId, type, anchorTs) : normalizeManeuverId('move', type, anchorTs),
      type,
      side_from: normalizeSideLabel(prev.side),
      side_to: normalizeSideLabel(next.side),
      start_ts: startTs,
      anchor_ts: anchorTs,
      deepest_ts: deep.deepestTs,
      end_ts: endTs,
      start_t: interpolateRelativeTime(samples, startTs),
      anchor_t: interpolateRelativeTime(samples, anchorTs),
      deepest_t: interpolateRelativeTime(samples, deep.deepestTs),
      end_t: interpolateRelativeTime(samples, endTs),
      duration_s: endTs - startTs,
      heading_delta_deg: headingDeltaDeg,
      sourceWind: {
        source: windAtAnchor?.source || null,
        directionDeg: finiteOrNull(windAtAnchor?.directionDeg),
        speedKts: finiteOrNull(windAtAnchor?.speedKts),
      },
      anchorSource: 'twa_crossing',
      preStats,
      duringStats,
      postStats,
      windDirectionDeg: finiteOrNull(windAtAnchor?.directionDeg),
      windSpeedKts: finiteOrNull(windAtAnchor?.speedKts),
      windSource: windAtAnchor?.source || null,
      overlayPoints: deep.overlayPoints,
      overlayAnchor: deep.anchorPoint,
      overlayDeepest: deep.deepestPoint,
    };

    if (opts?.includeSeries) {
      const chartStart = startTs - 5;
      const chartEnd = endTs + 5;
      move.com_y = toRelativeSeries(samples, chartStart, chartEnd, 'comY', startTs);
      move.trunk = toRelativeSeries(samples, chartStart, chartEnd, 'trunkDeg', startTs);
      move.rudder = toRelativeSeries(samples, chartStart, chartEnd, 'rudderDeg', startTs);
      move.boom = toRelativeSeries(samples, chartStart, chartEnd, 'boomDeg', startTs);
      move.heel = toRelativeSeries(samples, chartStart, chartEnd, 'heelDeg', startTs);
      move.pitch = toRelativeSeries(samples, chartStart, chartEnd, 'pitchDeg', startTs);
      move.heading = toRelativeSeries(samples, chartStart, chartEnd, 'headingDeg', startTs).map(point => ({
        t: point.t,
        v: point.v - (findNearestValue(samples, startTs, 'headingDeg') ?? point.v),
      }));
      move.sog = toRelativeSeries(samples, chartStart, chartEnd, 'sogKts', startTs);
      move.anchor_offset_s = Math.max(0, anchorTs - startTs);
    }

    moves.push(move);
    lastMoveEndTs = endTs;
  }

  const filteredMoves = collapseTransientSameDirectionMoves(collapseTransientOppositeMoves(moves));
  return { count: filteredMoves.length, moves: filteredMoves };
}

function sampleSort(a, b) {
  return (Number(a?.absTs) - Number(b?.absTs)) || ((Number(a?.relT) || 0) - (Number(b?.relT) || 0));
}

function resolveInterpolatedWind(samples, targetTs) {
  const rows = Array.isArray(samples) ? samples : [];
  if (!rows.length) return null;
  let lo = null;
  let hi = null;
  for (const sample of rows) {
    const ts = Number(sample?.absTs);
    if (!Number.isFinite(ts)) continue;
    if (ts <= targetTs) lo = sample;
    if (ts >= targetTs) {
      hi = sample;
      break;
    }
  }
  const base = hi || lo || rows[0];
  return {
    directionDeg: interpolateValueAtTs(lo || base, hi || base, 'windDirectionDeg', targetTs) ?? finiteOrNull(base?.windDirectionDeg),
    speedKts: interpolateValueAtTs(lo || base, hi || base, 'windSpeedKts', targetTs) ?? finiteOrNull(base?.windSpeedKts),
    source: String(base?.windSource || ''),
  };
}

function interpolateRelativeTime(samples, targetTs) {
  const target = Number(targetTs);
  const rows = Array.isArray(samples) ? samples : [];
  if (!rows.length || !Number.isFinite(target)) return null;
  let lo = null;
  let hi = null;
  for (const sample of rows) {
    const ts = Number(sample?.absTs);
    if (!Number.isFinite(ts)) continue;
    if (ts <= target) lo = sample;
    if (ts >= target) {
      hi = sample;
      break;
    }
  }
  if (!lo && !hi) return null;
  if (!lo) return finiteOrNull(hi?.relT);
  if (!hi) return finiteOrNull(lo?.relT);
  return interpolateValueAtTs(lo, hi, 'relT', target);
}

function uniqueStrings(values) {
  return [...new Set((Array.isArray(values) ? values : []).map(value => String(value || '')).filter(Boolean))];
}

function sampleEvenly(items, maxItems = 18) {
  const rows = Array.isArray(items) ? items.filter(Boolean) : [];
  const target = Math.max(1, Math.floor(Number(maxItems) || 1));
  if (rows.length <= target) return rows.slice();
  if (target === 1) return [rows[Math.floor(rows.length / 2)]];
  const out = [];
  let lastIdx = -1;
  for (let i = 0; i < target; i++) {
    const idx = Math.round((i * (rows.length - 1)) / (target - 1));
    if (idx === lastIdx) continue;
    out.push(rows[idx]);
    lastIdx = idx;
  }
  return out;
}

function buildVideoSecToAbsTsConverter(trackPoints) {
  const pts = (Array.isArray(trackPoints) ? trackPoints : [])
    .filter(point => Number.isFinite(Number(point?.video_s)) && Number.isFinite(Number(point?.ts)))
    .sort((a, b) => Number(a.video_s) - Number(b.video_s));
  if (!pts.length) return () => null;
  return (videoSec) => {
    const target = Number(videoSec);
    if (!Number.isFinite(target)) return null;
    let idx = 0;
    while (idx + 1 < pts.length && Number(pts[idx + 1].video_s) <= target) idx++;
    const lo = pts[idx];
    const hi = pts[Math.min(idx + 1, pts.length - 1)];
    const loVs = Number(lo?.video_s);
    const hiVs = Number(hi?.video_s);
    const loTs = Number(lo?.ts);
    const hiTs = Number(hi?.ts);
    if (!Number.isFinite(loVs) || !Number.isFinite(loTs)) return null;
    if (hi === lo || !Number.isFinite(hiVs) || !Number.isFinite(hiTs) || hiVs === loVs) return loTs;
    const frac = clamp((target - loVs) / (hiVs - loVs), 0, 1);
    return loTs + frac * (hiTs - loTs);
  };
}

function buildAbsTsToVideoSecConverter(trackPoints, fallbackStartTs = null) {
  const pts = (Array.isArray(trackPoints) ? trackPoints : [])
    .filter(point => Number.isFinite(Number(point?.video_s)) && Number.isFinite(Number(point?.ts)))
    .sort((a, b) => Number(a.ts) - Number(b.ts));
  const startTs = finiteOrNull(fallbackStartTs);
  if (!pts.length) {
    return (absTs) => (Number.isFinite(startTs) && Number.isFinite(Number(absTs)))
      ? Math.max(0, Number(absTs) - startTs)
      : null;
  }
  return (absTsRaw) => {
    const absTs = Number(absTsRaw);
    if (!Number.isFinite(absTs)) return null;
    let idx = 0;
    while (idx + 1 < pts.length && Number(pts[idx + 1].ts) <= absTs) idx++;
    const lo = pts[idx];
    const hi = pts[Math.min(idx + 1, pts.length - 1)];
    const loTs = Number(lo?.ts);
    const hiTs = Number(hi?.ts);
    const loVs = Number(lo?.video_s);
    const hiVs = Number(hi?.video_s);
    if (!Number.isFinite(loTs) || !Number.isFinite(loVs)) return null;
    if (hi === lo || !Number.isFinite(hiTs) || !Number.isFinite(hiVs) || hiTs === loTs) return loVs;
    const frac = clamp((absTs - loTs) / (hiTs - loTs), 0, 1);
    return loVs + frac * (hiVs - loVs);
  };
}

function buildMetricsTimelineRows(metricsRows, toAbsTs, fieldMap) {
  const out = [];
  for (const row of (Array.isArray(metricsRows) ? metricsRows : [])) {
    if (!row?.detected) continue;
    const videoSec = Number(row?.ts);
    const absTs = toAbsTs(videoSec);
    if (!Number.isFinite(videoSec) || !Number.isFinite(absTs)) continue;
    const nextRow = { absTs };
    let hasField = false;
    for (const [srcKey, dstKey] of Object.entries(fieldMap || {})) {
      const value = finiteOrNull(row?.[srcKey]);
      nextRow[dstKey] = value;
      if (Number.isFinite(value)) hasField = true;
    }
    if (hasField) out.push(nextRow);
  }
  return out.sort((a, b) => a.absTs - b.absTs);
}

function mergeAnalysisSamples(trackSamples, metricRows) {
  const out = (Array.isArray(trackSamples) ? trackSamples : []).map(sample => ({ ...sample }));
  if (!out.length || !Array.isArray(metricRows) || !metricRows.length) return out;
  for (const metricRow of metricRows) {
    const nearest = findNearestSample(out, metricRow.absTs, 1.5);
    if (!nearest) continue;
    if (Number.isFinite(metricRow?.trunkDeg)) nearest.trunkDeg = metricRow.trunkDeg;
    if (Number.isFinite(metricRow?.rudderDeg)) nearest.rudderDeg = metricRow.rudderDeg;
    if (Number.isFinite(metricRow?.boomDeg)) nearest.boomDeg = metricRow.boomDeg;
    if (Number.isFinite(metricRow?.comY)) nearest.comY = metricRow.comY;
  }
  return out;
}

function pickBestOverlappingVideo(maneuver, videoFilesById, videoTracksByFile) {
  const startTs = Number(maneuver?.start_ts);
  const endTs = Number(maneuver?.end_ts);
  const videoIds = uniqueStrings(maneuver?.video_file_ids);
  let best = null;
  for (const videoId of videoIds) {
    const file = videoFilesById.get(String(videoId));
    const track = videoTracksByFile.get(String(videoId));
    if (!file || !track) continue;
    const trackStart = finiteOrNull(track?.ts_start ?? file?.ts_start);
    const durationSec = finiteOrNull(file?.duration_sec);
    const trackEnd = finiteOrNull(track?.ts_end) ?? (Number.isFinite(trackStart) && Number.isFinite(durationSec) ? trackStart + durationSec : null);
    if (!Number.isFinite(trackStart) || !Number.isFinite(trackEnd)) continue;
    const overlap = Math.max(0, Math.min(endTs, trackEnd) - Math.max(startTs, trackStart));
    if (!best || overlap > best.overlapSec) {
      best = { file, track, overlapSec: overlap };
    }
  }
  return best;
}

export async function detectProjectManeuvers(projectId, { windContext = {}, settings = null, onProgress = null } = {}) {
  if (!projectId) return [];
  const report = typeof onProgress === 'function' ? onProgress : () => {};
  const [tracks, matches, fileMeta, athletes, files, cvConfigRec, analysisRows] = await Promise.all([
    listTracks(projectId, 'csv'),
    listMatches(projectId),
    getFileMeta(projectId),
    getAthletes(projectId),
    listFiles(projectId),
    getCvConfig(projectId),
    db.maneuverAnalyses.where('project_id').equals(projectId).toArray(),
  ]);
  const nextSettings = normalizeManeuverDetectionSettings(
    settings || cvConfigRec?.config?.maneuverDetection || DEFAULT_MANEUVER_DETECTION_SETTINGS
  );
  const athleteById = new Map((athletes || []).map(athlete => [String(athlete.id), athlete]));
  const fileById = new Map((files || []).map(file => [String(file.id), file]));
  const videoIdsByCsvId = new Map();
  for (const match of (matches || [])) {
    const csvId = String(match?.csv_file_id || '');
    const videoId = String(match?.video_file_id || '');
    if (!csvId || !videoId) continue;
    if (!videoIdsByCsvId.has(csvId)) videoIdsByCsvId.set(csvId, []);
    videoIdsByCsvId.get(csvId).push(videoId);
  }
  const readyAnalysisIds = new Set(
    (analysisRows || [])
      .filter(row => Number(row?.schema_version) >= MANEUVER_ANALYSIS_SCHEMA_VERSION)
      .map(row => String(row?.maneuver_id || ''))
  );
  const maneuvers = [];

  for (let idx = 0; idx < tracks.length; idx++) {
    const track = tracks[idx];
    const trackFileId = String(track?.file_id || '');
    if (!trackFileId) continue;
    const athleteId = String(fileMeta?.[trackFileId]?.athlete_id || '');
    if (!athleteId) continue;
    const athlete = athleteById.get(athleteId);
    const file = fileById.get(trackFileId);
    const points = await getTrackPoints(track.id);
    const samples = buildTrackPointSamples(points, { trackFileId, windContext });
    const detected = detectManeuversFromSamples(samples, {
      settings: nextSettings,
      trackFileId,
      includeSeries: false,
    });
    const videoFileIds = uniqueStrings(videoIdsByCsvId.get(trackFileId) || []);
    for (const move of detected.moves) {
      maneuvers.push({
        id: move.id,
        project_id: projectId,
        athlete_id: athleteId,
        athlete_name: athlete?.name || file?.filename || 'Athlete',
        type: move.type,
        track_file_id: trackFileId,
        video_file_ids: videoFileIds,
        start_ts: move.start_ts,
        anchor_ts: move.anchor_ts,
        deepest_ts: move.deepest_ts,
        end_ts: move.end_ts,
        side_from: move.side_from,
        side_to: move.side_to,
        sourceWind: move.sourceWind,
        anchorSource: move.anchorSource,
        heading_delta_deg: move.heading_delta_deg,
        preStats: move.preStats,
        duringStats: move.duringStats,
        postStats: move.postStats,
        deepReady: readyAnalysisIds.has(String(move.id)),
      });
    }
    report(`Detecting maneuvers (${idx + 1}/${Math.max(1, tracks.length)})`, (idx + 1) / Math.max(1, tracks.length));
  }

  maneuvers.sort((a, b) => Number(a.anchor_ts || 0) - Number(b.anchor_ts || 0));
  return maneuvers;
}

export async function buildManeuverAnalysis(projectId, maneuverId, { force = false, windContext = {} } = {}) {
  if (!projectId || !maneuverId) return null;
  if (!force) {
    const cached = await getManeuverAnalysis(projectId, maneuverId);
    const cachedPoseReady = !!cached && (
      (Array.isArray(cached?.pose?.keypointXY) && cached.pose.keypointXY.length > 0)
      || (Array.isArray(cached?.pose?.comXY) && cached.pose.comXY.length > 0)
      || (Array.isArray(cached?.pose?.overlayFrames) && cached.pose.overlayFrames.some(frame => frame?.skeleton))
      || (Array.isArray(cached?.pose?.keyframes) && cached.pose.keyframes.some(frame => frame?.skeleton))
    );
    const cachedOverlayReady = !!cached
      && Array.isArray(cached?.gpsOverlay?.points)
      && cached.gpsOverlay.points.some(point => Number.isFinite(Number(point?.x)) && Number.isFinite(Number(point?.y)));
    const cachedHasVideo = !!String(cached?.mediaRefs?.primaryVideoFileId || '');
    if (
      cached
      && Number(cached?.schema_version) >= MANEUVER_ANALYSIS_SCHEMA_VERSION
      && cachedOverlayReady
      && (!cachedHasVideo || cachedPoseReady)
    ) return cached;
  }

  const maneuver = await db.maneuvers.get(String(maneuverId));
  if (!maneuver || String(maneuver.project_id) !== String(projectId)) return null;

  const [files, tracks, trackPoints, matches, cvConfigRec] = await Promise.all([
    listFiles(projectId),
    listTracks(projectId),
    (async () => {
      const track = await db.tracks.where('file_id').equals(String(maneuver.track_file_id)).first();
      return track ? getTrackPoints(track.id) : [];
    })(),
    listMatches(projectId),
    getCvConfig(projectId),
  ]);

  const settings = normalizeManeuverDetectionSettings(
    cvConfigRec?.config?.maneuverDetection || DEFAULT_MANEUVER_DETECTION_SETTINGS
  );
  const fileById = new Map((files || []).map(file => [String(file.id), file]));
  const trackByFileId = new Map((tracks || []).map(track => [String(track.file_id), track]));
  const matchedVideoIds = uniqueStrings(maneuver.video_file_ids || (matches || [])
    .filter(match => String(match?.csv_file_id || '') === String(maneuver.track_file_id || ''))
    .map(match => match.video_file_id));

  const bestVideo = pickBestOverlappingVideo(
    { ...maneuver, video_file_ids: matchedVideoIds },
    fileById,
    trackByFileId,
  );

  const baseTrackSamples = buildTrackPointSamples(trackPoints, {
    trackFileId: maneuver.track_file_id,
    windContext,
  });
  const maneuverDurationS = Math.max(0, Number(maneuver.end_ts) - Number(maneuver.start_ts));
  const posePaddingS = Math.max(MANEUVER_POSE_WINDOW_PAD_SEC, maneuverDurationS * 0.3);
  const timelinePaddingS = Math.max(10, maneuverDurationS * 1.15);
  const overlayPaddingS = Math.max(18, maneuverDurationS * 1.9);
  const expandedStartTs = Number(maneuver.start_ts) - timelinePaddingS;
  const expandedEndTs = Number(maneuver.end_ts) + timelinePaddingS;
  const gpsSamples = getWindowSamples(baseTrackSamples, expandedStartTs, expandedEndTs);
  let mergedSamples = gpsSamples;
  let metricTimeline = null;
  let mediaRefs = {
    primaryVideoFileId: null,
    videoStartSec: null,
    videoEndSec: null,
    videoAnchorSec: null,
    available: false,
  };
  let pose = {
    available: false,
    keyframes: [],
    frameCount: 0,
    comSideSwitchRelS: null,
  };

  if (bestVideo?.track) {
    const videoTrackPoints = await getTrackPoints(bestVideo.track.id);
    const absTsToVideoSec = buildAbsTsToVideoSecConverter(videoTrackPoints, bestVideo.file?.ts_start);
    const videoStartSec = absTsToVideoSec(Number(maneuver.start_ts) - posePaddingS);
    const videoEndSec = absTsToVideoSec(Number(maneuver.end_ts) + posePaddingS);
    const videoAnchorSec = absTsToVideoSec(maneuver.anchor_ts);
    mediaRefs = {
      primaryVideoFileId: String(bestVideo.file.id),
      videoStartSec: finiteOrNull(videoStartSec),
      videoEndSec: finiteOrNull(videoEndSec),
      videoAnchorSec: finiteOrNull(videoAnchorSec),
      available: Number.isFinite(videoAnchorSec),
    };

    if (Number.isFinite(videoStartSec) && Number.isFinite(videoEndSec) && videoEndSec >= videoStartSec) {
      const [metricRows, skeletonFrames] = await Promise.all([
        loadMetrics(projectId, bestVideo.file.id),
        loadSkeletonFrames(projectId, bestVideo.file.id, Math.max(0, videoStartSec - 5), videoEndSec + 5),
      ]);
      const toAbsTs = buildVideoSecToAbsTsConverter(videoTrackPoints);
      metricTimeline = buildMetricsTimelineRows(metricRows, toAbsTs, {
        trunk_angle: 'trunkDeg',
        rudder_angle: 'rudderDeg',
        boom_angle: 'boomDeg',
        com_y: 'comY',
      });
      mergedSamples = mergeAnalysisSamples(gpsSamples, metricTimeline);
      pose = buildPoseSummary(skeletonFrames, toAbsTs, maneuver, { padSec: posePaddingS });
    }
  }

  const detected = detectManeuversFromSamples(mergedSamples, {
    settings,
    trackFileId: maneuver.track_file_id,
    includeSeries: true,
  });
  const canonicalMove = detected.moves.find(move => String(move.id) === String(maneuver.id))
    || detected.moves.find(move => Math.abs(Number(move.anchor_ts) - Number(maneuver.anchor_ts)) <= 1.0)
    || null;
  const move = canonicalMove || {
    ...maneuver,
    com_y: [],
    trunk: [],
    rudder: [],
    boom: [],
    heel: [],
    pitch: [],
    heading: [],
    sog: [],
    anchor_offset_s: Math.max(0, Number(maneuver.anchor_ts) - Number(maneuver.start_ts)),
    overlayPoints: [],
    overlayAnchor: null,
    overlayDeepest: null,
  };
  const baseStartTs = Number(maneuver.start_ts);
  const chartStart = baseStartTs - 5;
  const chartEnd = Number(maneuver.end_ts) + 5;
  const metricRudderSeries = metricTimeline
    ? toRelativeSeries(metricTimeline, chartStart, chartEnd, 'rudderDeg', baseStartTs)
    : [];
  const metricBoomSeries = metricTimeline
    ? toRelativeSeries(metricTimeline, chartStart, chartEnd, 'boomDeg', baseStartTs)
    : [];
  const vmgSeriesRaw = mergedSamples
    .filter(sample => Number(sample?.absTs) >= expandedStartTs && Number(sample?.absTs) <= expandedEndTs)
    .map(sample => ({
      t: Number(sample.absTs) - Number(maneuver.anchor_ts),
      v: Number(sample.vmgAbsKts),
    }))
    .filter(sample => Number.isFinite(sample.t) && Number.isFinite(sample.v));
  const vmgSeries = vmgSeriesRaw.length > 4
    ? smoothTimelineSeriesMedian(vmgSeriesRaw, 5)
    : vmgSeriesRaw;
  const overlayWindDirectionDeg = finiteOrNull(move.windDirectionDeg ?? maneuver?.sourceWind?.directionDeg);
  const overlayGeometry = computeDeepestPoint(
    mergedSamples,
    Number(maneuver.start_ts) - overlayPaddingS,
    Number(maneuver.end_ts) + overlayPaddingS,
    Number(maneuver.anchor_ts),
    overlayWindDirectionDeg,
  );
  const coreGeometry = computeDeepestPoint(
    mergedSamples,
    Number(maneuver.start_ts),
    Number(maneuver.end_ts),
    Number(maneuver.anchor_ts),
    overlayWindDirectionDeg,
  );

  const analysis = {
    schema_version: MANEUVER_ANALYSIS_SCHEMA_VERSION,
    maneuver_id: String(maneuver.id),
    project_id: projectId,
    generated_at: new Date().toISOString(),
    analysisWindow: {
      start_ts: Number(maneuver.start_ts),
      anchor_ts: Number(maneuver.anchor_ts),
      end_ts: Number(maneuver.end_ts),
      duration_s: Math.max(0, Number(maneuver.end_ts) - Number(maneuver.start_ts)),
    },
    maneuver: {
      ...maneuver,
      type: maneuver.type,
      side_from: maneuver.side_from,
      side_to: maneuver.side_to,
      preStats: move.preStats || maneuver.preStats || null,
      duringStats: move.duringStats || maneuver.duringStats || null,
      postStats: move.postStats || maneuver.postStats || null,
    },
    timelines: {
      anchorOffsetS: finiteOrNull(move.anchor_offset_s),
      sog: move.sog || [],
      heading: move.heading || [],
      rudder: metricRudderSeries.length >= 2 ? metricRudderSeries : (move.rudder || []),
      boom: metricBoomSeries.length >= 2 ? metricBoomSeries : (move.boom || []),
      trunk: move.trunk || [],
      pitch: move.pitch || [],
      com_y: move.com_y || [],
      twa: mergedSamples
        .filter(sample => Number(sample?.absTs) >= expandedStartTs && Number(sample?.absTs) <= expandedEndTs)
        .map(sample => ({
          t: Number(sample.absTs) - Number(maneuver.anchor_ts),
          v: Number(sample.signedTwaDeg),
        }))
        .filter(sample => Number.isFinite(sample.t) && Number.isFinite(sample.v)),
      vmg: vmgSeries,
      heel: mergedSamples
        .filter(sample => Number(sample?.absTs) >= expandedStartTs && Number(sample?.absTs) <= expandedEndTs)
        .map(sample => ({
          t: Number(sample.absTs) - Number(maneuver.anchor_ts),
          v: Number(sample.heelDeg),
        }))
        .filter(sample => Number.isFinite(sample.t) && Number.isFinite(sample.v)),
      pitch: mergedSamples
        .filter(sample => Number(sample?.absTs) >= expandedStartTs && Number(sample?.absTs) <= expandedEndTs)
        .map(sample => ({
          t: Number(sample.absTs) - Number(maneuver.anchor_ts),
          v: Number(sample.pitchDeg),
        }))
        .filter(sample => Number.isFinite(sample.t) && Number.isFinite(sample.v)),
    },
    pose,
    gpsOverlay: {
      points: overlayGeometry.overlayPoints || move.overlayPoints || [],
      anchor: overlayGeometry.anchorPoint || move.overlayAnchor || null,
      deepest: coreGeometry.deepestPoint || move.overlayDeepest || overlayGeometry.deepestPoint || null,
      windDirectionDeg: overlayWindDirectionDeg,
      alignmentAnchor: 'twa-crossing',
    },
    mediaRefs,
  };

  await saveManeuverAnalysis(projectId, maneuver.id, analysis);
  return analysis;
}

function buildPoseSummary(frames, toAbsTs, maneuver, { padSec = MANEUVER_POSE_WINDOW_PAD_SEC } = {}) {
  const sourceFrames = (Array.isArray(frames) ? frames : [])
    .map(frame => ({
      ...frame,
      absTs: toAbsTs(Number(frame?.ts)),
    }))
    .filter(frame => Number.isFinite(frame.absTs) && frame?.skeleton && Object.keys(frame.skeleton).length > 0)
    .sort((a, b) => Number(a.absTs) - Number(b.absTs));
  if (!sourceFrames.length) {
    return {
      available: false,
      keyframes: [],
      keypointXY: [],
      comXY: [],
      frameCount: 0,
      comSideSwitchRelS: null,
    };
  }

  const paddedStartTs = Number(maneuver.start_ts) - Math.max(0, Number(padSec) || 0);
  const paddedEndTs = Number(maneuver.end_ts) + Math.max(0, Number(padSec) || 0);
  const targets = [
    Number(maneuver.start_ts),
    Number(maneuver.anchor_ts),
    Number(maneuver.end_ts),
    Number(maneuver.deepest_ts),
  ].filter(Number.isFinite);
  const labels = ['Start', 'Anchor', 'End', 'Deepest'];
  const keyframes = [];
  for (let i = 0; i < targets.length; i++) {
    const frame = findNearestSample(sourceFrames, targets[i], 2.5);
    if (!frame) continue;
    keyframes.push({
      label: labels[i] || `Frame ${i + 1}`,
      relS: Number(frame.absTs) - Number(maneuver.anchor_ts),
      skeleton: frame.skeleton,
    });
  }
  const overlayFrames = sampleEvenly(
    sourceFrames.filter(frame => Number(frame.absTs) >= paddedStartTs && Number(frame.absTs) <= paddedEndTs),
    24,
  ).map(frame => ({
    relS: Number(frame.absTs) - Number(maneuver.anchor_ts),
    skeleton: frame.skeleton,
  }));
  const keypointXY = [];
  const comXY = [];
  for (const frame of sourceFrames) {
    for (const point of Object.values(frame?.skeleton || {})) {
      const x = finiteOrNull(point?.[0]);
      const y = finiteOrNull(point?.[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      keypointXY.push([x, y]);
    }
    const com = computeCenterOfMass(frame?.skeleton || null);
    if (Array.isArray(com) && Number.isFinite(Number(com[0])) && Number.isFinite(Number(com[1]))) {
      comXY.push([Number(com[0]), Number(com[1])]);
    }
  }

  let comSideSwitchRelS = null;
  let prev = null;
  for (const frame of sourceFrames) {
    const comY = finiteOrNull(frame?.metrics?.com_y);
    if (!Number.isFinite(comY)) continue;
    if (prev && Number.isFinite(prev.comY) && prev.comY !== 0 && comY !== 0 && Math.sign(prev.comY) !== Math.sign(comY)) {
      const frac = clamp((0 - prev.comY) / (comY - prev.comY), 0, 1);
      const absTs = Number(prev.absTs) + frac * (Number(frame.absTs) - Number(prev.absTs));
      comSideSwitchRelS = absTs - Number(maneuver.anchor_ts);
      break;
    }
    prev = { absTs: frame.absTs, comY };
  }

  return {
    available: overlayFrames.length > 0 || keyframes.length > 0,
    keyframes,
    overlayFrames,
    keypointXY,
    comXY,
    frameCount: sourceFrames.length,
    comSideSwitchRelS,
  };
}
