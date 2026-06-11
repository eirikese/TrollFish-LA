const TWO_PI = Math.PI * 2;
const RAD_TO_DEG = 180 / Math.PI;
const DEG_TO_RAD = Math.PI / 180;

export const KNOT_TO_MS = 0.514444;
export const DEFAULT_DEAD_ZONE_ANGLE_RAD = Math.PI / 6;

const ILCA7_TWA_DEG = new Float64Array([0, 30, 35, 45, 60, 75, 90, 110, 120, 135, 150, 160, 170, 180]);
const ILCA7_TWS_KTS = new Float64Array([4, 6, 8, 10, 12, 14, 16, 20]);
const ILCA7_POLAR_SPEEDS_MS = [
  new Float64Array([0.0, 0.0, 1.8, 2.6, 3.2, 3.6, 3.8, 3.7, 3.6, 3.4, 3.1, 2.9, 2.8, 2.7].map(v => v * KNOT_TO_MS)),
  new Float64Array([0.0, 0.0, 2.3, 3.2, 3.8, 4.3, 4.6, 4.7, 4.8, 4.9, 4.8, 4.6, 4.3, 4.0].map(v => v * KNOT_TO_MS)),
  new Float64Array([0.0, 0.0, 2.6, 3.6, 4.4, 5.0, 5.5, 5.8, 6.0, 6.4, 6.2, 5.9, 5.6, 5.2].map(v => v * KNOT_TO_MS)),
  new Float64Array([0.0, 0.0, 2.8, 3.9, 4.8, 5.5, 6.0, 6.4, 6.8, 7.4, 7.2, 6.9, 6.5, 6.0].map(v => v * KNOT_TO_MS)),
  new Float64Array([0.0, 0.0, 3.0, 4.1, 5.0, 5.8, 6.3, 6.8, 7.2, 8.0, 8.0, 7.7, 7.2, 6.6].map(v => v * KNOT_TO_MS)),
  new Float64Array([0.0, 0.0, 3.1, 4.2, 5.1, 5.9, 6.4, 7.0, 7.5, 8.5, 8.6, 8.3, 7.8, 7.1].map(v => v * KNOT_TO_MS)),
  new Float64Array([0.0, 0.0, 3.2, 4.3, 5.2, 6.0, 6.6, 7.2, 7.8, 9.0, 9.2, 8.8, 8.3, 7.6].map(v => v * KNOT_TO_MS)),
  new Float64Array([0.0, 0.0, 3.3, 4.4, 5.3, 6.1, 6.8, 7.5, 8.2, 9.5, 9.8, 9.3, 8.7, 8.0].map(v => v * KNOT_TO_MS)),
];
const DEFAULT_POLAR_TWA_CENTERS_DEG = Object.freeze([35, 45, 60, 75, 90, 110, 120, 135, 150, 160, 170, 180]);

const _maxPolarSpeedCache = new Map();

function nowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') return performance.now();
  return Date.now();
}

function createYieldController(budgetMs = 12) {
  let lastYieldAt = nowMs();
  return {
    async maybeYield(force = false) {
      const now = nowMs();
      if (!force && now - lastYieldAt < budgetMs) return;
      await new Promise(resolve => {
        if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
          window.requestAnimationFrame(() => resolve());
          return;
        }
        setTimeout(resolve, 0);
      });
      lastYieldAt = nowMs();
    },
  };
}

function wrapPhase(angleRad) {
  let out = angleRad % TWO_PI;
  if (out < 0) out += TWO_PI;
  return out;
}

function angleDifference(angle1, angle2) {
  let diff = wrapPhase(angle1) - wrapPhase(angle2);
  if (diff > Math.PI) diff -= TWO_PI;
  else if (diff < -Math.PI) diff += TWO_PI;
  return diff;
}

function mirrorAngleToHalfCircle(angleRad) {
  const wrapped = wrapPhase(angleRad);
  return wrapped > Math.PI ? TWO_PI - wrapped : wrapped;
}

function wrapDegrees(value) {
  let out = value % 360;
  if (out < 0) out += 360;
  return out;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function bearingBetweenPoints(lat1Deg, lon1Deg, lat2Deg, lon2Deg) {
  const lat1 = lat1Deg * DEG_TO_RAD;
  const lat2 = lat2Deg * DEG_TO_RAD;
  const dLon = (lon2Deg - lon1Deg) * DEG_TO_RAD;
  const x = Math.sin(dLon) * Math.cos(lat2);
  const y = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return wrapDegrees(Math.atan2(x, y) * RAD_TO_DEG);
}

function courseFromPositions(latitude, longitude) {
  const out = new Float64Array(latitude.length);
  out.fill(Number.NaN);
  let prevValid = -1;
  for (let i = 0; i < latitude.length; i++) {
    if (!Number.isFinite(latitude[i]) || !Number.isFinite(longitude[i])) continue;
    if (prevValid >= 0) {
      const bearingDeg = bearingBetweenPoints(latitude[prevValid], longitude[prevValid], latitude[i], longitude[i]);
      for (let j = prevValid; j <= i; j++) out[j] = bearingDeg;
    }
    prevValid = i;
  }
  return out;
}

function forwardFillNumeric(values, fillValue = null) {
  const out = new Float64Array(values.length);
  out.set(values);
  let firstValid = -1;
  for (let i = 0; i < out.length; i++) {
    if (Number.isFinite(out[i])) {
      firstValid = i;
      break;
    }
  }
  if (firstValid < 0) {
    if (fillValue == null) return out;
    out.fill(fillValue);
    return out;
  }
  for (let i = 0; i < firstValid; i++) out[i] = fillValue == null ? out[firstValid] : fillValue;
  for (let i = firstValid + 1; i < out.length; i++) {
    if (!Number.isFinite(out[i])) out[i] = out[i - 1];
  }
  return out;
}

function sanitizeSpeedColumn(speedMs, maxReasonableSpeedMs = 15) {
  const out = new Float64Array(speedMs.length);
  for (let i = 0; i < speedMs.length; i++) {
    const value = Number(speedMs[i]);
    out[i] = Number.isFinite(value) && value >= 0 && value <= maxReasonableSpeedMs ? value : 0;
  }
  return out;
}

function inclusiveGrid(start, stop, step) {
  if (!(step > 0)) throw new Error('Grid step must be positive');
  const count = Math.floor((stop - start) / step + 0.5);
  const out = new Float64Array(count + 1);
  for (let i = 0; i <= count; i++) out[i] = start + i * step;
  return out;
}

function wrappedGrid(centerDeg, halfSpanDeg, stepDeg) {
  const values = inclusiveGrid(centerDeg - halfSpanDeg, centerDeg + halfSpanDeg, stepDeg);
  const uniq = [];
  const seen = new Set();
  for (let i = 0; i < values.length; i++) {
    const wrapped = wrapDegrees(values[i]);
    const key = wrapped.toFixed(6);
    if (seen.has(key)) continue;
    seen.add(key);
    uniq.push(wrapped);
  }
  return Float64Array.from(uniq);
}

function prepareTwsInterpolation(speedGridKts) {
  const lowIndex = new Int16Array(speedGridKts.length);
  const highIndex = new Int16Array(speedGridKts.length);
  const alpha = new Float64Array(speedGridKts.length);
  for (let i = 0; i < speedGridKts.length; i++) {
    const clipped = clamp(speedGridKts[i], ILCA7_TWS_KTS[0], ILCA7_TWS_KTS[ILCA7_TWS_KTS.length - 1]);
    let hi = 1;
    while (hi < ILCA7_TWS_KTS.length - 1 && ILCA7_TWS_KTS[hi] < clipped) hi++;
    const lo = hi - 1;
    const span = ILCA7_TWS_KTS[hi] - ILCA7_TWS_KTS[lo];
    lowIndex[i] = lo;
    highIndex[i] = hi;
    alpha[i] = span > 0 ? (clipped - ILCA7_TWS_KTS[lo]) / span : 0;
  }
  return { lowIndex, highIndex, alpha };
}

function interpolateRowAtAngles(row, anglesDeg) {
  const out = new Float64Array(anglesDeg.length);
  for (let i = 0; i < anglesDeg.length; i++) {
    const angle = clamp(anglesDeg[i], ILCA7_TWA_DEG[0], ILCA7_TWA_DEG[ILCA7_TWA_DEG.length - 1]);
    let hi = 1;
    while (hi < ILCA7_TWA_DEG.length - 1 && ILCA7_TWA_DEG[hi] < angle) hi++;
    const lo = hi - 1;
    const span = ILCA7_TWA_DEG[hi] - ILCA7_TWA_DEG[lo];
    const alpha = span > 0 ? (angle - ILCA7_TWA_DEG[lo]) / span : 0;
    out[i] = row[lo] * (1 - alpha) + row[hi] * alpha;
  }
  return out;
}

function interpolateSpeedByAngle(twaDeg) {
  return ILCA7_POLAR_SPEEDS_MS.map(row => interpolateRowAtAngles(row, twaDeg));
}

function twaReliability(twaDeg) {
  const upwind = clamp((twaDeg - 32) / 18, 0, 1);
  const downwind = clamp((180 - twaDeg) / 12, 0, 1);
  return upwind * downwind;
}

function weightedAngleMeanDeg(items) {
  let sumSin = 0;
  let sumCos = 0;
  for (const item of items) {
    const weight = Number(item?.weight);
    const angleDeg = Number(item?.angleDeg);
    if (!(weight > 0) || !Number.isFinite(angleDeg)) continue;
    const angleRad = angleDeg * DEG_TO_RAD;
    sumSin += Math.sin(angleRad) * weight;
    sumCos += Math.cos(angleRad) * weight;
  }
  if (Math.abs(sumSin) < 1e-9 && Math.abs(sumCos) < 1e-9) return Number.NaN;
  return wrapDegrees(Math.atan2(sumSin, sumCos) * RAD_TO_DEG);
}

function weightedMean(items) {
  let sum = 0;
  let weightSum = 0;
  for (const item of items) {
    const weight = Number(item?.weight);
    const value = Number(item?.value);
    if (!(weight > 0) || !Number.isFinite(value)) continue;
    sum += value * weight;
    weightSum += weight;
  }
  return weightSum > 0 ? sum / weightSum : Number.NaN;
}

function quantileSorted(sortedValues, q) {
  if (!sortedValues.length) return Number.NaN;
  if (sortedValues.length === 1) return sortedValues[0];
  const pos = clamp(q, 0, 1) * (sortedValues.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sortedValues[lo];
  const alpha = pos - lo;
  return sortedValues[lo] * (1 - alpha) + sortedValues[hi] * alpha;
}

function angleBinEdgesFromCenters(centersDeg) {
  const centers = Array.from(centersDeg || [])
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!centers.length) throw new Error('At least one angle center is required');
  if (centers.length === 1) {
    return Float64Array.from([
      Math.max(0, centers[0] - 5),
      Math.min(180, centers[0] + 5),
    ]);
  }
  const mids = [];
  for (let i = 0; i < centers.length - 1; i++) mids.push(0.5 * (centers[i] + centers[i + 1]));

  let lowerEdge = centers[0] - 0.5 * (centers[1] - centers[0]);
  let upperEdge = centers[centers.length - 1] + 0.5 * (centers[centers.length - 1] - centers[centers.length - 2]);

  // Keep terminal bins from collapsing when the natural half-step would extend past 0/180.
  if (lowerEdge < 0) {
    const overflow = -lowerEdge;
    if (mids.length) mids[0] += overflow;
    lowerEdge = 0;
  }
  if (upperEdge > 180) {
    const overflow = upperEdge - 180;
    if (mids.length) mids[mids.length - 1] -= overflow;
    upperEdge = 180;
  }

  const edges = [lowerEdge, ...mids, upperEdge];
  return Float64Array.from(edges);
}

function referenceSpeedMsAtTwa(trueWindSpeedMs, twaDeg) {
  const speedByTws = interpolateSpeedByAngle(Float64Array.from([Number(twaDeg)]));
  const twsInterp = prepareTwsInterpolation(Float64Array.from([Number(trueWindSpeedMs) / KNOT_TO_MS]));
  const lowIdx = twsInterp.lowIndex[0];
  const highIdx = twsInterp.highIndex[0];
  const alpha = twsInterp.alpha[0];
  return speedByTws[lowIdx][0] * (1 - alpha) + speedByTws[highIdx][0] * alpha;
}

function generateAthleteSpecificPolar(track, result, sourceName = 'session', options = {}) {
  const statisticQuantile = Number.isFinite(Number(options.statisticQuantile))
    ? Number(options.statisticQuantile)
    : 0.8;
  if (!(statisticQuantile > 0 && statisticQuantile < 1)) {
    throw new Error('statisticQuantile must be between 0 and 1');
  }

  const centers = Array.from(
    options.twaCentersDeg && (Array.isArray(options.twaCentersDeg) || ArrayBuffer.isView(options.twaCentersDeg))
      ? options.twaCentersDeg
      : DEFAULT_POLAR_TWA_CENTERS_DEG,
  )
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  const edges = angleBinEdgesFromCenters(centers);
  const minSamplesPerBin = Math.max(1, Math.round(Number(options.minSamplesPerBin) || 12));
  const athletePoints = [];

  for (let centerIdx = 0; centerIdx < centers.length; centerIdx++) {
    const lowerEdge = edges[centerIdx];
    const upperEdge = edges[centerIdx + 1];
    const isLast = centerIdx === centers.length - 1;
    const speedSamples = [];

    for (let i = 0; i < track.sampleCount; i++) {
      if (!result.inlierMask?.[i]) continue;
      const twaDeg = Number(result.twaDeg?.[i]);
      const speedMs = Number(track.speedMs?.[i]);
      if (!Number.isFinite(twaDeg) || !Number.isFinite(speedMs)) continue;
      const inBin = isLast
        ? (twaDeg >= lowerEdge && twaDeg <= upperEdge)
        : (twaDeg >= lowerEdge && twaDeg < upperEdge);
      if (!inBin) continue;
      speedSamples.push(speedMs);
    }

    if (speedSamples.length < minSamplesPerBin) continue;
    speedSamples.sort((a, b) => a - b);
    const athleteSpeedMs = quantileSorted(speedSamples, statisticQuantile);
    const referenceSpeedMs = referenceSpeedMsAtTwa(result.speedMs, centers[centerIdx]);
    athletePoints.push({
      twaDeg: centers[centerIdx],
      athleteSpeedKts: athleteSpeedMs / KNOT_TO_MS,
      referenceSpeedKts: referenceSpeedMs / KNOT_TO_MS,
      performancePct: referenceSpeedMs > 1e-9 ? (athleteSpeedMs / referenceSpeedMs) * 100 : Number.NaN,
      sampleCount: speedSamples.length,
    });
  }

  athletePoints.sort((a, b) => a.twaDeg - b.twaDeg);

  return {
    sourceName,
    estimatedTrueWindKts: result.speedMs / KNOT_TO_MS,
    statisticLabel: `P${Math.round(statisticQuantile * 100)} inlier speed`,
    points: athletePoints,
  };
}

function maxPolarSpeedMs(trueWindSpeedMs) {
  const cacheKey = Math.round(trueWindSpeedMs * 1000);
  if (_maxPolarSpeedCache.has(cacheKey)) return _maxPolarSpeedCache.get(cacheKey);
  const candidateAngles = inclusiveGrid(35, 180, 1);
  const speedByTws = interpolateSpeedByAngle(candidateAngles);
  const twsInterp = prepareTwsInterpolation(Float64Array.from([trueWindSpeedMs / KNOT_TO_MS]));
  const lo = twsInterp.lowIndex[0];
  const hi = twsInterp.highIndex[0];
  const a = twsInterp.alpha[0];
  let maxSpeed = 0;
  for (let i = 0; i < candidateAngles.length; i++) {
    const speed = speedByTws[lo][i] * (1 - a) + speedByTws[hi][i] * a;
    if (speed > maxSpeed) maxSpeed = speed;
  }
  _maxPolarSpeedCache.set(cacheKey, maxSpeed);
  return maxSpeed;
}

function detectTowOutliers(track, predictedSpeedMs, twaDeg, trueWindSpeedMs, baseMask) {
  const out = new Uint8Array(track.sampleCount);
  const maxPolarSpeed = maxPolarSpeedMs(trueWindSpeedMs);
  const hardMarginMs = 0.75 * KNOT_TO_MS;
  for (let i = 0; i < track.sampleCount; i++) {
    if (!baseMask[i]) continue;
    const residual = track.speedMs[i] - predictedSpeedMs[i];
    const trustedAngle = twaDeg[i] >= 45;
    const softMarginMs = 1.0 * KNOT_TO_MS + 0.1 * predictedSpeedMs[i];
    const hardOverspeed = track.speedMs[i] > maxPolarSpeed + hardMarginMs;
    const softOverspeed = trustedAngle && residual > softMarginMs;
    if (hardOverspeed || softOverspeed) out[i] = 1;
  }
  return out;
}

function detectLowConfidenceSamples(track, twaDeg, predictedSpeedMs, towOutlierMask, baseMask) {
  const out = new Uint8Array(track.sampleCount);
  for (let i = 0; i < track.sampleCount; i++) {
    if (!baseMask[i] || towOutlierMask[i]) continue;
    const residual = track.speedMs[i] - predictedSpeedMs[i];
    const twaConfidence = twaDeg[i] >= 40 && twaDeg[i] <= 172;
    const underspeedMarginMs = 0.8 * KNOT_TO_MS + 0.18 * predictedSpeedMs[i];
    const maneuver = track.turnRateDegS[i] > 16 || track.accelerationMs2[i] > 0.9;
    const underspeed = residual < -underspeedMarginMs;
    if (!twaConfidence || maneuver || underspeed) out[i] = 1;
  }
  return out;
}

function summarizeEstimate(result) {
  return {
    directionDeg: result.directionDeg,
    speedKts: result.speedMs / KNOT_TO_MS,
    fitScore: result.score,
    sampleCount: result.sampleCount,
    inlierCount: result.inlierCount,
    towOutlierCount: result.towOutlierCount,
    lowConfidenceCount: result.lowConfidenceCount,
    meanAbsoluteErrorKts: result.meanAbsoluteErrorKts,
    medianAbsoluteErrorKts: result.medianAbsoluteErrorKts,
  };
}

function estimateWeight(estimate) {
  const inlierRatio = estimate.sampleCount > 0 ? estimate.inlierCount / estimate.sampleCount : 0;
  const scoreWeight = 1 / (1 + Math.max(0, estimate.fitScore) * 12);
  return Math.max(estimate.inlierCount, 1) * clamp(inlierRatio * scoreWeight, 0.15, 1);
}

async function searchTrueWind(track, directionGridDeg, speedGridKts, sampleWeights, options = {}, yieldController = null) {
  const twsInterp = prepareTwsInterpolation(speedGridKts);
  let bestSearch = null;
  const sampleCount = track.sampleCount;

  for (let dIdx = 0; dIdx < directionGridDeg.length; dIdx++) {
    const directionDeg = directionGridDeg[dIdx];
    const directionRad = directionDeg * DEG_TO_RAD;
    const twaDeg = new Float64Array(sampleCount);
    const combinedWeights = new Float64Array(sampleCount);
    let weightSum = 0;

    for (let i = 0; i < sampleCount; i++) {
      const twa = mirrorAngleToHalfCircle(directionRad - track.headingRad[i]) * RAD_TO_DEG;
      twaDeg[i] = twa;
      const combined = sampleWeights[i] * twaReliability(twa);
      combinedWeights[i] = combined;
      weightSum += combined;
    }

    if (!(weightSum > 1e-9)) {
      if (yieldController && (dIdx & 3) === 3) await yieldController.maybeYield();
      continue;
    }

    const speedByTws = interpolateSpeedByAngle(twaDeg);
    let bestSpeedIdx = -1;
    let bestScore = Number.POSITIVE_INFINITY;

    for (let sIdx = 0; sIdx < speedGridKts.length; sIdx++) {
      const lowRow = speedByTws[twsInterp.lowIndex[sIdx]];
      const highRow = speedByTws[twsInterp.highIndex[sIdx]];
      const alpha = twsInterp.alpha[sIdx];
      let loss = 0;

      for (let i = 0; i < sampleCount; i++) {
        const weight = combinedWeights[i];
        if (!(weight > 0)) continue;
        const predicted = lowRow[i] * (1 - alpha) + highRow[i] * alpha;
        const residual = track.speedMs[i] - predicted;
        const absResidual = Math.abs(residual);
        const huberLoss = absResidual <= 0.35
          ? 0.5 * absResidual * absResidual
          : 0.35 * (absResidual - 0.175);
        const overspeedMargin = 0.45 + 0.08 * predicted;
        const overspeedPenalty = Math.max(residual - overspeedMargin, 0);
        loss += (huberLoss + 1.75 * overspeedPenalty * overspeedPenalty) * weight;
      }

      let score = loss / weightSum;
      if (Number.isFinite(options.priorDirectionDeg) && options.priorDirectionWeight > 0) {
        const directionDeltaDeg = Math.abs(angleDifference(directionDeg * DEG_TO_RAD, options.priorDirectionDeg * DEG_TO_RAD)) * RAD_TO_DEG;
        score += options.priorDirectionWeight * directionDeltaDeg * directionDeltaDeg;
      }
      if (Number.isFinite(options.priorSpeedKts) && options.priorSpeedWeight > 0) {
        const speedDelta = speedGridKts[sIdx] - options.priorSpeedKts;
        score += options.priorSpeedWeight * speedDelta * speedDelta;
      }

      if (score < bestScore) {
        bestScore = score;
        bestSpeedIdx = sIdx;
      }
    }

    if (bestSpeedIdx >= 0 && (!bestSearch || bestScore < bestSearch.score)) {
      const lowRow = speedByTws[twsInterp.lowIndex[bestSpeedIdx]];
      const highRow = speedByTws[twsInterp.highIndex[bestSpeedIdx]];
      const alpha = twsInterp.alpha[bestSpeedIdx];
      const predictionMs = new Float64Array(sampleCount);
      for (let i = 0; i < sampleCount; i++) predictionMs[i] = lowRow[i] * (1 - alpha) + highRow[i] * alpha;
      bestSearch = {
        score: bestScore,
        directionDeg,
        speedKts: speedGridKts[bestSpeedIdx],
        predictionMs,
        twaDeg,
      };
    }

    if (yieldController && (dIdx & 3) === 3) await yieldController.maybeYield();
  }

  if (!bestSearch) throw new Error('The track did not contain enough reliable sailing samples to estimate the wind');
  return bestSearch;
}

async function estimateTrueWind(track, options = {}, yieldController = null) {
  if (!track || track.sampleCount === 0) throw new Error('Cannot estimate wind from an empty track');
  const disableTowFiltering = !!options.disableTowFiltering;

  const baseMask = new Uint8Array(track.sampleCount);
  for (let i = 0; i < track.sampleCount; i++) {
    baseMask[i] = Number.isFinite(track.speedMs[i]) && Number.isFinite(track.headingRad[i]) && track.speedMs[i] >= 0.45 * KNOT_TO_MS ? 1 : 0;
  }
  const baseWeights = baseSampleWeights(track);
  let usableWeightCount = 0;
  for (let i = 0; i < baseWeights.length; i++) {
    baseWeights[i] *= baseMask[i];
    if (baseWeights[i] > 0) usableWeightCount++;
  }
  if (!usableWeightCount) throw new Error('No usable sailing samples were found in the track');

  const coarseDirectionGridDeg = Number.isFinite(options.coarseDirectionCenterDeg) && Number.isFinite(options.coarseDirectionHalfSpanDeg)
    ? wrappedGrid(options.coarseDirectionCenterDeg, options.coarseDirectionHalfSpanDeg, options.coarseDirectionStepDeg ?? 4)
    : inclusiveGrid(0, 356, options.coarseDirectionStepDeg ?? 4);
  const coarseSpeedGridKts = inclusiveGrid(options.minTrueWindSpeedKts ?? 4, options.maxTrueWindSpeedKts ?? 20, options.coarseSpeedStepKts ?? 0.5);
  const coarseSearch = await searchTrueWind(track, coarseDirectionGridDeg, coarseSpeedGridKts, baseWeights, {
    priorDirectionDeg: options.priorDirectionDeg,
    priorDirectionWeight: options.priorDirectionWeight ?? 0,
    priorSpeedKts: options.priorSpeedKts,
    priorSpeedWeight: options.priorSpeedWeight ?? 0,
  }, yieldController);

  const firstTowMask = disableTowFiltering
    ? new Uint8Array(track.sampleCount)
    : detectTowOutliers(track, coarseSearch.predictionMs, coarseSearch.twaDeg, coarseSearch.speedKts * KNOT_TO_MS, baseMask);
  const refinedWeights = new Float64Array(baseWeights.length);
  for (let i = 0; i < baseWeights.length; i++) refinedWeights[i] = firstTowMask[i] ? 0 : baseWeights[i];

  const debiasedCoarse = await searchTrueWind(track, coarseDirectionGridDeg, coarseSpeedGridKts, refinedWeights, {
    priorDirectionDeg: options.priorDirectionDeg,
    priorDirectionWeight: options.priorDirectionWeight ?? 0,
    priorSpeedKts: options.priorSpeedKts,
    priorSpeedWeight: options.priorSpeedWeight ?? 0,
  }, yieldController);

  const refineDirectionHalfSpanDeg = options.refineDirectionHalfSpanDeg ?? 6;
  const refineSpeedHalfSpanKts = options.refineSpeedHalfSpanKts ?? 2;
  const minTrueWindSpeedKts = options.minTrueWindSpeedKts ?? 4;
  const maxTrueWindSpeedKts = options.maxTrueWindSpeedKts ?? 20;
  const directionGridDeg = wrappedGrid(
    debiasedCoarse.directionDeg,
    refineDirectionHalfSpanDeg,
    options.refineDirectionStepDeg ?? 0.25,
  );
  const speedGridKts = inclusiveGrid(
    Math.max(minTrueWindSpeedKts, debiasedCoarse.speedKts - refineSpeedHalfSpanKts),
    Math.min(maxTrueWindSpeedKts, debiasedCoarse.speedKts + refineSpeedHalfSpanKts),
    options.refineSpeedStepKts ?? 0.1,
  );
  const refinedSearch = await searchTrueWind(track, directionGridDeg, speedGridKts, refinedWeights, {
    priorDirectionDeg: options.priorDirectionDeg,
    priorDirectionWeight: options.priorDirectionWeight ?? 0,
    priorSpeedKts: options.priorSpeedKts,
    priorSpeedWeight: options.priorSpeedWeight ?? 0,
  }, yieldController);

  const towOutlierMask = disableTowFiltering
    ? new Uint8Array(track.sampleCount)
    : detectTowOutliers(track, refinedSearch.predictionMs, refinedSearch.twaDeg, refinedSearch.speedKts * KNOT_TO_MS, baseMask);
  const lowConfidenceMask = detectLowConfidenceSamples(track, refinedSearch.twaDeg, refinedSearch.predictionMs, towOutlierMask, baseMask);

  let inlierCount = 0;
  let towOutlierCount = 0;
  let lowConfidenceCount = 0;
  let absResidualSum = 0;
  const absResiduals = [];
  const inlierMask = new Uint8Array(track.sampleCount);
  const residualSpeedMs = new Float64Array(track.sampleCount);
  for (let i = 0; i < track.sampleCount; i++) {
    residualSpeedMs[i] = track.speedMs[i] - refinedSearch.predictionMs[i];
    if (towOutlierMask[i]) towOutlierCount++;
    if (lowConfidenceMask[i]) lowConfidenceCount++;
    if (baseMask[i] && !towOutlierMask[i] && !lowConfidenceMask[i]) {
      inlierMask[i] = 1;
      inlierCount++;
      const absResidual = Math.abs(residualSpeedMs[i]);
      absResidualSum += absResidual;
      absResiduals.push(absResidual);
    }
  }
  absResiduals.sort((a, b) => a - b);
  const medianAbsoluteResidual = absResiduals.length
    ? absResiduals[Math.floor(absResiduals.length / 2)]
    : Number.NaN;

  return {
    directionDeg: refinedSearch.directionDeg,
    speedMs: refinedSearch.speedKts * KNOT_TO_MS,
    score: refinedSearch.score,
    predictionMs: refinedSearch.predictionMs,
    twaDeg: refinedSearch.twaDeg,
    inlierMask,
    residualSpeedMs,
    sampleCount: track.sampleCount,
    inlierCount,
    towOutlierCount,
    lowConfidenceCount,
    meanAbsoluteErrorKts: inlierCount > 0 ? (absResidualSum / inlierCount) / KNOT_TO_MS : Number.NaN,
    medianAbsoluteErrorKts: Number.isFinite(medianAbsoluteResidual) ? medianAbsoluteResidual / KNOT_TO_MS : Number.NaN,
  };
}

function baseSampleWeights(track) {
  const out = new Float64Array(track.sampleCount);
  for (let i = 0; i < track.sampleCount; i++) {
    const speedWeight = clamp((track.speedMs[i] / KNOT_TO_MS - 0.8) / 2.5, 0, 1);
    const heelWeight = clamp((Math.abs(track.heelDeg[i]) - 2) / 8, 0, 1);
    const steadyTurnWeight = clamp(1 - track.turnRateDegS[i] / 18, 0, 1);
    const steadySpeedWeight = clamp(1 - track.accelerationMs2[i] / 1.2, 0, 1);
    const headingCourseDelta = Math.abs(angleDifference(track.courseRad[i], track.headingRad[i])) * RAD_TO_DEG;
    const leewayWeight = clamp(1 - headingCourseDelta / 45, 0, 1);
    const dtWeight = clamp(1 - Math.max(track.dtSeconds[i] - 1.5, 0) / 4, 0, 1);
    out[i] = (0.1 + 0.4 * speedWeight + 0.2 * heelWeight + 0.15 * steadyTurnWeight + 0.1 * steadySpeedWeight + 0.05 * leewayWeight) * dtWeight;
  }
  return out;
}

function subsetTrack(track, startIdx, endIdx) {
  const size = Math.max(0, endIdx - startIdx);
  const absoluteTs = track.absoluteTs.slice(startIdx, endIdx);
  const elapsedSeconds = new Float64Array(size);
  const dtSeconds = new Float64Array(size);
  const baseElapsed = size > 0 ? track.elapsedSeconds[startIdx] : 0;
  for (let i = 0; i < size; i++) elapsedSeconds[i] = track.elapsedSeconds[startIdx + i] - baseElapsed;
  for (let i = 0; i < size; i++) dtSeconds[i] = track.dtSeconds[startIdx + i];
  if (size > 1) {
    const sample = Array.from(dtSeconds.slice(1)).sort((a, b) => a - b);
    dtSeconds[0] = Math.max(sample.length ? sample[Math.floor(sample.length / 2)] : dtSeconds[1], 1e-3);
  } else if (size === 1) {
    dtSeconds[0] = Math.max(dtSeconds[0] || 0.5, 1e-3);
  }
  return {
    schemaName: track.schemaName,
    absoluteTs,
    latitude: track.latitude.slice(startIdx, endIdx),
    longitude: track.longitude.slice(startIdx, endIdx),
    speedMs: track.speedMs.slice(startIdx, endIdx),
    courseRad: track.courseRad.slice(startIdx, endIdx),
    headingRad: track.headingRad.slice(startIdx, endIdx),
    heelDeg: track.heelDeg.slice(startIdx, endIdx),
    trimDeg: track.trimDeg.slice(startIdx, endIdx),
    elapsedSeconds,
    dtSeconds,
    turnRateDegS: track.turnRateDegS.slice(startIdx, endIdx),
    accelerationMs2: track.accelerationMs2.slice(startIdx, endIdx),
    sampleCount: size,
  };
}

function windowRepresentativePosition(track, startIdx, endIdx) {
  const lats = [];
  const lons = [];
  for (let i = startIdx; i < endIdx; i++) {
    if (Number.isFinite(track.latitude[i]) && Number.isFinite(track.longitude[i])) {
      lats.push(track.latitude[i]);
      lons.push(track.longitude[i]);
    }
  }
  if (!lats.length) return { lat: Number.NaN, lon: Number.NaN };
  lats.sort((a, b) => a - b);
  lons.sort((a, b) => a - b);
  return {
    lat: lats[Math.floor(lats.length / 2)],
    lon: lons[Math.floor(lons.length / 2)],
  };
}

function smoothWindTimeSeries(points, span = 3) {
  if (!Array.isArray(points) || points.length <= 2 || span <= 1) return points || [];
  const halfSpan = Math.floor(span / 2);
  return points.map((point, index) => {
    const start = Math.max(0, index - halfSpan);
    const stop = Math.min(points.length, index + halfSpan + 1);
    const window = points.slice(start, stop);
    const directionDeg = weightedAngleMeanDeg(window.map(item => ({
      angleDeg: item.directionDeg,
      weight: Math.max(item.inlierCount || 1, 1),
    })));
    const speeds = window
      .map(item => Number(item.speedKts))
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    return {
      ...point,
      directionDeg: Number.isFinite(directionDeg) ? directionDeg : point.directionDeg,
      speedKts: speeds.length ? speeds[Math.floor(speeds.length / 2)] : point.speedKts,
    };
  });
}

async function estimateTrueWindSeries(track, sourceName = 'session', options = {}, sessionEstimate = null, yieldController = null) {
  if (!track || track.sampleCount === 0) {
    return { sourceName, windowSeconds: options.windowSeconds ?? 300, stepSeconds: options.stepSeconds ?? 180, points: [] };
  }

  const windowSeconds = options.windowSeconds ?? 300;
  const stepSeconds = options.stepSeconds ?? 180;
  const minWindowSamples = options.minWindowSamples ?? 180;
  const minInlierCount = options.minInlierCount ?? 40;
  const maxDirectionStepDeg = options.maxDirectionStepDeg ?? 18;
  if (!(windowSeconds > 0) || !(stepSeconds > 0)) throw new Error('windowSeconds and stepSeconds must be positive');

  const globalResult = sessionEstimate || await estimateTrueWind(track, {}, yieldController);
  const totalDuration = track.elapsedSeconds[track.sampleCount - 1] - track.elapsedSeconds[0];
  const centers = [];
  if (totalDuration <= windowSeconds) {
    centers.push(0.5 * (track.elapsedSeconds[0] + track.elapsedSeconds[track.sampleCount - 1]));
  } else {
    const startCenter = track.elapsedSeconds[0] + windowSeconds * 0.5;
    const endCenter = track.elapsedSeconds[track.sampleCount - 1] - windowSeconds * 0.5;
    for (let center = startCenter; center <= endCenter + 1e-6; center += stepSeconds) centers.push(center);
  }

  const points = [];
  let directionAnchorDeg = globalResult.directionDeg;
  let speedAnchorKts = globalResult.speedMs / KNOT_TO_MS;
  const elapsed = track.elapsedSeconds;
  let startIdx = 0;
  let endIdx = 0;

  for (let cIdx = 0; cIdx < centers.length; cIdx++) {
    const centerSeconds = centers[cIdx];
    const windowStart = centerSeconds - 0.5 * windowSeconds;
    const windowEnd = centerSeconds + 0.5 * windowSeconds;
    while (startIdx < elapsed.length && elapsed[startIdx] < windowStart) startIdx++;
    while (endIdx < elapsed.length && elapsed[endIdx] <= windowEnd) endIdx++;
    const count = endIdx - startIdx;
    if (count < minWindowSamples) {
      if (yieldController && (cIdx & 1) === 1) await yieldController.maybeYield();
      continue;
    }

    const windowTrack = subsetTrack(track, startIdx, endIdx);
    let localResult;
    try {
      localResult = await estimateTrueWind(windowTrack, {
        coarseDirectionStepDeg: 6,
        coarseSpeedStepKts: 0.5,
        refineDirectionHalfSpanDeg: Math.max(4, 0.5 * maxDirectionStepDeg),
        refineDirectionStepDeg: 0.5,
        refineSpeedHalfSpanKts: 1.5,
        refineSpeedStepKts: 0.1,
        minTrueWindSpeedKts: Math.max(2, speedAnchorKts - 4),
        maxTrueWindSpeedKts: Math.min(24, speedAnchorKts + 4),
        coarseDirectionCenterDeg: directionAnchorDeg,
        coarseDirectionHalfSpanDeg: maxDirectionStepDeg,
        priorDirectionDeg: directionAnchorDeg,
        priorDirectionWeight: 0.00035,
        priorSpeedKts: speedAnchorKts,
        priorSpeedWeight: 0.012,
        ...(options.windEstimateOptions || {}),
      }, yieldController);
    } catch {
      if (yieldController && (cIdx & 1) === 1) await yieldController.maybeYield();
      continue;
    }
    if (localResult.inlierCount < minInlierCount) {
      if (yieldController && (cIdx & 1) === 1) await yieldController.maybeYield();
      continue;
    }

    directionAnchorDeg = localResult.directionDeg;
    speedAnchorKts = localResult.speedMs / KNOT_TO_MS;

    let representativeIndex = startIdx;
    let bestDelta = Number.POSITIVE_INFINITY;
    for (let i = startIdx; i < endIdx; i++) {
      const delta = Math.abs(elapsed[i] - centerSeconds);
      if (delta < bestDelta) {
        bestDelta = delta;
        representativeIndex = i;
      }
    }
    const position = windowRepresentativePosition(track, startIdx, endIdx);
    points.push({
      ts: track.absoluteTs[representativeIndex],
      elapsedSeconds: elapsed[representativeIndex],
      lat: position.lat,
      lon: position.lon,
      directionDeg: localResult.directionDeg,
      speedKts: localResult.speedMs / KNOT_TO_MS,
      fitScore: localResult.score,
      sampleCount: count,
      inlierCount: localResult.inlierCount,
    });

    if (yieldController && (cIdx & 1) === 1) await yieldController.maybeYield();
  }

  return {
    sourceName,
    windowSeconds,
    stepSeconds,
    points: smoothWindTimeSeries(points),
  };
}

export function buildWindTrack(points, sourceName = 'csv') {
  if (!Array.isArray(points) || !points.length) return null;
  const filtered = points
    .map(point => ({
      ts: Number(point?.ts),
      lat: Number(point?.lat),
      lon: Number(point?.lon),
      sog: Number(point?.sog),
      cog: Number(point?.cog),
      hdg: Number(point?.hdg),
      heel: Number(point?.heel),
      trim: Number(point?.trim),
    }))
    .filter(point => Number.isFinite(point.ts) && Number.isFinite(point.lat) && Number.isFinite(point.lon))
    .sort((a, b) => a.ts - b.ts);
  if (!filtered.length) return null;

  const absoluteTs = new Float64Array(filtered.length);
  const latitude = new Float64Array(filtered.length);
  const longitude = new Float64Array(filtered.length);
  const rawSpeedMs = new Float64Array(filtered.length);
  const headingDeg = new Float64Array(filtered.length);
  const courseDeg = new Float64Array(filtered.length);
  const heelDeg = new Float64Array(filtered.length);
  const trimDeg = new Float64Array(filtered.length);

  for (let i = 0; i < filtered.length; i++) {
    const point = filtered[i];
    absoluteTs[i] = point.ts;
    latitude[i] = point.lat;
    longitude[i] = point.lon;
    rawSpeedMs[i] = Number.isFinite(point.sog) ? point.sog * KNOT_TO_MS : Number.NaN;
    headingDeg[i] = Number.isFinite(point.hdg) ? wrapDegrees(point.hdg) : Number.NaN;
    courseDeg[i] = Number.isFinite(point.cog) ? wrapDegrees(point.cog) : Number.NaN;
    heelDeg[i] = Number.isFinite(point.heel) ? point.heel : 0;
    trimDeg[i] = Number.isFinite(point.trim) ? point.trim : 0;
  }

  const speedMs = sanitizeSpeedColumn(rawSpeedMs);
  const inferredCourseDeg = courseFromPositions(latitude, longitude);
  for (let i = 0; i < courseDeg.length; i++) {
    if (!Number.isFinite(courseDeg[i])) courseDeg[i] = inferredCourseDeg[i];
  }
  const headingFilled = forwardFillNumeric(headingDeg, Number.isFinite(courseDeg[0]) ? courseDeg[0] : 0);
  for (let i = 0; i < courseDeg.length; i++) {
    if (!Number.isFinite(courseDeg[i])) courseDeg[i] = headingFilled[i];
  }
  const courseFilled = forwardFillNumeric(courseDeg, headingFilled[0]);

  const elapsedSeconds = new Float64Array(filtered.length);
  const dtSeconds = new Float64Array(filtered.length);
  const turnRateDegS = new Float64Array(filtered.length);
  const accelerationMs2 = new Float64Array(filtered.length);
  const headingRad = new Float64Array(filtered.length);
  const courseRad = new Float64Array(filtered.length);

  const baseTs = absoluteTs[0];
  for (let i = 0; i < filtered.length; i++) {
    elapsedSeconds[i] = absoluteTs[i] - baseTs;
    headingRad[i] = headingFilled[i] * DEG_TO_RAD;
    courseRad[i] = courseFilled[i] * DEG_TO_RAD;
  }
  for (let i = 0; i < filtered.length; i++) {
    if (i === 0) {
      dtSeconds[i] = filtered.length > 1 ? Math.max(absoluteTs[1] - absoluteTs[0], 1e-3) : 0.5;
      turnRateDegS[i] = 0;
      accelerationMs2[i] = 0;
      continue;
    }
    const dt = Math.max(absoluteTs[i] - absoluteTs[i - 1], 1e-3);
    dtSeconds[i] = dt;
    turnRateDegS[i] = Math.abs(angleDifference(headingRad[i], headingRad[i - 1])) * RAD_TO_DEG / dt;
    accelerationMs2[i] = Math.abs(speedMs[i] - speedMs[i - 1]) / dt;
  }

  return {
    schemaName: sourceName,
    absoluteTs,
    latitude,
    longitude,
    speedMs,
    courseRad,
    headingRad,
    heelDeg,
    trimDeg,
    elapsedSeconds,
    dtSeconds,
    turnRateDegS,
    accelerationMs2,
    sampleCount: filtered.length,
  };
}

export async function estimateWindFromCsvPoints(points, options = {}) {
  const track = buildWindTrack(points, options.sourceName || 'csv');
  if (!track || track.sampleCount < (options.minTrackSamples ?? 120)) return null;

  const yieldController = createYieldController(options.yieldBudgetMs ?? 12);
  const windEstimateOptions = {
    ...(options.windEstimateOptions || {}),
    disableTowFiltering: !!options.disableTowFiltering,
  };
  const sessionEstimate = await estimateTrueWind(track, windEstimateOptions, yieldController);
  const localSeries = await estimateTrueWindSeries(
    track,
    options.sourceName || 'session',
    {
      windowSeconds: options.windowSeconds ?? 300,
      stepSeconds: options.stepSeconds ?? 180,
      minWindowSamples: options.minWindowSamples ?? 180,
      minInlierCount: options.minInlierCount ?? 40,
      maxDirectionStepDeg: options.maxDirectionStepDeg ?? 18,
      windEstimateOptions,
    },
    sessionEstimate,
    yieldController,
  );
  await yieldController.maybeYield(true);

  const session = summarizeEstimate(sessionEstimate);
  return {
    sourceName: options.sourceName || 'csv',
    sampleCount: track.sampleCount,
    tsStart: track.absoluteTs[0],
    tsEnd: track.absoluteTs[track.absoluteTs.length - 1],
    session,
    localSeries,
    qualityWeight: estimateWeight(session),
  };
}

export async function buildAthletePolarFromPoints(points, options = {}) {
  const sourceName = options.sourceName || 'session';
  const track = buildWindTrack(points, sourceName);
  if (!track || track.sampleCount < (options.minTrackSamples ?? 80)) return null;

  const yieldController = createYieldController(options.yieldBudgetMs ?? 12);
  const windResult = await estimateTrueWind(track, options.windEstimateOptions || {}, yieldController);
  const polar = generateAthleteSpecificPolar(track, windResult, sourceName, options);
  await yieldController.maybeYield(true);

  const curvePointCount = polar.points.filter(point =>
    Number.isFinite(Number(point?.athleteSpeedKts)) &&
    Number(point?.sampleCount) > 0,
  ).length;

  return {
    sourceName,
    trackSampleCount: track.sampleCount,
    tsStart: track.absoluteTs[0],
    tsEnd: track.absoluteTs[track.absoluteTs.length - 1],
    estimatedTrueWindKts: polar.estimatedTrueWindKts,
    statisticLabel: polar.statisticLabel,
    pointCount: polar.points.length,
    curvePointCount,
    wind: summarizeEstimate(windResult),
    points: polar.points,
  };
}

export function combineSessionWindEstimates(estimates) {
  const rows = (Array.isArray(estimates) ? estimates : [])
    .filter(item => item?.session && Number.isFinite(item.session.directionDeg) && Number.isFinite(item.session.speedKts))
    .map(item => ({
      angleDeg: item.session.directionDeg,
      value: item.session.speedKts,
      weight: Number(item.qualityWeight) > 0 ? Number(item.qualityWeight) : estimateWeight(item.session),
    }));
  if (!rows.length) return null;
  const directionDeg = weightedAngleMeanDeg(rows);
  const speedKts = weightedMean(rows);
  const totalWeight = rows.reduce((sum, row) => sum + row.weight, 0);
  return {
    directionDeg,
    speedKts,
    sourceCount: rows.length,
    weight: totalWeight,
  };
}

export function combineLocalWindEstimates(candidates) {
  const rows = (Array.isArray(candidates) ? candidates : [])
    .filter(item => Number.isFinite(item?.directionDeg) && Number.isFinite(item?.speedKts))
    .map(item => ({
      angleDeg: item.directionDeg,
      value: item.speedKts,
      weight: Math.max(Number(item.inlierCount) || 0, 1) * clamp(1 / (1 + Math.max(0, Number(item.fitScore) || 0) * 16), 0.1, 1),
    }));
  if (!rows.length) return null;
  return {
    directionDeg: weightedAngleMeanDeg(rows),
    speedKts: weightedMean(rows),
    sourceCount: rows.length,
    weight: rows.reduce((sum, row) => sum + row.weight, 0),
  };
}
