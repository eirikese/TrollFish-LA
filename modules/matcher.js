/**
 * TrollFish — matcher.js
 * GPS track matching — direct port from matcher.py
 *
 * Finds the best time offset between a video GPS track and a CSV GPS track
 * using coarse+fine sweep with haversine distance scoring.
 */

import { MATCH_SAMPLE_POINTS } from './config.js';

// ── Haversine distance (metres) ───────────────────────────────────────

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}


// ── Subsample a track to N evenly-spaced points ───────────────────────

function subsample(points, n) {
  if (points.length <= n) return points;
  const step = (points.length - 1) / (n - 1);
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push(points[Math.round(i * step)]);
  }
  return out;
}

function validGpsPoints(points) {
  return (Array.isArray(points) ? points : [])
    .filter(p => (
      Number.isFinite(Number(p?.ts)) &&
      Number.isFinite(Number(p?.lat)) &&
      Number.isFinite(Number(p?.lon))
    ))
    .sort((a, b) => Number(a.ts) - Number(b.ts));
}


// ── Find CSV point nearest to a given timestamp ───────────────────────

function findNearest(sortedPts, ts) {
  let lo = 0, hi = sortedPts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedPts[mid].ts < ts) lo = mid + 1;
    else hi = mid;
  }
  // Check neighbours
  let best = lo;
  let bestDist = Math.abs(sortedPts[lo].ts - ts);
  if (lo > 0) {
    const d = Math.abs(sortedPts[lo - 1].ts - ts);
    if (d < bestDist) { best = lo - 1; bestDist = d; }
  }
  if (lo < sortedPts.length - 1) {
    const d = Math.abs(sortedPts[lo + 1].ts - ts);
    if (d < bestDist) { best = lo + 1; bestDist = d; }
  }
  return { point: sortedPts[best], dt: bestDist };
}


// ── Score a given offset ──────────────────────────────────────────────

/**
 * Score a candidate time offset.
 * @param {object[]} videoPts — subsampled video GPS points
 * @param {object[]} csvPts — sorted CSV GPS points
 * @param {number} offset — seconds to add to video timestamps
 * @returns {{ score, median_distance_m, p90_distance_m, coverage, sample_count } | null}
 */
function scoreOffset(videoPts, csvPts, offset) {
  const maxDt = 5.0; // max allowed time gap in seconds
  const csvTsStart = csvPts[0].ts;
  const csvTsEnd = csvPts[csvPts.length - 1].ts;

  const distances = [];
  for (const vp of videoPts) {
    const ts = vp.ts + offset;
    if (ts < csvTsStart - maxDt || ts > csvTsEnd + maxDt) continue;
    const { point, dt } = findNearest(csvPts, ts);
    if (dt > maxDt) continue;
    distances.push(haversineM(vp.lat, vp.lon, point.lat, point.lon));
  }

  if (distances.length < 3) return null;

  distances.sort((a, b) => a - b);
  const coverage = distances.length / videoPts.length;
  const median = distances[Math.floor(distances.length / 2)];
  const p90 = distances[Math.floor(distances.length * 0.9)];

  // Combined score: lower is better — median + penalty for poor coverage + p90 weight
  const score = median * (1.0 + (1.0 - coverage) * 2.0) + p90 * 0.1;

  return {
    score,
    median_distance_m: median,
    p90_distance_m: p90,
    coverage,
    sample_count: distances.length,
  };
}


// ── Best offset: coarse + fine sweep ──────────────────────────────────

function bestOffset(videoPts, csvPts) {
  // Subsample video for speed
  const videoSub = subsample(videoPts, MATCH_SAMPLE_POINTS);

  // Coarse sweep: -300s to +300s in 5s steps
  let bestResult = null;
  let bestOffsetVal = 0;

  for (let off = -300; off <= 300; off += 5) {
    const result = scoreOffset(videoSub, csvPts, off);
    if (result && (bestResult === null || result.score < bestResult.score)) {
      bestResult = result;
      bestOffsetVal = off;
    }
  }

  // ── Epoch-alignment sweep ──────────────────────────────────────
  // If video and CSV timestamps are in different epochs (e.g. wrong
  // GoPro clock), the ±300s sweep above won't find a match.
  // Compute the offset that would align midpoints and sweep around it.
  const videoMidTs = (videoPts[0].ts + videoPts[videoPts.length - 1].ts) / 2;
  const csvMidTs = (csvPts[0].ts + csvPts[csvPts.length - 1].ts) / 2;
  const epochOff = Math.round(csvMidTs - videoMidTs);

  if (Math.abs(epochOff) > 300) {
    for (let off = epochOff - 300; off <= epochOff + 300; off += 5) {
      const result = scoreOffset(videoSub, csvPts, off);
      if (result && (bestResult === null || result.score < bestResult.score)) {
        bestResult = result;
        bestOffsetVal = off;
      }
    }
  }

  if (bestResult === null) return null;

  // Fine sweep: ±6s around best in 0.5s steps
  for (let off = bestOffsetVal - 6; off <= bestOffsetVal + 6; off += 0.5) {
    const result = scoreOffset(videoSub, csvPts, off);
    if (result && result.score < bestResult.score) {
      bestResult = result;
      bestOffsetVal = off;
    }
  }

  return {
    ...bestResult,
    offset_seconds: bestOffsetVal,
  };
}


// ── Public API ────────────────────────────────────────────────────────

/**
 * Match video GPS tracks against CSV GPS tracks.
 *
 * @param {{ file_id: string, points: object[] }[]} videoTracks
 * @param {{ file_id: string, points: object[] }[]} csvTracks
 * @param {number} [maxRankPerVideo=3] — top N CSV candidates per video
 * @returns {object[]} — match results sorted by video_file_id and rank
 */
export function matchVideoTracksToCsv(videoTracks, csvTracks, maxRankPerVideo = 3) {
  const results = [];

  for (const video of videoTracks) {
    const videoPts = validGpsPoints(video.points);
    if (videoPts.length < 12) continue;

    const candidates = [];

    for (const csv of csvTracks) {
      const csvPts = validGpsPoints(csv.points);
      if (csvPts.length < 12) continue;

      const best = bestOffset(videoPts, csvPts);
      if (!best) continue;

      candidates.push({
        video_file_id: video.file_id,
        csv_file_id: csv.file_id,
        score: best.score,
        median_distance_m: best.median_distance_m,
        p90_distance_m: best.p90_distance_m,
        coverage: best.coverage,
        offset_seconds: best.offset_seconds,
        sample_count: best.sample_count,
      });
    }

    // Sort by score (ascending = best first)
    candidates.sort((a, b) => a.score - b.score);

    for (let rank = 0; rank < Math.min(candidates.length, maxRankPerVideo); rank++) {
      candidates[rank].rank = rank + 1;
      results.push(candidates[rank]);
    }
  }

  return results;
}
