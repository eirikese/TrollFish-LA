/**
 * TrollFish — pose-engine.js
 * Browser-side MediaPipe Pose processing pipeline.
 *
 * Processes a video file at SKELETON_TARGET_FPS using MediaPipe Tasks Vision,
 * then places, smooths, and computes metrics for each skeleton frame.
 *
 * Uses the CDN build of @mediapipe/tasks-vision for WASM/WebGPU.
 * Model file served from the app's local vendor directory.
 *
 * On iPad Safari WebGL cannot be used in Workers, so everything runs on
 * the main thread. We yield control after each frame to keep UI responsive.
 *
 * Supports 'lite', 'full', and 'heavy' pose_landmarker models.
 * Real-time mode: detectLive() runs on each video frame during playback.
 */

import {
  SKELETON_TARGET_FPS, SKELETON_HIP_PLANE_Z, SKELETON_LOWER_PLANE_Z,
  DEFAULT_CV_CONFIG, getCalibration, isGoPro13Model,
} from './config.js';
import {
  defaultCameraPoseAndRotation,
  computePlacedSkeletonSymmetricRaycast,
  placeSkeletonOnBoat,
} from './rayplane.js';
import { SkeletonPlacementKalman, KALMAN_DEFAULTS } from './skeleton-filter.js?v=20260604pose2';
import { computeFrameMetrics, scaleSkeletonToAthleteHeight } from './skeleton-metrics.js?v=20260604pose2';
import * as DB from './db.js';
import * as Storage from './storage.js';
import * as FM from './file-manager.js';
import * as AutoPnP from './autopnp-engine.js?v=20260604pose2';
import * as Rudder from './rudder-engine.js?v=20260604pose2';
import * as Boom from './boom-engine.js?v=20260604pose2';

// ── MediaPipe lazy loader ─────────────────────────────────────────────

const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';

/** Model paths — lite from Google Storage, full served locally */
const MODEL_PATHS = {
  full: new URL('../vendor/mediapipe/pose_landmarker_full.task', import.meta.url).href,
  lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
  heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
};
const PLAYBACK_STALL_FALLBACK_MS = 4000;
const SEEK_WAIT_VISIBLE_MS = 4000;
const SEEK_WAIT_HIDDEN_MS = 15000;

let _PoseLandmarker = null;
let _FilesetResolver = null;

/** Cached landmarkers keyed by model name */
const _landmarkerCache = {}; // 'full' | 'lite' | 'heavy' → PoseLandmarker
let _currentModel = null;   // the currently active model name
let _loadingPromise = null;
const _landmarkerTsState = new WeakMap(); // PoseLandmarker -> { lastOutMs, offsetMs }

/**
 * MediaPipe VIDEO mode requires timestamps to be strictly increasing per graph.
 * We keep a per-landmarker offset so segment/full runs can restart at lower
 * video-local times without violating monotonic timestamp requirements.
 */
function _nextLandmarkerTimestampMs(landmarker, proposedMs) {
  const raw = Number.isFinite(Number(proposedMs)) ? Math.max(0, Math.round(Number(proposedMs))) : 0;
  let state = _landmarkerTsState.get(landmarker);
  if (!state) {
    state = { lastOutMs: Number.NEGATIVE_INFINITY, offsetMs: 0 };
    _landmarkerTsState.set(landmarker, state);
  }

  let out = raw + state.offsetMs;
  if (!(out > state.lastOutMs)) {
    state.offsetMs += (state.lastOutMs + 1 - out);
    out = raw + state.offsetMs;
  }

  state.lastOutMs = out;
  return out;
}

function _detectForVideoSafe(landmarker, source, timestampMs) {
  const safeTsMs = _nextLandmarkerTimestampMs(landmarker, timestampMs);
  return landmarker.detectForVideo(source, safeTsMs);
}

/**
 * Lazy-load the MediaPipe WASM runtime and create a PoseLandmarker
 * for the given model variant.
 *
 * @param {'full'|'lite'|'heavy'} model — model variant (default: 'full')
 * @returns {Promise<PoseLandmarker>}
 */
async function ensurePoseLandmarker(model = 'full') {
  const requested = String(model || 'full').toLowerCase();
  const key = requested === 'lite' ? 'lite' : (requested === 'heavy' ? 'heavy' : 'full');

  // Return cached if same model requested
  if (_landmarkerCache[key]) {
    _currentModel = key;
    return _landmarkerCache[key];
  }

  // If already loading this model, wait
  if (_loadingPromise && _currentModel === key) return _loadingPromise;

  _loadingPromise = (async () => {
    // Load WASM runtime once
    if (!_PoseLandmarker || !_FilesetResolver) {
      const mod = await import(`${MEDIAPIPE_CDN}/vision_bundle.mjs`);
      _PoseLandmarker = mod.PoseLandmarker;
      _FilesetResolver = mod.FilesetResolver;
    }

    const vision = await _FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);

    let landmarker;
    try {
      landmarker = await _PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_PATHS[key],
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
    } catch (gpuErr) {
      console.warn('[Pose] GPU delegate failed, falling back to CPU:', gpuErr.message);
      landmarker = await _PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_PATHS[key],
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
    }

    _landmarkerCache[key] = landmarker;
    _currentModel = key;
    return landmarker;
  })();

  try {
    const result = await _loadingPromise;
    return result;
  } catch (e) {
    _loadingPromise = null;
    throw e;
  }
}

/**
 * Return the currently selected model name.
 */
export function getCurrentModel() {
  return _currentModel || 'full';
}


// ── Real-time single-frame detection ──────────────────────────────────

/**
 * Run pose detection on a single canvas/video element for real-time overlay.
 * Returns normalized + world landmarks or null.
 *
 * @param {HTMLCanvasElement|HTMLVideoElement} source
 * @param {number} timestampMs — monotonically increasing ms
 * @param {'full'|'lite'|'heavy'} model
 * @returns {{ normLm: Array, worldLm: Array } | null}
 */
export async function detectLive(source, timestampMs, model = 'lite') {
  const landmarker = await ensurePoseLandmarker(model);
  const result = _detectForVideoSafe(landmarker, source, timestampMs);
  if (!result.landmarks?.length || !result.worldLandmarks?.length) return null;
  return { normLm: result.landmarks[0], worldLm: result.worldLandmarks[0] };
}


// ── Frame extraction ──────────────────────────────────────────────────

/**
 * Extract frames from a video at the target FPS.
 * Uses playback-based extraction with requestVideoFrameCallback
 * for maximum speed, falling back to seek-based for compatibility.
 *
 * @param {HTMLVideoElement} videoEl — loaded video element
 * @param {number} fps — target extraction FPS
 * @param {(frame: {canvas, ts_s, frameIdx}) => Promise<void>} onFrame
 * @param {(pct: number) => void} onProgress
 * @param {{ cancelled: boolean }} cancelToken
 */
async function extractFrames(videoEl, fps, onFrame, onProgress, cancelToken, startSec = 0, endSec = Infinity, opts = {}) {
  const duration = videoEl.duration;
  if (!isFinite(duration) || duration <= 0) throw new Error('Video has no duration');

  const effectiveStart = Math.max(0, startSec);
  const effectiveEnd = Math.min(duration, isFinite(endSec) ? endSec : duration);
  const effectiveDuration = effectiveEnd - effectiveStart;
  if (effectiveDuration <= 0) throw new Error('Empty time range');

  const step = 1.0 / fps;
  const totalFrames = Math.ceil(effectiveDuration * fps);

  // Create offscreen canvas for frame capture (MediaPipe path) - use smaller
  // size for speed. Angle models can also request a frozen full-res canvas.
  const configuredMaxDim = Number(opts?.poseInputMaxDim);
  const maxDim = Number.isFinite(configuredMaxDim)
    ? Math.max(240, Math.min(960, Math.round(configuredMaxDim)))
    : 480;
  let cw = videoEl.videoWidth;
  let ch = videoEl.videoHeight;
  if (Math.max(cw, ch) > maxDim) {
    const scale = maxDim / Math.max(cw, ch);
    cw = Math.round(cw * scale);
    ch = Math.round(ch * scale);
  }

  const canvas = document.createElement('canvas');
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true, willReadFrequently: false });
  const fullFramePredicate = typeof opts?.captureFullResolutionFrame === 'function'
    ? opts.captureFullResolutionFrame
    : (opts?.captureFullResolutionFrame ? () => true : null);
  const captureFullResolutionFrame = Boolean(fullFramePredicate);
  let fullCanvas = null;
  let fullCtx = null;
  if (captureFullResolutionFrame) {
    fullCanvas = document.createElement('canvas');
    fullCanvas.width = videoEl.videoWidth;
    fullCanvas.height = videoEl.videoHeight;
    fullCtx = fullCanvas.getContext('2d', { willReadFrequently: false });
    if (fullCtx) {
      fullCtx.imageSmoothingEnabled = false;
    } else {
      fullCanvas = null;
    }
  }

  // Try playback-based extraction (faster — avoids seek overhead)
  const hasVFRC = typeof videoEl.requestVideoFrameCallback === 'function';
  const preferSeek = Boolean(opts?.preferSeek);
  const canUsePlayback = hasVFRC && !_isDocumentHidden() && !preferSeek;

  if (canUsePlayback) {
    const playbackResult = await _extractViaPlayback(
      videoEl, fps, step, totalFrames, effectiveEnd,
      canvas, ctx, fullCanvas, fullCtx, fullFramePredicate, cw, ch, onFrame, onProgress, cancelToken, effectiveStart
    );
    if (!cancelToken.cancelled && playbackResult && playbackResult.completed === false) {
      await _extractViaSeek(
        videoEl, fps, step, totalFrames, effectiveEnd,
        canvas, ctx, fullCanvas, fullCtx, fullFramePredicate, cw, ch, onFrame, onProgress, cancelToken,
        playbackResult.nextTargetTime, playbackResult.frameIdx
      );
    }
  } else {
    await _extractViaSeek(videoEl, fps, step, totalFrames, effectiveEnd, canvas, ctx, fullCanvas, fullCtx, fullFramePredicate, cw, ch, onFrame, onProgress, cancelToken, effectiveStart, 0);
  }
}

function _isDocumentHidden() {
  return typeof document !== 'undefined' && document.visibilityState === 'hidden';
}

function _waitForVideoSeek(videoEl, targetTime, timeoutMs = SEEK_WAIT_VISIBLE_MS) {
  return new Promise((resolve) => {
    let settled = false;
    const target = Number(targetTime);
    const withinTarget = () => {
      const current = Number(videoEl?.currentTime);
      return !videoEl?.seeking && Number.isFinite(current) && Math.abs(current - target) <= 0.08;
    };
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(Boolean(ok));
    };
    const onReady = () => {
      if (withinTarget()) finish(true);
    };
    const timer = setTimeout(() => finish(withinTarget()), timeoutMs);
    const cleanup = () => {
      clearTimeout(timer);
      try { videoEl?.removeEventListener('seeked', onReady); } catch {}
      try { videoEl?.removeEventListener('timeupdate', onReady); } catch {}
      try { videoEl?.removeEventListener('canplay', onReady); } catch {}
    };

    try { videoEl?.addEventListener('seeked', onReady); } catch {}
    try { videoEl?.addEventListener('timeupdate', onReady); } catch {}
    try { videoEl?.addEventListener('canplay', onReady); } catch {}
    if (withinTarget()) finish(true);
  });
}

/**
 * Fast: play video at max speed, sample frames via requestVideoFrameCallback.
 */
async function _extractViaPlayback(videoEl, fps, step, totalFrames, endTime, canvas, ctx, fullCanvas, fullCtx, fullFramePredicate, cw, ch, onFrame, onProgress, cancelToken, startTime = 0) {
  return new Promise((resolve, reject) => {
    let nextTargetTime = startTime;
    let frameIdx = 0;
    let processing = false;
    const playbackRate = Math.min(16, Math.max(1, Math.floor(30 / fps)));
    const rangeDuration = endTime - startTime;
    let settled = false;
    let stallTimer = null;

    const finish = (result) => {
      if (settled) return;
      settled = true;
      cleanup();
      videoEl.pause();
      resolve(result);
    };
    const fail = (err) => {
      if (settled) return;
      settled = true;
      cleanup();
      videoEl.pause();
      reject(err);
    };
    const armStallFallback = () => {
      if (stallTimer) clearTimeout(stallTimer);
      stallTimer = setTimeout(() => {
        if (settled || cancelToken.cancelled) return;
        const mediaTime = Number(videoEl.currentTime);
        const done = (Number.isFinite(mediaTime) && mediaTime >= endTime - 0.05) || nextTargetTime >= endTime - 1e-6;
        if (done) {
          finish({ completed: true, nextTargetTime: endTime, frameIdx });
          return;
        }
        console.info(`[Pose] Playback extraction stalled${_isDocumentHidden() ? ' in background tab' : ''}; resuming via seek mode at ${nextTargetTime.toFixed(2)}s`);
        finish({
          completed: false,
          nextTargetTime: Math.max(startTime, Math.min(endTime, nextTargetTime)),
          frameIdx,
        });
      }, PLAYBACK_STALL_FALLBACK_MS);
    };
    const onVisibilityChange = () => {
      if (_isDocumentHidden()) armStallFallback();
    };
    const cleanup = () => {
      if (stallTimer) clearTimeout(stallTimer);
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', onVisibilityChange);
      }
    };

    videoEl.currentTime = startTime;
    videoEl.muted = true;
    videoEl.playbackRate = playbackRate;
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisibilityChange);
    }

    function onVideoFrame(now, metadata) {
      if (cancelToken.cancelled) {
        finish({ completed: true, nextTargetTime, frameIdx });
        return;
      }
      armStallFallback();
      const currentTime = metadata.mediaTime;

      if (currentTime >= nextTargetTime && !processing) {
        processing = true;
        const idx = frameIdx++;
        const ts_s = currentTime;
        ctx.drawImage(videoEl, 0, 0, cw, ch);
        if (fullCtx && fullCanvas && fullFramePredicate?.({ frameIdx: idx, ts_s })) {
          fullCtx.drawImage(videoEl, 0, 0, fullCanvas.width, fullCanvas.height);
        }
        nextTargetTime = startTime + idx * step + step;

        onFrame({ canvas, fullCanvas, ts_s, frameIdx: idx, width: cw, height: ch }).then(() => {
          processing = false;
          if (onProgress) onProgress(Math.min(1, (currentTime - startTime) / rangeDuration));
        }).catch((err) => {
          processing = false;
          fail(err);
        });
      }

      if (currentTime < endTime - 0.05 && !cancelToken.cancelled) {
        videoEl.requestVideoFrameCallback(onVideoFrame);
      } else {
        finish({ completed: true, nextTargetTime: endTime, frameIdx });
      }
    }

    armStallFallback();
    videoEl.requestVideoFrameCallback(onVideoFrame);
    videoEl.play().catch(fail);
  });
}

/**
 * Fallback: seek to each target time (slower but universal).
 */
async function _extractViaSeek(videoEl, fps, step, totalFrames, endTime, canvas, ctx, fullCanvas, fullCtx, fullFramePredicate, cw, ch, onFrame, onProgress, cancelToken, startTime = 0, startFrameIdx = 0) {
  let frameIdx = Math.max(0, Number(startFrameIdx) || 0);
  for (let t = startTime; t < endTime; t += step) {
    if (cancelToken.cancelled) break;

    videoEl.currentTime = t;
    const seekOk = await _waitForVideoSeek(
      videoEl,
      t,
      _isDocumentHidden() ? SEEK_WAIT_HIDDEN_MS : SEEK_WAIT_VISIBLE_MS
    );
    if (!seekOk) {
      console.warn(`[Pose] Seek extraction timed out near ${t.toFixed(2)}s; skipping frame`);
      frameIdx++;
      if (onProgress) onProgress(Math.min(1, frameIdx / totalFrames));
      continue;
    }

    ctx.drawImage(videoEl, 0, 0, cw, ch);
    if (fullCtx && fullCanvas && fullFramePredicate?.({ frameIdx, ts_s: t })) {
      fullCtx.drawImage(videoEl, 0, 0, fullCanvas.width, fullCanvas.height);
    }
    await onFrame({ canvas, fullCanvas, ts_s: t, frameIdx, width: cw, height: ch });

    frameIdx++;
    if (onProgress) onProgress(frameIdx / totalFrames);

    // Yield every 4 frames to reduce overhead
    if (frameIdx % 4 === 0) await new Promise(r => setTimeout(r, 0));
  }
}


// ── Main processing pipeline ──────────────────────────────────────────

/** Running state for cancellation */
const _activeRuns = new Map(); // fileId → { cancelled: boolean }

/** Live skeleton frame listeners — for 3D viewer updates during processing */
const _frameListeners = new Map(); // fileId → Set<(frame) => void>
const JSONL_APPEND_BATCH_LINES = 24;
const AUTOPNP_PREPASS_CACHE_LIMIT = 64;
const _autoPnpPrepassCache = new Map();

function _roundRangeKey(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(3) : 'inf';
}

function _buildAutoPnpPrepassCacheKey({
  projectId,
  fileId,
  calibrationVariant,
  effectiveStart,
  effectiveEnd,
  videoDuration,
  autoPnpCfg,
}) {
  const cfgKey = [
    Number(autoPnpCfg?.avg_frames) || '',
    Number(autoPnpCfg?.min_valid_frames) || '',
    Number(autoPnpCfg?.sample_frames ?? autoPnpCfg?.max_sample_frames) || '',
  ].join(':');
  return [
    projectId,
    fileId,
    calibrationVariant,
    _roundRangeKey(effectiveStart),
    _roundRangeKey(effectiveEnd),
    _roundRangeKey(videoDuration),
    cfgKey,
  ].join('|');
}

function _getCachedAutoPnpPrepass(key) {
  if (!_autoPnpPrepassCache.has(key)) return null;
  const value = _autoPnpPrepassCache.get(key);
  _autoPnpPrepassCache.delete(key);
  _autoPnpPrepassCache.set(key, value);
  return value;
}

function _setCachedAutoPnpPrepass(key, value) {
  _autoPnpPrepassCache.set(key, value);
  while (_autoPnpPrepassCache.size > AUTOPNP_PREPASS_CACHE_LIMIT) {
    const firstKey = _autoPnpPrepassCache.keys().next().value;
    _autoPnpPrepassCache.delete(firstKey);
  }
}

function _applyAutoPnpSnapshot(snapshot, R_wc, camPos) {
  if (!snapshot || !Array.isArray(snapshot.R_wc) || !Array.isArray(snapshot.camPos)) return null;
  for (let j = 0; j < 9; j++) R_wc[j] = snapshot.R_wc[j];
  camPos[0] = snapshot.camPos[0];
  camPos[1] = snapshot.camPos[1];
  camPos[2] = snapshot.camPos[2];
  return Number.isFinite(Number(snapshot.cameraYawDeg)) ? Number(snapshot.cameraYawDeg) : null;
}

/**
 * Subscribe to live skeleton frames emitted during processing.
 * @param {string} fileId
 * @param {(frame: {ts, skeleton}) => void} callback
 * @returns {() => void} unsubscribe function
 */
export function onLiveFrame(fileId, callback) {
  if (!_frameListeners.has(fileId)) _frameListeners.set(fileId, new Set());
  _frameListeners.get(fileId).add(callback);
  return () => { _frameListeners.get(fileId)?.delete(callback); };
}

function _emitFrame(fileId, frame) {
  const listeners = _frameListeners.get(fileId);
  if (listeners) for (const cb of listeners) { try { cb(frame); } catch {} }
}

function _createJsonlBatchWriter(path, flushEvery = JSONL_APPEND_BATCH_LINES) {
  const normalizedPath = String(path || '').trim();
  const threshold = Math.max(1, Math.round(Number(flushEvery) || JSONL_APPEND_BATCH_LINES));
  const pending = [];

  async function flushPending(force = false) {
    if (!pending.length) return;
    if (!force && pending.length < threshold) return;
    const chunk = pending.join('\n');
    pending.length = 0;
    await Storage.appendLine(normalizedPath, chunk);
  }

  return {
    async push(line) {
      pending.push(String(line ?? ''));
      if (pending.length >= threshold) await flushPending(true);
    },
    async flush() {
      await flushPending(true);
    },
  };
}

function _splitOpfsPath(path) {
  const parts = String(path || '').replace(/\\/g, '/').split('/').filter(Boolean);
  const filename = parts.pop() || '';
  return { dirParts: parts, filename };
}

function _tsKey(ts) {
  const t = Number(ts);
  if (!Number.isFinite(t)) return null;
  return Math.round(t * 1000); // millisecond bucket
}

function _skeletonPointCount(obj) {
  if (!obj || !obj.skeleton || typeof obj.skeleton !== 'object') return 0;
  return Object.keys(obj.skeleton).length;
}

function _recordScore(obj) {
  if (!obj || typeof obj !== 'object') return 0;
  let score = 0;
  if (obj.detected === true) score += 10;
  const skelPts = _skeletonPointCount(obj);
  if (skelPts > 0) score += 100 + skelPts;
  return score;
}

function _normalizeRecordsByTs(records) {
  const bestByTs = new Map(); // tsKey -> { obj, score, skelPts }
  for (const obj of records || []) {
    const key = _tsKey(obj?.ts);
    if (key == null) continue;
    const score = _recordScore(obj);
    const skelPts = _skeletonPointCount(obj);
    const prev = bestByTs.get(key);
    if (!prev || score > prev.score || (score === prev.score && skelPts >= prev.skelPts)) {
      bestByTs.set(key, { obj, score, skelPts });
    }
  }
  return [...bestByTs.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, rec]) => rec.obj);
}

function _isStorageWriteError(err) {
  const msg = String(err?.message || err || '').toLowerCase();
  return (
    msg.includes('storage') ||
    msg.includes('quota') ||
    msg.includes('indexeddb') ||
    msg.includes('opfs') ||
    msg.includes('appendline') ||
    msg.includes('writefile')
  );
}

function _normalizeRangeList(ranges, tolerance = 0.35) {
  const ordered = (ranges || [])
    .map((range) => Array.isArray(range) ? range : [range?.start, range?.end])
    .map(([start, end]) => [Number(start), Number(end)])
    .filter(([start, end]) => Number.isFinite(start) && Number.isFinite(end))
    .map(([start, end]) => [Math.max(0, start), Math.max(Math.max(0, start), end)])
    .sort((a, b) => a[0] - b[0]);

  if (!ordered.length) return [];
  const merged = [ordered[0]];
  for (let i = 1; i < ordered.length; i++) {
    const [start, end] = ordered[i];
    const last = merged[merged.length - 1];
    if (start <= last[1] + tolerance) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

function _clipRangeList(ranges, clipStart, clipEnd) {
  const start = Number(clipStart);
  const end = Number(clipEnd);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return [];
  const clipped = [];
  for (const range of ranges || []) {
    const pair = Array.isArray(range) ? range : [range?.start, range?.end];
    const lo = Number(pair[0]);
    const hi = Number(pair[1]);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) continue;
    const nextStart = Math.max(start, lo);
    const nextEnd = Math.min(end, hi);
    if (nextEnd >= nextStart) clipped.push([nextStart, nextEnd]);
  }
  return _normalizeRangeList(clipped);
}

function _rangeListCovers(ranges, startSec, endSec, tolerance = 0.75) {
  const start = Number(startSec);
  const end = Number(endSec);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return false;
  const ordered = _normalizeRangeList(ranges, tolerance);
  if (!ordered.length) return false;
  let coveredUntil = start;
  for (const [segStart, segEnd] of ordered) {
    if (segEnd < coveredUntil - tolerance) continue;
    if (segStart > coveredUntil + tolerance) return false;
    coveredUntil = Math.max(coveredUntil, segEnd);
    if (coveredUntil >= end - tolerance) return true;
  }
  return coveredUntil >= end - tolerance;
}

function _rangeListCoveredDuration(ranges, startSec, endSec, tolerance = 0.75) {
  const start = Number(startSec);
  const end = Number(endSec);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return 0;
  const ordered = _normalizeRangeList(ranges, tolerance)
    .map(([segStart, segEnd]) => [segStart - tolerance, segEnd + tolerance]);
  let coveredUntil = start;
  let covered = 0;
  for (const [segStart, segEnd] of ordered) {
    const clippedStart = Math.max(start, segStart);
    const clippedEnd = Math.min(end, segEnd);
    if (clippedEnd <= coveredUntil) continue;
    if (clippedStart > coveredUntil) coveredUntil = clippedStart;
    if (clippedEnd > coveredUntil) {
      covered += clippedEnd - coveredUntil;
      coveredUntil = clippedEnd;
      if (coveredUntil >= end) break;
    }
  }
  return Math.max(0, Math.min(end - start, covered));
}

function _rangeListCoverageRatio(ranges, startSec, endSec, tolerance = 0.75) {
  const start = Number(startSec);
  const end = Number(endSec);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return 0;
  const span = Math.max(0, end - start);
  if (span <= 1e-6) return _rangeListCovers(ranges, start, end, tolerance) ? 1 : 0;
  return _rangeListCoveredDuration(ranges, start, end, tolerance) / span;
}

function _median(values) {
  if (!Array.isArray(values) || values.length === 0) return null;
  const arr = values.filter(v => Number.isFinite(Number(v))).map(Number).sort((a, b) => a - b);
  if (!arr.length) return null;
  const mid = Math.floor(arr.length / 2);
  return arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
}

function _clip(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function _wrapAngle180(angleDeg) {
  const angle = Number(angleDeg);
  if (!Number.isFinite(angle)) return null;
  return ((angle + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
}

function _mapLegacyRudderAngleToSigned(angleDeg) {
  const angle = Number(angleDeg);
  if (!Number.isFinite(angle)) return null;
  return 90.0 - angle;
}

function _mapLegacyBoomAzimuthToSigned(angleDeg) {
  const angle = Number(angleDeg);
  if (!Number.isFinite(angle)) return null;
  return _wrapAngle180(angle - 180.0);
}

function _mapLegacyBoomCameraCenteredToMinusX(angleDeg) {
  const angle = Number(angleDeg);
  if (!Number.isFinite(angle)) return null;
  return _wrapAngle180(angle + 90.0);
}

function _mapLegacyBoomXCenteredToMinusX(angleDeg) {
  const angle = Number(angleDeg);
  if (!Number.isFinite(angle)) return null;
  return _wrapAngle180(angle - 180.0);
}

export function normalizeMetricRudderConvention(row) {
  if (!row || typeof row !== 'object') return row;

  let out = row;

  if (row?.rudder_angle_system !== 'signed_centered_v1') {
    const rudder = _mapLegacyRudderAngleToSigned(row?.rudder_angle);
    const rudderRaw = _mapLegacyRudderAngleToSigned(row?.rudder_angle_raw);
    out = {
      ...out,
      rudder_angle: rudder ?? row?.rudder_angle ?? null,
      rudder_angle_raw: rudderRaw ?? row?.rudder_angle_raw ?? null,
      rudder_angle_system: 'signed_centered_v1',
    };
  }

  if (row?.boom_angle_system !== 'boom_minus_x_centered_v3') {
    const legacyCameraCentered = row?.boom_angle_system === 'signed_centered_v1';
    const legacyXCentered = row?.boom_angle_system === 'boom_x_centered_v2';
    const boom = legacyCameraCentered
      ? _mapLegacyBoomCameraCenteredToMinusX(row?.boom_angle)
      : (legacyXCentered
        ? _mapLegacyBoomXCenteredToMinusX(row?.boom_angle)
        : _mapLegacyBoomAzimuthToSigned(row?.boom_angle));
    const boomRaw = legacyCameraCentered
      ? _mapLegacyBoomCameraCenteredToMinusX(row?.boom_angle_raw)
      : (legacyXCentered
        ? _mapLegacyBoomXCenteredToMinusX(row?.boom_angle_raw)
        : _mapLegacyBoomAzimuthToSigned(row?.boom_angle_raw));
    out = {
      ...out,
      boom_angle: boom ?? row?.boom_angle ?? null,
      boom_angle_raw: boomRaw ?? row?.boom_angle_raw ?? null,
      boom_angle_system: 'boom_minus_x_centered_v3',
    };
  }

  return out;
}

function _filterRudderAngle(rawAngle, filterState) {
  const raw = Number(rawAngle);
  if (!Number.isFinite(raw)) return { value: null, outlier: false };

  const st = filterState || { recent: [], ema: null };
  st.recent.push(raw);
  if (st.recent.length > 4) st.recent.shift();

  let candidate = raw;
  let outlier = false;
  if (st.recent.length >= 4) {
    const med = _median(st.recent);
    if (med != null) {
      const dev = st.recent.map(v => Math.abs(v - med));
      const mad = _median(dev) ?? 0;
      const robustSigma = 1.4826 * mad;
      // Keep spike protection, but let real steering inputs move through quickly.
      const thresh = Math.max(32.0, 7.0 * robustSigma);
      if (Math.abs(raw - med) > thresh) {
        outlier = true;
        candidate = med + Math.sign(raw - med) * thresh;
      }
    }
  }

  const alpha = 0.10; // keep only a light memory of the previous filtered value
  if (!Number.isFinite(st.ema)) st.ema = candidate;
  else st.ema = alpha * st.ema + (1.0 - alpha) * candidate;

  const value = _clip(st.ema, 0.0, 180.0);
  return { value, outlier };
}

async function _normalizeJsonlFileByTs(path) {
  const text = await Storage.readFileText(path);
  if (!text) return 0;
  const rows = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      rows.push(JSON.parse(trimmed));
    } catch {}
  }
  if (rows.length === 0) return 0;
  const normalized = _normalizeRecordsByTs(rows);
  const out = normalized.map(r => JSON.stringify(r)).join('\n');
  const { dirParts, filename } = _splitOpfsPath(path);
  await Storage.writeFile(dirParts, filename, out ? `${out}\n` : '');
  return normalized.length;
}

/**
 * Get info about all currently running jobs.
 * @returns {Array<{fileId, progress, message, startTime}>}
 */
export function getActiveJobs() {
  const jobs = [];
  for (const [fileId, token] of _activeRuns) {
    jobs.push({
      fileId,
      progress: token.progress || 0,
      message: token.message || 'Processing...',
      startTime: token.startTime || Date.now(),
      cancelled: token.cancelled,
      segmentName: token.segmentName || null,
      startSec: Number.isFinite(Number(token.startSec)) ? Number(token.startSec) : null,
      endSec: Number.isFinite(Number(token.endSec)) ? Number(token.endSec) : null,
    });
  }
  return jobs;
}

/**
 * Process a video file: MediaPipe pose → placement → Kalman → metrics.
 * Results are streamed to OPFS as JSONL.
 *
 * @param {string} projectId
 * @param {string} fileId
 * @param {Object} cvConfig — project CV config (or DEFAULT_CV_CONFIG)
 * @param {(msg: string, pct: number) => void} onProgress
 * @param {{ fps?: number, model?: 'full'|'lite'|'heavy', athleteWeight?: number, athleteHeight?: number }} opts — optional overrides
 * @returns {{ frameCount: number, outputPath: string }}
 */
export async function processVideo(projectId, fileId, cvConfig = null, onProgress = null, opts = {}) {
  // Accept either a bare config object or a DB cvConfig row ({project_id, config,
  // updated_at}); unwrap the latter so per-project settings actually take effect.
  const rawCfg = (cvConfig && typeof cvConfig === 'object' && cvConfig.config && typeof cvConfig.config === 'object')
    ? cvConfig.config
    : cvConfig;
  const cfg = { ...DEFAULT_CV_CONFIG, ...(rawCfg || {}) };
  const fileRec = await DB.getFile(fileId);
  const segmentName = typeof opts.segmentName === 'string' ? opts.segmentName.trim() : '';
  const startSec = Number.isFinite(Number(opts.startSec)) ? Math.max(0, Number(opts.startSec)) : null;
  const endSec = Number.isFinite(Number(opts.endSec)) ? Math.max(0, Number(opts.endSec)) : null;
  const cancelToken = {
    cancelled: false,
    progress: 0,
    message: 'Starting...',
    startTime: Date.now(),
    segmentName: segmentName || null,
    startSec,
    endSec,
  };
  const report = (msg, pct) => {
    if (typeof msg === 'string' && msg.trim()) cancelToken.message = msg;
    const pctNum = Number(pct);
    if (Number.isFinite(pctNum)) cancelToken.progress = Math.max(0, Math.min(1, pctNum));
    if (onProgress) onProgress(msg, pct);
  };
  _activeRuns.set(fileId, cancelToken);

  try {
    // 1. Load MediaPipe
    const model = opts.model || 'full';
    report(`Loading MediaPipe (${model})...`, 0);
    const landmarker = await ensurePoseLandmarker(model);

    // 2. Load calibration & compute camera pose
    const cameraText = [
      fileRec?.device_make,
      fileRec?.device_model,
      cfg.camera_model,
    ].filter(Boolean).join(' ');
    const calibrationVariant = isGoPro13Model(cameraText) ? 'gopro13' : 'default';
    report(calibrationVariant === 'gopro13' ? 'Loading GoPro 13 calibration...' : 'Loading calibration...', 0.02);
    const calib = await getCalibration(calibrationVariant);
    // Flatten K to row-major 9-element array
    const K = [
      calib.K[0][0], calib.K[0][1], calib.K[0][2],
      calib.K[1][0], calib.K[1][1], calib.K[1][2],
      calib.K[2][0], calib.K[2][1], calib.K[2][2],
    ];
    const imgW = calib.img_size[0];
    const imgH = calib.img_size[1];

    const { pos: camPos, R_wc } = defaultCameraPoseAndRotation(
      cfg.camera_pitch_deg, cfg.camera_yaw_deg, cfg.camera_roll_deg
    );
    if (cfg.camera_position) {
      camPos[0] = cfg.camera_position[0];
      camPos[1] = cfg.camera_position[1];
      camPos[2] = cfg.camera_position[2];
    }

    // Manual keypoint-calibrated pose: a fixed camera pose solved from the user's
    // hand-corrected boat keypoints. The camera is rigidly mounted, so one pose
    // applies to the whole clip — lock it in and skip Auto-PnP entirely.
    const manualPose = cfg.manual_camera_pose;
    const manualPoseLocked = !!(manualPose && Array.isArray(manualPose.camPos) && Array.isArray(manualPose.R_wc));
    if (manualPoseLocked) {
      camPos[0] = Number(manualPose.camPos[0]);
      camPos[1] = Number(manualPose.camPos[1]);
      camPos[2] = Number(manualPose.camPos[2]);
      // R_wc may be a nested 3×3 or a flat 9 array — flatten to row-major 9.
      const m = manualPose.R_wc;
      const flat = (Array.isArray(m[0])) ? [m[0][0],m[0][1],m[0][2], m[1][0],m[1][1],m[1][2], m[2][0],m[2][1],m[2][2]] : m;
      for (let j = 0; j < 9; j++) R_wc[j] = Number(flat[j]);
      console.log(`[Pose] using MANUAL keypoint-calibrated camera pose: camPos=[${camPos.map(v=>v.toFixed(3))}] (Auto-PnP disabled)`);
    }

    // Auto-PnP config — force-disabled when a manual pose is locked in.
    const autoPnpCfg = cfg.auto_camera_pnp || {};
    const autoPnpEnabled = !manualPoseLocked && autoPnpCfg.enabled !== false;
    const autoPnpInterval = Math.max(1, Math.round(Number(autoPnpCfg.interval_frames) || 30)); // every N frames
    let autoPnpReady = false;
    let cameraYawDeg = Number(cfg.camera_yaw_deg);
    if (!Number.isFinite(cameraYawDeg)) cameraYawDeg = 0;
    console.log(`[Pose] Auto-PnP config: enabled=${autoPnpEnabled}, interval=${autoPnpInterval}`);
    if (autoPnpEnabled) {
      try {
        report('Loading YOLO model for Auto-PnP...', 0.03);
        autoPnpReady = await AutoPnP.preload();
        console.log(`[Pose] Auto-PnP preload: ${autoPnpReady ? 'READY' : 'NOT READY (preload returned false)'}`);
      } catch (e) {
        console.warn('[Pose] Auto-PnP preload FAILED, continuing without it:', e);
      }
    } else {
      console.log('[Pose] Auto-PnP is DISABLED in config');
    }

    let rudderReady = false;
    let boomReady = false;
    const rudderPredictionEnabled = opts?.enableRudderPrediction !== false;
    const boomPredictionEnabled = opts?.enableBoomPrediction !== false;
    report('Loading angle models...', 0.04);
    const [rudderLoadResult, boomLoadResult] = await Promise.all([
      (async () => {
        if (!rudderPredictionEnabled) {
          console.log('[Pose] Rudder predictions disabled by settings');
          return false;
        }
        try {
          const ready = await Rudder.preload();
          if (ready) {
            console.log('[Pose] Rudder model preload: READY', Rudder.getModelInfo());
          } else {
            console.warn('[Pose] Rudder model preload: NOT READY');
          }
          return !!ready;
        } catch (e) {
          console.warn('[Pose] Rudder model preload FAILED, continuing without rudder predictions:', e);
          return false;
        }
      })(),
      (async () => {
        if (!boomPredictionEnabled) {
          console.log('[Pose] Boom predictions disabled by settings');
          return false;
        }
        try {
          const ready = await Boom.preload();
          if (ready) {
            console.log('[Pose] Boom model preload: READY', Boom.getModelInfo());
          } else {
            console.warn('[Pose] Boom model preload: NOT READY');
          }
          return !!ready;
        } catch (e) {
          console.warn('[Pose] Boom model preload FAILED, continuing without boom predictions:', e);
          return false;
        }
      })(),
    ]);
    rudderReady = rudderLoadResult;
    boomReady = boomLoadResult;
    if (boomReady && typeof Boom.resetSmoothing === 'function') {
      Boom.resetSmoothing();
    }

    const zHip = cfg.hip_plane_z ?? SKELETON_HIP_PLANE_Z;
    const zAnkle = cfg.lower_plane_z ?? SKELETON_LOWER_PLANE_Z;
    const athleteMass = Number.isFinite(Number(opts.athleteWeight))
      ? Number(opts.athleteWeight)
      : (cfg.athlete_weight ?? 75);
    const athleteHeight = Number.isFinite(Number(opts.athleteHeight))
      ? Number(opts.athleteHeight)
      : (Number.isFinite(Number(cfg.athlete_height)) ? Number(cfg.athlete_height) : null);
    const boatCom = [typeof cfg.boat_com === 'number' ? cfg.boat_com : -1.114, 0, 0];

    // 3. Create video element for frame extraction
    report('Loading video...', 0.05);
    const videoUrl = await FM.createVideoURL(fileId);
    if (!videoUrl) throw new Error('Video file not accessible — please re-pick the file');

    const videoEl = document.createElement('video');
    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.preload = 'auto';
    videoEl.src = videoUrl;

    await new Promise((resolve, reject) => {
      videoEl.onloadedmetadata = resolve;
      videoEl.onerror = () => reject(new Error('Failed to load video'));
      setTimeout(() => reject(new Error('Video load timeout')), 30000);
    });

    const videoDuration = videoEl.duration;
    const fps = (opts.fps && opts.fps >= 1 && opts.fps <= 30) ? opts.fps : SKELETON_TARGET_FPS;

    // 4. Set up Kalman filter
    const kalmanParams = { ...KALMAN_DEFAULTS, ...(cfg.skeleton_filter || {}) };
    const smoother = new SkeletonPlacementKalman(kalmanParams);
    const dt = 1.0 / fps;

    // 5. Prepare OPFS output path
    const outputDir = `${projectId}/${fileId}`;
    const skeletonPath = `${outputDir}/skeleton.jsonl`;
    const metricsPath = `${outputDir}/metrics.jsonl`;
    const skeletonWriter = _createJsonlBatchWriter(skeletonPath);
    const metricsWriter = _createJsonlBatchWriter(metricsPath);

    const rangeStart = opts.startSec ?? 0;
    const rangeEnd = opts.endSec ?? Infinity;
    const effectiveStart = Math.max(0, rangeStart);
    const effectiveEnd = Math.min(videoDuration, isFinite(rangeEnd) ? rangeEnd : videoDuration);
    const isSegmentRun = effectiveStart > 1e-6 || effectiveEnd < (videoDuration - 1e-6);

    // Full runs replace previous output; segmented runs are merged in-place.
    if (!isSegmentRun) {
      try { await Storage.deleteFile(skeletonPath); } catch {}
      try { await Storage.deleteFile(metricsPath); } catch {}
    }

    // ── Auto-PnP pre-pass: calibrate camera before processing ─────────

    let autoPnpPrepassApplied = false;
    if (autoPnpReady && !cancelToken.cancelled) {
      const prepassCacheKey = _buildAutoPnpPrepassCacheKey({
        projectId,
        fileId,
        calibrationVariant,
        effectiveStart,
        effectiveEnd,
        videoDuration,
        autoPnpCfg,
      });
      const cachedPrepass = _getCachedAutoPnpPrepass(prepassCacheKey);

      if (cachedPrepass?.status === 'ok') {
        const cachedYaw = _applyAutoPnpSnapshot(cachedPrepass, R_wc, camPos);
        if (cachedYaw != null) cameraYawDeg = cachedYaw;
        autoPnpPrepassApplied = true;
        console.log(`[Pose] Auto-PnP pre-pass cache hit: pos=[${camPos.map(v => v.toFixed(3))}], yaw=${cameraYawDeg.toFixed(1)}`);
      } else if (cachedPrepass?.status === 'failed') {
        console.log('[Pose] Auto-PnP pre-pass cache hit: previous range had too few valid frames; using default camera pose');
      } else {
        report('Calibrating camera from boat keypoints...', 0.06);
        const minValidFrames = Math.max(1, Math.round(Number(autoPnpCfg.min_valid_frames) || 5));
        const targetValid = Math.max(1, Math.min(20, minValidFrames));
        const configuredSampleCount = Number(autoPnpCfg.sample_frames ?? autoPnpCfg.max_sample_frames);
        const sampleCount = Math.max(
          targetValid,
          Math.min(30, Math.round(Number.isFinite(configuredSampleCount) && configuredSampleCount > 0 ? configuredSampleCount : targetValid * 4)),
        );
        const sampleStep = (effectiveEnd - effectiveStart) / (sampleCount + 1);
        const validPoses = [];

        for (let i = 1; i <= sampleCount && validPoses.length < targetValid; i++) {
          if (cancelToken.cancelled) break;
          const t = effectiveStart + i * sampleStep;
          videoEl.currentTime = t;
          const seekOk = await _waitForVideoSeek(
            videoEl,
            t,
            _isDocumentHidden() ? SEEK_WAIT_HIDDEN_MS : SEEK_WAIT_VISIBLE_MS,
          );
          if (!seekOk) {
            console.warn(`[AutoPnP pre-pass] seek timed out near ${t.toFixed(1)}s; sampling current frame`);
          }

          try {
            const result = await AutoPnP.estimateCameraPose(videoEl);
            if (result) {
              validPoses.push(result);
              console.log(`[AutoPnP pre-pass] ${i}/${sampleCount} at ${t.toFixed(1)}s: VALID (fullres-video) pos=[${result.camPos.map(v=>v.toFixed(3))}] err=${result.meanErrorPx.toFixed(1)}px (${validPoses.length}/${targetValid})`);
            } else {
              console.log(`[AutoPnP pre-pass] ${i}/${sampleCount} at ${t.toFixed(1)}s: rejected`);
            }
          } catch (e) {
            console.warn(`[AutoPnP pre-pass] ${i}/${sampleCount}: error`, e.message);
          }
          report(`Calibrating camera... (${validPoses.length}/${targetValid} valid)`, 0.06 + 0.015 * (i / sampleCount));
        }

        if (validPoses.length >= Math.min(3, targetValid)) {
          const _median = arr => {
            const s = [...arr].sort((a, b) => a - b);
            const m = Math.floor(s.length / 2);
            return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
          };
          const medPitch = _median(validPoses.map(r => r.angles.pitch_deg));
          const medYaw   = _median(validPoses.map(r => r.angles.yaw_deg));
          const medRoll  = _median(validPoses.map(r => r.angles.roll_deg));
          const medX = _median(validPoses.map(r => r.camPos[0]));
          const medY = _median(validPoses.map(r => r.camPos[1]));
          const medZ = _median(validPoses.map(r => r.camPos[2]));

          const { R_wc: newR } = defaultCameraPoseAndRotation(medPitch, medYaw, medRoll);
          for (let j = 0; j < 9; j++) R_wc[j] = newR[j];
          camPos[0] = medX; camPos[1] = medY; camPos[2] = medZ;
          cameraYawDeg = Number.isFinite(medYaw) ? medYaw : cameraYawDeg;
          autoPnpPrepassApplied = true;

          _setCachedAutoPnpPrepass(prepassCacheKey, {
            status: 'ok',
            camPos: [camPos[0], camPos[1], camPos[2]],
            R_wc: Array.from(R_wc),
            cameraYawDeg,
            validPoseCount: validPoses.length,
          });
          console.log(`[Pose] Auto-PnP pre-pass DONE: calibrated from ${validPoses.length} frames — pos=[${medX.toFixed(3)}, ${medY.toFixed(3)}, ${medZ.toFixed(3)}], angles=p${medPitch.toFixed(1)} y${medYaw.toFixed(1)} r${medRoll.toFixed(1)}`);
        } else {
          _setCachedAutoPnpPrepass(prepassCacheKey, {
            status: 'failed',
            validPoseCount: validPoses.length,
          });
          console.log(`[Pose] Auto-PnP pre-pass: only ${validPoses.length} valid — using default camera pose`);
        }
      }
    }

    // 6. Process frames
    report('Processing frames...', 0.08);
    let frameCount = 0;
    let errorCount = 0;
    let autoPnpFrameCount = 0;
    const rudderFilterState = { recent: [], ema: null };

    // Determine time range to process (segment or full video)
    if (isFinite(rangeStart) || isFinite(rangeEnd)) {
      console.log(`[Pose] Processing segment: ${rangeStart.toFixed(1)}s – ${isFinite(rangeEnd) ? rangeEnd.toFixed(1) + 's' : 'end'}`);
    }

    try {
      await extractFrames(videoEl, fps, async ({ canvas, fullCanvas, ts_s, frameIdx, width, height }) => {
        try {
        // Auto-PnP: periodically re-estimate camera pose from boat keypoints
        const shouldRunFrameAutoPnp = autoPnpReady && frameIdx % autoPnpInterval === 0 && !(autoPnpPrepassApplied && frameIdx === 0);
        const frameAutoPnpPromise = shouldRunFrameAutoPnp ? (async () => {
          try {
            // Use current camera state as initial guess for LM solver
            const currentPose = {
              pitch: cfg.camera_pitch_deg ?? 14.7,
              yaw: cameraYawDeg,
              roll: cfg.camera_roll_deg ?? 0,
              x: camPos[0], y: camPos[1], z: camPos[2],
            };
            const pnpResult = await AutoPnP.estimateCameraPose(fullCanvas || videoEl, currentPose);
            if (pnpResult && pnpResult.camPos && pnpResult.R_wc) {
              // Update rotation matrix first (row-major flat ← nested 3×3)
              for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
                R_wc[r * 3 + c] = pnpResult.R_wc[r][c];
              }
              // Update camera position (only after R_wc succeeds)
              camPos[0] = pnpResult.camPos[0];
              camPos[1] = pnpResult.camPos[1];
              camPos[2] = pnpResult.camPos[2];
              if (Number.isFinite(Number(pnpResult?.angles?.yaw_deg))) {
                cameraYawDeg = Number(pnpResult.angles.yaw_deg);
              }
              autoPnpFrameCount++;
              if (autoPnpFrameCount <= 5 || autoPnpFrameCount % 10 === 0) {
                console.log(`[AutoPnP] frame ${frameIdx}: updated (fullres-video) camPos=[${camPos.map(v=>v.toFixed(3))}], err=${pnpResult.meanErrorPx?.toFixed(1)}px (total updates: ${autoPnpFrameCount})`);
              }
              return true;
            } else {
              if (frameIdx < 5 * autoPnpInterval) {
                console.log(`[AutoPnP] frame ${frameIdx}: no result (model returned null)`);
              }
            }
          } catch (e) {
            console.warn(`[AutoPnP] frame ${frameIdx}: ERROR`, e.message || e);
          }
          return false;
        })() : Promise.resolve(false);

        let rudderPrediction = {
          model_angle_deg: null,
          corrected_angle_deg: null,
          camera_yaw_deg: Number.isFinite(cameraYawDeg) ? cameraYawDeg : null,
        };
        let boomPrediction = {
          angle_deg: null,
          angle_raw_deg: null,
          model_angle_deg: null,
          raw_output: null,
          outlier: false,
        };
        const predictRudderForCurrentCamera = async () => {
          if (!rudderReady) return;
          try {
            rudderPrediction = await Rudder.predictRudderAngle(canvas, cameraYawDeg);
            const rawCorrected = Number(rudderPrediction.corrected_angle_deg);
            const filt = _filterRudderAngle(rawCorrected, rudderFilterState);
            rudderPrediction.corrected_angle_raw_deg = _mapLegacyRudderAngleToSigned(rawCorrected);
            rudderPrediction.corrected_angle_deg = _mapLegacyRudderAngleToSigned(filt.value);
            rudderPrediction.outlier = filt.outlier;
          } catch (rudderErr) {
            if (frameIdx < 5) {
              console.warn(`[Rudder] frame ${frameIdx}: prediction failed`, rudderErr?.message || rudderErr);
            }
          }
        };
        const predictBoomForFrame = async () => {
          if (!boomReady) return;
          try {
            // The pose canvas is downscaled for MediaPipe speed. PilotNet needs
            // the sampled full-resolution frame so its narrow ROI stays stable.
            boomPrediction = await Boom.predictBoomAngle(fullCanvas || videoEl);
          } catch (boomErr) {
            if (frameIdx < 5) {
              console.warn(`[Boom] frame ${frameIdx}: prediction failed`, boomErr?.message || boomErr);
            }
          }
        };
        const boomPredictionPromise = predictBoomForFrame();
        let rudderPredictionPromise = shouldRunFrameAutoPnp ? null : predictRudderForCurrentCamera();

        // Run MediaPipe
        const timestampMs = Math.round(ts_s * 1000);
        const result = _detectForVideoSafe(landmarker, canvas, timestampMs);
        await frameAutoPnpPromise;
        rudderPrediction.camera_yaw_deg = Number.isFinite(cameraYawDeg) ? cameraYawDeg : null;
        if (!rudderPredictionPromise) rudderPredictionPromise = predictRudderForCurrentCamera();
        await Promise.all([rudderPredictionPromise, boomPredictionPromise]);

        if (!result.landmarks || result.landmarks.length === 0 ||
            !result.worldLandmarks || result.worldLandmarks.length === 0) {
          // No pose detected
          await Promise.all([
            skeletonWriter.push(JSON.stringify({
              frame: frameIdx, ts: ts_s, detected: false,
            })),
            metricsWriter.push(JSON.stringify({
              frame: frameIdx, ts: ts_s, detected: false,
              rudder_angle: rudderPrediction.corrected_angle_deg,
              rudder_angle_raw: rudderPrediction.corrected_angle_raw_deg ?? rudderPrediction.corrected_angle_deg,
              rudder_model_angle: rudderPrediction.model_angle_deg,
              rudder_camera_yaw_deg: rudderPrediction.camera_yaw_deg,
              rudder_angle_system: 'signed_centered_v1',
              rudder_outlier: Boolean(rudderPrediction.outlier),
              boom_angle: boomPrediction.angle_deg,
              boom_angle_raw: boomPrediction.angle_raw_deg ?? boomPrediction.angle_deg,
              boom_model_angle: boomPrediction.model_angle_deg,
              boom_raw_output: boomPrediction.raw_output,
              boom_angle_system: boomPrediction.angle_system || 'boom_minus_x_centered_v3',
              boom_outlier: Boolean(boomPrediction.outlier),
            })),
          ]); 
          // Still update Kalman (predict only)
          smoother.smooth(null, {}, dt);
          frameCount++;
          return;
        }

        // Extract landmarks from first person
        const normLm = result.landmarks[0]; // NormalizedLandmark[33]
        const worldLm = result.worldLandmarks[0]; // Landmark[33]

        // Convert to our format: index → [x,y,z]
        const worldDict = {};
        const normDict = {};
        const confDict = {};
        for (let i = 0; i < 33; i++) {
          if (worldLm[i]) {
            worldDict[i] = [worldLm[i].x, worldLm[i].y, worldLm[i].z];
          }
          if (normLm[i]) {
            normDict[i] = [normLm[i].x, normLm[i].y, normLm[i].z, normLm[i].visibility ?? 0.5];
            confDict[i] = normLm[i].visibility ?? 0.5;
          }
        }

        // Place skeleton on boat
        let placed = computePlacedSkeletonSymmetricRaycast(
          worldDict, normDict, K, imgW, imgH, camPos, R_wc, zHip, zAnkle
        );
        let placementMethod = placed ? 'raycast' : null;
        if (!placed) {
          placed = placeSkeletonOnBoat(worldDict);
          if (placed) placementMethod = 'fallback';
        }
        // Log placement method for first several frames
        if (frameIdx < 5 || (frameIdx < 100 && frameIdx % 20 === 0)) {
          const hip23 = placed?.[23], hip24 = placed?.[24];
          const hipPos = (hip23 && hip24) ? `hip=[(${hip23.map(v=>v.toFixed(3))}),(${hip24.map(v=>v.toFixed(3))})]` : 'no hips';
          console.log(`[Pose] frame ${frameIdx} ts=${ts_s.toFixed(2)}: placement=${placementMethod || 'NONE'}, ${hipPos}`);
        }

        // Kalman smooth
        let smoothed = null;
        if (placed) {
          smoothed = smoother.smooth(placed, confDict, dt);
        } else {
          smoother.smooth(null, {}, dt);
        }

        // Use smoothed if available, otherwise fall back to placed
        const skeletonSource = smoothed || placed;
        const heightScaled = scaleSkeletonToAthleteHeight(skeletonSource, athleteHeight);
        const skeletonForMetrics = heightScaled.skeleton || skeletonSource;

        // Compute metrics
        const metrics = skeletonForMetrics
          ? computeFrameMetrics(skeletonForMetrics, athleteMass, boatCom)
          : { trunk_angle: null, com_x: null, com_y: null, com_z: null,
              pitch_moment: null, roll_moment: null, sitting_score: null };
        const metricsWithRudder = {
          ...metrics,
          skeleton_height_m: heightScaled.measuredHeight ?? null,
          skeleton_height_target_m: heightScaled.targetHeight ?? null,
          skeleton_height_scale: heightScaled.applied ? heightScaled.scale : null,
          rudder_angle: rudderPrediction.corrected_angle_deg,
          rudder_angle_raw: rudderPrediction.corrected_angle_raw_deg ?? rudderPrediction.corrected_angle_deg,
          rudder_model_angle: rudderPrediction.model_angle_deg,
          rudder_camera_yaw_deg: rudderPrediction.camera_yaw_deg,
          rudder_angle_system: 'signed_centered_v1',
          rudder_outlier: Boolean(rudderPrediction.outlier),
          boom_angle: boomPrediction.angle_deg,
          boom_angle_raw: boomPrediction.angle_raw_deg ?? boomPrediction.angle_deg,
          boom_model_angle: boomPrediction.model_angle_deg,
          boom_raw_output: boomPrediction.raw_output,
          boom_angle_system: boomPrediction.angle_system || 'boom_minus_x_centered_v3',
          boom_outlier: Boolean(boomPrediction.outlier),
        };

        // Serialize skeleton (compact: only xyz per landmark)
        const skelOut = {};
        if (skeletonForMetrics) {
          for (const [idx, pt] of Object.entries(skeletonForMetrics)) {
            skelOut[idx] = [Math.round(pt[0]*1e4)/1e4, Math.round(pt[1]*1e4)/1e4, Math.round(pt[2]*1e4)/1e4];
          }
        }
        const skelKeys = Object.keys(skelOut).length;
        // Debug: log first few frames' skeleton output
        if (frameIdx < 3) {
          console.log(`[Pose] frame ${frameIdx}: smoothed=${!!smoothed}, placed=${!!placed}, skelOut keys=${skelKeys}, hip23=${JSON.stringify(skelOut[23])}, hip24=${JSON.stringify(skelOut[24])}`);
        }

        await Promise.all([
          skeletonWriter.push(JSON.stringify({
            frame: frameIdx, ts: ts_s, detected: true, skeleton: skelOut,
          })),
          metricsWriter.push(JSON.stringify({
            frame: frameIdx, ts: ts_s, detected: true, ...metricsWithRudder,
          })),
        ]);

        // Emit live frame for 3D viewer
        if (skelKeys > 0) {
          _emitFrame(fileId, { ts: ts_s, skeleton: skelOut, metrics: metricsWithRudder });
        }

        frameCount++;
        } catch (e) {
          if (_isStorageWriteError(e)) throw e;
          errorCount++;
          if (errorCount > 50) throw new Error(`Too many frame errors: ${e.message}`);
        }
      }, (pct) => {
        report(`Processing: ${Math.round(pct * 100)}%`, 0.08 + pct * 0.9);
      }, cancelToken, rangeStart, rangeEnd, {
        captureFullResolutionFrame: ({ frameIdx }) => Boolean(
          boomReady ||
          (autoPnpReady && frameIdx % autoPnpInterval === 0 && !(autoPnpPrepassApplied && frameIdx === 0))
        ),
        poseInputMaxDim: opts?.poseInputMaxDim,
        preferSeek: isSegmentRun && opts?.exactSegmentSeek === true,
      });
    } finally {
      await Promise.all([
        skeletonWriter.flush(),
        metricsWriter.flush(),
      ]);
    }

    // 7. Clean up
    URL.revokeObjectURL(videoUrl);
    videoEl.src = '';

    // 7b. For segmented processing, normalize merged JSONL files:
    // sort by timestamp and dedupe per timestamp bucket.
    if (isSegmentRun) {
      try {
        const skelCount = await _normalizeJsonlFileByTs(skeletonPath);
        const metricCount = await _normalizeJsonlFileByTs(metricsPath);
        console.log(`[Pose] merged segmented output: skeleton=${skelCount}, metrics=${metricCount}`);
      } catch (normErr) {
        console.warn('[Pose] segmented output normalization failed:', normErr);
      }
    }

    // 8. Update DB with run status
    const existingRun = await DB.getCvRun(projectId, fileId);
    let processedRanges = cancelToken.cancelled
      ? _normalizeRangeList(existingRun?.processed_ranges || [])
      : (isSegmentRun
        ? _normalizeRangeList([...(existingRun?.processed_ranges || []), [effectiveStart, effectiveEnd]])
        : _normalizeRangeList([[0, videoDuration]]));
    if (!cancelToken.cancelled && isSegmentRun) {
      try {
        const actualCoverage = await getSkeletonCoverage(projectId, fileId);
        const clippedCoverage = _clipRangeList(actualCoverage.map(c => [c.start, c.end]), effectiveStart, effectiveEnd);
        processedRanges = _normalizeRangeList([...(existingRun?.processed_ranges || []), ...clippedCoverage]);
        const coverageRatio = _rangeListCoverageRatio(clippedCoverage, effectiveStart, effectiveEnd, 1.0);
        if (!_rangeListCovers(clippedCoverage, effectiveStart, effectiveEnd, 1.0) && coverageRatio < 0.86) {
          console.warn(
            `[Pose] segment coverage incomplete for ${fileId}: requested=${effectiveStart.toFixed(2)}-${effectiveEnd.toFixed(2)} ` +
            `saved=${clippedCoverage.map(([s, e]) => `${s.toFixed(2)}-${e.toFixed(2)}`).join(', ') || 'none'}`
          );
        }
      } catch (coverageErr) {
        console.warn('[Pose] could not verify saved segment coverage:', coverageErr);
      }
    }
    await DB.upsertCvRun(projectId, fileId, {
      status: cancelToken.cancelled ? 'cancelled' : 'completed',
      frame_count: frameCount,
      fps,
      duration: videoDuration,
      error_count: errorCount,
      autopnp_updates: autoPnpFrameCount,
      processed_ranges: processedRanges,
    });

    console.log(`[Pose] DONE: ${frameCount} frames, ${errorCount} errors, ${autoPnpFrameCount} auto-PnP updates, output: ${skeletonPath}`);
    report('Done', 1);
    return { frameCount, skeletonPath, metricsPath };

  } catch (e) {
    await DB.upsertCvRun(projectId, fileId, {
      status: 'error',
      error: e.message,
    });
    throw e;
  } finally {
    _activeRuns.delete(fileId);
  }
}

function _lineOutsideVideoRange(line, startSec, endSec) {
  if (!line || !line.trim()) return false;
  try {
    const obj = JSON.parse(line);
    const ts = Number(obj?.ts);
    if (!Number.isFinite(ts)) return true;
    return ts < startSec || ts > endSec;
  } catch {
    return true;
  }
}

async function _filterJsonlOutsideRange(path, startSec, endSec) {
  const text = await Storage.readFileText(path);
  if (text == null) return 0;
  const kept = text
    .split(/\r?\n/)
    .filter(line => _lineOutsideVideoRange(line, startSec, endSec));
  const nextText = kept.length ? `${kept.join('\n')}\n` : '';
  const slash = path.lastIndexOf('/');
  const dirParts = slash >= 0 ? path.slice(0, slash).split('/').filter(Boolean) : [];
  const filename = slash >= 0 ? path.slice(slash + 1) : path;
  await Storage.writeFile(dirParts, filename, nextText);
  return kept.length;
}

function _subtractRangeList(ranges, startSec, endSec, tolerance = 0.02) {
  const start = Number(startSec);
  const end = Number(endSec);
  if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return _normalizeRangeList(ranges || []);
  const out = [];
  for (const range of ranges || []) {
    const s = Number(Array.isArray(range) ? range[0] : range?.start);
    const e = Number(Array.isArray(range) ? range[1] : range?.end);
    if (!Number.isFinite(s) || !Number.isFinite(e) || !(e > s)) continue;
    if (e <= start + tolerance || s >= end - tolerance) {
      out.push([s, e]);
      continue;
    }
    if (s < start - tolerance) out.push([s, Math.max(s, start)]);
    if (e > end + tolerance) out.push([Math.min(e, end), e]);
  }
  return _normalizeRangeList(out);
}

export async function deleteProcessedRange(projectId, fileId, startSec, endSec) {
  const start = Math.max(0, Number(startSec));
  const end = Number(endSec);
  if (!projectId || !fileId || !Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return;
  const outputDir = `${projectId}/${fileId}`;
  await Promise.all([
    _filterJsonlOutsideRange(`${outputDir}/skeleton.jsonl`, start, end),
    _filterJsonlOutsideRange(`${outputDir}/metrics.jsonl`, start, end),
  ]);
  const existingRun = await DB.getCvRun(projectId, fileId);
  if (existingRun) {
    const processedRanges = _subtractRangeList(existingRun.processed_ranges || [], start, end);
    await DB.upsertCvRun(projectId, fileId, {
      ...existingRun,
      status: processedRanges.length ? existingRun.status : 'partial',
      processed_ranges: processedRanges,
      updated_at: new Date().toISOString(),
    });
  }
}

/**
 * Cancel a running skeleton job for a file.
 */
export function cancelProcessing(fileId) {
  const token = _activeRuns.get(fileId);
  if (token) token.cancelled = true;
}

/**
 * Check if skeleton processing is currently running for a file.
 */
export function isProcessing(fileId) {
  return _activeRuns.has(fileId);
}


// ── Load skeleton frames for 3D viewer ────────────────────────────────

/**
 * Load skeleton JSONL from OPFS for a specific time range.
 * @param {string} projectId
 * @param {string} fileId
 * @param {number} startTs — video-local seconds
 * @param {number} endTs — video-local seconds
 * @returns {Array<{frame, ts, skeleton}>}
 */
export async function loadSkeletonFrames(projectId, fileId, startTs = 0, endTs = Infinity) {
  const path = `${projectId}/${fileId}/skeleton.jsonl`;
  console.log(`[Pose] loadSkeletonFrames: path="${path}", range=[${startTs}, ${endTs}]`);
  try {
    const text = await Storage.readFileText(path);
    if (!text) { console.warn(`[Pose] loadSkeletonFrames: OPFS returned null/empty for "${path}"`); return []; }
    const lines = text.split('\n').filter(l => l.trim());
    let detected = 0, empty = 0, filtered = 0;
    const allDetectedFrames = [];
    const frames = [];
    for (const line of lines) {
      try {
        const obj = JSON.parse(line);
        if (!obj.detected) { filtered++; continue; }
        if (!obj.skeleton || Object.keys(obj.skeleton).length === 0) { empty++; continue; }
        detected++;
        allDetectedFrames.push(obj);
        if (obj.ts >= startTs && obj.ts <= endTs) {
          frames.push(obj);
        } else {
          filtered++;
        }
      } catch {}
    }
    let normalized = _normalizeRecordsByTs(frames);
    if (!normalized.length && detected > 0) {
      const fallback = _normalizeRecordsByTs(allDetectedFrames);
      if (fallback.length) {
        const fbLo = Number(fallback[0]?.ts);
        const fbHi = Number(fallback[fallback.length - 1]?.ts);
        const reqMid = (Number(startTs) + Number(endTs)) / 2;
        const outOfDomain = Number.isFinite(fbLo) && Number.isFinite(fbHi)
          && Number.isFinite(reqMid)
          && (reqMid < fbLo || reqMid > fbHi);
        if (outOfDomain) {
          console.warn(`[Pose] loadSkeletonFrames: no ts-range matches for [${startTs}, ${endTs}] but found ${fallback.length} skeleton frames in-domain [${fbLo.toFixed(3)}, ${fbHi.toFixed(3)}]; returning unfiltered frames as fallback.`);
        } else {
          console.warn(`[Pose] loadSkeletonFrames: range filtering returned 0 despite ${detected} detected skeleton lines; returning unfiltered frames as fallback.`);
        }
        normalized = fallback;
      }
    }
    const tsLo = normalized.length ? Number(normalized[0].ts) : NaN;
    const tsHi = normalized.length ? Number(normalized[normalized.length - 1].ts) : NaN;
    const tsRange = (Number.isFinite(tsLo) && Number.isFinite(tsHi)) ? `${tsLo.toFixed(1)}-${tsHi.toFixed(1)}` : 'none';
    console.log(`[Pose] loadSkeletonFrames: ${lines.length} lines total, ${detected} with skeleton, ${empty} empty skel, ${filtered} filtered, ${normalized.length} returned (ts range: ${tsRange})`);
    return normalized;
  } catch (e) {
    console.error(`[Pose] loadSkeletonFrames: ERROR reading "${path}":`, e);
    return [];
  }
}

/**
 * Load metrics JSONL from OPFS.
 */
export async function loadMetrics(projectId, fileId) {
  const path = `${projectId}/${fileId}/metrics.jsonl`;
  try {
    const text = await Storage.readFileText(path);
    if (!text) return [];
    const lines = text.split('\n').filter(l => l.trim());
    const rows = lines.map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
    const normalizedRows = rows.map(normalizeMetricRudderConvention);
    return _normalizeRecordsByTs(normalizedRows);
  } catch {
    return [];
  }
}

/**
 * Get processed coverage from skeleton.jsonl.
 * Returns array of {start, end} (video seconds) for processed segments,
 * regardless of whether pose detection succeeded on every frame.
 */
export async function getSkeletonCoverage(projectId, fileId) {
  const path = `${projectId}/${fileId}/skeleton.jsonl`;
  try {
    const text = await Storage.readFileText(path);
    if (!text) return [];
    const lines = text.split('\n').filter(l => l.trim());
    if (lines.length === 0) return [];

    const rows = [];
    for (const line of lines) {
      try {
        const obj = JSON.parse(line);
        if (!Number.isFinite(Number(obj.ts))) continue;
        rows.push(obj);
      } catch {}
    }
    const frames = _normalizeRecordsByTs(rows);
    if (frames.length === 0) return [];

    const tsVals = frames.map(f => Number(f.ts)).filter(Number.isFinite).sort((a, b) => a - b);
    if (tsVals.length === 0) return [];
    if (tsVals.length === 1) return [{ start: tsVals[0], end: tsVals[0] }];

    const dts = [];
    for (let i = 1; i < tsVals.length; i++) {
      const dt = tsVals[i] - tsVals[i - 1];
      if (dt > 0) dts.push(dt);
    }
    dts.sort((a, b) => a - b);
    const dtMed = dts.length ? dts[Math.floor(dts.length / 2)] : (1 / Math.max(1, SKELETON_TARGET_FPS));
    const gapThreshold = Math.max(1.0, dtMed * 4.0);

    const out = [];
    let start = tsVals[0];
    let prev = tsVals[0];
    for (let i = 1; i < tsVals.length; i++) {
      const t = tsVals[i];
      if ((t - prev) > gapThreshold) {
        out.push({ start, end: prev });
        start = t;
      }
      prev = t;
    }
    out.push({ start, end: prev });
    return out;
  } catch {
    return [];
  }
}
