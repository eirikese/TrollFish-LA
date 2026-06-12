/**
 * TrollFish — pipeline.js
 * Orchestrates the file ingest pipeline (browser-side).
 *
 * Replaces the server-side worker.py + pipeline.py:
 *   1. Register files in IndexedDB
 *   2. Parse GPS from CSVs (csv-gps.js)
 *   3. Extract GPS from GoPro videos (gopro-gps.js)
 *   4. Match video tracks to CSV tracks (matcher.js)
 *
 * All processing happens on the main thread or in Web Workers.
 * Progress is reported via callbacks.
 */

import * as DB from './db.js';
import * as FM from './file-manager.js';
import { parseCsvTrack } from './csv-gps.js?v=20260527detail1';
// import { parseGoProVideoTrack } from './gopro-gps.js';
import { matchVideoTracksToCsv } from './matcher.js?v=20260527detail1';
import { parseAppleMobileVideoMetadata } from './mobile-video-metadata.js';
import { MAP_MAX_POINTS_PER_TRACK } from './config.js';

const CSV_INGEST_CONCURRENCY = 4;
const VIDEO_INGEST_CONCURRENCY = 2;

// ── File classification ───────────────────────────────────────────────

const VIDEO_EXTS = new Set(['.mp4', '.mov', '.avi', '.mkv', '.m4v']);
const CSV_EXTS = new Set(['.csv', '.tsv', '.txt']);

function classifyFile(filename) {
  const ext = ('.' + filename.split('.').pop()).toLowerCase();
  if (VIDEO_EXTS.has(ext)) return 'video';
  if (CSV_EXTS.has(ext)) return 'csv';
  return 'unknown';
}

function normalizeEpochSeconds(ts) {
  const n = Number(ts);
  if (!Number.isFinite(n)) return null;
  const a = Math.abs(n);
  if (a > 1e14) return n / 1e6; // microseconds
  if (a > 1e11) return n / 1e3; // milliseconds
  return n;
}

function sanitizeVideoRange(tsStartRaw, tsEndRaw, durationSec = null) {
  let start = normalizeEpochSeconds(tsStartRaw);
  let end = normalizeEpochSeconds(tsEndRaw);
  const durN = Number(durationSec);
  const dur = Number.isFinite(durN) && durN > 0 ? durN : null;

  if (start == null && end != null && dur != null) {
    start = end - dur;
  } else if (start == null && end != null) {
    start = end - 600;
  }
  if (start != null && (end == null || end <= start) && dur != null) {
    end = start + dur;
  } else if (start != null && end == null) {
    end = start + 600;
  }
  if (start != null && end != null) {
    const span = end - start;
    const maxSpan = dur != null ? Math.max(900, dur * 6.0) : 12 * 3600;
    if (!(span > 0) || span > maxSpan) {
      end = dur != null ? (start + dur) : (start + maxSpan);
    }
  }
  return { start, end };
}

function sanitizeVideoGpsPoints(points, durationSec = null) {
  if (!Array.isArray(points) || points.length === 0) return [];
  const durN = Number(durationSec);
  const dur = Number.isFinite(durN) && durN > 0 ? durN : null;

  const cleaned = [];
  for (const p of points) {
    const lat = Number(p?.lat);
    const lon = Number(p?.lon);
    const videoS = Number(p?.video_s);
    const ts = normalizeEpochSeconds(p?.ts);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) continue;
    if (!Number.isFinite(videoS) || videoS < -1 || videoS > 24 * 3600) continue;
    if (!Number.isFinite(ts)) continue;
    cleaned.push({ ...p, ts, video_s: videoS });
  }
  if (cleaned.length < 2) return [];

  cleaned.sort((a, b) => a.video_s - b.video_s);
  let deduped = [];
  let lastVs = null;
  for (const p of cleaned) {
    if (lastVs != null && Math.abs(p.video_s - lastVs) <= 1e-6) {
      deduped[deduped.length - 1] = p;
    } else {
      deduped.push(p);
      lastVs = p.video_s;
    }
  }
  if (deduped.length < 2) return [];

  // ── Trim GPS warm-up outliers using median-offset filter ──────────
  // Good GPS points satisfy ts ≈ video_s + constant_offset.
  // Warm-up junk (wrong epoch / no-fix) will have wildly different offsets.
  // Compute median offset and remove points that deviate too much.
  const trimThresh = dur != null ? Math.max(120, dur * 0.5) : 300;
  const offsets = deduped.map(p => p.ts - p.video_s);
  const sortedOffsets = [...offsets].sort((a, b) => a - b);
  const medOffset = sortedOffsets[Math.floor(sortedOffsets.length / 2)];
  const beforeTrim = deduped.length;
  deduped = deduped.filter(p => Math.abs((p.ts - p.video_s) - medOffset) <= trimThresh);
  if (deduped.length < beforeTrim) {
    console.log(`[Pipeline] trimmed ${beforeTrim - deduped.length} GPS warm-up outlier(s) (median offset=${medOffset.toFixed(1)}s, thresh=${trimThresh.toFixed(1)}s)`);
  }
  if (deduped.length < 2) return [];

  // ── Span checks on trimmed data ──────────────────────────────────
  const vsStart = deduped[0].video_s;
  const vsEnd = deduped[deduped.length - 1].video_s;
  const tsStart = deduped[0].ts;
  const tsEnd = deduped[deduped.length - 1].ts;
  const vsSpan = vsEnd - vsStart;
  const tsSpan = tsEnd - tsStart;
  if (!(vsSpan > 0) || !(tsSpan > 0)) return [];

  const maxVideoSpan = dur != null ? Math.max(900, dur * 3.0) : 8 * 3600;
  const maxTsSpan = dur != null ? Math.max(900, dur * 6.0) : 24 * 3600;
  if (vsSpan > maxVideoSpan || tsSpan > maxTsSpan) {
    console.warn(`[Pipeline] rejecting suspicious GPS track: videoSpan=${vsSpan.toFixed(1)}s tsSpan=${tsSpan.toFixed(1)}s dur=${dur ?? 'n/a'}s`);
    return [];
  }

  // Residual check on surviving points
  const residuals = deduped.map(p => Math.abs((p.ts - p.video_s) - medOffset)).sort((a, b) => a - b);
  const p95 = residuals[Math.floor((residuals.length - 1) * 0.95)];
  const maxResidual = dur != null ? Math.max(120, dur * 0.5) : 180;
  if (p95 > maxResidual) {
    console.warn(`[Pipeline] rejecting GPS track due to unstable ts/video alignment (p95=${p95.toFixed(2)}s, max=${maxResidual.toFixed(2)}s)`);
    return [];
  }

  return deduped;
}

async function getProjectCsvTimeRange(projectId) {
  const csvTracks = await DB.listTracks(projectId, 'csv');
  let minTs = null;
  let maxTs = null;
  for (const track of csvTracks) {
    const start = normalizeEpochSeconds(track?.ts_start);
    const end = normalizeEpochSeconds(track?.ts_end);
    if (start == null || end == null) continue;
    minTs = minTs == null ? start : Math.min(minTs, start);
    maxTs = maxTs == null ? end : Math.max(maxTs, end);
  }
  return minTs != null && maxTs != null ? { minTs, maxTs } : null;
}

function rangesOverlap(startA, endA, startB, endB, slackSec = 5) {
  if (![startA, endA, startB, endB].every(Number.isFinite)) return false;
  return endA >= startB - slackSec && startA <= endB + slackSec;
}

function isContinuousCsvTrack(track) {
  const fileText = String(track?.filename || '').toLowerCase();
  const meta = track?.meta || {};
  const sessionText = String(meta.api_session_id || meta.session_id || '').toLowerCase();
  const sourceText = String(meta.source || '').toLowerCase();
  return (
    /\bcont-\d{4}-\d{2}-\d{2}\b/.test(fileText) ||
    /\bcont-\d{4}-\d{2}-\d{2}\b/.test(sessionText) ||
    sourceText === 'continuous'
  );
}

function looksLikeAppleCaptureFilename(filename) {
  const base = String(filename || '').trim().toUpperCase();
  if (!base) return false;
  return /^(IMG|VID|PXL|MVIMG|IMG_E)_\d{3,}/.test(base);
}

function isStoredPlaybackOnlyVideo(file) {
  if (file?.force_analyze) return false;   // user opted this video into analysis (overrides flag + heuristic)
  if (file?.external_playback) return true;
  const captureStartTs = normalizeEpochSeconds(file?.capture_start_ts);
  if (captureStartTs == null) return false;
  const source = String(file?.capture_ts_source || '').trim().toLowerCase();
  if (!source || source === 'file_last_modified') return false;
  return !!file?.playback_only && (
    /apple/i.test(String(file?.device_make || '')) ||
    /iphone|ipad|ipod/i.test(String(file?.device_model || '')) ||
    looksLikeAppleCaptureFilename(file?.filename)
  );
}

function shouldRefreshStoredMobileMetadata(file) {
  if (file?.kind !== 'video') return false;
  if (!looksLikeAppleCaptureFilename(file?.filename)) return false;
  const source = String(file?.capture_ts_source || '').trim().toLowerCase();
  return !source || source === 'file_last_modified' || source === 'quicktime_mvhd';
}

async function maybeRefreshStoredMobileMetadata(fileRec) {
  if (!shouldRefreshStoredMobileMetadata(fileRec)) return fileRec;

  const file = await FM.getFileForReading(fileRec.id);
  if (!file) return fileRec;

  let appleMeta = null;
  try {
    appleMeta = await parseAppleMobileVideoMetadata(file);
  } catch (err) {
    console.warn(`[Pipeline] mobile metadata refresh skipped for ${fileRec.filename}:`, err);
    return fileRec;
  }
  const captureStartTs = normalizeEpochSeconds(appleMeta?.captureStartTs);
  if (captureStartTs == null) return fileRec;

  const nextSource = String(appleMeta?.captureTsSource || '');
  const prevCaptureTs = normalizeEpochSeconds(fileRec?.capture_start_ts);
  const prevEstStart = normalizeEpochSeconds(fileRec?.est_start_ts);
  const nextPlaybackOnly = !!appleMeta?.isLikelyIphonePlaybackVideo;
  const sourceChanged = nextSource && nextSource !== String(fileRec?.capture_ts_source || '');
  const captureChanged = prevCaptureTs == null || Math.abs(prevCaptureTs - captureStartTs) > 1;
  const estChanged = prevEstStart == null || Math.abs(prevEstStart - captureStartTs) > 1;
  const playbackChanged = !!fileRec?.playback_only !== nextPlaybackOnly;
  const makeChanged = String(fileRec?.device_make || '') !== String(appleMeta?.make || '');
  const modelChanged = String(fileRec?.device_model || '') !== String(appleMeta?.model || '');
  if (!sourceChanged && !captureChanged && !estChanged && !playbackChanged && !makeChanged && !modelChanged) {
    return fileRec;
  }

  const nextFields = {
    capture_start_ts: captureStartTs,
    est_start_ts: captureStartTs,
    capture_ts_source: nextSource || fileRec?.capture_ts_source || null,
    playback_only: nextPlaybackOnly,
    external_playback: nextPlaybackOnly,
    device_make: appleMeta?.make || null,
    device_model: appleMeta?.model || null,
  };
  await DB.updateFileFields(fileRec.id, nextFields);
  return { ...fileRec, ...nextFields };
}

function getVideoIngestConcurrency() {
  const cores = Number(globalThis.navigator?.hardwareConcurrency);
  if (Number.isFinite(cores) && cores <= 4) return 1;
  return VIDEO_INGEST_CONCURRENCY;
}

async function runLimited(records, concurrency, worker) {
  const limit = Math.max(1, Math.min(records.length || 1, Math.floor(Number(concurrency) || 1)));
  let next = 0;
  const runners = Array.from({ length: limit }, async () => {
    while (next < records.length) {
      const idx = next++;
      await worker(records[idx], idx);
    }
  });
  await Promise.all(runners);
}


// ── Ingest pipeline ───────────────────────────────────────────────────

/**
 * Ingest files into a project.
 * Registers them in IndexedDB, parses GPS, and runs matching.
 *
 * @param {string} projectId
 * @param {{ file: File, fileId: string }[]} pickedFiles — from file-manager.registerPickedFiles()
 * @param {(msg: string, progress: number) => void} [onProgress]
 * @returns {{ files: object[], errors: string[] }}
 */
export async function ingestFiles(projectId, pickedFiles, onProgress = null) {
  const report = (msg, pct) => { if (onProgress) onProgress(msg, pct); };
  const errors = [];
  const insertedFiles = [];

  // Step 1: Register files in DB
  report('Registering files...', 0);
  for (const { file, fileId } of pickedFiles) {
    const kind = classifyFile(file.name);
    try {
      const rec = await DB.insertFile({
        id: fileId,
        project_id: projectId,
        filename: file.name,
        kind,
        size_bytes: file.size,
        status: 'uploaded',
      });
      insertedFiles.push(rec);
    } catch (e) {
      errors.push(`Failed to register ${file.name}: ${e.message}`);
    }
  }

  // Step 2: Parse GPS tracks
  const totalSteps = insertedFiles.length + 1; // +1 for matching
  let completedSteps = 0;
  const activeProgress = new Map();

  const reportProcessingProgress = (rec, msg, progress = 0) => {
    const clamped = Math.max(0, Math.min(1, Number(progress) || 0));
    activeProgress.set(rec.id, clamped);
    const inFlight = [...activeProgress.values()].reduce((sum, value) => sum + value, 0);
    report(msg, ((completedSteps + inFlight) / totalSteps) * 0.9);
  };

  async function processOne(rec) {
    reportProcessingProgress(rec, `Processing ${rec.filename}...`, 0.02);
    try {
      if (rec.kind === 'csv') {
        await processCsvFile(projectId, rec);
      } else if (rec.kind === 'video') {
        await processVideoFile(projectId, rec, (videoProgress) => {
          reportProcessingProgress(rec, `Extracting GPS from ${rec.filename}...`, videoProgress);
        });
      } else {
        throw new Error('Unsupported file type');
      }
      await DB.updateFileStatus(rec.id, 'processed');
    } catch (e) {
      errors.push(`${rec.filename}: ${e.message}`);
      if (e?.code === 'playback_only_outside_csv_range') {
        try { await DB.deleteFile(projectId, rec.id); } catch {}
        try { await FM.forgetFile(rec.id); } catch {}
      } else {
        await DB.updateFileStatus(rec.id, 'error', e.message);
      }
    } finally {
      activeProgress.delete(rec.id);
      completedSteps++;
      report(`Processed ${completedSteps}/${insertedFiles.length} files`, (completedSteps / totalSteps) * 0.9);
    }
  }

  const csvFiles = insertedFiles.filter(rec => rec.kind === 'csv');
  const videoFiles = insertedFiles.filter(rec => rec.kind === 'video');
  const otherFiles = insertedFiles.filter(rec => rec.kind !== 'csv' && rec.kind !== 'video');

  await runLimited(csvFiles, CSV_INGEST_CONCURRENCY, processOne);
  await runLimited(videoFiles, getVideoIngestConcurrency(), processOne);
  await runLimited(otherFiles, 1, processOne);

  // Step 3: Run matching
  report('Matching GPS tracks...', 0.92);
  try {
    await runMatching(projectId);
  } catch (e) {
    errors.push(`Matching failed: ${e.message}`);
  }

  report('Done', 1);
  return { files: insertedFiles, errors };
}


// ── CSV processing ────────────────────────────────────────────────────

async function processCsvFile(projectId, fileRec) {
  const text = await FM.readFileAsText(fileRec.id);
  if (!text) throw new Error('Could not read file');

  const { points, metadata } = parseCsvTrack(text, fileRec.filename);
  if (points.length === 0) {
    throw new Error(`No GPS points found (${metadata.reason || 'unknown reason'})`);
  }

  await DB.upsertTrack({
    file_id: fileRec.id,
    project_id: projectId,
    kind: 'csv',
    points,
    meta: metadata,
  });
}


// ── Video processing ──────────────────────────────────────────────────

/**
 * Probe a video file for its duration using a temporary <video> element.
 * @param {File} file
 * @returns {Promise<number|null>} duration in seconds, or null
 */
function probeVideoDuration(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    let resolved = false;
    const done = (val) => { if (resolved) return; resolved = true; try { URL.revokeObjectURL(url); } catch {} resolve(val); };
    video.onloadedmetadata = () => {
      const d = video.duration;
      done(Number.isFinite(d) && d > 0 ? d : null);
    };
    video.onerror = (e) => { console.warn('probeVideoDuration error:', e); done(null); };
    // Timeout fallback — some files may hang
    setTimeout(() => { if (!resolved) console.warn('probeVideoDuration timed out for', file.name); done(null); }, 15000);
    video.src = url;
    video.load(); // Explicit load — required when element is not in DOM
  });
}

async function processVideoFile(projectId, fileRec, onProgress) {
  const file = FM.getFileRef(fileRec.id);
  if (!file) throw new Error('Video file not accessible — please re-pick the file');

  // Probe video metadata (duration + origin timestamp from File object)
  const duration = await probeVideoDuration(file);
  let appleMeta = null;
  try {
    appleMeta = await parseAppleMobileVideoMetadata(file);
  } catch (err) {
    console.warn(`[Pipeline] Apple metadata probe skipped for ${fileRec.filename}:`, err);
  }
  const lastModTsRaw = file.lastModified ? file.lastModified / 1000 : null;
  const lastModTs = normalizeEpochSeconds(lastModTsRaw);
  const embeddedStartTs = normalizeEpochSeconds(appleMeta?.captureStartTs);
  const playbackOnly = !!appleMeta?.isLikelyIphonePlaybackVideo && embeddedStartTs != null;
  const estStartTs = embeddedStartTs ?? ((lastModTs && duration) ? lastModTs - duration : lastModTs);
  const captureTsSource = embeddedStartTs != null
    ? String(appleMeta?.captureTsSource || 'embedded')
    : (lastModTs != null ? 'file_last_modified' : null);

  if (playbackOnly) {
    const csvRange = await getProjectCsvTimeRange(projectId);
    if (csvRange) {
      const phoneStartTs = embeddedStartTs;
      const phoneEndTs = embeddedStartTs + (Number.isFinite(Number(duration)) && Number(duration) > 0 ? Number(duration) : 0);
      if (!rangesOverlap(phoneStartTs, phoneEndTs, csvRange.minTs, csvRange.maxTs)) {
        const err = new Error(
          `Rejected phone media because ${new Date(phoneStartTs * 1000).toLocaleString()} ` +
          `falls outside the imported CSV time range ` +
          `(${new Date(csvRange.minTs * 1000).toLocaleString()} - ${new Date(csvRange.maxTs * 1000).toLocaleString()})`
        );
        err.code = 'playback_only_outside_csv_range';
        throw err;
      }
    }
  }

  console.log(
    `[Pipeline] probed ${fileRec.filename}: duration=${duration}, ` +
    `lastMod=${lastModTs}, embeddedStart=${embeddedStartTs}, estStart=${estStartTs}, playbackOnly=${playbackOnly}`
  );
  await DB.updateFileFields(fileRec.id, {
    duration_sec: duration || null,
    last_modified_ts: lastModTs,
    est_start_ts: estStartTs,
    capture_start_ts: embeddedStartTs,
    capture_ts_source: captureTsSource,
    playback_only: playbackOnly,
    external_playback: playbackOnly,
    device_make: appleMeta?.make || null,
    device_model: appleMeta?.model || null,
  });

  try {
    const { parseGoProVideoTrack } = await import('./gopro-gps.js?v=20260602speed2');
    const { points, metadata } = await parseGoProVideoTrack(file, onProgress);
    const sanePoints = sanitizeVideoGpsPoints(points, duration);
    if (sanePoints.length > 0) {
      console.log(`[Pipeline] extracted ${sanePoints.length} ${metadata?.gps_format || 'GPS'} point(s) from ${fileRec.filename}`);
      const deviceUpdates = {};
      if (metadata?.device_make) deviceUpdates.device_make = metadata.device_make;
      if (metadata?.device_model) deviceUpdates.device_model = metadata.device_model;
      if (Object.keys(deviceUpdates).length) {
        await DB.updateFileFields(fileRec.id, deviceUpdates);
      }
      await DB.upsertTrack({
        file_id: fileRec.id,
        project_id: projectId,
        kind: 'video',
        points: sanePoints,
        meta: {
          ...metadata,
          point_count_raw: points.length,
          point_count_sane: sanePoints.length,
        },
      });
    } else if (points.length > 0) {
      console.warn(`[Pipeline] ignoring invalid GPS timeline for ${fileRec.filename}; falling back to file metadata timestamps`);
    } else {
      console.warn(`[Pipeline] no GoPro GPS points found for ${fileRec.filename}; video can play, but CSV matching needs GPS telemetry`);
    }
  } catch (e) {
    console.warn(`GPS extraction skipped for ${fileRec.filename}: ${e.message}`);
  }
}


// ── Matching ──────────────────────────────────────────────────────────

/**
 * Run GPS matching for all video↔CSV track pairs in a project.
 */
export async function runMatching(projectId) {
  const files = await DB.listFiles(projectId);
  const playbackOnlyIds = new Set(
    files
      .filter(file => file.kind === 'video' && isStoredPlaybackOnlyVideo(file))
      .map(file => String(file.id))
  );
  const allTracks = await DB.listTracks(projectId);
  const videoTracks = [];
  const csvTracks = [];

  for (const track of allTracks) {
    const points = await DB.getTrackPoints(track.id);
    const file = files.find(f => f.id === track.file_id);
    const entry = {
      file_id: track.file_id,
      points,
      filename: file?.filename || '',
      meta: track.meta || {},
    };
    if (track.kind === 'video') {
      if (!playbackOnlyIds.has(String(track.file_id))) videoTracks.push(entry);
    } else if (track.kind === 'csv') {
      csvTracks.push(entry);
    }
  }

  if (videoTracks.length === 0 || csvTracks.length === 0) {
    await DB.replaceMatches(projectId, []);
    return;
  }

  const continuousCsvTracks = csvTracks.filter(isContinuousCsvTrack);
  let matches = [];
  if (continuousCsvTracks.length > 0) {
    const nonContinuousCsvTracks = csvTracks.filter(track => !isContinuousCsvTrack(track));
    for (const videoTrack of videoTracks) {
      const contMatches = matchVideoTracksToCsv([videoTrack], continuousCsvTracks);
      matches.push(...(contMatches.length ? contMatches : matchVideoTracksToCsv([videoTrack], nonContinuousCsvTracks)));
    }
  } else {
    matches = matchVideoTracksToCsv(videoTracks, csvTracks);
  }
  await DB.replaceMatches(projectId, matches);
}


// ── Build map data (replaces /api/projects/{id}/map-data) ─────────────

/**
 * Build the map data structure that app.js expects.
 * @param {string} projectId
 * @returns {object} — same shape as the server's MapDataResponse
 */
export async function buildMapData(projectId) {
  let files = await DB.listFiles(projectId);
  if (files.length) {
    files = await Promise.all(files.map(file => maybeRefreshStoredMobileMetadata(file)));
  }
  for (const file of files) {
    if (file?.kind !== 'video') continue;
    if (!file.playback_only || isStoredPlaybackOnlyVideo(file)) continue;
    await DB.updateFileFields(file.id, { playback_only: false, external_playback: false });
    file.playback_only = false;
    file.external_playback = false;
  }
  const tracks = await DB.listTracks(projectId);
  let matches = await DB.listMatches(projectId);
  const playbackOnlyIds = new Set(
    files
      .filter(file => file.kind === 'video' && isStoredPlaybackOnlyVideo(file))
      .map(file => String(file.id))
  );
  // \u2500\u2500 Re-run matching if any video with GPS has no match \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // The matcher now supports epoch-alignment, so re-matching can pick up
  // videos whose GoPro clock was years off (which the original \u00b1300s sweep missed).
  const videoFileIdsWithGps = new Set(
    tracks
      .filter(t => t.kind === 'video' && !playbackOnlyIds.has(String(t.file_id)))
      .map(t => t.file_id)
  );
  const csvFileIdsWithGps = new Set(tracks.filter(t => t.kind === 'csv').map(t => t.file_id));
  const matchedVideoIds = new Set(matches.map(m => m.video_file_id));
  const unmatchedVideoIds = [...videoFileIdsWithGps].filter(id => !matchedVideoIds.has(id));
  if (unmatchedVideoIds.length > 0 && csvFileIdsWithGps.size > 0) {
    console.log(`[Pipeline] ${unmatchedVideoIds.length} video(s) with GPS have no match \u2014 re-running matcher with epoch-alignment`);
    await runMatching(projectId);
    matches = await DB.listMatches(projectId);
  }

  const finalMatchedVideoIds = new Set(matches.map(m => m.video_file_id));
  const finalUnmatchedVideoIds = [...videoFileIdsWithGps].filter(id => !finalMatchedVideoIds.has(id));
  if (finalUnmatchedVideoIds.length > 0) {
    console.log(`[Pipeline] ${finalUnmatchedVideoIds.length} video(s) have no CSV match yet; keeping them on the timeline without CSV overlay`);
  }

  // Build best-match lookup: video_file_id → best csv_file_id
  const bestMatchMap = {};
  for (const m of matches) {
    if (!bestMatchMap[m.video_file_id] || m.rank < bestMatchMap[m.video_file_id].rank) {
      bestMatchMap[m.video_file_id] = m;
    }
  }

  const videos = [];
  const csvs = [];

  // ── Pre-compute CSV epoch range for cross-referencing ──────────
  // This lets us decide whether video GPS timestamps or file.lastModified
  // is more trustworthy when they disagree.
  let csvEpochMin = Infinity, csvEpochMax = -Infinity;
  for (const t of tracks) {
    if (t.kind !== 'csv') continue;
    const s = normalizeEpochSeconds(t.ts_start);
    const e = normalizeEpochSeconds(t.ts_end);
    if (Number.isFinite(s) && s < csvEpochMin) csvEpochMin = s;
    if (Number.isFinite(e) && e > csvEpochMax) csvEpochMax = e;
  }
  const hasCsvEpoch = csvEpochMin < Infinity && csvEpochMax > -Infinity;

  for (const track of tracks) {
    let points = await DB.getTrackPoints(track.id);
    const file = files.find(f => f.id === track.file_id);
    const fileDur = Number(file?.duration_sec);
    const dur = Number.isFinite(fileDur) && fileDur > 0 ? fileDur : null;

    let tsStart = track.ts_start;
    let tsEnd = track.ts_end;
    if (track.kind === 'video') {
      const sane = sanitizeVideoGpsPoints(points, dur);
      if (sane.length >= 2) {
        points = sane;
        tsStart = sane[0].ts;
        tsEnd = sane[sane.length - 1].ts;

        // ── Rebase GPS timestamps using matched CSV offset ───────
        const bestMatch = bestMatchMap[track.file_id];

        // Ensure we do not shift raw GPS time for external (playback only) videos 
        // because we want to use the pure GPS time from the video to place it on the timeline correctly.
        const isExternal = isStoredPlaybackOnlyVideo(file);
        if (!isExternal && bestMatch && Number.isFinite(bestMatch.offset_seconds) && Math.abs(bestMatch.offset_seconds) > 60) {
          const off = bestMatch.offset_seconds;
          console.warn(
            `[Pipeline] rebasing GPS timestamps for ${file?.filename} using matched CSV offset: ` +
            `${off.toFixed(1)}s (GPS epoch ${tsStart}..${tsEnd})`
          );
          for (const p of points) {
            p.ts = p.ts + off;
          }
          tsStart = points[0].ts;
          tsEnd = points[points.length - 1].ts;
        } else {
          // ── Smart rebase: check GPS vs CSV epoch vs lastModified ──
          // GPS timestamps come from GPS satellites (always correct
          // after lock). file.lastModified comes from the camera's
          // internal clock (can be wrong, e.g. 2016 on a 2026 GoPro).
          // If CSVs exist, compare both against the CSV epoch to decide
          // which is trustworthy.
          const fileLastMod = normalizeEpochSeconds(file?.last_modified_ts);
          const MAX_DRIFT = 7 * 86400; // 7 days
          const gpsNearCsv = hasCsvEpoch && Math.abs(tsEnd - csvEpochMax) < MAX_DRIFT;
          const lastModNearCsv = hasCsvEpoch && fileLastMod != null && Math.abs(fileLastMod - csvEpochMax) < MAX_DRIFT;

          if (gpsNearCsv) {
            // GPS timestamps are close to CSV epoch → GPS is correct, keep as-is
            // (This handles wrong-clock GoPros where GPS satellite time is 2026
            //  but File.lastModified is 2016 due to misconfigured camera clock)
            console.log(
              `[Pipeline] GPS timestamps for ${file?.filename} are near CSV epoch ` +
              `(${tsStart.toFixed(0)}..${tsEnd.toFixed(0)} vs CSV ${csvEpochMin.toFixed(0)}..${csvEpochMax.toFixed(0)}) — keeping as-is`
            );
          } else if (lastModNearCsv && fileLastMod != null && Math.abs(tsEnd - fileLastMod) > MAX_DRIFT) {
            // lastModified is close to CSV but GPS is far → GPS clock was wrong
            const estStart = (dur != null) ? fileLastMod - dur : fileLastMod;
            console.warn(
              `[Pipeline] rebasing GPS timestamps for ${file?.filename} using lastModified: ` +
              `GPS epoch ${tsStart}..${tsEnd} vs lastMod ${fileLastMod} (drift=${Math.abs(tsEnd - fileLastMod).toFixed(0)}s)`
            );
            for (const p of points) {
              if (Number.isFinite(p.video_s)) {
                p.ts = estStart + p.video_s;
              }
            }
            tsStart = points[0].ts;
            tsEnd = points[points.length - 1].ts;
          } else if (!hasCsvEpoch && fileLastMod != null && Math.abs(tsEnd - fileLastMod) > MAX_DRIFT) {
            // No CSV reference — fall back to lastModified rebase as last resort
            const estStart = (dur != null) ? fileLastMod - dur : fileLastMod;
            console.warn(
              `[Pipeline] rebasing GPS timestamps for ${file?.filename} using lastModified (no CSV ref): ` +
              `GPS epoch ${tsStart}..${tsEnd} vs lastMod ${fileLastMod} (drift=${Math.abs(tsEnd - fileLastMod).toFixed(0)}s)`
            );
            for (const p of points) {
              if (Number.isFinite(p.video_s)) {
                p.ts = estStart + p.video_s;
              }
            }
            tsStart = points[0].ts;
            tsEnd = points[points.length - 1].ts;
          }
          // else: GPS and lastModified are close enough — no rebase needed
        }
      } else {
        const r = sanitizeVideoRange(track.ts_start, track.ts_end, dur);
        tsStart = r.start;
        tsEnd = r.end;
      }
    }

    const telemetryPoints = points;

    // Subsample for map display
    if (points.length > MAP_MAX_POINTS_PER_TRACK) {
      const step = (points.length - 1) / (MAP_MAX_POINTS_PER_TRACK - 1);
      const sub = [];
      for (let i = 0; i < MAP_MAX_POINTS_PER_TRACK; i++) {
        sub.push(points[Math.round(i * step)]);
      }
      points = sub;
    }

    const playbackOnly = isStoredPlaybackOnlyVideo(file);
    const mapTrack = {
      id: track.file_id,
      file_id: track.file_id,
      filename: file?.filename ?? 'unknown',
      point_count: track.point_count,
      ts_start: tsStart,
      ts_end: tsEnd,
      size_bytes: file?.size_bytes ?? 0,
      points,
      telemetry_points: telemetryPoints,
      duration_sec: dur,
      playback_only: playbackOnly,
      external_playback: !!file?.external_playback,
      capture_ts_source: file?.capture_ts_source || null,
      capture_start_ts: normalizeEpochSeconds(file?.capture_start_ts),
      device_make: file?.device_make || null,
      device_model: file?.device_model || null,
      track_meta: track?.meta || null,
    };

    if (track.kind === 'video') {
      mapTrack.best_match_csv_id = playbackOnly ? null : (bestMatchMap[track.file_id]?.csv_file_id ?? null);
      videos.push(mapTrack);
    } else {
      csvs.push(mapTrack);
    }
  }

  // Include files that have NO track (e.g. non-GoPro videos, failed CSVs)
  const fileIdsWithTracks = new Set(tracks.map(t => t.file_id));
  for (const file of files) {
    if (fileIdsWithTracks.has(file.id)) continue;
    // For videos without GPS, use estimated timestamps from File.lastModified + duration
    const durN = Number(file.duration_sec);
    const dur = Number.isFinite(durN) && durN > 0 ? durN : null;
    const estStartRaw = file.est_start_ts || file.last_modified_ts || null;
    const estStart = normalizeEpochSeconds(estStartRaw);
    const estEndRaw = (estStart != null && dur != null) ? estStart + dur : null;
    const r = sanitizeVideoRange(estStart, estEndRaw, dur);
    const playbackOnly = isStoredPlaybackOnlyVideo(file);
    const stub = {
      id: file.id,
      file_id: file.id,
      filename: file.filename,
      point_count: 0,
      ts_start: r.start,
      ts_end: r.end,
      points: [],
      size_bytes: file.size_bytes,
      duration_sec: dur,
      has_gps: false,
      playback_only: playbackOnly,
      external_playback: !!file.external_playback,
      capture_ts_source: file.capture_ts_source || null,
      capture_start_ts: normalizeEpochSeconds(file.capture_start_ts),
      device_make: file.device_make || null,
      device_model: file.device_model || null,
    };
    if (file.kind === 'video') {
      stub.best_match_csv_id = null;
      videos.push(stub);
    } else if (file.kind === 'csv') {
      csvs.push(stub);
    }
  }

  return {
    project_id: projectId,
    videos,
    csvs,
    matches,
    files,
  };
}


// ── Build CV statuses (replaces /api/projects/{id}/cv/statuses) ───────

export async function buildCvStatuses(projectId) {
  const runs = await DB.listCvRuns(projectId);
  const result = {};
  for (const run of runs) {
    result[run.file_id] = run;
  }
  return result;
}


// ── Delete segment (file + associated data) ───────────────────────────

export async function deleteSegment(projectId, fileId) {
  await DB.deleteFile(projectId, fileId);
  // Re-run matching since tracks changed
  await runMatching(projectId);
}
