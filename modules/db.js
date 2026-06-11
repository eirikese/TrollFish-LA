/**
 * TrollFish — db.js
 * Browser-side IndexedDB storage using Dexie.js
 *
 * Mirrors the SQLite schema from the Python server:
 *   projects, files, tracks, matches, cvConfig, cvRuns, athletes, segments
 *
 * Usage:
 *   import { db } from './db.js';
 *   const projects = await db.projects.toArray();
 */

import Dexie from '../vendor/dexie.mjs';

/** @type {Dexie} */
export const db = new Dexie('TrollFish');

// ── Schema definition ─────────────────────────────────────────────────
// Dexie auto-creates the primary key from the first column.
// Only indexed fields need to be listed; all other properties are stored
// but not indexed.

db.version(1).stores({
  // project_id is auto-generated UUID
  projects: 'id, name, created_at',

  // files: video & CSV registrations (no file content stored here)
  files: 'id, project_id, kind, filename, status, created_at',

  // GPS tracks extracted from files (one track per file)
  tracks: 'id, file_id, project_id, kind, [project_id+kind]',

  // GPS track points stored separately for large arrays
  // key = track_id, value = points array
  trackPoints: 'track_id',

  // video↔CSV alignment matches
  matches: 'id, project_id, video_file_id, csv_file_id, [project_id+video_file_id]',

  // per-project CV config (one row per project)
  cvConfig: 'project_id',

  // per-file CV run status
  cvRuns: 'id, [project_id+file_id], project_id, status',

  // athletes per project
  athletes: 'id, project_id',

  // segments per project (absolute epoch timestamp ranges)
  segments: 'id, project_id',

  // per-file metadata (athlete assignment, etc.)
  fileMeta: '[project_id+file_id], project_id',

  // persistent File System Access API handles (Chrome/Edge only)
  fileHandles: 'file_id',
});

db.version(2).stores({
  projects: 'id, name, created_at',
  files: 'id, project_id, kind, filename, status, created_at',
  tracks: 'id, file_id, project_id, kind, [project_id+kind]',
  trackPoints: 'track_id',
  matches: 'id, project_id, video_file_id, csv_file_id, [project_id+video_file_id]',
  cvConfig: 'project_id',
  cvRuns: 'id, [project_id+file_id], project_id, status',
  athletes: 'id, project_id',
  segments: 'id, project_id',
  fileMeta: '[project_id+file_id], project_id',
  fileHandles: 'file_id',
  maneuvers: 'id, project_id, athlete_id, type, anchor_ts, [project_id+anchor_ts], [project_id+athlete_id], [project_id+type]',
  maneuverAnalyses: 'maneuver_id, project_id, generated_at, [project_id+maneuver_id]',
});

db.version(3).stores({
  projects: 'id, name, created_at',
  files: 'id, project_id, kind, filename, status, created_at',
  tracks: 'id, file_id, project_id, kind, [project_id+kind]',
  trackPoints: 'track_id',
  matches: 'id, project_id, video_file_id, csv_file_id, [project_id+video_file_id]',
  cvConfig: 'project_id',
  cvRuns: 'id, [project_id+file_id], project_id, status',
  athletes: 'id, project_id',
  segments: 'id, project_id',
  fileMeta: '[project_id+file_id], project_id',
  fileHandles: 'file_id',
  maneuvers: 'id, project_id, athlete_id, type, anchor_ts, [project_id+anchor_ts], [project_id+athlete_id], [project_id+type]',
  maneuverAnalyses: 'maneuver_id, project_id, generated_at, [project_id+maneuver_id]',
  windEstimates: '[project_id+csv_file_id], project_id',
});


// ── Helper: generate UUID ─────────────────────────────────────────────
export function uuid() {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();

  // Fallback for browsers/contexts without crypto.randomUUID()
  const bytes = new Uint8Array(16);
  if (c && typeof c.getRandomValues === 'function') {
    c.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  }

  // RFC 4122 v4 bits
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = [...bytes].map(b => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}

/** ISO string without microseconds */
export function nowISO() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}


// ═══════════════════════════════════════════════════════════════════════
// Convenience methods (mirror the Python MetadataStore API)
// ═══════════════════════════════════════════════════════════════════════

// ── Projects ──────────────────────────────────────────────────────────

export async function createProject(name) {
  const id = uuid();
  const created_at = nowISO();
  await db.projects.add({ id, name, created_at });
  return { id, name, created_at };
}

export async function getProject(projectId) {
  return db.projects.get(projectId) ?? null;
}

export async function listProjects() {
  return db.projects.orderBy('created_at').reverse().toArray();
}

export async function deleteProject(projectId) {
  await db.transaction('rw',
    db.projects, db.files, db.tracks, db.trackPoints, db.matches,
    db.cvConfig, db.cvRuns, db.athletes, db.segments, db.fileMeta,
    db.maneuvers, db.maneuverAnalyses, db.windEstimates,
    async () => {
      // Gather dependent IDs
      const fileIds = (await db.files.where('project_id').equals(projectId).toArray()).map(f => f.id);
      const trackIds = (await db.tracks.where('project_id').equals(projectId).toArray()).map(t => t.id);

      await db.trackPoints.where('track_id').anyOf(trackIds).delete();
      await db.cvRuns.where('project_id').equals(projectId).delete();
      await db.matches.where('project_id').equals(projectId).delete();
      await db.tracks.where('project_id').equals(projectId).delete();
      await db.files.where('project_id').equals(projectId).delete();
      await db.athletes.where('project_id').equals(projectId).delete();
      await db.segments.where('project_id').equals(projectId).delete();
      await db.fileMeta.where('project_id').equals(projectId).delete();
      await db.maneuvers.where('project_id').equals(projectId).delete();
      await db.maneuverAnalyses.where('project_id').equals(projectId).delete();
      await db.windEstimates.where('project_id').equals(projectId).delete();
      await db.cvConfig.delete(projectId);
      await db.projects.delete(projectId);
    }
  );
}


// ── Files ─────────────────────────────────────────────────────────────

/**
 * Register a file (video or CSV) in the project.
 * No actual file content is stored — just metadata.
 * @param {{id?:string, project_id:string, filename:string, kind:string, size_bytes:number, status?:string}} opts
 */
export async function insertFile({ id, project_id, filename, kind, size_bytes, status = 'uploaded' }) {
  const fileId = id || uuid();
  const created_at = nowISO();
  const rec = { id: fileId, project_id, filename, kind, size_bytes, status, error: null, created_at };
  await db.files.add(rec);
  return rec;
}

export async function listFiles(projectId) {
  return db.files.where('project_id').equals(projectId).sortBy('created_at');
}

export async function getFile(fileId) {
  return db.files.get(fileId) ?? null;
}

export async function updateFileStatus(fileId, status, error = null) {
  await db.files.update(fileId, { status, error });
}

/** Update arbitrary fields on a file record (e.g. duration_sec, last_modified_ts). */
export async function updateFileFields(fileId, fields) {
  await db.files.update(fileId, fields);
}

export async function deleteFile(projectId, fileId) {
  await db.transaction('rw',
    db.files, db.tracks, db.trackPoints, db.matches, db.cvRuns, db.fileMeta,
    async () => {
      const track = await db.tracks.where('file_id').equals(fileId).first();
      if (track) {
        await db.trackPoints.delete(track.id);
        await db.tracks.delete(track.id);
      }
      await db.matches.where('project_id').equals(projectId).filter(m =>
        m.video_file_id === fileId || m.csv_file_id === fileId
      ).delete();
      await db.cvRuns.where({ project_id: projectId, file_id: fileId }).delete();
      await db.fileMeta.delete([projectId, fileId]);
      await db.files.delete(fileId);
    }
  );
}


// ── Tracks ────────────────────────────────────────────────────────────

/**
 * Insert or update a GPS track. Points are stored in a separate table.
 * @param {{file_id:string, project_id:string, kind:string, points:object[], meta:object}} opts
 */
export async function upsertTrack({ file_id, project_id, kind, points, meta }) {
  const now = nowISO();
  const point_count = points.length;
  const ts_start = point_count > 0 ? points[0].ts : null;
  const ts_end = point_count > 0 ? points[point_count - 1].ts : null;

  // Check for existing track for this file
  const existing = await db.tracks.where('file_id').equals(file_id).first();
  const id = existing ? existing.id : uuid();

  const trackRow = {
    id, file_id, project_id, kind, point_count,
    ts_start, ts_end, meta, created_at: existing?.created_at ?? now, updated_at: now,
  };

  await db.transaction('rw', db.tracks, db.trackPoints, async () => {
    await db.tracks.put(trackRow);
    await db.trackPoints.put({ track_id: id, points });
  });

  return trackRow;
}

export async function listTracks(projectId, kind = null) {
  if (kind) {
    return db.tracks.where({ project_id: projectId, kind }).toArray();
  }
  return db.tracks.where('project_id').equals(projectId).toArray();
}

export async function getTrackByFileId(fileId) {
  return db.tracks.where('file_id').equals(fileId).first() ?? null;
}

export async function getTrackPoints(trackId) {
  const rec = await db.trackPoints.get(trackId);
  return rec?.points ?? [];
}


// ── Matches ───────────────────────────────────────────────────────────

export async function replaceMatches(projectId, matches) {
  const created_at = nowISO();
  await db.transaction('rw', db.matches, async () => {
    await db.matches.where('project_id').equals(projectId).delete();
    if (matches.length === 0) return;
    const rows = matches.map(m => ({
      id: uuid(),
      project_id: projectId,
      ...m,
      created_at,
    }));
    await db.matches.bulkAdd(rows);
  });
}

export async function listMatches(projectId) {
  return db.matches.where('project_id').equals(projectId).toArray();
}


// ── CV Config ─────────────────────────────────────────────────────────

export async function getCvConfig(projectId) {
  return db.cvConfig.get(projectId) ?? null;
}

export async function upsertCvConfig(projectId, config) {
  const updated_at = nowISO();
  await db.cvConfig.put({ project_id: projectId, config, updated_at });
  return { project_id: projectId, config, updated_at };
}


// ── CV Runs ───────────────────────────────────────────────────────────

export async function upsertCvRun(projectId, fileId, fields) {
  const now = nowISO();
  const existing = await db.cvRuns.where({ project_id: projectId, file_id: fileId }).first();
  const id = existing ? existing.id : uuid();
  const row = {
    ...existing,
    id,
    project_id: projectId,
    file_id: fileId,
    created_at: existing?.created_at ?? now,
    ...fields,
  };
  await db.cvRuns.put(row);
  return row;
}

export async function getCvRun(projectId, fileId) {
  return db.cvRuns.where({ project_id: projectId, file_id: fileId }).first() ?? null;
}

export async function listCvRuns(projectId) {
  return db.cvRuns.where('project_id').equals(projectId).toArray();
}


// ── Athletes ──────────────────────────────────────────────────────────

export async function getAthletes(projectId) {
  return db.athletes.where('project_id').equals(projectId).toArray();
}

export async function saveAthletes(projectId, athletesList) {
  await db.transaction('rw', db.athletes, async () => {
    await db.athletes.where('project_id').equals(projectId).delete();
    if (athletesList.length === 0) return;
    const rows = athletesList.map(a => ({
      ...a,
      id: a.id || uuid(),
      project_id: projectId,
    }));
    await db.athletes.bulkAdd(rows);
  });
}


// ── Segments ──────────────────────────────────────────────────────────

export async function getSegments(projectId) {
  return db.segments.where('project_id').equals(projectId).toArray();
}

export async function saveSegments(projectId, segmentsList) {
  await db.transaction('rw', db.segments, async () => {
    await db.segments.where('project_id').equals(projectId).delete();
    if (segmentsList.length === 0) return;
    const rows = segmentsList.map(s => ({
      ...s,
      id: s.id || uuid(),
      project_id: projectId,
    }));
    await db.segments.bulkAdd(rows);
  });
}


// ── Maneuvers ─────────────────────────────────────────────────────────

export async function getManeuvers(projectId) {
  return db.maneuvers
    .where('project_id')
    .equals(projectId)
    .sortBy('anchor_ts');
}

export async function saveManeuvers(projectId, maneuverRows) {
  await db.transaction('rw', db.maneuvers, async () => {
    await db.maneuvers.where('project_id').equals(projectId).delete();
    if (!Array.isArray(maneuverRows) || maneuverRows.length === 0) return;
    const rows = maneuverRows.map(row => ({
      ...row,
      id: row.id || uuid(),
      project_id: projectId,
    }));
    await db.maneuvers.bulkPut(rows);
  });
}

export async function getManeuverAnalysis(projectId, maneuverId) {
  const direct = await db.maneuverAnalyses.get(String(maneuverId));
  if (direct && String(direct.project_id) === String(projectId)) return direct;
  return db.maneuverAnalyses
    .where('[project_id+maneuver_id]')
    .equals([projectId, String(maneuverId)])
    .first() ?? null;
}

export async function saveManeuverAnalysis(projectId, maneuverId, analysis) {
  const row = {
    ...analysis,
    maneuver_id: String(maneuverId),
    project_id: projectId,
    generated_at: analysis?.generated_at || nowISO(),
  };
  await db.maneuverAnalyses.put(row);
  return row;
}


// ── File metadata ─────────────────────────────────────────────────────

export async function getFileMeta(projectId) {
  const rows = await db.fileMeta.where('project_id').equals(projectId).toArray();
  const result = {};
  for (const row of rows) result[row.file_id] = row.meta;
  return result;
}

export async function setFileMeta(projectId, fileId, meta) {
  await db.fileMeta.put({ project_id: projectId, file_id: fileId, meta });
  return meta;
}

// ── Wind estimates ───────────────────────────────────────────────────

export async function getWindEstimate(projectId, csvFileId) {
  return db.windEstimates.get([projectId, csvFileId]) ?? null;
}

export async function upsertWindEstimate(projectId, csvFileId, signature, estimate) {
  const row = {
    project_id: projectId,
    csv_file_id: csvFileId,
    signature: String(signature || ''),
    estimate,
    updated_at: nowISO(),
  };
  await db.windEstimates.put(row);
  return row;
}
