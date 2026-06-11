/**
 * TrollFish — file-manager.js
 * Hybrid file handle manager for video files.
 *
 * Chrome/Edge: Uses File System Access API for persistent file handles
 *              stored in IndexedDB. Videos never need re-picking.
 *
 * Safari/iPad: No persistent handles. Stores file identity {name, size,
 *              lastModified} and prompts user to re-pick when needed.
 *
 * CSV files are small and read once, so they're stored as text in IndexedDB.
 */

import { db, uuid } from './db.js';
import * as Storage from './storage.js';

// ── Feature detection ─────────────────────────────────────────────────

/** True if the browser supports persistent file handles (Chrome/Edge) */
export const HAS_FILE_SYSTEM_ACCESS = typeof window.showOpenFilePicker === 'function';

/**
 * True if OPFS is available for blob persistence.
 * Set lazily on first call to persistToOPFS / restoreFromOPFS.
 */
let _opfsAvailable = null;
const forgottenFileIds = new Set();
async function _checkOPFS() {
  if (_opfsAvailable !== null) return _opfsAvailable;
  _opfsAvailable = await Storage.isAvailable();
  return _opfsAvailable;
}
let _opfsPersistQueue = Promise.resolve();

/**
 * Persist a video/CSV File to OPFS so it survives page refresh.
 * Fire-and-forget: errors are non-fatal.
 * @param {string} fileId
 * @param {File} file
 */
async function persistToOPFS(fileId, file) {
  try {
    if (forgottenFileIds.has(fileId)) return;
    if (!await _checkOPFS()) return;
    // Write the Blob directly to avoid full in-memory ArrayBuffer copies.
    await Storage.writeFile(['_blobs'], `${fileId}.bin`, file);
    if (forgottenFileIds.has(fileId)) {
      try { await Storage.deleteFile(['_blobs'], `${fileId}.bin`); } catch {}
      return;
    }
    // Store identity metadata alongside
    const meta = JSON.stringify({ name: file.name, type: file.type, lastModified: file.lastModified });
    await Storage.writeFile(['_blobs'], `${fileId}.meta`, meta);
    if (forgottenFileIds.has(fileId)) {
      try { await Storage.deleteFile(['_blobs'], `${fileId}.bin`); } catch {}
      try { await Storage.deleteFile(['_blobs'], `${fileId}.meta`); } catch {}
      return;
    }
    console.log(`[FM] persisted ${file.name} (${(file.size/1e6).toFixed(1)} MB) to OPFS`);
  } catch (e) {
    console.warn('[FM] OPFS persist failed:', e);
  }
}

function queuePersistToOPFS(fileId, file) {
  _opfsPersistQueue = _opfsPersistQueue
    .then(() => persistToOPFS(fileId, file))
    .catch(() => {});
  return _opfsPersistQueue;
}

/**
 * Restore a File from OPFS if it exists.
 * @param {string} fileId
 * @returns {File|null}
 */
async function restoreFromOPFS(fileId) {
  try {
    if (!await _checkOPFS()) return null;
    const metaText = await Storage.readFileText(['_blobs'], `${fileId}.meta`);
    if (!metaText) return null;
    const meta = JSON.parse(metaText);
    const blob = await Storage.readFileBlob(['_blobs'], `${fileId}.bin`);
    if (!blob) return null;
    const file = new File([blob], meta.name, { type: meta.type, lastModified: meta.lastModified });
    console.log(`[FM] restored ${file.name} from OPFS`);
    return file;
  } catch {
    return null;
  }
}

// ── Internal handle store (IndexedDB via separate Dexie table) ────────
// We store file handles in the existing db but handles are a special
// type — they need permission re-checking after page reload.

const handleCache = new Map(); // fileId -> { handle: FileSystemFileHandle, file: File }

/**
 * Store a FileSystemFileHandle for later use (Chrome only).
 * @param {string} fileId — our internal file UUID
 * @param {FileSystemFileHandle} handle
 */
async function storeHandle(fileId, handle) {
  // NOTE: do NOT overwrite handleCache here — registerPickedFiles already set it
  // with both handle AND file. We only persist the handle to IDB.
  try {
    await db.table('fileHandles').put({ file_id: fileId, handle });
  } catch {
    // Table may not exist yet on first run — that's ok, we'll create it
  }
}

/**
 * Retrieve a file from its handle. On Chrome, verifies permission.
 * @param {string} fileId
 * @returns {File|null}
 */
export async function getFileForReading(fileId) {
  // 1. Check memory cache
  let entry = handleCache.get(fileId);
  if (entry?.file) return entry.file;

  // 2. Try to get handle from IDB (Chrome/Edge persistent handles)
  if (!entry?.handle) {
    try {
      const stored = await db.table('fileHandles').get(fileId);
      if (stored?.handle) {
        entry = { handle: stored.handle, file: null };
        handleCache.set(fileId, entry);
      }
    } catch {
      // Table missing or handle missing
    }
  }

  if (entry?.handle) {
    // 3. Verify/request permission on the handle
    const handle = entry.handle;
    try {
      const perm = await handle.queryPermission({ mode: 'read' });
      if (perm === 'granted') {
        const file = await handle.getFile();
        entry.file = file;
        return file;
      }
      const req = await handle.requestPermission({ mode: 'read' });
      if (req === 'granted') {
        const file = await handle.getFile();
        entry.file = file;
        return file;
      }
    } catch {
      // Permission denied or handle stale — fall through to OPFS
    }
  }

  // 4. Try OPFS blob restoration (works on all browsers)
  const restored = await restoreFromOPFS(fileId);
  if (restored) {
    if (!entry) entry = { handle: null, file: restored };
    else entry.file = restored;
    handleCache.set(fileId, entry);
    return restored;
  }

  return null;
}


/**
 * Register files from the user's file picker or drop zone.
 * Works on both Chrome (persistent handles) and Safari (File objects only).
 *
 * @param {FileList|File[]} files — from input.files or DataTransfer.files
 * @param {FileSystemFileHandle[]|null} handles — from showOpenFilePicker (Chrome only)
 * @returns {{ file: File, fileId: string, handle?: FileSystemFileHandle }[]}
 */
export function registerPickedFiles(files, handles = null) {
  const results = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const fileId = uuid();
    const handle = handles ? handles[i] : null;
    forgottenFileIds.delete(fileId);

    if (handle) {
      handleCache.set(fileId, { handle, file });
      // Fire-and-forget persist handle to IDB
      storeHandle(fileId, handle).catch(() => {});
    } else {
      // No handle — cache the File object
      handleCache.set(fileId, { handle: null, file });
    }

    // Persist the actual blob to OPFS (works on all browsers)
    // Video files survive page refresh this way.
    queuePersistToOPFS(fileId, file);

    results.push({ file, fileId, handle });
  }
  return results;
}


/**
 * Read a file as text (for CSV parsing).
 * @param {string} fileId
 * @returns {string|null}
 */
export async function readFileAsText(fileId) {
  const file = await getFileForReading(fileId);
  if (!file) return null;
  return file.text();
}

/**
 * Create an object URL for video playback.
 * Caller must revoke when done.
 * @param {string} fileId
 * @returns {string|null} — blob: URL
 */
export async function createVideoURL(fileId) {
  const file = await getFileForReading(fileId);
  if (!file) return null;
  return URL.createObjectURL(file);
}

/**
 * Get raw File reference if still in memory (same session).
 * Used for GoPro GPS extraction that needs ArrayBuffer chunks.
 * @param {string} fileId
 * @returns {File|null}
 */
export function getFileRef(fileId) {
  const entry = handleCache.get(fileId);
  return entry?.file ?? null;
}

/**
 * Check whether a file is still accessible.
 * On Safari, this is only valid within the same session.
 * On Chrome, handles persist across sessions.
 * @param {string} fileId
 * @returns {boolean}
 */
export function isFileAvailable(fileId) {
  const entry = handleCache.get(fileId);
  if (!entry) return false;
  return !!(entry.file || entry.handle);
}

/**
 * Get file identity info for display / re-pick matching.
 * @param {string} fileId
 * @returns {{ name: string, size: number, lastModified: number }|null}
 */
export async function getFileIdentity(fileId) {
  const file = await getFileForReading(fileId);
  if (!file) return null;
  return { name: file.name, size: file.size, lastModified: file.lastModified };
}

/**
 * Return file IDs that have NO File object in-memory (stale after refresh).
 * Works on all browsers — does not require File System Access API.
 * @param {string[]} fileIds
 * @returns {string[]}
 */
export function getStaleFileIds(fileIds) {
  return fileIds.filter(fid => {
    const entry = handleCache.get(fid);
    return !entry?.file;
  });
}

/**
 * Re-pick files by matching user-selected File objects to known DB records.
 * Used on Firefox/Safari where persistent handles don't exist.
 *
 * @param {File[]} files — freshly picked files from <input> or showOpenFilePicker
 * @param {{ id: string, filename: string, size_bytes: number }[]} dbRecords — known files from DB
 * @returns {{ reconnected: string[], unmatched: string[] }}
 */
export function reconnectByRepick(files, dbRecords) {
  const reconnected = [];
  const unmatched = [];
  // Build lookup from dbRecords
  const byNameSize = new Map();
  for (const rec of dbRecords) {
    const key = `${rec.filename}|${rec.size_bytes}`;
    byNameSize.set(key, rec);
  }
  for (const file of files) {
    const key = `${file.name}|${file.size}`;
    const rec = byNameSize.get(key);
    if (rec) {
      handleCache.set(rec.id, { handle: null, file });
      reconnected.push(rec.id);
      byNameSize.delete(key); // consume match
    }
  }
  // Any DB records not matched
  for (const rec of byNameSize.values()) {
    unmatched.push(rec.id);
  }
  return { reconnected, unmatched };
}

/**
 * Silently reconnect handles that are already granted (no user gesture needed).
 * Call this early on page load to restore files whose permission persists.
 *
 * @param {string[]} fileIds
 * @returns {{ reconnected: string[], needsGesture: string[] }}
 */
export async function autoReconnect(fileIds) {
  const reconnected = [];
  const needsGesture = [];
  for (const fid of fileIds) {
    let entry = handleCache.get(fid);
    if (entry?.file) { reconnected.push(fid); continue; }
    // Try loading from IDB if not in memory (Chrome/Edge handles)
    if (!entry?.handle) {
      try {
        const stored = await db.table('fileHandles').get(fid);
        if (stored?.handle) {
          entry = { handle: stored.handle, file: null };
          handleCache.set(fid, entry);
        }
      } catch { /* table missing */ }
    }
    // Try handle permission (Chrome/Edge)
    if (entry?.handle) {
      try {
        const perm = await entry.handle.queryPermission({ mode: 'read' });
        if (perm === 'granted') {
          entry.file = await entry.handle.getFile();
          reconnected.push(fid);
          continue;
        }
      } catch { /* stale handle */ }
    }
    // Try OPFS blob (all browsers)
    const restored = await restoreFromOPFS(fid);
    if (restored) {
      if (!entry) entry = { handle: null, file: restored };
      else entry.file = restored;
      handleCache.set(fid, entry);
      reconnected.push(fid);
      continue;
    }
    needsGesture.push(fid);
  }
  return { reconnected, needsGesture };
}

/**
 * Reconnect file handles after a page refresh.
 * Must be called from a user gesture (click handler) because
 * requestPermission() requires transient activation.
 *
 * @param {string[]} fileIds — IDs of files to reconnect
 * @returns {{ reconnected: string[], failed: string[] }}
 */
export async function reconnectHandles(fileIds) {
  const reconnected = [];
  const failed = [];
  for (const fid of fileIds) {
    let entry = handleCache.get(fid);
    // Try loading from IDB if not in memory
    if (!entry?.handle) {
      try {
        const stored = await db.table('fileHandles').get(fid);
        if (stored?.handle) {
          entry = { handle: stored.handle, file: null };
          handleCache.set(fid, entry);
        }
      } catch { /* table missing */ }
    }
    if (!entry?.handle) { failed.push(fid); continue; }
    try {
      const perm = await entry.handle.queryPermission({ mode: 'read' });
      if (perm === 'granted') {
        entry.file = await entry.handle.getFile();
        reconnected.push(fid);
        continue;
      }
      const req = await entry.handle.requestPermission({ mode: 'read' });
      if (req === 'granted') {
        entry.file = await entry.handle.getFile();
        reconnected.push(fid);
      } else {
        failed.push(fid);
      }
    } catch {
      failed.push(fid);
    }
  }
  return { reconnected, failed };
}

/**
 * Check how many files have stale handles that need reconnection.
 * @param {string[]} fileIds
 * @returns {number} count of files needing reconnection
 */
export async function countStaleHandles(fileIds) {
  let stale = 0;
  for (const fid of fileIds) {
    let entry = handleCache.get(fid);
    if (entry?.file) continue; // already have File object
    // Check IDB for stored handle
    if (!entry?.handle) {
      try {
        const stored = await db.table('fileHandles').get(fid);
        if (stored?.handle) {
          entry = { handle: stored.handle, file: null };
          handleCache.set(fid, entry);
        }
      } catch { /* table missing */ }
    }
    if (entry?.handle) {
      try {
        const perm = await entry.handle.queryPermission({ mode: 'read' });
        if (perm !== 'granted') stale++;
      } catch {
        stale++;
      }
    } else {
      stale++; // no handle at all
    }
  }
  return stale;
}

/**
 * Forget a registered file and remove any persisted blob copies.
 * Best-effort cleanup used when ingest rejects a file after registration.
 * @param {string} fileId
 */
export async function forgetFile(fileId) {
  forgottenFileIds.add(fileId);
  handleCache.delete(fileId);
  try {
    await db.table('fileHandles').delete(fileId);
  } catch {}
  try {
    await Storage.deleteFile(['_blobs'], `${fileId}.bin`);
  } catch {}
  try {
    await Storage.deleteFile(['_blobs'], `${fileId}.meta`);
  } catch {}
}

// ── Initialize fileHandles table (add to Dexie schema in next version) ──
// We do a dynamic upgrade so db.js doesn't need editing
try {
  if (!db.tables.some(t => t.name === 'fileHandles')) {
    // This will naturally happen on first use via version upgrade
    // For now, we handle the missing table gracefully above
  }
} catch {
  // Ignore during module load
}
