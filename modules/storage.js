/**
 * TrollFish — storage.js
 * OPFS (Origin Private File System) wrapper for large binary blobs
 * (skeleton JSONL, metrics CSV, etc.)
 *
 * Falls back to IndexedDB blobs when OPFS is not available (insecure
 * context, iPad Safari over HTTP, etc.).
 */

import Dexie from '../vendor/dexie.mjs';

// ═══════════════════════════════════════════════════════════════════════
// OPFS backend
// ═══════════════════════════════════════════════════════════════════════

let _opfsRoot = null;
let _opfsChecked = false;
let _forceIdbFallback = false;

function _setIdbFallback(reason = '') {
  _forceIdbFallback = true;
  _opfsRoot = null;
  _opfsChecked = true;
  const msg = reason ? ` (${reason})` : '';
  console.warn(`[Storage] Falling back to IndexedDB${msg}`);
}

async function _probeOpfsWritable(root) {
  if (!root) return false;
  const probeName = '.trollfish_opfs_probe';
  try {
    const fileHandle = await root.getFileHandle(probeName, { create: true });

    if (typeof fileHandle.createWritable === 'function') {
      const writable = await fileHandle.createWritable();
      await writable.close();
      try { await root.removeEntry(probeName); } catch {}
      return true;
    }

    if (typeof fileHandle.createSyncAccessHandle === 'function') {
      try {
        const ah = await fileHandle.createSyncAccessHandle();
        ah.close();
        try { await root.removeEntry(probeName); } catch {}
        return true;
      } catch {
        return false;
      }
    }

    return false;
  } catch {
    return false;
  }
}

/** Get the OPFS root directory handle (lazy init) */
async function getRoot() {
  if (_opfsRoot) return _opfsRoot;
  if (_forceIdbFallback) return null;
  if (_opfsChecked) return null; // already tried and failed
  _opfsChecked = true;
  if (typeof navigator !== 'undefined' && navigator.storage?.getDirectory) {
    try {
      _opfsRoot = await navigator.storage.getDirectory();
      const writableOk = await _probeOpfsWritable(_opfsRoot);
      if (!writableOk) {
        _setIdbFallback('OPFS writable stream API unavailable');
        return null;
      }
      return _opfsRoot;
    } catch {
      // OPFS not available, fall through
    }
  }
  console.warn('[Storage] OPFS unavailable — using IndexedDB fallback for blob storage');
  return null;
}

/**
 * Ensure a nested directory path exists inside OPFS.
 * @param {string[]} parts — path segments
 * @returns {FileSystemDirectoryHandle|null}
 */
async function ensureDir(parts) {
  let dir = await getRoot();
  if (!dir) return null; // will use IDB fallback
  for (const segment of parts) {
    dir = await dir.getDirectoryHandle(segment, { create: true });
  }
  return dir;
}


// ═══════════════════════════════════════════════════════════════════════
// IndexedDB fallback backend
// ═══════════════════════════════════════════════════════════════════════

const _idb = new Dexie('TrollFish_Storage');
_idb.version(1).stores({
  blobs: 'path', // path = "dir1/dir2/.../filename"
});
_idb.version(2).stores({
  blobs: 'path',
  chunks: '++id,path,seq,[path+seq]',
});

const _idbTextEncoder = new TextEncoder();
const _idbTextDecoder = new TextDecoder();
const _idbChunkSeqCache = new Map(); // path -> next seq
const _idbAppendChains = new Map(); // path -> Promise chain

/** Build a flat path key from dirParts + filename */
function _idbKey(dirParts, filename) {
  return [...dirParts, filename].join('/');
}

/** Build a directory prefix for listing */
function _idbDirPrefix(dirParts) {
  return dirParts.join('/') + '/';
}

function _toText(data) {
  if (data == null) return '';
  if (typeof data === 'string') return data;
  if (data instanceof ArrayBuffer) return _idbTextDecoder.decode(data);
  if (ArrayBuffer.isView(data)) return _idbTextDecoder.decode(data);
  try {
    return _idbTextDecoder.decode(new Uint8Array(data));
  } catch {
    return String(data);
  }
}

function _toArrayBuffer(data) {
  if (data == null) return new ArrayBuffer(0);
  if (data instanceof ArrayBuffer) return data;
  if (ArrayBuffer.isView(data)) return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  if (typeof data === 'string') return _idbTextEncoder.encode(data).buffer;
  return new Uint8Array(data).buffer;
}

function _storageError(context, err) {
  const name = String(err?.name || '');
  const msg = String(err?.message || err || '').trim();
  const merged = `${name} ${msg}`.toLowerCase();
  const quotaLike = merged.includes('quota') || merged.includes('storage') || merged.includes('space');
  if (quotaLike) {
    return new Error(`${context}: browser storage quota exceeded`);
  }
  return new Error(`${context}: ${msg || name || 'unknown error'}`);
}

function _withAppendLock(path, fn) {
  const prev = _idbAppendChains.get(path) || Promise.resolve();
  const next = prev.catch(() => {}).then(fn);
  _idbAppendChains.set(path, next);
  return next.finally(() => {
    if (_idbAppendChains.get(path) === next) _idbAppendChains.delete(path);
  });
}

async function _deleteIdbChunksForPath(path) {
  await _idb.chunks.where('path').equals(path).delete();
  _idbChunkSeqCache.delete(path);
}

async function _collectIdbText(key) {
  const rec = await _idb.blobs.get(key);
  const chunks = await _idb.chunks.where('path').equals(key).sortBy('seq');
  const hasChunks = chunks.length > 0;
  if (!rec && chunks.length === 0) return null;
  if (rec?.type === 'binary' && !hasChunks) return null;
  const base = rec
    ? (rec.type === 'text' ? _toText(rec.data) : '')
    : '';
  if (chunks.length === 0) return base;
  let out = base;
  for (const c of chunks) out += String(c?.text || '');
  return out;
}

async function _nextChunkSeq(path) {
  if (_idbChunkSeqCache.has(path)) return _idbChunkSeqCache.get(path);
  const last = await _idb.chunks
    .where('[path+seq]')
    .between([path, Dexie.minKey], [path, Dexie.maxKey])
    .last();
  const next = Number.isFinite(Number(last?.seq)) ? Number(last.seq) + 1 : 1;
  _idbChunkSeqCache.set(path, next);
  return next;
}

async function _writeIdbFile(dirParts, filename, data) {
  const key = _idbKey(dirParts, filename);
  let value = data;
  if (data instanceof Blob) value = await data.arrayBuffer();
  else if (!(typeof data === 'string' || data instanceof ArrayBuffer || ArrayBuffer.isView(data))) {
    value = new Uint8Array(data).buffer;
  }

  await _idb.transaction('rw', _idb.blobs, _idb.chunks, async () => {
    await _idb.blobs.put({
      path: key,
      data: typeof value === 'string' ? value : _toArrayBuffer(value),
      type: typeof value === 'string' ? 'text' : 'binary',
    });
    await _deleteIdbChunksForPath(key);
  });
}

async function _appendIdbLine(dirParts, filename, line) {
  const key = _idbKey(dirParts, filename);
  return _withAppendLock(key, async () => {
    const seq = await _nextChunkSeq(key);
    await _idb.transaction('rw', _idb.blobs, _idb.chunks, async () => {
      const rec = await _idb.blobs.get(key);
      if (!rec) {
        await _idb.blobs.put({ path: key, data: '', type: 'text' });
      } else if (rec.type !== 'text') {
        await _idb.blobs.put({ path: key, data: _toText(rec.data), type: 'text' });
      }
      await _idb.chunks.put({ path: key, seq, text: `${line}\n` });
    });
    _idbChunkSeqCache.set(key, seq + 1);
  });
}


// ═══════════════════════════════════════════════════════════════════════
// Unified API (auto-selects OPFS or IDB)
// ═══════════════════════════════════════════════════════════════════════

/**
 * Write a string/blob to storage.
 * @param {string[]} dirParts — directory path segments
 * @param {string} filename
 * @param {string|Blob|ArrayBuffer|Uint8Array} data
 */
export async function writeFile(dirParts, filename, data) {
  const dir = await ensureDir(dirParts);
  if (!dir) {
    // IDB fallback
    await _writeIdbFile(dirParts, filename, data);
    return;
  }

  let fileHandle;
  try {
    fileHandle = await dir.getFileHandle(filename, { create: true });
  } catch (err) {
    console.warn('[Storage] OPFS getFileHandle failed, using IndexedDB for this file:', err);
    await _writeIdbFile(dirParts, filename, data);
    return;
  }

  // Prefer synchronous access handle (faster, available in workers)
  if (typeof fileHandle.createSyncAccessHandle === 'function') {
    try {
      const accessHandle = await fileHandle.createSyncAccessHandle();
      const encoder = new TextEncoder();
      const bytes = typeof data === 'string' ? encoder.encode(data) : new Uint8Array(data);
      accessHandle.truncate(0);
      accessHandle.write(bytes);
      accessHandle.flush();
      accessHandle.close();
      return;
    } catch {
      // Sync access not available on main thread, fall through
    }
  }

  // Writable stream fallback (works on main thread)
  if (typeof fileHandle.createWritable !== 'function') {
    _setIdbFallback('fileHandle.createWritable() missing');
    await _writeIdbFile(dirParts, filename, data);
    return;
  }
  try {
    const writable = await fileHandle.createWritable();
    await writable.write(data);
    await writable.close();
  } catch (err) {
    console.warn('[Storage] OPFS write failed, using IndexedDB for this file:', err);
    await _writeIdbFile(dirParts, filename, data);
  }
}

/**
 * Read a file as text.
 * @param {string[]|string} dirPartsOrPath
 * @param {string} [filename]
 * @returns {string|null}
 */
export async function readFileText(dirPartsOrPath, filename) {
  let dirParts;
  if (typeof dirPartsOrPath === 'string' && filename === undefined) {
    const sp = _splitPath(dirPartsOrPath);
    dirParts = sp.dirParts; filename = sp.filename;
  } else if (Array.isArray(dirPartsOrPath)) {
    dirParts = dirPartsOrPath;
  } else {
    return null;
  }

  const dir = await ensureDir(dirParts);
  if (!dir) {
    // IDB fallback
    try {
      return await _collectIdbText(_idbKey(dirParts, filename));
    } catch { return null; }
  }

  try {
    const fileHandle = await dir.getFileHandle(filename);
    const file = await fileHandle.getFile();
    return await file.text();
  } catch {
    return null;
  }
}

/**
 * Read a file as an ArrayBuffer.
 * @param {string[]} dirParts
 * @param {string} filename
 * @returns {ArrayBuffer|null}
 */
export async function readFileBuffer(dirParts, filename) {
  const dir = await ensureDir(dirParts);
  if (!dir) {
    // IDB fallback
    try {
      const key = _idbKey(dirParts, filename);
      const text = await _collectIdbText(key);
      if (text != null) return _idbTextEncoder.encode(text).buffer;

      const rec = await _idb.blobs.get(key);
      if (!rec) return null;
      return _toArrayBuffer(rec.data);
    } catch { return null; }
  }

  try {
    const fileHandle = await dir.getFileHandle(filename);
    const file = await fileHandle.getFile();
    return await file.arrayBuffer();
  } catch {
    return null;
  }
}

/**
 * Read a file as a Blob/File without forcing an ArrayBuffer copy.
 * @param {string[]} dirParts
 * @param {string} filename
 * @returns {Blob|null}
 */
export async function readFileBlob(dirParts, filename) {
  const dir = await ensureDir(dirParts);
  if (!dir) {
    // IDB fallback
    try {
      const key = _idbKey(dirParts, filename);
      const text = await _collectIdbText(key);
      if (text != null) return new Blob([text], { type: 'text/plain' });

      const rec = await _idb.blobs.get(key);
      if (!rec) return null;
      if (rec.data instanceof Blob) return rec.data;
      if (rec.data instanceof ArrayBuffer) return new Blob([rec.data]);
      if (ArrayBuffer.isView(rec.data)) return new Blob([rec.data.buffer]);
      if (typeof rec.data === 'string') return new Blob([rec.data], { type: 'text/plain' });
      return null;
    } catch {
      return null;
    }
  }

  try {
    const fileHandle = await dir.getFileHandle(filename);
    const file = await fileHandle.getFile();
    return file;
  } catch {
    return null;
  }
}

/**
 * Delete a file.
 * @param {string[]|string} dirPartsOrPath
 * @param {string} [filename]
 */
export async function deleteFile(dirPartsOrPath, filename) {
  let dirParts;
  if (typeof dirPartsOrPath === 'string' && filename === undefined) {
    const sp = _splitPath(dirPartsOrPath);
    dirParts = sp.dirParts; filename = sp.filename;
  } else if (Array.isArray(dirPartsOrPath)) {
    dirParts = dirPartsOrPath;
  } else {
    return;
  }

  const dir = await ensureDir(dirParts);
  if (!dir) {
    const key = _idbKey(dirParts, filename);
    try {
      await _idb.transaction('rw', _idb.blobs, _idb.chunks, async () => {
        await _idb.blobs.delete(key);
        await _deleteIdbChunksForPath(key);
      });
    } catch {}
    return;
  }

  try {
    await dir.removeEntry(filename);
  } catch {
    // Ignore if doesn't exist
  }
}

/**
 * Delete an entire directory tree.
 * @param {string[]} dirParts
 */
export async function deleteDir(dirParts) {
  const root = await getRoot();
  if (!root) {
    // IDB fallback: delete all keys with this prefix
    try {
      const prefix = _idbDirPrefix(dirParts);
      const [blobKeys, chunkKeys] = await Promise.all([
        _idb.blobs.where('path').startsWith(prefix).primaryKeys(),
        _idb.chunks.where('path').startsWith(prefix).primaryKeys(),
      ]);
      if (blobKeys.length) await _idb.blobs.bulkDelete(blobKeys);
      if (chunkKeys.length) await _idb.chunks.bulkDelete(chunkKeys);
      for (const key of _idbChunkSeqCache.keys()) {
        if (key.startsWith(prefix)) _idbChunkSeqCache.delete(key);
      }
    } catch {}
    return;
  }

  try {
    if (dirParts.length === 0) return;
    const parentParts = dirParts.slice(0, -1);
    const dirName = dirParts[dirParts.length - 1];
    const parent = dirParts.length === 1 ? root : await ensureDir(parentParts);
    if (parent) {
      await parent.removeEntry(dirName, { recursive: true });
    }
  } catch {
    // Ignore
  }
}

/**
 * List files in a directory.
 * @param {string[]} dirParts
 * @returns {string[]}
 */
export async function listFiles(dirParts) {
  const dir = await ensureDir(dirParts);
  if (!dir) {
    // IDB fallback: list keys with this prefix, extract filenames
    try {
      const prefix = _idbDirPrefix(dirParts);
      const [blobKeys, chunkKeys] = await Promise.all([
        _idb.blobs.where('path').startsWith(prefix).primaryKeys(),
        _idb.chunks.where('path').startsWith(prefix).uniqueKeys(),
      ]);
      const keys = [...new Set([...blobKeys, ...chunkKeys])];
      const names = [];
      for (const key of keys) {
        const rest = key.slice(prefix.length);
        if (!rest.includes('/')) names.push(rest);
      }
      return names;
    } catch { return []; }
  }

  try {
    const names = [];
    for await (const [name, handle] of dir) {
      if (handle.kind === 'file') names.push(name);
    }
    return names;
  } catch {
    return [];
  }
}

/**
 * Check if OPFS is available in this browser.
 */
export async function isAvailable() {
  return !!(await getRoot());
}

/**
 * Return storage backend + quota/usage diagnostics for debugging.
 */
export async function getDebugInfo() {
  const root = await getRoot();
  let usage = null;
  let quota = null;
  try {
    if (typeof navigator !== 'undefined' && navigator.storage?.estimate) {
      const est = await navigator.storage.estimate();
      usage = Number.isFinite(Number(est?.usage)) ? Number(est.usage) : null;
      quota = Number.isFinite(Number(est?.quota)) ? Number(est.quota) : null;
    }
  } catch {}
  return {
    backend: root ? 'opfs' : 'idb',
    forceIdbFallback: _forceIdbFallback,
    opfsChecked: _opfsChecked,
    usage,
    quota,
  };
}


// ── Path-string convenience API ───────────────────────────────────────

function _splitPath(path) {
  const parts = path.replace(/\\/g, '/').split('/').filter(Boolean);
  const filename = parts.pop();
  return { dirParts: parts, filename };
}

/**
 * Append a line of text to a file (for JSONL streaming).
 * @param {string[]|string} dirPartsOrPath
 * @param {string} filenameOrLine
 * @param {string} [lineIfNeeded]
 */
export async function appendLine(dirPartsOrPath, filenameOrLine, lineIfNeeded) {
  let dirParts, filename, line;
  if (typeof dirPartsOrPath === 'string' && lineIfNeeded === undefined) {
    const sp = _splitPath(dirPartsOrPath);
    dirParts = sp.dirParts;
    filename = sp.filename;
    line = filenameOrLine;
  } else if (Array.isArray(dirPartsOrPath)) {
    dirParts = dirPartsOrPath;
    filename = filenameOrLine;
    line = lineIfNeeded;
  } else {
    throw new Error('appendLine: invalid arguments');
  }

  const dir = await ensureDir(dirParts);
  if (!dir) {
    // IDB fallback: append as incremental chunks (avoids O(n^2) rewrites)
    try {
      await _appendIdbLine(dirParts, filename, line);
    } catch (e) {
      throw _storageError(`[Storage IDB] appendLine failed for "${_idbKey(dirParts, filename)}"`, e);
    }
    return;
  }

  let fileHandle;
  try {
    fileHandle = await dir.getFileHandle(filename, { create: true });
  } catch (err) {
    console.warn('[Storage] OPFS getFileHandle failed, using IndexedDB for append:', err);
    try {
      await _appendIdbLine(dirParts, filename, line);
    } catch (e) {
      throw _storageError(`[Storage] appendLine failed for "${_idbKey(dirParts, filename)}"`, e);
    }
    return;
  }
  if (typeof fileHandle.createWritable !== 'function') {
    _setIdbFallback('fileHandle.createWritable() missing in appendLine');
    try {
      await _appendIdbLine(dirParts, filename, line);
    } catch (e) {
      throw _storageError(`[Storage] appendLine failed for "${_idbKey(dirParts, filename)}"`, e);
    }
    return;
  }
  try {
    const file = await fileHandle.getFile();
    const writable = await fileHandle.createWritable({ keepExistingData: true });
    writable.seek(file.size);
    await writable.write(line + '\n');
    await writable.close();
  } catch (err) {
    console.warn('[Storage] OPFS append failed, using IndexedDB for this write:', err);
    try {
      await _appendIdbLine(dirParts, filename, line);
    } catch (e) {
      throw _storageError(`[Storage] appendLine failed for "${_idbKey(dirParts, filename)}"`, e);
    }
  }
}
