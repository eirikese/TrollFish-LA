/**
 * TrollFish — gopro-gps.js
 * Browser-native GoPro GPS telemetry extractor.
 *
 * Parses GoPro MP4 files to extract GPS data from the GPMF
 * (GoPro Metadata Format) telemetry stream embedded in the file.
 *
 * This is a self-contained implementation — no npm dependencies.
 *
 * Flow:
 *   1. Parse MP4 box structure to find GoPro metadata track ('GoPro MET')
 *   2. Read sample table (stbl) to locate GPMF data chunks
 *   3. Parse GPMF binary KLV stream for GPS5 (lat, lon, alt, speed2d, speed3d)
 *   4. Combine with GPSU (UTC timestamp) and SCAL (scale factors)
 *   5. Return array of { ts, lat, lon, video_s } points
 *
 * Reference: https://github.com/gopro/gpmf-parser
 */

import { filterGpsOutliers } from './csv-gps.js';

const MAX_GPMF_SAMPLE_SIZE_BYTES = 10_000_000;
const MAX_GPMF_READ_BATCH_BYTES = 32 * 1024 * 1024;
const TEXT_DECODER = new TextDecoder();

// ── MP4 Box Parser ────────────────────────────────────────────────────

/**
 * Read a 4-byte big-endian uint from a DataView.
 */
function readU32(dv, offset) {
  return dv.getUint32(offset, false);
}

/**
 * Read 4 ASCII chars as a string.
 */
function read4CC(dv, offset) {
  return String.fromCharCode(
    dv.getUint8(offset), dv.getUint8(offset + 1),
    dv.getUint8(offset + 2), dv.getUint8(offset + 3)
  );
}

/**
 * Parse top-level MP4 boxes from a File object.
 * Yields { type, offset, size } for each box.
 *
 * Reads lazily from the File using slice() — doesn't load whole file.
 */
async function* parseBoxes(file, start = 0, end = null) {
  end = end ?? file.size;
  let pos = start;

  while (pos < end - 8) {
    const headerBuf = await readSlice(file, pos, Math.min(pos + 16, end));
    const dv = new DataView(headerBuf);
    let size = readU32(dv, 0);
    const type = read4CC(dv, 4);

    let headerLen = 8;
    if (size === 1 && headerBuf.byteLength >= 16) {
      // 64-bit extended size
      const hi = readU32(dv, 8);
      const lo = readU32(dv, 12);
      size = hi * 0x100000000 + lo;
      headerLen = 16;
    } else if (size === 0) {
      size = end - pos; // extends to end of file/parent
    }

    if (size < headerLen) break; // corrupted

    yield { type, offset: pos, size, headerLen, dataStart: pos + headerLen, dataEnd: pos + size };
    pos += size;
  }
}

async function readSlice(file, start, end) {
  const blob = file.slice(start, end);
  return blob.arrayBuffer();
}

async function readSampleBuffers(file, sampleOffsets, sampleSizes) {
  const samples = new Array(sampleOffsets.length);
  let i = 0;

  while (i < sampleOffsets.length) {
    const offset = sampleOffsets[i];
    const size = i < sampleSizes.length ? sampleSizes[i] : 0;
    if (!Number.isFinite(offset) || size <= 0 || size >= MAX_GPMF_SAMPLE_SIZE_BYTES) {
      samples[i] = new ArrayBuffer(0);
      i++;
      continue;
    }

    const batchStart = offset;
    let batchEnd = offset + size;
    const batchIndexes = [i];
    i++;

    while (i < sampleOffsets.length) {
      const nextOffset = sampleOffsets[i];
      const nextSize = i < sampleSizes.length ? sampleSizes[i] : 0;
      if (
        !Number.isFinite(nextOffset) ||
        nextSize <= 0 ||
        nextSize >= MAX_GPMF_SAMPLE_SIZE_BYTES ||
        nextOffset !== batchEnd ||
        (batchEnd + nextSize - batchStart) > MAX_GPMF_READ_BATCH_BYTES
      ) {
        break;
      }
      batchIndexes.push(i);
      batchEnd += nextSize;
      i++;
    }

    const batchBuf = await readSlice(file, batchStart, batchEnd);
    for (const sampleIdx of batchIndexes) {
      const start = sampleOffsets[sampleIdx] - batchStart;
      const end = start + sampleSizes[sampleIdx];
      samples[sampleIdx] = batchBuf.slice(start, end);
    }
  }

  return samples;
}


// ── MP4 structure traversal ───────────────────────────────────────────

/**
 * Find the GoPro metadata track and extract GPMF samples.
 * @param {File} file
 * @returns {{ samples: ArrayBuffer[], timescale: number, durations: number[], sampleTimes: number[] }}
 */
async function extractGpmfSamples(file) {
  // Step 1: Find moov box
  let moovBox = null;
  for await (const box of parseBoxes(file)) {
    if (box.type === 'moov') {
      moovBox = box;
      break;
    }
  }
  if (!moovBox) throw new Error('No moov box found — not a valid MP4.');

  // Step 2: Read entire moov into memory (typically <5MB)
  const moovBuf = await readSlice(file, moovBox.dataStart, moovBox.dataEnd);
  const moovDv = new DataView(moovBuf);

  // Step 3: Find GoPro metadata trak
  const traks = findChildBoxes(moovBuf, 0, moovBuf.byteLength, 'trak');
  let metaTrak = null;

  for (const trak of traks) {
    // Look for handler type 'meta' or name containing 'GoPro'
    const hdlrBoxes = findDescendantBoxes(moovBuf, trak.relStart, trak.relEnd, 'hdlr');
    for (const hdlr of hdlrBoxes) {
      const hdlrData = moovBuf.slice(hdlr.relStart, hdlr.relEnd);
      const hdlrDv = new DataView(hdlrData);
      if (hdlrData.byteLength >= 12) {
        const handlerType = read4CC(hdlrDv, 8); // skip version(4) + pre_defined(4)
        if (handlerType === 'meta') {
          // Check the name field for 'GoPro'
          const nameStart = 24;
          if (hdlrData.byteLength > nameStart) {
            const nameBytes = new Uint8Array(hdlrData.slice(nameStart));
            const name = new TextDecoder().decode(nameBytes).replace(/\0/g, '');
            if (name.includes('GoPro') || name.includes('gpmf')) {
              metaTrak = trak;
              break;
            }
          }
          // Even without GoPro in name, 'meta' handler is likely the one
          metaTrak = trak;
        }
      }
    }
    if (metaTrak) break;
  }

  if (!metaTrak) throw new Error('No GoPro metadata track found in MP4.');

  // Step 4: Parse sample table (stbl) to get chunk offsets and sample sizes
  const stblBoxes = findDescendantBoxes(moovBuf, metaTrak.relStart, metaTrak.relEnd, 'stbl');
  if (stblBoxes.length === 0) throw new Error('No stbl box in metadata track.');
  const stbl = stblBoxes[0];

  // Get timescale from mdhd
  const mdhdBoxes = findDescendantBoxes(moovBuf, metaTrak.relStart, metaTrak.relEnd, 'mdhd');
  let timescale = 1000;
  if (mdhdBoxes.length > 0) {
    const mdhdData = moovBuf.slice(mdhdBoxes[0].relStart, mdhdBoxes[0].relEnd);
    const mdhdDv = new DataView(mdhdData);
    const version = mdhdDv.getUint8(0);
    if (version === 0) {
      timescale = mdhdDv.getUint32(12, false);
    } else {
      timescale = mdhdDv.getUint32(20, false);
    }
  }

  // stsz — sample sizes
  const stszBoxes = findDescendantBoxes(moovBuf, stbl.relStart, stbl.relEnd, 'stsz');
  const sampleSizes = [];
  if (stszBoxes.length > 0) {
    const stszData = moovBuf.slice(stszBoxes[0].relStart, stszBoxes[0].relEnd);
    const stszDv = new DataView(stszData);
    const defaultSize = stszDv.getUint32(4, false);
    const count = stszDv.getUint32(8, false);
    for (let i = 0; i < count; i++) {
      sampleSizes.push(defaultSize > 0 ? defaultSize : stszDv.getUint32(12 + i * 4, false));
    }
  }

  // stco / co64 — chunk offsets
  let chunkOffsets = [];
  const stcoBoxes = findDescendantBoxes(moovBuf, stbl.relStart, stbl.relEnd, 'stco');
  const co64Boxes = findDescendantBoxes(moovBuf, stbl.relStart, stbl.relEnd, 'co64');

  if (stcoBoxes.length > 0) {
    const stcoData = moovBuf.slice(stcoBoxes[0].relStart, stcoBoxes[0].relEnd);
    const stcoDv = new DataView(stcoData);
    const count = stcoDv.getUint32(4, false);
    for (let i = 0; i < count; i++) {
      chunkOffsets.push(stcoDv.getUint32(8 + i * 4, false));
    }
  } else if (co64Boxes.length > 0) {
    const co64Data = moovBuf.slice(co64Boxes[0].relStart, co64Boxes[0].relEnd);
    const co64Dv = new DataView(co64Data);
    const count = co64Dv.getUint32(4, false);
    for (let i = 0; i < count; i++) {
      const hi = co64Dv.getUint32(8 + i * 8, false);
      const lo = co64Dv.getUint32(8 + i * 8 + 4, false);
      chunkOffsets.push(hi * 0x100000000 + lo);
    }
  }

  // stsc — sample-to-chunk mapping
  const stscBoxes = findDescendantBoxes(moovBuf, stbl.relStart, stbl.relEnd, 'stsc');
  const stscEntries = [];
  if (stscBoxes.length > 0) {
    const stscData = moovBuf.slice(stscBoxes[0].relStart, stscBoxes[0].relEnd);
    const stscDv = new DataView(stscData);
    const count = stscDv.getUint32(4, false);
    for (let i = 0; i < count; i++) {
      stscEntries.push({
        firstChunk: stscDv.getUint32(8 + i * 12, false),
        samplesPerChunk: stscDv.getUint32(8 + i * 12 + 4, false),
        sampleDescriptionIndex: stscDv.getUint32(8 + i * 12 + 8, false),
      });
    }
  }

  // stts — sample durations
  const sttsBoxes = findDescendantBoxes(moovBuf, stbl.relStart, stbl.relEnd, 'stts');
  const durations = [];
  if (sttsBoxes.length > 0) {
    const sttsData = moovBuf.slice(sttsBoxes[0].relStart, sttsBoxes[0].relEnd);
    const sttsDv = new DataView(sttsData);
    const entryCount = sttsDv.getUint32(4, false);
    for (let i = 0; i < entryCount; i++) {
      const sampleCount = sttsDv.getUint32(8 + i * 8, false);
      const sampleDelta = sttsDv.getUint32(8 + i * 8 + 4, false);
      for (let j = 0; j < sampleCount; j++) {
        durations.push(sampleDelta);
      }
    }
  }

  // Build sample offset map using stsc
  const sampleOffsets = buildSampleOffsets(chunkOffsets, sampleSizes, stscEntries);

  // Compute sample times from durations and timescale
  const sampleTimes = [];
  let ts = 0;
  for (let i = 0; i < sampleSizes.length; i++) {
    sampleTimes.push(ts / timescale);
    ts += (i < durations.length ? durations[i] : (durations[durations.length - 1] || 1));
  }

  // Step 5: Read sample data from the file. GPMF samples are commonly
  // contiguous, so batching adjacent reads avoids thousands of File.slice()
  // round trips on long videos while preserving the exact sample buffers.
  const samples = await readSampleBuffers(file, sampleOffsets, sampleSizes);

  return { samples, timescale, sampleTimes };
}


/**
 * Build per-sample file offsets from chunk offsets + stsc + stsz.
 */
function buildSampleOffsets(chunkOffsets, sampleSizes, stscEntries) {
  if (stscEntries.length === 0 || chunkOffsets.length === 0) {
    // Simple: one sample per chunk
    return chunkOffsets.slice(0, sampleSizes.length);
  }

  const offsets = [];
  let sampleIdx = 0;

  for (let chunkIdx = 0; chunkIdx < chunkOffsets.length; chunkIdx++) {
    // Determine samples-per-chunk for this chunk
    let samplesInChunk = stscEntries[0].samplesPerChunk;
    for (const entry of stscEntries) {
      if (chunkIdx + 1 >= entry.firstChunk) {
        samplesInChunk = entry.samplesPerChunk;
      } else {
        break;
      }
    }

    let pos = chunkOffsets[chunkIdx];
    for (let s = 0; s < samplesInChunk && sampleIdx < sampleSizes.length; s++) {
      offsets.push(pos);
      pos += sampleSizes[sampleIdx];
      sampleIdx++;
    }
  }

  return offsets;
}


/**
 * Find child boxes within a buffer region.
 */
function findChildBoxes(buf, start, end, targetType = null) {
  const dv = new DataView(buf);
  const results = [];
  let pos = start;

  while (pos < end - 8) {
    let size = readU32(dv, pos);
    const type = read4CC(dv, pos + 4);
    let headerLen = 8;

    if (size === 1 && pos + 16 <= end) {
      const hi = readU32(dv, pos + 8);
      const lo = readU32(dv, pos + 12);
      size = hi * 0x100000000 + lo;
      headerLen = 16;
    } else if (size === 0) {
      size = end - pos;
    }

    if (size < headerLen) break;

    if (!targetType || type === targetType) {
      results.push({
        type,
        relStart: pos + headerLen,
        relEnd: Math.min(pos + size, end),
      });
    }

    pos += size;
  }

  return results;
}

/**
 * Recursively find descendant boxes of a given type.
 */
function findDescendantBoxes(buf, start, end, targetType) {
  const results = [];
  const children = findChildBoxes(buf, start, end);
  for (const child of children) {
    if (child.type === targetType) {
      results.push(child);
    }
    // Recurse into container boxes
    const containers = ['moov', 'trak', 'mdia', 'minf', 'stbl', 'udta', 'edts'];
    if (containers.includes(child.type)) {
      results.push(...findDescendantBoxes(buf, child.relStart, child.relEnd, targetType));
    }
  }
  return results;
}


// ── GPMF Binary Parser ───────────────────────────────────────────────

/**
 * Parse a GPMF binary stream (one sample).
 * Returns an array of KLV entries.
 */
function parseGpmfStream(buffer) {
  const dv = new DataView(buffer);
  const entries = [];
  let pos = 0;

  while (pos + 8 <= buffer.byteLength) {
    const fourCC = read4CC(dv, pos);
    const typeChar = String.fromCharCode(dv.getUint8(pos + 4));
    const structSize = dv.getUint8(pos + 5);
    const repeat = dv.getUint16(pos + 6, false);

    const dataSize = structSize * repeat;
    const padded = dataSize + ((4 - (dataSize % 4)) % 4); // pad to 4 bytes

    pos += 8;

    if (pos + dataSize > buffer.byteLength) break;

    const rawData = buffer.slice(pos, pos + dataSize);

    // Nested container (type 0x00)
    if (typeChar === '\0' && dataSize > 0) {
      const nested = parseGpmfStream(rawData);
      entries.push({ fourCC, type: 'container', children: nested });
    } else {
      entries.push({
        fourCC,
        type: typeChar,
        structSize,
        repeat,
        data: rawData,
      });
    }

    pos += padded;
  }

  return entries;
}


/**
 * Extract GPS data from parsed GPMF entries for one sample.
 * Looks for DEVC > STRM > GPS5 (or GPSA, GPS9 in newer firmware).
 *
 * IMPORTANT: Each STRM has its own SCAL — we must use the SCAL from the
 * same STRM that contains GPS5/GPS9, NOT a SCAL from an unrelated sensor stream.
 */
function extractGpsFromGpmf(entries) {
  const result = { gps5: null, gps9: null, gpsu: null, scale: null, typeDef: null };

  /**
   * Check a flat list of entries (one STRM's children) for GPS data + siblings.
   */
  function extractFromStrm(children) {
    let gps5 = null, gps9 = null, gpsu = null, scale = null, typeDef = null;
    for (const e of children) {
      if (e.fourCC === 'GPS5' && e.data) gps5 = e;
      if (e.fourCC === 'GPS9' && e.data) gps9 = e;
      if (e.fourCC === 'GPSU' && e.data) gpsu = e;
      if (e.fourCC === 'SCAL' && e.data) scale = e;
      if (e.fourCC === 'TYPE' && e.data) typeDef = readAsciiEntry(e);
    }
    return { gps5, gps9, gpsu, scale, typeDef };
  }

  /**
   * Walk the GPMF tree looking for STRM containers that hold GPS data.
   */
  function walk(entries) {
    for (const entry of entries) {
      if (entry.children) {
        // If this is a STRM-like container, check its direct children for GPS data
        const probe = extractFromStrm(entry.children);
        if (probe.gps5 || probe.gps9) {
          result.gps5 = probe.gps5;
          result.gps9 = probe.gps9;
          result.scale = probe.scale;
          result.typeDef = probe.typeDef;
          // GPSU is sometimes in the same STRM
          if (probe.gpsu) result.gpsu = probe.gpsu;
          return; // found the GPS stream — stop searching
        }
        // Not the GPS stream — recurse deeper (e.g. into DEVC containers)
        walk(entry.children);
        if (result.gps5 || result.gps9) return; // found in a nested level
      }
      // GPSU can also appear at the DEVC level in some firmware versions
      if (!result.gpsu && entry.fourCC === 'GPSU' && entry.data) {
        result.gpsu = entry;
      }
    }
  }

  walk(entries);
  return result;
}

function extractDeviceInfoFromGpmf(entries) {
  const info = {};
  function walk(items) {
    for (const entry of items || []) {
      if (entry.children) walk(entry.children);
      if (!entry?.data) continue;
      if (entry.fourCC === 'DVNM') info.device_model = readAsciiEntry(entry).trim();
      if (entry.fourCC === 'MKNM') info.device_make = readAsciiEntry(entry).trim();
      if (entry.fourCC === 'FWRE') info.firmware = readAsciiEntry(entry).trim();
    }
  }
  walk(entries);
  if (info.device_model && !info.device_make) info.device_make = 'GoPro';
  return info;
}

function readAsciiEntry(entry) {
  if (!entry?.data) return '';
  return TEXT_DECODER.decode(new Uint8Array(entry.data)).replace(/\0/g, '');
}

/**
 * Parse GPSU timestamp (UTC date string in GPMF format).
 * Format: "yymmddHHMMSS.sss" — 16 bytes ASCII
 * Returns epoch seconds.
 */
function parseGpsu(gpsuEntry) {
  if (!gpsuEntry?.data) return null;
  const bytes = new Uint8Array(gpsuEntry.data);
  const str = TEXT_DECODER.decode(bytes).replace(/\0/g, '');
  if (str.length < 12) return null;

  const yy = parseInt(str.slice(0, 2), 10);
  const mm = parseInt(str.slice(2, 4), 10);
  const dd = parseInt(str.slice(4, 6), 10);
  const HH = parseInt(str.slice(6, 8), 10);
  const MM = parseInt(str.slice(8, 10), 10);
  const SS = parseInt(str.slice(10, 12), 10);
  let ms = 0;
  if (str.length > 13 && str[12] === '.') {
    ms = parseInt(str.slice(13).padEnd(3, '0').slice(0, 3), 10);
  }

  const year = yy + 2000;
  const d = new Date(Date.UTC(year, mm - 1, dd, HH, MM, SS, ms));
  return d.getTime() / 1000;
}

function readScaleValues(scaleEntry, maxCount, defaults) {
  const scales = defaults.slice(0, maxCount);
  while (scales.length < maxCount) scales.push(1);
  if (!scaleEntry?.data) return scales;

  const scaleDv = new DataView(scaleEntry.data);
  const scaleType = scaleEntry.type;
  if (scaleType === 'l' || scaleType === 'L' || scaleType === 'f' || scaleType === 'F') {
    const nScales = Math.min(maxCount, Math.floor(scaleEntry.data.byteLength / 4));
    for (let i = 0; i < nScales; i++) {
      const raw = (scaleType === 'f' || scaleType === 'F')
        ? scaleDv.getFloat32(i * 4, false)
        : scaleDv.getInt32(i * 4, false);
      scales[i] = Number.isFinite(raw) && raw !== 0 ? raw : 1;
    }
  } else if (scaleType === 's' || scaleType === 'S') {
    const nScales = Math.min(maxCount, Math.floor(scaleEntry.data.byteLength / 2));
    for (let i = 0; i < nScales; i++) {
      const raw = scaleDv.getInt16(i * 2, false);
      scales[i] = raw !== 0 ? raw : 1;
    }
  }
  return scales;
}

function gpsDaysSecondsToEpoch(days, secondsOfDay) {
  if (!Number.isFinite(days) || !Number.isFinite(secondsOfDay)) return null;
  if (days < 0 || days > 50000 || secondsOfDay < 0 || secondsOfDay >= 90000) return null;
  return Date.UTC(2000, 0, 1) / 1000 + days * 86400 + secondsOfDay;
}

function bearingDeg(lat1, lon1, lat2, lon2) {
  const toRad = d => d * Math.PI / 180;
  const toDeg = r => r * 180 / Math.PI;
  const phi1 = toRad(lat1);
  const phi2 = toRad(lat2);
  const dLon = toRad(lon2 - lon1);
  const y = Math.sin(dLon) * Math.cos(phi2);
  const x = Math.cos(phi1) * Math.sin(phi2) -
    Math.sin(phi1) * Math.cos(phi2) * Math.cos(dLon);
  return (toDeg(Math.atan2(y, x)) + 360) % 360;
}

function addDerivedCog(points) {
  if (!Array.isArray(points) || points.length < 2) return points || [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (Number.isFinite(Number(p?.cog))) continue;
    const next = points[i + 1];
    const prev = points[i - 1];
    const ref = next || prev;
    if (!ref) continue;
    const a = next ? p : ref;
    const b = next ? ref : p;
    const lat1 = Number(a?.lat), lon1 = Number(a?.lon);
    const lat2 = Number(b?.lat), lon2 = Number(b?.lon);
    if (![lat1, lon1, lat2, lon2].every(Number.isFinite)) continue;
    if (Math.abs(lat1 - lat2) < 1e-12 && Math.abs(lon1 - lon2) < 1e-12) continue;
    p.cog = bearingDeg(lat1, lon1, lat2, lon2);
  }
  return points;
}

const GPMF_TYPE_SIZES = {
  b: 1, B: 1,
  s: 2, S: 2,
  l: 4, L: 4, f: 4, F: 4, q: 4,
  d: 8, j: 8, J: 8, Q: 8,
};

function expandGpmfTypeDef(typeDef) {
  return String(typeDef || '')
    .replace(/([A-Za-z?])\[(\d+)\]/g, (_m, type, repeat) => type.repeat(Number(repeat) || 1))
    .split('')
    .filter(type => GPMF_TYPE_SIZES[type]);
}

function readGpmfTypedValue(dv, offset, type) {
  switch (type) {
    case 'b': return dv.getInt8(offset);
    case 'B': return dv.getUint8(offset);
    case 's': return dv.getInt16(offset, false);
    case 'S': return dv.getUint16(offset, false);
    case 'l': return dv.getInt32(offset, false);
    case 'L': return dv.getUint32(offset, false);
    case 'f': return dv.getFloat32(offset, false);
    case 'd': return dv.getFloat64(offset, false);
    case 'j': {
      const hi = dv.getInt32(offset, false);
      const lo = dv.getUint32(offset + 4, false);
      return hi * 0x100000000 + lo;
    }
    case 'J':
    case 'Q': {
      const hi = dv.getUint32(offset, false);
      const lo = dv.getUint32(offset + 4, false);
      return hi * 0x100000000 + lo;
    }
    case 'q': {
      const raw = dv.getInt32(offset, false);
      return raw / 65536;
    }
    default:
      return NaN;
  }
}

function scaleGps9SensorValue(raw, scale, fallbackScale = 1) {
  const s = Number.isFinite(Number(scale)) && Number(scale) !== 0 ? Number(scale) : fallbackScale;
  return Number(raw) / s;
}

function decodeGps9Days(rawDays, scale) {
  let days = Number(rawDays);
  if (Number.isFinite(days) && days >= 0 && days <= 50000) return days;

  const scaled = scaleGps9SensorValue(rawDays, scale, 1);
  return Number.isFinite(scaled) && scaled >= 0 && scaled <= 50000 ? scaled : days;
}

function decodeGps9Seconds(rawSeconds, scale) {
  const seconds = Number(rawSeconds);
  const scaled = scaleGps9SensorValue(rawSeconds, scale, 1);

  if (Number.isFinite(scaled) && scaled >= 0 && scaled < 90000) return scaled;
  if (Number.isFinite(seconds) && seconds >= 0 && seconds < 90000) return seconds;
  if (Number.isFinite(seconds) && seconds >= 0 && seconds < 90000000) return seconds / 1000;

  return seconds;
}

function pushGps9Point(points, raw, scales) {
  if (!Array.isArray(raw) || raw.length < 9) return;

  const lat = scaleGps9SensorValue(raw[0], scales[0], 10000000);
  const lon = scaleGps9SensorValue(raw[1], scales[1], 10000000);
  const alt = scaleGps9SensorValue(raw[2], scales[2], 1000);
  const speed2d = scaleGps9SensorValue(raw[3], scales[3], 1000);
  const speed3d = scaleGps9SensorValue(raw[4], scales[4], 1000);
  const days = decodeGps9Days(raw[5], scales[5]);
  const secondsOfDay = decodeGps9Seconds(raw[6], scales[6]);
  const dop = scaleGps9SensorValue(raw[7], scales[7], 100);
  const fix = raw[8];

  if (Number.isFinite(fix) && fix < 2) return;
  if (lat === 0 && lon === 0) return;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;
  const ts = gpsDaysSecondsToEpoch(days, secondsOfDay);
  if (!Number.isFinite(ts)) return;

  points.push({
    lat,
    lon,
    alt,
    speed2d,
    speed3d,
    ts,
    dop,
    fix,
  });
}

/**
 * Decode GPS5 data (5 x int32 per sample: lat, lon, alt, speed2d, speed3d).
 */
function decodeGps5(gps5Entry, scaleEntry) {
  if (!gps5Entry?.data) return [];

  const scales = readScaleValues(scaleEntry, 5, [10000000, 10000000, 1000, 1000, 1000]);

  const count = gps5Entry.repeat;
  const dv = new DataView(gps5Entry.data);
  const points = [];

  for (let i = 0; i < count; i++) {
    const offset = i * 20; // 5 x int32 = 20 bytes
    if (offset + 20 > gps5Entry.data.byteLength) break;

    const lat = dv.getInt32(offset, false) / scales[0];
    const lon = dv.getInt32(offset + 4, false) / scales[1];
    const alt = dv.getInt32(offset + 8, false) / scales[2];
    const speed2d = dv.getInt32(offset + 12, false) / scales[3];
    const speed3d = dv.getInt32(offset + 16, false) / scales[4];

    // Validate
    if (lat === 0 && lon === 0) continue;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) continue;

    points.push({ lat, lon, alt, speed2d, speed3d });
  }

  return points;
}

/**
 * Decode GPS9 data used by newer GoPro firmware:
 * lat, lon, alt, 2D speed, 3D speed, days since 2000, seconds since midnight, DOP, fix.
 */
function decodeGps9(gps9Entry, scaleEntry, typeDef = null) {
  if (!gps9Entry?.data) return [];

  const scales = readScaleValues(scaleEntry, 9, [
    10000000, 10000000, 1000, 1000, 1000, 1, 1, 100, 1,
  ]);
  const fieldCount = 9;
  const typedFields = gps9Entry.type === '?' ? expandGpmfTypeDef(typeDef).slice(0, fieldCount) : [];
  if (typedFields.length === fieldCount) {
    const typedStructSize = typedFields.reduce((sum, type) => sum + GPMF_TYPE_SIZES[type], 0);
    const structSize = Math.max(typedStructSize, Number(gps9Entry.structSize) || typedStructSize);
    const count = Math.min(
      Number(gps9Entry.repeat) || 0,
      Math.floor(gps9Entry.data.byteLength / structSize),
    );
    const dv = new DataView(gps9Entry.data);
    const points = [];

    for (let i = 0; i < count; i++) {
      const baseOffset = i * structSize;
      if (baseOffset + typedStructSize > gps9Entry.data.byteLength) break;

      const raw = [];
      let fieldOffset = baseOffset;
      for (const type of typedFields) {
        raw.push(readGpmfTypedValue(dv, fieldOffset, type));
        fieldOffset += GPMF_TYPE_SIZES[type];
      }
      pushGps9Point(points, raw, scales);
    }

    return points;
  }

  const bytesPerField = 4;
  const minStructSize = fieldCount * bytesPerField;
  const structSize = Math.max(minStructSize, Number(gps9Entry.structSize) || minStructSize);
  const count = Math.min(
    Number(gps9Entry.repeat) || 0,
    Math.floor(gps9Entry.data.byteLength / structSize),
  );
  const dv = new DataView(gps9Entry.data);
  const points = [];

  for (let i = 0; i < count; i++) {
    const offset = i * structSize;
    if (offset + minStructSize > gps9Entry.data.byteLength) break;

    const raw = [];
    for (let f = 0; f < fieldCount; f++) {
      raw.push(dv.getInt32(offset + f * bytesPerField, false));
    }

    pushGps9Point(points, raw, scales);
  }

  return points;
}


// ── Main entry point ──────────────────────────────────────────────────

/**
 * Extract GPS track from a GoPro MP4 file.
 *
 * @param {File} file — the video File object
 * @param {(progress: number) => void} [onProgress] — optional 0..1 callback
 * @returns {{ points: object[], metadata: object }}
 */
export async function parseGoProVideoTrack(file, onProgress = null) {
  if (onProgress) onProgress(0);

  const { samples, sampleTimes } = await extractGpmfSamples(file);

  if (onProgress) onProgress(0.3);

  // Parse each GPMF sample and extract GPS
  let allPoints = [];
  let gpsFormat = null;
  let deviceInfo = {};

  for (let i = 0; i < samples.length; i++) {
    if (samples[i].byteLength === 0) continue;

    const entries = parseGpmfStream(samples[i]);
    if (!deviceInfo.device_model) {
      deviceInfo = { ...deviceInfo, ...extractDeviceInfoFromGpmf(entries) };
    }
    const { gps5, gps9, gpsu, scale, typeDef } = extractGpsFromGpmf(entries);

    if (!gps5 && !gps9) continue;

    const baseTs = parseGpsu(gpsu);
    const videoS = sampleTimes[i] ?? 0;
    const gpsPoints = gps9 ? decodeGps9(gps9, scale, typeDef) : decodeGps5(gps5, scale);
    if (!gpsFormat) gpsFormat = gps9 ? 'GPS9' : 'GPS5';

    if (gpsPoints.length === 0) continue;

    // Distribute GPS points evenly across the sample duration
    const nextVideoS = (i + 1 < sampleTimes.length) ? sampleTimes[i + 1] : videoS + 1;
    const dt = (nextVideoS - videoS) / Math.max(1, gpsPoints.length);

    for (let j = 0; j < gpsPoints.length; j++) {
      const p = gpsPoints[j];
      const t = videoS + j * dt;

      // Timestamp: GPS9 carries absolute UTC per point; GPS5 uses GPSU + offset.
      const ts = Number.isFinite(p.ts) ? p.ts : (baseTs != null ? (baseTs + j * dt) : null);

      if (ts == null && baseTs == null) continue;

      const pt = {
        ts: ts ?? 0,
        lat: p.lat,
        lon: p.lon,
        video_s: t,
      };
      // Propagate SOG (speed2d from GPS telemetry, m/s → knots)
      if (Number.isFinite(p.speed2d) && p.speed2d >= 0) {
        pt.sog = p.speed2d * 1.94384;
      }
      allPoints.push(pt);
    }

    if (onProgress) onProgress(0.3 + 0.6 * (i / samples.length));
  }

  // Sort by video_s
  allPoints.sort((a, b) => a.video_s - b.video_s);

  // Dedup by video_s
  {
    const deduped = [];
    let lastVS = null;
    for (const p of allPoints) {
      if (lastVS == null || Math.abs(p.video_s - lastVS) > 1e-9) {
        deduped.push(p);
        lastVS = p.video_s;
      }
    }
    allPoints = deduped;
  }

  // Filter GPS outliers
  const nBefore = allPoints.length;
  allPoints = filterGpsOutliers(allPoints);
  addDerivedCog(allPoints);
  const nRemoved = nBefore - allPoints.length;

  if (onProgress) onProgress(1);

  return {
    points: allPoints,
    metadata: {
      source: file.name,
      point_count: allPoints.length,
      has_video_s: true,
      gps_format: allPoints.length ? gpsFormat : 'none',
      outliers_removed: nRemoved,
      sample_count: samples.length,
      device_make: deviceInfo.device_make || null,
      device_model: deviceInfo.device_model || null,
      firmware: deviceInfo.firmware || null,
    },
  };
}
