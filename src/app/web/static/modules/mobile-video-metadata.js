const QUICKTIME_UNIX_EPOCH_DELTA_SEC = 2082844800;
const APPLE_CREATIONDATE_KEY = 'com.apple.quicktime.creationdate';
const APPLE_MAKE_KEY = 'com.apple.quicktime.make';
const APPLE_MODEL_KEY = 'com.apple.quicktime.model';

function readU32(dv, offset) {
  return dv.getUint32(offset, false);
}

function readU64(dv, offset) {
  const hi = BigInt(dv.getUint32(offset, false));
  const lo = BigInt(dv.getUint32(offset + 4, false));
  return (hi << 32n) | lo;
}

function read4CC(dv, offset) {
  return String.fromCharCode(
    dv.getUint8(offset),
    dv.getUint8(offset + 1),
    dv.getUint8(offset + 2),
    dv.getUint8(offset + 3),
  );
}

async function readSlice(file, start, end) {
  return file.slice(start, end).arrayBuffer();
}

async function* parseBoxes(file, start = 0, end = null) {
  const limit = end ?? file.size;
  let pos = start;
  while (pos < limit - 8) {
    const headerBuf = await readSlice(file, pos, Math.min(pos + 16, limit));
    if (headerBuf.byteLength < 8) break;
    const dv = new DataView(headerBuf);
    let size = readU32(dv, 0);
    const type = read4CC(dv, 4);
    let headerLen = 8;

    if (size === 1 && headerBuf.byteLength >= 16) {
      const hi = readU32(dv, 8);
      const lo = readU32(dv, 12);
      size = hi * 0x100000000 + lo;
      headerLen = 16;
    } else if (size === 0) {
      size = limit - pos;
    }

    if (size < headerLen) break;

    yield {
      type,
      offset: pos,
      size,
      headerLen,
      dataStart: pos + headerLen,
      dataEnd: pos + size,
    };
    pos += size;
  }
}

function findChildBoxes(buf, start, end, targetType = null) {
  const dv = new DataView(buf);
  const results = [];
  let pos = start;
  while (pos < end - 8) {
    let size = readU32(dv, pos);
    const type = read4CC(dv, pos + 4);
    const typeCode = readU32(dv, pos + 4);
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
        typeCode,
        relStart: pos + headerLen,
        relEnd: Math.min(pos + size, end),
      });
    }

    pos += size;
  }
  return results;
}

function sanitizeDecodedText(text) {
  return String(text || '')
    .replace(/\u0000+/g, ' ')
    .replace(/[\u0001-\u001f\u007f]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function decodeTextPayload(payload) {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload || new ArrayBuffer(0));
  if (!bytes.length) return '';

  const utf8 = sanitizeDecodedText(new TextDecoder('utf-8', { fatal: false }).decode(bytes));
  if (utf8) return utf8;

  if (bytes.length >= 2) {
    const utf16be = [];
    for (let i = 0; i + 1 < bytes.length; i += 2) {
      utf16be.push((bytes[i] << 8) | bytes[i + 1]);
    }
    const utf16 = sanitizeDecodedText(String.fromCharCode(...utf16be));
    if (utf16) return utf16;
  }

  return '';
}

function decodeBoxTextValue(buf, box) {
  const children = findChildBoxes(buf, box.relStart, box.relEnd);
  const dataBox = children.find(child => child.type === 'data');
  if (dataBox) {
    const payloadStart = Math.min(dataBox.relEnd, dataBox.relStart + 8);
    return decodeTextPayload(buf.slice(payloadStart, dataBox.relEnd));
  }
  return decodeTextPayload(buf.slice(box.relStart, box.relEnd));
}

function parseMetadataDateValue(rawValue) {
  const text = sanitizeDecodedText(rawValue);
  if (!text) return null;

  const candidates = [];
  const isoMatch = text.match(/\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?: ?(?:Z|[+-]\d{2}:?\d{2}))?/);
  if (isoMatch?.[0]) candidates.push(isoMatch[0]);
  candidates.push(text);

  for (let candidate of candidates) {
    candidate = String(candidate || '').trim();
    if (!candidate) continue;
    if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/.test(candidate)) {
      candidate = candidate.replace(' ', 'T');
    }
    candidate = candidate.replace(/([+-]\d{2})(\d{2})$/, '$1:$2');
    const ms = Date.parse(candidate);
    if (Number.isFinite(ms)) return ms / 1000;
  }

  return null;
}

function quickTimeSecondsToUnix(seconds) {
  const value = Number(seconds);
  return Number.isFinite(value) ? (value - QUICKTIME_UNIX_EPOCH_DELTA_SEC) : null;
}

function parseMvhdCreationTs(moovBuf) {
  const mvhd = findChildBoxes(moovBuf, 0, moovBuf.byteLength, 'mvhd')[0];
  if (!mvhd) return null;

  const dv = new DataView(moovBuf, mvhd.relStart, mvhd.relEnd - mvhd.relStart);
  if (dv.byteLength < 8) return null;

  const version = dv.getUint8(0);
  if (version === 1) {
    if (dv.byteLength < 20) return null;
    return quickTimeSecondsToUnix(readU64(dv, 4));
  }
  return quickTimeSecondsToUnix(readU32(dv, 4));
}

function parseKeysBox(buf, box) {
  const dv = new DataView(buf, box.relStart, box.relEnd - box.relStart);
  if (dv.byteLength < 8) return [];
  const entryCount = readU32(dv, 4);
  const keys = [];
  let pos = 8;

  for (let i = 0; i < entryCount && pos + 8 <= dv.byteLength; i++) {
    const size = readU32(dv, pos);
    const namespace = read4CC(dv, pos + 4);
    if (size < 8 || pos + size > dv.byteLength) break;
    const raw = buf.slice(box.relStart + pos + 8, box.relStart + pos + size);
    const key = decodeTextPayload(raw);
    keys.push(namespace === 'mdta' ? key : `${namespace}:${key}`);
    pos += size;
  }

  return keys;
}

function parseMetaBoxValues(buf, metaBox) {
  const values = {};
  const childStart = Math.min(metaBox.relEnd, metaBox.relStart + 4);
  if (childStart >= metaBox.relEnd) return values;

  const children = findChildBoxes(buf, childStart, metaBox.relEnd);
  const keysBox = children.find(child => child.type === 'keys');
  const ilstBox = children.find(child => child.type === 'ilst');
  const keys = keysBox ? parseKeysBox(buf, keysBox) : [];

  if (ilstBox) {
    const items = findChildBoxes(buf, ilstBox.relStart, ilstBox.relEnd);
    for (const item of items) {
      const value = decodeBoxTextValue(buf, item);
      if (!value) continue;
      if (item.typeCode >= 1 && item.typeCode <= keys.length) {
        values[keys[item.typeCode - 1]] = value;
      } else {
        values[item.type] = value;
      }
    }
  }

  return values;
}

function parseUdtaBoxValues(buf, udtaBox) {
  const out = {};
  const children = findChildBoxes(buf, udtaBox.relStart, udtaBox.relEnd);
  for (const child of children) {
    if (child.type === 'meta') continue;
    if (child.type === 'manu' || child.type === 'modl' || child.type === '\u00A9day' || child.type === 'IDIT' || child.type === 'date') {
      const value = decodeBoxTextValue(buf, child);
      if (value) out[child.type] = value;
    }
  }
  return out;
}

function pickFirstNonEmpty(...values) {
  for (const value of values) {
    const text = sanitizeDecodedText(value);
    if (text) return text;
  }
  return null;
}

function looksLikeAppleCaptureFilename(name) {
  const base = String(name || '').trim().toUpperCase();
  if (!base) return false;
  return /^(IMG|VID|PXL|MVIMG|IMG_E)_\d{3,}/.test(base);
}

export async function parseAppleMobileVideoMetadata(file) {
  if (!(file instanceof File) || file.size <= 0) return null;

  let moovBox = null;
  for await (const box of parseBoxes(file)) {
    if (box.type === 'moov') {
      moovBox = box;
      break;
    }
  }
  if (!moovBox) return null;

  const moovBuf = await readSlice(file, moovBox.dataStart, moovBox.dataEnd);
  const moovChildren = findChildBoxes(moovBuf, 0, moovBuf.byteLength);
  const mdtaValues = {};
  const udtaValues = {};

  for (const child of moovChildren) {
    if (child.type === 'meta') {
      Object.assign(mdtaValues, parseMetaBoxValues(moovBuf, child));
      continue;
    }
    if (child.type !== 'udta') continue;

    Object.assign(udtaValues, parseUdtaBoxValues(moovBuf, child));
    const udtaMetaBoxes = findChildBoxes(moovBuf, child.relStart, child.relEnd, 'meta');
    for (const metaBox of udtaMetaBoxes) {
      Object.assign(mdtaValues, parseMetaBoxValues(moovBuf, metaBox));
    }
  }

  const make = pickFirstNonEmpty(mdtaValues[APPLE_MAKE_KEY], udtaValues.manu);
  const model = pickFirstNonEmpty(mdtaValues[APPLE_MODEL_KEY], udtaValues.modl);
  const creationText = pickFirstNonEmpty(
    udtaValues.date,
    mdtaValues[APPLE_CREATIONDATE_KEY],
    mdtaValues['\u00A9day'],
    udtaValues['\u00A9day'],
    mdtaValues.IDIT,
    udtaValues.IDIT,
  );
  const creationTs = parseMetadataDateValue(creationText);
  const mvhdCreationTs = parseMvhdCreationTs(moovBuf);
  const captureStartTs = creationTs ?? mvhdCreationTs ?? null;
  const captureTsSource = udtaValues.date && creationTs != null
    ? 'quicktime_user_date'
    : (creationTs != null
      ? 'quicktime_creationdate'
      : (mvhdCreationTs != null ? 'quicktime_mvhd' : null));

  const hasAppleQuickTimeKeys = Boolean(
    mdtaValues[APPLE_CREATIONDATE_KEY]
    || mdtaValues[APPLE_MAKE_KEY]
    || mdtaValues[APPLE_MODEL_KEY]
  );
  const isAppleDevice = /apple/i.test(make || '') || /iphone|ipad|ipod/i.test(model || '') || hasAppleQuickTimeKeys;
  const filenameLooksApple = looksLikeAppleCaptureFilename(file.name);
  const isLikelyIphonePlaybackVideo = Boolean(
    captureStartTs != null && (isAppleDevice || filenameLooksApple)
  );

  return {
    captureStartTs,
    captureTsSource,
    creationText,
    make,
    model,
    isAppleDevice,
    isLikelyIphonePlaybackVideo,
  };
}
