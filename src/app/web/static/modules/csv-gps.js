/**
 * TrollFish — csv-gps.js
 * Browser-side CSV GPS parser — direct port from parsers.py
 *
 * Supports three formats:
 *   A. Vakaros-native: timestamp, latitude, longitude
 *   B. Sensor/IMU logger: lat, lon + iso_time or timestamp_ms
 *   C. Generic fallback: flexible column matching
 *
 * Also includes GPS outlier filter and timestamp normalization.
 */

import { MAX_PLAUSIBLE_SPEED_MS } from './config.js';

// ── Helpers ───────────────────────────────────────────────────────────

function parseFloat_(v) {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function normalizeIsoTimezone(raw) {
  let s = (raw || '').trim();
  if (!s) return s;
  if (s.endsWith('Z')) return s.slice(0, -1) + '+00:00';
  // +0000 -> +00:00
  if (s.length >= 5) {
    const c = s[s.length - 5];
    if ((c === '+' || c === '-') && s[s.length - 3] !== ':') {
      const tail = s.slice(-4);
      if (/^\d{4}$/.test(tail)) {
        return s.slice(0, -5) + s.slice(-5, -2) + ':' + s.slice(-2);
      }
    }
  }
  return s;
}

function parseNumericEpoch(raw) {
  if (!/^[+-]?\d+(\.\d+)?$/.test(raw)) return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  const a = Math.abs(n);
  if (a > 1e14) return n / 1e6; // microseconds
  if (a > 1e11) return n / 1e3; // milliseconds
  if (a > 1e9) return n;        // seconds
  return null;
}

function parseIsoLike(s) {
  // Accept:
  // YYYY-MM-DD
  // YYYY-MM-DD HH:MM[:SS[.sss]]
  // YYYY-MM-DDTHH:MM[:SS[.sss]]
  // with optional timezone Z / ±HH:MM / ±HHMM
  const m = s.match(
    /^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2})(?::?(\d{2}))?(?::?(\d{2})(?:[.,](\d{1,6}))?)?)?(?:\s*(Z|[+\-]\d{2}:?\d{2}))?$/
  );
  if (!m) return null;

  const year = Number(m[1]);
  const month = Number(m[2]);
  const day = Number(m[3]);
  const hour = Number(m[4] || 0);
  const minute = Number(m[5] || 0);
  const second = Number(m[6] || 0);
  const frac = m[7] || '';
  const ms = frac ? Number(frac.padEnd(3, '0').slice(0, 3)) : 0;
  const tz = m[8] || '';

  if (
    !Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day) ||
    !Number.isFinite(hour) || !Number.isFinite(minute) || !Number.isFinite(second) || !Number.isFinite(ms)
  ) return null;
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 60) return null;

  // If timezone is present, parse as absolute UTC.
  if (tz) {
    if (tz === 'Z') return Date.UTC(year, month - 1, day, hour, minute, second, ms) / 1000;
    const tzNorm = tz.includes(':') ? tz : `${tz.slice(0, 3)}:${tz.slice(3)}`;
    const sign = tzNorm[0] === '-' ? -1 : 1;
    const tzh = Number(tzNorm.slice(1, 3));
    const tzm = Number(tzNorm.slice(4, 6));
    if (!Number.isFinite(tzh) || !Number.isFinite(tzm)) return null;
    const offMin = sign * (tzh * 60 + tzm);
    const utcMs = Date.UTC(year, month - 1, day, hour, minute, second, ms) - offMin * 60000;
    return utcMs / 1000;
  }

  // No timezone: preserve previous browser-local interpretation.
  const localMs = new Date(year, month - 1, day, hour, minute, second, ms).getTime();
  if (!Number.isFinite(localMs)) return null;
  return localMs / 1000;
}

/**
 * Flexible timestamp parser — handles ISO 8601 variants, epoch ms.
 * Returns epoch seconds or null.
 */
export function parseTimestamp(value) {
  const raw = (value == null ? '' : String(value)).trim();
  if (!raw) return null;

  const numericEpoch = parseNumericEpoch(raw);
  if (numericEpoch != null) return numericEpoch;

  let s = normalizeIsoTimezone(raw);
  // Normalize common variants before parsing.
  s = s.replace(/^(\d{4}):(\d{2}):(\d{2})/, '$1-$2-$3'); // YYYY:MM:DD
  s = s.replace(/^(\d{4})\/(\d{2})\/(\d{2})/, '$1-$2-$3'); // YYYY/MM/DD
  if (/^\d{4}-\d{2}-\d{2}\s+\d/.test(s)) s = s.replace(' ', 'T'); // Safari-safe ISO

  // Try JS Date.parse (handles most ISO formats)
  const ms = Date.parse(s);
  if (Number.isFinite(ms)) return ms / 1000;

  // Deterministic fallback parser for iPad/Safari and non-standard separators.
  const manual = parseIsoLike(s.replace('T', ' '));
  if (manual != null) return manual;

  return null;
}

function parseTsVakaros(s) {
  return parseTimestamp(s);
}

function parseTsSensor(s) {
  const raw = (s || '').trim();
  if (!raw) return null;
  // Pure numeric → epoch seconds/milliseconds/microseconds (magnitude-based)
  const num = Number(raw);
  if (Number.isFinite(num) && /^[+-]?\d+(\.\d+)?$/.test(raw)) {
    const a = Math.abs(num);
    if (a > 1e14) return num / 1e6; // microseconds
    if (a > 1e11) return num / 1e3; // milliseconds
    if (a > 1e9) return num;        // seconds
    return num / 1000;              // legacy fallback (small values as ms)
  }
  return parseTimestamp(raw);
}


// ── Haversine (fast equirectangular approx) ───────────────────────────

function haversineFast(lat1, lon1, lat2, lon2) {
  const dlat = (lat2 - lat1) * 111320.0;
  const dlon = (lon2 - lon1) * 111320.0 * Math.cos(((lat1 + lat2) / 2.0) * Math.PI / 180);
  return Math.sqrt(dlat * dlat + dlon * dlon);
}


// ── GPS outlier filter ────────────────────────────────────────────────

/**
 * Multi-pass GPS outlier removal — ported from parsers.py
 * Removes points implying speed > maxSpeed m/s.
 */
export function filterGpsOutliers(pts, maxSpeed = MAX_PLAUSIBLE_SPEED_MS, maxPasses = 3) {
  if (pts.length < 3) return pts;

  for (let pass = 0; pass < maxPasses; pass++) {
    const n = pts.length;
    if (n < 3) break;
    const keep = new Array(n).fill(true);

    // Interior: both-neighbour check
    for (let i = 1; i < n - 1; i++) {
      const dtAB = Math.max(0.1, Math.abs(pts[i].ts - pts[i - 1].ts));
      const dtBC = Math.max(0.1, Math.abs(pts[i + 1].ts - pts[i].ts));
      const dAB = haversineFast(pts[i - 1].lat, pts[i - 1].lon, pts[i].lat, pts[i].lon);
      const dBC = haversineFast(pts[i].lat, pts[i].lon, pts[i + 1].lat, pts[i + 1].lon);
      if ((dAB / dtAB) > maxSpeed && (dBC / dtBC) > maxSpeed) {
        keep[i] = false;
      }
    }

    // Edge: first point
    {
      const dt = Math.max(0.1, Math.abs(pts[1].ts - pts[0].ts));
      const d = haversineFast(pts[0].lat, pts[0].lon, pts[1].lat, pts[1].lon);
      if (d / dt > maxSpeed) keep[0] = false;
    }

    // Edge: last point
    {
      const dt = Math.max(0.1, Math.abs(pts[n - 1].ts - pts[n - 2].ts));
      const d = haversineFast(pts[n - 2].lat, pts[n - 2].lon, pts[n - 1].lat, pts[n - 1].lon);
      if (d / dt > maxSpeed) keep[n - 1] = false;
    }

    const newPts = pts.filter((_, i) => keep[i]);
    if (newPts.length === pts.length) break;
    pts = newPts;
  }

  return pts;
}


// ── ms → epoch interpolation ──────────────────────────────────────────

function msToEpoch(msValue, anchors) {
  if (!anchors || anchors.length === 0) return null;
  if (anchors.length === 1) {
    return anchors[0][1] + (msValue - anchors[0][0]) / 1000;
  }

  // Binary search for insertion point
  let lo = 0, hi = anchors.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (anchors[mid][0] < msValue) lo = mid + 1;
    else hi = mid;
  }
  const idx = lo;

  let m0, t0, m1, t1;
  if (idx <= 0) {
    [m0, t0] = anchors[0];
    [m1, t1] = anchors[1];
  } else if (idx >= anchors.length) {
    [m0, t0] = anchors[anchors.length - 2];
    [m1, t1] = anchors[anchors.length - 1];
  } else {
    [m0, t0] = anchors[idx - 1];
    [m1, t1] = anchors[idx];
  }

  if (Math.abs(m1 - m0) < 1e-9) return t0 + (msValue - m0) / 1000;
  const slope = (t1 - t0) / (m1 - m0);
  return t0 + slope * (msValue - m0);
}


// ── Column picker ─────────────────────────────────────────────────────

function pickColumn(columnsLower, candidates) {
  for (const c of candidates) {
    if (c in columnsLower) return columnsLower[c];
  }
  return null;
}


// ── Main CSV parser ───────────────────────────────────────────────────

/**
 * Parse CSV text into a track.
 * @param {string} csvText — raw CSV text
 * @param {string} [sourceName] — filename for metadata
 * @returns {{ points: object[], metadata: object }}
 */
export function parseCsvTrack(csvText, sourceName = 'unknown') {
  // Parse CSV
  const lines = csvText.split(/\r?\n/);
  if (lines.length < 2) {
    return { points: [], metadata: { source: sourceName, reason: 'empty_csv' } };
  }

  // Parse header
  const headerLine = lines[0];
  const fieldnames = headerLine.split(',').map(f => f.trim().replace(/^"|"$/g, ''));
  if (fieldnames.length === 0) {
    return { points: [], metadata: { source: sourceName, reason: 'no_header' } };
  }

  // Parse rows
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const values = parseCSVLine(line);
    const row = {};
    for (let j = 0; j < fieldnames.length; j++) {
      row[fieldnames[j]] = j < values.length ? values[j] : '';
    }
    rows.push(row);
  }

  if (rows.length === 0) {
    return { points: [], metadata: { source: sourceName, reason: 'empty_csv' } };
  }

  // Build case-insensitive column map
  const columnsLower = {};
  for (const name of fieldnames) {
    if (name) columnsLower[name.trim().toLowerCase()] = name;
  }

  const has = (...names) => names.every(n => n in columnsLower);

  // Detect optional columns
  const sogCol = pickColumn(columnsLower, ['sog_mps', 'sog', 'sog_kts']);
  const heelCol = pickColumn(columnsLower, ['heel', 'heel_deg', 'roll_deg']);
  const trimCol = pickColumn(columnsLower, ['trim', 'trim_deg', 'pitch', 'pitch_deg']);
  const cogCol = pickColumn(columnsLower, ['cog', 'cog_deg', 'course', 'course_over_ground']);
  const hdgCol = pickColumn(columnsLower, ['hdg_true', 'hdg', 'heading', 'heading_deg', 'yaw', 'yaw_deg', 'mag_hdg']);
  const rudderCol = pickColumn(columnsLower, ['rudder_angle', 'rudder_angle_deg', 'rudder', 'rudder_deg']);
  const boomCol = pickColumn(columnsLower, ['boom_angle', 'boom_angle_deg', 'boom', 'boom_deg']);
  const trunkCol = pickColumn(columnsLower, ['trunk_angle', 'trunk_angle_deg', 'torso_angle', 'torso_angle_deg', 'trunk', 'torso']);

  // Detect format
  let latCol, lonCol, tsCol, tsMsCol, parseFn, detectedFormat;

  if (has('timestamp', 'latitude', 'longitude')) {
    // Format A: Vakaros-native
    latCol = columnsLower['latitude'];
    lonCol = columnsLower['longitude'];
    tsCol = columnsLower['timestamp'];
    tsMsCol = null;
    parseFn = parseTsVakaros;
    detectedFormat = 'vakaros_native';
  } else if (has('lat', 'lon') && (has('iso_time') || has('timestamp_ms'))) {
    // Format B: Sensor/IMU logger
    latCol = columnsLower['lat'];
    lonCol = columnsLower['lon'];
    tsCol = columnsLower['iso_time'] || columnsLower['timestamp_ms'];
    // Enable anchor interpolation when both iso_time and timestamp_ms exist
    tsMsCol = (columnsLower['iso_time'] && columnsLower['timestamp_ms'])
      ? columnsLower['timestamp_ms'] : null;
    parseFn = parseTsSensor;
    detectedFormat = 'sensor_logger';
  } else {
    // Generic fallback
    latCol = pickColumn(columnsLower, ['latitude', 'lat', 'gps_lat', 'gpslatitude']);
    lonCol = pickColumn(columnsLower, ['longitude', 'lon', 'lng', 'gps_lon', 'gpslongitude']);
    tsCol = pickColumn(columnsLower, ['timestamp', 'datetime', 'time', 'date_time', 'iso_time', 'gps_time', 'utc_time']);
    tsMsCol = pickColumn(columnsLower, ['timestamp_ms', 'time_ms', 'elapsed_ms', 'ms']);
    parseFn = parseTimestamp;
    detectedFormat = 'generic';
  }

  if (!latCol || !lonCol) {
    return {
      points: [],
      metadata: {
        source: sourceName, reason: 'missing_lat_lon',
        columns: fieldnames, detected_format: detectedFormat,
      },
    };
  }

  // Build ms→epoch anchors (for generic and sensor_logger formats)
  let anchors = [];
  if ((detectedFormat === 'generic' || detectedFormat === 'sensor_logger') && tsMsCol) {
    const isoCol = columnsLower['iso_time'];
    for (const row of rows) {
      const msVal = parseFloat_(row[tsMsCol]);
      if (msVal == null) continue;
      let epoch = tsCol ? parseTimestamp(row[tsCol]) : null;
      if (epoch == null && isoCol) epoch = parseTimestamp(row[isoCol]);
      if (epoch != null) anchors.push([msVal, epoch]);
    }
    anchors.sort((a, b) => a[0] - b[0]);
    // Dedup
    const deduped = [];
    let lastMs = null;
    for (const item of anchors) {
      if (lastMs == null || Math.abs(item[0] - lastMs) > 1e-9) {
        deduped.push(item);
        lastMs = item[0];
      }
    }
    anchors = deduped;
  }

  // Parse rows → points
  let points = [];
  for (const row of rows) {
    let ts;
    if (detectedFormat === 'generic') {
      ts = tsCol ? parseTimestamp(row[tsCol]) : null;
      if (ts == null) ts = parseTimestamp(row['iso_time']);
      if (ts == null && tsMsCol) {
        const msVal = parseFloat_(row[tsMsCol]);
        if (msVal != null) ts = msToEpoch(msVal, anchors);
      }
    } else if (detectedFormat === 'sensor_logger' && tsMsCol && anchors.length >= 1) {
      // Prefer direct iso_time, fall back to anchor interpolation via timestamp_ms
      ts = parseFn(row[tsCol] || '');
      if (ts == null) {
        const msVal = parseFloat_(row[tsMsCol]);
        if (msVal != null) ts = msToEpoch(msVal, anchors);
      }
    } else {
      ts = parseFn(row[tsCol] || '');
    }

    if (ts == null) continue;

    const lat = parseFloat_(row[latCol]);
    const lon = parseFloat_(row[lonCol]);
    const hasGps = lat != null && lon != null && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
    const pt = { ts };
    if (hasGps) {
      pt.lat = lat;
      pt.lon = lon;
    }
    if (hasGps && sogCol != null) {
      const v = parseFloat_(row[sogCol]);
      if (v != null) {
        // Normalise SOG to knots — sog_mps is m/s, convert; sog_kts is already knots
        const sogLower = sogCol.toLowerCase();
        pt.sog = sogLower === 'sog_mps' ? v * 1.94384 : v;
      }
    }
    if (heelCol != null) { const v = parseFloat_(row[heelCol]); if (v != null) pt.heel = v; }
    if (trimCol != null) { const v = parseFloat_(row[trimCol]); if (v != null) pt.trim = v; }
    let cogValue = null;
    if (cogCol != null) {
      const v = parseFloat_(row[cogCol]);
      if (v != null) {
        cogValue = v;
        pt.cog = v;
      }
    }
    let hdgValue = null;
    if (hdgCol != null) {
      const v = parseFloat_(row[hdgCol]);
      if (v != null) hdgValue = v;
    }
    if (hdgValue != null && (Math.abs(hdgValue) > 1e-9 || cogValue == null)) pt.hdg = hdgValue;
    else if (cogValue != null) pt.hdg = cogValue;
    if (rudderCol != null) { const v = parseFloat_(row[rudderCol]); if (v != null) pt.rudder = v; }
    if (boomCol != null) { const v = parseFloat_(row[boomCol]); if (v != null) pt.boom = v; }
    if (trunkCol != null) { const v = parseFloat_(row[trunkCol]); if (v != null) pt.trunk = v; }

    if (!hasGps && !['sog', 'heel', 'trim', 'cog', 'hdg', 'rudder', 'boom', 'trunk'].some(key => pt[key] != null)) continue;
    points.push(pt);
  }

  // Sort by timestamp
  points.sort((a, b) => a.ts - b.ts);

  // Dedup by timestamp
  {
    const deduped = [];
    let lastTs = null;
    for (const p of points) {
      if (lastTs == null || Math.abs(p.ts - lastTs) > 1e-9) {
        deduped.push(p);
        lastTs = p.ts;
      } else {
        Object.assign(deduped[deduped.length - 1], p);
      }
    }
    points = deduped;
  }

  // Remove GPS outliers without dropping telemetry-only rows from GPS dropouts.
  const nBefore = points.length;
  const gpsPoints = points.filter(p => Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lon)));
  const telemetryOnlyPoints = points.filter(p => !(Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lon))));
  const filteredGpsPoints = filterGpsOutliers(gpsPoints);
  points = [...filteredGpsPoints, ...telemetryOnlyPoints].sort((a, b) => a.ts - b.ts);
  const nRemoved = nBefore - points.length;

  return {
    points,
    metadata: {
      source: sourceName,
      detected_format: detectedFormat,
      columns: fieldnames,
      lat_col: latCol,
      lon_col: lonCol,
      ts_col: tsCol,
      ts_ms_col: tsMsCol,
      sog_col: sogCol,
      heel_col: heelCol,
      trim_col: trimCol,
      cog_col: cogCol,
      hdg_col: hdgCol,
      rudder_col: rudderCol,
      boom_col: boomCol,
      trunk_col: trunkCol,
      anchor_count: anchors.length,
      row_count: rows.length,
      point_count: points.length,
      gps_point_count: filteredGpsPoints.length,
      telemetry_only_point_count: telemetryOnlyPoints.length,
      outliers_removed: nRemoved,
    },
  };
}


// ── Robust CSV line parser (handles quoted fields) ────────────────────

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (i + 1 < line.length && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += c;
      }
    } else {
      if (c === '"') {
        inQuotes = true;
      } else if (c === ',') {
        result.push(current.trim());
        current = '';
      } else {
        current += c;
      }
    }
  }
  result.push(current.trim());
  return result;
}
