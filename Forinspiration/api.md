# TrollSports Live — Data API

Base URL: `https://<your-deployment>.trollsports-live.pages.dev/api/data`

All endpoints are read-only. POST/PUT/PATCH/DELETE return 405.

---

## Authentication

Every request (except `OPTIONS` preflight) requires an API key. Pass it one of two ways:

```
?api_key=<key>
Authorization: Bearer <key>
```

Missing or wrong key → `401 UNAUTHORIZED`.

---

## Response Envelope

Every response is JSON with this shape:

```json
{ "success": true, "data": [...], "meta": { "count": 10, "has_more": false } }
{ "success": false, "error": "NOT_FOUND", "message": "Session not found" }
```

`meta` fields vary by endpoint (documented below). Error codes: `UNAUTHORIZED` (401), `BAD_REQUEST` (400), `NOT_FOUND` (404), `METHOD_NOT_ALLOWED` (405), `RATE_LIMITED` (429), `DB_UNAVAILABLE` (503), `INTERNAL_ERROR` (500).

---

## Rate Limits

| Bucket | Limit |
|---|---|
| Unauthenticated (wrong / no key) | 10 req/min/IP |
| Authenticated | 120 req/min/IP |

Exceeding the limit returns `429 RATE_LIMITED` with a `retry_after` field (seconds until the next window).

---

## Endpoints

### `GET /api/data/health`

DB connectivity check. Useful to verify your API key and that the service is up.

```bash
curl "https://<host>/api/data/health?api_key=<key>"
```

```json
{ "success": true, "data": { "status": "ok" }, "meta": { "timestamp": 1746350000000 } }
```

---

### `GET /api/data/sessions`

List all sessions and recordings, newest first.

**Query parameters**

| Param | Default | Description |
|---|---|---|
| `limit` | 50 | Results per page (1–200) |
| `offset` | 0 | Pagination offset |
| `type` | all | Filter: `recording`, `highlight`, `upload`, or `legacy` |
| `from` | — | Only include sessions with `start_time ≥ from` (Unix ms) |
| `to` | — | Only include sessions with `start_time ≤ to` (Unix ms) |
| `athlete` | — | Substring match on athlete names (max 100 chars) |

**Example**

```bash
# First page
curl "https://<host>/api/data/sessions?api_key=<key>&limit=20"

# Filter by athlete and date range
curl "https://<host>/api/data/sessions?api_key=<key>&athlete=Eirik&from=1740000000000&to=1746400000000"
```

**Response data item**

```json
{
  "id":            "rec-abc123",
  "display_name":  "Tuesday practice",
  "type":          "recording",
  "source":        "browser-recording",
  "start_time":    1746300000000,
  "end_time":      1746303600000,
  "unit_ids":      ["unit-01", "unit-02"],
  "athlete_names": { "unit-01": "Eirik", "unit-02": "Jonas" },
  "location":      "Oslofjord"
}
```

**Meta**

```json
{ "count": 20, "has_more": true, "limit": 20, "offset": 0 }
```

Use `offset += limit` and repeat while `has_more: true` to page through all results.

---

### `GET /api/data/sessions/:id`

Metadata for one session or recording, plus summary counts for marks and wind samples.

```bash
curl "https://<host>/api/data/sessions/rec-abc123?api_key=<key>"
```

**Response data** — same fields as the list endpoint plus:

```json
{
  "duration_ms": 3600000,
  "summary": { "marks": 3, "wind_samples": 720 }
}
```

---

### `GET /api/data/sessions/:id/telemetry`

Sensor data for a session. Returns downsampled data by default to keep response sizes manageable. Use `raw=true` for full resolution with paging.

**Query parameters**

| Param | Default | Description |
|---|---|---|
| `from` | session start | Sub-window start (Unix ms) |
| `to` | session end | Sub-window end (Unix ms). Max range: 24 hours |
| `units` | all | Comma-separated unit IDs to include (max 10, e.g. `unit-01,unit-02`) |
| `fields` | all | Comma-separated column names to return (see field list below) |
| `max_points` | 500 | Max data points per unit — drives the time-bucket size (50–5000) |
| `raw` | false | `true` = skip downsampling, return every row with paging |
| `limit` | 2000 | Rows per page when `raw=true` (max 5000) |
| `offset` | 0 | Pagination offset when `raw=true` |

**Available fields**

```
timestamp   unit_id     custom_name
lat         lon         alt
roll        pitch       yaw
sog         cog         hdop
gnss_ms     gnss_iso
mag_x       mag_y       mag_z
rudder_angle  boom_angle  torso_angle  seq
```

`timestamp` and `unit_id` are always included regardless of `fields`.

**Examples**

```bash
# GPS track only — 200 points per unit
curl "https://<host>/api/data/sessions/rec-abc123/telemetry?api_key=<key>&fields=lat,lon,sog,cog&max_points=200"

# IMU data for one unit, sub-window
curl "https://<host>/api/data/sessions/rec-abc123/telemetry?api_key=<key>&units=unit-01&fields=roll,pitch,yaw&from=1746300000000&to=1746301800000"

# Full resolution, first 100 rows
curl "https://<host>/api/data/sessions/rec-abc123/telemetry?api_key=<key>&raw=true&limit=100"

# Next page
curl "https://<host>/api/data/sessions/rec-abc123/telemetry?api_key=<key>&raw=true&limit=100&offset=100"
```

**Downsampled meta**

```json
{
  "count": 412,
  "downsampled": true,
  "bucket_ms": 1800,
  "max_points": 500,
  "from": 1746300000000,
  "to": 1746303600000,
  "units": "all",
  "fields": ["timestamp", "unit_id", "lat", "lon", "sog", "cog"]
}
```

**Raw meta**

```json
{
  "count": 100,
  "has_more": true,
  "downsampled": false,
  "from": 1746300000000,
  "to": 1746303600000,
  "units": ["unit-01"],
  "fields": ["timestamp", "unit_id", "roll", "pitch", "yaw"],
  "limit": 100,
  "offset": 0
}
```

---

### `GET /api/data/sessions/:id/marks`

Race course mark positions (top mark, start line A/B).

```bash
curl "https://<host>/api/data/sessions/rec-abc123/marks?api_key=<key>"
```

**Response data item**

```json
{ "mark_type": "top_mark", "lat": 59.9000, "lon": 10.6000, "timestamp": 1746300100000 }
```

`mark_type` values: `top_mark`, `start_line_a`, `start_line_b`.

---

### `GET /api/data/sessions/:id/wind`

Wind speed and direction samples (~5s intervals during a recording).

```bash
curl "https://<host>/api/data/sessions/rec-abc123/wind?api_key=<key>"
```

**Response data item**

```json
{
  "timestamp":      1746300000000,
  "wind_direction": 225.0,
  "wind_speed_kt":  12.5,
  "source":         "easywind"
}
```

`wind_direction` is in degrees, meteorological FROM convention (225 = wind blowing from SW).

---

## Using the API from JavaScript

```js
const BASE = 'https://<host>/api/data';
const KEY  = 'your-api-key';

async function apiFetch(path, params = {}) {
  const url = new URL(`${BASE}${path}`);
  url.searchParams.set('api_key', KEY);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const res = await fetch(url);
  const json = await res.json();
  if (!json.success) throw new Error(`${json.error}: ${json.message}`);
  return json;
}

// List sessions
const { data: sessions } = await apiFetch('/sessions', { limit: 20 });

// Get GPS track for the first session
const { data: telemetry } = await apiFetch(`/sessions/${sessions[0].id}/telemetry`, {
  fields: 'lat,lon,sog',
  max_points: 500,
});

// All marks
const { data: marks } = await apiFetch(`/sessions/${sessions[0].id}/marks`);
```

---

## Tips

- **Pagination**: use `offset += limit` and repeat while `meta.has_more === true`.
- **Timestamps**: all timestamps are Unix milliseconds (multiply by 1000 when passing to `new Date()`).
- **Large sessions**: use `fields` to request only the columns you need, and keep `max_points` ≤ 1000 for fast responses.
- **Sub-windows**: use `from` and `to` on the telemetry endpoint to load a narrow slice of a long session. Max window is 24 hours.
- **Multiple units**: `units=unit-01,unit-02` returns interleaved rows ordered by `unit_id, timestamp`. Group by `unit_id` client-side.

---

## CORS

All responses include `Access-Control-Allow-Origin: *` by default, so you can call the API from any browser page. If the deployment is locked to a specific origin, requests from other origins will be blocked.
