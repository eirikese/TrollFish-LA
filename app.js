/* TrollFish app.js — ES Module — Timeline + Dual Viewer rewrite */
/* ── Browser-side modules (IndexedDB, GPS parsing, matching) ──── */
import * as DB from './modules/db.js';
import * as FM from './modules/file-manager.js';
import * as Pipeline from './modules/pipeline.js?v=20260604pose2';
import * as PoseEngine from './modules/pose-engine.js?v=20260604pose2';
import * as Storage from './modules/storage.js';
import { matchVideoTracksToCsv } from './modules/matcher.js?v=20260527detail1';
import { buildReportData, generateDensityGrid } from './modules/report-builder.js?v=20260604pose2';
import { generatePdf, renderKdeHistogram } from './modules/report-pdf.js?v=20260527detail1';
import {
  DEFAULT_MANEUVER_DETECTION_SETTINGS,
  MANEUVER_ANALYSIS_SCHEMA_VERSION,
  buildManeuverAnalysis,
  detectProjectManeuvers,
  normalizeManeuverDetectionSettings,
} from './modules/maneuvers.js?v=20260527detail1';
import { computeCenterOfMass } from './modules/skeleton-metrics.js';
import { combineLocalWindEstimates, combineSessionWindEstimates, estimateWindFromCsvPoints } from './modules/wind-estimation.js?v=20260418wind2';
import { PALETTE as _PALETTE, SOG_MAX_KT as _SOG_MAX } from './modules/config.js';

const state = {
  projectId: null,
  projects: [],
  mapData: null,
  cvStatuses: {},
  cvConfig: {},
  windConfig: { disableTowFilteringTrusted: true },
  athletes: [],
  fileMeta: {},
  trackLayers: {},
  posMarkers: {},
  map: null,
  maneuverMap: null,
  maneuverMapLayers: {
    tracks: null,
    markers: null,
    selection: null,
    wind: null,
  },
  videoColors: {},
  jobId: null,
  jobPollTimer: null,
  editingAthleteId: null,
  // Project-level segments (absolute epoch timestamps)
  segments: [],
  segmentSelect: {
    active: false,
    step: null,      // 'start' | 'end'
    tsStart: null,
    tsEnd: null,
    startMarker: null,
    endMarker: null,
    highlightLayer: null,
  },
  mediapipeWorkers: 2,
  skeletonCoverage: {},
  trackVisibility: {},
  analysisSelected: [],
  maneuvers: [],
  maneuverSelected: [],
  maneuverChecked: [],
  maneuverComparePicking: false,
  maneuverCompareOffsets: {},
  maneuverViewZoom: 1,
  manualManeuverDraft: {
    active: false,
    step: 'start',
    trackFileId: null,
    trackPoints: [],
    tsStart: null,
    tsEnd: null,
    startMarker: null,
    endMarker: null,
    highlightLayer: null,
  },
  maneuverFilters: {
    athleteId: '',
    type: 'all',
    side: 'all',
  },
  maneuverDetection: { ...DEFAULT_MANEUVER_DETECTION_SETTINGS },
  maneuverLayerGroup: null,
  maneuverRefreshToken: 0,
  maneuverRefreshPromise: null,
  advancedMode: false,
  advancedPane: {
    mode: null,
    segmentId: null,
    heatmapLoadToken: 0,
    maneuverLoadToken: 0,
    maneuverIds: [],
  },
  inlineHeatmaps: {
    visible: false,
    loadToken: 0,
    segmentId: null,
    results: null,
    loading: false,
    renderedSegmentId: null,
    renderedLoadToken: 0,
    expandedOverlayType: null,
    expandedOverlayKey: null,
    previousVideoLayout: null,
  },
  reportOptions: {
    summaryStats: true,
    histograms: true,
    heatmaps: true,
    boomAngle: true,
    polarPlots: false,
    maneuverAnalysis: false,
    downloadCsv: false,
  },
  advancedFeatures: {
    maneuversTab: false,
    hull3d: false,
    windPanel: false,
    boomPredictions: true,
    rudderPredictions: true,
  },
  videoLayout: 'auto',
  hiddenVideoSlots: {},
  timelineStatsWindowSec: 1,
  timelineStatOverlayGraph: false,
  timelineSogGapStitching: false,
  poseInputMaxDim: 480,
  poseExactSegmentSeek: false,
  timelineMetricOverlayPrefs: {},
  uploadInProgress: false,
  apiCsv: {
    baseUrl: '',
    apiKey: '',
  },
  mapShouldAutoFit: true,
  mapBaseZoom: null,
  wind: {
    loading: false,
    promise: null,
    loadToken: 0,
    byCsvId: {},
    session: null,
    localNow: null,
    layerGroup: null,
    control: null,
  },
  _activeTrackId: null,
  _topbarSegmentId: null,
  _topbarSegmentName: '',
  reportOverlayOpen: false,
  phonePlayback: {
    enabled: false,
    selectedFileId: null,
    currentFileId: null,
    desiredFileId: null,
    videoEl: null,
    paneVisible: false,
    savedLeftWidthPx: null,
    savedActiveLeftWidthPx: null,
    savedShellWidthPx: null,
    savedVideoGridWidthPx: null,
    lastDriftCorrectAt: 0,
    switchReqId: 0,
  },
  segmentModal: {
    mode: 'create',
    segmentId: null,
  },
  wizardStep: 1,
  // Timeline state
  tl: {
    playing: false,
    playbackRate: 1,
    globalStart: null,   // epoch seconds (min ts across all data)
    globalEnd: null,      // epoch seconds (max ts across all data)
    currentTs: null,      // current absolute epoch second
    animFrameId: null,
    lastFrameTime: null,
    athleteSlots: [],     // [{athleteId, color, name, fileId, videoEl, paneEl, pts, tsStart, tsEnd, sogPts}]
    viewStart: null,       // zoomed view start (epoch sec), null = use globalStart
    viewEnd: null,         // zoomed view end (epoch sec), null = use globalEnd
  },
};

const PALETTE = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#00bcd4','#ff6b6b','#4ecdc4'];
const SOG_MAX_KT = 22; // SOG values above this are treated as outliers and ignored
const ATHLETE_LIBRARY_KEY = 'trollfish_athleteLibrary_v1';
const CSV_ATHLETE_MEMORY_KEY = 'trollfish_csvAthleteMemory_v1';
const CSV_MEMORY_PREFIX_LEN = 6;
const ADVANCED_MODE_KEY = 'trollfish_advancedMode_v1';
const ADVANCED_FEATURES_KEY = 'trollfish_advancedFeatures_v1';
const VIDEO_LAYOUT_KEY = 'trollfish_videoLayout_v1';
const HIDDEN_VIDEO_SLOTS_KEY_PREFIX = 'trollfish_hiddenVideoSlots_';
const TL_STATS_WINDOW_KEY = 'trollfish_tlStatsWindowSec_v1';
const TL_STATS_OVERLAY_GRAPH_KEY = 'trollfish_tlStatsOverlayGraph_v1';
const TL_SOG_GAP_STITCHING_KEY = 'trollfish_tlSogGapStitching_v1';
const POSE_INPUT_MAX_DIM_KEY = 'trollfish_poseInputMaxDim_v1';
const POSE_EXACT_SEGMENT_SEEK_KEY = 'trollfish_poseExactSegmentSeek_v1';
const API_CSV_CONFIG_KEY = 'trollfish_apiCsvConfig_v1';
const REPORT_OPTIONS_KEY = 'trollfish_reportOptions_v1';
const REPORT_HISTORY_KEY_PREFIX = 'trollfish_reportHistory_';
const REPORT_STORAGE_DIR = ['reports'];
const MANEUVER_SPLIT_KEY = 'trollfish_maneuverSplitPx_v1';
const DEFAULT_WIND_CONFIG = Object.freeze({
  disableTowFilteringTrusted: true,
});
const DEFAULT_ADVANCED_FEATURES = Object.freeze({
  maneuversTab: false,
  hull3d: false,
  windPanel: false,
  boomPredictions: true,
  rudderPredictions: true,
});
const IS_IPAD = (() => {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  return /iPad/i.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
})();
const USE_IPAD_VIDEO_WORKAROUNDS = IS_IPAD;
const IPAD_VIDEO_SWITCH_STAGGER_MS = 45;
const IPAD_SLOT_VIDEO_CACHE_LIMIT = 2;
const IPAD_HEAVY_VIDEO_PROJECT_THRESHOLD = 3;
const IPAD_LIGHT_VIDEO_URL_CACHE_LIMIT = 6;
const IPAD_HEAVY_VIDEO_URL_CACHE_LIMIT = 4;
const IPAD_HEAVY_SOG_POINT_LIMIT = 720;
let _ipadVideoSwitchQueue = Promise.resolve();
const _timelineProcessedMetricCache = new Map(); // `${projectId}:${fileId}` -> { loadToken, ready, promise }
let _timelineMetricLoadToken = 0;
const TIMELINE_METRIC_LOAD_ABORTED = 'timeline_metric_load_aborted';
const TIMELINE_PROCESSED_STATS_REFRESH_MS = 90;
// Timeline stat readouts refresh at 1 Hz and show the mean of the trailing second.
const TIMELINE_STATS_UI_REFRESH_MS = 1000;
const TIMELINE_STATS_MEAN_WINDOW_SEC = 1;
// During playback the timeline advances every animation frame (~60fps) for smooth
// video, but the secondary timeline UI (markers, SOG canvas, labels, layout, stats)
// does not need 60fps. Throttling it to ~20fps cuts per-frame DOM/canvas work
// roughly 3x — a big win on machines that struggle with dual video decode.
const TIMELINE_UI_REFRESH_MS = 50;
const TIMELINE_INSTANT_METRIC_MAX_GAP_SEC = 4.0;
const TIMELINE_VMG_MOTION_MAX_GAP_SEC = 8.0;
const TIMELINE_STAT_OVERLAY_WINDOW_SEC = 12;
const TIMELINE_STAT_OVERLAY_MAX_POINTS = 160;
const TIMELINE_STAT_OVERLAY_GRAPH_HEIGHT = 140;
const TIMELINE_STAT_OVERLAY_GRAPH_GAP = 10;
const TIMELINE_STAT_OVERLAY_GRAPH_PAD_TOP = 12;
const TIMELINE_STAT_OVERLAY_GRAPH_PAD_BOTTOM = 12;
const MANEUVER_POSE_TARGET_FPS = 15;
const PHONE_PLAYBACK_LEFT_MIN_WIDTH = 280;
const TIMELINE_TRACK_SERIES_OPTS = Object.freeze({
  gapMultiplier: 8,
  minInterpolationGapSec: 4,
  });

function getProjectVideoCount() {
  return Array.isArray(state.mapData?.videos) ? state.mapData.videos.length : 0;
}

function isIpadHeavyVideoProject(videoCount = getProjectVideoCount()) {
  return USE_IPAD_VIDEO_WORKAROUNDS && videoCount > IPAD_HEAVY_VIDEO_PROJECT_THRESHOLD;
}

function getVideoURLCacheLimit() {
  if (!USE_IPAD_VIDEO_WORKAROUNDS) return 64;
  return isIpadHeavyVideoProject() ? IPAD_HEAVY_VIDEO_URL_CACHE_LIMIT : IPAD_LIGHT_VIDEO_URL_CACHE_LIMIT;
}

function decimateSeriesByCount(points, maxPoints) {
  if (!Array.isArray(points) || points.length <= maxPoints || maxPoints < 3) return points || [];
  const out = [];
  const lastIdx = points.length - 1;
  const step = lastIdx / (maxPoints - 1);
  let prevIdx = -1;
  for (let i = 0; i < maxPoints; i++) {
    const idx = Math.max(0, Math.min(lastIdx, Math.round(i * step)));
    if (idx === prevIdx) continue;
    out.push(points[idx]);
    prevIdx = idx;
  }
  if (out[out.length - 1] !== points[lastIdx]) out.push(points[lastIdx]);
  return out;
}

function optimizeSogPointsForDevice(points) {
  if (!isIpadHeavyVideoProject()) return points;
  return decimateSeriesByCount(points, IPAD_HEAVY_SOG_POINT_LIMIT);
}

function invalidateTimelineProcessedMetricCache(fileId = null, projectId = state.projectId) {
  const pid = String(projectId || '');
  if (!fileId) {
    for (const key of [..._timelineProcessedMetricCache.keys()]) {
      if (!pid || key.startsWith(`${pid}:`)) _timelineProcessedMetricCache.delete(key);
    }
    return;
  }
  _timelineProcessedMetricCache.delete(`${pid}:${fileId}`);
}

function loadJsonLocal(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
}

function saveJsonLocal(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}

function clampTimelineStatsWindowSec(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 1;
  return Math.max(1, Math.min(10, Math.round(num)));
}

function loadTimelineStatsWindowSetting() {
  state.timelineStatsWindowSec = clampTimelineStatsWindowSec(loadJsonLocal(TL_STATS_WINDOW_KEY, 1));
  const input = el('tl-stats-window-input');
  if (input) input.value = String(state.timelineStatsWindowSec);
}

function setTimelineStatsWindowSec(value) {
  state.timelineStatsWindowSec = clampTimelineStatsWindowSec(value);
  saveJsonLocal(TL_STATS_WINDOW_KEY, state.timelineStatsWindowSec);
  const input = el('tl-stats-window-input');
  if (input) input.value = String(state.timelineStatsWindowSec);
  updateTimelineStats();
}

function loadTimelineStatOverlayGraphSetting() {
  state.timelineStatOverlayGraph = !!loadJsonLocal(TL_STATS_OVERLAY_GRAPH_KEY, false);
  const toggle = el('tl-stats-overlay-graph-toggle');
  if (toggle) toggle.checked = !!state.timelineStatOverlayGraph;
}

function setTimelineStatOverlayGraph(enabled) {
  state.timelineStatOverlayGraph = !!enabled;
  saveJsonLocal(TL_STATS_OVERLAY_GRAPH_KEY, state.timelineStatOverlayGraph);
  const toggle = el('tl-stats-overlay-graph-toggle');
  if (toggle) toggle.checked = !!state.timelineStatOverlayGraph;
  updateTimelineStats();
}

function loadTimelineSogGapStitchingSetting() {
  state.timelineSogGapStitching = !!loadJsonLocal(TL_SOG_GAP_STITCHING_KEY, false);
  const toggle = el('tl-sog-gap-stitching-toggle');
  if (toggle) toggle.checked = !!state.timelineSogGapStitching;
}

function setTimelineSogGapStitching(enabled) {
  state.timelineSogGapStitching = !!enabled;
  saveJsonLocal(TL_SOG_GAP_STITCHING_KEY, state.timelineSogGapStitching);
  const toggle = el('tl-sog-gap-stitching-toggle');
  if (toggle) toggle.checked = !!state.timelineSogGapStitching;
  buildTimeline();
  updateTimelineStats();
}

function normalizePoseInputMaxDim(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 480;
  if (num <= 360) return 360;
  if (num >= 640) return 640;
  return 480;
}

function loadPoseProcessingSettings() {
  state.poseInputMaxDim = normalizePoseInputMaxDim(loadJsonLocal(POSE_INPUT_MAX_DIM_KEY, 480));
  state.poseExactSegmentSeek = !!loadJsonLocal(POSE_EXACT_SEGMENT_SEEK_KEY, false);
  const sizeSelect = el('pose-input-size-select');
  if (sizeSelect) sizeSelect.value = String(state.poseInputMaxDim);
  const exactToggle = el('pose-exact-segment-seek-toggle');
  if (exactToggle) exactToggle.checked = !!state.poseExactSegmentSeek;
}

function setPoseInputMaxDim(value) {
  state.poseInputMaxDim = normalizePoseInputMaxDim(value);
  saveJsonLocal(POSE_INPUT_MAX_DIM_KEY, state.poseInputMaxDim);
  const sizeSelect = el('pose-input-size-select');
  if (sizeSelect) sizeSelect.value = String(state.poseInputMaxDim);
}

function setPoseExactSegmentSeek(enabled) {
  state.poseExactSegmentSeek = !!enabled;
  saveJsonLocal(POSE_EXACT_SEGMENT_SEEK_KEY, state.poseExactSegmentSeek);
  const exactToggle = el('pose-exact-segment-seek-toggle');
  if (exactToggle) exactToggle.checked = !!state.poseExactSegmentSeek;
}

function syncManeuverDetectionInputs() {
  const settings = normalizeManeuverDetectionSettings(state.maneuverDetection || DEFAULT_MANEUVER_DETECTION_SETTINGS);
  state.maneuverDetection = settings;
  const headingIn = el('maneuver-min-heading-input');
  const stableIn = el('maneuver-min-stable-input');
  const statsIn = el('maneuver-stats-window-input');
  if (headingIn) headingIn.value = String(settings.minHeadingDeltaDeg);
  if (stableIn) stableIn.value = String(settings.minStableSideSec);
  if (statsIn) statsIn.value = String(settings.statsWindowSec);

  const towToggle = el('advanced-tow-filter-toggle');
  if (towToggle) towToggle.checked = !state.windConfig?.disableTowFilteringTrusted;
}

function normalizeWindConfig(config = {}) {
  return {
    disableTowFilteringTrusted: config?.disableTowFilteringTrusted == null
      ? DEFAULT_WIND_CONFIG.disableTowFilteringTrusted
      : !!config.disableTowFilteringTrusted,
  };
}

async function loadProjectCvConfig() {
  state.cvConfig = {};
  state.maneuverDetection = { ...DEFAULT_MANEUVER_DETECTION_SETTINGS };
  state.windConfig = { ...DEFAULT_WIND_CONFIG };
  if (!state.projectId) {
    syncManeuverDetectionInputs();
    return;
  }
  try {
    const rec = await DB.getCvConfig(state.projectId);
    state.cvConfig = rec?.config ? { ...rec.config } : {};
  } catch {
    state.cvConfig = {};
  }
  state.maneuverDetection = normalizeManeuverDetectionSettings(
    state.cvConfig?.maneuverDetection || DEFAULT_MANEUVER_DETECTION_SETTINGS
  );
  state.windConfig = normalizeWindConfig(state.cvConfig?.windConfig || DEFAULT_WIND_CONFIG);
  syncManeuverDetectionInputs();
}

async function saveProjectCvConfig() {
  if (!state.projectId) return;
  state.cvConfig = {
    ...(state.cvConfig || {}),
    maneuverDetection: normalizeManeuverDetectionSettings(state.maneuverDetection),
    windConfig: normalizeWindConfig(state.windConfig || DEFAULT_WIND_CONFIG),
  };
  await DB.upsertCvConfig(state.projectId, state.cvConfig);
}

function getReportHistoryKey(projectId) {
  return `${REPORT_HISTORY_KEY_PREFIX}${projectId || ''}`;
}

function loadReportHistory(projectId) {
  if (!projectId) return [];
  const list = loadJsonLocal(getReportHistoryKey(projectId), []);
  if (!Array.isArray(list)) return [];
  return list
    .map(r => ({
      id: String(r?.id || ''),
      filename: String(r?.filename || ''),
      name: String(r?.name || 'Report'),
      createdAt: Number.isFinite(Number(r?.createdAt)) ? Number(r.createdAt) : Date.now(),
      segmentIds: Array.isArray(r?.segmentIds) ? r.segmentIds.map(String) : [],
    }))
    .filter(r => r.id && r.filename);
}

function saveReportHistory(projectId, reports) {
  if (!projectId) return;
  saveJsonLocal(getReportHistoryKey(projectId), reports.slice(0, 100));
}

function sanitizeFilenamePart(text) {
  return String(text || '')
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 120);
}

function buildReportDisplayName(segmentIds) {
  const byId = new Map((state.segments || []).map(s => [String(s.id), s]));
  const names = (segmentIds || [])
    .map(id => byId.get(String(id))?.name)
    .filter(Boolean);
  if (!names.length) return 'Report';
  if (names.length === 1) return names[0];
  if (names.length === 2) return `${names[0]} + ${names[1]}`;
  return `${names[0]} + ${names.length - 1} more`;
}

function getOrderedReportSegments(segmentIds) {
  const byId = new Map((state.segments || []).map(s => [String(s.id), s]));
  return (segmentIds || [])
    .map(id => byId.get(String(id)))
    .filter(Boolean)
    .sort((a, b) => {
      const aStart = Number.isFinite(Number(a?.tsStart)) ? Number(a.tsStart) : Number.POSITIVE_INFINITY;
      const bStart = Number.isFinite(Number(b?.tsStart)) ? Number(b.tsStart) : Number.POSITIVE_INFINITY;
      if (aStart !== bStart) return aStart < bStart ? -1 : 1;
      const aEnd = Number.isFinite(Number(a?.tsEnd)) ? Number(a.tsEnd) : Number.POSITIVE_INFINITY;
      const bEnd = Number.isFinite(Number(b?.tsEnd)) ? Number(b.tsEnd) : Number.POSITIVE_INFINITY;
      if (aEnd !== bEnd) return aEnd < bEnd ? -1 : 1;
      return String(a?.name || '').localeCompare(String(b?.name || ''));
    });
}

function buildReportAthleteFileStem(segmentIds) {
  const seen = new Set();
  const names = [];
  for (const seg of getOrderedReportSegments(segmentIds)) {
    const athleteNames = getSegmentAthletes(seg)
      .map(a => String(a?.name || '').trim())
      .filter(Boolean)
      .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
    for (const name of athleteNames) {
      const key = name.toLocaleLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      names.push(name);
    }
  }
  if (!names.length) return null;
  return sanitizeFilenamePart(names.join('_'));
}

function buildReportFileName(segmentIds, createdAt = Date.now()) {
  const base = buildReportAthleteFileStem(segmentIds)
    || sanitizeFilenamePart(buildReportDisplayName(segmentIds))
    || 'report';
  const date = new Date(createdAt).toISOString().split('T')[0];
  return `${base}_${date}_report.pdf`;
}

function buildReportCsvFileName(segmentIds, createdAt = Date.now()) {
  const base = buildReportAthleteFileStem(segmentIds)
    || sanitizeFilenamePart(buildReportDisplayName(segmentIds))
    || 'segments';
  const date = new Date(createdAt).toISOString().split('T')[0];
  return `${base}_${date}_boom_rudder_timeseries.csv`;
}

function buildManeuverCsvFileName(maneuvers = [], createdAt = Date.now()) {
  const first = Array.isArray(maneuvers) ? maneuvers[0] : null;
  const athlete = sanitizeFilenamePart(first?.athlete_name || 'maneuver');
  const type = sanitizeFilenamePart(getManeuverTypeLabel(first).toLowerCase()) || 'maneuver';
  const suffix = maneuvers.length > 1 ? `${maneuvers.length}_maneuvers` : type;
  const date = new Date(createdAt).toISOString().split('T')[0];
  return `${athlete || 'maneuver'}_${suffix}_${date}_boom_rudder_timeseries.csv`;
}

function formatDateTime(ts) {
  if (!Number.isFinite(Number(ts))) return '--';
  const d = new Date(Number(ts));
  return d.toLocaleString();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function csvEscape(value) {
  if (value == null) return '';
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function csvNumber(value, digits = null) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '';
  return Number.isFinite(Number(digits)) ? n.toFixed(Number(digits)) : String(n);
}

function makeCsv(headers, rows) {
  const lines = [
    headers.map(csvEscape).join(','),
    ...rows.map(row => headers.map(header => csvEscape(row?.[header])).join(',')),
  ];
  return `${lines.join('\r\n')}\r\n`;
}

function normalizeApiCsvBaseUrl(value) {
  return String(value || '').trim().replace(/\/+$/, '');
}

function loadApiCsvConfig() {
  const saved = loadJsonLocal(API_CSV_CONFIG_KEY, {});
  state.apiCsv = {
    baseUrl: normalizeApiCsvBaseUrl(saved?.baseUrl),
    apiKey: String(saved?.apiKey || '').trim(),
  };
  syncApiCsvInputs();
}

function saveApiCsvConfigFromInputs() {
  const baseUrl = normalizeApiCsvBaseUrl(el('api-csv-url')?.value);
  const apiKey = String(el('api-csv-key')?.value || '').trim();
  state.apiCsv = { baseUrl, apiKey };
  saveJsonLocal(API_CSV_CONFIG_KEY, state.apiCsv);
  syncApiCsvInputs('Saved. Video uploads with GPS time will fetch matching API CSV tracks.', 'ok');
}

function syncApiCsvInputs(message = null, tone = '') {
  const urlInput = el('api-csv-url');
  const keyInput = el('api-csv-key');
  const status = el('api-csv-status');
  if (urlInput && urlInput.value !== (state.apiCsv?.baseUrl || '')) urlInput.value = state.apiCsv?.baseUrl || '';
  if (keyInput && keyInput.value !== (state.apiCsv?.apiKey || '')) keyInput.value = state.apiCsv?.apiKey || '';
  if (status) {
    status.className = tone || '';
    status.textContent = message || (
      hasApiCsvConfig()
        ? 'Ready: uploaded videos with GPS time will fetch matching API telemetry as CSV tracks.'
        : 'Optional: enter the telemetry API URL and password to auto-fetch CSV tracks from video GPS time.'
    );
  }
}

function hasApiCsvConfig() {
  return !!normalizeApiCsvBaseUrl(state.apiCsv?.baseUrl) && !!String(state.apiCsv?.apiKey || '').trim();
}

function buildApiCsvUrl(path, params = {}) {
  const base = normalizeApiCsvBaseUrl(state.apiCsv?.baseUrl);
  if (!base) throw new Error('Enter the telemetry API URL.');
  const apiRoot = base.endsWith('/api/data') ? base : `${base}/api/data`;
  const url = new URL(`${apiRoot}${path.startsWith('/') ? path : `/${path}`}`);
  for (const [key, value] of Object.entries(params || {})) {
    if (value == null || value === '') continue;
    url.searchParams.set(key, String(value));
  }
  return url;
}

async function fetchApiCsvJson(path, params = {}) {
  const apiKey = String(state.apiCsv?.apiKey || '').trim();
  if (!apiKey) throw new Error('Enter the telemetry API password.');
  const res = await fetch(buildApiCsvUrl(path, params), {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  let json = null;
  try {
    json = await res.json();
  } catch {
    throw new Error(`Telemetry API returned ${res.status} instead of JSON.`);
  }
  if (!res.ok || !json?.success) {
    throw new Error(json?.message || json?.error || `Telemetry API error ${res.status}`);
  }
  return json;
}

function apiCsvFileSafe(value, fallback = 'api') {
  return sanitizeFilenamePart(String(value || '').trim()) || fallback;
}

function normalizeApiEpochMs(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  const abs = Math.abs(n);
  if (abs > 1e14) return Math.round(n / 1000);
  if (abs > 1e11) return Math.round(n);
  return Math.round(n * 1000);
}

function apiRowValue(row, names) {
  if (!row || typeof row !== 'object') return undefined;
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(row, name)) return row[name];
  }
  const lowerMap = new Map(Object.keys(row).map(key => [String(key).toLowerCase(), key]));
  for (const name of names) {
    const actual = lowerMap.get(String(name).toLowerCase());
    if (actual != null) return row[actual];
  }
  return undefined;
}

function apiRowNumber(row, names) {
  const n = Number(apiRowValue(row, names));
  return Number.isFinite(n) ? n : null;
}

function isContinuousApiSession(session) {
  const id = String(session?.id || '');
  const type = String(session?.type || '').toLowerCase();
  const source = String(session?.source || '').toLowerCase();
  return /^cont-\d{4}-\d{2}-\d{2}$/.test(id) || type === 'legacy' || source === 'continuous';
}

function buildContinuousApiSessionsForRange(fromMs, toMs) {
  const start = Number(fromMs);
  const end = Number(toMs);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return [];
  const dayMs = 24 * 60 * 60 * 1000;
  const firstDay = Math.floor(start / dayMs) * dayMs;
  const lastDay = Math.floor(end / dayMs) * dayMs;
  const sessions = [];
  for (let day = firstDay; day <= lastDay; day += dayMs) {
    const date = new Date(day).toISOString().slice(0, 10);
    sessions.push({
      id: `cont-${date}`,
      display_name: `Continuous ${date}`,
      type: 'legacy',
      source: 'continuous',
      start_time: day,
      end_time: day + dayMs - 1,
      athlete_names: {},
    });
  }
  return sessions;
}

function buildApiTelemetryCsv(unitId, rows, session, fromMs, toMs) {
  const headers = [
    'timestamp',
    'lat',
    'lon',
    'sog_mps',
    'cog',
    'roll_deg',
    'pitch_deg',
    'heading_deg',
    'rudder_angle',
    'boom_angle',
    'torso_angle',
    'unit_id',
    'custom_name',
    'gnss_iso',
  ];
  const csvRows = [];
  const points = [];
  for (const row of rows || []) {
    const tsMs = normalizeApiEpochMs(apiRowValue(row, ['timestamp', 'gnss_ms', 'time']));
    const lat = apiRowNumber(row, ['lat', 'latitude']);
    const lon = apiRowNumber(row, ['lon', 'lng', 'long', 'longitude']);
    const hasGps = lat != null && lon != null && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
    const cog = apiRowNumber(row, ['cog', 'COG', 'cog_deg', 'course', 'course_over_ground']);
    const yaw = apiRowNumber(row, ['yaw', 'yaw_deg', 'heading', 'heading_deg', 'hdg']);
    const heading = (yaw != null && (Math.abs(yaw) > 1e-9 || cog == null)) ? yaw : cog;
    if (tsMs != null && hasGps) {
      const pt = { ts: tsMs / 1000, lat, lon };
      const sog = apiRowNumber(row, ['sog', 'sog_mps', 'speed']);
      if (Number.isFinite(sog)) pt.sog = sog * 1.94384;
      if (Number.isFinite(cog)) pt.cog = cog;
      if (heading != null) pt.hdg = heading;
      points.push(pt);
    }
    csvRows.push({
      timestamp: tsMs != null ? tsMs : '',
      lat: apiRowValue(row, ['lat', 'latitude']),
      lon: apiRowValue(row, ['lon', 'lng', 'long', 'longitude']),
      sog_mps: hasGps ? apiRowValue(row, ['sog', 'sog_mps', 'speed']) : '',
      cog: apiRowValue(row, ['cog', 'COG', 'cog_deg', 'course', 'course_over_ground']),
      roll_deg: apiRowValue(row, ['roll', 'roll_deg']),
      pitch_deg: apiRowValue(row, ['pitch', 'pitch_deg']),
      heading_deg: heading ?? '',
      rudder_angle: apiRowValue(row, ['rudder_angle', 'rudder_angle_deg', 'rudder']),
      boom_angle: apiRowValue(row, ['boom_angle', 'boom_angle_deg', 'boom']),
      torso_angle: apiRowValue(row, ['torso_angle', 'torso_angle_deg', 'trunk_angle']),
      unit_id: apiRowValue(row, ['unit_id', 'unit']) ?? unitId,
      custom_name: apiRowValue(row, ['custom_name']) ?? session?.athlete_names?.[unitId] ?? '',
      gnss_iso: apiRowValue(row, ['gnss_iso']) || '',
    });
  }
  const customName = String(csvRows.find(row => row.custom_name)?.custom_name || session?.athlete_names?.[unitId] || '').trim();
  const startLabel = new Date(fromMs).toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
  const filename = `TrollSports_${apiCsvFileSafe(customName || unitId, 'unit')}_${apiCsvFileSafe(session?.id, 'session')}_${apiCsvFileSafe(unitId, 'unit')}_${startLabel}.csv`;
  return {
    file: new File([makeCsv(headers, csvRows)], filename, { type: 'text/csv', lastModified: Date.now() }),
    filename,
    unitId,
    customName,
    sessionId: session?.id || '',
    fromMs,
    toMs,
    points,
  };
}

function epochSecondsToIso(ts) {
  const n = Number(ts);
  if (!Number.isFinite(n)) return '';
  return new Date(n * 1000).toISOString();
}

function makeMetricTimelineRows(primaryRows = [], secondaryRows = [], extraSeries = []) {
  const byTime = new Map();
  const add = (rows, key) => {
    for (const point of (Array.isArray(rows) ? rows : [])) {
      const t = Number(point?.t);
      const v = Number(point?.v);
      if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
      const timeKey = t.toFixed(6);
      const row = byTime.get(timeKey) || { t };
      row[key] = v;
      byTime.set(timeKey, row);
    }
  };
  add(primaryRows, 'rudder');
  add(secondaryRows, 'boom');
  for (const series of (Array.isArray(extraSeries) ? extraSeries : [])) {
    if (!series) continue;
    add(series.rows, series.key);
  }
  return [...byTime.values()].sort((a, b) => Number(a.t) - Number(b.t));
}

function canvasToBlob(canvas, type = 'image/png', quality = 0.92) {
  if (!(canvas instanceof HTMLCanvasElement)) return Promise.resolve(null);
  return new Promise(resolve => {
    canvas.toBlob(blob => resolve(blob), type, quality);
  });
}

function isLikelyStorageError(err) {
  const msg = String(err?.message || err || '').toLowerCase();
  return (
    msg.includes('storage') ||
    msg.includes('quota') ||
    msg.includes('indexeddb') ||
    msg.includes('opfs')
  );
}

async function logStorageDebug(context = '') {
  if (typeof Storage.getDebugInfo !== 'function') return;
  try {
    const info = await Storage.getDebugInfo();
    const usageMb = Number.isFinite(info?.usage) ? (info.usage / (1024 * 1024)).toFixed(1) : '?';
    const quotaMb = Number.isFinite(info?.quota) ? (info.quota / (1024 * 1024)).toFixed(1) : '?';
    console.warn(`[StorageDebug] ${context} backend=${info?.backend} usage=${usageMb}MB quota=${quotaMb}MB forceIdb=${!!info?.forceIdbFallback}`);
  } catch {}
}

function normalizeAthleteName(name) {
  return String(name || '').trim();
}

function normalizeFilenameKey(name) {
  return String(name || '').trim().toLowerCase();
}

function csvFilenamePrefix(name) {
  const stem = normalizeFilenameKey(name).replace(/\.[^.]+$/, '');
  return stem.slice(0, CSV_MEMORY_PREFIX_LEN);
}

function loadAthleteLibrary() {
  const list = loadJsonLocal(ATHLETE_LIBRARY_KEY, []);
  if (!Array.isArray(list)) return [];
  const out = [];
  const seen = new Set();
  for (const raw of list) {
    const name = normalizeAthleteName(raw?.name);
    if (!name) continue;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      name,
      color: typeof raw?.color === 'string' ? raw.color : null,
      weight: Number.isFinite(Number(raw?.weight)) ? Number(raw.weight) : null,
      height: Number.isFinite(Number(raw?.height)) ? Number(raw.height) : null,
      updated_at: Number.isFinite(Number(raw?.updated_at)) ? Number(raw.updated_at) : Date.now(),
    });
  }
  return out;
}

function saveAthleteLibrary(list) {
  saveJsonLocal(ATHLETE_LIBRARY_KEY, list);
}

function mergeAthleteIntoLibrary(library, athlete) {
  const name = normalizeAthleteName(athlete?.name);
  if (!name) return false;
  const key = name.toLowerCase();
  const idx = library.findIndex(a => String(a.name || '').trim().toLowerCase() === key);
  const entry = {
    name,
    color: athlete?.color || null,
    weight: Number.isFinite(Number(athlete?.weight)) ? Number(athlete.weight) : null,
    height: Number.isFinite(Number(athlete?.height)) ? Number(athlete.height) : null,
    updated_at: Date.now(),
  };
  if (idx >= 0) library[idx] = { ...library[idx], ...entry };
  else library.push(entry);
  library.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
  return true;
}

function updateAthleteLibraryFromCurrentState() {
  const library = loadAthleteLibrary();
  let changed = false;
  for (const athlete of state.athletes) {
    changed = mergeAthleteIntoLibrary(library, athlete) || changed;
  }
  if (changed) saveAthleteLibrary(library.slice(0, 128));
}

function loadCsvAthleteMemory() {
  const raw = loadJsonLocal(CSV_ATHLETE_MEMORY_KEY, { exact: {}, prefix: {} });
  const exact = raw && typeof raw.exact === 'object' ? raw.exact : {};
  const prefix = raw && typeof raw.prefix === 'object' ? raw.prefix : {};
  return { exact, prefix };
}

function saveCsvAthleteMemory(memory) {
  saveJsonLocal(CSV_ATHLETE_MEMORY_KEY, memory);
}

function rememberCsvAthlete(csvFilename, athlete) {
  const name = normalizeAthleteName(athlete?.name);
  const fileKey = normalizeFilenameKey(csvFilename);
  const pref = csvFilenamePrefix(csvFilename);
  if (!name || !fileKey || !pref) return;
  const profile = {
    name,
    color: athlete?.color || null,
    weight: Number.isFinite(Number(athlete?.weight)) ? Number(athlete.weight) : null,
    height: Number.isFinite(Number(athlete?.height)) ? Number(athlete.height) : null,
    updated_at: Date.now(),
  };
  const memory = loadCsvAthleteMemory();
  memory.exact[fileKey] = profile;
  memory.prefix[pref] = profile;
  saveCsvAthleteMemory(memory);
}

function guessCsvAthleteProfile(csvFilename) {
  const fileKey = normalizeFilenameKey(csvFilename);
  const pref = csvFilenamePrefix(csvFilename);
  if (!fileKey || !pref) return null;
  const memory = loadCsvAthleteMemory();
  const remembered = memory.exact[fileKey] || memory.prefix[pref];
  if (remembered) return remembered;
  const stem = fileKey.replace(/\.[^.]+$/, '');
  const candidates = [
    ...(state.athletes || []),
    ...loadAthleteLibrary(),
  ];
  for (const athlete of candidates) {
    const name = normalizeAthleteName(athlete?.name);
    if (!name) continue;
    const key = name.toLowerCase();
    if (stem.includes(key) || key.split(/\s+/).filter(Boolean).every(part => stem.includes(part))) {
      return {
        name,
        color: athlete?.color || null,
        weight: Number.isFinite(Number(athlete?.weight)) ? Number(athlete.weight) : null,
        height: Number.isFinite(Number(athlete?.height)) ? Number(athlete.height) : null,
      };
    }
  }
  return null;
}

function findAthleteByName(name) {
  const key = normalizeAthleteName(name).toLowerCase();
  if (!key) return null;
  return state.athletes.find(a => normalizeAthleteName(a?.name).toLowerCase() === key) || null;
}

function parseOptionalPositiveNumber(value) {
  const num = parseFloat(value);
  return Number.isFinite(num) && num > 0 ? num : null;
}

function nextPaletteColor(seed = 0) {
  return PALETTE[Math.abs(seed) % PALETTE.length];
}

// ── Utils ──────────────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }
function fmt(s) { if(s == null || isNaN(s)) return '--:--'; const m=Math.floor(s/60),ss=Math.floor(s%60); return `${m}:${ss.toString().padStart(2,'0')}`; }
function fmtClock(epochSec) {
  if(epochSec == null) return '--:--:--';
  const d = new Date(epochSec * 1000);
  return d.toISOString().substring(11,19);
}

function normalizeEpochSecondsUI(ts) {
  const n = Number(ts);
  if (!Number.isFinite(n)) return null;
  const a = Math.abs(n);
  if (a > 1e14) return n / 1e6; // microseconds
  if (a > 1e11) return n / 1e3; // milliseconds
  return n;
}

function sanitizeVideoRangeUI(v) {
  let start = normalizeEpochSecondsUI(v?.ts_start);
  let end = normalizeEpochSecondsUI(v?.ts_end);
  const durN = Number(v?.duration_sec);
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

function normalizeMapVideoRanges(videos) {
  for (const v of (videos || [])) {
    const beforeStart = v.ts_start;
    const beforeEnd = v.ts_end;
    const { start, end } = sanitizeVideoRangeUI(v);
    v.ts_start = start;
    v.ts_end = end;
    if (Array.isArray(v.points)) {
      for (const p of v.points) {
        const nts = normalizeEpochSecondsUI(p.ts);
        if (nts != null) p.ts = nts;
      }
    }
    if (Array.isArray(v.telemetry_points)) {
      for (const p of v.telemetry_points) {
        const nts = normalizeEpochSecondsUI(p.ts);
        if (nts != null) p.ts = nts;
      }
    }
    if ((beforeStart !== v.ts_start || beforeEnd !== v.ts_end) && (v.ts_start != null || v.ts_end != null)) {
      console.warn(`[Timeline] normalized video range for ${v.filename || v.id}: ${beforeStart}..${beforeEnd} -> ${v.ts_start}..${v.ts_end}`);
    }
  }
}
// Video URL cache: fileId -> blob URL
const _videoURLCache = {};
const _videoURLPending = {};
const _videoURLLastUse = {};
const _videoURLFailLog = {}; // fileId -> last warn timestamp (throttle spam)
let _timelineMediaOverlayUrl = null;
let _timelineMediaOverlayFileId = null;

function touchVideoURL(fileId) {
  _videoURLLastUse[fileId] = Date.now();
}

function revokeVideoURL(fileId) {
  const url = _videoURLCache[fileId];
  if (url) {
    try { URL.revokeObjectURL(url); } catch {}
  }
  delete _videoURLCache[fileId];
  delete _videoURLLastUse[fileId];
  delete _videoURLPending[fileId];
}

function collectPinnedVideoIds(absTs = state.tl?.currentTs) {
  const pinned = new Set();
  for (const slot of (state.tl?.athleteSlots || [])) {
    if (slot.currentFileId) pinned.add(slot.currentFileId);
    if (slot._desiredFileId) pinned.add(slot._desiredFileId);
    for (const videoEl of getSlotVideoEls(slot)) {
      if (videoEl?._loadedFileId) pinned.add(videoEl._loadedFileId);
    }
    if (Number.isFinite(absTs)) {
      const current = findSlotVideoAtTime(slot, absTs);
      if (current?.id) pinned.add(current.id);
    }
  }
  const phonePlayback = state.phonePlayback || {};
  if (phonePlayback.currentFileId) pinned.add(phonePlayback.currentFileId);
  if (phonePlayback.desiredFileId) pinned.add(phonePlayback.desiredFileId);
  if (isPhonePlaybackEnabled()) {
    const phoneItem = getPhonePlaybackItemAtTs(absTs, phonePlayback.currentFileId);
    if (phoneItem?.id) pinned.add(phoneItem.id);
    const nextPhoneItem = getNextPhonePlaybackItemAfterTs(absTs);
    if (nextPhoneItem?.id && Number.isFinite(absTs)) {
      const nextStart = getTimelineMediaStartTs(nextPhoneItem);
      if (Number.isFinite(nextStart) && nextStart - absTs <= 20) pinned.add(nextPhoneItem.id);
    }
  }
  return pinned;
}

function pruneVideoURLCache() {
  const maxEntries = getVideoURLCacheLimit();
  const ids = Object.keys(_videoURLCache);
  if (ids.length <= maxEntries) return;
  const pinned = collectPinnedVideoIds(state.tl?.currentTs);
  const victims = ids
    .filter(id => !pinned.has(id))
    .sort((a, b) => (_videoURLLastUse[a] || 0) - (_videoURLLastUse[b] || 0));
  for (const id of victims) {
    if (Object.keys(_videoURLCache).length <= maxEntries) break;
    revokeVideoURL(id);
  }
}

async function getVideoURL(fileId) {
  if (_videoURLCache[fileId]) {
    touchVideoURL(fileId);
    return _videoURLCache[fileId];
  }
  if (_videoURLPending[fileId]) return _videoURLPending[fileId];

  _videoURLPending[fileId] = (async () => {
    try {
      const url = await FM.createVideoURL(fileId);
      if (url) {
        _videoURLCache[fileId] = url;
        touchVideoURL(fileId);
        delete _videoURLFailLog[fileId];
        pruneVideoURLCache();
        return url;
      }
      // Throttle the warning: log at most once per 5 seconds per fileId
      const now = Date.now();
      if (!_videoURLFailLog[fileId] || now - _videoURLFailLog[fileId] > 5000) {
        console.warn(`getVideoURL: no File for ${fileId} - file handle lost or not picked`);
        _videoURLFailLog[fileId] = now;
      }
    } catch (e) {
      console.error(`getVideoURL error for ${fileId}:`, e);
    } finally {
      delete _videoURLPending[fileId];
    }
    return null;
  })();
  return _videoURLPending[fileId];
}

function queueSlotVideoSwitch(task) {
  if (!USE_IPAD_VIDEO_WORKAROUNDS) return task();
  const run = () => Promise.resolve().then(task);
  _ipadVideoSwitchQueue = _ipadVideoSwitchQueue.then(run, run);
  return _ipadVideoSwitchQueue;
}

function getSlotVideoEls(slot) {
  if (!slot) return [];
  if (Array.isArray(slot._videoEls)) return slot._videoEls;
  return slot.videoEl ? [slot.videoEl] : [];
}

function getSlotActiveVideoEl(slot) {
  const videos = getSlotVideoEls(slot);
  if (!videos.length) return null;
  if (slot?.videoEl && videos.includes(slot.videoEl)) return slot.videoEl;
  const idx = Number.isFinite(slot?._activeVideoIdx) ? Number(slot._activeVideoIdx) : 0;
  return videos[Math.max(0, Math.min(videos.length - 1, idx))] || videos[0] || null;
}

function getSlotStandbyVideoEl(slot) {
  const videos = getSlotVideoEls(slot);
  const active = getSlotActiveVideoEl(slot);
  return videos.find(videoEl => videoEl !== active) || null;
}

function syncSlotActiveVideoRef(slot) {
  if (!slot) return null;
  slot.videoEl = getSlotActiveVideoEl(slot);
  return slot.videoEl;
}

function touchSlotVideoEl(videoEl) {
  if (!videoEl) return null;
  videoEl._lastUseAt = Date.now();
  return videoEl;
}

function getSlotVideoElForFile(slot, fileId) {
  return getSlotVideoEls(slot).find(videoEl => videoEl?._clipFileId === fileId) || null;
}

function insertSlotVideoEl(slot, videoEl) {
  const host = slot?._videoStageEl || slot?.paneEl || null;
  if (!host || !videoEl || videoEl.parentNode === host) return;
  const anchor = slot._overlayCanvas || slot._emptyEl || null;
  host.insertBefore(videoEl, anchor);
}

function getPinnedSlotVideoFileIds(slot, absTs = state.tl?.currentTs) {
  const pinned = new Set();
  if (!slot) return pinned;
  if (slot.currentFileId) pinned.add(slot.currentFileId);
  if (slot._desiredFileId) pinned.add(slot._desiredFileId);
  if (Number.isFinite(absTs)) {
    const current = findSlotVideoAtTime(slot, absTs);
    if (current?.id) pinned.add(current.id);
  }
  return pinned;
}

function pruneSlotVideoEls(slot, absTs = state.tl?.currentTs) {
  if (!USE_IPAD_VIDEO_WORKAROUNDS) return;
  const videos = getSlotVideoEls(slot);
  if (videos.length <= IPAD_SLOT_VIDEO_CACHE_LIMIT) return;
  const pinned = getPinnedSlotVideoFileIds(slot, absTs);
  const active = getSlotActiveVideoEl(slot);
  const victims = videos
    .filter(videoEl => videoEl && videoEl !== active && !pinned.has(videoEl._clipFileId))
    .sort((a, b) => (a._lastUseAt || 0) - (b._lastUseAt || 0));
  for (const videoEl of victims) {
    if (getSlotVideoEls(slot).length <= IPAD_SLOT_VIDEO_CACHE_LIMIT) break;
    pauseVideoElement(videoEl, true);
    if (videoEl.parentNode) videoEl.parentNode.removeChild(videoEl);
    slot._videoEls = getSlotVideoEls(slot).filter(v => v !== videoEl);
  }
}

function setVideoElementVisible(videoEl, visible) {
  if (!videoEl) return;
  if (USE_IPAD_VIDEO_WORKAROUNDS) {
    videoEl.style.display = 'block';
    videoEl.style.opacity = visible ? '1' : '0';
    videoEl.style.visibility = visible ? 'visible' : 'hidden';
    videoEl.style.pointerEvents = visible ? '' : 'none';
    videoEl.style.zIndex = visible ? '2' : '1';
    return;
  }
  videoEl.style.display = visible ? 'block' : 'none';
}

function showSlotActiveVideo(slot) {
  if (!slot?.paneEl) return;
  if (isVideoSlotHidden(slot)) {
    slot.paneEl.style.display = 'none';
    return;
  }
  const active = syncSlotActiveVideoRef(slot);
  slot.paneEl.dataset.hasVideoAtPlayhead = '1';
  slot.paneEl.style.display = 'flex';
  if (slot._emptyEl) slot._emptyEl.style.display = 'none';
  for (const videoEl of getSlotVideoEls(slot)) {
    setVideoElementVisible(videoEl, videoEl === active);
  }
}

function showSlotLoadingState(slot, message = 'Loading video...') {
  if (!slot?.paneEl) return;
  if (isVideoSlotHidden(slot)) {
    slot.paneEl.style.display = 'none';
    return;
  }
  slot.paneEl.dataset.hasVideoAtPlayhead = '1';
  slot.paneEl.style.display = 'flex';
  if (slot._emptyEl) {
    slot._emptyEl.textContent = message;
    slot._emptyEl.style.display = 'flex';
  }
  for (const videoEl of getSlotVideoEls(slot)) {
    setVideoElementVisible(videoEl, false);
  }
}

function showSlotEmptyState(slot, message = 'No video at this time') {
  if (!slot?.paneEl) return;
  slot.paneEl.dataset.hasVideoAtPlayhead = '0';
  slot.paneEl.style.display = (!isVideoSlotHidden(slot) && slot.videos?.length) ? 'flex' : 'none';
  if (slot._emptyEl) {
    slot._emptyEl.textContent = message;
    slot._emptyEl.style.display = slot.videos?.length ? 'flex' : 'none';
  }
  for (const videoEl of getSlotVideoEls(slot)) {
    setVideoElementVisible(videoEl, false);
  }
}

function pauseVideoElement(videoEl, clearSrc = false) {
  if (!videoEl) return;
  try { videoEl.pause(); } catch {}
  if (!clearSrc) return;
  try { videoEl.removeAttribute('src'); } catch {}
  try { videoEl.load(); } catch {}
  delete videoEl._loadedFileId;
  delete videoEl._loadedURL;
}

function pauseSlotVideos(slot, clearSrc = false) {
  for (const videoEl of getSlotVideoEls(slot)) {
    if (clearSrc && USE_IPAD_VIDEO_WORKAROUNDS) videoEl.preload = 'metadata';
    pauseVideoElement(videoEl, clearSrc);
  }
}

function clearSlotVideoSources(slot) {
  pauseSlotVideos(slot, true);
  slot.currentFileId = null;
  slot._desiredFileId = null;
}

function setSlotPlaybackRate(slot, rate) {
  for (const videoEl of getSlotVideoEls(slot)) {
    try { videoEl.playbackRate = rate; } catch {}
  }
}

function clampVideoTime(videoEl, videoSec) {
  let target = Math.max(0, Number(videoSec) || 0);
  if (videoEl && Number.isFinite(videoEl.duration) && videoEl.duration > 0) {
    target = Math.min(target, Math.max(0, videoEl.duration - 0.05));
  }
  return target;
}

function assignVideoTime(videoEl, target) {
  if (!videoEl) return false;
  const current = Number.isFinite(videoEl.currentTime) ? videoEl.currentTime : NaN;
  try {
    if (typeof videoEl.fastSeek === 'function' && Number.isFinite(current) && Math.abs(current - target) > 1.25) {
      videoEl.fastSeek(target);
    } else {
      videoEl.currentTime = target;
    }
    return true;
  } catch {
    return false;
  }
}

function waitForVideoReadyState(videoEl, minReadyState = 1, timeoutMs = 4000) {
  if (!videoEl) return Promise.resolve(false);
  if (videoEl.readyState >= minReadyState) return Promise.resolve(true);
  return new Promise(resolve => {
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      cleanup();
      resolve(ok);
    };
    const onState = () => {
      if (videoEl.readyState >= minReadyState) finish(true);
    };
    const onError = () => finish(false);
    const timer = setTimeout(() => finish(videoEl.readyState >= minReadyState), timeoutMs);
    const cleanup = () => {
      clearTimeout(timer);
      videoEl.removeEventListener('loadedmetadata', onState);
      videoEl.removeEventListener('loadeddata', onState);
      videoEl.removeEventListener('canplay', onState);
      videoEl.removeEventListener('seeked', onState);
      videoEl.removeEventListener('timeupdate', onState);
      videoEl.removeEventListener('error', onError);
      videoEl.removeEventListener('abort', onError);
      videoEl.removeEventListener('emptied', onError);
    };
    videoEl.addEventListener('loadedmetadata', onState);
    videoEl.addEventListener('loadeddata', onState);
    videoEl.addEventListener('canplay', onState);
    videoEl.addEventListener('seeked', onState);
    videoEl.addEventListener('timeupdate', onState);
    videoEl.addEventListener('error', onError);
    videoEl.addEventListener('abort', onError);
    videoEl.addEventListener('emptied', onError);
  });
}

function seekVideoElement(videoEl, videoSec) {
  if (!videoEl || !Number.isFinite(videoSec)) return;
  const target = clampVideoTime(videoEl, videoSec);
  const doSeek = () => {
    assignVideoTime(videoEl, target);
  };
  if (videoEl.readyState >= 1) doSeek();
  else videoEl.addEventListener('loadedmetadata', doSeek, { once: true });
}

async function seekVideoElementAccurate(videoEl, videoSec, timeoutMs = 2400) {
  if (!videoEl || !Number.isFinite(videoSec)) return false;
  const ready = await waitForVideoReadyState(videoEl, 1, timeoutMs);
  if (!ready && videoEl.readyState < 1) return false;
  const target = clampVideoTime(videoEl, videoSec);
  if (Math.abs((videoEl.currentTime || 0) - target) <= 0.05) return true;
  return new Promise(resolve => {
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      cleanup();
      resolve(ok);
    };
    const onState = () => {
      if (Math.abs((videoEl.currentTime || 0) - target) <= 0.12) finish(true);
    };
    const onError = () => finish(false);
    const timer = setTimeout(() => {
      finish(Math.abs((videoEl.currentTime || 0) - target) <= 0.2);
    }, timeoutMs);
    const cleanup = () => {
      clearTimeout(timer);
      videoEl.removeEventListener('seeked', onState);
      videoEl.removeEventListener('loadeddata', onState);
      videoEl.removeEventListener('timeupdate', onState);
      videoEl.removeEventListener('error', onError);
    };
    videoEl.addEventListener('seeked', onState);
    videoEl.addEventListener('loadeddata', onState);
    videoEl.addEventListener('timeupdate', onState);
    videoEl.addEventListener('error', onError);
    if (!assignVideoTime(videoEl, target)) {
      finish(false);
    }
  });
}

function tryPlaySlotVideo(slot, expectedFileId = slot?.currentFileId) {
  const videoEl = syncSlotActiveVideoRef(slot);
  if (!videoEl || !expectedFileId) return;
  if (slot.currentFileId !== expectedFileId) return;

  setSlotPlaybackRate(slot, state.tl.playbackRate);
  const p = videoEl.play();
  if (p?.catch) p.catch(() => {});

  if (!USE_IPAD_VIDEO_WORKAROUNDS) return;
  if (!videoEl.paused && videoEl.readyState >= 2) return;

  if (slot._playRetryTimer) {
    clearTimeout(slot._playRetryTimer);
    slot._playRetryTimer = null;
  }
  const token = (slot._playRetryToken || 0) + 1;
  slot._playRetryToken = token;
  const retry = () => {
    if (slot._playRetryToken !== token) return;
    if (!state.tl.playing) return;
    if (slot.currentFileId !== expectedFileId) return;
    setSlotPlaybackRate(slot, state.tl.playbackRate);
    const p2 = videoEl.play();
    if (p2?.catch) p2.catch(() => {});
  };
  videoEl.addEventListener('loadeddata', retry, { once: true });
  videoEl.addEventListener('canplay', retry, { once: true });
  slot._playRetryTimer = setTimeout(retry, 140);
}

function findLoadedSlotVideoElement(slot, fileId) {
  return getSlotVideoEls(slot).find(videoEl => videoEl?._clipFileId === fileId && videoEl?._loadedFileId === fileId && (videoEl.currentSrc || videoEl.src)) || null;
}

function findReadySlotVideoElement(slot, fileId, minReadyState = 1) {
  const videoEl = findLoadedSlotVideoElement(slot, fileId);
  return videoEl && videoEl.readyState >= minReadyState ? videoEl : null;
}

async function prepareVideoElementForPlayback(slot, videoEl, fileId, videoSec, { forceReload = false, background = false, guard = null } = {}) {
  if (!videoEl || !fileId) return { ok: false, missing: false };
  touchSlotVideoEl(videoEl);
  const url = await getVideoURL(fileId);
  if (!url) return { ok: false, missing: true };

  const sameSource = !forceReload
    && videoEl._loadedFileId === fileId
    && videoEl._loadedURL === url
    && (videoEl.currentSrc || videoEl.src);

  if (guard && !guard()) return { ok: false, missing: false, cancelled: true };

  await queueSlotVideoSwitch(async () => {
    const stillWanted = slot._desiredFileId == null || slot._desiredFileId === fileId || background;
    if (!stillWanted || (guard && !guard())) return;
    videoEl.preload = background ? 'metadata' : 'auto';
    if (!sameSource) {
      pauseVideoElement(videoEl, forceReload);
      videoEl.src = url;
      videoEl._loadedFileId = fileId;
      videoEl._loadedURL = url;
      try { videoEl.load(); } catch {}
    } else if (videoEl.readyState === 0) {
      try { videoEl.load(); } catch {}
    }
    if (USE_IPAD_VIDEO_WORKAROUNDS) {
      await new Promise(resolve => setTimeout(resolve, IPAD_VIDEO_SWITCH_STAGGER_MS));
    }
  });

  if (guard && !guard()) return { ok: false, missing: false, cancelled: true };

  let ready = await waitForVideoReadyState(videoEl, background ? 1 : 2, background ? 1800 : (USE_IPAD_VIDEO_WORKAROUNDS ? 3600 : 2800));
  if (!ready) {
    if (guard && !guard()) return { ok: false, missing: false, cancelled: true };
    try { videoEl.load(); } catch {}
    ready = await waitForVideoReadyState(videoEl, background ? 1 : 2, USE_IPAD_VIDEO_WORKAROUNDS ? 3200 : 2200);
  }

  if (Number.isFinite(videoSec)) {
    if (guard && !guard()) return { ok: false, missing: false, cancelled: true };
    await seekVideoElementAccurate(videoEl, videoSec, background ? 1200 : 2400);
    if (!background) {
      ready = await waitForVideoReadyState(videoEl, 2, USE_IPAD_VIDEO_WORKAROUNDS ? 2000 : 1200) || ready;
    }
  }

  const minReady = background ? 1 : 2;
  return { ok: ready || videoEl.readyState >= minReady, missing: false, url };
}

function swapSlotActiveVideo(slot, targetVideoEl, fileId) {
  const videos = getSlotVideoEls(slot);
  const idx = Math.max(0, videos.indexOf(targetVideoEl));
  slot._activeVideoIdx = idx;
  slot.currentFileId = fileId;
  slot._desiredFileId = null;
  slot.videoEl = touchSlotVideoEl(videos[idx] || targetVideoEl || null);
  showSlotActiveVideo(slot);
  for (const videoEl of videos) {
    if (videoEl !== slot.videoEl) {
      try { videoEl.pause(); } catch {}
    }
  }
  pruneSlotVideoEls(slot, state.tl.currentTs);
}

function collectUpcomingSlotVideos(slot, absTs = state.tl?.currentTs) {
  return [];
}

async function preloadUpcomingSlotVideo(slot, absTs = state.tl?.currentTs) {
  return;
}

async function switchSlotVideoSourceDefault(slot, fileId, videoSec, autoPlayAfterLoad, forceReload, requestId) {
  const videoEl = syncSlotActiveVideoRef(slot);
  const prepared = await prepareVideoElementForPlayback(slot, videoEl, fileId, videoSec, { forceReload });
  if (slot._switchReqId !== requestId || slot._desiredFileId !== fileId) return;
  if (prepared.cancelled) return;

  if (!prepared.ok) {
    console.warn(`[Video] no URL for ${fileId} in slot "${slot.name}"`);
    slot.currentFileId = null;
    slot._desiredFileId = null;
    showSlotEmptyState(slot, prepared.missing ? 'Video file not accessible - re-add file' : 'Video failed to load');
    return;
  }

  slot.currentFileId = fileId;
  slot._desiredFileId = null;
  showSlotActiveVideo(slot);
  if (autoPlayAfterLoad || state.tl.playing) tryPlaySlotVideo(slot, fileId);
  else try { videoEl.pause(); } catch {}
}

async function switchSlotVideoSourceIpad(slot, fileId, videoSec, autoPlayAfterLoad, forceReload, requestId) {
  let targetVideoEl = !forceReload ? findLoadedSlotVideoElement(slot, fileId) : null;
  const instantVideoEl = !forceReload ? findReadySlotVideoElement(slot, fileId, 1) : null;
  if (slot._switchReqId !== requestId || slot._desiredFileId !== fileId) return;
  if (instantVideoEl) {
    swapSlotActiveVideo(slot, instantVideoEl, fileId);
    if (Number.isFinite(videoSec)) seekVideoElement(instantVideoEl, videoSec);
    if (autoPlayAfterLoad || state.tl.playing) tryPlaySlotVideo(slot, fileId);
    else try { slot.videoEl.pause(); } catch {}
    return;
  }
  if (!targetVideoEl) targetVideoEl = touchSlotVideoEl(ensureSlotVideoEl(slot, fileId) || getSlotActiveVideoEl(slot));

  const prepared = await prepareVideoElementForPlayback(slot, targetVideoEl, fileId, videoSec, { forceReload });
  if (slot._switchReqId !== requestId || slot._desiredFileId !== fileId) return;
  if (prepared.cancelled) return;

  if (!prepared.ok) {
    console.warn(`[Video] no URL for ${fileId} in slot "${slot.name}"`);
    slot.currentFileId = null;
    slot._desiredFileId = null;
    showSlotEmptyState(slot, prepared.missing ? 'Video file not accessible - re-add file' : 'Video failed to load');
    return;
  }

  swapSlotActiveVideo(slot, targetVideoEl, fileId);
  if (autoPlayAfterLoad || state.tl.playing) tryPlaySlotVideo(slot, fileId);
  else try { slot.videoEl.pause(); } catch {}
}

function switchSlotVideoSource(slot, fileId, videoSec, autoPlayAfterLoad = false, forceReload = false) {
  if (!slot || !fileId) return Promise.resolve();
  if (!USE_IPAD_VIDEO_WORKAROUNDS && !slot?.videoEl) return Promise.resolve();
  const requestId = (slot._switchReqId || 0) + 1;
  slot._switchReqId = requestId;
  slot._desiredFileId = fileId;

  const run = USE_IPAD_VIDEO_WORKAROUNDS
    ? switchSlotVideoSourceIpad(slot, fileId, videoSec, autoPlayAfterLoad, forceReload, requestId)
    : switchSlotVideoSourceDefault(slot, fileId, videoSec, autoPlayAfterLoad, forceReload, requestId);

  return Promise.resolve(run).catch(err => {
    if (slot._switchReqId === requestId && slot._desiredFileId === fileId) {
      slot._desiredFileId = null;
    }
    console.warn(`[Video] switch failed for slot "${slot?.name || '?'}":`, err);
  });
}

function prewarmLikelyVideosAtTs(absTs) {
  if (!Number.isFinite(absTs)) return;
  const ids = new Set();
  const heavyProject = isIpadHeavyVideoProject();
  for (const slot of state.tl.athleteSlots || []) {
    const active = findSlotVideoAtTime(slot, absTs);
    if (active?.id) ids.add(active.id);
    if (heavyProject) continue;

    let next = null;
    for (const v of (slot.videos || [])) {
      if (v?.id == null || v.ts_start == null || v.ts_start < absTs) continue;
      if (!next || v.ts_start < next.ts_start) next = v;
    }
    if (next?.id && next.ts_start - absTs <= 20) ids.add(next.id);
  }
  if (isPhonePlaybackEnabled()) {
    const phoneItem = getPhonePlaybackItemAtTs(absTs, state.phonePlayback?.currentFileId);
    if (phoneItem?.id) ids.add(phoneItem.id);
    const nextPhoneItem = getNextPhonePlaybackItemAfterTs(absTs);
    if (nextPhoneItem?.id) {
      const phoneStart = getTimelineMediaStartTs(nextPhoneItem);
      if (Number.isFinite(phoneStart) && phoneStart - absTs <= 20) ids.add(nextPhoneItem.id);
    }
  }
  const limit = heavyProject ? Math.min(2, ids.size) : ids.size;
  let warmed = 0;
  for (const id of ids) {
    if (warmed >= limit) break;
    getVideoURL(id).catch(() => {});
    warmed++;
  }
}

function requestSlotVideoRecover(slot, reason = '') {
  if (!USE_IPAD_VIDEO_WORKAROUNDS) return;
  if (!slot?.videoEl || !slot.currentFileId) return;
  const videoEl = syncSlotActiveVideoRef(slot);
  const now = Date.now();
  if (slot._lastRecoverAt && now - slot._lastRecoverAt < 900) return;
  slot._lastRecoverAt = now;

  const fileId = slot.currentFileId;
  const currentTs = state.tl.currentTs;
  const vid = findSlotVideoAtTime(slot, currentTs);
  const videoSec = (vid && vid.id === fileId)
    ? absTs2VideoSec(vid, currentTs)
    : (Number.isFinite(videoEl.currentTime) ? videoEl.currentTime : null);

  if (reason) {
    console.warn(`[Video] recovering slot "${slot.name}" after ${reason}`);
  }
  slot._stuckSince = 0;
  slot._desiredFileId = fileId;
  switchSlotVideoSource(slot, fileId, videoSec, state.tl.playing, true);
}

function monitorSlotVideoHealth(slot, nowMs = performance.now()) {
  if (!USE_IPAD_VIDEO_WORKAROUNDS) return;
  if (!slot?.videoEl || !slot.currentFileId) return;
  const videoEl = syncSlotActiveVideoRef(slot);
  const hasSource = !!videoEl.currentSrc;
  const ready = videoEl.readyState >= 2;
  if (hasSource && ready) {
    slot._stuckSince = 0;
    return;
  }
  if (!slot._stuckSince) {
    slot._stuckSince = nowMs;
    return;
  }
  if (nowMs - slot._stuckSince > 1700) {
    slot._stuckSince = nowMs;
    requestSlotVideoRecover(slot, hasSource ? 'not-ready' : 'no-source');
  }
}

function createSlotPlaybackVideoEl(slot) {
  const video = document.createElement('video');
  video.muted = true;
  video.defaultMuted = true;
  video.playsInline = true;
  video.setAttribute('playsinline', '');
  video.setAttribute('webkit-playsinline', '');
  video.disablePictureInPicture = true;
  video.setAttribute('x-webkit-airplay', 'deny');
  video.preload = USE_IPAD_VIDEO_WORKAROUNDS ? 'metadata' : 'auto';
  video.controls = false;
  if (USE_IPAD_VIDEO_WORKAROUNDS) {
    video.style.position = 'absolute';
    video.style.inset = '0';
    video.style.transform = 'translateZ(0)';
    video.style.backfaceVisibility = 'hidden';
  }
  setVideoElementVisible(video, false);
  const recoverIfActive = (reason) => {
    if (!slot?.currentFileId) return;
    if (video !== getSlotActiveVideoEl(slot)) return;
    if (video._loadedFileId !== slot.currentFileId) return;
    requestSlotVideoRecover(slot, reason);
  };
  video.addEventListener('stalled', () => recoverIfActive('stalled'));
  video.addEventListener('error', () => recoverIfActive('error'));
  return video;
}

function ensureSlotVideoEl(slot, fileId) {
  if (!slot || !fileId) return null;
  let video = getSlotVideoElForFile(slot, fileId);
  if (video) return touchSlotVideoEl(video);
  const standby = getSlotStandbyVideoEl(slot);
  if (standby) {
    standby._clipFileId = fileId;
    return touchSlotVideoEl(standby);
  }
  video = createSlotPlaybackVideoEl(slot);
  video._clipFileId = fileId;
  try { video.playbackRate = state.tl.playbackRate; } catch {}
  if (!Array.isArray(slot._videoEls)) slot._videoEls = [];
  slot._videoEls.push(video);
  insertSlotVideoEl(slot, video);
  pruneSlotVideoEls(slot, state.tl?.currentTs);
  return touchSlotVideoEl(video);
}

function clearSlotPlaybackRequest(slot) {
  slot._switchReqId = (slot._switchReqId || 0) + 1;
  slot._preloadReqId = (slot._preloadReqId || 0) + 1;
  slot._desiredFileId = null;
  slot._playRetryToken = (slot._playRetryToken || 0) + 1;
  if (slot._playRetryTimer) {
    clearTimeout(slot._playRetryTimer);
    slot._playRetryTimer = null;
  }
}

function stopSlotPlayback(slot, clearSources = !USE_IPAD_VIDEO_WORKAROUNDS) {
  clearSlotPlaybackRequest(slot);
  slot._lastPreloadFileId = null;
  slot._lastPreloadAt = 0;
  pauseSlotVideos(slot, clearSources);
  slot.currentFileId = null;
  slot._stuckSince = 0;
}

function ensurePhonePlaybackDom() {
  const phone = state.phonePlayback;
  const pane = el('phone-playback-pane');
  const stage = el('phone-playback-stage');
  const empty = el('phone-playback-empty');
  const label = el('phone-playback-label');
  if (!phone || !pane || !stage || !empty || !label) return null;
  phone.paneEl = pane;
  phone.stageEl = stage;
  phone.emptyEl = empty;
  phone.labelEl = label;
  if (!phone.videoEl) {
    const video = document.createElement('video');
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.setAttribute('playsinline', '');
    video.setAttribute('webkit-playsinline', '');
    video.disablePictureInPicture = true;
    video.setAttribute('x-webkit-airplay', 'deny');
    video.preload = USE_IPAD_VIDEO_WORKAROUNDS ? 'metadata' : 'auto';
    video.controls = false;
    if (USE_IPAD_VIDEO_WORKAROUNDS) {
      video.style.position = 'absolute';
      video.style.inset = '0';
      video.style.transform = 'translateZ(0)';
      video.style.backfaceVisibility = 'hidden';
    }
    setVideoElementVisible(video, false);
    stage.insertBefore(video, empty);
    phone.videoEl = video;
  }
  return phone;
}

function getPhonePlaybackRightMinWidth() {
  if (window.matchMedia('(max-width: 760px)').matches) return 220;
  const phoneDividerW = el('phone-playback-divider')?.offsetWidth || 5;
  const shellW = Math.max(180, Number(state.phonePlayback?.savedShellWidthPx) || 260);
  const videoMinW = 240;
  return shellW + phoneDividerW + videoMinW;
}

function setPhonePlaybackLayoutActive(active) {
  const layout = el('layout');
  if (!layout) return;
  const isActive = layout.classList.contains('phone-playback-active');
  if (active === isActive) return;

  if (active) {
    layout.classList.add('phone-playback-active');
    lockPhonePlaybackGridWidth();
  } else {
    layout.classList.remove('phone-playback-active');
    unlockPhonePlaybackGridWidth();
  }

  syncPhonePlaybackShellSize();
  syncPhonePlaybackMapOffset();
  scheduleAnalysisMapResize();
}

function syncPhonePlaybackShellSize() {
  const layout = el('layout');
  const panelRight = el('panel-right');
  const shell = el('phone-playback-shell');
  const divider = el('phone-playback-divider');
  if (!layout || !panelRight || !shell) return;
  const active = layout.classList.contains('phone-playback-active');
  const compact = window.matchMedia('(max-width: 760px)').matches;
  if (!active || compact) {
    shell.style.width = '';
    shell.style.flex = '';
    syncPhonePlaybackMapOffset();
    return;
  }

  const dividerW = divider?.offsetWidth || 0;
  const totalW = panelRight.clientWidth - dividerW;
  if (!(totalW > 0)) return;
  const minW = 180;
  const maxW = Math.max(minW, totalW - 240);
  const lockedGridW = Number(state.phonePlayback?.savedVideoGridWidthPx);
  let nextW = Number.NaN;
  if (Number.isFinite(lockedGridW)) {
    const gridMinW = 240;
    const clampedGridW = Math.max(gridMinW, Math.min(lockedGridW, totalW - minW));
    if (clampedGridW !== lockedGridW) setPhonePlaybackGridWidth(clampedGridW);
    nextW = Math.max(minW, Math.min(totalW - clampedGridW, maxW));
  } else {
    const fallbackW = Math.round(totalW * 0.32);
    nextW = Math.max(minW, Math.min(Number(state.phonePlayback?.savedShellWidthPx) || fallbackW, maxW));
  }
  state.phonePlayback.savedShellWidthPx = nextW;
  shell.style.width = `${nextW}px`;
  shell.style.flex = `0 0 ${nextW}px`;
  syncPhonePlaybackMapOffset();
}

function syncPhonePlaybackMapOffset() {
  const layout = el('layout');
  const left = el('panel-left');
  const shell = el('phone-playback-shell');
  const divider = el('phone-playback-divider');
  if (!layout) return;
  const active = layout.classList.contains('phone-playback-active');
  const compact = window.matchMedia('(max-width: 760px)').matches;
  if (!active || compact) {
    layout.style.setProperty('--phone-playback-width', '0px');
    if (left && left.style.flex === 'none' && Number.isFinite(Number(state.phonePlayback?.savedLeftWidthPx))) {
      left.style.width = `${state.phonePlayback.savedLeftWidthPx}px`;
    }
    state.phonePlayback.savedLeftWidthPx = null;
    return;
  }

  const shellW = Math.max(0, Number(state.phonePlayback?.savedShellWidthPx) || shell?.getBoundingClientRect().width || 0);
  const dividerW = divider?.offsetWidth || 0;
  const totalW = shellW + dividerW;
  layout.style.setProperty('--phone-playback-width', `${totalW}px`);

  if (left && left.style.flex === 'none') {
    if (!Number.isFinite(Number(state.phonePlayback?.savedLeftWidthPx))) {
      state.phonePlayback.savedLeftWidthPx = left.getBoundingClientRect().width;
    }
    const base = Number(state.phonePlayback.savedLeftWidthPx) || left.getBoundingClientRect().width;
    left.style.width = `max(${PHONE_PLAYBACK_LEFT_MIN_WIDTH}px, calc(${Math.round(base)}px - var(--phone-playback-width, 0px)))`;
  }
}

function lockPhonePlaybackGridWidth() {
  const grid = el('video-grid');
  if (!grid) return;
  const width = Math.round(grid.getBoundingClientRect().width || 0);
  if (!(width > 0)) return;
  setPhonePlaybackGridWidth(width);
}

function setPhonePlaybackGridWidth(width) {
  const grid = el('video-grid');
  if (!grid) return;
  const next = Math.max(240, Math.round(Number(width) || 0));
  state.phonePlayback.savedVideoGridWidthPx = next;
  grid.style.flex = `0 0 ${next}px`;
  grid.style.width = `${next}px`;
}

function unlockPhonePlaybackGridWidth() {
  const grid = el('video-grid');
  if (!grid) return;
  grid.style.flex = '';
  grid.style.width = '';
  state.phonePlayback.savedVideoGridWidthPx = null;
}

function setPhonePlaybackVisible(item, visible, message = '') {
  const phone = ensurePhonePlaybackDom();
  if (!phone) return;
  const title = item ? `Phone | ${getTimelineMediaDisplayName(item)}` : 'Phone Playback';
  phone.labelEl.textContent = title;
  phone.paneEl.style.display = visible ? 'flex' : 'none';
  phone.paneVisible = !!visible;
  setPhonePlaybackLayoutActive(!!visible);
  syncPhonePlaybackShellSize();
  if (phone.emptyEl) {
    phone.emptyEl.textContent = message || 'No phone video at this time';
    phone.emptyEl.style.display = visible && message ? 'flex' : 'none';
  }
  if (phone.videoEl) setVideoElementVisible(phone.videoEl, visible && !message);
}

function pausePhonePlayback(clearSrc = false, clearState = false) {
  const phone = state.phonePlayback;
  if (phone.videoEl) pauseVideoElement(phone.videoEl, clearSrc);
  phone.lastDriftCorrectAt = 0;
  if (clearState) {
    phone.currentFileId = null;
    phone.desiredFileId = null;
  }
}

function hidePhonePlayback(clearState = false, clearSrc = false) {
  pausePhonePlayback(clearSrc, clearState);
  setPhonePlaybackVisible(getPhonePlaybackItemAtTs(state.tl?.currentTs, state.phonePlayback?.currentFileId), false);
}

async function switchPhonePlaybackSource(item, videoSec, autoPlayAfterLoad = false, forceReload = false) {
  const phone = ensurePhonePlaybackDom();
  if (!phone || !item?.id || !phone.videoEl) return;
  const fileId = String(item.id);
  const requestId = (phone.switchReqId || 0) + 1;
  phone.switchReqId = requestId;
  phone.desiredFileId = fileId;

  setPhonePlaybackVisible(item, true, 'Loading phone video...');

  const url = await getVideoURL(fileId);
  if (phone.switchReqId !== requestId || phone.desiredFileId !== fileId) return;

  if (!url) {
    phone.currentFileId = null;
    phone.desiredFileId = null;
    setPhonePlaybackVisible(item, true, 'Phone video file not accessible - re-add file');
    return;
  }

  const videoEl = phone.videoEl;
  const sameSource = !forceReload
    && videoEl._loadedFileId === fileId
    && videoEl._loadedURL === url
    && (videoEl.currentSrc || videoEl.src);

  videoEl.preload = 'auto';
  if (!sameSource) {
    pauseVideoElement(videoEl, forceReload);
    videoEl.src = url;
    videoEl._loadedFileId = fileId;
    videoEl._loadedURL = url;
    try { videoEl.load(); } catch {}
  } else if (videoEl.readyState === 0) {
    try { videoEl.load(); } catch {}
  }

  let ready = await waitForVideoReadyState(videoEl, 2, USE_IPAD_VIDEO_WORKAROUNDS ? 3600 : 2800);
  if (!ready) {
    try { videoEl.load(); } catch {}
    ready = await waitForVideoReadyState(videoEl, 2, USE_IPAD_VIDEO_WORKAROUNDS ? 3200 : 2200);
  }
  if (Number.isFinite(videoSec)) {
    await seekVideoElementAccurate(videoEl, videoSec, 2400);
  }

  if (phone.switchReqId !== requestId || phone.desiredFileId !== fileId) return;

  if (!ready && videoEl.readyState < 1) {
    phone.currentFileId = null;
    phone.desiredFileId = null;
    setPhonePlaybackVisible(item, true, 'Phone video failed to load');
    return;
  }

  phone.currentFileId = fileId;
  phone.desiredFileId = null;
  phone.lastDriftCorrectAt = 0;
  setPhonePlaybackVisible(item, true);
  try { videoEl.playbackRate = state.tl.playbackRate; } catch {}
  if (autoPlayAfterLoad || state.tl.playing) {
    const playPromise = videoEl.play();
    if (playPromise?.catch) playPromise.catch(() => {});
  } else {
    try { videoEl.pause(); } catch {}
  }
}

function syncPhonePlaybackToTimeline({ forceSeek = false, forceReload = false } = {}) {
  if (!isPhonePlaybackEnabled()) {
    hidePhonePlayback(true, false);
    return;
  }
  const item = getPhonePlaybackItemAtTs(state.tl.currentTs, state.phonePlayback?.currentFileId);
  if (!item) {
    state.phonePlayback.lastDriftCorrectAt = 0;
    hidePhonePlayback(false, false);
    return;
  }

  const phone = ensurePhonePlaybackDom();
  if (!phone || !phone.videoEl) return;

  if (!videoContainsAbsTs(item, state.tl.currentTs)) {
    phone.lastDriftCorrectAt = 0;
    hidePhonePlayback(false, false);
    return;
  }

  const videoSec = absTs2VideoSec(item, state.tl.currentTs);
  if (phone.currentFileId !== String(item.id) || forceReload) {
    void switchPhonePlaybackSource(item, videoSec, state.tl.playing, forceReload);
    return;
  }

  setPhonePlaybackVisible(item, true);
  if (Number.isFinite(videoSec)) {
    const currentTime = Number.isFinite(phone.videoEl.currentTime) ? phone.videoEl.currentTime : null;
    const driftSec = currentTime == null ? Number.POSITIVE_INFINITY : Math.abs(currentTime - videoSec);
    const nowMs = performance.now();
    const shouldCorrectDrift = state.tl.playing
      && driftSec > 0.18
      && nowMs - (phone.lastDriftCorrectAt || 0) > 180;
    if (forceSeek || !state.tl.playing || shouldCorrectDrift) {
      seekVideoElement(phone.videoEl, videoSec);
      if (shouldCorrectDrift || forceSeek || !state.tl.playing) phone.lastDriftCorrectAt = nowMs;
    }
  }
  try { phone.videoEl.playbackRate = state.tl.playbackRate; } catch {}
  if (state.tl.playing) {
    if (phone.videoEl.paused) {
      const playPromise = phone.videoEl.play();
      if (playPromise?.catch) playPromise.catch(() => {});
    }
  } else {
    if (forceSeek && Number.isFinite(videoSec)) seekVideoElement(phone.videoEl, videoSec);
    try { phone.videoEl.pause(); } catch {}
  }
}

function toggleTimelinePhonePlayback(item) {
  if (!getPlaybackTimelineMedia().length) return;
  state.phonePlayback.enabled = !isPhonePlaybackEnabled();
  state.phonePlayback.selectedFileId = state.phonePlayback.enabled && item?.id ? String(item.id) : null;
  if (!state.phonePlayback.enabled) {
    hidePhonePlayback(true, false);
  } else {
    syncPhonePlaybackToTimeline({ forceSeek: true, forceReload: false });
  }
  updatePhonePlaybackToggleButton();
  renderTimelineMediaLayer();
}

function updatePhonePlaybackToggleButton() {
  const btn = el('btn-tl-phone-playback');
  if (!btn) return;
  const hasMedia = getPlaybackTimelineMedia().length > 0;
  const enabled = isPhonePlaybackEnabled();
  btn.style.display = hasMedia ? '' : 'none';
  btn.disabled = !hasMedia;
  btn.classList.toggle('active', hasMedia && enabled);
  btn.classList.toggle('inactive', hasMedia && !enabled);
  btn.textContent = enabled ? 'External Video: ON' : 'External Video: OFF';
  btn.title = hasMedia
    ? (enabled ? 'Turn synced external playback off' : 'Turn synced external playback on')
    : 'No external playback clips found in this project';
}

// File reconnection after page refresh

/**
 * Silently reconnect any file handles that Chrome already permits (no gesture).
 * Returns the IDs that still need a user gesture.
 */
async function autoReconnectFiles() {
  if (!state.projectId) return [];
  const files = await DB.listFiles(state.projectId);
  const videoIds = files.filter(f => f.kind === 'video').map(f => f.id);
  if (!videoIds.length) return [];
  const { reconnected, needsGesture } = await FM.autoReconnect(videoIds);
  // Invalidate URL cache for reconnected files
  for (const fid of reconnected) revokeVideoURL(fid);
  return needsGesture;
}

async function checkFileReconnection() {
  if (!state.projectId) return;
  const files = await DB.listFiles(state.projectId);
  const videoIds = files.filter(f => f.kind === 'video').map(f => f.id);
  if (!videoIds.length) return;
  const stale = await FM.countStaleHandles(videoIds);
  if (stale === 0) return;
  showReconnectBanner(videoIds, stale);
}

function showReconnectBanner(videoFiles, count) {
  // Remove any existing banner
  const existing = document.getElementById('reconnect-banner');
  if (existing) existing.remove();

  const banner = document.createElement('div');
  banner.id = 'reconnect-banner';
  banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#1a1e2e;border-bottom:2px solid #f5a623;padding:10px 20px;display:flex;align-items:center;gap:12px;font-size:13px;color:#e2e6ea;flex-wrap:wrap;';

  const videoIds = videoFiles.map(f => f.id);
  const names = videoFiles.filter(f => FM.getStaleFileIds([f.id]).length).map(f => f.filename).join(', ');

  if (FM.HAS_FILE_SYSTEM_ACCESS) {
    // Chrome/Edge: grant permission via stored handles
    banner.innerHTML = `
      <svg style="width:20px;height:20px;flex-shrink:0" viewBox="0 0 24 24" fill="none" stroke="#f5a623" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <span>${count} video file${count>1?'s':''} need${count===1?'s':''} reconnection after page refresh.</span>
      <button id="btn-reconnect" style="background:#f5a623;color:#000;border:none;border-radius:4px;padding:6px 16px;cursor:pointer;font-weight:600;font-size:13px;">Grant File Access</button>
      <button id="btn-reconnect-dismiss" style="background:transparent;color:#8b949e;border:1px solid #8b949e;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:12px;">Dismiss</button>
    `;
    document.body.prepend(banner);
    document.getElementById('btn-reconnect').onclick = async () => {
      const btn = document.getElementById('btn-reconnect');
      btn.textContent = 'Reconnecting...';
      btn.disabled = true;
      const { reconnected, failed } = await FM.reconnectHandles(videoIds);
      _afterReconnect(reconnected, banner, btn, failed.length);
    };
  } else {
    // Firefox/Safari: re-pick files and match by name+size
    banner.innerHTML = `
      <svg style="width:20px;height:20px;flex-shrink:0" viewBox="0 0 24 24" fill="none" stroke="#f5a623" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <span>${count} video file${count>1?'s':''} lost after page refresh. Re-select to reconnect:</span>
      <span style="color:#8b949e;font-size:11px;">${names}</span>
      <button id="btn-reconnect" style="background:#f5a623;color:#000;border:none;border-radius:4px;padding:6px 16px;cursor:pointer;font-weight:600;font-size:13px;">Re-pick Files</button>
      <button id="btn-reconnect-dismiss" style="background:transparent;color:#8b949e;border:1px solid #8b949e;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:12px;">Dismiss</button>
    `;
    document.body.prepend(banner);
    document.getElementById('btn-reconnect').onclick = async () => {
      // Open a file picker (user gesture) and match picked files to DB records
      const input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      input.accept = '.mp4,.mov,.csv';
      input.onchange = () => {
        const files = Array.from(input.files);
        if (!files.length) return;
        const { reconnected, unmatched } = FM.reconnectByRepick(files, videoFiles);
        const btn = document.getElementById('btn-reconnect');
        _afterReconnect(reconnected, banner, btn, unmatched.length);
      };
      input.click();
    };
  }
  document.getElementById('btn-reconnect-dismiss').onclick = () => banner.remove();
}

function _afterReconnect(reconnected, banner, btn, failedCount) {
  for (const fid of reconnected) revokeVideoURL(fid);
  if (state.tl?.athleteSlots) {
    for (const slot of state.tl.athleteSlots) slot.currentFileId = null;
  }
  buildTimeline();
  if (failedCount === 0) {
    banner.remove();
  } else {
    if (btn) {
      btn.textContent = `${reconnected.length} OK, ${failedCount} failed — Retry?`;
      btn.disabled = false;
    }
  }
}

function haversineKm(la1,lo1,la2,lo2){
  const R=6371,dL=(la2-la1)*Math.PI/180,dO=(lo2-lo1)*Math.PI/180;
  const a=Math.sin(dL/2)**2+Math.cos(la1*Math.PI/180)*Math.cos(la2*Math.PI/180)*Math.sin(dO/2)**2;
  return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}
function bearing(la1,lo1,la2,lo2){
  const p1=la1*Math.PI/180,p2=la2*Math.PI/180,dl=(lo2-lo1)*Math.PI/180;
  return(Math.atan2(Math.sin(dl)*Math.cos(p2),Math.cos(p1)*Math.sin(p2)-Math.sin(p1)*Math.cos(p2)*Math.cos(dl))*180/Math.PI+360)%360;
}
function confirmDialog(title, msg, cb) {
  el('confirm-title').textContent = title;
  el('confirm-msg').textContent = msg;
  el('confirm-modal').classList.add('open');
  const ok = el('btn-conf-ok');
  const cancel = el('btn-conf-cancel');
  function cleanup(){ el('confirm-modal').classList.remove('open'); ok.onclick=null; cancel.onclick=null; }
  ok.onclick = ()=>{ cleanup(); cb(); };
  cancel.onclick = ()=>{ cleanup(); };
}

// ── Project management ─────────────────────────────────────────────────
async function loadProjects() {
  state.projects = await DB.listProjects();
  renderProjectMenu();
  updateProjectLabel();
  renderProfileStep();
  updateSetupUiState();
}

async function selectProject(id) {
  cancelManualManeuverSegmentation({ silent: true });
  state.projectId = id;
  if (id) localStorage.setItem('trollfish_projectId', id);
  else localStorage.removeItem('trollfish_projectId');
  clearTrackLayers();
  state.mapData = null; state.cvStatuses = {}; state.cvConfig = {}; state.windConfig = { ...DEFAULT_WIND_CONFIG }; state.athletes = []; state.fileMeta = {};
  state.trackVisibility = {};
  state.segments = [];
  state.maneuvers = [];
  state.maneuverSelected = [];
  state.maneuverChecked = [];
  state.maneuverCompareOffsets = {};
  state.manualManeuverDraft = {
    active: false,
    step: 'start',
    trackFileId: null,
    trackPoints: [],
    tsStart: null,
    tsEnd: null,
    startMarker: null,
    endMarker: null,
    highlightLayer: null,
  };
  state.maneuverRefreshToken++;
  state.maneuverRefreshPromise = null;
  state.maneuverDetection = { ...DEFAULT_MANEUVER_DETECTION_SETTINGS };
  state.timelineMetricOverlayPrefs = {};
  state.uploadInProgress = false;
  state.hiddenVideoSlots = loadHiddenVideoSlots(id);
  state.mapShouldAutoFit = true;
  state.mapBaseZoom = null;
  state.wind.loadToken++;
  state.wind.loading = false;
  state.wind.promise = null;
  state.wind.byCsvId = {};
  state.wind.session = null;
  state.wind.localNow = null;
  state.wind._lastLocalEvalTs = NaN;
  cancelSegmentCreation();
  invalidateTimelineProcessedMetricCache(null, '');
  tlStop();
  destroyVideoSlots();
  const hasPid = !!id;
  el('btn-del-proj').disabled = !hasPid;
  el('files-list').innerHTML = '';
  el('ath-list').innerHTML = '';
  updateProjectLabel();
  renderProjectMenu();
  renderProfileStep();
  showNoVideo(true);
  renderWindMapControl();
  if(!id) {
    state.wizardStep = 1;
    state.analysisSelected = [];
    renderAnalysisTab();
    renderManeuverPanel();
    renderManeuverMap();
    void renderManeuverWorkspace();
    updateSetupUiState();
    return;
  }
  await loadProjectCvConfig();
  await loadAthletes();
  await Promise.all([loadMapData(), loadSegments(), loadManeuvers()]);
  const autoApplied = await autoAssignCsvAthletesFromMemory();
  if (autoApplied) {
    renderTrackPanel();
    renderMap();
  }
  try {
    const savedSel = localStorage.getItem('trollfish_analysisSelected_' + id);
    if (savedSel) state.analysisSelected = JSON.parse(savedSel);
    else state.analysisSelected = [];
  } catch { state.analysisSelected = []; }
  sanitizeAnalysisSelection();
  renderFilesList();
  renderAthletes();
  renderSegmentPanel();
  renderManeuverPanel();
  renderManeuverMap();
  void renderManeuverWorkspace();
  renderAnalysisTab();
  renderProfileStep();

  // Reset wizard step based on project data
  const setup = getSetupState();
  if(!setup.hasAthletes) state.wizardStep = 2;
  else if(!setup.hasFiles) state.wizardStep = 3;
  else state.wizardStep = 4;

  // ── File reconnection: try OPFS blobs + handles silently, then prompt ──
  if (setup.hasFiles) {
    const files = await DB.listFiles(state.projectId);
    const videoFiles = files.filter(f => f.kind === 'video');

    // autoReconnect now tries: memory cache → IDB handles → OPFS blobs
    const needsGesture = await autoReconnectFiles();
    buildTimeline();
    void refreshManeuvers('project-selected');
    if (needsGesture.length > 0) {
      showReconnectBanner(videoFiles, needsGesture.length);
    }
  } else {
    buildTimeline();
  }

  // Deferred redraw so SOG canvas gets correct dimensions after layout
  requestAnimationFrame(() => { setTimeout(() => drawSogCanvas(), 50); });
  updateSetupUiState();
}

async function createProject(name) {
  try {
    const p = await DB.createProject(name);
    await loadProjects();
    await selectProject(p.id);
    // Switch to athlete wizard step for new profiles
    state.wizardStep = 2;
    switchView('view-upload');
  } catch (e) {
    console.error('Create project failed:', e);
    const msg = e?.message || String(e);
    alert(`Failed to create project: ${msg}`);
    throw e;
  }
}

async function deleteProject() {
  if(!state.projectId) return;
  const name = state.projects.find(p=>p.id===state.projectId)?.name || 'this project';
  confirmDialog('Delete Project', `Delete "${name}" and all its files? This cannot be undone.`, async ()=>{
    const deletingId = state.projectId;
    await DB.deleteProject(state.projectId);
    saveJsonLocal(getReportHistoryKey(deletingId), []);
    await Storage.deleteDir([...REPORT_STORAGE_DIR, deletingId]);
    state.projectId = null;
    await loadProjects();
    await selectProject('');
  });
}

// ── Data loading ───────────────────────────────────────────────────────
async function loadMapData() {
  if(!state.projectId) return;
  try {
    state.mapData = await Pipeline.buildMapData(state.projectId);
    if(state.mapData.videos) state.mapData.videos = state.mapData.videos.map(v=>({...v,id:v.file_id}));
    if(state.mapData.csvs)   state.mapData.csvs   = state.mapData.csvs.map(c=>({...c,id:c.file_id}));
    normalizeMapVideoRanges(state.mapData.videos || []);
    state.cvStatuses = await Pipeline.buildCvStatuses(state.projectId);
    state.fileMeta = await DB.getFileMeta(state.projectId);
    await autoAssignCsvAthletesFromApiMeta();
    await autoAssignCsvAthletesFromMemory();
    assignVideoColors();
    initTrackVisibility();
    await loadAllSkeletonCoverages();
    state.mapShouldAutoFit = true;
    renderMap();
    renderWindMapControl();
    renderTrackPanel();
    loadWindEstimates().catch(err => console.warn('[wind] load failed:', err));
  } catch(e){ console.error(e); }
}

function computeDayLabel(ts) {
  if(!ts) return null;
  const d = new Date(ts * 1000);
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,'0')}-${String(d.getUTCDate()).padStart(2,'0')}`;
}

function normalizeAthleteId(id) {
  if (id == null) return null;
  const s = String(id).trim();
  return s || null;
}

function isPlaybackOnlyVideo(video) {
  return !!video?.playback_only;
}

function playbackOnlyVideoTitle(video) {
  const device = [video?.device_make, video?.device_model].filter(Boolean).join(' ').trim();
  const source = String(video?.capture_ts_source || '');
  const sourceText = source === 'quicktime_user_date'
    ? 'embedded original capture time'
    : (source === 'quicktime_creationdate'
      ? 'embedded QuickTime capture time'
      : (source === 'quicktime_mvhd' ? 'QuickTime movie header time' : 'video timestamp'));
  return device
    ? `${device} playback-only video synced from ${sourceText}`
    : `Playback-only video synced from ${sourceText}`;
}

function getPlaybackTimelineMedia() {
  return (state.mapData?.videos || [])
    .filter(v => isPlaybackOnlyVideo(v))
    .slice()
    .sort((a, b) => {
      const aStart = Number.isFinite(Number(a?.ts_start)) ? Number(a.ts_start) : Number.POSITIVE_INFINITY;
      const bStart = Number.isFinite(Number(b?.ts_start)) ? Number(b.ts_start) : Number.POSITIVE_INFINITY;
      if (aStart !== bStart) return aStart - bStart;
      return String(a?.filename || '').localeCompare(String(b?.filename || ''));
    });
}

function getTimelineMediaStartTs(item) {
  const ts = Number(item?.ts_start ?? item?.capture_start_ts);
  return Number.isFinite(ts) ? ts : null;
}

function getTimelineMediaEndTs(item) {
  const end = getVideoEndTs(item);
  return Number.isFinite(Number(end)) ? Number(end) : getTimelineMediaStartTs(item);
}

function getTimelineMediaKindLabel(fileOrItem) {
  const type = String(fileOrItem?.type || '').toLowerCase();
  if (type.startsWith('image/')) return 'IMG';
  return 'VID';
}

function getTimelineMediaDisplayName(item) {
  return String(item?.filename || item?.name || 'Timeline media');
}

function isPhonePlaybackEnabled() {
  return !!state.phonePlayback?.enabled && getPlaybackTimelineMedia().length > 0;
}

function getPhonePlaybackItemAtTs(absTs, preferredFileId = null) {
  if (!Number.isFinite(absTs)) return null;
  const mediaItems = getPlaybackTimelineMedia();
  if (!mediaItems.length) return null;
  if (preferredFileId) {
    const preferred = mediaItems.find(item => String(item?.id || '') === String(preferredFileId || ''));
    if (preferred && videoContainsAbsTs(preferred, absTs)) return preferred;
  }
  for (let i = mediaItems.length - 1; i >= 0; i--) {
    if (videoContainsAbsTs(mediaItems[i], absTs)) return mediaItems[i];
  }
  return null;
}

function getNextPhonePlaybackItemAfterTs(absTs) {
  if (!Number.isFinite(absTs)) return null;
  for (const item of getPlaybackTimelineMedia()) {
    const startTs = getTimelineMediaStartTs(item);
    if (Number.isFinite(startTs) && startTs >= absTs) return item;
  }
  return null;
}

function getSelectedPhonePlaybackItem() {
  if (!isPhonePlaybackEnabled()) return null;
  return getPhonePlaybackItemAtTs(state.tl?.currentTs, state.phonePlayback?.currentFileId);
}

function isPhonePlaybackSelected(item) {
  return !!item?.id && isPhonePlaybackEnabled();
}

function findAthleteById(athId) {
  const norm = normalizeAthleteId(athId);
  if (!norm) return null;
  return state.athletes.find(a => normalizeAthleteId(a?.id) === norm) || null;
}

/**
 * Find all videos that overlap a segment's time range.
 * Returns [{vid, videoStartSec, videoEndSec}] — video-local start/end seconds.
 */
function findOverlappingVideos(seg, { analyzableOnly = false } = {}) {
  const MIN_POSE_RANGE_SEC = 0.35;
  if (!state.mapData) return [];
  const vids = state.mapData.videos || [];
  const result = [];
  const seen = new Set();
  for (const v of vids) {
    if (analyzableOnly && isPlaybackOnlyVideo(v)) continue;
    const vidId = String(v?.id || '');
    if (!vidId || seen.has(vidId)) continue;
    const vStart = v.ts_start;
    const durationSec = Number(v.duration_sec);
    const vEnd = v.ts_end ?? (
      Number.isFinite(Number(vStart)) && Number.isFinite(durationSec)
        ? Number(vStart) + durationSec
        : null
    );
    let overlaps = false;
    if (vStart != null && vEnd != null) {
      overlaps = vStart <= seg.tsEnd && vEnd >= seg.tsStart;
    } else {
      const pts = (v.points || []).filter(p => p.ts != null && p.ts >= seg.tsStart && p.ts <= seg.tsEnd);
      overlaps = pts.length >= 2;
    }
    if (overlaps) {
      const overlapStartTs = vStart != null ? Math.max(seg.tsStart, vStart) : seg.tsStart;
      const overlapEndTs = (vStart != null && vEnd != null) ? Math.min(seg.tsEnd, vEnd) : seg.tsEnd;
      const durationRaw = Number(v?.duration_sec);
      const durationLimit = Number.isFinite(durationRaw) && durationRaw > 0 ? durationRaw : null;
      let videoStartSec = absTs2VideoSec(v, overlapStartTs);
      let videoEndSec = absTs2VideoSec(v, overlapEndTs);

      if (!Number.isFinite(Number(videoStartSec))) {
        videoStartSec = vStart != null ? Math.max(0, overlapStartTs - vStart) : 0;
      }
      if (!Number.isFinite(Number(videoEndSec))) {
        videoEndSec = vStart != null ? Math.max(Number(videoStartSec) || 0, overlapEndTs - vStart) : undefined;
      }

      if (durationLimit != null) {
        videoStartSec = Math.max(0, Math.min(Number(videoStartSec) || 0, durationLimit));
        if (Number.isFinite(Number(videoEndSec))) {
          videoEndSec = Math.max(videoStartSec, Math.min(Number(videoEndSec), durationLimit));
        }
      } else {
        videoStartSec = Math.max(0, Number(videoStartSec) || 0);
        if (Number.isFinite(Number(videoEndSec))) {
          videoEndSec = Math.max(videoStartSec, Number(videoEndSec));
        }
      }

      if (Number.isFinite(Number(videoEndSec)) && (Number(videoEndSec) - Number(videoStartSec)) < MIN_POSE_RANGE_SEC) {
        if (durationLimit != null) {
          const safeStart = Math.max(0, Math.min(Number(videoStartSec) || 0, Math.max(0, durationLimit - MIN_POSE_RANGE_SEC)));
          const safeEnd = Math.min(durationLimit, safeStart + MIN_POSE_RANGE_SEC);
          videoStartSec = safeStart;
          videoEndSec = safeEnd;
        } else {
          videoEndSec = Number(videoStartSec) + MIN_POSE_RANGE_SEC;
        }
      }

      // If mapping collapses both endpoints to the same value, keep a tiny range instead of dropping overlap.
      if (Number.isFinite(Number(videoEndSec)) && !(Number(videoEndSec) > Number(videoStartSec))) {
        if (durationLimit != null) {
          const safeStart = Math.max(0, Math.min(Number(videoStartSec) || 0, Math.max(0, durationLimit - MIN_POSE_RANGE_SEC)));
          const safeEnd = Math.min(durationLimit, safeStart + MIN_POSE_RANGE_SEC);
          videoStartSec = safeStart;
          videoEndSec = safeEnd;
        } else {
          videoEndSec = Number(videoStartSec) + MIN_POSE_RANGE_SEC;
        }
      }

      if (Number.isFinite(Number(videoEndSec)) && !(Number(videoEndSec) > Number(videoStartSec))) {
        console.warn(`[Segment] skipped empty overlap range for ${vidId}: ${videoStartSec} - ${videoEndSec}`);
        continue;
      }

      result.push({ vid: v, videoStartSec, videoEndSec });
      seen.add(vidId);
    }
  }
  return result;
}

/**
 * Resolve athletes that overlap a segment.
 * Returns [{athleteId, name, color}].
 */
function getSegmentAthletes(seg) {
  const overlapping = findOverlappingVideos(seg, { analyzableOnly: true });
  const seen = new Map(); // athleteId -> {name, color}
  for (const { vid } of overlapping) {
    const meta = state.fileMeta[vid.id] || {};
    let athId = normalizeAthleteId(meta.athlete_id);
    if (!athId && vid.best_match_csv_id) {
      const csvMeta = state.fileMeta[vid.best_match_csv_id] || {};
      athId = normalizeAthleteId(csvMeta.athlete_id);
    }
    if (!athId) athId = '__unassigned_' + vid.id;
    if (seen.has(athId)) continue;
    const ath = findAthleteById(athId);
    const color = state.videoColors[vid.id] || '#f5a623';
    seen.set(athId, { athleteId: athId, name: ath?.name || vid.filename || 'Unassigned', color });
  }
  return [...seen.values()];
}

/* Return the primary athlete color for a segment (first overlapping athlete) */
function getSegmentColor(seg) {
  const athletes = getSegmentAthletes(seg);
  return athletes.length > 0 ? athletes[0].color : '#f5a623';
}

function getSegmentById(segId) {
  const id = String(segId ?? '');
  return (state.segments || []).find(seg => String(seg.id) === id) || null;
}

function getSegmentAtTs(ts = state.tl?.currentTs) {
  if (!Number.isFinite(Number(ts))) return null;
  const hits = (state.segments || [])
    .filter(seg => Number.isFinite(seg?.tsStart) && Number.isFinite(seg?.tsEnd) && ts >= seg.tsStart && ts <= seg.tsEnd)
    .sort((a, b) => ((a.tsEnd - a.tsStart) - (b.tsEnd - b.tsStart)) || (a.tsStart - b.tsStart));
  return hits[0] || null;
}

function fmtDurationCompact(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  if (total >= 3600) {
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    return `${h}h ${m}m`;
  }
  if (total >= 60) {
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}m ${s}s`;
  }
  return `${total}s`;
}

function getManeuverById(maneuverId) {
  const id = String(maneuverId || '');
  return (state.maneuvers || []).find(maneuver => String(maneuver?.id) === id) || null;
}

function getManeuverColor(maneuver) {
  const athlete = findAthleteById(maneuver?.athlete_id);
  if (athlete?.color) return athlete.color;
  return String(maneuver?.type) === 'jibe' ? '#e86030' : '#1a73e8';
}

function sanitizeManeuverSelection() {
  const validIds = new Set((state.maneuvers || []).map(maneuver => String(maneuver.id)));
  state.maneuverSelected = (Array.isArray(state.maneuverSelected) ? state.maneuverSelected : [])
    .map(String)
    .filter(id => validIds.has(id));
}

function sanitizeManeuverChecked() {
  const validIds = new Set((state.maneuvers || []).map(maneuver => String(maneuver.id)));
  state.maneuverChecked = (Array.isArray(state.maneuverChecked) ? state.maneuverChecked : [])
    .map(String)
    .filter((id, idx, arr) => validIds.has(id) && arr.indexOf(id) === idx);
}

function getSelectedManeuvers() {
  sanitizeManeuverSelection();
  return state.maneuverSelected
    .map(id => getManeuverById(id))
    .filter(Boolean);
}

function getCheckedManeuvers() {
  sanitizeManeuverChecked();
  return state.maneuverChecked
    .map(id => getManeuverById(id))
    .filter(Boolean);
}

function getComparableManeuverSelection() {
  const selected = getSelectedManeuvers();
  if (selected.length < 2) return null;
  const firstType = String(selected[0]?.type || '');
  if (!selected.every(move => String(move?.type || '') === firstType)) return null;
  return selected;
}

function getComparableCheckedManeuvers() {
  const checked = getCheckedManeuvers();
  if (checked.length < 2) return null;
  const firstType = String(checked[0]?.type || '');
  if (!checked.every(move => String(move?.type || '') === firstType)) return null;
  return checked;
}

function buildManeuverRoleMap(maneuvers = []) {
  const roles = new Map();
  (Array.isArray(maneuvers) ? maneuvers : []).forEach((move, idx) => {
    const letter = idx < 26 ? String.fromCharCode(65 + idx) : String(idx + 1);
    roles.set(String(move?.id || ''), `Move ${letter}`);
  });
  return roles;
}

function buildManeuverCompareOffsetKey(maneuvers = [], move = null) {
  const ids = (Array.isArray(maneuvers) ? maneuvers : [])
    .map(move => String(move?.id || ''))
    .filter(Boolean)
    .sort();
  if (ids.length < 2) return '';
  const moveId = String(move?.id || '');
  return moveId ? `${ids.join('|')}::${moveId}` : ids.join('|');
}

function getManeuverCompareOffsetSec(maneuvers = [], move = null) {
  const key = buildManeuverCompareOffsetKey(maneuvers, move);
  if (!key) return 0;
  const value = Number(state.maneuverCompareOffsets?.[key]);
  return Number.isFinite(value) ? value : 0;
}

function setManeuverCompareOffsetSec(maneuvers = [], move = null, value = 0) {
  const key = buildManeuverCompareOffsetKey(maneuvers, move);
  if (!key) return;
  if (!state.maneuverCompareOffsets || typeof state.maneuverCompareOffsets !== 'object') {
    state.maneuverCompareOffsets = {};
  }
  state.maneuverCompareOffsets[key] = Number.isFinite(Number(value)) ? Number(value) : 0;
}

function getVisibleManeuvers() {
  const filters = state.maneuverFilters || {};
  return (state.maneuvers || []).filter(maneuver => {
    if (filters.athleteId && String(maneuver?.athlete_id || '') !== String(filters.athleteId)) return false;
    if (filters.type && filters.type !== 'all' && String(maneuver?.type || '') !== String(filters.type)) return false;
    if (filters.side && filters.side !== 'all' && String(maneuver?.side_from || '').toLowerCase() !== String(filters.side).toLowerCase()) return false;
    return true;
  });
}

function getManeuverProcessState(maneuver) {
  if (!maneuver) return { key: 'unprocessed', label: 'Unprocessed' };
  const active = findManeuverOverlappingVideos(maneuver, { analyzableOnly: true })
    .some(overlap => isManeuverPoseRunActive(overlap?.vid?.id));
  if (active) return { key: 'processing', label: 'Processing' };
  return maneuver.deepReady
    ? { key: 'processed', label: 'Processed' }
    : { key: 'unprocessed', label: 'Unprocessed' };
}

function syncManeuverFilterOptions() {
  const select = el('maneuver-athlete-filter');
  if (!select) return;
  const previous = String(state.maneuverFilters?.athleteId || '');
  const maneuvers = state.maneuvers || [];
  const athleteIds = [...new Set(maneuvers.map(maneuver => String(maneuver?.athlete_id || '')).filter(Boolean))];
  const options = athleteIds.map(id => {
    const athlete = findAthleteById(id);
    const fallback = maneuvers.find(maneuver => String(maneuver?.athlete_id || '') === id)?.athlete_name || 'Athlete';
    return { id, label: athlete?.name || fallback };
  }).sort((a, b) => a.label.localeCompare(b.label));
  select.innerHTML = '<option value="">All athletes</option>';
  for (const option of options) {
    const node = document.createElement('option');
    node.value = option.id;
    node.textContent = option.label;
    select.appendChild(node);
  }
  select.value = options.some(option => option.id === previous) ? previous : '';
  state.maneuverFilters = {
    ...(state.maneuverFilters || {}),
    athleteId: select.value,
  };
}

function isManeuverWorkspaceActive() {
  return getSelectedManeuvers().length > 0;
}

function scheduleManeuverMapRefresh({ render = true } = {}) {
  const run = () => {
    if (!state.maneuverMap) return;
    try { state.maneuverMap.invalidateSize(false); } catch {}
    if (render) renderManeuverMap();
  };
  requestAnimationFrame(() => requestAnimationFrame(run));
  setTimeout(run, 100);
  setTimeout(run, 280);
}

function syncManeuverWorkspaceSurface(active = isManeuverWorkspaceActive()) {
  if (!active || !state.maneuverMap) return;
  scheduleManeuverMapRefresh({ render: false });
}

function inspectManeuver(maneuverId, { openWorkspace = true, preserveMapView = true, renderMapView = true } = {}) {
  const id = String(maneuverId || '');
  if (!id) return;
  setManeuverSelection([id], {
    openWorkspace,
    preserveMapView,
    renderMapView,
  });
}

function toggleManeuverCompareSelection(maneuverId, { checked = null, openWorkspace = false, preserveMapView = true } = {}) {
  const id = String(maneuverId || '');
  if (!id) return;
  const next = new Set((state.maneuverChecked || []).map(String));
  const shouldCheck = checked == null ? !next.has(id) : !!checked;
  if (shouldCheck) {
    const move = (state.maneuvers || []).find(maneuver => String(maneuver?.id || '') === id);
    const moveType = String(move?.type || '');
    const currentChecked = getCheckedManeuvers();
    const hasDifferentType = !!moveType && currentChecked.some(checkedMove => String(checkedMove?.type || '') !== moveType);
    if (hasDifferentType) next.clear();
    next.add(id);
  } else {
    next.delete(id);
  }
  state.maneuverChecked = [...next];
  sanitizeManeuverChecked();
  renderManeuverPanel();
  renderManeuverMap({ preserveView: preserveMapView });
  if (openWorkspace) openCheckedManeuverCompare({ preserveMapView });
}

function clearManeuverCompareSet() {
  state.maneuverChecked = [];
  state.maneuverComparePicking = false;
  sanitizeManeuverChecked();
  renderManeuverPanel();
  renderManeuverMap({ preserveView: true });
}

function startManeuverComparePicking({ reset = true } = {}) {
  state.maneuverComparePicking = true;
  if (reset) state.maneuverChecked = [];
  sanitizeManeuverChecked();
  renderManeuverPanel();
  renderManeuverMap({ preserveView: true });
}

function cancelManeuverComparePicking({ clear = false } = {}) {
  state.maneuverComparePicking = false;
  if (clear) state.maneuverChecked = [];
  sanitizeManeuverChecked();
  renderManeuverPanel();
  renderManeuverMap({ preserveView: true });
}

function openCheckedManeuverCompare({ preserveMapView = true, loadDeep = true, force = false } = {}) {
  const checked = getCheckedManeuvers();
  if (checked.length < 2) {
    alert('Pick at least two maneuvers before finishing Compare.');
    return false;
  }
  const type = String(checked[0]?.type || '');
  if (!checked.every(move => String(move?.type || '') === type)) {
    alert('Compare needs checked maneuvers of the same type. Use the type filter or clear the compare set.');
    return false;
  }
  state.maneuverComparePicking = false;
  setManeuverSelection(checked.map(move => move.id), {
    openWorkspace: false,
    preserveMapView,
    renderMapView: true,
    renderWorkspace: false,
  });
  void openManeuverWorkspace({ force, preserveMapView, loadDeep, renderMapView: true });
  return true;
}

function setManeuverSelection(ids, {
  openWorkspace = true,
  preserveMapView = false,
  renderMapView = true,
  scrollIntoView = false,
  renderWorkspace = true,
} = {}) {
  state.maneuverSelected = Array.isArray(ids) ? ids.map(String) : [];
  sanitizeManeuverSelection();
  sanitizeManeuverChecked();
  syncManeuverWorkspaceSurface();
  renderManeuverPanel();
  if (scrollIntoView) scrollFirstSelectedManeuverRowIntoView();
  if (renderMapView) renderManeuverMap({ preserveView: preserveMapView });
  renderMap();
  drawSogCanvas();
  if (!renderWorkspace) {
    return;
  }
  if (openWorkspace && state.maneuverSelected.length > 0) {
    void openManeuverWorkspace({ preserveMapView, renderMapView });
  } else {
    void renderManeuverWorkspace();
  }
}

function scrollFirstSelectedManeuverRowIntoView() {
  const list = el('maneuver-panel-list');
  if (!list) return;
  const selected = list.querySelector('.mnv-row.is-selected');
  if (!selected) return;
  try {
    selected.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
  } catch {}
}

function toggleManeuverChecked(maneuverId, checked) {
  toggleManeuverCompareSelection(maneuverId, {
    checked: !!checked,
    openWorkspace: false,
    preserveMapView: true,
  });
}

async function loadManeuvers() {
  if (!state.projectId) {
    state.maneuvers = [];
    state.maneuverSelected = [];
    state.maneuverChecked = [];
    state.maneuverComparePicking = false;
    return;
  }
  try {
    state.maneuvers = await DB.getManeuvers(state.projectId);
  } catch {
    state.maneuvers = [];
  }
  sanitizeManeuverSelection();
  sanitizeManeuverChecked();
}

async function refreshManeuvers(reason = 'sync') {
  if (!state.projectId) {
    state.maneuvers = [];
    state.maneuverChecked = [];
    state.maneuverComparePicking = false;
    renderManeuverPanel();
    renderManeuverMap();
    void renderManeuverWorkspace();
    return [];
  }
  const csvTracks = state.mapData?.csvs || [];
  if (!csvTracks.length) {
    state.maneuvers = [];
    state.maneuverChecked = [];
    state.maneuverComparePicking = false;
    await DB.saveManeuvers(state.projectId, []);
    renderManeuverPanel();
    renderManeuverMap();
    void renderManeuverWorkspace();
    renderMap();
    drawSogCanvas();
    return [];
  }

  const token = ++state.maneuverRefreshToken;
  const promise = (async () => {
    await ensureWindEstimatesReady();
    const detectedManeuvers = await detectProjectManeuvers(state.projectId, {
      windContext: buildReportWindContext(),
      settings: state.maneuverDetection,
    });
    const maneuvers = detectedManeuvers.filter(maneuver => !!pickBestManeuverVideo(maneuver, { analyzableOnly: true }));
    if (token !== state.maneuverRefreshToken) return state.maneuvers || [];
    const validAnalysisIds = await invalidateStaleManeuverAnalyses(maneuvers);
    for (const maneuver of maneuvers) {
      maneuver.deepReady = validAnalysisIds.has(String(maneuver.id));
    }
    state.maneuvers = maneuvers;
    await DB.saveManeuvers(state.projectId, maneuvers);
    sanitizeManeuverSelection();
    sanitizeManeuverChecked();
    renderManeuverPanel();
    renderManeuverMap();
    void renderManeuverWorkspace();
    renderMap();
    drawSogCanvas();
    console.log(`[Maneuver] refreshed ${maneuvers.length} video-backed move(s) (${reason})`);
    return maneuvers;
  })();
  state.maneuverRefreshPromise = promise;
  try {
    return await promise;
  } finally {
    if (state.maneuverRefreshPromise === promise) state.maneuverRefreshPromise = null;
  }
}

function renderManeuverPanel() {
  const title = el('maneuver-panel-title');
  const meta = el('maneuver-panel-meta');
  const list = el('maneuver-panel-list');
  const athleteFilter = el('maneuver-athlete-filter');
  const typeFilter = el('maneuver-type-filter');
  const sideFilter = el('maneuver-side-filter');
  const refreshBtn = el('btn-maneuver-redetect');
  const exportBtn = el('btn-maneuver-download-map');
  const csvBtn = el('btn-maneuver-download-csv');
  const processBtn = el('btn-maneuver-process-selected');
  const compareBtn = el('btn-maneuver-open-compare');
  const clearCompareBtn = el('btn-maneuver-clear-compare');
  const compareTray = el('maneuver-compare-tray');
  const sidebar = el('maneuver-sidebar');
  if (!list) return;
  const previousScrollTop = list.scrollTop || 0;
  const restoreScroll = () => {
    requestAnimationFrame(() => {
      try { list.scrollTop = previousScrollTop; } catch {}
    });
  };
  syncManeuverFilterOptions();
  const pickingCompare = !!state.maneuverComparePicking;
  if (title) title.textContent = pickingCompare ? 'Pick Maneuvers' : 'Maneuver Map';
  if (athleteFilter) athleteFilter.disabled = !(state.maneuvers || []).length;
  if (typeFilter) typeFilter.value = String(state.maneuverFilters?.type || 'all');
  if (typeFilter) typeFilter.disabled = !(state.maneuvers || []).length;
  if (sideFilter) sideFilter.value = String(state.maneuverFilters?.side || 'all');
  if (sideFilter) sideFilter.disabled = !(state.maneuvers || []).length;
  if (refreshBtn) refreshBtn.disabled = !state.projectId;
  const visible = getVisibleManeuvers();
  const checkedMoves = getCheckedManeuvers();
  const checkedCount = checkedMoves.length;
  const comparableChecked = getComparableCheckedManeuvers();
  if (sidebar) sidebar.classList.toggle('is-picking', pickingCompare);
  const checkedType = checkedMoves.length ? String(checkedMoves[0]?.type || '') : '';
  const checkedMixed = checkedMoves.length >= 2 && !checkedMoves.every(move => String(move?.type || '') === checkedType);
  const selectedCount = getSelectedManeuvers().length;
  if (exportBtn) {
    exportBtn.disabled = checkedCount < 2 || checkedMixed;
    exportBtn.title = checkedMixed ? 'Compare map export needs one maneuver type at a time.' : 'Download the checked compare overlay map.';
  }
  if (compareBtn) {
    compareBtn.disabled = pickingCompare ? !comparableChecked : !visible.length;
    compareBtn.classList.toggle('primary', pickingCompare ? !!comparableChecked : !!visible.length);
    compareBtn.textContent = pickingCompare
      ? (comparableChecked ? `Finish (${checkedCount})` : `Pick ${Math.max(0, 2 - checkedCount)} more`)
      : 'Compare';
    compareBtn.title = pickingCompare
      ? (checkedMixed
        ? 'Compare needs picked maneuvers of the same type.'
        : 'Finish selection, process the maneuver windows, and open the compare columns.')
      : 'Start picking maneuvers from the map for comparison.';
  }
  if (clearCompareBtn) {
    clearCompareBtn.disabled = !pickingCompare && checkedCount < 1;
    clearCompareBtn.textContent = pickingCompare ? 'Cancel' : 'Clear';
  }
  if (csvBtn) {
    const csvVisible = !!state.advancedMode && !!state.reportOptions?.downloadCsv;
    csvBtn.style.display = csvVisible ? '' : 'none';
    csvBtn.disabled = !csvVisible || (checkedCount < 1 && selectedCount < 1);
    csvBtn.title = checkedCount > 0
      ? 'Download boom/rudder time-series CSV for checked maneuvers.'
      : 'Download boom/rudder time-series CSV for the selected maneuver.';
  }
  if (processBtn) {
    processBtn.disabled = checkedCount < 1 && selectedCount < 1;
    processBtn.textContent = checkedCount > 0 ? `Process Checked (${checkedCount})` : 'Process Selected';
  }
  if (compareTray) {
    compareTray.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'maneuver-compare-tray-head';
    const label = document.createElement('span');
    label.textContent = pickingCompare ? 'Picking Compare' : 'Compare';
    const count = document.createElement('span');
    count.textContent = `${checkedCount} for compare`;
    head.append(label, count);
    compareTray.appendChild(head);

    const status = document.createElement('div');
    status.className = 'maneuver-compare-tray-status';
    if (!pickingCompare) {
      status.textContent = 'Press Compare, then click maneuver markers on the map.';
    } else if (!checkedCount) {
      status.textContent = 'Click maneuver markers on the map to add them here.';
    } else if (checkedCount === 1) {
      status.textContent = 'Pick one more maneuver of the same type, then Finish.';
    } else if (checkedMixed) {
      status.textContent = 'Mixed tack/jibe sets cannot be compared together.';
    } else {
      status.textContent = `${checkedCount} ${getManeuverTypeLabel(checkedMoves[0])} maneuvers ready.`;
    }
    compareTray.appendChild(status);

    if (checkedMoves.length) {
      const chipRow = document.createElement('div');
      chipRow.className = 'maneuver-compare-chip-row';
      const colors = buildManeuverCompareColors(checkedMoves);
      checkedMoves.forEach((move, idx) => {
        const chip = document.createElement('div');
        chip.className = 'maneuver-compare-chip';
        const dot = document.createElement('span');
        dot.className = 'maneuver-compare-chip-dot';
        dot.style.background = colors[idx] || getManeuverColor(move);
        const text = document.createElement('span');
        text.className = 'maneuver-compare-chip-label';
        text.textContent = `${getManeuverRoleLabel(idx)} · ${move?.athlete_name || 'Athlete'} ${fmtClock(move?.anchor_ts)}`;
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'maneuver-compare-chip-remove';
        removeBtn.textContent = 'x';
        removeBtn.title = 'Remove from compare';
        removeBtn.onclick = evt => {
          evt.stopPropagation();
          toggleManeuverChecked(move.id, false);
        };
        chip.append(dot, text, removeBtn);
        chipRow.appendChild(chip);
      });
      compareTray.appendChild(chipRow);
    }
  }
  if (meta) {
    meta.textContent = !state.projectId
      ? 'Select a project to inspect auto-detected maneuvers.'
      : (pickingCompare
        ? `${checkedCount} picked · click markers on the map`
        : `${visible.length} markers shown${(state.maneuvers || []).length !== visible.length ? ` · ${(state.maneuvers || []).length} total` : ''}`);
  }
  list.innerHTML = '';
  list.style.display = 'none';

  if (!state.projectId) {
    const empty = document.createElement('div');
    empty.className = 'mnv-empty';
    empty.textContent = 'Choose a project to see detected maneuvers here.';
    restoreScroll();
    return;
  }

  if (!(state.maneuvers || []).length) {
    restoreScroll();
    return;
  }
  if (!visible.length) {
    restoreScroll();
    return;
  }

  const selectedIds = new Set((state.maneuverSelected || []).map(String));
  const checkedIds = new Set((state.maneuverChecked || []).map(String));
  const roleMap = getManeuverSelectionRoleMap();
  const checkedRoleMap = buildManeuverRoleMap(checkedMoves);
  const compareColorMap = new Map();
  if (comparableChecked?.length >= 2) {
    const compareColors = buildManeuverCompareColors(comparableChecked);
    comparableChecked.forEach((move, idx) => {
      compareColorMap.set(String(move.id), compareColors[idx]);
    });
  }
  const makeSectionTitle = (label, count) => {
    const node = document.createElement('div');
    node.className = 'mnv-section-title';
    node.textContent = `${label}${Number.isFinite(Number(count)) ? ` (${count})` : ''}`;
    list.appendChild(node);
  };
  const renderRows = (rows) => {
  for (const maneuver of rows) {
    const row = document.createElement('div');
    const rowId = String(maneuver.id);
    const selected = selectedIds.has(rowId);
    const checked = checkedIds.has(rowId);
    row.className = `mnv-row${selected ? ' is-selected' : ''}${checked ? ' is-checked' : ''}`;
    row.dataset.maneuverId = rowId;
    row.onclick = evt => {
      toggleManeuverChecked(maneuver.id, !checkedIds.has(rowId));
    };

    const main = document.createElement('div');
    main.className = 'mnv-row-main';
    const color = compareColorMap.get(rowId) || getManeuverColor(maneuver);
    const entry = Number(maneuver?.preStats?.entrySpeedKts ?? maneuver?.duringStats?.entrySpeedKts);
    const exit = Number(maneuver?.postStats?.exitSpeedKts ?? maneuver?.duringStats?.exitSpeedKts);
    const role = roleMap.get(rowId) || '';
    const checkedRole = checkedRoleMap.get(rowId) || '';
    const windDir = formatManeuverNumber(maneuver?.sourceWind?.directionDeg, { digits: 0, suffix: '°', empty: '--' });
    const windSpeed = formatManeuverNumber(maneuver?.sourceWind?.speedKts, { digits: 1, suffix: ' kt', empty: '--' });
    const processState = getManeuverProcessState(maneuver);
    main.innerHTML = `
      <div class="mnv-row-overline">
        <span class="mnv-row-role">
          <span class="mnv-row-select-mark">${checked ? '&#10003;' : ''}</span>
          <span class="mnv-row-color-dot" style="background:${color};"></span>
          ${checked ? `<span class="mnv-row-role-badge">${checkedRole || 'Compare'}</span>` : '<span class="mnv-row-compare-label">Click to compare</span>'}
        </span>
        <span class="mnv-row-role">
          <span class="mnv-status-pill ${processState.key}">${processState.label}</span>
          <span class="mnv-row-anchor">${fmtClock(maneuver?.anchor_ts)}</span>
        </span>
      </div>
      <div class="mnv-row-title">
        <span class="mnv-type-chip ${String(maneuver?.type || '')}" style="background:${color};">${String(maneuver?.type || '').startsWith('j') ? 'J' : 'T'}</span>
        <span>${maneuver?.athlete_name || 'Athlete'} · ${String(maneuver?.type || '') === 'jibe' ? 'Jibe' : 'Tack'} at ${fmtClock(maneuver?.anchor_ts)}</span>
      </div>
      <div class="mnv-row-meta">
        <span class="mnv-row-meta-item">${maneuver?.sourceWind?.directionDeg != null ? `Wind ${windDir}${maneuver?.sourceWind?.speedKts != null ? ` · ${windSpeed}` : ''}` : 'Wind pending'}</span>
        <span class="mnv-row-meta-item">${maneuver?.side_from || '?'} → ${maneuver?.side_to || '?'}</span>
        <span class="mnv-row-meta-item">${fmtDurationCompact(maneuver?.duringStats?.durationS || maneuver?.duration_s)}</span>
      </div>
      <div class="mnv-row-meta">
        <span class="mnv-row-meta-item">Entry ${Number.isFinite(entry) ? `${entry.toFixed(1)} kt` : '--'}</span>
        <span class="mnv-row-meta-item">Exit ${Number.isFinite(exit) ? `${exit.toFixed(1)} kt` : '--'}</span>
        <span class="mnv-row-meta-item">Δ ${formatManeuverNumber(maneuver?.heading_delta_deg, { digits: 0, suffix: '°' })}</span>
      </div>
    `;
    const compareCopy = document.createElement('div');
    compareCopy.className = 'mnv-row-compare';
    compareCopy.textContent = checkedIds.has(rowId)
      ? `${checkedRole || 'Checked'} staged for compare`
      : (selected ? `${role || 'Selected'} open for analysis` : 'Click to add to compare');

    const actions = document.createElement('div');
    actions.className = 'mnv-row-actions';
    const analyzeBtn = document.createElement('button');
    analyzeBtn.type = 'button';
    analyzeBtn.className = 'mnv-row-inspect-btn';
    analyzeBtn.textContent = 'Analyze';
    analyzeBtn.onclick = evt => {
      evt.stopPropagation();
      inspectManeuver(maneuver.id, {
        openWorkspace: true,
        preserveMapView: true,
      });
    };
    actions.appendChild(analyzeBtn);

    row.append(main, compareCopy, actions);
    list.appendChild(row);
  }
  };

  const checkedVisible = visible.filter(maneuver => checkedIds.has(String(maneuver.id)));
  const remainingVisible = visible.filter(maneuver => !checkedIds.has(String(maneuver.id)));
  if (checkedVisible.length) {
    makeSectionTitle('Compare set', checkedVisible.length);
    renderRows(checkedVisible);
    makeSectionTitle('Other maneuvers', remainingVisible.length);
  }
  renderRows(remainingVisible);
  restoreScroll();
}

function getTrackItemByFileId(fileId) {
  const id = String(fileId || '');
  if (!id) return null;
  const csv = (state.mapData?.csvs || []).find(item => (
    String(item?.id || '') === id
    || String(item?.file_id || '') === id
  ));
  if (csv) return csv;
  return (state.mapData?.videos || []).find(item => (
    String(item?.id || '') === id
    || String(item?.file_id || '') === id
  )) || null;
}

async function getTrackPointsByFileId(fileId) {
  const id = String(fileId || '');
  if (!id) return [];
  const liveTrack = getTrackItemByFileId(id);
  if (Array.isArray(liveTrack?.points) && liveTrack.points.length) {
    return liveTrack.points;
  }
  try {
    const trackRow = await DB.getTrackByFileId(id);
    if (!trackRow?.id) return [];
    const points = await DB.getTrackPoints(trackRow.id);
    return Array.isArray(points) ? points : [];
  } catch {
    return [];
  }
}

function getExpandedManeuverOverlayRows(sourcePoints, maneuver) {
  const rows = (Array.isArray(sourcePoints) ? sourcePoints : [])
    .filter(point => (
      Number.isFinite(Number(point?.ts))
      && Number.isFinite(Number(point?.lat))
      && Number.isFinite(Number(point?.lon))
    ))
    .sort((a, b) => Number(a?.ts) - Number(b?.ts));
  if (!rows.length) return [];

  const startTs = Number(maneuver?.start_ts);
  const endTs = Number(maneuver?.end_ts);
  const anchorTs = Number(maneuver?.anchor_ts);
  const durationS = Number.isFinite(endTs - startTs) ? Math.max(0, endTs - startTs) : 0;
  const basePadSec = Math.max(14, durationS * 1.75);
  const windowStartTs = Number.isFinite(startTs)
    ? startTs - basePadSec
    : (Number.isFinite(anchorTs) ? anchorTs - basePadSec : Number(rows[0]?.ts));
  const windowEndTs = Number.isFinite(endTs)
    ? endTs + basePadSec
    : (Number.isFinite(anchorTs) ? anchorTs + basePadSec : Number(rows[rows.length - 1]?.ts));

  let selected = rows.filter(point => Number(point.ts) >= windowStartTs && Number(point.ts) <= windowEndTs);
  if (selected.length >= 2) return selected;

  if (Number.isFinite(anchorTs)) {
    selected = rows
      .slice()
      .sort((a, b) => Math.abs(Number(a?.ts) - anchorTs) - Math.abs(Number(b?.ts) - anchorTs))
      .slice(0, Math.min(140, rows.length))
      .sort((a, b) => Number(a?.ts) - Number(b?.ts));
    if (selected.length >= 2) return selected;
  }

  if (rows.length >= 2) return rows.slice();
  return selected;
}

function getManeuverTrackPoints(maneuver) {
  const track = getTrackItemByFileId(maneuver?.track_file_id);
  if (!Array.isArray(track?.points)) return [];
  return getExpandedManeuverOverlayRows(track.points, maneuver);
}

function resolveVideoAthleteId(video) {
  if (!video) return null;
  const direct = normalizeAthleteId(video?.athlete_id);
  if (direct) return direct;
  const fileMetaId = normalizeAthleteId(state.fileMeta?.[String(video?.id || '')]?.athlete_id);
  if (fileMetaId) return fileMetaId;
  const csvId = String(video?.best_match_csv_id || '');
  if (!csvId) return null;
  return normalizeAthleteId(state.fileMeta?.[csvId]?.athlete_id);
}

function getPoseAthleteOptionsForVideo(video) {
  const athlete = findAthleteById(resolveVideoAthleteId(video));
  if (!athlete) return {};
  const opts = {};
  if (Number.isFinite(Number(athlete.weight))) opts.athleteWeight = Number(athlete.weight);
  if (Number.isFinite(Number(athlete.height))) opts.athleteHeight = Number(athlete.height);
  return opts;
}

function pickNearestTrackPoint(points, latlng) {
  const rows = (Array.isArray(points) ? points : []).filter(point => (
    Number.isFinite(Number(point?.lat))
    && Number.isFinite(Number(point?.lon))
    && Number.isFinite(Number(point?.ts))
  ));
  if (!rows.length || !latlng) return null;
  let best = null;
  let bestDist = Number.POSITIVE_INFINITY;
  const targetLat = Number(latlng?.lat);
  const targetLon = Number(latlng?.lng ?? latlng?.lon);
  for (const point of rows) {
    const dLat = Number(point.lat) - targetLat;
    const dLon = Number(point.lon) - targetLon;
    const distSq = (dLat * dLat) + (dLon * dLon);
    if (distSq < bestDist) {
      best = point;
      bestDist = distSq;
    }
  }
  return best;
}

function clearManualManeuverDraftLayers() {
  const draft = state.manualManeuverDraft;
  const map = state.maneuverMap;
  if (!draft || !map) return;
  for (const key of ['startMarker', 'endMarker', 'highlightLayer']) {
    if (draft[key]) {
      try { map.removeLayer(draft[key]); } catch {}
      draft[key] = null;
    }
  }
}

function updateManualManeuverButtonState() {
  const btn = el('btn-maneuver-manual');
  if (!btn) return;
  const active = !!state.manualManeuverDraft?.active;
  btn.classList.toggle('active', active);
  btn.textContent = active ? 'Cancel Manual Segment' : 'Manual Segment';
}

function updateManualManeuverStatusText() {
  const status = el('maneuver-side-status');
  if (!status) return;
  const draft = state.manualManeuverDraft;
  if (!draft?.active) return;
  status.textContent = draft.step === 'start'
    ? 'Manual segment: click a track on the map to set START.'
    : 'Manual segment: click a track on the map to set END.';
}

function cancelManualManeuverSegmentation({ silent = false } = {}) {
  clearManualManeuverDraftLayers();
  state.manualManeuverDraft = {
    active: false,
    step: 'start',
    trackFileId: null,
    trackPoints: [],
    tsStart: null,
    tsEnd: null,
    startMarker: null,
    endMarker: null,
    highlightLayer: null,
  };
  updateManualManeuverButtonState();
  if (!silent) {
    const status = el('maneuver-side-status');
    if (status) status.textContent = 'Ready';
  }
  renderManeuverMap({ preserveView: true });
}

function startManualManeuverSegmentation() {
  if (!state.projectId || !state.maneuverMap) {
    alert('Select a project and open the Maneuver workspace before manual segmentation.');
    return;
  }
  clearManualManeuverDraftLayers();
  state.manualManeuverDraft = {
    active: true,
    step: 'start',
    trackFileId: null,
    trackPoints: [],
    tsStart: null,
    tsEnd: null,
    startMarker: null,
    endMarker: null,
    highlightLayer: null,
  };
  updateManualManeuverButtonState();
  updateManualManeuverStatusText();
  renderManeuverMap({ preserveView: true });
}

function toggleManualManeuverSegmentation() {
  if (state.manualManeuverDraft?.active) {
    cancelManualManeuverSegmentation();
  } else {
    startManualManeuverSegmentation();
  }
}

function drawManualManeuverDraftHighlight() {
  const draft = state.manualManeuverDraft;
  if (!draft?.active || !state.maneuverMap) return;
  if (draft.highlightLayer) {
    try { state.maneuverMap.removeLayer(draft.highlightLayer); } catch {}
    draft.highlightLayer = null;
  }
  const points = Array.isArray(draft.trackPoints) ? draft.trackPoints : [];
  if (!(Number.isFinite(draft.tsStart) && Number.isFinite(draft.tsEnd)) || points.length < 2) return;
  const seg = points.filter(point => Number(point?.ts) >= draft.tsStart && Number(point?.ts) <= draft.tsEnd);
  if (seg.length < 2) return;
  draft.highlightLayer = L.polyline(seg.map(point => [point.lat, point.lon]), {
    color: '#f59e0b',
    weight: 7,
    opacity: 0.92,
    pane: 'processedTrackPane',
    dashArray: '9 7',
  }).addTo(state.maneuverMap);
}

async function createManualManeuverFromDraft() {
  const draft = state.manualManeuverDraft;
  if (!state.projectId || !draft?.active) return;
  const startTs = Number(draft.tsStart);
  const endTs = Number(draft.tsEnd);
  const trackFileId = String(draft.trackFileId || '');
  if (!trackFileId || !(Number.isFinite(startTs) && Number.isFinite(endTs) && endTs > startTs)) return;

  const trackPoints = Array.isArray(draft.trackPoints) ? draft.trackPoints : [];
  const anchorTs = startTs + ((endTs - startTs) / 2);
  const startPt = findTrackPointNearTs(trackPoints, startTs);
  const endPt = findTrackPointNearTs(trackPoints, endTs);
  const headingStart = Number(startPt?.hdg ?? startPt?.cog);
  const headingEnd = Number(endPt?.hdg ?? endPt?.cog);
  const headingDelta = Number.isFinite(headingStart) && Number.isFinite(headingEnd)
    ? Math.abs((((headingEnd - headingStart + 540) % 360) - 180))
    : null;
  const sideFrom = Number(startPt?.twa) < 0 ? 'port' : 'starboard';
  const sideTo = Number(endPt?.twa) < 0 ? 'port' : 'starboard';
  const preferredType = String(state.maneuverFilters?.type || '').toLowerCase();
  const type = preferredType === 'jibe' || preferredType === 'tack'
    ? preferredType
    : (Number.isFinite(headingDelta) && headingDelta >= 95 ? 'jibe' : 'tack');
  const athleteId = String(state.fileMeta?.[trackFileId]?.athlete_id || '');
  const athlete = findAthleteById(athleteId);
  const videoFileIds = (state.mapData?.videos || [])
    .filter(video => String(video?.best_match_csv_id || '') === trackFileId)
    .map(video => String(video.id));
  const id = `${trackFileId}:${type}:manual:${Math.round(anchorTs * 10)}:${Date.now().toString(36).slice(-5)}`;

  const manualMove = {
    id,
    project_id: state.projectId,
    athlete_id: athleteId,
    athlete_name: athlete?.name || 'Athlete',
    type,
    track_file_id: trackFileId,
    video_file_ids: videoFileIds,
    start_ts: startTs,
    anchor_ts: anchorTs,
    deepest_ts: anchorTs,
    end_ts: endTs,
    side_from: sideFrom,
    side_to: sideTo,
    sourceWind: null,
    anchorSource: 'manual',
    heading_delta_deg: Number.isFinite(headingDelta) ? headingDelta : null,
    preStats: null,
    duringStats: {
      durationS: Math.max(0, endTs - startTs),
      headingDeltaDeg: Number.isFinite(headingDelta) ? headingDelta : null,
    },
    postStats: null,
    deepReady: false,
    isManual: true,
  };

  state.maneuvers = [...(state.maneuvers || []), manualMove]
    .sort((a, b) => Number(a?.anchor_ts || 0) - Number(b?.anchor_ts || 0));
  await DB.saveManeuvers(state.projectId, state.maneuvers);
  cancelManualManeuverSegmentation({ silent: true });
  setManeuverSelection([manualMove.id], { openWorkspace: true, preserveMapView: true });
  const status = el('maneuver-side-status');
  if (status) status.textContent = 'Manual maneuver created.';
}

function handleManualManeuverMapClick(trackFileId, trackPoints, nearestPoint) {
  const draft = state.manualManeuverDraft;
  if (!draft?.active || !nearestPoint || !state.maneuverMap) return;
  const ts = Number(nearestPoint.ts);
  if (!Number.isFinite(ts)) return;

  if (draft.step === 'start') {
    draft.trackFileId = String(trackFileId || '');
    draft.trackPoints = Array.isArray(trackPoints) ? trackPoints : [];
    draft.tsStart = ts;
    draft.tsEnd = null;
    draft.step = 'end';
    if (draft.startMarker) {
      try { state.maneuverMap.removeLayer(draft.startMarker); } catch {}
    }
    draft.startMarker = L.circleMarker([nearestPoint.lat, nearestPoint.lon], {
      radius: 9,
      color: '#ffffff',
      weight: 2.5,
      fillColor: '#2ea043',
      fillOpacity: 1,
      pane: 'splitMarkersPane',
    }).addTo(state.maneuverMap);
    updateManualManeuverStatusText();
    return;
  }

  if (String(draft.trackFileId || '') !== String(trackFileId || '')) {
    alert('Finish this manual segment on the same track where you set START.');
    return;
  }

  draft.tsEnd = ts;
  if (draft.tsStart > draft.tsEnd) {
    [draft.tsStart, draft.tsEnd] = [draft.tsEnd, draft.tsStart];
  }
  if (draft.endMarker) {
    try { state.maneuverMap.removeLayer(draft.endMarker); } catch {}
  }
  draft.endMarker = L.circleMarker([nearestPoint.lat, nearestPoint.lon], {
    radius: 9,
    color: '#ffffff',
    weight: 2.5,
    fillColor: '#e3342f',
    fillOpacity: 1,
    pane: 'splitMarkersPane',
  }).addTo(state.maneuverMap);
  drawManualManeuverDraftHighlight();
  void createManualManeuverFromDraft();
}

function latLonToLocalMeters(lat, lon, anchorLat, anchorLon) {
  const latRad = Number(anchorLat) * Math.PI / 180;
  const dLat = (Number(lat) - Number(anchorLat)) * Math.PI / 180;
  const dLon = (Number(lon) - Number(anchorLon)) * Math.PI / 180;
  const earthRadiusM = 6371000;
  return {
    x: dLon * Math.cos(latRad) * earthRadiusM,
    y: dLat * earthRadiusM,
  };
}

function rotatePlanPoint(x, y, degrees) {
  const rad = (Number(degrees) || 0) * Math.PI / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return {
    x: x * cos - y * sin,
    y: x * sin + y * cos,
  };
}

function buildTrackOverlayGeometry(maneuver, sourcePointsOverride = null) {
  const track = getTrackItemByFileId(maneuver?.track_file_id);
  const sourcePts = Array.isArray(sourcePointsOverride)
    ? sourcePointsOverride
    : (Array.isArray(track?.points) ? track.points : []);
  const rows = getExpandedManeuverOverlayRows(sourcePts, maneuver);
  const anchor = findTrackPointNearTs(sourcePts, maneuver?.anchor_ts)
    || findTrackPointNearTs(rows, maneuver?.anchor_ts)
    || rows[Math.max(0, Math.floor(rows.length / 2))]
    || null;
  if (!anchor || !rows.length) {
    return { points: [], deepest: null };
  }
  const anchorLat = Number(anchor.lat);
  const anchorLon = Number(anchor.lon);
  const rotateBy = -(Number(maneuver?.sourceWind?.directionDeg) || Number(maneuver?.windDirectionDeg) || 0);
  const points = rows.map(point => {
    const local = latLonToLocalMeters(point.lat, point.lon, anchorLat, anchorLon);
    const rotated = rotatePlanPoint(local.x, local.y, rotateBy);
    return {
      absTs: Number(point.ts),
      x: rotated.x,
      y: rotated.y,
      lat: Number(point.lat),
      lon: Number(point.lon),
    };
  });
  let deepest = points[0] || null;
  for (const point of points) {
    if (!deepest || Math.abs(Number(point?.x) || 0) > Math.abs(Number(deepest?.x) || 0)) deepest = point;
  }
  return { points, deepest };
}

async function resolveManeuverOverlayGeometry(maneuver, analysis) {
  const analysisPoints = Array.isArray(analysis?.gpsOverlay?.points) ? analysis.gpsOverlay.points : [];
  const filteredAnalysis = analysisPoints.filter(point => Number.isFinite(Number(point?.x)) && Number.isFinite(Number(point?.y)));
  if (filteredAnalysis.length >= 2) {
    return {
      points: filteredAnalysis,
      deepest: (analysis?.gpsOverlay?.deepest && Number.isFinite(Number(analysis.gpsOverlay.deepest?.x)) && Number.isFinite(Number(analysis.gpsOverlay.deepest?.y)))
        ? analysis.gpsOverlay.deepest
        : filteredAnalysis.reduce((best, point) => (!best || Math.abs(Number(point?.x) || 0) > Math.abs(Number(best?.x) || 0) ? point : best), null),
    };
  }

  const fallbackPoints = Array.isArray(maneuver?.overlayPoints) ? maneuver.overlayPoints : [];
  const filteredFallback = fallbackPoints.filter(point => Number.isFinite(Number(point?.x)) && Number.isFinite(Number(point?.y)));
  if (filteredFallback.length >= 2) {
    return {
      points: filteredFallback,
      deepest: (maneuver?.overlayDeepest && Number.isFinite(Number(maneuver.overlayDeepest?.x)) && Number.isFinite(Number(maneuver.overlayDeepest?.y)))
        ? maneuver.overlayDeepest
        : filteredFallback.reduce((best, point) => (!best || Math.abs(Number(point?.x) || 0) > Math.abs(Number(best?.x) || 0) ? point : best), null),
    };
  }

  let geometry = buildTrackOverlayGeometry(maneuver);
  if (geometry.points.length >= 2) return geometry;

  const dbTrackPoints = await getTrackPointsByFileId(maneuver?.track_file_id);
  geometry = buildTrackOverlayGeometry(maneuver, dbTrackPoints);
  return geometry;
}

function findManeuverOverlappingVideos(maneuver, { analyzableOnly = false } = {}) {
  const ids = new Set((Array.isArray(maneuver?.video_file_ids) ? maneuver.video_file_ids : []).map(String));
  const trackFileId = String(maneuver?.track_file_id || '');
  const allVideos = state.mapData?.videos || [];
  const videos = ids.size
    ? allVideos.filter(video => ids.has(String(video?.id || '')))
    : allVideos.filter(video => String(video?.best_match_csv_id || '') === trackFileId);
  const maneuverAthleteId = normalizeAthleteId(maneuver?.athlete_id);
  const athleteScopedVideos = maneuverAthleteId
    ? videos.filter(video => resolveVideoAthleteId(video) === maneuverAthleteId)
    : [];
  const candidateVideos = athleteScopedVideos.length ? athleteScopedVideos : videos;
  const result = [];
  for (const video of candidateVideos) {
    if (analyzableOnly && isPlaybackOnlyVideo(video)) continue;
    const overlapStartTs = Math.max(Number(maneuver?.start_ts), Number(video?.ts_start ?? maneuver?.start_ts));
    const overlapEndTs = Math.min(
      Number(maneuver?.end_ts),
      Number((video?.ts_end ?? ((Number(video?.ts_start) || 0) + (Number(video?.duration_sec) || 0))) || maneuver?.end_ts),
    );
    if (!(overlapEndTs >= overlapStartTs)) continue;
    const videoStartSec = absTs2VideoSec(video, overlapStartTs);
    const videoEndSec = absTs2VideoSec(video, overlapEndTs);
    result.push({
      vid: video,
      videoStartSec: Number.isFinite(Number(videoStartSec)) ? Number(videoStartSec) : 0,
      videoEndSec: Number.isFinite(Number(videoEndSec)) ? Number(videoEndSec) : Number(video?.duration_sec || 0),
      overlapSec: Math.max(0, overlapEndTs - overlapStartTs),
      athleteMatch: resolveVideoAthleteId(video) === maneuverAthleteId,
    });
  }
  return result.sort((a, b) => {
    if (a.athleteMatch !== b.athleteMatch) return a.athleteMatch ? -1 : 1;
    return Number(b.overlapSec || 0) - Number(a.overlapSec || 0);
  });
}

function pickBestManeuverVideo(maneuver, { analyzableOnly = true } = {}) {
  return findManeuverOverlappingVideos(maneuver, { analyzableOnly })[0] || null;
}

function findTrackPointNearTs(points, targetTs) {
  const rows = Array.isArray(points) ? points : [];
  const target = Number(targetTs);
  if (!rows.length || !Number.isFinite(target)) return null;
  let best = null;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (const point of rows) {
    const ts = Number(point?.ts);
    if (!Number.isFinite(ts) || !Number.isFinite(Number(point?.lat)) || !Number.isFinite(Number(point?.lon))) continue;
    const delta = Math.abs(ts - target);
    if (delta < bestDelta) {
      best = point;
      bestDelta = delta;
    }
  }
  return best;
}

function clearManeuverMapLayers() {
  if (!state.maneuverMap) return;
  for (const layer of Object.values(state.maneuverMapLayers || {})) {
    if (layer) {
      try { state.maneuverMap.removeLayer(layer); } catch {}
    }
  }
  state.maneuverMapLayers = {
    tracks: null,
    markers: null,
    selection: null,
    wind: null,
  };
}

function getManeuverMapWindDirectionDeg(visible = []) {
  const selected = getSelectedManeuvers();
  const source = selected.length ? selected : (Array.isArray(visible) ? visible : []);
  const dirs = source
    .map(move => Number(move?.sourceWind?.directionDeg))
    .filter(Number.isFinite);
  if (!dirs.length) {
    const sessionDir = Number(state.wind?.session?.directionDeg);
    return Number.isFinite(sessionDir) ? sessionDir : null;
  }
  let x = 0;
  let y = 0;
  for (const dir of dirs) {
    const rad = (dir * Math.PI) / 180;
    x += Math.cos(rad);
    y += Math.sin(rad);
  }
  if (Math.abs(x) < 1e-9 && Math.abs(y) < 1e-9) return null;
  let out = (Math.atan2(y, x) * 180) / Math.PI;
  if (out < 0) out += 360;
  return out;
}

function createManeuverWindArrowMarker(latlng, directionDeg) {
  if (!latlng || !Number.isFinite(Number(directionDeg))) return null;
  const fromDeg = Number(directionDeg);
  const toDeg = (fromDeg + 180) % 360;
  const html = `
    <div class="maneuver-map-wind" title="Wind from ${Math.round(fromDeg)}°">
      <div class="maneuver-map-wind-arrow" style="transform: rotate(${toDeg}deg);">↑</div>
      <div class="maneuver-map-wind-label">Wind ${Math.round(fromDeg)}°</div>
    </div>
  `;
  return L.marker(latlng, {
    icon: L.divIcon({ html, className: 'maneuver-map-wind-marker', iconSize: [40, 44], iconAnchor: [20, 22] }),
    pane: 'splitMarkersPane',
    interactive: false,
    keyboard: false,
  });
}

function getManeuverMapClusterRadiusPx() {
  const zoom = Number(state.maneuverMap?.getZoom?.());
  if (!Number.isFinite(zoom) || zoom >= 16) return 0;
  if (zoom >= 14) return 34;
  if (zoom >= 12) return 46;
  return 62;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function clusterManeuverMapMarkers(markerRows, selectedIds) {
  const rows = Array.isArray(markerRows) ? markerRows : [];
  const radius = getManeuverMapClusterRadiusPx();
  if (!state.maneuverMap || radius <= 0) return rows.map(row => ({ rows: [row], latlng: row.latlng }));
  const clusters = [];
  for (const row of rows) {
    const id = String(row?.maneuver?.id || '');
    if (selectedIds?.has(id)) {
      clusters.push({ rows: [row], latlng: row.latlng, point: state.maneuverMap.latLngToLayerPoint(row.latlng) });
      continue;
    }
    const point = state.maneuverMap.latLngToLayerPoint(row.latlng);
    let best = null;
    let bestDist = Number.POSITIVE_INFINITY;
    for (const cluster of clusters) {
      if (cluster.rows.some(item => selectedIds?.has(String(item?.maneuver?.id || '')))) continue;
      const dist = point.distanceTo(cluster.point);
      if (dist <= radius && dist < bestDist) {
        best = cluster;
        bestDist = dist;
      }
    }
    if (!best) {
      clusters.push({ rows: [row], latlng: row.latlng, point });
      continue;
    }
    best.rows.push(row);
    const lat = best.rows.reduce((sum, item) => sum + Number(item.latlng.lat), 0) / best.rows.length;
    const lng = best.rows.reduce((sum, item) => sum + Number(item.latlng.lng), 0) / best.rows.length;
    best.latlng = L.latLng(lat, lng);
    best.point = state.maneuverMap.latLngToLayerPoint(best.latlng);
  }
  return clusters;
}

function getManeuverSelectionRoleMap() {
  const roles = new Map();
  const selected = getSelectedManeuvers();
  if (!selected.length) return roles;
  if (selected.length === 1) {
    roles.set(String(selected[0]?.id || ''), 'Selected');
    return roles;
  }
  selected.forEach((move, idx) => {
    const letter = idx < 26 ? String.fromCharCode(65 + idx) : String(idx + 1);
    roles.set(String(move?.id || ''), `Move ${letter}`);
  });
  return roles;
}

function getManeuverRoleLabel(idx) {
  const index = Math.max(0, Number(idx) || 0);
  const letter = index < 26 ? String.fromCharCode(65 + index) : String(index + 1);
  return `Move ${letter}`;
}

function renderManeuverMap({ preserveView = false } = {}) {
  if (!state.maneuverMap) return;
  clearManeuverMapLayers();
  if (!state.mapData || !state.projectId) return;

  const visible = getVisibleManeuvers();
  if (!visible.length) return;

  const csvById = new Map((state.mapData?.csvs || []).map(csv => [String(csv.id), csv]));
  const trackGroup = L.layerGroup();
  const markerGroup = L.layerGroup();
  const selectionGroup = L.layerGroup();
  const selectedIds = new Set((state.maneuverSelected || []).map(String));
  const checkedIds = new Set((state.maneuverChecked || []).map(String));
  const roleMap = getManeuverSelectionRoleMap();
  const checkedRoleMap = buildManeuverRoleMap(getCheckedManeuvers());
  const comparableSelection = getComparableCheckedManeuvers();
  const compareColorMap = new Map();
  if (comparableSelection?.length >= 2) {
    const compareColors = buildManeuverCompareColors(comparableSelection);
    comparableSelection.forEach((move, idx) => {
      compareColorMap.set(String(move.id), compareColors[idx]);
    });
  }

  const contextBounds = [];
  const focusBounds = [];
  const markerRows = [];
  const visibleTrackIds = [...new Set(visible.map(maneuver => String(maneuver?.track_file_id || '')).filter(Boolean))];
  for (const trackFileId of visibleTrackIds) {
    const csv = csvById.get(trackFileId);
    const pts = Array.isArray(csv?.points)
      ? csv.points.filter(point => Number.isFinite(Number(point?.lat)) && Number.isFinite(Number(point?.lon)))
      : [];
    if (pts.length < 2) continue;
    const color = getManeuverColor(visible.find(maneuver => String(maneuver?.track_file_id || '') === trackFileId) || { track_file_id: trackFileId });
    const trackPolyline = L.polyline(pts.map(point => [point.lat, point.lon]), {
      color,
      weight: 3.5,
      opacity: 0.38,
      pane: 'baseTrackPane',
    });
    trackPolyline.on('click', evt => {
      if (!state.manualManeuverDraft?.active) return;
      const nearest = pickNearestTrackPoint(pts, evt?.latlng);
      if (!nearest) return;
      handleManualManeuverMapClick(trackFileId, pts, nearest);
    });
    trackPolyline.addTo(trackGroup);
    const step = Math.max(1, Math.floor(pts.length / 160));
    for (let idx = 0; idx < pts.length; idx += step) contextBounds.push([pts[idx].lat, pts[idx].lon]);
    const last = pts[pts.length - 1];
    if (last) contextBounds.push([last.lat, last.lon]);
  }

  for (const maneuver of visible) {
    const csv = csvById.get(String(maneuver?.track_file_id || ''));
    const sourcePts = Array.isArray(csv?.points) ? csv.points : [];
    const movePts = getManeuverTrackPoints(maneuver);
    const anchorPoint = findTrackPointNearTs(sourcePts, maneuver?.anchor_ts) || findTrackPointNearTs(movePts, maneuver?.anchor_ts);
    if (!anchorPoint) continue;

    const color = compareColorMap.get(String(maneuver.id)) || getManeuverColor(maneuver);
    const processState = getManeuverProcessState(maneuver);
    const markerLatLng = L.latLng(anchorPoint.lat, anchorPoint.lon);
    const clusterRadiusPx = getManeuverMapClusterRadiusPx();
    if (clusterRadiusPx > 0 && !selectedIds.has(String(maneuver.id)) && !checkedIds.has(String(maneuver.id)) && state.maneuverMap) {
      const markerPoint = state.maneuverMap.latLngToLayerPoint(markerLatLng);
      const overlapsExisting = markerRows.some(row => {
        if (!row?.latlng || selectedIds.has(String(row?.maneuver?.id || '')) || checkedIds.has(String(row?.maneuver?.id || ''))) return false;
        return markerPoint.distanceTo(state.maneuverMap.latLngToLayerPoint(row.latlng)) <= clusterRadiusPx;
      });
      if (overlapsExisting) {
        contextBounds.push([anchorPoint.lat, anchorPoint.lon]);
        continue;
      }
    }
    markerRows.push({ maneuver, latlng: markerLatLng });
    const isSelected = selectedIds.has(String(maneuver.id));
    const isChecked = checkedIds.has(String(maneuver.id));
    const markerHtml = `
      <div class="maneuver-map-chip-shell">
        <div class="maneuver-map-chip ${String(maneuver.type || '')} ${isSelected ? 'selected' : ''} ${isChecked ? 'checked' : ''}" style="background:${color};">${String(maneuver.type || '').startsWith('j') ? 'J' : 'T'}</div>
        <span class="maneuver-map-status-dot ${processState.key}" title="${processState.label}"></span>
      </div>
    `;
    const marker = L.marker([anchorPoint.lat, anchorPoint.lon], {
      icon: L.divIcon({ html: markerHtml, className: 'maneuver-map-marker', iconSize: null, iconAnchor: [0, 20] }),
      pane: 'splitMarkersPane',
    });
    const role = roleMap.get(String(maneuver.id)) || checkedRoleMap.get(String(maneuver.id)) || '';
    marker.bindTooltip(
      `${role ? `${role} · ` : ''}${maneuver?.athlete_name || 'Athlete'} ${getManeuverTypeLabel(maneuver)} · ${fmtClock(maneuver?.anchor_ts)} · ${state.maneuverComparePicking ? `click to ${isChecked ? 'remove' : 'compare'}` : 'click for stats'}`,
      {
        direction: 'top',
        offset: [0, -20],
        opacity: 0.96,
        className: 'maneuver-map-tooltip',
        permanent: selectedIds.has(String(maneuver.id)),
      },
    );
    marker.on('click', evt => {
      const originalEvent = evt?.originalEvent;
      if (state.maneuverComparePicking) {
        toggleManeuverCompareSelection(maneuver.id, {
          checked: !checkedIds.has(String(maneuver.id)),
          openWorkspace: false,
          preserveMapView: true,
        });
      } else {
        inspectManeuver(maneuver.id, {
          openWorkspace: true,
          preserveMapView: true,
          renderMapView: false,
        });
      }
      try { L.DomEvent.stop(originalEvent); } catch {
        try { originalEvent?.stopPropagation?.(); } catch {}
      }
    });
    marker.addTo(markerGroup);
    contextBounds.push([anchorPoint.lat, anchorPoint.lon]);

    if (!selectedIds.has(String(maneuver.id)) || movePts.length < 2) continue;
    L.polyline(movePts.map(point => [point.lat, point.lon]), {
      color,
      weight: 7,
      opacity: 0.96,
      dashArray: String(maneuver?.type || '') === 'jibe' ? '8 8' : null,
      pane: 'processedTrackPane',
    }).addTo(selectionGroup);
    for (const point of movePts) focusBounds.push([point.lat, point.lon]);

    const deepest = findTrackPointNearTs(movePts, maneuver?.deepest_ts);
    if (deepest) {
      L.circleMarker([deepest.lat, deepest.lon], {
        radius: 6.5,
        color: '#ffffff',
        weight: 2,
        fillColor: color,
        fillOpacity: 1,
        pane: 'splitMarkersPane',
      }).addTo(selectionGroup);
      focusBounds.push([deepest.lat, deepest.lon]);
    }
  }

  trackGroup.addTo(state.maneuverMap);
  markerGroup.addTo(state.maneuverMap);
  selectionGroup.addTo(state.maneuverMap);
  state.maneuverMapLayers = {
    tracks: trackGroup,
    markers: markerGroup,
    selection: selectionGroup,
  };
  if (state.manualManeuverDraft?.active) {
    drawManualManeuverDraftHighlight();
  }

  const fitPoints = focusBounds.length ? focusBounds : contextBounds;
  const windDirDeg = getManeuverMapWindDirectionDeg(visible);
  if (fitPoints.length && Number.isFinite(Number(windDirDeg))) {
    const center = L.latLngBounds(fitPoints).getCenter();
    const windMarker = createManeuverWindArrowMarker(center, windDirDeg);
    if (windMarker) {
      windMarker.addTo(state.maneuverMap);
      state.maneuverMapLayers.wind = windMarker;
    }
  }
  if (!fitPoints.length) return;
  if (preserveView) return;
  try {
    state.maneuverMap.fitBounds(L.latLngBounds(fitPoints), {
      padding: focusBounds.length ? [34, 34] : [26, 26],
      maxZoom: focusBounds.length ? 17 : 15,
      animate: false,
    });
  } catch {}
}

function getManeuverTypeLabel(typeOrManeuver) {
  const type = typeof typeOrManeuver === 'string' ? typeOrManeuver : String(typeOrManeuver?.type || '');
  return type === 'jibe' ? 'Jibe' : 'Tack';
}

function formatManeuverNumber(value, { digits = 1, suffix = '', signed = false, empty = '--' } = {}) {
  const num = Number(value);
  if (!Number.isFinite(num)) return empty;
  const sign = signed && num > 0 ? '+' : '';
  return `${sign}${num.toFixed(digits)}${suffix}`;
}

function formatManeuverDuration(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return '--';
  if (seconds >= 60) return fmtDurationCompact(seconds);
  return `${seconds.toFixed(1)}s`;
}

const MANEUVER_COMPARE_FALLBACK_COLORS = ['#0f766e', '#d97706', '#2563eb', '#d9485f', '#7c3aed', '#0891b2'];

function normalizeHexColor(value) {
  const input = String(value || '').trim();
  if (!input.startsWith('#')) return null;
  if (input.length === 4) {
    return `#${input[1]}${input[1]}${input[2]}${input[2]}${input[3]}${input[3]}`;
  }
  if (input.length === 7) return input;
  return null;
}

function hexToRgb(value) {
  const hex = normalizeHexColor(value);
  if (!hex) return null;
  const raw = hex.slice(1);
  return {
    r: Number.parseInt(raw.slice(0, 2), 16),
    g: Number.parseInt(raw.slice(2, 4), 16),
    b: Number.parseInt(raw.slice(4, 6), 16),
  };
}

function rgbToHex({ r, g, b }) {
  const toHex = component => Math.max(0, Math.min(255, Math.round(Number(component) || 0))).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function mixHexColors(base, target, ratio = 0.5) {
  const a = hexToRgb(base);
  const b = hexToRgb(target);
  if (!a || !b) return null;
  const t = Math.max(0, Math.min(1, Number(ratio) || 0));
  return rgbToHex({
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  });
}

function buildManeuverCompareColors(maneuvers = []) {
  const chosen = [];
  const used = new Set();
  for (let idx = 0; idx < maneuvers.length; idx++) {
    const maneuver = maneuvers[idx];
    const base = getManeuverColor(maneuver);
    const baseKey = String(normalizeHexColor(base) || base || '').toLowerCase();
    const sameAthleteSeen = chosen.some((_, prevIdx) => String(maneuvers[prevIdx]?.athlete_id || '') === String(maneuver?.athlete_id || ''));
    const duplicateColorSeen = !!baseKey && used.has(baseKey);
    const fallback = MANEUVER_COMPARE_FALLBACK_COLORS.find(candidate => {
      const key = String(normalizeHexColor(candidate) || candidate || '').toLowerCase();
      return key && key !== baseKey && !used.has(key);
    }) || MANEUVER_COMPARE_FALLBACK_COLORS[idx % MANEUVER_COMPARE_FALLBACK_COLORS.length];
    const candidates = (
      sameAthleteSeen || duplicateColorSeen
        ? [fallback, mixHexColors(base, fallback, 0.38), base]
        : [base, fallback]
    ).filter(Boolean);
    let picked = candidates[0];
    for (const candidate of candidates) {
      const key = String(normalizeHexColor(candidate) || candidate || '').toLowerCase();
      if (!key || !used.has(key)) {
        picked = candidate;
        break;
      }
    }
    const pickedKey = String(normalizeHexColor(picked) || picked || '').toLowerCase();
    if (pickedKey) used.add(pickedKey);
    chosen.push(picked);
  }
  return chosen;
}

function setManeuverSideKpis(kpis = []) {
  const wrap = el('maneuver-side-kpis');
  if (!wrap) return;
  wrap.innerHTML = '';
  for (const card of (Array.isArray(kpis) ? kpis : []).slice(0, 4)) {
    const node = document.createElement('div');
    node.className = 'maneuver-side-kpi-card';
    const label = document.createElement('div');
    label.className = 'maneuver-side-kpi-label';
    label.textContent = card?.label || '';
    const value = document.createElement('div');
    value.className = 'maneuver-side-kpi-value';
    value.textContent = card?.value || '--';
    node.append(label, value);
    wrap.appendChild(node);
  }
}

function setManeuverSidePanel({
  title = 'No maneuver selected',
  sub = 'Select one maneuver to inspect it, or multiple same-type maneuvers to compare them.',
  status = 'Ready',
  kpis = [],
  analyzeEnabled = false,
  segmentEnabled = false,
  analyzeLabel = 'Process Maneuver',
} = {}) {
  const titleEl = el('maneuver-side-title');
  const subEl = el('maneuver-side-sub');
  const statusEl = el('maneuver-side-status');
  const analyzeBtn = el('btn-maneuver-analyze');
  const segmentBtn = el('btn-maneuver-segment');
  const refreshBtn = el('btn-maneuver-refresh');

  if (titleEl) titleEl.textContent = title;
  if (subEl) subEl.textContent = sub;
  if (statusEl) statusEl.textContent = status;
  if (refreshBtn) refreshBtn.disabled = !state.projectId;
  if (analyzeBtn) {
    analyzeBtn.disabled = !analyzeEnabled;
    analyzeBtn.textContent = analyzeLabel;
  }
  if (segmentBtn) segmentBtn.disabled = !segmentEnabled;
  setManeuverSideKpis(kpis);
}

function createManeuverEmptyState(title, body) {
  const wrap = document.createElement('div');
  wrap.className = 'maneuver-stage-empty';
  const titleEl = document.createElement('div');
  titleEl.className = 'title';
  titleEl.textContent = title;
  const bodyEl = document.createElement('div');
  bodyEl.className = 'body';
  bodyEl.textContent = body;
  wrap.append(titleEl, bodyEl);
  return wrap;
}

function createManeuverInlineNote(body) {
  const wrap = document.createElement('div');
  wrap.className = 'maneuver-inline-note';
  wrap.textContent = body;
  return wrap;
}

function createManeuverActionState({
  title = 'Maneuver Analysis Ready',
  body = '',
  buttonLabel = 'Process',
  onClick = null,
  disabled = false,
} = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'maneuver-stage-empty';
  const titleEl = document.createElement('div');
  titleEl.className = 'title';
  titleEl.textContent = title;
  const bodyEl = document.createElement('div');
  bodyEl.className = 'body';
  bodyEl.textContent = body;
  wrap.append(titleEl, bodyEl);
  if (typeof onClick === 'function') {
    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.textContent = buttonLabel;
    btn.disabled = !!disabled;
    btn.style.marginTop = '6px';
    btn.onclick = () => onClick();
    wrap.appendChild(btn);
  }
  return wrap;
}

async function triggerManeuverPoseExtraction(maneuvers, allowedVideoIds = null) {
  if (!state.projectId) { alert('No project selected'); return; }
  const jobs = [];
  const seenJobs = new Set();
  let skippedReady = 0;
  for (const move of maneuvers) {
    const cachedAnalysis = await DB.getManeuverAnalysis(state.projectId, move?.id);
    if (hasCompleteManeuverProcessing(cachedAnalysis)) {
      skippedReady++;
      continue;
    }
    for (const ov of findManeuverOverlappingVideos(move, { analyzableOnly: true })) {
      const fileId = String(ov?.vid?.id || '');
      if (!fileId) continue;
      if (allowedVideoIds instanceof Set && !allowedVideoIds.has(fileId)) continue;
      const moveDurationSec = Math.max(0, Number(move?.end_ts) - Number(move?.start_ts));
      const padSec = Math.min(6, Math.max(3, moveDurationSec * 0.35));
      const durationLimit = Number(ov?.vid?.duration_sec);
      const startSec = Math.max(0, Number(ov.videoStartSec) - padSec);
      const rawEndSec = Number(ov.videoEndSec) + padSec;
      const endSec = Number.isFinite(durationLimit) && durationLimit > 0 ? Math.min(durationLimit, rawEndSec) : rawEndSec;
      const key = `${fileId}|${String(move?.id || '')}|${startSec.toFixed(2)}|${endSec.toFixed(2)}`;
      if (seenJobs.has(key)) continue;
      seenJobs.add(key);
      jobs.push({
        vid: ov.vid,
        maneuverId: String(move?.id || ''),
        videoStartSec: startSec,
        videoEndSec: endSec,
        segName: `${getManeuverTypeLabel(move)} ${fmtClock(move?.anchor_ts)}`,
      });
    }
  }
  if (jobs.length === 0) {
    alert(skippedReady > 0
      ? 'All selected maneuvers already have processed video/pose data.'
      : 'No analyzable video covers these maneuvers.');
    return;
  }
  seedQueueAggregateProgress(jobs.map(ov => ({
    fileId: ov.vid.id,
    startSec: ov.videoStartSec,
    endSec: ov.videoEndSec,
    segmentName: ov.segName,
  })));
  updateProgressRing(2);
  
  for (const ov of jobs) {
    runSkeletonForVideo(ov.vid.id, {
      startSec: ov.videoStartSec,
      endSec: ov.videoEndSec,
      segmentName: ov.segName,
      fps: getManeuverPoseFps(),
    })
      .then(async () => {
        await loadAllSkeletonCoverages();
        await refreshManeuvers('pose-processed');
        await openManeuverWorkspace({ force: true, preserveMapView: true, loadDeep: true });
        renderManeuverPanel();
        renderManeuverMap({ preserveView: true });
      })
      .catch(e => console.warn('Maneuver pose run failed:', e));
  }
  renderQueue();
  syncManeuverWorkspaceSurface(true);
}

function buildManeuverProcessActions({
  maneuvers = [],
  analyses = [],
  force = false,
  preserveMapView = true,
  emphasize = false,
} = {}) {
  const moves = Array.isArray(maneuvers) ? maneuvers.filter(Boolean) : [];
  if (!moves.length) return null;
  const analysisRows = Array.isArray(analyses) ? analyses : [];
  const active = moves.some(move => (
    findManeuverOverlappingVideos(move, { analyzableOnly: true })
      .some(overlap => isManeuverPoseRunActive(overlap?.vid?.id))
  ));
  const hasReadyPose = moves.every((move, idx) => (
    !!analysisRows[idx] && hasCompleteManeuverProcessing(analysisRows[idx])
  ));
  const btnGroup = document.createElement('div');
  btnGroup.style.display = 'flex';
  btnGroup.style.gap = '10px';
  btnGroup.style.alignItems = 'center';

  const btn = document.createElement('button');
  btn.className = `btn maneuver-process-btn${emphasize ? ' is-hero' : ''}`;
  btn.textContent = active
    ? 'Processing...'
    : (hasReadyPose ? 'Refresh Video/Pose' : (moves.length > 1 ? 'Process Compare' : 'Process Maneuver'));
  btn.disabled = active;
  btn.onclick = () => {
    // Unify logic: extract pose AND open workspace
    if (!hasReadyPose || emphasize) {
       triggerManeuverPoseExtraction(moves);
    } else {
       void openManeuverWorkspace({ force: true, preserveMapView: true, loadDeep: true });
    }
  };
  btnGroup.appendChild(btn);



  const metaText = active
    ? 'Pose processing is running for one of the selected maneuver windows. The workspace will refresh when it finishes.'
    : (hasReadyPose
      ? 'Pose data already exists for this maneuver window. Refresh Pose to rerun the bounded pose pass and rebuild the overlay.'
      : (emphasize
        ? ''
        : 'Preview map first, then process only when you want video, pose density, and timelines for this maneuver.'));

  const row = document.createElement('div');
  row.className = `maneuver-process-row${emphasize ? ' is-hero' : ''}`;
  row.appendChild(btnGroup);
  if (metaText) {
    const meta = document.createElement('div');
    meta.className = 'maneuver-inline-note';
    meta.style.padding = '10px 14px';
    meta.textContent = metaText;
    row.appendChild(meta);
  }
  return row;
}

function createManeuverSection(title, subtitle = '') {
  const section = document.createElement('section');
  section.className = 'maneuver-section-card';
  const head = document.createElement('div');
  head.className = 'maneuver-section-head';
  const heading = document.createElement('h3');
  heading.textContent = title;
  head.appendChild(heading);
  if (subtitle) {
    const meta = document.createElement('div');
    meta.className = 'heatmap-meta';
    meta.style.color = '#61758b';
    meta.textContent = subtitle;
    head.appendChild(meta);
  }
  const body = document.createElement('div');
  body.className = 'maneuver-section-body';
  section.append(head, body);
  return { section, body };
}

function createManeuverCanvas(width = 900, height = 420) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function drawManeuverDirectionArrow(ctx, from, to, color, { size = 12, offsetPx = 0 } = {}) {
  if (!ctx || !from || !to) return;
  const dx = Number(to.x) - Number(from.x);
  const dy = Number(to.y) - Number(from.y);
  const len = Math.hypot(dx, dy);
  if (!(len > 1e-6)) return;
  const ux = dx / len;
  const uy = dy / len;
  const offsetX = -uy * Number(offsetPx || 0);
  const offsetY = ux * Number(offsetPx || 0);
  const arrowX = Number(to.x) + offsetX;
  const arrowY = Number(to.y) + offsetY;
  const leftX = arrowX - ux * size - uy * (size * 0.5);
  const leftY = arrowY - uy * size + ux * (size * 0.5);
  const rightX = arrowX - ux * size + uy * (size * 0.5);
  const rightY = arrowY - uy * size - ux * (size * 0.5);
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(arrowX, arrowY);
  ctx.lineTo(leftX, leftY);
  ctx.lineTo(rightX, rightY);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function pickManeuverDirectionArrowPoints(pointsPx = [], centerPx = null, { laneIndex = 0, laneCount = 1 } = {}) {
  const points = (Array.isArray(pointsPx) ? pointsPx : []).filter(point => Number.isFinite(Number(point?.x)) && Number.isFinite(Number(point?.y)));
  if (points.length < 2) return null;
  const center = centerPx && Number.isFinite(Number(centerPx?.x)) && Number.isFinite(Number(centerPx?.y))
    ? { x: Number(centerPx.x), y: Number(centerPx.y) }
    : { x: 0, y: 0 };
  const total = Math.max(1, Number(laneCount) || 1);
  const idxOffset = total <= 1 ? 0 : (Number(laneIndex) - (total - 1) / 2);
  const minRadiusPx = 86 + (Math.abs(idxOffset) * 6);
  const startFrac = Math.max(0.46, Math.min(0.78, 0.5 + (idxOffset * 0.08)));
  const startIdx = Math.max(1, Math.floor(points.length * startFrac));
  for (let idx = startIdx; idx < points.length; idx++) {
    const point = points[idx];
    const radius = Math.hypot(point.x - center.x, point.y - center.y);
    if (radius < minRadiusPx) continue;
    const fromIdx = Math.max(0, idx - 1);
    if (fromIdx === idx) continue;
    return {
      from: points[fromIdx],
      to: point,
    };
  }
  return {
    from: points[Math.max(0, points.length - 2)],
    to: points[points.length - 1],
  };
}

function getManeuverTimeShift(analysis, key, align = 'anchor') {
  if (align !== 'anchor') return 0;
  const anchorOffset = Number(analysis?.timelines?.anchorOffsetS);
  if (!Number.isFinite(anchorOffset)) return 0;
  return ['sog', 'heading', 'rudder', 'boom', 'trunk', 'com_y'].includes(String(key || '')) ? anchorOffset : 0;
}

function getManeuverTimelineSeries(analysis, key, { align = 'anchor', extraShiftSec = 0 } = {}) {
  const rows = Array.isArray(analysis?.timelines?.[key]) ? analysis.timelines[key] : [];
  const shift = getManeuverTimeShift(analysis, key, align) - (Number.isFinite(Number(extraShiftSec)) ? Number(extraShiftSec) : 0);
  return rows
    .map(point => ({
      t: Number(point?.t) - shift,
      v: Number(point?.v),
    }))
    .filter(point => Number.isFinite(point.t) && Number.isFinite(point.v));
}

function findManeuverPoseFrame(analysis, label) {
  const target = String(label || '').toLowerCase();
  return (analysis?.pose?.keyframes || []).find(frame => String(frame?.label || '').toLowerCase() === target) || null;
}

function drawManeuverLineChart(canvas, seriesList, {
  title = '',
  yLabel = '',
  xLabel = 'Time vs anchor (s)',
  showLegend = true,
  baselineAt = 0,
} = {}) {
  if (!(canvas instanceof HTMLCanvasElement)) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const PAD = { top: 38, right: 22, bottom: 50, left: 60 };
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fcfdff';
  ctx.fillRect(0, 0, W, H);

  const normalized = (Array.isArray(seriesList) ? seriesList : [])
    .map(series => ({
      ...series,
      points: (Array.isArray(series?.points) ? series.points : [])
        .map(point => ({ t: Number(point?.t), v: Number(point?.v) }))
        .filter(point => Number.isFinite(point.t) && Number.isFinite(point.v)),
    }))
    .filter(series => series.points.length > 0);

  if (!normalized.length) {
    ctx.fillStyle = '#73849a';
    ctx.font = '13px "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No synchronized data in this maneuver window', W / 2, H / 2);
    return;
  }

  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const series of normalized) {
    for (const point of series.points) {
      if (point.t < xMin) xMin = point.t;
      if (point.t > xMax) xMax = point.t;
      if (point.v < yMin) yMin = point.v;
      if (point.v > yMax) yMax = point.v;
    }
  }
  if (!(xMax > xMin)) { xMin -= 1; xMax += 1; }
  if (!(yMax > yMin)) { yMin -= 1; yMax += 1; }
  if (Number.isFinite(baselineAt)) {
    yMin = Math.min(yMin, baselineAt);
    yMax = Math.max(yMax, baselineAt);
  }

  const xPad = Math.max(0.8, (xMax - xMin) * 0.08);
  const yPad = Math.max(0.35, (yMax - yMin) * 0.18);
  xMin -= xPad;
  xMax += xPad;
  yMin -= yPad;
  yMax += yPad;

  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;
  const xToPx = value => PAD.left + ((value - xMin) / Math.max(1e-6, xMax - xMin)) * pw;
  const yToPx = value => H - PAD.bottom - ((value - yMin) / Math.max(1e-6, yMax - yMin)) * ph;

  ctx.strokeStyle = '#dce6f1';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = PAD.top + (ph * i) / 5;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();
  }
  for (let i = 0; i <= 6; i++) {
    const x = PAD.left + (pw * i) / 6;
    ctx.beginPath();
    ctx.moveTo(x, PAD.top);
    ctx.lineTo(x, H - PAD.bottom);
    ctx.stroke();
  }

  if (Number.isFinite(Number(baselineAt)) && baselineAt >= yMin && baselineAt <= yMax) {
    ctx.strokeStyle = 'rgba(219, 152, 52, 0.58)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(PAD.left, yToPx(baselineAt));
    ctx.lineTo(W - PAD.right, yToPx(baselineAt));
    ctx.stroke();
  }

  if (0 >= xMin && 0 <= xMax) {
    ctx.strokeStyle = 'rgba(31, 41, 55, 0.55)';
    ctx.lineWidth = 1.6;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(xToPx(0), PAD.top);
    ctx.lineTo(xToPx(0), H - PAD.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#50657b';
    ctx.font = '11px "Segoe UI", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Anchor', Math.min(W - PAD.right - 42, xToPx(0) + 6), PAD.top + 6);
  }

  for (const series of normalized) {
    ctx.strokeStyle = series.color || '#5cc8ff';
    ctx.lineWidth = series.width || 3;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i < series.points.length; i++) {
      const point = series.points[i];
      const x = xToPx(point.t);
      const y = yToPx(point.v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.fillStyle = '#132338';
  ctx.font = '600 12px "Segoe UI", sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(title, PAD.left, 8);
  if (yLabel) {
    ctx.save();
    ctx.translate(14, PAD.top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#61758b';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
  ctx.textAlign = 'center';
  ctx.fillStyle = '#61758b';
  ctx.fillText(xLabel, PAD.left + pw / 2, H - 20);

  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#61758b';
  ctx.fillText(formatManeuverNumber(yMax, { digits: Math.abs(yMax - yMin) < 6 ? 2 : 1, empty: '--' }), PAD.left - 6, PAD.top + 8);
  ctx.fillText(formatManeuverNumber(yMin, { digits: Math.abs(yMax - yMin) < 6 ? 2 : 1, empty: '--' }), PAD.left - 6, H - PAD.bottom - 8);

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 6; i++) {
    const value = xMin + ((xMax - xMin) * i) / 6;
    ctx.fillText(formatManeuverNumber(value, { digits: 1, empty: '--' }), PAD.left + (pw * i) / 6, H - PAD.bottom + 4);
  }

  if (showLegend) {
    let x = PAD.left;
    const y = PAD.top - 14;
    for (const series of normalized) {
      ctx.fillStyle = series.color || '#5cc8ff';
      ctx.fillRect(x, y + 3, 12, 3);
      ctx.fillStyle = '#132338';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(series.label || 'Series', x + 14, y - 3);
      x += Math.max(120, ctx.measureText(series.label || 'Series').width + 32);
    }
  }
}

function drawManeuverOverlayMap(canvas, entries, {
  title = 'Normalized Overlay Map',
  paddingRatioX = 0.42,
  paddingRatioY = 0.46,
  arrowSpread = false,
} = {}) {
  if (!(canvas instanceof HTMLCanvasElement)) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const PAD = 34;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fcfdff';
  ctx.fillRect(0, 0, W, H);

  const normalized = (Array.isArray(entries) ? entries : [])
    .map(entry => ({
      color: entry?.color || '#5cc8ff',
      label: entry?.label || 'Maneuver',
      points: (Array.isArray(entry?.points) ? entry.points : [])
        .map(point => ({ x: Number(point?.x), y: Number(point?.y) }))
        .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y)),
      deepest: entry?.deepest && Number.isFinite(Number(entry.deepest?.x)) && Number.isFinite(Number(entry.deepest?.y))
        ? { x: Number(entry.deepest.x), y: Number(entry.deepest.y) }
        : null,
    }))
    .filter(entry => entry.points.length > 0);

  if (!normalized.length) {
    ctx.fillStyle = '#73849a';
    ctx.font = '13px "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No normalized track preview could be generated for this maneuver window', W / 2, H / 2);
    return;
  }

  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const entry of normalized) {
    for (const point of entry.points) {
      if (point.x < xMin) xMin = point.x;
      if (point.x > xMax) xMax = point.x;
      if (point.y < yMin) yMin = point.y;
      if (point.y > yMax) yMax = point.y;
    }
    if (entry.deepest) {
      xMin = Math.min(xMin, entry.deepest.x);
      xMax = Math.max(xMax, entry.deepest.x);
      yMin = Math.min(yMin, entry.deepest.y);
      yMax = Math.max(yMax, entry.deepest.y);
    }
  }
  xMin = Math.min(xMin, 0);
  xMax = Math.max(xMax, 0);
  yMin = Math.min(yMin, 0);
  yMax = Math.max(yMax, 0);
  if (!(xMax > xMin)) { xMin -= 10; xMax += 10; }
  if (!(yMax > yMin)) { yMin -= 10; yMax += 10; }
  const spanPadX = Math.max(24, (xMax - xMin) * Math.max(0, Number(paddingRatioX) || 0));
  const spanPadY = Math.max(28, (yMax - yMin) * Math.max(0, Number(paddingRatioY) || 0));
  xMin -= spanPadX;
  xMax += spanPadX;
  yMin -= spanPadY;
  yMax += spanPadY;

  const spanX = xMax - xMin;
  const spanY = yMax - yMin;
  const scale = Math.min((W - PAD * 2) / Math.max(1, spanX), (H - PAD * 2) / Math.max(1, spanY));
  const xToPx = value => PAD + (value - xMin) * scale;
  const yToPx = value => H - PAD - (value - yMin) * scale;
  const centerPx = { x: xToPx(0), y: yToPx(0) };

  ctx.strokeStyle = '#dce6f1';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const x = PAD + ((W - PAD * 2) * i) / 5;
    ctx.beginPath();
    ctx.moveTo(x, PAD);
    ctx.lineTo(x, H - PAD);
    ctx.stroke();
  }
  for (let i = 0; i <= 4; i++) {
    const y = PAD + ((H - PAD * 2) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(PAD, y);
    ctx.lineTo(W - PAD, y);
    ctx.stroke();
  }

  if (0 >= xMin && 0 <= xMax) {
    ctx.strokeStyle = 'rgba(31, 41, 55, 0.34)';
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(xToPx(0), PAD);
    ctx.lineTo(xToPx(0), H - PAD);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (0 >= yMin && 0 <= yMax) {
    ctx.strokeStyle = 'rgba(219, 152, 52, 0.34)';
    ctx.beginPath();
    ctx.moveTo(PAD, yToPx(0));
    ctx.lineTo(W - PAD, yToPx(0));
    ctx.stroke();
  }

  for (let entryIdx = 0; entryIdx < normalized.length; entryIdx++) {
    const entry = normalized[entryIdx];
    const pointsPx = entry.points.map(point => ({ x: xToPx(point.x), y: yToPx(point.y) }));
    ctx.strokeStyle = entry.color;
    ctx.lineWidth = 3.2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i < pointsPx.length; i++) {
      const point = pointsPx[i];
      const x = point.x;
      const y = point.y;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const arrow = pickManeuverDirectionArrowPoints(pointsPx, centerPx, {
      laneIndex: entryIdx,
      laneCount: normalized.length,
    });
    const arrowOffsetPx = arrowSpread && normalized.length > 1
      ? (entryIdx - ((normalized.length - 1) / 2)) * 10
      : 0;
    if (arrow) drawManeuverDirectionArrow(ctx, arrow.from, arrow.to, entry.color, { size: 14, offsetPx: arrowOffsetPx });
  }

  ctx.fillStyle = '#132338';
  ctx.beginPath();
  ctx.arc(centerPx.x, centerPx.y, 5.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#132338';
  ctx.font = '600 12px "Segoe UI", sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(title, PAD, 8);
  ctx.fillStyle = '#61758b';
  ctx.fillText('Direction arrows show travel through the maneuver window', PAD, H - 18);

  let legendX = PAD;
  for (const entry of normalized) {
    ctx.fillStyle = entry.color;
    ctx.fillRect(legendX, 24, 12, 3);
    ctx.fillStyle = '#132338';
    ctx.fillText(entry.label, legendX + 18, 16);
    legendX += Math.max(120, ctx.measureText(entry.label).width + 34);
  }
}

function getSkeletonBounds(skeletons = []) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const skeleton of skeletons) {
    for (const point of Object.values(skeleton || {})) {
      const x = Number(point?.[0]);
      const y = Number(point?.[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  if (!(maxX > minX) || !(maxY > minY)) return null;
  return { minX, maxX, minY, maxY };
}

function getManeuverPoseFrames(analysis) {
  const overlayFrames = Array.isArray(analysis?.pose?.overlayFrames) ? analysis.pose.overlayFrames : [];
  const keyframes = Array.isArray(analysis?.pose?.keyframes) ? analysis.pose.keyframes : [];
  return (overlayFrames.length ? overlayFrames : keyframes).filter(frame => frame?.skeleton);
}

function hasManeuverTimelineData(analysis, key, { align = 'anchor', minPoints = 2 } = {}) {
  return getManeuverTimelineSeries(analysis, key, { align }).length >= Math.max(1, Number(minPoints) || 1);
}

function buildDerivedManeuverPosePointClouds(analysis) {
  const frames = getManeuverPoseFrames(analysis);
  const keypointXY = [];
  const comXY = [];
  for (const frame of frames) {
    const skeleton = frame?.skeleton || null;
    if (!skeleton) continue;
    for (const point of Object.values(skeleton)) {
      const x = Number(point?.[0]);
      const y = Number(point?.[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      keypointXY.push([x, y]);
    }
    const com = computeCenterOfMass(skeleton);
    if (Array.isArray(com) && Number.isFinite(Number(com[0])) && Number.isFinite(Number(com[1]))) {
      comXY.push([Number(com[0]), Number(com[1])]);
    }
  }
  return { keypointXY, comXY, frameCount: frames.length };
}

function getManeuverPosePointClouds(analysis) {
  const cachedKeypoints = Array.isArray(analysis?.pose?.keypointXY) ? analysis.pose.keypointXY.filter(pair => Array.isArray(pair) && Number.isFinite(Number(pair[0])) && Number.isFinite(Number(pair[1]))) : [];
  const cachedCom = Array.isArray(analysis?.pose?.comXY) ? analysis.pose.comXY.filter(pair => Array.isArray(pair) && Number.isFinite(Number(pair[0])) && Number.isFinite(Number(pair[1]))) : [];
  if (cachedKeypoints.length || cachedCom.length) {
    return {
      keypointXY: cachedKeypoints,
      comXY: cachedCom,
      frameCount: Number.isFinite(Number(analysis?.pose?.frameCount)) ? Number(analysis.pose.frameCount) : getManeuverPoseFrames(analysis).length,
    };
  }
  return buildDerivedManeuverPosePointClouds(analysis);
}

function hasRenderableManeuverPose(analysis) {
  if (!analysis) return false;
  const clouds = getManeuverPosePointClouds(analysis);
  if (clouds.keypointXY.length || clouds.comXY.length) return true;
  return getManeuverPoseFrames(analysis).length > 0;
}

function hasRenderableManeuverControlPredictions(analysis) {
  if (!analysis) return false;
  const hasRudder = hasManeuverTimelineData(analysis, 'rudder', { align: 'anchor', minPoints: 2 });
  const hasBoom = hasManeuverTimelineData(analysis, 'boom', { align: 'anchor', minPoints: 2 });
  return hasRudder && (!areBoomPredictionsEnabled() || hasBoom);
}

function hasCompleteManeuverProcessing(analysis) {
  return hasRenderableManeuverPose(analysis) && hasRenderableManeuverControlPredictions(analysis);
}

function hasRenderableManeuverGpsOverlay(analysis) {
  const points = Array.isArray(analysis?.gpsOverlay?.points) ? analysis.gpsOverlay.points : [];
  return points.some(point => Number.isFinite(Number(point?.x)) && Number.isFinite(Number(point?.y)));
}

function isManeuverPoseRunActive(fileId) {
  const id = String(fileId || '');
  if (!id) return false;
  return !!(
    PoseEngine.isProcessing(id)
    || _startingSkeletonRuns.has(id)
    || ((_queuedSegmentRuns.get(id) || []).length > 0)
  );
}

function maneuverWorkspaceShouldRefreshForVideo(fileId) {
  const id = String(fileId || '');
  if (!id || !el('view-maneuvers')?.classList.contains('active-view')) return false;
  return getSelectedManeuvers().some(maneuver => (
    findManeuverOverlappingVideos(maneuver, { analyzableOnly: false })
      .some(overlap => String(overlap?.vid?.id || '') === id)
  ));
}

function createManeuverHeatmapJob(pointsXY, color, type = 'keypoints') {
  const points = Array.isArray(pointsXY) ? pointsXY.filter(pair => Array.isArray(pair) && Number.isFinite(Number(pair[0])) && Number.isFinite(Number(pair[1]))) : [];
  const hm = points.length
    ? generateDensityGrid(points, type === 'com'
      ? {
          grid_size_x: 5.0,
          grid_size_y: 3.0,
          grid_center_x: -2.0,
          grid_center_y: 0.0,
          resolution: 140,
          sigma_cells: 2.6,
        }
      : {
          grid_size_x: 5.0,
          grid_size_y: 3.0,
          grid_center_x: -1.5,
          grid_center_y: 0.0,
          resolution: 140,
          sigma_cells: 2.8,
        })
    : null;
  return { hm, points, color };
}

function createManeuverHeatmapTile(title, subtitle, job, renderJobs) {
  const tile = document.createElement('div');
  tile.className = 'maneuver-heatmap-tile';
  if (job?.color) tile.style.setProperty('--maneuver-heatmap-accent', job.color);

  const head = document.createElement('div');
  head.className = 'maneuver-heatmap-head';
  const titleEl = document.createElement('div');
  titleEl.className = 'maneuver-heatmap-title';
  titleEl.textContent = title;
  const subEl = document.createElement('div');
  subEl.className = 'maneuver-heatmap-sub';
  subEl.textContent = subtitle;
  head.append(titleEl, subEl);
  tile.appendChild(head);

  if (!job?.hm) {
    const empty = document.createElement('div');
    empty.className = 'maneuver-heatmap-empty';
    empty.textContent = subtitle || 'Pose data is not available for this maneuver window yet.';
    tile.appendChild(empty);
    return tile;
  }

  const wrap = document.createElement('div');
  wrap.className = 'maneuver-heatmap-canvas-wrap';
  const canvas = document.createElement('canvas');
  canvas.width = 960;
  canvas.height = 700;
  wrap.appendChild(canvas);
  tile.appendChild(wrap);
  renderJobs.push({
    canvas,
    wrap,
    hm: job.hm,
    points: job.points,
    color: job.color,
    frustumSize: 4.8,
    lightTheme: true,
  });
  return tile;
}

function queueManeuverHeatmapRender(renderJobs = []) {
  if (!renderJobs.length) return;
  void (async () => {
    let deps = null;
    try {
      deps = await loadHeatmapRenderDeps();
    } catch (err) {
      console.warn('[maneuvers] heatmap render deps unavailable:', err);
    }
    for (const job of renderJobs) {
      try {
        if (!deps) throw new Error('Heatmap render deps unavailable');
        await renderHeatmapWithBoat(job.canvas, job, job.frustumSize, deps, {
          backgroundColor: 0xf7fafc,
          gridColor: 0xdbe6f0,
          ambientIntensity: 0.9,
          directionalIntensity: 0.48,
          boatColor: 0xfdfefe,
          boatEmissive: 0xe4edf6,
          boatEmissiveIntensity: 0.22,
          boatOpacity: 0.96,
        });
      } catch (err) {
        console.warn('[maneuvers] heatmap render failed, using fallback image:', err);
        try { await renderHeatmapFallbackImage(job.wrap, job.canvas, job); } catch {}
      }
      await nextAnimationFrame();
    }
  })();
}

function toTopDownPosePoint(point) {
  const foreAft = Number(point?.[0]);
  const lateral = Number(point?.[1]);
  if (!Number.isFinite(foreAft) || !Number.isFinite(lateral)) return null;
  return { x: lateral, y: -foreAft };
}

function getTopDownPoseBounds(seriesList = []) {
  const bounds = {
    minX: -1.45,
    maxX: 1.45,
    minY: -0.15,
    maxY: 3.45,
  };
  for (const series of (Array.isArray(seriesList) ? seriesList : [])) {
    for (const frame of (Array.isArray(series?.frames) ? series.frames : [])) {
      for (const point of Object.values(frame?.skeleton || {})) {
        const plan = toTopDownPosePoint(point);
        if (!plan) continue;
        bounds.minX = Math.min(bounds.minX, plan.x);
        bounds.maxX = Math.max(bounds.maxX, plan.x);
        bounds.minY = Math.min(bounds.minY, plan.y);
        bounds.maxY = Math.max(bounds.maxY, plan.y);
      }
    }
  }
  return bounds;
}

function drawTopDownSkeletonFrame(ctx, skeleton, project, color, alpha = 0.22, lineWidth = 1.5) {
  if (!ctx || !skeleton) return;
  ctx.save();
  ctx.globalAlpha = Math.max(0.04, Math.min(1, alpha));
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const [aIdx, bIdx] of SKEL_CONNECTIONS) {
    const a = toTopDownPosePoint(skeleton?.[aIdx] ?? skeleton?.[String(aIdx)]);
    const b = toTopDownPosePoint(skeleton?.[bIdx] ?? skeleton?.[String(bIdx)]);
    if (!a || !b) continue;
    const aPx = project(a);
    const bPx = project(b);
    ctx.beginPath();
    ctx.moveTo(aPx.x, aPx.y);
    ctx.lineTo(bPx.x, bPx.y);
    ctx.stroke();
  }
  for (const point of Object.values(skeleton || {})) {
    const plan = toTopDownPosePoint(point);
    if (!plan) continue;
    const px = project(plan);
    ctx.beginPath();
    ctx.arc(px.x, px.y, Math.max(1.8, lineWidth * 0.95), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawBoatTopDownPoseOverlay(canvas, seriesEntries, { title = 'Boat-Top Pose Overlay' } = {}) {
  if (!(canvas instanceof HTMLCanvasElement)) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const PAD = 26;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const normalized = (Array.isArray(seriesEntries) ? seriesEntries : [])
    .map(entry => ({
      color: entry?.color || '#2563eb',
      label: entry?.label || 'Maneuver',
      frames: (Array.isArray(entry?.frames) ? entry.frames : []).filter(frame => frame?.skeleton),
    }))
    .filter(entry => entry.frames.length > 0);

  if (!normalized.length) {
    ctx.fillStyle = '#73849a';
    ctx.font = '13px "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No pose frames are available for this maneuver window', W / 2, H / 2);
    return;
  }

  const bounds = getTopDownPoseBounds(normalized);
  const spanX = Math.max(1, bounds.maxX - bounds.minX);
  const spanY = Math.max(1, bounds.maxY - bounds.minY);
  const scale = Math.min((W - PAD * 2) / spanX, (H - PAD * 2) / spanY);
  const xToPx = value => PAD + (value - bounds.minX) * scale;
  const yToPx = value => H - PAD - (value - bounds.minY) * scale;
  const project = point => ({ x: xToPx(point.x), y: yToPx(point.y) });

  ctx.strokeStyle = '#dde6f0';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = PAD + ((W - PAD * 2) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, PAD);
    ctx.lineTo(x, H - PAD);
    ctx.stroke();
  }
  for (let i = 0; i <= 5; i++) {
    const y = PAD + ((H - PAD * 2) * i) / 5;
    ctx.beginPath();
    ctx.moveTo(PAD, y);
    ctx.lineTo(W - PAD, y);
    ctx.stroke();
  }

  const hull = [
    { x: 0, y: 3.48 },
    { x: 0.44, y: 3.18 },
    { x: 0.82, y: 2.1 },
    { x: 1.06, y: 0.56 },
    { x: 0.7, y: -0.02 },
    { x: 0, y: -0.18 },
    { x: -0.7, y: -0.02 },
    { x: -1.06, y: 0.56 },
    { x: -0.82, y: 2.1 },
    { x: -0.44, y: 3.18 },
  ];
  ctx.fillStyle = 'rgba(237, 243, 250, 0.95)';
  ctx.strokeStyle = '#c5d2df';
  ctx.lineWidth = 2;
  ctx.beginPath();
  hull.forEach((point, idx) => {
    const px = project(point);
    if (idx === 0) ctx.moveTo(px.x, px.y);
    else ctx.lineTo(px.x, px.y);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.strokeStyle = 'rgba(19, 35, 56, 0.24)';
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  const stern = project({ x: 0, y: -0.15 });
  const bow = project({ x: 0, y: 3.45 });
  ctx.moveTo(stern.x, stern.y);
  ctx.lineTo(bow.x, bow.y);
  ctx.stroke();
  ctx.setLineDash([]);

  for (const entry of normalized) {
    const maxRel = Math.max(0.25, ...entry.frames.map(frame => Math.abs(Number(frame?.relS) || 0)));
    for (const frame of entry.frames) {
      const closeness = 1 - Math.min(1, Math.abs(Number(frame?.relS) || 0) / maxRel);
      drawTopDownSkeletonFrame(ctx, frame.skeleton, project, entry.color, 0.07 + closeness * 0.22, 1.35);
    }
    const anchorFrame = entry.frames.reduce((best, frame) => (
      !best || Math.abs(Number(frame?.relS) || 0) < Math.abs(Number(best?.relS) || 0)
        ? frame
        : best
    ), null);
    if (anchorFrame?.skeleton) drawTopDownSkeletonFrame(ctx, anchorFrame.skeleton, project, entry.color, 0.82, 2.35);
  }

  ctx.fillStyle = '#132338';
  ctx.font = '12px "Segoe UI", sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(title, PAD, 8);
  ctx.fillStyle = '#61758b';
  ctx.fillText('Placed skeletons overlaid in boat coordinates for the full maneuver window', PAD, H - 18);

  let legendX = PAD;
  for (const entry of normalized) {
    ctx.fillStyle = entry.color;
    ctx.fillRect(legendX, 24, 12, 3);
    ctx.fillStyle = '#132338';
    ctx.fillText(`${entry.label} · ${entry.frames.length} frames`, legendX + 18, 16);
    legendX += Math.max(160, ctx.measureText(`${entry.label} · ${entry.frames.length} frames`).width + 34);
  }
}

function drawSkeletonIntoBox(ctx, skeleton, bounds, box, color) {
  if (!skeleton || !bounds) return;
  const pad = 12;
  const sx = (box.w - pad * 2) / Math.max(1e-6, bounds.maxX - bounds.minX);
  const sy = (box.h - pad * 2) / Math.max(1e-6, bounds.maxY - bounds.minY);
  const scale = Math.min(sx, sy);
  const xOffset = box.x + (box.w - (bounds.maxX - bounds.minX) * scale) / 2;
  const yOffset = box.y + (box.h - (bounds.maxY - bounds.minY) * scale) / 2;
  const toPx = (point) => ({
    x: xOffset + (Number(point?.[0]) - bounds.minX) * scale,
    y: yOffset + (Number(point?.[1]) - bounds.minY) * scale,
  });

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.9;
  for (const [aIdx, bIdx] of SKEL_CONNECTIONS) {
    const a = skeleton?.[aIdx] ?? skeleton?.[String(aIdx)];
    const b = skeleton?.[bIdx] ?? skeleton?.[String(bIdx)];
    if (!a || !b) continue;
    const aPx = toPx(a);
    const bPx = toPx(b);
    ctx.beginPath();
    ctx.moveTo(aPx.x, aPx.y);
    ctx.lineTo(bPx.x, bPx.y);
    ctx.stroke();
  }
  ctx.fillStyle = color;
  for (const point of Object.values(skeleton || {})) {
    const x = Number(point?.[0]);
    const y = Number(point?.[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const px = toPx(point);
    ctx.beginPath();
    ctx.arc(px.x, px.y, 2.6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawPoseStrip(canvas, keyframes = [], color = '#5cc8ff') {
  if (!(canvas instanceof HTMLCanvasElement)) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b1420';
  ctx.fillRect(0, 0, W, H);

  const frames = (Array.isArray(keyframes) ? keyframes : []).filter(frame => frame?.skeleton);
  if (!frames.length) {
    ctx.fillStyle = '#9fb0c2';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No pose keyframes available', W / 2, H / 2);
    return;
  }

  const bounds = getSkeletonBounds(frames.map(frame => frame.skeleton));
  const cellW = W / frames.length;
  frames.forEach((frame, idx) => {
    const box = { x: idx * cellW + 6, y: 18, w: cellW - 12, h: H - 42 };
    drawSkeletonIntoBox(ctx, frame.skeleton, bounds, box, color);
    ctx.fillStyle = '#dce8f5';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${frame.label || 'Frame'} ${formatManeuverNumber(frame.relS, { digits: 1, suffix: 's', signed: true })}`, box.x + box.w / 2, H - 8);
  });
}

function drawPoseOverlay(canvas, overlays = []) {
  if (!(canvas instanceof HTMLCanvasElement)) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b1420';
  ctx.fillRect(0, 0, W, H);

  const frames = (Array.isArray(overlays) ? overlays : []).filter(item => item?.frame?.skeleton);
  if (!frames.length) {
    ctx.fillStyle = '#9fb0c2';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Pose overlay unavailable', W / 2, H / 2);
    return;
  }

  const bounds = getSkeletonBounds(frames.map(item => item.frame.skeleton));
  const box = { x: 18, y: 28, w: W - 36, h: H - 60 };
  for (const item of frames) {
    drawSkeletonIntoBox(ctx, item.frame.skeleton, bounds, box, item.color || '#5cc8ff');
  }

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.font = '13px sans-serif';
  let x = 18;
  for (const item of frames) {
    ctx.fillStyle = item.color || '#5cc8ff';
    ctx.fillRect(x, 10, 12, 3);
    ctx.fillStyle = '#dce8f5';
    ctx.fillText(item.label || 'Pose', x + 18, 2);
    x += Math.max(130, ctx.measureText(item.label || 'Pose').width + 36);
  }
}

async function createManeuverVideoCard(label, maneuver, analysis, color, {
  showOffsetSlider = false,
  initialOffsetSec = 0,
  onOffsetChange = null,
} = {}) {
  const card = document.createElement('div');
  card.className = 'maneuver-video-card';
  const meta = document.createElement('div');
  meta.className = 'maneuver-video-meta';

  const title = document.createElement('div');
  title.className = 'maneuver-video-title';
  title.textContent = label;

  const analyzableOverlaps = findManeuverOverlappingVideos(maneuver, { analyzableOnly: true });
  const playbackOverlaps = analyzableOverlaps.length
    ? analyzableOverlaps
    : findManeuverOverlappingVideos(maneuver, { analyzableOnly: false });
  const overlaps = playbackOverlaps;
  const fallback = overlaps[0] || null;
  const analysisFileId = String(analysis?.mediaRefs?.primaryVideoFileId || '');
  const candidateIds = [];
  if (analysisFileId) candidateIds.push(analysisFileId);
  for (const row of overlaps) {
    const id = String(row?.vid?.id || '');
    if (id) candidateIds.push(id);
  }
  const uniqueCandidateIds = [...new Set(candidateIds)];
  if (!uniqueCandidateIds.length) {
    meta.append(
      title,
      document.createTextNode('No athlete video is available for this maneuver window.'),
    );
    card.appendChild(meta);
    return { card, videoEl: null, anchorSec: null, durationSec: null };
  }

  const videoEl = document.createElement('video');
  videoEl.controls = true;
  videoEl.muted = true;
  videoEl.playsInline = true;
  videoEl.preload = 'metadata';
  let selectedOverlap = fallback;
  let videoUrl = '';
  for (const candidateId of uniqueCandidateIds) {
    try {
      videoUrl = await getVideoURL(candidateId);
      selectedOverlap = overlaps.find(row => String(row?.vid?.id || '') === String(candidateId)) || fallback;
      break;
    } catch {}
  }
  if (!videoUrl) {
    meta.append(title, document.createTextNode('The video file could not be opened in this browser session.'));
    card.appendChild(meta);
    return { card, videoEl: null, anchorSec: null, durationSec: null };
  }
  videoEl.src = videoUrl;

  const anchorSecRaw = Number(analysis?.mediaRefs?.videoAnchorSec);
  const startSecRaw = Number(analysis?.mediaRefs?.videoStartSec ?? selectedOverlap?.videoStartSec);
  const endSecRaw = Number(analysis?.mediaRefs?.videoEndSec ?? selectedOverlap?.videoEndSec);
  const anchorSec = Number.isFinite(anchorSecRaw) ? anchorSecRaw : (Number.isFinite(startSecRaw) ? startSecRaw : 0);
  const initialSeekSecRaw = Number.isFinite(startSecRaw)
    ? Math.max(0, startSecRaw - 1.0)
    : anchorSec;
  const durationSec = Number.isFinite(endSecRaw) && Number.isFinite(startSecRaw)
    ? Math.max(0, endSecRaw - startSecRaw)
    : Number(analysis?.analysisWindow?.duration_s);

  videoEl.addEventListener('loadedmetadata', () => {
    const seekTo = Math.max(0, Math.min(Number(videoEl.duration) || Infinity, initialSeekSecRaw));
    try { videoEl.currentTime = seekTo; } catch {}
  }, { once: true });

  const replayBtn = document.createElement('button');
  replayBtn.className = 'btn sm';
  replayBtn.textContent = 'Replay from Start';
  replayBtn.style.marginLeft = '8px';
  replayBtn.onclick = () => {
    if (Number.isFinite(startSecRaw)) {
      videoEl.currentTime = startSecRaw;
      videoEl.play().catch(() => {});
    }
  };

  const cardObj = {
    card,
    videoEl,
    anchorSec,
    durationSec,
    offsetSec: Number.isFinite(Number(initialOffsetSec)) ? Number(initialOffsetSec) : 0,
  };

  const controlsRow = document.createElement('div');
  controlsRow.style.marginTop = '6px';
  controlsRow.append(replayBtn);

  if (showOffsetSlider) {
    const sliderWrap = document.createElement('details');
    sliderWrap.className = 'maneuver-offset-menu';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = -20;
    slider.max = 20;
    slider.step = 0.1;
    slider.value = String(cardObj.offsetSec);

    const sliderLbl = document.createElement('span');
    sliderLbl.textContent = `Offset: ${cardObj.offsetSec > 0 ? '+' : ''}${cardObj.offsetSec.toFixed(1)}s`;
    const summary = document.createElement('summary');
    summary.textContent = sliderLbl.textContent;

    slider.oninput = () => {
      cardObj.offsetSec = Number(slider.value);
      sliderLbl.textContent = `Offset: ${cardObj.offsetSec > 0 ? '+' : ''}${cardObj.offsetSec.toFixed(1)}s`;
      summary.textContent = sliderLbl.textContent;
      try {
        const seekEvt = new Event('seeked');
        videoEl.dispatchEvent(seekEvt);
      } catch {}
      if (typeof onOffsetChange === 'function') onOffsetChange(cardObj.offsetSec, false);
    };
    slider.onchange = () => {
      if (typeof onOffsetChange === 'function') onOffsetChange(cardObj.offsetSec, true);
    };

    sliderWrap.append(summary, slider, sliderLbl);
    controlsRow.appendChild(sliderWrap);
  }

  meta.append(
    title,
    document.createTextNode(`${maneuver?.athlete_name || 'Athlete'} · ${getManeuverTypeLabel(maneuver)} · anchor ${formatManeuverNumber(anchorSec, { digits: 1, suffix: 's' })}`),
    controlsRow
  );

  if (Number.isFinite(durationSec)) {
    const detail = document.createElement('div');
    detail.textContent = `Window ${formatManeuverDuration(durationSec)}`;
    meta.appendChild(detail);
  }
  meta.style.borderTop = `2px solid ${color}`;
  card.append(videoEl, meta);
  return cardObj;
}

function syncManeuverVideoEntries(entries = []) {
  const videos = (Array.isArray(entries) ? entries : []).filter(item => item?.videoEl && Number.isFinite(Number(item?.anchorSec)));
  if (!videos.length) return null;
  let syncing = false;
  let driftFrame = 0;
  let lastDriftAt = 0;
  let masterEntry = videos[0] || null;
  const syncFrom = (source, mode = 'seek') => {
    if (syncing) return;
    const sourceEntry = videos.find(entry => entry.videoEl === source);
    if (!sourceEntry) return;
    masterEntry = sourceEntry;
    syncing = true;
    try {
      const rel = (Number(source.currentTime) || 0) - (Number(sourceEntry.anchorSec) + Number(sourceEntry.offsetSec || 0));
      for (const entry of videos) {
        if (entry.videoEl === source) continue;
        const target = Math.max(0, (Number(entry.anchorSec) || 0) + Number(entry.offsetSec || 0) + rel);
        const threshold = mode === 'drift' ? 0.24 : 0.14;
        if (Math.abs((Number(entry.videoEl.currentTime) || 0) - target) > threshold) {
          try { entry.videoEl.currentTime = target; } catch {}
        }
        if (mode === 'play') entry.videoEl.play().catch(() => {});
        if (mode === 'pause') entry.videoEl.pause();
      }
    } finally {
      setTimeout(() => { syncing = false; }, 0);
    }
  };

  const stopDriftSync = () => {
    if (!driftFrame) return;
    cancelAnimationFrame(driftFrame);
    driftFrame = 0;
  };

  const tickDriftSync = (now = performance.now()) => {
    const active = masterEntry && !masterEntry.videoEl.paused
      ? masterEntry
      : videos.find(entry => entry?.videoEl && !entry.videoEl.paused);
    if (!active) {
      stopDriftSync();
      return;
    }
    masterEntry = active;
    if (!lastDriftAt || now - lastDriftAt > 260) {
      lastDriftAt = now;
      syncFrom(active.videoEl, 'drift');
    }
    driftFrame = requestAnimationFrame(tickDriftSync);
  };

  const startDriftSync = sourceEntry => {
    if (sourceEntry) masterEntry = sourceEntry;
    if (!driftFrame) driftFrame = requestAnimationFrame(tickDriftSync);
  };

  for (const entry of videos) {
    entry.videoEl.addEventListener('seeked', () => syncFrom(entry.videoEl, 'seek'));
    entry.videoEl.addEventListener('play', () => {
      syncFrom(entry.videoEl, 'play');
      startDriftSync(entry);
    });
    entry.videoEl.addEventListener('pause', () => {
      syncFrom(entry.videoEl, 'pause');
      if (!videos.some(item => item?.videoEl && !item.videoEl.paused)) stopDriftSync();
    });
  }

  return {
    entries: videos,
    playAll() {
      const source = masterEntry || videos[0];
      if (!source?.videoEl) return;
      syncFrom(source.videoEl, 'play');
      source.videoEl.play().catch(() => {});
      startDriftSync(source);
    },
    pauseAll() {
      for (const entry of videos) {
        try { entry.videoEl.pause(); } catch {}
      }
      stopDriftSync();
    },
    replayAll() {
      const source = masterEntry || videos[0];
      const relStart = -Math.min(1, Number(source?.anchorSec) || 0);
      syncing = true;
      try {
        for (const entry of videos) {
          const target = Math.max(0, (Number(entry.anchorSec) || 0) + Number(entry.offsetSec || 0) + relStart);
          try { entry.videoEl.currentTime = target; } catch {}
        }
      } finally {
        syncing = false;
      }
      this.playAll();
    },
  };
}

function buildManeuverStatsBlock(title, stats) {
  const block = document.createElement('div');
  block.className = 'maneuver-stats-block';
  const heading = document.createElement('h4');
  heading.textContent = title;
  block.appendChild(heading);
  const phaseKey = String(title || '').trim().toLowerCase();
  const rows = phaseKey === 'before'
    ? [
        ['Avg SOG', formatManeuverNumber(stats?.avgSogKts, { digits: 1, suffix: ' kt' })],
        ['Avg VMG', formatManeuverNumber(stats?.avgVmgKts, { digits: 1, suffix: ' kt' })],
        ['Abs TWA', formatManeuverNumber(stats?.avgAbsTwaDeg, { digits: 1, suffix: '°' })],
        ['COG Δ', formatManeuverNumber(stats?.headingDeltaDeg, { digits: 1, suffix: '°' })],
        ['Avg RA', formatManeuverNumber(stats?.avgRudderDeg, { digits: 1, suffix: '°' })],
        ['Avg BA', formatManeuverNumber(stats?.avgBoomDeg, { digits: 1, suffix: '°' })],
        ['Entry speed', formatManeuverNumber(stats?.entrySpeedKts, { digits: 1, suffix: ' kt' })],
      ]
    : phaseKey === 'during'
      ? [
          ['Duration', formatManeuverNumber(stats?.durationS, { digits: 1, suffix: ' s' })],
          ['Avg SOG', formatManeuverNumber(stats?.avgSogKts, { digits: 1, suffix: ' kt' })],
          ['Avg VMG', formatManeuverNumber(stats?.avgVmgKts, { digits: 1, suffix: ' kt' })],
          ['Abs TWA', formatManeuverNumber(stats?.avgAbsTwaDeg, { digits: 1, suffix: '°' })],
          ['COG Δ', formatManeuverNumber(stats?.headingDeltaDeg, { digits: 1, suffix: '°' })],
          ['Avg RA', formatManeuverNumber(stats?.avgRudderDeg, { digits: 1, suffix: '°' })],
          ['Avg BA', formatManeuverNumber(stats?.avgBoomDeg, { digits: 1, suffix: '°' })],
          ['Min speed', formatManeuverNumber(stats?.minSpeedKts, { digits: 1, suffix: ' kt' })],
        ]
      : [
          ['Avg SOG', formatManeuverNumber(stats?.avgSogKts, { digits: 1, suffix: ' kt' })],
          ['Avg VMG', formatManeuverNumber(stats?.avgVmgKts, { digits: 1, suffix: ' kt' })],
          ['Abs TWA', formatManeuverNumber(stats?.avgAbsTwaDeg, { digits: 1, suffix: '°' })],
          ['COG Δ', formatManeuverNumber(stats?.headingDeltaDeg, { digits: 1, suffix: '°' })],
          ['Avg RA', formatManeuverNumber(stats?.avgRudderDeg, { digits: 1, suffix: '°' })],
          ['Avg BA', formatManeuverNumber(stats?.avgBoomDeg, { digits: 1, suffix: '°' })],
          ['Exit speed', formatManeuverNumber(stats?.exitSpeedKts, { digits: 1, suffix: ' kt' })],
        ];
  for (const [key, value] of rows) {
    const row = document.createElement('div');
    row.className = 'maneuver-stats-row';
    const keyEl = document.createElement('span');
    keyEl.className = 'key';
    keyEl.textContent = key;
    const valueEl = document.createElement('span');
    valueEl.textContent = value;
    row.append(keyEl, valueEl);
    block.appendChild(row);
  }
  return block;
}

function buildManeuverHero(maneuver, color, suffix = '') {
  const hero = document.createElement('div');
  hero.className = 'maneuver-hero';
  hero.style.background = String(maneuver?.type) === 'jibe'
    ? 'linear-gradient(135deg, rgba(145,69,29,.96), rgba(45,21,9,.98))'
    : 'linear-gradient(135deg, rgba(21,61,109,.96), rgba(10,27,43,.98))';
  const chip = document.createElement('div');
  chip.className = 'maneuver-hero-chip';
  chip.style.color = '#082032';
  chip.textContent = String(maneuver?.type || '').startsWith('j') ? 'J' : 'T';
  const text = document.createElement('div');
  const title = document.createElement('div');
  title.className = 'maneuver-hero-title';
  title.textContent = `${maneuver?.athlete_name || 'Athlete'} · ${getManeuverTypeLabel(maneuver)}${suffix ? ` ${suffix}` : ''}`;
  const sub = document.createElement('div');
  sub.className = 'maneuver-hero-sub';
  sub.textContent = `${fmtClock(maneuver?.anchor_ts)} · ${maneuver?.side_from || '?'} → ${maneuver?.side_to || '?'} · ${formatManeuverDuration(Number(maneuver?.end_ts) - Number(maneuver?.start_ts))}`;
  text.append(title, sub);
  hero.append(chip, text);
  if (color) hero.style.borderColor = color;
  return hero;
}

function buildManeuverSummaryGrid(maneuver) {
  const grid = document.createElement('div');
  grid.className = 'maneuver-summary-grid';
  const cards = [
    {
      label: 'Entry / Exit',
      value: `${formatManeuverNumber(maneuver?.duringStats?.entrySpeedKts, { digits: 1, suffix: ' kt' })} → ${formatManeuverNumber(maneuver?.duringStats?.exitSpeedKts, { digits: 1, suffix: ' kt' })}`,
      sub: `Min ${formatManeuverNumber(maneuver?.duringStats?.minSpeedKts, { digits: 1, suffix: ' kt' })}`,
    },
    {
      label: 'COG Delta',
      value: formatManeuverNumber(maneuver?.heading_delta_deg, { digits: 1, suffix: '°' }),
      sub: `${maneuver?.side_from || '?'} to ${maneuver?.side_to || '?'}`,
    },
    {
      label: 'Recovery',
      value: formatManeuverNumber(maneuver?.duringStats?.speedRecoveryTimeS, { digits: 1, suffix: ' s' }),
      sub: `Window ${formatManeuverDuration(maneuver?.duringStats?.durationS || maneuver?.duration_s)}`,
    },
    {
      label: 'Wind',
      value: formatManeuverNumber(maneuver?.sourceWind?.directionDeg, { digits: 0, suffix: '°' }),
      sub: `${formatManeuverNumber(maneuver?.sourceWind?.directionDeg, { digits: 0, suffix: '°' })} · ${formatManeuverNumber(maneuver?.sourceWind?.speedKts, { digits: 1, suffix: ' kt' })}`,
    },
  ];
  for (const item of cards) {
    const card = document.createElement('div');
    card.className = 'maneuver-summary-card';
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = item.label;
    const value = document.createElement('div');
    value.className = 'value';
    value.textContent = item.value;
    const sub = document.createElement('div');
    sub.className = 'sub';
    sub.textContent = item.sub;
    card.append(label, value, sub);
    grid.appendChild(card);
  }
  return grid;
}

function buildManeuverBrief(maneuver, { color = getManeuverColor(maneuver), prefix = '' } = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'maneuver-brief';
  const durationLabel = formatManeuverDuration(
    maneuver?.duringStats?.durationS ?? maneuver?.duration_s ?? (Number(maneuver?.end_ts) - Number(maneuver?.start_ts))
  );

  const main = document.createElement('div');
  main.className = 'maneuver-brief-main';
  const title = document.createElement('div');
  title.className = 'maneuver-brief-title';
  title.textContent = `${prefix ? `${prefix} · ` : ''}${maneuver?.athlete_name || 'Athlete'} ${getManeuverTypeLabel(maneuver)}`;
  const meta = document.createElement('div');
  meta.className = 'maneuver-brief-meta';
  meta.textContent = [
    fmtClock(maneuver?.anchor_ts),
    `${maneuver?.side_from || '?'} → ${maneuver?.side_to || '?'}`,
    durationLabel,
  ].filter(Boolean).join(' · ');
  main.append(title, meta);

  const chip = document.createElement('div');
  chip.className = 'maneuver-brief-chip';
  const typeChip = document.createElement('span');
  typeChip.className = `mnv-type-chip ${String(maneuver?.type || '')}`;
  typeChip.style.background = color;
  typeChip.textContent = String(maneuver?.type || '').startsWith('j') ? 'J' : 'T';
  const label = document.createElement('span');
  label.textContent = maneuver?.sourceWind?.directionDeg != null
    ? `Wind ${formatManeuverNumber(maneuver?.sourceWind?.directionDeg, { digits: 0, suffix: '°' })}`
    : 'Maneuver';
  chip.append(typeChip, label);

  wrap.append(main, chip);
  return wrap;
}

function buildManeuverSelectionState() {
  const selected = getSelectedManeuvers();
  if (selected.length >= 2 && selected.every(move => String(move?.type || '') === String(selected[0]?.type || ''))) {
    return { kind: 'compare', selected, comparable: selected, reason: null };
  }
  if (selected.length === 1) {
    return { kind: 'inspect', selected, comparable: null, reason: null };
  }
  if (selected.length === 0) {
    return { kind: 'empty', selected, comparable: null, reason: 'Click a maneuver marker for individual stats. Press Compare to pick markers for comparison, then Finish to process and open the columns.' };
  }
  if (selected.length >= 2) {
    return { kind: 'disabled', selected, comparable: null, reason: 'Compare is only available when all selected maneuvers are the same type.' };
  }
  return { kind: 'disabled', selected, comparable: null, reason: 'Select maneuvers of the same type to compare.' };
}

async function invalidateStaleManeuverAnalyses(nextManeuvers = []) {
  if (!state.projectId) return new Set();
  const rows = await DB.db.maneuverAnalyses.where('project_id').equals(state.projectId).toArray();
  const nextById = new Map((Array.isArray(nextManeuvers) ? nextManeuvers : []).map(move => [String(move.id), move]));
  const validIds = new Set();
  for (const row of rows) {
    const id = String(row?.maneuver_id || '');
    const move = nextById.get(id);
    const startTs = Number(row?.analysisWindow?.start_ts);
    const anchorTs = Number(row?.analysisWindow?.anchor_ts);
    const endTs = Number(row?.analysisWindow?.end_ts);
    const isValid = !!move
      && Number(row?.schema_version) >= MANEUVER_ANALYSIS_SCHEMA_VERSION
      && Math.abs((Number(move.start_ts) || 0) - startTs) <= 0.5
      && Math.abs((Number(move.anchor_ts) || 0) - anchorTs) <= 0.5
      && Math.abs((Number(move.end_ts) || 0) - endTs) <= 0.5
      && String(row?.maneuver?.type || move.type || '') === String(move.type || '');
    if (isValid) {
      validIds.add(id);
    } else if (id) {
      await DB.db.maneuverAnalyses.delete(id);
    }
  }
  return validIds;
}

async function ensureManeuverAnalysisReady(maneuver, { force = false } = {}) {
  if (!state.projectId || !maneuver?.id) return null;
  const bestVideo = pickBestManeuverVideo(maneuver, { analyzableOnly: true });
  const cached = force ? null : await DB.getManeuverAnalysis(state.projectId, maneuver.id);
  const cachedReady = !!cached
    && Number(cached?.schema_version) >= MANEUVER_ANALYSIS_SCHEMA_VERSION
    && hasRenderableManeuverGpsOverlay(cached)
    && (!bestVideo?.vid || hasCompleteManeuverProcessing(cached));
  if (cachedReady) return cached;
  const rebuildCachedAnalysis = !!cached && !cachedReady;
  if (!bestVideo?.vid) return null;
  if (bestVideo?.vid) {
    const cachedNeedsControlPredictions = !!cached
      && hasRenderableManeuverPose(cached)
      && !hasRenderableManeuverControlPredictions(cached);
    const poseWindowPadSec = Math.max(4, Number(maneuver?.duration_s || 0) * 0.3);
    const coverageStartSec = Math.max(0, Number(bestVideo.videoStartSec) - poseWindowPadSec);
    const durationLimit = Number.isFinite(Number(bestVideo.vid?.duration_sec)) ? Number(bestVideo.vid.duration_sec) : null;
    let coverageEndSec = Number.isFinite(Number(bestVideo.videoEndSec)) ? Number(bestVideo.videoEndSec) + poseWindowPadSec : coverageStartSec + Math.max(6, Number(maneuver?.duration_s || 0) + poseWindowPadSec);
    if (Number.isFinite(durationLimit)) coverageEndSec = Math.min(durationLimit, coverageEndSec);
    const coverage = state.skeletonCoverage?.[bestVideo.vid.id] || [];
    const coverageToleranceSec = 1.25;
    const coverageRatio = intervalCoverageRatio(coverage, coverageStartSec, coverageEndSec, coverageToleranceSec);
    const poseCoverageReady = intervalCoversRange(coverage, coverageStartSec, coverageEndSec, coverageToleranceSec)
      || coverageRatio >= 0.86;
    const poseProcessingActive = isManeuverPoseRunActive(bestVideo.vid.id)
      || hasQueuedSegmentOverlap(bestVideo.vid.id, coverageStartSec, coverageEndSec, coverageToleranceSec);
    if (poseProcessingActive && cached && hasRenderableManeuverGpsOverlay(cached)) {
      return cached;
    }
    if ((!poseCoverageReady || cachedNeedsControlPredictions) && !poseProcessingActive) {
      try {
        await runSkeletonForVideo(bestVideo.vid.id, {
          startSec: coverageStartSec,
          endSec: coverageEndSec,
          segmentName: `${getManeuverTypeLabel(maneuver)} ${maneuver?.athlete_name || 'Maneuver'}`,
          fps: getManeuverPoseFps(),
        });
        await loadAllSkeletonCoverages();
      } catch (err) {
        console.warn('[maneuver] bounded pose processing failed, continuing with available data:', err);
      }
    }
  }

  const analysis = await buildManeuverAnalysis(state.projectId, maneuver.id, {
    force: force || rebuildCachedAnalysis,
    windContext: buildReportWindContext(),
  });
  if (!analysis) return null;

  let touched = false;
  state.maneuvers = (state.maneuvers || []).map(move => {
    if (String(move?.id) !== String(maneuver.id)) return move;
    if (move.deepReady) return move;
    touched = true;
    return { ...move, deepReady: true };
  });
  if (touched) {
    await DB.saveManeuvers(state.projectId, state.maneuvers);
    renderManeuverPanel();
  }
  return analysis;
}

async function createSegmentFromManeuver(maneuver) {
  if (!state.projectId || !maneuver) return;
  const baseName = `${maneuver?.athlete_name || 'Athlete'} ${getManeuverTypeLabel(maneuver)} ${new Date(Number(maneuver?.anchor_ts || 0) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
  const segment = {
    id: DB.uuid(),
    name: baseName,
    tsStart: Number(maneuver.start_ts),
    tsEnd: Number(maneuver.end_ts),
  };
  state.segments = [...(state.segments || []), segment].sort((a, b) => Number(a.tsStart || 0) - Number(b.tsStart || 0));
  await saveSegments();
  renderSegmentPanel();
  renderAnalysisTab();
  renderMap();
  drawSogCanvas();
}

async function buildSingleManeuverContent(maneuver, analysis) {
  const color = getManeuverColor(maneuver);
  const fragment = document.createDocumentFragment();
  const heatmapJobs = [];
  const overlayGeometry = await resolveManeuverOverlayGeometry(maneuver, analysis);

  const mapSec = createManeuverSection('Maneuver Map', 'Local-wind normalized GPS overlay.');
  mapSec.body.appendChild(buildManeuverBrief(maneuver, { color }));
  const mapCard = document.createElement('div');
  mapCard.className = 'maneuver-map-surface';
  const mapCanvas = createManeuverCanvas(analysis ? 1100 : 1280, analysis ? 560 : 700);
  drawManeuverOverlayMap(mapCanvas, [{
    color,
    label: `${maneuver?.athlete_name || 'Athlete'} ${getManeuverTypeLabel(maneuver)}`,
    points: overlayGeometry.points,
    deepest: overlayGeometry.deepest,
  }], {
    title: analysis ? 'Normalized track overlay' : 'Track preview',
  });
  mapCard.appendChild(mapCanvas);

  const processRow = buildManeuverProcessActions({
    maneuvers: [maneuver],
    analyses: [analysis],
    emphasize: !analysis,
  });
  if (processRow && !analysis) mapSec.body.appendChild(processRow);
  mapSec.body.appendChild(mapCard);
  fragment.appendChild(mapSec.section);

  if (!analysis) {
    const preMediaSec = createManeuverSection('Video Clip', 'Clip preview is available even before pose processing is complete.');
    const preMediaGrid = document.createElement('div');
    preMediaGrid.className = 'maneuver-media-grid';
    const previewCard = await createManeuverVideoCard('Best overlapping video', maneuver, null, color);
    preMediaGrid.appendChild(previewCard.card);
    preMediaSec.body.appendChild(preMediaGrid);
    fragment.appendChild(preMediaSec.section);
    return fragment;
  }

  const mediaSec = createManeuverSection('Video + Pose Density', 'Synchronized video and boat-referenced pose density.');
  const mediaGrid = document.createElement('div');
  mediaGrid.className = 'maneuver-media-grid';
  const videoCard = await createManeuverVideoCard('Best overlapping video', maneuver, analysis, color);
  mediaGrid.appendChild(videoCard.card);
  const posePanel = document.createElement('div');
  posePanel.className = 'maneuver-heatmap-panel';
  const clouds = getManeuverPosePointClouds(analysis);
  const poseBusy = isManeuverPoseRunActive(analysis?.mediaRefs?.primaryVideoFileId);
  const heatmapGrid = document.createElement('div');
  heatmapGrid.className = 'maneuver-heatmap-grid';
  heatmapGrid.append(
    createManeuverHeatmapTile(
      'Athlete movement density',
      clouds.keypointXY.length
        ? `${clouds.frameCount || getManeuverPoseFrames(analysis).length} pose frames projected into boat space`
        : (poseBusy ? 'Pose processing is still running for this maneuver window.' : 'Pose frames were unavailable for this maneuver.'),
      createManeuverHeatmapJob(clouds.keypointXY, color, 'keypoints'),
      heatmapJobs,
    ),
    createManeuverHeatmapTile(
      'Center of mass density',
      clouds.comXY.length
        ? 'Center-of-mass positions over the full maneuver window'
        : (poseBusy ? 'Center-of-mass data will appear when pose processing finishes.' : 'Center-of-mass data was unavailable for this maneuver.'),
      createManeuverHeatmapJob(clouds.comXY, color, 'com'),
      heatmapJobs,
    ),
  );
  posePanel.appendChild(heatmapGrid);
  mediaGrid.appendChild(posePanel);
  mediaSec.body.appendChild(mediaGrid);
  fragment.appendChild(mediaSec.section);

  const summarySec = createManeuverSection('Summary Stats', 'Before, during, and after metrics from the shared maneuver detector.');
  const statsGrid = document.createElement('div');
  statsGrid.className = 'maneuver-stats-table';
  const statsSource = analysis?.maneuver || maneuver;
  statsGrid.append(
    buildManeuverStatsBlock('Before', statsSource?.preStats),
    buildManeuverStatsBlock('During', statsSource?.duringStats),
    buildManeuverStatsBlock('After', statsSource?.postStats),
  );
  summarySec.body.appendChild(statsGrid);
  fragment.appendChild(summarySec.section);

  const chartSec = createManeuverSection('Maneuver Timelines', 'Only timelines with usable data are shown.');
  const chartGrid = document.createElement('div');
  chartGrid.className = 'maneuver-chart-grid';
  const chartDefs = [
    { key: 'sog', title: 'Speed', yLabel: 'SOG (kt)', baselineAt: null },
    { key: 'vmg', title: 'VMG', yLabel: 'VMG (kt)', baselineAt: 0 },
    { key: 'twa', title: 'TWA Crossing', yLabel: 'Signed TWA (deg)', baselineAt: 0 },
    { key: 'heel', title: 'Heel', yLabel: 'Heel (deg)', baselineAt: 0 },
    { key: 'pitch', title: 'Pitch', yLabel: 'Pitch (deg)', baselineAt: 0 },
    { key: 'rudder', title: 'Rudder', yLabel: 'Rudder (deg)', baselineAt: 0 },
    ...(areBoomPredictionsEnabled() ? [{ key: 'boom', title: 'Boom', yLabel: 'Boom (deg)', baselineAt: 0 }] : []),
  ];
  const visibleChartDefs = chartDefs.filter(def => hasManeuverTimelineData(analysis, def.key, { align: 'anchor' }));
  if (!visibleChartDefs.length) {
    chartSec.body.appendChild(createManeuverInlineNote('This maneuver did not have enough synchronized telemetry to draw readable timelines.'));
  }
  for (const def of visibleChartDefs) {
    const card = document.createElement('div');
    card.className = 'maneuver-chart-card';
    const title = document.createElement('div');
    title.className = 'maneuver-chart-title';
    title.textContent = def.title;
    const canvas = createManeuverCanvas(940, 360);
    drawManeuverLineChart(canvas, [{
      label: maneuver?.athlete_name || 'Athlete',
      color,
      points: getManeuverTimelineSeries(analysis, def.key, { align: 'anchor' }),
    }], {
      title: def.title,
      yLabel: def.yLabel,
      baselineAt: def.baselineAt,
    });
    card.append(title, canvas);
    chartGrid.appendChild(card);
  }
  if (visibleChartDefs.length) chartSec.body.appendChild(chartGrid);
  if (processRow) {
    processRow.classList.add('is-footer');
    chartSec.body.appendChild(processRow);
  }
  fragment.appendChild(chartSec.section);

  queueManeuverHeatmapRender(heatmapJobs);
  return fragment;
}

function buildCompareQuickStatsTable(maneuvers, analyses, colors) {
  const table = document.createElement('table');
  table.className = 'maneuver-compare-matrix';
  const thead = document.createElement('thead');
  thead.innerHTML = `
    <tr>
      <th>Move</th>
      <th>Entry</th>
      <th>Min</th>
      <th>Exit</th>
      <th>Delta</th>
      <th>Recovery</th>
    </tr>
  `;
  const tbody = document.createElement('tbody');
  maneuvers.forEach((move, idx) => {
    const source = analyses[idx]?.maneuver || move;
    const during = source?.duringStats || {};
    const pre = source?.preStats || {};
    const post = source?.postStats || {};
    const entry = during?.entrySpeedKts ?? pre?.entrySpeedKts ?? pre?.avgSogKts;
    const exit = during?.exitSpeedKts ?? post?.exitSpeedKts ?? post?.avgSogKts;
    const min = during?.minSpeedKts;
    const delta = Number.isFinite(Number(entry)) && Number.isFinite(Number(exit))
      ? Number(exit) - Number(entry)
      : null;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="maneuver-compare-chip-dot" style="background:${colors[idx] || getManeuverColor(move)};"></span> ${getManeuverRoleLabel(idx)}</td>
      <td>${formatManeuverNumber(entry, { digits: 1, suffix: ' kt' })}</td>
      <td>${formatManeuverNumber(min, { digits: 1, suffix: ' kt' })}</td>
      <td>${formatManeuverNumber(exit, { digits: 1, suffix: ' kt' })}</td>
      <td>${formatManeuverNumber(delta, { digits: 1, suffix: ' kt', signed: true })}</td>
      <td>${formatManeuverNumber(during?.speedRecoveryTimeS, { digits: 1, suffix: ' s' })}</td>
    `;
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  return table;
}

function getManeuverCompareChartDefs() {
  return [
    { key: 'sog', title: 'Speed', yLabel: 'SOG (kt)', baselineAt: null },
    { key: 'vmg', title: 'VMG', yLabel: 'VMG (kt)', baselineAt: 0 },
    { key: 'twa', title: 'TWA Crossing', yLabel: 'Signed TWA (deg)', baselineAt: 0 },
    { key: 'heel', title: 'Heel', yLabel: 'Heel (deg)', baselineAt: 0 },
    { key: 'pitch', title: 'Pitch', yLabel: 'Pitch (deg)', baselineAt: 0 },
    { key: 'rudder', title: 'Rudder', yLabel: 'Rudder (deg)', baselineAt: 0 },
    ...(areBoomPredictionsEnabled() ? [{ key: 'boom', title: 'Boom', yLabel: 'Boom (deg)', baselineAt: 0 }] : []),
  ];
}

function closeManeuverGraphOverlay() {
  const overlay = document.querySelector('.maneuver-graph-overlay');
  if (!overlay) return;
  document.removeEventListener('keydown', overlay._onKeyDown);
  overlay.remove();
}

function openManeuverGraphOverlay(def, maneuvers = [], analyses = [], colors = []) {
  if (!def) return;
  closeManeuverGraphOverlay();
  const rows = maneuvers.map((move, idx) => ({
    move,
    analysis: analyses[idx],
    color: colors[idx] || getManeuverColor(move),
    points: getManeuverTimelineSeries(analyses[idx], def.key, {
      align: 'anchor',
      extraShiftSec: idx > 0 ? getManeuverCompareOffsetSec(maneuvers, move) : 0,
    }),
  }));
  if (!rows.some(row => row.points.length >= 2)) return;

  const overlay = document.createElement('div');
  overlay.className = 'maneuver-graph-overlay';
  const shell = document.createElement('div');
  shell.className = 'maneuver-graph-overlay-shell';

  const head = document.createElement('div');
  head.className = 'maneuver-graph-overlay-head';
  const titleWrap = document.createElement('div');
  const title = document.createElement('div');
  title.className = 'maneuver-graph-overlay-title';
  title.textContent = def.title;
  const sub = document.createElement('div');
  sub.className = 'maneuver-graph-overlay-sub';
  sub.textContent = `${maneuvers.length} ${getManeuverTypeLabel(maneuvers[0])} maneuvers aligned at anchor`;
  titleWrap.append(title, sub);
  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn';
  closeBtn.type = 'button';
  closeBtn.textContent = 'Close';
  closeBtn.onclick = closeManeuverGraphOverlay;
  head.append(titleWrap, closeBtn);

  const grid = document.createElement('div');
  grid.className = 'maneuver-graph-overlay-grid';
  grid.style.gridTemplateColumns = `repeat(${rows.length}, minmax(min(560px, 100%), 1fr))`;
  rows.forEach((row, idx) => {
    const card = document.createElement('section');
    card.className = 'inline-plot-overlay-card';
    card.style.borderColor = rgbaFromHex(row.color, 0.28);
    card.style.boxShadow = `inset 0 0 0 1px ${rgbaFromHex(row.color, 0.08)}`;

    const cardHead = document.createElement('div');
    cardHead.className = 'inline-plot-overlay-card-head';
    const dot = document.createElement('span');
    dot.className = 'inline-plot-overlay-card-dot';
    dot.style.background = row.color;
    const heading = document.createElement('div');
    heading.className = 'inline-plot-overlay-card-heading';
    const name = document.createElement('div');
    name.className = 'inline-plot-overlay-card-name';
    name.textContent = `${getManeuverRoleLabel(idx)} · ${row.move?.athlete_name || 'Athlete'}`;
    const meta = document.createElement('div');
    meta.className = 'inline-plot-overlay-card-meta';
    meta.textContent = row.points.length >= 2 ? `${row.points.length.toLocaleString()} samples` : 'Not enough synchronized data';
    heading.append(name, meta);
    cardHead.append(dot, heading);
    card.appendChild(cardHead);

    if (row.points.length < 2) {
      const empty = document.createElement('div');
      empty.className = 'inline-plot-overlay-card-empty';
      empty.textContent = 'Not enough data in this maneuver window.';
      card.appendChild(empty);
    } else {
      const media = document.createElement('div');
      media.className = 'inline-plot-overlay-card-media';
      const canvas = createManeuverCanvas(960, 460);
      drawManeuverLineChart(canvas, [{
        label: `${getManeuverRoleLabel(idx)} · ${row.move?.athlete_name || 'Athlete'}`,
        color: row.color,
        points: row.points,
      }], {
        title: def.title,
        yLabel: def.yLabel,
        baselineAt: def.baselineAt,
      });
      media.appendChild(canvas);
      card.appendChild(media);
    }
    grid.appendChild(card);
  });

  shell.append(head, grid);
  overlay.appendChild(shell);
  overlay._onKeyDown = evt => {
    if (evt.key === 'Escape') closeManeuverGraphOverlay();
  };
  document.addEventListener('keydown', overlay._onKeyDown);
  overlay.addEventListener('click', evt => {
    if (evt.target === overlay) closeManeuverGraphOverlay();
  });
  document.body.appendChild(overlay);
}

function createManeuverOverlayChartCard(def, maneuvers = [], analyses = [], colors = []) {
  const card = document.createElement('button');
  card.type = 'button';
  card.className = 'maneuver-chart-card maneuver-chart-card-button';
  const title = document.createElement('div');
  title.className = 'maneuver-chart-title';
  title.textContent = def.title;
  const canvas = createManeuverCanvas(760, 300);
  drawManeuverLineChart(canvas, maneuvers.map((move, idx) => ({
    label: `${getManeuverRoleLabel(idx)} · ${move?.athlete_name || `Move ${idx + 1}`}`,
    color: colors[idx] || getManeuverColor(move),
    points: getManeuverTimelineSeries(analyses[idx], def.key, {
      align: 'anchor',
      extraShiftSec: idx > 0 ? getManeuverCompareOffsetSec(maneuvers, move) : 0,
    }),
  })), {
    title: def.title,
    yLabel: def.yLabel,
    baselineAt: def.baselineAt,
  });
  card.append(title, canvas);
  card.title = 'Expand this timeline and split by maneuver';
  card.onclick = () => openManeuverGraphOverlay(def, maneuvers, analyses, colors);
  return card;
}

async function buildCompareManeuverContent(maneuvers, analyses) {
  const fragment = document.createDocumentFragment();
  const colors = buildManeuverCompareColors(maneuvers);
  const heatmapJobs = [];
  const videoEntries = [];
  _maneuverCompareScrollRatio = 0;

  const processRow = buildManeuverProcessActions({ maneuvers, analyses });

  const compareSec = createManeuverSection(`${getManeuverTypeLabel(maneuvers[0])} Compare`, 'One full column per selected maneuver. Timelines are overlayed in every column; Expand splits them.');
  compareSec.section.classList.add('maneuver-compare-stage');
  const rail = document.createElement('div');
  rail.className = 'maneuver-compare-column-rail maneuver-compare-direct-rail';
  const chartDefs = getManeuverCompareChartDefs();
  const visibleChartDefs = chartDefs.filter(def => analyses.some(analysis => hasManeuverTimelineData(analysis, def.key, { align: 'anchor' })));

  for (let i = 0; i < maneuvers.length; i++) {
    const move = maneuvers[i];
    const analysis = analyses[i];
    const color = colors[i] || getManeuverColor(move);
    const column = document.createElement('section');
    column.className = 'maneuver-compare-column';
    column.style.borderColor = rgbaFromHex(color, 0.2);

    const sticky = document.createElement('div');
    sticky.className = 'maneuver-compare-column-head';
    sticky.appendChild(buildManeuverBrief(move, {
      color,
      prefix: getManeuverRoleLabel(i),
    }));
    column.appendChild(sticky);

    const scroll = document.createElement('div');
    scroll.className = 'maneuver-compare-column-scroll';
    initManeuverCompareColumnSync(scroll);

    const entry = await createManeuverVideoCard(`${getManeuverRoleLabel(i)} video`, move, analysis, color, {
      showOffsetSlider: i > 0,
      initialOffsetSec: i > 0 ? getManeuverCompareOffsetSec(maneuvers, move) : 0,
      onOffsetChange: (offsetSec, committed) => {
        setManeuverCompareOffsetSec(maneuvers, move, offsetSec);
        if (committed) {
          void renderManeuverWorkspace({ loadDeep: false, force: false });
        }
      },
    });
    entry.label = getManeuverRoleLabel(i);
    videoEntries.push(entry);
    if (entry.card) entry.card.classList.add('maneuver-compare-video-card');
    column.appendChild(entry.card);

    const statsSource = analysis?.maneuver || move;
    const statsCard = document.createElement('div');
    statsCard.className = 'maneuver-compare-summary-card';
    const statsGrid = document.createElement('div');
    statsGrid.className = 'maneuver-stats-table';
    statsGrid.append(
      buildManeuverStatsBlock('Before', statsSource?.preStats),
      buildManeuverStatsBlock('During', statsSource?.duringStats),
      buildManeuverStatsBlock('After', statsSource?.postStats),
    );
    statsCard.appendChild(statsGrid);
    scroll.appendChild(statsCard);

    const chartStack = document.createElement('div');
    chartStack.className = 'maneuver-compare-column-charts';
    if (!visibleChartDefs.length) {
      chartStack.appendChild(createManeuverInlineNote('No synchronized timelines were available for these maneuvers.'));
    } else {
      for (const def of visibleChartDefs) {
        chartStack.appendChild(createManeuverOverlayChartCard(def, maneuvers, analyses, colors));
      }
    }
    scroll.appendChild(chartStack);

    const clouds = getManeuverPosePointClouds(analysis);
    const poseBusy = isManeuverPoseRunActive(analysis?.mediaRefs?.primaryVideoFileId);
    const posePanel = document.createElement('div');
    posePanel.className = 'maneuver-heatmap-panel';
    const heatmapGrid = document.createElement('div');
    heatmapGrid.className = 'maneuver-heatmap-grid';
    heatmapGrid.append(
      createManeuverHeatmapTile(
        'Athlete movement density',
        clouds.keypointXY.length
          ? `${clouds.frameCount || getManeuverPoseFrames(analysis).length} pose frames projected into boat space`
          : (poseBusy ? 'Pose processing is still running for this maneuver window.' : 'Pose frames were unavailable for this maneuver.'),
        createManeuverHeatmapJob(clouds.keypointXY, color, 'keypoints'),
        heatmapJobs,
      ),
      createManeuverHeatmapTile(
        'Center of mass density',
        clouds.comXY.length
          ? 'Center-of-mass positions over the full maneuver window'
          : (poseBusy ? 'Center-of-mass data will appear when pose processing finishes.' : 'Center-of-mass data was unavailable for this maneuver.'),
        createManeuverHeatmapJob(clouds.comXY, color, 'com'),
        heatmapJobs,
      ),
    );
    posePanel.appendChild(heatmapGrid);
    scroll.appendChild(posePanel);

    column.appendChild(scroll);
    rail.appendChild(column);
  }

  syncManeuverVideoEntries(videoEntries);
  compareSec.body.appendChild(rail);
  fragment.appendChild(compareSec.section);
  if (processRow) {
    processRow.classList.add('is-footer');
    fragment.appendChild(processRow);
  }

  requestAnimationFrame(() => syncManeuverCompareScrollPositions());

  queueManeuverHeatmapRender(heatmapJobs);
  return fragment;
}

async function renderManeuverWorkspace({ loadDeep = false, force = false } = {}) {
  const content = el('maneuver-analysis-content') || el('maneuver-content');
  const shell = el('maneuver-analysis-shell') || el('maneuver-wrap');
  if (!content) return;
  content.classList.remove('maneuver-stage-empty');
  const resetScroll = () => {
    if (!shell) return;
    requestAnimationFrame(() => {
      try { shell.scrollTop = 0; } catch {}
    });
  };

  const selection = buildManeuverSelectionState();
  const selected = selection.selected || [];
  const maneuverLayout = el('maneuver-view-layout');
  if (maneuverLayout) {
    maneuverLayout.classList.toggle('maneuver-compare-active', selection.kind === 'compare');
    if (selection.kind === 'compare') {
      maneuverLayout.style.setProperty('--maneuver-compare-map-col', selected.length >= 3 ? '26%' : '34%');
    } else {
      maneuverLayout.style.removeProperty('--maneuver-compare-map-col');
    }
  }
  const token = ++state.advancedPane.maneuverLoadToken;
  state.advancedPane.maneuverIds = selected.map(move => String(move.id));
  content.innerHTML = '';
  if (shell) shell.scrollTop = 0;
  syncManeuverWorkspaceSurface(selection.kind !== 'empty');

  if (selection.kind === 'empty') {
    content.appendChild(createManeuverEmptyState('Maneuver Workspace', selection.reason));
    resetScroll();
    return;
  }

  if (selection.kind === 'disabled') {
    content.appendChild(createManeuverEmptyState('Compare Not Available', selection.reason));
    resetScroll();
    return;
  }

  if (loadDeep) {
    content.appendChild(createManeuverEmptyState(
      selection.kind === 'compare' ? 'Preparing Compare Workspace' : 'Preparing Maneuver Workspace',
      'Loading cached analysis and bounded pose processing for the selected maneuver window.',
    ));
  }

  const targetMoves = selection.kind === 'compare' ? selected : [selected[0]];
  const analyses = [];
  for (const move of targetMoves) {
    if (loadDeep) analyses.push(await ensureManeuverAnalysisReady(move, { force }));
    else analyses.push(await DB.getManeuverAnalysis(state.projectId, move.id));
  }
  if (token !== state.advancedPane.maneuverLoadToken) return;

  content.innerHTML = '';
  if (selection.kind === 'compare') {
    content.appendChild(await buildCompareManeuverContent(selected, analyses));
    resetScroll();
    syncManeuverWorkspaceSurface(true);
    return;
  }

  content.appendChild(await buildSingleManeuverContent(selected[0], analyses[0] || null));
  resetScroll();
  syncManeuverWorkspaceSurface(true);
}

async function openManeuverWorkspace({ force = false, preserveMapView = false, loadDeep = false, renderMapView = true } = {}) {
  const alreadyActive = !!el('view-maneuvers')?.classList.contains('active-view');
  if (!alreadyActive) switchView('view-maneuvers');
  await renderManeuverWorkspace({ loadDeep, force });
  if (renderMapView) renderManeuverMap({ preserveView: preserveMapView || alreadyActive });
  return true;
}

function getManeuversForCsvExport() {
  const checked = getCheckedManeuvers();
  return checked.length ? checked : getSelectedManeuvers();
}

function buildManeuverBoomRudderCsv(maneuvers = [], analyses = []) {
  const headers = [
    'maneuver_id',
    'type',
    'athlete_id',
    'athlete_name',
    'side_from',
    'side_to',
    'time_vs_anchor_s',
    'abs_ts_s',
    'iso_time',
    'rudder_angle_deg',
    'boom_angle_deg',
  ];
  const rows = [];
  for (let idx = 0; idx < maneuvers.length; idx++) {
    const move = maneuvers[idx];
    const analysis = analyses[idx];
    const sourceMove = analysis?.maneuver || move;
    const anchorTs = Number(sourceMove?.anchor_ts ?? move?.anchor_ts);
    const timelineRows = makeMetricTimelineRows(
      getManeuverTimelineSeries(analysis, 'rudder', { align: 'anchor' }),
      getManeuverTimelineSeries(analysis, 'boom', { align: 'anchor' }),
    );
    for (const point of timelineRows) {
      const absTs = Number.isFinite(anchorTs) ? anchorTs + Number(point.t) : null;
      rows.push({
        maneuver_id: sourceMove?.id || move?.id || '',
        type: sourceMove?.type || move?.type || '',
        athlete_id: sourceMove?.athlete_id || move?.athlete_id || '',
        athlete_name: sourceMove?.athlete_name || move?.athlete_name || '',
        side_from: sourceMove?.side_from || move?.side_from || '',
        side_to: sourceMove?.side_to || move?.side_to || '',
        time_vs_anchor_s: csvNumber(point.t, 3),
        abs_ts_s: csvNumber(absTs, 3),
        iso_time: epochSecondsToIso(absTs),
        rudder_angle_deg: csvNumber(point.rudder, 6),
        boom_angle_deg: csvNumber(point.boom, 6),
      });
    }
  }
  return { csv: makeCsv(headers, rows), rowCount: rows.length };
}

async function downloadSelectedManeuverCsv() {
  if (!state.projectId) {
    alert('Select a project before exporting maneuver CSV.');
    return;
  }
  if (!state.advancedMode || !state.reportOptions?.downloadCsv) {
    alert('Enable Download CSV in Advanced Features first.');
    return;
  }
  const maneuvers = getManeuversForCsvExport();
  if (!maneuvers.length) {
    alert('Select or check at least one maneuver to export CSV.');
    return;
  }

  const btn = el('btn-maneuver-download-csv');
  if (btn) { btn.disabled = true; btn.textContent = 'Preparing CSV...'; }
  try {
    const analyses = [];
    for (const move of maneuvers) {
      let analysis = await DB.getManeuverAnalysis(state.projectId, move.id).catch(() => null);
      if (!analysis) {
        analysis = await buildManeuverAnalysis(state.projectId, move.id, {
          force: false,
          windContext: buildReportWindContext(),
        }).catch(err => {
          console.warn('[maneuver-csv] analysis build failed:', err);
          return null;
        });
      }
      analyses.push(analysis);
    }
    const { csv, rowCount } = buildManeuverBoomRudderCsv(maneuvers, analyses);
    if (!rowCount) {
      alert('No boom or rudder timeline data was found for the selected maneuver(s).');
      return;
    }
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    downloadBlob(blob, buildManeuverCsvFileName(maneuvers, Date.now()));
  } catch (err) {
    console.error('Maneuver CSV export failed:', err);
    alert('Maneuver CSV export failed: ' + (err?.message || err));
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<svg class="icon-svg"><use href="#ico-download"/></svg> Download CSV';
    }
    renderManeuverPanel();
  }
}

async function downloadCheckedManeuverCompareMap() {
  const checked = getCheckedManeuvers();
  if (checked.length < 2) {
    alert('Check at least two maneuvers to export a compare map.');
    return;
  }
  if (!state.projectId) {
    alert('Select a project before exporting a maneuver compare map.');
    return;
  }

  const analyses = [];
  for (const move of checked) {
    let analysis = await DB.getManeuverAnalysis(state.projectId, move.id).catch(() => null);
    if (!analysis || !hasRenderableManeuverGpsOverlay(analysis)) analysis = null;
    analyses.push(analysis);
  }
  const colors = buildManeuverCompareColors(checked);
  const overlayGeometries = await Promise.all(checked.map((move, idx) => resolveManeuverOverlayGeometry(move, analyses[idx])));
  const entries = checked.map((move, idx) => ({
    color: colors[idx],
    label: `${move?.athlete_name || `Move ${idx + 1}`} · ${getManeuverTypeLabel(move)} · ${fmtClock(move?.anchor_ts)}`,
    points: overlayGeometries[idx]?.points || [],
    deepest: overlayGeometries[idx]?.deepest || null,
  })).filter(entry => Array.isArray(entry.points) && entry.points.length >= 2);

  if (entries.length < 2) {
    alert('At least two checked maneuvers need usable overlay tracks before a compare map can be exported.');
    return;
  }

  const canvas = createManeuverCanvas(1700, 1060);
  drawManeuverOverlayMap(canvas, entries, {
    title: `${entries.length} Maneuver Overlay`,
    paddingRatioX: 0.2,
    paddingRatioY: 0.22,
    arrowSpread: true,
  });
  const blob = await canvasToBlob(canvas, 'image/png');
  if (!blob) {
    alert('The maneuver compare map could not be rendered for download.');
    return;
  }

  const projectLabel = sanitizeFilenamePart(state.projectName || state.projectId || 'project') || 'project';
  const stamp = new Date().toISOString().replace(/[:]/g, '-').replace(/\..+$/, '');
  downloadBlob(blob, `${projectLabel}_maneuver_overlay_${stamp}.png`);
}

function intervalCoversRange(intervals, startSec, endSec, tolerance = 0.75) {
  const start = Number(startSec);
  const end = Number.isFinite(Number(endSec)) ? Number(endSec) : start;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return false;
  const ordered = (intervals || [])
    .map(([s, e]) => [Number(s), Number(e)])
    .filter(([s, e]) => Number.isFinite(s) && Number.isFinite(e))
    .sort((a, b) => a[0] - b[0]);
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

function intervalCoveredDuration(intervals, startSec, endSec, tolerance = 0.75) {
  const start = Number(startSec);
  const end = Number.isFinite(Number(endSec)) ? Number(endSec) : start;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return 0;
  const rangeEnd = Math.max(start, end);
  const ordered = (intervals || [])
    .map(([s, e]) => [Number(s), Number(e)])
    .filter(([s, e]) => Number.isFinite(s) && Number.isFinite(e))
    .map(([s, e]) => [s - tolerance, e + tolerance])
    .sort((a, b) => a[0] - b[0]);
  if (!ordered.length || rangeEnd <= start) return 0;
  let coveredUntil = start;
  let covered = 0;
  for (const [segStart, segEnd] of ordered) {
    const clippedStart = Math.max(start, segStart);
    const clippedEnd = Math.min(rangeEnd, segEnd);
    if (clippedEnd <= coveredUntil) continue;
    if (clippedStart > coveredUntil) {
      coveredUntil = clippedStart;
    }
    if (clippedEnd > coveredUntil) {
      covered += clippedEnd - coveredUntil;
      coveredUntil = clippedEnd;
      if (coveredUntil >= rangeEnd) break;
    }
  }
  return Math.max(0, Math.min(rangeEnd - start, covered));
}

function intervalCoverageRatio(intervals, startSec, endSec, tolerance = 0.75) {
  const start = Number(startSec);
  const end = Number.isFinite(Number(endSec)) ? Number(endSec) : start;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return 0;
  const span = Math.max(0, end - start);
  if (span <= 1e-6) return intervalCoversRange(intervals, start, end, tolerance) ? 1 : 0;
  return intervalCoveredDuration(intervals, start, end, tolerance) / span;
}

function hasQueuedSegmentOverlap(fileId, startSec, endSec, tolerance = 0.75) {
  const runs = _queuedSegmentRuns.get(fileId) || [];
  const start = Number(startSec);
  const end = Number.isFinite(Number(endSec)) ? Number(endSec) : start;
  return runs.some(run => {
    const runStart = Number.isFinite(Number(run?.startSec)) ? Number(run.startSec) : 0;
    const runEnd = Number.isFinite(Number(run?.endSec)) ? Number(run.endSec) : Infinity;
    return runStart <= end + tolerance && runEnd >= start - tolerance;
  });
}

let _maneuverCompareScrollSyncLock = false;
let _maneuverCompareScrollRatio = 0;

function getVisibleManeuverCompareScrollCols() {
  return [...document.querySelectorAll('.maneuver-compare-column-scroll')].filter(col => col.offsetParent !== null);
}

function syncManeuverCompareScrollPositions(sourceCol = null) {
  const cols = getVisibleManeuverCompareScrollCols();
  if (cols.length < 2) return;
  const source = cols.includes(sourceCol) ? sourceCol : cols[0];
  const sourceMax = Math.max(0, source.scrollHeight - source.clientHeight);
  const ratio = sourceMax > 0 ? source.scrollTop / sourceMax : _maneuverCompareScrollRatio;
  _maneuverCompareScrollRatio = Math.max(0, Math.min(1, Number.isFinite(ratio) ? ratio : 0));

  _maneuverCompareScrollSyncLock = true;
  try {
    for (const col of cols) {
      if (col === source) continue;
      const targetMax = Math.max(0, col.scrollHeight - col.clientHeight);
      col.scrollTop = targetMax > 0 ? targetMax * _maneuverCompareScrollRatio : 0;
    }
  } finally {
    _maneuverCompareScrollSyncLock = false;
  }
}

function initManeuverCompareColumnSync(col) {
  if (!col || col.dataset.scrollSyncInit === '1') return;
  col.dataset.scrollSyncInit = '1';
  col.addEventListener('scroll', () => {
    if (_maneuverCompareScrollSyncLock) return;
    syncManeuverCompareScrollPositions(col);
  }, { passive: true });
}

function getKnownSkeletonCoverage(fileId) {
  const processedRanges = state.cvStatuses?.[fileId]?.processed_ranges;
  const liveCoverage = state.skeletonCoverage?.[fileId] || [];
  return [
    ...(Array.isArray(processedRanges) ? processedRanges : []),
    ...(Array.isArray(liveCoverage) ? liveCoverage : []),
  ];
}

function isSkeletonRangeReady(fileId, startSec, endSec, durationSec = null, tolerance = 1.25) {
  const start = Number(startSec) || 0;
  const duration = Number(durationSec);
  const end = Number.isFinite(Number(endSec))
    ? Number(endSec)
    : (Number.isFinite(duration) && duration > 0 ? duration : start);
  if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return false;
  const coverage = getKnownSkeletonCoverage(fileId);
  if (!coverage.length) return false;
  const statusKey = String(state.cvStatuses?.[fileId]?.status || '').toLowerCase();
  const ratio = intervalCoverageRatio(coverage, start, end, tolerance);
  return intervalCoversRange(coverage, start, end, tolerance)
    || (statusKey === 'completed' && ratio >= 0.9);
}

function getSegmentProcessingStatus(seg) {
  const overlapping = findOverlappingVideos(seg);
  const analyzableOverlapping = findOverlappingVideos(seg, { analyzableOnly: true });
  if (!overlapping.length) {
    return { key: 'missing', label: 'No video', detail: 'No overlapping videos', ready: false };
  }
  if (!analyzableOverlapping.length) {
    return { key: 'playback', label: 'Playback Only', detail: 'Only playback-only videos overlap', ready: false };
  }

  let readyCount = 0;
  let processingCount = 0;
  let partialCount = 0;

  for (const overlap of analyzableOverlapping) {
    const startSec = Number(overlap.videoStartSec) || 0;
    const endSec = Number.isFinite(Number(overlap.videoEndSec))
      ? Number(overlap.videoEndSec)
      : (Number.isFinite(Number(overlap.vid?.duration_sec)) ? Number(overlap.vid.duration_sec) : startSec);
    const coverage = getKnownSkeletonCoverage(overlap.vid.id);
    const statusKey = String(state.cvStatuses?.[overlap.vid.id]?.status || '').toLowerCase();
    const coverageTolerance = 1.25;
    const coverageRatio = intervalCoverageRatio(coverage, startSec, endSec, coverageTolerance);
    const running = PoseEngine.isProcessing(overlap.vid.id) || hasQueuedSegmentOverlap(overlap.vid.id, startSec, endSec);
    const covered = intervalCoversRange(coverage, startSec, endSec, coverageTolerance)
      || (statusKey === 'completed' && coverageRatio >= 0.9);

    if (covered && !running) readyCount++;
    else if (running) processingCount++;
    else if (coverage.length || statusKey === 'completed' || coverageRatio > 0) partialCount++;
  }

  const total = analyzableOverlapping.length;
  if (readyCount === total) {
    return { key: 'ready', label: 'Ready', detail: `${readyCount}/${total} videos ready`, ready: true };
  }
  if (processingCount > 0) {
    return {
      key: 'processing',
      label: 'Processing',
      detail: `${readyCount}/${total} ready${processingCount ? ` | ${processingCount} running` : ''}`,
      ready: false,
    };
  }
  if (readyCount > 0 || partialCount > 0) {
    return { key: 'partial', label: 'Partial', detail: `${readyCount}/${total} videos ready`, ready: false };
  }
  return { key: 'unprocessed', label: 'Unprocessed', detail: `0/${total} videos ready`, ready: false };
}

function isAnalysisViewActive() {
  return !!el('view-analysis')?.classList.contains('active-view');
}

function updateHeatmapButtonState(seg = (isAnalysisViewActive() ? getSegmentAtTs(state.tl?.currentTs) : null), inAnalysis = isAnalysisViewActive()) {
  const btn = el('btn-heatmaps');
  if (!btn) return;
  btn.style.display = state.advancedMode ? '' : 'none';
  btn.classList.toggle('active', !!state.inlineHeatmaps?.visible);
  btn.disabled = !state.advancedMode || !inAnalysis || (!seg && !state.inlineHeatmaps?.visible);
  btn.title = seg
    ? (state.inlineHeatmaps?.visible
      ? 'Hide the inline report heatmaps beside the videos'
      : 'Show the report heatmaps beside each visible video')
    : 'Move the playhead into a segment to show inline heatmaps';
}

function syncHeatmapViewerToCurrentSegment(seg, inAnalysis = isAnalysisViewActive()) {
  syncInlineHeatmapsToCurrentSegment(seg, inAnalysis);
}

function updateCurrentSegmentActions() {
  const wrap = el('current-seg-actions');
  const label = el('current-seg-label');
  const inAnalysis = isAnalysisViewActive();
  const seg = inAnalysis ? getSegmentAtTs(state.tl?.currentTs) : null;
  if (wrap && label) {
    if (!seg) {
      wrap.style.display = 'none';
      wrap.dataset.segmentId = '';
      state._topbarSegmentId = null;
      state._topbarSegmentName = '';
    } else {
      wrap.style.display = 'inline-flex';
      wrap.dataset.segmentId = String(seg.id);
      if (state._topbarSegmentId !== seg.id || state._topbarSegmentName !== seg.name) {
        label.textContent = seg.name;
        label.title = seg.name;
        state._topbarSegmentId = seg.id;
        state._topbarSegmentName = seg.name;
      }
    }
  }
  updateHeatmapButtonState(seg, inAnalysis);
  syncHeatmapViewerToCurrentSegment(seg, inAnalysis);
}

function assignVideoColors() {
  if(!state.mapData) return;
  state.videoColors = {};
  const vids = state.mapData.videos || [];
  const csvs = state.mapData.csvs || [];
  const colorByAthlete = new Map(); // normalized athleteId -> color
  let fallbackIdx = 0;
  const resolveAthleteColor = (athId) => {
    const norm = normalizeAthleteId(athId);
    if(!norm) return null;
    if(colorByAthlete.has(norm)) return colorByAthlete.get(norm);
    const ath = findAthleteById(norm);
    let color = null;
    if(ath) {
      const athIdx = state.athletes.findIndex(a => normalizeAthleteId(a?.id) === norm);
      color = ath.color || PALETTE[(athIdx >= 0 ? athIdx : colorByAthlete.size) % PALETTE.length];
    } else {
      color = PALETTE[fallbackIdx++ % PALETTE.length];
    }
    colorByAthlete.set(norm, color);
    return color;
  };

  for(const v of vids) {
    const meta = state.fileMeta[v.id] || {};
    let athIdForColor = isPlaybackOnlyVideo(v) ? null : normalizeAthleteId(meta.athlete_id);
    // Derive from matched CSV if video has no direct athlete
    if(!athIdForColor && !isPlaybackOnlyVideo(v) && v.best_match_csv_id) {
      const csvMeta = state.fileMeta[v.best_match_csv_id] || {};
      athIdForColor = normalizeAthleteId(csvMeta.athlete_id);
    }
    const athleteColor = resolveAthleteColor(athIdForColor);
    state.videoColors[v.id] = athleteColor || PALETTE[fallbackIdx++ % PALETTE.length];
  }
  const matchedVidByCSV = {};
  for(const v of vids) { if(v.best_match_csv_id) matchedVidByCSV[v.best_match_csv_id] = v.id; }
  for(const csv of csvs) {
    const vid = matchedVidByCSV[csv.id];
    if(vid) {
      state.videoColors[csv.id] = state.videoColors[vid];
      continue;
    }
    const csvMeta = state.fileMeta[csv.id] || {};
    const csvAthColor = resolveAthleteColor(csvMeta.athlete_id);
    state.videoColors[csv.id] = csvAthColor || '#2ea043';
  }
}

function initTrackVisibility() {
  const vids = state.mapData?.videos || [];
  const csvs = state.mapData?.csvs || [];
  const validIds = new Set([...vids.map(v => v.id), ...csvs.map(c => c.id)]);
  for(const k of Object.keys(state.trackVisibility)) {
    if(!validIds.has(k)) delete state.trackVisibility[k];
  }
  for(const v of vids) {
    if(v.id in state.trackVisibility) continue;
    // A video with its own GoPro GPS track defaults to visible (so the track shows
    // on every reload). A playback-only clip with no telemetry has nothing of its
    // own to draw, so its default doesn't matter — keep it visible too.
    state.trackVisibility[v.id] = true;
  }
  for(const csv of csvs) { if(!(csv.id in state.trackVisibility)) state.trackVisibility[csv.id] = true; }
}

function wrapWindDegrees(value) {
  let out = Number(value) % 360;
  if (out < 0) out += 360;
  return out;
}

function angleDifferenceDeg(a, b) {
  let diff = wrapWindDegrees(a) - wrapWindDegrees(b);
  if (diff > 180) diff -= 360;
  else if (diff < -180) diff += 360;
  return diff;
}

function mirrorAngleToHalfCircleDeg(angleDeg) {
  const wrapped = Math.abs(angleDifferenceDeg(angleDeg, 0));
  return wrapped > 180 ? 360 - wrapped : wrapped;
}

function hasResolvedWindEstimate(wind) {
  return Number.isFinite(Number(wind?.directionDeg)) && Number.isFinite(Number(wind?.speedKts));
}

function windCardinalLabel(directionDeg) {
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const wrapped = wrapWindDegrees(directionDeg);
  return dirs[Math.round(wrapped / 45) % dirs.length];
}

function formatWindDirection(directionDeg) {
  const wrapped = wrapWindDegrees(directionDeg);
  return `${Math.round(wrapped)} deg ${windCardinalLabel(wrapped)}`;
}

function windArrowRotationDeg(directionDeg) {
  return wrapWindDegrees(directionDeg + 180);
}

function findNearestWindPointByTs(points, absTs) {
  if (!Array.isArray(points) || !points.length || !Number.isFinite(absTs)) return null;
  let lo = 0;
  let hi = points.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (Number(points[mid]?.ts) < absTs) lo = mid + 1;
    else hi = mid;
  }
  const candidates = [points[lo], points[Math.max(0, lo - 1)], points[Math.min(points.length - 1, lo + 1)]];
  let best = null;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (const point of candidates) {
    const ts = Number(point?.ts);
    if (!Number.isFinite(ts)) continue;
    const delta = Math.abs(ts - absTs);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = point;
    }
  }
  return best;
}

function ensureWindMapControl() {
  if (!state.map) return null;
  if (state.wind.control && state.wind.control.getPosition?.() !== 'bottomleft') {
    try { state.map.removeControl(state.wind.control); } catch {}
    state.wind.control = null;
  }
  if (state.wind.control) return state.wind.control;
  const control = L.control({ position: 'bottomleft' });
  control.onAdd = function() {
    const root = L.DomUtil.create('div', 'leaflet-control wind-map-control');
    L.DomEvent.disableClickPropagation(root);
    L.DomEvent.disableScrollPropagation(root);
    return root;
  };
  control.addTo(state.map);
  state.wind.control = control;
  return control;
}

function renderWindMapControl() {
  if (!state.map) return;
  const csvCount = (state.mapData?.csvs || []).length;
  if (!state.advancedFeatures?.windPanel || !csvCount) {
    const root = state.wind.control?._container;
    if (root) {
      root.style.display = 'none';
      root.innerHTML = '';
    }
    return;
  }
  const control = ensureWindMapControl();
  const root = control?._container;
  if (!root) {
    return;
  }
  root.style.display = '';
  const session = state.wind.session;
  const localNow = state.wind.localNow;
  const resolvedCount = Object.keys(state.wind.byCsvId || {}).length;
  const hasSession = Number.isFinite(session?.directionDeg) && Number.isFinite(session?.speedKts);
  const hasLocal = Number.isFinite(localNow?.directionDeg) && Number.isFinite(localNow?.speedKts);
  const emptyBody = state.wind.loading
    ? 'Estimating wind from the imported CSV tracks.'
    : 'Need steadier CSV sailing samples to estimate wind.';
  const sessionArrowDeg = windArrowRotationDeg(hasSession ? session.directionDeg : 0);
  const localArrowDeg = windArrowRotationDeg(hasLocal ? localNow.directionDeg : (hasSession ? session.directionDeg : 0));
  root.innerHTML = `
    <div class="wind-card ${state.wind.loading ? 'is-loading' : ''}">
      <div class="wind-card-head">
        <div>
          <div class="wind-card-eyebrow">Wind</div>
          <div class="wind-card-title">${hasSession ? 'Likely wind' : 'Wind estimate'}</div>
        </div>
      </div>
      <div class="wind-card-body">
        <div class="wind-card-compass-wrap">
          <div class="wind-card-compass ${hasSession ? '' : 'is-empty'}">
            <div class="wind-card-compass-north">N</div>
            <div class="wind-card-compass-arrow" style="transform: translate(-50%, -50%) rotate(${sessionArrowDeg}deg);"></div>
          </div>
          <div class="wind-card-compass-label">Session</div>
        </div>
        <div class="wind-card-metrics">
          ${hasSession ? `
            <div class="wind-card-main">${formatWindDirection(session.directionDeg)}</div>
            <div class="wind-card-sub">${session.speedKts.toFixed(1)} kt session wind</div>
            <div class="wind-card-local-row">
              <div class="wind-card-local-stack">
                <div class="wind-card-local-rose ${hasLocal ? '' : 'is-empty'}">
                  <div class="wind-card-local-arrow" style="transform: translate(-50%, -50%) rotate(${localArrowDeg}deg);"></div>
                </div>
                <div class="wind-card-local-label">Now</div>
              </div>
              <div class="wind-card-row">
                <span class="wind-card-row-key">Playhead wind</span>
                <span class="wind-card-row-val">${hasLocal ? `${formatWindDirection(localNow.directionDeg)} | ${localNow.speedKts.toFixed(1)} kt` : 'Move the playhead into the session'}</span>
              </div>
            </div>
          ` : `
            <div class="wind-card-empty">${emptyBody}</div>
          `}
        </div>
      </div>
    </div>
  `;
}

async function setTowFilteringDisabledForTrustedSession(disabled) {
  const nextValue = !!disabled;
  if (!!state.windConfig?.disableTowFilteringTrusted === nextValue) {
    renderWindMapControl();
    return;
  }
  state.windConfig = normalizeWindConfig({
    ...(state.windConfig || {}),
    disableTowFilteringTrusted: nextValue,
  });
  try {
    await saveProjectCvConfig();
  } catch (err) {
    console.warn('[wind] failed to persist tow-filter setting', err);
  }
  state.wind.byCsvId = {};
  state.wind.session = null;
  state.wind.localNow = null;
  state.wind._localNowKey = '';
  state.wind._lastLocalEvalTs = NaN;
  renderWindMapControl();
  await loadWindEstimates();
}

function renderWindMapLayer() {
  if (state.wind.layerGroup && state.map) state.map.removeLayer(state.wind.layerGroup);
  state.wind.layerGroup = null;
  return;
}

function updateCurrentWindEstimate(absTs = state.tl?.currentTs, { forceRender = false } = {}) {
  const currentTs = Number(absTs);
  const candidates = [];
  for (const [csvId, wind] of Object.entries(state.wind.byCsvId || {})) {
    const points = wind?.localSeries?.points;
    if (!Array.isArray(points) || !points.length || !Number.isFinite(currentTs)) continue;
    const point = findNearestWindPointByTs(points, currentTs);
    if (!point) continue;
    const maxGap = Math.max(Number(wind?.localSeries?.stepSeconds) * 1.5 || 0, Number(wind?.localSeries?.windowSeconds) * 0.65 || 0, 90);
    const delta = Math.abs(Number(point.ts) - currentTs);
    if (!Number.isFinite(delta) || delta > maxGap) continue;
    candidates.push({ ...point, csvId });
  }
  const key = candidates
    .map(item => `${item.csvId}:${Math.round(Number(item.ts) || 0)}`)
    .sort()
    .join('|');
  if (!forceRender && key && state.wind._localNowKey === key) return state.wind.localNow;
  state.wind._localNowKey = key;
  const combined = combineLocalWindEstimates(candidates);
  state.wind.localNow = combined ? {
    ...combined,
    sampleTs: candidates.length ? candidates[0].ts : null,
  } : null;
  renderWindMapControl();
  if (forceRender || !state.tl?.playing) updateTimelineStats();
  return state.wind.localNow;
}

function maybeRefreshWindEstimate(force = false) {
  const currentTs = Number(state.tl?.currentTs);
  if (!Number.isFinite(currentTs)) {
    if (force) updateCurrentWindEstimate(currentTs, { forceRender: true });
    return;
  }
  if (!force && Number.isFinite(state.wind._lastLocalEvalTs) && Math.abs(currentTs - state.wind._lastLocalEvalTs) < 12) return;
  state.wind._lastLocalEvalTs = currentTs;
  updateCurrentWindEstimate(currentTs, { forceRender: force });
}

function buildWindTrackSignature(track, pointCount = null) {
  const pointCountNum = Number.isFinite(Number(pointCount))
    ? Number(pointCount)
    : (Number.isFinite(Number(track?.point_count)) ? Number(track.point_count) : 0);
  return [
    String(track?.updated_at || ''),
    pointCountNum,
    Number(track?.ts_start) || 0,
    Number(track?.ts_end) || 0,
    !!state.windConfig?.disableTowFilteringTrusted ? 'tow-off' : 'tow-on',
  ].join('|');
}

async function loadWindEstimates() {
  if (state.wind.promise) return state.wind.promise;
  const promise = (async () => {
    const projectId = state.projectId;
    const csvs = state.mapData?.csvs || [];
    const token = ++state.wind.loadToken;
    state.wind.loading = csvs.length > 0;
    state.wind.byCsvId = {};
    state.wind.session = null;
    state.wind.localNow = null;
    state.wind._localNowKey = '';
    state.wind._lastLocalEvalTs = NaN;
    renderWindMapControl();
    renderWindMapLayer();
    if (!projectId || !csvs.length) {
      state.wind.loading = false;
      renderWindMapControl();
      return;
    }

    const tracks = await DB.listTracks(projectId, 'csv');
    const trackByFileId = new Map(tracks.map(track => [track.file_id, track]));
    for (const csv of csvs) {
      if (token !== state.wind.loadToken) return;
      const track = trackByFileId.get(csv.id);
      if (!track) continue;
      const points = await DB.getTrackPoints(track.id);
      if (token !== state.wind.loadToken) return;
      try {
        const signature = buildWindTrackSignature(track, points.length);
        let result = null;
        let usedCache = false;
        try {
          const cached = await DB.getWindEstimate(projectId, csv.id);
          if (cached && String(cached.signature || '') === signature && cached.estimate) {
            result = cached.estimate;
            usedCache = true;
          }
        } catch (cacheErr) {
          console.warn('[wind] cache lookup failed for', csv.filename || csv.id, cacheErr);
        }
        if (!result) {
          result = await estimateWindFromCsvPoints(points, {
            sourceName: csv.filename || csv.id,
            disableTowFiltering: !!state.windConfig?.disableTowFilteringTrusted,
          });
          if (result) {
            try {
              await DB.upsertWindEstimate(projectId, csv.id, signature, result);
            } catch (cacheWriteErr) {
              console.warn('[wind] cache write failed for', csv.filename || csv.id, cacheWriteErr);
            }
          }
        }
        if (token !== state.wind.loadToken) return;
        if (!result) continue;
        state.wind.byCsvId[csv.id] = {
          ...result,
          fileId: csv.id,
          filename: csv.filename,
        };
        if (usedCache) {
          console.log('[wind] using cached estimate for', csv.filename || csv.id);
        }
        state.wind.session = combineSessionWindEstimates(Object.values(state.wind.byCsvId));
        updateCurrentWindEstimate(state.tl?.currentTs, { forceRender: true });
        renderWindMapLayer();
      } catch (err) {
        console.warn('[wind] estimate failed for', csv.filename || csv.id, err);
      }
    }

    if (token !== state.wind.loadToken) return;
    state.wind.loading = false;
    state.wind.session = combineSessionWindEstimates(Object.values(state.wind.byCsvId));
    updateCurrentWindEstimate(state.tl?.currentTs, { forceRender: true });
    renderWindMapLayer();
    renderWindMapControl();
    if (projectId === state.projectId) void refreshManeuvers('wind-updated');
  })();
  state.wind.promise = promise;
  try {
    return await promise;
  } finally {
    if (state.wind.promise === promise) state.wind.promise = null;
  }
}

async function ensureWindEstimatesReady() {
  const csvs = state.mapData?.csvs || [];
  if (!csvs.length) return;
  if (state.wind.promise) {
    await state.wind.promise;
    return;
  }
  if (Object.keys(state.wind.byCsvId || {}).length > 0 || state.wind.session) return;
  await loadWindEstimates();
}

function buildReportWindContext() {
  const session = Number.isFinite(Number(state.wind.session?.directionDeg))
    ? {
        directionDeg: Number(state.wind.session.directionDeg),
        speedKts: Number(state.wind.session.speedKts),
        sourceCount: Number(state.wind.session.sourceCount) || 0,
        weight: Number(state.wind.session.weight) || 0,
      }
    : null;
  const byCsvId = {};
  for (const [csvId, wind] of Object.entries(state.wind.byCsvId || {})) {
    const localPoints = Array.isArray(wind?.localSeries?.points)
      ? wind.localSeries.points.map(point => ({
          ts: Number(point?.ts),
          directionDeg: Number(point?.directionDeg),
          speedKts: Number(point?.speedKts),
          fitScore: Number(point?.fitScore),
          inlierCount: Number(point?.inlierCount),
        })).filter(point => Number.isFinite(point.ts) && Number.isFinite(point.directionDeg) && Number.isFinite(point.speedKts))
      : [];
    byCsvId[String(csvId)] = {
      localSeries: {
        windowSeconds: Number(wind?.localSeries?.windowSeconds) || 0,
        stepSeconds: Number(wind?.localSeries?.stepSeconds) || 0,
        points: localPoints,
      },
    };
  }
  return { session, byCsvId };
}

function renderTrackPanel() {
  const list = el('track-panel-list');
  const hdr  = el('track-panel-hdr');
  list.innerHTML = '';
  const vids = state.mapData?.videos || [];
  const csvs = state.mapData?.csvs || [];
  const csvById = Object.fromEntries(csvs.map(c => [c.id, c]));
  const csvMatchedBy = {};
  for(const v of vids) {
    if(v.best_match_csv_id) {
      if(!csvMatchedBy[v.best_match_csv_id]) csvMatchedBy[v.best_match_csv_id] = [];
      csvMatchedBy[v.best_match_csv_id].push(v.filename || v.id);
    }
  }
  function addRow(track, kind) {
    const visible = state.trackVisibility[track.id] ?? true;
    const color = state.videoColors[track.id] || '#888';
    const name = track.filename || track.id;
    const day = computeDayLabel(track.ts_start);
    let linkHint = '';
    if(kind === 'vid' && track.best_match_csv_id) {
      const csv = csvById[track.best_match_csv_id];
      if(csv) linkHint = `<span class="trk-link" title="GPS from: ${csv.filename||csv.id}"><svg class="icon-svg" style="width:11px;height:11px"><use href="#ico-link"/></svg></span>`;
    } else if(kind === 'csv' && csvMatchedBy[track.id]?.length) {
      linkHint = `<span class="trk-link" title="Videos: ${csvMatchedBy[track.id].join(', ')}"><svg class="icon-svg" style="width:11px;height:11px"><use href="#ico-link"/></svg></span>`;
    }
    // Athlete tag for track panel
    const meta = state.fileMeta[track.id] || {};
    let athId = (kind === 'vid' && isPlaybackOnlyVideo(track)) ? null : meta.athlete_id;
    if(kind === 'vid' && !isPlaybackOnlyVideo(track) && !athId && track.best_match_csv_id) {
      const csvMeta = state.fileMeta[track.best_match_csv_id] || {};
      athId = csvMeta.athlete_id;
    }
    const ath = state.athletes.find(a=>a.id===athId);
    const playbackOnly = kind === 'vid' && isPlaybackOnlyVideo(track);
    const athName = playbackOnly ? 'Playback only' : (ath ? ath.name : '—');
    const athClass = playbackOnly ? 'trk-athlete' : 'trk-athlete clickable';
    const athTitle = playbackOnly ? playbackOnlyVideoTitle(track) : 'Assign athlete';
    const fileKind = kind === 'vid' ? 'video' : 'csv';

    const row = document.createElement('div');
    row.className = 'trk-row';
    row.innerHTML = `
      <span class="trk-eye ${visible?'on':''}" title="${visible?'Hide':'Show'} track"><svg class="icon-svg" style="width:14px;height:14px"><use href="#ico-eye"/></svg></span>
      <span class="trk-dot" style="background:${color}"></span>
      <span class="trk-name" title="${name}">${name}</span>
      ${linkHint}
      <span class="${athClass}" title="${athTitle}">${athName}</span>
      ${day ? `<span class="trk-day">${day}</span>` : ''}
      <span class="trk-badge ${kind}">${kind.toUpperCase()}</span>
    `;
    row.querySelector('.trk-eye').onclick = () => {
      state.trackVisibility[track.id] = !state.trackVisibility[track.id];
      renderTrackPanel();
      renderMap();
    };
    if (!playbackOnly) {
      row.querySelector('.trk-athlete').onclick = (e) => {
        openAssignPopup(track.id, e.target, fileKind);
      };
    }
    // Click track name to seek & play
    row.querySelector('.trk-name').onclick = () => {
      playTrack(track, kind);
    };
    // Highlight the active track
    if(state._activeTrackId === track.id) row.classList.add('active-track');
    list.appendChild(row);
  }
  for(const v of vids) addRow(v, 'vid');
  for(const csv of csvs) addRow(csv, 'csv');
  const count = vids.length + csvs.length;
  const panel = el('track-panel');
  const isCollapsed = panel.classList.contains('collapsed');
  hdr.textContent = (isCollapsed ? '▸' : '▾') + ` Tracks (${count})`;
  if(count > 0) panel.style.maxHeight = '';
}

/**
 * Click a track to seek the timeline to its start and begin playing.
 * For video tracks this loads the video; for CSV tracks it seeks to the GPS track start.
 */
function playTrack(track, kind) {
  state._activeTrackId = track.id;
  renderTrackPanel(); // highlight active row

  // Make sure the track is visible on the map
  if(state.trackVisibility[track.id] === false) {
    state.trackVisibility[track.id] = true;
    renderMap();
  }

  // Seek timeline to this track's start
  const ts = track.ts_start;
  if(ts != null && state.tl.globalStart != null) {
    tlPause();
    tlSeekTo(ts);
    // Small delay to let video load, then play
    setTimeout(() => tlPlay(), 150);
  }

  // Ensure the map view includes this track
  if(track.points?.length) {
    const lats = track.points.map(p => p.lat).filter(v => v != null);
    const lons = track.points.map(p => p.lon).filter(v => v != null);
    if(lats.length && lons.length) {
      const bounds = [
        [Math.min(...lats), Math.min(...lons)],
        [Math.max(...lats), Math.max(...lons)]
      ];
      fitMapToBounds(bounds, { padding: [36, 36], maxZoom: 15, storeBase: true });
    }
  }
}

async function loadAllSkeletonCoverages() {
  const vids = (state.mapData?.videos || []);
  for(const v of vids) {
    try {
      const coverage = await PoseEngine.getSkeletonCoverage(state.projectId, v.id);
      state.skeletonCoverage[v.id] = coverage.map(c => [c.start, c.end]);
    } catch {
      state.skeletonCoverage[v.id] = [];
    }
  }
}

async function refreshSkeletonCoverageForVideo(fileId, expectedRange = null) {
  if (!state.projectId || !fileId) return [];
  const expectedStart = Number(expectedRange?.startSec);
  const expectedEnd = Number(expectedRange?.endSec);
  const wantsRange = Number.isFinite(expectedStart) && Number.isFinite(expectedEnd) && expectedEnd > expectedStart;

  for (let attempt = 0; attempt < 5; attempt++) {
    let coverage = [];
    try {
      coverage = await PoseEngine.getSkeletonCoverage(state.projectId, fileId);
      state.skeletonCoverage[fileId] = coverage.map(c => [c.start, c.end]);
    } catch {
      state.skeletonCoverage[fileId] = [];
    }
    const intervals = state.skeletonCoverage[fileId] || [];
    const hasExpectedCoverage = wantsRange
      ? intervals.some(([s, e]) => e >= expectedStart - 0.5 && s <= expectedEnd + 0.5)
      : intervals.length > 0;
    if (hasExpectedCoverage) return intervals;
    await new Promise(resolve => setTimeout(resolve, 120));
  }
  return state.skeletonCoverage[fileId] || [];
}

function isInCoverage(fileId, videoSec) {
  if(videoSec == null) return false;
  const intervals = state.skeletonCoverage[fileId];
  if(!intervals || !intervals.length) return false;
  for(const [s, e] of intervals) {
    if(videoSec >= s - 0.5 && videoSec <= e + 0.5) return true;
  }
  return false;
}

async function loadAthletes() {
  if(!state.projectId) return;
  try { state.athletes = await DB.getAthletes(state.projectId); }
  catch(e){ state.athletes=[]; }

  if (state.athletes.length === 0) {
    const library = loadAthleteLibrary();
    if (library.length) {
      state.athletes = library.slice(0, 24).map((a, idx) => ({
        id: DB.uuid(),
        name: a.name,
        weight: Number.isFinite(Number(a.weight)) ? Number(a.weight) : null,
        height: Number.isFinite(Number(a.height)) ? Number(a.height) : null,
        color: a.color || nextPaletteColor(idx),
      }));
      await DB.saveAthletes(state.projectId, state.athletes);
    }
  }
}

async function saveAthletes() {
  if(!state.projectId) return;
  await DB.saveAthletes(state.projectId, state.athletes);
  updateAthleteLibraryFromCurrentState();
}

async function setFileMeta(fileId, meta) {
  if(!state.projectId) return;
  await DB.setFileMeta(state.projectId, fileId, meta);
  // Reload full meta dict
  state.fileMeta = await DB.getFileMeta(state.projectId);

  const csvTrack = (state.mapData?.csvs || []).find(c => c.id === fileId);
  if (csvTrack && meta?.athlete_id) {
    const athlete = findAthleteById(meta.athlete_id);
    if (athlete) rememberCsvAthlete(csvTrack.filename || csvTrack.name || '', athlete);
  }

  assignVideoColors();
  renderTrackPanel();
  renderMap();
  updateSetupUiState();
}

async function setCsvAthleteAndLinkedVideos(csvFileId, athleteId) {
  if (!state.projectId || !csvFileId) return;
  const nextAthleteId = athleteId || null;
  const updates = [];
  const csvMeta = state.fileMeta?.[csvFileId] || {};
  updates.push([csvFileId, { ...csvMeta, athlete_id: nextAthleteId }]);

  for (const video of (state.mapData?.videos || [])) {
    if (String(video?.best_match_csv_id || '') !== String(csvFileId)) continue;
    const videoMeta = state.fileMeta?.[video.id] || {};
    updates.push([
      video.id,
      {
        ...videoMeta,
        athlete_id: nextAthleteId,
        manual_athlete: false,
      },
    ]);
  }

  for (const [fileId, meta] of updates) {
    await DB.setFileMeta(state.projectId, fileId, meta);
  }
  state.fileMeta = await DB.getFileMeta(state.projectId);

  const csvTrack = (state.mapData?.csvs || []).find(c => String(c.id) === String(csvFileId));
  if (csvTrack && nextAthleteId) {
    const athlete = findAthleteById(nextAthleteId);
    if (athlete) rememberCsvAthlete(csvTrack.filename || csvTrack.name || '', athlete);
  }

  assignVideoColors();
  renderTrackPanel();
  renderMap();
  updateSetupUiState();
}

async function setVideoExternalMode(fileId, { external = false, athleteId = null } = {}) {
  if (!state.projectId || !fileId) return;
  const prevMeta = state.fileMeta?.[fileId] || {};
  const nextMeta = { ...prevMeta };

  // Claiming a video for analysis (external=false) sets force_analyze so the
  // pipeline's auto re-flag loop and playback-only heuristic leave it analyzable,
  // even with no matching CSV. Marking it External clears the protection.
  await DB.updateFileFields(fileId, { playback_only: !!external, force_analyze: !external });

  if (external) {
    nextMeta.athlete_id = null;
    nextMeta.manual_athlete = false;
  } else {
    nextMeta.athlete_id = athleteId || null;
    nextMeta.manual_athlete = !!athleteId;
  }

  await setFileMeta(fileId, nextMeta);
  await Pipeline.runMatching(state.projectId);
  await loadMapData();
  renderFilesList();
  renderTrackPanel();
  renderMap();
  buildTimeline();
  renderAssignList();
  updateSetupUiState();
  void refreshManeuvers('assignment-updated');
}

async function autoAssignCsvAthletesFromMemory() {
  if (!state.projectId || !state.mapData) return false;
  const csvs = state.mapData.csvs || [];
  if (!csvs.length) return false;

  let athletesChanged = false;
  let metaChanged = false;
  const pendingMeta = [];

  for (const csv of csvs) {
    const current = state.fileMeta[csv.id] || {};
    if (normalizeAthleteId(current.athlete_id)) continue;

    const profile = guessCsvAthleteProfile(csv.filename || csv.name || '');
    if (!profile?.name) continue;

    let athlete = findAthleteByName(profile.name);
    if (!athlete) {
      athlete = {
        id: DB.uuid(),
        name: profile.name,
        weight: Number.isFinite(Number(profile.weight)) ? Number(profile.weight) : null,
        height: Number.isFinite(Number(profile.height)) ? Number(profile.height) : null,
        color: profile.color || nextPaletteColor(state.athletes.length),
      };
      state.athletes.push(athlete);
      athletesChanged = true;
    } else {
      if (!athlete.color && profile.color) {
        athlete.color = profile.color;
        athletesChanged = true;
      }
      if (!Number.isFinite(Number(athlete.height)) && Number.isFinite(Number(profile.height))) {
        athlete.height = Number(profile.height);
        athletesChanged = true;
      }
    }

    pendingMeta.push([csv.id, { ...current, athlete_id: athlete.id }]);
    for (const video of (state.mapData?.videos || [])) {
      if (String(video?.best_match_csv_id || '') !== String(csv.id)) continue;
      const videoMeta = state.fileMeta[video.id] || {};
      if (videoMeta.manual_athlete) continue;
      pendingMeta.push([video.id, { ...videoMeta, athlete_id: athlete.id, manual_athlete: false }]);
    }
    rememberCsvAthlete(csv.filename || csv.name || '', athlete);
    metaChanged = true;
  }

  if (!athletesChanged && !metaChanged) return false;
  if (athletesChanged) await DB.saveAthletes(state.projectId, state.athletes);
  for (const [fileId, meta] of pendingMeta) {
    await DB.setFileMeta(state.projectId, fileId, meta);
  }
  state.fileMeta = await DB.getFileMeta(state.projectId);
  updateAthleteLibraryFromCurrentState();
  assignVideoColors();
  return true;
}

function getApiCsvAthleteName(csv) {
  const meta = csv?.track_meta || csv?.meta || {};
  return String(meta.api_custom_name || meta.api_athlete_name || meta.custom_name || '').trim();
}

async function autoAssignCsvAthletesFromApiMeta() {
  if (!state.projectId || !state.mapData) return false;
  const csvs = state.mapData.csvs || [];
  if (!csvs.length) return false;

  let athletesChanged = false;
  let metaChanged = false;
  const pendingMeta = [];

  for (const csv of csvs) {
    const current = state.fileMeta[csv.id] || {};
    if (normalizeAthleteId(current.athlete_id)) continue;
    const name = getApiCsvAthleteName(csv);
    if (!name) continue;

    let athlete = findAthleteByName(name);
    if (!athlete) {
      athlete = {
        id: DB.uuid(),
        name,
        weight: null,
        height: null,
        color: nextPaletteColor(state.athletes.length),
      };
      state.athletes.push(athlete);
      athletesChanged = true;
    }
    pendingMeta.push([csv.id, { ...current, athlete_id: athlete.id }]);
    for (const video of (state.mapData?.videos || [])) {
      if (String(video?.best_match_csv_id || '') !== String(csv.id)) continue;
      const videoMeta = state.fileMeta[video.id] || {};
      if (videoMeta.manual_athlete) continue;
      pendingMeta.push([video.id, { ...videoMeta, athlete_id: athlete.id, manual_athlete: false }]);
    }
    rememberCsvAthlete(csv.filename || csv.name || '', athlete);
    metaChanged = true;
  }

  if (!athletesChanged && !metaChanged) return false;
  if (athletesChanged) await DB.saveAthletes(state.projectId, state.athletes);
  for (const [fileId, meta] of pendingMeta) {
    await DB.setFileMeta(state.projectId, fileId, meta);
  }
  state.fileMeta = await DB.getFileMeta(state.projectId);
  updateAthleteLibraryFromCurrentState();
  assignVideoColors();
  return true;
}

// ── Map rendering ──────────────────────────────────────────────────────
function clearTrackLayers() {
  for(const lg of Object.values(state.trackLayers)) { if(state.map) state.map.removeLayer(lg); }
  state.trackLayers = {};
  if (state.wind.layerGroup && state.map) state.map.removeLayer(state.wind.layerGroup);
  state.wind.layerGroup = null;
  for(const entry of Object.values(state.posMarkers)) {
    const m = getPositionMarkerLayer(entry);
    if(state.map && m) state.map.removeLayer(m);
  }
  state.posMarkers = {};
  // Clear segment highlight layers
  if(state.segmentLayerGroup && state.map) { state.map.removeLayer(state.segmentLayerGroup); }
  state.segmentLayerGroup = null;
  if(state.maneuverLayerGroup && state.map) { state.map.removeLayer(state.maneuverLayerGroup); }
  state.maneuverLayerGroup = null;
  // Note: segment selection markers are NOT cleared here — they persist across re-renders
}

function getPositionMarkerLayer(entry) {
  if (!entry) return null;
  if (typeof entry === 'object' && 'marker' in entry) return entry.marker || null;
  return entry;
}

function createArrowMarker(pts, color) {
  if (!state.map || !pts?.length) return null;
  const arrowHtml = `<div style="width:20px;height:20px;display:flex;align-items:center;justify-content:center;">
    <svg width="20" height="20" viewBox="0 0 20 20"><polygon points="10,2 4,18 10,13 16,18" fill="${color}" stroke="white" stroke-width="1.2"/></svg>
  </div>`;
  const arrowIcon = L.divIcon({html:arrowHtml,className:'',iconSize:[20,20],iconAnchor:[10,10]});
  return L.marker([pts[0].lat,pts[0].lon],{
    icon:arrowIcon,
    zIndexOffset:500,
    interactive:false,
    keyboard:false,
  }).addTo(state.map);
}

function ensurePositionMarker(fileId) {
  const entry = state.posMarkers[fileId];
  if (!entry) return null;
  if (entry.marker) return entry.marker;
  if (!entry.pts?.length || !entry.color) return null;
  entry.marker = createArrowMarker(entry.pts, entry.color);
  return entry.marker;
}

function syncActivePositionMarkers(activeFileIds) {
  const heavyProject = isIpadHeavyVideoProject();
  for (const [fid, entry] of Object.entries(state.posMarkers)) {
    if (!entry) continue;
    if (heavyProject && !activeFileIds.has(fid)) {
      if (entry.marker && state.map) state.map.removeLayer(entry.marker);
      entry.marker = null;
      continue;
    }
    const marker = activeFileIds.has(fid) ? ensurePositionMarker(fid) : getPositionMarkerLayer(entry);
    if (!marker) continue;
    const markerEl = marker.getElement?.();
    if (markerEl) markerEl.style.display = activeFileIds.has(fid) ? '' : 'none';
  }
}

function fitMapToBounds(latLngs, { padding = [48, 48], maxZoom = 14, storeBase = true } = {}) {
  if (!state.map || !latLngs?.length) return;
  try {
    const bounds = Array.isArray(latLngs) ? L.latLngBounds(latLngs) : latLngs;
    state.map.fitBounds(bounds, { padding, maxZoom, animate: false });
    if (storeBase) state.mapBaseZoom = state.map.getZoom();
  } catch {}
}

function shouldFollowPlaybackOnMap() {
  return false;
}

function keepPlaybackMarkersInView(activeFileIds) {
  if (!state.map || !shouldFollowPlaybackOnMap() || !activeFileIds?.size) return;
  const markers = [];
  for (const fileId of activeFileIds) {
    const marker = ensurePositionMarker(fileId);
    const latLng = marker?.getLatLng?.();
    if (latLng) markers.push(latLng);
  }
  if (!markers.length) return;
  const innerBounds = state.map.getBounds().pad(-0.2);
  if (markers.every(latLng => innerBounds.contains(latLng))) return;
  const target = markers.length === 1 ? markers[0] : L.latLngBounds(markers).getCenter();
  state.map.panTo(target, { animate: false });
}

function getVideoMapPoints(video, csvTracks = []) {
  const directPts = Array.isArray(video?.points)
    ? video.points.filter(point => (
      Number.isFinite(Number(point?.lat)) &&
      Number.isFinite(Number(point?.lon)) &&
      Number.isFinite(Number(point?.ts))
    ))
    : [];
  if (directPts.length >= 2) return directPts;
  if (!isPlaybackOnlyVideo(video)) return directPts;

  const csvs = Array.isArray(csvTracks) ? csvTracks : [];
  let candidates = [];
  if (video?.best_match_csv_id) {
    const matched = csvs.find(csv => String(csv?.id || '') === String(video.best_match_csv_id || ''));
    if (matched) candidates.push(matched);
  }
  if (!candidates.length) {
    const start = Number(video?.ts_start);
    const end = Number(video?.ts_end);
    candidates = csvs.filter(csv => {
      const csvStart = Number(csv?.ts_start);
      const csvEnd = Number(csv?.ts_end);
      if (!Number.isFinite(start) || !Number.isFinite(end)) return false;
      if (!Number.isFinite(csvStart) || !Number.isFinite(csvEnd)) return false;
      return end >= csvStart && start <= csvEnd;
    });
  }

  const start = Number(video?.ts_start);
  const end = Number(video?.ts_end);
  const sliceToVideoRange = points => {
    if (!Number.isFinite(start) || !Number.isFinite(end)) return points;
    const ranged = points.filter(point => Number(point.ts) >= start && Number(point.ts) <= end);
    return ranged.length >= 2 ? ranged : points;
  };
  let best = [];
  for (const csv of candidates) {
    const valid = validLatLonPoints(csv?.points).filter(point => Number.isFinite(Number(point?.ts)));
    if (valid.length < 2) continue;
    const sliced = sliceToVideoRange(valid);
    if (sliced.length > best.length) best = sliced;
  }
  return best;
}

function renderMap() {
  if(!state.map || !state.mapData) return;
  clearTrackLayers();
  const bounds = [];
  const csvs = state.mapData.csvs || [];
  const vids = state.mapData.videos || [];
  const heavyProject = isIpadHeavyVideoProject(vids.length);

  // CSV cut-ranges
  const csvCutRanges = {};
  for(const v of vids) {
    if(!v.best_match_csv_id || !v.points?.length) continue;
    const tss = v.points.map(p => p.ts).filter(t => t != null);
    if(!tss.length) continue;
    const cid = v.best_match_csv_id;
    if(!csvCutRanges[cid]) csvCutRanges[cid] = [];
    csvCutRanges[cid].push({ts_min: Math.min(...tss), ts_max: Math.max(...tss)});
  }
  function isCutTs(csvId, ts) {
    if(!ts) return false;
    return (csvCutRanges[csvId] || []).some(r => ts >= r.ts_min && ts <= r.ts_max);
  }
  function drawSegmented(pts, lg, getStyle, clickH) {
    if(!pts.length) return;
    let run = [pts[0]], runStyle = JSON.stringify(getStyle(pts[0]));
    for(let i = 1; i < pts.length; i++) {
      const s = JSON.stringify(getStyle(pts[i]));
      if(s !== runStyle) {
        if(run.length > 1) L.polyline(run.map(p=>[p.lat,p.lon]), JSON.parse(runStyle)).on('click',clickH).addTo(lg);
        run = [pts[i-1], pts[i]]; runStyle = s;
      } else { run.push(pts[i]); }
    }
    if(run.length > 1) L.polyline(run.map(p=>[p.lat,p.lon]), JSON.parse(runStyle)).on('click',clickH).addTo(lg);
  }

  function makeClickHandler(pts, v) {
    return function(e) {
      const ll = e.latlng;
      let bestIdx=0, bestDist=Infinity;
      pts.forEach((p,i)=>{ const d=(p.lat-ll.lat)**2+(p.lon-ll.lng)**2; if(d<bestDist){bestDist=d;bestIdx=i;} });
      const nearest = pts[bestIdx];
      if(state.segmentSelect.active && nearest.ts != null) {
        handleSegmentClick(nearest.ts, [nearest.lat, nearest.lon]);
      } else if(nearest.ts != null) {
        tlSeekTo(nearest.ts);
      }
    };
  }

  // CSV tracks
  for(const csv of csvs) {
    const csvMapPts = getCsvMapPoints(csv, vids);
    if(!csvMapPts.length) continue;
    if(state.trackVisibility[csv.id] === false) continue;
    const color = state.videoColors[csv.id] || '#2ea043';
    const lg = L.layerGroup();
    const clickH = makeClickHandler(csvMapPts, csv);
    const hasCuts = !!(csvCutRanges[csv.id]?.length);
    const coveredStyle = {color, weight:3, opacity:0.82, pane:'baseTrackPane'};
    const uncoveredStyle = {color:'#8a9099', weight:2, opacity:0.55, dashArray:'4 6', pane:'baseTrackPane'};
    if(hasCuts) {
      drawSegmented(csvMapPts, lg, p => (
        isCutTs(csv.id, p.ts)
          ? coveredStyle
          : uncoveredStyle
      ), clickH);
    } else {
      L.polyline(csvMapPts.map(p=>[p.lat,p.lon]),
        uncoveredStyle).on('click',clickH).addTo(lg);
    }
    bounds.push(...csvMapPts.map(p=>[p.lat,p.lon]));
    lg.addTo(state.map);
    state.trackLayers[csv.id] = lg;
  }

  // Video tracks
  for(const v of vids) {
    // A video's own GoPro GPS track should always render on the map — even if the
    // video is classified playback_only/external. Only skip a playback-only video
    // when it has no GPS of its own (getVideoMapPoints then returns no direct pts,
    // so a phone clip without telemetry stays off the map).
    const hasOwnGps = Array.isArray(v.points) && v.points.length >= 2;
    if (v.playback_only && !hasOwnGps) continue;
    const pts = getVideoMapPoints(v, csvs);
    if(!pts.length) continue;
    const color = state.videoColors[v.id] || '#1d8fd8';
    const lg = heavyProject ? null : L.layerGroup();
    const clickH = heavyProject ? null : makeClickHandler(pts, v);
    const hasDirectGps = Array.isArray(v.points) && v.points.length >= 2;
    const videoTrackStyle = hasDirectGps
      ? {color, weight:4, opacity:0.85, pane:'baseTrackPane'}
      : {color, weight:3, opacity:0.78, dashArray:'6 5', pane:'baseTrackPane'};

    if(!heavyProject && state.trackVisibility[v.id] !== false) {
      L.polyline(pts.map(p=>[p.lat,p.lon]),
        videoTrackStyle).on('click',clickH).addTo(lg);
      bounds.push(...pts.map(p=>[p.lat,p.lon]));
    } else if (heavyProject) {
      bounds.push([pts[0].lat, pts[0].lon], [pts[pts.length - 1].lat, pts[pts.length - 1].lon]);
    } else {
      bounds.push([pts[0].lat, pts[0].lon], [pts[pts.length - 1].lat, pts[pts.length - 1].lon]);
    }

    state.posMarkers[v.id] = {marker: null, pts, color};
    if (!heavyProject) {
      lg.addTo(state.map);
      state.trackLayers[v.id] = lg;
    }
  }
  renderWindMapLayer();

  // ── Segment highlights ──────────────────────────────────────────────
  // For each saved segment, draw thick athlete-colored polylines on matching tracks
  // with a single label per segment. All layers go into a tracked group.
  if (!heavyProject) {
    const segGroup = L.layerGroup();
    const csvTrackEntries = csvs.map(c => ({id: c.id, points: getCsvMapPoints(c, vids)})).filter(e => e.points.length);
    const videoTrackEntries = vids.map(v => ({id: v.id, points: v.points || []})).filter(e => e.points.length);
    for(const seg of state.segments) {
      const tsS = seg.tsStart, tsE = seg.tsEnd;
      if(tsS == null || tsE == null) continue;
      const segColor = getSegmentColor(seg);
      let bestSegPts = null, bestLen = 0;
      let segmentTrackEntries = csvTrackEntries;
      const hasCsvCoverage = csvTrackEntries.some(entry => {
        for (const point of entry.points) {
          const ts = Number(point?.ts);
          if (Number.isFinite(ts) && ts >= tsS && ts <= tsE) return true;
        }
        return false;
      });
      if (!hasCsvCoverage) segmentTrackEntries = videoTrackEntries;
      // First pass: draw all track highlights and find longest for label placement
      for(const entry of segmentTrackEntries) {
        const color = state.videoColors[entry.id] || segColor;
        const segPts = entry.points.filter(p => p.ts != null && p.ts >= tsS && p.ts <= tsE);
        if(segPts.length < 2) continue;
        // Thick highlighted line in athlete color with click-to-seek
        const poly = L.polyline(segPts.map(p=>[p.lat,p.lon]),{
          color, weight:11, opacity:0.88, pane:'processedTrackPane'
        });
        poly.on('click', function(e) {
          const ll = e.latlng;
          let bestIdx=0, bestDist=Infinity;
          segPts.forEach((p,i)=>{ const d=(p.lat-ll.lat)**2+(p.lon-ll.lng)**2; if(d<bestDist){bestDist=d;bestIdx=i;} });
          const nearest = segPts[bestIdx];
          if(state.segmentSelect.active && nearest.ts != null) {
            handleSegmentClick(nearest.ts, [nearest.lat, nearest.lon]);
          } else if(nearest.ts != null) {
            tlSeekTo(nearest.ts);
          }
        });
        poly.addTo(segGroup);
        if(segPts.length > bestLen) { bestLen = segPts.length; bestSegPts = segPts; }
      }
      // Single label at midpoint of longest matching track
      if(bestSegPts) {
        const midIdx = Math.floor(bestSegPts.length / 2);
        const midPt = bestSegPts[midIdx];
        const labelHtml = `<div class="seg-label-icon" style="color:${segColor};">${seg.name}</div>`;
        const icon = L.divIcon({html:labelHtml, className:'', iconSize:null, iconAnchor:[0,26]});
        L.marker([midPt.lat, midPt.lon], {icon, pane:'splitMarkersPane'}).addTo(segGroup);

        const startPt = bestSegPts[0];
        const endPt = bestSegPts[bestSegPts.length - 1];
        L.circleMarker([startPt.lat, startPt.lon], {
          radius: 7,
          color: '#ffffff',
          weight: 2.5,
          fillColor: '#2ea043',
          fillOpacity: 1,
          pane: 'splitMarkersPane',
        }).addTo(segGroup);
        L.circleMarker([endPt.lat, endPt.lon], {
          radius: 7,
          color: '#ffffff',
          weight: 2.5,
          fillColor: '#e3342f',
          fillOpacity: 1,
          pane: 'splitMarkersPane',
        }).addTo(segGroup);
      }
    }
    segGroup.addTo(state.map);
    state.segmentLayerGroup = segGroup;
  }

  const activeFileIds = new Set();
  for (const slot of (state.tl?.athleteSlots || [])) {
    if (slot.currentFileId) activeFileIds.add(slot.currentFileId);
    else if (Number.isFinite(state.tl?.currentTs)) {
      const active = findSlotVideoAtTime(slot, state.tl.currentTs);
      if (active?.id) activeFileIds.add(active.id);
    }
  }
  syncActivePositionMarkers(activeFileIds);

  if(bounds.length > 0 && state.mapShouldAutoFit) {
    fitMapToBounds(bounds, { padding: [52, 52], maxZoom: 14, storeBase: true });
    state.mapShouldAutoFit = false;
  }
  renderWindMapControl();
}

function movePositionMarker(fileId, videoSec) {
  const entry = state.posMarkers[fileId]; if(!entry) return;
  const m = ensurePositionMarker(fileId);
  const pts = entry.pts ?? (state.mapData?.videos||[]).find(v=>v.id===fileId)?.points;
  if(!m || !pts?.length) return;
  let bestIdx=0;
  for(let i=0;i<pts.length;i++){ if(pts[i].video_s<=videoSec) bestIdx=i; else break; }
  const best=pts[bestIdx];
  if(!best) return;
  m.setLatLng([best.lat, best.lon]);
  const next = pts[Math.min(bestIdx+1, pts.length-1)];
  const hdg = (next && next!==best) ? bearing(best.lat,best.lon,next.lat,next.lon) : 0;
  const markerEl = m.getElement();
  if(markerEl) {
    const svgEl = markerEl.querySelector('svg');
    if(svgEl) svgEl.style.transform = `rotate(${hdg}deg)`;
  }
}

function movePositionMarkerByTs(fileId, absTs) {
  const entry = state.posMarkers[fileId]; if(!entry) return;
  const m = ensurePositionMarker(fileId);
  const pts = entry.pts;
  if(!m || !pts?.length) return;
  let bestIdx=0;
  for(let i=0;i<pts.length;i++){ if(pts[i].ts<=absTs) bestIdx=i; else break; }
  const best=pts[bestIdx];
  if(!best) return;
  m.setLatLng([best.lat, best.lon]);
  const next = pts[Math.min(bestIdx+1, pts.length-1)];
  const hdg = (next && next!==best) ? bearing(best.lat,best.lon,next.lat,next.lon) : 0;
  const markerEl = m.getElement();
  if(markerEl) {
    const svgEl = markerEl.querySelector('svg');
    if(svgEl) svgEl.style.transform = `rotate(${hdg}deg)`;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// ██  SHARED TIMELINE + DUAL VIDEO VIEWER
// ═══════════════════════════════════════════════════════════════════════

function showNoVideo(show) {
  const nv = el('no-video');
  const vg = el('video-grid');
  const title = el('no-video-title');
  const subtitle = el('no-video-subtitle');
  const playbackCount = getPlaybackTimelineMedia().length;
  const timelineSlotCount = state.tl?.athleteSlots?.length || 0;
  if (title) {
    title.textContent = playbackCount > 0
      ? 'Playback-only phone media is available on the timeline'
      : (timelineSlotCount > 0
        ? 'CSV timeline data is available'
        : 'Upload videos and assign athletes to begin');
  }
  if (subtitle) {
    subtitle.textContent = playbackCount > 0
      ? `Use the External Video toggle in the top bar to arm synced playback.${timelineSlotCount > 0 ? ' CSV speed and live stats still run below.' : ''}`
      : (timelineSlotCount > 0
        ? 'CSV speed and live stats are still available in the timeline below.'
        : '');
  }
  if(nv) nv.style.display = show ? 'flex' : 'none';
  if(vg) vg.style.display = show ? 'none' : 'grid';
  if (!show) applyVideoPaneGridPositions();
}

function destroyVideoSlots() {
  const grid = el('video-grid');
  if(grid) grid.innerHTML = '';
  for(const slot of state.tl.athleteSlots) {
    stopSlotPlayback(slot, true);
  }
  hidePhonePlayback(false, false);
  closeInlineHeatmapPlotOverlay();
  if (state.inlineHeatmaps) {
    state.inlineHeatmaps.loadToken++;
    state.inlineHeatmaps.loading = false;
    state.inlineHeatmaps.segmentId = null;
    state.inlineHeatmaps.results = null;
    state.inlineHeatmaps.renderedSegmentId = null;
    state.inlineHeatmaps.renderedLoadToken = 0;
    state.inlineHeatmaps.previousVideoLayout = null;
  }
  state.tl.athleteSlots = [];
  state.tl.viewStart = null;
  state.tl.viewEnd = null;
  state.tl._lastPrewarmAt = 0;
  state.tl._pendingProcessedMetricRefresh = false;
  pruneVideoURLCache();
}

function getTimelineSlotOverlayPrefKey(slot) {
  if (!slot) return null;
  const athleteId = normalizeAthleteId(slot.athleteId);
  if (athleteId) return `ath:${athleteId}`;
  const videoId = String(slot.videos?.[0]?.id || slot.currentFileId || slot.name || '').trim();
  return videoId ? `video:${videoId}` : null;
}

function getTimelineSlotOverlayMetricSet(slot) {
  const prefKey = getTimelineSlotOverlayPrefKey(slot);
  const saved = prefKey ? state.timelineMetricOverlayPrefs?.[prefKey] : null;
  return new Set(Array.isArray(saved) ? saved : []);
}

function syncTimelineSlotOverlayMetricPrefs(slot) {
  const prefKey = getTimelineSlotOverlayPrefKey(slot);
  if (!prefKey) return;
  const next = [...(slot._selectedTimelineMetrics || new Set())];
  if (!next.length) delete state.timelineMetricOverlayPrefs[prefKey];
  else state.timelineMetricOverlayPrefs[prefKey] = next;
}

function getUnifiedTimelineOverlayMetricSet(slots = state.tl?.athleteSlots || []) {
  const validKeys = new Set(TIMELINE_STAT_DEFS.map(def => def.key));
  const selected = new Set();
  for (const slot of slots) {
    const metricSet = slot?._selectedTimelineMetrics instanceof Set
      ? slot._selectedTimelineMetrics
      : new Set();
    for (const key of metricSet) {
      if (validKeys.has(key)) selected.add(key);
    }
  }
  return selected;
}

function applyUnifiedTimelineOverlayMetricSet(metricKeys, slots = state.tl?.athleteSlots || []) {
  const validKeys = new Set(TIMELINE_STAT_DEFS.map(def => def.key));
  const normalized = new Set();
  for (const key of metricKeys || []) {
    if (validKeys.has(key)) normalized.add(key);
  }
  for (const slot of slots) {
    if (!slot) continue;
    slot._selectedTimelineMetrics = new Set(normalized);
    syncTimelineSlotOverlayMetricPrefs(slot);
  }
  return normalized;
}

function syncTimelineOverlayMetricSelectionAcrossSlots(slots = state.tl?.athleteSlots || []) {
  return applyUnifiedTimelineOverlayMetricSet(getUnifiedTimelineOverlayMetricSet(slots), slots);
}

/**
 * Build the shared timeline from map data.
 * Groups videos by athlete, computes global time range,
 * creates video panes and draws SOG canvas.
 */
function buildTimeline() {
  const tl = state.tl;
  destroyVideoSlots();

  const allTimelineVideos = state.mapData?.videos || [];
  const vids = allTimelineVideos.filter(v => !isPlaybackOnlyVideo(v));
  const csvs = state.mapData?.csvs || [];
  if(!allTimelineVideos.length && !csvs.length) {
    tl.athleteSlots = [];
    showNoVideo(true);
    drawSogCanvas();
    populateTimelineStats();
    return;
  }

  // Gather all timestamps for global range — use VIDEO file ranges only
  // so the scrub bar shows only the time span covered by video files
  const allTs = [];
  for(const v of allTimelineVideos) {
    if(v.ts_start != null) allTs.push(v.ts_start);
    if(v.ts_end != null) {
      allTs.push(v.ts_end);
    } else if(v.ts_start != null) {
      // ts_end missing (duration probe failed) — synthesize from duration_sec or default
      const dur = v.duration_sec || 600; // 10 min fallback
      allTs.push(v.ts_start + dur);
    }
  }
  const csvTs = [];
  for(const c of csvs) {
    if(c.ts_start != null) csvTs.push(c.ts_start);
    if(c.ts_end != null) csvTs.push(c.ts_end);
  }
  // Fallback: if no videos have timestamps, use CSV timestamps
  if(!allTs.length) {
    allTs.push(...csvTs);
  }
  if(!allTs.length) {
    tl.athleteSlots = [];
    showNoVideo(true);
    drawSogCanvas();
    populateTimelineStats();
    return;
  }
  // IQR-based outlier rejection (removes GPS warm-up junk timestamps)
  allTs.sort((a,b) => a - b);
  const q1 = allTs[Math.floor(allTs.length * 0.25)];
  const q3 = allTs[Math.floor(allTs.length * 0.75)];
  const iqr = q3 - q1;
  const fence = Math.max(iqr * 3, 3600); // at least 1 hour fence
  const loFence = q1 - fence;
  const hiFence = q3 + fence;
  const filtered = allTs.filter(t => t >= loFence && t <= hiFence);
  let gMin = filtered.length ? filtered[0] : allTs[0];
  let gMax = filtered.length ? filtered[filtered.length - 1] : allTs[allTs.length - 1];
  // If video-derived range is clearly wrong compared to CSV range, trust CSV.
  if (csvTs.length >= 2) {
    const csvSorted = [...csvTs].sort((a, b) => a - b);
    const cMin = csvSorted[0];
    const cMax = csvSorted[csvSorted.length - 1];
    const cSpan = cMax - cMin;
    const vSpan = gMax - gMin;
    const disjoint = gMax < (cMin - 3600) || gMin > (cMax + 3600);
    const videoMuchWider = cSpan > 0 && vSpan > Math.max(cSpan * 8, 12 * 3600);
    if ((disjoint || videoMuchWider) && Number.isFinite(cMin) && Number.isFinite(cMax) && cMax > cMin) {
      console.warn(`[Timeline] using CSV range due to suspicious video range: video=${gMin}..${gMax}, csv=${cMin}..${cMax}`);
      gMin = cMin;
      gMax = cMax;
    }
  }
  // Safety cap: if total span > 48 hours, something is wrong — fallback to
  // CSV range, individual video durations, or a hard cap.
  const MAX_REASONABLE_SPAN = 48 * 3600; // 48 hours
  if ((gMax - gMin) > MAX_REASONABLE_SPAN) {
    // Try CSV range first
    if (csvTs.length >= 2) {
      const csvSorted = [...csvTs].sort((a, b) => a - b);
      gMin = csvSorted[0];
      gMax = csvSorted[csvSorted.length - 1];
      console.warn(`[Timeline] span exceeded 48h cap, using CSV range: ${gMin}..${gMax}`);
    } else {
      // Use the tightest cluster of video timestamps (median ± total video duration)
      const totalDur = allTimelineVideos.reduce((s, v) => s + (v.duration_sec || 600), 0);
      const midTs = allTs[Math.floor(allTs.length / 2)];
      gMin = midTs - totalDur / 2;
      gMax = midTs + totalDur / 2;
      console.warn(`[Timeline] span exceeded 48h cap, using median cluster: ${gMin}..${gMax}`);
    }
  }
  if(!isFinite(gMin) || !isFinite(gMax)) {
    tl.athleteSlots = [];
    showNoVideo(true);
    drawSogCanvas();
    populateTimelineStats();
    return;
  }
  tl.globalStart = gMin;
  tl.globalEnd = gMax;
  if(tl.currentTs == null || tl.currentTs < gMin || tl.currentTs > gMax) tl.currentTs = gMin;

  // Group videos by athlete (derived from matched CSV's metadata)
  const athleteVids = new Map(); // athleteId -> [{vid, meta, pts}]
  const unassigned = [];
  for(const v of vids) {
    const meta = state.fileMeta[v.id] || {};
    let athId = isPlaybackOnlyVideo(v) ? null : normalizeAthleteId(meta.athlete_id);
    // Derive from matched CSV if video itself has no athlete
    if(!athId && !isPlaybackOnlyVideo(v) && v.best_match_csv_id) {
      const csvMeta = state.fileMeta[v.best_match_csv_id] || {};
      athId = normalizeAthleteId(csvMeta.athlete_id);
    }
    if(athId) {
      if(!athleteVids.has(athId)) athleteVids.set(athId, []);
      athleteVids.get(athId).push(v);
    } else {
      unassigned.push(v);
    }
  }
  const representedCsvIds = new Set();
  for (const v of vids) {
    if (isPlaybackOnlyVideo(v)) continue;
    if (v?.best_match_csv_id) representedCsvIds.add(v.best_match_csv_id);
  }
  const assignedCsvsByAthlete = new Map();
  for (const csv of csvs) {
    const csvMeta = state.fileMeta[csv.id] || {};
    const athId = normalizeAthleteId(csvMeta.athlete_id);
    if (!athId) continue;
    if (!assignedCsvsByAthlete.has(athId)) assignedCsvsByAthlete.set(athId, []);
    assignedCsvsByAthlete.get(athId).push(csv);
  }

  const athleteCsvs = new Map();
  const unassignedCsvs = [];
  for (const csv of csvs) {
    if (!csv || representedCsvIds.has(csv.id)) continue;
    const csvMeta = state.fileMeta[csv.id] || {};
    const athId = normalizeAthleteId(csvMeta.athlete_id);
    if (athId && !athleteVids.has(athId)) {
      if (!athleteCsvs.has(athId)) athleteCsvs.set(athId, []);
      athleteCsvs.get(athId).push(csv);
    } else if (!athId) {
      unassignedCsvs.push(csv);
    }
  }

  // Build athlete slots: one pane per athlete (or per unassigned video)
  const slots = [];
  for(const [athId, avids] of athleteVids) {
    const ath = findAthleteById(athId);
    const athIdx = state.athletes.findIndex(a => normalizeAthleteId(a?.id) === normalizeAthleteId(athId));
    const color = ath?.color || (athIdx >= 0 ? PALETTE[athIdx % PALETTE.length] : (state.videoColors[avids[0]?.id] || PALETTE[0]));
    // Sort videos by ts_start
    avids.sort((a,b)=>(a.ts_start||0)-(b.ts_start||0));
    // Keep the live timeline CSV-first like the original viewer: matched CSV
    // drives the smooth speed curve, with video GPS only as a fallback.
    const extraCsvTracks = [];
    const sogSourcePts = collectSogForAthlete(athId, avids, extraCsvTracks);
    const windCsvIds = collectSlotWindCsvIds(avids, assignedCsvsByAthlete.get(athId) || []);
    const sogPts = optimizeSogPointsForDevice(sogSourcePts);
    slots.push({
      athleteId: athId,
      name: ath?.name || 'Unknown',
      color,
      videos: avids,
      csvTracks: extraCsvTracks,
      windCsvIds,
      sogPts,
      statsSeries: {
        sog: buildNumericSeries(sogSourcePts, 'sog', TIMELINE_TRACK_SERIES_OPTS),
        motionDir: buildNumericSeries(collectMotionDirectionForAthlete(avids, extraCsvTracks), 'motionDir', TIMELINE_TRACK_SERIES_OPTS),
        heading: buildNumericSeries(collectTrackMetricForAthlete(avids, 'hdg', extraCsvTracks), 'hdg', TIMELINE_TRACK_SERIES_OPTS),
        cog: buildNumericSeries(collectTrackMetricForAthlete(avids, 'cog', extraCsvTracks), 'cog', TIMELINE_TRACK_SERIES_OPTS),
        heel: buildNumericSeries(collectTrackMetricForAthlete(avids, 'heel', extraCsvTracks), 'heel', TIMELINE_TRACK_SERIES_OPTS),
        pitch: buildNumericSeries(collectTrackMetricForAthlete(avids, 'trim', extraCsvTracks), 'trim', TIMELINE_TRACK_SERIES_OPTS),
        roll: buildNumericSeries([], 'roll_moment'),
        trunk: buildNumericSeries([], 'trunk_angle'),
        rudder: buildNumericSeries([], 'rudder_angle'),
        boom: buildNumericSeries([], 'boom_angle'),
      },
      _selectedTimelineMetrics: null,
      currentFileId: null,
      _lastVideoIdx: null,
      videoEl: null,
      paneEl: null,
    });
  }
  for (const [athId, athleteCsvTracks] of athleteCsvs) {
    const ath = findAthleteById(athId);
    const athIdx = state.athletes.findIndex(a => normalizeAthleteId(a?.id) === normalizeAthleteId(athId));
    const color = ath?.color || (athIdx >= 0 ? PALETTE[athIdx % PALETTE.length] : (state.videoColors[athleteCsvTracks[0]?.id] || PALETTE[slots.length % PALETTE.length]));
    athleteCsvTracks.sort((a, b) => (a.ts_start || 0) - (b.ts_start || 0));
    const sogSourcePts = collectSogForCsvTracks(athleteCsvTracks);
    const sogPts = optimizeSogPointsForDevice(sogSourcePts);
    slots.push({
      athleteId: athId,
      name: ath?.name || athleteCsvTracks[0]?.filename || 'Unknown',
      color,
      videos: [],
      windCsvIds: athleteCsvTracks.map(csv => csv.id),
      sogPts,
      statsSeries: {
        sog: buildNumericSeries(sogSourcePts, 'sog', TIMELINE_TRACK_SERIES_OPTS),
        motionDir: buildNumericSeries(collectMotionDirectionForCsvTracks(athleteCsvTracks), 'motionDir', TIMELINE_TRACK_SERIES_OPTS),
        heading: buildNumericSeries(collectTrackMetricForCsvTracks(athleteCsvTracks, 'hdg'), 'hdg', TIMELINE_TRACK_SERIES_OPTS),
        cog: buildNumericSeries(collectTrackMetricForCsvTracks(athleteCsvTracks, 'cog'), 'cog', TIMELINE_TRACK_SERIES_OPTS),
        heel: buildNumericSeries(collectTrackMetricForCsvTracks(athleteCsvTracks, 'heel'), 'heel', TIMELINE_TRACK_SERIES_OPTS),
        pitch: buildNumericSeries(collectTrackMetricForCsvTracks(athleteCsvTracks, 'trim'), 'trim', TIMELINE_TRACK_SERIES_OPTS),
        roll: buildNumericSeries([], 'roll_moment'),
        trunk: buildNumericSeries([], 'trunk_angle'),
        rudder: buildNumericSeries([], 'rudder_angle'),
        boom: buildNumericSeries([], 'boom_angle'),
      },
      _selectedTimelineMetrics: null,
      currentFileId: null,
      _lastVideoIdx: null,
      videoEl: null,
      paneEl: null,
    });
  }
  // Unassigned videos each get their own slot
  for(let i = 0; i < unassigned.length; i++) {
    const v = unassigned[i];
    const color = state.videoColors[v.id] || PALETTE[(slots.length+i) % PALETTE.length];
    const sogSourcePts = collectSogForVideo(v);
    const windCsvIds = collectSlotWindCsvIds([v]);
    const sogPts = optimizeSogPointsForDevice(sogSourcePts);
    slots.push({
      athleteId: null,
      name: v.filename || 'Unassigned',
      color,
      videos: [v],
      windCsvIds,
      sogPts,
      statsSeries: {
        sog: buildNumericSeries(sogSourcePts, 'sog', TIMELINE_TRACK_SERIES_OPTS),
        motionDir: buildNumericSeries(collectMotionDirectionForVideo(v), 'motionDir', TIMELINE_TRACK_SERIES_OPTS),
        heading: buildNumericSeries(collectTrackMetricForVideo(v, 'hdg'), 'hdg', TIMELINE_TRACK_SERIES_OPTS),
        cog: buildNumericSeries(collectTrackMetricForVideo(v, 'cog'), 'cog', TIMELINE_TRACK_SERIES_OPTS),
        heel: buildNumericSeries(collectTrackMetricForVideo(v, 'heel'), 'heel', TIMELINE_TRACK_SERIES_OPTS),
        pitch: buildNumericSeries(collectTrackMetricForVideo(v, 'trim'), 'trim', TIMELINE_TRACK_SERIES_OPTS),
        roll: buildNumericSeries([], 'roll_moment'),
        trunk: buildNumericSeries([], 'trunk_angle'),
        rudder: buildNumericSeries([], 'rudder_angle'),
        boom: buildNumericSeries([], 'boom_angle'),
      },
      _selectedTimelineMetrics: null,
      currentFileId: null,
      _lastVideoIdx: null,
      videoEl: null,
      paneEl: null,
    });
  }
  for (let i = 0; i < unassignedCsvs.length; i++) {
    const csv = unassignedCsvs[i];
    const color = state.videoColors[csv.id] || PALETTE[(slots.length + i) % PALETTE.length];
    const sogSourcePts = collectSogForCsvTracks([csv]);
    const sogPts = optimizeSogPointsForDevice(sogSourcePts);
    slots.push({
      athleteId: null,
      name: csv.filename || 'CSV Track',
      color,
      videos: [],
      windCsvIds: [csv.id],
      sogPts,
      statsSeries: {
        sog: buildNumericSeries(sogSourcePts, 'sog', TIMELINE_TRACK_SERIES_OPTS),
        motionDir: buildNumericSeries(collectMotionDirectionForCsvTracks([csv]), 'motionDir', TIMELINE_TRACK_SERIES_OPTS),
        heading: buildNumericSeries(collectTrackMetricForCsvTracks([csv], 'hdg'), 'hdg', TIMELINE_TRACK_SERIES_OPTS),
        cog: buildNumericSeries(collectTrackMetricForCsvTracks([csv], 'cog'), 'cog', TIMELINE_TRACK_SERIES_OPTS),
        heel: buildNumericSeries(collectTrackMetricForCsvTracks([csv], 'heel'), 'heel', TIMELINE_TRACK_SERIES_OPTS),
        pitch: buildNumericSeries(collectTrackMetricForCsvTracks([csv], 'trim'), 'trim', TIMELINE_TRACK_SERIES_OPTS),
        roll: buildNumericSeries([], 'roll_moment'),
        trunk: buildNumericSeries([], 'trunk_angle'),
        rudder: buildNumericSeries([], 'rudder_angle'),
        boom: buildNumericSeries([], 'boom_angle'),
      },
      _selectedTimelineMetrics: null,
      currentFileId: null,
      _lastVideoIdx: null,
      videoEl: null,
      paneEl: null,
    });
  }

  tl.athleteSlots = slots;
  syncVideoLayoutButton();
  for (const slot of slots) {
    slot._selectedTimelineMetrics = getTimelineSlotOverlayMetricSet(slot);
  }
  syncTimelineOverlayMetricSelectionAcrossSlots(slots);
  populateTimelineStats();
  refreshTimelineProcessedMetricStats();

  if(slots.length === 0) { showNoVideo(true); drawSogCanvas(); return; }
  showNoVideo(false);

  // Create video panes
  const grid = el('video-grid');
  grid.innerHTML = '';
  const heatmapPanel = el('inline-heatmap-panel');
  if (heatmapPanel) heatmapPanel.innerHTML = '';
  applyVideoLayout();

  if (!USE_IPAD_VIDEO_WORKAROUNDS) {
    console.log(`[Timeline] Creating ${slots.length} athlete slot(s):`);
    for(let si = 0; si < slots.length; si++) {
      const slot = slots[si];
      console.log(`  Slot ${si}: athlete="${slot.name}" id=${slot.athleteId} videos=${slot.videos.length} sogPts=${slot.sogPts.length} color=${slot.color}`);
      for(const v of slot.videos) console.log(`    vid=${v.id} file=${v.filename} ts=${v.ts_start}..${v.ts_end}`);
    }
  }

  for(const slot of slots) {
    const pane = document.createElement('div');
    pane.className = 'video-pane';
    pane.style.borderColor = slot.color;
    pane.draggable = true;

    const paneBody = document.createElement('div');
    paneBody.className = 'video-pane-body';
    pane.appendChild(paneBody);

    const stage = document.createElement('div');
    stage.className = 'video-stage';
    paneBody.appendChild(stage);

    const label = document.createElement('div');
    label.className = 'pane-label';
    label.style.color = slot.color;
    label.textContent = slot.name;
    stage.appendChild(label);

    const hideBtn = document.createElement('button');
    hideBtn.type = 'button';
    hideBtn.className = 'btn video-pane-hide';
    hideBtn.title = 'Hide this video from playback';
    hideBtn.innerHTML = '<svg class="icon-svg" style="width:12px;height:12px"><use href="#ico-x"/></svg>';
    hideBtn.onclick = (e) => {
      e.stopPropagation();
      setVideoSlotHidden(slot, true);
    };
    stage.appendChild(hideBtn);

    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.className = 'pose-overlay';
    stage.appendChild(overlayCanvas);

    const metricOverlay = document.createElement('div');
    metricOverlay.className = 'video-metric-overlay';
    metricOverlay.style.display = 'none';
    stage.appendChild(metricOverlay);

    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'No video at this time';
    empty.style.display = 'none';
    stage.appendChild(empty);

    const heatmapCol = document.createElement('aside');
    heatmapCol.className = 'video-pane-heatmaps';
    initInlineHeatmapColSync(heatmapCol);
    el('inline-heatmap-panel')?.appendChild(heatmapCol);

    const videos = USE_IPAD_VIDEO_WORKAROUNDS
      ? [createSlotPlaybackVideoEl(slot)]
      : [createSlotPlaybackVideoEl(slot)];
    for (const video of videos) insertSlotVideoEl({ paneEl: pane, _videoStageEl: stage, _overlayCanvas: overlayCanvas, _emptyEl: empty }, video);

    slot._videoEls = videos;
    slot._activeVideoIdx = 0;
    slot.videoEl = videos[0] || null;
    slot.currentFileId = null;
    slot._desiredFileId = null;
    slot.paneEl = pane;
    slot._videoStageEl = stage;
    slot._emptyEl = empty;
    slot._overlayCanvas = overlayCanvas;
    slot._metricOverlayEl = metricOverlay;
    slot._heatmapColEl = heatmapCol;
    pane.dataset.slotKey = getVideoSlotVisibilityKey(slot) || '';
    pane.style.display = 'none'; // hidden until tlSeekTo activates it
    grid.appendChild(pane);
    initVideoPaneDragReorder(pane);
  }

  // Now seek to current timestamp to load appropriate videos
  drawSogCanvas();
  tlSeekTo(tl.currentTs);
  updateTimeLabel();
  updateTimelineStats();
}

/**
 * Collect SOG data points for an athlete, merging CSV and video sources.
 * Returns [{ts, sog}] sorted by ts.
 */
/** Returns true if ts falls within any video's time range (with small padding) */
function isWithinVideoRange(ts, videos, pad = 2) {
  for (const v of videos) {
    if (v.ts_start != null && v.ts_end != null && ts >= v.ts_start - pad && ts <= v.ts_end + pad) return true;
  }
  return false;
}

function smoothSogSeries(pts) {
  if (!Array.isArray(pts) || pts.length < 3) return pts || [];
  const out = pts.map(p => ({ ts: p.ts, sog: p.sog }));
  for (let i = 1; i < pts.length - 1; i++) {
    const a = pts[i - 1].sog;
    const b = pts[i].sog;
    const c = pts[i + 1].sog;
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) continue;
    const neighborAvg = (a + c) / 2;
    const minN = Math.min(a, c);
    const maxN = Math.max(a, c);
    const isolatedDrop = b < minN * 0.45 && (neighborAvg - b) > 0.7;
    const isolatedSpike = b > maxN * 1.9 && (b - neighborAvg) > 0.7;
    if (isolatedDrop || isolatedSpike) out[i].sog = neighborAvg;
  }
  return out;
}

function finalizeSogPoints(rawPts) {
  if (!rawPts.length) return [];
  rawPts.sort((a, b) => a.ts - b.ts);
  const seen = new Set();
  const deduped = rawPts.filter(p => {
    if (!Number.isFinite(p?.ts) || !Number.isFinite(p?.sog)) return false;
    if (p.sog < 0 || p.sog > SOG_MAX_KT) return false;
    if (seen.has(p.ts)) return false;
    seen.add(p.ts);
    return true;
  });
  return smoothSogSeries(deduped);
}

function buildDerivedSogFromGps(points) {
  const pts = [];
  if (!points?.length) return pts;
  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i], p2 = points[i + 1];
    if (p1.ts == null || p2.ts == null) continue;
    const dt = p2.ts - p1.ts;
    if (dt <= 0) continue;
    const dkm = haversineKm(p1.lat, p1.lon, p2.lat, p2.lon);
    const kt = dkm / 1.852 / (dt / 3600);
    if (Number.isFinite(kt)) pts.push({ ts: (p1.ts + p2.ts) / 2, sog: kt });
  }
  return pts;
}

function sortedTelemetryPoints(points) {
  return (Array.isArray(points) ? points : [])
    .filter(point => Number.isFinite(Number(point?.ts)))
    .slice()
    .sort((a, b) => Number(a.ts) - Number(b.ts));
}

function medianTelemetryStep(points) {
  const pts = sortedTelemetryPoints(points);
  const gaps = [];
  for (let i = 1; i < pts.length; i++) {
    const dt = Number(pts[i].ts) - Number(pts[i - 1].ts);
    if (Number.isFinite(dt) && dt > 0) gaps.push(dt);
  }
  if (!gaps.length) return null;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

function csvGpsGapThreshold(csvPts) {
  const med = medianTelemetryStep(csvPts);
  if (!Number.isFinite(med) || med <= 0) return 4;
  return Math.max(2, Math.min(20, med * 4));
}

function tsFallsInCsvGpsGap(ts, csvPts, gapThreshold = null, alreadySorted = false) {
  const t = Number(ts);
  const pts = alreadySorted ? (Array.isArray(csvPts) ? csvPts : []) : sortedTelemetryPoints(csvPts);
  if (!Number.isFinite(t) || pts.length < 2) return false;
  const threshold = Number.isFinite(Number(gapThreshold)) ? Number(gapThreshold) : csvGpsGapThreshold(pts);
  let lo = 0, hi = pts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(pts[mid].ts) < t) lo = mid + 1;
    else hi = mid;
  }
  const next = pts[lo];
  const prev = pts[lo - 1];
  if (!prev && next) {
    const edgeGap = Number(next.ts) - t;
    return edgeGap > threshold && edgeGap <= 120;
  }
  if (prev && !next) {
    const edgeGap = t - Number(prev.ts);
    return edgeGap > threshold && edgeGap <= 120;
  }
  if (!prev || !next) return false;
  const gap = Number(next.ts) - Number(prev.ts);
  return gap > threshold && t > Number(prev.ts) && t < Number(next.ts);
}

function collectVideoGapFallbackPoints(csvPts, videoPts, mapper) {
  const csvSorted = sortedTelemetryPoints(csvPts);
  if (csvSorted.length < 2 || !Array.isArray(videoPts) || !videoPts.length) return [];
  const threshold = csvGpsGapThreshold(csvSorted);
  const out = [];
  for (const point of videoPts) {
    if (!tsFallsInCsvGpsGap(point?.ts, csvSorted, threshold, true)) continue;
    const mapped = mapper(point);
    if (mapped) out.push(mapped);
  }
  return out;
}

function mergeCsvMetricWithVideoGpsGaps(csvMetricPts, csvGpsPts, videoPts, mapper) {
  const metricSorted = sortedTelemetryPoints(csvMetricPts);
  if (metricSorted.length < 2 && Array.isArray(videoPts) && videoPts.length) {
    const fallbackAll = videoPts.map(mapper).filter(Boolean);
    return metricSorted.length ? metricSorted : fallbackAll;
  }
  const fallback = collectVideoGapFallbackPoints(metricSorted, videoPts, mapper);
  if (!fallback.length) return csvMetricPts;
  return [...metricSorted, ...fallback];
}

function pointMetricValue(point, metricKey) {
  if (!point) return null;
  const direct = Number(point[metricKey]);
  if (Number.isFinite(direct)) return direct;
  if (metricKey === 'hdg') {
    const cog = Number(point.cog);
    if (Number.isFinite(cog)) return cog;
  }
  return null;
}

function mergeMapPointsWithVideoGpsGaps(csvPts, videoPts) {
  const csvSorted = sortedTelemetryPoints(csvPts)
    .filter(point => Number.isFinite(Number(point?.lat)) && Number.isFinite(Number(point?.lon)));
  const videoFallback = collectVideoGapFallbackPoints(csvSorted, videoPts, point => {
    const lat = Number(point?.lat);
    const lon = Number(point?.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    return { ...point, lat, lon, _gpsFallbackSource: 'gopro' };
  });
  if (!videoFallback.length) return csvSorted;
  return sortedTelemetryPoints([...csvSorted, ...videoFallback]);
}

function getCsvMapPoints(csv, videos = []) {
  return sortedTelemetryPoints(validLatLonPoints(csv?.points));
}

function getTrackTelemetryPoints(track) {
  if (Array.isArray(track?.telemetry_points) && track.telemetry_points.length) return track.telemetry_points;
  return Array.isArray(track?.points) ? track.points : [];
}

function validLatLonPoints(points) {
  return (Array.isArray(points) ? points : []).filter(point => (
    Number.isFinite(Number(point?.lat)) &&
    Number.isFinite(Number(point?.lon))
  ));
}

function collectSogForAthlete(athId, avids, extraCsvTracks = []) {
  const csvPts = [];
  const csvGpsPts = [];
  const videoPts = [];
  const csvs = state.mapData?.csvs || [];

  // Prefer SOG from matched CSV tracks.
  const seenCsvIds = new Set();
  for (const v of avids) {
    if (v.best_match_csv_id) seenCsvIds.add(v.best_match_csv_id);
    for (const p of getTrackTelemetryPoints(v)) {
      if (p.ts != null && p.sog != null) {
        videoPts.push({ ts: p.ts, sog: p.sog });
      }
    }
  }
  for (const csv of extraCsvTracks || []) {
    if (csv?.id) seenCsvIds.add(csv.id);
  }
  for (const csvId of seenCsvIds) {
    const csv = csvs.find(c => c.id === csvId);
    if (csv) {
      for (const p of getTrackTelemetryPoints(csv)) {
        if (p.ts != null && p.sog != null) csvPts.push({ ts: p.ts, sog: p.sog });
        if (Number.isFinite(Number(p?.ts)) && Number.isFinite(Number(p?.lat)) && Number.isFinite(Number(p?.lon))) {
          csvGpsPts.push({ ts: Number(p.ts), lat: Number(p.lat), lon: Number(p.lon) });
        }
      }
    }
  }
  if (csvPts.length) return finalizeSogPoints(csvPts);
  if (videoPts.length) return finalizeSogPoints(videoPts);

  const derived = [];
  for (const v of avids) derived.push(...buildDerivedSogFromGps(getTrackTelemetryPoints(v)));
  return finalizeSogPoints(derived);
}

function collectSogForVideo(v) {
  const csvPts = [];
  const csvGpsPts = [];
  const videoPts = [];
  const csvs = state.mapData?.csvs || [];
  if (v.best_match_csv_id) {
    const csv = csvs.find(c => c.id === v.best_match_csv_id);
    if (csv) {
      for (const p of getTrackTelemetryPoints(csv)) {
        if (p.ts != null && p.sog != null) csvPts.push({ ts: p.ts, sog: p.sog });
        if (Number.isFinite(Number(p?.ts)) && Number.isFinite(Number(p?.lat)) && Number.isFinite(Number(p?.lon))) {
          csvGpsPts.push({ ts: Number(p.ts), lat: Number(p.lat), lon: Number(p.lon) });
        }
      }
    }
  }
  for (const p of getTrackTelemetryPoints(v)) {
    if (p.ts != null && p.sog != null) videoPts.push({ ts: p.ts, sog: p.sog });
  }
  if (csvPts.length) return finalizeSogPoints(csvPts);
  if (videoPts.length) return finalizeSogPoints(videoPts);
  return finalizeSogPoints(buildDerivedSogFromGps(getTrackTelemetryPoints(v)));
}

function collectSogForCsvTracks(csvTracks) {
  const csvPts = [];
  for (const csv of csvTracks || []) {
    for (const p of getTrackTelemetryPoints(csv)) {
      if (p?.ts != null && p?.sog != null) csvPts.push({ ts: p.ts, sog: p.sog });
    }
  }
  return finalizeSogPoints(csvPts);
}

function collectTrackMetricForAthlete(avids, metricKey, extraCsvTracks = []) {
  const csvPts = [];
  const csvGpsPts = [];
  const videoPts = [];
  const csvs = state.mapData?.csvs || [];
  const seenCsvIds = new Set();

  for (const v of avids || []) {
    if (v?.best_match_csv_id) seenCsvIds.add(v.best_match_csv_id);
    for (const p of getTrackTelemetryPoints(v)) {
      const vMetric = pointMetricValue(p, metricKey);
      if (p?.ts != null && vMetric != null) {
        videoPts.push({ ts: p.ts, [metricKey]: vMetric });
      }
    }
  }
  for (const csv of extraCsvTracks || []) {
    if (csv?.id) seenCsvIds.add(csv.id);
  }

  for (const csvId of seenCsvIds) {
    const csv = csvs.find(c => c.id === csvId);
    for (const p of getTrackTelemetryPoints(csv)) {
      const vMetric = pointMetricValue(p, metricKey);
      if (p?.ts != null && vMetric != null) {
        csvPts.push({ ts: p.ts, [metricKey]: vMetric });
      }
      if (Number.isFinite(Number(p?.ts)) && Number.isFinite(Number(p?.lat)) && Number.isFinite(Number(p?.lon))) {
        csvGpsPts.push({ ts: Number(p.ts), lat: Number(p.lat), lon: Number(p.lon) });
      }
    }
  }

  if (csvPts.length && (metricKey === 'cog' || metricKey === 'hdg')) {
    return mergeCsvMetricWithVideoGpsGaps(csvPts, csvGpsPts, videoPts, p => {
      const value = pointMetricValue(p, metricKey);
      return value != null ? { ts: Number(p.ts), [metricKey]: value, source: 'gopro_gap' } : null;
    });
  }
  return csvPts.length ? csvPts : videoPts;
}

function collectTrackMetricForCsvTracks(csvTracks, metricKey) {
  const csvPts = [];
  for (const csv of csvTracks || []) {
    for (const p of getTrackTelemetryPoints(csv)) {
      const vMetric = pointMetricValue(p, metricKey);
      if (p?.ts != null && vMetric != null) {
        csvPts.push({ ts: p.ts, [metricKey]: vMetric });
      }
    }
  }
  return csvPts;
}

function collectTrackMetricForVideo(video, metricKey) {
  const csvPts = [];
  const csvGpsPts = [];
  const videoPts = [];
  const csvs = state.mapData?.csvs || [];

  if (video?.best_match_csv_id) {
    const csv = csvs.find(c => c.id === video.best_match_csv_id);
    for (const p of getTrackTelemetryPoints(csv)) {
      const vMetric = pointMetricValue(p, metricKey);
      if (p?.ts != null && vMetric != null) {
        csvPts.push({ ts: p.ts, [metricKey]: vMetric });
      }
      if (Number.isFinite(Number(p?.ts)) && Number.isFinite(Number(p?.lat)) && Number.isFinite(Number(p?.lon))) {
        csvGpsPts.push({ ts: Number(p.ts), lat: Number(p.lat), lon: Number(p.lon) });
      }
    }
  }

  for (const p of getTrackTelemetryPoints(video)) {
    const vMetric = pointMetricValue(p, metricKey);
    if (p?.ts != null && vMetric != null) {
      videoPts.push({ ts: p.ts, [metricKey]: vMetric });
    }
  }

  if (csvPts.length && (metricKey === 'cog' || metricKey === 'hdg')) {
    return mergeCsvMetricWithVideoGpsGaps(csvPts, csvGpsPts, videoPts, p => {
      const value = pointMetricValue(p, metricKey);
      return value != null ? { ts: Number(p.ts), [metricKey]: value, source: 'gopro_gap' } : null;
    });
  }
  return csvPts.length ? csvPts : videoPts;
}

function collectMotionDirectionForCsvTracks(csvTracks) {
  const csvPts = [];
  for (const csv of csvTracks || []) {
    for (const point of getTrackTelemetryPoints(csv)) {
      const motionDir = extractMotionDirectionDeg(point);
      if (point?.ts != null && motionDir != null) {
        csvPts.push({ ts: point.ts, motionDir });
      }
    }
  }
  return csvPts;
}

function extractMotionDirectionDeg(point) {
  if (!point) return null;
  const cog = Number(point.cog);
  if (Number.isFinite(cog)) return wrapWindDegrees(cog);
  const hdg = Number(point.hdg);
  if (Number.isFinite(hdg)) return wrapWindDegrees(hdg);
  return null;
}

function collectMotionDirectionForAthlete(avids, extraCsvTracks = []) {
  const csvPts = [];
  const csvGpsPts = [];
  const videoPts = [];
  const csvs = state.mapData?.csvs || [];
  const seenCsvIds = new Set();

  for (const video of avids || []) {
    if (video?.best_match_csv_id) seenCsvIds.add(video.best_match_csv_id);
    for (const point of getTrackTelemetryPoints(video)) {
      const motionDir = extractMotionDirectionDeg(point);
      if (point?.ts != null && motionDir != null) {
        videoPts.push({ ts: point.ts, motionDir });
      }
    }
  }
  for (const csv of extraCsvTracks || []) {
    if (csv?.id) seenCsvIds.add(csv.id);
  }

  for (const csvId of seenCsvIds) {
    const csv = csvs.find(c => c.id === csvId);
    for (const point of getTrackTelemetryPoints(csv)) {
      const motionDir = extractMotionDirectionDeg(point);
      if (point?.ts != null && motionDir != null) {
        csvPts.push({ ts: point.ts, motionDir });
      }
      if (Number.isFinite(Number(point?.ts)) && Number.isFinite(Number(point?.lat)) && Number.isFinite(Number(point?.lon))) {
        csvGpsPts.push({ ts: Number(point.ts), lat: Number(point.lat), lon: Number(point.lon) });
      }
    }
  }

  if (csvPts.length) {
    return mergeCsvMetricWithVideoGpsGaps(csvPts, csvGpsPts, videoPts, point => {
      const motionDir = extractMotionDirectionDeg(point);
      return motionDir != null ? { ts: Number(point.ts), motionDir, source: 'gopro_gap' } : null;
    });
  }
  return videoPts;
}

function collectMotionDirectionForVideo(video) {
  const csvPts = [];
  const csvGpsPts = [];
  const videoPts = [];
  const csvs = state.mapData?.csvs || [];

  if (video?.best_match_csv_id) {
    const csv = csvs.find(c => c.id === video.best_match_csv_id);
    for (const point of getTrackTelemetryPoints(csv)) {
      const motionDir = extractMotionDirectionDeg(point);
      if (point?.ts != null && motionDir != null) {
        csvPts.push({ ts: point.ts, motionDir });
      }
      if (Number.isFinite(Number(point?.ts)) && Number.isFinite(Number(point?.lat)) && Number.isFinite(Number(point?.lon))) {
        csvGpsPts.push({ ts: Number(point.ts), lat: Number(point.lat), lon: Number(point.lon) });
      }
    }
  }

  for (const point of getTrackTelemetryPoints(video)) {
    const motionDir = extractMotionDirectionDeg(point);
    if (point?.ts != null && motionDir != null) {
      videoPts.push({ ts: point.ts, motionDir });
    }
  }

  if (csvPts.length) {
    return mergeCsvMetricWithVideoGpsGaps(csvPts, csvGpsPts, videoPts, point => {
      const motionDir = extractMotionDirectionDeg(point);
      return motionDir != null ? { ts: Number(point.ts), motionDir, source: 'gopro_gap' } : null;
    });
  }
  return videoPts;
}

function collectSlotWindCsvIds(videos, extraCsvTracks = []) {
  const out = [];
  const seen = new Set();
  for (const video of videos || []) {
    const csvId = video?.best_match_csv_id;
    if (!csvId || seen.has(csvId)) continue;
    seen.add(csvId);
    out.push(csvId);
  }
  for (const csv of extraCsvTracks || []) {
    const csvId = csv?.id;
    if (!csvId || seen.has(csvId)) continue;
    seen.add(csvId);
    out.push(csvId);
  }
  return out;
}

function buildNumericSeries(rawPts, valueKey, opts = null) {
  const ordered = (rawPts || [])
    .map(p => ({ ts: Number(p?.ts), v: Number(p?.[valueKey]) }))
    .filter(p => Number.isFinite(p.ts) && Number.isFinite(p.v))
    .sort((a, b) => a.ts - b.ts);

  const points = [];
  for (const p of ordered) {
    const prev = points[points.length - 1];
    if (prev && Math.abs(prev.ts - p.ts) <= 1e-6) prev.v = p.v;
    else points.push(p);
  }

  const gaps = [];
  for (let i = 1; i < points.length; i++) {
    const dt = points[i].ts - points[i - 1].ts;
    if (Number.isFinite(dt) && dt > 0) gaps.push(dt);
  }

  let medianGapSec = null;
  if (gaps.length) {
    gaps.sort((a, b) => a - b);
    medianGapSec = gaps[Math.floor(gaps.length / 2)];
  }

  const gapMultiplier = Number.isFinite(Number(opts?.gapMultiplier))
    ? Math.max(1, Number(opts.gapMultiplier))
    : 4;
  const minInterpolationGapSec = Number.isFinite(Number(opts?.minInterpolationGapSec))
    ? Math.max(0.25, Number(opts.minInterpolationGapSec))
    : 1.5;
  const maxGapSec = Number.isFinite(medianGapSec)
    ? Math.max(minInterpolationGapSec, medianGapSec * gapMultiplier)
    : Math.max(minInterpolationGapSec, 2.5);
  const areaPrefix = new Array(points.length).fill(0);
  const coveragePrefix = new Array(points.length).fill(0);
  for (let i = 1; i < points.length; i++) {
    areaPrefix[i] = areaPrefix[i - 1];
    coveragePrefix[i] = coveragePrefix[i - 1];
    const prev = points[i - 1];
    const next = points[i];
    const dt = next.ts - prev.ts;
    if (!(dt > 0) || dt > maxGapSec) continue;
    areaPrefix[i] += 0.5 * (prev.v + next.v) * dt;
    coveragePrefix[i] += dt;
  }
  return { points, areaPrefix, coveragePrefix, maxGapSec };
}

function lowerBoundSeriesTs(points, targetTs) {
  let lo = 0, hi = points.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].ts < targetTs) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBoundSeriesTs(points, targetTs) {
  let lo = 0, hi = points.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].ts <= targetTs) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function findNearestSeriesPoint(points, targetTs) {
  if (!Array.isArray(points) || !points.length || !Number.isFinite(Number(targetTs))) return null;
  const idx = lowerBoundSeriesTs(points, Number(targetTs));
  const prev = idx > 0 ? points[idx - 1] : null;
  const next = idx < points.length ? points[idx] : null;
  if (!prev) return next || null;
  if (!next) return prev;
  return Math.abs(prev.ts - targetTs) <= Math.abs(next.ts - targetTs) ? prev : next;
}

function getSeriesIntegralSnapshot(series, targetTs) {
  const points = series?.points || [];
  const areaPrefix = series?.areaPrefix || [];
  const coveragePrefix = series?.coveragePrefix || [];
  const maxGapSec = Number.isFinite(Number(series?.maxGapSec)) ? Number(series.maxGapSec) : Infinity;
  const ts = Number(targetTs);
  if (!points.length || !Number.isFinite(ts)) return { area: 0, coverage: 0 };

  const idx = lowerBoundSeriesTs(points, ts);
  if (idx <= 0) return { area: 0, coverage: 0 };
  if (idx >= points.length) {
    return {
      area: areaPrefix[points.length - 1] || 0,
      coverage: coveragePrefix[points.length - 1] || 0,
    };
  }

  const lo = points[idx - 1];
  const hi = points[idx];
  const baseArea = areaPrefix[idx - 1] || 0;
  const baseCoverage = coveragePrefix[idx - 1] || 0;
  const dt = hi.ts - lo.ts;
  if (!(dt > 0) || dt > maxGapSec || ts <= lo.ts) {
    return { area: baseArea, coverage: baseCoverage };
  }

  const frac = (ts - lo.ts) / dt;
  const vAtTs = lo.v + frac * (hi.v - lo.v);
  const partialDt = ts - lo.ts;
  return {
    area: baseArea + 0.5 * (lo.v + vAtTs) * partialDt,
    coverage: baseCoverage + partialDt,
  };
}

function computeSeriesWindowMean(series, startTs, endTs) {
  const points = series?.points || [];
  const start = Number(startTs);
  const end = Number(endTs);
  if (!points.length || !Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return null;

  const clampedStartTs = Math.max(start, points[0].ts);
  const clampedEndTs = Math.min(end, points[points.length - 1].ts);
  if (!(clampedEndTs > clampedStartTs)) return null;

  const startSnap = getSeriesIntegralSnapshot(series, clampedStartTs);
  const endSnap = getSeriesIntegralSnapshot(series, clampedEndTs);
  const coveredSec = endSnap.coverage - startSnap.coverage;
  if (!(coveredSec > 1e-6)) return null;
  return (endSnap.area - startSnap.area) / coveredSec;
}

const TIMELINE_STAT_DEFS = Object.freeze([
  { key: 'sog', label: 'SOG', suffix: ' kt', seriesKey: 'sog', overlayLabel: 'SOG' },
  {
    key: 'vmg',
    label: 'VMG',
    suffix: ' kt',
    live: true,
    overlayLabel: 'VMG',
    title: 'Live absolute VMG from current motion direction against the estimated wind',
  },
  {
    key: 'twa',
    label: 'TWA',
    suffix: '\u00B0',
    live: true,
    overlayLabel: 'TWA',
    title: 'Live true wind angle from current motion direction against the estimated wind',
  },
  { key: 'heading', label: 'COG', suffix: '\u00B0', seriesKey: 'heading', overlayLabel: 'COG' },
  { key: 'heel', label: 'Heel', suffix: '\u00B0', seriesKey: 'heel', overlayLabel: 'Heel' },
  { key: 'pitch', label: 'Pitch', suffix: '\u00B0', seriesKey: 'pitch', overlayLabel: 'Pitch' },
  { key: 'roll', label: 'Roll M', suffix: ' Nm', seriesKey: 'roll', overlayLabel: 'RM' },
  { key: 'trunk', label: 'Trunk', suffix: '\u00B0', seriesKey: 'trunk', overlayLabel: 'TA' },
  { key: 'rudder', label: 'Rudder', suffix: '\u00B0', seriesKey: 'rudder', overlayLabel: 'RA' },
  { key: 'boom', label: 'Boom', suffix: '\u00B0', seriesKey: 'boom', overlayLabel: 'BA' },
]);

function yieldTimelineMetricWork() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

function createEmptyProcessedTimelineMetricSet() {
  return { trunk_angle: [], roll_moment: [], rudder_angle: [], boom_angle: [] };
}

function createEmptyProcessedTimelineStatValues() {
  return { roll: null, trunk: null, rudder: null, boom: null };
}

function isTimelineMetricLoadAborted(err) {
  return String(err?.message || err || '') === TIMELINE_METRIC_LOAD_ABORTED;
}

function abortTimelineMetricLoadIfNeeded(loadToken) {
  if (loadToken !== _timelineMetricLoadToken || state.tl?.playing) {
    throw new Error(TIMELINE_METRIC_LOAD_ABORTED);
  }
}

function buildVideoSecToAbsTsConverter(video) {
  if (!video) return () => null;
  const telemetryPoints = getTrackTelemetryPoints(video);
  if (!telemetryPoints.length) {
    const startTs = Number(video.ts_start);
    return (videoSec) => (Number.isFinite(startTs) && Number.isFinite(Number(videoSec)))
      ? startTs + Number(videoSec)
      : null;
  }

  const pts = telemetryPoints
    .filter(p => Number.isFinite(Number(p?.video_s)) && Number.isFinite(Number(p?.ts)))
    .sort((a, b) => Number(a.video_s) - Number(b.video_s));
  if (!pts.length) {
    const startTs = Number(video.ts_start);
    return (videoSec) => (Number.isFinite(startTs) && Number.isFinite(Number(videoSec)))
      ? startTs + Number(videoSec)
      : null;
  }

  let idx = 0;
  return (videoSecRaw) => {
    const videoSec = Number(videoSecRaw);
    if (!Number.isFinite(videoSec)) return null;

    while (idx + 1 < pts.length && Number(pts[idx + 1].video_s) <= videoSec) idx++;
    const lo = pts[idx];
    const hi = pts[Math.min(idx + 1, pts.length - 1)];
    const loVs = Number(lo?.video_s);
    const loTs = Number(lo?.ts);
    if (!Number.isFinite(loVs) || !Number.isFinite(loTs)) return null;

    const hiVs = Number(hi?.video_s);
    const hiTs = Number(hi?.ts);
    if (hi === lo || !Number.isFinite(hiVs) || !Number.isFinite(hiTs) || hiVs === loVs) return loTs;

    const frac = (videoSec - loVs) / (hiVs - loVs);
    return loTs + frac * (hiTs - loTs);
  };
}

async function loadProcessedTimelineMetricPoints(projectId, video, loadToken = _timelineMetricLoadToken) {
  const pid = String(projectId || '');
  const fileId = String(video?.id || '');
  if (!pid || !fileId) return createEmptyProcessedTimelineMetricSet();

  const cacheKey = `${pid}:${fileId}`;
  const cached = _timelineProcessedMetricCache.get(cacheKey);
  if (cached && (cached.ready || cached.loadToken === loadToken)) {
    return cached.promise;
  }

  const entry = { loadToken, ready: false, promise: null };
  const loadPromise = (async () => {
    abortTimelineMetricLoadIfNeeded(loadToken);
    const text = await Storage.readFileText(`${pid}/${fileId}/metrics.jsonl`);
    abortTimelineMetricLoadIfNeeded(loadToken);
    if (!text) {
      entry.ready = true;
      return createEmptyProcessedTimelineMetricSet();
    }

    const toAbsTs = buildVideoSecToAbsTsConverter(video);
    const trunk = [];
    const roll = [];
    const rudder = [];
    const boom = [];
    const lines = text.split('\n');
    const yieldEvery = 600;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]?.trim();
      if (!line) continue;

      let row = null;
      try {
        row = JSON.parse(line);
      } catch {
        continue;
      }

      row = PoseEngine.normalizeMetricRudderConvention(row);
      if (!row?.detected) continue;
      const videoSec = Number(row.ts);
      if (!Number.isFinite(videoSec)) continue;
      const absTs = toAbsTs(videoSec);
      if (!Number.isFinite(absTs)) continue;

      const trunkVal = Number(row.trunk_angle);
      if (Number.isFinite(trunkVal)) trunk.push({ ts: absTs, trunk_angle: trunkVal });

      const rollVal = Number(row.roll_moment);
      if (Number.isFinite(rollVal)) roll.push({ ts: absTs, roll_moment: rollVal });

      const rudderVal = Number(row.rudder_angle);
      if (Number.isFinite(rudderVal)) rudder.push({ ts: absTs, rudder_angle: rudderVal });

      const boomVal = Number(row.boom_angle);
      if (Number.isFinite(boomVal)) boom.push({ ts: absTs, boom_angle: boomVal });

      if (i > 0 && i % yieldEvery === 0) {
        abortTimelineMetricLoadIfNeeded(loadToken);
        await yieldTimelineMetricWork();
      }
    }

    abortTimelineMetricLoadIfNeeded(loadToken);
    entry.ready = true;
    return { trunk_angle: trunk, roll_moment: roll, rudder_angle: rudder, boom_angle: boom };
  })().catch((err) => {
    if (_timelineProcessedMetricCache.get(cacheKey) === entry) {
      _timelineProcessedMetricCache.delete(cacheKey);
    }
    if (isTimelineMetricLoadAborted(err)) return null;
    console.warn(`[Timeline] metric load failed for ${fileId}:`, err);
    return createEmptyProcessedTimelineMetricSet();
  });

  entry.promise = loadPromise;
  _timelineProcessedMetricCache.set(cacheKey, entry);
  return loadPromise;
}

async function hydrateTimelineProcessedMetricSeries(slots, loadToken) {
  const projectId = state.projectId;
  if (!projectId || !Array.isArray(slots) || !slots.length) return;

  for (const slot of slots) {
    if (loadToken !== _timelineMetricLoadToken) return;
    const videos = Array.isArray(slot?.videos) ? slot.videos : [];
    if (!videos.length) {
      slot.statsSeries = {
        ...(slot.statsSeries || {}),
        roll: buildNumericSeries([], 'roll_moment'),
        trunk: buildNumericSeries([], 'trunk_angle'),
        rudder: buildNumericSeries([], 'rudder_angle'),
        boom: buildNumericSeries([], 'boom_angle'),
      };
      continue;
    }

    const trunk = [];
    const roll = [];
    const rudder = [];
    const boom = [];
    for (const video of videos) {
      if (loadToken !== _timelineMetricLoadToken) return;
      const set = await loadProcessedTimelineMetricPoints(projectId, video, loadToken);
      if (set == null) return;
      if (Array.isArray(set?.trunk_angle)) trunk.push(...set.trunk_angle);
      if (Array.isArray(set?.roll_moment)) roll.push(...set.roll_moment);
      if (Array.isArray(set?.rudder_angle)) rudder.push(...set.rudder_angle);
      if (Array.isArray(set?.boom_angle)) boom.push(...set.boom_angle);
      await yieldTimelineMetricWork();
    }

    slot.statsSeries = {
      ...(slot.statsSeries || {}),
      roll: buildNumericSeries(roll, 'roll_moment'),
      trunk: buildNumericSeries(trunk, 'trunk_angle'),
      rudder: buildNumericSeries(rudder, 'rudder_angle'),
      boom: buildNumericSeries(boom, 'boom_angle'),
    };
  }

  if (loadToken !== _timelineMetricLoadToken || slots !== state.tl?.athleteSlots) return;
  updateTimelineStats();
}

function refreshTimelineProcessedMetricStats() {
  const slots = state.tl?.athleteSlots || [];
  if (!slots.length) return;
  if (state.tl?.playing) {
    state.tl._pendingProcessedMetricRefresh = true;
    return;
  }
  state.tl._pendingProcessedMetricRefresh = false;
  const loadToken = ++_timelineMetricLoadToken;
  void hydrateTimelineProcessedMetricSeries(slots, loadToken);
}

function computeRollingMean(series, centerTs, windowSec = state.timelineStatsWindowSec) {
  const points = series?.points || [];
  if (!points.length || !Number.isFinite(Number(centerTs))) return null;
  const clampedWindow = clampTimelineStatsWindowSec(windowSec);
  const center = Number(centerTs);
  const halfWindow = clampedWindow / 2;
  const maxGapSec = Number.isFinite(Number(series?.maxGapSec))
    ? Number(series.maxGapSec)
    : Math.max(2.5, clampedWindow * 2.5);
  const centeredMean = computeSeriesWindowMean(series, center - halfWindow, center + halfWindow);
  if (centeredMean != null) return centeredMean;

  const nearest = findNearestSeriesPoint(points, center);
  if (!nearest || Math.abs(nearest.ts - center) > maxGapSec) return null;
  const fallbackMean = computeSeriesWindowMean(series, nearest.ts - halfWindow, nearest.ts + halfWindow);
  if (fallbackMean != null) return fallbackMean;
  return nearest.v;
}

function computeSeriesInstantValue(series, centerTs, maxGapSec = TIMELINE_INSTANT_METRIC_MAX_GAP_SEC) {
  const points = series?.points || [];
  if (!points.length || !Number.isFinite(Number(centerTs))) return null;
  const nearest = findNearestSeriesPoint(points, Number(centerTs));
  if (!nearest || Math.abs(nearest.ts - Number(centerTs)) > maxGapSec) return null;
  return nearest.v;
}

// Mean of the trailing `windowSec` seconds ending at `centerTs` — used for the
// 1 Hz timeline stat readouts so the numbers reflect the last second rather than
// a single instantaneous (and jumpy) sample. Falls back to the nearest sample
// when the window has no coverage (e.g. at the very start of a track).
function computeSeriesTrailingMean(series, centerTs, windowSec = TIMELINE_STATS_MEAN_WINDOW_SEC) {
  const center = Number(centerTs);
  if (!Number.isFinite(center)) return null;
  const win = Number(windowSec) > 0 ? Number(windowSec) : TIMELINE_STATS_MEAN_WINDOW_SEC;
  const mean = computeSeriesWindowMean(series, center - win, center);
  if (mean != null) return mean;
  return computeSeriesInstantValue(series, center);
}

function formatTimelineMetricValue(value, suffix = '') {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}${suffix}` : '--';
}

function slotHasTimelineMetricSeries(slot, metricKey) {
  const def = TIMELINE_STAT_DEFS.find(d => d.key === metricKey);
  if (!def) return false;
  if (def.live) return Boolean(slot?.statsSeries?.motionDir?.points?.length);
  return Boolean(slot?.statsSeries?.[def.seriesKey]?.points?.length);
}

function resolveSlotWindEstimate(slot, absTs) {
  const targetTs = Number(absTs);
  const candidateRows = [];
  const seen = new Set();
  const activeVideo = findSlotVideoAtTime(slot, targetTs);

  const enqueueCsv = (csvId, priority) => {
    if (!csvId || seen.has(csvId)) return;
    seen.add(csvId);
    candidateRows.push({ csvId, priority });
  };

  enqueueCsv(activeVideo?.best_match_csv_id, 0);
  for (const csvId of slot?.windCsvIds || []) enqueueCsv(csvId, 1);

  let best = null;
  let bestPriority = Number.POSITIVE_INFINITY;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (const row of candidateRows) {
    const wind = state.wind.byCsvId?.[row.csvId];
    const localSeries = wind?.localSeries;
    const point = findNearestWindPointByTs(localSeries?.points, targetTs);
    if (!point) continue;
    const maxGap = Math.max(
      Number(localSeries?.stepSeconds) * 1.5 || 0,
      Number(localSeries?.windowSeconds) * 0.65 || 0,
      90,
    );
    const delta = Math.abs(Number(point.ts) - targetTs);
    if (!Number.isFinite(delta) || delta > maxGap) continue;
    if (row.priority < bestPriority || (row.priority === bestPriority && delta < bestDelta)) {
      best = point;
      bestPriority = row.priority;
      bestDelta = delta;
    }
  }

  if (hasResolvedWindEstimate(best)) return best;
  if (hasResolvedWindEstimate(state.wind.localNow)) return state.wind.localNow;
  if (hasResolvedWindEstimate(state.wind.session)) return state.wind.session;
  return null;
}

function resolveLiveTimelineTwaDeg(slot, absTs) {
  const currentTs = Number(absTs);
  if (!Number.isFinite(currentTs)) return null;

  const motionSeries = slot?.statsSeries?.motionDir;
  const motionPoint = findNearestSeriesPoint(motionSeries?.points, currentTs);
  if (!motionPoint || Math.abs(motionPoint.ts - currentTs) > TIMELINE_VMG_MOTION_MAX_GAP_SEC) return null;

  const wind = resolveSlotWindEstimate(slot, currentTs);
  if (!hasResolvedWindEstimate(wind)) return null;

  return mirrorAngleToHalfCircleDeg(angleDifferenceDeg(motionPoint.v, wind.directionDeg));
}

function computeLiveTimelineTwa(slot, absTs) {
  return resolveLiveTimelineTwaDeg(slot, absTs);
}

function computeLiveTimelineVmg(slot, absTs, sogKts, twaDeg = resolveLiveTimelineTwaDeg(slot, absTs)) {
  const sog = Number(sogKts);
  if (!Number.isFinite(sog) || !Number.isFinite(twaDeg)) return null;
  return Math.abs(sog * Math.cos(twaDeg * Math.PI / 180));
}

function shouldRefreshProcessedTimelineStats(currentTs, windowSec) {
  if (!state.tl?.playing) return true;
  const currentTsNum = currentTs == null ? NaN : Number(currentTs);
  const now = performance.now();
  const lastAt = Number(state.tl?._processedStatsRefreshedAt);
  const lastTs = Number(state.tl?._processedStatsRefreshedTs);
  const lastWindow = Number(state.tl?._processedStatsWindowSec);
  if (!Number.isFinite(lastAt) || !Number.isFinite(lastTs)) return true;
  if (lastWindow !== windowSec) return true;
  if (!Number.isFinite(currentTsNum)) return true;
  if (currentTsNum < lastTs - 1e-6) return true;
  if (Math.abs(currentTsNum - lastTs) >= 0.25) return true;
  return (now - lastAt) >= TIMELINE_PROCESSED_STATS_REFRESH_MS;
}

function markProcessedTimelineStatsRefresh(currentTs, windowSec) {
  const currentTsNum = currentTs == null ? NaN : Number(currentTs);
  state.tl._processedStatsRefreshedAt = performance.now();
  state.tl._processedStatsRefreshedTs = Number.isFinite(currentTsNum) ? currentTsNum : null;
  state.tl._processedStatsWindowSec = windowSec;
}

function getTimelineOverlayGraphWindowSec() {
  return TIMELINE_STAT_OVERLAY_WINDOW_SEC;
}

function buildTimelineOverlaySeriesForSlot(slot, def, startTs, endTs) {
  if (!slot) return null;
  if (def.live) return buildTimelineOverlayLiveSeries(slot, def, startTs, endTs);

  const series = slot?.statsSeries?.[def.seriesKey];
  const points = series?.points || [];
  if (!points.length) return null;

  let startIdx = lowerBoundSeriesTs(points, startTs);
  if (startIdx > 0) startIdx -= 1;
  let endIdx = upperBoundSeriesTs(points, endTs);
  if (endIdx < points.length) endIdx += 1;

  let subset = points.slice(startIdx, endIdx).filter(p => p.ts >= startTs && p.ts <= endTs);
  if (!subset.length) subset = points.slice(startIdx, endIdx);
  if (!subset.length) return null;

  if (subset.length > TIMELINE_STAT_OVERLAY_MAX_POINTS) {
    subset = decimateSeriesByCount(subset, TIMELINE_STAT_OVERLAY_MAX_POINTS);
  }

  return {
    color: slot.color || '#c8cdd0',
    points: subset,
    maxGapSec: Number.isFinite(Number(series?.maxGapSec))
      ? Number(series.maxGapSec)
      : TIMELINE_INSTANT_METRIC_MAX_GAP_SEC,
  };
}

function buildTimelineOverlayLiveSeries(slot, def, startTs, endTs) {
  const motionSeries = slot?.statsSeries?.motionDir;
  const points = motionSeries?.points || [];
  if (!points.length) return null;

  let startIdx = lowerBoundSeriesTs(points, startTs);
  if (startIdx > 0) startIdx -= 1;
  let endIdx = upperBoundSeriesTs(points, endTs);
  if (endIdx < points.length) endIdx += 1;

  let subset = points.slice(startIdx, endIdx);
  if (!subset.length) return null;

  if (subset.length > TIMELINE_STAT_OVERLAY_MAX_POINTS) {
    subset = decimateSeriesByCount(subset, TIMELINE_STAT_OVERLAY_MAX_POINTS);
  }

  const out = [];
  for (const p of subset) {
    const ts = Number(p?.ts);
    if (!Number.isFinite(ts)) continue;
    let v = null;
    if (def.key === 'twa') {
      v = computeLiveTimelineTwa(slot, ts);
    } else if (def.key === 'vmg') {
      const sog = computeSeriesInstantValue(slot.statsSeries?.sog, ts);
      const twa = computeLiveTimelineTwa(slot, ts);
      v = computeLiveTimelineVmg(slot, ts, sog, twa);
    }
    if (Number.isFinite(v)) out.push({ ts, v: Number(v) });
  }
  if (!out.length) return null;

  return {
    color: slot.color || '#c8cdd0',
    points: out,
    maxGapSec: Number.isFinite(Number(motionSeries?.maxGapSec))
      ? Number(motionSeries.maxGapSec)
      : TIMELINE_INSTANT_METRIC_MAX_GAP_SEC,
  };
}

function buildTimelineOverlayGraphData(selectedMetrics, slots, currentTs, windowSec) {
  const ts = Number(currentTs);
  if (!Number.isFinite(ts) || !selectedMetrics?.size || !Array.isArray(slots) || !slots.length) return null;

  const spanSec = Math.max(1, Number(windowSec) || TIMELINE_STAT_OVERLAY_WINDOW_SEC);
  const startTs = ts - spanSec / 2;
  const endTs = ts + spanSec / 2;
  const metrics = {};

  for (const def of TIMELINE_STAT_DEFS) {
    if (!selectedMetrics.has(def.key)) continue;

    const seriesList = [];
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    for (const slot of slots) {
      const seriesInfo = buildTimelineOverlaySeriesForSlot(slot, def, startTs, endTs);
      if (!seriesInfo?.points?.length) continue;
      seriesList.push(seriesInfo);
      for (const p of seriesInfo.points) {
        if (!Number.isFinite(p?.v)) continue;
        if (p.v < min) min = p.v;
        if (p.v > max) max = p.v;
      }
    }

    if (!seriesList.length || !Number.isFinite(min) || !Number.isFinite(max)) continue;
    if (Math.abs(max - min) < 1e-6) {
      const pad = Math.max(1, Math.abs(max) * 0.05);
      min -= pad;
      max += pad;
    }

    metrics[def.key] = { min, max, series: seriesList };
  }

  return { startTs, endTs, windowSec: spanSec, metrics };
}

function getTimelineOverlayGraphTargetSlots(slots) {
  const visible = (slots || []).filter(slot => {
    const pane = slot?.paneEl;
    if (!pane || pane.style.display === 'none') return false;
    return Boolean(slot?._videoStageEl);
  });
  return visible.length ? visible : (slots || []);
}

function shouldUseTimelineMetricSharedPanel(slots = state.tl?.athleteSlots || []) {
  if (!state.timelineStatOverlayGraph) return false;
  const visibleCount = (slots || []).filter(slot => {
    const pane = slot?.paneEl;
    return pane && pane.style.display !== 'none' && !isVideoSlotHidden(slot);
  }).length;
  return visibleCount >= 1;
}

function computeTimelineOverlayGraphLayout(slot, count) {
  const stage = slot?._videoStageEl;
  const height = stage?.clientHeight || stage?.offsetHeight || 0;
  const baseGap = TIMELINE_STAT_OVERLAY_GRAPH_GAP;
  if (!Number.isFinite(height) || height <= 0 || !Number.isFinite(count) || count <= 0) {
    return { rowHeight: TIMELINE_STAT_OVERLAY_GRAPH_HEIGHT, gap: baseGap };
  }

  const gap = count > 1
    ? Math.min(baseGap, Math.max(4, Math.round(height * 0.03)))
    : 0;
  const usable = Math.max(0, height - TIMELINE_STAT_OVERLAY_GRAPH_PAD_TOP - TIMELINE_STAT_OVERLAY_GRAPH_PAD_BOTTOM);
  const raw = (usable - gap * Math.max(0, count - 1)) / Math.max(1, count);
  const rowHeight = Math.max(10, Math.min(TIMELINE_STAT_OVERLAY_GRAPH_HEIGHT, raw));
  return { rowHeight, gap };
}

function assignTimelineOverlayGraphSlots(selectedMetrics, slots) {
  for (const slot of slots || []) {
    slot._metricOverlayGraphKeys = new Set();
    slot._metricOverlayGraphLayout = null;
  }

  const order = TIMELINE_STAT_DEFS
    .filter(def => selectedMetrics?.has(def.key))
    .map(def => def.key);
  if (!order.length) return;

  const targetSlots = getTimelineOverlayGraphTargetSlots(slots);
  if (!targetSlots.length) return;

  let slotIdx = 0;
  for (const key of order) {
    const slot = targetSlots[slotIdx % targetSlots.length];
    slotIdx++;
    slot._metricOverlayGraphKeys.add(key);
  }

  for (const slot of targetSlots) {
    slot._metricOverlayGraphLayout = computeTimelineOverlayGraphLayout(
      slot,
      slot._metricOverlayGraphKeys.size
    );
  }
}

function formatTimelineOverlayAxisValue(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '';
  const abs = Math.abs(num);
  const digits = abs >= 100 ? 0 : 1;
  return num.toFixed(digits);
}

function renderTimelineOverlayMetricGraph(canvas, metricData, def, graphMeta) {
  if (!canvas || !metricData || !graphMeta) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;

  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const scale = Math.max(0.6, Math.min(1, rect.height / TIMELINE_STAT_OVERLAY_GRAPH_HEIGHT));
  const padLeft = Math.round(34 * scale);
  const padRight = Math.round(8 * scale);
  const padTop = Math.round(18 * scale);
  const padBottom = Math.round(16 * scale);
  const plotW = rect.width - padLeft - padRight;
  const plotH = rect.height - padTop - padBottom;
  if (plotW <= 8 || plotH <= 8) return;

  const min = metricData.min;
  const max = metricData.max;
  const span = max - min;
  if (!(span > 0)) return;

  const fontSize = Math.max(8, Math.round(11 * scale));
  ctx.font = `${fontSize}px Segoe UI, sans-serif`;
  ctx.fillStyle = 'rgba(255,255,255,0.72)';
  const unit = String(def?.suffix || '').trim();
  const title = unit ? `${def.label} (${unit})` : def.label;
  ctx.fillText(title, padLeft, 12);

  const ticks = 4;
  ctx.lineWidth = Math.max(1, Math.round(1 * scale));
  for (let i = 0; i < ticks; i++) {
    const t = ticks === 1 ? 0 : i / (ticks - 1);
    const y = padTop + (1 - t) * plotH;
    const val = min + span * t;

    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(rect.width - padRight, y);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText(formatTimelineOverlayAxisValue(val), 4, y + 3);
  }

  const midX = padLeft + plotW / 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  ctx.lineWidth = Math.max(1.5, Math.round(2 * scale));
  ctx.shadowColor = 'rgba(0,0,0,0.55)';
  ctx.shadowBlur = 3;
  ctx.beginPath();
  ctx.moveTo(midX, padTop);
  ctx.lineTo(midX, padTop + plotH);
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.beginPath();
  ctx.moveTo(midX, padTop - 1);
  ctx.lineTo(midX - 4 * scale, padTop - 7 * scale);
  ctx.lineTo(midX + 4 * scale, padTop - 7 * scale);
  ctx.closePath();
  ctx.fill();

  for (const series of metricData.series || []) {
    const pts = series?.points || [];
    if (!pts.length) continue;
    const color = series?.color || '#ffffff';
    ctx.strokeStyle = rgbaFromHex(color, 0.92);
    ctx.lineWidth = 2;
    ctx.beginPath();

    let started = false;
    let prev = null;
    const maxGap = Number(series?.maxGapSec) || 0;
    for (const p of pts) {
      if (!Number.isFinite(p?.ts) || !Number.isFinite(p?.v)) continue;
      const x = padLeft + ((p.ts - graphMeta.startTs) / graphMeta.windowSec) * plotW;
      const y = padTop + (1 - (p.v - min) / span) * plotH;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else if (prev && maxGap > 0 && (p.ts - prev.ts) > maxGap) {
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      prev = p;
    }
    if (started) ctx.stroke();
  }
}

function removeTimelineMetricOverlay(metricKey) {
  const selected = getUnifiedTimelineOverlayMetricSet();
  if (!selected.has(metricKey)) return;
  selected.delete(metricKey);
  applyUnifiedTimelineOverlayMetricSet(selected);
  updateTimelineStats();
}

function clearTimelineMetricSharedPanel() {
  const panel = el('timeline-stat-panel');
  if (panel) panel.innerHTML = '';
  const layout = el('layout');
  if (layout) layout.classList.remove('stat-panel-open');
  syncInlineHeatmapPanelLayout();
}

function renderTimelineMetricSharedPanel(selected, slots) {
  const panel = el('timeline-stat-panel');
  const layout = el('layout');
  if (!panel || !layout) return;

  const graphData = state.tl?._metricOverlayGraphData;
  if (!shouldUseTimelineMetricSharedPanel(slots) || !selected?.size || !graphData?.metrics) {
    panel.innerHTML = '';
    layout.classList.remove('stat-panel-open');
    syncInlineHeatmapPanelLayout();
    return;
  }

  layout.classList.add('stat-panel-open');
  syncInlineHeatmapPanelLayout();
  panel.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'timeline-stat-panel-head';
  const title = document.createElement('span');
  title.textContent = 'Stat graphs';
  const hint = document.createElement('span');
  hint.textContent = 'Click to remove';
  head.append(title, hint);
  panel.appendChild(head);

  let visibleCount = 0;
  for (const def of TIMELINE_STAT_DEFS) {
    const metricData = graphData.metrics?.[def.key];
    if (!selected.has(def.key) || !metricData) continue;
    const graph = document.createElement('div');
    graph.className = 'video-metric-graph';
    graph.title = `Remove ${def.label} graph`;
    graph.onclick = () => removeTimelineMetricOverlay(def.key);
    const canvas = document.createElement('canvas');
    canvas.className = 'video-metric-graph-canvas';
    graph.appendChild(canvas);
    panel.appendChild(graph);
    renderTimelineOverlayMetricGraph(canvas, metricData, def, graphData);
    visibleCount++;
  }

  if (!visibleCount) {
    const empty = document.createElement('div');
    empty.className = 'timeline-stat-panel-empty';
    empty.textContent = 'No graph data at this playhead.';
    panel.appendChild(empty);
  }
}

function updateSlotMetricOverlay(slot) {
  const wrap = slot?._metricOverlayEl;
  if (!wrap) return;

  const selected = getUnifiedTimelineOverlayMetricSet();
  const graphMode = !!state.timelineStatOverlayGraph;
  const useSharedPanel = graphMode && shouldUseTimelineMetricSharedPanel();
  wrap.classList.toggle('graph-mode', graphMode);
  wrap.classList.toggle('chip-click-mode', !graphMode);
  if (useSharedPanel) {
    for (const item of Object.values(slot._metricOverlayItems || {})) {
      if (item?.root) item.root.style.display = 'none';
    }
    for (const item of Object.values(slot._metricOverlayGraphs || {})) {
      if (item?.root) item.root.style.display = 'none';
    }
    wrap.style.display = 'none';
    return;
  }

  if (!slot._metricOverlayItems) slot._metricOverlayItems = {};
  if (!slot._metricOverlayGraphs) slot._metricOverlayGraphs = {};
  if (!selected.size) {
    for (const item of Object.values(slot._metricOverlayItems)) {
      if (item?.root) item.root.style.display = 'none';
    }
    for (const item of Object.values(slot._metricOverlayGraphs)) {
      if (item?.root) item.root.style.display = 'none';
    }
    wrap.style.display = 'none';
    return;
  }

  if (graphMode) {
    for (const item of Object.values(slot._metricOverlayItems)) {
      if (item?.root) item.root.style.display = 'none';
    }

    const graphData = state.tl?._metricOverlayGraphData;
    const allowed = slot?._metricOverlayGraphKeys;
    const layout = slot?._metricOverlayGraphLayout;
    if (layout?.gap != null) wrap.style.gap = `${layout.gap}px`;
    else wrap.style.gap = '';
    let visibleCount = 0;
    for (const def of TIMELINE_STAT_DEFS) {
      const metricData = graphData?.metrics?.[def.key];
      const item = slot._metricOverlayGraphs[def.key];
      if (!selected.has(def.key) || !metricData || (allowed && !allowed.has(def.key))) {
        if (item?.root) item.root.style.display = 'none';
        continue;
      }

      let refs = item;
      if (!refs) {
        const graph = document.createElement('div');
        graph.className = 'video-metric-graph';
        graph.title = `Remove ${def.label} graph`;
        graph.onclick = () => removeTimelineMetricOverlay(def.key);
        const canvas = document.createElement('canvas');
        canvas.className = 'video-metric-graph-canvas';
        graph.appendChild(canvas);
        refs = { root: graph, canvas };
        slot._metricOverlayGraphs[def.key] = refs;
      }
      if (refs.root.parentNode !== wrap) wrap.appendChild(refs.root);

      refs.root.style.display = 'block';
      if (layout?.rowHeight != null) refs.root.style.height = `${layout.rowHeight}px`;
      else refs.root.style.height = '';
      renderTimelineOverlayMetricGraph(refs.canvas, metricData, def, graphData);
      visibleCount++;
    }

    wrap.style.display = visibleCount ? 'flex' : 'none';
    return;
  }

  let visibleCount = 0;
  for (const def of TIMELINE_STAT_DEFS) {
    const item = slot._metricOverlayItems[def.key];
    if (!selected.has(def.key)) {
      if (item?.root) item.root.style.display = 'none';
      continue;
    }

    const value = slot?._timelineStatValues?.[def.key];

    let refs = item;
      if (!refs) {
        const chip = document.createElement('div');
        chip.className = 'video-metric-chip';
        chip.title = `Remove ${def.label}`;
        chip.setAttribute('role', 'button');
        chip.tabIndex = 0;
        chip.onclick = () => removeTimelineMetricOverlay(def.key);
        chip.onkeydown = (ev) => {
          if (ev.key !== 'Enter' && ev.key !== ' ') return;
          ev.preventDefault();
          removeTimelineMetricOverlay(def.key);
        };

        const key = document.createElement('span');
        key.className = 'video-metric-chip-key';
        key.textContent = def.overlayLabel;

        const val = document.createElement('span');
        val.className = 'video-metric-chip-val';

        chip.append(key, val);
        refs = { root: chip, value: val };
        slot._metricOverlayItems[def.key] = refs;
    }
    if (refs.root.parentNode !== wrap) wrap.appendChild(refs.root);

    const hasSeries = slotHasTimelineMetricSeries(slot, def.key);
    refs.root.style.display = 'inline-flex';
    refs.root.style.borderColor = rgbaFromHex(slot.color || '#c8cdd0', 0.42);
    refs.value.textContent = formatTimelineMetricValue(value, def.suffix);
    refs.value.style.color = value == null ? 'rgba(255,255,255,0.82)' : (slot.color || '#ffffff');
    refs.root.style.opacity = hasSeries ? '1' : '0.72';
    visibleCount++;
  }

  for (const item of Object.values(slot._metricOverlayGraphs)) {
    if (item?.root) item.root.style.display = 'none';
  }

  wrap.style.display = visibleCount ? 'flex' : 'none';
}

function toggleTimelineMetricOverlay(slot, metricKey) {
  const selected = getUnifiedTimelineOverlayMetricSet();
  if (selected.has(metricKey)) selected.delete(metricKey);
  else selected.add(metricKey);
  applyUnifiedTimelineOverlayMetricSet(selected);
  updateTimelineStats();
}

function populateTimelineStats() {
  const wrap = el('tl-stats');
  if (!wrap) return;
  wrap.innerHTML = '';

  const slots = state.tl?.athleteSlots || [];
  if (!slots.length) return;

  const row = document.createElement('div');
  row.className = 'tl-stats-row';

  for (const slot of slots) {
    const name = document.createElement('span');
    name.className = 'tl-ath-name';
    name.textContent = slot.name || 'Athlete';
    name.style.color = slot.color || '#c8cdd0';

    slot._timelineStatEls = {
      name,
      metrics: {},
    };
  }

  for (const def of TIMELINE_STAT_DEFS) {
    const group = document.createElement('div');
    group.className = 'tl-stat-group';

    const label = document.createElement('span');
    label.className = 'tl-stat-label';
    label.textContent = def.label;
    label.title = def.title || def.label;
    group.appendChild(label);

    for (const slot of slots) {
      const metric = document.createElement('button');
      metric.type = 'button';
      metric.className = 'tl-ath-metric';
      metric.title = `${slot.name || 'Athlete'} ${def.title || def.label}. Click to toggle the overlay in all video panes.`;
      const dot = document.createElement('span');
      dot.className = 'tl-ath-metric-dot';
      dot.style.background = slot.color || '#c8cdd0';
      const key = document.createElement('span');
      key.className = 'tl-ath-metric-key';
      key.textContent = def.label;
      const value = document.createElement('span');
      value.className = 'tl-ath-metric-val';
      value.style.color = slot.color || '#c8cdd0';
      metric.onclick = () => toggleTimelineMetricOverlay(slot, def.key);
      metric.append(dot, key, value);
      slot._timelineStatEls.metrics[def.key] = { button: metric, value };
      group.appendChild(metric);
    }
    row.appendChild(group);
  }

  wrap.appendChild(row);
  updateTimelineStats();
}

function updateTimelineStats(force = false) {
  const wrap = el('tl-stats');
  if (!wrap) return;

  const slots = state.tl?.athleteSlots || [];
  if (!slots.length) {
    wrap.innerHTML = '';
    clearTimelineMetricSharedPanel();
    return;
  }
  if (state.tl?.playing && !force) {
    const now = performance.now();
    const last = Number(state.tl._lastStatsUiUpdateAt) || 0;
    if (now - last < TIMELINE_STATS_UI_REFRESH_MS) return;
    state.tl._lastStatsUiUpdateAt = now;
  } else {
    state.tl._lastStatsUiUpdateAt = performance.now();
  }

  const currentTs = state.tl?.currentTs;
  const windowSec = clampTimelineStatsWindowSec(state.timelineStatsWindowSec);
  const refreshProcessedStats = shouldRefreshProcessedTimelineStats(currentTs, windowSec);
  const selectedMetrics = getUnifiedTimelineOverlayMetricSet(slots);
  const overlayGraphEnabled = !!state.timelineStatOverlayGraph;
  if (overlayGraphEnabled && selectedMetrics.size) {
    const overlayWindowSec = getTimelineOverlayGraphWindowSec();
    state.tl._metricOverlayGraphData = buildTimelineOverlayGraphData(
      selectedMetrics,
      slots,
      currentTs,
      overlayWindowSec,
    );
    assignTimelineOverlayGraphSlots(selectedMetrics, slots);
  } else {
    state.tl._metricOverlayGraphData = null;
    assignTimelineOverlayGraphSlots(new Set(), slots);
  }
  if (refreshProcessedStats) markProcessedTimelineStatsRefresh(currentTs, windowSec);
  wrap.title = 'SOG / Heel / Pitch show the nearest sample; COG / Roll M / Trunk / Rudder / Boom show nearest available sample; VMG shows absolute value and TWA uses current motion direction against the estimated wind';

  for (const slot of slots) {
    const refs = slot._timelineStatEls;
    if (!refs) continue;
    refs.name.textContent = slot.name || 'Athlete';
    refs.name.style.color = slot.color || '#c8cdd0';

    // Scalar metrics show the mean of the trailing second (computeSeriesTrailingMean).
    // Angular metrics (heading/cog/twa) keep nearest-sample + their existing angle
    // smoothing below, since arithmetic means are wrong across the 0/360 wrap.
    const processedStatValues = refreshProcessedStats
      ? {
          roll: computeSeriesTrailingMean(slot.statsSeries?.roll, currentTs),
          trunk: computeSeriesTrailingMean(slot.statsSeries?.trunk, currentTs),
          rudder: computeSeriesTrailingMean(slot.statsSeries?.rudder, currentTs),
          boom: computeSeriesTrailingMean(slot.statsSeries?.boom, currentTs),
        }
      : (slot._processedTimelineStatValues || createEmptyProcessedTimelineStatValues());
    slot._processedTimelineStatValues = processedStatValues;

    const statValues = {
      sog: computeSeriesTrailingMean(slot.statsSeries?.sog, currentTs),
      heading: computeSeriesInstantValue(slot.statsSeries?.heading, currentTs),
      cog: computeSeriesInstantValue(slot.statsSeries?.cog, currentTs),
      heel: computeSeriesTrailingMean(slot.statsSeries?.heel, currentTs),
      pitch: computeSeriesTrailingMean(slot.statsSeries?.pitch, currentTs),
      ...processedStatValues,
    };
    statValues.twa = computeLiveTimelineTwa(slot, currentTs);
    statValues.heading = smoothTimelineAngleValue(slot, 'heading', statValues.heading, currentTs, {
      maxHoldSec: 6,
      alpha: 0.42,
    });
    statValues.twa = smoothTimelineLinearValue(slot, 'twa', statValues.twa, currentTs, {
      maxHoldSec: 10,
      alpha: 0.38,
    });
    statValues.vmg = computeLiveTimelineVmg(slot, currentTs, statValues.sog, statValues.twa);
    slot._timelineStatValues = statValues;

    for (const def of TIMELINE_STAT_DEFS) {
      const metricRef = refs.metrics?.[def.key];
      if (!metricRef) continue;
      const hasSeries = slotHasTimelineMetricSeries(slot, def.key);
      const value = statValues[def.key];
      const text = formatTimelineMetricValue(value, def.suffix);
      if (metricRef.value.textContent !== text) metricRef.value.textContent = text;
      metricRef.value.classList.toggle('empty', value == null);
      metricRef.value.style.color = value == null ? '#6f7886' : (slot.color || '#c8cdd0');
      metricRef.button.disabled = !hasSeries;
      metricRef.button.classList.toggle('disabled', !hasSeries);

      const isActive = selectedMetrics.has(def.key);
      metricRef.button.classList.toggle('active', isActive);
      metricRef.button.style.borderColor = isActive ? (slot.color || '#c8cdd0') : 'transparent';
      metricRef.button.style.background = isActive ? rgbaFromHex(slot.color || '#c8cdd0', 0.15) : 'transparent';
    }

    updateSlotMetricOverlay(slot);
  }
  renderTimelineMetricSharedPanel(selectedMetrics, slots);
}

function smoothTimelineLinearValue(slot, metricKey, value, ts, { maxHoldSec = 6, alpha = 0.45 } = {}) {
  if (!slot) return Number.isFinite(Number(value)) ? Number(value) : null;
  const numTs = Number(ts);
  if (!slot._metricSmoothCache || typeof slot._metricSmoothCache !== 'object') slot._metricSmoothCache = {};
  const cache = slot._metricSmoothCache;
  const prev = cache[metricKey];
  const next = Number(value);
  if (!Number.isFinite(next)) {
    if (prev && Number.isFinite(prev.value) && Number.isFinite(numTs) && Math.abs(numTs - prev.ts) <= maxHoldSec) {
      return prev.value;
    }
    return null;
  }
  let out = next;
  if (prev && Number.isFinite(prev.value) && Number.isFinite(numTs) && Math.abs(numTs - prev.ts) <= maxHoldSec) {
    const a = Math.max(0, Math.min(1, Number(alpha)));
    out = prev.value + (next - prev.value) * a;
  }
  if (Number.isFinite(numTs)) cache[metricKey] = { value: out, ts: numTs };
  return out;
}

function smoothTimelineAngleValue(slot, metricKey, value, ts, { maxHoldSec = 6, alpha = 0.45 } = {}) {
  if (!slot) return Number.isFinite(Number(value)) ? wrapWindDegrees(Number(value)) : null;
  const numTs = Number(ts);
  if (!slot._metricSmoothCache || typeof slot._metricSmoothCache !== 'object') slot._metricSmoothCache = {};
  const cache = slot._metricSmoothCache;
  const prev = cache[metricKey];
  const nextRaw = Number(value);
  if (!Number.isFinite(nextRaw)) {
    if (prev && Number.isFinite(prev.value) && Number.isFinite(numTs) && Math.abs(numTs - prev.ts) <= maxHoldSec) {
      return wrapWindDegrees(prev.value);
    }
    return null;
  }
  let out = wrapWindDegrees(nextRaw);
  if (prev && Number.isFinite(prev.value) && Number.isFinite(numTs) && Math.abs(numTs - prev.ts) <= maxHoldSec) {
    const a = Math.max(0, Math.min(1, Number(alpha)));
    const delta = angleDifferenceDeg(out, prev.value);
    out = wrapWindDegrees(prev.value + delta * a);
  }
  if (Number.isFinite(numTs)) cache[metricKey] = { value: out, ts: numTs };
  return out;
}

// First index in a ts-sorted point array whose ts >= target (binary search).
function lowerBoundByTs(pts, target) {
  let lo = 0, hi = pts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (pts[mid].ts < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function sogGapThreshold(pts) {
  if (!pts || pts.length < 3) return 20;
  const gaps = [];
  for (let i = 1; i < pts.length && gaps.length < 200; i++) {
    const dt = pts[i].ts - pts[i - 1].ts;
    if (Number.isFinite(dt) && dt > 0) gaps.push(dt);
  }
  if (!gaps.length) return 20;
  gaps.sort((a, b) => a - b);
  const med = gaps[Math.floor(gaps.length / 2)] || 3;
  return Math.max(12, med * 8);
}

function rgbaFromHex(hex, alpha = 1) {
  if (typeof hex !== 'string' || !hex.startsWith('#') || (hex.length !== 7 && hex.length !== 4)) {
    return `rgba(120,130,140,${alpha})`;
  }
  const full = hex.length === 4
    ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`
    : hex;
  const r = parseInt(full.slice(1, 3), 16) || 0;
  const g = parseInt(full.slice(3, 5), 16) || 0;
  const b = parseInt(full.slice(5, 7), 16) || 0;
  return `rgba(${r},${g},${b},${alpha})`;
}

function shortVideoLabel(name, max = 18) {
  const raw = String(name || '');
  if (raw.length <= max) return raw;
  const extIdx = raw.lastIndexOf('.');
  const ext = extIdx >= 0 ? raw.slice(extIdx) : '';
  const stem = extIdx >= 0 ? raw.slice(0, extIdx) : raw;
  const room = Math.max(4, max - ext.length - 1);
  return `${stem.slice(0, room)}...${ext}`;
}

function drawCoverageCanvas() {
  const canvas = el('coverage-canvas');
  const wrap = el('sog-canvas-wrap');
  if (!canvas || !wrap) return;

  const w = wrap.clientWidth || wrap.offsetWidth || 800;
  const h = canvas.clientHeight || canvas.offsetHeight || 18;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(w * dpr));
  canvas.height = Math.max(1, Math.floor(h * dpr));
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const tl = state.tl;
  if (tl.globalStart == null || tl.globalEnd == null) return;
  const [vStart, vEnd] = tlViewRange();
  const dur = vEnd - vStart;
  if (dur <= 0 || !tl.athleteSlots.length) return;

  ctx.fillStyle = '#181c20';
  ctx.fillRect(0, 0, w, h);

  const rows = tl.athleteSlots.length;
  const gap = 1;
  const rowH = Math.max(4, (h - (rows - 1) * gap) / rows);
  ctx.font = '10px sans-serif';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < rows; i++) {
    const slot = tl.athleteSlots[i];
    const y = i * (rowH + gap);
    ctx.fillStyle = 'rgba(110,120,130,0.18)';
    ctx.fillRect(0, y, w, rowH);
    for (const v of slot.videos) {
      if (v.ts_start == null) continue;
      const endTs = v.ts_end ?? (Number.isFinite(v.duration_sec) ? v.ts_start + v.duration_sec : null);
      if (endTs == null) continue;
      const x0 = ((v.ts_start - vStart) / dur) * w;
      const x1 = ((endTs - vStart) / dur) * w;
      if (x1 <= 0 || x0 >= w) continue;
      const bx0 = Math.max(0, x0);
      const bx1 = Math.min(w, x1);
      const bw = bx1 - bx0;
      if (bw <= 0.75) continue;

      ctx.fillStyle = rgbaFromHex(slot.color, 0.62);
      ctx.fillRect(bx0, y, bw, rowH);

      if (!isIpadHeavyVideoProject() && bw > 36) {
        const label = shortVideoLabel(v.filename || v.id, Math.floor(bw / 7.2));
        ctx.save();
        ctx.beginPath();
        ctx.rect(bx0, y, bw, rowH);
        ctx.clip();
        ctx.fillStyle = 'rgba(255,255,255,0.93)';
        ctx.fillText(label, bx0 + 2, y + rowH / 2);
        ctx.restore();
      }
    }
  }

  // Row separators
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let i = 1; i < rows; i++) {
    const y = i * (rowH + gap) - 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
}

// Helper: get visible time range (zoom or full)
function tlViewRange() {
  const tl = state.tl;
  const vs = tl.viewStart ?? tl.globalStart;
  const ve = tl.viewEnd ?? tl.globalEnd;
  return [vs, ve];
}

function renderTimelineMediaLayer() {
  const layer = el('tl-media-layer');
  const wrap = el('sog-canvas-wrap');
  const coverage = el('coverage-canvas');
  if (!layer || !wrap || !coverage) return;

  const mediaItems = getPlaybackTimelineMedia();
  updatePhonePlaybackToggleButton();
  const tl = state.tl;
  if (!mediaItems.length || tl.globalStart == null || tl.globalEnd == null) {
    layer.innerHTML = '';
    layer.style.display = 'none';
    return;
  }

  const w = wrap.clientWidth || wrap.offsetWidth || 0;
  const [vStart, vEnd] = tlViewRange();
  const dur = vEnd - vStart;
  if (!(w > 0) || !(dur > 0)) {
    layer.innerHTML = '';
    layer.style.display = 'none';
    return;
  }

  layer.style.display = '';
  layer.innerHTML = '';
  const visibleItems = [];
  for (const item of mediaItems) {
    const startTs = getTimelineMediaStartTs(item);
    const endTs = getTimelineMediaEndTs(item);
    if (!Number.isFinite(startTs) || !Number.isFinite(endTs)) continue;
    if (endTs < vStart || startTs > vEnd) continue;
    visibleItems.push({ item, startTs, endTs });
  }

  if (!visibleItems.length) {
    layer.style.display = 'none';
    return;
  }

  const maxLanes = window.matchMedia('(max-width: 760px)').matches ? 1 : 2;
  const laneEnds = [];
  const assignments = [];
  for (const entry of visibleItems) {
    let lane = laneEnds.findIndex(lastEndTs => entry.startTs > lastEndTs + 0.25);
    if (lane < 0 && laneEnds.length < maxLanes) lane = laneEnds.length;
    if (lane < 0) {
      let bestIdx = 0;
      let bestEnd = laneEnds[0] ?? Number.POSITIVE_INFINITY;
      for (let i = 1; i < laneEnds.length; i++) {
        if (laneEnds[i] < bestEnd) {
          bestEnd = laneEnds[i];
          bestIdx = i;
        }
      }
      lane = bestIdx;
    }
    laneEnds[lane] = entry.endTs;
    assignments.push({ ...entry, lane });
  }

  const laneCount = Math.max(1, Math.min(maxLanes, laneEnds.length || 1));
  const coverageTop = coverage.offsetTop || 0;
  const coverageH = coverage.clientHeight || coverage.offsetHeight || 18;
  const padY = 2;
  const gapY = laneCount > 1 ? 2 : 0;
  const barH = Math.max(5, Math.floor((coverageH - padY * 2 - gapY * (laneCount - 1)) / laneCount));
  const playbackEnabled = isPhonePlaybackEnabled();
  const activeItem = getPhonePlaybackItemAtTs(tl.currentTs, state.phonePlayback?.currentFileId);

  assignments.forEach((entry, index) => {
    const { item, startTs, endTs, lane } = entry;
    const clampedStart = Math.max(vStart, Math.min(vEnd, startTs));
    const clampedEnd = Math.max(clampedStart, Math.min(vEnd, endTs));
    const x0 = ((clampedStart - vStart) / dur) * w;
    const x1 = ((clampedEnd - vStart) / dur) * w;
    const bw = Math.max(7, x1 - x0);
    const accent = state.videoColors[item.id] || PALETTE[index % PALETTE.length] || '#8db6ff';

    const marker = document.createElement('button');
    marker.type = 'button';
    marker.className = 'tl-media-segment';
    if (playbackEnabled) marker.classList.add('is-enabled');
    if (String(activeItem?.id || '') === String(item?.id || '') || _timelineMediaOverlayFileId === String(item?.id || '')) {
      marker.classList.add('is-active');
    }
    marker.style.left = `${Math.max(0, x0)}px`;
    marker.style.top = `${coverageTop + padY + lane * (barH + gapY)}px`;
    marker.style.width = `${bw}px`;
    marker.style.height = `${barH}px`;
    marker.style.setProperty('--tl-media-border', rgbaFromHex(accent, playbackEnabled ? 0.78 : 0.44));
    marker.style.setProperty('--tl-media-fill-soft', rgbaFromHex(accent, playbackEnabled ? 0.34 : 0.22));
    marker.style.setProperty('--tl-media-fill', rgbaFromHex(accent, playbackEnabled ? 0.9 : 0.64));
    const baseName = getTimelineMediaDisplayName(item).replace(/\.[^.]+$/, '');
    const labelChars = Math.max(0, Math.floor((bw - 28) / 5.8));
    const shortName = labelChars >= 8
      ? (baseName.length > labelChars ? `${baseName.slice(0, Math.max(3, labelChars - 3))}...` : baseName)
      : '';
    const actionText = `Click to jump playback to this clip start (${new Date(startTs * 1000).toLocaleTimeString()})`;
    marker.title = `${getTimelineMediaDisplayName(item)}\n${playbackOnlyVideoTitle(item)}\n${new Date(startTs * 1000).toLocaleString()} - ${new Date(endTs * 1000).toLocaleString()}\n${actionText}`;
    marker.innerHTML = `
      <span class="tl-media-segment-kind">${getTimelineMediaKindLabel(item)}</span>
      ${shortName ? `<span class="tl-media-segment-label">${shortName}</span>` : ''}
    `;
    marker.addEventListener('pointerdown', e => {
      e.preventDefault();
      e.stopPropagation();
    });
    marker.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      const clipStartTs = Number.isFinite(startTs) ? startTs : getTimelineMediaStartTs(item);
      if (Number.isFinite(clipStartTs)) tlSeekTo(clipStartTs);
    });
    layer.appendChild(marker);
  });
}

function closeTimelineMediaOverlay() {
  const modal = el('timeline-media-modal');
  const body = el('timeline-media-modal-body');
  const empty = el('timeline-media-modal-empty');
  if (body) body.innerHTML = '';
  if (empty) empty.style.display = 'none';
  if (modal) modal.classList.remove('open');
  if (_timelineMediaOverlayUrl) {
    try { URL.revokeObjectURL(_timelineMediaOverlayUrl); } catch {}
  }
  _timelineMediaOverlayUrl = null;
  _timelineMediaOverlayFileId = null;
  renderTimelineMediaLayer();
}

async function openTimelineMediaOverlay(item) {
  if (!item?.id) return;
  if (state.tl.playing) tlPause();

  closeTimelineMediaOverlay();

  const modal = el('timeline-media-modal');
  const title = el('timeline-media-modal-title');
  const subtitle = el('timeline-media-modal-subtitle');
  const body = el('timeline-media-modal-body');
  const empty = el('timeline-media-modal-empty');
  if (!modal || !title || !subtitle || !body || !empty) return;

  const startTs = getTimelineMediaStartTs(item);
  title.textContent = getTimelineMediaDisplayName(item);
  subtitle.textContent = [
    Number.isFinite(startTs) ? `Captured ${new Date(startTs * 1000).toLocaleString()}` : 'Capture time unavailable',
    playbackOnlyVideoTitle(item),
  ].filter(Boolean).join(' | ');

  modal.classList.add('open');
  empty.style.display = 'none';
  _timelineMediaOverlayFileId = String(item.id);
  renderTimelineMediaLayer();

  const file = await FM.getFileForReading(item.id);
  if (!file) {
    empty.style.display = 'flex';
    return;
  }

  const fileType = String(file.type || '').toLowerCase();
  const isImage = fileType.startsWith('image/') || /\.(jpe?g|png|gif|webp|heic|heif)$/i.test(file.name || '');
  _timelineMediaOverlayUrl = URL.createObjectURL(file);

  if (isImage) {
    const img = document.createElement('img');
    img.alt = getTimelineMediaDisplayName(item);
    img.src = _timelineMediaOverlayUrl;
    body.appendChild(img);
    return;
  }

  const video = document.createElement('video');
  video.controls = true;
  video.preload = 'metadata';
  video.playsInline = true;
  video.setAttribute('playsinline', '');
  video.src = _timelineMediaOverlayUrl;
  const seekSec = Number.isFinite(state.tl.currentTs) && videoContainsAbsTs(item, state.tl.currentTs)
    ? absTs2VideoSec(item, state.tl.currentTs)
    : 0;
  video.addEventListener('loadedmetadata', () => {
    if (!Number.isFinite(seekSec)) return;
    try {
      const maxSeek = Number.isFinite(video.duration) && video.duration > 0
        ? Math.max(0, video.duration - 0.05)
        : seekSec;
      video.currentTime = Math.max(0, Math.min(seekSec, maxSeek));
    } catch {}
  }, { once: true });
  body.appendChild(video);
}

function keepTimelinePlayheadInView(marginFrac = 0.16) {
  const tl = state.tl;
  if (tl.globalStart == null || tl.globalEnd == null || tl.currentTs == null) return false;
  if (tl.viewStart == null || tl.viewEnd == null) return false;

  const globalDur = tl.globalEnd - tl.globalStart;
  const viewDur = tl.viewEnd - tl.viewStart;
  if (!(globalDur > 0) || !(viewDur > 0) || viewDur >= globalDur - 0.5) return false;

  const margin = Math.max(1, Math.min(viewDur * 0.35, viewDur * marginFrac));
  let newStart = tl.viewStart;
  let newEnd = tl.viewEnd;

  if (tl.currentTs < tl.viewStart + margin) {
    newStart = tl.currentTs - margin;
    newEnd = newStart + viewDur;
  } else if (tl.currentTs > tl.viewEnd - margin) {
    newEnd = tl.currentTs + margin;
    newStart = newEnd - viewDur;
  } else {
    return false;
  }

  if (newStart < tl.globalStart) {
    newStart = tl.globalStart;
    newEnd = newStart + viewDur;
  }
  if (newEnd > tl.globalEnd) {
    newEnd = tl.globalEnd;
    newStart = newEnd - viewDur;
  }

  if (Math.abs(newStart - tl.viewStart) < 1e-6 && Math.abs(newEnd - tl.viewEnd) < 1e-6) return false;
  tl.viewStart = Math.max(tl.globalStart, newStart);
  tl.viewEnd = Math.min(tl.globalEnd, newEnd);
  return true;
}

// ── SOG Canvas drawing ─────────────────────────────────────────────────
function drawSogCanvas() {
  const canvas = el('sog-canvas');
  if(!canvas) return;
  const wrap = el('sog-canvas-wrap');
  const w = wrap.clientWidth || wrap.offsetWidth || 800;
  const h = canvas.clientHeight || canvas.offsetHeight || 72;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(w * dpr));
  canvas.height = Math.max(1, Math.floor(h * dpr));
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const tl = state.tl;
  if (tl.globalStart == null || tl.globalEnd == null) {
    drawCoverageCanvas();
    renderTimelineMediaLayer();
    return;
  }
  const [vStart, vEnd] = tlViewRange();
  const dur = vEnd - vStart;
  if (dur <= 0) {
    drawCoverageCanvas();
    renderTimelineMediaLayer();
    return;
  }
  const isZoomed = (tl.viewStart != null || tl.viewEnd != null) &&
    (vStart > tl.globalStart + 0.5 || vEnd < tl.globalEnd - 0.5);
  const PAD_TOP = 4;
  const PAD_BOT = 4;
  const plotH = Math.max(10, h - PAD_TOP - PAD_BOT);

  // Draw background grid
  ctx.strokeStyle = 'rgba(60,75,100,0.5)';
  ctx.lineWidth = 0.5;
  // Time marks: adaptive tick spacing
  const tickSec = dur > 3600 ? 300 : dur > 600 ? 60 : dur > 120 ? 30 : dur > 30 ? 10 : 5;
  for (let t = Math.ceil(vStart / tickSec) * tickSec; t <= vEnd; t += tickSec) {
    const x = (t - vStart) / dur * w;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillStyle = 'rgba(160,175,200,0.7)';
    ctx.font = '10px sans-serif';
    const label = fmtClock(t);
    ctx.fillText(label, x + 2, h - 2);
  }

  // Find max SOG across all slots for scaling. maxSog over the full series is a
  // fixed property of the data, so cache it per slot rather than rescanning every
  // (60fps) redraw — the full-series scan was a big cost when zoomed out.
  let maxSog = 0;
  for (const slot of tl.athleteSlots) {
    if (slot._sogMaxCache == null || slot._sogMaxCacheLen !== slot.sogPts.length) {
      let m = 0;
      for (const p of slot.sogPts) { if (p.sog > m) m = p.sog; }
      slot._sogMaxCache = m;
      slot._sogMaxCacheLen = slot.sogPts.length;
    }
    if (slot._sogMaxCache > maxSog) maxSog = slot._sogMaxCache;
  }
  if (maxSog < 1) maxSog = 10;
  maxSog *= 1.1; // 10% headroom

  // Draw SOG curves — one per athlete, each in their color. Only the points inside
  // the visible time window are walked (binary-searched), and they're bucketed to
  // ~1 column per pixel: drawing more points than the canvas is wide is wasted work
  // and was the cause of playback jitter when zoomed out (the whole track was being
  // stroked every frame). Cost now scales with canvas width, not point count/zoom.
  for (const slot of tl.athleteSlots) {
    const pts = slot.sogPts;
    if (!pts.length) continue;
    const gapLimit = sogGapThreshold(pts);
    // Window: include one point on each side so the line reaches the canvas edges.
    let startIdx = lowerBoundByTs(pts, vStart);
    if (startIdx > 0) startIdx--;
    let endIdx = lowerBoundByTs(pts, vEnd);
    if (endIdx < pts.length) endIdx++;

    ctx.beginPath();
    ctx.strokeStyle = slot.color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.9;
    let first = true;
    let prevTs = null;
    let curCol = -1;        // current pixel column being bucketed
    let colMinY = 0, colMaxY = 0, colMinX = 0;
    const yFor = sog => PAD_TOP + plotH - (sog / maxSog) * plotH;
    const flushCol = () => {
      if (curCol < 0) return;
      // Draw the column's vertical extent so peaks/troughs survive downsampling.
      if (first) { ctx.moveTo(colMinX, colMinY); first = false; }
      else ctx.lineTo(colMinX, colMinY);
      if (colMaxY !== colMinY) ctx.lineTo(colMinX, colMaxY);
    };
    for (let i = startIdx; i < endIdx; i++) {
      const p = pts[i];
      const gapBreak = (prevTs != null && (p.ts - prevTs) > gapLimit);
      prevTs = p.ts;
      const x = (p.ts - vStart) / dur * w;
      const y = yFor(p.sog);
      if (gapBreak) { flushCol(); curCol = -1; first = true; }
      const col = x | 0;
      if (col !== curCol) {
        flushCol();
        curCol = col;
        colMinX = x; colMinY = y; colMaxY = y;
      } else {
        if (y < colMinY) colMinY = y;
        if (y > colMaxY) colMaxY = y;
      }
    }
    flushCol();
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Draw segment regions as translucent vertical bands with all athlete colors
  for (const seg of state.segments) {
    if (seg.tsStart == null || seg.tsEnd == null) continue;
    const x1 = (seg.tsStart - vStart) / dur * w;
    const x2 = (seg.tsEnd - vStart) / dur * w;
    if (x2 < 0 || x1 > w) continue;
    const segAthletes = getSegmentAthletes(seg);
    const segCols = segAthletes.length > 0 ? segAthletes.map(a => a.color) : [getSegmentColor(seg)];
    const bandW = Math.min(w, x2) - Math.max(0, x1);
    const bandH = h;
    const stripeH = bandH / segCols.length;
    for (let si = 0; si < segCols.length; si++) {
      const col = segCols[si];
      const r3 = parseInt(col.slice(1, 3), 16) || 0;
      const g3 = parseInt(col.slice(3, 5), 16) || 0;
      const b3 = parseInt(col.slice(5, 7), 16) || 0;
      ctx.fillStyle = `rgba(${r3},${g3},${b3},0.12)`;
      ctx.fillRect(Math.max(0, x1), si * stripeH, bandW, stripeH);
    }
    const mainCol = segCols[0];
    if (x1 >= 0 && x1 <= w) {
      ctx.strokeStyle = mainCol; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, h); ctx.stroke();
    }
    if (x2 >= 0 && x2 <= w) {
      ctx.strokeStyle = mainCol; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x2, 0); ctx.lineTo(x2, h); ctx.stroke();
    }
    const cx = (Math.max(0, x1) + Math.min(w, x2)) / 2;
    const r3m = parseInt(mainCol.slice(1, 3), 16) || 0;
    const g3m = parseInt(mainCol.slice(3, 5), 16) || 0;
    const b3m = parseInt(mainCol.slice(5, 7), 16) || 0;
    ctx.fillStyle = `rgba(${r3m},${g3m},${b3m},0.85)`;
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(seg.name, cx, 10);
    ctx.textAlign = 'start';
  }

  // Draw pending segment selection markers on scrub bar
  const ss = state.segmentSelect;
  if (ss.active && ss.tsStart != null) {
    const sx = (ss.tsStart - vStart) / dur * w;
    if (sx >= 0 && sx <= w) {
      ctx.strokeStyle = '#2ea043'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
      ctx.fillStyle = '#2ea043'; ctx.font = 'bold 9px sans-serif';
      ctx.fillText('START', sx + 3, 20);
    }
    if (ss.tsEnd != null) {
      const ex = (ss.tsEnd - vStart) / dur * w;
      if (ex >= 0 && ex <= w) {
        ctx.strokeStyle = '#e3342f'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ex, 0); ctx.lineTo(ex, h); ctx.stroke();
        ctx.fillStyle = '#e3342f'; ctx.font = 'bold 9px sans-serif';
        ctx.fillText('END', ex + 3, 20);
      }
      const x1b = Math.max(0, Math.min(sx, ex));
      const x2b = Math.min(w, Math.max(sx, ex));
      ctx.fillStyle = 'rgba(245,166,35,0.2)';
      ctx.fillRect(x1b, 0, x2b - x1b, h);
    }
  }

  // SOG axis label
  ctx.fillStyle = 'rgba(139,148,158,0.5)';
  ctx.font = '9px sans-serif';
  ctx.fillText(`${maxSog.toFixed(0)} kt`, 2, 10);

  // Athlete legend (top right)
  if (tl.athleteSlots.length > 1) {
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'end';
    let legendY = 12;
    for (const slot of tl.athleteSlots) {
      ctx.fillStyle = slot.color;
      ctx.fillText(`* ${slot.name}`, w - 6, legendY);
      legendY += 13;
    }
    ctx.textAlign = 'start';
  }

  // If zoomed, draw a minimap indicator
  if (isZoomed) {
    const globalDur = tl.globalEnd - tl.globalStart;
    const mmW = Math.min(120, w * 0.15);
    const mmH = 10;
    const mmX = w - mmW - 4;
    const mmY = 2;
    ctx.fillStyle = 'rgba(22,27,34,0.85)';
    ctx.fillRect(mmX, mmY, mmW, mmH);
    ctx.strokeStyle = 'rgba(139,148,158,0.5)';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(mmX, mmY, mmW, mmH);
    const vx1 = mmX + ((vStart - tl.globalStart) / globalDur) * mmW;
    const vx2 = mmX + ((vEnd - tl.globalStart) / globalDur) * mmW;
    ctx.fillStyle = 'rgba(29,143,216,0.5)';
    ctx.fillRect(vx1, mmY, vx2 - vx1, mmH);
    ctx.fillStyle = 'rgba(139,148,158,0.7)';
    ctx.font = '8px sans-serif';
    ctx.fillText(`${(globalDur / dur).toFixed(1)}x`, mmX + 2, mmY + mmH + 9);
  }

  drawCoverageCanvas();
  updatePlayhead();
  renderTimelineMediaLayer();
}

function updatePlayhead() {
  const tl = state.tl;
  const playhead = el('sog-playhead');
  if(!playhead || tl.globalStart == null || tl.globalEnd == null) return;
  const [vStart, vEnd] = tlViewRange();
  const dur = vEnd - vStart;
  if(dur <= 0) return;
  const wrap = el('sog-canvas-wrap');
  const w = wrap.clientWidth || wrap.offsetWidth || 800;
  const frac = (tl.currentTs - vStart) / dur;
  playhead.style.left = Math.max(0, Math.min(w, frac * w)) + 'px';
  playhead.style.display = '';
}

function updateTimeLabel() {
  const tl = state.tl;
  const label = el('tl-time');
  if(!label) return;
  if(tl.globalStart == null || tl.globalEnd == null) { label.textContent = '--:-- / --:--'; return; }
  const elapsed = (tl.currentTs || tl.globalStart) - tl.globalStart;
  const total = tl.globalEnd - tl.globalStart;
  label.textContent = `${fmtClock(tl.currentTs)} — ${fmt(elapsed)} / ${fmt(total)}`;
}

// ── Timeline seeking and video switching ───────────────────────────────
function tlSeekTo(absTs) {
  const tl = state.tl;
  if(tl.globalStart == null || tl.globalEnd == null) return;
  tl.currentTs = Math.max(tl.globalStart, Math.min(tl.globalEnd, absTs));
  prewarmLikelyVideosAtTs(tl.currentTs);
  const activeFileIds = new Set();

  // For each athlete slot, find the right video and seek it
  for(const slot of tl.athleteSlots) {
    if (isVideoSlotHidden(slot)) {
      if (slot.paneEl) slot.paneEl.style.display = 'none';
      if (slot.currentFileId || slot._desiredFileId) stopSlotPlayback(slot, false);
      continue;
    }
    const vid = findSlotVideoAtTime(slot, tl.currentTs);
    if(vid) {
      const fileId = vid.id;
      activeFileIds.add(fileId);
      const videoSec = absTs2VideoSec(vid, tl.currentTs);
      const switchPending = slot._desiredFileId === fileId;
      if(slot.currentFileId !== fileId) {
        if (slot.currentFileId) showSlotActiveVideo(slot);
        else showSlotLoadingState(slot);
        if (!switchPending) {
          if (!USE_IPAD_VIDEO_WORKAROUNDS) {
            console.log(`[tlSeekTo] slot "${slot.name}": switching to video ${fileId} (${vid.filename}) at ${videoSec?.toFixed(1)}s`);
          }
          switchSlotVideoSource(slot, fileId, videoSec, tl.playing);
        }
      } else {
        showSlotActiveVideo(slot);
        seekVideoElement(syncSlotActiveVideoRef(slot), videoSec);
        if (tl.playing && slot.videoEl.paused) tryPlaySlotVideo(slot, fileId);
      }
      monitorSlotVideoHealth(slot, performance.now());
      // Move map marker
      if(videoSec != null) movePositionMarkerByTs(fileId, tl.currentTs);
    } else {
      if (slot.videos.length > 0) showSlotEmptyState(slot);
      else if (slot.paneEl) slot.paneEl.style.display = 'none';
      if (slot.currentFileId || slot._desiredFileId) stopSlotPlayback(slot, !USE_IPAD_VIDEO_WORKAROUNDS);
      else slot._stuckSince = 0;
    }
  }
  // If no panes are visible, show the empty state
  const anyVisible = tl.athleteSlots.some(s => s.paneEl && s.paneEl.style.display !== 'none');
  showNoVideo(!anyVisible);
  syncInlineHeatmapPanelLayout();

  // Hide arrows for videos not currently playing, show for active ones
  syncActivePositionMarkers(activeFileIds);
  keepPlaybackMarkersInView(activeFileIds);
  syncPhonePlaybackToTimeline({ forceSeek: true, forceReload: false });

  if (keepTimelinePlayheadInView()) drawSogCanvas();
  else {
    updatePlayhead();
    renderTimelineMediaLayer();
  }
  updateTimeLabel();
  maybeRefreshWindEstimate(true);
  updateTimelineStats();
  updateCurrentSegmentActions();
}

/**
 * Find which video is active at a given absolute timestamp.
 * Videos are sorted by ts_start.
 */
function _findVideoAtTimeLegacy(videos, absTs) {
  for(let i = videos.length - 1; i >= 0; i--) {
    const v = videos[i];
    if(v.ts_start == null) continue;
    // Normal case: both start and end known
    if(v.ts_end != null && absTs >= v.ts_start && absTs <= v.ts_end) return v;
    // Fallback: ts_end is null (duration probe failed) — use duration_sec or treat as open-ended
    if(v.ts_end == null) {
      const dur = v.duration_sec || 7200; // fallback: 2 hours max
      if(absTs >= v.ts_start && absTs <= v.ts_start + dur) return v;
    }
  }
  return null;
}

function getVideoEndTs(v) {
  if (!v || v.ts_start == null) return null;
  if (v.ts_end != null) return v.ts_end;
  const dur = Number.isFinite(Number(v.duration_sec)) ? Number(v.duration_sec) : 7200;
  return v.ts_start + dur;
}

function videoContainsAbsTs(v, absTs) {
  if (!v || v.ts_start == null || !Number.isFinite(absTs)) return false;
  const endTs = getVideoEndTs(v);
  return endTs != null && absTs >= v.ts_start && absTs <= endTs;
}

function findVideoIndexAtTime(videos, absTs, hintIdx = null) {
  if (!Array.isArray(videos) || !videos.length || !Number.isFinite(absTs)) return -1;
  if (Number.isInteger(hintIdx) && hintIdx >= 0 && hintIdx < videos.length) {
    const hinted = videos[hintIdx];
    if (videoContainsAbsTs(hinted, absTs)) return hintIdx;
    if (hinted?.ts_start != null && absTs >= hinted.ts_start) {
      for (let i = hintIdx + 1; i < videos.length; i++) {
        const v = videos[i];
        if (videoContainsAbsTs(v, absTs)) return i;
        if (v?.ts_start != null && v.ts_start > absTs) break;
      }
    }
    const hintedEnd = getVideoEndTs(hinted);
    if (hintedEnd == null || absTs <= hintedEnd) {
      for (let i = hintIdx - 1; i >= 0; i--) {
        const v = videos[i];
        if (videoContainsAbsTs(v, absTs)) return i;
        const endTs = getVideoEndTs(v);
        if (endTs != null && absTs > endTs) break;
      }
    }
  }
  for (let i = videos.length - 1; i >= 0; i--) {
    if (videoContainsAbsTs(videos[i], absTs)) return i;
  }
  return -1;
}

function findVideoAtTime(videos, absTs) {
  const idx = findVideoIndexAtTime(videos, absTs);
  return idx >= 0 ? videos[idx] : null;
}

function findSlotVideoAtTime(slot, absTs) {
  if (!slot) return null;
  const idx = findVideoIndexAtTime(slot.videos || [], absTs, slot._lastVideoIdx);
  slot._lastVideoIdx = idx >= 0 ? idx : null;
  return idx >= 0 ? slot.videos[idx] : null;
}

/**
 * Convert absolute epoch timestamp to video-local seconds.
 * Uses the video's points to find the mapping.
 */
function absTs2VideoSec(vid, absTs) {
  const pts = vid.points;
  if(!pts?.length) {
    // Fallback for videos without GPS: compute from ts_start directly
    if(vid.ts_start != null) return Math.max(0, absTs - vid.ts_start);
    return null;
  }
  // Find closest point by ts and use its video_s, then interpolate
  let loIdx = 0;
  for(let i = 0; i < pts.length; i++) {
    if(pts[i].ts != null && pts[i].ts <= absTs) loIdx = i;
    else break;
  }
  const lo = pts[loIdx];
  const hiIdx = Math.min(loIdx + 1, pts.length - 1);
  const hi = pts[hiIdx];
  if(lo.video_s == null) return null;
  if(hiIdx === loIdx || hi.ts == null || hi.video_s == null || hi.ts === lo.ts) return lo.video_s;
  // Linear interpolation
  const frac = (absTs - lo.ts) / (hi.ts - lo.ts);
  return lo.video_s + frac * (hi.video_s - lo.video_s);
}

/**
 * Convert video-local seconds to absolute epoch timestamp.
 */
function videoSec2AbsTs(vid, videoSec) {
  const pts = vid.points;
  if(!pts?.length) {
    // Fallback for videos without GPS: offset from ts_start
    if(vid.ts_start != null) return vid.ts_start + videoSec;
    return null;
  }
  let loIdx = 0;
  for(let i = 0; i < pts.length; i++) {
    if(pts[i].video_s != null && pts[i].video_s <= videoSec) loIdx = i;
    else break;
  }
  const lo = pts[loIdx];
  const hiIdx = Math.min(loIdx + 1, pts.length - 1);
  const hi = pts[hiIdx];
  if(lo.ts == null) return null;
  if(hiIdx === loIdx || hi.video_s == null || hi.ts == null || hi.video_s === lo.video_s) return lo.ts;
  const frac = (videoSec - lo.video_s) / (hi.video_s - lo.video_s);
  return lo.ts + frac * (hi.ts - lo.ts);
}

/**
 * Find the closest lat/lon on any track for a given absolute timestamp.
 * Returns [lat, lon] or null.
 */
function findLatLonForTs(ts) {
  if(!state.mapData) return null;
  const vids = state.mapData.videos || [];
  for(const v of vids) {
    const pts = validLatLonPoints(getTrackTelemetryPoints(v));
    if(!pts.length) continue;
    let bestIdx = 0;
    for(let i = 0; i < pts.length; i++) {
      if(pts[i].ts != null && pts[i].ts <= ts) bestIdx = i;
    }
    const p = pts[bestIdx];
    if(p && p.ts != null && Math.abs(p.ts - ts) < 60) {
      return [p.lat, p.lon];
    }
  }
  return null;
}

// ── Play / Pause ───────────────────────────────────────────────────────
function tlPlay() {
  const tl = state.tl;
  if(tl.playing) return;
  tl.playing = true;
  tl.lastFrameTime = performance.now();
  tl._lastUiRefreshAt = 0;       // force a UI refresh on the first played frame
  tl._lastStatsUiUpdateAt = 0;   // and a 1 Hz stats refresh right away
  tl._pendingProcessedMetricRefresh = true;
  _timelineMetricLoadToken++;
  prewarmLikelyVideosAtTs(tl.currentTs);
  syncPhonePlaybackToTimeline({ forceSeek: true, forceReload: false });

  // Play all active video elements
  let playDelayMs = 0;
  for(const slot of tl.athleteSlots) {
    if(slot.currentFileId && slot.videoEl) {
      if (USE_IPAD_VIDEO_WORKAROUNDS) {
        const delay = playDelayMs;
        setTimeout(() => {
          if (state.tl.playing) tryPlaySlotVideo(slot, slot.currentFileId);
        }, delay);
        playDelayMs += 55;
      } else {
        tryPlaySlotVideo(slot, slot.currentFileId);
      }
    }
  }
  const phoneItem = getSelectedPhonePlaybackItem();
  if (state.phonePlayback?.paneVisible && state.phonePlayback.videoEl && phoneItem && videoContainsAbsTs(phoneItem, tl.currentTs)) {
    try { state.phonePlayback.videoEl.playbackRate = state.tl.playbackRate; } catch {}
    const playPromise = state.phonePlayback.videoEl.play();
    if (playPromise?.catch) playPromise.catch(() => {});
  }

  updatePlayPauseIcon();
  tl.animFrameId = requestAnimationFrame(tlAnimLoop);
}

function tlPause() {
  const tl = state.tl;
  tl.playing = false;
  if(tl.animFrameId) { cancelAnimationFrame(tl.animFrameId); tl.animFrameId = null; }
  tl._lastPrewarmAt = 0;

  for(const slot of tl.athleteSlots) {
    slot._playRetryToken = (slot._playRetryToken || 0) + 1;
    if (slot._playRetryTimer) {
      clearTimeout(slot._playRetryTimer);
      slot._playRetryTimer = null;
    }
    pauseSlotVideos(slot, false);
  }
  pausePhonePlayback(false, false);

  clearAllOverlays();
  updatePlayPauseIcon();
  // Playback throttles the timeline UI to ~20fps, so on pause do one final refresh
  // to land the playhead, labels and stats exactly where playback stopped.
  tl._lastUiRefreshAt = 0;
  if (keepTimelinePlayheadInView()) drawSogCanvas();
  else updatePlayhead();
  updateTimeLabel();
  updateTimelineStats(true);
  if (tl._pendingProcessedMetricRefresh) refreshTimelineProcessedMetricStats();
}

function tlStop() {
  tlPause();
  state.tl.currentTs = state.tl.globalStart;
}

function tlToggle() {
  if(state.tl.playing) tlPause();
  else tlPlay();
}

function tlAnimLoop(now) {
  const tl = state.tl;
  if(!tl.playing) return;

  const dt = (now - (tl.lastFrameTime || now)) / 1000;
  tl.lastFrameTime = now;

  // Advance timeline by dt * playbackRate
  tl.currentTs += dt * tl.playbackRate;
  if (!tl._lastPrewarmAt || now - tl._lastPrewarmAt > 1200) {
    prewarmLikelyVideosAtTs(tl.currentTs + 6);
    tl._lastPrewarmAt = now;
  }

  if(tl.currentTs >= tl.globalEnd) {
    tl.currentTs = tl.globalEnd;
    tlPause();
    return;
  }
  const activeIdsLoop = new Set();
  const activeVidsLoop = [];

  // Per-frame: only the work needed to keep each slot's video switched to the
  // right clip and playing. Cosmetic style writes and the map-marker move are
  // deferred to the throttled UI block below so the decoder/compositor isn't
  // starved by main-thread work every frame.
  for(const slot of tl.athleteSlots) {
    if (isVideoSlotHidden(slot)) {
      if (slot.paneEl) slot.paneEl.style.display = 'none';
      if (slot.currentFileId || slot._desiredFileId) stopSlotPlayback(slot, false);
      continue;
    }
    const vid = findSlotVideoAtTime(slot, tl.currentTs);
    if(vid) {
      activeIdsLoop.add(vid.id);
      activeVidsLoop.push(vid);
      const switchPending = slot._desiredFileId === vid.id;
      if(slot.currentFileId !== vid.id) {
        const videoSec = absTs2VideoSec(vid, tl.currentTs);
        if (slot.currentFileId) showSlotActiveVideo(slot);
        else showSlotLoadingState(slot);
        if (!switchPending) switchSlotVideoSource(slot, vid.id, videoSec, true);
      } else if (slot.videoEl.paused) {
        showSlotActiveVideo(slot);
        tryPlaySlotVideo(slot, vid.id);
      }
      monitorSlotVideoHealth(slot, now);
    } else {
      if (slot.videos.length > 0) showSlotEmptyState(slot);
      else if (slot.paneEl) slot.paneEl.style.display = 'none';
      if (slot.currentFileId || slot._desiredFileId) stopSlotPlayback(slot, !USE_IPAD_VIDEO_WORKAROUNDS);
      else slot._stuckSince = 0;
    }
  }

  // Keep the phone (second) video synced every frame — it must track the master
  // video closely; the function already self-throttles its own drift seeks.
  syncPhonePlaybackToTimeline({ forceSeek: false, forceReload: false });

  // Throttle the heavier, non-critical timeline UI to ~20fps. The timeline clock and
  // video playback above still update every frame, so motion stays smooth; these
  // panels (map markers, SOG canvas, labels, stats, layout) just refresh less often.
  const lastUi = Number(tl._lastUiRefreshAt) || 0;
  if (now - lastUi >= TIMELINE_UI_REFRESH_MS) {
    tl._lastUiRefreshAt = now;
    for (const slot of tl.athleteSlots) {
      if (!isVideoSlotHidden(slot) && slot.currentFileId) showSlotActiveVideo(slot);
    }
    // If no panes are visible, show the empty state
    const anyVisLoop = tl.athleteSlots.some(s => s.paneEl && s.paneEl.style.display !== 'none');
    showNoVideo(!anyVisLoop);
    // Move map markers for the videos playing at the current time
    for (const vid of activeVidsLoop) {
      if (absTs2VideoSec(vid, tl.currentTs) != null) movePositionMarkerByTs(vid.id, tl.currentTs);
    }
    syncInlineHeatmapPanelLayout();
    // Hide arrows for videos not currently playing
    syncActivePositionMarkers(activeIdsLoop);
    keepPlaybackMarkersInView(activeIdsLoop);
    if (keepTimelinePlayheadInView()) drawSogCanvas();
    else updatePlayhead();
    updateTimeLabel();
    maybeRefreshWindEstimate(false);
    updateTimelineStats();
    updateCurrentSegmentActions();
  }

  // Real-time pose overlay — run detection on visible slots
  if (isRealtimePoseEnabled()) {
    for (const slot of tl.athleteSlots) {
      if (slot.currentFileId && slot.videoEl && !slot.videoEl.paused) {
        runRealtimeDetectionOnSlot(slot);  // fire-and-forget, guarded by busy flag
      }
    }
  }

  tl.animFrameId = requestAnimationFrame(tlAnimLoop);
}

function updatePlayPauseIcon() {
  const icon = el('tl-play-icon');
  if(!icon) return;
  const useEl = icon.querySelector('use');
  if(useEl) useEl.setAttribute('href', state.tl.playing ? '#ico-pause' : '#ico-play');
}

// ── SOG Canvas click / scrub ───────────────────────────────────────────
function initSogScrub() {
  const wrap = el('sog-canvas-wrap');
  if(!wrap) return;
  let scrubbing = false;

  function tsFromEvent(e) {
    const rect = wrap.getBoundingClientRect();
    const x = (e.clientX || (e.touches?.[0]?.clientX ?? 0)) - rect.left;
    const frac = Math.max(0, Math.min(1, x / rect.width));
    const tl = state.tl;
    if(tl.globalStart == null || tl.globalEnd == null) return null;
    const [vStart, vEnd] = tlViewRange();
    return vStart + frac * (vEnd - vStart);
  }

  function scrubFromEvent(e) {
    const ts = tsFromEvent(e);
    if(ts != null) tlSeekTo(ts);
  }

  wrap.addEventListener('pointerdown', e => {
    // Shift+click places segment markers on the scrub bar; plain clicks always seek.
    if(e.shiftKey) {
      const ts = tsFromEvent(e);
      if(ts != null) {
        if(!state.segmentSelect.active) startSegmentCreation();
        // Find closest lat/lon from any track at this ts for map marker
        const ll = findLatLonForTs(ts);
        handleSegmentClick(ts, ll);
      }
      return;
    }
    scrubbing = true;
    wrap.setPointerCapture(e.pointerId);
    const wasPlaying = state.tl.playing;
    if(wasPlaying) tlPause();
    wrap._wasPlaying = wasPlaying;
    scrubFromEvent(e);
  });
  wrap.addEventListener('pointermove', e => {
    if(!scrubbing) return;
    scrubFromEvent(e);
  });
  wrap.addEventListener('pointerup', e => {
    if(!scrubbing) return;
    scrubbing = false;
    if(wrap._wasPlaying) tlPlay();
  });

  // Zoom with mouse wheel
  wrap.addEventListener('wheel', e => {
    e.preventDefault();
    const tl = state.tl;
    if(tl.globalStart == null || tl.globalEnd == null) return;
    const globalDur = tl.globalEnd - tl.globalStart;
    if(globalDur <= 0) return;

    const [vStart, vEnd] = tlViewRange();
    const vDur = vEnd - vStart;
    const rect = wrap.getBoundingClientRect();
    const xFrac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const pivot = vStart + xFrac * vDur; // time at cursor

    const zoomFactor = e.deltaY > 0 ? 1.25 : 0.8; // scroll down = zoom out, up = zoom in
    let newDur = vDur * zoomFactor;
    // Clamp: min 10 seconds, max = full range
    newDur = Math.max(10, Math.min(globalDur, newDur));

    if(newDur >= globalDur - 0.5) {
      // Reset to full view
      tl.viewStart = null;
      tl.viewEnd = null;
    } else {
      let newStart = pivot - xFrac * newDur;
      let newEnd = newStart + newDur;
      // Clamp to global bounds
      if(newStart < tl.globalStart) { newStart = tl.globalStart; newEnd = newStart + newDur; }
      if(newEnd > tl.globalEnd) { newEnd = tl.globalEnd; newStart = newEnd - newDur; }
      newStart = Math.max(tl.globalStart, newStart);
      tl.viewStart = newStart;
      tl.viewEnd = newEnd;
    }
    drawSogCanvas();
  }, {passive: false});
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────
function initKeyboard() {
  document.addEventListener('keydown', e => {
    if (el('timeline-media-modal')?.classList.contains('open')) {
      if (e.code === 'Escape') {
        closeTimelineMediaOverlay();
        e.preventDefault();
      }
      return;
    }
    if (isInlineHeatmapPlotOverlayOpen()) {
      if (e.code === 'Escape') {
        closeInlineHeatmapPlotOverlay();
        e.preventDefault();
      }
      return;
    }
    if(state.reportOverlayOpen) return;
    if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT'||e.target.tagName==='TEXTAREA') return;
    const tl = state.tl;
    if(e.code === 'Space') {
      e.preventDefault();
      tlToggle();
    }
    if(e.code === 'Escape') {
      if(state.segmentSelect.active) { cancelSegmentCreation(); e.preventDefault(); }
    }
    if(e.code === 'KeyI') {
      e.preventDefault();
      setSegmentStartAtCurrentPlayhead();
    }
    if(e.code === 'KeyO') {
      e.preventDefault();
      setSegmentEndAtCurrentPlayhead();
    }
    if(e.code === 'ArrowLeft') {
      e.preventDefault();
      const step = e.shiftKey ? 1 : 10;
      if(tl.currentTs != null) tlSeekTo(tl.currentTs - step);
    }
    if(e.code === 'ArrowRight') {
      e.preventDefault();
      const step = e.shiftKey ? 1 : 10;
      if(tl.currentTs != null) tlSeekTo(tl.currentTs + step);
    }
  });
}

// ── Files tab ──────────────────────────────────────────────────────────
function renderFilesList() {
  const list = el('files-list');
  list.innerHTML = '';
  if(!state.mapData) return;
  const all = [...(state.mapData.videos||[]).map(v=>({...v,kind:'video'})),
               ...(state.mapData.csvs||[]).map(c=>({...c,kind:'csv'}))];
  const dayMap = new Map();
  for(const f of all) {
    const day = computeDayLabel(f.ts_start) || 'Unknown';
    if(!dayMap.has(day)) dayMap.set(day, []);
    dayMap.get(day).push(f);
  }
  const sortedDays = [...dayMap.keys()].sort();
  for(const day of sortedDays) {
    if(sortedDays.length > 1) {
      const hdr = document.createElement('div');
      hdr.className = 'day-group-hdr';
      hdr.textContent = day === 'Unknown' ? 'Unknown date' : day;
      list.appendChild(hdr);
    }
    for(const f of dayMap.get(day)) {
      const meta = state.fileMeta[f.id] || {};
      const playbackOnly = f.kind === 'video' && isPlaybackOnlyVideo(f);
      // For videos, derive athlete from matched CSV's metadata
      let athId = meta.athlete_id;
      if(f.kind === 'video' && !playbackOnly && !athId && f.best_match_csv_id) {
        const csvMeta = state.fileMeta[f.best_match_csv_id] || {};
        athId = csvMeta.athlete_id;
      }
      const ath = state.athletes.find(a=>a.id===athId);
      const athIdx = ath ? state.athletes.indexOf(ath) : -1;
      const athColor = ath ? (ath.color || PALETTE[athIdx % PALETTE.length]) : null;
      const status = state.cvStatuses[f.id]?.status||'';
      const div = document.createElement('div');
      div.className='file-item';
      const sz = f.size_bytes ? (f.size_bytes/1048576).toFixed(1)+'MB' : '';
      const colorDot = athColor ? `<span style="width:8px;height:8px;border-radius:50%;background:${athColor};flex-shrink:0;display:inline-block;"></span>` : '';
      // Both CSVs and videos are directly assignable
      const isCSV = f.kind === 'csv';
      const isManual = meta.manual_athlete;
      const athTagText = playbackOnly ? 'Playback only' : (ath ? ath.name : '—');
      const athTagTitle = playbackOnly ? playbackOnlyVideoTitle(f) : (ath ? ath.name : 'Assign athlete');
      const athTagExtra = playbackOnly
        ? ''
        : ((!isCSV && ath && !isManual) ? ' <span style="font-size:9px;opacity:.5">(auto)</span>' : '');
      const athTagClass = (f.kind === 'video' || !playbackOnly) ? 'f-athlete-tag clickable' : 'f-athlete-tag';
      div.innerHTML = `
        <span class="f-kind ${f.kind}">${f.kind==='video'?'VID':'CSV'}</span>
        ${colorDot}
        <span class="f-name" title="${f.filename||f.name||'unknown'}">${f.filename||f.name||'unknown'}</span>
        <span class="f-size">${sz}</span>
        <span class="${athTagClass}" data-fid="${f.id}" data-kind="${f.kind}" data-csv-id="${f.best_match_csv_id||''}" title="${athTagTitle}">${athTagText}${athTagExtra}</span>
        <button class="f-del-btn" data-fid="${f.id}" title="Delete file"><svg class="icon-svg" style="width:13px;height:13px"><use href="#ico-trash"/></svg></button>
      `;
      div.querySelector('.f-name').onclick = ()=>{
        if (f.kind === 'video' && playbackOnly) {
          openTimelineMediaOverlay(f).catch(err => {
            console.warn('[timeline-media] open from file list failed:', err);
            alert(`Could not open ${f.filename || f.name || 'the video'}.`);
          });
        }
      };
      div.querySelector('.f-del-btn').onclick = ()=>deleteFile(f.id, f.filename||f.name||f.id);
      const athTag = div.querySelector('.f-athlete-tag');
      if (athTagClass.includes('clickable')) athTag.onclick = (e)=>openAssignPopup(f.id, e.target, f.kind);
      list.appendChild(div);
    }
  }
}

async function deleteFile(fileId, name) {
  if(!state.projectId) return;
  confirmDialog('Delete File', `Delete "${name}"? This cannot be undone.`, async ()=>{
    try {
      await Pipeline.deleteSegment(state.projectId, fileId);
    } catch(e) {
      let msg = e.message || String(e);
      alert(`Could not delete "${name}":\n${msg}`);
      return;
    }
    delete state.trackVisibility[fileId];
    await loadMapData();
    renderFilesList();
    buildTimeline();
    updateSetupUiState();
  });
}

// ── Athlete assign popup ───────────────────────────────────────────────
function openAssignPopup(fileId, anchor, kind) {
  const video = (state.mapData?.videos || []).find(v => String(v?.id) === String(fileId));
  const playbackOnly = kind === 'video' && isPlaybackOnlyVideo(video);
  const popup = el('assign-popup');
  popup.innerHTML = '';

  if (kind === 'video') {
    const modeOpt = document.createElement('div');
    modeOpt.className='assign-opt';
    modeOpt.textContent = playbackOnly ? 'Analyze (connect athlete, no CSV)' : 'Mark as External Video';
    modeOpt.onclick = async()=>{
      if (playbackOnly) {
        await setVideoExternalMode(fileId, { external: false, athleteId: null });
      } else {
        await setVideoExternalMode(fileId, { external: true });
      }
      closePopup();
    };
    popup.appendChild(modeOpt);
  }

  const noneOpt = document.createElement('div');
  noneOpt.className='assign-opt'; noneOpt.textContent='— Unassign —';
  noneOpt.onclick = async()=>{
    if (kind === 'video') {
      await setVideoExternalMode(fileId, { external: false, athleteId: null });
    } else {
      await setCsvAthleteAndLinkedVideos(fileId, null);
    }
    closePopup(); renderFilesList(); buildTimeline(); void refreshManeuvers('assignment-updated');
  };
  popup.appendChild(noneOpt);
  for(const a of state.athletes){
    const opt = document.createElement('div');
    opt.className='assign-opt'; opt.textContent=`${a.name} (${a.weight??'?'}kg${a.height ? `, ${a.height}cm` : ''})`;
    opt.onclick = async()=>{
      if (kind === 'video') {
        await setVideoExternalMode(fileId, { external: false, athleteId: a.id });
      } else {
        await setCsvAthleteAndLinkedVideos(fileId, a.id);
      }
      closePopup(); renderFilesList(); buildTimeline(); void refreshManeuvers('assignment-updated');
    };
    popup.appendChild(opt);
  }
  const rect=anchor.getBoundingClientRect();
  popup.style.display='block';
  const pw = popup.offsetWidth, ph = popup.offsetHeight;
  let top = rect.bottom + 4, left = rect.left;
  if(left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
  if(left < 4) left = 4;
  if(top + ph > window.innerHeight - 8) top = rect.top - ph - 4;
  if(top < 4) top = 4;
  popup.style.top = top + 'px'; popup.style.left = left + 'px';
  function closePopup(){ popup.style.display='none'; document.removeEventListener('click',outsideClick); }
  function outsideClick(e){ if(!popup.contains(e.target)&&e.target!==anchor) closePopup(); }
  setTimeout(()=>document.addEventListener('click',outsideClick),50);
}

// ── Athletes tab ───────────────────────────────────────────────────────
function renderAthletes() {
  const list = el('ath-list');
  list.innerHTML = '';
  for(const a of state.athletes){
    const idx = state.athletes.indexOf(a);
    const color = a.color || PALETTE[idx % PALETTE.length];
    const div = document.createElement('div');
    div.className='ath-item';
      div.innerHTML = `
        <span class="ath-color" style="background:${color}" title="Change color"></span>
        <span class="ath-name" title="Edit athlete">${a.name}</span>
        <span class="ath-weight" title="Edit athlete">${a.weight!=null?a.weight+'kg':''}${a.height!=null?` / ${a.height}cm`:''}</span>
        <button class="ath-del" data-aid="${a.id}" title="Delete athlete"><svg class="icon-svg" style="width:13px;height:13px"><use href="#ico-x"/></svg></button>
      `;
      div.querySelector('.ath-color').onclick = (e) => openColorPicker(a, e.target);
      div.querySelector('.ath-name').onclick = () => openAthleteEdit(a.id);
      div.querySelector('.ath-weight').onclick = () => openAthleteEdit(a.id);
      div.querySelector('.ath-del').onclick = ()=>deleteAthlete(a.id);
      list.appendChild(div);
    }
}

const COLOR_OPTIONS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#00bcd4'];

function openColorPicker(athlete, anchor) {
  // Close any existing color picker
  document.querySelectorAll('.color-picker-popup').forEach(p => p.remove());
  const popup = document.createElement('div');
  popup.className = 'color-picker-popup';
  for(const c of COLOR_OPTIONS) {
    const swatch = document.createElement('div');
    swatch.className = 'color-swatch';
    if((athlete.color || PALETTE[state.athletes.indexOf(athlete) % PALETTE.length]) === c) swatch.classList.add('active');
    swatch.style.background = c;
    swatch.onclick = async () => {
      athlete.color = c;
      await saveAthletes();
      renderAthletes();
      assignVideoColors();
      renderFilesList();
      renderTrackPanel();
      renderMap();
      buildTimeline();
      popup.remove();
    };
    popup.appendChild(swatch);
  }
  document.body.appendChild(popup);
  const rect = anchor.getBoundingClientRect();
  let top = rect.bottom + 4, left = rect.left;
  if(left + 160 > window.innerWidth) left = window.innerWidth - 168;
  if(top + 60 > window.innerHeight) top = rect.top - 60;
  popup.style.top = top + 'px';
  popup.style.left = left + 'px';
  function outsideClick(e) { if(!popup.contains(e.target) && e.target !== anchor) { popup.remove(); document.removeEventListener('click', outsideClick); } }
  setTimeout(() => document.addEventListener('click', outsideClick), 50);
}

function deleteAthlete(id) {
  const ath=state.athletes.find(a=>a.id===id);
  if(!ath) return;
  confirmDialog('Delete Athlete', `Remove "${ath.name}"?`, async ()=>{
    state.athletes = state.athletes.filter(a=>a.id!==id);
    await saveAthletes();
    renderAthletes();
    renderFilesList();
    assignVideoColors();
    renderTrackPanel();
    renderMap();
    buildTimeline();
    updateSetupUiState();
  });
}

function openAthleteEdit(id) {
  state.editingAthleteId = id || null;
  const a = id ? state.athletes.find(a=>a.id===id) : null;
  el('ath-name-in').value = a?.name||'';
  el('ath-weight-in').value = a?.weight||'';
  el('ath-height-in').value = a?.height||'';
  el('ath-edit-row').classList.add('open');
  el('ath-name-in').focus();
}

async function saveAthlete() {
  const name = el('ath-name-in').value.trim();
  const weight = parseOptionalPositiveNumber(el('ath-weight-in').value);
  const height = parseOptionalPositiveNumber(el('ath-height-in').value);
  if(!name) return;
  if(state.editingAthleteId) {
    const a = state.athletes.find(a=>a.id===state.editingAthleteId);
    if(a){ a.name=name; a.weight=weight; a.height=height; }
  } else {
    state.athletes.push({id: DB.uuid(), name, weight, height});
  }
  await saveAthletes();
  renderAthletes();
  renderFilesList();
  assignVideoColors();
  renderTrackPanel();
  renderMap();
  buildTimeline();
  void refreshManeuvers('athlete-saved');
  el('ath-edit-row').classList.remove('open');
  state.editingAthleteId = null;
  updateSetupUiState();
}

function getSetupState() {
  const videos = state.mapData?.videos || [];
  const csvs = state.mapData?.csvs || [];
  const fileCount = videos.length + csvs.length;
  const assignedCsvs = csvs.filter(csv => normalizeAthleteId(state.fileMeta?.[csv.id]?.athlete_id)).length;
  const hasProject = !!state.projectId;
  const hasFiles = fileCount > 0;
  const hasAthletes = state.athletes.length > 0;
  const assignmentsReady = csvs.length === 0 || assignedCsvs === csvs.length;
  const missing = [];

  if (!hasProject) missing.push('Select or create a project');
  if (state.uploadInProgress) missing.push('Wait for file upload and matching to finish');
  if (!hasFiles) missing.push('Upload at least one video or CSV file');
  if (!hasAthletes) missing.push('Add at least one athlete');
  if (!assignmentsReady) missing.push('Assign every GPS / CSV file to an athlete');

  return {
    hasProject,
    hasFiles,
    hasAthletes,
    assignmentsReady,
    ready: missing.length === 0,
    missing,
    steps: {
      1: hasProject,
      2: hasAthletes,
      3: hasFiles && !state.uploadInProgress,
      4: hasProject && hasFiles && hasAthletes && assignmentsReady && !state.uploadInProgress,
    },
  };
}

function renderWizardStatus() {
  const setup = getSetupState();
  const currentStep = Math.max(1, Math.min(4, state.wizardStep || 1));
  document.querySelectorAll('.wiz-step').forEach(stepEl => {
    const stepNum = parseInt(stepEl.dataset.step, 10);
    stepEl.classList.remove('active', 'done', 'warn');
    if (stepNum === currentStep) stepEl.classList.add('active');
    if (setup.steps[stepNum]) stepEl.classList.add('done');
    if (stepNum === 4 && currentStep === 4 && !setup.ready) stepEl.classList.add('warn');
  });
  document.querySelectorAll('.wiz-connector').forEach(connector => {
    const after = parseInt(connector.dataset.after, 10);
    connector.classList.toggle('done', !!setup.steps[after]);
  });

  const uploadNav = document.querySelector('.nav-btn[data-view="view-upload"]');
  if (uploadNav) uploadNav.classList.toggle('setup-pending', state.uploadInProgress || !setup.hasFiles);
}

function syncWizardFooterButtons(step = state.wizardStep) {
  const total = 4;
  const backBtn = el('btn-wiz-back');
  const nextBtn = el('btn-wiz-next');
  if (!backBtn || !nextBtn) return;
  const currentStep = Math.max(1, Math.min(total, step || 1));
  const setup = getSetupState();
  backBtn.style.display = currentStep === 1 ? 'none' : '';
  if (currentStep === total) {
    nextBtn.style.display = 'none';
    return;
  }
  nextBtn.style.display = '';
  nextBtn.disabled =
    (currentStep === 1 && !setup.hasProject) ||
    (currentStep === 2 && !setup.hasAthletes) ||
    (currentStep === 3 && (!setup.hasFiles || state.uploadInProgress));
  nextBtn.innerHTML = 'Next <svg class="icon-svg" style="width:14px;height:14px"><use href="#ico-skip-fwd"/></svg>';
}

function updateSetupUiState() {
  syncWizardFooterButtons();
  renderWizardStatus();
  if (state.wizardStep === 4) renderReadySummary();
}

// ── Wizard step navigation ─────────────────────────────────────────────
function wizGoTo(step) {
  const total = 4;
  if (!state.projectId && step > 1) step = 1;
  step = Math.max(1, Math.min(total, step));
  state.wizardStep = step;

  // Show active page
  document.querySelectorAll('.wiz-page').forEach(p => p.classList.remove('active'));
  const page = el('wiz-page-' + step);
  if(page) page.classList.add('active');

  syncWizardFooterButtons(step);

  // Refresh content for each step
  if(step === 1) renderProfileStep();
  if(step === 2) renderAthletes();
  if(step === 3) renderFilesList();
  if(step === 4) renderReadySummary();
  renderWizardStatus();
}

function wizNext() {
  if(state.wizardStep < 4) wizGoTo(state.wizardStep + 1);
}
function wizBack() {
  if(state.wizardStep > 1) wizGoTo(state.wizardStep - 1);
}

// wizard step clicks
function initWizardStepClicks() {
  document.querySelectorAll('.wiz-step').forEach(s => {
    s.addEventListener('click', () => {
      const n = parseInt(s.dataset.step);
      wizGoTo(n);
    });
  });
}

// ── Step 4: Assign athletes to files ───────────────────────────────────
function renderAssignList() {
  const list = el('assign-list');
  if(!list) return;
  list.innerHTML = '';
  if(!state.mapData) { list.innerHTML = '<div class="empty-state" style="padding:20px;"><div>No files uploaded yet</div></div>'; return; }

  const csvs = (state.mapData.csvs || []).map(c => ({...c, kind: 'csv'}));
  const videos = (state.mapData.videos || []).map(v => ({...v, kind: 'video'}));

  if(csvs.length === 0 && videos.length === 0) {
    list.innerHTML = '<div class="empty-state" style="padding:20px;"><div>No files uploaded yet. Go back to Step 3.</div></div>';
    return;
  }

  // CSV files — user assigns athletes to these
  if(csvs.length) {
    const hdr = document.createElement('div');
    hdr.style.cssText = 'font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);padding:6px 0 4px;';
    hdr.textContent = 'GPS / CSV Files';
    list.appendChild(hdr);

    for(const f of csvs) {
      const meta = state.fileMeta[f.id] || {};
      const card = document.createElement('div');
      card.className = 'assign-card';
      const curAthId = meta.athlete_id || '';
      let options = '<option value="">— Select athlete —</option>';
      for(const a of state.athletes) {
        const sel = a.id === curAthId ? ' selected' : '';
        options += `<option value="${a.id}"${sel}>${a.name}${a.weight ? ' (' + a.weight + 'kg)' : ''}</option>`;
      }
      card.innerHTML = `
        <span class="f-kind csv">CSV</span>
        <div class="assign-card-info">
          <div class="assign-card-name">${f.filename || f.name || 'unknown'}</div>
        </div>
        <select class="assign-select" data-fid="${f.id}" data-kind="csv">${options}</select>
      `;
      const sel = card.querySelector('.assign-select');
      sel.onchange = async () => {
        const athId = sel.value || null;
        await setCsvAthleteAndLinkedVideos(f.id, athId);
        renderAssignList();
        buildTimeline();
        void refreshManeuvers('assignment-updated');
      };
      list.appendChild(card);
    }
  }

  // Videos — every video is directly assignable to an athlete. Assigning an
  // athlete claims the video, which unlocks its GoPro GPS track and pose/video
  // analysis (no CSV required). The "External" option keeps a clip as
  // playback-only. Videos auto-linked through a matched CSV still resolve their
  // athlete automatically unless overridden here.
  if(videos.length) {
    const hdr = document.createElement('div');
    hdr.style.cssText = 'font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);padding:12px 0 4px;';
    hdr.textContent = 'Video Files';
    list.appendChild(hdr);

    for(const f of videos) {
      const meta = state.fileMeta[f.id] || {};
      const isExternal = isPlaybackOnlyVideo(f);
      const curAthId = meta.athlete_id || '';
      const isManual = meta.manual_athlete;

      // Athlete auto-inherited from a matched CSV (only relevant when not manually set / external).
      let autoAthId = null;
      if(!curAthId && !isExternal && f.best_match_csv_id) {
        const csvMeta = state.fileMeta[f.best_match_csv_id] || {};
        autoAthId = csvMeta.athlete_id || null;
      }
      const autoAth = autoAthId ? state.athletes.find(a => a.id === autoAthId) : null;

      const card = document.createElement('div');
      card.className = 'assign-card';

      let options = '<option value="">— Select athlete to analyze —</option>';
      for(const a of state.athletes) {
        const sel = (!isExternal && a.id === curAthId) ? ' selected' : '';
        options += `<option value="${a.id}"${sel}>${a.name}${a.weight ? ' (' + a.weight + 'kg)' : ''}</option>`;
      }
      const extSel = isExternal ? ' selected' : '';
      options += `<option value="__external__"${extSel}>External — not analyzed</option>`;

      let subText = '';
      if(isExternal) {
        subText = '<div class="assign-card-auto">Playback only. Pick an athlete to enable GPS + video analysis (no CSV needed).</div>';
      } else if(autoAth && !isManual) {
        subText = `<div class="assign-card-auto">Auto-linked via matched CSV: ${autoAth.name}</div>`;
      }

      card.innerHTML = `
        <span class="f-kind video">VID</span>
        <div class="assign-card-info">
          <div class="assign-card-name">${f.filename || f.name || 'unknown'}</div>
          ${subText}
        </div>
        <select class="assign-select" data-fid="${f.id}" data-kind="video">${options}</select>
      `;
      const sel = card.querySelector('.assign-select');
      sel.onchange = async () => {
        if (sel.value === '__external__') {
          await setVideoExternalMode(f.id, { external: true });
          return;
        }
        await setVideoExternalMode(f.id, { external: false, athleteId: sel.value || null });
      };
      list.appendChild(card);
    }
  }
}

// ── Step 4: Ready summary ──────────────────────────────────────────────
function renderReadySummary() {
  const wrap = el('ready-summary');
  if(!wrap) return;
  const setup = getSetupState();
  const nVids = (state.mapData?.videos || []).length;
  const nCsvs = (state.mapData?.csvs || []).length;
  const nAth = state.athletes.length;
  const readyWrap = wrap.closest('.ready-wrap');
  const iconEl = el('ready-icon');
  const titleEl = el('ready-title');
  const subtitleEl = el('ready-subtitle');
  const goBtn = el('btn-go-analysis');
  if (readyWrap) readyWrap.classList.toggle('incomplete', !setup.ready);
  if (iconEl) iconEl.innerHTML = setup.ready ? '&#10003;' : '&#10007;';
  if (titleEl) titleEl.textContent = setup.ready ? "You're all set!" : 'Setup not complete';
  if (subtitleEl) {
    subtitleEl.textContent = setup.ready
      ? 'Your files are uploaded and athletes are assigned. Head to the Analysis view to explore GPS tracks, synced video playback, and skeleton data.'
      : 'Finish the missing setup items below before moving on to analysis or reports.';
  }
  if (goBtn) goBtn.disabled = !setup.ready;
  const missingHtml = setup.ready ? '' : `
    <div class="ready-missing">
      <div class="ready-missing-title">Still missing</div>
      <ul class="ready-missing-list">${setup.missing.map(item => `<li>${item}</li>`).join('')}</ul>
    </div>
  `;
  wrap.innerHTML = `
    <div class="ready-stat"><div class="ready-stat-num">${nVids}</div><div class="ready-stat-label">Videos</div></div>
    <div class="ready-stat"><div class="ready-stat-num">${nCsvs}</div><div class="ready-stat-label">CSV files</div></div>
    <div class="ready-stat"><div class="ready-stat-num">${nAth}</div><div class="ready-stat-label">Athletes</div></div>
    ${missingHtml}
  `;
}

async function fetchApiCsvFilesForVideoRange(range, existingFilenames, onProgress = null) {
  const startSec = Number(range?.startSec);
  const endSec = Number(range?.endSec);
  if (!Number.isFinite(startSec) || !Number.isFinite(endSec) || !(endSec > startSec)) return [];

  const fromMs = Math.round((startSec - 30) * 1000);
  const toMs = Math.round((endSec + 30) * 1000);
  const sessionSearchFrom = Math.round((startSec - 24 * 3600) * 1000);
  const sessionSearchTo = toMs;
  const fetched = [];

  async function fetchSessionTelemetryCsvs(sessionsToFetch, progressBase = 0.38) {
    const out = [];
    for (const session of sessionsToFetch) {
      const sessionId = String(session?.id || '');
      if (!sessionId) continue;
      onProgress?.(`Fetching telemetry for ${session.display_name || sessionId}...`, progressBase);

      const rows = [];
      let offset = 0;
      const limit = 5000;
      for (let page = 0; page < 20; page++) {
        const telemetry = await fetchApiCsvJson(`/sessions/${encodeURIComponent(sessionId)}/telemetry`, {
          from: fromMs,
          to: toMs,
          raw: 'true',
          limit,
          offset,
          fields: 'lat,lon,sog,cog,roll,pitch,yaw,rudder_angle,boom_angle,torso_angle,gnss_ms,gnss_iso,custom_name',
        });
        rows.push(...(telemetry.data || []));
        if (!telemetry.meta?.has_more) break;
        offset += limit;
      }

      const byUnit = new Map();
      for (const row of rows) {
        const unitId = String(apiRowValue(row, ['unit_id', 'unit']) || '').trim();
        const tsMs = normalizeApiEpochMs(apiRowValue(row, ['timestamp', 'gnss_ms', 'time']));
        if (!unitId || tsMs == null) continue;
        if (!byUnit.has(unitId)) byUnit.set(unitId, []);
        byUnit.get(unitId).push(row);
      }

      for (const [unitId, unitRows] of byUnit) {
        const gpsRows = unitRows.filter(row => (
          apiRowNumber(row, ['lat', 'latitude']) != null &&
          apiRowNumber(row, ['lon', 'lng', 'long', 'longitude']) != null
        ));
        if (gpsRows.length < 2) continue;
        const csvFile = buildApiTelemetryCsv(unitId, unitRows, session, fromMs, toMs);
        if (existingFilenames.has(csvFile.filename)) continue;
        existingFilenames.add(csvFile.filename);
        out.push(csvFile);
      }
    }
    return out;
  }

  const directContinuousSessions = buildContinuousApiSessionsForRange(fromMs, toMs);
  if (directContinuousSessions.length) {
    onProgress?.(`Checking continuous telemetry source${directContinuousSessions.length === 1 ? '' : 's'} first...`, 0.16);
    try {
      const continuousFiles = await fetchSessionTelemetryCsvs(directContinuousSessions, 0.24);
      if (continuousFiles.length) {
        console.log('[API CSV] using direct continuous telemetry files', continuousFiles.map(item => item.filename));
        return continuousFiles;
      }
      console.log('[API CSV] direct continuous telemetry probe returned no usable GPS rows');
    } catch (err) {
      console.log('[API CSV] direct continuous telemetry probe failed; falling back to session search', err?.message || err);
    }
  }

  onProgress?.('Querying telemetry sessions...', 0.12);
  const sessionsRes = await fetchApiCsvJson('/sessions', {
    limit: 200,
    from: sessionSearchFrom,
    to: sessionSearchTo,
  });
  const sessions = (sessionsRes.data || []).filter(session => {
    const s = normalizeApiEpochMs(session?.start_time);
    const e = normalizeApiEpochMs(session?.end_time) ?? Number.POSITIVE_INFINITY;
    if (!Number.isFinite(s)) return false;
    return (e == null || e >= fromMs) && s <= toMs;
  });
  const continuousSessions = sessions.filter(isContinuousApiSession);
  const sessionsToFetch = continuousSessions.length ? continuousSessions : sessions;
  if (continuousSessions.length) {
    onProgress?.(`Found ${continuousSessions.length} continuous telemetry source${continuousSessions.length === 1 ? '' : 's'}; skipping short recordings.`, 0.22);
  }

  fetched.push(...await fetchSessionTelemetryCsvs(sessionsToFetch, 0.38));

  return fetched;
}

async function keepBestApiCsvFilesForUploadedVideos(csvFiles, ranges) {
  if (!Array.isArray(csvFiles) || csvFiles.length <= 1) return csvFiles || [];
  const videoTracks = [];
  for (const range of ranges || []) {
    const track = await DB.getTrackByFileId(range.fileId);
    if (!track) continue;
    const points = await DB.getTrackPoints(track.id);
    if (Array.isArray(points) && points.length >= 12) {
      videoTracks.push({ file_id: String(range.fileId), points });
    }
  }
  const csvTracks = csvFiles
    .map((item, idx) => ({ file_id: `api-csv-${idx}`, points: item.points || [], item }))
    .filter(track => Array.isArray(track.points) && track.points.length >= 12);

  if (!videoTracks.length || !csvTracks.length) {
    console.log('[API CSV] could not pre-match generated CSVs; keeping all candidates');
    return csvFiles;
  }

  const matches = matchVideoTracksToCsv(videoTracks, csvTracks, 1);
  const selectedIds = new Set(matches.map(match => String(match.csv_file_id || '')));
  if (!selectedIds.size) {
    console.log('[API CSV] generated CSV candidates did not match uploaded video GPS; keeping all candidates');
    return csvFiles;
  }

  const selected = csvTracks
    .filter(track => selectedIds.has(track.file_id))
    .map(track => track.item);
  console.log('[API CSV] keeping best matched CSV files', selected.map(item => item.filename), matches);
  return selected.length ? selected : csvFiles;
}

async function importApiCsvsForUploadedVideos(insertedFiles, onProgress = null) {
  if (!state.projectId || !hasApiCsvConfig()) {
    console.log('[API CSV] skipped: project/API config missing', {
      hasProject: !!state.projectId,
      hasConfig: hasApiCsvConfig(),
      baseUrl: state.apiCsv?.baseUrl || '',
    });
    syncApiCsvInputs(null, '');
    return { imported: 0, skipped: true };
  }

  const uploadedVideoIds = new Set((insertedFiles || [])
    .filter(file => file?.kind === 'video')
    .map(file => String(file.id)));
  if (!uploadedVideoIds.size) {
    console.log('[API CSV] skipped: no uploaded video files in this batch');
    return { imported: 0, skipped: true };
  }

  const files = await DB.listFiles(state.projectId);
  const existingFilenames = new Set(files.map(file => String(file.filename || '')));
  const ranges = [];
  for (const file of files) {
    if (!uploadedVideoIds.has(String(file.id))) continue;
    const track = await DB.getTrackByFileId(file.id);
    let startSec = Number(track?.ts_start);
    let endSec = Number(track?.ts_end);
    if (!Number.isFinite(startSec)) startSec = Number(file.capture_start_ts ?? file.est_start_ts);
    if (!Number.isFinite(endSec)) {
      const dur = Number(file.duration_sec);
      if (Number.isFinite(startSec) && Number.isFinite(dur) && dur > 0) endSec = startSec + dur;
    }
    if (Number.isFinite(startSec) && Number.isFinite(endSec) && endSec > startSec) {
      ranges.push({ fileId: file.id, filename: file.filename, startSec, endSec });
    }
  }
  if (!ranges.length) {
    console.log('[API CSV] skipped: no usable uploaded video time ranges');
    syncApiCsvInputs('API CSV skipped: no uploaded video GPS time was available.', 'err');
    return { imported: 0, skipped: true };
  }
  console.log('[API CSV] fetching telemetry for uploaded video ranges', ranges);

  let csvFiles = [];
  try {
    for (let i = 0; i < ranges.length; i++) {
      const range = ranges[i];
      const prefix = ranges.length > 1 ? `${i + 1}/${ranges.length}: ` : '';
      const generated = await fetchApiCsvFilesForVideoRange(range, existingFilenames, (msg, frac) => {
        onProgress?.(`${prefix}${msg}`, Math.min(0.88, Math.max(0.05, frac)));
      });
      csvFiles.push(...generated);
    }
  } catch (err) {
    syncApiCsvInputs(`API CSV failed: ${err.message}`, 'err');
    throw err;
  }

  if (!csvFiles.length) {
    console.log('[API CSV] no generated CSV files for uploaded video ranges');
    syncApiCsvInputs('API CSV: no matching telemetry rows found for the uploaded video time.', 'err');
    return { imported: 0, skipped: false };
  }
  console.log('[API CSV] generated CSV files', csvFiles.map(item => item.filename));
  csvFiles = await keepBestApiCsvFilesForUploadedVideos(csvFiles, ranges);

  onProgress?.(`Importing ${csvFiles.length} API CSV track${csvFiles.length === 1 ? '' : 's'}...`, 0.9);
  const picked = FM.registerPickedFiles(csvFiles.map(item => item.file), null);
  const ingestResult = await Pipeline.ingestFiles(state.projectId, picked, (msg, pct) => {
    onProgress?.(`API CSV: ${msg}`, 0.9 + Math.min(0.09, Math.max(0, pct) * 0.09));
  });
  for (let i = 0; i < picked.length; i++) {
    const row = csvFiles[i];
    const fileId = picked[i]?.fileId;
    if (!fileId) continue;
    const track = await DB.getTrackByFileId(fileId);
    if (!track) continue;
    const points = await DB.getTrackPoints(track.id);
    await DB.upsertTrack({
      file_id: fileId,
      project_id: state.projectId,
      kind: 'csv',
      points,
      meta: {
        ...(track.meta || {}),
        source: 'trollsports_api',
        api_session_id: row.sessionId,
        api_unit_id: row.unitId,
        api_custom_name: row.customName,
        api_from_ms: row.fromMs,
        api_to_ms: row.toMs,
      },
    });
  }
  await Pipeline.runMatching(state.projectId);
  if (ingestResult.errors?.length) {
    console.warn('[API CSV] ingest errors:', ingestResult.errors);
  }
  syncApiCsvInputs(`Imported ${csvFiles.length} API CSV track${csvFiles.length === 1 ? '' : 's'}.`, 'ok');
  return { imported: csvFiles.length, errors: ingestResult.errors || [] };
}

// ── Drop zone for file upload ──────────────────────────────────────────
let _uploadChain = Promise.resolve();
let _activeUploadCount = 0;
let _filePickerOpen = false;

function initDropZone() {
  const zone = el('drop-zone');
  const fileIn = el('file-input');
  if(!zone || !fileIn) return;

  zone.addEventListener('click', async () => {
    if(!state.projectId) { alert('Please select or create a project first.'); return; }
    // Use File System Access API for persistent handles when available
    if (FM.HAS_FILE_SYSTEM_ACCESS) {
      if (_filePickerOpen) return;
      _filePickerOpen = true;
      try {
        const handles = await window.showOpenFilePicker({
          multiple: true,
          types: [
            { description: 'Videos & CSV', accept: { 'video/*': ['.mp4', '.mov'], 'text/csv': ['.csv'] } },
          ],
        });
        if (!handles.length) return;
        const files = await Promise.all(handles.map(h => h.getFile()));
        await handleFileUpload(files, handles);
      } catch (e) {
        if (e.name !== 'AbortError') console.warn('File picker error:', e);
      } finally {
        _filePickerOpen = false;
      }
    } else {
      fileIn.click();
    }
  });

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
  });
  zone.addEventListener('drop', async e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if(!state.projectId) { alert('Please select or create a project first.'); return; }
    const files = Array.from(e.dataTransfer.files).filter(f => /\.(mp4|mov|csv)$/i.test(f.name));
    if(!files.length) return;
    await handleFileUpload(files, null);
  });

  fileIn.addEventListener('change', async e => {
    const files = Array.from(e.target.files); e.target.value = '';
    if(!files.length) return;
    await handleFileUpload(files, null);
  });
}

async function handleFileUpload(files, handles = null) {
  _uploadChain = _uploadChain.then(() => processFileUpload(files, handles));
  return _uploadChain.catch(err => {
    console.error('[Upload] queued upload failed:', err);
  });
}

async function processFileUpload(files, handles = null) {
  const zone = el('drop-zone');
  _activeUploadCount++;
  state.uploadInProgress = _activeUploadCount > 0;
  updateSetupUiState();
  try {
    zone.innerHTML = '<div class="drop-zone-text" style="color:var(--accent)">Processing files...</div>';
    // Register files with the file manager (keeps File references / handles)
    const picked = FM.registerPickedFiles(files, handles);
    console.log('[Upload] registered', picked.length, 'files:', picked.map(p => `${p.fileId} (${p.file.name})`));
    // Run local ingest pipeline (GPS parsing + matching)
    const result = await Pipeline.ingestFiles(state.projectId, picked, (msg, pct) => {
      updateProgressRing(Math.round(pct * 100));
      if(zone) zone.innerHTML = `<div class="drop-zone-text" style="color:var(--accent)">${msg}</div>`;
    });
    console.log('[Upload] ingest done, errors:', result.errors.length);
    if (result.errors.length) {
      console.warn('Ingest errors:', result.errors);
      // Show brief toast for user visibility
      const errMsg = result.errors.map(e => '• ' + e).join('\n');
      setTimeout(() => alert('Some files had issues:\n' + errMsg), 100);
    }
    const hasUploadedVideo = result.files.some(file => file.kind === 'video');
    if (hasUploadedVideo && hasApiCsvConfig()) {
      try {
        await importApiCsvsForUploadedVideos(result.files, (msg, pct) => {
          updateProgressRing(Math.round(pct * 100));
          if(zone) zone.innerHTML = `<div class="drop-zone-text" style="color:var(--accent)">${msg}</div>`;
        });
      } catch (err) {
        console.warn('[API CSV] automatic import failed:', err);
      }
    } else if (hasUploadedVideo) {
      syncApiCsvInputs(null, '');
    }
    await loadMapData();
    const vids = state.mapData?.videos || [];
    console.log('[Upload] mapData loaded, videos:', vids.length, vids.map(v => `${v.id} ts=${v.ts_start}-${v.ts_end} dur=${v.duration_sec}`));
    renderAthletes(); renderFilesList(); buildTimeline();
    void refreshManeuvers('files-ingested');
    zone.innerHTML = `<span class="drop-zone-icon"><svg class="icon-svg" style="width:48px;height:48px"><use href="#ico-upload"/></svg></span><div class="drop-zone-text"><strong>Click to browse</strong> or drag & drop more files</div><div class="drop-zone-hint">Accepts .mp4, .mov, .csv files</div>`;
    updateProgressRing(100);
    setTimeout(() => updateProgressRing(0), 2000);
  } catch(err) {
    alert('Processing failed: ' + err.message);
    zone.innerHTML = `<span class="drop-zone-icon"><svg class="icon-svg" style="width:48px;height:48px"><use href="#ico-upload"/></svg></span><div class="drop-zone-text"><strong>Click to browse</strong> or drag & drop files here</div><div class="drop-zone-hint">Accepts .mp4, .mov, .csv files</div>`;
    updateProgressRing(0);
  } finally {
    _activeUploadCount = Math.max(0, _activeUploadCount - 1);
    state.uploadInProgress = _activeUploadCount > 0;
    updateSetupUiState();
  }
}

// ── Upload (browser-side via pipeline.js) ──────────────────────────────
function _truncName(name, max) {
  if (name.length <= max) return name;
  const ext = name.lastIndexOf('.') !== -1 ? name.slice(name.lastIndexOf('.')) : '';
  return name.slice(0, max - ext.length - 1) + '…' + ext;
}

// NOTE: _uploadSingleFile and uploadFiles removed — replaced by
// FM.registerPickedFiles() + Pipeline.ingestFiles() in handleFileUpload()

// ── Job polling ────────────────────────────────────────────────────────
function setBadge(text, cls) {
  // Legacy badge removed — use progress ring
  if (cls === 'running') updateProgressRing(state._lastJobPct || 5);
  else if (cls === 'completed') { updateProgressRing(100); setTimeout(() => updateProgressRing(0), 3000); }
  else if (cls === 'failed') { updateProgressRing(0); }
}

function startJobPoll(jobId, onComplete) {
  state.jobId = jobId;
  state._lastJobPct = 0;
  state._lastJobStatus = {status:'pending', progress:0, message:'Starting…'};
  state._onJobComplete = onComplete || null;
  if(state.jobPollTimer) clearInterval(state.jobPollTimer);
  updateProgressRing(2);
  renderQueue();
  // Poll immediately, then every 1.5s
  pollJob(jobId);
  state.jobPollTimer = setInterval(()=>pollJob(jobId), 1500);
}

async function pollJob(jobId) {
  // Browser-side: jobs are processed locally. Polling is a no-op.
  // This function is kept for compatibility but does nothing.
  if (state.jobPollTimer) { clearInterval(state.jobPollTimer); state.jobPollTimer = null; }
  state.jobId = null;
  state._lastJobStatus = null;
  updateProgressRing(0);
  renderQueue();
}

function getWorkersValue() { return parseInt(el('workers-input')?.value||'2')||2; }
function getSkeletonFps() { return parseInt(el('skeleton-fps-input')?.value||'10')||10; }
function getPoseInputMaxDim() {
  const selected = Number(el('pose-input-size-select')?.value);
  const value = Number.isFinite(selected) ? selected : Number(state.poseInputMaxDim);
  if (value <= 360) return 360;
  if (value >= 640) return 640;
  return 480;
}
function isPoseExactSegmentSeekEnabled() {
  return !!state.poseExactSegmentSeek;
}
function getManeuverPoseFps() {
  return Math.max(getSkeletonFps(), MANEUVER_POSE_TARGET_FPS);
}
function getSelectedModel() {
  const value = String(el('pose-model-select')?.value || 'lite').toLowerCase();
  if (value === 'heavy') return 'heavy';
  if (value === 'full') return 'full';
  return 'lite';
}
function areBoomPredictionsEnabled() {
  return state.advancedFeatures?.boomPredictions !== false;
}
function areRudderPredictionsEnabled() {
  return state.advancedFeatures?.rudderPredictions !== false;
}
function isRealtimePoseEnabled() { return false; }

// ── Real-time pose overlay drawing ─────────────────────────────────────

const POSE_CONNECTIONS_2D = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
  [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
  [27,29],[28,30],[29,31],[30,32],[27,31],[28,32]
];

function drawPoseLandmarksOnCanvas(ctx, normLm, w, h) {
  ctx.clearRect(0, 0, w, h);
  if (!normLm || normLm.length === 0) return;

  // Draw connections (bones)
  ctx.strokeStyle = 'rgba(0, 255, 128, 0.7)';
  ctx.lineWidth = 2;
  for (const [a, b] of POSE_CONNECTIONS_2D) {
    if (a >= normLm.length || b >= normLm.length) continue;
    const la = normLm[a], lb = normLm[b];
    if (la.visibility < 0.3 || lb.visibility < 0.3) continue;
    ctx.beginPath();
    ctx.moveTo(la.x * w, la.y * h);
    ctx.lineTo(lb.x * w, lb.y * h);
    ctx.stroke();
  }

  // Draw landmarks (joints)
  ctx.fillStyle = 'rgba(255, 64, 64, 0.85)';
  for (let i = 0; i < normLm.length; i++) {
    const lm = normLm[i];
    if (lm.visibility < 0.3) continue;
    const x = lm.x * w, y = lm.y * h;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function syncOverlayCanvasSize(slot) {
  const canvas = slot._overlayCanvas;
  const video = slot.videoEl;
  if (!canvas || !video) return;
  const rect = video.getBoundingClientRect();
  if (canvas.width !== rect.width || canvas.height !== rect.height) {
    canvas.width = rect.width;
    canvas.height = rect.height;
  }
}

function clearOverlayCanvas(slot) {
  const c = slot._overlayCanvas;
  if (c) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height); }
}

// Track busy flag per slot to avoid overlapping detections
let _realtimeBusy = new WeakMap();

async function runRealtimeDetectionOnSlot(slot) {
  if (!slot.videoEl || slot.videoEl.paused || slot.videoEl.ended) return;
  if (!slot._overlayCanvas) return;
  if (_realtimeBusy.get(slot)) return; // skip if previous detection still running

  _realtimeBusy.set(slot, true);
  try {
    syncOverlayCanvasSize(slot);
    const ts = performance.now();
    const result = await PoseEngine.detectLive(slot.videoEl, ts, getSelectedModel());
    if (result && result.normLm) {
      const ctx = slot._overlayCanvas.getContext('2d');
      drawPoseLandmarksOnCanvas(ctx, result.normLm, slot._overlayCanvas.width, slot._overlayCanvas.height);
    } else {
      clearOverlayCanvas(slot);
    }
  } catch (e) {
    console.warn('Realtime pose detection error:', e);
  } finally {
    _realtimeBusy.set(slot, false);
  }
}

function clearAllOverlays() {
  if (!state.tl?.athleteSlots) return;
  for (const slot of state.tl.athleteSlots) {
    clearOverlayCanvas(slot);
  }
}

// ── End real-time pose overlay ──────────────────────────────────────────

// Queue segment-scoped processing requests per file to avoid dropping runs
// when users create multiple segments quickly.
const _queuedSegmentRuns = new Map(); // fileId -> Array<{startSec?, endSec?, segmentName?}>
const _startingSkeletonRuns = new Map(); // fileId -> {startTime, progress, message, startSec?, endSec?, segmentName?}
const MIN_SEGMENT_POSE_RANGE_SEC = 0.35;

function normalizeSegmentRunOpts(segmentOpts = {}) {
  const startSec = Number.isFinite(Number(segmentOpts?.startSec)) ? Math.max(0, Number(segmentOpts.startSec)) : null;
  const endSec = Number.isFinite(Number(segmentOpts?.endSec)) ? Math.max(startSec ?? 0, Number(segmentOpts.endSec)) : null;
  const segmentName = typeof segmentOpts?.segmentName === 'string' ? segmentOpts.segmentName.trim() : '';
  const fpsRaw = Number(segmentOpts?.fps);
  const fps = Number.isFinite(fpsRaw) ? Math.max(1, Math.min(30, fpsRaw)) : null;
  return {
    startSec,
    endSec,
    segmentName: segmentName || null,
    fps,
  };
}

function hasSegmentRunBounds(segmentOpts = {}) {
  return Number.isFinite(Number(segmentOpts?.startSec)) || Number.isFinite(Number(segmentOpts?.endSec));
}

function sameSegmentRunRequest(a, b, tolerance = 0.35) {
  const aNorm = normalizeSegmentRunOpts(a);
  const bNorm = normalizeSegmentRunOpts(b);
  const aHasBounds = hasSegmentRunBounds(aNorm);
  const bHasBounds = hasSegmentRunBounds(bNorm);
  if (aHasBounds !== bHasBounds) return false;
  if (!aHasBounds) return true;

  const aStart = aNorm.startSec ?? 0;
  const bStart = bNorm.startSec ?? 0;
  const aEnd = Number.isFinite(aNorm.endSec) ? aNorm.endSec : Infinity;
  const bEnd = Number.isFinite(bNorm.endSec) ? bNorm.endSec : Infinity;
  const sameEnd = (!Number.isFinite(aEnd) && !Number.isFinite(bEnd)) || Math.abs(aEnd - bEnd) <= tolerance;
  const aFps = Number.isFinite(Number(aNorm.fps)) ? Number(aNorm.fps) : null;
  const bFps = Number.isFinite(Number(bNorm.fps)) ? Number(bNorm.fps) : null;
  const sameFps = (aFps == null && bFps == null) || Math.abs((aFps ?? 0) - (bFps ?? 0)) <= 0.1;
  return Math.abs(aStart - bStart) <= tolerance && sameEnd && sameFps;
}

function getStartingSegmentJobs() {
  const out = [];
  for (const [fileId, job] of _startingSkeletonRuns.entries()) {
    out.push({
      fileId,
      progress: Number.isFinite(Number(job?.progress)) ? Number(job.progress) : 0,
      message: String(job?.message || 'Preparing...'),
      startTime: Number.isFinite(Number(job?.startTime)) ? Number(job.startTime) : Date.now(),
      cancelled: false,
      segmentName: job?.segmentName || null,
      startSec: Number.isFinite(Number(job?.startSec)) ? Number(job.startSec) : null,
      endSec: Number.isFinite(Number(job?.endSec)) ? Number(job.endSec) : null,
      pendingStart: true,
    });
  }
  return out;
}

function hasEquivalentQueuedSegmentRun(fileId, segmentOpts = {}) {
  const runs = _queuedSegmentRuns.get(fileId) || [];
  return runs.some(run => sameSegmentRunRequest(run, segmentOpts));
}

function hasEquivalentStartingSegmentRun(fileId, segmentOpts = {}) {
  const run = _startingSkeletonRuns.get(fileId);
  return !!run && sameSegmentRunRequest(run, segmentOpts);
}

function hasEquivalentActiveSegmentRun(fileId, segmentOpts = {}) {
  const active = PoseEngine.getActiveJobs().find(job => String(job.fileId) === String(fileId));
  return !!active && sameSegmentRunRequest(active, segmentOpts);
}

function getQueuedSegmentJobs() {
  const out = [];
  for (const [fileId, runs] of _queuedSegmentRuns.entries()) {
    const queue = Array.isArray(runs) ? runs : [];
    for (let i = 0; i < queue.length; i++) {
      out.push({ fileId, queueIdx: i, queueLen: queue.length, opts: queue[i] || {} });
    }
  }
  return out;
}

function dequeueQueuedSegmentRun(fileId, queueIdx) {
  const runs = _queuedSegmentRuns.get(fileId);
  if(!runs || !runs.length) return false;
  if(queueIdx < 0 || queueIdx >= runs.length) return false;
  runs.splice(queueIdx, 1);
  if(runs.length === 0) _queuedSegmentRuns.delete(fileId);
  else _queuedSegmentRuns.set(fileId, runs);
  return true;
}

async function runSkeletonForVideo(fileId, segmentOpts = {}) {
  if(!state.projectId) { alert('No project selected'); return; }
  if(!fileId) { alert('No video selected'); return; }
  const fileRec = (state.mapData?.videos || []).find(v => String(v?.id) === String(fileId))
    || await DB.getFile(fileId);
  if (isPlaybackOnlyVideo(fileRec)) {
    alert('This video is playback-only and is excluded from athlete analysis.');
    return;
  }
  const normalizedSegmentOpts = normalizeSegmentRunOpts(segmentOpts);
  if (hasSegmentRunBounds(normalizedSegmentOpts)) {
    const durationLimit = Number(fileRec?.duration_sec);
    const hasDurationLimit = Number.isFinite(durationLimit) && durationLimit > 0;
    let startSec = Number.isFinite(Number(normalizedSegmentOpts.startSec))
      ? Math.max(0, Number(normalizedSegmentOpts.startSec))
      : null;
    let endSec = Number.isFinite(Number(normalizedSegmentOpts.endSec))
      ? Math.max(0, Number(normalizedSegmentOpts.endSec))
      : null;

    if (hasDurationLimit) {
      if (startSec != null) startSec = Math.min(startSec, durationLimit);
      if (endSec != null) endSec = Math.min(endSec, durationLimit);
    }
    if (startSec != null && endSec != null && (endSec - startSec) < MIN_SEGMENT_POSE_RANGE_SEC) {
      if (hasDurationLimit) {
        const safeStart = Math.max(0, Math.min(startSec, Math.max(0, durationLimit - MIN_SEGMENT_POSE_RANGE_SEC)));
        const safeEnd = Math.min(durationLimit, safeStart + MIN_SEGMENT_POSE_RANGE_SEC);
        startSec = safeStart;
        endSec = safeEnd;
      } else {
        endSec = startSec + MIN_SEGMENT_POSE_RANGE_SEC;
      }
    }
    if (startSec != null && endSec != null && !(endSec > startSec)) {
      console.warn(`[Pose] skipped empty segment range for ${fileId}: start=${startSec}, end=${endSec}`);
      return;
    }
    normalizedSegmentOpts.startSec = startSec;
    normalizedSegmentOpts.endSec = endSec;
  }
  const segName = normalizedSegmentOpts.segmentName || '';
  const runBusy = PoseEngine.isProcessing(fileId) || _startingSkeletonRuns.has(fileId);
  if(runBusy) {
    const hasSegmentBounds = hasSegmentRunBounds(normalizedSegmentOpts);
    if (hasSegmentBounds) {
      if (
        hasEquivalentStartingSegmentRun(fileId, normalizedSegmentOpts) ||
        hasEquivalentActiveSegmentRun(fileId, normalizedSegmentOpts) ||
        hasEquivalentQueuedSegmentRun(fileId, normalizedSegmentOpts)
      ) {
        console.log(`[Pose] skipped duplicate segment processing request for ${fileId}`);
        renderQueue();
        return;
      }
      const q = _queuedSegmentRuns.get(fileId) || [];
      q.push({
        startSec: normalizedSegmentOpts.startSec,
        endSec: normalizedSegmentOpts.endSec,
        segmentName: segName || undefined,
        fps: normalizedSegmentOpts.fps ?? undefined,
      });
      _queuedSegmentRuns.set(fileId, q);
      console.log(`[Pose] queued segment processing for ${fileId} (queue=${q.length})`);
      renderQueue();
      return;
    }
    alert('Already processing this video');
    return;
  }
  _startingSkeletonRuns.set(fileId, {
    startTime: Date.now(),
    progress: 0.01,
    message: 'Preparing processing...',
    startSec: normalizedSegmentOpts.startSec,
    endSec: normalizedSegmentOpts.endSec,
    segmentName: segName || null,
  });
  renderQueue(); // show immediately in queue
  try {
    const cvConfig = await DB.getCvConfig(state.projectId) || {};
    const starting = _startingSkeletonRuns.get(fileId);
    if (starting) {
      starting.progress = 0.03;
      starting.message = 'Loading processing settings...';
    }
    renderQueue();
    const fps = Number.isFinite(Number(normalizedSegmentOpts.fps))
      ? Number(normalizedSegmentOpts.fps)
      : getSkeletonFps();
    const model = getSelectedModel();
    const opts = {
      fps,
      model,
      poseInputMaxDim: getPoseInputMaxDim(),
      exactSegmentSeek: isPoseExactSegmentSeekEnabled(),
      enableBoomPrediction: areBoomPredictionsEnabled(),
      enableRudderPrediction: areRudderPredictionsEnabled(),
      ...getPoseAthleteOptionsForVideo(fileRec),
    };
    // If segment bounds provided (video-local seconds), limit processing
    if (normalizedSegmentOpts.startSec != null) opts.startSec = normalizedSegmentOpts.startSec;
    if (normalizedSegmentOpts.endSec != null) opts.endSec = normalizedSegmentOpts.endSec;
    if (segName) opts.segmentName = segName;
    const processingPromise = PoseEngine.processVideo(state.projectId, fileId, cvConfig, (msg, pct) => {
      // Queue renders itself via polling now; just update badge
      const badge = el('badge');
      if(badge) badge.textContent = msg;
    }, opts);
    _startingSkeletonRuns.delete(fileId);
    // Kick queue render again after starting the async job so polling always arms.
    renderQueue();
    await processingPromise;
    await refreshSkeletonCoverageForVideo(fileId, normalizedSegmentOpts);
    invalidateTimelineProcessedMetricCache(fileId);
    refreshTimelineProcessedMetricStats();
    // Reload data
    state.cvStatuses = await Pipeline.buildCvStatuses(state.projectId);
    await loadAllSkeletonCoverages();
    renderQueue();
    renderSegmentPanel();
    renderAnalysisTab();
    updateCurrentSegmentActions();
    drawSogCanvas();
    renderManeuverPanel();
    renderManeuverMap({ preserveView: true });
    if (maneuverWorkspaceShouldRefreshForVideo(fileId)) {
      void renderManeuverWorkspace({ loadDeep: true, force: true });
      setTimeout(() => {
        void renderManeuverWorkspace({ loadDeep: false, force: true });
        renderManeuverPanel();
        renderManeuverMap({ preserveView: true });
      }, 180);
    }
  } catch(e) {
    console.error('Skeleton processing failed:', e);
    if (isLikelyStorageError(e)) {
      await logStorageDebug('runSkeletonForVideo');
      alert('Skeleton processing failed due to browser storage limits. Try deleting old reports and reloading the page.');
    } else {
      alert('Skeleton processing failed: ' + e.message);
    }
    renderQueue();
    renderSegmentPanel();
    renderAnalysisTab();
    updateCurrentSegmentActions();
  } finally {
    _startingSkeletonRuns.delete(fileId);
    const q = _queuedSegmentRuns.get(fileId);
    if (q && q.length > 0) {
      const next = q.shift();
      if (q.length === 0) _queuedSegmentRuns.delete(fileId);
      // Fire next queued segment run after current one finishes.
      runSkeletonForVideo(fileId, next || {}).catch(err => {
        console.warn(`Queued skeleton processing failed for ${fileId}:`, err);
      });
    }
  }
}

/**
 * Run skeleton processing for all video files in the current project.
 */
/**
 * Process all segments: for each segment, process ALL overlapping videos
 * within the segment's time bounds. This is the primary workflow.
 */
async function runSkeletonForAllVideos() {
  if(!state.projectId) { alert('No project selected'); return; }
  const btn = el('btn-run-skeleton');
  if(btn) { btn.disabled = true; btn.textContent = 'Processing...'; }
  renderQueue();

  try {
    state.cvStatuses = await Pipeline.buildCvStatuses(state.projectId);
    // If segments exist, process only segment time ranges (multi-athlete aware)
    if (state.segments.length > 0) {
      let totalJobs = 0, done = 0, skipped = 0;
      // Collect unique (fileId, startSec, endSec) jobs across all segments
      const jobs = []; // [{fileId, filename, startSec, endSec, segName}]
      const dedupe = new Set();
      for (const seg of state.segments) {
        const overlapping = findOverlappingVideos(seg, { analyzableOnly: true });
        for (const { vid, videoStartSec, videoEndSec } of overlapping) {
          const key = `${vid.id}:${videoStartSec.toFixed(1)}:${(videoEndSec ?? 'end')}`;
          if (dedupe.has(key)) continue;
          dedupe.add(key);
          if (isSkeletonRangeReady(vid.id, videoStartSec, videoEndSec, vid.duration_sec)) {
            skipped++;
            continue;
          }
          jobs.push({
            fileId: vid.id,
            filename: vid.filename || vid.id.slice(0, 8),
            startSec: videoStartSec,
            endSec: videoEndSec,
            segName: seg.name,
          });
        }
      }
      totalJobs = jobs.length;
      if (totalJobs === 0) {
        alert(skipped > 0
          ? `All ${skipped} video-segment job${skipped === 1 ? '' : 's'} already have processed coverage.`
          : 'No analyzable videos overlap any segments.');
        return;
      }
      jobs.sort((a, b) => {
        const fileCmp = String(a.fileId).localeCompare(String(b.fileId));
        if (fileCmp !== 0) return fileCmp;
        return (Number(a.startSec) || 0) - (Number(b.startSec) || 0);
      });
      seedQueueAggregateProgress(jobs.map(job => ({
        fileId: job.fileId,
        startSec: job.startSec,
        endSec: job.endSec,
        segmentName: job.segName,
      })));

      console.log(`[Skeleton] processing ${totalJobs} video-segment jobs across ${state.segments.length} segments (${skipped} already covered)`);
      for (const job of jobs) {
        if (PoseEngine.isProcessing(job.fileId)) {
          console.log(`[Skeleton] ${job.filename} already processing, queueing...`);
        }
        const badge = el('badge');
        if (badge) badge.textContent = `${done + 1}/${totalJobs}: ${job.filename} (${job.segName})`;
        try {
          await runSkeletonForVideo(job.fileId, { startSec: job.startSec, endSec: job.endSec, segmentName: job.segName });
          done++;
        } catch (e) {
          console.warn(`Skeleton failed for ${job.filename} in ${job.segName}:`, e);
        }
      }
      state.cvStatuses = await Pipeline.buildCvStatuses(state.projectId);
      await loadAllSkeletonCoverages();
      renderQueue();
      renderAnalysisTab();
      drawSogCanvas();
      alert(`Skeleton processing complete: ${done}/${totalJobs} jobs processed${skipped ? `, ${skipped} already ready` : ''}.`);

    } else {
      // No segments: process all videos at full length (fallback)
      const files = await DB.listFiles(state.projectId);
      const allVideos = files.filter(f => f.kind === 'video' && !f.playback_only);
      const videos = allVideos.filter(v => !isSkeletonRangeReady(v.id, 0, v.duration_sec, v.duration_sec));
      const skipped = allVideos.length - videos.length;
      if (allVideos.length === 0) { alert('No analyzable video files in project.'); return; }
      if (videos.length === 0) {
        alert(`All ${skipped} video${skipped === 1 ? '' : 's'} already have processed coverage.`);
        return;
      }
      seedQueueAggregateProgress(videos.map(v => ({ fileId: v.id })));

      let done = 0;
      for (const v of videos) {
        if (PoseEngine.isProcessing(v.id)) continue;
        try {
          const cvConfig = await DB.getCvConfig(state.projectId) || {};
          const fps = getSkeletonFps();
          const model = getSelectedModel();
          const badge = el('badge');
          if (badge) badge.textContent = `Skeleton ${done + 1}/${videos.length}: ${v.filename}`;
          const processingPromise = PoseEngine.processVideo(state.projectId, v.id, cvConfig, (msg, pct) => {
            if (badge) badge.textContent = `${v.filename}: ${msg}`;
          }, {
            fps,
            model,
            poseInputMaxDim: getPoseInputMaxDim(),
            exactSegmentSeek: isPoseExactSegmentSeekEnabled(),
            enableBoomPrediction: areBoomPredictionsEnabled(),
            enableRudderPrediction: areRudderPredictionsEnabled(),
            ...getPoseAthleteOptionsForVideo(v),
          });
          // Kick queue render again after starting the async job so polling always arms.
          renderQueue();
          await processingPromise;
          invalidateTimelineProcessedMetricCache(v.id);
          refreshTimelineProcessedMetricStats();
          done++;
        } catch (e) {
          console.warn(`Skeleton failed for ${v.filename}:`, e);
        }
      }
      state.cvStatuses = await Pipeline.buildCvStatuses(state.projectId);
      await loadAllSkeletonCoverages();
      renderQueue();
      renderAnalysisTab();
      drawSogCanvas();
      alert(`Skeleton processing complete: ${done}/${videos.length} videos processed${skipped ? `, ${skipped} already ready` : ''}.`);
    }
  } catch(e) {
    console.error('Skeleton processing failed:', e);
    alert('Skeleton processing failed: ' + e.message);
    renderQueue();
  } finally {
    if(btn) { btn.disabled = false; btn.innerHTML = '<svg class="icon-svg" style="width:16px;height:16px"><use href="#ico-process"></use></svg> Run Skeleton'; }
  }
}

/**
 * Reprocess a single segment: find all overlapping videos and run
 * skeleton/MediaPipe/AutoPnP for each within the segment's time bounds.
 */
async function reprocessSegment(seg) {
  if (!state.projectId) { alert('No project selected'); return; }
  if (state.inlineHeatmaps?.visible) setInlineHeatmapsVisible(false);
  if (state.advancedPane?.mode === 'heatmaps') closeStlViewer();
  const overlapping = findOverlappingVideos(seg, { analyzableOnly: true });
  if (overlapping.length === 0) {
    alert(`No analyzable videos overlap segment "${seg.name}".`);
    return;
  }
  console.log(`[Segment] reprocessing "${seg.name}": ${overlapping.length} video(s)`);
  renderQueue();
  seedQueueAggregateProgress(overlapping.map(({ vid, videoStartSec, videoEndSec }) => ({
    fileId: vid.id,
    startSec: videoStartSec,
    endSec: videoEndSec,
    segmentName: seg.name,
  })));
  const jobs = [];
  for (const { vid, videoStartSec, videoEndSec } of overlapping) {
    console.log(`[Segment] processing ${vid.filename}: video_s ${videoStartSec.toFixed(1)} - ${videoEndSec?.toFixed(1) ?? 'end'}`);
    jobs.push(
      runSkeletonForVideo(vid.id, { startSec: videoStartSec, endSec: videoEndSec, segmentName: seg.name })
        .catch(e => {
          console.warn(`Skeleton failed for ${vid.filename} in segment "${seg.name}":`, e);
          throw e;
        })
    );
  }
  renderSegmentPanel();
  renderAnalysisTab();
  updateCurrentSegmentActions();

  const results = await Promise.allSettled(jobs);
  state.cvStatuses = await Pipeline.buildCvStatuses(state.projectId);
  await loadAllSkeletonCoverages();
  renderQueue();
  renderSegmentPanel();
  renderAnalysisTab();
  updateCurrentSegmentActions();
  drawSogCanvas();

  const failures = results.filter(r => r.status === 'rejected');
  if (failures.length) {
    alert(`Processing finished with ${failures.length} failed job${failures.length === 1 ? '' : 's'} for segment "${seg.name}".`);
  }
}

// ── Segment creation (project-level, absolute timestamps) ──────────────

async function loadSegments() {
  if(!state.projectId) { state.segments = []; return; }
  try { state.segments = await DB.getSegments(state.projectId); }
  catch(e) { state.segments = []; }
}

async function saveSegments() {
  if(!state.projectId) return;
  await DB.saveSegments(state.projectId, state.segments);
}

function getNextSegmentName() {
  const existing = state.segments.map(s => s.name);
  for(let i = 1; ; i++) {
    const name = `Segment ${i}`;
    if(!existing.includes(name)) return name;
  }
}

function startSegmentCreation() {
  const ss = state.segmentSelect;
  // If already active, cancel first
  if(ss.active) { cancelSegmentCreation(); return; }
  ss.active = true;
  ss.step = 'start';
  ss.tsStart = null; ss.tsEnd = null;
  clearSegmentSelectMarkers();
  el('btn-new-segment').classList.add('active');
  el('seg-status').classList.add('active');
  el('seg-status-text').textContent = 'Click track or scrub bar to set START';
  // Switch to analysis view if not already there
  switchView('view-analysis');
  drawSogCanvas();
}

function cancelSegmentCreation() {
  const ss = state.segmentSelect;
  ss.active = false; ss.step = null; ss.tsStart = null; ss.tsEnd = null;
  clearSegmentSelectMarkers();
  el('btn-new-segment').classList.remove('active');
  el('seg-status').classList.remove('active');
  drawSogCanvas();
}

function clearSegmentSelectMarkers() {
  const ss = state.segmentSelect;
  if(ss.startMarker && state.map) { state.map.removeLayer(ss.startMarker); ss.startMarker = null; }
  if(ss.endMarker && state.map) { state.map.removeLayer(ss.endMarker); ss.endMarker = null; }
  if(ss.highlightLayer && state.map) { state.map.removeLayer(ss.highlightLayer); ss.highlightLayer = null; }
}

function handleSegmentClick(ts, latlng) {
  if(ts == null) return;
  const ss = state.segmentSelect;
  if(!ss.active) return;

  if(ss.step === 'start') {
    ss.tsStart = ts;
    ss.step = 'end';
    // Place green start marker on map
    if(ss.startMarker && state.map) state.map.removeLayer(ss.startMarker);
    if(latlng && state.map) {
      ss.startMarker = L.circleMarker(latlng, {
        radius:10,
        color:'#ffffff',
        weight:2.5,
        fillColor:'#2ea043',
        fillOpacity:1,
        pane:'splitMarkersPane',
      }).addTo(state.map);
    }
    el('seg-status-text').textContent = 'Click track or scrub bar to set END';
    drawSogCanvas();
  } else if(ss.step === 'end') {
    ss.tsEnd = ts;
    // Auto-swap if start > end
    if(ss.tsStart > ss.tsEnd) {
      [ss.tsStart, ss.tsEnd] = [ss.tsEnd, ss.tsStart];
    }
    // Place red end marker on map
    if(ss.endMarker && state.map) state.map.removeLayer(ss.endMarker);
    if(latlng && state.map) {
      ss.endMarker = L.circleMarker(latlng, {
        radius:10,
        color:'#ffffff',
        weight:2.5,
        fillColor:'#e3342f',
        fillOpacity:1,
        pane:'splitMarkersPane',
      }).addTo(state.map);
    }
    // Draw highlight between start and end
    drawSegmentSelectHighlight();
    drawSogCanvas();
    // Open naming modal
    openSegmentNamingModal();
  }
}

function setSegmentStartAtCurrentPlayhead() {
  const ts = Number(state.tl?.currentTs);
  if (!Number.isFinite(ts)) return;
  if (!state.segmentSelect.active) startSegmentCreation();
  const ss = state.segmentSelect;
  ss.step = 'start';
  ss.tsStart = null;
  ss.tsEnd = null;
  if (ss.endMarker && state.map) { state.map.removeLayer(ss.endMarker); ss.endMarker = null; }
  if (ss.highlightLayer && state.map) { state.map.removeLayer(ss.highlightLayer); ss.highlightLayer = null; }
  handleSegmentClick(ts, findLatLonForTs(ts));
}

function setSegmentEndAtCurrentPlayhead() {
  const ts = Number(state.tl?.currentTs);
  if (!Number.isFinite(ts)) return;
  const ss = state.segmentSelect;
  if (!ss.active || ss.tsStart == null) return;
  ss.step = 'end';
  handleSegmentClick(ts, findLatLonForTs(ts));
}

function drawSegmentSelectHighlight() {
  const ss = state.segmentSelect;
  if(ss.highlightLayer && state.map) { state.map.removeLayer(ss.highlightLayer); ss.highlightLayer = null; }
  if(ss.tsStart == null || ss.tsEnd == null || !state.mapData) return;
  const vids = state.mapData.videos || [];
  const lg = L.layerGroup();
  for(const v of vids) {
    const color = state.videoColors[v.id] || '#f5a623';
    const pts = (v.points || []).filter(p => p.ts != null && p.ts >= ss.tsStart && p.ts <= ss.tsEnd);
    if(pts.length >= 2) {
      L.polyline(pts.map(p=>[p.lat,p.lon]), {color, weight:10, opacity:0.85}).addTo(lg);
    }
  }
  lg.addTo(state.map);
  ss.highlightLayer = lg;
}

function closeSegmentModal({ cancelCreation = false } = {}) {
  el('segment-modal').classList.remove('open');
  if (cancelCreation && state.segmentModal.mode === 'create') cancelSegmentCreation();
  state.segmentModal.mode = 'create';
  state.segmentModal.segmentId = null;
}

function openSegmentNamingModal() {
  const ss = state.segmentSelect;
  if(ss.tsStart == null || ss.tsEnd == null) return;
  const dur = ss.tsEnd - ss.tsStart;
  state.segmentModal.mode = 'create';
  state.segmentModal.segmentId = null;
  el('segment-modal-title').textContent = 'Name this segment';
  el('btn-seg-modal-save').textContent = 'Save Segment';
  el('seg-modal-range').textContent = `${fmtClock(ss.tsStart)} → ${fmtClock(ss.tsEnd)} (${dur.toFixed(1)}s)`;
  el('seg-name-in').value = getNextSegmentName();
  el('segment-modal').classList.add('open');
  setTimeout(() => { el('seg-name-in').focus(); el('seg-name-in').select(); }, 50);
}

function openSegmentRenameModal(segId) {
  const seg = getSegmentById(segId);
  if (!seg) return;
  state.segmentModal.mode = 'rename';
  state.segmentModal.segmentId = String(seg.id);
  el('segment-modal-title').textContent = 'Rename segment';
  el('btn-seg-modal-save').textContent = 'Save';
  const dur = Math.max(0, (Number(seg.tsEnd) || 0) - (Number(seg.tsStart) || 0));
  el('seg-modal-range').textContent = `${fmtClock(seg.tsStart)} - ${fmtClock(seg.tsEnd)} (${dur.toFixed(1)}s)`;
  el('seg-name-in').value = seg.name || '';
  el('segment-modal').classList.add('open');
  setTimeout(() => { el('seg-name-in').focus(); el('seg-name-in').select(); }, 50);
}

async function renameSegment(segId, nextName) {
  const seg = getSegmentById(segId);
  const name = String(nextName || '').trim();
  if (!seg || !name || name === seg.name) return;
  seg.name = name;
  await saveSegments();
  renderMap();
  renderSegmentPanel();
  renderAnalysisTab();
  drawSogCanvas();
  updateCurrentSegmentActions();
}

async function saveNewSegment() {
  const modalMode = state.segmentModal.mode;
  const segmentId = state.segmentModal.segmentId;
  const ss = state.segmentSelect;
  const fallbackName = modalMode === 'rename' ? (getSegmentById(segmentId)?.name || '') : getNextSegmentName();
  const name = el('seg-name-in').value.trim() || fallbackName;
  closeSegmentModal();

  if (modalMode === 'rename') {
    await renameSegment(segmentId, name);
    return;
  }

  const seg = {
    id: DB.uuid(),
    name,
    tsStart: ss.tsStart,
    tsEnd: ss.tsEnd,
  };
  state.segments.push(seg);
  state.analysisSelected = [...new Set([...(state.analysisSelected || []), seg.id])];
  saveAnalysisSelection();
  await saveSegments();
  cancelSegmentCreation();

  // Auto-start processing pipeline for ALL overlapping videos in the segment
  try {
    reprocessSegment(seg);
  } catch(e) { console.warn('Auto-process after segment creation failed:', e); }

  renderMap();
  renderSegmentPanel();
  renderAnalysisTab();
  drawSogCanvas();
  updateCurrentSegmentActions();
}

async function deleteSegment(segId) {
  const seg = getSegmentById(segId);
  const overlapping = seg ? findOverlappingVideos(seg, { analyzableOnly: true }) : [];
  state.segments = state.segments.filter(s => s.id !== segId);
  const segKey = String(segId);
  state.analysisSelected = (state.analysisSelected || []).map(String).filter(id => id !== segKey);
  saveAnalysisSelection();
  await saveSegments();
  for (const { vid, videoStartSec, videoEndSec } of overlapping) {
    try {
      const duration = Number(vid?.duration_sec);
      const endSec = Number.isFinite(Number(videoEndSec))
        ? Number(videoEndSec)
        : (Number.isFinite(duration) ? duration : null);
      if (Number.isFinite(Number(videoStartSec)) && Number.isFinite(Number(endSec)) && endSec > videoStartSec) {
        await PoseEngine.deleteProcessedRange(state.projectId, vid.id, videoStartSec, endSec);
        invalidateTimelineProcessedMetricCache(vid.id);
      }
    } catch (err) {
      console.warn(`[Segment] failed to delete processed data for ${vid?.filename || vid?.id}:`, err);
    }
  }
  state.cvStatuses = await Pipeline.buildCvStatuses(state.projectId);
  await loadAllSkeletonCoverages();
  refreshTimelineProcessedMetricStats();
  renderMap();
  renderSegmentPanel();
  renderAnalysisTab();
  drawSogCanvas();
  updateCurrentSegmentActions();
}

function renderSegmentPanel() {
  const list = el('segment-panel-list');
  if(!list) return;
  list.innerHTML = '';
  const panel = el('segment-panel');
  const hdr = el('segment-panel-hdr');
  hdr.textContent = `\u2702 Segments (${state.segments.length})`;
  if(!state.segments.length) { panel.classList.add('collapsed'); return; }
  for(const seg of state.segments) {
    const athletes = getSegmentAthletes(seg);
    const overlapping = findOverlappingVideos(seg);
    const row = document.createElement('div');
    row.className = 'seg-row';
    const seekToSegment = () => {
      if(seg.tsStart != null) tlSeekTo(seg.tsStart);
    };
    row.onclick = seekToSegment;
    const nameEl = document.createElement('span');
    nameEl.className = 'seg-name';
    nameEl.textContent = seg.name;
    nameEl.onclick = seekToSegment;
    const rangeEl = document.createElement('span');
    rangeEl.className = 'seg-range';
    const dur = ((seg.tsEnd || 0) - (seg.tsStart || 0));
    const vidInfo = `${overlapping.length}v`;
    const athColors = athletes.map(a =>
      `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${a.color};" title="${a.name}"></span>`
    ).join(' ');
    rangeEl.innerHTML = `${fmtClock(seg.tsStart)} (${dur.toFixed(0)}s) ${athColors} <span style="color:var(--muted);font-size:10px;">${vidInfo}</span>`;
    // Process button
    const procBtn = document.createElement('button');
    procBtn.className = 'seg-del'; // reuse small button style
    procBtn.title = 'Process all videos in this segment';
    procBtn.innerHTML = '&#x25B6;';
    procBtn.style.color = 'var(--accent, #2a7fc4)';
    procBtn.onclick = (e) => { e.stopPropagation(); reprocessSegment(seg); };
    const delBtn = document.createElement('button');
    delBtn.className = 'seg-del';
    delBtn.title = 'Delete segment';
    delBtn.innerHTML = '<svg class="icon-svg" style="width:10px;height:10px"><use href="#ico-x"/></svg>';
    delBtn.onclick = (e) => { e.stopPropagation(); deleteSegment(seg.id); };
    row.append(nameEl, rangeEl, procBtn, delBtn);
    list.appendChild(row);
  }
}

// ── Analysis tab ─────────────────────────────────────────────────────
function getAllSplits() {
  // Returns project-level segments with enriched data and ALL athlete info
  return state.segments.map(seg => {
    const athletes = getSegmentAthletes(seg);
    const overlapping = findOverlappingVideos(seg);
    const color = athletes.length > 0 ? athletes[0].color : '#f5a623';
    const athleteNames = athletes.map(a => a.name).filter(n => !n.startsWith('__'));
    return {
      ...seg,
      start_s: seg.tsStart,
      end_s: seg.tsEnd,
      duration: ((seg.tsEnd || 0) - (seg.tsStart || 0)).toFixed(1),
      color,
      athletes,
      athleteName: athleteNames.join(', ') || '--',
      videoCount: overlapping.length,
    };
  });
}

function renderAnalysisTab() {
  const splits = getAllSplits();
  const listEl = el('analysis-list');
  const emptyEl = el('analysis-empty');
  const reportBtn = el('btn-gen-report-all');
  const csvBtn = el('btn-download-report-csv');
  listEl.innerHTML = '';
  const readyIds = new Set(splits.filter(seg => getSegmentProcessingStatus(seg).ready).map(seg => String(seg.id)));
  sanitizeAnalysisSelection(readyIds);
  const selected = new Set((state.analysisSelected || []).map(String));

  if(!splits.length) {
    emptyEl.style.display = '';
    if(reportBtn) {
      reportBtn.disabled = true;
      reportBtn.title = 'Create segments first.';
    }
    if (csvBtn) {
      csvBtn.disabled = true;
      csvBtn.style.display = 'none';
      csvBtn.title = 'Create processed segments first.';
    }
    loadPdfReports();
    updateCurrentSegmentActions();
    return;
  }
  emptyEl.style.display = 'none';
  loadPdfReports();
  const statusById = new Map(splits.map(seg => [String(seg.id), getSegmentProcessingStatus(seg)]));
  const syncReportButton = () => {
    if (!reportBtn) return;
    reportBtn.disabled = selected.size === 0;
    reportBtn.title = selected.size === 0 ? 'Select at least one ready segment.' : 'Generate report';
    if (csvBtn) {
      const csvVisible = !!state.advancedMode && !!state.reportOptions?.downloadCsv;
      csvBtn.style.display = csvVisible ? '' : 'none';
      csvBtn.disabled = !csvVisible || selected.size === 0;
      csvBtn.title = selected.size === 0 ? 'Select at least one ready segment.' : 'Download boom/rudder time-series CSV';
    }
  };
  syncReportButton();

  const hdr = document.createElement('div');
  hdr.className = 'analysis-section-hdr';
  hdr.textContent = `Segments (${splits.length})`;
  listEl.appendChild(hdr);

  for(const s of splits) {
    const status = statusById.get(String(s.id)) || { key: 'unprocessed', label: 'Unprocessed', detail: 'Not processed', ready: false };
    const row = document.createElement('div');
    row.className = 'an-split';
    const include = document.createElement('input');
    include.type = 'checkbox';
    include.className = 'an-split-include';
    include.title = status.ready ? 'Include in report' : 'Finish processing before including this segment in a report';
    include.checked = selected.has(String(s.id));
    include.disabled = !status.ready;
    include.onchange = () => {
      const set = new Set((state.analysisSelected || []).map(String));
      if (include.checked) set.add(String(s.id));
      else set.delete(String(s.id));
      state.analysisSelected = [...set];
      saveAnalysisSelection();
      selected.clear();
      for (const id of state.analysisSelected) selected.add(String(id));
      syncReportButton();
    };

    // Show a color dot per athlete in the segment
    const dotWrap = document.createElement('span');
    dotWrap.style.display = 'flex';
    dotWrap.style.flexDirection = 'column';
    dotWrap.style.gap = '2px';
    dotWrap.style.alignItems = 'center';
    if (s.athletes && s.athletes.length > 0) {
      for (const a of s.athletes) {
        const dot = document.createElement('span');
        dot.className = 'an-split-color';
        dot.style.background = a.color;
        dot.title = a.name;
        dotWrap.appendChild(dot);
      }
    } else {
      const dot = document.createElement('span');
      dot.className = 'an-split-color';
      dot.style.background = s.color;
      dotWrap.appendChild(dot);
    }

    const info = document.createElement('div');
    info.className = 'an-split-info';
    const dur = parseFloat(s.duration);
    const athLabel = s.athleteName || '--';
    info.innerHTML = `
      <div class="an-split-name">${s.name}</div>
      <div class="an-split-meta">
        <span>${fmtClock(s.tsStart)} → ${fmtClock(s.tsEnd)}</span>
        <span>${dur.toFixed(1)}s</span>
        <span>${s.videoCount || 0} video${s.videoCount !== 1 ? 's' : ''}</span>
      </div>
      <div class="an-split-meta" style="margin-top:2px;">
        <span style="color:var(--muted);">${athLabel}</span>
      </div>
    `;

    const nameEl = info.querySelector('.an-split-name');
    if (nameEl) {
      const head = document.createElement('div');
      head.className = 'an-split-head';
      nameEl.replaceWith(head);
      head.appendChild(nameEl);
      const statusEl = document.createElement('span');
      statusEl.className = `seg-status-pill ${status.key}`;
      statusEl.textContent = status.label;
      head.appendChild(statusEl);
    }
    const metaRows = info.querySelectorAll('.an-split-meta');
    if (metaRows[0]?.firstElementChild) metaRows[0].firstElementChild.textContent = `${fmtClock(s.tsStart)} - ${fmtClock(s.tsEnd)}`;
    if (metaRows[1]) {
      const detail = document.createElement('span');
      detail.textContent = status.detail;
      metaRows[1].appendChild(detail);
    }

    const actions = document.createElement('div');
    actions.className = 'an-split-actions';

    const renameBtn = document.createElement('button');
    renameBtn.className = 'btn sm';
    renameBtn.textContent = 'Rename';
    renameBtn.onclick = () => openSegmentRenameModal(s.id);

    const procBtn = document.createElement('button');
    procBtn.className = 'btn sm';
    procBtn.textContent = status.key === 'ready' ? 'Reprocess' : 'Process';
    procBtn.disabled = status.key === 'processing';
    procBtn.onclick = () => reprocessSegment(s);

    const delBtn = document.createElement('button');
    delBtn.className = 'btn danger sm';
    delBtn.textContent = 'Delete';
    delBtn.onclick = () => deleteSegment(s.id);

    actions.append(renameBtn, procBtn, delBtn);
    row.append(include, dotWrap, info, actions);
    listEl.appendChild(row);
  }

  updateCurrentSegmentActions();
}

// ── PDF Report generation (browser-side) ─────────────────────────────

function setReportBlockingOverlay(open, pct = 0) {
  const overlay = el('report-blocking-overlay');
  const number = el('report-blocking-number');
  state.reportOverlayOpen = !!open;
  if (!overlay || !number) return;
  overlay.classList.toggle('open', !!open);
  if (open) {
    const displayPct = Math.max(0, Math.min(100, Math.round(Number(pct) || 0)));
    number.textContent = `${displayPct}%`;
    number.classList.toggle('done', displayPct >= 100);
  } else {
    number.textContent = '';
    number.classList.remove('done');
  }
}

async function generateReportAll() {
  if(!state.projectId) return;
  if(!state.segments || !state.segments.length) { alert('No segments available.'); return; }
  const readyIds = new Set((state.segments || []).filter(seg => getSegmentProcessingStatus(seg).ready).map(seg => String(seg.id)));
  sanitizeAnalysisSelection(readyIds);
  const segmentIds = (state.analysisSelected || []).map(String);
  if(!segmentIds.length) { alert('Select at least one segment for the report.'); return; }
  const unfinished = segmentIds
    .map(id => getSegmentById(id))
    .filter(Boolean)
    .map(seg => ({ seg, status: getSegmentProcessingStatus(seg) }))
    .filter(({ status }) => !status.ready);
  if (unfinished.length) {
    alert(`Finish processing before generating a report:\n${unfinished.map(({ seg, status }) => `- ${seg.name}: ${status.label}`).join('\n')}`);
    renderAnalysisTab();
    return;
  }

  const btn = el('btn-gen-report-all');
  if(btn) { btn.disabled = true; btn.textContent = 'Generating...'; }
  updateProgressRing(5);
  setReportBlockingOverlay(true, 5);

  try {
    const setReportProgress = (pct) => {
      const clamped = Math.max(0, Math.min(100, Math.round(pct)));
      updateProgressRing(clamped);
      setReportBlockingOverlay(true, clamped);
    };
    const yieldUi = () => new Promise(resolve => setTimeout(resolve, 0));
    setReportProgress(8);
    await ensureWindEstimatesReady();

    // 1. Build report data from IndexedDB + OPFS
    const polarPlotsEnabled = !!state.reportOptions?.polarPlots;
    const maneuverAnalysisEnabled = !!state.reportOptions?.maneuverAnalysis;
    const reportData = await buildReportData(
      state.projectId,
      segmentIds,
      (msg, pct) => {
        setReportProgress(8 + pct * 52);
      },
      {
        includeDensityImages: false,
        includeLegacyVisuals: false,
        polarPlots: polarPlotsEnabled,
        maneuverAnalysis: maneuverAnalysisEnabled,
        wind: buildReportWindContext(),
      },
    );
    setReportProgress(60);
    await yieldUi();

    // 2. Generate PDF
    const blob = await generatePdf(reportData, {
      includeSummaryStats: state.reportOptions?.summaryStats !== false,
      includeHistograms: state.reportOptions?.histograms !== false,
      includeHeatmaps: state.reportOptions?.heatmaps !== false,
      includeBoomAngle: state.reportOptions?.boomAngle !== false,
      polarPlots: polarPlotsEnabled,
      maneuverAnalysis: maneuverAnalysisEnabled,
      onProgress: (msg, pct) => {
        setReportProgress(60 + pct * 34);
      },
    });

    setReportProgress(95);
    await yieldUi();

    // 3. Persist in storage + history (best-effort on Safari/iPad)
    const createdAt = Date.now();
    const reportName = buildReportDisplayName(segmentIds);
    const stamp = new Date(createdAt).toISOString().replace(/[:.]/g, '-');
    const fileStem = sanitizeFilenamePart(reportName) || 'Report';
    const storedFilename = `${stamp}_${fileStem}.pdf`;
    let reportStored = false;
    try {
      if (typeof Storage.getDebugInfo === 'function') {
        const info = await Storage.getDebugInfo();
        const usage = Number(info?.usage);
        const quota = Number(info?.quota);
        const blobSize = Number(blob?.size || 0);
        if (Number.isFinite(usage) && Number.isFinite(quota) && quota > 0 && blobSize > 0) {
          const freeBytes = Math.max(0, quota - usage);
          const neededBytes = Math.ceil(blobSize * 1.1); // small safety margin
          if (freeBytes < neededBytes) {
            throw new Error(`Report storage skipped: low browser storage (${Math.round(freeBytes / 1048576)}MB free)`);
          }
        }
      }
      await Storage.writeFile([...REPORT_STORAGE_DIR, state.projectId], storedFilename, blob);

      const history = loadReportHistory(state.projectId);
      history.unshift({
        id: DB.uuid(),
        filename: storedFilename,
        name: reportName,
        createdAt,
        segmentIds,
      });
      saveReportHistory(state.projectId, history);
      reportStored = true;
    } catch (storageErr) {
      console.warn('[Report] Could not persist report in browser storage/history:', storageErr);
      await logStorageDebug('generateReportAll.persistReport');
    }

    // 4. Trigger download
    downloadBlob(blob, buildReportFileName(segmentIds, createdAt));
    if (!reportStored) {
      alert('Report generated and downloaded, but could not be saved in report history on this browser.');
    }

    setReportProgress(100);
    setTimeout(() => updateProgressRing(0), 2000);

    // Refresh reports list
    loadPdfReports();

  } catch(e) {
    console.error('Report generation failed:', e);
    alert('Report generation failed: ' + e.message);
    updateProgressRing(0);
  } finally {
    setReportBlockingOverlay(false, 0);
    if(btn) {
      btn.disabled = false;
      btn.innerHTML = '<svg class="icon-svg" style="width:18px;height:18px"><use href="#ico-activity"/></svg> Generate Report';
    }
  }
}

function buildSegmentBoomRudderCsv(reportData) {
  const headers = [
    'segment_id',
    'segment_name',
    'athlete_id',
    'athlete_name',
    'time_s',
    'abs_ts_s',
    'iso_time',
    'rudder_angle_deg',
    'boom_angle_deg',
    'heel_moment_nm',
    'pitch_moment_nm',
    'trunk_angle_deg',
  ];
  const rows = [];
  for (const seg of (reportData?.segments || [])) {
    const startS = Number(seg?.start_s);
    const timelineRows = makeMetricTimelineRows(seg?.rudder_timeline, seg?.boom_timeline, [
      { key: 'heel_moment', rows: seg?.moment_timeline },
      { key: 'pitch_moment', rows: seg?.pitch_moment_timeline },
      { key: 'trunk_angle', rows: seg?.trunk_angle_timeline },
    ]);
    for (const point of timelineRows) {
      const absTs = Number.isFinite(startS) ? startS + Number(point.t) : null;
      rows.push({
        segment_id: seg?.split_id || '',
        segment_name: seg?.name || '',
        athlete_id: seg?.athlete_id || '',
        athlete_name: seg?.athlete_name || '',
        time_s: csvNumber(point.t, 3),
        abs_ts_s: csvNumber(absTs, 3),
        iso_time: epochSecondsToIso(absTs),
        rudder_angle_deg: csvNumber(point.rudder, 6),
        boom_angle_deg: csvNumber(point.boom, 6),
        heel_moment_nm: csvNumber(point.heel_moment, 3),
        pitch_moment_nm: csvNumber(point.pitch_moment, 3),
        trunk_angle_deg: csvNumber(point.trunk_angle, 3),
      });
    }
  }
  return { csv: makeCsv(headers, rows), rowCount: rows.length };
}

function getSelectedReadyReportSegmentIds() {
  const readyIds = new Set((state.segments || []).filter(seg => getSegmentProcessingStatus(seg).ready).map(seg => String(seg.id)));
  sanitizeAnalysisSelection(readyIds);
  return (state.analysisSelected || []).map(String).filter(id => readyIds.has(id));
}

async function downloadSelectedReportCsv() {
  if (!state.projectId) return;
  if (!state.advancedMode || !state.reportOptions?.downloadCsv) {
    alert('Enable Download CSV in Advanced Features first.');
    return;
  }
  const segmentIds = getSelectedReadyReportSegmentIds();
  if (!segmentIds.length) {
    alert('Select at least one processed segment for CSV export.');
    renderAnalysisTab();
    return;
  }

  const btn = el('btn-download-report-csv');
  if (btn) { btn.disabled = true; btn.textContent = 'Preparing CSV...'; }
  updateProgressRing(5);
  setReportBlockingOverlay(true, 5);
  try {
    await ensureWindEstimatesReady();
    const reportData = await buildReportData(
      state.projectId,
      segmentIds,
      (msg, pct) => {
        const clamped = Math.max(5, Math.min(95, Math.round(5 + pct * 90)));
        updateProgressRing(clamped);
        setReportBlockingOverlay(true, clamped);
      },
      {
        includeDensityImages: false,
        includeLegacyVisuals: false,
        polarPlots: false,
        maneuverAnalysis: false,
        timelineMaxPoints: null,
        wind: buildReportWindContext(),
      },
    );
    const { csv, rowCount } = buildSegmentBoomRudderCsv(reportData);
    if (!rowCount) {
      alert('No boom or rudder timeline data was found for the selected segment(s).');
      return;
    }
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    downloadBlob(blob, buildReportCsvFileName(segmentIds, Date.now()));
    updateProgressRing(100);
    setTimeout(() => updateProgressRing(0), 1500);
  } catch (err) {
    console.error('CSV export failed:', err);
    alert('CSV export failed: ' + (err?.message || err));
    updateProgressRing(0);
  } finally {
    setReportBlockingOverlay(false, 0);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<svg class="icon-svg"><use href="#ico-download"/></svg> Download CSV';
    }
    renderAnalysisTab();
  }
}

async function loadPdfReports() {
  if(!state.projectId) return;
  const reportsEl = el('analysis-reports');
  const bodyEl = el('analysis-reports-body');
  if(!reportsEl || !bodyEl) return;
  const hasSegments = state.segments && state.segments.length > 0;
  const history = loadReportHistory(state.projectId);

  if (!history.length) {
    if(hasSegments) {
      bodyEl.innerHTML = `<div style="padding:12px 18px;font-size:13px;">
        <p style="margin:0 0 8px 0;color:var(--text);">Generate a report to store it in your history.</p>
        <p style="margin:0;color:var(--muted);font-size:12px;">Reports are named automatically based on selected segments.</p>
      </div>`;
    } else {
      bodyEl.innerHTML = '<div style="padding:12px 18px;color:var(--muted);font-size:13px;">Create segments first to generate reports.</div>';
    }
    return;
  }

  bodyEl.innerHTML = '';
  for (const rep of history) {
    const row = document.createElement('div');
    row.className = 'report-row';
    row.innerHTML = `
      <div class="report-row-main">
        <div class="report-row-name">${rep.name || 'Report'}</div>
      </div>
      <div class="report-row-actions">
        <button class="btn sm" data-action="download"><svg class="icon-svg"><use href="#ico-download"/></svg> Download</button>
        <button class="btn sm danger" data-action="delete"><svg class="icon-svg"><use href="#ico-trash"/></svg> Delete</button>
      </div>
    `;

    const dlBtn = row.querySelector('button[data-action="download"]');
    dlBtn.onclick = async () => {
      const blob = await Storage.readFileBlob([...REPORT_STORAGE_DIR, state.projectId], rep.filename);
      if (!blob) {
        alert('Report file is no longer available.');
        return;
      }
      const createdAt = rep.createdAt || Date.now();
      const fallbackStem = sanitizeFilenamePart(rep.name || 'report') || 'report';
      const fallbackDate = new Date(createdAt).toISOString().split('T')[0];
      const downloadName = Array.isArray(rep.segmentIds) && rep.segmentIds.length
        ? buildReportFileName(rep.segmentIds, createdAt)
        : `${fallbackStem}_${fallbackDate}_report.pdf`;
      downloadBlob(blob, downloadName);
    };

    const delBtn = row.querySelector('button[data-action="delete"]');
    delBtn.onclick = async () => {
      await Storage.deleteFile([...REPORT_STORAGE_DIR, state.projectId], rep.filename);
      const nextHistory = loadReportHistory(state.projectId).filter(r => r.id !== rep.id);
      saveReportHistory(state.projectId, nextHistory);
      loadPdfReports();
    };

    bodyEl.appendChild(row);
  }
}

// ── STL resize handle ────────────────────────────────────────────────
function initStlResize() {
  const handle = el('stl-resize-handle');
  if(!handle) return;
  const left = el('panel-left');
  let dragging = false;
  handle.addEventListener('pointerdown', e => {
    dragging = true;
    handle.setPointerCapture(e.pointerId);
    handle.classList.add('dragging');
  });
  handle.addEventListener('pointermove', e => {
    if(!dragging) return;
    const rect = left.getBoundingClientRect();
    const controlsH = el('map-controls').getBoundingClientRect().height;
    const total = rect.height - controlsH - handle.offsetHeight;
    const mapH = Math.max(80, Math.min(total - 80, e.clientY - rect.top));
    const stlH = total - mapH;
    el('map').style.flex = `0 0 ${mapH}px`;
    el('stl-inline').style.flex = `0 0 ${stlH}px`;
    if(state.map) state.map.invalidateSize();
    window.dispatchEvent(new Event('resize'));
  });
  handle.addEventListener('pointerup', () => { dragging = false; handle.classList.remove('dragging'); });
}

// ── Resizable divider ──────────────────────────────────────────────────
function initDivider() {
  const div = el('divider');
  const left = el('panel-left');
  const layout = el('layout');
  if(!div || !left || !layout) return;

  let dragging = false;
  let activePointerId = null;
  let startX = 0;
  let startW = 0;

  const endDrag = () => {
    if(!dragging) return;
    dragging = false;
    activePointerId = null;
    div.classList.remove('dragging');
    document.body.style.userSelect = '';
  };

  div.addEventListener('pointerdown', e => {
    if(e.button !== undefined && e.button !== 0) return;
    dragging = true;
    activePointerId = e.pointerId;
    startX = e.clientX;
    startW = left.offsetWidth;
    div.classList.add('dragging');
    div.setPointerCapture(e.pointerId);
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  div.addEventListener('pointermove', e => {
    if(!dragging || e.pointerId !== activePointerId) return;
    const minW = layout.classList.contains('phone-playback-active') ? PHONE_PLAYBACK_LEFT_MIN_WIDTH : 200;
    const outerDividerW = div.offsetWidth || 0;
    const rightMinW = layout.classList.contains('phone-playback-active') ? getPhonePlaybackRightMinWidth() : 200;
    const maxW = Math.max(minW, layout.offsetWidth - outerDividerW - rightMinW);
    const w = Math.max(minW, Math.min(startW + (e.clientX - startX), maxW));
    left.style.width = w + 'px';
    left.style.flex = 'none';
    if (layout.classList.contains('phone-playback-active')) {
      state.phonePlayback.savedActiveLeftWidthPx = w;
      const shell = el('phone-playback-shell');
      const panelRight = el('panel-right');
      const playbackDivider = el('phone-playback-divider');
      const lockedGridW = Number(state.phonePlayback?.savedVideoGridWidthPx);
      if (shell && panelRight && Number.isFinite(lockedGridW)) {
        const dividerW = playbackDivider?.offsetWidth || 0;
        const totalW = panelRight.clientWidth - dividerW;
        const minShellW = 180;
        const maxShellW = Math.max(minShellW, totalW - 240);
        const clampedGridW = Math.max(240, Math.min(lockedGridW, totalW - minShellW));
        if (clampedGridW !== lockedGridW) setPhonePlaybackGridWidth(clampedGridW);
        const nextShellW = Math.max(minShellW, Math.min(totalW - clampedGridW, maxShellW));
        state.phonePlayback.savedShellWidthPx = nextShellW;
        shell.style.width = `${nextShellW}px`;
        shell.style.flex = `0 0 ${nextShellW}px`;
      }
    }
    if(state.map) state.map.invalidateSize();
    // Keep STL canvas sizing in sync while dragging divider.
    window.dispatchEvent(new Event('resize'));
  });

  div.addEventListener('pointerup', endDrag);
  div.addEventListener('pointercancel', endDrag);
  div.addEventListener('lostpointercapture', endDrag);
}

function initPhonePlaybackDivider() {
  const div = el('phone-playback-divider');
  const shell = el('phone-playback-shell');
  const panelRight = el('panel-right');
  const layout = el('layout');
  if (!div || !shell || !panelRight || !layout) return;

  let dragging = false;
  let activePointerId = null;
  let startX = 0;
  let startW = 0;

  const endDrag = () => {
    if (!dragging) return;
    dragging = false;
    activePointerId = null;
    div.classList.remove('dragging');
    document.body.style.userSelect = '';
    syncPhonePlaybackMapOffset();
    scheduleAnalysisMapResize();
  };

  div.addEventListener('pointerdown', e => {
    if (e.button !== undefined && e.button !== 0) return;
    if (!layout.classList.contains('phone-playback-active')) return;
    if (window.matchMedia('(max-width: 760px)').matches) return;
    dragging = true;
    activePointerId = e.pointerId;
    startX = e.clientX;
    startW = shell.getBoundingClientRect().width;
    div.classList.add('dragging');
    div.setPointerCapture(e.pointerId);
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  div.addEventListener('pointermove', e => {
    if (!dragging || e.pointerId !== activePointerId) return;
    const dividerW = div.offsetWidth || 0;
    const totalW = panelRight.clientWidth - dividerW;
    const minW = 180;
    const maxW = Math.max(minW, totalW - 240);
    const nextW = Math.max(minW, Math.min(startW + (e.clientX - startX), maxW));
    state.phonePlayback.savedShellWidthPx = nextW;
    shell.style.width = `${nextW}px`;
    shell.style.flex = `0 0 ${nextW}px`;
    setPhonePlaybackGridWidth(totalW - nextW);
    syncPhonePlaybackMapOffset();
    window.dispatchEvent(new Event('resize'));
  });

  div.addEventListener('pointerup', endDrag);
  div.addEventListener('pointercancel', endDrag);
  div.addEventListener('lostpointercapture', endDrag);
}

// ── Navigation (view switching) ─────────────────────────────────────────
function switchView(viewId) {
  if (viewId !== 'view-analysis') closeInlineHeatmapPlotOverlay();
  document.querySelectorAll('.view-panel').forEach(v => v.classList.remove('active-view'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  const view = el(viewId);
  if (view) view.classList.add('active-view');
  const btn = document.querySelector(`.nav-btn[data-view="${viewId}"]`);
  if (btn) btn.classList.add('active');
  // Refresh content when switching views
  if (viewId === 'view-analysis') {
    if (state.map) state.map.invalidateSize();
    drawSogCanvas();
    renderWindMapControl();
  }
  if (viewId === 'view-maneuvers') {
    initManeuverMap();
    scheduleManeuverMapRefresh();
    requestAnimationFrame(() => {
      renderManeuverPanel();
      void renderManeuverWorkspace();
    });
  }
  if (viewId === 'view-report') renderAnalysisTab();
  if (viewId === 'view-upload') { wizGoTo(state.wizardStep); }
  const segmentBtn = el('btn-new-segment');
  if (segmentBtn) segmentBtn.disabled = viewId !== 'view-analysis';
  updateCurrentSegmentActions();
}

function initNavigation() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  function positionProjectMenu() {
    const menu = el('project-menu');
    const btn = el('btn-hamburger');
    if(!menu || !btn) return;

    const compact = window.matchMedia('(max-width: 1180px)').matches;
    menu.classList.toggle('floating', compact);
    if(!compact) {
      menu.style.left = '';
      menu.style.top = '';
      return;
    }

    const rect = btn.getBoundingClientRect();
    const margin = 8;
    const menuWidth = Math.min(Math.max(260, menu.offsetWidth || 260), Math.max(260, window.innerWidth - margin * 2));
    const left = Math.max(margin, Math.min(rect.left, window.innerWidth - menuWidth - margin));
    const top = Math.min(window.innerHeight - 12, rect.bottom + 4);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
  }

  // Hamburger project menu
  el('btn-hamburger').onclick = () => {
    const menu = el('project-menu');
    const opening = !menu.classList.contains('open');
    menu.classList.toggle('open');
    if(opening) positionProjectMenu();
  };
  document.addEventListener('click', e => {
    const menu = el('project-menu');
    if (menu.classList.contains('open') && !el('project-wrap').contains(e.target)) menu.classList.remove('open');
  });
  window.addEventListener('resize', () => {
    const menu = el('project-menu');
    if(menu?.classList.contains('open')) positionProjectMenu();
  });
  el('topbar')?.addEventListener('scroll', () => {
    const menu = el('project-menu');
    if(menu?.classList.contains('open')) positionProjectMenu();
  }, { passive: true });
}

function renderProjectMenu() {
  const list = el('project-menu-list');
  list.innerHTML = '';
  for (const p of state.projects) {
    const opt = document.createElement('div');
    opt.className = 'proj-opt' + (p.id === state.projectId ? ' selected' : '');
    opt.textContent = p.name;
    opt.onclick = async () => {
      el('project-menu').classList.remove('open');
      await selectProject(p.id);
    };
    list.appendChild(opt);
  }
}

function updateProjectLabel() {
  const proj = state.projects.find(p => p.id === state.projectId);
  el('project-label').textContent = proj ? proj.name : 'No project';
  el('btn-del-proj').disabled = !state.projectId;
}

function renderProfileStep() {
  const list = el('profile-list');
  if (!list) return;
  list.innerHTML = '';

  if (!state.projects.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.style.padding = '20px';
    empty.innerHTML = '<div>No profiles yet.</div><div style="font-size:13px;color:var(--muted);">Create your first profile below.</div>';
    list.appendChild(empty);
    return;
  }

  for (const p of state.projects) {
    const row = document.createElement('div');
    row.className = 'profile-item' + (p.id === state.projectId ? ' active' : '');
    const created = p.created_at ? new Date(p.created_at).toLocaleDateString() : '--';
    row.innerHTML = `
      <div style="min-width:0;">
        <div class="profile-item-name">${p.name}</div>
        <div class="profile-item-date">Created ${created}</div>
      </div>
      <button class="btn sm">${p.id === state.projectId ? 'Selected' : 'Select'}</button>
    `;
    const btn = row.querySelector('button');
    btn.disabled = p.id === state.projectId;
    btn.onclick = async (e) => {
      e.stopPropagation();
      await selectProject(p.id);
      wizGoTo(Math.max(2, state.wizardStep || 2));
    };
    row.onclick = () => btn.click();
    list.appendChild(row);
  }
}

function sanitizeAnalysisSelection(allowedIds = null) {
  const validIds = allowedIds instanceof Set
    ? new Set([...allowedIds].map(String))
    : new Set((state.segments || []).map(s => String(s.id)));
  const incoming = Array.isArray(state.analysisSelected) ? state.analysisSelected.map(String) : [];
  const filtered = incoming.filter(id => validIds.has(id));
  const hasStored = !!(state.projectId && localStorage.getItem('trollfish_analysisSelected_' + state.projectId) != null);
  if (filtered.length === 0 && validIds.size > 0 && !hasStored) {
    state.analysisSelected = [...validIds];
  } else {
    state.analysisSelected = filtered;
  }
  saveAnalysisSelection();
}

function saveAnalysisSelection() {
  if (!state.projectId) return;
  try {
    localStorage.setItem('trollfish_analysisSelected_' + state.projectId, JSON.stringify(state.analysisSelected || []));
  } catch {}
}

function loadAdvancedModeSetting() {
  state.advancedMode = !!loadJsonLocal(ADVANCED_MODE_KEY, false);
}

function normalizeAdvancedFeatures(raw = {}) {
  return {
    maneuversTab: !!raw?.maneuversTab,
    hull3d: !!raw?.hull3d,
    windPanel: !!raw?.windPanel,
    boomPredictions: raw?.boomPredictions !== false,
    rudderPredictions: raw?.rudderPredictions !== false,
  };
}

function loadAdvancedFeaturesSetting() {
  state.advancedFeatures = normalizeAdvancedFeatures(loadJsonLocal(ADVANCED_FEATURES_KEY, DEFAULT_ADVANCED_FEATURES));
}

function saveAdvancedFeaturesSetting() {
  saveJsonLocal(ADVANCED_FEATURES_KEY, normalizeAdvancedFeatures(state.advancedFeatures || DEFAULT_ADVANCED_FEATURES));
}

function setAdvancedFeature(key, enabled) {
  state.advancedFeatures = normalizeAdvancedFeatures({
    ...(state.advancedFeatures || {}),
    [key]: !!enabled,
  });
  saveAdvancedFeaturesSetting();
  applyAdvancedModeVisibility();
}

function getHiddenVideoSlotsKey(projectId = state.projectId) {
  return `${HIDDEN_VIDEO_SLOTS_KEY_PREFIX}${projectId || ''}`;
}

function loadHiddenVideoSlots(projectId = state.projectId) {
  const raw = loadJsonLocal(getHiddenVideoSlotsKey(projectId), {});
  return raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
}

function saveHiddenVideoSlots() {
  if (!state.projectId) return;
  saveJsonLocal(getHiddenVideoSlotsKey(state.projectId), state.hiddenVideoSlots || {});
}

function getVideoSlotVisibilityKey(slot) {
  const athleteId = normalizeAthleteId(slot?.athleteId);
  if (athleteId) return `ath:${athleteId}`;
  const videoId = String(slot?.videos?.[0]?.id || slot?.currentFileId || slot?.name || '').trim();
  return videoId ? `video:${videoId}` : null;
}

function isVideoSlotHidden(slot) {
  const key = getVideoSlotVisibilityKey(slot);
  return !!(key && state.hiddenVideoSlots?.[key]);
}

function setVideoSlotHidden(slot, hidden) {
  const key = getVideoSlotVisibilityKey(slot);
  if (!key) return;
  if (hidden) state.hiddenVideoSlots[key] = true;
  else delete state.hiddenVideoSlots[key];
  saveHiddenVideoSlots();
  applyVideoSlotVisibility();
  drawSogCanvas();
}

function showAllVideoSlots() {
  state.hiddenVideoSlots = {};
  saveHiddenVideoSlots();
  applyVideoSlotVisibility();
  if (Number.isFinite(Number(state.tl?.currentTs))) tlSeekTo(state.tl.currentTs);
  drawSogCanvas();
}

function normalizeVideoLayout(value) {
  return ['stack', 'row', '2plus1', 'auto'].includes(String(value || '')) ? String(value) : 'auto';
}

function loadVideoLayoutSetting() {
  state.videoLayout = normalizeVideoLayout(loadJsonLocal(VIDEO_LAYOUT_KEY, 'auto'));
  renderVideoLayoutPopover();
  syncVideoLayoutButton();
}

function setVideoLayout(value) {
  state.videoLayout = normalizeVideoLayout(value);
  saveJsonLocal(VIDEO_LAYOUT_KEY, state.videoLayout);
  applyVideoLayout();
  syncVideoLayoutButton();
  renderVideoLayoutPopover();
}

function setTemporaryVideoLayout(value) {
  state.videoLayout = normalizeVideoLayout(value);
  applyVideoLayout();
  syncVideoLayoutButton();
  renderVideoLayoutPopover();
}

function getVisibleTimelineVideoPaneCount() {
  return (state.tl?.athleteSlots || []).filter(slot => {
    const pane = slot?.paneEl;
    if (!pane || pane.style.display === 'none') return false;
    if (isVideoSlotHidden(slot)) return false;
    return pane.dataset.hasVideoAtPlayhead === '1';
  }).length;
}

function applyHeatmapVideoLayoutOverride(nextVisible) {
  if (nextVisible) {
    const shouldOverride = normalizeVideoLayout(state.videoLayout) === 'row'
      && getVisibleTimelineVideoPaneCount() >= 3;
    if (shouldOverride && !state.inlineHeatmaps.previousVideoLayout) {
      state.inlineHeatmaps.previousVideoLayout = state.videoLayout;
      setTemporaryVideoLayout('2plus1');
    }
    return;
  }

  const previous = state.inlineHeatmaps?.previousVideoLayout;
  state.inlineHeatmaps.previousVideoLayout = null;
  if (previous) setTemporaryVideoLayout(previous);
}

function applyVideoLayout() {
  const grid = el('video-grid');
  if (!grid) return;
  grid.classList.remove('layout-stack', 'layout-row', 'layout-2plus1', 'layout-auto');
  grid.classList.add(`layout-${normalizeVideoLayout(state.videoLayout)}`);
  applyVideoPaneGridPositions();
}

function applyVideoPaneGridPositions() {
  const grid = el('video-grid');
  if (!grid) return;
  const panes = Array.from(grid.querySelectorAll('.video-pane'));
  for (const pane of panes) {
    pane.style.gridColumn = '';
    pane.style.gridRow = '';
  }
  if (normalizeVideoLayout(state.videoLayout) !== '2plus1') return;

  const visible = panes.filter(pane => pane.style.display !== 'none');
  if (visible.length === 1) {
    visible[0].style.gridColumn = '1 / span 2';
    visible[0].style.gridRow = '1 / span 2';
    return;
  }
  if (visible.length === 2) {
    visible[0].style.gridColumn = '1';
    visible[0].style.gridRow = '1 / span 2';
    visible[1].style.gridColumn = '2';
    visible[1].style.gridRow = '1 / span 2';
    return;
  }
  if (visible.length >= 3) {
    visible[0].style.gridColumn = '1';
    visible[0].style.gridRow = '1';
    visible[1].style.gridColumn = '1';
    visible[1].style.gridRow = '2';
    visible[2].style.gridColumn = '2';
    visible[2].style.gridRow = '1 / span 2';
  }
}

function getVideoLayoutLabel(value = state.videoLayout) {
  return {
    stack: 'Layout: Stack',
    row: 'Layout: Side by side',
    '2plus1': 'Layout: 2 + 1',
    auto: 'Layout: Auto grid',
  }[normalizeVideoLayout(value)] || 'Layout';
}

function syncVideoLayoutButton() {
  const btn = el('btn-video-layout');
  if (btn) btn.textContent = getVideoLayoutLabel();
  const wrap = el('video-layout-wrap');
  const hasMultipleSlots = (state.tl?.athleteSlots || []).length > 1;
  if (wrap) wrap.style.display = hasMultipleSlots ? '' : 'none';
  if (!hasMultipleSlots) closeVideoLayoutPopover();
  document.querySelectorAll('.layout-choice').forEach(choice => {
    choice.classList.toggle('active', normalizeVideoLayout(choice.dataset.layout) === normalizeVideoLayout(state.videoLayout));
  });
}

function getVideoSlotPaneByKey(slotKey) {
  return Array.from(document.querySelectorAll('#video-grid .video-pane'))
    .find(pane => pane.dataset.slotKey === slotKey) || null;
}

function moveVideoSlotPaneToIndex(slotKey, index) {
  const grid = el('video-grid');
  const pane = getVideoSlotPaneByKey(slotKey);
  if (!grid || !pane) return;
  const wasHidden = !!state.hiddenVideoSlots?.[slotKey];
  if (wasHidden) {
    delete state.hiddenVideoSlots[slotKey];
    saveHiddenVideoSlots();
  }
  const panes = Array.from(grid.querySelectorAll('.video-pane')).filter(item => item !== pane);
  const targetIndex = Math.max(0, Math.min(Number(index) || 0, panes.length));
  const before = panes[targetIndex] || null;
  grid.insertBefore(pane, before);
  if (wasHidden) {
    applyVideoSlotVisibility();
    if (Number.isFinite(Number(state.tl?.currentTs))) tlSeekTo(state.tl.currentTs);
    drawSogCanvas();
  }
  applyVideoPaneGridPositions();
}

function labelVideoLayoutPreviewCells(preview, layout) {
  if (!preview) return;
  const slots = state.tl?.athleteSlots || [];
  preview.innerHTML = '';
  const count = layout === 'auto' ? 4 : 3;
  const paneOrder = Array.from(document.querySelectorAll('#video-grid .video-pane')).map(p => p.dataset.slotKey);
  const orderedSlots = [...slots].sort((a, b) => {
    const ai = paneOrder.indexOf(getVideoSlotVisibilityKey(a));
    const bi = paneOrder.indexOf(getVideoSlotVisibilityKey(b));
    return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi);
  });
  for (let i = 0; i < count; i++) {
    const cell = document.createElement('span');
    cell.className = 'layout-drop-cell';
    cell.dataset.layout = layout;
    cell.dataset.slotIndex = String(i);
    const slot = orderedSlots[i];
    cell.textContent = slot?.name || `Slot ${i + 1}`;
    if (slot?.color) cell.style.borderColor = rgbaFromHex(slot.color, 0.45);
    cell.addEventListener('dragover', e => {
      e.preventDefault();
      cell.classList.add('drag-over');
      try { e.dataTransfer.dropEffect = 'move'; } catch {}
    });
    cell.addEventListener('dragleave', () => cell.classList.remove('drag-over'));
    cell.addEventListener('drop', e => {
      e.preventDefault();
      cell.classList.remove('drag-over');
      const slotKey = e.dataTransfer?.getData('text/plain') || '';
      if (!slotKey) return;
      setVideoLayout(layout);
      moveVideoSlotPaneToIndex(slotKey, i);
      renderVideoLayoutPopover();
    });
    preview.appendChild(cell);
  }
}

function renderVideoLayoutPopover() {
  document.querySelectorAll('.layout-choice').forEach(choice => {
    const layout = normalizeVideoLayout(choice.dataset.layout);
    const preview = choice.querySelector('.snap-preview');
    labelVideoLayoutPreviewCells(preview, layout);
    choice.classList.toggle('active', layout === normalizeVideoLayout(state.videoLayout));
  });
  const tray = el('video-layout-slot-tray');
  if (!tray) return;
  tray.innerHTML = '';
  for (const slot of state.tl?.athleteSlots || []) {
    const key = getVideoSlotVisibilityKey(slot);
    if (!key) continue;
    const chip = document.createElement('span');
    chip.className = 'video-layout-chip';
    chip.draggable = true;
    chip.dataset.slotKey = key;
    chip.title = 'Drag into a layout slot';
    const dot = document.createElement('span');
    dot.className = 'video-layout-chip-dot';
    dot.style.background = slot.color || '#8b97ad';
    const name = document.createElement('span');
    name.className = 'video-layout-chip-name';
    name.textContent = slot.name || 'Athlete';
    chip.append(dot, name);
    chip.addEventListener('dragstart', e => {
      try {
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', key);
      } catch {}
    });
    tray.appendChild(chip);
  }
}

function closeVideoLayoutPopover() {
  el('video-layout-popover')?.classList.remove('open');
}

function toggleVideoLayoutPopover() {
  const pop = el('video-layout-popover');
  if (!pop) return;
  renderVideoLayoutPopover();
  pop.classList.toggle('open');
}

function initVideoPaneDragReorder(pane) {
  const grid = el('video-grid');
  if (!pane || !grid) return;
  pane.addEventListener('dragstart', e => {
    pane.classList.add('dragging');
    try {
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', pane.dataset.slotKey || '');
    } catch {}
  });
  pane.addEventListener('dragend', () => {
    pane.classList.remove('dragging');
    document.querySelectorAll('.video-pane.drag-over').forEach(elm => elm.classList.remove('drag-over'));
    applyVideoPaneGridPositions();
    renderVideoLayoutPopover();
  });
  pane.addEventListener('dragover', e => {
    e.preventDefault();
    const dragging = grid.querySelector('.video-pane.dragging');
    if (!dragging || dragging === pane) return;
    pane.classList.add('drag-over');
    const rect = pane.getBoundingClientRect();
    const before = state.videoLayout === 'stack'
      ? e.clientY < rect.top + rect.height / 2
      : e.clientX < rect.left + rect.width / 2;
    grid.insertBefore(dragging, before ? pane : pane.nextSibling);
  });
  pane.addEventListener('dragleave', () => pane.classList.remove('drag-over'));
  pane.addEventListener('drop', e => {
    e.preventDefault();
    pane.classList.remove('drag-over');
  });
}

function applyVideoSlotVisibility() {
  for (const slot of state.tl?.athleteSlots || []) {
    if (!slot?.paneEl) continue;
    const hidden = isVideoSlotHidden(slot);
    slot.paneEl.dataset.hiddenByUser = hidden ? '1' : '';
    if (hidden) {
      slot.paneEl.style.display = 'none';
      stopSlotPlayback(slot, false);
    } else if (slot.paneEl.dataset.hasVideoAtPlayhead === '1') {
      slot.paneEl.style.display = 'flex';
    }
  }
  const anyVisible = (state.tl?.athleteSlots || []).some(s => s.paneEl && s.paneEl.style.display !== 'none');
  showNoVideo(!anyVisible);
  syncInlineHeatmapPanelLayout();
}

function loadReportOptionsSetting() {
  const raw = loadJsonLocal(REPORT_OPTIONS_KEY, {});
  state.reportOptions = {
    summaryStats: raw?.summaryStats == null ? true : !!raw.summaryStats,
    histograms: raw?.histograms == null ? true : !!raw.histograms,
    heatmaps: raw?.heatmaps == null ? true : !!raw.heatmaps,
    boomAngle: raw?.boomAngle == null ? true : !!raw.boomAngle,
    polarPlots: !!raw?.polarPlots,
    maneuverAnalysis: !!(raw?.maneuverAnalysis ?? raw?.tackAnalysis),
    downloadCsv: !!raw?.downloadCsv,
  };
  const polarToggle = el('report-opt-polar-plots');
  if (polarToggle) polarToggle.checked = !!state.reportOptions.polarPlots;
  const tackToggle = el('report-opt-tack-analysis');
  if (tackToggle) tackToggle.checked = !!state.reportOptions.maneuverAnalysis;
  const csvToggle = el('report-opt-download-csv');
  if (csvToggle) csvToggle.checked = !!state.reportOptions.downloadCsv;
  syncReportMenuToggles();
}

function saveReportOptionsSetting() {
  saveJsonLocal(REPORT_OPTIONS_KEY, {
    summaryStats: state.reportOptions?.summaryStats !== false,
    histograms: state.reportOptions?.histograms !== false,
    heatmaps: state.reportOptions?.heatmaps !== false,
    boomAngle: state.reportOptions?.boomAngle !== false,
    polarPlots: !!state.reportOptions?.polarPlots,
    maneuverAnalysis: !!state.reportOptions?.maneuverAnalysis,
    downloadCsv: !!state.reportOptions?.downloadCsv,
  });
}

function syncReportMenuToggles() {
  for (const [id, key] of [
    ['report-menu-summary-stats', 'summaryStats'],
    ['report-menu-histograms', 'histograms'],
    ['report-menu-heatmaps', 'heatmaps'],
    ['report-menu-boom-angle', 'boomAngle'],
    ['report-menu-polar-plots', 'polarPlots'],
    ['report-menu-tack-analysis', 'maneuverAnalysis'],
  ]) {
    const toggle = el(id);
    if (toggle) toggle.checked = !!state.reportOptions?.[key];
  }
}

function applyReportOptionsVisibility() {
  const wrap = el('analysis-report-advanced');
  if (wrap) wrap.style.display = state.advancedMode ? 'flex' : 'none';
  const polarToggle = el('report-opt-polar-plots');
  if (polarToggle) {
    polarToggle.disabled = !state.advancedMode;
    polarToggle.checked = !!state.reportOptions?.polarPlots;
  }
  const tackToggle = el('report-opt-tack-analysis');
  if (tackToggle) {
    tackToggle.disabled = !state.advancedMode;
    tackToggle.checked = !!state.reportOptions?.maneuverAnalysis;
  }
  const csvToggle = el('report-opt-download-csv');
  if (csvToggle) {
    csvToggle.disabled = !state.advancedMode;
    csvToggle.checked = !!state.reportOptions?.downloadCsv;
  }
  const reportCsvBtn = el('btn-download-report-csv');
  if (reportCsvBtn) reportCsvBtn.style.display = state.advancedMode && state.reportOptions?.downloadCsv ? '' : 'none';
  const maneuverCsvBtn = el('btn-maneuver-download-csv');
  if (maneuverCsvBtn) maneuverCsvBtn.style.display = state.advancedMode && state.reportOptions?.downloadCsv ? '' : 'none';
  syncReportMenuToggles();
}

function setReportPolarPlots(enabled) {
  state.reportOptions = {
    ...(state.reportOptions || {}),
    polarPlots: !!enabled,
  };
  saveReportOptionsSetting();
  applyReportOptionsVisibility();
}

function setReportIncludeOption(key, enabled) {
  if (!key) return;
  state.reportOptions = {
    ...(state.reportOptions || {}),
    [key]: !!enabled,
  };
  saveReportOptionsSetting();
  applyReportOptionsVisibility();
}

function setReportTackAnalysis(enabled) {
  state.reportOptions = {
    ...(state.reportOptions || {}),
    maneuverAnalysis: !!enabled,
  };
  saveReportOptionsSetting();
  applyReportOptionsVisibility();
}

function setReportDownloadCsv(enabled) {
  state.reportOptions = {
    ...(state.reportOptions || {}),
    downloadCsv: !!enabled,
  };
  saveReportOptionsSetting();
  applyReportOptionsVisibility();
  renderAnalysisTab();
  renderManeuverPanel();
}

function applyAdvancedModeVisibility() {
  const advanced = !!state.advancedMode;
  const toggle = el('advanced-mode-toggle');
  if (toggle) toggle.checked = advanced;
  const featureManeuversToggle = el('feature-maneuvers-toggle');
  if (featureManeuversToggle) featureManeuversToggle.checked = !!state.advancedFeatures?.maneuversTab;
  const featureHullToggle = el('feature-hull3d-toggle');
  if (featureHullToggle) featureHullToggle.checked = !!state.advancedFeatures?.hull3d;
  const featureWindPanelToggle = el('feature-wind-panel-toggle');
  if (featureWindPanelToggle) featureWindPanelToggle.checked = !!state.advancedFeatures?.windPanel;
  const featureBoomPredictionsToggle = el('feature-boom-predictions-toggle');
  if (featureBoomPredictionsToggle) featureBoomPredictionsToggle.checked = areBoomPredictionsEnabled();
  const featureRudderPredictionsToggle = el('feature-rudder-predictions-toggle');
  if (featureRudderPredictionsToggle) featureRudderPredictionsToggle.checked = areRudderPredictionsEnabled();
  const poseInputSizeSelect = el('pose-input-size-select');
  if (poseInputSizeSelect) poseInputSizeSelect.value = String(getPoseInputMaxDim());
  const poseExactSegmentSeekToggle = el('pose-exact-segment-seek-toggle');
  if (poseExactSegmentSeekToggle) poseExactSegmentSeekToggle.checked = isPoseExactSegmentSeekEnabled();
  const settingsWrap = el('ath-settings-wrap');
  if (settingsWrap) settingsWrap.classList.toggle('open', advanced);

  const trackPanel = el('track-panel');
  if (trackPanel) trackPanel.style.display = advanced ? '' : 'none';

  const stlBtn = el('btn-stl');
  if (stlBtn) stlBtn.style.display = advanced && state.advancedFeatures?.hull3d ? '' : 'none';
  const heatmapBtn = el('btn-heatmaps');
  if (heatmapBtn) heatmapBtn.style.display = advanced ? '' : 'none';
  const maneuverNav = document.querySelector('.nav-btn[data-view="view-maneuvers"]');
  if (maneuverNav) maneuverNav.style.display = state.advancedFeatures?.maneuversTab ? '' : 'none';
  if (!state.advancedFeatures?.maneuversTab && el('view-maneuvers')?.classList.contains('active-view')) {
    switchView('view-analysis');
  }

  if (!advanced) {
    closeStlViewer();
    setInlineHeatmapsVisible(false);
  } else if (!state.advancedFeatures?.hull3d) {
    closeStlViewer();
  }
  applyReportOptionsVisibility();
  renderManeuverPanel();
  void renderManeuverWorkspace();
  updateCurrentSegmentActions();
  renderWindMapLayer();
  renderWindMapControl();
  renderMap();
  renderManeuverPanel();
  renderManeuverMap();
  drawSogCanvas();
}

function setAdvancedMode(enabled) {
  state.advancedMode = !!enabled;
  saveJsonLocal(ADVANCED_MODE_KEY, state.advancedMode);
  applyAdvancedModeVisibility();
}

// ── Progress ring + queue ──────────────────────────────────────────────
function updateProgressRing(pct, labelOverride = null) {
  const fill = el('ring-fill');
  const label = el('progress-pct');
  if (!fill) return;
  const offset = 100 - Math.min(100, Math.max(0, pct));
  fill.setAttribute('stroke-dashoffset', offset);
  if (labelOverride != null && String(labelOverride).trim() !== '') {
    label.textContent = String(labelOverride);
  } else {
    label.textContent = pct > 0 ? Math.round(pct) + '%' : '';
  }
}

let _queuePollTimer = null;
let _queueAggregateProgress = null;

function buildQueueProgressJobKey(jobLike) {
  const fileId = String(jobLike?.fileId || '');
  const startRaw = jobLike?.startSec ?? jobLike?.opts?.startSec;
  const endRaw = jobLike?.endSec ?? jobLike?.opts?.endSec;
  const startSec = Number.isFinite(Number(startRaw)) ? Number(startRaw).toFixed(3) : 'start';
  const endSec = Number.isFinite(Number(endRaw)) ? Number(endRaw).toFixed(3) : 'end';
  const segmentName = String(jobLike?.segmentName ?? jobLike?.opts?.segmentName ?? '').trim();
  return `${fileId}|${startSec}|${endSec}|${segmentName}`;
}

function seedQueueAggregateProgress(jobLikes = []) {
  const keys = [];
  const seen = new Set();
  for (const job of jobLikes) {
    const key = buildQueueProgressJobKey(job);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    keys.push(key);
  }
  if (!keys.length) return;
  _queueAggregateProgress = {
    totalJobs: keys.length,
    expectedKeys: new Set(keys),
    startedKeys: new Set(),
    completedKeys: new Set(),
  };
}

function computeQueueAggregateProgress(activeJobs, queuedJobs) {
  const currentJobs = [
    ...activeJobs.map(job => ({
      key: buildQueueProgressJobKey(job),
      progress: Math.max(0, Math.min(1, Number(job?.progress) || 0)),
    })),
    ...queuedJobs.map(job => ({
      key: buildQueueProgressJobKey(job),
      progress: 0,
    })),
  ];
  if (!_queueAggregateProgress) {
    _queueAggregateProgress = {
      totalJobs: 0,
      expectedKeys: new Set(),
      startedKeys: new Set(),
      completedKeys: new Set(),
    };
  }

  const currentMap = new Map();
  for (const job of currentJobs) {
    if (!job.key) continue;
    currentMap.set(job.key, Math.max(currentMap.get(job.key) ?? 0, job.progress));
    if (!_queueAggregateProgress.expectedKeys.has(job.key)) {
      _queueAggregateProgress.expectedKeys.add(job.key);
      _queueAggregateProgress.totalJobs += 1;
    }
    _queueAggregateProgress.startedKeys.add(job.key);
    _queueAggregateProgress.completedKeys.delete(job.key);
  }

  for (const key of _queueAggregateProgress.startedKeys) {
    if (!currentMap.has(key)) _queueAggregateProgress.completedKeys.add(key);
  }

  const inFlightProgress = Array.from(currentMap.values()).reduce((sum, progress) => sum + progress, 0);
  const pct = ((_queueAggregateProgress.completedKeys.size + inFlightProgress) / Math.max(1, _queueAggregateProgress.totalJobs)) * 100;
  return Math.round(Math.max(0, Math.min(100, pct)));
}

function renderQueue() {
  const list = el('queue-list');
  const empty = el('queue-empty');
  if (!list) return;
  list.innerHTML = '';

  // Gather active pose-engine jobs
  const poseJobs = PoseEngine.getActiveJobs();
  const startingJobs = getStartingSegmentJobs().filter(job => !poseJobs.some(active => String(active.fileId) === String(job.fileId)));
  const queuedJobs = getQueuedSegmentJobs();
  const visibleActiveJobs = [...startingJobs, ...poseJobs];
  const totalJobs = visibleActiveJobs.length + queuedJobs.length;
  const hasAny = totalJobs > 0;

  if (hasAny) {
    empty.style.display = 'none';
    const weightedPct = computeQueueAggregateProgress(visibleActiveJobs, queuedJobs);
    updateProgressRing(weightedPct);

    for (const job of visibleActiveJobs) {
      const pct = Math.round((job.progress || 0) * 100);
      const elapsed = ((Date.now() - job.startTime) / 1000) | 0;
      const mins = (elapsed / 60) | 0;
      const secs = elapsed % 60;
      const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
      const isPendingStart = !!job.pendingStart;
      const accentColor = job.cancelled ? 'var(--red,#e3342f)' : 'var(--accent,#2a7fc4)';
      const statusText = job.cancelled
        ? 'Cancelling…'
        : (isPendingStart ? `${job.message || 'Starting...'} — ${timeStr}` : `${pct}% — ${timeStr}`);
      // Look up filename from file metadata
      const meta = state.fileMeta?.[job.fileId];
      const fileName = meta?.filename || job.fileId.slice(0, 8);
      const segmentName = typeof job.segmentName === 'string' ? job.segmentName.trim() : '';
      const name = segmentName || fileName;
      const title = segmentName ? `${segmentName} (${fileName})` : name;
      const detailParts = [];
      if (segmentName) detailParts.push(`Video: ${fileName}`);
      if (job.message) detailParts.push(job.message);
      const startTxt = Number.isFinite(Number(job.startSec)) ? Number(job.startSec).toFixed(1) : null;
      const endTxt = Number.isFinite(Number(job.endSec)) ? Number(job.endSec).toFixed(1) : null;
      if (startTxt != null || endTxt != null) detailParts.push(`Segment range ${startTxt ?? '--'}s \u2192 ${endTxt ?? 'end'}s`);
      const detailLine = detailParts.join(' | ');
      const item = document.createElement('div');
      item.className = 'queue-item';
      item.style.flexWrap = 'wrap';
      item.innerHTML = `
        <span class="queue-item-name" title="${title}">🦴 ${name}</span>
        <span class="queue-item-status" style="color:${accentColor};font-weight:600;">${statusText}</span>
        ${(!job.cancelled && !isPendingStart) ? `<button class="queue-cancel" title="Cancel">✕</button>` : ''}
        <div style="width:100%;height:3px;background:var(--border);border-radius:2px;margin-top:4px;">
          <div style="height:100%;width:${pct}%;background:${accentColor};border-radius:2px;transition:width .3s;"></div>
        </div>
        <div style="width:100%;font-size:10px;color:var(--muted);margin-top:2px;">${detailLine}</div>
      `;
      if (!job.cancelled && !isPendingStart) {
        item.querySelector('.queue-cancel').onclick = () => {
          PoseEngine.cancelProcessing(job.fileId);
          renderQueue();
        };
      }
      list.appendChild(item);
    }

    for (const qJob of queuedJobs) {
      const meta = state.fileMeta?.[qJob.fileId];
      const fileName = meta?.filename || qJob.fileId.slice(0, 8);
      const segmentName = typeof qJob.opts?.segmentName === 'string' ? qJob.opts.segmentName.trim() : '';
      const name = segmentName || fileName;
      const title = segmentName ? `${segmentName} (${fileName})` : name;
      const startTxt = Number.isFinite(Number(qJob.opts?.startSec)) ? Number(qJob.opts.startSec).toFixed(1) : '--';
      const endTxt = Number.isFinite(Number(qJob.opts?.endSec)) ? Number(qJob.opts.endSec).toFixed(1) : 'end';
      const rangeTxt = `Segment range ${startTxt}s \u2192 ${endTxt}s`;
      const detailLine = segmentName ? `Video: ${fileName} | ${rangeTxt}` : rangeTxt;
      const item = document.createElement('div');
      item.className = 'queue-item';
      item.style.flexWrap = 'wrap';
      item.innerHTML = `
        <span class="queue-item-name" title="${title}">🕒 ${name}</span>
        <span class="queue-item-status" style="font-weight:600;">Queued ${qJob.queueIdx + 1}/${qJob.queueLen}</span>
        <button class="queue-cancel" title="Remove from queue">✕</button>
        <div style="width:100%;font-size:10px;color:var(--muted);margin-top:2px;">${detailLine}</div>
      `;
      item.querySelector('.queue-cancel').onclick = () => {
        if (dequeueQueuedSegmentRun(qJob.fileId, qJob.queueIdx)) renderQueue();
      };
      list.appendChild(item);
    }

    // Start polling if not already
    if (!_queuePollTimer) {
      _queuePollTimer = setInterval(() => {
        renderQueue();
      }, 600);
    }
  } else {
    empty.style.display = '';
    const idlePct = _queueAggregateProgress ? computeQueueAggregateProgress([], []) : 0;
    if (_queueAggregateProgress && _queueAggregateProgress.totalJobs > 0 && _queueAggregateProgress.completedKeys.size >= _queueAggregateProgress.totalJobs) {
      _queueAggregateProgress = null;
      updateProgressRing(0);
    } else if (!_queueAggregateProgress) {
      updateProgressRing(0);
    } else {
      updateProgressRing(idlePct);
    }
    // Stop polling when nothing is active
    if (_queuePollTimer) { clearInterval(_queuePollTimer); _queuePollTimer = null; }
  }
}

async function cancelJob(jobId) {
  // Cancel by fileId for pose jobs
  if (jobId) PoseEngine.cancelProcessing(jobId);
  renderQueue();
}

function initQueueDropdown() {
  function positionQueueDropdown() {
    const dd = el('queue-dropdown');
    const ring = el('progress-ring');
    if(!dd || !ring) return;

    const compact = window.matchMedia('(max-width: 1180px)').matches;
    dd.classList.toggle('floating', compact);
    if(!compact) {
      dd.style.left = '';
      dd.style.top = '';
      dd.style.right = '';
      return;
    }

    const rect = ring.getBoundingClientRect();
    const margin = 8;
    const maxWidth = Math.max(220, window.innerWidth - margin * 2);
    const width = Math.max(220, Math.min(dd.offsetWidth || 320, maxWidth));
    const left = Math.max(margin, Math.min(rect.right - width, window.innerWidth - width - margin));
    const top = Math.min(window.innerHeight - 12, rect.bottom + 4);
    dd.style.left = `${left}px`;
    dd.style.top = `${top}px`;
    dd.style.right = 'auto';
  }

  el('progress-ring').onclick = () => {
    const dd = el('queue-dropdown');
    const opening = !dd.classList.contains('open');
    dd.classList.toggle('open');
    if(opening) positionQueueDropdown();
    renderQueue();
  };
  document.addEventListener('click', e => {
    const dd = el('queue-dropdown');
    if (dd.classList.contains('open') && !el('progress-wrap').contains(e.target)) dd.classList.remove('open');
  });
  window.addEventListener('resize', () => {
    const dd = el('queue-dropdown');
    if(dd?.classList.contains('open')) positionQueueDropdown();
  });
  el('topbar')?.addEventListener('scroll', () => {
    const dd = el('queue-dropdown');
    if(dd?.classList.contains('open')) positionQueueDropdown();
  }, { passive: true });
}

function initTimelineStatsWheelScroll() {
  const wrap = el('tl-stats');
  if (!wrap || wrap.dataset.wheelInit === '1') return;
  wrap.dataset.wheelInit = '1';

  wrap.addEventListener('wheel', (e) => {
    if (!wrap) return;
    const maxScrollLeft = Math.max(0, wrap.scrollWidth - wrap.clientWidth);
    if (maxScrollLeft <= 1) return;

    let delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
    if (!Number.isFinite(delta) || Math.abs(delta) < 0.01) return;

    if (e.deltaMode === 1) delta *= 16;
    else if (e.deltaMode === 2) delta *= wrap.clientWidth;

    const nextScrollLeft = Math.max(0, Math.min(maxScrollLeft, wrap.scrollLeft + delta));
    if (Math.abs(nextScrollLeft - wrap.scrollLeft) < 0.5) return;

    e.preventDefault();
    wrap.scrollLeft = nextScrollLeft;
  }, { passive: false });
}

// ── Init ───────────────────────────────────────────────────────────────
function initLeafletTrackPanes(mapInstance) {
  if (!mapInstance) return;
  mapInstance.createPane('baseTrackPane').style.zIndex = '390';
  mapInstance.createPane('processedTrackPane').style.zIndex = '405';
  mapInstance.createPane('windPane').style.zIndex = '430';
  mapInstance.createPane('splitMarkersPane').style.zIndex = '450';
}

function initMap() {
  state.map = L.map('map',{zoomControl:true,preferCanvas:false,keyboard:false}).setView([55.5,10.5],7);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{attribution:'© CartoDB',maxZoom:18}).addTo(state.map);
  initLeafletTrackPanes(state.map);
}

function initManeuverMap() {
  if (state.maneuverMap || !el('maneuver-map')) return;
  state.maneuverMap = L.map('maneuver-map', { zoomControl: true, preferCanvas: false, keyboard: false }).setView([55.5, 10.5], 7);
  try { state.maneuverMap.zoomControl.setPosition('topright'); } catch {}
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', { attribution: '© CartoDB', maxZoom: 18 }).addTo(state.maneuverMap);
  initLeafletTrackPanes(state.maneuverMap);
  state.maneuverMap.on('zoomend', () => renderManeuverMap({ preserveView: true }));
  const shell = el('maneuver-map-shell');
  if (shell && window.ResizeObserver && !state.maneuverMapResizeObserver) {
    state.maneuverMapResizeObserver = new ResizeObserver(() => {
      scheduleManeuverMapRefresh({ render: false });
    });
    state.maneuverMapResizeObserver.observe(shell);
  }
  scheduleManeuverMapRefresh();
}

function applyManeuverViewZoom() {
  const content = el('maneuver-analysis-content');
  if (!content) return;
  const zoom = Math.max(0.65, Math.min(1.8, Number(state.maneuverViewZoom) || 1));
  state.maneuverViewZoom = zoom;
  if ('zoom' in content.style) {
    content.style.zoom = Math.abs(zoom - 1) < 0.001 ? '' : String(zoom);
    content.style.transform = '';
    content.style.width = '';
    return;
  }
  content.style.transformOrigin = 'top left';
  content.style.transform = Math.abs(zoom - 1) < 0.001 ? '' : `scale(${zoom})`;
  content.style.width = Math.abs(zoom - 1) < 0.001 ? '' : `${100 / zoom}%`;
}

function setManeuverViewZoom(nextZoom, pivotEvent = null) {
  const shell = el('maneuver-analysis-shell');
  const previousZoom = Math.max(0.65, Math.min(1.8, Number(state.maneuverViewZoom) || 1));
  const zoom = Math.max(0.65, Math.min(1.8, Number(nextZoom) || previousZoom));
  if (Math.abs(zoom - previousZoom) < 0.002) return;

  let pivotX = null;
  let pivotY = null;
  let contentX = null;
  let contentY = null;
  if (shell && pivotEvent) {
    const rect = shell.getBoundingClientRect();
    const eventX = Number(pivotEvent.clientX);
    const eventY = Number(pivotEvent.clientY);
    if (Number.isFinite(eventX) && Number.isFinite(eventY)) {
      pivotX = Math.max(0, Math.min(rect.width, eventX - rect.left));
      pivotY = Math.max(0, Math.min(rect.height, eventY - rect.top));
      contentX = (shell.scrollLeft + pivotX) / previousZoom;
      contentY = (shell.scrollTop + pivotY) / previousZoom;
    }
  }

  state.maneuverViewZoom = zoom;
  applyManeuverViewZoom();

  if (shell && contentX != null && contentY != null) {
    requestAnimationFrame(() => {
      shell.scrollLeft = Math.max(0, contentX * zoom - pivotX);
      shell.scrollTop = Math.max(0, contentY * zoom - pivotY);
    });
  }
}

function initManeuverViewTrackpadZoom() {
  const shell = el('maneuver-analysis-shell');
  if (!shell || shell.dataset.trackpadZoomInit === '1') return;
  shell.dataset.trackpadZoomInit = '1';
  shell.addEventListener('wheel', evt => {
    if (!evt.ctrlKey && !evt.metaKey) return;
    evt.preventDefault();
    evt.stopPropagation();
    const delta = Number(evt.deltaY);
    if (!Number.isFinite(delta) || Math.abs(delta) < 0.01) return;
    const factor = Math.exp(-delta * 0.002);
    setManeuverViewZoom((Number(state.maneuverViewZoom) || 1) * factor, evt);
  }, { passive: false, capture: true });

  shell.addEventListener('gesturestart', evt => {
    evt.preventDefault();
    shell.dataset.gestureStartZoom = String(Number(state.maneuverViewZoom) || 1);
  }, { passive: false });
  shell.addEventListener('gesturechange', evt => {
    evt.preventDefault();
    const startZoom = Number(shell.dataset.gestureStartZoom) || Number(state.maneuverViewZoom) || 1;
    const scale = Number(evt.scale);
    if (!Number.isFinite(scale) || scale <= 0) return;
    setManeuverViewZoom(startZoom * scale, evt);
  }, { passive: false });
}

function initManeuverViewDivider() {
  const layout = el('maneuver-view-layout');
  const divider = el('maneuver-view-divider');
  if (!layout || !divider || divider.dataset.resizeInit === '1') return;
  divider.dataset.resizeInit = '1';

  const applySaved = () => {
    const saved = Number(loadJsonLocal(MANEUVER_SPLIT_KEY, null));
    if (!Number.isFinite(saved) || saved <= 0) return;
    const rect = layout.getBoundingClientRect();
    const max = Math.max(320, rect.width - 390);
    const px = Math.max(280, Math.min(saved, max));
    layout.style.setProperty('--maneuver-map-col', `${px}px`);
    scheduleManeuverMapRefresh({ render: false });
  };
  applySaved();

  let dragging = false;
  const onMove = evt => {
    if (!dragging) return;
    const rect = layout.getBoundingClientRect();
    const clientX = evt.touches?.[0]?.clientX ?? evt.clientX;
    const raw = Number(clientX) - rect.left;
    const max = Math.max(320, rect.width - 390);
    const px = Math.max(280, Math.min(raw, max));
    layout.style.setProperty('--maneuver-map-col', `${px}px`);
    saveJsonLocal(MANEUVER_SPLIT_KEY, px);
    scheduleManeuverMapRefresh({ render: false });
    evt.preventDefault();
  };
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    scheduleManeuverMapRefresh({ render: false });
  };
  const start = evt => {
    dragging = true;
    divider.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    onMove(evt);
  };

  divider.addEventListener('mousedown', start);
  divider.addEventListener('touchstart', start, { passive: false });
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove, { passive: false });
  window.addEventListener('mouseup', stop);
  window.addEventListener('touchend', stop);
  window.addEventListener('resize', applySaved);
}

async function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  if (['localhost', '127.0.0.1'].includes(window.location.hostname)) {
    try {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map(reg => reg.unregister().catch(() => {})));
    } catch {}
    return;
  }
  try {
    const registration = await navigator.serviceWorker.register(
      new URL('./sw.js?v=20260527detail1', import.meta.url).toString(),
      { updateViaCache: 'none' },
    );
    registration.update().catch(() => {});
  } catch (err) {
    console.warn('[PWA] service worker registration failed:', err);
  }
}

async function init() {
  registerServiceWorker();
  initMap();
  initDivider();
  initPhonePlaybackDivider();
  initNavigation();
  initKeyboard();
  initSogScrub();
  initQueueDropdown();
  initTimelineStatsWheelScroll();
  initManeuverViewTrackpadZoom();
  initManeuverViewDivider();
  loadAdvancedModeSetting();
  loadAdvancedFeaturesSetting();
  loadVideoLayoutSetting();
  loadReportOptionsSetting();
  loadTimelineStatsWindowSetting();
  loadTimelineStatOverlayGraphSetting();
  loadTimelineSogGapStitchingSetting();
  loadPoseProcessingSettings();
  loadApiCsvConfig();

  await loadProjects();

  const savedProjectId = localStorage.getItem('trollfish_projectId');
  if (savedProjectId && state.projects.some(p => p.id === savedProjectId)) {
    await selectProject(savedProjectId);
  }

  // New project
  el('btn-new-proj').onclick = ()=>{ el('new-proj-name').value=''; el('new-proj-modal').classList.add('open'); setTimeout(()=>el('new-proj-name').focus(),50); };
  el('btn-new-cancel').onclick = ()=>el('new-proj-modal').classList.remove('open');
  el('btn-new-confirm').onclick = async()=>{
    const name=el('new-proj-name').value.trim(); if(!name) return;
    const btn = el('btn-new-confirm');
    btn.disabled = true;
    try {
      await createProject(name);
      el('new-proj-modal').classList.remove('open');
    } finally {
      btn.disabled = false;
    }
  };
  el('new-proj-name').addEventListener('keydown',e=>{ if(e.key==='Enter') el('btn-new-confirm').click(); });
  const inlineProfileInput = el('profile-name-in');
  const createProfileInline = async () => {
    const name = inlineProfileInput?.value?.trim();
    if (!name) return;
    try {
      await createProject(name);
      if (inlineProfileInput) inlineProfileInput.value = '';
    } catch {}
  };
  const createProfileBtn = el('btn-create-profile-inline');
  if (createProfileBtn) createProfileBtn.onclick = () => createProfileInline();
  inlineProfileInput?.addEventListener('keydown', e => { if (e.key === 'Enter') createProfileInline(); });

  // Delete project
  el('btn-del-proj').onclick = ()=>deleteProject();

  // Upload — drop zone and wizard
  initDropZone();
  const apiCsvSaveBtn = el('btn-api-csv-save');
  if (apiCsvSaveBtn) apiCsvSaveBtn.onclick = () => saveApiCsvConfigFromInputs();
  el('api-csv-url')?.addEventListener('change', () => saveApiCsvConfigFromInputs());
  el('api-csv-key')?.addEventListener('change', () => saveApiCsvConfigFromInputs());
  initWizardStepClicks();
  el('btn-wiz-next').onclick = () => wizNext();
  el('btn-wiz-back').onclick = () => wizBack();
  el('btn-go-analysis').onclick = () => switchView('view-analysis');

  // Refresh
  el('btn-refresh').onclick = async()=>{
    await loadMapData();
    renderFilesList();
    buildTimeline();
    void refreshManeuvers('manual-refresh');
  };

  // Athletes
  el('btn-add-ath').onclick = ()=>openAthleteEdit(null);
  el('btn-save-ath').onclick = ()=>saveAthlete();
  el('btn-cancel-ath').onclick = ()=>{ el('ath-edit-row').classList.remove('open'); state.editingAthleteId=null; };
  el('ath-weight-in').addEventListener('keydown',e=>{ if(e.key==='Enter') saveAthlete(); });
  el('ath-height-in').addEventListener('keydown',e=>{ if(e.key==='Enter') saveAthlete(); });

  // Workers
  const workersIn = el('workers-input');
  if(workersIn) workersIn.addEventListener('change',e=>{ state.mediapipeWorkers=parseInt(e.target.value)||2; });
  const poseInputSizeSelect = el('pose-input-size-select');
  if (poseInputSizeSelect) poseInputSizeSelect.addEventListener('change', e => setPoseInputMaxDim(e.target.value));
  const poseExactSegmentSeekToggle = el('pose-exact-segment-seek-toggle');
  if (poseExactSegmentSeekToggle) poseExactSegmentSeekToggle.addEventListener('change', e => setPoseExactSegmentSeek(e.target.checked));
  const tlStatsWindowIn = el('tl-stats-window-input');
  if (tlStatsWindowIn) tlStatsWindowIn.addEventListener('change', e => setTimelineStatsWindowSec(e.target.value));
  const tlStatOverlayToggle = el('tl-stats-overlay-graph-toggle');
  if (tlStatOverlayToggle) tlStatOverlayToggle.addEventListener('change', e => setTimelineStatOverlayGraph(e.target.checked));
  const tlSogGapStitchingToggle = el('tl-sog-gap-stitching-toggle');
  if (tlSogGapStitchingToggle) tlSogGapStitchingToggle.addEventListener('change', e => setTimelineSogGapStitching(e.target.checked));
  const advTowToggle = el('advanced-tow-filter-toggle');
  if (advTowToggle) advTowToggle.addEventListener('change', e => setTowFilteringDisabledForTrustedSession(!e.target.checked));
  const advToggle = el('advanced-mode-toggle');
  if (advToggle) advToggle.addEventListener('change', e => setAdvancedMode(e.target.checked));
  const maneuverFeatureToggle = el('feature-maneuvers-toggle');
  if (maneuverFeatureToggle) maneuverFeatureToggle.addEventListener('change', e => setAdvancedFeature('maneuversTab', e.target.checked));
  const hullFeatureToggle = el('feature-hull3d-toggle');
  if (hullFeatureToggle) hullFeatureToggle.addEventListener('change', e => setAdvancedFeature('hull3d', e.target.checked));
  const windPanelFeatureToggle = el('feature-wind-panel-toggle');
  if (windPanelFeatureToggle) windPanelFeatureToggle.addEventListener('change', e => setAdvancedFeature('windPanel', e.target.checked));
  const boomPredictionsFeatureToggle = el('feature-boom-predictions-toggle');
  if (boomPredictionsFeatureToggle) boomPredictionsFeatureToggle.addEventListener('change', e => setAdvancedFeature('boomPredictions', e.target.checked));
  const rudderPredictionsFeatureToggle = el('feature-rudder-predictions-toggle');
  if (rudderPredictionsFeatureToggle) rudderPredictionsFeatureToggle.addEventListener('change', e => setAdvancedFeature('rudderPredictions', e.target.checked));
  const videoLayoutBtn = el('btn-video-layout');
  if (videoLayoutBtn) videoLayoutBtn.addEventListener('click', e => {
    e.stopPropagation();
    toggleVideoLayoutPopover();
  });
  document.querySelectorAll('.layout-choice').forEach(choice => {
    choice.addEventListener('click', e => {
      setVideoLayout(choice.dataset.layout);
      renderVideoLayoutPopover();
    });
  });
  document.addEventListener('click', e => {
    const wrap = el('video-layout-wrap');
    if (wrap && !wrap.contains(e.target)) closeVideoLayoutPopover();
  });
  const showAllVideosBtn = el('btn-video-show-all');
  if (showAllVideosBtn) showAllVideosBtn.onclick = () => showAllVideoSlots();
  const reportPolarToggle = el('report-opt-polar-plots');
  if (reportPolarToggle) reportPolarToggle.addEventListener('change', e => setReportPolarPlots(e.target.checked));
  const reportTackToggle = el('report-opt-tack-analysis');
  if (reportTackToggle) reportTackToggle.addEventListener('change', e => setReportTackAnalysis(e.target.checked));
  const reportCsvToggle = el('report-opt-download-csv');
  if (reportCsvToggle) reportCsvToggle.addEventListener('change', e => setReportDownloadCsv(e.target.checked));
  const reportSettingsBtn = el('btn-report-settings');
  if (reportSettingsBtn) reportSettingsBtn.addEventListener('click', e => {
    e.stopPropagation();
    el('report-settings-popover')?.classList.toggle('open');
  });
  for (const [id, key] of [
    ['report-menu-summary-stats', 'summaryStats'],
    ['report-menu-histograms', 'histograms'],
    ['report-menu-heatmaps', 'heatmaps'],
    ['report-menu-boom-angle', 'boomAngle'],
    ['report-menu-polar-plots', 'polarPlots'],
    ['report-menu-tack-analysis', 'maneuverAnalysis'],
  ]) {
    el(id)?.addEventListener('change', e => setReportIncludeOption(key, e.target.checked));
  }
  document.addEventListener('click', e => {
    const wrap = el('report-settings-wrap');
    if (wrap && !wrap.contains(e.target)) el('report-settings-popover')?.classList.remove('open');
  });
  const maneuverDetectionInputs = [
    ['maneuver-min-heading-input', 'minHeadingDeltaDeg'],
    ['maneuver-min-stable-input', 'minStableSideSec'],
    ['maneuver-stats-window-input', 'statsWindowSec'],
  ];
  for (const [id, key] of maneuverDetectionInputs) {
    const input = el(id);
    if (!input) continue;
    input.addEventListener('change', async e => {
      state.maneuverDetection = normalizeManeuverDetectionSettings({
        ...(state.maneuverDetection || {}),
        [key]: e.target.value,
      });
      syncManeuverDetectionInputs();
      await saveProjectCvConfig();
      void refreshManeuvers('settings-changed');
    });
  }
  applyAdvancedModeVisibility();

  // Segment creation
  el('btn-new-segment').onclick = () => startSegmentCreation();
  el('btn-seg-cancel').onclick = () => cancelSegmentCreation();
  el('btn-current-seg-rename').onclick = () => {
    const segId = el('current-seg-actions')?.dataset.segmentId;
    if (segId) openSegmentRenameModal(segId);
  };
  el('btn-current-seg-delete').onclick = () => {
    const segId = el('current-seg-actions')?.dataset.segmentId;
    if (segId) deleteSegment(segId);
  };

  // Segment panel header toggle
  el('segment-panel-hdr').onclick = () => {
    el('segment-panel').classList.toggle('collapsed');
  };

  const maneuverAthleteFilter = el('maneuver-athlete-filter');
  if (maneuverAthleteFilter) {
    maneuverAthleteFilter.addEventListener('change', e => {
      state.maneuverFilters = { ...(state.maneuverFilters || {}), athleteId: String(e.target.value || '') };
      renderManeuverPanel();
      renderManeuverMap();
    });
  }
  const maneuverTypeFilter = el('maneuver-type-filter');
  if (maneuverTypeFilter) {
    maneuverTypeFilter.addEventListener('change', e => {
      state.maneuverFilters = { ...(state.maneuverFilters || {}), type: String(e.target.value || 'all') };
      renderManeuverPanel();
      renderManeuverMap();
    });
  }
  const maneuverSideFilter = el('maneuver-side-filter');
  if (maneuverSideFilter) {
    maneuverSideFilter.addEventListener('change', e => {
      state.maneuverFilters = { ...(state.maneuverFilters || {}), side: String(e.target.value || 'all') };
      renderManeuverPanel();
      renderManeuverMap();
    });
  }
  const openCompareBtn = el('btn-maneuver-open-compare');
  if (openCompareBtn) {
    openCompareBtn.onclick = () => {
      if (!state.maneuverComparePicking) {
        startManeuverComparePicking({ reset: true });
        return;
      }
      openCheckedManeuverCompare({ preserveMapView: true, loadDeep: true, force: false });
    };
  }
  const clearCompareBtn = el('btn-maneuver-clear-compare');
  if (clearCompareBtn) {
    clearCompareBtn.onclick = () => {
      if (state.maneuverComparePicking) cancelManeuverComparePicking({ clear: true });
      else clearManeuverCompareSet();
    };
  }
  const maneuverRedetectBtn = el('btn-maneuver-redetect');
  if (maneuverRedetectBtn) maneuverRedetectBtn.onclick = () => { void refreshManeuvers('manual-rebuild'); };
  const maneuverDownloadMapBtn = el('btn-maneuver-download-map');
  if (maneuverDownloadMapBtn) maneuverDownloadMapBtn.onclick = () => { void downloadCheckedManeuverCompareMap(); };
  const maneuverDownloadCsvBtn = el('btn-maneuver-download-csv');
  if (maneuverDownloadCsvBtn) maneuverDownloadCsvBtn.onclick = () => { void downloadSelectedManeuverCsv(); };
  const maneuverProcessSelectedBtn = el('btn-maneuver-process-selected');
  if (maneuverProcessSelectedBtn) {
    maneuverProcessSelectedBtn.onclick = () => {
      const checked = getCheckedManeuvers();
      const selected = getSelectedManeuvers();
      const moves = checked.length ? checked : selected;
      if (!moves.length) {
        alert('Select or check at least one maneuver to process.');
        return;
      }
      void triggerManeuverPoseExtraction(moves);
    };
  }
  const maneuverManualBtn = el('btn-maneuver-manual');
  if (maneuverManualBtn) {
    maneuverManualBtn.onclick = () => toggleManualManeuverSegmentation();
    updateManualManeuverButtonState();
  }

  // Track panel header toggle
  el('track-panel-hdr').onclick = () => {
    el('track-panel').classList.toggle('collapsed');
    renderTrackPanel();
  };

  // Timeline controls
  el('btn-tl-play').onclick = ()=>tlToggle();
  el('btn-tl-skip-back').onclick = ()=>{ if(state.tl.currentTs != null) tlSeekTo(state.tl.currentTs - 10); };
  el('btn-tl-skip-fwd').onclick = ()=>{ if(state.tl.currentTs != null) tlSeekTo(state.tl.currentTs + 10); };
  if (el('btn-tl-phone-playback')) {
    el('btn-tl-phone-playback').onclick = () => toggleTimelinePhonePlayback();
  }
  el('tl-speed-select').addEventListener('change', e => {
    state.tl.playbackRate = parseFloat(e.target.value) || 1;
    for(const slot of state.tl.athleteSlots) {
      if(slot.videoEl) setSlotPlaybackRate(slot, state.tl.playbackRate);
    }
    if (state.phonePlayback?.videoEl) {
      try { state.phonePlayback.videoEl.playbackRate = state.tl.playbackRate; } catch {}
    }
  });

  // STL
  el('btn-stl').onclick = ()=>{
    const modal = el('stl-inline');
    if(modal.classList.contains('open') && state.advancedPane.mode === 'stl') closeStlViewer();
    else openStlViewer();
  };
  el('btn-heatmaps').onclick = () => {
    const modal = el('stl-inline');
    if (modal.classList.contains('open') && state.advancedPane.mode === 'heatmaps') closeStlViewer();
    setInlineHeatmapsVisible(!state.inlineHeatmaps?.visible);
  };
  el('btn-maneuver-refresh').onclick = () => { void refreshManeuvers('manual-rebuild'); };
  el('btn-maneuver-analyze').onclick = () => {
    void openManeuverWorkspace({ force: true, preserveMapView: true, loadDeep: true });
  };
  el('btn-maneuver-segment').onclick = async () => {
    const selected = getSelectedManeuvers();
    if (selected.length !== 1) return;
    await createSegmentFromManeuver(selected[0]);
  };
  el('btn-stl-close').onclick = ()=> closeStlViewer();
  initStlResize();

  // Segment modal
  el('btn-seg-modal-cancel').onclick = () => closeSegmentModal({ cancelCreation: true });
  el('segment-modal').addEventListener('click', e => { if(e.target === el('segment-modal')) closeSegmentModal({ cancelCreation: true }); });
  el('seg-name-in').addEventListener('keydown', e => { if(e.key === 'Enter') saveNewSegment(); });
  el('btn-seg-modal-save').onclick = () => saveNewSegment();

  // Analysis / Report
  el('btn-gen-report-all').onclick = () => generateReportAll();
  const reportCsvBtn = el('btn-download-report-csv');
  if (reportCsvBtn) reportCsvBtn.onclick = () => { void downloadSelectedReportCsv(); };

  // Modal closes
  el('new-proj-modal').addEventListener('click',e=>{ if(e.target===el('new-proj-modal')) el('new-proj-modal').classList.remove('open'); });
  el('confirm-modal').addEventListener('click',e=>{ if(e.target===el('confirm-modal')) el('confirm-modal').classList.remove('open'); });
  el('btn-inline-plot-overlay-close').onclick = () => closeInlineHeatmapPlotOverlay();
  el('inline-plot-overlay').addEventListener('click', e => {
    if (e.target === el('inline-plot-overlay')) closeInlineHeatmapPlotOverlay();
  });
  el('btn-timeline-media-close').onclick = () => closeTimelineMediaOverlay();
  el('timeline-media-modal').addEventListener('click', e => {
    if (e.target === el('timeline-media-modal')) closeTimelineMediaOverlay();
  });

  // Redraw SOG canvas on window resize
  window.addEventListener('resize', () => {
    syncPhonePlaybackShellSize();
    drawSogCanvas();
  });
  updatePhonePlaybackToggleButton();
  updateSetupUiState();
  updateCurrentSegmentActions();
}

// ── STL Viewer ─────────────────────────────────────────────────────────
const SKEL_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],
  [15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],
  [30,32],[27,31],[28,32]
];

// ── Hull 3D camera-pose tuning (live preview) ──────────────────────────
// Live-preview offset applied to displayed skeletons in the 3D viewer so the
// user sees the effect of the tuning sliders immediately. The EXACT placement is
// produced by re-processing with cfg.camera_pose_offset; this preview is a
// first-order approximation (exact for pure translation): each placed world point
// p is re-expressed as if the camera rotated by the small offset about its own
// position and then translated — p' = R_off·(p − camPos) + camPos + posOffset.
const STL_TUNE_DEFAULT = Object.freeze({ pitch_deg:0, yaw_deg:0, roll_deg:0, x_m:0, y_m:0, z_m:0, hip_z_m:0, ankle_z_m:0 });
const STL_TUNE_CAM_POS = [-3.194, 0.02, 0.585]; // nominal camera world position (config default)
let stlTunePreview = { ...STL_TUNE_DEFAULT };

function stlTuneOffsetActive(o = stlTunePreview) {
  return Object.keys(STL_TUNE_DEFAULT).some(k => Number(o?.[k]) || 0);
}

// Build a world-frame rotation matrix (row-major 9) from small pitch/yaw/roll
// offsets, matching the camera-relative convention Rz(roll)·Ry(yaw)·Rx(-pitch)
// projected into world axes (X fwd, Y lat, Z up).
function stlTuneRotMat(o) {
  const d2r = Math.PI / 180;
  const rx = -(Number(o.pitch_deg) || 0) * d2r; // pitch about lateral (Y)
  const ry = (Number(o.yaw_deg) || 0) * d2r;    // yaw about vertical (Z)
  const rz = (Number(o.roll_deg) || 0) * d2r;   // roll about fore-aft (X)
  const cx=Math.cos(rx), sx=Math.sin(rx), cy=Math.cos(ry), sy=Math.sin(ry), cz=Math.cos(rz), sz=Math.sin(rz);
  // Compose about world axes: roll→X, yaw→Z, pitch→Y
  const Rroll = [1,0,0, 0,cz,-sz, 0,sz,cz];
  const Ryaw  = [cy,-sy,0, sy,cy,0, 0,0,1];
  const Rpit  = [cx,0,sx, 0,1,0, -sx,0,cx];
  const mul = (A,B)=>[
    A[0]*B[0]+A[1]*B[3]+A[2]*B[6], A[0]*B[1]+A[1]*B[4]+A[2]*B[7], A[0]*B[2]+A[1]*B[5]+A[2]*B[8],
    A[3]*B[0]+A[4]*B[3]+A[5]*B[6], A[3]*B[1]+A[4]*B[4]+A[5]*B[7], A[3]*B[2]+A[4]*B[5]+A[5]*B[8],
    A[6]*B[0]+A[7]*B[3]+A[8]*B[6], A[6]*B[1]+A[7]*B[4]+A[8]*B[7], A[6]*B[2]+A[7]*B[5]+A[8]*B[8],
  ];
  return mul(Rroll, mul(Ryaw, Rpit));
}

// Apply the live-preview offset to a landmark dict {idx:[x,y,z]} → new dict.
function stlTuneApplyPreview(lm) {
  if (!lm || !stlTuneOffsetActive()) return lm;
  const o = stlTunePreview;
  const R = stlTuneRotMat(o);
  const cp = STL_TUNE_CAM_POS;
  const dx = Number(o.x_m)||0, dy = Number(o.y_m)||0, dz0 = Number(o.z_m)||0;
  const hipZ = Number(o.hip_z_m)||0;
  const out = {};
  for (const k of Object.keys(lm)) {
    const p = lm[k];
    if (!p) { out[k] = p; continue; }
    const vx = p[0]-cp[0], vy = p[1]-cp[1], vz = p[2]-cp[2];
    const rx = R[0]*vx + R[1]*vy + R[2]*vz;
    const ry = R[3]*vx + R[4]*vy + R[5]*vz;
    const rz = R[6]*vx + R[7]*vy + R[8]*vz;
    out[k] = [rx+cp[0]+dx, ry+cp[1]+dy, rz+cp[2]+dz0+hipZ];
  }
  return out;
}

// Slider id → (offset key, output formatter)
const STL_TUNE_FIELDS = [
  ['tune-pitch',  'pitch_deg', v => `${v.toFixed(1)}°`],
  ['tune-yaw',    'yaw_deg',   v => `${v.toFixed(1)}°`],
  ['tune-roll',   'roll_deg',  v => `${v.toFixed(1)}°`],
  ['tune-x',      'x_m',       v => `${v.toFixed(2)} m`],
  ['tune-y',      'y_m',       v => `${v.toFixed(2)} m`],
  ['tune-z',      'z_m',       v => `${v.toFixed(2)} m`],
  ['tune-hipz',   'hip_z_m',   v => `${v.toFixed(2)} m`],
  ['tune-anklez', 'ankle_z_m', v => `${v.toFixed(2)} m`],
];

function stlTuneSyncInputsFromState() {
  for (const [id, key, fmt] of STL_TUNE_FIELDS) {
    const input = el(id);
    const out = el(`${id}-out`);
    const v = Number(stlTunePreview[key]) || 0;
    if (input) input.value = String(v);
    if (out) out.textContent = fmt(v);
  }
}

function setStlTuneStatus(msg, kind = '') {
  const elx = el('stl-tune-status');
  if (!elx) return;
  elx.textContent = msg;
  elx.style.color = kind === 'ok' ? 'var(--green)' : (kind === 'err' ? 'var(--red, #e74c3c)' : 'var(--muted)');
}

// Wire the Hull 3D tuning panel. onPreviewChange re-renders displayed skeletons.
function setupStlTuningControls(onPreviewChange) {
  // Initialise preview from the saved per-project offset.
  const saved = state.cvConfig?.camera_pose_offset;
  stlTunePreview = { ...STL_TUNE_DEFAULT, ...(saved && typeof saved === 'object' ? saved : {}) };
  stlTuneSyncInputsFromState();

  for (const [id, key, fmt] of STL_TUNE_FIELDS) {
    const input = el(id);
    const out = el(`${id}-out`);
    if (!input) continue;
    input.oninput = () => {
      const v = Number(input.value) || 0;
      stlTunePreview[key] = v;
      if (out) out.textContent = fmt(v);
      onPreviewChange?.();
      setStlTuneStatus('Live preview — Save to keep, Re-process to bake into analysis.');
    };
  }

  const resetBtn = el('btn-tune-reset');
  if (resetBtn) resetBtn.onclick = () => {
    stlTunePreview = { ...STL_TUNE_DEFAULT };
    stlTuneSyncInputsFromState();
    onPreviewChange?.();
    setStlTuneStatus('Reset to no offset (not yet saved).');
  };

  const saveBtn = el('btn-tune-save');
  if (saveBtn) saveBtn.onclick = async () => {
    try {
      await persistCameraPoseOffset();
      setStlTuneStatus('Saved. Re-process to apply to skeleton positions.', 'ok');
    } catch (e) {
      console.error('[tune] save failed', e);
      setStlTuneStatus(`Save failed: ${e.message || e}`, 'err');
    }
  };

  const reprocBtn = el('btn-tune-reprocess');
  if (reprocBtn) reprocBtn.onclick = async () => {
    try {
      await persistCameraPoseOffset();
      setStlTuneStatus('Saved — re-processing videos with new tuning…', 'ok');
      // Clear the live preview so the 3D view shows the freshly-baked result,
      // not preview-on-top-of-baked.
      stlTunePreview = { ...STL_TUNE_DEFAULT };
      stlTuneSyncInputsFromState();
      onPreviewChange?.();
      await runSkeletonForAllVideos();
      setStlTuneStatus('Re-process complete. Tuning is baked into the saved skeletons.', 'ok');
    } catch (e) {
      console.error('[tune] reprocess failed', e);
      setStlTuneStatus(`Re-process failed: ${e.message || e}`, 'err');
    }
  };
}

// Persist the current preview offset into the project cvConfig.
async function persistCameraPoseOffset() {
  if (!state.projectId) throw new Error('No project selected');
  state.cvConfig = {
    ...(state.cvConfig || {}),
    camera_pose_offset: { ...STL_TUNE_DEFAULT, ...stlTunePreview },
  };
  await DB.upsertCvConfig(state.projectId, state.cvConfig);
}

async function loadSkeletonFrames(projectId, fileId) {
  if(!projectId || !fileId) return null;
  try {
    const frames = await PoseEngine.loadSkeletonFrames(projectId, fileId);
    if(!frames || frames.length === 0) return null;
    // Convert to format expected by 3D viewer: [{video_s, lm:{0:[x,y,z],...}}]
    return frames.map(f => ({
      video_s: f.ts,
      lm: f.skeleton,
    }));
  } catch {
    return null;
  }
}

function skelBinarySearch(frames, t, maxGapSec = 0.75) {
  if(!frames || !frames.length || !Number.isFinite(t)) return null;
  let lo = 0, hi = frames.length - 1;
  while(lo < hi) {
    const m = Math.floor((lo + hi + 1) / 2);
    if(frames[m].video_s <= t) lo = m;
    else hi = m - 1;
  }
  const a = frames[lo] || null;
  const b = frames[Math.min(lo + 1, frames.length - 1)] || null;
  let best = a;
  if (a && b && b !== a) {
    best = Math.abs(b.video_s - t) < Math.abs(a.video_s - t) ? b : a;
  }
  if (!best) return null;
  return Math.abs(best.video_s - t) <= maxGapSec ? best : null;
}

window.stlSetCamera = () => {};
let _stlRenderer = null;  // track WebGL renderer to prevent context exhaustion
let _stlAnimId = null;
let _stlResizeHandler = null;
let _stlResizeObserver = null;
let _heatmapRenderDepsPromise = null;
let _heatmapHullGeomPromise = null;

const ADVANCED_PANE_TITLES = Object.freeze({
  stl: 'Hull + Skeleton 3D',
  heatmaps: 'Segment Heatmaps',
  maneuvers: 'Maneuver Workspace',
});

function _disposeStlRenderer() {
  if(_stlAnimId) { cancelAnimationFrame(_stlAnimId); _stlAnimId = null; }
  if(_stlRenderer) {
    try {
      const gl = _stlRenderer.getContext();
      const ext = gl.getExtension('WEBGL_lose_context');
      if(ext) ext.loseContext();
    } catch(e) {}
    _stlRenderer.dispose();
    try {
      if(_stlRenderer.domElement?.parentNode) _stlRenderer.domElement.parentNode.removeChild(_stlRenderer.domElement);
    } catch(e) {}
    _stlRenderer = null;
  }
  // Remove any orphaned canvases from wrap
  const wrap = el('stl-3d-wrap');
  if(wrap) { while(wrap.firstChild) wrap.removeChild(wrap.firstChild); }
}

function nextAnimationFrame() {
  return new Promise(resolve => requestAnimationFrame(resolve));
}

function cleanupStlViewerRuntime() {
  const modal = el('stl-inline');
  if (modal?._liveUnsubs) {
    for (const fn of modal._liveUnsubs) fn();
    modal._liveUnsubs = null;
  }
  _disposeStlRenderer();
  if(_stlResizeHandler) { window.removeEventListener('resize', _stlResizeHandler); _stlResizeHandler = null; }
  if(_stlResizeObserver) { try { _stlResizeObserver.disconnect(); } catch(e) {} _stlResizeObserver = null; }
  window.stlSetCamera = () => {};
}

function setAdvancedPaneMode(mode = null) {
  const nextMode = mode || null;
  const prevMode = state.advancedPane?.mode || null;
  if (prevMode === 'stl' && nextMode !== 'stl') cleanupStlViewerRuntime();
  state.advancedPane.mode = nextMode;

  const stlLayout = el('stl-layout');
  const heatmapLayout = el('heatmap-layout');
  const maneuverLayout = el('maneuver-layout');
  if (stlLayout) stlLayout.classList.toggle('active', nextMode === 'stl');
  if (heatmapLayout) heatmapLayout.classList.toggle('active', nextMode === 'heatmaps');
  if (maneuverLayout) maneuverLayout.classList.toggle('active', nextMode === 'maneuvers');

  const title = el('advanced-pane-title');
  if (title) title.textContent = ADVANCED_PANE_TITLES[nextMode] || ADVANCED_PANE_TITLES.maneuvers;
}

async function ensureAdvancedPaneOpen(mode) {
  if (!state.advancedMode) return false;
  const modal = el('stl-inline');
  if (!modal) return false;
  setAdvancedPaneMode(mode);
  const wasOpen = modal.classList.contains('open');
  if (!wasOpen) {
    modal.classList.add('open');
    el('panel-left')?.classList.add('stl-open');
    await nextAnimationFrame();
    if (state.map) state.map.invalidateSize();
  }
  return !wasOpen;
}

function buildHeatmapSegmentSummary(seg) {
  if (!seg) {
    return 'The same keypoint and COM heatmaps used in report generation, limited to the current segment only.';
  }
  const athleteCount = getSegmentAthletes(seg).length;
  const videoCount = findOverlappingVideos(seg).length;
  const proc = getSegmentProcessingStatus(seg);
  const athleteLabel = `${athleteCount} athlete${athleteCount === 1 ? '' : 's'}`;
  const videoLabel = `${videoCount} overlapping video${videoCount === 1 ? '' : 's'}`;
  return `${athleteLabel} | ${videoLabel} | ${proc.label} (${proc.detail})`;
}

function setHeatmapPanelInfo(seg = null, { summary = null, status = 'Ready', statusTone = '' } = {}) {
  const nameEl = el('heatmap-segment-name');
  const rangeEl = el('heatmap-segment-range');
  const summaryEl = el('heatmap-segment-summary');
  const statusEl = el('heatmap-panel-status');

  if (nameEl) nameEl.textContent = seg?.name || 'No segment selected';
  if (rangeEl) {
    if (seg) {
      const duration = Math.max(0, Number(seg.tsEnd) - Number(seg.tsStart));
      rangeEl.textContent = `${fmtClock(seg.tsStart)} -> ${fmtClock(seg.tsEnd)} (${duration.toFixed(1)}s)`;
    } else {
      rangeEl.textContent = 'Move the playhead into a segment in Analysis to preview report heatmaps.';
    }
  }
  if (summaryEl) summaryEl.textContent = summary ?? buildHeatmapSegmentSummary(seg);
  if (statusEl) {
    statusEl.textContent = status;
    statusEl.className = `heatmap-meta${statusTone ? ` ${statusTone}` : ''}`;
  }
}

function replaceHeatmapContent(node) {
  const content = el('heatmap-content');
  if (!content) return null;
  content.replaceChildren(node);
  return content;
}

function createHeatmapState(title, body, kind = '') {
  const root = document.createElement('div');
  root.className = `heatmap-state${kind ? ` ${kind}` : ''}`;

  const titleEl = document.createElement('div');
  titleEl.className = 'heatmap-state-title';
  titleEl.textContent = title;

  const bodyEl = document.createElement('div');
  bodyEl.className = 'heatmap-state-body';
  bodyEl.textContent = body;

  root.append(titleEl, bodyEl);
  return root;
}

function renderHeatmapLoadingState(seg, msg = 'Preparing heatmaps...', pct = null) {
  const pctText = Number.isFinite(Number(pct)) ? `${Math.round(Number(pct) * 100)}%` : '';
  const status = pctText ? `${pctText} | ${msg}` : msg;
  setHeatmapPanelInfo(seg, { status, statusTone: 'loading' });
  replaceHeatmapContent(createHeatmapState(
    pctText ? `Building Heatmaps ${pctText}` : 'Building Heatmaps',
    msg || 'Preparing report-style heatmaps for the current segment.',
    'loading',
  ));
}

function resolveHeatmapImageSrc(imageB64) {
  const raw = String(imageB64 || '');
  if (!raw) return '';
  return raw.startsWith('data:') ? raw : `data:image/png;base64,${raw}`;
}

function normalizeDisplayColor(color) {
  const value = String(color || '').trim();
  return value || null;
}

function resolveHeatmapAthleteColor(result, seg, segmentAthletes = null) {
  const athleteId = normalizeAthleteId(result?.athlete_id);
  const athleteName = normalizeAthleteName(result?.athlete_name).toLowerCase();
  const segAthletes = Array.isArray(segmentAthletes) ? segmentAthletes : getSegmentAthletes(seg);

  const athlete = athleteId ? findAthleteById(athleteId) : null;
  const athleteColor = normalizeDisplayColor(athlete?.color);
  if (athleteColor) return athleteColor;

  const segAthleteById = athleteId
    ? segAthletes.find(entry => normalizeAthleteId(entry?.athleteId) === athleteId)
    : null;
  const segAthleteColorById = normalizeDisplayColor(segAthleteById?.color);
  if (segAthleteColorById) return segAthleteColorById;

  const segAthleteByName = athleteName
    ? segAthletes.find(entry => normalizeAthleteName(entry?.name).toLowerCase() === athleteName)
    : null;
  const segAthleteColorByName = normalizeDisplayColor(segAthleteByName?.color);
  if (segAthleteColorByName) return segAthleteColorByName;

  const namedAthlete = athleteName ? findAthleteByName(athleteName) : null;
  const namedAthleteColor = normalizeDisplayColor(namedAthlete?.color);
  if (namedAthleteColor) return namedAthleteColor;

  const fileIds = Array.isArray(result?.file_id) ? result.file_id : [result?.file_id];
  for (const fileId of fileIds) {
    const fileColor = normalizeDisplayColor(state.videoColors?.[fileId]);
    if (fileColor) return fileColor;
  }

  return '#5cc8ff';
}

const INLINE_HEATMAP_HULL_X = Object.freeze([-4.2, -3.5, -1.5, 0.0, 0.5, 0.0, -1.5, -3.5, -4.2]);
const INLINE_HEATMAP_HULL_Y = Object.freeze([0, 0.6, 0.73, 0.5, 0, -0.5, -0.73, -0.6, 0]);
const INLINE_HEATMAP_HULL_BOUNDS = Object.freeze({
  minFore: -4.35,
  maxFore: 0.75,
  maxSide: 0.95,
});
const INLINE_HEATMAP_CAMERA_ZOOM_OUT = 1.28;
const INLINE_HEATMAP_DRAW_INSET = 0.88;
const INLINE_HEATMAP_BOAT_LENGTH_M = 4.2;
const INLINE_HEATMAP_BOAT_CENTER_X = -(INLINE_HEATMAP_BOAT_LENGTH_M / 2);
const INLINE_HEATMAP_BOAT_CENTER_X_BIAS = 0.0;
const INLINE_HEATMAP_SCREEN_Y_BIAS = 0.0;

function clearSlotInlineHeatmaps(slot) {
  if (slot?._heatmapColEl) slot._heatmapColEl.replaceChildren();
}

function scheduleAnalysisMapResize() {
  if (!state.map || !isAnalysisViewActive()) return;
  requestAnimationFrame(() => {
    try { state.map.invalidateSize(); } catch {}
  });
  setTimeout(() => {
    try { state.map.invalidateSize(); } catch {}
  }, 140);
  window.dispatchEvent(new Event('resize'));
}

function syncInlineHeatmapPanelLayout() {
  const layout = el('layout');
  const panel = el('inline-heatmap-panel');
  if (!layout || !panel) return;
  const activeCols = [];
  const visible = !!state.inlineHeatmaps?.visible && !!state.advancedMode;
  for (const slot of state.tl?.athleteSlots || []) {
    const show = visible &&
      slot?.paneEl &&
      slot.paneEl.style.display !== 'none' &&
      slot.paneEl.classList.contains('show-heatmaps');
    if (slot?._heatmapColEl) {
      slot._heatmapColEl.style.display = show ? 'flex' : 'none';
      if (show) activeCols.push(slot._heatmapColEl);
    }
  }
  layout.classList.toggle('heatmaps-open', visible && activeCols.length > 0);
  const colW = 280;
  const gap = Math.max(0, activeCols.length - 1) * 6;
  const heatmapWidth = activeCols.length ? Math.min(680, activeCols.length * colW + gap + 10) : 0;
  const statOpen = layout.classList.contains('stat-panel-open');
  const statWidth = statOpen ? 300 : 0;
  layout.style.setProperty('--inline-heatmap-width', `${heatmapWidth}px`);
  layout.style.setProperty('--side-tools-width', `${heatmapWidth + statWidth}px`);
  scheduleAnalysisMapResize();
}

function setInlineHeatmapsVisible(visible) {
  const nextVisible = !!visible && !!state.advancedMode;
  state.inlineHeatmaps.visible = nextVisible;
  state.inlineHeatmaps.loadToken++;
  if (!nextVisible) {
    closeInlineHeatmapPlotOverlay();
    state.inlineHeatmaps.loading = false;
    state.inlineHeatmaps.renderedSegmentId = null;
    state.inlineHeatmaps.renderedLoadToken = 0;
  }

  for (const slot of state.tl?.athleteSlots || []) {
    if (slot?.paneEl) slot.paneEl.classList.toggle('show-heatmaps', nextVisible);
    if (!nextVisible) clearSlotInlineHeatmaps(slot);
  }

  syncInlineHeatmapPanelLayout();
  updateHeatmapButtonState();
  if (nextVisible) {
    syncInlineHeatmapsToCurrentSegment();
  }
  if (!nextVisible) {
    scheduleAnalysisMapResize();
  }
}

function renderInlineHeatmapStateForSlot(slot, title, body, color = null) {
  const col = slot?._heatmapColEl;
  if (!col) return;
  col.replaceChildren();

  const stateCard = document.createElement('section');
  stateCard.className = 'video-side-heatmap';
  stateCard.style.borderColor = rgbaFromHex(color || slot?.color || '#5cc8ff', 0.28);
  stateCard.style.boxShadow = `inset 0 0 0 1px ${rgbaFromHex(color || slot?.color || '#5cc8ff', 0.08)}`;

  const heading = document.createElement('div');
  heading.className = 'video-side-heatmap-title';
  heading.style.color = color || slot?.color || '#d9e6f2';
  heading.textContent = title;

  const empty = document.createElement('div');
  empty.className = 'video-side-heatmap-empty';
  empty.textContent = body;

  stateCard.append(heading, empty);
  col.appendChild(stateCard);
  queueInlineHeatmapScrollSync(col);
}

function renderInlineHeatmapLoading(seg, msg = 'Preparing report heatmaps...', pct = null) {
  const pctText = Number.isFinite(Number(pct)) ? `${Math.round(Number(pct) * 100)}%` : '';
  const status = pctText ? `${pctText} | ${msg}` : msg;
  for (const slot of state.tl?.athleteSlots || []) {
    if (!slot?.paneEl?.classList.contains('show-heatmaps') || slot.paneEl.style.display === 'none') continue;
    renderInlineHeatmapStateForSlot(
      slot,
      pctText ? `Building ${pctText}` : 'Building Heatmaps',
      status || `Loading ${seg?.name || 'segment'} heatmaps...`,
      slot?.color || '#5cc8ff',
    );
  }
  queueInlineHeatmapScrollSync();
}

const INLINE_HEATMAP_SUMMARY_DEFS = Object.freeze([
  { key: 'sog', label: 'Avg SOG', decimals: 1, suffix: ' kt' },
  { key: 'heel', label: 'Avg Heel', decimals: 1, suffix: '\u00B0' },
  { key: 'moment_roll', label: 'Avg RM', decimals: 0, suffix: ' Nm' },
  { key: 'trunk_angle', label: 'Avg TA', decimals: 1, suffix: '\u00B0' },
  { key: 'rudder', label: 'Avg RA', decimals: 1, suffix: '\u00B0' },
  { key: 'boom', label: 'Avg BA', decimals: 1, suffix: '\u00B0' },
]);

const INLINE_HEATMAP_PLOT_DEFS = Object.freeze([
  { key: 'trunk', title: 'Trunk Angle Distribution', xlabel: 'Trunk Angle', timelineKey: 'trunk_angle_timeline' },
  { key: 'rudder', title: 'Rudder Angle Distribution', xlabel: 'Rudder Angle', timelineKey: 'rudder_timeline' },
  { key: 'boom', title: 'Boom Angle Distribution', xlabel: 'Boom Angle', timelineKey: 'boom_timeline' },
  { key: 'roll', title: 'Rolling Moment Distribution', xlabel: 'Rolling Moment', timelineKey: 'moment_timeline', mapValue: v => Math.abs(Number(v)) },
  { key: 'heel', title: 'Heel Angle Distribution', xlabel: 'Heel Angle', timelineKey: 'heel_timeline' },
  { key: 'sog', title: 'SOG Distribution', xlabel: 'Speed Over Ground', timelineKey: 'sog_timeline' },
]);

const INLINE_HEATMAP_TILE_DEFS = Object.freeze([
  { key: 'keypoint', title: 'All keypoints', resultKey: 'keypoint_heatmap' },
  { key: 'com', title: 'Center of mass', resultKey: 'com_heatmap' },
]);

const INLINE_VMG_MODE_META = Object.freeze({
  upwind: { label: 'Upwind', tone: '#6fd0ff' },
  downwind: { label: 'Downwind', tone: '#f7b552' },
  reach: { label: 'Reach', tone: '#8fa1b4' },
});

let _inlineHeatmapScrollSyncLock = false;
let _inlineHeatmapScrollRatio = 0;
let _inlineHeatmapOverlayRenderToken = 0;

function getInlineHeatmapPlotDef(plotKey) {
  return INLINE_HEATMAP_PLOT_DEFS.find(def => def.key === plotKey) || null;
}

function getInlineHeatmapTileDef(tileKey) {
  return INLINE_HEATMAP_TILE_DEFS.find(def => def.key === tileKey) || null;
}

function getInlineHeatmapPlotValues(def, result) {
  if (!def) return [];
  const rawSeries = Array.isArray(result?.[def.timelineKey]) ? result[def.timelineKey] : [];
  return rawSeries
    .map(entry => (typeof def.mapValue === 'function' ? def.mapValue(entry?.v) : Number(entry?.v)))
    .filter(v => Number.isFinite(v));
}

function getInlineVmgModeMeta(mode) {
  return INLINE_VMG_MODE_META[String(mode || '').toLowerCase()] || INLINE_VMG_MODE_META.reach;
}

function findNearestInlineVmgPoint(points, targetT, maxGapSec = 6) {
  if (!Array.isArray(points) || !points.length || !Number.isFinite(Number(targetT))) return null;
  let lo = 0;
  let hi = points.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(points[mid]?.t) < Number(targetT)) lo = mid + 1;
    else hi = mid;
  }
  const prev = lo > 0 ? points[lo - 1] : null;
  const next = lo < points.length ? points[lo] : null;
  const best = !prev ? next : (!next ? prev : (Math.abs(Number(prev.t) - Number(targetT)) <= Math.abs(Number(next.t) - Number(targetT)) ? prev : next));
  if (!best) return null;
  return Math.abs(Number(best.t) - Number(targetT)) <= maxGapSec ? best : null;
}

function resolveInlineVmgSummaryMode(summary) {
  const dominant = String(summary?.dominant_mode || '').toLowerCase();
  if (dominant === 'upwind' || dominant === 'downwind') return dominant;
  const upwindCoverage = Number(summary?.upwind?.coverage_s) || 0;
  const downwindCoverage = Number(summary?.downwind?.coverage_s) || 0;
  if (upwindCoverage <= 0 && downwindCoverage <= 0) return 'reach';
  return upwindCoverage >= downwindCoverage ? 'upwind' : 'downwind';
}

function buildInlineHeatmapVmgKey(result) {
  const athleteId = normalizeAthleteId(result?.athlete_id);
  if (athleteId) return `id:${athleteId}`;
  return `name:${normalizeAthleteName(result?.athlete_name).toLowerCase()}`;
}

function formatInlineVmgValue(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} kt` : '--';
}

function formatInlineVmgGap(value) {
  if (!Number.isFinite(Number(value))) return '--';
  return Number(value) <= 0.05 ? 'Leader' : `-${Number(value).toFixed(1)} kt`;
}

function buildInlineHeatmapVmgContext(results, seg) {
  const rows = Array.isArray(results) ? results : [];
  const segStartTs = Number(seg?.tsStart);
  const segEndTs = Number(seg?.tsEnd);
  const currentTs = Number(state.tl?.currentTs);
  const relT = Number.isFinite(currentTs) && Number.isFinite(segStartTs) && Number.isFinite(segEndTs) && currentTs >= segStartTs && currentTs <= segEndTs
    ? (currentTs - segStartTs)
    : null;

  const baseRows = rows.map(result => {
    const currentPoint = relT == null ? null : findNearestInlineVmgPoint(result?.vmg_timeline, relT, 12);
    const localPoint = relT == null ? null : findNearestInlineVmgPoint(result?.vmg_local_timeline, relT, 12);
    const currentMode = String(currentPoint?.mode || resolveInlineVmgSummaryMode(result?.vmg_summary) || 'reach').toLowerCase();
    return {
      key: buildInlineHeatmapVmgKey(result),
      result,
      currentPoint,
      localPoint,
      currentMode: (currentMode === 'upwind' || currentMode === 'downwind') ? currentMode : 'reach',
    };
  });

  let compareMode = null;
  const liveModes = baseRows
    .map(row => String(row.currentPoint?.mode || '').toLowerCase())
    .filter(mode => mode === 'upwind' || mode === 'downwind');
  if (liveModes.length) {
    const upCount = liveModes.filter(mode => mode === 'upwind').length;
    compareMode = upCount >= (liveModes.length - upCount) ? 'upwind' : 'downwind';
  } else {
    const upwindCoverage = baseRows.reduce((sum, row) => sum + (Number(row.result?.vmg_summary?.upwind?.coverage_s) || 0), 0);
    const downwindCoverage = baseRows.reduce((sum, row) => sum + (Number(row.result?.vmg_summary?.downwind?.coverage_s) || 0), 0);
    if (upwindCoverage > 0 || downwindCoverage > 0) compareMode = upwindCoverage >= downwindCoverage ? 'upwind' : 'downwind';
  }
  if (compareMode) {
    const hasModeData = baseRows.some(row => !!row.result?.vmg_summary?.[compareMode]?.eligible && Number.isFinite(Number(row.result?.vmg_summary?.[compareMode]?.avg)));
    if (!hasModeData) {
      const fallbackMode = compareMode === 'upwind' ? 'downwind' : 'upwind';
      compareMode = baseRows.some(row => !!row.result?.vmg_summary?.[fallbackMode]?.eligible && Number.isFinite(Number(row.result?.vmg_summary?.[fallbackMode]?.avg)))
        ? fallbackMode
        : null;
    }
  }

  const mappedRows = baseRows.map(row => {
    const compareSummary = compareMode ? row.result?.vmg_summary?.[compareMode] : null;
    const liveValue = row.currentPoint && row.currentPoint.mode === compareMode && Number.isFinite(Number(row.currentPoint.v))
      ? Number(row.currentPoint.v)
      : null;
    const compareValue = liveValue != null
      ? liveValue
      : ((compareSummary?.eligible && Number.isFinite(Number(compareSummary?.avg))) ? Number(compareSummary.avg) : null);
    const bestStable = compareMode && Number.isFinite(Number(compareSummary?.best_stable))
      ? Number(compareSummary.best_stable)
      : null;
    const localValue = row.localPoint && row.localPoint.mode === compareMode && Number.isFinite(Number(row.localPoint.v))
      ? Number(row.localPoint.v)
      : null;
    return {
      ...row,
      compareMode,
      compareValue,
      bestStable,
      localValue,
      sourceLabel: liveValue != null
        ? `Now ${getInlineVmgModeMeta(compareMode).label}`
        : (compareMode ? `Avg ${getInlineVmgModeMeta(compareMode).label}` : 'VMG'),
    };
  });

  const leaderRow = mappedRows.reduce((best, row) => {
    if (!Number.isFinite(Number(row.compareValue))) return best;
    return (!best || row.compareValue > best.compareValue) ? row : best;
  }, null);
  const leaderValue = Number(leaderRow?.compareValue);
  const scaleMax = mappedRows.reduce((mx, row) => {
    const values = [mx, Number(row.compareValue), Number(row.bestStable)];
    return Math.max(...values.filter(Number.isFinite), 0);
  }, 0);
  const map = new Map();
  for (const row of mappedRows) {
    map.set(row.key, {
      ...row,
      leaderGap: Number.isFinite(leaderValue) && Number.isFinite(Number(row.compareValue))
        ? Math.max(0, leaderValue - Number(row.compareValue))
        : null,
      scalePct: scaleMax > 0 && Number.isFinite(Number(row.compareValue))
        ? Math.max(0, Math.min(100, (Number(row.compareValue) / scaleMax) * 100))
        : 0,
      bestPct: scaleMax > 0 && Number.isFinite(Number(row.bestStable))
        ? Math.max(0, Math.min(100, (Number(row.bestStable) / scaleMax) * 100))
        : 0,
    });
  }
  return {
    compareMode,
    leaderKey: leaderRow?.key || null,
    leaderValue: Number.isFinite(leaderValue) ? leaderValue : null,
    scaleMax,
    relT,
    rows: map,
  };
}

function createInlineHeatmapVmgCard(result, color, vmgContext) {
  const card = document.createElement('section');
  card.className = 'video-side-heatmap video-side-vmg';
  card.style.borderColor = rgbaFromHex(color || '#5cc8ff', 0.28);
  card.style.boxShadow = `inset 0 0 0 1px ${rgbaFromHex(color || '#5cc8ff', 0.08)}`;

  const heading = document.createElement('div');
  heading.className = 'video-side-heatmap-title';
  heading.textContent = 'VMG';
  heading.style.color = color || '#d9e6f2';
  card.appendChild(heading);

  const row = vmgContext?.rows?.get(buildInlineHeatmapVmgKey(result));
  if (!row || !vmgContext?.compareMode) {
    const empty = document.createElement('div');
    empty.className = 'video-side-heatmap-empty';
    empty.textContent = 'VMG needs a scored upwind or downwind section with enough confident wind data.';
    card.appendChild(empty);
    return card;
  }

  const currentMeta = getInlineVmgModeMeta(row.currentMode);
  const compareMeta = getInlineVmgModeMeta(vmgContext.compareMode);

  const top = document.createElement('div');
  top.className = 'video-side-vmg-top';

  const modePill = document.createElement('span');
  modePill.className = `video-side-vmg-pill ${row.currentMode === 'reach' ? 'is-neutral' : ''}`;
  modePill.textContent = currentMeta.label;
  modePill.style.borderColor = rgbaFromHex(currentMeta.tone, row.currentMode === 'reach' ? 0.14 : 0.32);
  modePill.style.color = currentMeta.tone;

  const source = document.createElement('span');
  source.className = 'video-side-vmg-source';
  source.textContent = row.sourceLabel;

  top.append(modePill, source);
  card.appendChild(top);

  const main = document.createElement('div');
  main.className = 'video-side-vmg-main';
  main.textContent = formatInlineVmgValue(row.compareValue);
  main.style.color = Number.isFinite(Number(row.compareValue)) ? (color || '#d9e6f2') : '#8fa1b4';
  card.appendChild(main);

  const sub = document.createElement('div');
  sub.className = 'video-side-vmg-sub';
  sub.textContent = Number.isFinite(Number(vmgContext.scaleMax))
    ? `${compareMeta.label} scale | leader ${formatInlineVmgValue(vmgContext.leaderValue)}`
    : 'Neutral while reaching';
  card.appendChild(sub);

  const meter = document.createElement('div');
  meter.className = 'video-side-vmg-meter';

  const bestMarker = document.createElement('div');
  bestMarker.className = 'video-side-vmg-best-marker';
  bestMarker.style.left = `${row.bestPct}%`;
  if (!Number.isFinite(Number(row.bestStable))) bestMarker.style.display = 'none';
  meter.appendChild(bestMarker);

  const fill = document.createElement('div');
  fill.className = `video-side-vmg-fill ${row.currentMode === 'reach' ? 'is-neutral' : ''}`;
  fill.style.width = `${row.scalePct}%`;
  fill.style.background = `linear-gradient(90deg, ${rgbaFromHex(compareMeta.tone, 0.34)}, ${compareMeta.tone})`;
  meter.appendChild(fill);
  card.appendChild(meter);

  const metaGrid = document.createElement('div');
  metaGrid.className = 'video-side-vmg-meta';
  const metaRows = [
    ['Gap', formatInlineVmgGap(row.leaderGap)],
    ['Best stable', formatInlineVmgValue(row.bestStable)],
    ['Local now', formatInlineVmgValue(row.localValue)],
  ];
  for (const [labelText, valueText] of metaRows) {
    const item = document.createElement('div');
    item.className = 'video-side-vmg-meta-item';
    const key = document.createElement('div');
    key.className = 'video-side-vmg-meta-key';
    key.textContent = labelText;
    const value = document.createElement('div');
    value.className = 'video-side-vmg-meta-val';
    value.textContent = valueText;
    if (labelText === 'Gap' && valueText === 'Leader') value.style.color = color || '#d9e6f2';
    item.append(key, value);
    metaGrid.appendChild(item);
  }
  card.appendChild(metaGrid);
  return card;
}

function isInlineHeatmapPlotOverlayOpen() {
  return !!state.inlineHeatmaps?.expandedOverlayKey && !!el('inline-plot-overlay')?.classList.contains('open');
}

function getVisibleInlineHeatmapPlotSeries(plotKey) {
  const def = getInlineHeatmapPlotDef(plotKey);
  if (!def) return [];

  const seg = getSegmentById(state.inlineHeatmaps?.segmentId) || getSegmentAtTs(state.tl?.currentTs);
  const segmentAthletes = getSegmentAthletes(seg);
  const results = Array.isArray(state.inlineHeatmaps?.results) ? state.inlineHeatmaps.results : [];
  const rows = [];

  for (const slot of state.tl?.athleteSlots || []) {
    if (!slot?._heatmapColEl) continue;
    if (!slot?.paneEl?.classList.contains('show-heatmaps')) continue;
    if (slot.paneEl.style.display === 'none') continue;
    const result = findHeatmapResultForSlot(slot, results);
    rows.push({
      athleteName: normalizeAthleteName(result?.athlete_name) || normalizeAthleteName(slot?.name) || 'Athlete',
      color: result ? resolveHeatmapAthleteColor(result, seg, segmentAthletes) : (slot?.color || '#5cc8ff'),
      values: getInlineHeatmapPlotValues(def, result),
    });
  }
  return rows;
}

function getVisibleInlineHeatmapTileSeries(tileKey) {
  const def = getInlineHeatmapTileDef(tileKey);
  if (!def) return [];

  const seg = getSegmentById(state.inlineHeatmaps?.segmentId) || getSegmentAtTs(state.tl?.currentTs);
  const segmentAthletes = getSegmentAthletes(seg);
  const results = Array.isArray(state.inlineHeatmaps?.results) ? state.inlineHeatmaps.results : [];
  const rows = [];

  for (const slot of state.tl?.athleteSlots || []) {
    if (!slot?._heatmapColEl) continue;
    if (!slot?.paneEl?.classList.contains('show-heatmaps')) continue;
    if (slot.paneEl.style.display === 'none') continue;
    const result = findHeatmapResultForSlot(slot, results);
    rows.push({
      athleteName: normalizeAthleteName(result?.athlete_name) || normalizeAthleteName(slot?.name) || 'Athlete',
      color: result ? resolveHeatmapAthleteColor(result, seg, segmentAthletes) : (slot?.color || '#5cc8ff'),
      hm: result?.[def.resultKey] || null,
    });
  }
  return rows;
}

function closeInlineHeatmapPlotOverlay() {
  _inlineHeatmapOverlayRenderToken++;
  state.inlineHeatmaps.expandedOverlayType = null;
  state.inlineHeatmaps.expandedOverlayKey = null;
  const overlay = el('inline-plot-overlay');
  const grid = el('inline-plot-overlay-grid');
  document.body.classList.remove('inline-plot-overlay-open');
  if (grid) {
    grid.replaceChildren();
    grid.style.gridTemplateColumns = '';
  }
  if (overlay) overlay.classList.remove('open');
}

async function renderInlineHeatmapPlotOverlay() {
  const overlayType = state.inlineHeatmaps?.expandedOverlayType;
  const overlayKey = state.inlineHeatmaps?.expandedOverlayKey;
  const overlay = el('inline-plot-overlay');
  const titleEl = el('inline-plot-overlay-title');
  const subtitleEl = el('inline-plot-overlay-subtitle');
  const grid = el('inline-plot-overlay-grid');
  if (!overlay || !titleEl || !subtitleEl || !grid) return;
  const renderToken = ++_inlineHeatmapOverlayRenderToken;

  let def = null;
  let rows = [];
  if (overlayType === 'heatmap') {
    def = getInlineHeatmapTileDef(overlayKey);
    rows = getVisibleInlineHeatmapTileSeries(overlayKey);
  } else {
    def = getInlineHeatmapPlotDef(overlayKey);
    rows = getVisibleInlineHeatmapPlotSeries(overlayKey);
  }
  if (!def || !rows.length) {
    closeInlineHeatmapPlotOverlay();
    return;
  }

  const seg = getSegmentById(state.inlineHeatmaps?.segmentId) || getSegmentAtTs(state.tl?.currentTs);
  const segName = normalizeAthleteName(seg?.name) || 'Current segment';
  titleEl.textContent = def.title;
  subtitleEl.textContent = rows.length > 1 ? `${segName} | ${rows.length} athletes` : segName;

  grid.replaceChildren();
  grid.style.gridTemplateColumns = `repeat(${rows.length}, minmax(min(560px, 100%), 1fr))`;

  for (const row of rows) {
    const card = document.createElement('section');
    card.className = 'inline-plot-overlay-card';
    card.style.borderColor = rgbaFromHex(row.color || '#5cc8ff', 0.28);
    card.style.boxShadow = `inset 0 0 0 1px ${rgbaFromHex(row.color || '#5cc8ff', 0.08)}`;

    const head = document.createElement('div');
    head.className = 'inline-plot-overlay-card-head';

    const dot = document.createElement('span');
    dot.className = 'inline-plot-overlay-card-dot';
    dot.style.background = row.color || '#5cc8ff';

    const heading = document.createElement('div');
    heading.className = 'inline-plot-overlay-card-heading';

    const nameEl = document.createElement('div');
    nameEl.className = 'inline-plot-overlay-card-name';
    nameEl.textContent = row.athleteName;

    const metaEl = document.createElement('div');
    metaEl.className = 'inline-plot-overlay-card-meta';
    if (overlayType === 'heatmap') {
      const hm = row.hm;
      const gridX = Number(hm?.grid_size_x);
      const gridY = Number(hm?.grid_size_y);
      metaEl.textContent = (Number.isFinite(gridX) && Number.isFinite(gridY))
        ? `${gridX.toFixed(1)}m x ${gridY.toFixed(1)}m window`
        : 'No processed heatmap available';
    } else {
      metaEl.textContent = row.values.length >= 3
        ? `${row.values.length.toLocaleString()} samples`
        : 'Not enough processed data';
    }

    heading.append(nameEl, metaEl);
    head.append(dot, heading);
    card.appendChild(head);

    if (overlayType === 'heatmap') {
      if (!row.hm) {
        const empty = document.createElement('div');
        empty.className = 'inline-plot-overlay-card-empty';
        empty.textContent = 'No processed pose data was available for this heatmap in the current segment.';
        card.appendChild(empty);
      } else {
        const media = document.createElement('div');
        media.className = 'inline-plot-overlay-card-media';

        const canvas = document.createElement('canvas');
        const canvasWidth = 720;
        const canvasHeight = Math.max(920, Math.round(canvasWidth * ((Number(row.hm.grid_size_x) || 5.0) / Math.max(0.1, Number(row.hm.grid_size_y) || 3.0))));
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        canvas.setAttribute('aria-label', `${row.athleteName} ${def.title}`);
        canvas.dataset.inlineOverlayHeatmap = '1';
        canvas._inlineHeatmap = row.hm;
        canvas._inlineHeatmapColor = row.color || '#5cc8ff';
        canvas._inlineHeatmapTitle = def.title;

        media.appendChild(canvas);
        card.appendChild(media);
        card.appendChild(createHeatmapLegend(row.hm, row.color));
      }
    } else if (row.values.length < 3) {
      const empty = document.createElement('div');
      empty.className = 'inline-plot-overlay-card-empty';
      empty.textContent = 'Not enough data in this segment.';
      card.appendChild(empty);
    } else {
      const media = document.createElement('div');
      media.className = 'inline-plot-overlay-card-media';

      const img = document.createElement('img');
      img.alt = `${row.athleteName} ${def.title}`;
      img.decoding = 'async';
      img.src = renderKdeHistogram(
        [{ values: row.values, label: row.athleteName, color: row.color }],
        null,
        def.xlabel,
        def.title,
      );
      media.appendChild(img);
      card.appendChild(media);
    }

    grid.appendChild(card);
  }

  overlay.classList.add('open');
  document.body.classList.add('inline-plot-overlay-open');

  if (overlayType !== 'heatmap') return;

  let deps = null;
  try {
    deps = await loadHeatmapRenderDeps();
  } catch (err) {
    console.warn('[heatmaps] expanded overlay WebGL deps unavailable, using flat fallback:', err);
  }

  const canvases = [...grid.querySelectorAll('canvas[data-inline-overlay-heatmap="1"]')];
  for (let idx = 0; idx < canvases.length; idx++) {
    if (renderToken !== _inlineHeatmapOverlayRenderToken) return;
    const canvas = canvases[idx];
    const hm = canvas?._inlineHeatmap;
    const color = canvas?._inlineHeatmapColor || '#5cc8ff';
    const title = canvas?._inlineHeatmapTitle || def.title;
    if (!hm) continue;
    try {
      await renderInlineHeatmapCanvas(canvas, { canvas, hm, color, title }, deps);
    } catch (err) {
      console.warn('[heatmaps] expanded overlay render failed:', err);
    }
    if (idx + 1 < canvases.length) await nextAnimationFrame();
  }
}

function openInlineHeatmapPlotOverlay(plotKey) {
  const def = getInlineHeatmapPlotDef(plotKey);
  if (!def) return;
  state.inlineHeatmaps.expandedOverlayType = 'plot';
  state.inlineHeatmaps.expandedOverlayKey = def.key;
  renderInlineHeatmapPlotOverlay().catch(err => {
    console.warn('[heatmaps] failed to render expanded overlay:', err);
  });
}

function openInlineHeatmapTileOverlay(tileKey) {
  const def = getInlineHeatmapTileDef(tileKey);
  if (!def) return;
  state.inlineHeatmaps.expandedOverlayType = 'heatmap';
  state.inlineHeatmaps.expandedOverlayKey = def.key;
  renderInlineHeatmapPlotOverlay().catch(err => {
    console.warn('[heatmaps] failed to render expanded overlay:', err);
  });
}

function getVisibleInlineHeatmapCols() {
  const cols = [];
  for (const slot of state.tl?.athleteSlots || []) {
    const col = slot?._heatmapColEl;
    if (!col) continue;
    if (!slot?.paneEl?.classList.contains('show-heatmaps')) continue;
    if (slot.paneEl.style.display === 'none') continue;
    cols.push(col);
  }
  return cols;
}

function syncInlineHeatmapScrollPositions(sourceCol = null) {
  const cols = getVisibleInlineHeatmapCols();
  if (cols.length < 2) return;
  const source = cols.includes(sourceCol) ? sourceCol : cols[0];
  if (!source) return;
  const sourceMax = Math.max(0, source.scrollHeight - source.clientHeight);
  const ratio = sourceMax > 0 ? source.scrollTop / sourceMax : _inlineHeatmapScrollRatio;
  _inlineHeatmapScrollRatio = Math.max(0, Math.min(1, Number.isFinite(ratio) ? ratio : 0));

  _inlineHeatmapScrollSyncLock = true;
  try {
    for (const col of cols) {
      if (col === source) continue;
      const targetMax = Math.max(0, col.scrollHeight - col.clientHeight);
      col.scrollTop = targetMax > 0 ? targetMax * _inlineHeatmapScrollRatio : 0;
    }
  } finally {
    _inlineHeatmapScrollSyncLock = false;
  }
}

function queueInlineHeatmapScrollSync(sourceCol = null) {
  nextAnimationFrame().then(() => {
    if (_inlineHeatmapScrollSyncLock) return;
    syncInlineHeatmapScrollPositions(sourceCol);
  }).catch(() => {});
}

function initInlineHeatmapColSync(col) {
  if (!col || col.dataset.scrollSyncInit === '1') return;
  col.dataset.scrollSyncInit = '1';
  col.addEventListener('scroll', () => {
    if (_inlineHeatmapScrollSyncLock) return;
    syncInlineHeatmapScrollPositions(col);
  }, { passive: true });
}

function formatInlineHeatmapSummaryValue(statBlock, decimals = 1, suffix = '') {
  const avg = Number(statBlock?.avg);
  return Number.isFinite(avg) ? `${avg.toFixed(decimals)}${suffix}` : '--';
}

function createInlineHeatmapSectionLabel(text, color = null) {
  const label = document.createElement('div');
  label.className = 'video-side-section-label';
  label.textContent = text;
  if (color) label.style.color = color;
  return label;
}

function createInlineHeatmapSummaryGrid(result, color) {
  const grid = document.createElement('div');
  grid.className = 'video-side-stats-grid';
  for (const def of INLINE_HEATMAP_SUMMARY_DEFS) {
    const chip = document.createElement('div');
    chip.className = 'video-side-stat-chip';
    chip.style.borderColor = rgbaFromHex(color || '#5cc8ff', 0.16);

    const key = document.createElement('div');
    key.className = 'video-side-stat-key';
    key.textContent = def.label;

    const value = document.createElement('div');
    value.className = 'video-side-stat-value';
    value.textContent = formatInlineHeatmapSummaryValue(result?.[def.key], def.decimals, def.suffix);
    value.style.color = color || '#d9e6f2';

    chip.append(key, value);
    grid.appendChild(chip);
  }
  return grid;
}

function createInlineHeatmapPlotTile(def, values, athleteName, color, plotJobs) {
  const tile = document.createElement('section');
  tile.className = 'video-side-plot';
  tile.style.borderColor = rgbaFromHex(color || '#5cc8ff', 0.16);

  const title = document.createElement('div');
  title.className = 'video-side-plot-title';
  title.textContent = def.title;
  tile.appendChild(title);

  if (!Array.isArray(values) || values.length < 3) {
    const empty = document.createElement('div');
    empty.className = 'video-side-plot-empty';
    empty.textContent = 'Not enough data in this segment.';
    tile.appendChild(empty);
    return tile;
  }

  const media = document.createElement('div');
  media.className = 'video-side-plot-media';
  const img = document.createElement('img');
  img.alt = def.title;
  media.appendChild(img);
  tile.appendChild(media);

  tile.classList.add('expandable');
  tile.tabIndex = 0;
  tile.setAttribute('role', 'button');
  tile.setAttribute('aria-haspopup', 'dialog');
  tile.title = 'Click to expand this plot for all athletes';
  tile.addEventListener('click', () => openInlineHeatmapPlotOverlay(def.key));
  tile.addEventListener('keydown', e => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    openInlineHeatmapPlotOverlay(def.key);
  });

  plotJobs.push({
    img,
    plotKey: def.key,
    title: def.title,
    xlabel: def.xlabel,
    athleteName: athleteName || 'Athlete',
    color: color || '#5cc8ff',
    values,
  });
  return tile;
}

function buildInlineHeatmapStatsCard(result, color, plotJobs, vmgContext = null) {
  const card = document.createElement('section');
  card.className = 'video-side-heatmap video-side-heatmap-stats';
  card.style.borderColor = rgbaFromHex(color || '#5cc8ff', 0.28);
  card.style.boxShadow = `inset 0 0 0 1px ${rgbaFromHex(color || '#5cc8ff', 0.08)}`;

  const heading = document.createElement('div');
  heading.className = 'video-side-heatmap-title';
  heading.textContent = 'Stats';
  heading.style.color = color || '#d9e6f2';
  card.appendChild(heading);

  card.appendChild(createInlineHeatmapSummaryGrid(result, color));

  const plots = document.createElement('div');
  plots.className = 'video-side-plots';
  for (const def of INLINE_HEATMAP_PLOT_DEFS) {
    const values = getInlineHeatmapPlotValues(def, result);
    plots.appendChild(createInlineHeatmapPlotTile(def, values, result?.athlete_name, color, plotJobs));
  }
  card.appendChild(plots);
  return card;
}

function buildInlineHeatmapTile(parent, def, hm, color) {
  const tile = document.createElement('section');
  tile.className = 'video-side-heatmap';
  tile.style.borderColor = rgbaFromHex(color || '#5cc8ff', 0.28);
  tile.style.boxShadow = `inset 0 0 0 1px ${rgbaFromHex(color || '#5cc8ff', 0.08)}`;

  const heading = document.createElement('div');
  heading.className = 'video-side-heatmap-title';
  heading.textContent = def?.title || 'Heatmap';
  heading.style.color = color || '#d9e6f2';
  tile.appendChild(heading);

  if (!hm) {
    const empty = document.createElement('div');
    empty.className = 'video-side-heatmap-empty';
    empty.textContent = 'No processed pose data was available for this heatmap in the current segment.';
    tile.appendChild(empty);
    parent.appendChild(tile);
    return null;
  }

  const wrap = document.createElement('div');
  wrap.className = 'video-side-heatmap-canvas-wrap';

  const canvas = document.createElement('canvas');
  canvas.width = 440;
  canvas.height = Math.max(660, Math.round(440 * ((Number(hm.grid_size_x) || 5.0) / Math.max(0.1, Number(hm.grid_size_y) || 3.0))));
  canvas.setAttribute('aria-label', def?.title || 'Heatmap');
  wrap.appendChild(canvas);
  wrap.appendChild(createHeatmapLegend(hm, color));

  if (def?.key) {
    tile.classList.add('expandable');
    tile.tabIndex = 0;
    tile.setAttribute('role', 'button');
    tile.setAttribute('aria-haspopup', 'dialog');
    tile.title = 'Click to expand this heatmap for all athletes';
    tile.addEventListener('click', () => openInlineHeatmapTileOverlay(def.key));
    tile.addEventListener('keydown', e => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      e.preventDefault();
      openInlineHeatmapTileOverlay(def.key);
    });
  }

  tile.appendChild(wrap);
  parent.appendChild(tile);
  return { canvas, hm, color, title: def?.title || 'Heatmap', overlayKey: def?.key || null };
}

function findHeatmapResultForSlot(slot, results) {
  const rows = Array.isArray(results) ? results : [];
  const slotAthleteId = normalizeAthleteId(slot?.athleteId);
  if (slotAthleteId) {
    const byId = rows.find(result => normalizeAthleteId(result?.athlete_id) === slotAthleteId);
    if (byId) return byId;
  }

  const slotName = normalizeAthleteName(slot?.name).toLowerCase();
  if (slotName) {
    const byName = rows.find(result => normalizeAthleteName(result?.athlete_name).toLowerCase() === slotName);
    if (byName) return byName;
  }

  const slotFileIds = new Set((slot?.videos || []).map(video => String(video?.id || '')).filter(Boolean));
  const intersecting = rows.filter(result => {
    const fileIds = Array.isArray(result?.file_id) ? result.file_id : [result?.file_id];
    return fileIds.some(fileId => slotFileIds.has(String(fileId || '')));
  });
  return intersecting.length === 1 ? intersecting[0] : null;
}

function getInlineHeatmapViewBox(hm, canvasWidth, canvasHeight) {
  const gridSizeX = Number(hm?.grid_size_x) || 5.0;
  const gridSizeY = Number(hm?.grid_size_y) || 3.0;
  const gridCenterX = Number.isFinite(Number(hm?.grid_center_x)) ? Number(hm.grid_center_x) : -1.0;
  const gridCenterY = Number.isFinite(Number(hm?.grid_center_y)) ? Number(hm.grid_center_y) : 0.0;
  const planeMinFore = gridCenterX - gridSizeX / 2;
  const planeMaxFore = gridCenterX + gridSizeX / 2;
  const contentMinFore = Math.min(planeMinFore, INLINE_HEATMAP_HULL_BOUNDS.minFore);
  const contentMaxFore = Math.max(planeMaxFore, INLINE_HEATMAP_HULL_BOUNDS.maxFore);
  const contentCenterFore = INLINE_HEATMAP_BOAT_CENTER_X + INLINE_HEATMAP_BOAT_CENTER_X_BIAS;
  const foreRadius = Math.max(
    Math.abs(contentMinFore - contentCenterFore),
    Math.abs(contentMaxFore - contentCenterFore),
  );
  const foreSpan = Math.max(0.1, (foreRadius * 2) + 0.5);
  const sideHalfSpan = Math.max(
    INLINE_HEATMAP_HULL_BOUNDS.maxSide,
    Math.abs(gridCenterY) + gridSizeY / 2,
  ) + 0.25;
  const sideSpan = Math.max(0.1, sideHalfSpan * 2);
  const contentAspect = sideSpan / foreSpan;
  const availW = Math.max(40, canvasWidth - 10);
  const availH = Math.max(40, canvasHeight - 10);
  let drawW = availW;
  let drawH = drawW / Math.max(0.001, contentAspect);
  if (drawH > availH) {
    drawH = availH;
    drawW = drawH * contentAspect;
  }
  const offsetX = (canvasWidth - drawW) / 2;
  const offsetY = (canvasHeight - drawH) / 2;
  const sideMin = -sideHalfSpan;
  const sideMax = sideHalfSpan;
  const foreMin = contentCenterFore - foreSpan / 2;
  const foreMax = contentCenterFore + foreSpan / 2;

  return {
    drawX: offsetX,
    drawY: offsetY,
    drawW,
    drawH,
    sideMin,
    sideMax,
    foreMin,
    foreMax,
    contentCenterFore,
    foreSpan,
    sideSpan,
  };
}

function computeInlineHeatmapFrustum(job, renderWidth, renderHeight) {
  const hm = job?.hm || {};
  const width = Number(renderWidth) || 360;
  const height = Number(renderHeight) || 600;
  const aspect = Math.max(0.25, width / Math.max(1, height));
  const view = getInlineHeatmapViewBox(hm, width, height);
  return {
    frustumSize: (Math.max(view.sideSpan, view.foreSpan / aspect) + 0.25) * INLINE_HEATMAP_CAMERA_ZOOM_OUT,
    targetX: view.contentCenterFore,
  };
}

function traceInlineRoundedRect(ctx, x, y, width, height, radius = 10) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function drawInlineHeatmapHullBowUp(ctx, viewport) {
  const projSideX = side => viewport.drawX + ((side - viewport.sideMin) / Math.max(0.001, viewport.sideMax - viewport.sideMin)) * viewport.drawW;
  const projForeY = fore => viewport.drawY + ((viewport.foreMax - fore) / Math.max(0.001, viewport.foreMax - viewport.foreMin)) * viewport.drawH;

  ctx.save();
  ctx.strokeStyle = 'rgba(216,226,238,0.80)';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let i = 0; i < INLINE_HEATMAP_HULL_X.length; i++) {
    const px = projSideX(INLINE_HEATMAP_HULL_Y[i]);
    const py = projForeY(INLINE_HEATMAP_HULL_X[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = 'rgba(216,226,238,0.28)';
  ctx.lineWidth = 0.75;
  const cx = projSideX(0);
  ctx.beginPath();
  ctx.moveTo(cx, projForeY(-5.0));
  ctx.lineTo(cx, projForeY(1.0));
  ctx.stroke();
  ctx.restore();
}

async function renderInlineHeatmapCanvas(canvas, job, deps = null) {
  if (!canvas || !job?.hm) return false;
  if (deps) {
    try {
      const width = canvas.width || 360;
      const height = canvas.height || 600;
      const viewport = getInlineHeatmapViewBox(job.hm, width, height);
      const sourceCanvas = document.createElement('canvas');
      sourceCanvas.width = Math.max(440, Math.round(viewport.drawH));
      sourceCanvas.height = Math.max(264, Math.round(viewport.drawW));

      const { frustumSize, targetX } = computeInlineHeatmapFrustum(job, sourceCanvas.width, sourceCanvas.height);
      await renderHeatmapWithBoat(sourceCanvas, job, frustumSize, deps, { targetX });

      const ctx3d = canvas.getContext('2d');
      if (!ctx3d) return false;
      ctx3d.clearRect(0, 0, width, height);
      ctx3d.fillStyle = '#0b0f14';
      ctx3d.fillRect(0, 0, width, height);

      const bg3d = ctx3d.createLinearGradient(0, 0, 0, height);
      bg3d.addColorStop(0, 'rgba(19,28,38,0.92)');
      bg3d.addColorStop(1, 'rgba(8,12,18,0.98)');
      ctx3d.fillStyle = bg3d;
      ctx3d.fillRect(0, 0, width, height);

      ctx3d.save();
      ctx3d.fillStyle = 'rgba(255,255,255,0.02)';
      ctx3d.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx3d.lineWidth = 1;
      traceInlineRoundedRect(ctx3d, viewport.drawX, viewport.drawY, viewport.drawW, viewport.drawH, 10);
      ctx3d.fill();
      ctx3d.stroke();
      ctx3d.clip();

      const scale = Math.min(
        viewport.drawW / Math.max(1, sourceCanvas.height),
        viewport.drawH / Math.max(1, sourceCanvas.width),
      ) * INLINE_HEATMAP_DRAW_INSET;
      ctx3d.translate(
        viewport.drawX + viewport.drawW / 2,
        viewport.drawY + viewport.drawH / 2 + (viewport.drawH * INLINE_HEATMAP_SCREEN_Y_BIAS),
      );
      ctx3d.rotate(-Math.PI / 2);
      ctx3d.drawImage(
        sourceCanvas,
        -sourceCanvas.width * scale / 2,
        -sourceCanvas.height * scale / 2,
        sourceCanvas.width * scale,
        sourceCanvas.height * scale,
      );
      ctx3d.restore();

      return true;
    } catch (err) {
      console.warn('[heatmaps] inline 3D render failed, using flat fallback:', err);
    }
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) return false;

  const width = canvas.width || 360;
  const height = canvas.height || 600;
  const viewport = getInlineHeatmapViewBox(job.hm, width, height);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0, 0, width, height);

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, 'rgba(19,28,38,0.92)');
  bg.addColorStop(1, 'rgba(8,12,18,0.98)');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.02)';
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  traceInlineRoundedRect(ctx, viewport.drawX, viewport.drawY, viewport.drawW, viewport.drawH, 10);
  ctx.fill();
  ctx.stroke();
  ctx.clip();

  const textureCanvas = await buildPdfStyleHeatmapTexture(job, Math.round(viewport.drawH), Math.round(viewport.drawW));
  if (textureCanvas) {
    ctx.save();
    ctx.translate(viewport.drawX + viewport.drawW / 2, viewport.drawY + viewport.drawH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.drawImage(textureCanvas, -viewport.drawH / 2, -viewport.drawW / 2, viewport.drawH, viewport.drawW);
    ctx.restore();
  }
  ctx.restore();

  const glow = ctx.createLinearGradient(0, 0, 0, height);
  const [r, g, b] = heatmapRgbFromHex(job.color || '#5cc8ff');
  glow.addColorStop(0, `rgba(${r},${g},${b},0.10)`);
  glow.addColorStop(0.5, 'rgba(255,255,255,0)');
  glow.addColorStop(1, `rgba(${r},${g},${b},0.05)`);
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);

  drawInlineHeatmapHullBowUp(ctx, viewport);
  return true;
}

function renderInlineHeatmapsForResults(seg, results, loadToken) {
  if (loadToken !== state.inlineHeatmaps?.loadToken || !state.inlineHeatmaps?.visible) return;
  state.inlineHeatmaps.renderedSegmentId = seg?.id != null ? String(seg.id) : null;
  state.inlineHeatmaps.renderedLoadToken = loadToken;
  const segmentAthletes = getSegmentAthletes(seg);
  const heatmapJobs = [];
  const plotJobs = [];
  for (const slot of state.tl?.athleteSlots || []) {
    if (!slot?._heatmapColEl) continue;
    const result = findHeatmapResultForSlot(slot, results);
    if (!result) {
      renderInlineHeatmapStateForSlot(
        slot,
        slot?.name || 'Heatmaps',
        'No processed segment heatmaps are available for this video yet.',
        slot?.color || '#5cc8ff',
      );
      continue;
    }

    const color = resolveHeatmapAthleteColor(result, seg, segmentAthletes);
    slot._heatmapColEl.replaceChildren();
    slot._heatmapColEl.appendChild(createInlineHeatmapSectionLabel('Heatmaps', color));
    const kpJob = buildInlineHeatmapTile(slot._heatmapColEl, getInlineHeatmapTileDef('keypoint'), result?.keypoint_heatmap, color);
    const comJob = buildInlineHeatmapTile(slot._heatmapColEl, getInlineHeatmapTileDef('com'), result?.com_heatmap, color);
    if (kpJob) heatmapJobs.push(kpJob);
    if (comJob) heatmapJobs.push(comJob);
    slot._heatmapColEl.appendChild(createInlineHeatmapSectionLabel('Stats', color));
    slot._heatmapColEl.appendChild(buildInlineHeatmapStatsCard(result, color, plotJobs, null));
  }
  queueInlineHeatmapScrollSync();
  if (state.inlineHeatmaps?.expandedOverlayKey) {
    renderInlineHeatmapPlotOverlay().catch(err => {
      console.warn('[heatmaps] failed to refresh expanded overlay:', err);
    });
  }

  (async () => {
    let deps = null;
    try {
      deps = await loadHeatmapRenderDeps();
    } catch (err) {
      console.warn('[heatmaps] inline WebGL deps unavailable, using flat fallback:', err);
    }
    for (let idx = 0; idx < heatmapJobs.length; idx++) {
      if (loadToken !== state.inlineHeatmaps?.loadToken || !state.inlineHeatmaps?.visible) return;
      try {
        await renderInlineHeatmapCanvas(heatmapJobs[idx].canvas, heatmapJobs[idx], deps);
      } catch (err) {
        console.warn('[heatmaps] inline render failed:', err);
      }
      if (idx + 1 < heatmapJobs.length) await nextAnimationFrame();
    }
    for (let idx = 0; idx < plotJobs.length; idx++) {
      if (loadToken !== state.inlineHeatmaps?.loadToken || !state.inlineHeatmaps?.visible) return;
      try {
        plotJobs[idx].img.src = renderKdeHistogram(
          [{ values: plotJobs[idx].values, label: plotJobs[idx].athleteName, color: plotJobs[idx].color }],
          null,
          plotJobs[idx].xlabel,
          plotJobs[idx].title,
        );
      } catch (err) {
        console.warn('[heatmaps] inline distribution render failed:', err);
      }
      if (idx + 1 < plotJobs.length) await nextAnimationFrame();
    }
    queueInlineHeatmapScrollSync();
  })();
}

async function syncInlineHeatmapsToCurrentSegment(seg = (isAnalysisViewActive() ? getSegmentAtTs(state.tl?.currentTs) : null), inAnalysis = isAnalysisViewActive()) {
  const enabled = !!state.inlineHeatmaps?.visible && !!state.advancedMode && !!inAnalysis;

  for (const slot of state.tl?.athleteSlots || []) {
    if (slot?.paneEl) slot.paneEl.classList.toggle('show-heatmaps', enabled);
  }
  if (!enabled) {
    closeInlineHeatmapPlotOverlay();
    return;
  }

  if (!seg) {
    closeInlineHeatmapPlotOverlay();
    for (const slot of state.tl?.athleteSlots || []) {
      renderInlineHeatmapStateForSlot(
        slot,
        'Heatmaps Preview',
        'Move the timeline playhead into a segment to show the report heatmaps beside each video.',
        slot?.color || '#5cc8ff',
      );
    }
    state.inlineHeatmaps.segmentId = null;
    state.inlineHeatmaps.results = null;
    state.inlineHeatmaps.loading = false;
    state.inlineHeatmaps.renderedSegmentId = null;
    state.inlineHeatmaps.renderedLoadToken = 0;
    updateHeatmapButtonState(seg, inAnalysis);
    return;
  }

  const segId = String(seg.id);
  if (state.inlineHeatmaps?.expandedOverlayKey && state.inlineHeatmaps.segmentId && state.inlineHeatmaps.segmentId !== segId) {
    closeInlineHeatmapPlotOverlay();
  }
  if (state.inlineHeatmaps.segmentId === segId && Array.isArray(state.inlineHeatmaps.results)) {
    const alreadyRendered = state.inlineHeatmaps.renderedSegmentId === segId
      && state.inlineHeatmaps.renderedLoadToken === state.inlineHeatmaps.loadToken;
    if (!alreadyRendered) {
      renderInlineHeatmapsForResults(seg, state.inlineHeatmaps.results, state.inlineHeatmaps.loadToken);
    }
    updateHeatmapButtonState(seg, inAnalysis);
    return;
  }
  if (state.inlineHeatmaps.segmentId === segId && state.inlineHeatmaps.loading) {
    updateHeatmapButtonState(seg, inAnalysis);
    return;
  }

  const loadToken = ++state.inlineHeatmaps.loadToken;
  state.inlineHeatmaps.segmentId = segId;
  state.inlineHeatmaps.results = null;
  state.inlineHeatmaps.loading = true;
  state.inlineHeatmaps.renderedSegmentId = null;
  state.inlineHeatmaps.renderedLoadToken = 0;
  renderInlineHeatmapLoading(seg, 'Collecting report heatmaps...', 0);
  updateHeatmapButtonState(seg, inAnalysis);

  try {
    renderInlineHeatmapLoading(seg, 'Collecting segment telemetry...', 0.02);
    await ensureWindEstimatesReady();
    const reportData = await buildReportData(
      state.projectId,
      [segId],
      (msg, pct) => {
        if (loadToken !== state.inlineHeatmaps?.loadToken || !state.inlineHeatmaps?.visible) return;
        renderInlineHeatmapLoading(seg, msg, pct);
      },
      {
        includeDensityImages: true,
        includeLegacyVisuals: false,
        wind: buildReportWindContext(),
      },
    );
    if (loadToken !== state.inlineHeatmaps?.loadToken || !state.inlineHeatmaps?.visible) return;
    const segmentResults = Array.isArray(reportData?.segments)
      ? reportData.segments.filter(item => String(item?.split_id) === segId)
      : [];
    state.inlineHeatmaps.loading = false;
    state.inlineHeatmaps.results = segmentResults;
    renderInlineHeatmapsForResults(seg, segmentResults, loadToken);
  } catch (err) {
    if (loadToken !== state.inlineHeatmaps?.loadToken || !state.inlineHeatmaps?.visible) return;
    state.inlineHeatmaps.loading = false;
    console.error('[heatmaps] inline preview failed:', err);
    for (const slot of state.tl?.athleteSlots || []) {
      renderInlineHeatmapStateForSlot(
        slot,
        'Heatmaps Failed',
        err?.message || 'The segment heatmaps could not be generated for this video.',
        slot?.color || '#5cc8ff',
      );
    }
  }
}

const HEATMAP_DENSITY_LEVELS = Object.freeze([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]);

function heatmapRgbFromHex(hex) {
  if (typeof hex !== 'string' || !hex.startsWith('#') || (hex.length !== 7 && hex.length !== 4)) {
    return [92, 200, 255];
  }
  const full = hex.length === 4
    ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`
    : hex;
  return [
    parseInt(full.slice(1, 3), 16) || 0,
    parseInt(full.slice(3, 5), 16) || 0,
    parseInt(full.slice(5, 7), 16) || 0,
  ];
}

function buildHeatmapLegendGradient(color) {
  const [r, g, b] = heatmapRgbFromHex(color);
  return `linear-gradient(to right,
    rgba(${r},${g},${b},0.10) 0%,
    rgba(${r},${g},${b},0.28) 14%,
    rgba(${r},${g},${b},0.52) 34%,
    rgba(${r},${g},${b},0.78) 68%,
    rgba(${r},${g},${b},1.00) 100%)`;
}

function createHeatmapLegend(hm, color) {
  const legend = document.createElement('div');
  legend.className = 'heatmap-legend';

  const high = document.createElement('span');
  high.textContent = 'High';
  if (color) high.style.color = color;
  const gradient = document.createElement('div');
  gradient.className = 'gradient';
  if (color) gradient.style.background = buildHeatmapLegendGradient(color);
  const low = document.createElement('span');
  low.textContent = 'Low';
  const meta = document.createElement('span');
  meta.textContent = `${hm.grid_size_x}m x ${hm.grid_size_y}m | ${hm.point_count} pts`;

  legend.append(high, gradient, low, meta);
  return legend;
}

function clamp01(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  return Math.max(0, Math.min(1, num));
}

function estimateHeatmapDensityFromBuilderPixel(r, g, b, a) {
  const alpha = clamp01((Number(a) || 0) / 255);
  if (alpha <= 0.01) return 0;

  let density = null;
  if (r <= 6 && b >= 248 && g <= 228) {
    density = (g / 225) * 0.25;
  } else if (r <= 6 && g >= 248) {
    density = 0.25 + ((255 - b) / 255) * 0.25;
  } else if (g >= 248 && b <= 6 && r < 250) {
    density = 0.50 + (r / 255) * 0.25;
  } else if (r >= 248 && b <= 6) {
    density = 0.75 + ((255 - g) / 255) * 0.25;
  }

  if (!Number.isFinite(density)) {
    density = (alpha - 0.8) / 0.2;
  }
  return clamp01(density);
}

function quantizeHeatmapDensity(density) {
  const d = clamp01(density);
  let level = 0;
  for (const threshold of HEATMAP_DENSITY_LEVELS) {
    if (d >= threshold) level = threshold;
  }
  return level;
}

function remapHeatmapStrength(density) {
  const d = clamp01(density);
  if (d <= 0) return 0;
  // Lift lower densities so the preview reads clearly on the dark panel.
  return clamp01(Math.pow(d, 0.62));
}

function loadHeatmapSourceImage(job) {
  if (job?._heatmapSourceImagePromise) return job._heatmapSourceImagePromise;
  const src = resolveHeatmapImageSrc(job?.hm?.image_b64);
  if (!src) return Promise.resolve(null);
  job._heatmapSourceImagePromise = new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  }).catch(err => {
    job._heatmapSourceImagePromise = null;
    throw err;
  });
  return job._heatmapSourceImagePromise;
}

async function buildPdfStyleHeatmapTexture(job, targetW = 240, targetH = null) {
  const hm = job?.hm || {};
  const gridSizeX = Number(hm.grid_size_x) || 5.0;
  const gridSizeY = Number(hm.grid_size_y) || 3.0;
  const width = Math.max(120, Math.round(targetW || Number(hm.width) || 240));
  const height = Math.max(72, Math.round(targetH || (width * (gridSizeY / gridSizeX))));
  const color = job?.color || '#5cc8ff';
  const cacheKey = `${width}x${height}:${color}`;

  if (!job._styledHeatmapCache) job._styledHeatmapCache = {};
  if (job._styledHeatmapCache[cacheKey]) return job._styledHeatmapCache[cacheKey];

  const sourceImg = await loadHeatmapSourceImage(job);
  if (!sourceImg) return null;

  const textureCanvas = document.createElement('canvas');
  textureCanvas.width = width;
  textureCanvas.height = height;
  const ctx = textureCanvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(sourceImg, 0, 0, width, height);

  const imgData = ctx.getImageData(0, 0, width, height);
  const pixels = imgData.data;
  const [rBase, gBase, bBase] = heatmapRgbFromHex(color);

  for (let i = 0; i < pixels.length; i += 4) {
    const density = estimateHeatmapDensityFromBuilderPixel(
      pixels[i],
      pixels[i + 1],
      pixels[i + 2],
      pixels[i + 3],
    );
    const level = quantizeHeatmapDensity(density);
    if (level < HEATMAP_DENSITY_LEVELS[0]) {
      pixels[i] = 0;
      pixels[i + 1] = 0;
      pixels[i + 2] = 0;
      pixels[i + 3] = 0;
      continue;
    }
    const strength = remapHeatmapStrength(level);
    const tint = 0.30 + 0.70 * strength;
    pixels[i] = Math.round(255 + (rBase - 255) * tint);
    pixels[i + 1] = Math.round(255 + (gBase - 255) * tint);
    pixels[i + 2] = Math.round(255 + (bBase - 255) * tint);
    pixels[i + 3] = Math.round(255 * Math.min(1, 0.18 + 0.82 * strength));
  }

  ctx.putImageData(imgData, 0, 0);
  job._styledHeatmapCache[cacheKey] = textureCanvas;
  return textureCanvas;
}

async function renderFlatHeatmapFallback(canvas, job) {
  if (!canvas) return false;
  const ctx = canvas.getContext('2d');
  if (!ctx) return false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = job?.lightTheme ? '#f7fafc' : '#0b0f14';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const textureCanvas = await buildPdfStyleHeatmapTexture(job, canvas.width, canvas.height);
  if (!textureCanvas) return false;
  ctx.drawImage(textureCanvas, 0, 0, canvas.width, canvas.height);
  return true;
}

function appendHeatmapTile(parent, title, hm, points, color, renderJobs) {
  const tile = document.createElement('section');
  tile.className = 'heatmap-tile';
  tile.style.borderColor = rgbaFromHex(color || '#5cc8ff', 0.28);
  tile.style.boxShadow = `inset 0 0 0 1px ${rgbaFromHex(color || '#5cc8ff', 0.08)}`;

  const heading = document.createElement('div');
  heading.className = 'heatmap-tile-title';
  heading.textContent = title;
  heading.style.color = color || '#d9e6f2';
  tile.appendChild(heading);

  if (!hm) {
    const empty = document.createElement('div');
    empty.className = 'heatmap-empty-tile';
    empty.style.border = `1px solid ${rgbaFromHex(color || '#5cc8ff', 0.14)}`;
    empty.textContent = 'No processed pose data was available for this heatmap in the current segment.';
    tile.appendChild(empty);
    parent.appendChild(tile);
    return;
  }

  const wrap = document.createElement('div');
  wrap.className = 'heatmap-canvas-wrap';

  const canvas = document.createElement('canvas');
  canvas.width = 720;
  canvas.height = 432;
  canvas.setAttribute('aria-label', title);
  wrap.appendChild(canvas);
  wrap.appendChild(createHeatmapLegend(hm, color));

  tile.appendChild(wrap);
  parent.appendChild(tile);
  renderJobs.push({ canvas, hm, points, color, frustumSize: 4.4, wrap });
}

function buildHeatmapCard(result, seg, segmentAthletes, renderJobs) {
  const card = document.createElement('section');
  card.className = 'heatmap-card';

  const color = resolveHeatmapAthleteColor(result, seg, segmentAthletes);
  card.style.setProperty('--heatmap-accent', color);
  card.style.setProperty('--heatmap-accent-soft', rgbaFromHex(color, 0.18));
  card.style.borderColor = rgbaFromHex(color, 0.45);
  card.style.boxShadow = `0 12px 30px rgba(0,0,0,.22), 0 0 0 1px ${rgbaFromHex(color, 0.08)}`;

  const head = document.createElement('div');
  head.className = 'heatmap-card-head';

  const dot = document.createElement('span');
  dot.className = 'heatmap-card-dot';
  dot.style.background = color;

  const title = document.createElement('div');
  title.className = 'heatmap-card-title';
  title.textContent = result?.athlete_name || 'Athlete';

  const subtitle = document.createElement('div');
  subtitle.className = 'heatmap-card-subtitle';
  subtitle.textContent = `${Math.max(0, Number(seg?.tsEnd) - Number(seg?.tsStart)).toFixed(1)}s segment`;

  head.append(dot, title, subtitle);

  const body = document.createElement('div');
  body.className = 'heatmap-card-body';
  appendHeatmapTile(body, 'All keypoints', result?.keypoint_heatmap, result?.kp_xy, color, renderJobs);
  appendHeatmapTile(body, 'Center of mass', result?.com_heatmap, result?.com_xy, color, renderJobs);

  card.append(head, body);
  return card;
}

function loadHeatmapRenderDeps() {
  if (!_heatmapRenderDepsPromise) {
    _heatmapRenderDepsPromise = Promise.all([
      import(new URL('./vendor/three.module.js', import.meta.url).href),
      import(new URL('./vendor/STLLoader.js', import.meta.url).href),
    ])
      .then(([THREE, stl]) => ({ THREE, STLLoader: stl.STLLoader }))
      .catch(err => {
        _heatmapRenderDepsPromise = null;
        throw err;
      });
  }
  return _heatmapRenderDepsPromise;
}

function loadHeatmapHullGeometry(STLLoader) {
  if (!_heatmapHullGeomPromise) {
    _heatmapHullGeomPromise = new Promise(resolve => {
      new STLLoader().load(new URL('./Hull.stl', import.meta.url).href, geom => resolve(geom), undefined, () => resolve(null));
    });
  }
  return _heatmapHullGeomPromise;
}

async function renderHeatmapFallbackImage(wrap, canvas, job) {
  if (await renderFlatHeatmapFallback(canvas, job)) return;
  const hm = job?.hm;
  const src = resolveHeatmapImageSrc(hm?.image_b64);
  if (!src || !wrap || !canvas) return;
  const img = document.createElement('img');
  img.alt = 'Heatmap preview';
  img.style.display = 'block';
  img.style.width = '100%';
  img.style.height = 'auto';
  img.style.aspectRatio = job?.lightTheme ? '16 / 9' : '5 / 3';
  img.style.borderRadius = '8px';
  img.style.background = job?.lightTheme ? '#f7fafc' : '#0b0f14';
  await new Promise((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = reject;
    img.src = src;
  });
  if (canvas.parentNode === wrap) wrap.replaceChild(img, canvas);
}

async function renderHeatmapWithBoat(canvas, job, frustumSize, deps, renderOpts = null) {
  const { THREE, STLLoader } = deps || await loadHeatmapRenderDeps();
  const opts = renderOpts || {};
  const hm = job?.hm || {};
  const w = canvas.width || 720;
  const h = canvas.height || 432;
  const targetX = Number.isFinite(Number(opts?.targetX)) ? Number(opts.targetX) : Number(hm.grid_center_x) || 0;
  const backgroundColor = Number.isFinite(Number(opts?.backgroundColor)) ? Number(opts.backgroundColor) : 0x0b0f14;
  const gridColor = Number.isFinite(Number(opts?.gridColor)) ? Number(opts.gridColor) : 0x1e2430;
  const ambientIntensity = Number.isFinite(Number(opts?.ambientIntensity)) ? Number(opts.ambientIntensity) : 0.7;
  const directionalIntensity = Number.isFinite(Number(opts?.directionalIntensity)) ? Number(opts.directionalIntensity) : 0.5;
  const boatColor = Number.isFinite(Number(opts?.boatColor)) ? Number(opts.boatColor) : 0xd9e6f2;
  const boatEmissive = Number.isFinite(Number(opts?.boatEmissive)) ? Number(opts.boatEmissive) : 0x1e2b38;
  const boatEmissiveIntensity = Number.isFinite(Number(opts?.boatEmissiveIntensity)) ? Number(opts.boatEmissiveIntensity) : 0.32;
  const boatOpacity = Number.isFinite(Number(opts?.boatOpacity)) ? Number(opts.boatOpacity) : 0.88;
  const cameraUp = Array.isArray(opts?.cameraUp) && opts.cameraUp.length === 3
    ? opts.cameraUp.map(v => Number(v) || 0)
    : [0, 0, -1];

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(backgroundColor);

  const aspect = w / h;
  const cam = new THREE.OrthographicCamera(
    -frustumSize * aspect / 2, frustumSize * aspect / 2,
    frustumSize / 2, -frustumSize / 2, 0.1, 100,
  );
  cam.position.set(targetX, 10, 0);
  cam.lookAt(targetX, 0, 0);
  cam.up.set(cameraUp[0], cameraUp[1], cameraUp[2]);

  scene.add(new THREE.AmbientLight(0xffffff, ambientIntensity));
  const dl = new THREE.DirectionalLight(0xffffff, directionalIntensity);
  dl.position.set(2, 8, -2);
  scene.add(dl);
  scene.add(new THREE.GridHelper(10, 20, gridColor, gridColor));

  const boatGeom = await loadHeatmapHullGeometry(STLLoader);
  let boatMaterial = null;
  if (boatGeom) {
    boatMaterial = new THREE.MeshPhongMaterial({
      color: boatColor,
      emissive: boatEmissive,
      emissiveIntensity: boatEmissiveIntensity,
      specular: 0xffffff,
      shininess: 95,
      transparent: true,
      opacity: boatOpacity,
    });
    const mesh = new THREE.Mesh(boatGeom, boatMaterial);
    mesh.scale.set(0.01, 0.01, 0.01);
    const rot = new THREE.Matrix4();
    rot.set(0,0,-1,0, 0,1,0,0, 1,0,0,0, 0,0,0,1);
    mesh.applyMatrix4(rot);
    mesh.position.set(-2.974, 0, 0);
    scene.add(mesh);
  }

  const textureSource = await buildPdfStyleHeatmapTexture(job, 340, Math.round(340 * ((Number(hm.grid_size_y) || 3.0) / (Number(hm.grid_size_x) || 5.0))))
    || await new Promise((resolve, reject) => {
      const src = resolveHeatmapImageSrc(hm?.image_b64);
      if (!src) {
        reject(new Error('No heatmap texture source available'));
        return;
      }
      const elImg = new Image();
      elImg.onload = () => resolve(elImg);
      elImg.onerror = reject;
      elImg.src = src;
    });
  const tex = textureSource instanceof HTMLCanvasElement
    ? new THREE.CanvasTexture(textureSource)
    : new THREE.Texture(textureSource);
  tex.needsUpdate = true;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;

  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(hm.grid_size_x, hm.grid_size_y),
    new THREE.MeshBasicMaterial({
      map: tex,
      transparent: true,
      opacity: 0.94,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.set(hm.grid_center_x, 0.01, -hm.grid_center_y);
  scene.add(plane);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  try {
    renderer.setSize(w, h, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.render(scene, cam);
  } finally {
    renderer.dispose();
    tex.dispose();
    plane.geometry.dispose();
    plane.material.dispose();
    if (boatMaterial) boatMaterial.dispose();
  }
}

async function renderHeatmapPreview(seg, results, loadToken) {
  if (loadToken !== state.advancedPane.heatmapLoadToken || state.advancedPane.mode !== 'heatmaps') return;
  const sortedResults = (results || []).slice().sort((a, b) => String(a?.athlete_name || '').localeCompare(String(b?.athlete_name || '')));
  if (!sortedResults.length) {
    setHeatmapPanelInfo(seg, { status: 'No segment heatmaps available yet', statusTone: 'error' });
    replaceHeatmapContent(createHeatmapState(
      'No Heatmaps Yet',
      'Process the videos that overlap this segment to populate the report-style heatmaps.',
      'error',
    ));
    return;
  }

  const renderJobs = [];
  const grid = document.createElement('div');
  grid.className = 'heatmap-grid';
  const segmentAthletes = getSegmentAthletes(seg);
  for (const result of sortedResults) {
    grid.appendChild(buildHeatmapCard(result, seg, segmentAthletes, renderJobs));
  }
  replaceHeatmapContent(grid);
  setHeatmapPanelInfo(seg, {
    status: renderJobs.length ? `Rendering ${renderJobs.length} heatmap${renderJobs.length === 1 ? '' : 's'}...` : 'No pose density yet',
    statusTone: renderJobs.length ? 'loading' : 'error',
  });

  if (!renderJobs.length) return;

  await nextAnimationFrame();
  let deps = null;
  try {
    deps = await loadHeatmapRenderDeps();
  } catch (err) {
    console.warn('[heatmaps] failed to load WebGL deps:', err);
  }

  for (let idx = 0; idx < renderJobs.length; idx++) {
    const job = renderJobs[idx];
    if (loadToken !== state.advancedPane.heatmapLoadToken || state.advancedPane.mode !== 'heatmaps') return;
    try {
      if (!deps) throw new Error('Heatmap render deps unavailable');
      await renderHeatmapWithBoat(job.canvas, job, job.frustumSize, deps);
    } catch (err) {
      console.warn('[heatmaps] 3D render failed, using image fallback:', err);
      try { await renderHeatmapFallbackImage(job.wrap, job.canvas, job); } catch {}
    }
    setHeatmapPanelInfo(seg, {
      status: `Rendered ${idx + 1}/${renderJobs.length} heatmaps`,
      statusTone: idx + 1 === renderJobs.length ? 'ready' : 'loading',
    });
    if (idx + 1 < renderJobs.length) await nextAnimationFrame();
  }

  setHeatmapPanelInfo(seg, {
    status: `${sortedResults.length} athlete${sortedResults.length === 1 ? '' : 's'} ready`,
    statusTone: 'ready',
  });
}

async function openHeatmapViewer({ forceReload = false, silentNoSegment = false } = {}) {
  if (!state.advancedMode) return;
  const modal = el('stl-inline');
  const seg = isAnalysisViewActive() ? getSegmentAtTs(state.tl?.currentTs) : null;
  const nextSegId = seg?.id != null ? String(seg.id) : null;

  if (
    modal?.classList.contains('open') &&
    state.advancedPane.mode === 'heatmaps' &&
    !forceReload &&
    nextSegId &&
    String(state.advancedPane.segmentId || '') === nextSegId
  ) {
    return;
  }

  await ensureAdvancedPaneOpen('heatmaps');
  const loadToken = ++state.advancedPane.heatmapLoadToken;
  state.advancedPane.segmentId = nextSegId;

  if (!seg) {
    setHeatmapPanelInfo(null, {
      status: silentNoSegment ? 'Waiting for a segment' : 'Move into a segment to preview heatmaps',
    });
    replaceHeatmapContent(createHeatmapState(
      'Heatmaps Preview',
      'Move the timeline playhead into a segment in Analysis, then open Heatmaps to preview that segment only.',
    ));
    return;
  }

  renderHeatmapLoadingState(seg, 'Collecting report data...', 0);

  try {
    renderHeatmapLoadingState(seg, 'Collecting segment telemetry...', 0.02);
    await ensureWindEstimatesReady();
    const reportData = await buildReportData(
      state.projectId,
      [String(seg.id)],
      (msg, pct) => {
        if (loadToken !== state.advancedPane.heatmapLoadToken || state.advancedPane.mode !== 'heatmaps') return;
        renderHeatmapLoadingState(seg, msg, pct);
      },
      {
        includeDensityImages: true,
        includeLegacyVisuals: false,
        wind: buildReportWindContext(),
      },
    );
    if (loadToken !== state.advancedPane.heatmapLoadToken || state.advancedPane.mode !== 'heatmaps') return;
    const segmentResults = Array.isArray(reportData?.segments)
      ? reportData.segments.filter(item => String(item?.split_id) === String(seg.id))
      : [];
    await renderHeatmapPreview(seg, segmentResults, loadToken);
  } catch (err) {
    if (loadToken !== state.advancedPane.heatmapLoadToken || state.advancedPane.mode !== 'heatmaps') return;
    console.error('[heatmaps] preview failed:', err);
    setHeatmapPanelInfo(seg, { status: 'Heatmap preview failed', statusTone: 'error' });
    replaceHeatmapContent(createHeatmapState(
      'Heatmaps Failed',
      err?.message || 'The heatmaps could not be generated for this segment.',
      'error',
    ));
  }
}

function closeStlViewer() {
  const modal = el('stl-inline');
  state.advancedPane.heatmapLoadToken++;
  state.advancedPane.segmentId = null;
  setAdvancedPaneMode(null);
  if (modal) modal.classList.remove('open');
  el('panel-left')?.classList.remove('stl-open');
  el('map').style.flex = '';
  el('stl-inline').style.flex = '';
  setTimeout(()=>{ if(state.map) state.map.invalidateSize(); }, 60);
}

async function openStlViewer() {
  if (!state.advancedMode) return;
  const modal = el('stl-inline');
  if(modal.classList.contains('open') && state.advancedPane.mode === 'stl' && _stlRenderer) return;

  // Always force-cleanup any previous WebGL state
  cleanupStlViewerRuntime();
  await ensureAdvancedPaneOpen('stl');
  await nextAnimationFrame();

  const {
    Scene, PerspectiveCamera, WebGLRenderer,
    AmbientLight, DirectionalLight,
    Mesh, MeshPhongMaterial, MeshStandardMaterial, Color,
    GridHelper, AxesHelper,
    BufferGeometry, BufferAttribute,
    LineSegments, LineBasicMaterial, Points, PointsMaterial,
    SphereGeometry, Matrix4, Vector3
  } = await import(new URL('./vendor/three.module.js', import.meta.url).href);
  const {STLLoader}      = await import(new URL('./vendor/STLLoader.js', import.meta.url).href);
  const {OrbitControls}  = await import(new URL('./vendor/OrbitControls.js', import.meta.url).href);
  if (!modal.classList.contains('open') || state.advancedPane.mode !== 'stl') return;

  const wrap = el('stl-3d-wrap');
  // Clear any orphaned canvases
  while(wrap.firstChild) wrap.removeChild(wrap.firstChild);
  const W = wrap.clientWidth  || wrap.offsetWidth  || 800;
  const H = wrap.clientHeight || wrap.offsetHeight || 600;

  const scene    = new Scene();
  scene.background = new Color(0x11151c);
  const camera   = new PerspectiveCamera(50, W / H, 0.01, 500);
  camera.position.set(2, 2, 3);

  let renderer;
  try {
    renderer = new WebGLRenderer({antialias: true, powerPreference: 'high-performance'});
  } catch(e) {
    console.error('WebGL context creation failed:', e);
    wrap.innerHTML = '<div style="padding:24px;color:#e74c3c;text-align:center;">WebGL not available.<br>Try closing other 3D tabs or restart your browser.</div>';
    return;
  }
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  wrap.appendChild(renderer.domElement);
  _stlRenderer = renderer;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(-1.5, 0.3, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  scene.add(new AmbientLight(0xffffff, 0.6));
  const dl = new DirectionalLight(0xffffff, 0.9);
  dl.position.set(4, 8, 4);
  scene.add(dl);

  const grid = new GridHelper(12, 12, 0x2b3448, 0x1b2231);
  scene.add(grid);
  scene.add(new AxesHelper(1.5));

  let sceneNeedsRender = true;
  controls.addEventListener('change', () => { sceneNeedsRender = true; });

  let hullMesh = null;
  const stlLoader = new STLLoader();
  stlLoader.load(new URL('./Hull.stl', import.meta.url).href, geom => {
    geom.computeVertexNormals();
    hullMesh = new Mesh(geom, new MeshPhongMaterial({
      color: 0x2a7fc4, specular: 0x112233, shininess: 120
    }));
    hullMesh.scale.set(0.01, 0.01, 0.01);
    const rotMat = new Matrix4();
    rotMat.set(0,0,-1,0, 0,1,0,0, 1,0,0,0, 0,0,0,1);
    hullMesh.applyMatrix4(rotMat);
    hullMesh.position.set(-2.974, 0, 0);
    scene.add(hullMesh);
    sceneNeedsRender = true;
  }, undefined, err => console.warn('Hull STL load failed:', err));

  // ── Multi-skeleton support: one set of meshes per athlete slot ──
  const NC = SKEL_CONNECTIONS.length;
  const NJ = 33;

  // Each entry: { slot, fileId, color, frames, boneMesh, jointMesh, comSphere, bonePos, jointPos, boneGeom, jointGeom }
  const skelEntries = [];

  function hexToInt(hex) {
    return parseInt(hex.replace('#',''), 16);
  }

  function createSkeletonMeshes(colorHex) {
    const c = hexToInt(colorHex);
    // Lighter variant for joints
    const r = (c>>16)&0xff, g = (c>>8)&0xff, b = c&0xff;
    const jr = Math.min(255, r + 60), jg = Math.min(255, g + 60), jb = Math.min(255, b + 60);
    const jointColor = (jr<<16) | (jg<<8) | jb;

    const bPos = new Float32Array(NC * 6);
    const jPos = new Float32Array(NJ * 3);
    const bGeom = new BufferGeometry();
    const bAttr = new BufferAttribute(bPos, 3);
    bAttr.setUsage(35048);
    bGeom.setAttribute('position', bAttr);
    bGeom.setDrawRange(0, NC * 2);
    const jGeom = new BufferGeometry();
    const jAttr = new BufferAttribute(jPos, 3);
    jAttr.setUsage(35048);
    jGeom.setAttribute('position', jAttr);
    const bMesh = new LineSegments(bGeom, new LineBasicMaterial({color: c}));
    const jMesh = new Points(jGeom, new PointsMaterial({color: jointColor, size: 0.04, sizeAttenuation: true}));
    bMesh.visible = false;
    jMesh.visible = false;
    scene.add(bMesh);
    scene.add(jMesh);

    const com = new Mesh(
      new SphereGeometry(0.03, 14, 14),
      new MeshPhongMaterial({color: c, emissive: Math.floor(c * 0.25)})
    );
    com.visible = false;
    scene.add(com);

    return { boneMesh: bMesh, jointMesh: jMesh, comSphere: com, bonePos: bPos, jointPos: jPos, boneGeom: bGeom, jointGeom: jGeom };
  }

  function computeCOM(lm) {
    function pt(i) { return lm[i] || null; }
    function mid(a, b) {
      const pa = pt(a), pb = pt(b);
      if(pa && pb) return [(pa[0]+pb[0])/2, (pa[1]+pb[1])/2, (pa[2]+pb[2])/2];
      return pa || pb || null;
    }
    function addSeg(segs, p1, p2, mf, cf) { if(p1 && p2) segs.push([p1, p2, mf, cf]); }
    function segCom(p1, p2, cf) { return [p1[0]+cf*(p2[0]-p1[0]), p1[1]+cf*(p2[1]-p1[1]), p1[2]+cf*(p2[2]-p1[2])]; }
    const midHip = mid(23, 24), midShoulder = mid(11, 12), headTop = mid(7, 8);
    const segs = [];
    addSeg(segs, midShoulder, headTop, 0.081, 1.000);
    addSeg(segs, midShoulder, midHip, 0.497, 0.500);
    addSeg(segs, pt(11), pt(13), 0.028, 0.436);
    addSeg(segs, pt(13), pt(15), 0.016, 0.430);
    addSeg(segs, pt(15), pt(17), 0.006, 0.506);
    addSeg(segs, pt(12), pt(14), 0.028, 0.436);
    addSeg(segs, pt(14), pt(16), 0.016, 0.430);
    addSeg(segs, pt(16), pt(18), 0.006, 0.506);
    addSeg(segs, pt(23), pt(25), 0.100, 0.433);
    addSeg(segs, pt(25), pt(27), 0.0465, 0.433);
    addSeg(segs, pt(27), pt(31), 0.0145, 0.500);
    addSeg(segs, pt(24), pt(26), 0.100, 0.433);
    addSeg(segs, pt(26), pt(28), 0.0465, 0.433);
    addSeg(segs, pt(28), pt(32), 0.0145, 0.500);
    let totalMass = 0, cx = 0, cy = 0, cz = 0;
    for(const [p1, p2, mf, cf] of segs) {
      const sc = segCom(p1, p2, cf);
      cx += mf * sc[0]; cy += mf * sc[1]; cz += mf * sc[2]; totalMass += mf;
    }
    if(totalMass < 0.35) return null;
    return [cx/totalMass, cy/totalMass, cz/totalMass];
  }

  function updateEntryMesh(entry, lm) {
    const { bonePos, jointPos, boneGeom, jointGeom, comSphere } = entry;
    const HIDE = -9999; // off-screen Y for missing landmarks
    if (lm) lm = stlTuneApplyPreview(lm); // live tuning preview (no-op when offset is zero)
    if(!lm) {
      for(let i = 0; i < NJ; i++) {
        jointPos[i*3] = 0; jointPos[i*3+1] = HIDE; jointPos[i*3+2] = 0;
      }
      for(let c = 0; c < NC; c++) {
        bonePos[c*6] = 0; bonePos[c*6+1] = HIDE; bonePos[c*6+2] = 0;
        bonePos[c*6+3] = 0; bonePos[c*6+4] = HIDE; bonePos[c*6+5] = 0;
      }
      comSphere.position.set(0, HIDE, 0);
      boneGeom.attributes.position.needsUpdate = true;
      jointGeom.attributes.position.needsUpdate = true;
      sceneNeedsRender = true;
      return { com: null, trunkDeg: null };
    }
    for(let i = 0; i < NJ; i++) {
      const p = lm[i];
      if (p) { jointPos[i*3] = p[0]; jointPos[i*3+1] = p[2]; jointPos[i*3+2] = -p[1]; }
      else   { jointPos[i*3] = 0;    jointPos[i*3+1] = HIDE; jointPos[i*3+2] = 0; }
    }
    for(let c = 0; c < NC; c++) {
      const [a, b] = SKEL_CONNECTIONS[c];
      const pa = lm[a], pb = lm[b];
      if (pa && pb) {
        bonePos[c*6] = pa[0]; bonePos[c*6+1] = pa[2]; bonePos[c*6+2] = -pa[1];
        bonePos[c*6+3] = pb[0]; bonePos[c*6+4] = pb[2]; bonePos[c*6+5] = -pb[1];
      } else {
        bonePos[c*6] = 0; bonePos[c*6+1] = HIDE; bonePos[c*6+2] = 0;
        bonePos[c*6+3] = 0; bonePos[c*6+4] = HIDE; bonePos[c*6+5] = 0;
      }
    }
    boneGeom.attributes.position.needsUpdate = true;
    jointGeom.attributes.position.needsUpdate = true;
    const com = computeCOM(lm);
    if(com) { comSphere.position.set(com[0], com[2], -com[1]); }
    else { comSphere.position.set(0, HIDE, 0); }
    sceneNeedsRender = true;
    return { com, trunkDeg: computeTrunkAngle(lm) };
  }

  function computeTrunkAngle(lm) {
    const l11=lm[11],l12=lm[12],l23=lm[23],l24=lm[24];
    if(!l11||!l12||!l23||!l24) return null;
    const sx=(l11[0]+l12[0])/2, sy=(l11[1]+l12[1])/2, sz=(l11[2]+l12[2])/2;
    const hx=(l23[0]+l24[0])/2, hy=(l23[1]+l24[1])/2, hz=(l23[2]+l24[2])/2;
    const dx=sx-hx, dy=sy-hy, dz=sz-hz;
    const len=Math.sqrt(dx*dx+dy*dy+dz*dz);
    if(len<1e-6) return null;
    const dot = dz / len;
    return Math.acos(Math.max(-1,Math.min(1,dot)))*180/Math.PI;
  }

  function updateDialAndMetrics() { /* metrics removed from UI */ }

  // Load skeleton frames for all athlete slots that have a video
  const skelStatusEl = el('stl-skel-status');
  skelStatusEl.textContent = 'Loading skeletons…';
  skelStatusEl.className = '';

  const slotsWithVideo = state.tl.athleteSlots.filter(s => s.currentFileId && s.videoEl);
  // Also collect all video fileIds from the project (in case slots are stale / video lost)
  const allVideoFileIds = (state.mapData?.videos || []).map(v => v.id);
  const skelEntriesByFile = new Map(); // fileId -> entry
  const skelLoadInFlight = new Set(); // fileIds currently loading from OPFS

  function refreshSkelStatus() {
    if(!skelEntries.length) {
      skelStatusEl.textContent = 'No skeleton — run Pose first';
      skelStatusEl.className = '';
      return;
    }
    const names = skelEntries.map(e => e.name).join(', ');
    const total = skelEntries.reduce((s, e) => s + (e.frames?.length || 0), 0);
    skelStatusEl.textContent = `${skelEntries.length} skeleton${skelEntries.length>1?'s':''} (${names}) — ${total} frames`;
    skelStatusEl.className = 'ok';
  }

  let skelStatusRefreshQueued = false;
  function scheduleSkelStatusRefresh() {
    if (skelStatusRefreshQueued) return;
    skelStatusRefreshQueued = true;
    requestAnimationFrame(() => {
      skelStatusRefreshQueued = false;
      if (!modal.classList.contains('open')) return;
      refreshSkelStatus();
    });
  }

  function slotForFile(fileId) {
    return state.tl.athleteSlots.find(s => s.videos?.some(vv => vv.id === fileId)) || null;
  }

  function createOrGetEntry(fileId, slotHint = null) {
    if(!fileId) return null;
    let entry = skelEntriesByFile.get(fileId);
    if(entry) {
      if(!entry.slot && slotHint) entry.slot = slotHint;
      return entry;
    }

    const slot = slotHint || slotForFile(fileId);
    const color = slot?.color || state.videoColors[fileId] || PALETTE[skelEntries.length % PALETTE.length];
    const videoMeta = (state.mapData?.videos || []).find(v => v.id === fileId);
    const name = slot?.name || videoMeta?.filename || state.fileMeta?.[fileId]?.filename || String(fileId).slice(0, 8);
    const meshes = createSkeletonMeshes(color);
    entry = { slot: slot || null, fileId, color, name, frames: [], ...meshes };
    const showSkel = el('stl-tog-skel')?.checked ?? true;
    const showJoints = el('stl-tog-joints')?.checked ?? true;
    const showCom = el('stl-tog-com')?.checked ?? true;
    entry.boneMesh.visible = showSkel;
    entry.jointMesh.visible = showJoints;
    entry.comSphere.visible = showCom;
    skelEntries.push(entry);
    skelEntriesByFile.set(fileId, entry);
    return entry;
  }

  function insertViewerFrameSorted(entry, viewerFrame) {
    if(!entry || !viewerFrame) return false;
    const ts = Number(viewerFrame.video_s);
    if(!Number.isFinite(ts)) return false;
    const frame = { video_s: ts, lm: viewerFrame.lm || null };
    const frames = entry.frames || (entry.frames = []);
    const EPS = 1e-3;

    if(frames.length === 0) {
      frames.push(frame);
      return true;
    }

    const lastTs = Number(frames[frames.length - 1]?.video_s);
    if(Number.isFinite(lastTs) && ts > lastTs + EPS) {
      frames.push(frame);
      return true;
    }

    let lo = 0, hi = frames.length - 1;
    while(lo <= hi) {
      const mid = (lo + hi) >> 1;
      const mt = Number(frames[mid]?.video_s);
      if(mt < ts) lo = mid + 1;
      else hi = mid - 1;
    }

    if(lo < frames.length && Math.abs(Number(frames[lo].video_s) - ts) <= EPS) {
      frames[lo] = frame;
      return false;
    }
    if(lo > 0 && Math.abs(Number(frames[lo - 1].video_s) - ts) <= EPS) {
      frames[lo - 1] = frame;
      return false;
    }
    frames.splice(lo, 0, frame);
    return true;
  }

  function mergeViewerFrames(entry, frames) {
    if(!entry || !Array.isArray(frames) || !frames.length) return 0;
    let added = 0;
    for(const fr of frames) {
      if(insertViewerFrameSorted(entry, fr)) added++;
    }
    return added;
  }

  async function ensureEntryFramesLoaded(fileId, slotHint = null) {
    if(!state.projectId || !fileId) return null;
    const existing = skelEntriesByFile.get(fileId);
    if(existing && existing.frames?.length) {
      if(!existing.slot && slotHint) existing.slot = slotHint;
      return existing;
    }
    if(skelLoadInFlight.has(fileId)) return existing || null;

    skelLoadInFlight.add(fileId);
    try {
      const frames = await loadSkeletonFrames(state.projectId, fileId);
      if(!modal.classList.contains('open')) return existing || null;
      if(!frames || !frames.length) return existing || null;
      const entry = createOrGetEntry(fileId, slotHint);
      const before = entry.frames.length;
      mergeViewerFrames(entry, frames);
      if(before === 0 && entry.frames.length) updateEntryMesh(entry, entry.frames[0].lm);
      refreshSkelStatus();
      sceneNeedsRender = true;
      return entry;
    } catch(e) {
      console.warn(`[3D] failed to load skeleton for ${fileId}:`, e);
      return existing || null;
    } finally {
      skelLoadInFlight.delete(fileId);
    }
  }

  console.log(`[3D] openStlViewer: slotsWithVideo=${slotsWithVideo.length}, allVideoFileIds=${JSON.stringify(allVideoFileIds)}, projectId=${state.projectId}`);
  for (const slot of state.tl.athleteSlots) {
    console.log(`[3D]   slot "${slot.name}": currentFileId=${slot.currentFileId}, hasVideoEl=${!!slot.videoEl}`);
  }

  // First, load skeletons for currently-playing slots
  for(const slot of slotsWithVideo) {
    const fileId = slot.currentFileId;
    await ensureEntryFramesLoaded(fileId, slot);
    const entry = skelEntriesByFile.get(fileId);
    console.log(`[3D] slot "${slot.name}" (${fileId}): ${entry?.frames?.length || 0} frames`);
  }

  // If no skeletons found via active slots, warm-cache in background.
  // Do not block viewer startup; users may seek into a segment immediately.
  if (skelEntries.length === 0 && state.projectId) {
    console.log(`[3D] No skeletons from active slots — starting non-blocking fallback for ${allVideoFileIds.length} project videos`);
    (async () => {
      for (const fid of allVideoFileIds) {
        if(!modal.classList.contains('open')) break;
        // Prioritize currently active timeline videos and avoid duplicate loads.
        if (skelEntriesByFile.has(fid)) continue;
        await ensureEntryFramesLoaded(fid, slotForFile(fid));
      }
    })().catch(e => console.warn('[3D] background fallback load failed:', e));
  }

  refreshSkelStatus();
  if(skelEntries.length) {
    // Target camera on first skeleton's hip (guard against incomplete landmarks)
    const lm0 = skelEntries[0].frames[0]?.lm;
    if (lm0 && lm0[23] && lm0[24]) {
      const hx = (lm0[23][0]+lm0[24][0])/2;
      const hy = (lm0[23][2]+lm0[24][2])/2;
      const hz = -(lm0[23][1]+lm0[24][1])/2;
      controls.target.set(hx, hy + 0.3, hz);
      camera.position.set(hx + 2.5, hy + 1.5, hz + 2.5);
      controls.update();
    }
  }

  // ── Live frame subscription: show skeleton frames as they arrive during processing ──
  const _liveUnsubs = [];
  const liveFileIds = new Set(allVideoFileIds);
  for (const slot of slotsWithVideo) if (slot.currentFileId) liveFileIds.add(slot.currentFileId);
  for (const fileId of liveFileIds) {
    // Subscribe to live frames from pose-engine
    const unsub = PoseEngine.onLiveFrame(fileId, (frame) => {
      if (!modal.classList.contains('open')) return;
      const slot = slotForFile(fileId);
      const entry = createOrGetEntry(fileId, slot);
      if (!entry) return;
      // Convert to the format used by the 3D viewer: { video_s, lm }
      const viewerFrame = { video_s: Number(frame.ts), lm: frame.skeleton };
      insertViewerFrameSorted(entry, viewerFrame);
      // Update mesh immediately with the latest frame
      const isActiveInSlot = entry.slot ? entry.slot.currentFileId === entry.fileId : true;
      if (isActiveInSlot) updateEntryMesh(entry, viewerFrame.lm);
      scheduleSkelStatusRefresh();
      sceneNeedsRender = true;
    });
    _liveUnsubs.push(unsub);
  }
  // Store unsub functions for cleanup when viewer closes
  modal._liveUnsubs = _liveUnsubs;

  const BOAT_CTR = new Vector3(-1.5, 0.3, 0);

  function animCameraTo(pos, tgt, ms = 450) {
    const p0 = camera.position.clone();
    const t0 = controls.target.clone();
    const t1 = performance.now();
    function step(now) {
      const raw = Math.min((now - t1) / ms, 1);
      const t = raw < 0.5 ? 4*raw*raw*raw : 1 - Math.pow(-2*raw+2,3)/2;
      camera.position.lerpVectors(p0, pos, t);
      controls.target.lerpVectors(t0, tgt, t);
      controls.update();
      sceneNeedsRender = true;
      if(raw < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  window.stlSetCamera = (name) => {
    const tgt = BOAT_CTR.clone();
    switch(name) {
      case 'default': animCameraTo(new Vector3(2.5, 2, 3.5), tgt); break;
      case 'side': animCameraTo(new Vector3(-1.5, 1.5, -4), tgt); break;
      case 'top': animCameraTo(new Vector3(-1.5, 5, 0.1), tgt); break;
      case 'front': animCameraTo(new Vector3(1.5, 1, 0), tgt); break;
      case 'rear': animCameraTo(new Vector3(-5, 1, 0), tgt); break;
      case 'athlete':
        if(skelEntries.length) {
          const lm = skelEntries[0].frames[0]?.lm;
          if (lm && lm[23] && lm[24]) {
            const ax=(lm[23][0]+lm[24][0])/2;
            const ay=(lm[23][2]+lm[24][2])/2;
            const az=-(lm[23][1]+lm[24][1])/2;
            animCameraTo(new Vector3(ax+2.2,ay+1.2,az+1.5), new Vector3(ax,ay+0.3,az));
          }
        }
        break;
    }
  };

  function tog(id, fn) {
    const cb = document.getElementById(id);
    if(cb) cb.addEventListener('change', e => { fn(e.target.checked); sceneNeedsRender = true; });
  }
  tog('stl-tog-hull', v => { if(hullMesh) hullMesh.visible = v; });
  tog('stl-tog-skel', v => { for(const e of skelEntries) e.boneMesh.visible = v; });
  tog('stl-tog-joints', v => { for(const e of skelEntries) e.jointMesh.visible = v; });
  tog('stl-tog-com', v => { for(const e of skelEntries) e.comSphere.visible = v; });
  tog('stl-tog-grid', v => { grid.visible = v; });

  // ── Camera pose tuning controls ──────────────────────────────────────
  // Force every displayed skeleton to re-render with the current preview offset.
  const refreshTunedMeshes = () => {
    for (const entry of skelEntries) {
      const lm = entry._lastFrame?.lm || entry.frames?.[0]?.lm;
      if (lm) updateEntryMesh(entry, lm);
    }
    sceneNeedsRender = true;
  };
  setupStlTuningControls(refreshTunedMeshes);

  let animId;
  let lastRenderTs = 0;
  function animate(ts = 0) {
    animId = requestAnimationFrame(animate);
    _stlAnimId = animId;
    controls.update();

    // Lazy-load skeleton frames for newly active slot videos while viewer is open.
    for (const slot of state.tl.athleteSlots) {
      if (slot.currentFileId && !skelEntriesByFile.has(slot.currentFileId)) {
        ensureEntryFramesLoaded(slot.currentFileId, slot);
      }
    }

    // Update each skeleton from its own video element
    let firstMetrics = null;
    let firstLm = null;
    for(const entry of skelEntries) {
      if(entry.slot && entry.slot.currentFileId && entry.fileId !== entry.slot.currentFileId) {
        updateEntryMesh(entry, null);
        continue;
      }
      if(!entry.frames?.length) continue;
      const vid = entry.slot?.videoEl;
      const hasVideoClock = !!vid && vid.readyState >= 1 && Number.isFinite(vid.currentTime);
      const isPlaying = !!vid && !vid.paused && !vid.ended && vid.readyState >= 2;
      const t = hasVideoClock ? vid.currentTime : NaN;

      // On iPad/Safari playback can stay paused at t=0 in hidden/inline video contexts.
      // Keep a stable fallback frame visible when we don't have a reliable running clock.
      let frame = hasVideoClock ? skelBinarySearch(entry.frames, t, 0.75) : null;
      if(!frame && !isPlaying) {
        if(hasVideoClock) frame = skelBinarySearch(entry.frames, t, Number.POSITIVE_INFINITY);
        if(!frame) frame = entry._lastFrame || entry.frames[0] || null;
      }

      if(frame) {
        const m = updateEntryMesh(entry, frame.lm);
        entry._lastFrame = frame;
        if(!firstMetrics) { firstMetrics = m; firstLm = frame.lm; }
      } else {
        updateEntryMesh(entry, null);
      }
    }
    if(firstMetrics && firstLm) {
      updateDialAndMetrics(firstMetrics.trunkDeg, firstLm, firstMetrics.com);
    }
    const anyPlaying = skelEntries.some(e => e.slot?.videoEl && !e.slot.videoEl.paused);
    const elapsed = ts - lastRenderTs;
    const minMs = anyPlaying ? 33 : 250;
    if(sceneNeedsRender || elapsed >= minMs) {
      renderer.render(scene, camera);
      sceneNeedsRender = false;
      lastRenderTs = ts;
    }
  }
  animate();

  function onResize() {
    if(!modal.classList.contains('open')) return;
    const w = wrap.clientWidth || wrap.offsetWidth;
    const h = wrap.clientHeight || wrap.offsetHeight;
    if(!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    sceneNeedsRender = true;
  }
  window.addEventListener('resize', onResize);
  _stlResizeHandler = onResize;
  if(_stlResizeObserver) { try { _stlResizeObserver.disconnect(); } catch(e) {} _stlResizeObserver = null; }
  if(window.ResizeObserver) {
    _stlResizeObserver = new ResizeObserver(() => onResize());
    _stlResizeObserver.observe(wrap);
  }
  setTimeout(onResize, 50);
}

// ── Boot ───────────────────────────────────────────────────────────────
init();





