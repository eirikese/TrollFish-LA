/**
 * TrollFish - boom-engine.js
 * Browser-side PilotNet ONNX inference for boom azimuth prediction.
 *
 * Uses the same frame source and PilotNet preprocessing as the rudder engine:
 * ROI crop -> resize -> YUV -> normalize [-1, 1] -> NCHW float32.
 */

import { BOOM_MODEL_VERSION } from './config.js';

const ONNX_RUNTIME_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js';

function _assetUrl(path) {
  const url = new URL(path, import.meta.url);
  url.searchParams.set('v', BOOM_MODEL_VERSION);
  return url.href;
}

const BOOM_MODEL_URLS = [
  _assetUrl('../vendor/onnx/best_pilotnet_boom.onnx'),
  _assetUrl('./best_pilotnet_boom.onnx'),
];
const BOOM_META_URLS = [
  _assetUrl('../vendor/onnx/best_pilotnet_boom.onnx.json'),
  _assetUrl('./best_pilotnet_boom.onnx.json'),
];

const DEFAULT_META = Object.freeze({
  angle_mean: 1.043214201927185,
  angle_std: 48.37992477416992,
  input_size: [66, 200],       // [h, w]
  roi: [230, 400, 500, 1420], // [top, bottom, left, right] in 1920x1080 reference space
  roi_reference_size: [1080, 1920],
  onnx_output_units: 'degrees',
  onnx_output_range: '[-180, 180)',
  target_mode: 'sincos_centered',
  target_center_deg: 180.0,
  onnx_output_angle_system: 'raw_boom_azimuth_deg',
});

const BOOM_ZERO_AZIMUTH_DEG = 180.0;
const BOOM_MAX_FRAME_DELTA_DEG = 45.0;
const BOOM_EMA_ALPHA = 0.28;
const BOOM_ANGLE_SYSTEM = 'boom_minus_x_centered_v3';

const _state = {
  ort: null,
  session: null,
  inputName: null,
  outputName: null,
  angleMean: DEFAULT_META.angle_mean,
  angleStd: DEFAULT_META.angle_std,
  inputH: DEFAULT_META.input_size[0],
  inputW: DEFAULT_META.input_size[1],
  roiCrop: DEFAULT_META.roi.slice(),
  roiRefH: DEFAULT_META.roi_reference_size[0],
  roiRefW: DEFAULT_META.roi_reference_size[1],
  modelOutputsDegrees: true,
  targetMode: DEFAULT_META.target_mode,
  targetCenterDeg: DEFAULT_META.target_center_deg,
  outputAngleSystem: DEFAULT_META.onnx_output_angle_system,
  loadPromise: null,
  scratchCanvas: null,
  scratchCtx: null,
  tensorData: null,
  tensorDataInUse: false,
  roiCacheKey: '',
  roiCache: null,
  warnedLowResolutionSource: false,
  filter: {
    emaSin: null,
    emaCos: null,
    lastAngleDeg: null,
    recent: [],
  },
};

function _isFiniteNumber(v) {
  const n = Number(v);
  return Number.isFinite(n);
}

function _wrapAngle180(angleDeg) {
  const angle = Number(angleDeg);
  if (!Number.isFinite(angle)) return null;
  return ((angle + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
}

function _angleDeltaDeg(nextDeg, prevDeg) {
  const delta = _wrapAngle180(Number(nextDeg) - Number(prevDeg));
  return Number.isFinite(delta) ? delta : null;
}

function _medianNumber(values) {
  const arr = (values || [])
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!arr.length) return null;
  const mid = Math.floor(arr.length / 2);
  return arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) * 0.5;
}

function _circularMedianAround(values, referenceDeg) {
  const reference = Number(referenceDeg);
  if (!Number.isFinite(reference)) return _medianNumber(values);
  const unwrapped = (values || [])
    .map((value) => {
      const delta = _angleDeltaDeg(value, reference);
      return Number.isFinite(delta) ? reference + delta : null;
    })
    .filter(Number.isFinite);
  const med = _medianNumber(unwrapped);
  return Number.isFinite(med) ? _wrapAngle180(med) : null;
}

function _numberOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function _boomAzimuthToMinusXCentered(azimuthDeg) {
  const azimuth = Number(azimuthDeg);
  if (!Number.isFinite(azimuth)) return null;
  return _wrapAngle180(azimuth - BOOM_ZERO_AZIMUTH_DEG);
}

function _outputIsCenteredModelAngle() {
  const system = String(_state.outputAngleSystem || '').toLowerCase();
  return system.includes('model_angle')
    || system.includes('centered')
    || system.includes('minus_x')
    || system.includes('signed')
    || (
      String(_state.targetMode || '').toLowerCase().includes('centered')
      && system !== 'raw_boom_azimuth_deg'
    );
}

function _modelOutputToRawAzimuth(rawOutput) {
  const raw = Number(rawOutput);
  if (!Number.isFinite(raw)) return null;

  let angle = _state.modelOutputsDegrees
    ? raw
    : (raw * _state.angleStd) + _state.angleMean;

  if (!Number.isFinite(angle)) return null;
  if (_outputIsCenteredModelAngle()) {
    angle += Number(_state.targetCenterDeg) || 0.0;
  }
  return _wrapAngle180(angle);
}

function _smoothSignedAngle(rawSignedDeg) {
  const st = _state.filter;
  if (!_isFiniteNumber(rawSignedDeg)) return { value: null, raw: null, outlier: false };
  const raw = Number(rawSignedDeg);
  if (!Number.isFinite(raw)) return { value: null, raw: null, outlier: false };

  let candidate = raw;
  let outlier = false;
  if (Number.isFinite(st.lastAngleDeg)) {
    const delta = _angleDeltaDeg(raw, st.lastAngleDeg);
    if (Number.isFinite(delta) && Math.abs(delta) > BOOM_MAX_FRAME_DELTA_DEG) {
      candidate = _wrapAngle180(st.lastAngleDeg + Math.sign(delta) * BOOM_MAX_FRAME_DELTA_DEG);
      outlier = true;
    }
  }

  st.recent.push(candidate);
  if (st.recent.length > 5) st.recent.shift();
  if (st.recent.length >= 3) {
    const med = _circularMedianAround(st.recent, candidate);
    if (Number.isFinite(med)) candidate = med;
  }

  if (!Number.isFinite(st.lastAngleDeg)) {
    st.lastAngleDeg = candidate;
  } else {
    const delta = _angleDeltaDeg(candidate, st.lastAngleDeg);
    st.lastAngleDeg = Number.isFinite(delta)
      ? _wrapAngle180(st.lastAngleDeg + BOOM_EMA_ALPHA * delta)
      : candidate;
  }

  return { value: st.lastAngleDeg, raw, outlier };
}

async function _loadScriptOnce(src) {
  if (window.ort) return;

  const existing = document.querySelector(`script[src="${src}"]`);
  if (existing) {
    await new Promise((resolve, reject) => {
      if (window.ort) {
        resolve();
        return;
      }
      existing.addEventListener('load', resolve, { once: true });
      existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
    });
    return;
  }

  await new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });
}

async function _ensureOrt() {
  if (_state.ort) return _state.ort;
  await _loadScriptOnce(ONNX_RUNTIME_CDN);
  if (!window.ort) {
    throw new Error('ONNX Runtime Web unavailable after script load');
  }
  _state.ort = window.ort;
  return _state.ort;
}

function _parseMeta(metaRaw) {
  const meta = { ...DEFAULT_META, ...(metaRaw || {}) };

  const angleMean = Number(meta.angle_mean);
  const angleStd = Number(meta.angle_std);
  const inputSize = Array.isArray(meta.input_size) && meta.input_size.length === 2 ? meta.input_size : DEFAULT_META.input_size;
  const roiRaw = Array.isArray(meta.roi_crop) && meta.roi_crop.length === 4
    ? meta.roi_crop
    : (Array.isArray(meta.roi) && meta.roi.length === 4 ? meta.roi : DEFAULT_META.roi);
  const roiRef = Array.isArray(meta.roi_reference_size) && meta.roi_reference_size.length === 2
    ? meta.roi_reference_size
    : DEFAULT_META.roi_reference_size;
  const outputUnits = String(meta.onnx_output_units || meta.output_units || '').trim().toLowerCase();
  const modelOutputsDegrees = meta.output_is_degrees === true
    || outputUnits.includes('deg')
    || outputUnits === 'degrees';
  const targetMode = String(meta.target_mode || DEFAULT_META.target_mode);
  const outputAngleSystem = String(meta.onnx_output_angle_system || meta.output_angle_system || DEFAULT_META.onnx_output_angle_system);
  const targetCenterDeg = _numberOrDefault(meta.target_center_deg, DEFAULT_META.target_center_deg);

  return {
    angleMean: Number.isFinite(angleMean) ? angleMean : DEFAULT_META.angle_mean,
    angleStd: Math.abs(angleStd) >= 1e-8 ? angleStd : DEFAULT_META.angle_std,
    inputH: Math.max(8, _numberOrDefault(inputSize[0], DEFAULT_META.input_size[0])),
    inputW: Math.max(8, _numberOrDefault(inputSize[1], DEFAULT_META.input_size[1])),
    roiCrop: [
      Math.round(_numberOrDefault(roiRaw[0], DEFAULT_META.roi[0])),
      Math.round(_numberOrDefault(roiRaw[1], DEFAULT_META.roi[1])),
      Math.round(_numberOrDefault(roiRaw[2], DEFAULT_META.roi[2])),
      Math.round(_numberOrDefault(roiRaw[3], DEFAULT_META.roi[3])),
    ],
    roiRefH: Math.max(1, Math.round(_numberOrDefault(roiRef[0], DEFAULT_META.roi_reference_size[0]))),
    roiRefW: Math.max(1, Math.round(_numberOrDefault(roiRef[1], DEFAULT_META.roi_reference_size[1]))),
    modelOutputsDegrees,
    targetMode,
    targetCenterDeg,
    outputAngleSystem,
  };
}

async function _loadMeta() {
  let lastErr = null;
  for (const url of BOOM_META_URLS) {
    try {
      const resp = await fetch(url, { cache: 'no-cache' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const metaRaw = await resp.json();
      return _parseMeta(metaRaw);
    } catch (err) {
      lastErr = err;
    }
  }
  console.warn('[Boom] Failed to load metadata, using defaults:', lastErr?.message || lastErr);
  return _parseMeta(null);
}

function _createScratchContext(inputW, inputH) {
  if (!_state.scratchCanvas || !_state.scratchCtx || _state.scratchCanvas.width !== inputW || _state.scratchCanvas.height !== inputH) {
    const canvas = document.createElement('canvas');
    canvas.width = inputW;
    canvas.height = inputH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Could not create boom preprocessing canvas context');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    _state.scratchCanvas = canvas;
    _state.scratchCtx = ctx;
  }
  return _state.scratchCtx;
}

function _sourceSize(source) {
  const w = Number(source?.videoWidth ?? source?.naturalWidth ?? source?.width ?? 0);
  const h = Number(source?.videoHeight ?? source?.naturalHeight ?? source?.height ?? 0);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 1 || h <= 1) return null;
  return { w, h };
}

function _resolveScaledRoi(roiCrop, srcW, srcH, refW, refH) {
  const sx = srcW / Math.max(1, refW);
  const sy = srcH / Math.max(1, refH);

  let top = Math.round(roiCrop[0] * sy);
  let bottom = Math.round(roiCrop[1] * sy);
  let left = Math.round(roiCrop[2] * sx);
  let right = Math.round(roiCrop[3] * sx);

  top = Math.max(0, Math.min(srcH, top));
  bottom = Math.max(0, Math.min(srcH, bottom));
  left = Math.max(0, Math.min(srcW, left));
  right = Math.max(0, Math.min(srcW, right));

  if (bottom - top < 2 || right - left < 2) return null;
  return { top, bottom, left, right };
}

function _getScaledRoiForSource(sz) {
  const key = [
    Math.round(sz.w),
    Math.round(sz.h),
    _state.roiRefW,
    _state.roiRefH,
    _state.roiCrop.join(','),
  ].join(':');
  if (_state.roiCacheKey === key) return _state.roiCache;
  const roi = _resolveScaledRoi(
    _state.roiCrop,
    sz.w,
    sz.h,
    _state.roiRefW,
    _state.roiRefH
  );
  _state.roiCacheKey = key;
  _state.roiCache = roi;
  return roi;
}

function _leaseTensorData(length) {
  if (!_state.tensorDataInUse) {
    if (!_state.tensorData || _state.tensorData.length !== length) {
      _state.tensorData = new Float32Array(length);
    }
    _state.tensorDataInUse = true;
    return {
      data: _state.tensorData,
      release: () => { _state.tensorDataInUse = false; },
    };
  }
  return {
    data: new Float32Array(length),
    release: () => {},
  };
}

function _preprocessToTensor(source) {
  const sz = _sourceSize(source);
  if (!sz) return null;

  if (
    !_state.warnedLowResolutionSource
    && (sz.w < _state.roiRefW * 0.75 || sz.h < _state.roiRefH * 0.75)
  ) {
    _state.warnedLowResolutionSource = true;
    console.warn(
      `[Boom] Predicting from a ${Math.round(sz.w)}x${Math.round(sz.h)} source; ` +
      `this model was trained against ${_state.roiRefW}x${_state.roiRefH} GoPro frames. ` +
      'Use the original video/image source when possible.'
    );
  }

  const roi = _getScaledRoiForSource(sz);
  if (!roi) return null;

  const ctx = _createScratchContext(_state.inputW, _state.inputH);
  ctx.drawImage(
    source,
    roi.left,
    roi.top,
    roi.right - roi.left,
    roi.bottom - roi.top,
    0,
    0,
    _state.inputW,
    _state.inputH
  );

  const rgba = ctx.getImageData(0, 0, _state.inputW, _state.inputH).data;
  const pixels = _state.inputW * _state.inputH;
  const tensorLease = _leaseTensorData(3 * pixels);
  const x = tensorLease.data;

  // Approximate OpenCV COLOR_BGR2YUV using canvas RGB source.
  for (let i = 0; i < pixels; i++) {
    const off = i * 4;
    const r = rgba[off];
    const g = rgba[off + 1];
    const b = rgba[off + 2];

    const y = 0.114 * b + 0.587 * g + 0.299 * r;
    const u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0;
    const v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0;

    x[i] = (y / 127.5) - 1.0;
    x[pixels + i] = (Math.max(0, Math.min(255, u)) / 127.5) - 1.0;
    x[2 * pixels + i] = (Math.max(0, Math.min(255, v)) / 127.5) - 1.0;
  }

  try {
    return {
      tensor: new _state.ort.Tensor('float32', x, [1, 3, _state.inputH, _state.inputW]),
      release: tensorLease.release,
    };
  } catch (err) {
    tensorLease.release();
    throw err;
  }
}

async function _ensureSession() {
  if (_state.session) return _state.session;
  if (_state.loadPromise) return _state.loadPromise;

  _state.loadPromise = (async () => {
    const ort = await _ensureOrt();
    ort.env.wasm.numThreads = 1;

    const parsedMeta = await _loadMeta();
    _state.angleMean = parsedMeta.angleMean;
    _state.angleStd = parsedMeta.angleStd;
    _state.inputH = parsedMeta.inputH;
    _state.inputW = parsedMeta.inputW;
    _state.roiCrop = parsedMeta.roiCrop;
    _state.roiRefH = parsedMeta.roiRefH;
    _state.roiRefW = parsedMeta.roiRefW;
    _state.modelOutputsDegrees = parsedMeta.modelOutputsDegrees;
    _state.targetMode = parsedMeta.targetMode;
    _state.targetCenterDeg = parsedMeta.targetCenterDeg;
    _state.outputAngleSystem = parsedMeta.outputAngleSystem;

    let session = null;
    let lastErr = null;
    for (const url of BOOM_MODEL_URLS) {
      try {
        session = await ort.InferenceSession.create(url, {
          executionProviders: ['wasm'],
        });
        break;
      } catch (wasmErr) {
        lastErr = wasmErr;
        try {
          session = await ort.InferenceSession.create(url, {
            executionProviders: ['webgl'],
          });
          break;
        } catch (webglErr) {
          lastErr = webglErr;
        }
      }
    }
    if (!session) {
      throw lastErr || new Error('Could not load boom ONNX model');
    }

    _state.session = session;
    _state.inputName = session.inputNames?.[0] || 'input';
    _state.outputName = session.outputNames?.[0] || 'angle_deg';
    return session;
  })();

  try {
    return await _state.loadPromise;
  } catch (err) {
    _state.loadPromise = null;
    throw err;
  }
}

export async function preload() {
  await _ensureSession();
  return true;
}

export function getModelInfo() {
  return {
    loaded: Boolean(_state.session),
    input_size: [_state.inputH, _state.inputW],
    roi_crop: _state.roiCrop.slice(),
    angle_mean: _state.angleMean,
    angle_std: _state.angleStd,
    model_outputs_degrees: _state.modelOutputsDegrees,
    target_mode: _state.targetMode,
    target_center_deg: _state.targetCenterDeg,
    onnx_output_angle_system: _state.outputAngleSystem,
    output_angle_system: BOOM_ANGLE_SYSTEM,
    model_version: BOOM_MODEL_VERSION,
    zero_degrees: 'boom azimuth on the negative X axis',
    raw_model_angle_system: 'boom_azimuth_deg_atan2_z_x_v1',
  };
}

export function resetSmoothing() {
  _state.filter.emaSin = null;
  _state.filter.emaCos = null;
  _state.filter.lastAngleDeg = null;
  _state.filter.recent = [];
}

/**
 * Predict boom angle from a frame source.
 *
 * The current ONNX export outputs raw boom azimuth using the dataset convention:
 * atan2(z, x), wrapped to [-180, 180). If a future export outputs the centered
 * model angle directly, metadata tells us to add target_center_deg first. The
 * returned angle_deg is always recentered around -X, so 0 deg is the boom
 * pointing along negative X.
 *
 * @param {HTMLCanvasElement|HTMLVideoElement|ImageBitmap} source
 * @returns {Promise<{angle_deg:number|null, angle_raw_deg:number|null, model_angle_deg:number|null, raw_output:number|null, angle_system:string, outlier:boolean}>}
 */
export async function predictBoomAngle(source) {
  const session = _state.session || await _ensureSession();
  if (!session) {
    return { angle_deg: null, angle_raw_deg: null, model_angle_deg: null, raw_output: null, angle_system: BOOM_ANGLE_SYSTEM, outlier: false };
  }

  const prepared = _preprocessToTensor(source);
  if (!prepared) {
    return { angle_deg: null, angle_raw_deg: null, model_angle_deg: null, raw_output: null, angle_system: BOOM_ANGLE_SYSTEM, outlier: false };
  }

  const { tensor, release } = prepared;
  const feeds = { [_state.inputName]: tensor };
  let outMap = null;
  try {
    outMap = await session.run(feeds);
  } finally {
    release?.();
  }
  const out = outMap?.[_state.outputName];
  const raw = Number(out?.data?.[0]);
  if (!_isFiniteNumber(raw)) {
    return { angle_deg: null, angle_raw_deg: null, model_angle_deg: null, raw_output: null, angle_system: BOOM_ANGLE_SYSTEM, outlier: false };
  }

  const modelAzimuth = _modelOutputToRawAzimuth(raw);
  const rawSigned = _boomAzimuthToMinusXCentered(modelAzimuth);
  const filtered = _smoothSignedAngle(rawSigned);

  return {
    angle_deg: Number.isFinite(filtered.value) ? filtered.value : null,
    angle_raw_deg: Number.isFinite(filtered.raw) ? filtered.raw : null,
    model_angle_deg: Number.isFinite(modelAzimuth) ? modelAzimuth : null,
    raw_output: raw,
    angle_system: BOOM_ANGLE_SYSTEM,
    outlier: Boolean(filtered.outlier),
  };
}
