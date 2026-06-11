/**
 * TrollFish — rudder-engine.js
 * Browser-side PilotNet ONNX inference for rudder angle prediction.
 *
 * Mirrors rudder_labeler_standalone.py runtime behavior:
 *   - ROI crop -> resize -> YUV -> normalize [-1, 1]
 *   - denormalize model output via angle_mean/angle_std
 *   - optional runtime yaw correction: corrected = clip(pred - camera_yaw, 0..180)
 */

const ONNX_RUNTIME_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js';
const RUDDER_MODEL_URL = new URL('../vendor/onnx/best_pilotnet.onnx', import.meta.url).href;
const RUDDER_META_URL = new URL('../vendor/onnx/best_pilotnet_meta.json', import.meta.url).href;

const DEFAULT_META = Object.freeze({
  angle_mean: 88.94429755662736,
  angle_std: 9.87913755812305,
  input_size: [66, 200],      // [h, w]
  roi_crop: [880, 1080, 200, 1685], // [top, bottom, left, right] in 1920x1080 reference space
  angle_domain: 'corrected_no_runtime_yaw',
  runtime_yaw_correction: false,
  roi_reference_size: [1080, 1920], // [h, w]
});

const _state = {
  ort: null,
  session: null,
  inputName: null,
  outputName: null,
  angleMean: DEFAULT_META.angle_mean,
  angleStd: DEFAULT_META.angle_std,
  inputH: DEFAULT_META.input_size[0],
  inputW: DEFAULT_META.input_size[1],
  roiCrop: DEFAULT_META.roi_crop.slice(),
  runtimeYawCorrection: DEFAULT_META.runtime_yaw_correction,
  roiRefH: DEFAULT_META.roi_reference_size[0],
  roiRefW: DEFAULT_META.roi_reference_size[1],
  loadPromise: null,
  scratchCanvas: null,
  scratchCtx: null,
  tensorData: null,
  tensorDataInUse: false,
  roiCacheKey: '',
  roiCache: null,
};

function _isFiniteNumber(v) {
  const n = Number(v);
  return Number.isFinite(n);
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
  const roiCrop = Array.isArray(meta.roi_crop) && meta.roi_crop.length === 4 ? meta.roi_crop : DEFAULT_META.roi_crop;
  const roiRef = Array.isArray(meta.roi_reference_size) && meta.roi_reference_size.length === 2
    ? meta.roi_reference_size
    : DEFAULT_META.roi_reference_size;

  const angleDomain = String(meta.angle_domain || '');
  const runtimeYawCorrection = Boolean(
    meta.runtime_yaw_correction ?? (angleDomain === 'uncorrected_runtime_yaw_subtract')
  );

  return {
    angleMean: Number.isFinite(angleMean) ? angleMean : DEFAULT_META.angle_mean,
    angleStd: Math.abs(angleStd) >= 1e-8 ? angleStd : DEFAULT_META.angle_std,
    inputH: Math.max(8, Number(inputSize[0]) || DEFAULT_META.input_size[0]),
    inputW: Math.max(8, Number(inputSize[1]) || DEFAULT_META.input_size[1]),
    roiCrop: [
      Math.round(Number(roiCrop[0]) || DEFAULT_META.roi_crop[0]),
      Math.round(Number(roiCrop[1]) || DEFAULT_META.roi_crop[1]),
      Math.round(Number(roiCrop[2]) || DEFAULT_META.roi_crop[2]),
      Math.round(Number(roiCrop[3]) || DEFAULT_META.roi_crop[3]),
    ],
    runtimeYawCorrection,
    roiRefH: Math.max(1, Math.round(Number(roiRef[0]) || DEFAULT_META.roi_reference_size[0])),
    roiRefW: Math.max(1, Math.round(Number(roiRef[1]) || DEFAULT_META.roi_reference_size[1])),
  };
}

async function _loadMeta() {
  try {
    const resp = await fetch(RUDDER_META_URL, { cache: 'no-cache' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const metaRaw = await resp.json();
    return _parseMeta(metaRaw);
  } catch (err) {
    console.warn('[Rudder] Failed to load metadata, using defaults:', err?.message || err);
    return _parseMeta(null);
  }
}

function _createScratchContext(inputW, inputH) {
  if (!_state.scratchCanvas || !_state.scratchCtx || _state.scratchCanvas.width !== inputW || _state.scratchCanvas.height !== inputH) {
    const canvas = document.createElement('canvas');
    canvas.width = inputW;
    canvas.height = inputH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Could not create rudder preprocessing canvas context');
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

function _toCorrectedAngle(modelAngleDeg, cameraYawDeg) {
  let a = Number(modelAngleDeg);
  if (!Number.isFinite(a)) return null;

  if (_state.runtimeYawCorrection) {
    const yaw = Number(cameraYawDeg);
    if (Number.isFinite(yaw)) a -= yaw;
  }

  if (a < 0.0) a = 0.0;
  if (a > 180.0) a = 180.0;
  return a;
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
    _state.runtimeYawCorrection = parsedMeta.runtimeYawCorrection;
    _state.roiRefH = parsedMeta.roiRefH;
    _state.roiRefW = parsedMeta.roiRefW;

    let session;
    try {
      session = await ort.InferenceSession.create(RUDDER_MODEL_URL, {
        executionProviders: ['webgl'],
      });
    } catch {
      session = await ort.InferenceSession.create(RUDDER_MODEL_URL, {
        executionProviders: ['wasm'],
      });
    }

    _state.session = session;
    _state.inputName = session.inputNames?.[0] || 'input';
    _state.outputName = session.outputNames?.[0] || 'output';
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
    runtime_yaw_correction: _state.runtimeYawCorrection,
  };
}

/**
 * Predict rudder angle from a frame source.
 *
 * @param {HTMLCanvasElement|HTMLVideoElement|ImageBitmap} source
 * @param {number|null} cameraYawDeg
 * @returns {Promise<{model_angle_deg:number|null, corrected_angle_deg:number|null, camera_yaw_deg:number|null}>}
 */
export async function predictRudderAngle(source, cameraYawDeg = null) {
  const session = _state.session || await _ensureSession();
  if (!session) {
    return {
      model_angle_deg: null,
      corrected_angle_deg: null,
      camera_yaw_deg: _isFiniteNumber(cameraYawDeg) ? Number(cameraYawDeg) : null,
    };
  }

  const prepared = _preprocessToTensor(source);
  if (!prepared) {
    return {
      model_angle_deg: null,
      corrected_angle_deg: null,
      camera_yaw_deg: _isFiniteNumber(cameraYawDeg) ? Number(cameraYawDeg) : null,
    };
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
  if (!Number.isFinite(raw)) {
    return {
      model_angle_deg: null,
      corrected_angle_deg: null,
      camera_yaw_deg: _isFiniteNumber(cameraYawDeg) ? Number(cameraYawDeg) : null,
    };
  }

  const modelAngle = (raw * _state.angleStd) + _state.angleMean;
  const yaw = _isFiniteNumber(cameraYawDeg) ? Number(cameraYawDeg) : null;
  const corrected = _toCorrectedAngle(modelAngle, yaw);

  return {
    model_angle_deg: Number.isFinite(modelAngle) ? modelAngle : null,
    corrected_angle_deg: Number.isFinite(corrected) ? corrected : null,
    camera_yaw_deg: yaw,
  };
}
