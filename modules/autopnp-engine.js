/**
 * TrollFish — autopnp-engine.js
 * Browser-side Auto Camera PnP estimation using YOLO boat keypoints.
 *
 * Faithfully ports the Python processing.py pipeline:
 *   DLT algebraic initialization → RANSAC outlier rejection → LM refinement,
 *   multi-candidate scoring with soft plausibility penalties,
 *   fisheye undistortion of YOLO keypoints before solving.
 *
 * Correction pipeline: extract angles via BASE_R_WC, roll+90° offset,
 * y/z position swap, and bounds validation.
 *
 * The 9 keypoints and their 3D object coordinates match the Python pipeline:
 *   frontdeck, porttop, portmid, portlow, starboardtop, starboardmid,
 *   starboardlow, portback, starboardback
 */

import { getCalibration } from './config.js';
import { defaultCameraPoseAndRotation } from './rayplane.js';

// ── Constants (mirrored from Python processing_core.py) ───────────────

const ONNX_MODEL_URL = new URL('../vendor/onnx/boat_keypoints.onnx', import.meta.url).href;
const MODEL_INPUT_SIZE = 640;
const MIN_KPT_CONF = 0.8;
const MIN_PAIRS = 6;

const KEYPOINT_LABELS = [
  'frontdeck', 'porttop', 'portmid', 'portlow',
  'starboardtop', 'starboardmid', 'starboardlow',
  'portback', 'starboardback',
];

/** 3D object points for each keypoint (boat reference frame, in meters) */
const OBJECT_POINTS_3D = {
  frontdeck:     [-0.531,  0.000,  0.001],
  porttop:       [-1.174, -0.005, -0.308],
  portmid:       [-1.182, -0.060, -0.294],
  portlow:       [-1.189, -0.176, -0.295],
  portback:      [-2.169, -0.007, -0.284],
  starboardtop:  [-1.165, -0.005,  0.334],
  starboardmid:  [-1.162, -0.060,  0.312],
  starboardlow:  [-1.177, -0.185,  0.313],
  starboardback: [-2.169, -0.012,  0.306],
};

const BOUNDS = {
  pitch_min_deg: 10.0, pitch_max_deg: 23.0,
  yaw_abs_max_deg: 10.0, roll_abs_max_deg: 5.0,
  x_min_m: -3.4, x_max_m: -3.1,
  y_abs_max_m: 0.1, z_min_m: 0.5, z_max_m: 0.8,
  max_reproj_err_px: 30.0,
};

/** Default initial camera pose (same as rayplane.js default) */
export const DEFAULT_INITIAL_POSE = {
  pitch: 14.7, yaw: 0.0, roll: 0.0,
  x: -3.194, y: 0.02, z: 0.585,
};

// ── ONNX Runtime lazy loader ──────────────────────────────────────────

let _ort = null;
let _session = null;
let _loadingPromise = null;
const _preprocessState = {
  canvas: null,
  ctx: null,
  tensorData: null,
  tensorDataInUse: false,
};

async function ensureOrt() {
  if (_ort) return _ort;
  // Load ONNX Runtime Web from CDN
  if (!window.ort) {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js';
    document.head.appendChild(script);
    await new Promise((resolve, reject) => {
      script.onload = resolve;
      script.onerror = () => reject(new Error('Failed to load ONNX Runtime Web'));
    });
  }
  _ort = window.ort;
  return _ort;
}

async function ensureSession() {
  if (_session) return _session;
  if (_loadingPromise) return _loadingPromise;

  _loadingPromise = (async () => {
    const ort = await ensureOrt();
    ort.env.wasm.numThreads = 1;
    // Try WebGL first for GPU acceleration, fall back to WASM
    let session;
    try {
      session = await ort.InferenceSession.create(ONNX_MODEL_URL, {
        executionProviders: ['webgl'],
      });
    } catch {
      session = await ort.InferenceSession.create(ONNX_MODEL_URL, {
        executionProviders: ['wasm'],
      });
    }
    _session = session;
    return session;
  })();

  try {
    return await _loadingPromise;
  } catch (e) {
    _loadingPromise = null;
    throw e;
  }
}


// ── Image preprocessing ───────────────────────────────────────────────

/**
 * Preprocess a canvas/video frame for YOLO input.
 * YOLO expects [1, 3, 640, 640] normalized float32 tensor.
 */
function preprocessFrame(source, inputSize) {
  if (!_preprocessState.canvas || !_preprocessState.ctx || _preprocessState.canvas.width !== inputSize || _preprocessState.canvas.height !== inputSize) {
    const canvas = document.createElement('canvas');
    canvas.width = inputSize;
    canvas.height = inputSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not create AutoPnP preprocessing canvas context');
    _preprocessState.canvas = canvas;
    _preprocessState.ctx = ctx;
  }
  const canvas = _preprocessState.canvas;
  const ctx = _preprocessState.ctx;

  // Letterbox: fit source into square
  const sw = source.videoWidth || source.width;
  const sh = source.videoHeight || source.height;
  const scale = Math.min(inputSize / sw, inputSize / sh);
  const nw = Math.round(sw * scale);
  const nh = Math.round(sh * scale);
  const dx = Math.round((inputSize - nw) / 2);
  const dy = Math.round((inputSize - nh) / 2);

  ctx.fillStyle = '#808080'; // gray letterbox
  ctx.fillRect(0, 0, inputSize, inputSize);
  ctx.drawImage(source, 0, 0, sw, sh, dx, dy, nw, nh);

  const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
  const data = imageData.data;
  const pixels = inputSize * inputSize;
  const tensorLength = 3 * pixels;
  let float32;
  let release = () => {};
  if (!_preprocessState.tensorDataInUse) {
    if (!_preprocessState.tensorData || _preprocessState.tensorData.length !== tensorLength) {
      _preprocessState.tensorData = new Float32Array(tensorLength);
    }
    _preprocessState.tensorDataInUse = true;
    float32 = _preprocessState.tensorData;
    release = () => { _preprocessState.tensorDataInUse = false; };
  } else {
    float32 = new Float32Array(tensorLength);
  }

  // CHW format, normalized to [0, 1]
  for (let i = 0; i < pixels; i++) {
    float32[i] = data[i * 4] / 255.0;           // R
    float32[pixels + i] = data[i * 4 + 1] / 255.0; // G
    float32[pixels * 2 + i] = data[i * 4 + 2] / 255.0; // B
  }

  return { tensor: float32, release, scale, dx, dy, origW: sw, origH: sh };
}


// ── YOLO output parsing ──────────────────────────────────────────────

/**
 * Parse YOLOv8-pose output tensor.
 * Output shape: [1, 32, 8400] for 9 keypoints (4 box + 1 conf + 9*3 kpts = 32)
 * Returns best detection's keypoints in original image coordinates.
 */
function parseYoloPoseOutput(outputData, numDetections, meta) {
  const { scale, dx, dy } = meta;
  const numKpts = 9;
  // Output: [1, 4+1+numKpts*3, numDetections]
  // Channels: cx, cy, w, h, conf, kpt0_x, kpt0_y, kpt0_conf, kpt1_x, ...
  const channels = 4 + 1 + numKpts * 3; // 32

  let bestConf = 0;
  let bestIdx = -1;

  for (let i = 0; i < numDetections; i++) {
    const conf = outputData[4 * numDetections + i];
    if (conf > bestConf) { bestConf = conf; bestIdx = i; }
  }

  if (bestIdx < 0 || bestConf < 0.25) return null;

  const keypoints = [];
  for (let k = 0; k < numKpts; k++) {
    const baseChannel = 5 + k * 3;
    const kx = outputData[baseChannel * numDetections + bestIdx];
    const ky = outputData[(baseChannel + 1) * numDetections + bestIdx];
    const kconf = outputData[(baseChannel + 2) * numDetections + bestIdx];

    // Convert from letterboxed coords back to original image coords
    const origX = (kx - dx) / scale;
    const origY = (ky - dy) / scale;

    keypoints.push({ x: origX, y: origY, conf: kconf });
  }

  return { confidence: bestConf, keypoints };
}


// ── Geometric keypoint binding ────────────────────────────────────────

/**
 * Bind detected keypoints to semantic labels based on geometry.
 * Frontdeck = highest Y (nearest top of image).
 * Left/right split by X. Within each side: back = furthest from front,
 * remaining sorted by Y for top/mid/low.
 */
function bindKeypointsToLabels(keypoints) {
  const n = keypoints.length;
  if (n < 9) return null;

  const pts = keypoints.slice(0, 9);

  // Front = highest in image (smallest Y)
  let frontIdx = 0;
  for (let i = 1; i < n && i < 9; i++) {
    if (pts[i].y < pts[frontIdx].y) frontIdx = i;
  }

  const remaining = [];
  for (let i = 0; i < 9; i++) if (i !== frontIdx) remaining.push(i);

  // Sort remaining by X to split left/right
  remaining.sort((a, b) => pts[a].x - pts[b].x);
  const leftSide = remaining.slice(0, 4);
  const rightSide = remaining.slice(4, 8);

  function splitSide(indices) {
    const dists = indices.map(i => {
      const ddx = pts[i].x - pts[frontIdx].x;
      const ddy = pts[i].y - pts[frontIdx].y;
      return { i, d: Math.sqrt(ddx*ddx + ddy*ddy) };
    });
    dists.sort((a, b) => b.d - a.d);
    const backIdx = dists[0].i;
    const rail = dists.slice(1).map(d => d.i);
    rail.sort((a, b) => pts[a].y - pts[b].y);
    return { top: rail[0], mid: rail[1], low: rail[2], back: backIdx };
  }

  const left = splitSide(leftSide);
  const right = splitSide(rightSide);

  return {
    frontdeck: frontIdx,
    porttop: left.top, portmid: left.mid, portlow: left.low, portback: left.back,
    starboardtop: right.top, starboardmid: right.mid, starboardlow: right.low, starboardback: right.back,
  };
}


// ── Rodrigues rotation helpers ────────────────────────────────────────

/** Rodrigues vector (axis*angle) → 3×3 rotation matrix (flat row-major). */
function _rodriguesExp(v) {
  const theta = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (theta < 1e-12) return [1,0,0, 0,1,0, 0,0,1];
  const k0 = v[0]/theta, k1 = v[1]/theta, k2 = v[2]/theta;
  const c = Math.cos(theta), s = Math.sin(theta), t = 1 - c;
  return [
    c + k0*k0*t,     k0*k1*t - k2*s, k0*k2*t + k1*s,
    k1*k0*t + k2*s,  c + k1*k1*t,    k1*k2*t - k0*s,
    k2*k0*t - k1*s,  k2*k1*t + k0*s, c + k2*k2*t,
  ];
}

/** 3×3 rotation matrix (flat row-major) → Rodrigues vector. */
function _rodriguesLog(R) {
  const trace = R[0] + R[4] + R[8];
  const cosTheta = Math.max(-1, Math.min(1, (trace - 1) / 2));
  const theta = Math.acos(cosTheta);
  if (theta < 1e-10) return [0, 0, 0];
  const s2 = 2 * Math.sin(theta);
  if (Math.abs(s2) < 1e-12) {
    // theta ≈ π — pick dominant axis
    const d0 = R[0], d1 = R[4], d2 = R[8];
    if (d0 >= d1 && d0 >= d2) return [theta, 0, 0];
    if (d1 >= d2) return [0, theta, 0];
    return [0, 0, theta];
  }
  const f = theta / s2;
  return [f * (R[7] - R[5]), f * (R[2] - R[6]), f * (R[3] - R[1])];
}

/** Flat 3×3 transpose. */
function _mat3T(M) { return [M[0],M[3],M[6], M[1],M[4],M[7], M[2],M[5],M[8]]; }

/** Flat 3×3 × vec3. */
function _mat3vec(M, v) {
  return [M[0]*v[0]+M[1]*v[1]+M[2]*v[2],
          M[3]*v[0]+M[4]*v[1]+M[5]*v[2],
          M[6]*v[0]+M[7]*v[1]+M[8]*v[2]];
}

/** Flat 3×3 × flat 3×3. */
function _mat3mul(A, B) {
  return [
    A[0]*B[0]+A[1]*B[3]+A[2]*B[6], A[0]*B[1]+A[1]*B[4]+A[2]*B[7], A[0]*B[2]+A[1]*B[5]+A[2]*B[8],
    A[3]*B[0]+A[4]*B[3]+A[5]*B[6], A[3]*B[1]+A[4]*B[4]+A[5]*B[7], A[3]*B[2]+A[4]*B[5]+A[5]*B[8],
    A[6]*B[0]+A[7]*B[3]+A[8]*B[6], A[6]*B[1]+A[7]*B[4]+A[8]*B[7], A[6]*B[2]+A[7]*B[5]+A[8]*B[8],
  ];
}

// ── Base rotation & correction constants (matching Python) ────────────

/** Same as Python AUTO_CAMERA_BASE_R_WC (and rayplane.js BASE_ROT). */
const BASE_R_WC = [0,0,1, -1,0,0, 0,-1,0];

const ROLL_OFFSET_DEG = 90.0;

// ── Fisheye undistortion (matching cv2.fisheye.undistortPoints) ─────────

/**
 * Undistort 2D pixel points from fisheye-distorted image coordinates
 * to undistorted image coordinates.
 *
 * Matches cv2.fisheye.undistortPoints(pts, K, D, P=K_new).
 *
 * OpenCV fisheye model:
 *   θ_d = θ(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
 * where θ = atan(r) and r = sqrt(x²+y²) in normalized pinhole coords.
 *
 * @param {number[][]} pts  - Array of [u, v] distorted pixel coordinates
 * @param {number[][]} K    - 3×3 raw camera matrix (distorted)
 * @param {number[][]} D    - 4×1 distortion coefficients [[k1],[k2],[k3],[k4]]
 * @param {number[][]} K_new - 3×3 new camera matrix (undistorted)
 * @returns {number[][]} Array of [u', v'] undistorted pixel coordinates
 */
function _fisheyeUndistortPoints(pts, K, D, K_new) {
  const fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
  const fx_n = K_new[0][0], fy_n = K_new[1][1], cx_n = K_new[0][2], cy_n = K_new[1][2];
  const k1 = D[0][0], k2 = D[1][0], k3 = D[2][0], k4 = D[3][0];

  return pts.map(([u, v]) => {
    // 1. Normalize with raw camera matrix
    const x_d = (u - cx) / fx;
    const y_d = (v - cy) / fy;
    const theta_d = Math.sqrt(x_d * x_d + y_d * y_d);

    if (theta_d < 1e-10) {
      // Point at optical center — no distortion
      return [fx_n * x_d + cx_n, fy_n * y_d + cy_n];
    }

    // 2. Solve for θ from θ_d = θ(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
    //    via Newton's method
    let theta = theta_d;
    for (let i = 0; i < 20; i++) {
      const t2 = theta * theta;
      const t4 = t2 * t2;
      const t6 = t4 * t2;
      const t8 = t4 * t4;
      const f  = theta * (1 + k1*t2 + k2*t4 + k3*t6 + k4*t8) - theta_d;
      const fp = 1 + 3*k1*t2 + 5*k2*t4 + 7*k3*t6 + 9*k4*t8;
      if (Math.abs(fp) < 1e-15) break;
      const dt = f / fp;
      theta -= dt;
      if (Math.abs(dt) < 1e-12) break;
    }

    // 3. Undistorted normalized coords: r = tan(θ), scale = r / θ_d
    const r = Math.tan(theta);
    const scale = r / theta_d;
    const x_u = scale * x_d;
    const y_u = scale * y_d;

    // 4. Re-project with new camera matrix
    return [fx_n * x_u + cx_n, fy_n * y_u + cy_n];
  });
}

// ── Linear algebra helpers ────────────────────────────────────────────

/**
 * Jacobi eigenvalue decomposition for an n×n SYMMETRIC matrix.
 * Returns { values: Float64Array, vectors: Float64Array[] (column-major) }.
 * Eigenvectors are columns of the returned 2-D array (vectors[row][col]).
 */
function _jacobiEigen(S_in, n) {
  const S = Array.from({ length: n }, (_, i) => Float64Array.from(S_in[i]));
  const V = Array.from({ length: n }, (_, i) => {
    const r = new Float64Array(n);
    r[i] = 1;
    return r;
  });
  for (let sweep = 0; sweep < 50; sweep++) {
    let offNorm = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) offNorm += S[i][j] * S[i][j];
    if (offNorm < 1e-28) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(S[p][q]) < 1e-15) continue;
        const d = S[q][q] - S[p][p];
        let t;
        if (Math.abs(S[p][q]) < 1e-15 * Math.abs(d)) {
          t = S[p][q] / d;
        } else {
          const phi = d / (2 * S[p][q]);
          t = 1 / (Math.abs(phi) + Math.sqrt(1 + phi * phi));
          if (phi < 0) t = -t;
        }
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        const tau = s / (1 + c);
        const Spq = S[p][q];
        S[p][p] -= t * Spq;
        S[q][q] += t * Spq;
        S[p][q] = 0;
        S[q][p] = 0;
        for (let i = 0; i < n; i++) {
          if (i !== p && i !== q) {
            const Sip = S[i][p], Siq = S[i][q];
            S[i][p] = S[p][i] = Sip - s * (Siq + tau * Sip);
            S[i][q] = S[q][i] = Siq + s * (Sip - tau * Siq);
          }
        }
        for (let i = 0; i < n; i++) {
          const Vip = V[i][p], Viq = V[i][q];
          V[i][p] = Vip - s * (Viq + tau * Vip);
          V[i][q] = Viq + s * (Vip - tau * Viq);
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = S[i][i];
  return { values, vectors: V };
}

/** Gauss elimination with partial pivoting for n×n system Ax = b. */
function _solveLinearNxN(A, b) {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    if (Math.abs(aug[col][col]) < 1e-15) return null;
    for (let row = col + 1; row < n; row++) {
      const f = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) aug[row][j] -= f * aug[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = aug[i][n];
    for (let j = i + 1; j < n; j++) s -= aug[i][j] * x[j];
    x[i] = s / aug[i][i];
  }
  return x;
}


// ── DLT PnP solver (algebraic initialization) ────────────────────────

/**
 * Direct Linear Transform PnP. Solves for [R|t] algebraically.
 * Matches OpenCV SOLVEPNP_ITERATIVE's internal DLT initialization.
 *
 * @param {number[][]} objPts3D - 3D object points
 * @param {number[][]} imgPts2D - 2D image points (pixels)
 * @param {number[]} K - flat 9-element camera intrinsic matrix
 * @returns {{ rvec: number[3], tvec: number[3] } | null}
 */
function _solvePnPDLT(objPts3D, imgPts2D, K) {
  const n = objPts3D.length;
  if (n < 6) return null;

  const fx = K[0], fy = K[4], cx = K[2], cy = K[5];

  // ── Hartley normalization: center & scale 3D and 2D points ──
  // 3D: centroid and isotropic scale
  let cx3 = 0, cy3 = 0, cz3 = 0;
  for (const [X, Y, Z] of objPts3D) { cx3 += X; cy3 += Y; cz3 += Z; }
  cx3 /= n; cy3 /= n; cz3 /= n;
  let avgD3 = 0;
  for (const [X, Y, Z] of objPts3D) {
    avgD3 += Math.sqrt((X-cx3)**2 + (Y-cy3)**2 + (Z-cz3)**2);
  }
  avgD3 /= n;
  const s3 = avgD3 > 1e-10 ? Math.sqrt(3) / avgD3 : 1.0;

  // 2D: K-normalize, then center and scale
  const imgKN = imgPts2D.map(([u, v]) => [(u - cx) / fx, (v - cy) / fy]);
  let mx2 = 0, my2 = 0;
  for (const [x, y] of imgKN) { mx2 += x; my2 += y; }
  mx2 /= n; my2 /= n;
  let avgD2 = 0;
  for (const [x, y] of imgKN) { avgD2 += Math.sqrt((x-mx2)**2 + (y-my2)**2); }
  avgD2 /= n;
  const s2 = avgD2 > 1e-10 ? Math.SQRT2 / avgD2 : 1.0;

  // Build 2n × 12 system: M q = 0  (with normalized points)
  // q = [r00,r01,r02,tx, r10,r11,r12,ty, r20,r21,r22,tz]
  const rows = 2 * n, cols = 12;
  const M = new Array(rows);
  for (let i = 0; i < n; i++) {
    const X = s3 * (objPts3D[i][0] - cx3);
    const Y = s3 * (objPts3D[i][1] - cy3);
    const Z = s3 * (objPts3D[i][2] - cz3);
    const xn = s2 * (imgKN[i][0] - mx2);
    const yn = s2 * (imgKN[i][1] - my2);
    M[2 * i]     = [-X, -Y, -Z, -1, 0, 0, 0, 0, xn*X, xn*Y, xn*Z, xn];
    M[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, yn*X, yn*Y, yn*Z, yn];
  }

  // Compute M^T M (12 × 12 symmetric)
  const MtM = Array.from({ length: cols }, () => new Array(cols).fill(0));
  for (let i = 0; i < cols; i++) {
    for (let j = i; j < cols; j++) {
      let s = 0;
      for (let k = 0; k < rows; k++) s += M[k][i] * M[k][j];
      MtM[i][j] = s;
      MtM[j][i] = s;
    }
  }

  // Eigenvector of M^T M with smallest eigenvalue = null-space of M
  const { values, vectors } = _jacobiEigen(MtM, cols);
  let minIdx = 0;
  for (let i = 1; i < cols; i++) {
    if (values[i] < values[minIdx]) minIdx = i;
  }
  const q = new Array(cols);
  for (let i = 0; i < cols; i++) q[i] = vectors[i][minIdx];

  // Extract R' and t' from null vector (in normalized space)
  const Rn = [q[0],q[1],q[2], q[4],q[5],q[6], q[8],q[9],q[10]];
  const tn = [q[3], q[7], q[11]];

  // ── Denormalize: P = T2_inv @ [R'|t'] @ T3 ──
  // Step 1: A = [R'|t'] @ T3 → A_R = s3*R', A_t = -s3*(R'@centroid3d) + t'
  const A_R = Rn.map(v => v * s3);
  const A_t = [
    -(A_R[0]*cx3 + A_R[1]*cy3 + A_R[2]*cz3) + tn[0],
    -(A_R[3]*cx3 + A_R[4]*cy3 + A_R[5]*cz3) + tn[1],
    -(A_R[6]*cx3 + A_R[7]*cy3 + A_R[8]*cz3) + tn[2],
  ];
  // Step 2: P = T2_inv @ A
  const R_raw = [
    A_R[0]/s2 + mx2*A_R[6], A_R[1]/s2 + mx2*A_R[7], A_R[2]/s2 + mx2*A_R[8],
    A_R[3]/s2 + my2*A_R[6], A_R[4]/s2 + my2*A_R[7], A_R[5]/s2 + my2*A_R[8],
    A_R[6],                  A_R[7],                  A_R[8],
  ];
  const t_raw = [
    A_t[0]/s2 + mx2*A_t[2],
    A_t[1]/s2 + my2*A_t[2],
    A_t[2],
  ];

  // Determine scale from R row norms (should be ~1 each for a valid rotation)
  const n0 = Math.sqrt(R_raw[0]*R_raw[0] + R_raw[1]*R_raw[1] + R_raw[2]*R_raw[2]);
  const n1 = Math.sqrt(R_raw[3]*R_raw[3] + R_raw[4]*R_raw[4] + R_raw[5]*R_raw[5]);
  const n2 = Math.sqrt(R_raw[6]*R_raw[6] + R_raw[7]*R_raw[7] + R_raw[8]*R_raw[8]);
  const avgNorm = (n0 + n1 + n2) / 3;
  if (avgNorm < 1e-10) return null;
  const sc = 1 / avgNorm;
  for (let i = 0; i < 9; i++) R_raw[i] *= sc;
  const tvec = [t_raw[0] * sc, t_raw[1] * sc, t_raw[2] * sc];

  // Project R_raw onto SO(3) via polar decomposition:
  // R^T R = V Σ² V^T, then R_valid = R_raw * V * diag(1/σᵢ) * V^T
  const RtR = new Array(3).fill(null).map(() => new Array(3).fill(0));
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += R_raw[k * 3 + i] * R_raw[k * 3 + j];
      RtR[i][j] = s;
    }
  const eig3 = _jacobiEigen(RtR, 3);
  // Sort eigenvalues descending
  const order = [0, 1, 2].sort((a, b) => eig3.values[b] - eig3.values[a]);
  // V matrix (columns)
  const V3 = Array.from({ length: 3 }, (_, r) =>
    [eig3.vectors[r][order[0]], eig3.vectors[r][order[1]], eig3.vectors[r][order[2]]]
  );
  const sigma = order.map(k => Math.sqrt(Math.max(0, eig3.values[k])));
  // U = R_raw * V * diag(1/σ)
  const U3 = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += R_raw[i * 3 + k] * V3[k][j];
      U3[i][j] = sigma[j] > 1e-10 ? s / sigma[j] : 0;
    }
  // R_valid = U * V^T
  const R_valid = new Array(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += U3[i][k] * V3[j][k];
      R_valid[i * 3 + j] = s;
    }
  // Ensure det(R_valid) = +1
  const det = R_valid[0]*(R_valid[4]*R_valid[8]-R_valid[5]*R_valid[7])
    - R_valid[1]*(R_valid[3]*R_valid[8]-R_valid[5]*R_valid[6])
    + R_valid[2]*(R_valid[3]*R_valid[7]-R_valid[4]*R_valid[6]);
  if (det < 0) {
    for (let i = 0; i < 9; i++) R_valid[i] = -R_valid[i];
    tvec[0] = -tvec[0]; tvec[1] = -tvec[1]; tvec[2] = -tvec[2];
  }

  // Ensure points are in front of camera (positive z_cam)
  let nFront = 0;
  for (let i = 0; i < n; i++) {
    const [X, Y, Z] = objPts3D[i];
    const zc = R_valid[6]*X + R_valid[7]*Y + R_valid[8]*Z + tvec[2];
    if (zc > 0) nFront++;
  }
  if (nFront < n / 2) {
    for (let i = 0; i < 9; i++) R_valid[i] = -R_valid[i];
    tvec[0] = -tvec[0]; tvec[1] = -tvec[1]; tvec[2] = -tvec[2];
  }

  const rvec = _rodriguesLog(R_valid);
  return { rvec, tvec };
}


// ── LM refinement (Levenberg-Marquardt, same parameterization as OpenCV) ─

/**
 * Refine rvec/tvec via Levenberg-Marquardt.
 * Used as a second stage after DLT initialization.
 */
function _refinePnPLM(objPts3D, imgPts2D, K, initRvec, initTvec) {
  const n = objPts3D.length;
  if (n < 4) return null;

  const fx = K[0], fy = K[4], cx = K[2], cy = K[5];
  let params = [...initRvec, ...initTvec];

  function computeResiduals(p) {
    const R = _rodriguesExp([p[0], p[1], p[2]]);
    const t = [p[3], p[4], p[5]];
    const res = new Array(2 * n);
    for (let i = 0; i < n; i++) {
      const [X, Y, Z] = objPts3D[i];
      const xc = R[0]*X + R[1]*Y + R[2]*Z + t[0];
      const yc = R[3]*X + R[4]*Y + R[5]*Z + t[1];
      const zc = R[6]*X + R[7]*Y + R[8]*Z + t[2];
      if (Math.abs(zc) < 1e-10) { res[2*i] = 1e4; res[2*i+1] = 1e4; continue; }
      res[2*i]   = fx * xc / zc + cx - imgPts2D[i][0];
      res[2*i+1] = fy * yc / zc + cy - imgPts2D[i][1];
    }
    return res;
  }

  const EPS = [1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4];
  const m = 2 * n;
  let lambda = 1e-3;
  let prevCost = Infinity;

  for (let iter = 0; iter < 120; iter++) {
    const r = computeResiduals(params);
    const cost = r.reduce((s, v) => s + v * v, 0);
    if (cost < 0.01 || (iter > 5 && Math.abs(prevCost - cost) < 1e-8 * cost)) break;
    prevCost = cost;

    const J = new Array(m);
    for (let i = 0; i < m; i++) J[i] = new Array(6);
    for (let j = 0; j < 6; j++) {
      const pp = [...params]; pp[j] += EPS[j];
      const rp = computeResiduals(pp);
      for (let i = 0; i < m; i++) J[i][j] = (rp[i] - r[i]) / EPS[j];
    }

    const JTJ = Array.from({ length: 6 }, () => new Array(6).fill(0));
    const JTr = new Array(6).fill(0);
    for (let a = 0; a < 6; a++) {
      for (let b = 0; b < 6; b++) {
        let s = 0;
        for (let k = 0; k < m; k++) s += J[k][a] * J[k][b];
        JTJ[a][b] = s;
      }
      let s = 0;
      for (let k = 0; k < m; k++) s += J[k][a] * r[k];
      JTr[a] = s;
    }
    for (let i = 0; i < 6; i++) JTJ[i][i] *= (1 + lambda);

    const delta = _solveLinearNxN(JTJ, JTr.map(v => -v));
    if (!delta) break;

    const trial = params.map((v, i) => v + delta[i]);
    const trialCost = computeResiduals(trial).reduce((s, v) => s + v * v, 0);

    if (trialCost < cost) {
      params = trial;
      lambda = Math.max(lambda * 0.3, 1e-7);
    } else {
      lambda = Math.min(lambda * 3.0, 1e6);
    }
  }

  return {
    rvec: [params[0], params[1], params[2]],
    tvec: [params[3], params[4], params[5]],
  };
}


// ── RANSAC PnP ───────────────────────────────────────────────────────

/**
 * RANSAC wrapper around DLT: sample 6 points, solve DLT, count inliers.
 * Matching cv2.solvePnPRansac behaviour.
 *
 * @returns {{ rvec, tvec, inlierIdx: number[] } | null}
 */
function _solvePnPRansac(objPts3D, imgPts2D, K, nIters, reprojThresh) {
  const n = objPts3D.length;
  if (n < 6) return null;

  const fx = K[0], fy = K[4], cx = K[2], cy = K[5];
  let bestInliers = [];
  let bestRvec = null, bestTvec = null;

  function reprojErrors(rvec, tvec) {
    const R = _rodriguesExp(rvec);
    const t = tvec;
    const errs = new Array(n);
    for (let i = 0; i < n; i++) {
      const [X, Y, Z] = objPts3D[i];
      const xc = R[0]*X + R[1]*Y + R[2]*Z + t[0];
      const yc = R[3]*X + R[4]*Y + R[5]*Z + t[1];
      const zc = R[6]*X + R[7]*Y + R[8]*Z + t[2];
      if (Math.abs(zc) < 1e-10) { errs[i] = 1e6; continue; }
      const du = fx * xc / zc + cx - imgPts2D[i][0];
      const dv = fy * yc / zc + cy - imgPts2D[i][1];
      errs[i] = Math.sqrt(du * du + dv * dv);
    }
    return errs;
  }

  for (let iter = 0; iter < nIters; iter++) {
    // Sample 6 random indices
    const sample = [];
    const used = new Set();
    while (sample.length < 6) {
      const idx = Math.floor(Math.random() * n);
      if (!used.has(idx)) { used.add(idx); sample.push(idx); }
    }
    const sObj = sample.map(i => objPts3D[i]);
    const sImg = sample.map(i => imgPts2D[i]);
    const dlt = _solvePnPDLT(sObj, sImg, K);
    if (!dlt) continue;

    const errs = reprojErrors(dlt.rvec, dlt.tvec);
    const inliers = [];
    for (let i = 0; i < n; i++) {
      if (errs[i] < reprojThresh) inliers.push(i);
    }
    if (inliers.length > bestInliers.length) {
      bestInliers = inliers;
      bestRvec = dlt.rvec;
      bestTvec = dlt.tvec;
      // Early stop if all inliers
      if (inliers.length === n) break;
    }
  }

  if (bestInliers.length < 4 || !bestRvec) return null;
  return { rvec: bestRvec, tvec: bestTvec, inlierIdx: bestInliers };
}


// ── Multi-candidate PnP solver (matches Python _solve_pnp_pose) ─────

/**
 * Compute reprojection errors for all points given rvec/tvec.
 * @returns {number[]} per-point pixel errors
 */
function _reprojectionErrors(objPts, imgPts, K, rvec, tvec) {
  const fx = K[0], fy = K[4], cx = K[2], cy = K[5];
  const R = _rodriguesExp(rvec);
  const t = tvec;
  return objPts.map(([X, Y, Z], i) => {
    const xc = R[0]*X + R[1]*Y + R[2]*Z + t[0];
    const yc = R[3]*X + R[4]*Y + R[5]*Z + t[1];
    const zc = R[6]*X + R[7]*Y + R[8]*Z + t[2];
    if (Math.abs(zc) < 1e-10) return 1e6;
    const du = fx * xc / zc + cx - imgPts[i][0];
    const dv = fy * yc / zc + cy - imgPts[i][1];
    return Math.sqrt(du * du + dv * dv);
  });
}

/** Soft range-violation penalty (matching Python _range_violation). */
function _rangeViolation(v, lo, hi) {
  const span = Math.max(1e-6, Math.abs(hi - lo));
  if (v < lo) return (lo - v) / span;
  if (v > hi) return (v - hi) / span;
  return 0;
}
function _absViolation(v, lim) {
  return Math.max(Math.abs(v) - lim, 0) / Math.max(1e-6, lim);
}

/**
 * Build and score a PnP candidate exactly matching Python _build_candidate.
 * Lower score = better.
 */
function _buildCandidate(method, rvec, tvec, inlierIdx, objPts, imgPts, K) {
  const residuals = _reprojectionErrors(objPts, imgPts, K, rvec, tvec);
  const sorted = [...residuals].sort((a, b) => a - b);
  const medianAll = sorted[Math.floor(sorted.length / 2)];
  const inlierResids = inlierIdx.map(i => residuals[i]);
  const inlierErr = inlierResids.length
    ? inlierResids.reduce((a, b) => a + b, 0) / inlierResids.length
    : medianAll;

  // Compute corrected pose for plausibility scoring
  const R_cw = _rodriguesExp(rvec);
  const R_wc_raw = _mat3T(R_cw);
  const camPos_raw = _mat3vec(R_wc_raw, [-tvec[0], -tvec[1], -tvec[2]]);
  const rawAngles = _cameraAnglesFromRwc(R_wc_raw);
  const corrRoll = -(rawAngles.roll_deg + ROLL_OFFSET_DEG);
  const { R_wc: flatCorr } = defaultCameraPoseAndRotation(
    rawAngles.pitch_deg, rawAngles.yaw_deg, corrRoll
  );
  const camPos_corr = [camPos_raw[0], camPos_raw[2], camPos_raw[1]];
  const corrAngles = _cameraAnglesFromRwc(flatCorr);
  corrAngles.yaw_deg = _wrapDeg(corrAngles.yaw_deg);
  corrAngles.roll_deg = _wrapDeg(corrAngles.roll_deg);

  const b = BOUNDS;
  const [x, y, z] = camPos_corr;
  let violation = 0;
  violation += _rangeViolation(corrAngles.pitch_deg, b.pitch_min_deg, b.pitch_max_deg);
  violation += _absViolation(corrAngles.yaw_deg, b.yaw_abs_max_deg);
  violation += _absViolation(corrAngles.roll_deg, b.roll_abs_max_deg);
  violation += _rangeViolation(x, b.x_min_m, b.x_max_m);
  violation += _absViolation(y, b.y_abs_max_m);
  violation += _rangeViolation(z, b.z_min_m, b.z_max_m);
  violation += Math.max(inlierErr - b.max_reproj_err_px, 0) / Math.max(1e-6, b.max_reproj_err_px);

  const inlierRatio = inlierIdx.length / objPts.length;
  const lowInlierPenalty = Math.max(0, 0.70 - inlierRatio) * 25;
  const score = medianAll + 25 * violation + lowInlierPenalty;

  return {
    method,
    rvec: [...rvec],
    tvec: [...tvec],
    score,
    medianErrorPx: medianAll,
    inlierErr,
    numInliers: inlierIdx.length,
    numPairs: objPts.length,
  };
}

/**
 * Compute initial rvec/tvec from an approximate camera pose in corrected space.
 * The corrected-to-raw conversion reverses the y/z swap and roll+90° offset,
 * providing a good starting point for LM that bypasses the DLT entirely.
 */
function _computeInitialRvecTvec(initialPose) {
  const p = initialPose || DEFAULT_INITIAL_POSE;
  const pitch = p.pitch ?? 14.7;
  const yaw   = p.yaw   ?? 0.0;
  const roll  = p.roll  ?? 0.0;
  const x     = p.x     ?? -3.194;
  const y     = p.y     ?? 0.02;
  const z     = p.z     ?? 0.585;

  // Raw position: un-swap y/z
  const rawPos = [x, z, y];
  // Raw roll: corrected_roll = -(raw_roll + 90) → raw_roll = -corrected_roll - 90
  const rawRoll = -roll - 90.0;
  const { R_wc: R_wc_raw } = defaultCameraPoseAndRotation(pitch, yaw, rawRoll);

  // R_cw = R_wc^T, tvec = -R_cw @ rawPos
  const R_cw = _mat3T(R_wc_raw);
  const tvec = _mat3vec(R_cw, [-rawPos[0], -rawPos[1], -rawPos[2]]);
  const rvec = _rodriguesLog(R_cw);
  return { rvec, tvec };
}

/**
 * Multi-candidate PnP solver matching Python _solve_pnp_pose.
 * Tries RANSAC+DLT+LM, DLT+LM, and Pose-prior+LM, scores candidates
 * with soft plausibility penalties, returns best.
 *
 * @returns {{ rvec, tvec, method, medianErrorPx, numInliers, numPairs, score } | null}
 */
function _solvePnPPose(objPts, imgPts, K, initialPose = null) {
  const n = objPts.length;
  if (n < 4) return null;

  const candidates = [];
  const allIdx = Array.from({ length: n }, (_, i) => i);

  // Candidate A: RANSAC + DLT, then LM on inliers (matching Python ransac_epnp)
  if (n >= 6) {
    const ransac = _solvePnPRansac(objPts, imgPts, K, 400, 3.0);
    if (ransac && ransac.inlierIdx.length >= 4) {
      let rv = ransac.rvec, tv = ransac.tvec;
      // Refine with LM on inliers
      const inObj = ransac.inlierIdx.map(i => objPts[i]);
      const inImg = ransac.inlierIdx.map(i => imgPts[i]);
      const ref = _refinePnPLM(inObj, inImg, K, rv, tv);
      if (ref) { rv = ref.rvec; tv = ref.tvec; }
      candidates.push(_buildCandidate('ransac_dlt', rv, tv, ransac.inlierIdx, objPts, imgPts, K));
    }
  }

  // Candidate B: DLT on all points + LM refine (matching Python iterative_all)
  const dlt = _solvePnPDLT(objPts, imgPts, K);
  if (dlt) {
    let rv = dlt.rvec, tv = dlt.tvec;
    const ref = _refinePnPLM(objPts, imgPts, K, rv, tv);
    if (ref) { rv = ref.rvec; tv = ref.tvec; }
    candidates.push(_buildCandidate('iterative_all', rv, tv, allIdx, objPts, imgPts, K));
  }

  // Candidate C: Pose-prior + LM (initialize from approximate camera pose)
  {
    const { rvec: initRv, tvec: initTv } = _computeInitialRvecTvec(initialPose);
    const ref = _refinePnPLM(objPts, imgPts, K, initRv, initTv);
    if (ref) {
      candidates.push(_buildCandidate('pose_prior_lm', ref.rvec, ref.tvec, allIdx, objPts, imgPts, K));
    }
  }

  if (candidates.length === 0) return null;

  // Pick best by score (lower is better)
  return candidates.reduce((a, b) => a.score < b.score ? a : b);
}


// ── Correction & validation (matching Python _attempt_auto_camera_pnp) ─

/** Extract angles from R_wc via BASE_R_WC^T @ R_wc → Euler XYZ. */
function _cameraAnglesFromRwc(R_wc_flat) {
  const BT = _mat3T(BASE_R_WC);
  const R_rel = _mat3mul(BT, R_wc_flat);
  const ry = Math.asin(Math.max(-1, Math.min(1, -R_rel[6])));
  const cy = Math.cos(ry);
  let rx, rz;
  if (Math.abs(cy) > 1e-8) {
    rx = Math.atan2(R_rel[7], R_rel[8]);
    rz = Math.atan2(R_rel[3], R_rel[0]);
  } else {
    rx = Math.atan2(-R_rel[5], R_rel[4]);
    rz = 0;
  }
  const DEG = 180 / Math.PI;
  return { pitch_deg: -rx * DEG, yaw_deg: ry * DEG, roll_deg: rz * DEG };
}

function _wrapDeg(a) { return ((a + 180) % 360 + 360) % 360 - 180; }

function _getSourceSize(source) {
  const w = Number(source?.videoWidth ?? source?.width ?? 0);
  const h = Number(source?.videoHeight ?? source?.height ?? 0);
  return { w: Number.isFinite(w) ? w : 0, h: Number.isFinite(h) ? h : 0 };
}

function _scaleIntrinsics(K, sx, sy) {
  return [
    [K[0][0] * sx, K[0][1] * sx, K[0][2] * sx],
    [K[1][0] * sy, K[1][1] * sy, K[1][2] * sy],
    [K[2][0],      K[2][1],      K[2][2]],
  ];
}

function _intrinsicsForSource(calib, source) {
  if (!calib?.K || !calib?.K_undist) return null;
  const { w: srcW, h: srcH } = _getSourceSize(source);
  const calW = Number(calib?.img_size?.[0] ?? srcW);
  const calH = Number(calib?.img_size?.[1] ?? srcH);
  let sx = 1.0, sy = 1.0;
  if (srcW > 0 && srcH > 0 && calW > 0 && calH > 0) {
    sx = srcW / calW;
    sy = srcH / calH;
  }
  return {
    K: _scaleIntrinsics(calib.K, sx, sy),
    K_undist: _scaleIntrinsics(calib.K_undist, sx, sy),
  };
}

/**
 * Apply Python-style correction to raw PnP result:
 *  1) R_cw from rvec, R_wc = R_cw^T, camPos = -R_wc @ tvec
 *  2) Extract angles via BASE_R_WC
 *  3) Corrected roll = raw_roll + 90°
 *  4) Rebuild R_wc via defaultCameraPoseAndRotation
 *  5) Swap y/z in position
 *  6) Extract corrected angles, validate
 */
function _correctAndValidate(rvec, tvec, errPx) {
  const R_cw = _rodriguesExp(rvec);
  const R_wc_raw = _mat3T(R_cw);
  const camPos_raw = _mat3vec(R_wc_raw, [-tvec[0], -tvec[1], -tvec[2]]);
  const rawAngles = _cameraAnglesFromRwc(R_wc_raw);

  const corrRoll = -(rawAngles.roll_deg + ROLL_OFFSET_DEG);
  const { R_wc: flatCorr } = defaultCameraPoseAndRotation(
    rawAngles.pitch_deg, rawAngles.yaw_deg, corrRoll
  );

  const camPos_corr = [camPos_raw[0], camPos_raw[2], camPos_raw[1]];
  const corrAngles = _cameraAnglesFromRwc(flatCorr);
  corrAngles.yaw_deg = _wrapDeg(corrAngles.yaw_deg);
  corrAngles.roll_deg = _wrapDeg(corrAngles.roll_deg);

  const issues = _validatePose(camPos_corr, corrAngles, errPx);

  const R_wc_nested = [
    [flatCorr[0], flatCorr[1], flatCorr[2]],
    [flatCorr[3], flatCorr[4], flatCorr[5]],
    [flatCorr[6], flatCorr[7], flatCorr[8]],
  ];

  return { camPos: camPos_corr, R_wc: R_wc_nested, angles: corrAngles,
           rawAngles, rawCamPos: camPos_raw, issues };
}

/** Validate a corrected camera pose against physical bounds. */
function _validatePose(camPos, angles, errPx) {
  const b = BOUNDS;
  const issues = [];
  const [x, y, z] = camPos;
  const { pitch_deg: p, yaw_deg: yw, roll_deg: r } = angles;
  if (p < b.pitch_min_deg || p > b.pitch_max_deg) issues.push(`pitch=${p.toFixed(1)}°`);
  if (Math.abs(yw) > b.yaw_abs_max_deg) issues.push(`yaw=${yw.toFixed(1)}°`);
  if (Math.abs(r) > b.roll_abs_max_deg) issues.push(`roll=${r.toFixed(1)}°`);
  if (x < b.x_min_m || x > b.x_max_m) issues.push(`x=${x.toFixed(3)}m`);
  if (Math.abs(y) > b.y_abs_max_m) issues.push(`y=${y.toFixed(3)}m`);
  if (z < b.z_min_m || z > b.z_max_m) issues.push(`z=${z.toFixed(3)}m`);
  if (!isFinite(errPx) || errPx > b.max_reproj_err_px) issues.push(`err=${errPx?.toFixed(1)}px`);
  return issues;
}


// ── Public API ────────────────────────────────────────────────────────

/**
 * Detect boat keypoints in a video frame / canvas.
 * @param {HTMLCanvasElement|HTMLVideoElement} source
 * @returns {{ keypoints: Array<{x,y,conf}>, confidence: number, labels: Object } | null}
 */
export async function detectBoatKeypoints(source) {
  const session = _session || await ensureSession();
  const ort = _ort || await ensureOrt();

  const preprocessed = preprocessFrame(source, MODEL_INPUT_SIZE);
  let inputTensor;
  try {
    inputTensor = new ort.Tensor('float32', preprocessed.tensor, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  } catch (err) {
    preprocessed.release?.();
    throw err;
  }

  const feeds = {};
  const inputName = session.inputNames[0];
  feeds[inputName] = inputTensor;

  let results;
  try {
    results = await session.run(feeds);
  } finally {
    preprocessed.release?.();
  }
  const outputName = session.outputNames[0];
  const output = results[outputName];

  // Output shape: [1, 32, 8400]
  const data = output.data;
  const numDetections = output.dims[2];

  const detection = parseYoloPoseOutput(data, numDetections, preprocessed);
  if (!detection) return null;

  const labels = bindKeypointsToLabels(detection.keypoints);
  return { keypoints: detection.keypoints, confidence: detection.confidence, labels };
}

/**
 * Run full auto-PnP: detect keypoints → undistort → DLT+RANSAC+LM → correct → validate.
 * Faithfully matches Python _attempt_auto_camera_pnp + _solve_pnp_pose.
 *
 * @param {HTMLCanvasElement|HTMLVideoElement} source
 * @param {Object} [initialPose] (unused — DLT provides algebraic init)
 * @returns {{ camPos, R_wc (nested 3×3), angles, meanErrorPx, … } | null}
 */
export async function estimateCameraPose(source, initialPose = null) {
  const det = await detectBoatKeypoints(source);
  if (!det || !det.labels) return null;

  const calib = await getCalibration();
  const scaledIntrinsics = _intrinsicsForSource(calib, source);
  if (!scaledIntrinsics || !calib?.D) {
    console.error('[AutoPnP] calibration.json missing K/K_undist/D');
    return null;
  }
  const { K: K_dist, K_undist } = scaledIntrinsics;
  const K = [
    K_undist[0][0], K_undist[0][1], K_undist[0][2],
    K_undist[1][0], K_undist[1][1], K_undist[1][2],
    K_undist[2][0], K_undist[2][1], K_undist[2][2],
  ];

  const primaryLabels = det.labels;
  const swappedLabels = _swapPortStarboard(primaryLabels);
  const mappings = [
    { name: 'geometry', labels: primaryLabels },
    { name: 'geometry_swapped', labels: swappedLabels },
  ];

  let bestCandidate = null;
  let firstRejectedDebug = null;

  for (const { name: mapName, labels } of mappings) {
    // Build 3D-2D correspondences from distorted YOLO keypoints
    const objPts = [];
    const imgPtsDist = [];
    for (const label of KEYPOINT_LABELS) {
      const kptIdx = labels[label];
      if (kptIdx == null) continue;
      const kpt = det.keypoints[kptIdx];
      if (!kpt || kpt.conf < MIN_KPT_CONF) continue;
      if (!isFinite(kpt.x) || !isFinite(kpt.y)) continue;
      objPts.push(OBJECT_POINTS_3D[label]);
      imgPtsDist.push([kpt.x, kpt.y]);
    }
    if (objPts.length < MIN_PAIRS) continue;

    // Undistort YOLO keypoint pixel coords (fisheye → rectilinear)
    const imgPts = _fisheyeUndistortPoints(imgPtsDist, K_dist, calib.D, K_undist);

    // Multi-candidate solve (DLT + RANSAC + LM + pose-prior, matching Python _solve_pnp_pose)
    const solved = _solvePnPPose(objPts, imgPts, K, initialPose);
    if (!solved) continue;

    // Apply correction (roll+90°, y/z swap) and validate against bounds
    const corrected = _correctAndValidate(solved.rvec, solved.tvec, solved.medianErrorPx);

    const candidate = {
      ...corrected,
      meanErrorPx: solved.medianErrorPx,
      numPairs: solved.numPairs,
      numInliers: solved.numInliers,
      solveMethod: solved.method,
      mapping: mapName,
      keypoints: det.keypoints,
      labels,
      boatConfidence: det.confidence,
    };

    if (corrected.issues.length === 0) {
      if (!bestCandidate || solved.score < bestCandidate._score) {
        bestCandidate = candidate;
        bestCandidate._score = solved.score;
      }
    } else if (!firstRejectedDebug) {
      firstRejectedDebug = { solved, corrected };
    }
  }

  if (!bestCandidate) {
    if (firstRejectedDebug) {
      const { solved, corrected: c } = firstRejectedDebug;
      console.log(`[AutoPnP] REJECTED (${solved.method}): ${c.issues.join(', ')} | corrected pos=[${c.camPos.map(v=>v.toFixed(3))}] angles=p${c.angles.pitch_deg.toFixed(1)} y${c.angles.yaw_deg.toFixed(1)} r${c.angles.roll_deg.toFixed(1)} | raw pos=[${c.rawCamPos.map(v=>v.toFixed(3))}] rawAngles=p${c.rawAngles.pitch_deg.toFixed(1)} y${c.rawAngles.yaw_deg.toFixed(1)} r${c.rawAngles.roll_deg.toFixed(1)} err=${solved.medianErrorPx.toFixed(1)}px`);
    }
    return null;
  }

  console.log(`[AutoPnP] ACCEPTED (${bestCandidate.mapping}/${bestCandidate.solveMethod}): pos=[${bestCandidate.camPos.map(v=>v.toFixed(3))}], angles=p${bestCandidate.angles.pitch_deg.toFixed(1)} y${bestCandidate.angles.yaw_deg.toFixed(1)} r${bestCandidate.angles.roll_deg.toFixed(1)}, err=${bestCandidate.meanErrorPx.toFixed(1)}px (${bestCandidate.numPairs} pts, ${bestCandidate.numInliers} inliers)`);

  return {
    camPos: bestCandidate.camPos,
    R_wc: bestCandidate.R_wc,
    angles: bestCandidate.angles,
    meanErrorPx: bestCandidate.meanErrorPx,
    numPairs: bestCandidate.numPairs,
    keypoints: bestCandidate.keypoints,
    labels: bestCandidate.labels,
    boatConfidence: bestCandidate.boatConfidence,
  };
}

/** Swap port/starboard labels in a mapping (matching Python). */
function _swapPortStarboard(labels) {
  const s = { ...labels };
  const pairs = [['porttop','starboardtop'],['portmid','starboardmid'],
                 ['portlow','starboardlow'],['portback','starboardback']];
  for (const [a, b] of pairs) { const tmp = s[a]; s[a] = s[b]; s[b] = tmp; }
  return s;
}

/**
 * Check if the ONNX model is loaded and ready.
 */
export function isReady() {
  return !!_session;
}

/**
 * Pre-load the ONNX model (call during app init for faster first use).
 */
export async function preload() {
  try {
    await ensureSession();
    return true;
  } catch (e) {
    console.warn('AutoPnP model preload failed:', e);
    return false;
  }
}
