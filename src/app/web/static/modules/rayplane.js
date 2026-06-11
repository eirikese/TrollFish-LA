/**
 * TrollFish — rayplane.js
 * Ray-plane intersection & skeleton placement — port from Python rayplane.py
 *
 * Coordinate system:
 *   World/Boat frame: X = fore-aft (negative = forward/bow),
 *                     Y = lateral (port/starboard), Z = vertical (up)
 *   Camera base orientation: cam Z → world X, cam X → world −Y, cam Y → world −Z
 */

// ── Matrix / Vector helpers (inline, no dependency on gl-matrix) ──────

/** 3×3 matrix multiply: C = A * B (row-major arrays of length 9) */
function mat3mul(A, B) {
  return [
    A[0]*B[0]+A[1]*B[3]+A[2]*B[6], A[0]*B[1]+A[1]*B[4]+A[2]*B[7], A[0]*B[2]+A[1]*B[5]+A[2]*B[8],
    A[3]*B[0]+A[4]*B[3]+A[5]*B[6], A[3]*B[1]+A[4]*B[4]+A[5]*B[7], A[3]*B[2]+A[4]*B[5]+A[5]*B[8],
    A[6]*B[0]+A[7]*B[3]+A[8]*B[6], A[6]*B[1]+A[7]*B[4]+A[8]*B[7], A[6]*B[2]+A[7]*B[5]+A[8]*B[8],
  ];
}

/** 3×3 transpose */
function mat3T(M) {
  return [M[0],M[3],M[6], M[1],M[4],M[7], M[2],M[5],M[8]];
}

/** 3×3 * vec3 */
function mat3vec(M, v) {
  return [
    M[0]*v[0]+M[1]*v[1]+M[2]*v[2],
    M[3]*v[0]+M[4]*v[1]+M[5]*v[2],
    M[6]*v[0]+M[7]*v[1]+M[8]*v[2],
  ];
}

function vec3add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function vec3sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function vec3scale(v, s) { return [v[0]*s, v[1]*s, v[2]*s]; }
function vec3dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function vec3cross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function vec3len(v) { return Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
function vec3norm(v) { const l=vec3len(v); return l<1e-12?[0,0,0]:[v[0]/l,v[1]/l,v[2]/l]; }

// ── Rotation matrices ─────────────────────────────────────────────────

function rotX(deg) {
  const r=deg*Math.PI/180, c=Math.cos(r), s=Math.sin(r);
  return [1,0,0, 0,c,-s, 0,s,c];
}
function rotY(deg) {
  const r=deg*Math.PI/180, c=Math.cos(r), s=Math.sin(r);
  return [c,0,s, 0,1,0, -s,0,c];
}
function rotZ(deg) {
  const r=deg*Math.PI/180, c=Math.cos(r), s=Math.sin(r);
  return [c,-s,0, s,c,0, 0,0,1];
}

// ── Camera pose ───────────────────────────────────────────────────────

// Base rotation: camera Z→world X, camera X→world −Y, camera Y→world −Z
const BASE_ROT = [0,0,1, -1,0,0, 0,-1,0];

/**
 * Default camera pose and rotation matrix (camera-to-world).
 * @returns {{pos: number[3], R_wc: number[9]}}
 */
export function defaultCameraPoseAndRotation(pitch = 14.7, yaw = 0.0, roll = 0.0) {
  const pos = [-3.194, 0.02, 0.585];
  // Relative rotation: Rz(roll) @ Ry(yaw) @ Rx(-pitch)
  const R_rel = mat3mul(rotZ(roll), mat3mul(rotY(yaw), rotX(-pitch)));
  const R_wc = mat3mul(BASE_ROT, R_rel);
  return { pos, R_wc };
}

// ── Ray construction ──────────────────────────────────────────────────

/**
 * Convert MediaPipe normalized [0,1] coords to a unit ray in camera frame.
 * Uses the undistorted intrinsic matrix K.
 * @param {number} xNorm — normalized x [0,1]
 * @param {number} yNorm — normalized y [0,1]
 * @param {number} W — image width
 * @param {number} H — image height
 * @param {number[]} K — 3×3 intrinsic matrix (row-major, 9 elements)
 * @returns {number[3]} unit ray in camera frame
 */
export function rayFromNormLandmark(xNorm, yNorm, W, H, K) {
  const u = xNorm * W;
  const v = yNorm * H;
  const fx = K[0], fy = K[4], cx = K[2], cy = K[5];
  const xn = (u - cx) / fx;
  const yn = (v - cy) / fy;
  return vec3norm([xn, yn, 1.0]);
}

/**
 * Intersect a camera-frame ray with a world-space z = Z0 plane.
 * @param {number[3]} d_c — ray direction in camera frame
 * @param {number[9]} R_wc — camera-to-world rotation (row-major 3×3)
 * @param {number[3]} t_wc — camera position in world
 * @param {number} Z0 — plane height
 * @returns {number[3]|null} world intersection point
 */
export function intersectWorldZPlane(d_c, R_wc, t_wc, Z0, eps = 1e-9) {
  const d_w = mat3vec(R_wc, d_c);
  const dz = d_w[2];
  if (Math.abs(dz) < eps) return null; // parallel
  const t = (Z0 - t_wc[2]) / dz;
  if (t < 0) return null; // behind camera
  return vec3add(t_wc, vec3scale(d_w, t));
}

// ── Rotation alignment ────────────────────────────────────────────────

/**
 * Rodrigues rotation matrix mapping unit vector a → b.
 */
function rotationFromAToB(a, b) {
  const d = vec3dot(a, b);
  if (d > 0.9999) return [1,0,0,0,1,0,0,0,1]; // identity
  if (d < -0.9999) {
    // anti-parallel: rotate 180° about any perpendicular axis
    let perp = Math.abs(a[0]) < 0.9 ? vec3cross(a, [1,0,0]) : vec3cross(a, [0,1,0]);
    perp = vec3norm(perp);
    // 180° rotation about perp: R = 2*p*pT - I
    return [
      2*perp[0]*perp[0]-1, 2*perp[0]*perp[1],   2*perp[0]*perp[2],
      2*perp[1]*perp[0],   2*perp[1]*perp[1]-1,  2*perp[1]*perp[2],
      2*perp[2]*perp[0],   2*perp[2]*perp[1],    2*perp[2]*perp[2]-1,
    ];
  }
  const v = vec3cross(a, b);
  const s = vec3len(v);
  const c = d;
  // skew-symmetric matrix of v
  const vx = [0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0];
  // vx² = v⊗v - s²I → (1-c)/s² factor
  const f = (1 - c) / (s * s);
  const vx2 = mat3mul(vx, vx);
  return [
    1 + vx[0] + vx2[0]*f, vx[1] + vx2[1]*f,     vx[2] + vx2[2]*f,
    vx[3] + vx2[3]*f,     1 + vx[4] + vx2[4]*f,  vx[5] + vx2[5]*f,
    vx[6] + vx2[6]*f,     vx[7] + vx2[7]*f,      1 + vx[8] + vx2[8]*f,
  ];
}

/**
 * Two-vector rotation: align primary a1→b1 and secondary a2→b2.
 */
function rotationFromTwoVectors(a1, a2, b1, b2) {
  // Gram-Schmidt: orthonormalize secondary w.r.t. primary
  const a1n = vec3norm(a1);
  const b1n = vec3norm(b1);
  let a2orth = vec3sub(a2, vec3scale(a1n, vec3dot(a2, a1n)));
  let b2orth = vec3sub(b2, vec3scale(b1n, vec3dot(b2, b1n)));
  const la2 = vec3len(a2orth), lb2 = vec3len(b2orth);
  if (la2 < 1e-9 || lb2 < 1e-9) return rotationFromAToB(a1n, b1n);
  a2orth = vec3scale(a2orth, 1/la2);
  b2orth = vec3scale(b2orth, 1/lb2);
  const a3 = vec3cross(a1n, a2orth);
  const b3 = vec3cross(b1n, b2orth);
  // Source basis S = [a1n|a2orth|a3] cols, Target B = [b1n|b2orth|b3] cols
  // R = B * S^T (row-major)
  const S = [a1n[0],a2orth[0],a3[0], a1n[1],a2orth[1],a3[1], a1n[2],a2orth[2],a3[2]];
  const B = [b1n[0],b2orth[0],b3[0], b1n[1],b2orth[1],b3[1], b1n[2],b2orth[2],b3[2]];
  return mat3mul(B, mat3T(S));
}

// ── Skeleton placement ────────────────────────────────────────────────

/**
 * Primary placement path: symmetric raycast.
 * Preserves MediaPipe body orientation, solves global translation from
 * ray-plane constraints on hips and lower limbs.
 *
 * @param {Object} worldLandmarks — {0..32: [x,y,z]} in camera/MP world frame
 * @param {Object} normLandmarks  — {0..32: [x,y,z,vis]} normalized image coords
 * @param {number[9]} K — intrinsic matrix (row-major)
 * @param {number} W — image width
 * @param {number} H — image height
 * @param {number[3]} camPos — camera position in world
 * @param {number[9]} R_wc — camera-to-world rotation
 * @param {number} zHip — hip plane height
 * @param {number} zAnkle — ankle/foot plane height
 * @returns {Object|null} {0..32: [x,y,z]} in world frame
 */
export function computePlacedSkeletonSymmetricRaycast(
  worldLandmarks, normLandmarks, K, W, H, camPos, R_wc, zHip, zAnkle
) {
  // We need at least hips + one lower limb landmark
  const lm23w = worldLandmarks[23];
  const lm24w = worldLandmarks[24];
  if (!lm23w || !lm24w) return null;

  // Get MP body orientation (world coords are in camera frame)
  // We only need to solve the global translation; rotation comes from MP

  // Compute hip midpoint in MP world coords
  const hipMidMp = vec3scale(vec3add(lm23w, lm24w), 0.5);

  // Try to get raycast anchor points for hip plane
  const hipIntersections = [];
  for (const idx of [23, 24]) {
    const nrm = normLandmarks[idx];
    if (!nrm) continue;
    const ray = rayFromNormLandmark(nrm[0], nrm[1], W, H, K);
    const pt = intersectWorldZPlane(ray, R_wc, camPos, zHip);
    if (pt) hipIntersections.push({ idx, pt });
  }

  const ankleIntersections = [];
  for (const idx of [27, 28, 31, 32]) {
    const nrm = normLandmarks[idx];
    if (!nrm) continue;
    const ray = rayFromNormLandmark(nrm[0], nrm[1], W, H, K);
    const pt = intersectWorldZPlane(ray, R_wc, camPos, zAnkle);
    if (pt) ankleIntersections.push({ idx, pt });
  }

  if (hipIntersections.length === 0 && ankleIntersections.length === 0) return null;

  // Compute weighted translation offset
  // Hip anchor: average of hip intersections → should map to hip midpoint
  let translationSum = [0, 0, 0];
  let totalWeight = 0;

  if (hipIntersections.length >= 1) {
    const hipWorldAvg = vec3scale(
      hipIntersections.reduce((s, h) => vec3add(s, h.pt), [0,0,0]),
      1 / hipIntersections.length
    );
    // The MP world coords are in camera frame. Transform to world:
    const hipMpWorld = mat3vec(R_wc, hipMidMp);
    const offset = vec3sub(hipWorldAvg, hipMpWorld);
    // Set Z to zHip explicitly
    offset[2] = zHip - hipMpWorld[2];
    translationSum = vec3add(translationSum, vec3scale(offset, 2.0)); // weight=2 for hips
    totalWeight += 2.0;
  }

  if (ankleIntersections.length >= 1) {
    // Average ankle intersection
    const ankleWorldAvg = vec3scale(
      ankleIntersections.reduce((s, a) => vec3add(s, a.pt), [0,0,0]),
      1 / ankleIntersections.length
    );
    // Ankle midpoint in MP world
    const anklePts = [27, 28].filter(i => worldLandmarks[i]).map(i => worldLandmarks[i]);
    if (anklePts.length > 0) {
      const ankleMidMp = vec3scale(anklePts.reduce((s, p) => vec3add(s, p), [0,0,0]), 1/anklePts.length);
      const ankleMpWorld = mat3vec(R_wc, ankleMidMp);
      const offset = vec3sub(ankleWorldAvg, ankleMpWorld);
      offset[2] = zAnkle - ankleMpWorld[2];
      translationSum = vec3add(translationSum, vec3scale(offset, 1.0)); // weight=1 for ankles
      totalWeight += 1.0;
    }
  }

  if (totalWeight < 0.01) return null;
  const translation = vec3scale(translationSum, 1 / totalWeight);

  // Apply: transform all MP landmarks to world and add translation
  const placed = {};
  for (let i = 0; i < 33; i++) {
    const lm = worldLandmarks[i];
    if (!lm) continue;
    const worldPt = vec3add(mat3vec(R_wc, lm), translation);
    placed[i] = worldPt;
  }

  // Flip disambiguation: check upright (shoulders above hips, head above hips)
  const shouldersZ = _avgZ(placed, [11, 12]);
  const hipsZ = _avgZ(placed, [23, 24]);
  const headZ = _avgZ(placed, [0, 7, 8]);
  if (shouldersZ !== null && hipsZ !== null && shouldersZ < hipsZ) {
    // Upside down — try flipping 180° around world X
    // This is a rare edge case; skip for now
  }

  return placed;
}

function _avgZ(skeleton, indices) {
  let sum = 0, cnt = 0;
  for (const i of indices) {
    if (skeleton[i]) { sum += skeleton[i][2]; cnt++; }
  }
  return cnt > 0 ? sum / cnt : null;
}

/**
 * Fallback placement: rigid body transform using anchor + direction vectors.
 */
export function placeSkeletonOnBoat(
  worldLandmarks,
  anchorIdx = 24, dirIdx = 28,
  anchorBoat = [0, 0, 0.06], dirBoat = [0, 0, 0],
  rollIdx = null, rollBoat = null
) {
  const srcAnchor = worldLandmarks[anchorIdx];
  const srcDir = worldLandmarks[dirIdx];
  if (!srcAnchor || !srcDir) return null;

  const srcVec = vec3norm(vec3sub(srcDir, srcAnchor));
  const tgtVec = vec3norm(vec3sub(dirBoat, anchorBoat));
  if (vec3len(srcVec) < 1e-9 || vec3len(tgtVec) < 1e-9) return null;

  let R;
  if (rollIdx != null && rollBoat && worldLandmarks[rollIdx]) {
    const srcRoll = worldLandmarks[rollIdx];
    R = rotationFromTwoVectors(
      vec3sub(srcDir, srcAnchor), vec3sub(srcRoll, srcAnchor),
      vec3sub(dirBoat, anchorBoat), vec3sub(rollBoat, anchorBoat)
    );
  } else {
    R = rotationFromAToB(srcVec, tgtVec);
  }

  // Apply: R * (pt - srcAnchor) + anchorBoat
  const placed = {};
  for (let i = 0; i < 33; i++) {
    const lm = worldLandmarks[i];
    if (!lm) continue;
    placed[i] = vec3add(mat3vec(R, vec3sub(lm, srcAnchor)), anchorBoat);
  }

  // Flip check: shoulders should be above hips
  const sZ = _avgZ(placed, [11, 12]);
  const hZ = _avgZ(placed, [23, 24]);
  if (sZ !== null && hZ !== null && sZ < hZ - 0.05) {
    // Flip 180° around the primary axis direction
    const flipR = rotationFromAToB([0, 0, 1], [0, 0, -1]);
    const midHip = placed[23] && placed[24]
      ? vec3scale(vec3add(placed[23], placed[24]), 0.5) : anchorBoat;
    for (let i = 0; i < 33; i++) {
      if (!placed[i]) continue;
      placed[i] = vec3add(mat3vec(flipR, vec3sub(placed[i], midHip)), midHip);
    }
  }

  return placed;
}

// ── Exports for testing / direct use ──────────────────────────────────

export {
  mat3mul, mat3T, mat3vec,
  vec3add, vec3sub, vec3scale, vec3dot, vec3cross, vec3len, vec3norm,
  rotX, rotY, rotZ,
  rotationFromAToB, rotationFromTwoVectors,
  rayFromNormLandmark as rayFromNormLandmarkUndistorted,
};
