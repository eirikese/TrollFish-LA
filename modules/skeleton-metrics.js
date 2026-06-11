/**
 * TrollFish — skeleton-metrics.js
 * Biomechanical metrics from placed skeleton — port from Python skeleton_metrics.py
 */

// ── Helper ────────────────────────────────────────────────────────────

function _point3(v) {
  if (!v || !Array.isArray(v) || v.length < 3) return null;
  if (!isFinite(v[0]) || !isFinite(v[1]) || !isFinite(v[2])) return null;
  return v;
}

function _mid(a, b) {
  if (!a || !b) return null;
  return [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2];
}

function _sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function _len(v) { return Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
function _dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function _lerp(a, b, t) { return [a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*(b[2]-a[2])]; }
function _dist(a, b) { return (!a || !b) ? null : _len(_sub(a, b)); }

function _avgPoints(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  return valid.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0])
    .map(v => v / valid.length);
}

// ── Trunk vector ──────────────────────────────────────────────────────

/**
 * Trunk vector from hip → shoulder (single-side, lm23→lm11 by default).
 * Returns unit vector or null.
 */
export function computeTrunkVector(skeleton, upperIdx = 11, lowerIdx = 23) {
  const upper = _point3(skeleton[upperIdx]);
  const lower = _point3(skeleton[lowerIdx]);
  if (!upper || !lower) return null;
  const v = _sub(upper, lower);
  const l = _len(v);
  if (l < 1e-6) return null;
  return [v[0]/l, v[1]/l, v[2]/l];
}

/**
 * Angle of trunk from vertical (degrees). 0° = upright, 90° = horizontal.
 */
export function trunkAngleToVertical(skeleton, verticalAxis = [0, 0, 1]) {
  const tv = computeTrunkVector(skeleton);
  if (!tv) return null;
  const d = _dot(tv, verticalAxis);
  const clamped = Math.max(-1, Math.min(1, d));
  return Math.acos(clamped) * (180 / Math.PI);
}

// ── Center of mass ────────────────────────────────────────────────────

/**
 * Biomechanical segment table for COM estimation.
 * Each entry: [name, fromIdx|[idx,idx], toIdx|[idx,idx], massFraction, comFraction]
 * If fromIdx/toIdx is an array, it's the midpoint of those two landmarks.
 */
const COM_SEGMENTS = [
  ['head_neck',    [11,12],  [7,8],    0.081, 1.000],
  ['trunk',        [11,12],  [23,24],  0.497, 0.500],
  ['upper_arm_L',  11,        13,       0.028, 0.436],
  ['upper_arm_R',  12,        14,       0.028, 0.436],
  ['forearm_L',    13,        15,       0.016, 0.430],
  ['forearm_R',    14,        16,       0.016, 0.430],
  ['hand_L',       15,        17,       0.006, 0.506],
  ['hand_R',       16,        18,       0.006, 0.506],
  ['thigh_L',      23,        25,       0.100, 0.433],
  ['thigh_R',      24,        26,       0.100, 0.433],
  ['shank_L',      25,        27,       0.0465, 0.433],
  ['shank_R',      26,        28,       0.0465, 0.433],
  ['foot_L',       27,        31,       0.0145, 0.500],
  ['foot_R',       28,        32,       0.0145, 0.500],
];

function _getSegPt(skeleton, spec) {
  if (Array.isArray(spec)) {
    return _mid(_point3(skeleton[spec[0]]), _point3(skeleton[spec[1]]));
  }
  return _point3(skeleton[spec]);
}

/**
 * Compute center of mass using segmental analysis.
 * Tolerant of missing segments — renormalizes mass fractions.
 * Returns [x,y,z] or null.
 */
export function computeCenterOfMass(skeleton) {
  let totalMass = 0;
  let com = [0, 0, 0];

  for (const [, fromSpec, toSpec, massFrac, comFrac] of COM_SEGMENTS) {
    const from = _getSegPt(skeleton, fromSpec);
    const to = _getSegPt(skeleton, toSpec);
    if (!from || !to) continue;
    const segCom = _lerp(from, to, comFrac);
    com[0] += segCom[0] * massFrac;
    com[1] += segCom[1] * massFrac;
    com[2] += segCom[2] * massFrac;
    totalMass += massFrac;
  }

  // Guard: need at least 35% represented mass
  if (totalMass < 0.35) return null;

  return [com[0]/totalMass, com[1]/totalMass, com[2]/totalMass];
}

// ── Skeleton height normalization ─────────────────────────────────────

function _normalizeAthleteHeightM(height) {
  const h = Number(height);
  if (!Number.isFinite(h) || h <= 0) return null;
  // Accept either metres (1.82) or centimetres (182).
  const metres = h > 3 ? h / 100 : h;
  return metres >= 1.0 && metres <= 2.4 ? metres : null;
}

/**
 * Estimate standing body height from limb-chain lengths in the placed skeleton.
 * This is posture-tolerant: a crouched/sitting frame still contributes its
 * anatomical segment lengths rather than just its vertical bounding box.
 */
export function estimateSkeletonHeight(skeleton) {
  if (!skeleton) return null;
  const midShoulder = _mid(_point3(skeleton[11]), _point3(skeleton[12]));
  const head = _avgPoints([
    _point3(skeleton[0]), _point3(skeleton[7]), _point3(skeleton[8]),
    _point3(skeleton[9]), _point3(skeleton[10]),
  ]);
  const neckHead = _dist(midShoulder, head);
  const headTopAllowance = neckHead != null ? neckHead * 0.35 : 0.12;

  const chains = [];
  for (const [hipIdx, kneeIdx, ankleIdx] of [[23, 25, 27], [24, 26, 28]]) {
    const ankle = _point3(skeleton[ankleIdx]);
    const knee = _point3(skeleton[kneeIdx]);
    const hip = _point3(skeleton[hipIdx]);
    const legA = _dist(ankle, knee);
    const legB = _dist(knee, hip);
    const trunk = _dist(hip, midShoulder);
    if (legA != null && legB != null && trunk != null && neckHead != null) {
      chains.push(legA + legB + trunk + neckHead + headTopAllowance);
    }
  }

  if (chains.length) return chains.reduce((sum, v) => sum + v, 0) / chains.length;

  const midAnkle = _avgPoints([_point3(skeleton[27]), _point3(skeleton[28])]);
  if (midAnkle && head) return _dist(midAnkle, head) + headTopAllowance;
  return null;
}

/**
 * Uniformly scale a placed skeleton to the athlete's known body height.
 * Returns the original skeleton unchanged when no valid height is supplied.
 */
export function scaleSkeletonToAthleteHeight(skeleton, athleteHeight, opts = {}) {
  const targetM = _normalizeAthleteHeightM(athleteHeight);
  if (!skeleton || targetM == null) return { skeleton, scale: 1, measuredHeight: null, targetHeight: targetM, applied: false };

  const measured = estimateSkeletonHeight(skeleton);
  if (!Number.isFinite(measured) || measured <= 0) {
    return { skeleton, scale: 1, measuredHeight: measured ?? null, targetHeight: targetM, applied: false };
  }

  const rawScale = targetM / measured;
  const minScale = Number.isFinite(Number(opts.minScale)) ? Number(opts.minScale) : 0.65;
  const maxScale = Number.isFinite(Number(opts.maxScale)) ? Number(opts.maxScale) : 1.45;
  const scale = Math.max(minScale, Math.min(maxScale, rawScale));
  if (Math.abs(scale - 1) < 0.015) {
    return { skeleton, scale: 1, measuredHeight: measured, targetHeight: targetM, applied: false };
  }

  const anchor = _avgPoints([
    _point3(skeleton[27]), _point3(skeleton[28]), _point3(skeleton[31]), _point3(skeleton[32]),
  ]) || _mid(_point3(skeleton[23]), _point3(skeleton[24])) || [0, 0, 0];
  const scaled = {};
  for (const [idx, pt] of Object.entries(skeleton)) {
    const p = _point3(pt);
    if (!p) continue;
    scaled[idx] = [
      anchor[0] + (p[0] - anchor[0]) * scale,
      anchor[1] + (p[1] - anchor[1]) * scale,
      anchor[2] + (p[2] - anchor[2]) * scale,
    ];
  }

  return { skeleton: scaled, scale, measuredHeight: measured, targetHeight: targetM, applied: true };
}

/**
 * Check if COM is in a plausible range.
 */
export function isPlausibleCom(com) {
  if (!com) return false;
  // Thresholds from processing_core.py
  if (com[0] < -3 || com[0] > 0) return false;   // com_x
  if (com[1] < -2 || com[1] > 2) return false;    // com_y
  if (com[2] < -0.5 || com[2] > 0.5) return false; // com_z (relative to boat deck)
  return true;
}

// ── Moments ───────────────────────────────────────────────────────────

const GRAVITY = 9.81;

/**
 * Compute pitch and roll moments.
 * @param {Object} skeleton — placed skeleton {0..32: [x,y,z]}
 * @param {number} athleteMass — kg
 * @param {number[3]} boatCom — boat center of mass [x,y,z]
 * @returns {{pitchMoment: number, rollMoment: number, com: number[3]}|null}
 */
export function computeMoments(skeleton, athleteMass = 80, boatCom = [0, 0, 0]) {
  const com = computeCenterOfMass(skeleton);
  if (!com || !isPlausibleCom(com)) return null;

  const pitchMoment = athleteMass * GRAVITY * (com[0] - boatCom[0]);
  const rollMoment = athleteMass * GRAVITY * com[1];

  return { pitchMoment, rollMoment, com };
}

// ── Trunk angle using midpoints (matches processing_core.py) ─────────

/**
 * Trunk angle using bilateral midpoints (more robust than single-side).
 */
export function trunkAngleBilateral(skeleton) {
  const midShoulder = _mid(_point3(skeleton[11]), _point3(skeleton[12]));
  const midHip = _mid(_point3(skeleton[23]), _point3(skeleton[24]));
  if (!midShoulder || !midHip) return null;
  const v = _sub(midShoulder, midHip);
  const l = _len(v);
  if (l < 1e-6) return null;
  const cosA = v[2] / l; // dot with [0,0,1]
  return Math.acos(Math.max(-1, Math.min(1, cosA))) * (180 / Math.PI);
}

// ── Sitting score ─────────────────────────────────────────────────────

/**
 * Estimate sitting score from knee flexion angle.
 * ~170°+ = standing (score 0), ~90° = deeply seated (score 1).
 */
export function estimateSittingScore(skeleton) {
  const angles = [];
  // Left knee: hip23 → knee25 → ankle27
  const lk = _kneeAngle(skeleton, 23, 25, 27);
  if (lk !== null) angles.push(lk);
  // Right knee: hip24 → knee26 → ankle28
  const rk = _kneeAngle(skeleton, 24, 26, 28);
  if (rk !== null) angles.push(rk);
  if (angles.length === 0) return 0;
  const avgAngle = angles.reduce((s, a) => s + a, 0) / angles.length;
  return Math.max(0, Math.min(1, (145 - avgAngle) / 55));
}

function _kneeAngle(skeleton, hipIdx, kneeIdx, ankleIdx) {
  const hip = _point3(skeleton[hipIdx]);
  const knee = _point3(skeleton[kneeIdx]);
  const ankle = _point3(skeleton[ankleIdx]);
  if (!hip || !knee || !ankle) return null;
  const v1 = _sub(hip, knee);
  const v2 = _sub(ankle, knee);
  const l1 = _len(v1), l2 = _len(v2);
  if (l1 < 1e-6 || l2 < 1e-6) return null;
  const cosA = _dot(v1, v2) / (l1 * l2);
  return Math.acos(Math.max(-1, Math.min(1, cosA))) * (180 / Math.PI);
}

// ── Full per-frame metrics ────────────────────────────────────────────

/**
 * Compute all metrics for a single placed skeleton frame.
 * @returns {Object} {trunk_angle, com_x, com_y, com_z, pitch_moment, roll_moment, sitting_score}
 */
export function computeFrameMetrics(skeleton, athleteMass = 80, boatCom = [0, 0, 0]) {
  const trunkAngle = trunkAngleBilateral(skeleton);
  const com = computeCenterOfMass(skeleton);
  const sitting = estimateSittingScore(skeleton);
  const moments = com && isPlausibleCom(com)
    ? {
        pitchMoment: athleteMass * GRAVITY * (com[0] - boatCom[0]),
        rollMoment: athleteMass * GRAVITY * com[1],
      }
    : null;

  return {
    trunk_angle: trunkAngle,
    com_x: com?.[0] ?? null,
    com_y: com?.[1] ?? null,
    com_z: com?.[2] ?? null,
    pitch_moment: moments?.pitchMoment ?? null,
    roll_moment: moments?.rollMoment ?? null,
    sitting_score: sitting,
  };
}
