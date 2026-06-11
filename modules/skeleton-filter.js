/**
 * TrollFish — skeleton-filter.js
 * Kalman smoothing for 33-joint skeleton — port from Python skeleton_filter.py
 *
 * Each landmark gets a 6D constant-velocity Kalman filter (x,y,z,vx,vy,vz).
 * Confidence-adaptive measurement noise, Mahalanobis gating, bone-length
 * plausibility checks.
 */

// ── Defaults ──────────────────────────────────────────────────────────

export const KALMAN_DEFAULTS = Object.freeze({
  enabled: true,
  process_noise_acc: 1.5,
  measurement_noise: 0.03,
  use_landmark_confidence: true,
  min_landmark_confidence: 0.05,
  confidence_floor: 0.10,
  confidence_power: 1.0,
  max_confidence_noise_scale: 12.0,
  gate_sigma: 6.0,
  max_consecutive_misses: 20,
  initial_velocity_std: 1.0,
  velocity_decay: 0.97,
  max_speed: 6.0,
  max_measurement_jump: 0.75,
  reacquire_frames: 2,
  reacquire_max_jump: 0.35,
});

// ── Bone length plausibility ──────────────────────────────────────────

const BONE_LENGTH_LIMITS = [
  [[11,12], 0.10, 0.85],
  [[23,24], 0.10, 0.80],
  [[11,13], 0.10, 0.65],
  [[13,15], 0.10, 0.65],
  [[12,14], 0.10, 0.65],
  [[14,16], 0.10, 0.65],
  [[15,17], 0.04, 0.45],
  [[16,18], 0.04, 0.45],
  [[23,25], 0.15, 0.95],
  [[25,27], 0.15, 0.95],
  [[24,26], 0.15, 0.95],
  [[26,28], 0.15, 0.95],
  [[27,31], 0.06, 0.55],
  [[28,32], 0.06, 0.55],
  [[11,23], 0.12, 1.10],
  [[12,24], 0.12, 1.10],
];

function dist3(a, b) {
  const dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

// ── KalmanPoint3D ─────────────────────────────────────────────────────

class KalmanPoint3D {
  constructor(params = KALMAN_DEFAULTS) {
    this.p = params;
    this.state = null;   // Float64Array(6): [x,y,z,vx,vy,vz]
    this.P = null;       // Float64Array(36): 6×6 covariance (row-major)
    this.misses = 0;
    this.hits = 0;
    this._reacq_buf = [];
  }

  /** Predict + update step. Returns smoothed [x,y,z] or null. */
  step(measurement, dt, confidence = 1.0) {
    if (!this.state) {
      if (!measurement) return null;
      this._initState(measurement);
      return [this.state[0], this.state[1], this.state[2]];
    }

    // Predict
    this._predict(dt);

    if (!measurement) {
      this.misses++;
      if (this.misses > this.p.max_consecutive_misses) {
        this.state = null;
        this.P = null;
      }
      return this.state ? [this.state[0], this.state[1], this.state[2]] : null;
    }

    // Innovation gate: reject huge jumps before update
    const innov = [
      measurement[0]-this.state[0],
      measurement[1]-this.state[1],
      measurement[2]-this.state[2],
    ];
    const innovDist = Math.sqrt(innov[0]*innov[0]+innov[1]*innov[1]+innov[2]*innov[2]);

    // Mahalanobis gating (simplified — use diagonal of P for speed)
    const pDiag = [this.P[0], this.P[7], this.P[14]]; // variances of x,y,z
    const rScale = this._noiseScale(confidence);
    const r = (this.p.measurement_noise * rScale) ** 2;
    let mahala = 0;
    for (let i = 0; i < 3; i++) {
      const s = pDiag[i] + r;
      if (s > 1e-12) mahala += (innov[i] * innov[i]) / s;
    }
    mahala = Math.sqrt(mahala);

    if (mahala > this.p.gate_sigma || innovDist > this.p.max_measurement_jump) {
      // Gated out — treat as miss
      this.misses++;
      if (this.misses > this.p.max_consecutive_misses) {
        this.state = null; this.P = null;
        return null;
      }
      // Try reacquire
      this._reacq_buf.push(measurement);
      if (this._reacq_buf.length >= this.p.reacquire_frames) {
        if (this._checkReacquire()) {
          this._initState(measurement);
          this._reacq_buf = [];
          return [this.state[0], this.state[1], this.state[2]];
        }
        this._reacq_buf.shift();
      }
      return [this.state[0], this.state[1], this.state[2]];
    }

    // Standard Kalman update
    this._update(measurement, r);
    this.misses = 0;
    this.hits++;
    this._reacq_buf = [];
    return [this.state[0], this.state[1], this.state[2]];
  }

  _initState(m) {
    this.state = new Float64Array([m[0], m[1], m[2], 0, 0, 0]);
    const vStd = this.p.initial_velocity_std;
    const mNoise = this.p.measurement_noise;
    // Diagonal P
    this.P = new Float64Array(36);
    this.P[0] = mNoise * mNoise;
    this.P[7] = mNoise * mNoise;
    this.P[14] = mNoise * mNoise;
    this.P[21] = vStd * vStd;
    this.P[28] = vStd * vStd;
    this.P[35] = vStd * vStd;
    this.misses = 0;
    this.hits = 1;
  }

  _predict(dt) {
    const s = this.state;
    const decay = this.p.velocity_decay;
    // State prediction: x += v*dt, v *= decay
    s[0] += s[3] * dt;
    s[1] += s[4] * dt;
    s[2] += s[5] * dt;
    s[3] *= decay;
    s[4] *= decay;
    s[5] *= decay;
    // Speed clamp
    const speed = Math.sqrt(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]);
    if (speed > this.p.max_speed) {
      const f = this.p.max_speed / speed;
      s[3]*=f; s[4]*=f; s[5]*=f;
    }
    // P = F*P*F' + Q  (simplified: only update relevant blocks)
    const P = this.P;
    const q = this.p.process_noise_acc;
    const dt2 = dt*dt, dt3 = dt2*dt, dt4 = dt3*dt;
    // Process noise for constant-acceleration model
    const q_pp = q*q * dt4/4;  // position variance
    const q_pv = q*q * dt3/2;  // position-velocity covariance
    const q_vv = q*q * dt2;    // velocity variance
    // For each axis (0,1,2):
    for (let i = 0; i < 3; i++) {
      const pi = i, vi = i+3;
      const pp = pi*6+pi, pv = pi*6+vi, vp = vi*6+pi, vv = vi*6+vi;
      // P' = F*P*F' + Q where F = [[1,dt],[0,1]] per axis
      const oldPP = P[pp], oldPV = P[pv], oldVP = P[vp], oldVV = P[vv];
      P[pp] = oldPP + dt*(oldPV + oldVP) + dt2*oldVV + q_pp;
      P[pv] = oldPV + dt*oldVV + q_pv;
      P[vp] = oldVP + dt*oldVV + q_pv;
      P[vv] = oldVV + q_vv;
    }
  }

  _update(z, r) {
    const s = this.state;
    const P = this.P;
    // H = [I3 | 0_3x3], so innovation = z - s[0:3]
    // S = P[0:3,0:3] + R*I3
    // K = P[:,0:3] * S^-1
    // For efficiency, we do axis-independent scalar updates (diagonal R)
    for (let i = 0; i < 3; i++) {
      const pi = i;
      const innov = z[i] - s[i];
      const sii = P[pi*6+pi] + r;
      if (sii < 1e-15) continue;
      const kInv = 1.0 / sii;
      // Kalman gain column for measurement i
      const K = new Float64Array(6);
      for (let j = 0; j < 6; j++) K[j] = P[j*6+pi] * kInv;
      // State update
      for (let j = 0; j < 6; j++) s[j] += K[j] * innov;
      // P update: P = (I - K*H_i) * P
      for (let j = 0; j < 6; j++) {
        for (let k = 0; k < 6; k++) {
          P[j*6+k] -= K[j] * P[pi*6+k];
        }
      }
    }
  }

  _noiseScale(confidence) {
    if (!this.p.use_landmark_confidence) return 1.0;
    const c = Math.max(confidence, this.p.confidence_floor);
    const scale = Math.pow(1.0 / c, this.p.confidence_power);
    return Math.min(scale, this.p.max_confidence_noise_scale);
  }

  _checkReacquire() {
    const buf = this._reacq_buf;
    if (buf.length < 2) return true;
    for (let i = 1; i < buf.length; i++) {
      const d = dist3(buf[i], buf[i-1]);
      if (d > this.p.reacquire_max_jump) return false;
    }
    return true;
  }
}


// ── SkeletonPlacementKalman ───────────────────────────────────────────

export class SkeletonPlacementKalman {
  constructor(params = KALMAN_DEFAULTS) {
    this.params = { ...KALMAN_DEFAULTS, ...params };
    this.trackers = [];
    for (let i = 0; i < 33; i++) {
      this.trackers.push(new KalmanPoint3D(this.params));
    }
  }

  /**
   * Smooth a single frame.
   * @param {Object} skeleton — {0..32: [x,y,z]}
   * @param {Object} confidence — {0..32: float} visibility values
   * @param {number} dt — time delta in seconds (1/fps)
   * @returns {Object|null} smoothed skeleton or null if implausible
   */
  smooth(skeleton, confidence, dt) {
    if (!skeleton) {
      // Predict-only for all trackers
      const result = {};
      let anyValid = false;
      for (let i = 0; i < 33; i++) {
        const pos = this.trackers[i].step(null, dt, 0);
        if (pos) { result[i] = pos; anyValid = true; }
      }
      return anyValid ? result : null;
    }

    const result = {};
    for (let i = 0; i < 33; i++) {
      const m = skeleton[i] || null;
      const c = confidence[i] ?? 0.5;
      const pos = this.trackers[i].step(m, dt, c);
      if (pos) result[i] = pos;
    }

    // Plausibility check
    if (this._isPoseImplausible(result)) {
      this.reset();
      return null;
    }

    return result;
  }

  reset() {
    for (const t of this.trackers) {
      t.state = null;
      t.P = null;
      t.misses = 0;
      t.hits = 0;
      t._reacq_buf = [];
    }
  }

  _isPoseImplausible(skeleton) {
    // Max axis span check
    let count = 0;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < 33; i++) {
      const p = skeleton[i];
      if (!p) continue;
      count++;
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
      if (p[2] < minZ) minZ = p[2];
      if (p[2] > maxZ) maxZ = p[2];
    }
    if (count < 5) return false;
    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const spanZ = maxZ - minZ;
    if (spanX > 4.5 || spanY > 4.5 || spanZ > 4.5) return true;

    // Pelvis radius check
    const hip23 = skeleton[23], hip24 = skeleton[24];
    if (hip23 && hip24) {
      const pelvisR = dist3(hip23, [0,0,0]);
      if (pelvisR > 2.4) return true;
    }

    // Bone length violations
    let violations = 0, checked = 0;
    for (const [[a, b], minL, maxL] of BONE_LENGTH_LIMITS) {
      const pa = skeleton[a], pb = skeleton[b];
      if (!pa || !pb) continue;
      checked++;
      const d = dist3(pa, pb);
      if (d < minL || d > maxL) violations++;
    }
    if (checked >= 4 && violations >= 3 && violations / checked > 0.45) return true;

    return false;
  }
}
