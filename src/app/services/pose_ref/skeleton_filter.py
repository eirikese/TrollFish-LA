"""
Skeleton placement smoothing filters.

This module intentionally replaces the previous raycast-profile Kalman flow.
It smooths already-placed 3D skeleton landmarks frame-to-frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


SKELETON_KALMAN_DEFAULTS = {
    "enabled": True,
    "process_noise_acc": 1.5,           # m/s^2
    "measurement_noise": 0.03,          # m
    "use_landmark_confidence": True,    # scale measurement noise by lm visibility
    "min_landmark_confidence": 0.05,    # below this confidence: heavy downweighting
    "confidence_floor": 0.10,           # prevents extreme noise inflation near zero
    "confidence_power": 1.0,            # noise scale = (1/conf)^power
    "max_confidence_noise_scale": 12.0, # upper bound on confidence noise multiplier
    "gate_sigma": 6.0,                  # Mahalanobis sigma threshold
    "max_consecutive_misses": 20,       # frames
    "initial_velocity_std": 1.0,        # m/s
    "velocity_decay": 0.97,             # [0,1], 1=no damping
    "max_speed": 6.0,                   # m/s velocity clamp
    "max_measurement_jump": 0.75,       # m per-frame absolute innovation cap
    "reacquire_frames": 2,              # consecutive frames required after track loss
    "reacquire_max_jump": 0.35,         # m consistency gate while reacquiring
}

_BONE_LENGTH_LIMITS = {
    # (landmark_a, landmark_b): (min_len_m, max_len_m)
    (11, 12): (0.10, 0.85),  # shoulder width
    (23, 24): (0.10, 0.80),  # hip width
    (11, 13): (0.10, 0.65),  # upper arm
    (13, 15): (0.10, 0.65),  # forearm
    (12, 14): (0.10, 0.65),
    (14, 16): (0.10, 0.65),
    (15, 17): (0.04, 0.45),  # hand
    (16, 18): (0.04, 0.45),
    (23, 25): (0.15, 0.95),  # thigh
    (25, 27): (0.15, 0.95),  # shank
    (24, 26): (0.15, 0.95),
    (26, 28): (0.15, 0.95),
    (27, 31): (0.06, 0.55),  # foot
    (28, 32): (0.06, 0.55),
    (11, 23): (0.12, 1.10),  # torso side
    (12, 24): (0.12, 1.10),
}

_POSE_MAX_AXIS_SPAN = 4.5
_POSE_MAX_RADIUS_FROM_PELVIS = 2.4
_POSE_MIN_BONES_FOR_CHECK = 4
_POSE_MIN_BONE_VIOLATIONS = 3
_POSE_MAX_BONE_VIOLATION_RATIO = 0.45


def normalize_skeleton_filter_params(params: Optional[dict]) -> dict:
    """Clamp/normalize smoothing parameters."""
    out = dict(SKELETON_KALMAN_DEFAULTS)
    if not isinstance(params, dict):
        return out

    out["enabled"] = bool(params.get("enabled", out["enabled"]))
    try:
        out["process_noise_acc"] = float(
            np.clip(float(params.get("process_noise_acc", out["process_noise_acc"])), 1e-4, 100.0)
        )
    except Exception:
        pass
    try:
        out["measurement_noise"] = float(
            np.clip(float(params.get("measurement_noise", out["measurement_noise"])), 1e-4, 10.0)
        )
    except Exception:
        pass
    out["use_landmark_confidence"] = bool(
        params.get("use_landmark_confidence", out["use_landmark_confidence"])
    )
    try:
        out["min_landmark_confidence"] = float(
            np.clip(float(params.get("min_landmark_confidence", out["min_landmark_confidence"])), 0.0, 1.0)
        )
    except Exception:
        pass
    try:
        out["confidence_floor"] = float(
            np.clip(float(params.get("confidence_floor", out["confidence_floor"])), 1e-4, 1.0)
        )
    except Exception:
        pass
    try:
        out["confidence_power"] = float(
            np.clip(float(params.get("confidence_power", out["confidence_power"])), 0.0, 4.0)
        )
    except Exception:
        pass
    try:
        out["max_confidence_noise_scale"] = float(
            np.clip(
                float(params.get("max_confidence_noise_scale", out["max_confidence_noise_scale"])),
                1.0,
                100.0,
            )
        )
    except Exception:
        pass
    try:
        out["gate_sigma"] = float(
            np.clip(float(params.get("gate_sigma", out["gate_sigma"])), 0.1, 60.0)
        )
    except Exception:
        pass
    try:
        out["max_consecutive_misses"] = int(
            np.clip(int(params.get("max_consecutive_misses", out["max_consecutive_misses"])), 0, 300)
        )
    except Exception:
        pass
    try:
        out["initial_velocity_std"] = float(
            np.clip(float(params.get("initial_velocity_std", out["initial_velocity_std"])), 1e-3, 50.0)
        )
    except Exception:
        pass
    try:
        out["velocity_decay"] = float(
            np.clip(float(params.get("velocity_decay", out["velocity_decay"])), 0.0, 1.0)
        )
    except Exception:
        pass
    try:
        out["max_speed"] = float(
            np.clip(float(params.get("max_speed", out["max_speed"])), 1e-3, 100.0)
        )
    except Exception:
        pass
    try:
        out["max_measurement_jump"] = float(
            np.clip(float(params.get("max_measurement_jump", out["max_measurement_jump"])), 1e-3, 10.0)
        )
    except Exception:
        pass
    try:
        out["reacquire_frames"] = int(
            np.clip(int(params.get("reacquire_frames", out["reacquire_frames"])), 1, 10)
        )
    except Exception:
        pass
    try:
        out["reacquire_max_jump"] = float(
            np.clip(float(params.get("reacquire_max_jump", out["reacquire_max_jump"])), 1e-3, 5.0)
        )
    except Exception:
        pass
    return out


def _coerce_measurement(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        v = np.asarray(value, dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(v)):
        return None
    return v


def _is_pose_impossible(skeleton: Dict[int, Optional[np.ndarray]]) -> bool:
    """Return True for clearly implausible exploded poses."""
    if not isinstance(skeleton, dict) or not skeleton:
        return True

    valid_points = []
    for idx in range(33):
        p = _coerce_measurement(skeleton.get(idx))
        if p is not None:
            valid_points.append(p)

    # Not enough geometry for a robust plausibility judgment.
    if len(valid_points) < 4:
        return False

    pts = np.vstack(valid_points)
    span = np.max(pts, axis=0) - np.min(pts, axis=0)
    if not np.all(np.isfinite(span)):
        return True
    if float(np.max(span)) > _POSE_MAX_AXIS_SPAN:
        return True

    hip_l = _coerce_measurement(skeleton.get(23))
    hip_r = _coerce_measurement(skeleton.get(24))
    if hip_l is not None and hip_r is not None:
        pelvis = 0.5 * (hip_l + hip_r)
    elif hip_l is not None:
        pelvis = hip_l
    elif hip_r is not None:
        pelvis = hip_r
    else:
        pelvis = np.median(pts, axis=0)
    radius = np.linalg.norm(pts - pelvis[None, :], axis=1)
    if not np.all(np.isfinite(radius)):
        return True
    if float(np.percentile(radius, 95.0)) > _POSE_MAX_RADIUS_FROM_PELVIS:
        return True

    checked = 0
    violations = 0
    for (a, b), (lo, hi) in _BONE_LENGTH_LIMITS.items():
        pa = _coerce_measurement(skeleton.get(a))
        pb = _coerce_measurement(skeleton.get(b))
        if pa is None or pb is None:
            continue
        checked += 1
        length = float(np.linalg.norm(pa - pb))
        if not np.isfinite(length) or length < lo or length > hi:
            violations += 1

    if checked >= _POSE_MIN_BONES_FOR_CHECK:
        ratio = float(violations) / float(max(1, checked))
        if violations >= _POSE_MIN_BONE_VIOLATIONS and ratio >= _POSE_MAX_BONE_VIOLATION_RATIO:
            return True

    return False


class _KalmanPoint3D:
    """Constant-velocity Kalman tracker for one 3D point."""

    def __init__(
        self,
        dt: float,
        process_noise_acc: float,
        measurement_noise: float,
        use_landmark_confidence: bool,
        min_landmark_confidence: float,
        confidence_floor: float,
        confidence_power: float,
        max_confidence_noise_scale: float,
        gate_sigma: float,
        max_consecutive_misses: int,
        initial_velocity_std: float,
        velocity_decay: float,
        max_speed: float,
        max_measurement_jump: float,
        reacquire_frames: int,
        reacquire_max_jump: float,
    ):
        self.dt = float(dt)
        self.process_noise_acc = float(process_noise_acc)
        self.measurement_noise = float(measurement_noise)
        self.use_landmark_confidence = bool(use_landmark_confidence)
        self.min_landmark_confidence = float(np.clip(min_landmark_confidence, 0.0, 1.0))
        self.confidence_floor = float(np.clip(confidence_floor, 1e-6, 1.0))
        self.confidence_power = float(max(0.0, confidence_power))
        self.max_confidence_noise_scale = float(max(1.0, max_confidence_noise_scale))
        self.gate_sigma = float(gate_sigma)
        self.max_consecutive_misses = int(max_consecutive_misses)
        self.initial_velocity_std = float(initial_velocity_std)
        self.velocity_decay = float(np.clip(float(velocity_decay), 0.0, 1.0))
        self.max_speed = float(max(1e-6, max_speed))
        self.max_measurement_jump = float(max(1e-6, max_measurement_jump))
        self.reacquire_frames = int(max(1, reacquire_frames))
        self.reacquire_max_jump = float(max(1e-6, reacquire_max_jump))

        self._initialised = False
        self._ever_initialised = False
        self._miss_count = 0
        self._candidate_z: Optional[np.ndarray] = None
        self._candidate_count = 0

        # State [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.R = np.eye(3, dtype=np.float64) * (self.measurement_noise ** 2)

        self.F = np.eye(6, dtype=np.float64)
        self.Q = np.eye(6, dtype=np.float64)
        self._rebuild_model()

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    def _rebuild_model(self):
        dt = self.dt
        q = self.process_noise_acc ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            self.Q[i, i] = dt4 * q * 0.25
            self.Q[i, i + 3] = dt3 * q * 0.5
            self.Q[i + 3, i] = dt3 * q * 0.5
            self.Q[i + 3, i + 3] = dt2 * q

    def set_dt(self, dt: float):
        dt = float(dt)
        if abs(dt - self.dt) <= 1e-9:
            return
        self.dt = dt
        self._rebuild_model()

    def _predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self._apply_velocity_constraints()

    def _apply_velocity_constraints(self, decay: bool = True):
        if decay:
            self.x[3:] *= self.velocity_decay
        speed = float(np.linalg.norm(self.x[3:]))
        if speed > self.max_speed:
            self.x[3:] *= (self.max_speed / speed)

    def _initialise(self, z: np.ndarray):
        self.x[:3] = z
        self.x[3:] = 0.0
        iv = self.initial_velocity_std ** 2
        self.P = np.diag([0.01, 0.01, 0.01, iv, iv, iv]).astype(np.float64)
        self._initialised = True
        self._ever_initialised = True
        self._miss_count = 0
        self._clear_reacquire_state()

    def _clear_reacquire_state(self):
        self._candidate_z = None
        self._candidate_count = 0

    def _reset_track(self):
        self._initialised = False
        self._miss_count = 0
        self.x[:] = 0.0
        self.P = np.eye(6, dtype=np.float64)
        self._clear_reacquire_state()

    def _required_reacquire_frames(self) -> int:
        if not self._ever_initialised:
            return 1
        return self.reacquire_frames

    def _try_reacquire(self, z: np.ndarray) -> bool:
        required = self._required_reacquire_frames()
        if required <= 1:
            self._initialise(z)
            return True

        if self._candidate_z is None:
            self._candidate_z = z.copy()
            self._candidate_count = 1
            return False

        jump = float(np.linalg.norm(z - self._candidate_z))
        if jump <= self.reacquire_max_jump:
            self._candidate_count += 1
            self._candidate_z = 0.5 * (self._candidate_z + z)
            if self._candidate_count >= required:
                self._initialise(self._candidate_z)
                return True
            return False

        self._candidate_z = z.copy()
        self._candidate_count = 1
        return False

    def _measurement_noise_sigma(self, confidence: Optional[float]) -> Optional[float]:
        if not self.use_landmark_confidence:
            return self.measurement_noise
        if confidence is None or not np.isfinite(confidence):
            return self.measurement_noise

        conf = float(np.clip(confidence, 0.0, 1.0))
        if conf < self.min_landmark_confidence:
            return self.measurement_noise * self.max_confidence_noise_scale

        conf_eff = max(self.confidence_floor, conf)
        scale = float((1.0 / conf_eff) ** self.confidence_power)
        scale = float(np.clip(scale, 1.0, self.max_confidence_noise_scale))
        return self.measurement_noise * scale

    def _is_low_confidence(self, confidence: Optional[float]) -> bool:
        if not self.use_landmark_confidence:
            return False
        if confidence is None or not np.isfinite(confidence):
            return False
        return float(np.clip(confidence, 0.0, 1.0)) < self.min_landmark_confidence

    def _update(self, z: np.ndarray, measurement_sigma: Optional[float] = None) -> bool:
        y = z - (self.H @ self.x)
        if float(np.linalg.norm(y)) > self.max_measurement_jump:
            return False
        if measurement_sigma is not None and np.isfinite(measurement_sigma):
            R_use = np.eye(3, dtype=np.float64) * (float(measurement_sigma) ** 2)
        else:
            R_use = self.R
        S = self.H @ self.P @ self.H.T + R_use
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        d2 = float(y.T @ S_inv @ y)
        if d2 > (self.gate_sigma ** 2):
            return False

        K = self.P @ self.H.T @ S_inv
        self.x = self.x + (K @ y)
        I_KH = np.eye(6, dtype=np.float64) - (K @ self.H)
        self.P = I_KH @ self.P @ I_KH.T + K @ R_use @ K.T
        self._apply_velocity_constraints(decay=False)
        return True

    def reset(self):
        self._ever_initialised = False
        self._reset_track()

    def step(
        self,
        measurement: Optional[np.ndarray],
        dt: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        if dt is not None:
            self.set_dt(float(dt))

        if measurement is not None:
            z = _coerce_measurement(measurement)
            if z is None:
                measurement = None
            else:
                low_conf = self._is_low_confidence(confidence)
                meas_sigma = self._measurement_noise_sigma(confidence)
                if meas_sigma is None:
                    measurement = None
                else:
                    if not self._initialised:
                        # Avoid initializing/reacquiring from weak detections.
                        if low_conf:
                            return None
                        if self._try_reacquire(z):
                            return self.position
                        return None
                    self._predict()
                    accepted = self._update(z, measurement_sigma=meas_sigma)
                    if accepted:
                        self._miss_count = 0
                        self._clear_reacquire_state()
                    else:
                        self._miss_count += 1
                        if self._miss_count > self.max_consecutive_misses:
                            self._reset_track()
                            if (not low_conf) and self._try_reacquire(z):
                                return self.position
                    if self._initialised:
                        return self.position
                    return None

        # No usable measurement this frame.
        if not self._initialised:
            self._clear_reacquire_state()
            return None
        self._predict()
        self._miss_count += 1
        if self._miss_count > self.max_consecutive_misses:
            self._reset_track()
            return None
        return self.position


class SkeletonPlacementKalman:
    """Kalman smoother for placed skeleton 3D landmark positions."""

    def __init__(
        self,
        fps: float = 30.0,
        process_noise_acc: float = 1.5,
        measurement_noise: float = 0.03,
        use_landmark_confidence: bool = True,
        min_landmark_confidence: float = 0.05,
        confidence_floor: float = 0.10,
        confidence_power: float = 1.0,
        max_confidence_noise_scale: float = 12.0,
        gate_sigma: float = 6.0,
        max_consecutive_misses: int = 20,
        initial_velocity_std: float = 1.0,
        velocity_decay: float = 0.97,
        max_speed: float = 6.0,
        max_measurement_jump: float = 0.75,
        reacquire_frames: int = 2,
        reacquire_max_jump: float = 0.35,
        landmark_count: int = 33,
    ):
        self.landmark_count = int(max(1, landmark_count))
        self.dt = 1.0 / max(float(fps), 1.0)
        self.trackers: List[_KalmanPoint3D] = [
            _KalmanPoint3D(
                dt=self.dt,
                process_noise_acc=process_noise_acc,
                measurement_noise=measurement_noise,
                use_landmark_confidence=use_landmark_confidence,
                min_landmark_confidence=min_landmark_confidence,
                confidence_floor=confidence_floor,
                confidence_power=confidence_power,
                max_confidence_noise_scale=max_confidence_noise_scale,
                gate_sigma=gate_sigma,
                max_consecutive_misses=max_consecutive_misses,
                initial_velocity_std=initial_velocity_std,
                velocity_decay=velocity_decay,
                max_speed=max_speed,
                max_measurement_jump=max_measurement_jump,
                reacquire_frames=reacquire_frames,
                reacquire_max_jump=reacquire_max_jump,
            )
            for _ in range(self.landmark_count)
        ]

    def reset(self):
        for tr in self.trackers:
            tr.reset()

    def smooth(
        self,
        skeleton: Optional[Dict[int, np.ndarray]],
        landmark_confidence: Optional[Dict[int, float]] = None,
    ) -> Optional[Dict[int, np.ndarray]]:
        """
        Smooth one frame.

        Args:
            skeleton: Landmark dict index->3D point (or None).
        Returns:
            Smoothed skeleton dict with 33 keys or None when no landmarks are trackable.
        """
        if skeleton is None:
            skel_in = {}
        else:
            skel_in = skeleton

        out: Dict[int, Optional[np.ndarray]] = {}
        any_valid = False
        for idx in range(self.landmark_count):
            meas = _coerce_measurement(skel_in.get(idx))
            conf = None
            if landmark_confidence is not None:
                try:
                    c = landmark_confidence.get(idx)
                    if c is not None:
                        conf = float(c)
                except Exception:
                    conf = None
            pos = self.trackers[idx].step(meas, dt=self.dt, confidence=conf)
            if pos is not None and np.all(np.isfinite(pos)):
                out[idx] = np.asarray(pos, dtype=np.float64)
                any_valid = True
            else:
                out[idx] = None

        if not any_valid:
            return None
        if _is_pose_impossible(out):
            # Hard fail-safe for occasional global tracker explosions.
            self.reset()
            return None
        return out


def apply_skeleton_filter_sequence(
    raw_skeletons: List[Optional[Dict[int, np.ndarray]]],
    fps: float,
    params: Optional[dict] = None,
    landmark_confidences: Optional[List[Optional[Dict[int, float]]]] = None,
) -> List[Optional[Dict[int, np.ndarray]]]:
    """Apply the skeleton smoother across a sequence."""
    cfg = normalize_skeleton_filter_params(params)
    if not cfg["enabled"]:
        return list(raw_skeletons)

    smoother = SkeletonPlacementKalman(
        fps=fps,
        process_noise_acc=cfg["process_noise_acc"],
        measurement_noise=cfg["measurement_noise"],
        use_landmark_confidence=cfg["use_landmark_confidence"],
        min_landmark_confidence=cfg["min_landmark_confidence"],
        confidence_floor=cfg["confidence_floor"],
        confidence_power=cfg["confidence_power"],
        max_confidence_noise_scale=cfg["max_confidence_noise_scale"],
        gate_sigma=cfg["gate_sigma"],
        max_consecutive_misses=cfg["max_consecutive_misses"],
        initial_velocity_std=cfg["initial_velocity_std"],
        velocity_decay=cfg["velocity_decay"],
        max_speed=cfg["max_speed"],
        max_measurement_jump=cfg["max_measurement_jump"],
        reacquire_frames=cfg["reacquire_frames"],
        reacquire_max_jump=cfg["reacquire_max_jump"],
    )
    if landmark_confidences is None:
        return [smoother.smooth(skel) for skel in raw_skeletons]

    out = []
    for i, skel in enumerate(raw_skeletons):
        conf = landmark_confidences[i] if i < len(landmark_confidences) else None
        out.append(smoother.smooth(skel, landmark_confidence=conf))
    return out
