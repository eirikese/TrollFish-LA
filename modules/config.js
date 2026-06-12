/**
 * TrollFish — config.js
 * Application constants ported from Python config.py + models.py
 */

export const SKELETON_TARGET_FPS = 10.0;
export const SKELETON_LOWER_PLANE_Z = 0.0;
export const SKELETON_HIP_PLANE_Z = 0.06;
export const MAP_MAX_POINTS_PER_TRACK = 2500;
export const MATCH_SAMPLE_POINTS = 320;

export const APP_ASSET_VERSION = '20260524boom5';
export const BOOM_MODEL_VERSION = '20260524-centered-boom1';

/** Maximum plausible GPS speed in m/s (~50 knots, used for outlier filter) */
export const MAX_PLAUSIBLE_SPEED_MS = 25.0;

/** Default camera parameters from ProjectCvConfig */
export const DEFAULT_CV_CONFIG = Object.freeze({
  pose_model: 'full',
  pose_mode: '3d',
  pose_min_confidence: 0.8,
  camera_position: [-3.194, 0.02, 0.585],
  camera_pitch_deg: 14.7,
  camera_yaw_deg: 0.0,
  camera_roll_deg: 0.0,
  camera_R_wc: null,
  lower_plane_z: 0.0,
  hip_plane_z: 0.07,
  lower_landmark: 'ankle',
  athlete_weight: 75.0,
  athlete_height: null,
  boat_com: -1.114,
  mediapipe_workers: null,
  skeleton_filter: {},
  contact_params: {},
  seated_x_stabilizer: {},
  lateral_y_stabilizer: {},
  auto_camera_pnp: {
    enabled: true,
    interval_frames: 10000,
    avg_frames: 5,
    min_valid_frames: 5,
  },
  // Fixed camera pose solved from the user's hand-corrected boat keypoints
  // (Hull 3D → Calibrate keypoints). When set, the pose engine uses it for the
  // whole clip and disables Auto-PnP. null = use Auto-PnP / defaults.
  //   { camPos:[x,y,z], R_wc:[9 or 3×3], keypoints:[{x,y,label}], frameTs, meanErrorPx }
  manual_camera_pose: null,
});

/** Calibration files will be loaded lazily and cached here */
const _calibrations = {};

/**
 * Load the GoPro fisheye calibration data.
 * Returns { K: number[][], D: number[][], rms: number, img_size: number[] }
 */
export function isGoPro13Model(value) {
  const text = String(value || '').toLowerCase();
  return /\bhero\s*13\b|\bhero13\b|\bgopro\s*13\b/.test(text);
}

export async function getCalibration(variant = 'default') {
  const key = variant === 'gopro13' ? 'gopro13' : 'default';
  if (_calibrations[key]) return _calibrations[key];
  const filename = key === 'gopro13' ? '../gopro13calib.json' : '../calibration.json';
  const resp = await fetch(new URL(filename, import.meta.url));
  if (!resp.ok) throw new Error(`Failed to load ${filename.replace('../', '')}`);
  _calibrations[key] = await resp.json();
  return _calibrations[key];
}

/**
 * SOG display cap (knots). Values above this are treated as outliers.
 */
export const SOG_MAX_KT = 22;

/**
 * Colour palette for tracks/athletes.
 */
export const PALETTE = Object.freeze([
  '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
  '#1abc9c', '#e67e22', '#00bcd4', '#ff6b6b', '#4ecdc4',
]);
