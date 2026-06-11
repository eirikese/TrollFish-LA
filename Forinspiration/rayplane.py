import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd


POSE_CSV_PATH = "GX010235_pose.csv"


def _rot_x(deg):
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def _rot_y(deg):
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def _rot_z(deg):
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)


def default_camera_pose_and_rotation(pitch=8.0, yaw=0.0, roll=0.0):
    """Default camera pose and rotation camera->world.

    Args:
        pitch: Down-tilt angle in degrees (positive = looking down).
        yaw: Left/right yaw in degrees.
        roll: Roll angle in degrees.
    """
    camera_pos = np.array([-3.194, 0.0, 0.585], dtype=np.float64)
    R_wc = np.array([
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0],
    ], dtype=np.float64)

    # Relative camera rotation in XYZ convention:
    # Rx uses negative pitch to preserve legacy behavior.
    R_rel = _rot_z(roll) @ _rot_y(yaw) @ _rot_x(-pitch)
    R_wc = R_wc @ R_rel
    return camera_pos, R_wc


def load_fisheye_undistorted_intrinsics(npz_path="gopro_fisheye_calib.npz", balance=0.0, new_size=None):
    """Load fisheye intrinsics and return (K_undist, (W,H)).

    Expects an .npz with keys: K, D, img_size.
    """
    cal = np.load(npz_path)
    K, D = cal["K"], cal["D"]
    img_size = tuple(cal["img_size"])
    if new_size is None:
        new_size = img_size

    R = np.eye(3)
    K_new = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_size, R, balance=float(balance), new_size=new_size
    )
    W, H = new_size
    return K_new, (int(W), int(H))



def ray_from_norm_landmark_undistorted(x_norm, y_norm, W, H, K):
    """
    x_norm, y_norm: MediaPipe normalized landmark in [0,1] (image coordinates)
    W, H: undistorted image width/height
    K: intrinsics for the undistorted image (fx,fy,cx,cy)
    returns: unit ray direction d_c in camera frame
    """
    u = float(x_norm) * W
    v = float(y_norm) * H

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    x_n = (u - cx) / fx
    y_n = (v - cy) / fy

    d_c = np.array([x_n, y_n, 1.0], dtype=np.float64)
    d_c /= np.linalg.norm(d_c)
    return d_c


def get_landmark_norm(row, landmark_idx):
    x_norm = row.get(f"lm{landmark_idx}_norm_x")
    y_norm = row.get(f"lm{landmark_idx}_norm_y")
    z_norm = row.get(f"lm{landmark_idx}_norm_z")
    if pd.isna(x_norm) or pd.isna(y_norm) or pd.isna(z_norm):
        return None
    return np.array([x_norm, y_norm, z_norm], dtype=np.float64)


def get_landmark_world_cam(row, landmark_idx):
    """Returns MediaPipe 'world' coords as stored in CSV (camera frame)."""
    x_world = row.get(f"lm{landmark_idx}_world_x")
    y_world = row.get(f"lm{landmark_idx}_world_y")
    z_world = row.get(f"lm{landmark_idx}_world_z")
    if pd.isna(x_world) or pd.isna(y_world) or pd.isna(z_world):
        return None
    return np.array([x_world, y_world, z_world], dtype=np.float64)

def intersect_world_z_plane(d_c, R_wc, t_wc, Z0, eps=1e-9):
    """
    R_wc: 3x3 rotation camera->world
    t_wc: (3,) translation camera origin in world
    """
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3,3)
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3)

    o_w = t_wc
    d_w = R_wc @ d_c

    dz = d_w[2]
    if abs(dz) < eps:
        return None

    t = (Z0 - o_w[2]) / dz
    if t < 0:
        return None

    return o_w + t * d_w


def get_landmark_position(row, landmark_idx):
    """Extract landmark world coordinates and transform to boat frame."""
    world_coords = get_landmark_world_cam(row, landmark_idx)
    norm_coords = get_landmark_norm(row, landmark_idx)
    if world_coords is None or norm_coords is None:
        return None
    return world_coords, norm_coords

def load_pose_data(csv_path):
    """Load pose data from CSV and transform to boat coordinates."""
    if not os.path.exists(csv_path):
        csv_path = Path(__file__).parent.parent / csv_path
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} frames from pose CSV")    
    return df


def rotation_from_a_to_b(a, b, eps=1e-9):
    """
    Returns R such that R @ a_hat = b_hat (as close as possible).
    a, b: 3-vectors
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return None  # undefined rotation

    a_hat = a / na
    b_hat = b / nb

    v = np.cross(a_hat, b_hat)
    c = np.dot(a_hat, b_hat)  # cos(theta)
    s = np.linalg.norm(v)     # sin(theta)

    # If vectors are already aligned
    if s < eps and c > 0:
        return np.eye(3)

    # If vectors are opposite, choose an arbitrary orthogonal axis
    if s < eps and c < 0:
        # Pick a vector not parallel to a_hat
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a_hat)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a_hat, tmp)
        axis /= np.linalg.norm(axis)

        # 180-degree rotation around 'axis': R = I + 2*[axis]_x^2
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + 2 * (K @ K)

    # General case: Rodrigues formula
    k = v / s
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + K * s + (K @ K) * (1 - c)
    return R


def _normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def _rotation_about_axis(axis, angle_rad):
    """Rodrigues rotation matrix for rotation around a unit axis."""
    axis = _normalize(axis)
    if axis is None:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def rotation_from_two_vectors(a1, a2, b1, b2, eps=1e-9):
    """Return R such that R @ a1_hat ≈ b1_hat and R also aligns the roll using a2->b2.

    a1/b1 define the primary axis; a2/b2 define the secondary axis to disambiguate roll.
    """
    a1 = np.asarray(a1, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    e1a = _normalize(a1, eps=eps)
    e1b = _normalize(b1, eps=eps)
    if e1a is None or e1b is None:
        return None

    a2_ortho = a2 - np.dot(a2, e1a) * e1a
    b2_ortho = b2 - np.dot(b2, e1b) * e1b

    e2a = _normalize(a2_ortho, eps=eps)
    e2b = _normalize(b2_ortho, eps=eps)
    if e2a is None or e2b is None:
        return None

    e3a = np.cross(e1a, e2a)
    e3b = np.cross(e1b, e2b)
    e3a = _normalize(e3a, eps=eps)
    e3b = _normalize(e3b, eps=eps)
    if e3a is None or e3b is None:
        return None

    A = np.column_stack([e1a, e2a, e3a])
    B = np.column_stack([e1b, e2b, e3b])
    return B @ A.T


def place_skeleton_on_boat(
    row,
    anchor_idx=24,
    dir_idx=28,
    anchor_boat=None,
    dir_boat=None,
    use_scale=False,
    roll_idx=None,
    roll_boat=None,
    src_anchor=None,
    src_dir=None,
    src_roll=None,
):
    """
    Returns dict: landmark_idx -> 3D point in boat/world coords
    anchor_boat: boat/world position for landmark anchor_idx (e.g., lm24_intersect)
    dir_boat: boat/world position for landmark dir_idx (e.g., lm28_intersect)
    
    Roll is now computed directly from MediaPipe world coordinates, not from
    raycast intersections, to preserve the actual body roll orientation.
    """

    # Source anchor and direction in MediaPipe-world (from CSV), optionally overridden.
    if src_anchor is None:
        src_anchor_pos = get_landmark_position(row, anchor_idx)
        if src_anchor_pos is None:
            return None
        src_anchor, _ = src_anchor_pos
    else:
        src_anchor = np.asarray(src_anchor, dtype=np.float64).reshape(3)

    if src_dir is None:
        src_dir_pos = get_landmark_position(row, dir_idx)
        if src_dir_pos is None:
            return None
        src_dir, _ = src_dir_pos
    else:
        src_dir = np.asarray(src_dir, dtype=np.float64).reshape(3)

    if anchor_boat is None or dir_boat is None:
        return None
    anchor_boat = np.asarray(anchor_boat, dtype=np.float64).reshape(3)
    dir_boat = np.asarray(dir_boat, dtype=np.float64).reshape(3)

    v_src_primary = src_dir - src_anchor
    v_tgt_primary = dir_boat - anchor_boat

    R = None
    if src_roll is None and roll_idx is not None:
        src_roll_pos = get_landmark_position(row, int(roll_idx))
        if src_roll_pos is not None:
            src_roll, _ = src_roll_pos
    if src_roll is not None and roll_boat is not None:
        src_roll = np.asarray(src_roll, dtype=np.float64).reshape(3)
        roll_boat = np.asarray(roll_boat, dtype=np.float64).reshape(3)
        v_src_secondary = src_roll - src_anchor
        v_tgt_secondary = roll_boat - anchor_boat
        if np.linalg.norm(v_src_secondary) > 1e-9 and np.linalg.norm(v_tgt_secondary) > 1e-9:
            R = rotation_from_two_vectors(
                v_src_primary,
                v_src_secondary,
                v_tgt_primary,
                v_tgt_secondary,
            )

    if R is None:
        R = rotation_from_a_to_b(v_src_primary, v_tgt_primary)
        if R is None:
            return None

    s = 1.0
    if use_scale:
        ns = np.linalg.norm(v_src_primary)
        nt = np.linalg.norm(v_tgt_primary)
        if ns > 1e-9:
            s = nt / ns

    # Resolve 180-degree roll ambiguity by choosing the more upright candidate.
    # This prevents frequent upside-down flips when only weak roll cues are available.
    src_cache = {}
    for idx in (0, 11, 12, 23, 24, 27, 28, 31, 32):
        pos = get_landmark_position(row, idx)
        if pos is not None:
            src_cache[idx] = np.asarray(pos[0], dtype=np.float64).reshape(3)

    def _transform_point(R_mat, p_src):
        return anchor_boat + s * (R_mat @ (p_src - src_anchor))

    def _upright_metrics(R_mat):
        hip_l = src_cache.get(23)
        hip_r = src_cache.get(24)
        if hip_l is None or hip_r is None:
            return None
        hip_mid = 0.5 * (_transform_point(R_mat, hip_l) + _transform_point(R_mat, hip_r))

        metrics = {
            "shoulder_delta": None,
            "head_delta": None,
            "ankle_delta": None,
        }

        sh_l = src_cache.get(11)
        sh_r = src_cache.get(12)
        if sh_l is not None and sh_r is not None:
            sh_mid = 0.5 * (_transform_point(R_mat, sh_l) + _transform_point(R_mat, sh_r))
            metrics["shoulder_delta"] = float(sh_mid[2] - hip_mid[2])

        head = src_cache.get(0)
        if head is not None:
            head_p = _transform_point(R_mat, head)
            metrics["head_delta"] = float(head_p[2] - hip_mid[2])

        ankles = []
        for idx in (27, 28, 31, 32):
            p = src_cache.get(idx)
            if p is not None:
                ankles.append(_transform_point(R_mat, p))
        if ankles:
            ankle_mid = np.mean(np.asarray(ankles, dtype=np.float64), axis=0)
            metrics["ankle_delta"] = float(hip_mid[2] - ankle_mid[2])

        if all(v is None for v in metrics.values()):
            return None
        return metrics

    def _upright_rank(metrics):
        # Lower penalty is better; secondary key is larger positive separation.
        penalty = 0.0
        bonus = 0.0
        for key, base_w, miss_pen in (
            ("shoulder_delta", 20.0, 4.0),
            ("head_delta", 16.0, 3.0),
            ("ankle_delta", 10.0, 2.0),
        ):
            v = metrics.get(key)
            if v is None:
                penalty += miss_pen
                continue
            if v < 0.0:
                penalty += base_w + (-v)
            else:
                bonus += v
        return (penalty, -bonus)

    axis = _normalize(v_tgt_primary)
    if axis is not None:
        R_flip = _rotation_about_axis(axis, np.pi) @ R
        metrics_base = _upright_metrics(R)
        metrics_flip = _upright_metrics(R_flip)
        if metrics_base is not None and metrics_flip is not None:
            if _upright_rank(metrics_flip) < _upright_rank(metrics_base):
                R = R_flip
        elif metrics_flip is not None and metrics_base is None:
            R = R_flip

    placed = {}
    # Assuming MediaPipe pose has 33 landmarks: 0..32
    for i in range(33):
        p_i, _ = get_landmark_position(row, i)
        if p_i is None:
            placed[i] = None
            continue
        placed[i] = anchor_boat + s * (R @ (p_i - src_anchor))
    return placed



if __name__ == "__main__":
    # Demo / debug runner
    try:
        K, (W, H) = load_fisheye_undistorted_intrinsics("gopro_fisheye_calib.npz", balance=0.0)
    except Exception as e:
        raise RuntimeError(f"Failed to load fisheye calibration: {e}")

    camera_pos, R_wc = default_camera_pose_and_rotation()

    df = load_pose_data(POSE_CSV_PATH)
    for frame_idx in range(len(df)):
        row = df.iloc[frame_idx]
        lm24_world, lm24_norm = get_landmark_position(row, 24)
        lm24_ray = ray_from_norm_landmark_undistorted(
            lm24_norm[0], lm24_norm[1], W, H, K)
        lm24_intersect = intersect_world_z_plane(
            lm24_ray, R_wc=R_wc, t_wc=camera_pos, Z0=0.12
        )
        lm28_world, lm28_norm = get_landmark_position(row, 28)
        lm28_ray = ray_from_norm_landmark_undistorted(
            lm28_norm[0], lm28_norm[1], W, H, K)
        lm28_intersect = intersect_world_z_plane(
            lm28_ray, R_wc=R_wc, t_wc=camera_pos, Z0=-0.01
        )
        
        # print(f"Frame {frame_idx}: LM28 world={lm28_world}, intersect={lm28_intersect}")
        if lm24_intersect is None or lm28_intersect is None:
            print(f"Frame {frame_idx}: No intersection for LM24 or LM28")
            continue
        direction_vector = lm28_intersect - lm24_intersect
        direction_vector /= np.linalg.norm(direction_vector)
        
        print(f"Frame {frame_idx}: Direction vector from LM24 to LM28: {direction_vector}")
        placed = place_skeleton_on_boat(
            row,
            anchor_idx=24,
            dir_idx=28,
            anchor_boat=lm24_intersect,
            dir_boat=lm28_intersect,
            use_scale=False 
        )

        if placed is None:
            print(f"Frame {frame_idx}: Could not place skeleton")
            continue

        # Example: print where hip (24) and ankle (28) ended up
        print("Placed 24:", placed[24], "target anchor:", lm24_intersect)
        print("Placed 28:", placed[28], "target dir:", lm28_intersect)
