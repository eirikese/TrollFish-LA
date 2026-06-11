import numpy as np
from typing import Dict, Tuple, Optional


def _coerce_point3(value) -> Optional[np.ndarray]:
    """Return finite 3D point or None."""
    if value is None:
        return None
    try:
        p = np.asarray(value, dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(p)):
        return None
    return p


def compute_trunk_vector(
    skeleton: Dict[int, np.ndarray],
    upper_idx: int = 11,   # shoulders midpoint
    lower_idx: int = 23,   # hips midpoint
) -> Optional[np.ndarray]:
    """
    Returns normalized trunk direction vector (lower -> upper).
    """
    if upper_idx not in skeleton or lower_idx not in skeleton:
        return None

    p_upper = _coerce_point3(skeleton[upper_idx])
    p_lower = _coerce_point3(skeleton[lower_idx])
    if p_upper is None or p_lower is None:
        return None

    v = p_upper - p_lower
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return None

    return v / norm


def trunk_angle_to_vertical(
    skeleton: Dict[int, np.ndarray],
    vertical_axis: np.ndarray = np.array([0, 0, 1]),
) -> Optional[float]:
    """
    Returns trunk angle in degrees relative to vertical (Z).
    """
    v_trunk = compute_trunk_vector(skeleton)
    if v_trunk is None:
        return None

    vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
    cos_theta = np.clip(np.dot(v_trunk, vertical_axis), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    return np.degrees(angle_rad)


def compute_center_of_mass(skeleton: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """
    Estimate center of mass from 3D skeleton landmarks using a segmental method.

    This is robust to missing limbs: available segments are accumulated and then
    renormalized by the represented mass fraction.
    """
    def seg_com(p1, p2, fraction):
        return p1 + fraction * (p2 - p1)

    if not isinstance(skeleton, dict) or not skeleton:
        return None

    def point(idx: int) -> Optional[np.ndarray]:
        return _coerce_point3(skeleton.get(idx))

    def midpoint_or_single(a: int, b: int) -> Optional[np.ndarray]:
        pa = point(a)
        pb = point(b)
        if pa is not None and pb is not None:
            return 0.5 * (pa + pb)
        if pa is not None:
            return pa
        if pb is not None:
            return pb
        return None

    segments = []

    def add_segment(p1: Optional[np.ndarray], p2: Optional[np.ndarray], mass_f: float, com_frac: float):
        if p1 is None or p2 is None:
            return
        segments.append((p1, p2, mass_f, com_frac))

    mid_hip = midpoint_or_single(23, 24)
    mid_shoulder = midpoint_or_single(11, 12)
    head_top = midpoint_or_single(7, 8)

    # Axial segments
    add_segment(mid_shoulder, head_top, 0.081, 1.000)  # Head and neck
    add_segment(mid_shoulder, mid_hip, 0.497, 0.500)   # Trunk

    # Arms
    add_segment(point(11), point(13), 0.028, 0.436)  # Left upper arm
    add_segment(point(13), point(15), 0.016, 0.430)  # Left forearm
    add_segment(point(15), point(17), 0.006, 0.506)  # Left hand

    add_segment(point(12), point(14), 0.028, 0.436)  # Right upper arm
    add_segment(point(14), point(16), 0.016, 0.430)  # Right forearm
    add_segment(point(16), point(18), 0.006, 0.506)  # Right hand

    # Legs
    add_segment(point(23), point(25), 0.100, 0.433)   # Left thigh
    add_segment(point(25), point(27), 0.0465, 0.433)  # Left shank
    add_segment(point(27), point(31), 0.0145, 0.500)  # Left foot

    add_segment(point(24), point(26), 0.100, 0.433)   # Right thigh
    add_segment(point(26), point(28), 0.0465, 0.433)  # Right shank
    add_segment(point(28), point(32), 0.0145, 0.500)  # Right foot

    total_mass = 0.0
    com_sum = np.zeros(3, dtype=np.float64)
    for p1, p2, mass_f, com_frac in segments:
        seg_com_pos = seg_com(p1, p2, com_frac)
        com_sum += float(mass_f) * seg_com_pos
        total_mass += float(mass_f)

    # Guard against tiny represented mass fractions causing unstable COM.
    if total_mass < 0.35:
        return None

    return com_sum / total_mass
