"""Per-frame geometry features (angles, directions, distances, posture) for DTW."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from app.comparison_types import FrameFeatures, PoseFrame
from app.normalization import (
    _LEFT_ANKLE,
    _LEFT_ELBOW,
    _LEFT_HIP,
    _LEFT_KNEE,
    _LEFT_SHOULDER,
    _LEFT_WRIST,
    _RIGHT_ANKLE,
    _RIGHT_ELBOW,
    _RIGHT_HIP,
    _RIGHT_KNEE,
    _RIGHT_SHOULDER,
    _RIGHT_WRIST,
)

_REL_GATE = 0.35
"""Below this joint reliability, treat geometry as unknown for that joint."""


def _min_rel(rel: np.ndarray, *indices: int) -> float:
    m = 1.0
    for i in indices:
        if 0 <= i < len(rel):
            m = min(m, float(rel[i]))
        else:
            return 0.0
    return m


def _point_w(xy: np.ndarray, rel: np.ndarray, idx: int) -> Optional[Tuple[float, float, float]]:
    if idx >= xy.shape[0] or idx >= rel.shape[0]:
        return None
    if rel[idx] < _REL_GATE:
        return None
    if not np.isfinite(xy[idx, 0]) or not np.isfinite(xy[idx, 1]):
        return None
    return float(xy[idx, 0]), float(xy[idx, 1]), float(rel[idx])


def _angle_at(
    xy: np.ndarray,
    rel: np.ndarray,
    ia: int,
    ib: int,
    ic: int,
) -> Tuple[float, float]:
    """Interior angle at B (radians) and weight ``min(rel)``."""
    w = _min_rel(rel, ia, ib, ic)
    if w < _REL_GATE:
        return float("nan"), 0.0
    pa = _point_w(xy, rel, ia)
    pb = _point_w(xy, rel, ib)
    pc = _point_w(xy, rel, ic)
    if pa is None or pb is None or pc is None:
        return float("nan"), 0.0
    ax, ay, _ = pa
    bx, by, _ = pb
    cx, cy, _ = pc
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)
    la = math.hypot(ba[0], ba[1])
    lc = math.hypot(bc[0], bc[1])
    if la < 1e-8 or lc < 1e-8:
        return float("nan"), 0.0
    dot = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / (la * lc)))
    ang = math.acos(dot)
    return ang, w


def _unit_dir(
    xy: np.ndarray,
    rel: np.ndarray,
    i0: int,
    i1: int,
) -> Tuple[float, float, float]:
    """Unit vector from i0→i1 and weight."""
    w = _min_rel(rel, i0, i1)
    if w < _REL_GATE:
        return float("nan"), float("nan"), 0.0
    p0 = _point_w(xy, rel, i0)
    p1 = _point_w(xy, rel, i1)
    if p0 is None or p1 is None:
        return float("nan"), float("nan"), 0.0
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-8:
        return float("nan"), float("nan"), 0.0
    return dx / L, dy / L, w


def _midpoint(
    xy: np.ndarray,
    rel: np.ndarray,
    a: int,
    b: int,
) -> Optional[Tuple[float, float, float]]:
    pa = _point_w(xy, rel, a)
    pb = _point_w(xy, rel, b)
    if pa is None or pb is None:
        return None
    w = min(pa[2], pb[2])
    return (pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, w


def _dist(
    xy: np.ndarray,
    rel: np.ndarray,
    i0: int,
    i1: int,
) -> Tuple[float, float]:
    w = _min_rel(rel, i0, i1)
    if w < _REL_GATE:
        return float("nan"), 0.0
    p0 = _point_w(xy, rel, i0)
    p1 = _point_w(xy, rel, i1)
    if p0 is None or p1 is None:
        return float("nan"), 0.0
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return d, w


# --- Feature layout: total dimension and group masks ---
FEATURE_DIM = 41

# Slice ranges (exclusive end).
SL_ANGLES = slice(0, 11)  # 11 angles
SL_DIR = slice(11, 33)  # 11 * 2 = 22
SL_DIST = slice(33, 41)  # 8 distances


def build_frame_features(frame: PoseFrame) -> FrameFeatures:
    xy = frame.joints_norm_xy
    rel = frame.reliability
    v = np.full(FEATURE_DIM, np.nan, dtype=np.float64)
    wv = np.zeros(FEATURE_DIM, dtype=np.float64)

    def put_angle(slot: int, ang: float, wt: float) -> None:
        v[slot] = ang
        wv[slot] = wt

    def put_vec(slot: int, ux: float, uy: float, wt: float) -> None:
        v[slot] = ux
        v[slot + 1] = uy
        wv[slot] = wt
        wv[slot + 1] = wt

    def put_scalar(slot: int, val: float, wt: float) -> None:
        v[slot] = val
        wv[slot] = wt

    # Angles 0..10
    pairs = [
        (0, _LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST),
        (1, _RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST),
        (2, _LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE),
        (3, _RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE),
        (4, _LEFT_HIP, _LEFT_SHOULDER, _LEFT_ELBOW),
        (5, _RIGHT_HIP, _RIGHT_SHOULDER, _RIGHT_ELBOW),
        (6, _LEFT_SHOULDER, _LEFT_HIP, _LEFT_KNEE),
        (7, _RIGHT_SHOULDER, _RIGHT_HIP, _RIGHT_KNEE),
    ]
    for slot, ia, ib, ic in pairs:
        ang, wt = _angle_at(xy, rel, ia, ib, ic)
        put_angle(slot, ang, wt)

    # Shoulder / hip line tilt (absolute orientation)
    ls = _point_w(xy, rel, _LEFT_SHOULDER)
    rs = _point_w(xy, rel, _RIGHT_SHOULDER)
    if ls and rs and min(ls[2], rs[2]) >= _REL_GATE:
        put_angle(8, math.atan2(rs[1] - ls[1], rs[0] - ls[0]), min(ls[2], rs[2]))
    else:
        put_angle(8, float("nan"), 0.0)

    lh = _point_w(xy, rel, _LEFT_HIP)
    rh = _point_w(xy, rel, _RIGHT_HIP)
    if lh and rh and min(lh[2], rh[2]) >= _REL_GATE:
        put_angle(9, math.atan2(rh[1] - lh[1], rh[0] - lh[0]), min(lh[2], rh[2]))
    else:
        put_angle(9, float("nan"), 0.0)

    mid_s = _midpoint(xy, rel, _LEFT_SHOULDER, _RIGHT_SHOULDER)
    mid_h = _midpoint(xy, rel, _LEFT_HIP, _RIGHT_HIP)
    if mid_s and mid_h and min(mid_s[2], mid_h[2]) >= _REL_GATE:
        put_angle(
            10,
            math.atan2(mid_s[1] - mid_h[1], mid_s[0] - mid_h[0]),
            min(mid_s[2], mid_h[2]),
        )
    else:
        put_angle(10, float("nan"), 0.0)

    # Directions 11..32 (11 vectors)
    dir_specs = [
        (11, _LEFT_SHOULDER, _LEFT_ELBOW),
        (13, _LEFT_ELBOW, _LEFT_WRIST),
        (15, _RIGHT_SHOULDER, _RIGHT_ELBOW),
        (17, _RIGHT_ELBOW, _RIGHT_WRIST),
        (19, _LEFT_HIP, _LEFT_KNEE),
        (21, _LEFT_KNEE, _LEFT_ANKLE),
        (23, _RIGHT_HIP, _RIGHT_KNEE),
        (25, _RIGHT_KNEE, _RIGHT_ANKLE),
    ]
    base = 11
    for i, (slot, a, b) in enumerate(dir_specs):
        ux, uy, wt = _unit_dir(xy, rel, a, b)
        put_vec(base + i * 2, ux, uy, wt)

    off = 11 + len(dir_specs) * 2  # 11 + 16 = 27
    mid_s = _midpoint(xy, rel, _LEFT_SHOULDER, _RIGHT_SHOULDER)
    mid_h = _midpoint(xy, rel, _LEFT_HIP, _RIGHT_HIP)
    if mid_s and mid_h and min(mid_s[2], mid_h[2]) >= _REL_GATE:
        dx = mid_h[0] - mid_s[0]
        dy = mid_h[1] - mid_s[1]
        L = math.hypot(dx, dy)
        wt = min(mid_s[2], mid_h[2])
        if L >= 1e-8:
            put_vec(off, dx / L, dy / L, wt)
        else:
            put_vec(off, float("nan"), float("nan"), 0.0)
    else:
        put_vec(off, float("nan"), float("nan"), 0.0)
    off += 2

    ux, uy, wt = _unit_dir(xy, rel, _LEFT_SHOULDER, _RIGHT_SHOULDER)
    put_vec(off, ux, uy, wt)
    off += 2
    ux, uy, wt = _unit_dir(xy, rel, _LEFT_HIP, _RIGHT_HIP)
    put_vec(off, ux, uy, wt)

    # Distances 33..40
    d, wt = _dist(xy, rel, _LEFT_WRIST, _RIGHT_WRIST)
    put_scalar(33, d, wt)
    d, wt = _dist(xy, rel, _LEFT_ANKLE, _RIGHT_ANKLE)
    put_scalar(34, d, wt)

    mid_s2 = _midpoint(xy, rel, _LEFT_SHOULDER, _RIGHT_SHOULDER)
    mid_h2 = _midpoint(xy, rel, _LEFT_HIP, _RIGHT_HIP)
    torso_c: Optional[Tuple[float, float, float]] = None
    if mid_s2 and mid_h2:
        torso_c = (
            (mid_s2[0] + mid_h2[0]) * 0.5,
            (mid_s2[1] + mid_h2[1]) * 0.5,
            min(mid_s2[2], mid_h2[2]),
        )

    if torso_c is not None and torso_c[2] >= _REL_GATE:
        mt = (torso_c[0], torso_c[1])
        for j, wrist in enumerate((_LEFT_WRIST, _RIGHT_WRIST)):
            pw = _point_w(xy, rel, wrist)
            slot = 35 + j
            if pw is None:
                put_scalar(slot, float("nan"), 0.0)
            else:
                wt = min(torso_c[2], pw[2])
                put_scalar(slot, math.hypot(pw[0] - mt[0], pw[1] - mt[1]), wt)
    else:
        for j in range(2):
            put_scalar(35 + j, float("nan"), 0.0)

    d, wt = _dist(xy, rel, _LEFT_HIP, _LEFT_ANKLE)
    put_scalar(37, d, wt)
    d, wt = _dist(xy, rel, _RIGHT_HIP, _RIGHT_ANKLE)
    put_scalar(38, d, wt)

    # Hand height vs shoulders (positive = wrist above shoulder in image coords = smaller y in math?)
    # In normalized frame, y increases downward; "higher" = smaller y.
    for j, (wrist, shoulder) in enumerate(
        ((_LEFT_WRIST, _LEFT_SHOULDER), (_RIGHT_WRIST, _RIGHT_SHOULDER))
    ):
        slot = 39 + j
        pw = _point_w(xy, rel, wrist)
        ps = _point_w(xy, rel, shoulder)
        if pw and ps and min(pw[2], ps[2]) >= _REL_GATE:
            # shoulder_y - wrist_y: positive when wrist is above shoulder
            put_scalar(slot, ps[1] - pw[1], min(pw[2], ps[2]))
        else:
            put_scalar(slot, float("nan"), 0.0)

    masks = _build_group_masks()
    return FrameFeatures(
        frame_index=frame.frame_index,
        time_sec=frame.time_sec,
        vector=v,
        dim_weight=wv,
        group_masks=masks,
    )


def _build_group_masks() -> dict[str, np.ndarray]:
    g_angles = np.zeros(FEATURE_DIM, dtype=bool)
    # Joint-chain angles only (shoulder/hip line + torso axis live under posture).
    g_angles[0:8] = True
    g_dir = np.zeros(FEATURE_DIM, dtype=bool)
    g_dir[SL_DIR] = True
    g_dist = np.zeros(FEATURE_DIM, dtype=bool)
    g_dist[SL_DIST] = True
    g_post = np.zeros(FEATURE_DIM, dtype=bool)
    g_post[[8, 9, 10]] = True

    g_arms = np.zeros(FEATURE_DIM, dtype=bool)
    g_arms[[0, 1, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18]] = True
    g_arms[[33, 35, 36, 39, 40]] = True

    g_legs = np.zeros(FEATURE_DIM, dtype=bool)
    g_legs[[2, 3, 6, 7, 19, 20, 21, 22, 23, 24, 25, 26]] = True
    g_legs[[34, 37, 38]] = True

    g_torso = np.zeros(FEATURE_DIM, dtype=bool)
    g_torso[[8, 9, 10, 27, 28, 29, 30, 31, 32]] = True

    return {
        "angles": g_angles,
        "directions": g_dir,
        "distances": g_dist,
        "posture": g_post,
        "arms": g_arms,
        "legs": g_legs,
        "torso": g_torso,
    }
