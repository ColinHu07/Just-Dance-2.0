"""Torso-centered, scale-normalized 2D skeletons for cross-subject comparison."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from app import pose_utils

# MediaPipe pose landmark indices (match pose_utils).
_NOSE = 0
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28

_MIN_SCALE = 0.04
"""Minimum torso scale in normalized image units to avoid blow-ups."""


def _reliability_for_index(landmarks: Sequence[Any], idx: int) -> float:
    if idx >= len(landmarks):
        return 0.0
    return float(pose_utils.landmark_reliability(landmarks[idx]))


def _visible_xy(landmarks: Sequence[Any], idx: int) -> Optional[Tuple[float, float, float]]:
    p = pose_utils.norm_xy_if_visible(landmarks, idx)
    if p is None:
        return None
    r = _reliability_for_index(landmarks, idx)
    return (p[0], p[1], r)


def compute_normalization_params(
    landmarks: Optional[Sequence[Any]],
) -> Tuple[Optional[Tuple[float, float]], float, np.ndarray]:
    """
    Return ``(origin_xy, scale, reliability_33)`` for this frame.

    Origin: hip midpoint if both hips visible; else shoulder midpoint; else torso centroid
    from ``body_center_normalized``. Scale: mean of shoulder width and hip width in norm
    space when both endpoints exist; else whichever exists; else ``_MIN_SCALE``.
    """
    rel = np.zeros(33, dtype=np.float64)
    if not landmarks:
        return None, _MIN_SCALE, rel

    for i in range(min(33, len(landmarks))):
        rel[i] = _reliability_for_index(landmarks, i)

    origin = pose_utils.body_center_normalized(list(landmarks))
    if origin is None:
        return None, _MIN_SCALE, rel

    ls = _visible_xy(landmarks, _LEFT_SHOULDER)
    rs = _visible_xy(landmarks, _RIGHT_SHOULDER)
    lh = _visible_xy(landmarks, _LEFT_HIP)
    rh = _visible_xy(landmarks, _RIGHT_HIP)

    shoulder_w = 0.0
    if ls and rs:
        shoulder_w = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
    hip_w = 0.0
    if lh and rh:
        hip_w = math.hypot(rh[0] - lh[0], rh[1] - lh[1])

    scales: List[float] = []
    if shoulder_w > 1e-6:
        scales.append(shoulder_w)
    if hip_w > 1e-6:
        scales.append(hip_w)
    if scales:
        scale = max(_MIN_SCALE, float(sum(scales) / len(scales)))
    else:
        scale = _MIN_SCALE

    return origin, scale, rel


def normalize_landmarks_to_xy(
    landmarks: Optional[Sequence[Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map raw landmarks to a ``(33, 2)`` array in torso-normalized coordinates.

    Rows are NaN when the landmark is not visible under the standard visibility gate.
    """
    xy = np.full((33, 2), np.nan, dtype=np.float64)
    rel = np.zeros(33, dtype=np.float64)
    if not landmarks:
        return xy, rel

    origin, scale, rel = compute_normalization_params(landmarks)
    if origin is None:
        return xy, rel

    ox, oy = origin
    for i in range(min(33, len(landmarks))):
        p = pose_utils.norm_xy_if_visible(landmarks, i)
        if p is None:
            rel[i] = 0.0
            continue
        xy[i, 0] = (p[0] - ox) / scale
        xy[i, 1] = (p[1] - oy) / scale
    return xy, rel
