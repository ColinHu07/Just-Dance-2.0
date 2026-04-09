"""Aligned overlay blending and DTW path helpers for visual comparison."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# User performance layer opacity (reference stays fully opaque as the base).
DEFAULT_USER_OVERLAY_ALPHA = 0.4


def subsample_alignment_path(path: np.ndarray, max_steps: int = 480) -> np.ndarray:
    """
    Reduce DTW path length for smoother playback and lower seek load.

    Uniformly samples row indices along the path while keeping endpoints.
    """
    if path is None or path.size == 0:
        return path
    n = int(path.shape[0])
    if n <= max_steps:
        return np.asarray(path, dtype=np.int64).copy()
    t = np.linspace(0, n - 1, max_steps)
    ix = np.unique(np.round(t).astype(np.int64))
    return np.asarray(path, dtype=np.int64)[ix]


def blend_overlay_bgr(
    reference_bgr: np.ndarray,
    user_bgr: np.ndarray,
    *,
    user_alpha: float = DEFAULT_USER_OVERLAY_ALPHA,
) -> np.ndarray:
    """
    Stack user on reference: ``out = ref * (1 - α) + user * α`` after resizing user to ref.
    """
    if reference_bgr is None or user_bgr is None:
        raise ValueError("Missing frame(s) for overlay.")
    a = float(np.clip(user_alpha, 0.0, 1.0))
    ref = reference_bgr.astype(np.float32)
    usr = user_bgr
    if usr.shape[0] != ref.shape[0] or usr.shape[1] != ref.shape[1]:
        usr = cv2.resize(usr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    usr = usr.astype(np.float32)
    out = ref * (1.0 - a) + usr * a
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def read_bgr_at_index(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    """Seek and read one BGR frame; returns ``None`` on failure."""
    if cap is None or not cap.isOpened():
        return None
    idx = max(0, int(frame_index))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, bgr = cap.read()
    if not ret or bgr is None:
        return None
    return bgr


def overlay_pair_from_caps(
    cap_ref: cv2.VideoCapture,
    cap_user: cv2.VideoCapture,
    ref_index: int,
    user_index: int,
    *,
    user_alpha: float = DEFAULT_USER_OVERLAY_ALPHA,
    flip_reference_horizontal: bool = False,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Read both frames and blend. Returns ``(image_or_none, both_ok)``.

    When ``flip_reference_horizontal`` is True, the reference frame is mirrored so the
    overlay matches practice preview when following a mirrored reference.
    """
    r = read_bgr_at_index(cap_ref, ref_index)
    u = read_bgr_at_index(cap_user, user_index)
    if r is None or u is None:
        return None, False
    if flip_reference_horizontal:
        r = cv2.flip(r, 1)
    return blend_overlay_bgr(r, u, user_alpha=user_alpha), True
