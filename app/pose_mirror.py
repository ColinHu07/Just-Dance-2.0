"""Mirror MediaPipe-style normalized pose data for practice/scoring alignment."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.comparison_types import PoseFrame, PoseSequence

# MediaPipe Pose (33 landmarks): left/right pairs to swap before negating x.
# Nose (0) is unpaired; after swaps, x is negated for all rows.
_MEDIAPIPE_LR_SWAP_PAIRS: Tuple[Tuple[int, int], ...] = (
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
)


def _swap_rows_xy_rel(
    xy: np.ndarray,
    rel: np.ndarray,
    a: int,
    b: int,
) -> None:
    if a >= xy.shape[0] or b >= xy.shape[0]:
        return
    xy[[a, b], :] = xy[[b, a], :]
    if rel.shape[0] > max(a, b):
        rel[[a, b]] = rel[[b, a]]


def mirror_joints_norm_xy(
    joints_norm_xy: np.ndarray,
    reliability: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return mirrored copies: swap left/right semantics, then negate x (image horizontal flip).

    Operates on torso-normalized ``(33, 2)`` coordinates and parallel reliability vector.
    """
    xy = np.array(joints_norm_xy, dtype=np.float64, copy=True)
    rel = np.array(reliability, dtype=np.float64, copy=True)
    for a, b in _MEDIAPIPE_LR_SWAP_PAIRS:
        _swap_rows_xy_rel(xy, rel, a, b)
    xy[:, 0] *= -1.0
    return xy, rel


def mirror_pose_frame(frame: PoseFrame) -> PoseFrame:
    """Return a new frame with mirrored geometry; ``landmarks_raw`` is cleared."""
    xy, rel = mirror_joints_norm_xy(frame.joints_norm_xy, frame.reliability)
    return PoseFrame(
        frame_index=frame.frame_index,
        time_sec=frame.time_sec,
        image_width=frame.image_width,
        image_height=frame.image_height,
        landmarks_raw=None,
        joints_norm_xy=xy,
        reliability=rel,
    )


def mirror_pose_sequence(seq: PoseSequence) -> PoseSequence:
    """Return a new sequence with each frame mirrored (for scoring vs unmirrored user video)."""
    frames = [mirror_pose_frame(f) for f in seq.frames]
    return PoseSequence(
        source_path=seq.source_path,
        fps=seq.fps,
        frames=frames,
        video_width=seq.video_width,
        video_height=seq.video_height,
    )
