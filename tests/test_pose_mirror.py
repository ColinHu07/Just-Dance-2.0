"""Tests for mirrored pose geometry (swap + flip) used in library scoring."""

from __future__ import annotations

import numpy as np

from app.comparison_types import PoseFrame, PoseSequence
from app.pose_mirror import mirror_pose_frame, mirror_pose_sequence
from app.scoring import compare_pose_sequences


def _base_norm_xy() -> np.ndarray:
    xy = np.zeros((33, 2), dtype=np.float64)
    xy[11] = (-0.5, -0.2)
    xy[12] = (0.5, -0.2)
    xy[23] = (-0.2, 0.3)
    xy[24] = (0.2, 0.3)
    xy[13] = (-0.9, -0.1)
    xy[15] = (-1.2, 0.1)
    xy[14] = (0.9, -0.1)
    xy[16] = (1.2, 0.1)
    xy[25] = (-0.25, 0.9)
    xy[27] = (-0.25, 1.3)
    xy[26] = (0.25, 0.9)
    xy[28] = (0.25, 1.3)
    return xy


def _frame(xy: np.ndarray) -> PoseFrame:
    rel = np.ones(33, dtype=np.float64) * 0.95
    return PoseFrame(
        frame_index=0,
        time_sec=0.0,
        image_width=640,
        image_height=480,
        landmarks_raw=None,
        joints_norm_xy=xy,
        reliability=rel,
    )


def test_double_mirror_restores_coordinates() -> None:
    xy = _base_norm_xy()
    f = _frame(xy)
    once = mirror_pose_frame(f)
    twice = mirror_pose_frame(once)
    np.testing.assert_allclose(twice.joints_norm_xy, f.joints_norm_xy, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(twice.reliability, f.reliability, rtol=1e-9, atol=1e-9)


def test_mirrored_reference_matches_mirrored_user_in_scoring() -> None:
    """mirrored(ref) vs mirror(ref) should match as well as ref vs ref."""
    rel = np.ones(33, dtype=np.float64) * 0.95
    frames = []
    for k in range(16):
        xy = _base_norm_xy().copy()
        xy[15] += (0.02 * k, 0.01 * k)
        frames.append(
            PoseFrame(
                frame_index=k,
                time_sec=k / 30.0,
                image_width=640,
                image_height=480,
                landmarks_raw=None,
                joints_norm_xy=xy,
                reliability=rel,
            )
        )
    ref = PoseSequence("ref", 30.0, frames, 640, 480)
    mir = mirror_pose_sequence(ref)
    baseline = compare_pose_sequences(ref, ref)
    mirrored_pair = compare_pose_sequences(mir, mir)
    assert mirrored_pair.overall_score >= 98.0
    assert abs(mirrored_pair.overall_score - baseline.overall_score) < 2.0


def test_mirrored_ref_vs_same_mirrored_motion_scores_high() -> None:
    ref = PoseSequence(
        "r",
        30.0,
        [_frame(_base_norm_xy()) for _ in range(12)],
        640,
        480,
    )
    mir = mirror_pose_sequence(ref)
    r = compare_pose_sequences(mir, mir)
    assert r.overall_score >= 99.0
