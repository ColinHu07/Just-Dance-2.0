"""Sanity checks for geometry-based dance comparison (no video I/O)."""

from __future__ import annotations

import numpy as np
import pytest

from app import comparison_view
from app.comparison_types import PoseFrame, PoseSequence
from app.scoring import compare_pose_sequences
from app.sequence_features import build_frame_features


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


def _seq(name: str, n: int, *, warp_right_arm: bool = False, stride: int = 1) -> PoseSequence:
    rel = np.ones(33, dtype=np.float64) * 0.95
    frames: list[PoseFrame] = []
    t = 0
    for k in range(n):
        xy = _base_norm_xy().copy()
        if warp_right_arm:
            xy[14] += (0.5, 0.3)
            xy[16] += (0.7, 0.2)
        frames.append(
            PoseFrame(
                frame_index=t,
                time_sec=t / 30.0,
                image_width=640,
                image_height=480,
                landmarks_raw=None,
                joints_norm_xy=xy,
                reliability=rel,
            )
        )
        t += stride
    return PoseSequence(name, 30.0, frames, 640, 480)


def test_identical_sequences_score_near_perfect() -> None:
    a = _seq("a", 18)
    b = _seq("b", 18)
    r = compare_pose_sequences(a, b)
    assert r.overall_score >= 99.0
    assert r.breakdown.arms >= 95.0


def test_feedback_uses_supportive_coach_language() -> None:
    a = _seq("a", 18)
    r = compare_pose_sequences(a, a)
    text = " ".join(r.explanation_lines)
    assert "bright spot" in text
    assert "Timing felt locked in" in text


def test_warped_arm_scores_lower_than_identical() -> None:
    ref = _seq("ref", 18)
    bad = _seq("bad", 18, warp_right_arm=True)
    good = compare_pose_sequences(ref, ref)
    worse = compare_pose_sequences(ref, bad)
    assert worse.overall_score < good.overall_score - 2.0


def test_temporal_stretch_still_high_similarity() -> None:
    """User clip with 2× frames (repeated motion) should align via DTW."""
    ref = _seq("ref", 15, stride=1)
    slow = _seq("slow", 29, stride=1)  # ~2× length, same pose each step
    r = compare_pose_sequences(ref, slow)
    assert r.overall_score >= 92.0
    assert r.breakdown.timing >= 70.0


def test_empty_sequence_returns_zero() -> None:
    empty = PoseSequence("e", 30.0, [], 640, 480)
    full = _seq("f", 5)
    r = compare_pose_sequences(empty, full)
    assert r.overall_score == 0.0


def test_subsample_alignment_path_shortens_long_paths() -> None:
    path = np.arange(2000, dtype=np.int64).reshape(-1, 2)
    sub = comparison_view.subsample_alignment_path(path, max_steps=100)
    assert sub.shape[0] <= 100
    assert sub.shape[1] == 2


def test_blend_overlay_preserves_shape() -> None:
    ref = np.zeros((60, 80, 3), dtype=np.uint8)
    ref[:, :] = (255, 0, 0)
    usr = np.zeros((30, 40, 3), dtype=np.uint8)
    usr[:, :] = (0, 255, 0)
    out = comparison_view.blend_overlay_bgr(ref, usr, user_alpha=0.4)
    assert out.shape == ref.shape


def test_low_confidence_reference_joint_features_are_inactive() -> None:
    xy = _base_norm_xy()
    rel = np.ones(33, dtype=np.float64) * 0.95
    rel[15] = 0.4
    frame = PoseFrame(0, 0.0, 640, 480, None, xy, rel)

    feats = build_frame_features(frame)

    wrist_dependent_slots = [0, 13, 14, 33, 35, 39]
    assert np.all(feats.dim_weight[wrist_dependent_slots] == 0.0)
    assert np.isnan(feats.vector[0])


def test_user_extra_joint_gets_visibility_penalty_when_reference_lacks_it() -> None:
    rel_ref = np.ones(33, dtype=np.float64) * 0.95
    rel_ref[15] = 0.0
    rel_user = np.ones(33, dtype=np.float64) * 0.95

    ref_frames: list[PoseFrame] = []
    user_frames: list[PoseFrame] = []
    for k in range(12):
        ref_xy = _base_norm_xy()
        user_xy = _base_norm_xy()
        user_xy[15] += (4.0, -3.0)
        ref_frames.append(
            PoseFrame(k, k / 30.0, 640, 480, None, ref_xy, rel_ref.copy())
        )
        user_frames.append(
            PoseFrame(k, k / 30.0, 640, 480, None, user_xy, rel_user.copy())
        )

    ref = PoseSequence("ref", 30.0, ref_frames, 640, 480)
    user = PoseSequence("user", 30.0, user_frames, 640, 480)
    result = compare_pose_sequences(ref, user)

    assert 95.0 <= result.overall_score < 99.0
    assert any("extra visible limbs" in line for line in result.explanation_lines)


def test_user_extra_visible_leg_penalizes_leg_breakdown() -> None:
    rel_ref = np.ones(33, dtype=np.float64) * 0.95
    rel_ref[26] = 0.0
    rel_ref[28] = 0.0
    rel_user = np.ones(33, dtype=np.float64) * 0.95

    ref = PoseSequence(
        "ref",
        30.0,
        [
            PoseFrame(k, k / 30.0, 640, 480, None, _base_norm_xy(), rel_ref.copy())
            for k in range(12)
        ],
        640,
        480,
    )
    user = PoseSequence(
        "user",
        30.0,
        [
            PoseFrame(k, k / 30.0, 640, 480, None, _base_norm_xy(), rel_user.copy())
            for k in range(12)
        ],
        640,
        480,
    )

    result = compare_pose_sequences(ref, user)

    assert result.overall_score < 98.0
    assert result.breakdown.legs < 98.0
