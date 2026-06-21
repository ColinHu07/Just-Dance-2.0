"""Temporal drawing guards for pose overlays."""

from __future__ import annotations

from app.pose_utils import (
    PoseDrawTemporalState,
    _POSE_INITIAL_LOCK_FRAMES,
    _compact_lower_leg_segments,
    _decross_torso_cache_for_drawing,
    _leg_suppression_flags,
    _midpoint_from_cache,
    _needs_compact_body_draw,
    _pose_is_confident_enough,
)


class _Landmark:
    def __init__(self, x: float = 0.5, y: float = 0.5, score: float = 0.95) -> None:
        self.x = x
        self.y = y
        self.visibility = score
        self.presence = score


def _body_landmarks(score: float = 0.95) -> list[_Landmark]:
    landmarks = [_Landmark(score=score) for _ in range(33)]
    landmarks[11] = _Landmark(0.42, 0.28, score)
    landmarks[12] = _Landmark(0.58, 0.28, score)
    landmarks[23] = _Landmark(0.45, 0.55, score)
    landmarks[24] = _Landmark(0.55, 0.55, score)
    landmarks[25] = _Landmark(0.44, 0.72, score)
    landmarks[26] = _Landmark(0.56, 0.72, score)
    landmarks[27] = _Landmark(0.43, 0.9, score)
    landmarks[28] = _Landmark(0.57, 0.9, score)
    return landmarks


def test_pose_startup_gate_waits_before_first_draw() -> None:
    state = PoseDrawTemporalState()

    for _ in range(_POSE_INITIAL_LOCK_FRAMES - 1):
        assert not state.ready_to_draw_pose(0)

    assert state.ready_to_draw_pose(0)
    assert state.ready_to_draw_pose(0)


def test_pose_startup_gate_resets_if_unlocked_track_disappears() -> None:
    state = PoseDrawTemporalState()

    for _ in range(3):
        assert not state.ready_to_draw_pose(0)

    state.begin_frame(set())

    assert not state.ready_to_draw_pose(0)


def test_pose_frame_confidence_gate_accepts_stable_body() -> None:
    assert _pose_is_confident_enough(_body_landmarks())


def test_pose_frame_confidence_gate_rejects_low_confidence_body() -> None:
    assert not _pose_is_confident_enough(_body_landmarks(score=0.4))


def test_pose_frame_confidence_gate_rejects_sparse_core() -> None:
    landmarks = _body_landmarks()
    for idx in (23, 24, 25, 26, 27, 28):
        landmarks[idx].visibility = 0.2
        landmarks[idx].presence = 0.2

    assert not _pose_is_confident_enough(landmarks)


def test_profile_leg_suppression_keeps_partly_visible_leg() -> None:
    landmarks = _body_landmarks()
    landmarks[11].x = 0.5
    landmarks[12].x = 0.53
    landmarks[26].visibility = 0.7
    landmarks[26].presence = 0.7
    landmarks[28].visibility = 0.25
    landmarks[28].presence = 0.25

    assert _leg_suppression_flags(landmarks) == (False, False)


def test_profile_leg_suppression_drops_fully_lost_leg() -> None:
    landmarks = _body_landmarks()
    landmarks[11].x = 0.5
    landmarks[12].x = 0.53
    landmarks[26].visibility = 0.2
    landmarks[26].presence = 0.2
    landmarks[28].visibility = 0.2
    landmarks[28].presence = 0.2

    assert _leg_suppression_flags(landmarks) == (False, True)


def test_torso_decross_swaps_lower_body_for_drawing_only() -> None:
    cache = {
        11: ((220, 100), "solid"),
        12: ((120, 100), "solid"),
        23: ((130, 250), "solid"),
        24: ((230, 250), "solid"),
        25: ((125, 360), "solid"),
        26: ((235, 360), "solid"),
        27: ((120, 470), "solid"),
        28: ((240, 470), "solid"),
    }

    corrected = _decross_torso_cache_for_drawing(cache, frame_width=360)

    assert corrected[23][0] == (230, 250)
    assert corrected[24][0] == (130, 250)
    assert corrected[25][0] == (235, 360)
    assert corrected[26][0] == (125, 360)
    assert cache[23][0] == (130, 250)


def test_compact_body_draw_triggers_when_torso_pairs_overlap() -> None:
    landmarks = _body_landmarks()
    landmarks[11].x = 0.5
    landmarks[12].x = 0.55

    assert _needs_compact_body_draw(landmarks)


def test_midpoint_from_cache_preserves_held_style() -> None:
    cache = {
        11: ((10, 20), "solid"),
        12: ((30, 40), "held"),
    }

    assert _midpoint_from_cache(cache, 11, 12) == ((20, 30), "held")


def test_compact_leg_draw_skips_rotating_upper_leg_fan() -> None:
    assert _compact_lower_leg_segments(
        [
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
        ]
    ) == [(25, 27), (26, 28)]
