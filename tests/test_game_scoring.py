"""Game-style score conversion for frontend results."""

from __future__ import annotations

import numpy as np

from app.comparison_types import ComparisonResult, ScoreBreakdown
from app.game_scoring import build_game_score


def _result(overall: float, frames: list[float]) -> ComparisonResult:
    bd = ScoreBreakdown(
        overall=overall,
        timing=overall,
        arms=overall,
        legs=overall,
        torso_posture=overall,
        joint_angles=overall,
        limb_directions=overall,
        relative_distances=overall,
        movement=overall,
    )
    return ComparisonResult(
        overall_score=overall,
        breakdown=bd,
        explanation_lines=[],
        per_frame_similarity=np.asarray(frames, dtype=np.float64),
        alignment_path=np.zeros((len(frames), 2), dtype=np.int64),
        dtw_total_cost=0.0,
        dtw_mean_cost=0.0,
        timing_mean_abs_lag_frames=0.0,
    )


def test_zero_similarity_gets_no_game_points() -> None:
    score = build_game_score(_result(0.0, []))

    assert score.points == 0
    assert score.rank == "Practice"
    assert score.hit_counts["Miss"] == 1


def test_marvelous_windows_award_full_points() -> None:
    score = build_game_score(_result(98.0, [98.0] * 64))

    assert score.points == score.max_points
    assert score.rank == "S+"
    assert score.hit_counts["Marvelous"] > 0
    assert score.hit_counts["Miss"] == 0


def test_raw_cap_keeps_bad_run_from_over_scoring() -> None:
    score = build_game_score(_result(32.0, [80.0] * 64))

    assert score.points < score.max_points * 0.5
    assert score.rank in {"Practice", "D"}


def test_decent_strict_similarity_gets_playable_rank() -> None:
    score = build_game_score(_result(61.0, [65.0] * 64))

    assert score.points > 0
    assert score.rank == "B"
