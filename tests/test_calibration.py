"""Calibration/preflight scan helpers."""

from __future__ import annotations

from app.calibration import (
    CalibrationReport,
    _build_recommendations,
    _grade_from_score,
    _has_full_body,
    _people_separated,
    _sample_indices,
)


class _Landmark:
    def __init__(self, x: float = 0.5, y: float = 0.5, score: float = 0.95) -> None:
        self.x = x
        self.y = y
        self.visibility = score
        self.presence = score


def _person(cx: float) -> list[_Landmark]:
    lms = [_Landmark(cx, 0.5, 0.95) for _ in range(33)]
    lms[11] = _Landmark(cx - 0.05, 0.25)
    lms[12] = _Landmark(cx + 0.05, 0.25)
    lms[23] = _Landmark(cx - 0.04, 0.5)
    lms[24] = _Landmark(cx + 0.04, 0.5)
    lms[25] = _Landmark(cx - 0.04, 0.7)
    lms[26] = _Landmark(cx + 0.04, 0.7)
    lms[27] = _Landmark(cx - 0.04, 0.9)
    lms[28] = _Landmark(cx + 0.04, 0.9)
    return lms


def test_sample_indices_are_bounded_and_evenly_spread() -> None:
    idx = _sample_indices(100, max_samples=5)
    assert idx == [0, 25, 50, 74, 99]


def test_grade_from_score() -> None:
    assert _grade_from_score(90) == "Great"
    assert _grade_from_score(70) == "Usable"
    assert _grade_from_score(50) == "Risky"


def test_full_body_requires_lower_body() -> None:
    lms = _person(0.5)
    assert _has_full_body(lms)
    lms[28].visibility = 0.1
    lms[27].visibility = 0.1
    assert not _has_full_body(lms)


def test_people_separated_for_multiplayer_calibration() -> None:
    assert _people_separated([_person(0.3), _person(0.7)], 2)
    assert not _people_separated([_person(0.45), _person(0.5)], 2)


def test_report_marks_usable_as_ready() -> None:
    report = CalibrationReport(
        source_path="/tmp/x.mp4",
        expected_people=1,
        samples_analyzed=10,
        grade="Usable",
        score=75,
        any_pose_coverage=0.9,
        expected_people_coverage=0.9,
        torso_coverage=0.9,
        full_body_coverage=0.75,
        arms_coverage=0.8,
        legs_coverage=0.7,
        centered_coverage=0.9,
        separation_coverage=1.0,
        median_people_detected=1,
        max_people_detected=1,
    )
    assert report.is_ready
    assert "Usable calibration" in report.one_line()


def test_multiplayer_recommendation_mentions_spacing() -> None:
    recs = _build_recommendations(
        expected_people=4,
        expected_people_coverage=0.95,
        torso_coverage=0.95,
        full_body_coverage=0.9,
        arms_coverage=0.9,
        legs_coverage=0.9,
        centered_coverage=0.9,
        separation_coverage=0.2,
    )
    assert any("Space dancers" in r for r in recs)
