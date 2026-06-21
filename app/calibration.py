"""Video preflight scan for pose trackability and multiplayer readiness."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2

from app import pose_utils
from app import video_utils

_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28

_VISIBLE_GATE = 0.55
_MAX_SCAN_SAMPLES = 72
_CENTER_MIN_X = 0.18
_CENTER_MAX_X = 0.82
_CENTER_MIN_Y = 0.16
_CENTER_MAX_Y = 0.86
_MIN_MULTI_SEPARATION = 0.11


@dataclass
class CalibrationReport:
    """Summary of whether a video is ready for pose extraction/scoring."""

    source_path: str
    expected_people: int
    samples_analyzed: int
    grade: str
    score: float
    any_pose_coverage: float
    expected_people_coverage: float
    torso_coverage: float
    full_body_coverage: float
    arms_coverage: float
    legs_coverage: float
    centered_coverage: float
    separation_coverage: float
    median_people_detected: int
    max_people_detected: int
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.grade in {"Great", "Usable"}

    @property
    def is_risky(self) -> bool:
        return not self.is_ready

    def one_line(self) -> str:
        return (
            f"{self.grade} calibration ({self.score:.0f}/100): "
            f"pose {self.any_pose_coverage:.0%}, full body {self.full_body_coverage:.0%}, "
            f"legs {self.legs_coverage:.0%}, people {self.expected_people_coverage:.0%}"
        )

    def details_text(self) -> str:
        lines = [
            self.one_line(),
            (
                f"Samples: {self.samples_analyzed}  |  Expected dancers: {self.expected_people}  |  "
                f"Median detected: {self.median_people_detected}  |  Max detected: {self.max_people_detected}"
            ),
            (
                f"Torso {self.torso_coverage:.0%}  |  Arms {self.arms_coverage:.0%}  |  "
                f"Legs {self.legs_coverage:.0%}  |  Centered {self.centered_coverage:.0%}  |  "
                f"Separated {self.separation_coverage:.0%}"
            ),
        ]
        if self.recommendations:
            lines.append("Suggestions: " + " ".join(self.recommendations))
        return "\n".join(lines)


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, count / total))


def _is_visible(landmarks: Sequence[object], idx: int) -> bool:
    if idx >= len(landmarks):
        return False
    lm = landmarks[idx]
    if pose_utils.landmark_reliability(lm) < _VISIBLE_GATE:
        return False
    return math.isfinite(float(lm.x)) and math.isfinite(float(lm.y))


def _visible_count(landmarks: Sequence[object], indices: Sequence[int]) -> int:
    return sum(1 for idx in indices if _is_visible(landmarks, idx))


def _person_center(landmarks: Sequence[object]) -> Optional[Tuple[float, float]]:
    return pose_utils.body_center_normalized(list(landmarks))


def _has_torso(landmarks: Sequence[object]) -> bool:
    shoulders = _visible_count(landmarks, (_LEFT_SHOULDER, _RIGHT_SHOULDER))
    hips = _visible_count(landmarks, (_LEFT_HIP, _RIGHT_HIP))
    return shoulders == 2 and hips >= 1


def _has_full_body(landmarks: Sequence[object]) -> bool:
    return _has_torso(landmarks) and _visible_count(
        landmarks,
        (_LEFT_KNEE, _RIGHT_KNEE, _LEFT_ANKLE, _RIGHT_ANKLE),
    ) >= 3


def _is_centered(landmarks: Sequence[object]) -> bool:
    center = _person_center(landmarks)
    if center is None:
        return False
    return (
        _CENTER_MIN_X <= center[0] <= _CENTER_MAX_X
        and _CENTER_MIN_Y <= center[1] <= _CENTER_MAX_Y
    )


def _select_people(
    persons: List[List[object]],
    expected_people: int,
    width: int,
    height: int,
    previous_center: Optional[Tuple[float, float]],
) -> tuple[List[List[object]], Optional[Tuple[float, float]]]:
    if not persons:
        return [], previous_center
    if expected_people <= 1:
        idx = pose_utils.select_center_person_index(
            persons,
            width,
            height,
            previous_center,
        )
        if idx is None:
            return [], previous_center
        return [persons[idx]], _person_center(persons[idx])

    with_centers: list[tuple[float, List[object]]] = []
    for person in persons:
        center = _person_center(person)
        if center is not None:
            with_centers.append((center[0], person))
    with_centers.sort(key=lambda item: item[0])
    return [person for _, person in with_centers[:expected_people]], previous_center


def _people_separated(persons: Sequence[Sequence[object]], expected_people: int) -> bool:
    if expected_people <= 1:
        return True
    if len(persons) < expected_people:
        return False
    centers = [_person_center(p) for p in persons[:expected_people]]
    centers = [c for c in centers if c is not None]
    if len(centers) < expected_people:
        return False
    xs = sorted(c[0] for c in centers)
    return all((b - a) >= _MIN_MULTI_SEPARATION for a, b in zip(xs, xs[1:]))


def _sample_indices(frame_count: int, max_samples: int = _MAX_SCAN_SAMPLES) -> List[int]:
    if frame_count <= 0:
        return list(range(max_samples))
    n = max(1, min(max_samples, frame_count))
    if n == 1:
        return [0]
    return sorted({round(i * (frame_count - 1) / (n - 1)) for i in range(n)})


def _median_int(values: Sequence[int]) -> int:
    if not values:
        return 0
    s = sorted(values)
    return int(s[len(s) // 2])


def _grade_from_score(score: float) -> str:
    if score >= 85.0:
        return "Great"
    if score >= 68.0:
        return "Usable"
    return "Risky"


def _build_recommendations(
    *,
    expected_people: int,
    expected_people_coverage: float,
    torso_coverage: float,
    full_body_coverage: float,
    arms_coverage: float,
    legs_coverage: float,
    centered_coverage: float,
    separation_coverage: float,
) -> List[str]:
    recs: List[str] = []
    if expected_people_coverage < 0.8:
        recs.append("Make sure every dancer is visible before the dance starts.")
    if full_body_coverage < 0.7:
        recs.append("Back up or rotate the camera so head, hands, knees, and feet stay in frame.")
    if torso_coverage < 0.85:
        recs.append("Improve lighting and keep the torso unobstructed.")
    if legs_coverage < 0.65:
        recs.append("Keep feet and knees visible; avoid pants/backgrounds that blend together.")
    if arms_coverage < 0.65:
        recs.append("Keep hands and elbows away from the body during the countdown scan.")
    if centered_coverage < 0.75:
        recs.append("Center the dancer in the frame.")
    if expected_people > 1 and separation_coverage < 0.75:
        recs.append("Space dancers farther apart so IDs do not swap.")
    if not recs:
        recs.append("Looks ready for pose extraction.")
    return recs


def scan_video_calibration(
    path: str,
    *,
    expected_people: int = 1,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    max_samples: int = _MAX_SCAN_SAMPLES,
) -> CalibrationReport:
    """Sample a video and estimate whether pose tracking/scoring will be reliable."""
    expected_people = max(1, min(4, int(expected_people)))
    cap = video_utils.open_capture(path)
    landmarker = None
    try:
        meta = video_utils.read_metadata(cap, path)
        fps, _ = video_utils.resolve_writer_fps(cap, meta)
        frame_dt_ms = max(1, round(1000.0 / fps))
        indices = _sample_indices(meta.frame_count, max_samples=max_samples)
        landmarker = pose_utils.create_pose_landmarker(
            for_video=True,
            detection_mode=pose_utils.DetectionMode.ALL_PEOPLE,
        )

        previous_center: Optional[Tuple[float, float]] = None
        samples = 0
        any_pose = 0
        expected_visible = 0
        torso_ok = 0
        full_body_ok = 0
        arms_ok = 0
        legs_ok = 0
        centered_ok = 0
        separated_ok = 0
        people_counts: list[int] = []

        for step, frame_index in enumerate(indices):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Cancelled.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            timestamp_ms = int(round(frame_index * frame_dt_ms))
            persons = pose_utils.extract_pose_persons(
                landmarker,
                frame,
                timestamp_ms=timestamp_ms,
            )
            people_counts.append(len(persons))
            selected, previous_center = _select_people(
                persons,
                expected_people,
                meta.width,
                meta.height,
                previous_center,
            )
            samples += 1
            if persons:
                any_pose += 1
            if len(persons) >= expected_people:
                expected_visible += 1
            if _people_separated(selected, expected_people):
                separated_ok += 1
            if not selected:
                continue

            per_person_torso = [_has_torso(p) for p in selected]
            per_person_full = [_has_full_body(p) for p in selected]
            per_person_arms = [
                _visible_count(p, (_LEFT_ELBOW, _LEFT_WRIST, _RIGHT_ELBOW, _RIGHT_WRIST)) >= 3
                for p in selected
            ]
            per_person_legs = [
                _visible_count(p, (_LEFT_KNEE, _LEFT_ANKLE, _RIGHT_KNEE, _RIGHT_ANKLE)) >= 3
                for p in selected
            ]
            per_person_center = [_is_centered(p) for p in selected]

            if len(selected) >= expected_people and all(per_person_torso[:expected_people]):
                torso_ok += 1
            if len(selected) >= expected_people and all(per_person_full[:expected_people]):
                full_body_ok += 1
            if len(selected) >= expected_people and all(per_person_arms[:expected_people]):
                arms_ok += 1
            if len(selected) >= expected_people and all(per_person_legs[:expected_people]):
                legs_ok += 1
            if len(selected) >= expected_people and all(per_person_center[:expected_people]):
                centered_ok += 1

            if progress_cb and (step % 4 == 0 or step == len(indices) - 1):
                pct = min(100, int(100 * (step + 1) / max(1, len(indices))))
                progress_cb(pct, f"Calibration samples: {step + 1}/{len(indices)}")

        expected_people_coverage = _ratio(expected_visible, samples)
        torso_coverage = _ratio(torso_ok, samples)
        full_body_coverage = _ratio(full_body_ok, samples)
        arms_coverage = _ratio(arms_ok, samples)
        legs_coverage = _ratio(legs_ok, samples)
        centered_coverage = _ratio(centered_ok, samples)
        separation_coverage = _ratio(separated_ok, samples)
        any_pose_coverage = _ratio(any_pose, samples)
        multi_weight = separation_coverage if expected_people > 1 else 1.0
        score = 100.0 * (
            0.18 * any_pose_coverage
            + 0.18 * expected_people_coverage
            + 0.18 * torso_coverage
            + 0.20 * full_body_coverage
            + 0.09 * arms_coverage
            + 0.09 * legs_coverage
            + 0.05 * centered_coverage
            + 0.03 * multi_weight
        )
        grade = _grade_from_score(score)
        recs = _build_recommendations(
            expected_people=expected_people,
            expected_people_coverage=expected_people_coverage,
            torso_coverage=torso_coverage,
            full_body_coverage=full_body_coverage,
            arms_coverage=arms_coverage,
            legs_coverage=legs_coverage,
            centered_coverage=centered_coverage,
            separation_coverage=separation_coverage,
        )

        return CalibrationReport(
            source_path=str(path),
            expected_people=expected_people,
            samples_analyzed=samples,
            grade=grade,
            score=score,
            any_pose_coverage=any_pose_coverage,
            expected_people_coverage=expected_people_coverage,
            torso_coverage=torso_coverage,
            full_body_coverage=full_body_coverage,
            arms_coverage=arms_coverage,
            legs_coverage=legs_coverage,
            centered_coverage=centered_coverage,
            separation_coverage=separation_coverage,
            median_people_detected=_median_int(people_counts),
            max_people_detected=max(people_counts) if people_counts else 0,
            recommendations=recs,
        )
    finally:
        if landmarker is not None:
            landmarker.close()
        cap.release()
