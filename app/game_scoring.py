"""Frontend game-style scoring built on top of raw pose similarity."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np

from app.comparison_types import ComparisonResult


@dataclass(frozen=True)
class HitBand:
    name: str
    threshold: float
    points: int


@dataclass(frozen=True)
class GameScore:
    points: int
    max_points: int
    accuracy: float
    rank: str
    hit_counts: dict[str, int]
    max_combo: int
    total_windows: int
    raw_similarity: float

    @property
    def hit_summary(self) -> str:
        parts = []
        for band in HIT_BANDS:
            count = self.hit_counts.get(band.name, 0)
            if count:
                parts.append(f"{count} {band.name}")
        misses = self.hit_counts.get("Miss", 0)
        if misses:
            parts.append(f"{misses} Miss")
        return " · ".join(parts) if parts else "No hits"


HIT_BANDS: tuple[HitBand, ...] = (
    HitBand("Marvelous", 95.0, 10000),
    HitBand("Perfect", 90.0, 8500),
    HitBand("Great", 80.0, 6500),
    HitBand("Good", 70.0, 4500),
    HitBand("OK", 55.0, 2000),
)
MAX_HIT_POINTS = HIT_BANDS[0].points


def _rank_for_accuracy(accuracy: float) -> str:
    if accuracy >= 90.0:
        return "S+"
    if accuracy >= 80.0:
        return "S"
    if accuracy >= 68.0:
        return "A"
    if accuracy >= 50.0:
        return "B"
    if accuracy >= 35.0:
        return "C"
    if accuracy >= 20.0:
        return "D"
    return "Practice"


def _window_means(values: np.ndarray, *, target_windows: int = 32) -> list[float]:
    if values.size == 0:
        return []
    window = max(1, int(ceil(values.size / max(1, target_windows))))
    means: list[float] = []
    for start in range(0, values.size, window):
        chunk = values[start : start + window]
        if chunk.size:
            means.append(float(np.nanmean(chunk)))
    return means


def _judge(score: float) -> tuple[str, int]:
    for band in HIT_BANDS:
        if score >= band.threshold:
            return band.name, band.points
    return "Miss", 0


def _safe_scores(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=0.0)
    return np.clip(arr, 0.0, 100.0)


def build_game_score(result: ComparisonResult) -> GameScore:
    raw = max(0.0, min(100.0, float(result.overall_score)))
    per_frame = _safe_scores(result.per_frame_similarity)
    if raw <= 0.0 or per_frame.size == 0:
        hit_counts = {band.name: 0 for band in HIT_BANDS}
        hit_counts["Miss"] = 1
        return GameScore(
            points=0,
            max_points=MAX_HIT_POINTS,
            accuracy=0.0,
            rank="Practice",
            hit_counts=hit_counts,
            max_combo=0,
            total_windows=1,
            raw_similarity=raw,
        )

    # Blend each moment with the final capped score. This keeps the display
    # game-like while respecting global gates such as missing full-body tracking.
    window_scores = [
        max(0.0, min(100.0, 0.65 * score + 0.35 * raw))
        for score in _window_means(per_frame)
    ]
    if not window_scores:
        window_scores = [raw]

    hit_counts = {band.name: 0 for band in HIT_BANDS}
    hit_counts["Miss"] = 0
    points = 0
    combo = 0
    max_combo = 0
    for score in window_scores:
        label, award = _judge(score)
        hit_counts[label] = hit_counts.get(label, 0) + 1
        points += award
        if award > 0:
            combo += 1
            max_combo = max(max_combo, combo)
        else:
            combo = 0

    total_windows = len(window_scores)
    max_points = total_windows * MAX_HIT_POINTS
    point_accuracy = (points / max_points * 100.0) if max_points else 0.0
    # The hit windows drive points, but rank should not make a visibly decent
    # attempt feel like a total failure just because the pose model is strict.
    accuracy = max(point_accuracy, raw * 0.85)
    return GameScore(
        points=int(points),
        max_points=int(max_points),
        accuracy=accuracy,
        rank=_rank_for_accuracy(accuracy),
        hit_counts=hit_counts,
        max_combo=max_combo,
        total_windows=total_windows,
        raw_similarity=raw,
    )
