"""Structured types for pose sequences, features, and comparison results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class JointObservation:
    """One body landmark after normalization (origin at hip/shoulder center, isotropic scale)."""

    index: int
    x: float
    y: float
    z: float
    reliability: float


@dataclass
class PoseFrame:
    """Single time step: raw detector landmarks plus normalized geometry for scoring."""

    frame_index: int
    time_sec: float
    image_width: int
    image_height: int
    landmarks_raw: Optional[List[Any]]
    """Normalized-plane joints (33×2) in torso-centered, scale-normalized 2D; NaN if unknown."""
    joints_norm_xy: np.ndarray
    """Per-landmark reliability in ``[0, 1]`` aligned with MediaPipe indices."""
    reliability: np.ndarray


@dataclass
class PoseSequence:
    """Full video track for one clip."""

    source_path: str
    fps: float
    frames: List[PoseFrame]
    video_width: int
    video_height: int


@dataclass
class FrameFeatures:
    """Per-frame geometry features for DTW and grouped scoring."""

    frame_index: int
    time_sec: float
    # Flat vector for DTW (NaN = inactive dimension).
    vector: np.ndarray
    # Same length as vector: min(ref,user) reliability product per dimension.
    dim_weight: np.ndarray
    # Group masks into vector slices (name -> bool mask).
    group_masks: dict[str, np.ndarray] = field(default_factory=dict)
    # Raw per-landmark source confidence, used for visibility/silhouette mismatch checks.
    joint_reliability: Optional[np.ndarray] = None


@dataclass
class ScoreBreakdown:
    """0–100 style subscores (may be NaN if no valid data in that group)."""

    overall: float
    timing: float
    arms: float
    legs: float
    torso_posture: float
    joint_angles: float
    limb_directions: float
    relative_distances: float
    movement: float = 100.0


@dataclass
class ComparisonResult:
    """Output of reference vs user comparison after DTW alignment."""

    overall_score: float
    breakdown: ScoreBreakdown
    explanation_lines: List[str]
    per_frame_similarity: np.ndarray
    """Aligned pairs ``(ref_index, user_index)`` along the DTW path."""
    alignment_path: np.ndarray
    dtw_total_cost: float
    dtw_mean_cost: float
    timing_mean_abs_lag_frames: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "overall_score": float(self.overall_score),
            "breakdown": {
                "timing": float(self.breakdown.timing),
                "arms": float(self.breakdown.arms),
                "legs": float(self.breakdown.legs),
                "torso_posture": float(self.breakdown.torso_posture),
                "joint_angles": float(self.breakdown.joint_angles),
                "limb_directions": float(self.breakdown.limb_directions),
                "relative_distances": float(self.breakdown.relative_distances),
                "movement": float(self.breakdown.movement),
            },
            "explanation_lines": list(self.explanation_lines),
            "per_frame_similarity": self.per_frame_similarity.astype(float).tolist(),
            "alignment_path": self.alignment_path.astype(int).tolist(),
            "dtw_total_cost": float(self.dtw_total_cost),
            "dtw_mean_cost": float(self.dtw_mean_cost),
            "timing_mean_abs_lag_frames": float(self.timing_mean_abs_lag_frames),
        }
