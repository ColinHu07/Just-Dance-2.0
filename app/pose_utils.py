"""MediaPipe Pose (Tasks API) — multi-person detection and skeleton drawing."""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision import PoseLandmarksConnections
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.core import image as mp_image

from app import video_utils

# Skip drawing / center math when visibility or presence is below this.
_VISIBILITY_THRESHOLD = 0.5

# MediaPipe pose indices (same as PoseLandmark enum).
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_TORSO_INDICES = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)

_MAX_POSES = 6

_POINT_RADIUS = 5
_LINE_COLOR = (0, 220, 0)
_POINT_COLOR = (0, 255, 255)
_LINE_THICKNESS = 3

# Slightly stronger styling for center-highlight (preview, “All People” mode).
_HIGHLIGHT_LINE_COLOR = (0, 140, 255)
_HIGHLIGHT_POINT_COLOR = (255, 200, 0)
_HIGHLIGHT_LINE_THICKNESS = 4
_HIGHLIGHT_POINT_RADIUS = 6

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_FILENAME = "pose_landmarker_lite.task"


class DetectionMode(str, Enum):
    """Overlay strategy (legacy single-pose vs multi-person pipelines)."""

    # Original app behavior: MediaPipe ``num_poses=1``, draw the single returned pose.
    LEGACY_SINGLE = "legacy_single"
    ALL_PEOPLE = "all_people"
    CENTER_ONLY = "center_only"


@dataclass
class AnnotateResult:
    """Output of ``annotate_frame``."""

    image: np.ndarray
    num_people: int
    """Index of the person treated as body-center (selection / highlight). None if nobody detected."""
    center_person_index: Optional[int]


def ensure_pose_model_path() -> str:
    """Download the bundled pose landmarker model once into ``temp/``; return path."""
    video_utils.ensure_app_dirs()
    dest = video_utils.TEMP_DIR / _MODEL_FILENAME
    if dest.exists() and dest.stat().st_size > 500_000:
        return str(dest)
    try:
        urllib.request.urlretrieve(_POSE_MODEL_URL, dest)
    except Exception as e:
        raise RuntimeError(
            "Could not download pose model. Check your network connection.\n"
            f"URL: {_POSE_MODEL_URL}\n"
            f"Error: {e}"
        ) from e
    if not dest.exists() or dest.stat().st_size < 500_000:
        raise RuntimeError("Pose model file is missing or too small after download.")
    return str(dest)


def create_pose_landmarker(
    *,
    for_video: bool,
    detection_mode: DetectionMode,
) -> PoseLandmarker:
    """
    Create a PoseLandmarker configured for the selected overlay mode.

    **Legacy Single Person** uses ``num_poses=1`` (original pre-multi-person setup).
    **All People** and **Center Person Only** use ``num_poses=_MAX_POSES``.
    """
    model_path = ensure_pose_model_path()
    run_mode = RunningMode.VIDEO if for_video else RunningMode.IMAGE
    num_poses = 1 if detection_mode == DetectionMode.LEGACY_SINGLE else _MAX_POSES
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=run_mode,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)


def _landmark_visible(landmark: Any) -> bool:
    vis = getattr(landmark, "visibility", None)
    if vis is None:
        vis = getattr(landmark, "presence", None)
    if vis is None:
        return True
    return vis >= _VISIBILITY_THRESHOLD


def _landmark_xy(landmark: Any, w: int, h: int) -> Optional[tuple[int, int]]:
    if not _landmark_visible(landmark):
        return None
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return x, y


def _norm_xy_if_visible(landmarks: Sequence[Any], idx: int) -> Optional[Tuple[float, float]]:
    if idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if not _landmark_visible(lm):
        return None
    return float(lm.x), float(lm.y)


def body_center_normalized(landmarks: List[Any]) -> Optional[Tuple[float, float]]:
    """
    Stable body center in normalized [0,1] coordinates:
    1) midpoint of left/right hip if both visible
    2) else midpoint of left/right shoulder if both visible
    3) else average of visible torso landmarks (shoulders + hips)
    """
    if not landmarks:
        return None

    lh = _norm_xy_if_visible(landmarks, _LEFT_HIP)
    rh = _norm_xy_if_visible(landmarks, _RIGHT_HIP)
    if lh is not None and rh is not None:
        return (lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0

    ls = _norm_xy_if_visible(landmarks, _LEFT_SHOULDER)
    rs = _norm_xy_if_visible(landmarks, _RIGHT_SHOULDER)
    if ls is not None and rs is not None:
        return (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0

    pts: List[Tuple[float, float]] = []
    for i in _TORSO_INDICES:
        p = _norm_xy_if_visible(landmarks, i)
        if p is not None:
            pts.append(p)
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def select_center_person_index(
    persons: List[List[Any]],
    width: int,
    height: int,
) -> Optional[int]:
    """
    Pick the person whose body center is closest to the image center (pixel space).
    Falls back to index 0 when people exist but no center could be computed.
    """
    if not persons:
        return None

    cx = width * 0.5
    cy = height * 0.5
    best_i: Optional[int] = None
    best_d2 = math.inf

    for i, lm in enumerate(persons):
        nc = body_center_normalized(lm)
        if nc is None:
            continue
        px = nc[0] * width
        py = nc[1] * height
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    if best_i is not None:
        return best_i
    return 0


def extract_pose_persons(
    landmarker: PoseLandmarker,
    bgr: np.ndarray,
    *,
    timestamp_ms: Optional[int],
) -> List[List[Any]]:
    """Run the landmarker and return a list of per-person landmark lists (may be empty)."""
    if bgr is None or bgr.size == 0:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)

    if timestamp_ms is None:
        result = landmarker.detect(mp_img)
    else:
        result = landmarker.detect_for_video(mp_img, timestamp_ms)

    if not result.pose_landmarks:
        return []
    # Defensive copy: list of lists of landmarks
    return [list(pl) for pl in result.pose_landmarks]


def _annotate_legacy_single(
    landmarker: PoseLandmarker,
    bgr: np.ndarray,
    *,
    timestamp_ms: Optional[int],
) -> AnnotateResult:
    """
    Pre-multi-person behavior: one inference pass, draw ``pose_landmarks[0]`` only.

    Expects a landmarker created with ``num_poses=1`` (see ``create_pose_landmarker`` +
    ``DetectionMode.LEGACY_SINGLE``).
    """
    if bgr is None or bgr.size == 0:
        return AnnotateResult(image=bgr, num_people=0, center_person_index=None)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)

    if timestamp_ms is None:
        result = landmarker.detect(mp_img)
    else:
        result = landmarker.detect_for_video(mp_img, timestamp_ms)

    out = bgr.copy()
    if result.pose_landmarks:
        _draw_pose_landmarks(out, result.pose_landmarks[0])
        return AnnotateResult(image=out, num_people=1, center_person_index=0)
    return AnnotateResult(image=out, num_people=0, center_person_index=None)


def _draw_pose_landmarks(
    bgr: np.ndarray,
    landmarks: List[Any],
    *,
    line_color: Tuple[int, int, int] = _LINE_COLOR,
    point_color: Tuple[int, int, int] = _POINT_COLOR,
    line_thickness: int = _LINE_THICKNESS,
    point_radius: int = _POINT_RADIUS,
) -> None:
    h, w = bgr.shape[:2]
    if not landmarks:
        return

    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        a, b = conn.start, conn.end
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        pa = _landmark_xy(landmarks[a], w, h)
        pb = _landmark_xy(landmarks[b], w, h)
        if pa is None or pb is None:
            continue
        cv2.line(bgr, pa, pb, line_color, line_thickness, lineType=cv2.LINE_AA)

    for l in landmarks:
        pt = _landmark_xy(l, w, h)
        if pt is None:
            continue
        cv2.circle(bgr, pt, point_radius, point_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, pt, point_radius, (0, 80, 0), 1, lineType=cv2.LINE_AA)


def annotate_frame(
    landmarker: PoseLandmarker,
    bgr: np.ndarray,
    *,
    timestamp_ms: Optional[int] = None,
    mode: DetectionMode = DetectionMode.LEGACY_SINGLE,
    highlight_center_person: bool = False,
) -> AnnotateResult:
    """
    Run pose on one BGR frame and return overlay + detection stats.

    ``LEGACY_SINGLE`` uses the dedicated single-pose path (``num_poses=1`` landmarker).

    ``highlight_center_person``: in ``ALL_PEOPLE`` mode, draw the center-selected
    person with slightly stronger colors (intended for preview only).
    """
    if bgr is None or bgr.size == 0:
        return AnnotateResult(image=bgr, num_people=0, center_person_index=None)

    if mode == DetectionMode.LEGACY_SINGLE:
        return _annotate_legacy_single(landmarker, bgr, timestamp_ms=timestamp_ms)

    persons = extract_pose_persons(landmarker, bgr, timestamp_ms=timestamp_ms)
    h, w = bgr.shape[:2]
    out = bgr.copy()

    if not persons:
        return AnnotateResult(image=out, num_people=0, center_person_index=None)

    center_idx = select_center_person_index(persons, w, h)

    if mode == DetectionMode.CENTER_ONLY:
        if 0 <= center_idx < len(persons):
            _draw_pose_landmarks(out, persons[center_idx])
        return AnnotateResult(
            image=out,
            num_people=len(persons),
            center_person_index=center_idx,
        )

    # ALL_PEOPLE
    for i, plm in enumerate(persons):
        is_center = i == center_idx
        if highlight_center_person and is_center:
            _draw_pose_landmarks(
                out,
                plm,
                line_color=_HIGHLIGHT_LINE_COLOR,
                point_color=_HIGHLIGHT_POINT_COLOR,
                line_thickness=_HIGHLIGHT_LINE_THICKNESS,
                point_radius=_HIGHLIGHT_POINT_RADIUS,
            )
        else:
            _draw_pose_landmarks(out, plm)

    return AnnotateResult(
        image=out,
        num_people=len(persons),
        center_person_index=center_idx,
    )
