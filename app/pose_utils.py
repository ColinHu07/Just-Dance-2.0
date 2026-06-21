"""MediaPipe Pose (Tasks API) — multi-person detection and skeleton drawing.

Visualization notes
-------------------
**Head** — Dense MediaPipe face landmarks (indices 0–10) are not drawn. A 4-point
compass model (N/E/S/W) approximates head position, extent, and tilt.

**Body stabilization** (video mode) — Torso, arms, legs, and head compass use
confidence gating, short holds when joints flicker, partial segments, and EMA
smoothing with **body-part-specific** tuning. **Arms and legs** additionally
use **per-limb timers**: after ~``_LIMB_HARD_DROP_SEC`` at ``stabilization_fps``
with all core joints below ``_LIMB_ALL_LOST_SCORE``, that limb stops drawing until
a core joint is reliable again (no indefinite ghost limbs). Legs still omit the
weaker limb in side-on ``profile`` poses.
"""

from __future__ import annotations

import math
import urllib.request
import mediapipe as mp
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode


from app import video_utils

# Skip drawing / center math when visibility or presence is below this.
_VISIBILITY_THRESHOLD = 0.55

# Narrow shoulder/hip span (normalized x) → treat as side-on; be conservative on legs.
_PROFILE_SHOULDER_W = 0.06
_PROFILE_HIP_W = 0.052
_LEG_DOMINANCE_GAP = 0.32  # suppress only clearly lost weaker legs in profile
_LEG_SUPPRESS_DISTAL_SCORE = 0.46

# Per-limb absence: brief joint hold (seconds at source FPS), then hard drop (no limb draw).
_LIMB_HARD_DROP_SEC = 1.0
# While a limb is "weak" but not yet dropped, allow joint holds up to this many frames-worth.
_LIMB_OCCLUSION_HOLD_SEC = 0.22
# Core joints must all score below this to count as "fully lost" for absent counter.
_LIMB_ALL_LOST_SCORE = 0.42
_DEFAULT_STABILIZATION_FPS = 30.0

# When a pose first appears after a title card/fade-in, MediaPipe can return a
# plausible but awkward first body. Hold drawing briefly until the track settles.
_POSE_INITIAL_LOCK_FRAMES = 10

# Whole-pose confidence gate. Per-joint gates still decide which limbs draw, but
# these thresholds stop low-confidence frames from drawing any skeleton at all.
_POSE_FRAME_MIN_TORSO_AVG_SCORE = 0.58
_POSE_FRAME_MIN_CORE_AVG_SCORE = 0.56
_POSE_FRAME_MIN_CORE_VISIBLE = 5

# MediaPipe pose indices (same as PoseLandmark enum).
_NOSE = 0
_LEFT_EYE_CLUSTER = (1, 2, 3)
_RIGHT_EYE_CLUSTER = (4, 5, 6)
_LEFT_EAR = 7
_RIGHT_EAR = 8
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_TORSO_INDICES = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
_CORE_BODY_INDICES = (
    _LEFT_SHOULDER,
    _RIGHT_SHOULDER,
    _LEFT_HIP,
    _RIGHT_HIP,
    _LEFT_KNEE,
    _RIGHT_KNEE,
    _LEFT_ANKLE,
    _RIGHT_ANKLE,
)

# Face landmark indices use a simplified head model instead of dense points.
_FACE_INDICES = frozenset(range(0, 11))

# Full skeleton (except head compass) is drawn via stabilization — no raw connection pass.
_TORSO_SEGMENTS: Tuple[Tuple[int, int], ...] = (
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
)
_LEFT_ARM_SEGMENTS: Tuple[Tuple[int, int], ...] = (
    (11, 13),
    (13, 15),
)
_RIGHT_ARM_SEGMENTS: Tuple[Tuple[int, int], ...] = (
    (12, 14),
    (14, 16),
)
_LEFT_LEG_SEGMENTS: Tuple[Tuple[int, int], ...] = (
    (_LEFT_HIP, _LEFT_KNEE),
    (_LEFT_KNEE, _LEFT_ANKLE),
)
_RIGHT_LEG_SEGMENTS: Tuple[Tuple[int, int], ...] = (
    (_RIGHT_HIP, _RIGHT_KNEE),
    (_RIGHT_KNEE, _RIGHT_ANKLE),
)

_MAX_POSES = 6

_POINT_RADIUS = 5
_LINE_COLOR = (0, 220, 0)
_POINT_COLOR = (0, 255, 255)
_LINE_THICKNESS = 3

# Simplified head diamond (compass) — minimal styling.
_HEAD_POINT_RADIUS = 4
_HEAD_DIAMOND_COLOR = (200, 200, 60)
_HEAD_DIAMOND_THICK = 1
_HEAD_COMPASS_MAX_SPAN = 0.075
_TORSO_DECROSS_MIN_PAIR_SPAN_PX = 9
_COMPACT_DRAW_SHOULDER_W = 0.085
_COMPACT_DRAW_HIP_W = 0.075

# Estimated / held leg segments (temporal fill-in).
_FAINT_LINE_SCALE = 0.65  # relative to line_thickness

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


@dataclass
class JointTrack:
    """Short temporal memory for a skeleton landmark (hold + optional EMA smoothing)."""

    last_px: Optional[Tuple[int, int]] = None
    miss_streak: int = 0
    smooth_xy: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class BodyPartPolicy:
    """
    Region-specific stabilization: solid visibility threshold, brief hold length,
    and EMA weight on the current frame when visible (higher = snappier).
    """

    solid_threshold: float
    max_hold_frames: int
    smoothing_alpha: float
    max_jump_fraction: float


# Torso / shoulders / hips — strongest smoothing, longest hold.
TORSO_POLICY = BodyPartPolicy(
    solid_threshold=0.5,
    max_hold_frames=5,
    smoothing_alpha=0.5,
    max_jump_fraction=0.18,
)
# Arms — medium responsiveness; partial arms when wrist/elbow weak.
ARM_POLICY = BodyPartPolicy(
    solid_threshold=0.58,
    max_hold_frames=2,
    smoothing_alpha=0.72,
    max_jump_fraction=0.42,
)
# Legs — match prior behavior; profile suppression stays separate.
LEG_POLICY = BodyPartPolicy(
    solid_threshold=0.58,
    max_hold_frames=2,
    smoothing_alpha=0.68,
    max_jump_fraction=0.32,
)
# Simplified head compass — light smoothing only.
HEAD_COMPASS_POLICY = BodyPartPolicy(
    solid_threshold=0.0,
    max_hold_frames=2,
    smoothing_alpha=0.74,
    max_jump_fraction=0.18,
)


@dataclass
class LimbTrack:
    """Absence / drop state for one arm or leg chain."""

    consecutive_lost_frames: int = 0
    dropped: bool = False


@dataclass
class PoseStartupLock:
    """Short startup gate before drawing a newly detected body track."""

    warmup_frames: int = 0
    locked_once: bool = False


# Per-limb stabilization: ``core`` drives absence/hold/drop; ``mask`` clears cache when dropped.
_LIMB_SPECS: Dict[str, Dict[str, Any]] = {
    "L_arm": {
        "core": (_LEFT_SHOULDER, 13, 15),
        "mask": (13, 15, 17, 19, 21),
        "policy": ARM_POLICY,
    },
    "R_arm": {
        "core": (_RIGHT_SHOULDER, 14, 16),
        "mask": (14, 16, 18, 20, 22),
        "policy": ARM_POLICY,
    },
    "L_leg": {
        "core": (_LEFT_KNEE, _LEFT_ANKLE),
        "mask": (25, 27, 29, 31),
        "policy": LEG_POLICY,
    },
    "R_leg": {
        "core": (_RIGHT_KNEE, _RIGHT_ANKLE),
        "mask": (26, 28, 30, 32),
        "policy": LEG_POLICY,
    },
}

_LIMB_NAME_FOR_JOINT: Dict[int, str] = {}
for _idx in (13, 15, 17, 19, 21):
    _LIMB_NAME_FOR_JOINT[_idx] = "L_arm"
for _idx in (14, 16, 18, 20, 22):
    _LIMB_NAME_FOR_JOINT[_idx] = "R_arm"
for _idx in (25, 27, 29, 31):
    _LIMB_NAME_FOR_JOINT[_idx] = "L_leg"
for _idx in (26, 28, 30, 32):
    _LIMB_NAME_FOR_JOINT[_idx] = "R_leg"


class PoseDrawTemporalState:
    """
    Per-video stabilization: landmark tracks keyed by (person_index, landmark_index),
    plus separate tracks for head compass labels ``N``/``E``/``S``/``W``.

    Pass the same instance across ``annotate_frame`` calls while processing a clip;
    omit for single-frame preview.
    """

    def __init__(self) -> None:
        self._tracks: Dict[Tuple[int, int], JointTrack] = {}
        self._head_tracks: Dict[Tuple[int, str], JointTrack] = {}
        self._limb_tracks: Dict[Tuple[int, str], LimbTrack] = {}
        self._startup_locks: Dict[int, PoseStartupLock] = {}
        self.last_center_norm: Optional[Tuple[float, float]] = None

    def begin_frame(self, active_person_indices: set[int]) -> None:
        stale = [k for k in self._tracks if k[0] not in active_person_indices]
        for k in stale:
            del self._tracks[k]
        stale_h = [k for k in self._head_tracks if k[0] not in active_person_indices]
        for k in stale_h:
            del self._head_tracks[k]
        stale_l = [k for k in self._limb_tracks if k[0] not in active_person_indices]
        for k in stale_l:
            del self._limb_tracks[k]
        stale_s = [
            k
            for k, v in self._startup_locks.items()
            if k not in active_person_indices and not v.locked_once
        ]
        for k in stale_s:
            del self._startup_locks[k]

    def track_for(self, person: int, lm_index: int) -> JointTrack:
        k = (person, lm_index)
        if k not in self._tracks:
            self._tracks[k] = JointTrack()
        return self._tracks[k]

    def head_track_for(self, person: int, compass_label: str) -> JointTrack:
        k = (person, compass_label)
        if k not in self._head_tracks:
            self._head_tracks[k] = JointTrack()
        return self._head_tracks[k]

    def limb_track_for(self, person: int, limb_name: str) -> LimbTrack:
        k = (person, limb_name)
        if k not in self._limb_tracks:
            self._limb_tracks[k] = LimbTrack()
        return self._limb_tracks[k]

    def startup_lock_for(self, person: int) -> PoseStartupLock:
        if person not in self._startup_locks:
            self._startup_locks[person] = PoseStartupLock()
        return self._startup_locks[person]

    def ready_to_draw_pose(self, person: int) -> bool:
        lock = self.startup_lock_for(person)
        if lock.locked_once:
            return True
        lock.warmup_frames += 1
        if lock.warmup_frames >= _POSE_INITIAL_LOCK_FRAMES:
            lock.locked_once = True
            return True
        return False


def _policy_for_landmark_index(idx: int) -> BodyPartPolicy:
    if idx in (11, 12, 23, 24):
        return TORSO_POLICY
    if idx in (13, 15, 17, 19, 21, 14, 16, 18, 20, 22):
        return ARM_POLICY
    if idx in (25, 27, 29, 31, 26, 28, 30, 32):
        return LEG_POLICY
    return ARM_POLICY


def resolve_joint_display(
    landmark: Any,
    w: int,
    h: int,
    *,
    person_index: int,
    lm_index: int,
    temporal_state: Optional[PoseDrawTemporalState],
    policy: BodyPartPolicy,
    max_hold_override: Optional[int] = None,
) -> Tuple[Optional[Tuple[int, int]], str]:
    """Return (pixel, style) with style ``solid`` | ``held`` | ``none``."""
    px_now = _landmark_px_from_score(landmark, w, h, policy.solid_threshold)
    max_hold = (
        max_hold_override if max_hold_override is not None else policy.max_hold_frames
    )

    if temporal_state is None:
        if px_now is None:
            return None, "none"
        return px_now, "solid"

    tr = temporal_state.track_for(person_index, lm_index)
    if px_now is not None:
        sx, sy = float(px_now[0]), float(px_now[1])
        if tr.smooth_xy is not None:
            ox, oy = tr.smooth_xy
            max_jump = max(24.0, policy.max_jump_fraction * min(w, h))
            if math.hypot(sx - ox, sy - oy) > max_jump:
                px_now = None
        if px_now is None:
            tr.miss_streak += 1
            if tr.last_px is not None and max_hold > 0 and tr.miss_streak <= max_hold:
                return tr.last_px, "held"
            tr.last_px = None
            tr.smooth_xy = None
            return None, "none"
        if tr.smooth_xy is None:
            tr.smooth_xy = (sx, sy)
        else:
            a = policy.smoothing_alpha
            oxoy = tr.smooth_xy
            ox, oy = oxoy[0], oxoy[1]
            tr.smooth_xy = (a * sx + (1.0 - a) * ox, a * sy + (1.0 - a) * oy)
        ox_i = int(round(tr.smooth_xy[0]))
        oy_i = int(round(tr.smooth_xy[1]))
        out = (max(0, min(w - 1, ox_i)), max(0, min(h - 1, oy_i)))
        tr.last_px = out
        tr.miss_streak = 0
        return out, "solid"

    tr.miss_streak += 1
    if tr.last_px is not None and max_hold > 0 and tr.miss_streak <= max_hold:
        return tr.last_px, "held"

    tr.last_px = None
    tr.smooth_xy = None
    return None, "none"


def _reset_joint_tracks_for_indices(
    state: PoseDrawTemporalState,
    person: int,
    indices: Sequence[int],
) -> None:
    for idx in indices:
        k = (person, idx)
        if k not in state._tracks:
            continue
        t = state._tracks[k]
        t.smooth_xy = None
        t.last_px = None
        t.miss_streak = 0


def _effective_joint_max_hold(
    temporal_state: PoseDrawTemporalState,
    person_index: int,
    lm_index: int,
    policy: BodyPartPolicy,
    fps: float,
) -> int:
    """Longer joint hold during short limb occlusion; zero when limb is hard-dropped."""
    limb = _LIMB_NAME_FOR_JOINT.get(lm_index)
    if limb is None:
        return policy.max_hold_frames
    lt = temporal_state.limb_track_for(person_index, limb)
    if lt.dropped:
        return 0
    drop_cap = max(1, int(round(fps * _LIMB_HARD_DROP_SEC)))
    if lt.consecutive_lost_frames > 0:
        extended = max(policy.max_hold_frames, int(round(fps * _LIMB_OCCLUSION_HOLD_SEC)))
        return min(extended, drop_cap)
    return policy.max_hold_frames


def _update_limb_drop_state(
    temporal_state: PoseDrawTemporalState,
    landmarks: List[Any],
    *,
    person_index: int,
    fps: float,
    suppress_left_leg: bool,
    suppress_right_leg: bool,
) -> None:
    """
    Track per-limb “all core joints lost” streak; after ``_LIMB_HARD_DROP_SEC`` worth
    of frames, mark limb dropped until a core joint is reliable again.
    """
    fps = max(1.0, min(120.0, fps))
    drop_frames = max(1, int(round(fps * _LIMB_HARD_DROP_SEC)))

    for limb_name, spec in _LIMB_SPECS.items():
        if limb_name == "L_leg" and suppress_left_leg:
            lt = temporal_state.limb_track_for(person_index, limb_name)
            lt.consecutive_lost_frames = 0
            lt.dropped = False
            continue
        if limb_name == "R_leg" and suppress_right_leg:
            lt = temporal_state.limb_track_for(person_index, limb_name)
            lt.consecutive_lost_frames = 0
            lt.dropped = False
            continue

        th = spec["policy"].solid_threshold
        core: Tuple[int, ...] = spec["core"]
        scores: List[float] = []
        for ci in core:
            if ci < len(landmarks):
                scores.append(_joint_score(landmarks[ci]))
        if not scores:
            continue

        strong = max(scores) >= th
        all_lost = all(s < _LIMB_ALL_LOST_SCORE for s in scores)

        lt = temporal_state.limb_track_for(person_index, limb_name)
        was_dropped = lt.dropped

        if strong:
            if was_dropped:
                _reset_joint_tracks_for_indices(
                    temporal_state, person_index, spec["mask"]
                )
            lt.dropped = False
            lt.consecutive_lost_frames = 0
            continue

        if all_lost:
            lt.consecutive_lost_frames += 1
            if lt.consecutive_lost_frames > drop_frames:
                lt.dropped = True
        # else: middling visibility — do not inflate lost streak (partial limb / noise)


def _apply_limb_drop_mask(
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    temporal_state: PoseDrawTemporalState,
    *,
    person_index: int,
    suppress_left_leg: bool,
    suppress_right_leg: bool,
) -> None:
    for limb_name, spec in _LIMB_SPECS.items():
        if limb_name == "L_leg" and suppress_left_leg:
            continue
        if limb_name == "R_leg" and suppress_right_leg:
            continue
        if temporal_state.limb_track_for(person_index, limb_name).dropped:
            for idx in spec["mask"]:
                cache[idx] = (None, "none")


def _build_joint_cache(
    landmarks: List[Any],
    indices: set[int],
    w: int,
    h: int,
    person_index: int,
    temporal_state: Optional[PoseDrawTemporalState],
    stabilization_fps: float,
) -> Dict[int, Tuple[Optional[Tuple[int, int]], str]]:
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]] = {}
    s_fps = max(1.0, min(120.0, stabilization_fps))
    for idx in sorted(indices):
        if idx >= len(landmarks):
            cache[idx] = (None, "none")
            continue
        pol = _policy_for_landmark_index(idx)
        mh: Optional[int] = None
        if temporal_state is not None:
            mh = _effective_joint_max_hold(
                temporal_state, person_index, idx, pol, s_fps
            )
        cache[idx] = resolve_joint_display(
            landmarks[idx],
            w,
            h,
            person_index=person_index,
            lm_index=idx,
            temporal_state=temporal_state,
            policy=pol,
            max_hold_override=mh,
        )
    return cache


def _draw_segments_from_cache(
    bgr: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    *,
    line_color: Tuple[int, int, int],
    line_thickness: int,
) -> None:
    faint_thickness = max(1, int(line_thickness * _FAINT_LINE_SCALE))
    faint_line = tuple(max(0, c - 55) for c in line_color)
    for a, b in segments:
        pa, sa = cache.get(a, (None, "none"))
        pb, sb = cache.get(b, (None, "none"))
        if pa is None or pb is None:
            continue
        if sa == "held" or sb == "held":
            cv2.line(bgr, pa, pb, faint_line, faint_thickness, lineType=cv2.LINE_AA)
        else:
            cv2.line(bgr, pa, pb, line_color, line_thickness, lineType=cv2.LINE_AA)


def _draw_styled_line(
    bgr: np.ndarray,
    pa: Tuple[int, int],
    pb: Tuple[int, int],
    *,
    style_a: str,
    style_b: str,
    line_color: Tuple[int, int, int],
    line_thickness: int,
) -> None:
    faint_thickness = max(1, int(line_thickness * _FAINT_LINE_SCALE))
    faint_line = tuple(max(0, c - 55) for c in line_color)
    if style_a == "held" or style_b == "held":
        cv2.line(bgr, pa, pb, faint_line, faint_thickness, lineType=cv2.LINE_AA)
    else:
        cv2.line(bgr, pa, pb, line_color, line_thickness, lineType=cv2.LINE_AA)


def _midpoint_from_cache(
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    a: int,
    b: int,
) -> Tuple[Optional[Tuple[int, int]], str]:
    pa, sa = cache.get(a, (None, "none"))
    pb, sb = cache.get(b, (None, "none"))
    if pa is None and pb is None:
        return None, "none"
    if pa is None:
        return pb, sb
    if pb is None:
        return pa, sa
    style = "held" if sa == "held" or sb == "held" else "solid"
    return ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2), style


def _draw_compact_torso_from_cache(
    bgr: np.ndarray,
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    *,
    line_color: Tuple[int, int, int],
    line_thickness: int,
) -> None:
    shoulder_mid, shoulder_style = _midpoint_from_cache(
        cache, _LEFT_SHOULDER, _RIGHT_SHOULDER
    )
    hip_mid, hip_style = _midpoint_from_cache(cache, _LEFT_HIP, _RIGHT_HIP)
    if shoulder_mid is not None and hip_mid is not None:
        _draw_styled_line(
            bgr,
            shoulder_mid,
            hip_mid,
            style_a=shoulder_style,
            style_b=hip_style,
            line_color=line_color,
            line_thickness=line_thickness,
        )
    _draw_segments_from_cache(
        bgr,
        ((_LEFT_SHOULDER, _RIGHT_SHOULDER), (_LEFT_HIP, _RIGHT_HIP)),
        cache,
        line_color=line_color,
        line_thickness=line_thickness,
    )


def _compact_lower_leg_segments(
    segments: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    return [
        seg
        for seg in segments
        if seg in ((_LEFT_KNEE, _LEFT_ANKLE), (_RIGHT_KNEE, _RIGHT_ANKLE))
    ]


def _draw_compact_legs_from_cache(
    bgr: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    *,
    line_color: Tuple[int, int, int],
    line_thickness: int,
) -> None:
    _draw_segments_from_cache(
        bgr,
        _compact_lower_leg_segments(segments),
        cache,
        line_color=line_color,
        line_thickness=line_thickness,
    )


def _draw_joint_dots_from_cache(
    bgr: np.ndarray,
    indices: set[int],
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    point_color: Tuple[int, int, int],
    point_radius: int,
) -> None:
    faint_point = tuple(max(0, c - 40) for c in point_color)
    for idx in sorted(indices):
        px, st = cache.get(idx, (None, "none"))
        if px is None:
            continue
        pr = max(2, point_radius - 1) if st == "held" else point_radius
        col = faint_point if st == "held" else point_color
        cv2.circle(bgr, px, pr, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, px, pr, (0, 60, 0), 1, lineType=cv2.LINE_AA)


def _pair_x_polarity(
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    left_idx: int,
    right_idx: int,
    *,
    min_span_px: float,
) -> Optional[int]:
    left_px, _ = cache.get(left_idx, (None, "none"))
    right_px, _ = cache.get(right_idx, (None, "none"))
    if left_px is None or right_px is None:
        return None
    dx = left_px[0] - right_px[0]
    if abs(dx) < min_span_px:
        return None
    return 1 if dx > 0 else -1


def _swap_cache_entries(
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    a: int,
    b: int,
) -> None:
    cache[a], cache[b] = cache.get(b, (None, "none")), cache.get(a, (None, "none"))


def _decross_torso_cache_for_drawing(
    cache: Dict[int, Tuple[Optional[Tuple[int, int]], str]],
    *,
    frame_width: int,
) -> Dict[int, Tuple[Optional[Tuple[int, int]], str]]:
    """
    Keep the rendered torso from flipping when MediaPipe swaps one left/right pair.

    Pose labels are anatomical, but in side-on/crouched frames the lower-body labels
    can briefly invert while shoulders stay stable. For overlay drawing only, swap the
    lower-body pairs so shoulder and hip x-order agree and torso lines do not cross.
    """
    min_span = max(float(_TORSO_DECROSS_MIN_PAIR_SPAN_PX), frame_width * 0.025)
    shoulder_pol = _pair_x_polarity(
        cache, _LEFT_SHOULDER, _RIGHT_SHOULDER, min_span_px=min_span
    )
    hip_pol = _pair_x_polarity(
        cache, _LEFT_HIP, _RIGHT_HIP, min_span_px=min_span
    )
    if shoulder_pol is None or hip_pol is None or shoulder_pol == hip_pol:
        return cache

    corrected = dict(cache)
    for a, b in (
        (_LEFT_HIP, _RIGHT_HIP),
        (_LEFT_KNEE, _RIGHT_KNEE),
        (_LEFT_ANKLE, _RIGHT_ANKLE),
    ):
        _swap_cache_entries(corrected, a, b)
    return corrected


def _stabilize_head_compass_pixel(
    px: Tuple[int, int],
    w: int,
    h: int,
    *,
    person_index: int,
    label: str,
    temporal_state: Optional[PoseDrawTemporalState],
) -> Tuple[Tuple[int, int], str]:
    """Light EMA on head compass points in pixel space (video only)."""
    if temporal_state is None:
        return px, "solid"
    tr = temporal_state.head_track_for(person_index, label)
    sx, sy = float(px[0]), float(px[1])
    pol = HEAD_COMPASS_POLICY
    if tr.smooth_xy is None:
        tr.smooth_xy = (sx, sy)
    else:
        a = pol.smoothing_alpha
        ox, oy = tr.smooth_xy
        tr.smooth_xy = (a * sx + (1.0 - a) * ox, a * sy + (1.0 - a) * oy)
    ox_i = int(round(tr.smooth_xy[0]))
    oy_i = int(round(tr.smooth_xy[1]))
    out = (max(0, min(w - 1, ox_i)), max(0, min(h - 1, oy_i)))
    tr.last_px = out
    tr.miss_streak = 0
    return out, "solid"


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
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
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


def _joint_score(landmark: Any) -> float:
    v = getattr(landmark, "visibility", None)
    if v is None:
        v = getattr(landmark, "presence", None)
    if v is None:
        return 1.0
    return float(v)


def landmark_reliability(landmark: Any) -> float:
    """MediaPipe visibility / presence in ``[0, 1]`` (defaults to 1.0 if absent)."""
    return _joint_score(landmark)


def norm_xy_if_visible(landmarks: Sequence[Any], idx: int) -> Optional[Tuple[float, float]]:
    """Normalized image ``(x, y)`` in ``[0, 1]`` if the landmark passes the visibility gate."""
    return _norm_xy_if_visible(landmarks, idx)


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


def _landmark_px_from_score(
    landmark: Any, w: int, h: int, min_score: float
) -> Optional[Tuple[int, int]]:
    if _joint_score(landmark) < min_score:
        return None
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return x, y


def _centroid_norm_points(landmarks: Sequence[Any], indices: Sequence[int]) -> Optional[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in indices:
        if i >= len(landmarks):
            continue
        p = _norm_xy_if_visible(landmarks, i)
        if p is not None:
            pts.append(p)
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def _head_compass_norm(landmarks: List[Any]) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Four compass points in normalized [0,1]²: N (top), E (right), S (chin-ward), W (left).
    Uses ears → fallback eyes; center from visible head landmarks; no dense face dots.
    """
    if not landmarks:
        return None

    left_side = _centroid_norm_points(landmarks, [_LEFT_EAR]) or _centroid_norm_points(
        landmarks, list(_LEFT_EYE_CLUSTER)
    )
    right_side = _centroid_norm_points(landmarks, [_RIGHT_EAR]) or _centroid_norm_points(
        landmarks, list(_RIGHT_EYE_CLUSTER)
    )

    head_pts: List[Tuple[float, float]] = []
    for i in _FACE_INDICES:
        if i < len(landmarks):
            p = _norm_xy_if_visible(landmarks, i)
            if p is not None:
                head_pts.append(p)
    center: Optional[Tuple[float, float]] = None
    if head_pts:
        center = (
            sum(p[0] for p in head_pts) / len(head_pts),
            sum(p[1] for p in head_pts) / len(head_pts),
        )
    else:
        nose = _norm_xy_if_visible(landmarks, _NOSE)
        center = nose

    if center is None:
        return None

    # Head horizontal extent
    if left_side and right_side:
        if left_side[0] <= right_side[0]:
            w_pt, e_pt = left_side, right_side
        else:
            w_pt, e_pt = right_side, left_side
        span = max(
            0.028,
            math.hypot(e_pt[0] - w_pt[0], e_pt[1] - w_pt[1]) * 0.58,
        )
        span = min(span, _HEAD_COMPASS_MAX_SPAN)
    elif left_side:
        w_pt = left_side
        span = max(0.028, 2.0 * abs(center[0] - w_pt[0]), 0.04)
        span = min(span, _HEAD_COMPASS_MAX_SPAN)
        e_pt = (center[0] + (center[0] - w_pt[0]), center[1])
    elif right_side:
        e_pt = right_side
        span = max(0.028, 2.0 * abs(e_pt[0] - center[0]), 0.04)
        span = min(span, _HEAD_COMPASS_MAX_SPAN)
        w_pt = (center[0] - (e_pt[0] - center[0]), center[1])
    else:
        span = 0.038
        w_pt = (center[0] - span * 0.95, center[1])
        e_pt = (center[0] + span * 0.95, center[1])

    nose = _norm_xy_if_visible(landmarks, _NOSE)
    north_lift = span * 1.05
    south_drop = span * 0.78
    if nose is not None:
        # Slightly bias “chin” toward nose direction from center (vertical component).
        vy = max(1e-6, nose[1] - center[1])
        south_drop = max(south_drop, min(span * 1.1, vy * 1.15))

    n_pt = (center[0], center[1] - north_lift)
    s_pt = (center[0], center[1] + south_drop)

    def _clamp(p: Tuple[float, float]) -> Tuple[float, float]:
        return (max(0.0, min(1.0, p[0])), max(0.0, min(1.0, p[1])))

    return {
        "N": _clamp(n_pt),
        "E": _clamp(e_pt),
        "S": _clamp(s_pt),
        "W": _clamp(w_pt),
    }


def _shoulder_hip_widths_norm(landmarks: List[Any]) -> Tuple[float, float]:
    ls = _norm_xy_if_visible(landmarks, _LEFT_SHOULDER)
    rs = _norm_xy_if_visible(landmarks, _RIGHT_SHOULDER)
    sh_w = abs(ls[0] - rs[0]) if ls and rs else 1.0
    lh = _norm_xy_if_visible(landmarks, _LEFT_HIP)
    rh = _norm_xy_if_visible(landmarks, _RIGHT_HIP)
    hip_w = abs(lh[0] - rh[0]) if lh and rh else 1.0
    return sh_w, hip_w


def _is_profile_like_pose(landmarks: List[Any]) -> bool:
    sh_w, hip_w = _shoulder_hip_widths_norm(landmarks)
    return sh_w < _PROFILE_SHOULDER_W or hip_w < _PROFILE_HIP_W


def _needs_compact_body_draw(landmarks: List[Any]) -> bool:
    sh_w, hip_w = _shoulder_hip_widths_norm(landmarks)
    return sh_w < _COMPACT_DRAW_SHOULDER_W or hip_w < _COMPACT_DRAW_HIP_W


def _leg_chain_avg_score(landmarks: List[Any], hip: int, knee: int, ankle: int) -> float:
    idxs = [hip, knee, ankle]
    scores = [_joint_score(landmarks[i]) for i in idxs if i < len(landmarks)]
    return sum(scores) / len(scores) if scores else 0.0


def _leg_distal_lost(landmarks: List[Any], knee: int, ankle: int) -> bool:
    if knee >= len(landmarks) or ankle >= len(landmarks):
        return True
    return (
        _joint_score(landmarks[knee]) < _LEG_SUPPRESS_DISTAL_SCORE
        and _joint_score(landmarks[ankle]) < _LEG_SUPPRESS_DISTAL_SCORE
    )


def _leg_suppression_flags(landmarks: List[Any]) -> Tuple[bool, bool]:
    """In side-like views, omit the likely occluded (weaker) leg entirely."""
    if not _is_profile_like_pose(landmarks):
        return False, False
    left_m = _leg_chain_avg_score(landmarks, _LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE)
    right_m = _leg_chain_avg_score(landmarks, _RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE)
    suppress_l = (
        left_m + _LEG_DOMINANCE_GAP < right_m
        and _leg_distal_lost(landmarks, _LEFT_KNEE, _LEFT_ANKLE)
    )
    suppress_r = (
        right_m + _LEG_DOMINANCE_GAP < left_m
        and _leg_distal_lost(landmarks, _RIGHT_KNEE, _RIGHT_ANKLE)
    )
    return suppress_l, suppress_r


def _draw_head_compass(
    bgr: np.ndarray,
    landmarks: List[Any],
    *,
    point_color: Tuple[int, int, int],
    dot_radius: int = _HEAD_POINT_RADIUS,
    temporal_state: Optional[PoseDrawTemporalState] = None,
    person_index: int = 0,
) -> None:
    h, w = bgr.shape[:2]
    model = _head_compass_norm(landmarks)
    if model is None:
        return

    pix: Dict[str, Tuple[int, int]] = {}
    for name, p in model.items():
        x = int(p[0] * w)
        y = int(p[1] * h)
        raw = (max(0, min(w - 1, x)), max(0, min(h - 1, y)))
        stab, _ = _stabilize_head_compass_pixel(
            raw,
            w,
            h,
            person_index=person_index,
            label=name,
            temporal_state=temporal_state,
        )
        pix[name] = stab

    r = max(3, min(dot_radius, 10))
    for name in ("N", "E", "S", "W"):
        pt = pix.get(name)
        if pt is None:
            continue
        cv2.circle(bgr, pt, r, point_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, pt, r, (30, 100, 30), 1, lineType=cv2.LINE_AA)

    order = ("N", "E", "S", "W", "N")
    for a, b in zip(order[:-1], order[1:]):
        pa, pb = pix.get(a), pix.get(b)
        if pa and pb:
            cv2.line(bgr, pa, pb, _HEAD_DIAMOND_COLOR, _HEAD_DIAMOND_THICK, lineType=cv2.LINE_AA)


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
    previous_center: Optional[Tuple[float, float]] = None,
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
        frame_center_d2 = (px - cx) ** 2 + (py - cy) ** 2
        if previous_center is not None:
            pcx = previous_center[0] * width
            pcy = previous_center[1] * height
            prev_d2 = (px - pcx) ** 2 + (py - pcy) ** 2
            d2 = 0.35 * frame_center_d2 + 0.65 * prev_d2
        else:
            d2 = frame_center_d2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    if best_i is not None:
        return best_i
    return 0


def _pose_has_usable_torso(landmarks: Sequence[Any]) -> bool:
    """Avoid drawing hallucinated partial poses that lack a stable body anchor."""
    ls = _norm_xy_if_visible(landmarks, _LEFT_SHOULDER)
    rs = _norm_xy_if_visible(landmarks, _RIGHT_SHOULDER)
    lh = _norm_xy_if_visible(landmarks, _LEFT_HIP)
    rh = _norm_xy_if_visible(landmarks, _RIGHT_HIP)
    shoulder_pair = ls is not None and rs is not None
    hip_count = int(lh is not None) + int(rh is not None)
    if not shoulder_pair or hip_count < 1:
        return False
    sh_w = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
    if sh_w < 0.025:
        return False
    return True


def _pose_is_confident_enough(landmarks: Sequence[Any]) -> bool:
    """Frame-level confidence gate before drawing a whole skeleton."""
    if not _pose_has_usable_torso(landmarks):
        return False

    torso_scores = [
        _joint_score(landmarks[i])
        for i in _TORSO_INDICES
        if i < len(landmarks)
    ]
    if not torso_scores:
        return False
    if sum(torso_scores) / len(torso_scores) < _POSE_FRAME_MIN_TORSO_AVG_SCORE:
        return False

    core_scores = [
        _joint_score(landmarks[i])
        for i in _CORE_BODY_INDICES
        if i < len(landmarks)
    ]
    if not core_scores:
        return False
    visible_core = sum(score >= _VISIBILITY_THRESHOLD for score in core_scores)
    if visible_core < _POSE_FRAME_MIN_CORE_VISIBLE:
        return False
    if sum(core_scores) / len(core_scores) < _POSE_FRAME_MIN_CORE_AVG_SCORE:
        return False

    return True


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
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    if timestamp_ms is None:
        result = landmarker.detect(mp_img)
    else:
        result = landmarker.detect_for_video(mp_img, timestamp_ms)

    if not result.pose_landmarks:
        return []
    # Defensive copy: list of lists of landmarks
    return [list(pl) for pl in result.pose_landmarks]


def pick_landmarks_for_scoring(
    landmarker: PoseLandmarker,
    bgr: np.ndarray,
    *,
    timestamp_ms: Optional[int],
    detection_mode: DetectionMode,
) -> Optional[List[Any]]:
    """
    Return one person's MediaPipe landmark list for geometry / comparison pipelines.

    **LEGACY_SINGLE** — uses the single pose from the landmarker (``num_poses=1``).
    **CENTER_ONLY** / **ALL_PEOPLE** — runs multi-pose detection and returns the
    **center-selected** person (same rule as ``select_center_person_index``).
    """
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]

    if detection_mode == DetectionMode.LEGACY_SINGLE:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if timestamp_ms is None:
            result = landmarker.detect(mp_img)
        else:
            result = landmarker.detect_for_video(mp_img, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return list(result.pose_landmarks[0])

    persons = extract_pose_persons(landmarker, bgr, timestamp_ms=timestamp_ms)
    if not persons:
        return None
    idx = select_center_person_index(persons, w, h)
    if idx is None:
        return persons[0]
    return persons[idx]


def _annotate_legacy_single(
    landmarker: PoseLandmarker,
    bgr: np.ndarray,
    *,
    timestamp_ms: Optional[int],
    temporal_state: Optional[PoseDrawTemporalState] = None,
    stabilization_fps: Optional[float] = None,
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
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    if timestamp_ms is None:
        result = landmarker.detect(mp_img)
    else:
        result = landmarker.detect_for_video(mp_img, timestamp_ms)

    out = bgr.copy()
    if result.pose_landmarks:
        if temporal_state is not None:
            temporal_state.begin_frame({0})
        _draw_pose_landmarks(
            out,
            result.pose_landmarks[0],
            temporal_state=temporal_state,
            person_index=0,
            stabilization_fps=stabilization_fps,
        )
        return AnnotateResult(image=out, num_people=1, center_person_index=0)
    if temporal_state is not None:
        temporal_state.begin_frame(set())
    return AnnotateResult(image=out, num_people=0, center_person_index=None)


def _draw_pose_landmarks(
    bgr: np.ndarray,
    landmarks: List[Any],
    *,
    line_color: Tuple[int, int, int] = _LINE_COLOR,
    point_color: Tuple[int, int, int] = _POINT_COLOR,
    line_thickness: int = _LINE_THICKNESS,
    point_radius: int = _POINT_RADIUS,
    temporal_state: Optional[PoseDrawTemporalState] = None,
    person_index: int = 0,
    stabilization_fps: Optional[float] = None,
) -> None:
    h, w = bgr.shape[:2]
    if not landmarks or not _pose_is_confident_enough(landmarks):
        return
    if temporal_state is not None and not temporal_state.ready_to_draw_pose(person_index):
        return

    suppress_left, suppress_right = _leg_suppression_flags(landmarks)
    compact_body = _needs_compact_body_draw(landmarks)

    segment_layers: List[Sequence[Tuple[int, int]]] = []
    if not compact_body:
        segment_layers.append(_TORSO_SEGMENTS)
    segment_layers.extend((_LEFT_ARM_SEGMENTS, _RIGHT_ARM_SEGMENTS))
    leg_layers: List[Sequence[Tuple[int, int]]] = []
    if not suppress_left:
        leg_layers.append(_LEFT_LEG_SEGMENTS)
    if not suppress_right:
        leg_layers.append(_RIGHT_LEG_SEGMENTS)
    if not compact_body:
        segment_layers.extend(leg_layers)

    all_idx: set[int] = set()
    index_layers: List[Sequence[Tuple[int, int]]] = list(segment_layers)
    if compact_body:
        index_layers.extend(leg_layers)
        index_layers.append(_TORSO_SEGMENTS)
    for layer in index_layers:
        for a, b in layer:
            all_idx.add(a)
            all_idx.add(b)

    s_fps = (
        stabilization_fps
        if stabilization_fps is not None
        else _DEFAULT_STABILIZATION_FPS
    )
    if temporal_state is not None:
        _update_limb_drop_state(
            temporal_state,
            landmarks,
            person_index=person_index,
            fps=s_fps,
            suppress_left_leg=suppress_left,
            suppress_right_leg=suppress_right,
        )

    cache = _build_joint_cache(
        landmarks, all_idx, w, h, person_index, temporal_state, s_fps
    )
    if temporal_state is not None:
        _apply_limb_drop_mask(
            cache,
            temporal_state,
            person_index=person_index,
            suppress_left_leg=suppress_left,
            suppress_right_leg=suppress_right,
        )
    cache = _decross_torso_cache_for_drawing(cache, frame_width=w)

    if compact_body:
        _draw_compact_torso_from_cache(
            bgr,
            cache,
            line_color=line_color,
            line_thickness=line_thickness,
        )
        for layer in (_LEFT_ARM_SEGMENTS, _RIGHT_ARM_SEGMENTS):
            _draw_segments_from_cache(
                bgr,
                layer,
                cache,
                line_color=line_color,
                line_thickness=line_thickness,
            )
        compact_leg_segments = tuple(seg for layer in leg_layers for seg in layer)
        _draw_compact_legs_from_cache(
            bgr,
            compact_leg_segments,
            cache,
            line_color=line_color,
            line_thickness=line_thickness,
        )
    else:
        for layer in segment_layers:
            _draw_segments_from_cache(
                bgr,
                layer,
                cache,
                line_color=line_color,
                line_thickness=line_thickness,
            )

    _draw_joint_dots_from_cache(bgr, all_idx, cache, point_color, point_radius)

    _draw_head_compass(
        bgr,
        landmarks,
        point_color=point_color,
        dot_radius=point_radius,
        temporal_state=temporal_state,
        person_index=person_index,
    )


def annotate_frame(
    landmarker: PoseLandmarker,
    bgr: np.ndarray,
    *,
    timestamp_ms: Optional[int] = None,
    mode: DetectionMode = DetectionMode.LEGACY_SINGLE,
    highlight_center_person: bool = False,
    temporal_state: Optional[PoseDrawTemporalState] = None,
    stabilization_fps: Optional[float] = None,
) -> AnnotateResult:
    """
    Run pose on one BGR frame and return overlay + detection stats.

    ``LEGACY_SINGLE`` uses the dedicated single-pose path (``num_poses=1`` landmarker).

    ``highlight_center_person``: in ``ALL_PEOPLE`` mode, draw the center-selected
    person with slightly stronger colors (intended for preview only).

    ``temporal_state``: pass a persistent ``PoseDrawTemporalState`` while encoding
    video for torso/arm/leg hold + EMA smoothing and light head compass smoothing;
    omit for single-frame preview.

    ``stabilization_fps``: source/output frame rate used to convert limb drop/hold
    timings from seconds to frames (e.g. pass writer FPS when processing video).
    """
    if bgr is None or bgr.size == 0:
        return AnnotateResult(image=bgr, num_people=0, center_person_index=None)

    if mode == DetectionMode.LEGACY_SINGLE:
        return _annotate_legacy_single(
            landmarker,
            bgr,
            timestamp_ms=timestamp_ms,
            temporal_state=temporal_state,
            stabilization_fps=stabilization_fps,
        )

    persons = extract_pose_persons(landmarker, bgr, timestamp_ms=timestamp_ms)
    h, w = bgr.shape[:2]
    out = bgr.copy()

    if not persons:
        if temporal_state is not None:
            temporal_state.begin_frame(set())
        return AnnotateResult(image=out, num_people=0, center_person_index=None)

    if temporal_state is not None:
        temporal_state.begin_frame(set(range(len(persons))))

    prev_center = temporal_state.last_center_norm if temporal_state is not None else None
    center_idx = select_center_person_index(persons, w, h, prev_center)
    if temporal_state is not None and center_idx is not None:
        temporal_state.last_center_norm = body_center_normalized(persons[center_idx])

    if mode == DetectionMode.CENTER_ONLY:
        if 0 <= center_idx < len(persons):
            _draw_pose_landmarks(
                out,
                persons[center_idx],
                temporal_state=temporal_state,
                person_index=center_idx,
                stabilization_fps=stabilization_fps,
            )
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
                temporal_state=temporal_state,
                person_index=i,
                stabilization_fps=stabilization_fps,
            )
        else:
            _draw_pose_landmarks(
                out,
                plm,
                temporal_state=temporal_state,
                person_index=i,
                stabilization_fps=stabilization_fps,
            )

    return AnnotateResult(
        image=out,
        num_people=len(persons),
        center_person_index=center_idx,
    )
