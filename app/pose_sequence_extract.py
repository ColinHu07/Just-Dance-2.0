"""Extract a time-ordered pose sequence from a video file (single dancer)."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import cv2

from app import pose_utils
from app import video_utils
from app.comparison_types import PoseFrame, PoseSequence
from app.normalization import normalize_landmarks_to_xy

logger = logging.getLogger(__name__)

# Scoring uses center-selected person for stable main-dancer tracking.
_SCORING_DETECTION_MODE = pose_utils.DetectionMode.CENTER_ONLY


def extract_pose_sequence_from_video(
    path: str,
    *,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> PoseSequence:
    """
    Decode ``path``, run MediaPipe pose each frame, return normalized ``PoseSequence``.

    ``progress_cb(percent_int, message)`` is optional (percent 0–100).
    ``cancel_check`` returns True to abort early (raises ``RuntimeError``).
    """
    video_utils.ensure_app_dirs()
    cap = video_utils.open_capture(path)
    try:
        meta = video_utils.read_metadata(cap, path)
        out_fps, fps_note = video_utils.resolve_writer_fps(cap, meta)
        if fps_note and progress_cb:
            progress_cb(0, fps_note)

        w, h = meta.width, meta.height
        total = meta.frame_count if meta.frame_count > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(1, total)

        landmarker = pose_utils.create_pose_landmarker(
            for_video=True,
            detection_mode=_SCORING_DETECTION_MODE,
        )
        frame_dt_ms = max(1, round(1000.0 / out_fps))
        timestamp_ms = 0
        frames: List[PoseFrame] = []
        frame_index = 0

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("Cancelled.")
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break
                if bgr.shape[1] != w or bgr.shape[0] != h:
                    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

                lms = pose_utils.pick_landmarks_for_scoring(
                    landmarker,
                    bgr,
                    timestamp_ms=timestamp_ms,
                    detection_mode=_SCORING_DETECTION_MODE,
                )
                xy, rel = normalize_landmarks_to_xy(lms)
                t_sec = frame_index / out_fps if out_fps > 0 else 0.0
                frames.append(
                    PoseFrame(
                        frame_index=frame_index,
                        time_sec=t_sec,
                        image_width=w,
                        image_height=h,
                        landmarks_raw=lms,
                        joints_norm_xy=xy,
                        reliability=rel,
                    )
                )
                timestamp_ms += frame_dt_ms
                frame_index += 1
                if progress_cb and (frame_index % 3 == 0 or frame_index == total):
                    pct = min(100, int(100 * frame_index / total))
                    progress_cb(pct, f"Pose frames: {frame_index}/{total}")
        finally:
            landmarker.close()

        return PoseSequence(
            source_path=str(path),
            fps=float(out_fps),
            frames=frames,
            video_width=w,
            video_height=h,
        )
    finally:
        cap.release()
