"""Video I/O and metadata helpers (OpenCV)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# OpenCV sometimes returns garbage FPS; keep resolved values in a sane range.
_FPS_MIN = 1.0
_FPS_MAX = 120.0
_FALLBACK_FPS = 25.0


class VideoOpenError(Exception):
    """Raised when a video file cannot be opened or read."""


@dataclass
class VideoMetadata:
    path: str
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_sec(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps

    def summary(self) -> str:
        fc_str = str(self.frame_count) if self.frame_count > 0 else "?"
        if self.frame_count > 0 and self.fps > 0:
            dur_str = f"{self.duration_sec:.1f}s"
        else:
            dur_str = "?"
        return (
            f"{self.filename}  |  {self.width}×{self.height}  |  "
            f"{self.fps:.2f} fps  |  {fc_str} frames  |  {dur_str}"
        )


def ensure_app_dirs() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def open_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoOpenError(f"Could not open video:\n{path}")
    return cap


def read_metadata(cap: cv2.VideoCapture, path: str) -> VideoMetadata:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fc < 0:
        fc = 0
    p = Path(path)
    return VideoMetadata(
        path=str(p.resolve()),
        filename=p.name,
        width=w,
        height=h,
        fps=fps,
        frame_count=max(0, fc),
    )


def read_first_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


def _fps_is_plausible(fps: float) -> bool:
    return _FPS_MIN <= fps <= _FPS_MAX


def estimate_fps_from_frame_gaps(cap: cv2.VideoCapture, max_reads: int = 48) -> float:
    """
    When container FPS is missing, estimate from ``CAP_PROP_POS_MSEC`` deltas
    over consecutive reads. Leaves the capture at frame 0.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    deltas_ms: list[float] = []
    prev_m = -1.0
    for _ in range(max_reads):
        m = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        ret, _ = cap.read()
        if not ret:
            break
        if prev_m >= 0.0 and m > prev_m:
            deltas_ms.append(m - prev_m)
        prev_m = m
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not deltas_ms:
        return 0.0
    deltas_ms.sort()
    mid = deltas_ms[len(deltas_ms) // 2]
    if mid < 1.0:
        return 0.0
    fps = 1000.0 / mid
    if not _fps_is_plausible(fps):
        return 0.0
    return fps


def resolve_writer_fps(cap: cv2.VideoCapture, meta: VideoMetadata) -> tuple[float, str]:
    """
    Pick an output FPS as close to the source as OpenCV allows.

    Returns ``(fps, user_message)`` where ``user_message`` is non-empty if a
    fallback or estimate was used (caller may show in status).
    """
    candidates: list[float] = [
        float(meta.fps),
        float(cap.get(cv2.CAP_PROP_FPS)),
    ]
    for c in candidates:
        if _fps_is_plausible(c):
            return c, ""

    est = estimate_fps_from_frame_gaps(cap)
    if est > 0.0:
        return est, (
            f"Source FPS missing — estimated {est:.2f} fps from frame timestamps for export."
        )

    return _FALLBACK_FPS, (
        f"Source FPS missing — using {_FALLBACK_FPS:.0f} fps fallback for export "
        "(duration may not match the original file)."
    )


def resolve_playback_fps(
    cap: cv2.VideoCapture,
    source_meta: Optional[VideoMetadata] = None,
) -> tuple[float, str]:
    """
    FPS for in-app timer playback of an already-encoded file.
    Prefer the file's metadata; then source metadata; then estimate; then fallback.
    """
    file_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if _fps_is_plausible(file_fps):
        return file_fps, ""

    if source_meta is not None and _fps_is_plausible(float(source_meta.fps)):
        return float(source_meta.fps), (
            "Processed file has no FPS tag — using loaded source FPS for playback timer."
        )

    est = estimate_fps_from_frame_gaps(cap)
    if est > 0.0:
        return est, "Processed FPS unclear — estimated from frames for playback timer."

    return _FALLBACK_FPS, (
        f"Using {_FALLBACK_FPS:.0f} fps fallback for playback timer."
    )


def create_writer(
    out_path: str, width: int, height: int, fps: float
) -> tuple[cv2.VideoWriter, int, int]:
    """Create a VideoWriter; may adjust dimensions slightly for codec compatibility."""
    ensure_app_dirs()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    safe_fps = float(fps) if _fps_is_plausible(fps) else _FALLBACK_FPS
    w, h = int(width), int(height)
    writer = cv2.VideoWriter(out_path, fourcc, safe_fps, (w, h))
    if not writer.isOpened():
        raise VideoOpenError(f"Could not create output video:\n{out_path}")
    return writer, w, h
