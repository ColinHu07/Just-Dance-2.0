"""Video I/O and metadata helpers (OpenCV)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DANCE_LIBRARY_DIR = PROJECT_ROOT / "dance_library"

logger = logging.getLogger(__name__)

# OpenCV sometimes returns garbage FPS; keep resolved values in a sane range.
_FPS_MIN = 1.0
_FPS_MAX = 120.0
_FALLBACK_FPS = 30.0


class VideoOpenError(Exception):
    """Raised when a video file cannot be opened or read."""


@dataclass
class RobustVideoWriterResult:
    """Open VideoWriter after platform fallbacks; ``path`` is the file actually opened."""

    writer: cv2.VideoWriter
    path: str
    width: int
    height: int
    fps: float
    codec_label: str  # e.g. "mp4v + .mp4"


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
    DANCE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)


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


def read_frame_at_index(path: str, frame_index: int) -> Optional[np.ndarray]:
    """Open ``path``, seek to ``frame_index``, read one BGR frame, then close."""
    cap = open_capture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame
    finally:
        cap.release()


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


def _normalize_writer_dimensions(width: int, height: int) -> tuple[int, int, bool]:
    """Ensure positive even dimensions for codec compatibility; may shrink by one pixel per axis."""
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise VideoOpenError(
            f"Invalid output size for video writer: width={width!r}, height={height!r} "
            "(both must be integers > 0)."
        )
    w0, h0 = w, h
    if w % 2:
        w = max(2, w - 1)
    if h % 2:
        h = max(2, h - 1)
    adjusted = (w, h) != (w0, h0)
    return w, h, adjusted


def _sanitize_writer_fps(fps: float) -> tuple[float, bool]:
    """Return FPS safe for VideoWriter; use 30.0 if missing or out of range."""
    f = float(fps)
    if f > 0 and _fps_is_plausible(f):
        return f, False
    logger.warning(
        "Output FPS invalid or zero (%r); using %.1f for encoding.",
        fps,
        _FALLBACK_FPS,
    )
    return float(_FALLBACK_FPS), True


def create_writer(
    out_path: str, width: int, height: int, fps: float
) -> RobustVideoWriterResult:
    """
    Create a VideoWriter with macOS-friendly codec/container fallbacks.

    Tries, in order: mp4v+.mp4, avc1+.mp4, MJPG+.avi, XVID+.avi.
    Ensures parent directories exist, validates FPS and frame size, and normalizes
    to even width/height when needed.
    """
    ensure_app_dirs()
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    safe_fps, fps_fallback = _sanitize_writer_fps(fps)
    w, h, size_adjusted = _normalize_writer_dimensions(width, height)
    if size_adjusted:
        logger.info(
            "Adjusted output dimensions to even size for codec: %s×%s (requested %s×%s).",
            w,
            h,
            int(width),
            int(height),
        )

    stem = out.stem
    parent = out.parent
    # Ordered fallbacks for macOS OpenCV builds where mp4v/mp4 may fail.
    attempts: list[tuple[str, str, str]] = [
        ("mp4v", ".mp4", "mp4v + MP4 (.mp4)"),
        ("avc1", ".mp4", "avc1 + MP4 (.mp4)"),
        ("MJPG", ".avi", "MJPG + AVI (.avi)"),
        ("XVID", ".avi", "XVID + AVI (.avi)"),
    ]

    failed_lines: list[str] = []
    for fourcc_chars, ext, label in attempts:
        candidate = parent / f"{stem}{ext}"
        path_str = str(candidate)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_chars)
        writer = cv2.VideoWriter(path_str, fourcc, safe_fps, (w, h))
        opened = writer.isOpened()
        if opened:
            logger.info(
                "VideoWriter opened OK: %s | path=%s | fps=%.3f | size=%d×%d",
                label,
                path_str,
                safe_fps,
                w,
                h,
            )
            return RobustVideoWriterResult(
                writer=writer,
                path=path_str,
                width=w,
                height=h,
                fps=safe_fps,
                codec_label=label,
            )
        writer.release()
        msg = f"{label} @ {path_str}"
        logger.warning("VideoWriter failed: %s", msg)
        failed_lines.append(msg)

    detail = "; ".join(failed_lines)
    raise VideoOpenError(
        "Could not create output video after trying all codec/container fallbacks.\n"
        f"Attempts: {detail}\n"
        f"fps={safe_fps} (fallback_used={fps_fallback}), frame_size={w}×{h}, "
        f"requested_path={out_path}"
    )
