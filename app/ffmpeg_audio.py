"""Mux processed (silent) video with audio from the original file using ffmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AudioMuxOutcome:
    """Result of merging source audio onto the OpenCV-encoded video."""

    output_path: str
    silent_intermediate_path: str
    merged_audio: bool
    user_messages: list[str]
    dialog_warning: str | None  # Short message for a non-blocking main-thread dialog

    def all_status_text(self) -> str:
        return " ".join(self.user_messages)


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _find_ffprobe() -> str | None:
    return shutil.which("ffprobe")


def media_file_has_audio_stream(path: str) -> bool:
    """Return True if the file appears to contain at least one audio stream."""
    ffprobe = _find_ffprobe()
    try:
        if ffprobe:
            r = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "csv=p=0",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode == 0 and (r.stdout or "").strip():
                return True

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return False
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        err = (r.stderr or "") + (r.stdout or "")
        return "Audio:" in err
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("Could not probe audio for %s: %s", path, e)
        return False


def _build_final_mux_path(silent_processed: Path) -> Path:
    """Place muxed file next to the silent file; keep container suffix for stream copy."""
    return silent_processed.parent / f"{silent_processed.stem}_with_audio{silent_processed.suffix}"


def _run_ffmpeg_mux(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )


def mux_processed_with_source_audio(
    *,
    original_video_path: str,
    silent_processed_path: str,
    final_output_path: str | None = None,
) -> AudioMuxOutcome:
    """
    Combine video from ``silent_processed_path`` with audio from ``original_video_path``.

    Preferred: ``-map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -shortest``
    Fallback: re-encode audio to AAC for MP4 compatibility.
    """
    silent = Path(silent_processed_path).resolve()
    original = Path(original_video_path).resolve()
    if not silent.is_file():
        return AudioMuxOutcome(
            output_path=str(silent),
            silent_intermediate_path=str(silent),
            merged_audio=False,
            user_messages=["Silent processed file missing; cannot mux audio."],
            dialog_warning=None,
        )
    if not original.is_file():
        return AudioMuxOutcome(
            output_path=str(silent),
            silent_intermediate_path=str(silent),
            merged_audio=False,
            user_messages=["Original video missing; using silent processed file."],
            dialog_warning=None,
        )

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        msg = (
            "ffmpeg was not found on PATH. Install ffmpeg to merge original audio "
            "(e.g. on macOS: brew install ffmpeg). Using silent processed video only."
        )
        logger.warning(msg)
        return AudioMuxOutcome(
            output_path=str(silent),
            silent_intermediate_path=str(silent),
            merged_audio=False,
            user_messages=[msg],
            dialog_warning="Processed video has no audio: ffmpeg not found. Install ffmpeg to merge audio.",
        )

    if not media_file_has_audio_stream(str(original)):
        msg = "Source video has no audio; exported video-only file."
        logger.info(msg)
        return AudioMuxOutcome(
            output_path=str(silent),
            silent_intermediate_path=str(silent),
            merged_audio=False,
            user_messages=[msg],
            dialog_warning=None,
        )

    out = Path(final_output_path).resolve() if final_output_path else _build_final_mux_path(silent)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() == silent.resolve():
        out = _build_final_mux_path(silent)
    try:
        if out.is_file():
            out.unlink()
    except OSError as e:
        logger.warning("Could not remove existing mux target %s: %s", out, e)

    base = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(silent),
        "-i",
        str(original),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
    ]

    attempts: list[tuple[str, list[str]]] = [
        (
            "copy video + copy audio",
            base + ["-c:v", "copy", "-c:a", "copy", str(out)],
        ),
        (
            "copy video + AAC audio",
            base + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)],
        ),
    ]

    last_err = ""
    for label, cmd in attempts:
        logger.info("ffmpeg mux try: %s", label)
        r = _run_ffmpeg_mux(cmd)
        if r.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            logger.info("ffmpeg mux succeeded (%s) -> %s", label, out)
            msg = f"Merged original audio with processed video ({label})."
            return AudioMuxOutcome(
                output_path=str(out),
                silent_intermediate_path=str(silent),
                merged_audio=True,
                user_messages=[msg],
                dialog_warning=None,
            )
        last_err = (r.stderr or r.stdout or "").strip()
        logger.warning("ffmpeg mux failed (%s): %s", label, last_err[:500] if last_err else r.returncode)

    short_err = last_err[:400] + ("…" if last_err and len(last_err) > 400 else "")
    msg = (
        "Could not merge audio with ffmpeg; using silent processed video. "
        f"Last error: {short_err or 'unknown'}"
    )
    return AudioMuxOutcome(
        output_path=str(silent),
        silent_intermediate_path=str(silent),
        merged_audio=False,
        user_messages=[msg],
        dialog_warning=(
            "Audio merge failed; your saved file is video-only (no sound). "
            "See logs for ffmpeg details."
        ),
    )


def finalize_with_optional_audio(
    *,
    original_video_path: str,
    silent_processed_path: str,
    keep_original_audio: bool,
) -> AudioMuxOutcome:
    """
    If ``keep_original_audio`` is True, run mux; otherwise return the silent path unchanged.
    """
    if not keep_original_audio:
        return AudioMuxOutcome(
            output_path=str(Path(silent_processed_path).resolve()),
            silent_intermediate_path=str(Path(silent_processed_path).resolve()),
            merged_audio=False,
            user_messages=["Original audio merge skipped (Keep Original Audio is off)."],
            dialog_warning=None,
        )
    return mux_processed_with_source_audio(
        original_video_path=original_video_path,
        silent_processed_path=silent_processed_path,
    )
