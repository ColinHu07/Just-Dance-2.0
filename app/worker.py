"""Background QThread worker for full-video pose overlay."""

from __future__ import annotations

import logging

import cv2

from PySide6.QtCore import QThread, Signal

from app import ffmpeg_audio
from app import pose_utils
from app import video_utils

logger = logging.getLogger(__name__)


class ProcessVideoWorker(QThread):
    """Reads source video, writes pose-annotated video to ``out_path``."""

    progress = Signal(int)  # 0–100, or -1 when total frame count is unknown
    status = Signal(str)
    finished_ok = Signal(str)  # output path (muxed with audio when possible)
    failed = Signal(str)
    people_detected = Signal(int)  # last frame's count (throttled)
    mux_warning = Signal(str)  # main-thread dialog for non-fatal ffmpeg / mux issues

    def __init__(
        self,
        src_path: str,
        out_path: str,
        total_frames_hint: int = 0,
        *,
        detection_mode: pose_utils.DetectionMode = pose_utils.DetectionMode.LEGACY_SINGLE,
        keep_original_audio: bool = True,
    ) -> None:
        super().__init__()
        self._src_path = src_path
        self._out_path = out_path
        self._total_frames_hint = max(0, total_frames_hint)
        self._detection_mode = detection_mode
        self._keep_original_audio = keep_original_audio

    def run(self) -> None:
        cap: cv2.VideoCapture | None = None
        writer: cv2.VideoWriter | None = None
        actual_out_path = ""
        try:
            video_utils.ensure_app_dirs()
            cap = video_utils.open_capture(self._src_path)
            meta = video_utils.read_metadata(cap, self._src_path)
            w, h = meta.width, meta.height
            out_fps, fps_note = video_utils.resolve_writer_fps(cap, meta)
            if fps_note:
                self.status.emit(fps_note)

            total = self._total_frames_hint or meta.frame_count
            if total <= 0:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            use_percent = total > 0

            wr = video_utils.create_writer(self._out_path, w, h, out_fps)
            writer = wr.writer
            ww, hh = wr.width, wr.height
            actual_out_path = wr.path
            logger.info(
                "Encoding with %s | fps=%.3f | %d×%d | %s",
                wr.codec_label,
                wr.fps,
                ww,
                hh,
                actual_out_path,
            )
            status_parts: list[str] = []
            if actual_out_path != self._out_path:
                status_parts.append(
                    f"Output file: {actual_out_path} ({wr.codec_label})"
                )
            if (ww, hh) != (w, h):
                status_parts.append(f"Output size {ww}×{hh} (source was {w}×{h})")
            if status_parts:
                self.status.emit(" — ".join(status_parts))

            pose = pose_utils.create_pose_landmarker(
                for_video=True,
                detection_mode=self._detection_mode,
            )
            frame_dt_ms = max(1, round(1000.0 / wr.fps))
            timestamp_ms = 0
            pose_draw_state = pose_utils.PoseDrawTemporalState()
            try:
                frame_index = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                while True:
                    if self.isInterruptionRequested():
                        break
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if frame.shape[1] != ww or frame.shape[0] != hh:
                        frame = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_AREA)
                    result = pose_utils.annotate_frame(
                        pose,
                        frame,
                        timestamp_ms=timestamp_ms,
                        mode=self._detection_mode,
                        highlight_center_person=False,
                        temporal_state=pose_draw_state,
                        stabilization_fps=wr.fps,
                    )
                    out = result.image
                    timestamp_ms += frame_dt_ms
                    writer.write(out)
                    frame_index += 1

                    if frame_index == 1 or frame_index % 12 == 0:
                        self.people_detected.emit(result.num_people)

                    if use_percent:
                        pct = min(100, int(100 * frame_index / total))
                        if frame_index % 2 == 0 or pct == 100:
                            self.progress.emit(pct)
                    else:
                        if frame_index % 15 == 0:
                            self.progress.emit(-1)
                            self.status.emit(f"Processed {frame_index} frames…")

                if self.isInterruptionRequested():
                    self.failed.emit("Processing was cancelled.")
                else:
                    self.progress.emit(100)
                    if writer is not None:
                        writer.release()
                        writer = None

                    if (
                        self._keep_original_audio
                        and ffmpeg_audio.find_ffmpeg()
                        and ffmpeg_audio.media_file_has_audio_stream(self._src_path)
                    ):
                        self.status.emit("Merging original audio (ffmpeg)…")

                    outcome = ffmpeg_audio.finalize_with_optional_audio(
                        original_video_path=self._src_path,
                        silent_processed_path=actual_out_path,
                        keep_original_audio=self._keep_original_audio,
                    )
                    if outcome.dialog_warning:
                        self.mux_warning.emit(outcome.dialog_warning)
                    summary = " ".join(outcome.user_messages + [f"Output: {outcome.output_path}"])
                    self.status.emit(summary)
                    self.finished_ok.emit(outcome.output_path)
            finally:
                pose.close()
        except video_utils.VideoOpenError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"Processing failed: {e}")
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
