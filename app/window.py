"""Main application window: load, preview, process, play, export."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app import pose_utils
from app import ui_utils
from app import video_utils
from app.worker import ProcessVideoWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dance Pose Desktop")
        self.setMinimumSize(720, 560)
        self.setAcceptDrops(True)

        self._video_path: str | None = None
        self._meta: video_utils.VideoMetadata | None = None
        self._preview_source_bgr = None
        self._processed_path: str | None = None
        self._worker: ProcessVideoWorker | None = None
        self._play_cap: cv2.VideoCapture | None = None
        self._play_timer: QTimer | None = None
        self._play_base_fps: float = 25.0  # processed clip, for timer interval
        self._raw_first_frame = None  # BGR, for re-running preview when mode changes
        self._last_preview_annotation: pose_utils.AnnotateResult | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.path_label = QLabel("No video loaded.")
        self.path_label.setWordWrap(True)
        root.addWidget(self.path_label)

        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        root.addWidget(self.meta_label)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Detection mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(
            "Legacy Single Person",
            pose_utils.DetectionMode.LEGACY_SINGLE,
        )
        self.mode_combo.addItem(
            "All People",
            pose_utils.DetectionMode.ALL_PEOPLE,
        )
        self.mode_combo.addItem(
            "Center Person Only",
            pose_utils.DetectionMode.CENTER_ONLY,
        )
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self._on_detection_mode_changed)
        mode_row.addWidget(self.mode_combo, stretch=1)
        root.addLayout(mode_row)

        self.people_label = QLabel("People detected: —")
        self.people_label.setWordWrap(True)
        root.addWidget(self.people_label)

        self.status_label = QLabel("Load a video file to begin.")
        root.addWidget(self.status_label)

        row = QHBoxLayout()
        self.btn_load = QPushButton("Load Video")
        self.btn_process = QPushButton("Process Video")
        self.btn_process.setEnabled(False)
        self.btn_play = QPushButton("Play Processed Video")
        self.btn_play.setEnabled(False)
        self.btn_export = QPushButton("Export Processed Video")
        self.btn_export.setEnabled(False)
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_process)
        row.addWidget(self.btn_play)
        row.addWidget(self.btn_export)
        root.addLayout(row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Playback speed (processed):"))
        self.playback_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_speed_slider.setRange(25, 200)
        self.playback_speed_slider.setValue(100)
        self.playback_speed_slider.setSingleStep(1)
        self.playback_speed_slider.setPageStep(5)
        self.playback_speed_slider.setToolTip(
            "In-app preview only: 0.25× to 2.00×. Does not change exported files."
        )
        self.playback_speed_slider.setEnabled(False)
        self.playback_speed_slider.valueChanged.connect(self._on_playback_speed_changed)
        speed_row.addWidget(self.playback_speed_slider, stretch=1)
        self.playback_speed_label = QLabel("Speed: 1.00x")
        speed_row.addWidget(self.playback_speed_label)
        self.btn_speed_reset = QPushButton("1×")
        self.btn_speed_reset.setToolTip("Reset playback speed to 1.00×")
        self.btn_speed_reset.setEnabled(False)
        self.btn_speed_reset.clicked.connect(self._on_playback_speed_reset)
        speed_row.addWidget(self.btn_speed_reset)
        root.addLayout(speed_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(320)
        self.video_label.setStyleSheet("background-color: #222; color: #888;")
        self.video_label.setText("Video preview")
        root.addWidget(self.video_label, stretch=1)

        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_process.clicked.connect(self._on_process_clicked)
        self.btn_play.clicked.connect(self._on_play_clicked)
        self.btn_export.clicked.connect(self._on_export_clicked)

        self._update_playback_speed_label()

    # --- lifecycle ---

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._preview_source_bgr is not None:
            self._refresh_preview()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_playback()
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            if not self._worker.wait(8000):
                self._worker.terminate()
                self._worker.wait(2000)
        event.accept()

    # --- drag and drop ---

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self._load_video_path(path)
        event.acceptProposedAction()

    # --- slots ---

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)",
        )
        if path:
            self._load_video_path(path)

    def _on_process_clicked(self) -> None:
        if not self._video_path or self._worker is not None and self._worker.isRunning():
            return
        video_utils.ensure_app_dirs()
        out_name = f"pose_overlay_{uuid.uuid4().hex[:12]}.mp4"
        out_path = str(video_utils.TEMP_DIR / out_name)

        self._stop_playback()
        self._processed_path = None
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.playback_speed_slider.setEnabled(False)
        self.btn_speed_reset.setEnabled(False)

        total = self._meta.frame_count if self._meta else 0
        self._worker = ProcessVideoWorker(
            self._video_path,
            out_path,
            total_frames_hint=total,
            detection_mode=self._detection_mode(),
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.status.connect(self._on_worker_status)
        self._worker.people_detected.connect(self._on_worker_people_sample)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.finished.connect(self._on_worker_thread_finished)
        self._worker.failed.connect(self._on_worker_failed)

        self.btn_process.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if total <= 0:
            self.progress.setRange(0, 0)
        self.status_label.setText("Processing…")
        self._worker.start()

    def _on_play_clicked(self) -> None:
        if not self._processed_path:
            QMessageBox.information(self, "Playback", "No processed video yet. Process a video first.")
            return
        if self._play_timer is not None and self._play_timer.isActive():
            self._stop_playback()
            self.status_label.setText("Playback stopped.")
            return

        self._play_cap = cv2.VideoCapture(self._processed_path)
        if not self._play_cap.isOpened():
            QMessageBox.warning(self, "Playback", f"Could not open:\n{self._processed_path}")
            self._play_cap = None
            return

        self._play_base_fps, pb_note = video_utils.resolve_playback_fps(
            self._play_cap,
            self._meta,
        )
        interval_ms = self._compute_play_interval_ms()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._play_timer.start(interval_ms)
        msg = "Playing processed video… (click Play again to stop)"
        if pb_note:
            msg = f"{pb_note} {msg}"
        self.status_label.setText(msg)
        self.btn_play.setText("Stop Playback")

    def _on_export_clicked(self) -> None:
        if not self._processed_path or not Path(self._processed_path).is_file():
            QMessageBox.warning(self, "Export", "No processed video file to export.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export processed video",
            str(video_utils.OUTPUTS_DIR / "dance_pose_overlay.mp4"),
            "MP4 video (*.mp4);;All files (*.*)",
        )
        if not dest:
            return
        try:
            shutil.copy2(self._processed_path, dest)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self.status_label.setText(f"Exported to: {dest}")
        QMessageBox.information(self, "Export", f"Saved:\n{dest}")

    def _on_worker_progress(self, value: int) -> None:
        if value < 0:
            self.progress.setRange(0, 0)
            return
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        self.progress.setValue(value)

    def _on_worker_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_worker_finished(self, path: str) -> None:
        self._processed_path = path
        self.btn_play.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.playback_speed_slider.setEnabled(True)
        self.btn_speed_reset.setEnabled(True)
        self.status_label.setText(f"Done. Output: {path}")

    def _on_worker_people_sample(self, n: int) -> None:
        self.people_label.setText(f"Processing… last sampled frame: {n} people")

    def _on_detection_mode_changed(self, _index: int) -> None:
        if self._raw_first_frame is not None:
            self._refresh_pose_preview()

    def _on_worker_failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if message != "Processing was cancelled.":
            QMessageBox.warning(self, "Processing", message)
        self.status_label.setText("Processing failed." if message != "Processing was cancelled." else "Cancelled.")

    def _on_worker_thread_finished(self) -> None:
        self._worker = None
        self.btn_process.setEnabled(self._video_path is not None)
        self.btn_load.setEnabled(True)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        self._refresh_people_label_from_preview()

    def _on_play_tick(self) -> None:
        if self._play_cap is None:
            return
        ret, frame = self._play_cap.read()
        if not ret or frame is None:
            self._stop_playback()
            self.status_label.setText("Playback finished.")
            return
        self._set_preview_from_bgr(frame)

    # --- helpers ---

    def _load_video_path(self, path: str) -> None:
        self._stop_playback()
        cap = None
        try:
            cap = video_utils.open_capture(path)
            meta = video_utils.read_metadata(cap, path)
            first = video_utils.read_first_frame(cap)
            if first is None:
                raise video_utils.VideoOpenError("Video opened but first frame could not be read.")
        except video_utils.VideoOpenError as e:
            QMessageBox.warning(self, "Could not load video", str(e))
            self.status_label.setText("Load failed.")
            self._raw_first_frame = None
            self._last_preview_annotation = None
            return
        finally:
            if cap is not None:
                cap.release()

        self._video_path = str(Path(path).resolve())
        self._meta = meta
        self._processed_path = None
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.playback_speed_slider.setEnabled(False)
        self.btn_speed_reset.setEnabled(False)
        self.btn_play.setText("Play Processed Video")
        self.btn_process.setEnabled(True)

        self.path_label.setText(self._video_path)
        self.meta_label.setText(meta.summary())

        self._raw_first_frame = first.copy()
        self._refresh_pose_preview()
        self.status_label.setText(
            "Preview shows pose overlay on the first frame. Click Process Video to render the full clip."
        )

    def _detection_mode(self) -> pose_utils.DetectionMode:
        data = self.mode_combo.currentData()
        if isinstance(data, pose_utils.DetectionMode):
            return data
        if isinstance(data, str):
            return pose_utils.DetectionMode(data)
        return pose_utils.DetectionMode.LEGACY_SINGLE

    def _refresh_pose_preview(self) -> None:
        if self._raw_first_frame is None:
            return
        try:
            mode = self._detection_mode()
            pose = pose_utils.create_pose_landmarker(
                for_video=False,
                detection_mode=mode,
            )
            try:
                highlight = mode == pose_utils.DetectionMode.ALL_PEOPLE
                result = pose_utils.annotate_frame(
                    pose,
                    self._raw_first_frame,
                    timestamp_ms=None,
                    mode=mode,
                    highlight_center_person=highlight,
                )
            finally:
                pose.close()
        except RuntimeError as e:
            QMessageBox.critical(self, "Pose model", str(e))
            result = pose_utils.AnnotateResult(
                image=self._raw_first_frame.copy(),
                num_people=0,
                center_person_index=None,
            )

        self._last_preview_annotation = result
        self._set_people_label_from_result(result, preview=True)
        self._set_preview_from_bgr(result.image)

    def _set_people_label_from_result(
        self,
        result: pose_utils.AnnotateResult,
        *,
        preview: bool,
    ) -> None:
        prefix = "Preview (first frame): " if preview else ""
        mode = self._detection_mode()
        if mode == pose_utils.DetectionMode.LEGACY_SINGLE:
            if result.num_people == 0:
                self.people_label.setText(
                    f"{prefix}No pose detected (legacy single-pose, num_poses=1)"
                )
            else:
                self.people_label.setText(
                    f"{prefix}1 pose overlaid (legacy single-pose; not multi-counted)"
                )
            return

        if result.num_people == 0:
            self.people_label.setText(f"{prefix}0 people detected")
            return
        k = result.center_person_index
        human_k = (k + 1) if k is not None else "?"
        if mode == pose_utils.DetectionMode.CENTER_ONLY:
            self.people_label.setText(
                f"{prefix}{result.num_people} people detected "
                f"(showing center person #{human_k})"
            )
        else:
            self.people_label.setText(
                f"{prefix}{result.num_people} people detected "
                f"(center highlight: person #{human_k})"
            )

    def _refresh_people_label_from_preview(self) -> None:
        if self._last_preview_annotation is not None:
            self._set_people_label_from_result(
                self._last_preview_annotation,
                preview=True,
            )
        elif self._raw_first_frame is None:
            self.people_label.setText("People detected: —")

    def _set_preview_from_bgr(self, bgr) -> None:
        self._preview_source_bgr = bgr
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self._preview_source_bgr is None:
            return
        pm = ui_utils.bgr_to_qpixmap(self._preview_source_bgr)
        if pm.isNull():
            return
        target = self.video_label.size()
        if target.width() < 10 or target.height() < 10:
            return
        scaled = pm.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self.video_label.setText("")

    def _stop_playback(self) -> None:
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer.deleteLater()
            self._play_timer = None
        if self._play_cap is not None:
            self._play_cap.release()
            self._play_cap = None
        self.btn_play.setText("Play Processed Video")

    def _playback_speed_ratio(self) -> float:
        v = self.playback_speed_slider.value() / 100.0
        return max(0.25, min(2.0, v))

    def _compute_play_interval_ms(self) -> int:
        """QTimer interval: real-time frame duration divided by user speed factor."""
        speed = self._playback_speed_ratio()
        effective_fps = self._play_base_fps * speed
        return max(1, round(1000.0 / effective_fps))

    def _update_playback_speed_label(self) -> None:
        self.playback_speed_label.setText(
            f"Speed: {self._playback_speed_ratio():.2f}x"
        )

    def _on_playback_speed_changed(self, _value: int) -> None:
        self._update_playback_speed_label()
        if self._play_timer is not None and self._play_timer.isActive():
            self._play_timer.setInterval(self._compute_play_interval_ms())

    def _on_playback_speed_reset(self) -> None:
        self.playback_speed_slider.setValue(100)
