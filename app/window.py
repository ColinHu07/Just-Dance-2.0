"""Main application window: load, preview, process, play, export."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app import comparison_view
from app.dance_library import (
    delete_dance,
    list_dances,
    load_dance,
    save_dance_from_reference,
)
from app import pose_utils
from app import ui_utils
from app import video_utils
from app.comparison_types import ComparisonResult, PoseSequence
from app.comparison_worker import (
    CompareSequencesWorker,
    ExtractSequenceWorker,
    save_comparison_json,
)
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
        self._play_base_fps: float = 30.0  # processed clip, for timer interval (fallback only)
        self._raw_first_frame = None  # BGR, for re-running preview when mode changes
        self._last_preview_annotation: pose_utils.AnnotateResult | None = None

        self._ref_path: str | None = None
        self._user_path: str | None = None
        self._ref_meta: video_utils.VideoMetadata | None = None
        self._user_meta: video_utils.VideoMetadata | None = None
        self._ref_sequence: PoseSequence | None = None
        self._user_sequence: PoseSequence | None = None
        self._last_comparison: ComparisonResult | None = None
        self._extract_worker: ExtractSequenceWorker | None = None
        self._compare_worker: CompareSequencesWorker | None = None
        self._extract_target: str = "ref"

        self._ref_preview_bgr = None
        self._user_preview_bgr = None
        self._overlay_display_bgr = None
        self._overlay_pairs: np.ndarray | None = None
        self._overlay_cap_ref: cv2.VideoCapture | None = None
        self._overlay_cap_user: cv2.VideoCapture | None = None
        self._overlay_play_idx: int = 0
        self._overlay_play_timer: QTimer | None = None

        self._active_library_dance_id: str | None = None
        self._active_library_dance_name: str | None = None
        self._ref_practice_cap: cv2.VideoCapture | None = None
        self._ref_practice_timer: QTimer | None = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root = QVBoxLayout(inner)
        scroll.setWidget(inner)
        self.setCentralWidget(scroll)
        self._scroll_area = scroll

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

        audio_row = QHBoxLayout()
        self.keep_audio_checkbox = QCheckBox("Keep Original Audio")
        self.keep_audio_checkbox.setChecked(True)
        self.keep_audio_checkbox.setToolTip(
            "After OpenCV finishes the silent overlay clip, ffmpeg (if installed) can mux in "
            "the original soundtrack. Turn off to keep the processed file video-only."
        )
        audio_row.addWidget(self.keep_audio_checkbox)
        audio_row.addStretch(1)
        root.addLayout(audio_row)

        self.preview_audio_note = QLabel(
            "In-app playback is silent (frame preview only). With “Keep Original Audio” on, "
            "the saved processed file includes sound — open it in QuickTime or another player to hear it."
        )
        self.preview_audio_note.setWordWrap(True)
        self.preview_audio_note.setStyleSheet("color: #888;")
        root.addWidget(self.preview_audio_note)

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

        compare_box = QGroupBox("Dance comparison (reference vs your performance)")
        compare_layout = QVBoxLayout(compare_box)
        self.compare_ref_label = QLabel("Reference: (not loaded)")
        self.compare_ref_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_ref_label)
        self.compare_user_label = QLabel("User: (not loaded)")
        self.compare_user_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_user_label)

        lib_box = QGroupBox("Dance library (save once, reuse for practice & scoring)")
        lib_layout = QVBoxLayout(lib_box)
        lib_row1 = QHBoxLayout()
        lib_row1.addWidget(QLabel("Saved dances:"))
        self.library_combo = QComboBox()
        self.library_combo.setMinimumWidth(240)
        lib_row1.addWidget(self.library_combo, stretch=1)
        self.btn_save_dance = QPushButton("Save Current Reference as Dance")
        self.btn_load_library_dance = QPushButton("Load Selected Dance")
        self.btn_delete_library_dance = QPushButton("Delete Selected")
        lib_row1.addWidget(self.btn_save_dance)
        lib_row1.addWidget(self.btn_load_library_dance)
        lib_row1.addWidget(self.btn_delete_library_dance)
        lib_layout.addLayout(lib_row1)
        lib_row2 = QHBoxLayout()
        self.mirror_practice_checkbox = QCheckBox("Mirror reference for practice")
        self.mirror_scoring_checkbox = QCheckBox("Mirror reference for scoring")
        self.mirror_practice_checkbox.setChecked(True)
        self.mirror_scoring_checkbox.setChecked(True)
        self.mirror_practice_checkbox.setToolTip(
            "Horizontally flip the reference video in the preview, practice playback, and aligned overlay."
        )
        self.mirror_scoring_checkbox.setToolTip(
            "Compare your pose against a horizontally mirrored reference (swap left/right joints + flip x)."
        )
        lib_row2.addWidget(self.mirror_practice_checkbox)
        lib_row2.addWidget(self.mirror_scoring_checkbox)
        self.btn_ref_practice_play = QPushButton("Play reference (practice)")
        self.btn_ref_practice_play.setEnabled(False)
        self.btn_ref_practice_play.setToolTip("Watch the reference clip in the Reference preview (silent).")
        lib_row2.addWidget(self.btn_ref_practice_play)
        lib_row2.addStretch(1)
        lib_layout.addLayout(lib_row2)
        self.flow_mode_label = QLabel(
            "Practice: watch the reference. Load your video, process both sides, then compare to score."
        )
        self.flow_mode_label.setWordWrap(True)
        self.flow_mode_label.setStyleSheet("color: #aaa;")
        lib_layout.addWidget(self.flow_mode_label)
        compare_layout.addWidget(lib_box)

        dual_row = QHBoxLayout()
        ref_col = QVBoxLayout()
        ref_col.addWidget(QLabel("<b>Reference</b>"))
        self.ref_preview_label = QLabel()
        self.ref_preview_label.setMinimumHeight(200)
        self.ref_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ref_preview_label.setStyleSheet("background-color: #222; color: #888;")
        self.ref_preview_label.setText("No video loaded")
        self.ref_preview_status = QLabel("Load a reference file to preview frame 0.")
        self.ref_preview_status.setWordWrap(True)
        ref_col.addWidget(self.ref_preview_label, stretch=1)
        ref_col.addWidget(self.ref_preview_status)
        user_col = QVBoxLayout()
        user_col.addWidget(QLabel("<b>User</b>"))
        self.user_preview_label = QLabel()
        self.user_preview_label.setMinimumHeight(200)
        self.user_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.user_preview_label.setStyleSheet("background-color: #222; color: #888;")
        self.user_preview_label.setText("No video loaded")
        self.user_preview_status = QLabel("Load a user performance file to preview frame 0.")
        self.user_preview_status.setWordWrap(True)
        user_col.addWidget(self.user_preview_label, stretch=1)
        user_col.addWidget(self.user_preview_status)
        dual_row.addLayout(ref_col, stretch=1)
        dual_row.addLayout(user_col, stretch=1)
        compare_layout.addLayout(dual_row)

        cmp_row = QHBoxLayout()
        self.btn_load_ref = QPushButton("Load Reference Video")
        self.btn_load_user = QPushButton("Load User Video")
        self.btn_process_ref = QPushButton("Process Reference")
        self.btn_process_user = QPushButton("Process User")
        self.btn_compare = QPushButton("Compare Videos")
        self.btn_export_scores = QPushButton("Export scores JSON")
        self.btn_process_ref.setEnabled(False)
        self.btn_process_user.setEnabled(False)
        self.btn_compare.setEnabled(False)
        self.btn_export_scores.setEnabled(False)
        cmp_row.addWidget(self.btn_load_ref)
        cmp_row.addWidget(self.btn_load_user)
        cmp_row.addWidget(self.btn_process_ref)
        cmp_row.addWidget(self.btn_process_user)
        cmp_row.addWidget(self.btn_compare)
        cmp_row.addWidget(self.btn_export_scores)
        compare_layout.addLayout(cmp_row)
        self.compare_status_label = QLabel("Load both videos, process each into pose data, then compare.")
        self.compare_status_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_status_label)
        self.compare_scores_label = QLabel("")
        self.compare_scores_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_scores_label)
        self.compare_explain = QTextEdit()
        self.compare_explain.setReadOnly(True)
        self.compare_explain.setMaximumHeight(88)
        self.compare_explain.setPlaceholderText("Comparison notes will appear here.")
        compare_layout.addWidget(self.compare_explain)

        self.overlay_group = QGroupBox("Aligned overlay view (DTW time alignment)")
        overlay_layout = QVBoxLayout(self.overlay_group)
        overlay_layout.addWidget(
            QLabel(
                "After Compare Videos: reference frame (opaque) + user frame (~40% opacity). "
                "Pairs use the DTW alignment path, not raw frame numbers."
            )
        )
        self.overlay_video_label = QLabel()
        self.overlay_video_label.setMinimumHeight(260)
        self.overlay_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_video_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        self.overlay_video_label.setText("Run Compare Videos to build the aligned overlay.")
        overlay_layout.addWidget(self.overlay_video_label)
        overlay_btns = QHBoxLayout()
        self.btn_overlay_play = QPushButton("Play overlay")
        self.btn_overlay_play.setEnabled(False)
        self.btn_overlay_play.setToolTip(
            "Play through DTW-aligned frame pairs (subsampled for smooth preview)."
        )
        self.btn_overlay_play.clicked.connect(self._on_overlay_play_clicked)
        self.overlay_frame_info = QLabel("")
        self.overlay_frame_info.setWordWrap(True)
        overlay_btns.addWidget(self.btn_overlay_play)
        overlay_btns.addWidget(self.overlay_frame_info, stretch=1)
        overlay_layout.addLayout(overlay_btns)
        self.overlay_group.setVisible(False)
        compare_layout.addWidget(self.overlay_group)

        root.addWidget(compare_box)

        self.btn_load_ref.clicked.connect(self._on_load_ref_clicked)
        self.btn_load_user.clicked.connect(self._on_load_user_clicked)
        self.btn_process_ref.clicked.connect(self._on_process_ref_clicked)
        self.btn_process_user.clicked.connect(self._on_process_user_clicked)
        self.btn_compare.clicked.connect(self._on_compare_clicked)
        self.btn_export_scores.clicked.connect(self._on_export_scores_clicked)

        self.btn_save_dance.clicked.connect(self._on_save_dance_clicked)
        self.btn_load_library_dance.clicked.connect(self._on_load_library_dance_clicked)
        self.btn_delete_library_dance.clicked.connect(self._on_delete_library_dance_clicked)
        self.btn_ref_practice_play.clicked.connect(self._on_ref_practice_play_clicked)
        self.mirror_practice_checkbox.stateChanged.connect(self._on_mirror_practice_changed)

        root.addWidget(QLabel("<b>Pose overlay preview (main workflow)</b>"))
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(260)
        self.video_label.setStyleSheet("background-color: #222; color: #888;")
        self.video_label.setText("Video preview")
        root.addWidget(self.video_label, stretch=1)

        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_process.clicked.connect(self._on_process_clicked)
        self.btn_play.clicked.connect(self._on_play_clicked)
        self.btn_export.clicked.connect(self._on_export_clicked)

        self._update_playback_speed_label()

        self._refresh_library_combo()
        self._set_compare_ui_busy(False)

    # --- lifecycle ---

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._preview_source_bgr is not None:
            self._refresh_preview()
        self._refresh_comparison_preview_panels()
        if self._overlay_display_bgr is not None:
            self._refresh_overlay_preview()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_playback()
        self._stop_ref_practice_playback()
        self._teardown_overlay()
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            if not self._worker.wait(8000):
                self._worker.terminate()
                self._worker.wait(2000)
        if self._extract_worker is not None and self._extract_worker.isRunning():
            self._extract_worker.requestInterruption()
            self._extract_worker.wait(6000)
        if self._compare_worker is not None and self._compare_worker.isRunning():
            self._compare_worker.requestInterruption()
            self._compare_worker.wait(6000)
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
        if self._comparison_workers_busy():
            QMessageBox.information(
                self,
                "Busy",
                "Finish pose extraction or comparison first, then process the overlay video.",
            )
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
            keep_original_audio=self.keep_audio_checkbox.isChecked(),
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.status.connect(self._on_worker_status)
        self._worker.people_detected.connect(self._on_worker_people_sample)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.finished.connect(self._on_worker_thread_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.mux_warning.connect(self._on_mux_warning)

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

        self._stop_ref_practice_playback()
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
        msg = (
            "Playing processed video (no sound in-app)… "
            "(click Play again to stop)"
        )
        if pb_note:
            msg = f"{pb_note} {msg}"
        self.status_label.setText(msg)
        self.btn_play.setText("Stop Playback")

    def _on_export_clicked(self) -> None:
        if not self._processed_path or not Path(self._processed_path).is_file():
            QMessageBox.warning(self, "Export", "No processed video file to export.")
            return
        processed = Path(self._processed_path)
        default_name = f"dance_pose_overlay{processed.suffix}"
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export processed video",
            str(video_utils.OUTPUTS_DIR / default_name),
            "Video (*.mp4 *.avi);;MP4 (*.mp4);;AVI (*.avi);;All files (*.*)",
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
        # Final status text is set by the worker (includes mux / output path).

    def _on_mux_warning(self, message: str) -> None:
        QMessageBox.warning(self, "Audio merge", message)

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

    # --- dance comparison ---

    def _comparison_workers_busy(self) -> bool:
        ex = self._extract_worker is not None and self._extract_worker.isRunning()
        co = self._compare_worker is not None and self._compare_worker.isRunning()
        return ex or co

    def _set_compare_ui_busy(self, busy: bool) -> None:
        if busy:
            self._stop_overlay_playback()
            self.btn_overlay_play.setEnabled(False)
        self.btn_load_ref.setEnabled(not busy)
        self.btn_load_user.setEnabled(not busy)
        self.btn_process_ref.setEnabled(not busy and self._ref_path is not None)
        self.btn_process_user.setEnabled(not busy and self._user_path is not None)
        self.btn_compare.setEnabled(
            not busy
            and self._ref_sequence is not None
            and self._user_sequence is not None
        )
        self.btn_export_scores.setEnabled(
            not busy and self._last_comparison is not None
        )
        self.btn_save_dance.setEnabled(not busy and self._ref_sequence is not None)
        self.btn_load_library_dance.setEnabled(not busy)
        self.btn_delete_library_dance.setEnabled(not busy and self.library_combo.count() > 0)
        self.btn_ref_practice_play.setEnabled(
            not busy and self._ref_path is not None and Path(self._ref_path).is_file()
        )
        if not busy:
            ov = (
                self._overlay_pairs is not None
                and self._overlay_pairs.size > 0
            )
            self.btn_overlay_play.setEnabled(bool(ov))

    def _on_load_ref_clicked(self) -> None:
        if self._comparison_workers_busy():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open reference video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)",
        )
        if not path:
            return
        self._load_comparison_video(path, is_reference=True)

    def _on_load_user_clicked(self) -> None:
        if self._comparison_workers_busy():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open user performance video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)",
        )
        if not path:
            return
        self._load_comparison_video(path, is_reference=False)

    def _load_comparison_video(self, path: str, *, is_reference: bool) -> None:
        cap = None
        try:
            cap = video_utils.open_capture(path)
            meta = video_utils.read_metadata(cap, path)
            first = video_utils.read_first_frame(cap)
        except video_utils.VideoOpenError as e:
            QMessageBox.warning(self, "Could not load video", str(e))
            return
        finally:
            if cap is not None:
                cap.release()

        self._teardown_overlay()
        self.overlay_group.setVisible(False)

        resolved = str(Path(path).resolve())
        if is_reference:
            self._stop_ref_practice_playback()
            self._active_library_dance_id = None
            self._active_library_dance_name = None
            self._ref_path = resolved
            self._ref_meta = meta
            self._ref_sequence = None
            self._ref_preview_bgr = first.copy() if first is not None else None
            self.compare_ref_label.setText(
                f"Reference: {resolved}\n{meta.summary()} — pose not extracted yet."
            )
            self.ref_preview_status.setText(
                f"Showing frame 0  ·  {meta.summary()}"
                if first is not None
                else "Could not read frame 0."
            )
        else:
            self._user_path = resolved
            self._user_meta = meta
            self._user_sequence = None
            self._user_preview_bgr = first.copy() if first is not None else None
            self.compare_user_label.setText(
                f"User: {resolved}\n{meta.summary()} — pose not extracted yet."
            )
            self.user_preview_status.setText(
                f"Showing frame 0  ·  {meta.summary()}"
                if first is not None
                else "Could not read frame 0."
            )
        self._last_comparison = None
        self.compare_scores_label.setText("")
        self.compare_explain.clear()
        self._refresh_comparison_preview_panels()
        QTimer.singleShot(0, self._refresh_comparison_preview_panels)
        self._set_compare_ui_busy(False)
        self.compare_status_label.setText(
            "Click Process Reference / Process User to extract poses, then Compare Videos."
        )

    def _on_process_ref_clicked(self) -> None:
        if not self._ref_path or self._comparison_workers_busy():
            return
        self._extract_target = "ref"
        self._start_extract_sequence(self._ref_path, "reference")

    def _on_process_user_clicked(self) -> None:
        if not self._user_path or self._comparison_workers_busy():
            return
        self._extract_target = "user"
        self._start_extract_sequence(self._user_path, "user")

    def _start_extract_sequence(self, path: str, label: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Wait for video overlay processing to finish before extracting poses.",
            )
            return
        self._extract_worker = ExtractSequenceWorker(path, label=label)
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.finished_ok.connect(self._on_extract_finished)
        self._extract_worker.failed.connect(self._on_extract_failed)
        self._extract_worker.finished.connect(self._on_extract_thread_finished)
        self._set_compare_ui_busy(True)
        self.btn_process.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.compare_status_label.setText(f"Extracting pose sequence ({label})…")
        self._extract_worker.start()

    def _on_extract_progress(self, pct: int, message: str) -> None:
        self.progress.setValue(max(0, min(100, pct)))
        self.compare_status_label.setText(message)

    def _on_extract_finished(self, seq: PoseSequence) -> None:
        n = len(seq.frames)
        if self._extract_target == "ref":
            self._ref_sequence = seq
            meta_s = self._ref_meta.summary() if self._ref_meta else ""
            self.compare_ref_label.setText(
                f"Reference: {seq.source_path}\n{meta_s}\nPose frames extracted: {n}"
            )
        else:
            self._user_sequence = seq
            meta_s = self._user_meta.summary() if self._user_meta else ""
            self.compare_user_label.setText(
                f"User: {seq.source_path}\n{meta_s}\nPose frames extracted: {n}"
            )
        self.compare_status_label.setText(
            f"Pose extraction finished ({n} frames). You can compare when both clips are ready."
        )

    def _on_extract_failed(self, message: str) -> None:
        if message != "Cancelled.":
            QMessageBox.warning(self, "Pose extraction", message)
        self.compare_status_label.setText("Pose extraction stopped.")

    def _on_extract_thread_finished(self) -> None:
        self._extract_worker = None
        self.btn_load.setEnabled(True)
        self.btn_process.setEnabled(self._video_path is not None)
        self._set_compare_ui_busy(False)

    def _on_compare_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Wait for overlay processing to finish before comparing.",
            )
            return
        if (
            self._ref_sequence is None
            or self._user_sequence is None
            or self._comparison_workers_busy()
        ):
            return
        self._teardown_overlay()
        self.overlay_group.setVisible(False)
        self._compare_worker = CompareSequencesWorker(
            self._ref_sequence,
            self._user_sequence,
            mirror_reference_for_scoring=self.mirror_scoring_checkbox.isChecked(),
        )
        self._compare_worker.progress.connect(self.compare_status_label.setText)
        self._compare_worker.finished_ok.connect(self._on_compare_finished)
        self._compare_worker.failed.connect(self._on_compare_failed)
        self._compare_worker.finished.connect(self._on_compare_thread_finished)
        self._set_compare_ui_busy(True)
        self.btn_process.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setRange(0, 0)
        self._compare_worker.start()

    def _on_compare_finished(self, result: ComparisonResult) -> None:
        self._last_comparison = result
        bd = result.breakdown
        pf = result.per_frame_similarity
        extra = ""
        if pf.size > 0:
            extra = (
                f"\nPer-frame similarity (DTW path): mean {float(pf.mean()):.1f}%, "
                f"min {float(pf.min()):.1f}%"
            )
        self.compare_scores_label.setText(
            f"Overall similarity: {result.overall_score:.1f}%\n"
            f"Timing score: {bd.timing:.1f}%  |  Arms: {bd.arms:.1f}%  |  "
            f"Legs: {bd.legs:.1f}%  |  Torso/posture: {bd.torso_posture:.1f}%\n"
            f"(Angles: {bd.joint_angles:.1f}%  ·  Directions: {bd.limb_directions:.1f}%  ·  "
            f"Distances: {bd.relative_distances:.1f}%)\n"
            f"DTW mean step cost: {result.dtw_mean_cost:.2f}  ·  "
            f"Mean abs frame lag (approx.): {result.timing_mean_abs_lag_frames:.2f}"
            f"{extra}"
        )
        self.compare_explain.setPlainText("\n".join(result.explanation_lines))
        self.compare_status_label.setText("Comparison complete.")
        self._setup_overlay_from_comparison(result)

    def _on_compare_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Comparison", message)
        self.compare_status_label.setText("Comparison failed.")
        self._teardown_overlay()
        self.overlay_group.setVisible(False)

    def _on_compare_thread_finished(self) -> None:
        self._compare_worker = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.btn_load.setEnabled(True)
        self.btn_process.setEnabled(self._video_path is not None)
        self._set_compare_ui_busy(False)

    def _on_export_scores_clicked(self) -> None:
        if self._last_comparison is None:
            return
        default = str(video_utils.OUTPUTS_DIR / "dance_comparison_scores.json")
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export comparison JSON",
            default,
            "JSON (*.json);;All files (*.*)",
        )
        if not dest:
            return
        try:
            save_comparison_json(self._last_comparison, dest)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self.compare_status_label.setText(f"Scores exported to: {dest}")

    # --- comparison previews & aligned overlay ---

    def _ref_bgr_for_practice_display(self, bgr):
        if bgr is None:
            return None
        if self.mirror_practice_checkbox.isChecked():
            return cv2.flip(bgr, 1)
        return bgr

    def _refresh_comparison_preview_panels(self) -> None:
        ref_show = self._ref_bgr_for_practice_display(self._ref_preview_bgr)
        self._set_bgr_on_preview_label(self.ref_preview_label, ref_show, "No video loaded")
        self._set_bgr_on_preview_label(self.user_preview_label, self._user_preview_bgr, "No video loaded")

    def _set_bgr_on_preview_label(self, label: QLabel, bgr, empty_text: str) -> None:
        if bgr is None:
            label.clear()
            label.setText(empty_text)
            return
        pm = ui_utils.bgr_to_qpixmap(bgr)
        if pm.isNull():
            label.setText(empty_text)
            return
        tgt = label.size()
        if tgt.width() < 20 or tgt.height() < 20:
            return
        scaled = pm.scaled(
            tgt,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.setText("")

    def _refresh_overlay_preview(self) -> None:
        if self._overlay_display_bgr is None:
            return
        pm = ui_utils.bgr_to_qpixmap(self._overlay_display_bgr)
        if pm.isNull():
            return
        tgt = self.overlay_video_label.size()
        if tgt.width() < 20 or tgt.height() < 20:
            return
        scaled = pm.scaled(
            tgt,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.overlay_video_label.setPixmap(scaled)
        self.overlay_video_label.setText("")

    def _stop_overlay_playback(self) -> None:
        if self._overlay_play_timer is not None:
            self._overlay_play_timer.stop()
            self._overlay_play_timer.deleteLater()
            self._overlay_play_timer = None
        self.btn_overlay_play.setText("Play overlay")

    def _teardown_overlay(self) -> None:
        self._stop_overlay_playback()
        if self._overlay_cap_ref is not None:
            self._overlay_cap_ref.release()
            self._overlay_cap_ref = None
        if self._overlay_cap_user is not None:
            self._overlay_cap_user.release()
            self._overlay_cap_user = None
        self._overlay_pairs = None
        self._overlay_display_bgr = None
        self._overlay_play_idx = 0
        self.btn_overlay_play.setEnabled(False)

    def _setup_overlay_from_comparison(self, result: ComparisonResult) -> None:
        self._teardown_overlay()
        if (
            not self._ref_path
            or not self._user_path
            or result.alignment_path is None
            or result.alignment_path.size == 0
        ):
            self.overlay_group.setVisible(False)
            return

        pairs = comparison_view.subsample_alignment_path(result.alignment_path, max_steps=500)
        if pairs.size == 0:
            self.overlay_group.setVisible(False)
            return

        self._overlay_pairs = pairs
        try:
            self._overlay_cap_ref = video_utils.open_capture(self._ref_path)
            self._overlay_cap_user = video_utils.open_capture(self._user_path)
        except video_utils.VideoOpenError as e:
            QMessageBox.warning(
                self,
                "Overlay",
                f"Could not open videos for the aligned overlay preview:\n{e}",
            )
            self._teardown_overlay()
            self.overlay_group.setVisible(False)
            return

        rf = float(self._ref_sequence.fps) if self._ref_sequence else 30.0
        uf = float(self._user_sequence.fps) if self._user_sequence else 30.0
        self._overlay_play_fps = max(8.0, min(60.0, (rf + uf) * 0.5))

        self.overlay_group.setVisible(True)
        self._overlay_play_idx = 0
        self.btn_overlay_play.setEnabled(True)
        self._display_overlay_at_current_index()
        QTimer.singleShot(0, self._refresh_overlay_preview)

    def _display_overlay_at_current_index(self) -> None:
        if (
            self._overlay_pairs is None
            or self._overlay_cap_ref is None
            or self._overlay_cap_user is None
        ):
            return
        n = int(self._overlay_pairs.shape[0])
        k = int(self._overlay_play_idx) % n
        ri = int(self._overlay_pairs[k, 0])
        uj = int(self._overlay_pairs[k, 1])
        blended, ok = comparison_view.overlay_pair_from_caps(
            self._overlay_cap_ref,
            self._overlay_cap_user,
            ri,
            uj,
            user_alpha=comparison_view.DEFAULT_USER_OVERLAY_ALPHA,
            flip_reference_horizontal=self.mirror_practice_checkbox.isChecked(),
        )
        if not ok or blended is None:
            self.overlay_frame_info.setText(
                f"Read failed at ref #{ri}, user #{uj} (codec/seek issue)."
            )
            return
        self._overlay_display_bgr = blended
        self.overlay_frame_info.setText(
            f"Step {k + 1}/{n} (subsampled DTW)  ·  ref frame {ri}  ·  user frame {uj}  ·  "
            f"user α={comparison_view.DEFAULT_USER_OVERLAY_ALPHA:.2f}"
        )
        self._refresh_overlay_preview()

    def _on_overlay_play_clicked(self) -> None:
        if self._overlay_pairs is None or self._overlay_pairs.shape[0] == 0:
            return
        if self._overlay_play_timer is not None and self._overlay_play_timer.isActive():
            self._stop_overlay_playback()
            return
        self._stop_playback()
        self._stop_ref_practice_playback()
        self.btn_play.setText("Play Processed Video")
        self._overlay_play_timer = QTimer(self)
        self._overlay_play_timer.timeout.connect(self._on_overlay_tick)
        interval_ms = max(1, round(1000.0 / self._overlay_play_fps))
        self._overlay_play_timer.start(interval_ms)
        self.btn_overlay_play.setText("Pause overlay")

    def _on_overlay_tick(self) -> None:
        if self._overlay_pairs is None:
            return
        n = int(self._overlay_pairs.shape[0])
        self._overlay_play_idx = (self._overlay_play_idx + 1) % n
        self._display_overlay_at_current_index()

    def _on_mirror_practice_changed(self, _state: int) -> None:
        self._refresh_comparison_preview_panels()
        if self._overlay_pairs is not None and self._overlay_pairs.size > 0:
            self._display_overlay_at_current_index()

    # --- dance library & reference practice playback ---

    def _refresh_library_combo(self, *, select_id: str | None = None) -> None:
        self.library_combo.blockSignals(True)
        self.library_combo.clear()
        self.library_combo.addItem("(select a saved dance)", None)
        for e in list_dances():
            label = f"{e.name}  ·  {e.created_at[:10] if len(e.created_at) >= 10 else e.created_at}"
            self.library_combo.addItem(label, e.dance_id)
        if select_id:
            for i in range(self.library_combo.count()):
                if self.library_combo.itemData(i) == select_id:
                    self.library_combo.setCurrentIndex(i)
                    break
        self.library_combo.blockSignals(False)
        self.btn_delete_library_dance.setEnabled(self.library_combo.count() > 1)

    def _on_save_dance_clicked(self) -> None:
        if self._comparison_workers_busy():
            return
        if self._ref_sequence is None or self._ref_path is None or self._ref_meta is None:
            QMessageBox.information(
                self,
                "Save dance",
                "Load a reference video and click **Process Reference** first.",
            )
            return
        text, ok = QInputDialog.getText(self, "Save dance", "Dance name:")
        if not ok:
            return
        name = text.strip()
        if not name:
            QMessageBox.warning(self, "Save dance", "Enter a name for this dance.")
            return
        try:
            md = save_dance_from_reference(
                name=name,
                reference_video_path=self._ref_path,
                sequence=self._ref_sequence,
                meta=self._ref_meta,
                mirror_for_practice=self.mirror_practice_checkbox.isChecked(),
                mirror_for_scoring=self.mirror_scoring_checkbox.isChecked(),
            )
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self.compare_status_label.setText(
            f"Saved dance “{md.name}” to the library ({md.dance_id[:8]}…)."
        )
        self._refresh_library_combo(select_id=md.dance_id)

    def _on_load_library_dance_clicked(self) -> None:
        if self._comparison_workers_busy():
            return
        dance_id = self.library_combo.currentData()
        if not dance_id:
            QMessageBox.information(self, "Library", "Choose a saved dance from the list.")
            return
        try:
            md, seq = load_dance(str(dance_id))
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Library", str(e))
            return
        except Exception as e:
            QMessageBox.warning(self, "Library", f"Could not load dance:\n{e}")
            return

        self._teardown_overlay()
        self.overlay_group.setVisible(False)
        self._stop_ref_practice_playback()

        path = seq.source_path
        cap = None
        try:
            cap = video_utils.open_capture(path)
            meta = video_utils.read_metadata(cap, path)
            first = video_utils.read_first_frame(cap)
        except video_utils.VideoOpenError as e:
            QMessageBox.warning(
                self,
                "Library",
                f"Saved pose data is present but the video file could not be opened:\n{e}",
            )
            return
        finally:
            if cap is not None:
                cap.release()

        self._ref_path = path
        self._ref_meta = meta
        self._ref_sequence = seq
        self._active_library_dance_id = md.dance_id
        self._active_library_dance_name = md.name
        self._ref_preview_bgr = first.copy() if first is not None else None
        self.mirror_practice_checkbox.setChecked(md.mirror_for_practice)
        self.mirror_scoring_checkbox.setChecked(md.mirror_for_scoring)

        lib_line = f"Library dance: {md.name}"
        self.compare_ref_label.setText(
            f"{lib_line}\n{path}\n{meta.summary()}\n"
            f"Pose frames loaded from library: {len(seq.frames)} (no re-extraction needed)."
        )
        self.ref_preview_status.setText(
            f"Showing frame 0  ·  {meta.summary()}"
            if first is not None
            else "Could not read frame 0."
        )
        self._last_comparison = None
        self.compare_scores_label.setText("")
        self.compare_explain.clear()
        self.compare_status_label.setText(
            "Reference loaded from library. Watch **Play reference (practice)**, then load and process your video."
        )
        self._refresh_comparison_preview_panels()
        QTimer.singleShot(0, self._refresh_comparison_preview_panels)
        self._set_compare_ui_busy(False)

    def _on_delete_library_dance_clicked(self) -> None:
        if self._comparison_workers_busy():
            return
        dance_id = self.library_combo.currentData()
        if not dance_id:
            QMessageBox.information(self, "Library", "Choose a dance to delete.")
            return
        name = self.library_combo.currentText()
        r = QMessageBox.question(
            self,
            "Delete dance",
            f"Remove this dance from the library?\n\n{name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if r != QMessageBox.StandardButton.Yes:
            return
        if self._active_library_dance_id == dance_id:
            self._active_library_dance_id = None
            self._active_library_dance_name = None
        delete_dance(str(dance_id))
        self._refresh_library_combo()
        self.compare_status_label.setText("Dance removed from library.")

    def _stop_ref_practice_playback(self) -> None:
        if self._ref_practice_timer is not None:
            self._ref_practice_timer.stop()
            self._ref_practice_timer.deleteLater()
            self._ref_practice_timer = None
        if self._ref_practice_cap is not None:
            self._ref_practice_cap.release()
            self._ref_practice_cap = None
        self.btn_ref_practice_play.setText("Play reference (practice)")

    def _on_ref_practice_play_clicked(self) -> None:
        if not self._ref_path or not Path(self._ref_path).is_file():
            QMessageBox.information(self, "Practice", "Load a reference video first.")
            return
        if self._ref_practice_timer is not None and self._ref_practice_timer.isActive():
            self._stop_ref_practice_playback()
            self.ref_preview_status.setText("Practice playback stopped.")
            return

        self._stop_playback()
        self._stop_overlay_playback()
        self.btn_play.setText("Play Processed Video")

        self._ref_practice_cap = cv2.VideoCapture(self._ref_path)
        if not self._ref_practice_cap.isOpened():
            QMessageBox.warning(self, "Practice", f"Could not open:\n{self._ref_path}")
            self._ref_practice_cap = None
            return

        pb_fps, pb_note = video_utils.resolve_playback_fps(self._ref_practice_cap, self._ref_meta)
        interval_ms = max(1, round(1000.0 / max(8.0, min(60.0, pb_fps))))
        self._ref_practice_timer = QTimer(self)
        self._ref_practice_timer.timeout.connect(self._on_ref_practice_tick)
        self._ref_practice_timer.start(interval_ms)
        self.btn_ref_practice_play.setText("Stop reference playback")
        msg = "Playing reference for practice (silent)…"
        if pb_note:
            msg = f"{pb_note} {msg}"
        self.ref_preview_status.setText(msg)

    def _on_ref_practice_tick(self) -> None:
        if self._ref_practice_cap is None:
            return
        ret, frame = self._ref_practice_cap.read()
        if not ret or frame is None:
            self._ref_practice_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._ref_practice_cap.read()
        if not ret or frame is None:
            self._stop_ref_practice_playback()
            self.ref_preview_status.setText("Practice playback ended (read error).")
            return
        disp = self._ref_bgr_for_practice_display(frame)
        self._set_bgr_on_preview_label(self.ref_preview_label, disp, "")
        fi = int(self._ref_practice_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.ref_preview_status.setText(f"Practice playback · ~frame {fi}")
