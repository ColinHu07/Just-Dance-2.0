"""Main application window: load, preview, process, play, export."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QIcon, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
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
    QStackedLayout,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app import comparison_view
from app.calibration import CalibrationReport
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
    CalibrationScanWorker,
    CompareSequencesWorker,
    ExtractSequenceWorker,
    save_comparison_json,
)
from app.style import APP_STYLESHEET
from app.worker import ProcessVideoWorker


FRONTEND_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FRONTEND_BACKDROP_VIDEO_PATH = FRONTEND_ASSETS_DIR / "frontend-dance-bg.mp4"
FRONTEND_BACKDROP_IMAGE_PATH = FRONTEND_ASSETS_DIR / "frontend-dance-bg.png"
FRONTEND_MUSIC_PATH = FRONTEND_ASSETS_DIR / "frontend-bg-music.mp3"

FRONTEND_COMPANY_ORDER = ("ALL", "SM", "JYP", "YG", "HYBE", "OTHER")
FRONTEND_COMPANY_LABELS = {
    "ALL": "ALL",
    "SM": "SM",
    "JYP": "JYP",
    "YG": "YG",
    "HYBE": "HYBE",
    "OTHER": "MORE",
}
FRONTEND_ARTIST_COMPANIES = {
    "SM": {
        "aespa": ("aespa", "aespadrama", "aespawhiplash"),
        "NCT": ("nct", "nct127", "nctdream", "wayv"),
        "Red Velvet": ("redvelvet",),
        "RIIZE": ("riize",),
        "EXO": ("exo",),
        "SHINee": ("shinee",),
        "Girls' Generation": ("girlsgeneration", "snsd"),
    },
    "JYP": {
        "TWICE": ("twice",),
        "ITZY": ("itzy",),
        "Stray Kids": ("straykids", "skz"),
        "NMIXX": ("nmixx",),
        "NiziU": ("niziu",),
    },
    "YG": {
        "BLACKPINK": ("blackpink",),
        "BABYMONSTER": ("babymonster", "baemon"),
        "TREASURE": ("treasure",),
        "BIGBANG": ("bigbang",),
        "2NE1": ("2ne1",),
    },
    "HYBE": {
        "BTS": ("bts", "bangtan"),
        "SEVENTEEN": ("seventeen", "svt"),
        "LE SSERAFIM": ("lesserafim", "leserafim"),
        "NewJeans": ("newjeans",),
        "TXT": ("txt", "tomorrowxtogether"),
        "ENHYPEN": ("enhypen",),
        "ILLIT": ("illit",),
        "BOYNEXTDOOR": ("boynextdoor",),
    },
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dance Pose Desktop")
        self.setMinimumSize(980, 700)
        self.setAcceptDrops(True)
        self.setStyleSheet(APP_STYLESHEET)

        self._video_path: str | None = None
        self._meta: video_utils.VideoMetadata | None = None
        self._preview_source_bgr = None
        self._processed_path: str | None = None
        self._worker: ProcessVideoWorker | None = None
        self._play_cap: cv2.VideoCapture | None = None
        self._play_timer: QTimer | None = None
        self._play_base_fps: float = 30.0  # processed clip, for timer interval (fallback only)
        self._play_started_at: float = 0.0
        self._play_last_frame_index: int = -1
        self._play_current_frame_index: int = 0
        self._play_total_frames: int = 0
        self._seek_is_dragging: bool = False
        self._raw_first_frame = None  # BGR, for re-running preview when mode changes
        self._last_preview_annotation: pose_utils.AnnotateResult | None = None

        self._ref_path: str | None = None
        self._user_path: str | None = None
        self._ref_meta: video_utils.VideoMetadata | None = None
        self._user_meta: video_utils.VideoMetadata | None = None
        self._ref_sequence: PoseSequence | None = None
        self._user_sequence: PoseSequence | None = None
        self._last_comparison: ComparisonResult | None = None
        self._ref_calibration: CalibrationReport | None = None
        self._user_calibration: CalibrationReport | None = None
        self._frontend_calibration: CalibrationReport | None = None
        self._pending_frontend_start_after_calibration: bool = False
        self._frontend_start_ready_after_calibration: bool = False
        self._calibration_worker: CalibrationScanWorker | None = None
        self._calibration_target: str = "ref"
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
        self._ref_practice_base_fps: float = 30.0
        self._ref_practice_started_at: float = 0.0
        self._ref_practice_last_frame_index: int = -1
        self._ref_practice_total_frames: int = 0
        self._frontend_backdrop_cap: cv2.VideoCapture | None = None
        self._frontend_backdrop_timer: QTimer | None = None
        self._frontend_backdrop_bgr = None
        self._frontend_backdrop_static_pixmap: QPixmap | None = None
        self._frontend_backdrop_fps: float = 24.0
        self._frontend_backdrop_total_frames: int = 0
        self._frontend_backdrop_started_at: float = 0.0
        self._frontend_backdrop_last_frame_index: int = -1
        self._frontend_backdrop_ready: bool = False
        self._frontend_audio_output: QAudioOutput | None = None
        self._frontend_music_player: QMediaPlayer | None = None
        self._frontend_music_should_play: bool = False
        self._frontend_stage_bgr = None
        self._frontend_selected_company: str = "ALL"
        self._frontend_selected_artist: str = "ALL"
        self._frontend_company_buttons: dict[str, QPushButton] = {}
        self._frontend_artist_buttons: dict[str, QPushButton] = {}
        self._frontend_dance_card_buttons: dict[str, QPushButton] = {}
        self._frontend_player_buttons: dict[int, QPushButton] = {}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QWidget()
        inner.setObjectName("AppSurface")
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root = QVBoxLayout(inner)
        root.setContentsMargins(28, 24, 28, 32)
        root.setSpacing(16)
        scroll.setWidget(inner)
        self.setCentralWidget(scroll)
        self._scroll_area = scroll

        hero = QWidget()
        hero.setObjectName("Hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(22, 18, 22, 18)
        hero_layout.setSpacing(18)
        hero_text = QVBoxLayout()
        hero_text.setSpacing(4)
        app_title = QLabel("Dance Pose Desktop")
        app_title.setObjectName("AppTitle")
        hero_text.addWidget(app_title)
        app_subtitle = QLabel("Pose overlay, practice playback, and geometry scoring")
        app_subtitle.setObjectName("AppSubtitle")
        hero_text.addWidget(app_subtitle)
        hero_layout.addLayout(hero_text, stretch=1)
        app_pill = QLabel("Local analysis")
        app_pill.setObjectName("HeaderPill")
        app_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_layout.addWidget(app_pill)
        root.addWidget(hero)

        mode_bar = QWidget()
        mode_bar.setObjectName("ModeBar")
        mode_layout = QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        self.btn_mode_frontend = QPushButton("Frontend")
        self.btn_mode_backend = QPushButton("Backend")
        for btn in (self.btn_mode_frontend, self.btn_mode_backend):
            btn.setCheckable(True)
            btn.setProperty("modeToggle", True)
        mode_layout.addWidget(self.btn_mode_frontend)
        mode_layout.addWidget(self.btn_mode_backend)
        mode_layout.addStretch(1)
        root.addWidget(mode_bar)

        self.mode_stack = QStackedWidget()
        self.frontend_page = self._build_frontend_page()
        self.backend_page = QWidget()
        self.backend_page.setObjectName("BackendPage")
        backend_layout = QVBoxLayout(self.backend_page)
        backend_layout.setContentsMargins(0, 0, 0, 0)
        backend_layout.setSpacing(16)

        overlay_box = QGroupBox("Pose overlay")
        overlay_layout = QVBoxLayout(overlay_box)
        overlay_layout.setSpacing(12)

        self.path_label = QLabel("No video loaded.")
        self.path_label.setObjectName("FilePath")
        self.path_label.setWordWrap(True)
        overlay_layout.addWidget(self.path_label)

        self.meta_label = QLabel("")
        self.meta_label.setObjectName("MutedLabel")
        self.meta_label.setWordWrap(True)
        overlay_layout.addWidget(self.meta_label)

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
        overlay_layout.addLayout(mode_row)

        audio_row = QHBoxLayout()
        self.keep_audio_checkbox = QCheckBox("Keep Original Audio")
        self.keep_audio_checkbox.setChecked(True)
        self.keep_audio_checkbox.setToolTip(
            "After OpenCV finishes the silent overlay clip, ffmpeg (if installed) can mux in "
            "the original soundtrack. Turn off to keep the processed file video-only."
        )
        audio_row.addWidget(self.keep_audio_checkbox)
        audio_row.addStretch(1)
        overlay_layout.addLayout(audio_row)

        self.preview_audio_note = QLabel(
            "In-app playback is silent (frame preview only). With “Keep Original Audio” on, "
            "the saved processed file includes sound — open it in QuickTime or another player to hear it."
        )
        self.preview_audio_note.setWordWrap(True)
        self.preview_audio_note.setObjectName("MutedLabel")
        overlay_layout.addWidget(self.preview_audio_note)

        self.people_label = QLabel("People detected: —")
        self.people_label.setObjectName("PeopleLabel")
        self.people_label.setWordWrap(True)
        overlay_layout.addWidget(self.people_label)

        self.status_label = QLabel("Load a video file to begin.")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        overlay_layout.addWidget(self.status_label)

        row = QHBoxLayout()
        self.btn_load = QPushButton("Load video")
        self.btn_load.setProperty("variant", "primary")
        self.btn_process = QPushButton("Render overlay")
        self.btn_process.setProperty("variant", "accent")
        self.btn_process.setEnabled(False)
        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_export = QPushButton("Export")
        self.btn_export.setEnabled(False)
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_process)
        row.addWidget(self.btn_play)
        row.addWidget(self.btn_export)
        overlay_layout.addLayout(row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Playback speed:"))
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
        self.btn_speed_reset.setProperty("variant", "compact")
        self.btn_speed_reset.setToolTip("Reset playback speed to 1.00×")
        self.btn_speed_reset.setEnabled(False)
        self.btn_speed_reset.clicked.connect(self._on_playback_speed_reset)
        speed_row.addWidget(self.btn_speed_reset)
        overlay_layout.addLayout(speed_row)

        seek_row = QHBoxLayout()
        self.seek_current_label = QLabel("0:00")
        self.seek_current_label.setObjectName("MutedLabel")
        seek_row.addWidget(self.seek_current_label)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setToolTip("Drag to jump to a different point in the processed video.")
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        seek_row.addWidget(self.seek_slider, stretch=1)
        self.seek_duration_label = QLabel("0:00")
        self.seek_duration_label.setObjectName("MutedLabel")
        seek_row.addWidget(self.seek_duration_label)
        overlay_layout.addLayout(seek_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        overlay_layout.addWidget(self.progress)

        overlay_title = QLabel("Preview")
        overlay_title.setObjectName("PanelTitle")
        overlay_layout.addWidget(overlay_title)
        self.video_label = QLabel()
        self._configure_video_label(self.video_label, min_height=340, max_height=420)
        self.video_label.setText("Video preview")
        overlay_layout.addWidget(self.video_label)
        backend_layout.addWidget(overlay_box)

        compare_box = QGroupBox("Dance comparison (reference vs your performance)")
        compare_layout = QVBoxLayout(compare_box)
        compare_layout.setSpacing(12)
        self.compare_ref_label = QLabel("Reference: (not loaded)")
        self.compare_ref_label.setObjectName("FilePath")
        self.compare_ref_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_ref_label)
        self.compare_user_label = QLabel("User: (not loaded)")
        self.compare_user_label.setObjectName("FilePath")
        self.compare_user_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_user_label)

        lib_box = QGroupBox("Dance library (save once, reuse for practice && scoring)")
        lib_box.setObjectName("LibraryBox")
        lib_layout = QVBoxLayout(lib_box)
        lib_layout.setSpacing(10)
        lib_row1 = QHBoxLayout()
        lib_row1.addWidget(QLabel("Saved dances:"))
        self.library_combo = QComboBox()
        self.library_combo.setMinimumWidth(240)
        lib_row1.addWidget(self.library_combo, stretch=1)
        self.btn_save_dance = QPushButton("Save dance")
        self.btn_load_library_dance = QPushButton("Load dance")
        self.btn_delete_library_dance = QPushButton("Delete")
        self.btn_delete_library_dance.setProperty("variant", "danger")
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
        self.btn_ref_practice_play = QPushButton("Practice play")
        self.btn_ref_practice_play.setEnabled(False)
        self.btn_ref_practice_play.setToolTip("Watch the reference clip in the Reference preview (silent).")
        lib_row2.addWidget(self.btn_ref_practice_play)
        lib_row2.addStretch(1)
        lib_layout.addLayout(lib_row2)
        self.flow_mode_label = QLabel(
            "Practice: watch the reference. Load your video, process both sides, then compare to score."
        )
        self.flow_mode_label.setWordWrap(True)
        self.flow_mode_label.setObjectName("MutedLabel")
        lib_layout.addWidget(self.flow_mode_label)
        compare_layout.addWidget(lib_box)

        dual_row = QHBoxLayout()
        ref_col = QVBoxLayout()
        ref_title = QLabel("Reference")
        ref_title.setObjectName("PanelTitle")
        ref_col.addWidget(ref_title)
        self.ref_preview_label = QLabel()
        self._configure_video_label(self.ref_preview_label, min_height=200, max_height=260)
        self.ref_preview_label.setText("No video loaded")
        self.ref_preview_status = QLabel("Load a reference file to preview frame 0.")
        self.ref_preview_status.setObjectName("MutedLabel")
        self.ref_preview_status.setWordWrap(True)
        ref_col.addWidget(self.ref_preview_label, stretch=1)
        ref_col.addWidget(self.ref_preview_status)
        user_col = QVBoxLayout()
        user_title = QLabel("Performance")
        user_title.setObjectName("PanelTitle")
        user_col.addWidget(user_title)
        self.user_preview_label = QLabel()
        self._configure_video_label(self.user_preview_label, min_height=200, max_height=260)
        self.user_preview_label.setText("No video loaded")
        self.user_preview_status = QLabel("Load a user performance file to preview frame 0.")
        self.user_preview_status.setObjectName("MutedLabel")
        self.user_preview_status.setWordWrap(True)
        user_col.addWidget(self.user_preview_label, stretch=1)
        user_col.addWidget(self.user_preview_status)
        dual_row.addLayout(ref_col, stretch=1)
        dual_row.addLayout(user_col, stretch=1)
        compare_layout.addLayout(dual_row)

        calib_row = QHBoxLayout()
        calib_row.addWidget(QLabel("Calibration dancers:"))
        self.expected_people_combo = QComboBox()
        for n in range(1, 5):
            label = "1 dancer" if n == 1 else f"{n} dancers"
            self.expected_people_combo.addItem(label, n)
        self.expected_people_combo.setToolTip(
            "Scan expects this many dancers to be visible and separated. "
            "Comparison scoring still uses the center dancer for now."
        )
        self.expected_people_combo.currentIndexChanged.connect(
            self._on_expected_people_changed
        )
        calib_row.addWidget(self.expected_people_combo)
        self.btn_scan_ref = QPushButton("Scan reference")
        self.btn_scan_user = QPushButton("Scan performance")
        self.btn_scan_ref.setEnabled(False)
        self.btn_scan_user.setEnabled(False)
        calib_row.addWidget(self.btn_scan_ref)
        calib_row.addWidget(self.btn_scan_user)
        calib_row.addStretch(1)
        compare_layout.addLayout(calib_row)

        self.ref_calibration_label = QLabel("Reference calibration: not scanned")
        self.ref_calibration_label.setObjectName("MutedLabel")
        self.ref_calibration_label.setWordWrap(True)
        compare_layout.addWidget(self.ref_calibration_label)
        self.user_calibration_label = QLabel("Performance calibration: not scanned")
        self.user_calibration_label.setObjectName("MutedLabel")
        self.user_calibration_label.setWordWrap(True)
        compare_layout.addWidget(self.user_calibration_label)

        cmp_row = QHBoxLayout()
        self.btn_load_ref = QPushButton("Load reference")
        self.btn_load_user = QPushButton("Load performance")
        self.btn_process_ref = QPushButton("Extract reference")
        self.btn_process_user = QPushButton("Extract performance")
        self.btn_compare = QPushButton("Compare")
        self.btn_compare.setProperty("variant", "primary")
        self.btn_export_scores = QPushButton("Export JSON")
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
        self.compare_status_label.setObjectName("StatusLabel")
        self.compare_status_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_status_label)
        self.compare_scores_label = QLabel("")
        self.compare_scores_label.setObjectName("ScoreCard")
        self.compare_scores_label.setWordWrap(True)
        compare_layout.addWidget(self.compare_scores_label)
        self.compare_explain = QTextEdit()
        self.compare_explain.setReadOnly(True)
        self.compare_explain.setMaximumHeight(88)
        self.compare_explain.setPlaceholderText("Coach notes will appear here.")
        compare_layout.addWidget(self.compare_explain)

        self.overlay_group = QGroupBox("Aligned overlay view (DTW time alignment)")
        overlay_layout = QVBoxLayout(self.overlay_group)
        overlay_layout.setSpacing(12)
        overlay_layout.addWidget(
            QLabel(
                "After Compare Videos: reference frame (opaque) + user frame (~40% opacity). "
                "Pairs use the DTW alignment path, not raw frame numbers."
            )
        )
        self.overlay_video_label = QLabel()
        self._configure_video_label(self.overlay_video_label, min_height=260, max_height=340)
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

        backend_layout.addWidget(compare_box)

        self.mode_stack.addWidget(self.frontend_page)
        self.mode_stack.addWidget(self.backend_page)
        root.addWidget(self.mode_stack)

        self.btn_load_ref.clicked.connect(self._on_load_ref_clicked)
        self.btn_load_user.clicked.connect(self._on_load_user_clicked)
        self.btn_scan_ref.clicked.connect(self._on_scan_ref_clicked)
        self.btn_scan_user.clicked.connect(self._on_scan_user_clicked)
        self.btn_process_ref.clicked.connect(self._on_process_ref_clicked)
        self.btn_process_user.clicked.connect(self._on_process_user_clicked)
        self.btn_compare.clicked.connect(self._on_compare_clicked)
        self.btn_export_scores.clicked.connect(self._on_export_scores_clicked)

        self.btn_save_dance.clicked.connect(self._on_save_dance_clicked)
        self.btn_load_library_dance.clicked.connect(self._on_load_library_dance_clicked)
        self.btn_delete_library_dance.clicked.connect(self._on_delete_library_dance_clicked)
        self.btn_ref_practice_play.clicked.connect(self._on_ref_practice_play_clicked)
        self.mirror_practice_checkbox.stateChanged.connect(self._on_mirror_practice_changed)

        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_process.clicked.connect(self._on_process_clicked)
        self.btn_play.clicked.connect(self._on_play_clicked)
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_mode_frontend.clicked.connect(lambda: self._set_app_mode("frontend"))
        self.btn_mode_backend.clicked.connect(lambda: self._set_app_mode("backend"))
        self.frontend_dance_combo.currentIndexChanged.connect(
            self._on_frontend_dance_changed
        )
        self.frontend_players_combo.currentIndexChanged.connect(
            self._on_frontend_players_changed
        )
        self.btn_frontend_calibrate.clicked.connect(self._on_frontend_calibrate_clicked)
        self.btn_frontend_start.clicked.connect(self._on_frontend_start_clicked)

        self._update_playback_speed_label()

        self._refresh_library_combo()
        self._set_compare_ui_busy(False)
        self._set_app_mode("frontend")

    def _build_frontend_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("FrontendPage")
        stack = QStackedLayout(page)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)

        self.frontend_backdrop_label = QLabel()
        self.frontend_backdrop_label.setObjectName("FrontendBackdrop")
        self.frontend_backdrop_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
        )
        self.frontend_backdrop_label.setScaledContents(False)
        self.frontend_backdrop_label.setMinimumSize(0, 0)
        self.frontend_backdrop_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        if FRONTEND_BACKDROP_IMAGE_PATH.is_file():
            self._frontend_backdrop_static_pixmap = QPixmap(str(FRONTEND_BACKDROP_IMAGE_PATH))
        stack.addWidget(self.frontend_backdrop_label)

        content = QWidget()
        content.setObjectName("FrontendContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 22, 28, 24)
        layout.setSpacing(14)

        game_box = QWidget()
        game_box.setObjectName("FrontendGameShell")
        game_layout = QVBoxLayout(game_box)
        game_layout.setContentsMargins(0, 0, 0, 0)
        game_layout.setSpacing(12)

        marquee = QHBoxLayout()
        marquee.setSpacing(18)
        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        kicker = QLabel("STAGE SELECT")
        kicker.setObjectName("FrontendKicker")
        title_stack.addWidget(kicker)
        title = QLabel("Choose Your Dance")
        title.setObjectName("FrontendTitle")
        title_stack.addWidget(title)
        self.frontend_song_meta_label = QLabel("Pick a company, artist, and saved dance.")
        self.frontend_song_meta_label.setObjectName("FrontendSongMeta")
        title_stack.addWidget(self.frontend_song_meta_label)
        marquee.addLayout(title_stack, stretch=1)

        self.frontend_ready_badge = QLabel("READY")
        self.frontend_ready_badge.setObjectName("FrontendReadyBadge")
        self.frontend_ready_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        marquee.addWidget(self.frontend_ready_badge)
        game_layout.addLayout(marquee)

        company_row = QHBoxLayout()
        company_row.setSpacing(8)
        self._frontend_company_buttons = {}
        for company in FRONTEND_COMPANY_ORDER:
            btn = QPushButton(FRONTEND_COMPANY_LABELS.get(company, company))
            btn.setCheckable(True)
            btn.setObjectName("CompanyTab")
            btn.setProperty("company", company)
            btn.clicked.connect(
                lambda _checked=False, c=company: self._set_frontend_company_filter(c)
            )
            self._frontend_company_buttons[company] = btn
            company_row.addWidget(btn)
        company_row.addStretch(1)
        game_layout.addLayout(company_row)

        artist_wrap = QWidget()
        artist_wrap.setObjectName("ArtistStrip")
        artist_wrap_layout = QHBoxLayout(artist_wrap)
        artist_wrap_layout.setContentsMargins(10, 8, 10, 8)
        artist_wrap_layout.setSpacing(8)
        artist_label = QLabel("Artists")
        artist_label.setObjectName("FrontendControlLabel")
        artist_wrap_layout.addWidget(artist_label)
        self.frontend_artist_buttons_row = QHBoxLayout()
        self.frontend_artist_buttons_row.setSpacing(8)
        artist_wrap_layout.addLayout(self.frontend_artist_buttons_row, stretch=1)
        game_layout.addWidget(artist_wrap)

        middle = QHBoxLayout()
        middle.setSpacing(14)

        selector_panel = QWidget()
        selector_panel.setObjectName("DanceSelector")
        selector_layout = QVBoxLayout(selector_panel)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(8)

        selector_header = QHBoxLayout()
        library_title = QLabel("Saved Dances")
        library_title.setObjectName("FrontendPanelTitle")
        selector_header.addWidget(library_title)
        selector_header.addStretch(1)
        self.frontend_library_count_label = QLabel("0 tracks")
        self.frontend_library_count_label.setObjectName("FrontendMiniPill")
        selector_header.addWidget(self.frontend_library_count_label)
        selector_layout.addLayout(selector_header)

        self.frontend_dance_cards_scroll = QScrollArea()
        self.frontend_dance_cards_scroll.setObjectName("DanceCardScroll")
        self.frontend_dance_cards_scroll.setWidgetResizable(True)
        self.frontend_dance_cards_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.frontend_dance_cards_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        cards_host = QWidget()
        cards_host.setObjectName("DanceCardsHost")
        self.frontend_dance_cards_grid = QGridLayout(cards_host)
        self.frontend_dance_cards_grid.setContentsMargins(0, 0, 4, 0)
        self.frontend_dance_cards_grid.setHorizontalSpacing(10)
        self.frontend_dance_cards_grid.setVerticalSpacing(10)
        self.frontend_dance_cards_scroll.setWidget(cards_host)
        selector_layout.addWidget(self.frontend_dance_cards_scroll, stretch=1)
        middle.addWidget(selector_panel, stretch=3)

        play_panel = QWidget()
        play_panel.setObjectName("PlayPanel")
        play_layout = QVBoxLayout(play_panel)
        play_layout.setContentsMargins(0, 0, 0, 0)
        play_layout.setSpacing(10)

        self.frontend_song_title_label = QLabel("Select a dance")
        self.frontend_song_title_label.setObjectName("FrontendSongTitle")
        self.frontend_song_title_label.setWordWrap(True)
        play_layout.addWidget(self.frontend_song_title_label)

        self.frontend_stage_label = QLabel()
        self._configure_video_label(
            self.frontend_stage_label,
            min_height=210,
            max_height=260,
        )
        self.frontend_stage_label.setObjectName("FrontendStage")
        self.frontend_stage_label.setText("Select a dance")
        play_layout.addWidget(self.frontend_stage_label)

        self.frontend_status_label = QLabel("Select a saved dance.")
        self.frontend_status_label.setObjectName("FrontendStatus")
        self.frontend_status_label.setWordWrap(True)
        play_layout.addWidget(self.frontend_status_label)

        player_row = QHBoxLayout()
        player_row.setSpacing(8)
        players_label = QLabel("Players")
        players_label.setObjectName("FrontendControlLabel")
        player_row.addWidget(players_label)
        self._frontend_player_buttons = {}
        for n in range(1, 5):
            btn = QPushButton(str(n))
            btn.setCheckable(True)
            btn.setObjectName("PlayerChip")
            btn.setProperty("playerCount", n)
            btn.clicked.connect(lambda _checked=False, count=n: self._set_frontend_players(count))
            self._frontend_player_buttons[n] = btn
            player_row.addWidget(btn)
        player_row.addStretch(1)
        play_layout.addLayout(player_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.btn_frontend_calibrate = QPushButton("Calibrate")
        self.btn_frontend_calibrate.setObjectName("FrontendActionButton")
        self.btn_frontend_start = QPushButton("Start")
        self.btn_frontend_start.setObjectName("FrontendStartButton")
        self.btn_frontend_start.setProperty("variant", "primary")
        action_row.addWidget(self.btn_frontend_calibrate)
        action_row.addWidget(self.btn_frontend_start)
        play_layout.addLayout(action_row)
        play_layout.addStretch(1)
        middle.addWidget(play_panel, stretch=2)

        game_layout.addLayout(middle, stretch=1)

        self.frontend_dance_combo = QComboBox()
        self.frontend_dance_combo.setVisible(False)
        game_layout.addWidget(self.frontend_dance_combo)
        self.frontend_players_combo = QComboBox()
        for n in range(1, 5):
            label = "1 player" if n == 1 else f"{n} players"
            self.frontend_players_combo.addItem(label, n)
        self.frontend_players_combo.setVisible(False)
        game_layout.addWidget(self.frontend_players_combo)

        layout.addWidget(game_box)
        layout.addStretch(1)
        stack.addWidget(content)

        self.frontend_loading_overlay = QWidget()
        self.frontend_loading_overlay.setObjectName("FrontendLoadingOverlay")
        loading_layout = QVBoxLayout(self.frontend_loading_overlay)
        loading_layout.setContentsMargins(28, 28, 28, 28)
        loading_layout.setSpacing(12)
        loading_layout.addStretch(1)
        self.frontend_loading_title = QLabel("Loading stage")
        self.frontend_loading_title.setObjectName("FrontendLoadingTitle")
        self.frontend_loading_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.frontend_loading_title)
        self.frontend_loading_status = QLabel("Cueing video")
        self.frontend_loading_status.setObjectName("FrontendLoadingStatus")
        self.frontend_loading_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.frontend_loading_status)
        self.frontend_loading_progress = QProgressBar()
        self.frontend_loading_progress.setObjectName("FrontendLoadingProgress")
        self.frontend_loading_progress.setRange(0, 0)
        self.frontend_loading_progress.setTextVisible(False)
        loading_layout.addWidget(
            self.frontend_loading_progress, alignment=Qt.AlignmentFlag.AlignCenter
        )
        loading_layout.addStretch(1)
        stack.addWidget(self.frontend_loading_overlay)
        self.frontend_loading_overlay.hide()

        stack.setCurrentWidget(content)
        return page

    def _configure_video_label(
        self,
        label: QLabel,
        *,
        min_height: int,
        max_height: int,
    ) -> None:
        label.setObjectName("VideoPreview")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumWidth(0)
        label.setMinimumHeight(min_height)
        label.setMaximumHeight(max_height)
        label.setScaledContents(False)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def _set_app_mode(self, mode: str) -> None:
        frontend = mode == "frontend"
        self.mode_stack.setCurrentWidget(self.frontend_page if frontend else self.backend_page)
        self.btn_mode_frontend.setChecked(frontend)
        self.btn_mode_backend.setChecked(not frontend)
        self.btn_mode_frontend.setProperty("active", frontend)
        self.btn_mode_backend.setProperty("active", not frontend)
        for btn in (self.btn_mode_frontend, self.btn_mode_backend):
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        if frontend:
            self._resize_frontend_page()
            self._set_frontend_music_enabled(True)
            self._start_frontend_backdrop_video()
            self._refresh_frontend_backdrop()
            self._refresh_frontend_stage()
        else:
            self._set_frontend_music_enabled(False)
            self._set_frontend_loading(False)
            self._stop_frontend_backdrop_video()
            self.mode_stack.setMinimumHeight(0)
            self.mode_stack.setMaximumHeight(16777215)

    def _resize_frontend_page(self) -> None:
        if not hasattr(self, "frontend_page"):
            return
        viewport_h = 720
        if hasattr(self, "_scroll_area") and self._scroll_area is not None:
            viewport_h = max(520, self._scroll_area.viewport().height())
        target_h = max(500, min(760, viewport_h - 190))
        self.mode_stack.setMinimumHeight(target_h)
        self.mode_stack.setMaximumHeight(target_h)
        self.frontend_page.setMinimumHeight(target_h)
        self.frontend_page.setMaximumHeight(target_h)
        self.frontend_backdrop_label.setMinimumHeight(target_h)
        self.frontend_backdrop_label.setMaximumHeight(target_h)

    def _selected_frontend_dance_id(self) -> str | None:
        if not hasattr(self, "frontend_dance_combo"):
            return None
        dance_id = self.frontend_dance_combo.currentData()
        return str(dance_id) if dance_id else None

    def _frontend_expected_people(self) -> int:
        if not hasattr(self, "frontend_players_combo"):
            return self._expected_people()
        data = self.frontend_players_combo.currentData()
        try:
            return max(1, min(4, int(data)))
        except (TypeError, ValueError):
            return 1

    def _refresh_frontend_controls(self) -> None:
        if not hasattr(self, "btn_frontend_start"):
            return
        busy = self._comparison_workers_busy()
        has_dance = self._selected_frontend_dance_id() is not None
        running = (
            self._ref_practice_timer is not None
            and self._ref_practice_timer.isActive()
        )
        self.frontend_dance_combo.setEnabled(not busy and not running)
        self.frontend_players_combo.setEnabled(not busy and not running)
        for btn in self._frontend_company_buttons.values():
            btn.setEnabled(not busy and not running)
        for btn in self._frontend_artist_buttons.values():
            btn.setEnabled(not busy and not running)
        for btn in self._frontend_dance_card_buttons.values():
            btn.setEnabled(not busy and not running)
        for btn in self._frontend_player_buttons.values():
            btn.setEnabled(not busy and not running)
        self.btn_frontend_calibrate.setEnabled(not busy and not running and has_dance)
        self.btn_frontend_start.setEnabled((not busy and has_dance) or running)
        self.btn_frontend_start.setText("Stop" if running else "Start")

    def _sync_backend_library_combo_to_frontend(self, dance_id: str | None) -> None:
        if not dance_id:
            return
        for i in range(self.library_combo.count()):
            if self.library_combo.itemData(i) == dance_id:
                self.library_combo.blockSignals(True)
                self.library_combo.setCurrentIndex(i)
                self.library_combo.blockSignals(False)
                return

    def _set_frontend_loading(self, loading: bool, status: str = "Cueing video") -> None:
        if not hasattr(self, "frontend_loading_overlay"):
            return
        self.frontend_loading_status.setText(status)
        self.frontend_loading_overlay.setVisible(loading)
        if loading:
            self.frontend_loading_overlay.raise_()

    def _mark_frontend_backdrop_ready(self) -> None:
        if self._frontend_backdrop_ready:
            return
        self._frontend_backdrop_ready = True
        self._set_frontend_loading(False)
        self._play_frontend_music_if_ready()

    def _ensure_frontend_music_player(self) -> None:
        if self._frontend_music_player is not None or not FRONTEND_MUSIC_PATH.is_file():
            return
        self._frontend_audio_output = QAudioOutput(self)
        self._frontend_audio_output.setVolume(0.22)
        self._frontend_music_player = QMediaPlayer(self)
        self._frontend_music_player.setAudioOutput(self._frontend_audio_output)
        self._frontend_music_player.setSource(
            QUrl.fromLocalFile(str(FRONTEND_MUSIC_PATH))
        )
        try:
            self._frontend_music_player.setLoops(-1)
        except Exception:
            pass
        self._frontend_music_player.mediaStatusChanged.connect(
            self._on_frontend_music_status_changed
        )

    def _set_frontend_music_enabled(self, enabled: bool) -> None:
        self._frontend_music_should_play = enabled
        if enabled:
            self._ensure_frontend_music_player()
            self._play_frontend_music_if_ready()
            return
        if self._frontend_music_player is not None:
            self._frontend_music_player.pause()

    def _play_frontend_music_if_ready(self) -> None:
        if not self._frontend_music_should_play or not self._frontend_backdrop_ready:
            return
        self._ensure_frontend_music_player()
        if self._frontend_music_player is None:
            return
        if (
            self._frontend_music_player.playbackState()
            != QMediaPlayer.PlaybackState.PlayingState
        ):
            self._frontend_music_player.play()

    def _on_frontend_music_status_changed(self, status) -> None:
        if (
            status == QMediaPlayer.MediaStatus.EndOfMedia
            and self._frontend_music_should_play
        ):
            self._frontend_music_player.setPosition(0)
            self._frontend_music_player.play()
        elif status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            self._play_frontend_music_if_ready()

    def _set_frontend_stage_from_bgr(self, bgr, empty_text: str = "Select a dance") -> None:
        self._frontend_stage_bgr = bgr.copy() if bgr is not None else None
        self._refresh_frontend_stage(empty_text)

    def _refresh_frontend_stage(self, empty_text: str = "Select a dance") -> None:
        if not hasattr(self, "frontend_stage_label"):
            return
        if self._frontend_stage_bgr is None:
            self.frontend_stage_label.clear()
            self.frontend_stage_label.setText(empty_text)
            return
        self._set_bgr_on_preview_label(
            self.frontend_stage_label,
            self._frontend_stage_bgr,
            empty_text,
        )

    def _start_frontend_backdrop_video(self) -> None:
        if not hasattr(self, "frontend_backdrop_label"):
            return
        if (
            self._frontend_backdrop_timer is not None
            and self._frontend_backdrop_timer.isActive()
        ):
            if self._frontend_backdrop_bgr is not None:
                self._mark_frontend_backdrop_ready()
            return
        self._frontend_backdrop_ready = False
        self._set_frontend_loading(True, "Cueing video")
        if not FRONTEND_BACKDROP_VIDEO_PATH.is_file():
            self._frontend_backdrop_bgr = None
            if self._refresh_frontend_backdrop():
                self._mark_frontend_backdrop_ready()
            else:
                self._set_frontend_loading(False)
            return

        cap = cv2.VideoCapture(str(FRONTEND_BACKDROP_VIDEO_PATH))
        if not cap.isOpened():
            cap.release()
            self._frontend_backdrop_bgr = None
            if self._refresh_frontend_backdrop():
                self._mark_frontend_backdrop_ready()
            else:
                self._set_frontend_loading(False)
            return

        self._frontend_backdrop_cap = cap
        self._frontend_backdrop_fps = max(1.0, float(cap.get(cv2.CAP_PROP_FPS) or 24.0))
        self._frontend_backdrop_total_frames = max(
            0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        )
        self._frontend_backdrop_started_at = time.monotonic()
        self._frontend_backdrop_last_frame_index = -1
        # Keep the repaint heartbeat modest; the target frame is chosen from
        # wall-clock time below, so the video stays real-time even if we skip.
        interval_ms = 100
        self._frontend_backdrop_timer = QTimer(self)
        self._frontend_backdrop_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._frontend_backdrop_timer.timeout.connect(self._on_frontend_backdrop_tick)
        self._frontend_backdrop_timer.start(interval_ms)
        self._on_frontend_backdrop_tick()

    def _stop_frontend_backdrop_video(self) -> None:
        if self._frontend_backdrop_timer is not None:
            self._frontend_backdrop_timer.stop()
            self._frontend_backdrop_timer.deleteLater()
            self._frontend_backdrop_timer = None
        if self._frontend_backdrop_cap is not None:
            self._frontend_backdrop_cap.release()
            self._frontend_backdrop_cap = None
        self._frontend_backdrop_total_frames = 0
        self._frontend_backdrop_started_at = 0.0
        self._frontend_backdrop_last_frame_index = -1
        self._frontend_backdrop_ready = False

    def _on_frontend_backdrop_tick(self) -> None:
        if self._frontend_backdrop_cap is None:
            return
        if self._frontend_backdrop_started_at <= 0.0:
            self._frontend_backdrop_started_at = time.monotonic()

        elapsed = max(0.0, time.monotonic() - self._frontend_backdrop_started_at)
        target_index = int(elapsed * self._frontend_backdrop_fps)
        if self._frontend_backdrop_total_frames > 0:
            target_index %= self._frontend_backdrop_total_frames

        if target_index == self._frontend_backdrop_last_frame_index:
            return
        if (
            self._frontend_backdrop_total_frames > 0
            and target_index < self._frontend_backdrop_last_frame_index
        ):
            self._frontend_backdrop_cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        elif target_index > self._frontend_backdrop_last_frame_index + 1:
            skipped = target_index - self._frontend_backdrop_last_frame_index - 1
            if skipped <= 12:
                for _ in range(skipped):
                    if not self._frontend_backdrop_cap.grab():
                        break
            else:
                self._frontend_backdrop_cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)

        ret, frame = self._frontend_backdrop_cap.read()
        if not ret or frame is None:
            self._frontend_backdrop_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frontend_backdrop_started_at = time.monotonic()
            self._frontend_backdrop_last_frame_index = -1
            ret, frame = self._frontend_backdrop_cap.read()
        if not ret or frame is None:
            self._stop_frontend_backdrop_video()
            self._frontend_backdrop_bgr = None
            if self._refresh_frontend_backdrop():
                self._mark_frontend_backdrop_ready()
            else:
                self._set_frontend_loading(False)
            return
        current = int(self._frontend_backdrop_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self._frontend_backdrop_last_frame_index = max(target_index, current)
        self._frontend_backdrop_bgr = frame
        if self._refresh_frontend_backdrop():
            self._mark_frontend_backdrop_ready()

    def _refresh_frontend_backdrop(self) -> bool:
        if not hasattr(self, "frontend_backdrop_label"):
            return False
        tgt = self.frontend_backdrop_label.size()
        if tgt.width() < 20 or tgt.height() < 20:
            return False
        if self._frontend_backdrop_bgr is not None:
            pm = ui_utils.bgr_to_qpixmap(self._frontend_backdrop_bgr)
        else:
            pm = self._frontend_backdrop_static_pixmap or QPixmap()
        if pm.isNull():
            self.frontend_backdrop_label.clear()
            return False
        scaled = pm.scaled(
            tgt,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.frontend_backdrop_label.setPixmap(scaled)
        return True

    # --- lifecycle ---

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._preview_source_bgr is not None:
            self._refresh_preview()
        self._refresh_comparison_preview_panels()
        if self.mode_stack.currentWidget() is self.frontend_page:
            self._resize_frontend_page()
        self._refresh_frontend_backdrop()
        self._refresh_frontend_stage()
        if self._overlay_display_bgr is not None:
            self._refresh_overlay_preview()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_playback()
        self._stop_ref_practice_playback()
        self._set_frontend_music_enabled(False)
        if self._frontend_music_player is not None:
            self._frontend_music_player.stop()
        self._stop_frontend_backdrop_video()
        self._teardown_overlay()
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            if not self._worker.wait(8000):
                self._worker.terminate()
                self._worker.wait(2000)
        if self._calibration_worker is not None and self._calibration_worker.isRunning():
            self._calibration_worker.requestInterruption()
            self._calibration_worker.wait(6000)
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
        self._reset_seek_ui()
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
        self._play_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self._play_total_frames <= 0:
            self._play_total_frames = max(
                0, int(self._play_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )
        if self._play_total_frames > 0 and self._play_current_frame_index >= self._play_total_frames - 1:
            self._play_current_frame_index = 0
        start_frame = max(0, min(self._play_current_frame_index, max(0, self._play_total_frames - 1)))
        self._play_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self._play_started_at = time.monotonic()
        speed = self._playback_speed_ratio()
        self._play_started_at -= start_frame / max(1e-6, self._play_base_fps * speed)
        self._play_last_frame_index = start_frame - 1

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._play_timer.start(self._compute_play_interval_ms())
        self._on_play_tick()
        msg = (
            "Playing processed video in real time (no sound in-app)… "
            "(click Play again to stop)"
        )
        if pb_note:
            msg = f"{pb_note} {msg}"
        self.status_label.setText(msg)
        self.btn_play.setText("Stop")

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
        self._setup_processed_seek(path)
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
        target_index = self._clock_target_frame_index(
            self._play_started_at,
            self._play_base_fps,
            loop=False,
            total_frames=self._play_total_frames,
        )
        if target_index < 0:
            return
        if self._play_total_frames > 0 and target_index >= self._play_total_frames:
            self._play_current_frame_index = max(0, self._play_total_frames - 1)
            self._update_seek_ui(self._play_current_frame_index)
            self._stop_playback()
            self.status_label.setText("Playback finished.")
            return
        if target_index <= self._play_last_frame_index:
            return
        if target_index > self._play_last_frame_index + 1:
            self._play_cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ret, frame = self._play_cap.read()
        if not ret or frame is None:
            self._stop_playback()
            self.status_label.setText("Playback finished.")
            return
        current = int(self._play_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self._play_last_frame_index = max(target_index, current)
        self._play_current_frame_index = self._play_last_frame_index
        self._update_seek_ui(self._play_current_frame_index)
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
        self._reset_seek_ui()
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.playback_speed_slider.setEnabled(False)
        self.btn_speed_reset.setEnabled(False)
        self.btn_play.setText("Play")
        self.btn_process.setEnabled(True)

        self.path_label.setText(self._video_path)
        self.meta_label.setText(meta.summary())

        self._raw_first_frame = first.copy()
        self._refresh_pose_preview()
        self.status_label.setText(
            "Preview shows pose overlay on the first frame. Click Render overlay to process the full clip."
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
        self._play_started_at = 0.0
        self._play_last_frame_index = -1
        self.btn_play.setText("Play")

    def _playback_speed_ratio(self) -> float:
        v = self.playback_speed_slider.value() / 100.0
        return max(0.25, min(2.0, v))

    def _compute_play_interval_ms(self) -> int:
        """Short UI heartbeat; frame index is chosen from wall-clock time."""
        return 16

    def _clock_target_frame_index(
        self,
        started_at: float,
        base_fps: float,
        *,
        loop: bool,
        total_frames: int,
    ) -> int:
        if started_at <= 0.0:
            return -1
        elapsed = max(0.0, time.monotonic() - started_at)
        idx = int(elapsed * max(1.0, base_fps) * self._playback_speed_ratio())
        if loop and total_frames > 0:
            return idx % total_frames
        return idx

    def _sync_processed_playback_clock(self) -> None:
        if self._play_cap is None or self._play_started_at <= 0.0:
            return
        current = max(0, int(self._play_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        speed = self._playback_speed_ratio()
        self._play_started_at = time.monotonic() - (
            current / max(1e-6, self._play_base_fps * speed)
        )
        self._play_last_frame_index = current
        self._play_current_frame_index = current

    def _update_playback_speed_label(self) -> None:
        self.playback_speed_label.setText(
            f"Speed: {self._playback_speed_ratio():.2f}x"
        )

    def _on_playback_speed_changed(self, _value: int) -> None:
        self._update_playback_speed_label()
        if self._play_timer is not None and self._play_timer.isActive():
            self._sync_processed_playback_clock()
            self._play_timer.setInterval(self._compute_play_interval_ms())
        if self._ref_practice_timer is not None and self._ref_practice_timer.isActive():
            self._sync_ref_practice_clock()

    def _on_playback_speed_reset(self) -> None:
        self.playback_speed_slider.setValue(100)

    def _format_time(self, seconds: float) -> str:
        total = max(0, int(round(seconds)))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _duration_seconds(self) -> float:
        if self._play_base_fps <= 0 or self._play_total_frames <= 0:
            return 0.0
        return self._play_total_frames / self._play_base_fps

    def _frame_for_seek_value(self, value: int) -> int:
        if self._play_total_frames <= 1:
            return 0
        frac = max(0.0, min(1.0, value / max(1, self.seek_slider.maximum())))
        return int(round(frac * (self._play_total_frames - 1)))

    def _seek_value_for_frame(self, frame_index: int) -> int:
        if self._play_total_frames <= 1:
            return 0
        frac = max(0.0, min(1.0, frame_index / (self._play_total_frames - 1)))
        return int(round(frac * self.seek_slider.maximum()))

    def _reset_seek_ui(self) -> None:
        self._seek_is_dragging = False
        self._play_current_frame_index = 0
        self._play_total_frames = 0
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(0)
        self.seek_slider.blockSignals(False)
        self.seek_slider.setEnabled(False)
        self.seek_current_label.setText("0:00")
        self.seek_duration_label.setText("0:00")

    def _setup_processed_seek(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._reset_seek_ui()
            return
        try:
            self._play_base_fps, _ = video_utils.resolve_playback_fps(cap, self._meta)
            self._play_total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self._play_current_frame_index = 0
            enabled = self._play_total_frames > 1 and self._play_base_fps > 0
            self.seek_slider.setEnabled(enabled)
            self.seek_duration_label.setText(self._format_time(self._duration_seconds()))
            self._update_seek_ui(0)
            if enabled:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._set_preview_from_bgr(frame)
        finally:
            cap.release()

    def _update_seek_ui(self, frame_index: int) -> None:
        if self._play_total_frames <= 0:
            self.seek_current_label.setText("0:00")
            return
        frame_index = max(0, min(frame_index, self._play_total_frames - 1))
        self.seek_current_label.setText(
            self._format_time(frame_index / max(1e-6, self._play_base_fps))
        )
        self.seek_duration_label.setText(self._format_time(self._duration_seconds()))
        if not self._seek_is_dragging:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self._seek_value_for_frame(frame_index))
            self.seek_slider.blockSignals(False)

    def _on_seek_pressed(self) -> None:
        if not self.seek_slider.isEnabled():
            return
        self._seek_is_dragging = True

    def _on_seek_moved(self, value: int) -> None:
        if self._play_total_frames <= 0:
            return
        frame = self._frame_for_seek_value(value)
        self.seek_current_label.setText(
            self._format_time(frame / max(1e-6, self._play_base_fps))
        )

    def _on_seek_released(self) -> None:
        if not self.seek_slider.isEnabled() or self._processed_path is None:
            self._seek_is_dragging = False
            return
        frame = self._frame_for_seek_value(self.seek_slider.value())
        self._seek_is_dragging = False
        self._seek_processed_video(frame)

    def _seek_processed_video(self, frame_index: int) -> None:
        if self._processed_path is None or self._play_total_frames <= 0:
            return
        frame_index = max(0, min(frame_index, self._play_total_frames - 1))
        self._play_current_frame_index = frame_index
        self._play_last_frame_index = frame_index
        if self._play_cap is not None:
            self._play_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self._play_cap.read()
            if ret and frame is not None:
                self._set_preview_from_bgr(frame)
                self._play_current_frame_index = max(
                    frame_index,
                    int(self._play_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
                )
                self._play_last_frame_index = self._play_current_frame_index
            self._sync_processed_playback_clock()
        else:
            frame = video_utils.read_frame_at_index(self._processed_path, frame_index)
            if frame is not None:
                self._set_preview_from_bgr(frame)
        self._update_seek_ui(self._play_current_frame_index)

    # --- dance comparison ---

    def _comparison_workers_busy(self) -> bool:
        ca = (
            self._calibration_worker is not None
            and self._calibration_worker.isRunning()
        )
        ex = self._extract_worker is not None and self._extract_worker.isRunning()
        co = self._compare_worker is not None and self._compare_worker.isRunning()
        return ca or ex or co

    def _set_compare_ui_busy(self, busy: bool) -> None:
        if busy:
            self._stop_overlay_playback()
            self.btn_overlay_play.setEnabled(False)
        self.btn_load_ref.setEnabled(not busy)
        self.btn_load_user.setEnabled(not busy)
        self.expected_people_combo.setEnabled(not busy)
        self.btn_scan_ref.setEnabled(not busy and self._ref_path is not None)
        self.btn_scan_user.setEnabled(not busy and self._user_path is not None)
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
        self._refresh_frontend_controls()

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

    def _expected_people(self) -> int:
        data = self.expected_people_combo.currentData()
        try:
            return max(1, min(4, int(data)))
        except (TypeError, ValueError):
            return 1

    def _on_expected_people_changed(self, _index: int) -> None:
        self._ref_calibration = None
        self._user_calibration = None
        self._frontend_calibration = None
        if hasattr(self, "frontend_players_combo"):
            expected = self._expected_people()
            self.frontend_players_combo.blockSignals(True)
            for i in range(self.frontend_players_combo.count()):
                if self.frontend_players_combo.itemData(i) == expected:
                    self.frontend_players_combo.setCurrentIndex(i)
                    break
            self.frontend_players_combo.blockSignals(False)
        self._refresh_calibration_labels()
        self.compare_status_label.setText(
            "Calibration setting changed. Scan loaded videos again before extracting poses."
        )
        if hasattr(self, "frontend_status_label"):
            self.frontend_status_label.setText("Player count changed. Calibrate again before starting.")

    def _on_scan_ref_clicked(self) -> None:
        if not self._ref_path or self._comparison_workers_busy():
            return
        self._start_calibration_scan(self._ref_path, "reference", "ref")

    def _on_scan_user_clicked(self) -> None:
        if not self._user_path or self._comparison_workers_busy():
            return
        self._start_calibration_scan(self._user_path, "performance", "user")

    def _start_calibration_scan(self, path: str, label: str, target: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Wait for video overlay processing to finish before scanning.",
            )
            return
        self._calibration_target = target
        self._calibration_worker = CalibrationScanWorker(
            path,
            label=label,
            expected_people=self._expected_people(),
        )
        self._calibration_worker.progress.connect(self._on_calibration_progress)
        self._calibration_worker.finished_ok.connect(self._on_calibration_finished)
        self._calibration_worker.failed.connect(self._on_calibration_failed)
        self._calibration_worker.finished.connect(self._on_calibration_thread_finished)
        self._set_compare_ui_busy(True)
        self.btn_process.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.compare_status_label.setText(f"Scanning {label} calibration…")
        if target == "frontend" and hasattr(self, "frontend_status_label"):
            self.frontend_status_label.setText(f"Calibrating {label}…")
        self._refresh_frontend_controls()
        self._calibration_worker.start()

    def _on_calibration_progress(self, pct: int, message: str) -> None:
        self.progress.setValue(max(0, min(100, pct)))
        self.compare_status_label.setText(message)
        if self._calibration_target == "frontend" and hasattr(self, "frontend_status_label"):
            self.frontend_status_label.setText(message)

    def _on_calibration_finished(self, report: CalibrationReport) -> None:
        if self._calibration_target == "ref":
            self._ref_calibration = report
        elif self._calibration_target == "user":
            self._user_calibration = report
        else:
            self._frontend_calibration = report
        self._refresh_calibration_labels()
        self.compare_status_label.setText(report.one_line())
        if self._calibration_target == "frontend" and hasattr(self, "frontend_status_label"):
            self.frontend_status_label.setText(report.one_line())
            if self._pending_frontend_start_after_calibration:
                self._pending_frontend_start_after_calibration = False
                if report.is_ready:
                    self._frontend_start_ready_after_calibration = True

    def _on_calibration_failed(self, message: str) -> None:
        if message != "Cancelled.":
            QMessageBox.warning(self, "Calibration scan", message)
        self.compare_status_label.setText("Calibration scan stopped.")
        if self._calibration_target == "frontend" and hasattr(self, "frontend_status_label"):
            self.frontend_status_label.setText("Calibration stopped.")
        self._pending_frontend_start_after_calibration = False
        self._frontend_start_ready_after_calibration = False

    def _on_calibration_thread_finished(self) -> None:
        target = self._calibration_target
        self._calibration_worker = None
        self.btn_load.setEnabled(True)
        self.btn_process.setEnabled(self._video_path is not None)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_compare_ui_busy(False)
        self._refresh_frontend_controls()
        if target == "frontend" and self._frontend_start_ready_after_calibration:
            self._frontend_start_ready_after_calibration = False
            QTimer.singleShot(0, self._on_frontend_start_clicked)

    def _refresh_calibration_labels(self) -> None:
        ref_text = (
            "Reference calibration: " + self._ref_calibration.details_text()
            if self._ref_calibration is not None
            else "Reference calibration: not scanned"
        )
        user_text = (
            "Performance calibration: " + self._user_calibration.details_text()
            if self._user_calibration is not None
            else "Performance calibration: not scanned"
        )
        self.ref_calibration_label.setText(ref_text)
        self.user_calibration_label.setText(user_text)

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
            self._ref_calibration = None
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
            self._user_calibration = None
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
        self._refresh_calibration_labels()
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
        if not self._confirm_calibration_ready("reference", self._ref_calibration):
            return
        self._extract_target = "ref"
        self._start_extract_sequence(self._ref_path, "reference")

    def _on_process_user_clicked(self) -> None:
        if not self._user_path or self._comparison_workers_busy():
            return
        if not self._confirm_calibration_ready("performance", self._user_calibration):
            return
        self._extract_target = "user"
        self._start_extract_sequence(self._user_path, "user")

    def _confirm_calibration_ready(
        self,
        label: str,
        report: CalibrationReport | None,
    ) -> bool:
        if report is not None and report.is_ready:
            return True
        if report is None:
            text = (
                f"The {label} video has not been calibration-scanned yet.\n\n"
                "Scanning checks whether the dancer is visible, centered, and trackable. "
                "Proceed with pose extraction anyway?"
            )
        else:
            text = (
                f"The {label} video calibration is Risky.\n\n"
                f"{report.details_text()}\n\n"
                "Proceed with pose extraction anyway?"
            )
        choice = QMessageBox.question(
            self,
            "Calibration warning",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return choice == QMessageBox.StandardButton.Yes

    def _confirm_pair_calibration_ready(self) -> bool:
        warnings: list[str] = []
        if self._ref_calibration is None:
            warnings.append("Reference has not been scanned.")
        elif self._ref_calibration.is_risky:
            warnings.append("Reference scan is Risky.")
        if self._user_calibration is None:
            warnings.append("Performance has not been scanned.")
        elif self._user_calibration.is_risky:
            warnings.append("Performance scan is Risky.")
        if not warnings:
            return True
        choice = QMessageBox.question(
            self,
            "Calibration warning",
            "\n".join(warnings)
            + "\n\nScores may be less trustworthy. Compare anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return choice == QMessageBox.StandardButton.Yes

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
        if not self._confirm_pair_calibration_ready():
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
        self.btn_play.setText("Play")
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

    def _normalize_catalog_text(self, text: str) -> str:
        return "".join(ch for ch in text.casefold() if ch.isalnum())

    def _classify_frontend_dance(self, dance) -> tuple[str, str]:
        haystack = self._normalize_catalog_text(
            " ".join(
                [
                    getattr(dance, "name", ""),
                    getattr(dance, "source_path", ""),
                    getattr(dance, "video_path", ""),
                ]
            )
        )
        for company, artists in FRONTEND_ARTIST_COMPANIES.items():
            for artist, aliases in artists.items():
                for alias in aliases:
                    if self._normalize_catalog_text(alias) in haystack:
                        return company, artist
        return "OTHER", "Other"

    def _frontend_catalog_entries(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for dance in list_dances():
            company, artist = self._classify_frontend_dance(dance)
            duration = self._format_time(dance.duration_sec) if dance.duration_sec > 0 else ""
            thumb = video_utils.DANCE_LIBRARY_DIR / dance.folder_name / "thumbnail.jpg"
            entries.append(
                {
                    "id": dance.dance_id,
                    "name": dance.name,
                    "company": company,
                    "artist": artist,
                    "duration": duration,
                    "thumb": thumb if thumb.is_file() else None,
                    "created_at": dance.created_at,
                }
            )
        return sorted(
            entries,
            key=lambda e: (
                FRONTEND_COMPANY_ORDER.index(str(e["company"]))
                if str(e["company"]) in FRONTEND_COMPANY_ORDER
                else 99,
                str(e["artist"]).casefold(),
                str(e["name"]).casefold(),
            ),
        )

    def _filtered_frontend_entries(
        self,
        entries: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        company = self._frontend_selected_company
        artist = self._frontend_selected_artist
        filtered = [
            e
            for e in entries
            if company == "ALL" or str(e["company"]) == company
        ]
        if artist != "ALL":
            filtered = [e for e in filtered if str(e["artist"]) == artist]
        return filtered

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget()
            child_layout = item.layout()
            if child is not None:
                child.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _refresh_frontend_button_state(self, btn: QPushButton, selected: bool) -> None:
        btn.setChecked(selected)
        btn.setProperty("selected", "true" if selected else "false")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _set_frontend_company_filter(self, company: str) -> None:
        self._frontend_selected_company = company
        self._frontend_selected_artist = "ALL"
        self._refresh_frontend_dance_combo()

    def _set_frontend_artist_filter(self, artist: str) -> None:
        self._frontend_selected_artist = artist
        self._refresh_frontend_dance_combo()

    def _set_frontend_players(self, count: int) -> None:
        if not hasattr(self, "frontend_players_combo"):
            return
        for i in range(self.frontend_players_combo.count()):
            if int(self.frontend_players_combo.itemData(i)) == count:
                self.frontend_players_combo.setCurrentIndex(i)
                return

    def _sync_frontend_player_buttons(self) -> None:
        expected = self._frontend_expected_people()
        for count, btn in self._frontend_player_buttons.items():
            self._refresh_frontend_button_state(btn, count == expected)

    def _sync_frontend_dance_card_selection(self) -> None:
        selected = self._selected_frontend_dance_id()
        for dance_id, btn in self._frontend_dance_card_buttons.items():
            self._refresh_frontend_button_state(btn, dance_id == selected)

    def _select_frontend_dance_id(self, dance_id: str) -> None:
        for i in range(self.frontend_dance_combo.count()):
            if self.frontend_dance_combo.itemData(i) == dance_id:
                self.frontend_dance_combo.setCurrentIndex(i)
                self._sync_frontend_dance_card_selection()
                return

    def _refresh_frontend_filters(self, entries: list[dict[str, object]]) -> None:
        company_counts = {company: 0 for company in FRONTEND_COMPANY_ORDER}
        company_counts["ALL"] = len(entries)
        for entry in entries:
            company = str(entry["company"])
            company_counts[company] = company_counts.get(company, 0) + 1

        available_companies = {
            "ALL",
            "SM",
            "JYP",
            "YG",
            "HYBE",
        }
        if company_counts.get("OTHER", 0) > 0:
            available_companies.add("OTHER")
        if self._frontend_selected_company not in available_companies:
            self._frontend_selected_company = "ALL"

        for company, btn in self._frontend_company_buttons.items():
            count = company_counts.get(company, 0)
            btn.setText(f"{FRONTEND_COMPANY_LABELS.get(company, company)}\n{count}")
            btn.setVisible(company in available_companies)
            self._refresh_frontend_button_state(
                btn, company == self._frontend_selected_company
            )

        self._clear_layout(self.frontend_artist_buttons_row)
        self._frontend_artist_buttons = {}
        scoped = [
            e
            for e in entries
            if self._frontend_selected_company == "ALL"
            or str(e["company"]) == self._frontend_selected_company
        ]
        artist_counts: dict[str, int] = {"ALL": len(scoped)}
        for entry in scoped:
            artist = str(entry["artist"])
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        if (
            self._frontend_selected_artist != "ALL"
            and self._frontend_selected_artist not in artist_counts
        ):
            self._frontend_selected_artist = "ALL"
        for artist in sorted(
            artist_counts,
            key=lambda value: (value != "ALL", value.casefold()),
        ):
            label = "ALL" if artist == "ALL" else artist
            btn = QPushButton(f"{label}  {artist_counts[artist]}")
            btn.setCheckable(True)
            btn.setObjectName("ArtistChip")
            btn.clicked.connect(
                lambda _checked=False, a=artist: self._set_frontend_artist_filter(a)
            )
            self._frontend_artist_buttons[artist] = btn
            self._refresh_frontend_button_state(
                btn, artist == self._frontend_selected_artist
            )
            self.frontend_artist_buttons_row.addWidget(btn)
        self.frontend_artist_buttons_row.addStretch(1)

    def _refresh_frontend_dance_cards(self, entries: list[dict[str, object]]) -> None:
        self._clear_layout(self.frontend_dance_cards_grid)
        self._frontend_dance_card_buttons = {}
        filtered = self._filtered_frontend_entries(entries)
        self.frontend_library_count_label.setText(
            "1 track" if len(filtered) == 1 else f"{len(filtered)} tracks"
        )
        if not filtered:
            empty = QLabel("No saved dances in this group yet.")
            empty.setObjectName("FrontendEmptyState")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.frontend_dance_cards_grid.addWidget(empty, 0, 0, 1, 2)
            return

        for idx, entry in enumerate(filtered):
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setObjectName("DanceCard")
            btn.setProperty("company", str(entry["company"]))
            duration = str(entry["duration"])
            meta = f"{entry['artist']} · {duration}" if duration else str(entry["artist"])
            btn.setText(f"{entry['name']}\n{meta}\n{entry['company']}")
            btn.setMinimumHeight(126)
            btn.setMinimumWidth(230)
            thumb = entry.get("thumb")
            if isinstance(thumb, Path) and thumb.is_file():
                btn.setIcon(QIcon(str(thumb)))
                btn.setIconSize(QSize(72, 72))
            dance_id = str(entry["id"])
            btn.clicked.connect(
                lambda _checked=False, did=dance_id: self._select_frontend_dance_id(did)
            )
            self._frontend_dance_card_buttons[dance_id] = btn
            row, col = divmod(idx, 2)
            self.frontend_dance_cards_grid.addWidget(btn, row, col)
        self.frontend_dance_cards_grid.setColumnStretch(0, 1)
        self.frontend_dance_cards_grid.setColumnStretch(1, 1)
        self._sync_frontend_dance_card_selection()

    def _refresh_frontend_dance_combo(self, *, select_id: str | None = None) -> None:
        if not hasattr(self, "frontend_dance_combo"):
            return
        entries = self._frontend_catalog_entries()
        selected = select_id or self._selected_frontend_dance_id()
        if select_id:
            for entry in entries:
                if entry["id"] == select_id:
                    self._frontend_selected_company = str(entry["company"])
                    self._frontend_selected_artist = str(entry["artist"])
                    break
        self.frontend_dance_combo.blockSignals(True)
        self.frontend_dance_combo.clear()
        self.frontend_dance_combo.addItem("(select a dance)", None)
        for entry in entries:
            duration = str(entry["duration"])
            label = f"{entry['name']}  ·  {duration}" if duration else str(entry["name"])
            self.frontend_dance_combo.addItem(label, str(entry["id"]))
        if selected:
            for i in range(self.frontend_dance_combo.count()):
                if self.frontend_dance_combo.itemData(i) == selected:
                    self.frontend_dance_combo.setCurrentIndex(i)
                    break
        self.frontend_dance_combo.blockSignals(False)
        self._refresh_frontend_filters(entries)
        self._refresh_frontend_dance_cards(entries)
        self._sync_frontend_player_buttons()
        if self._selected_frontend_dance_id():
            self._on_frontend_dance_changed(self.frontend_dance_combo.currentIndex())
        else:
            self.frontend_song_title_label.setText("Select a dance")
            self.frontend_song_meta_label.setText("Pick a company, artist, and saved dance.")
            self.frontend_ready_badge.setText("READY")
            self.frontend_status_label.setText("Select a saved dance.")
            self._set_frontend_stage_from_bgr(None, "Select a dance")
        self._refresh_frontend_controls()

    def _on_frontend_dance_changed(self, _index: int) -> None:
        self._frontend_calibration = None
        dance_id = self._selected_frontend_dance_id()
        if dance_id is None:
            self.frontend_song_title_label.setText("Select a dance")
            self.frontend_song_meta_label.setText("Pick a company, artist, and saved dance.")
            self.frontend_ready_badge.setText("READY")
            self.frontend_status_label.setText("Select a saved dance.")
            self._set_frontend_stage_from_bgr(None, "Select a dance")
            self._sync_frontend_dance_card_selection()
            self._refresh_frontend_controls()
            return

        self._sync_backend_library_combo_to_frontend(dance_id)
        try:
            md, seq = load_dance(dance_id)
        except Exception as e:
            self.frontend_song_title_label.setText("Saved dance unavailable")
            self.frontend_song_meta_label.setText("Check the backend library.")
            self.frontend_ready_badge.setText("WAIT")
            self.frontend_status_label.setText(f"Could not load saved dance: {e}")
            self._set_frontend_stage_from_bgr(None, "Saved dance unavailable")
            self._refresh_frontend_controls()
            return

        video_path = md.video_path or seq.source_path
        first = video_utils.read_frame_at_index(video_path, 0) if video_path else None
        company, artist = self._classify_frontend_dance(md)
        duration = self._format_time(md.duration_sec) if md.duration_sec > 0 else ""
        meta_parts = [FRONTEND_COMPANY_LABELS.get(company, company), artist]
        if duration:
            meta_parts.append(duration)
        self.frontend_song_title_label.setText(md.name)
        self.frontend_song_meta_label.setText(" · ".join(meta_parts))
        self.frontend_ready_badge.setText("READY")
        if first is not None:
            if md.mirror_for_practice:
                first = cv2.flip(first, 1)
            self._set_frontend_stage_from_bgr(first, "Ready")
        else:
            self._set_frontend_stage_from_bgr(None, "Preview unavailable")
        players = self._frontend_expected_people()
        player_text = "1 player" if players == 1 else f"{players} players"
        self.frontend_status_label.setText(f"{md.name} selected · {player_text}")
        self._sync_frontend_dance_card_selection()
        self._refresh_frontend_controls()

    def _on_frontend_players_changed(self, _index: int) -> None:
        expected = self._frontend_expected_people()
        for i in range(self.expected_people_combo.count()):
            if self.expected_people_combo.itemData(i) == expected:
                self.expected_people_combo.setCurrentIndex(i)
                break
        self._frontend_calibration = None
        self.frontend_status_label.setText("Player count changed. Calibrate again before starting.")
        self._sync_frontend_player_buttons()
        self._refresh_frontend_controls()

    def _on_frontend_calibrate_clicked(self) -> None:
        if self._comparison_workers_busy():
            return
        dance_id = self._selected_frontend_dance_id()
        if not dance_id:
            self._pending_frontend_start_after_calibration = False
            QMessageBox.information(self, "Game", "Choose a saved dance first.")
            return
        self._sync_backend_library_combo_to_frontend(dance_id)
        try:
            md, seq = load_dance(dance_id)
        except Exception as e:
            self._pending_frontend_start_after_calibration = False
            QMessageBox.warning(self, "Game", f"Could not load saved dance:\n{e}")
            return
        video_path = md.video_path or seq.source_path
        if not video_path or not Path(video_path).is_file():
            self._pending_frontend_start_after_calibration = False
            QMessageBox.warning(self, "Game", "The saved dance video file is missing.")
            return
        self._frontend_calibration = None
        self._start_calibration_scan(video_path, "game setup", "frontend")

    def _on_frontend_start_clicked(self) -> None:
        if (
            self._ref_practice_timer is not None
            and self._ref_practice_timer.isActive()
        ):
            self._stop_ref_practice_playback()
            self.frontend_status_label.setText("Stopped.")
            self._refresh_frontend_controls()
            return

        if self._comparison_workers_busy():
            return
        dance_id = self._selected_frontend_dance_id()
        if not dance_id:
            QMessageBox.information(self, "Game", "Choose a saved dance first.")
            return

        if self._frontend_calibration is None:
            self._pending_frontend_start_after_calibration = True
            self._on_frontend_calibrate_clicked()
            return
        if self._frontend_calibration.is_risky:
            choice = QMessageBox.question(
                self,
                "Calibration warning",
                self._frontend_calibration.details_text()
                + "\n\nStart anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return

        self._sync_backend_library_combo_to_frontend(dance_id)
        self._on_load_library_dance_clicked()
        if self._active_library_dance_id != dance_id or not self._ref_path:
            return
        self._set_app_mode("frontend")
        self._on_ref_practice_play_clicked()
        players = self._frontend_expected_people()
        name = self._active_library_dance_name or "Dance"
        self.frontend_status_label.setText(f"{name} · {players} player(s)")
        self._refresh_frontend_controls()

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
        self._refresh_frontend_dance_combo(select_id=select_id)

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
            "Reference loaded from library. Use Practice play, then load and process your video."
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
        self._ref_practice_started_at = 0.0
        self._ref_practice_last_frame_index = -1
        self._ref_practice_total_frames = 0
        self.btn_ref_practice_play.setText("Practice play")
        self._refresh_frontend_controls()

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
        self.btn_play.setText("Play")

        self._ref_practice_cap = cv2.VideoCapture(self._ref_path)
        if not self._ref_practice_cap.isOpened():
            QMessageBox.warning(self, "Practice", f"Could not open:\n{self._ref_path}")
            self._ref_practice_cap = None
            return

        pb_fps, pb_note = video_utils.resolve_playback_fps(self._ref_practice_cap, self._ref_meta)
        self._ref_practice_base_fps = max(1.0, float(pb_fps))
        self._ref_practice_total_frames = max(
            0, int(self._ref_practice_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self._ref_practice_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._ref_practice_started_at = time.monotonic()
        self._ref_practice_last_frame_index = -1
        self._ref_practice_timer = QTimer(self)
        self._ref_practice_timer.timeout.connect(self._on_ref_practice_tick)
        self._ref_practice_timer.start(16)
        self.btn_ref_practice_play.setText("Stop practice")
        self._refresh_frontend_controls()
        msg = "Playing reference for practice in real time (silent)…"
        if pb_note:
            msg = f"{pb_note} {msg}"
        self.ref_preview_status.setText(msg)

    def _on_ref_practice_tick(self) -> None:
        if self._ref_practice_cap is None:
            return
        target_index = self._clock_target_frame_index(
            self._ref_practice_started_at,
            self._ref_practice_base_fps,
            loop=True,
            total_frames=self._ref_practice_total_frames,
        )
        if target_index < 0:
            return
        if target_index < self._ref_practice_last_frame_index:
            self._ref_practice_cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        elif target_index > self._ref_practice_last_frame_index + 1:
            self._ref_practice_cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        elif target_index == self._ref_practice_last_frame_index:
            return
        ret, frame = self._ref_practice_cap.read()
        if not ret or frame is None:
            self._ref_practice_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._ref_practice_started_at = time.monotonic()
            self._ref_practice_last_frame_index = -1
            ret, frame = self._ref_practice_cap.read()
        if not ret or frame is None:
            self._stop_ref_practice_playback()
            self.ref_preview_status.setText("Practice playback ended (read error).")
            return
        current = int(self._ref_practice_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self._ref_practice_last_frame_index = max(target_index, current)
        disp = self._ref_bgr_for_practice_display(frame)
        self._set_bgr_on_preview_label(self.ref_preview_label, disp, "")
        self._set_frontend_stage_from_bgr(disp, "")
        self.ref_preview_status.setText(f"Practice playback · ~frame {self._ref_practice_last_frame_index + 1}")

    def _sync_ref_practice_clock(self) -> None:
        if self._ref_practice_cap is None or self._ref_practice_started_at <= 0.0:
            return
        current = max(0, int(self._ref_practice_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        speed = self._playback_speed_ratio()
        self._ref_practice_started_at = time.monotonic() - (
            current / max(1e-6, self._ref_practice_base_fps * speed)
        )
        self._ref_practice_last_frame_index = current
