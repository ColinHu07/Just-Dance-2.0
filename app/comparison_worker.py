"""Background threads for pose-sequence extraction and dance comparison."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app import video_utils
from app.calibration import scan_video_calibration
from app.comparison_types import ComparisonResult, PoseSequence
from app.pose_sequence_extract import extract_pose_sequence_from_video
from app.pose_mirror import mirror_pose_sequence
from app.scoring import compare_pose_sequences

logger = logging.getLogger(__name__)


class ExtractSequenceWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(object)  # PoseSequence
    failed = Signal(str)

    def __init__(self, video_path: str, *, label: str = "video") -> None:
        super().__init__()
        self._path = video_path
        self._label = label

    def run(self) -> None:
        try:
            video_utils.ensure_app_dirs()

            def cb(pct: int, msg: str) -> None:
                self.progress.emit(pct, f"[{self._label}] {msg}")

            seq = extract_pose_sequence_from_video(
                self._path,
                progress_cb=cb,
                cancel_check=self.isInterruptionRequested,
            )
            self.finished_ok.emit(seq)
        except RuntimeError as e:
            if "Cancelled" in str(e):
                self.failed.emit("Cancelled.")
            else:
                self.failed.emit(str(e))
        except video_utils.VideoOpenError as e:
            self.failed.emit(str(e))
        except Exception as e:
            logger.exception("Extract sequence failed")
            self.failed.emit(f"Pose extraction failed: {e}")


class CalibrationScanWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        video_path: str,
        *,
        label: str = "video",
        expected_people: int = 1,
    ) -> None:
        super().__init__()
        self._path = video_path
        self._label = label
        self._expected_people = expected_people

    def run(self) -> None:
        try:
            video_utils.ensure_app_dirs()

            def cb(pct: int, msg: str) -> None:
                self.progress.emit(pct, f"[{self._label}] {msg}")

            report = scan_video_calibration(
                self._path,
                expected_people=self._expected_people,
                progress_cb=cb,
                cancel_check=self.isInterruptionRequested,
            )
            self.finished_ok.emit(report)
        except RuntimeError as e:
            if "Cancelled" in str(e):
                self.failed.emit("Cancelled.")
            else:
                self.failed.emit(str(e))
        except video_utils.VideoOpenError as e:
            self.failed.emit(str(e))
        except Exception as e:
            logger.exception("Calibration scan failed")
            self.failed.emit(f"Calibration scan failed: {e}")


class CompareSequencesWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(object)  # ComparisonResult
    failed = Signal(str)

    def __init__(
        self,
        ref: PoseSequence,
        user: PoseSequence,
        *,
        mirror_reference_for_scoring: bool = False,
    ) -> None:
        super().__init__()
        self._ref = ref
        self._user = user
        self._mirror_reference_for_scoring = mirror_reference_for_scoring

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                self.failed.emit("Cancelled.")
                return
            self.progress.emit("Aligning and scoring…")
            ref = (
                mirror_pose_sequence(self._ref)
                if self._mirror_reference_for_scoring
                else self._ref
            )
            result = compare_pose_sequences(ref, self._user)
            self.finished_ok.emit(result)
        except Exception as e:
            logger.exception("Compare failed")
            self.failed.emit(f"Comparison failed: {e}")


def save_comparison_json(result: ComparisonResult, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result.to_json_dict(), f, indent=2)
