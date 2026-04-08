"""Qt helpers: NumPy/OpenCV images to QPixmap."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Convert a BGR uint8 image to QPixmap (RGB888 copy, safe lifetime)."""
    if bgr is None or bgr.size == 0:
        return QPixmap()
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected H×W×3 BGR image")
    # OpenCV is BGR; Qt label expects RGB
    # As contiguous array so QImage buffer stays valid for .copy()
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())
