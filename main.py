#!/usr/bin/env python3
"""Entry point for Dance Pose Desktop."""

import sys

from PySide6.QtWidgets import QApplication

from app.window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Dance Pose Desktop")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
