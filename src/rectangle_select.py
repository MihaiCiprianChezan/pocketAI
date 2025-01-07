import datetime
from pathlib import Path
import time
import uuid

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget
from screenshot import ScreenshotUtil


class RectangleOverlay(QWidget):
    selected = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.screenshot_util = ScreenshotUtil()
        # Make widget frameless and stay on top
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Set a semi-transparent background to capture mouse events
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.start_point = None
        self.current_rect = QRect()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Save starting point
            self.start_point = event.globalPos()
            self.current_rect = QRect()

    def mouseMoveEvent(self, event):
        if self.start_point and event.buttons() & Qt.LeftButton:
            # Update rectangle while dragging
            self.current_rect = QRect(self.start_point, event.globalPos())
            self.update()

    def get_partial(self, x, y, w, h):
        formatted_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        file_name = Path(f"screenshot_{uuid.uuid4().hex}_{formatted_time}.png")
        dest_file = str(self.screenshot_util.SCREENSHOT_FOLDER / file_name)
        self.screenshot_util.capture_partial(x, y, w, h, dest_file)

    def mouseReleaseEvent(self, event):
        if self.start_point and event.button() == Qt.LeftButton:
            # Finalize rectangle and print its position and dimensions
            self.current_rect = QRect(self.start_point, event.globalPos())
           # print(f"[RectangleOverlay] Rectangle drawn at ({self.current_rect.x()}, {self.current_rect.y()}) "
           #        f"with width {self.current_rect.width()} and height {self.current_rect.height()}")

            self.get_coords(
                self.current_rect.x(),
                self.current_rect.y(),
                self.current_rect.width(),
                self.current_rect.height()
            )

            self.start_point = None
            self.current_rect = QRect()
            self.update()

            painter = QPainter(self)
            painter.setBrush(QColor(0, 0, 0, 120))  # Black with 100 alpha (semi-transparent)
            painter.setPen(Qt.NoPen)
            painter.drawRect(self.rect())

            self.hide()

    def get_coords(self, x, y, width, height):
        """
        Emits the signal with coordinates upon selection.
        """
        self.selected.emit(x, y, width, height)

    def paintEvent(self, event):
        """Draw the semi-transparent overlay and the selection rectangle."""
        painter = QPainter(self)

        # 1. Draw the semi-transparent gray overlay (entire screen)
        painter.setBrush(QColor(0, 0, 0, 120))  # Black with 100 alpha (semi-transparent)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        # 2. Clear the area inside the selection rectangle (remove dimming within the rectangle)
        if not self.current_rect.isNull():
            painter.setCompositionMode(QPainter.CompositionMode_Clear)  # Clears the rectangle area from the overlay
            painter.drawRect(self.current_rect)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)  # Restore normal drawing mode

            # 3. Draw the red border of the selection rectangle
            painter.setBrush(Qt.NoBrush)  # No fill for the rectangle
            painter.setPen(QPen(Qt.red, 2))  # Red border with 2px thickness
            painter.drawRect(self.current_rect)


def show_rectangle_overlay():
    app = QApplication.instance()  # Check for existing app instance
    if not app:
        app = QApplication([])

    overlay = RectangleOverlay()
    overlay.show()

    app.exec()  # Run the app if it isn't already running


if __name__ == "__main__":
    show_rectangle_overlay()
