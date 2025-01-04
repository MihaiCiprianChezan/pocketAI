import time

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QPen, QColor, QBrush


class RectangleOverlay(QWidget):
    def __init__(self):
        super().__init__()

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

    def mouseReleaseEvent(self, event):
        if self.start_point and event.button() == Qt.LeftButton:
            # Finalize rectangle and print its position and dimensions
            self.current_rect = QRect(self.start_point, event.globalPos())
            print(f"Rectangle drawn at ({self.current_rect.x()}, {self.current_rect.y()}) "
                  f"with width {self.current_rect.width()} and height {self.current_rect.height()}")
            # Close the overlay after completing the rectangle
            painter = QPainter(self)
            semi_transparent_brush = QBrush(QColor(0, 0, 0, 100))
            painter.fillRect(self.rect(), semi_transparent_brush)
            self.start_point = None
            self.current_rect = QRect()
            self.update()
            time.sleep(.3)
            self.hide()


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

    #
    # def paintEvent(self, event):
    #     painter = QPainter(self)
    #
    #     # Fill the entire screen with a semi-transparent layer
    #     semi_transparent_brush = QBrush(QColor(0, 0, 0, 100))  # Black with transparency
    #     painter.fillRect(self.rect(), semi_transparent_brush)
    #
    #     # Draw the dynamic rectangle
    #     if not self.current_rect.isNull():
    #         pen = QPen(QColor(255, 0, 0), 2)  # Red rectangle with 2px border
    #         painter.setPen(pen)
    #         painter.drawRect(self.current_rect)  # Draw user-rectangle


def show_rectangle_overlay():
    app = QApplication.instance()  # Check for existing app instance
    if not app:
        app = QApplication([])

    overlay = RectangleOverlay()
    overlay.show()

    app.exec()  # Run the app if it isn't already running


if __name__ == "__main__":
    show_rectangle_overlay()