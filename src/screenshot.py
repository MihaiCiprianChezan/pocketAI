import io

from PIL import Image
from PySide6.QtCore import QBuffer
from PySide6.QtGui import QGuiApplication
from app_logger import AppLogger

class ScreenshotUtil:
    def __init__(self, logger=None):
        # Ensure there is a QApplication instance for QScreen to work
        self.logger = logger if logger else AppLogger()
        self.app = QGuiApplication.instance()
        if self.app is None:
            self.app = QGuiApplication([])
        self.name = self.__class__.__name__

    def capture(self, save_path="screenshot-full-screen.png"):
        """Captures the entire desktop screen and saves the screenshot to the specified path."""
        screen = QGuiApplication.primaryScreen()  # Get the primary screen
        if screen is None:
            self.logger.debug(f"[{self.name}] No screen detected!")
            return False

        # Grab the entire screen (window ID 0 means the whole desktop)
        screenshot = screen.grabWindow(0)
        if screenshot.isNull():
            self.logger.debug(f"[{self.name}] Failed to capture the screen")
            return False

        # Save the screenshot to an image file
        self._save_qpixmap_as_image(screenshot, save_path)
        self.logger.debug(f"[{self.name}] Screenshot saved to: {save_path}")
        return True

    def _save_qpixmap_as_image(self, qpixmap, save_path):
        """
        Converts a QPixmap to a Pillow Image and saves it to the given path.
        """
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)  # Open buffer in read-write mode
        qpixmap.save(buffer, "png")  # Save QPixmap into the buffer as PNG

        # Convert QByteArray from the buffer into bytes
        byte_array = buffer.data()
        buffer.close()

        # Use Pillow to save the image from the buffer
        img = Image.open(io.BytesIO(byte_array))  # Use io.BytesIO with the image bytes for Pillow
        img.save(save_path)

    def capture_partial(self, x, y, width, height, save_path="screenshot-partial.png"):
        """Captures a partial screen area and saves the screenshot to the specified path.

        Args:
            x (int): Top-left corner X coordinate.
            y (int): Top-left corner Y coordinate.
            width (int): Width of the capture area.
            height (int): Height of the capture area.
            save_path (str): File path to save the screenshot.
        """
        screen = QGuiApplication.primaryScreen()  # Get the primary screen
        if screen is None:
            self.logger.debug(f"[{self.name}]No screen detected!")
            return False

        # Grab a specific portion of the screen
        screenshot = screen.grabWindow(0, x, y, width, height)
        if screenshot.isNull():
            self.logger.debug(f"[{self.name}] Failed to capture the screen")
            return False

        # Save the partial screenshot
        self._save_qpixmap_as_image(screenshot, save_path)
        self.logger.debug(f"[{self.name}] Partial screenshot saved to: {save_path}")
        return True

if __name__ == "__main__":
    # Create an instance of the DesktopScreenshot class
    screenshot_manager = ScreenshotUtil()

    # Capture the full desktop screen
    screenshot_manager.capture("full_screenshot.png")

    # Capture a specific region of the screen
    # Example: Capture a region starting from (100, 100) with width 500 and height 300
    screenshot_manager.capture_partial(100, 100, 500, 300, "partial_screenshot.png")