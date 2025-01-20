import random
import sys

from PySide6.QtCore import QCoreApplication, QMutex, QMutexLocker, QSize, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QAction, QColor, QMouseEvent, QMovie, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsColorizeEffect, QLabel, QMenu, QMessageBox, QVBoxLayout, QWidget

from app_logger import AppLogger


class EnergyBall(QWidget):
    def __init__(self, gif_path="./images/opti200.gif"):
        super().__init__()
        self.name = self.__class__.__name__
        self.logger = AppLogger()
        self.circle_color = QColor(0, 0, 0, 180)
        # Configure the transparent, frameless overlay window
        self.pulsating = None
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")  # Fully transparent widget
        # Load the GIF and dynamically set its size
        self.movie = QMovie(gif_path)
        self.movie.start()
        self.original_size = (self.movie.frameRect().width(), self.movie.frameRect().height())
        # Initialize QLabel
        self.label = QLabel(self)
        self.label.setAttribute(Qt.WA_TranslucentBackground, True)
        self.label.setMovie(self.movie)
        # self.label.hide()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        # State variables for interaction
        self.offset = None
        self.current_color = QColor(0, 0, 0)  # Keeps the current overlay color
        self.is_dragging = False  # Initialize dragging state
        self.color_effect = None
        self.color_animation = None
        self.animation_lock = QMutex()
        # Initialize position
        self.init_position()

    def receive_command(self, command, params):
        if command == "reset_colorized":
            self.reset_colorized(command, params)
        elif command == "change_color":
            self.handle_change_color(command, params)
        elif command == "start_pulsating":
            self.start_pulsating(command, params)
        elif command == "stop_pulsating":
            self.stop_pulsating(command, params)
        elif command == "zoom_effect":
            self.zoom_effect_wrapper(command, params)
        elif command == "rectangle_selection":
            self.rectangle_selection(command, params)
        elif command == "rectangle_selection_timeout":
            self.rectangle_selection_timeout(command, params)
        elif command == "exit":
            QCoreApplication.quit()

    def handle_change_color(self, command, params):
        color = params.get("color", (0, 0, 0))
        if color == (0, 0, 0):
            self.reset_colorized(command, params)
        else:
            self.change_color(command, params)

    # Slot to handle color change
    def change_color(self, command=None, params={}):
        color = params.get("color", (0, 0, 0))
        self.set_colorized(QColor(*color))

    # Slot to start pulsating
    def start_pulsating(self, command=None, _params=None):
        self.pulsate_effect()

    # Slot to stop pulsating
    def stop_pulsating(self, command=None, _params=None):
        if getattr(self, "pulsating", False):
            self.pulsating = False

    # Slot to handle zoom effect
    def zoom_effect_wrapper(self, command, params={}):
        zoom_factor = params.get("factor", 1.1)
        duration = params.get("duration", 100)
        self.zoom_effect(duration, zoom_factor)

    # Slot to handle screeshot
    def rectangle_selection(self, command, params={}):
        rect = params.get("RectangleOverlay", None)
        if rect:
            self.logger.debug(f"[{self.name}] Initializing rectangle selection using RectangleOverlay attribute")
            params['RectangleOverlay'].show()
        else:
            self.logger.debug(f"[{self.name}] RectangleOverlay attribute not found. Cannot initialize rectangle selection.")

    # Slot to handle screeshot hide
    def rectangle_selection_timeout(self, command, params={}):
        rect = params.get("RectangleOverlay", None)
        if rect:
            self.logger.debug(f"[{self.name}] Initializing rectangle selection using RectangleOverlay attribute")
            params['RectangleOverlay'].hide()
        else:
            self.logger.debug(f"[{self.name}] RectangleOverlay attribute not found. Cannot initialize rectangle selection.")

    def show_context_menu(self, position):
        """
        Display a context menu when right-clicking on the energy ball.
        """
        # Create the menu
        menu = QMenu(self)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.quit)
        menu.addAction(exit_action)
        # Display the menu at the requested position (mouse position or event position)
        menu.exec(self.mapToGlobal(position))

    def init_position(self):
        """
        Set the initial position of the widget to the bottom-right corner of the screen with padding.
        """
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        screen_width, screen_height = screen_geometry.width(), screen_geometry.height()
        padding = 100
        widget_width, widget_height = self.original_size
        self.move(screen_width - widget_width - padding, screen_height - widget_height - padding)

    def set_colorized(self, target_color: QColor, duration=500):
        """
        Smoothly transitions the label from fully transparent to the target color.
        Handles calls from multiple threads safely.
        """
        with QMutexLocker(self.animation_lock):  # Lock the mutex
            # Remove any existing effect to avoid conflicts
            existing_effect = self.label.graphicsEffect()
            if existing_effect:
                self.label.setGraphicsEffect(None)

                # Safely delete the existing effect
                if isinstance(existing_effect, QGraphicsColorizeEffect):
                    try:
                        existing_effect.deleteLater()
                    except RuntimeError:
                        # Skip deletion if the object was already deleted
                        self.logger.warning("[WARNING] Attempted to delete an already-deleted QGraphicsColorizeEffect.")

            # If an animation is already running, stop it
            if hasattr(self, 'color_animation') and self.color_animation is not None:
                self.color_animation.stop()
                try:
                    self.color_animation.deleteLater()
                except RuntimeError:
                    self.logger.warning("[WARNING] Attempted to delete an already-deleted QVariantAnimation.")
                self.color_animation = None

            # Create a local graphics effect (not stored in `self`)
            color_effect = QGraphicsColorizeEffect(self)
            color_effect.setColor(target_color)
            self.label.setGraphicsEffect(color_effect)

            # Create a new animation
            self.color_animation = QVariantAnimation(self)
            self.color_animation.setStartValue(0.0)
            self.color_animation.setEndValue(1.0)
            self.color_animation.setDuration(duration)

            # Update the effect's strength during animation
            def update_strength(strength):
                if color_effect:  # Check if color_effect still exists
                    try:
                        color_effect.setStrength(strength)
                    except RuntimeError:
                        self.logger.debug(f"[{self.name}] QGraphicsColorizeEffect already deleted during animation.")

            self.color_animation.valueChanged.connect(update_strength)

            # Debugging info for when the animation completes
            def on_animation_complete():
                if color_effect:
                    try:
                        self.logger.debug(f"[{self.name}] Transition to {target_color} complete.")
                    except RuntimeError:
                        self.logger.debug(f"[{self.name}] QGraphicsColorizeEffect already deleted at animation completion.")

            self.color_animation.finished.connect(on_animation_complete)

            # Start the animation
            self.color_animation.start()

    def reset_color(self):
        """
        Reset the widget to its original appearance by fading out.
        """
        self.set_colorized(QColor(0, 0, 0))  # Transition smoothly back to black (reset state)

    def reset_colorized(self, command, _params=None, duration=500):
        """
        Smoothly transition to no graphic effect.
        """
        # Check if there is an existing effect
        if not self.label.graphicsEffect() or not isinstance(self.label.graphicsEffect(), QGraphicsColorizeEffect):
            return  # No effect to remove
        # Create an animation to reduce the effect's strength smoothly
        animation = QVariantAnimation(self)
        animation.setStartValue(1.0)  # Full strength of the effect
        animation.setEndValue(0.0)  # No effect (reset state)
        animation.setDuration(duration)  # Duration in milliseconds

        def fade_out_effect(opacity):
            # Validate opacity
            if not (0.0 <= opacity <= 1.0):
                raise ValueError("Opacity must be between 0.0 and 1.0")
            if self.label is None:
                raise AttributeError("Label is not set.")
            effect = self.label.graphicsEffect()
            if effect is None:
                # Apply a new graphics effect if one does not exist
                effect = QGraphicsColorizeEffect()
                self.label.setGraphicsEffect(effect)
            if isinstance(effect, QGraphicsColorizeEffect):
                effect.setStrength(opacity)
                if opacity == 0.0:
                    # Optionally remove the effect when faded out
                    self.label.setGraphicsEffect(None)

        # Connect the animation's valueChanged signal to dynamically update the effect
        animation.valueChanged.connect(fade_out_effect)
        animation.start()

    def paintEvent(self, event):
        """
        Ensure the circle remains smoothly centered on the widget and proportional to the ball.
        """
        # Draw the circle
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.circle_color)
        painter.setPen(Qt.NoPen)

        # Always draw the circle centered in the widget
        widget_center = self.rect().center()
        radius = round(max(self.label.width(), self.label.height()) // 2.2)  # Circle matches ball size + padding
        painter.drawEllipse(widget_center.x() - radius, widget_center.y() - radius, 2 * radius, 2 * radius)

    def mousePressEvent(self, event: QMouseEvent):
        """
        Enable dragging the widget when left-clicked, or show context menu on right-click.
        """
        if event.button() == Qt.LeftButton:  # Left click for dragging
            self.is_dragging = True
            self.offset = event.globalPosition().toPoint() - self.pos()
        elif event.button() == Qt.RightButton:  # Right click to show the context menu
            self.show_context_menu(event.pos())
        else:
            super().mousePressEvent(event)  # Call the base class in other situations

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Move the widget as the mouse drags it.
        """
        if self.is_dragging:
            self.move(event.globalPosition().toPoint() - self.offset)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Stop dragging when the mouse is released or close the application
        on right-click with proper checks.
        """
        if event.button() == Qt.LeftButton:
            if hasattr(self, 'is_dragging'):  # Ensure the attribute exists
                self.is_dragging = False
        elif event.button() == Qt.RightButton:
            # Optional: Show a confirmation dialog before closing
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Are you sure you want to exit the application?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                QApplication.quit()
        # Always call the base class implementation to handle remaining logic
        super().mouseReleaseEvent(event)

    def zoom_effect(self, duration=100, zoom_factor=1.1):
        """
        Creates an animated zoom-in and zoom-out effect by scaling the QMovie and QLabel.
        This ensures the ball zooms relative to its center only and avoids conflicts with pulsating.
        """
        original_width, original_height = self.original_size
        zoomed_width = int(original_width * zoom_factor)
        zoomed_height = int(original_height * zoom_factor)

        # Centering is calculated only once, not during every frame
        delta_width = (zoomed_width - original_width) // 2
        delta_height = (zoomed_height - original_height) // 2

        # Create the zoom animation
        animation = QVariantAnimation(self)
        animation.setStartValue(QSize(original_width, original_height))
        animation.setEndValue(QSize(zoomed_width, zoomed_height))
        animation.setDuration(duration)

        def resize_frames(size):
            self.movie.setScaledSize(size)
            self.label.resize(size)

        # Update the widget size (but avoid rapidly changing position)
        animation.valueChanged.connect(resize_frames)

        # Reverse Zoom
        def reverse_zoom():
            reverse_animation = QVariantAnimation(self)
            reverse_animation.setStartValue(QSize(zoomed_width, zoomed_height))
            reverse_animation.setEndValue(QSize(original_width, original_height))
            reverse_animation.setDuration(duration)

            def reverse_resize(size):
                self.movie.setScaledSize(size)
                self.label.resize(size)

            reverse_animation.valueChanged.connect(reverse_resize)
            reverse_animation.start()

        animation.finished.connect(reverse_zoom)
        animation.start()

    def pulsate_effect(self):
        """
        Starts or stops a pulsating effect on the energy ball with random delays between pulses.
        """
        # Toggle the pulsating state
        if getattr(self, "pulsating", False):  # If pulsating is active, stop it
            self.pulsating = False
            if hasattr(self, "pulse_timer"):
                self.pulse_timer.stop()
                self.logger.debug(f"[{self.name}] Pulsating stopped.")
            return  # Exit if we are stopping pulsating
        # Initialize pulsating effect
        self.logger.debug(f"[{self.name}] Pulsating started.")
        self.pulsating = True
        # Create a timer for random intervals
        if not hasattr(self, "pulse_timer"):
            self.pulse_timer = QTimer(self)

        # Define the pulsate animation logic
        def single_pulse(zoom_factor=1.03, duration=1000):
            if not self.pulsating:  # Stop if pulsating was toggled off
                return
            original_width, original_height = self.original_size
            zoomed_width = int(original_width * zoom_factor)
            zoomed_height = int(original_height * zoom_factor)
            # Create the animation
            animation = QVariantAnimation(self)
            animation.setStartValue(QSize(original_width, original_height))  # Start at original size
            animation.setEndValue(QSize(zoomed_width, zoomed_height))  # End at zoomed size
            animation.setDuration(duration)  # Duration of each pulse (grow-shrink cycle)

            def resize_frames(size):
                self.movie.setScaledSize(size)  # Resize the QMovie dynamically

            animation.valueChanged.connect(resize_frames)
            # Reverse animation logic after the animation ends
            animation.finished.connect(lambda: resize_frames(QSize(original_width, original_height)))
            animation.start()

        # Schedule the next pulse with random intervals
        def schedule_next_pulse(rand_a=100, rand_b=1000):
            if self.pulsating:  # Continue pulsating
                single_pulse()
                random_delay = random.randint(rand_a, rand_b)  # Add a random delay (500ms to 1000ms)
                self.pulse_timer.start(random_delay)

        # Connect the timer timeout to the schedule function
        self.pulse_timer.timeout.connect(schedule_next_pulse)
        # Trigger the first pulse immediately
        schedule_next_pulse()

    def keyPressEvent(self, event):
        """
        Handle key events for interaction.
        """
        if event.key() == Qt.Key_1:  # Red
            self.set_colorized(QColor(100, 0, 0))
        elif event.key() == Qt.Key_2:  # Green
            self.set_colorized(QColor(0, 50, 0))
        elif event.key() == Qt.Key_3:  # Blue
            self.set_colorized(QColor(0, 0, 100))
        elif event.key() == Qt.Key_4:  # Yellow (mix of red and green)
            self.set_colorized(QColor(100, 100, 0))
        elif event.key() == Qt.Key_5:  # Cyan (mix of green and blue)
            self.set_colorized(QColor(0, 100, 100))
        elif event.key() == Qt.Key_6:  # Magenta (mix of red and blue)
            self.set_colorized(QColor(100, 0, 100))
        elif event.key() == Qt.Key_7:  # White (all colors combined)
            self.set_colorized(QColor(100, 100, 100))
        elif event.key() == Qt.Key_8:  # Dim Gray (low intensity mix of all colors)
            self.set_colorized(QColor(50, 50, 50))
        elif event.key() == Qt.Key_Space:  # Reset to initial uncolorized state
            self.label.setGraphicsEffect(None)  # Remove any color overlay
        elif event.key() == Qt.Key_9:  # Toggle pulsating effect
            self.pulsate_effect()  # New pulsating effect toggle
            self.logger.debug(f"[{self.name}] Pulsate toggle key detected!")  # Debugging output
        elif event.key() == Qt.Key_0:  # Zoom in and out
            self.zoom_effect()
            self.logger.debug(f"[{self.name}] Zoom key detected!")  # Debug
        elif event.key() == Qt.Key_Return:  # Zoom in and out
            self.stop_pulsating()
            self.logger.debug(f"[{self.name}] Stopping pulsating!")  # Debug
        elif event.key() == Qt.Key_Escape:  # Exit
            self.close()
            QCoreApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    energy_ball = EnergyBall()
    energy_ball.show()
    sys.exit(app.exec())
