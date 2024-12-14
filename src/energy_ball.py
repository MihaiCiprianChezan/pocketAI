import random
import sys

from PySide6.QtCore import QSize, QVariantAnimation, QTimer, QCoreApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QMouseEvent, QAction
from PySide6.QtGui import QMovie
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMessageBox
from PySide6.QtWidgets import QGraphicsColorizeEffect
from PySide6.QtWidgets import QMenu


class EnergyBall(QWidget):
    def __init__(self, gif_path="./images/opti100.gif"):
        super().__init__()
        self.circle_color = QColor(0, 0, 0, 127)
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

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # State variables for interaction
        self.is_dragging = False
        self.offset = None
        self.current_color = QColor(0, 0, 0)  # Keeps the current overlay color
        self.is_dragging = False  # Initialize dragging state
        # Initialize position
        self.init_position()

    def receive_command(self, command, params):
        if command == "reset_colorized":
            self.reset_colorized(command, params)
        elif command == "change_color":
            self.change_color(command, params)
        elif command == "start_pulsating":
            self.start_pulsating(command, params)
        elif command == "stop_pulsating":
            self.stop_pulsating(command, params)
        elif command == "zoom_effect":
            self.zoom_effect_wrapper(command, params)
        elif command == "exit":
            QCoreApplication.quit()

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

    def set_colorized(self, target_color: QColor, duration=800):
        """
        Smoothly transition to the target color using QVariantAnimation.
        :param target_color: QColor to transition to.
        """
        animation = QVariantAnimation(self)
        animation.setStartValue(self.current_color)
        animation.setEndValue(target_color)
        animation.setDuration(duration)  # Transition duration in ms
        animation.valueChanged.connect(self.apply_color)
        animation.start()

    def apply_color(self, color: QColor):
        """
        Update the color overlay dynamically during the animation.
        :param color: QColor to apply during the transition.
        """
        self.current_color = color
        # Apply the color overlay effect
        color_effect = QGraphicsColorizeEffect(self)
        color_effect.setColor(color)
        self.label.setGraphicsEffect(color_effect)

    def reset_color(self):
        """
        Reset the widget to its original appearance by fading out.
        """
        self.set_colorized(QColor(0, 0, 0))  # Transition smoothly back to black (reset state)

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
        """
        original_width, original_height = self.original_size
        zoomed_width = int(original_width * zoom_factor)
        zoomed_height = int(original_height * zoom_factor)

        # Create the animation for the zoom effect
        animation = QVariantAnimation(self)
        animation.setStartValue(QSize(original_width, original_height))  # Start at original size
        animation.setEndValue(QSize(zoomed_width, zoomed_height))  # End at zoomed size
        animation.setDuration(duration)  # Duration of 1 second

        # Handle value changes during the animation
        def resize_frames(size):
            self.movie.setScaledSize(size)

        # Connect animation changes to the resize logic
        animation.valueChanged.connect(resize_frames)

        # Reverse the animation after 1 second
        def reverse_zoom():
            reverse_animation = QVariantAnimation(self)
            reverse_animation.setStartValue(QSize(zoomed_width, zoomed_height))
            reverse_animation.setEndValue(QSize(original_width, original_height))
            reverse_animation.setDuration(duration)  # Reverse duration of 1 second
            reverse_animation.valueChanged.connect(resize_frames)
            reverse_animation.start()

        # Trigger the reverse animation after finishing the first one
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
                print("Pulsating stopped.")
            return  # Exit if we are stopping pulsating

        # Initialize pulsating effect
        print("Pulsating started.")
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

    def reset_colorized(self, command, _params=None, duration=900):
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
            if self.label.graphicsEffect() and isinstance(self.label.graphicsEffect(), QGraphicsColorizeEffect):
                self.label.graphicsEffect().setStrength(opacity)  # Set effect strength dynamically
            if opacity == 0.0:  # At the end of the transition, remove the effect
                self.label.setGraphicsEffect(None)

        # Connect the animation's valueChanged signal to dynamically update the effect
        animation.valueChanged.connect(fade_out_effect)
        animation.start()

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
            print("Pulsate toggle key detected!")  # Debugging output
        elif event.key() == Qt.Key_0:  # Zoom in and out
            self.zoom_effect()
            print("Zoom key detected!")  # Debug
        elif event.key() == Qt.Key_Return:  # Zoom in and out
            self.stop_pulsating()
            print("stoping pulsating!")  # Debug
        elif event.key() == Qt.Key_Escape:  # Exit
            self.close()
            QCoreApplication.quit()


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.circle_color)
        painter.setPen(Qt.NoPen)
        radius = max(self.width()-20, self.height()-20) // 2
        painter.drawEllipse(self.rect().center(), radius, radius)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    energy_ball = EnergyBall()
    energy_ball.show()
    sys.exit(app.exec())
