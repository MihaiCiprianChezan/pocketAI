import sys
import threading
from PySide6.QtWidgets import QApplication
from floating_energy_ball import FloatingEnergyBall
from voice_util_app import VoiceUtilApp

import warnings

# warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class VoiceUtilThread(threading.Thread):
    def __init__(self, voice_app):
        super().__init__()
        self.voice_app = voice_app

    def run(self):
        """
        Run the voice utility app in its own thread.
        """
        self.voice_app.run()


def main():
    # Create the QApplication
    app = QApplication(sys.argv)

    # Initialize FloatingEnergyBall
    energy_ball = FloatingEnergyBall("images/opti100.gif")
    energy_ball.show()

    # Initialize VoiceUtilApp
    voice_util_app = VoiceUtilApp()
    voice_thread = VoiceUtilThread(voice_util_app)

    # Connect signals from VoiceUtilApp to FloatingEnergyBall slots
    voice_util_app.send_command_to_ball.connect(energy_ball.receive_command)

    # Start the voice processing thread
    voice_thread.start()

    # Run the Qt event loop
    sys.exit(app.exec())

    # Properly clean up (wait for threads to finish)
    voice_thread.join()

if __name__ == "__main__":
    main()