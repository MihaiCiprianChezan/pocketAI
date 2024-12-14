import io
import threading
import warnings
import numpy as np
import pygame
import speech_recognition as sr
import torch
import whisper
from gtts import gTTS

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="Some parameters are on the meta device because they were offloaded to the cpu.")
warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the Whisper model
model = whisper.load_model("base", device=device)
model = model.to(device)

class SpeechProcessor:
    # Constants
    FADEOUT_DURATION_MS = 500
    SAMPLE_RATE = 16000
    WAIT_DURATION_MS = 2000
    FADEOUT_STEPS = 10

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=self.SAMPLE_RATE)
        self.stop_playback_event = threading.Event()
        self.playback_thread = None
        self.fadeout_thread = None
        self.mixer_lock = threading.Lock()
        # Initialize Pygame mixer
        self._initialize_mixer()


    def _initialize_mixer(self):
        """Ensure the Pygame mixer is initialized."""
        with self.mixer_lock:
            if not pygame.mixer.get_init():
                try:
                    pygame.mixer.init()  # Initialize mixer
                    print("Pygame mixer initialized successfully.")
                    print(f"Audio device: {pygame.mixer.get_init()}")
                except pygame.error as e:
                    print(f"Error initializing Pygame mixer: {e}")  # Log any mixer error
                    return False
            return True

    def is_playing(self):
        with self.mixer_lock:
            if not pygame.mixer.get_init():
                print("Mixer not initialized — cannot check if audio is playing.")
                return False
            return pygame.mixer.music.get_busy()

    def recognize_speech(self):
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source)
            print("Processing audio...")
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
            result = model.transcribe(audio_data)
            return result["text"].strip()
        except Exception as e:
            print(f"Error recognizing speech: {e}")
            return ""

    def _fade_out(self, sound, duration_ms):
        """Manually fade out the audio by decreasing volume gradually over duration."""
        if not sound:
            return
        total_steps = self.FADEOUT_STEPS
        step_delay = duration_ms // total_steps
        initial_volume = sound.get_volume()  # Get the current volume
        volume_step = initial_volume / total_steps
        for step in range(total_steps):
            if not pygame.mixer.get_busy():  # If the sound is no longer playing, stop fading
                break
            new_volume = max(0, initial_volume - volume_step * (step + 1))
            sound.set_volume(new_volume)  # Update the volume
            pygame.time.wait(step_delay)  # Wait for the next step
        sound.stop()  # Stop playback after the fade-out is complete

    def read_text(self, text, call_before=None, call_back=None, lang="en"):
        try:
            tts = gTTS(text=text, lang=lang)
            audio_stream = io.BytesIO()
            tts.write_to_fp(audio_stream)
            audio_stream.seek(0)
            if call_before:
                print("read_text() Calling before function...")
                call_before()
            self.play_stream(audio_stream.read(), call_back=call_back)
        except Exception as e:
            print(f"Error generating text-to-speech audio: {e}")

    def play_stream(self, audio_stream, call_back=None, fade_out_duration_ms=FADEOUT_DURATION_MS):
        def playback_worker(stream):
            try:
                # Initialize sound stream
                pygame.mixer.init()
                pygame_sound = pygame.mixer.Sound(io.BytesIO(stream))
                pygame_sound.set_volume(1.0)  # Start at full volume
                pygame_sound.play()  # Start playback
                print("Playing audio stream...")

                while pygame.mixer.get_busy():  # Wait while audio is playing
                    if self.stop_playback_event.is_set():  # Stop requested
                        self._fade_out(pygame_sound, fade_out_duration_ms)
                        break
                    pygame.time.wait(100)  # Wait before re-checking if sound is playing
            except Exception as e:
                print(f"Error during audio playback: {e}")
            finally:
                pygame.mixer.quit()  # Cleanup the mixer
                self.stop_playback_event.clear()
            if call_back:
                print("play_stream() Calling after function...")
                call_back()

        self.playback_thread = threading.Thread(target=playback_worker, args=(audio_stream,), daemon=True)
        self.playback_thread.start()

    def stop_sound(self, call_back=None):
        with self.mixer_lock:
            if not pygame.mixer.get_init():
                # print("Mixer not initialized — cannot stop sound.")
                return
            print(f"Stopping audio playback with fade-out...")
            self.stop_playback_event.set()
            # Make sure we wait a bit longer than the fade
            pygame.time.wait(self.FADEOUT_DURATION_MS + 100)
        if call_back:
            call_back()

    def shutdown(self):
        """Cleanly shut down the mixer."""
        with self.mixer_lock:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
                print("Pygame mixer shut down.")

    @staticmethod
    def wait(duration_ms):
        pygame.time.wait(duration_ms)

# Usage
if __name__ == "__main__":
    processor = SpeechProcessor()

    def test_playback():
        print("Testing playback...")

        # processor.read_text("hmm... ooh... ahh... uhh.. aha... sure... oh... ok... yup... yes... right...")
        # pygame.time.wait(1000)
        # processor.stop_sound()
        # # pygame.time.wait(1000)
        # processor.read_text("This is a test. Let's see if the fade-out works properly.")
        # pygame.time.wait(processor.WAIT_DURATION_MS)
        # processor.stop_sound()
        # # pygame.time.wait(WAIT_DURATION_MS)
        # processor.read_text("This is the second sound test. Let's see if the start of the second sound works properly.")
        # pygame.time.wait(processor.WAIT_DURATION_MS)
        # processor.stop_sound()
        # # pygame.time.wait(WAIT_DURATION_MS)
        # processor.read_text("This is third test... works properly?")

        from varstore import THINKING_SOUNDS
        processor.read_text(" ".join(THINKING_SOUNDS))


    threading.Thread(target=test_playback, daemon=True).start()

    while True:
        pass
