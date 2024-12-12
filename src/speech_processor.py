
import os
import threading
from gc import callbacks
from time import sleep
from uuid import uuid4
import numpy as np
import pygame
import speech_recognition as sr
import whisper
from gtts import gTTS

device = "cpu"
print("Torch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("CUDA device name:", torch.cuda.get_device_name(0))
    device = "cuda"

# Initialize the Whisper model
model = whisper.load_model("base", device="cuda, fp16=False)

class SpeechProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=self.sample_rate)
        self.playback_thread = None
        self.stop_playback_event = threading.Event()

        # Initialize Pygame mixer
        pygame.mixer.init()

    def is_playing(self):
        """Check if playback is actively running."""
        return (
                self.playback_thread is not None and
                self.playback_thread.is_alive() and
                pygame.mixer.music.get_busy()
        )

    def recognize_speech(self):
        """Recognize speech input using the microphone and Whisper model."""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source)

            print("Processing audio...")

            # Get the raw audio bytes and convert them to a NumPy array
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe the audio using the Whisper model
            result = model.transcribe(audio_data)
            return result["text"].strip()

        except Exception as e:
            print(f"Error recognizing speech: {e}")
            return ""

    def _play_audio(self, audio_file):
        """Play an audio file with fadeout support for interruption."""
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()  # Start playing the audio

            print(f"Playing audio: {audio_file}")

            while pygame.mixer.music.get_busy():  # Audio is playing
                if self.stop_playback_event.is_set():  # Check for fadeout request
                    print("Fading out audio playback...")

                    # Ensure fadeout completes over 2000ms
                    fade_duration = 1000
                    pygame.mixer.music.fadeout(fade_duration)  # Perform the fadeout
                    sleep(fade_duration / 1000)  # Wait for the fadeout to finish

                    # Ensure the music is stopped completely
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    return

                sleep(0.1)  # Reduce CPU usage by sleeping

            # Stop playback fully after track finishes if no fadeout needed
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()

        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            self.stop_playback_event.clear()  # Reset the fadeout flag

    def _start_playback(self, audio_file, callback=None):
        """Start audio playback in a new thread and interrupt any ongoing playback."""
        if self.playback_thread and self.playback_thread.is_alive():
            print("Stopping previous playback with fadeout...")
            self.stop_playback_event.set()  # Trigger fadeout in the playback thread
            self.playback_thread.join()  # Wait for the previous playback thread to terminate

        # Reset the stop event for subsequent fadeout handling
        self.stop_playback_event.clear()

        # Start a new playback thread
        self.playback_thread = threading.Thread(target=self._play_audio, args=(audio_file,), daemon=True)
        self.playback_thread.start()

        # Start a cleanup thread to safely delete the temporary audio file
        def cleanup(callback_f=callback):
            self.playback_thread.join()  # Wait for playback to complete
            try:
                if callback_f:
                    print('callback...')
                    callback_f()
                if os.path.exists(audio_file):  # Verify if file exists before removing
                    os.remove(audio_file)
                    print(f"Audio file {audio_file} deleted.")
                else:
                    print(f"Audio file {audio_file} already deleted or not found.")
            except Exception as e:
                print(f"Error deleting audio file {audio_file}: {e}")

        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()

    def stop_sound(self):
        """Stop the playback immediately and clear playback resources."""
        print("Stopping audio playback...")
        self.stop_playback_event.set()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()  # Wait for the playback thread to terminate

        pygame.mixer.music.stop()
        pygame.mixer.music.unload()  # Clear mixer cache
        print("Playback stopped.")

    def read_text(self, text, callbefore=None, callback=None):
        """Convert the given text to speech and play it."""
        if text:
            try:
                # Generate an audio file from text
                audio_file = f"{uuid4().hex}.mp3"
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(audio_file)
                if callbefore:
                    print('callbefore...')
                    callbefore()
                # Start playing the audio in parallel
                self._start_playback(audio_file, callback)

            except Exception as e:
                print(f"Error during text-to-speech processing: {e}")
        else:
            print("No text provided to read.")