import json
import threading
import time
import traceback
import pyaudio
import pyttsx3
import speech_recognition as sr
import torch
import vosk
from threading import Lock

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class SpeechProcessor:
    # Constants
    DEFAULT_VOICE = "Microsoft Sonia (Natural) - English (United Kingdom)"  # Or your preferred default
    DEFAULT_RATE = 180
    WAIT_DURATION_MS = 2000
    FADEOUT_DURATION_MS = 500
    SAMPLE_RATE = 16000
    FADEOUT_STEPS = 10

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=self.SAMPLE_RATE)
        self.engine = pyttsx3.init('sapi5')
        self.voices = self._list_voices()
        self.current_voice = self.DEFAULT_VOICE
        self.rate = self.DEFAULT_RATE
        self.stop_playback_event = threading.Event()
        self.playback_thread = None
        self.mixer_lock = threading.Lock()

        # Vosk Initialization
        self.recognition_model_path = "../models/vosk-model-small-en-us-0.15"
        try:
            self.model = vosk.Model(self.recognition_model_path)
            self.recognizer_vosk = vosk.KaldiRecognizer(self.model, 16000)
        except Exception as e:
            print(f"Error initializing Vosk: {e}, {traceback.format_exc()}")
            self.model = None
        self.set_voice(self.DEFAULT_VOICE)
        self.set_rate(self.DEFAULT_RATE)

    def is_playing(self):
        try:
            return self.engine.isBusy()
        except Exception as e:
            print(f"Error checking pyttsx3 engine status: {e}, {traceback.format_exc()}")
            return False

    def _list_voices(self):
        """Fetch available voices."""
        voices = self.engine.getProperty('voices')
        voice_dict = {}
        for voice in voices:
            voice_dict[voice.name] = {"id": voice.id, "gender": voice.gender, "languages": voice.languages}
        return voice_dict

    def list_voices(self):
        """List available voices."""
        for name, details in self.voices.items():
            print(f"Name: {name}, ID: {details['id']}, Gender: {details['gender']}, Languages: {details['languages']}")

    def set_voice(self, voice_name):
        if voice_name in self.voices:
            self.engine.setProperty('voice', self.voices[voice_name]['id'])
            self.current_voice = voice_name
            print(f"Voice set to: {voice_name}")
        else:
            print(f"Voice '{voice_name}' not found.")

    def set_rate(self, rate):
        self.engine.setProperty('rate', rate)
        self.rate = rate
        print(f"Rate set to: {rate}")

    def recognize_speech(self):
        if self.model:
            return self.recognize_speech_vosk()
        else:
            raise Exception("Vosk not initialized, cannot recognize speech.")

    def recognize_speech_vosk(self):
        """Recognize speech using Vosk."""
        try:
            audio_interface = pyaudio.PyAudio()
            stream = audio_interface.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
            stream.start_stream()
            print("Listening for speech ...")
            while True:
                data = stream.read(4000, exception_on_overflow=False)
                if len(data) == 0:
                    break
                if self.recognizer_vosk.AcceptWaveform(data):
                    result = json.loads(self.recognizer_vosk.Result())
                    text = result.get("text", "").strip()
                    if text:
                        return text
        except Exception as e:
            print(f"Vosk recognition error: {e}, {traceback.format_exc()}")
            return ""
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            audio_interface.terminate()

    def read_text(self, text, call_before=None, call_back=None, lang="en"):
        """
        Reads the given text aloud using a text-to-speech engine. Prior to reading, it ensures no other playback is currently
        in progress. The method supports optional pre- and post-speech callback functionalities and allows specifying the
        language for the voice used in text-to-speech.
        """
        self.stop_sound()

        with self.mixer_lock:
            def _read_text_inner(text_to_read):
                try:
                    # Pre-speech actions
                    if call_before and callable(call_before):
                        call_before()
                    # Speech
                    self.set_voice(self.current_voice)
                    self.engine.say(text_to_read)
                    self.engine.startLoop(False)
                    while self.engine.isBusy() and not self.stop_playback_event.is_set():
                        self.engine.iterate()
                        self.wait(100)
                    self.engine.endLoop()
                    self.stop_playback_event.clear()
                    # Post-speech actions
                    if call_back and callable(call_back):
                        call_back()
                except Exception as e:
                    print(f"Error during text reading: {e}, {traceback.format_exc()}")
                    self.stop_playback_event.clear()

            if lang:
                # TODO: set the appropriate voice for the appropriate language
                pass

            if self.playback_thread and self.playback_thread.is_alive():
                self.stop_sound()
                self.playback_thread.join()

            self.playback_thread = threading.Thread(target=_read_text_inner, args=(text,), daemon=True)
            self.stop_playback_event.clear()
            self.playback_thread.start()

    def stop_sound(self, call_back=None):
        self.stop_playback_event.set()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()
            self.playback_thread = None
        while self.playback_thread and self.playback_thread.is_alive():
            self.wait(100)

        if call_back and callable(call_back):
            call_back()

    def shutdown(self):
        with self.mixer_lock:
            self.engine.stop()
            print("(i) TTS engine shut down.")

    @staticmethod
    def wait(duration_ms):
        time.sleep(duration_ms / 1000)


if __name__ == "__main__":
    processor = SpeechProcessor()


    def before_text():
        print("About to start reading the text...")


    def after_text():
        print("Finished reading the text.")


    def test_threads():
        # Function to simulate concurrent reads/stops
        print("Starting thread tests...")

        # Store references to all threads for monitoring
        threads = []

        def play_and_stop(index, text, wait_time):
            print(f"[Thread-{index}] Starting playback for message: {text}")
            processor.stop_sound()
            processor.read_text(text, before_text, after_text)
            time.sleep(wait_time)
            processor.stop_sound()
            print(f"[Thread-{index}] Stopped playback.")

        for i in range(5):
            t = threading.Thread(target=play_and_stop, args=(i, f"Message from Thread-{i}", 2), daemon=True)
            threads.append(t)
            t.start()
            time.sleep(0.5)

        for t in threads:
            t.join()

        print("All threads completed without issues.")


    def repeat_stop_test():
        # Repeatedly play and stop to test robustness
        print("\nStarting repeated play/stop test...")
        try:
            for i in range(10):
                print(f"Iteration {i + 1}")
                processor.read_text(f"Test message {i + 1}", before_text, after_text)
                time.sleep(1)
                processor.stop_sound()
                assert processor.playback_thread is None or not processor.playback_thread.is_alive(), \
                    f"Playback thread not cleaned up properly in iteration {i + 1}"
        except AssertionError as e:
            print(f"Error detected: {e}, {traceback.format_exc()}")
        else:
            print("Repeated play/stop test passed successfully.")


    print("\n--- Basic Playback Test ---")
    processor.read_text("Hello, this is a test of the text-to-speech system.", before_text, after_text)
    time.sleep(2.5)
    processor.stop_sound()

    print("\n--- Immediate Playback After Stop ---")
    processor.read_text("This is a test to ensure smooth transition between messages.", before_text, after_text)
    time.sleep(6)

    print("\n--- Thread and Resource Robustness Tests ---")
    test_threads()
    repeat_stop_test()

    print("\n--- All Tests Completed ---")
