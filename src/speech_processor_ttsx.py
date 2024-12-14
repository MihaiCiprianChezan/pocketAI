import io
import threading
import warnings
import numpy as np
import pyttsx3
import speech_recognition as sr
import torch
import whisper
import vosk
import pyaudio
import json

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore",
                        message="Some parameters are on the meta device because they were offloaded to the cpu.")
warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the Whisper model (for transcription if needed)
model = whisper.load_model("base", device=device)
model = model.to(device)


class SpeechProcessorTTSX:
    # Constants (taken from both provided examples)
    DEFAULT_VOICE = "Microsoft Sonia (Natural) - English (United Kingdom)"  # Or your preferred default
    DEFAULT_RATE = 180
    WAIT_DURATION_MS = 2000
    FADEOUT_DURATION_MS = 500
    SAMPLE_RATE = 16000
    FADEOUT_STEPS = 10

    def __init__(self):
        self.recognizer = sr.Recognizer()  # From the first example
        self.microphone = sr.Microphone(sample_rate=self.SAMPLE_RATE)
        self.engine = pyttsx3.init('sapi5')
        self.voices = self._list_voices()
        self.current_voice = self.DEFAULT_VOICE
        self.rate = self.DEFAULT_RATE
        self.stop_playback_event = threading.Event()
        self.playback_thread = None
        self.mixer_lock = threading.Lock()  # not really useful but existing in original version so added for compatibility

        # Vosk Initialization (from the second example)
        self.recognition_model_path = "../models/vosk-model-en-us-daanzu-20200905-lgraph"
        try:
            self.model = vosk.Model(self.recognition_model_path)
            self.recognizer_vosk = vosk.KaldiRecognizer(self.model, 16000)
        except Exception as e:
            print(f"Error initializing Vosk: {e}")
            self.model = None  # Indicate Vosk initialization failed

        self.set_voice(self.DEFAULT_VOICE)  # Initialize voice after listing
        self.set_rate(self.DEFAULT_RATE)

    def _initialize_mixer(self):
        """Dummy method for compatibility."""
        return True

    def is_playing(self):
        try:
            return self.engine.isBusy()
        except Exception as e:
            print(f"Error checking pyttsx3 engine status: {e}")
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
        if self.model:  # Use Vosk if initialized
            return self.recognize_speech_vosk()
        else:  # Fallback to Whisper
            return self.recognize_speech_whisper()

    def recognize_speech_vosk(self):
        try:
            audio_interface = pyaudio.PyAudio()
            stream = audio_interface.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                                          frames_per_buffer=8000)
            stream.start_stream()

            print("Listening for speech (Vosk)...")
            while True:
                data = stream.read(4000, exception_on_overflow=False)
                if len(data) == 0:
                    break
                if self.recognizer_vosk.AcceptWaveform(data):
                    result = json.loads(self.recognizer_vosk.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"Recognized: {text}")
                        return text
        except Exception as e:
            print(f"Vosk recognition error: {e}")
            return ""
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            audio_interface.terminate()

    def recognize_speech_whisper(self):
        try:
            with self.microphone as source:
                print("Listening for speech (Whisper)...")
                audio = self.recognizer.listen(source)
            print("Processing audio...")
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
            result = model.transcribe(audio_data)
            text = result["text"].strip()
            print(f"Recognized: {text}")
            return text
        except Exception as e:
            print(f"Whisper recognition error: {e}")
            return ""

    def _fade_out(self, sound, duration_ms):
        """Dummy method for compatibility."""
        pass

    def read_text(self, text, call_before=None, call_back=None, lang="en"):
        def _read_text_inner(text_to_read):
            try:
                if call_before:
                    call_before()  # Pre-speech actions

                self.engine.say(text_to_read)
                self.engine.startLoop(False)
                while self.engine.isBusy() and not self.stop_playback_event.is_set():
                    self.engine.iterate()
                self.engine.endLoop()
                self.stop_playback_event.clear()

                if call_back:
                    call_back()  # Post-speech actions

            except Exception as e:
                print(f"Error during text reading: {e}")
                self.stop_playback_event.clear()  # Ensure it's cleared on error

        if self.playback_thread and self.playback_thread.is_alive():
            self.stop_sound(lambda: self.read_text(text, call_before, call_back, lang))  # Queue next text
        else:
            self.playback_thread = threading.Thread(target=_read_text_inner, args=(text,), daemon=True)
            self.stop_playback_event.clear()
            self.playback_thread.start()

    def stop_sound(self, call_back=None):
        self.stop_playback_event.set()  # Signal the thread to stop
        if self.playback_thread:
            self.playback_thread.join()  # Correctly placed join call. Waits until current speech finishes or stops.
            self.playback_thread = None  # Reset the thread variable after joining

        if call_back:
            call_back()

    def shutdown(self):
        with self.mixer_lock:  # Acquire the mixer lock (even though mixer use is removed)
            self.engine.stop()
            print("TTS engine shut down.")

    @staticmethod
    def wait(duration_ms):
        import time
        time.sleep(duration_ms / 1000)

#
# # Usage example
# if __name__ == "__main__":
#     processor = SpeechProcessorTTSX()
#
#
#     def test_playback():
#         print("Testing playback...")
#
#         # print("Available Voices:")
#         # processor.list_voices()
#         #
#         # # Change voice dynamically
#         # # processor.set_voice("Microsoft Sonia (Natural)")
#         # processor.read_text("The voice has been changed to Microsoft Sonia.")
#         #
#         # # Change speech rate dynamically
#         # processor.set_rate(200)
#         # processor.read_text("The speech rate has been increased to 200 words per minute.")
#         #
#         # # Reset to default voice and rate
#         # processor.set_voice(processor.DEFAULT_VOICE)
#         # processor.set_rate(processor.DEFAULT_RATE)
#         # processor.read_text("We are now using the default configuration again.")
#         #
#
#         try:
#             processor.read_text("hmm... ooh... ahh... uhh.. aha... sure... oh... ok... yup... yes... right...")
#             processor.wait(1000)
#             processor.stop_sound()
#
#             processor.read_text("This is a test. Let's see if the fade-out works properly.")
#             processor.wait(processor.WAIT_DURATION_MS)
#             processor.stop_sound()
#
#             processor.read_text("This is the second sound test. Let's see if the start of the second sound works properly.")
#             processor.wait(processor.WAIT_DURATION_MS)
#             processor.stop_sound()
#
#             processor.read_text("This is the third test... works properly?")
#
#         # Reading THINKING_SOUNDS (if available)
#
#             from varstore import THINKING_SOUNDS
#             processor.read_text(" ".join(THINKING_SOUNDS))
#         except ImportError:
#             print("THINKING_SOUNDS variable could not be imported from varstore.")
#
#
#     # Run the test playback in a separate thread
#     threading.Thread(target=test_playback, daemon=True).start()
#
#     while True:
#         pass

if __name__ == "__main__":
    processor = SpeechProcessorTTSX()


    def before_text():
        print("About to start reading the text...")


    def after_text():
        print("Finished reading the text.")


    # First playback
    processor.read_text("Hello, this is a test of the text-to-speech system.", before_text, after_text)

    # Simulate waiting before stopping
    import time

    time.sleep(2.5)  # Let it play for a second
    processor.stop_sound()

    # Immediate second playback
    processor.read_text("This is the second message after stopping the first one.", before_text, after_text)
    time.sleep(6)