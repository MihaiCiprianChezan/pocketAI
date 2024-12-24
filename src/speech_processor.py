import io
import json
import os
import sys
import threading
import traceback
import warnings

import numpy as np
import pyaudio

from utils import MODELS_DIR

with open(os.devnull, "w") as f:
    sys.stdout = f
    import pygame  # The Pygame module initializes here

    sys.stdout = sys.__stdout__  # Restore stdout
import pyttsx3
import speech_recognition as sr
import torch
import vosk
import whisper
from fastpunct import FastPunct
from gtts import gTTS

from app_logger import AppLogger

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="Some parameters are on the meta device because they were offloaded to the cpu.")
warnings.filterwarnings("ignore", category=FutureWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"


class SpeechProcessor:
    # Constants
    FADEOUT_DURATION_MS = 500
    SAMPLE_RATE = 16000
    # WAIT_DURATION_MS = 2000
    FADEOUT_STEPS = 10

    def __init__(self, use_vosk=True):
        self.logger = AppLogger()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=self.SAMPLE_RATE)
        self.stop_playback_event = threading.Event()
        self.mixer_lock = threading.Lock()
        self.use_vosk = use_vosk
        self.fastpunct = FastPunct()
        self.playback_thread = None
        self._initialize_mixer()
        self.file_lock = threading.Lock()
        self.logger.debug(f"[SOUND_PROCESSOR] Using device: {device}")
        if self.use_vosk:
            # Vosk initialization
            # self.recognition_model_path = "../models/vosk-model-small-en-us-0.15"
            self.recognition_model_path = str(MODELS_DIR / "vosk-model-small-en-us-0.15")
            # self.recognition_model_path = str(MODELS_DIR / "vosk-model-en-us-0.22-lgraph")
            try:
                self.model_vosk = vosk.Model(self.recognition_model_path)
                self.recognizer_vosk = vosk.KaldiRecognizer(self.model_vosk, 16000)
                self.logger.debug("[SOUND_PROCESSOR] Vosk model initialized successfully.")
            except Exception as e:
                self.logger.debug(f"[SOUND_PROCESSOR] Error initializing Vosk: {e}, {traceback.format_exc()}")
                self.model_vosk = None
        else:
            # Whisper initialization
            self.model_whisper = whisper.load_model("base", device=device).to(device)

    def _initialize_mixer(self):
        """Ensure the Pygame mixer is initialized."""
        with self.mixer_lock:
            if not pygame.mixer.get_init():
                try:
                    pygame.mixer.init()  # Initialize mixer
                    self.logger.debug("[SOUND_PROCESSOR] Pygame mixer initialized successfully.")
                    self.logger.debug(f"[SOUND_PROCESSOR] Audio device: {pygame.mixer.get_init()}")
                except pygame.error as e:
                    self.logger.debug(f"[SOUND_PROCESSOR] Error initializing Pygame mixer: {e}, {traceback.format_exc()}")  # Log any mixer error
                    return False
            return True

    def is_playing(self):
        with self.mixer_lock:
            if not pygame.mixer.get_init():
                self.logger.debug("[SOUND_PROCESSOR] Mixer not initialized — cannot check if audio is playing.")
                return False
            return pygame.mixer.music.get_busy()

    def recognize_speech(self, use_vosk=True):
        """Recognize speech from the microphone using Whisper or Vosk."""
        if use_vosk or self.use_vosk:
            if not self.model_vosk:
                self.logger.debug("[SOUND_PROCESSOR] osk model is not initialized. Falling back to Whisper.")
                return self._recognize_speech_whisper()
            return self._recognize_speech_vosk()
        else:
            return self._recognize_speech_whisper()

    def _recognize_speech_whisper(self):
        """Recognize speech using Whisper."""
        try:
            with self.microphone as source:
                self.logger.debug("[SOUND_PROCESSOR] Listening with Whisper...")
                audio = self.recognizer.listen(source)
            self.logger.debug("[SOUND_PROCESSOR] Processing audio with Whisper...")
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model_whisper.transcribe(audio_data)
            return result["text"].strip()
        except Exception as e:
            self.logger.debug(f"[SOUND_PROCESSOR] Error recognizing speech with Whisper: {e}, {traceback.format_exc()}")
            return ""

    def restore_punctuation(self, text):
        """
        Adds punctuation and capitalization to the transcribed text using fastPunct.
        """
        try:
            punctuated_text = self.fastpunct.punct(text, correct=True)  # correct=True enables capitalization
            return punctuated_text
        except Exception as e:
            self.logger.debug(f"[SOUND_PROCESSOR] Error while restoring punctuation: {e}")
            return text

    def _recognize_speech_vosk(self):
        """Recognize speech using Vosk."""
        try:
            audio_interface = pyaudio.PyAudio()
            stream = audio_interface.open(
                format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
            stream.start_stream()
            self.logger.debug("[SOUND_PROCESSOR] Listening with Vosk...")
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
            self.logger.debug(f"[SOUND_PROCESSOR] Error recognizing speech with Vosk: {e}, {traceback.format_exc()}")
            return ""
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            audio_interface.terminate()

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
        if sound:
            sound.stop()  # Stop playback after the fade-out is complete

    def read_text(self, text, call_before=None, call_back=None, lang="en", tld="co.uk"):
        try:
            tts = gTTS(text=text, lang=lang, tld=tld)
            audio_stream = io.BytesIO()
            tts.write_to_fp(audio_stream)
            audio_stream.seek(0)
            if call_before:
                self.logger.debug("[SOUND_PROCESSOR] : read_text.call_before() ...")
                call_before()
            self.play_stream(audio_stream.read(), call_back=call_back)
        except Exception as e:
            self.logger.debug(f"[SOUND_PROCESSOR] Error generating text-to-speech audio: {e}, {traceback.format_exc()}")

    def play_stream(self, audio_stream, call_back=None, fade_out_duration_ms=FADEOUT_DURATION_MS, do_not_interrupt=False):
        def playback_worker(stream):
            try:
                with self.mixer_lock:
                    # Initialize the mixer only if it's not already initialized
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                        self.logger.debug("[SOUND_PROCESSOR] Mixer initialized for playback.")

                    # Load and play the sound
                    pygame_sound = pygame.mixer.Sound(io.BytesIO(stream))
                    pygame_sound.set_volume(1.0)
                    pygame_sound.play()
                    self.logger.debug("[SOUND_PROCESSOR] Playing audio stream...")

                # Wait for the playback to complete or stop if requested
                while pygame.mixer.get_busy():
                    if self.stop_playback_event.is_set() and not do_not_interrupt:  # Do not interrupt if flagged
                        self.logger.debug("[SOUND_PROCESSOR] Stop requested. Initiating fade-out...")
                        with self.mixer_lock:
                            self._fade_out(pygame_sound, fade_out_duration_ms)
                        break
                    pygame.time.wait(100)

            except pygame.error as e:
                self.logger.debug(f"[SOUND_PROCESSOR] Pygame error during playback: {e}")
            except Exception as e:
                self.logger.debug(f"[SOUND_PROCESSOR] Unhandled error during playback: {e}, {traceback.format_exc()}")
            finally:
                with self.mixer_lock:
                    # Quit mixer after playback is done
                    if pygame.mixer.get_init():
                        pygame.mixer.quit()
                        self.logger.debug("[SOUND_PROCESSOR] Mixer quit after playback.")
                self.stop_playback_event.clear()

                # Execute callback after playback ends
                if call_back and callable(call_back):
                    self.logger.debug("[SOUND_PROCESSOR] play_stream.call_back() ...")
                    call_back()

        # If `do_not_interrupt` is False, stop any currently playing sound
        if not do_not_interrupt:
            if self.playback_thread and self.playback_thread.is_alive():
                self.logger.debug("[SOUND_PROCESSOR] Interrupting previous playback thread...")
                self.stop_playback_event.set()
                self.playback_thread.join()  # Wait for the previous thread to properly finish

        # If `do_not_interrupt` is True, wait for any ongoing playback to finish
        if do_not_interrupt and self.playback_thread and self.playback_thread.is_alive():
            self.logger.debug("[SOUND_PROCESSOR] Waiting for previous sound to complete...")
            self.playback_thread.join()  # Wait until the active playback finishes completely

        # Start a new playback thread
        self.stop_playback_event.clear()
        self.playback_thread = threading.Thread(target=playback_worker, args=(audio_stream,), daemon=True)
        self.playback_thread.start()

    def stop_sound(self, call_back=None):
        with self.mixer_lock:
            if not pygame.mixer.get_init():
                self.logger.debug("[SOUND_PROCESSOR] Mixer not initialized — cannot stop sound.")
                return
            self.logger.debug(f"[SOUND_PROCESSOR] Stopping audio playback with fade-out...")
            self.stop_playback_event.set()
            # Make sure we wait a bit longer than the fade
            pygame.time.wait(self.FADEOUT_DURATION_MS + 100)
        if call_back and callable(call_back):
            self.logger.debug("[SOUND_PROCESSOR] : stop_sound.call_back() ...")
            call_back()

    def shutdown(self):
        """Cleanly shut down the mixer."""
        with self.mixer_lock:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
                self.logger.debug("[SOUND_PROCESSOR] Pygame mixer shut down.")

    @staticmethod
    def wait(duration_ms):
        pygame.time.wait(duration_ms)


class SpeechProcessorTTSX3(SpeechProcessor):
    # DEFAULT_VOICE = "Microsoft Ryan (Natural) - English (United Kingdom)"  # Or your preferred default
    DEFAULT_VOICE = "Microsoft Sonia (Natural) - English (United Kingdom)"  # Or your preferred default
    DEFAULT_RATE = 180
    SUPPORTED_LANGUAGES = {
        "en": "Microsoft Sonia (Natural) - English (United Kingdom)",
        # "en": "Microsoft Ryan (Natural) - English (United Kingdom)",
        "es": "Microsoft Dalia (Natural) - Spanish (Mexico)",
        "fr": "Microsoft Denise (Natural) - French (France)",
        "de": "Microsoft Katja (Natural) - German (Germany)",
        "zh": "Microsoft Xiaoxiao (Natural) - Chinese (Simplified, China)"
    }

    def __init__(self):
        super().__init__()
        self.engine = pyttsx3.init('sapi5')
        self.current_voice = self.DEFAULT_VOICE
        self.rate = self.DEFAULT_RATE
        self.mixer_lock = threading.Lock()
        self.stop_playback_event = threading.Event()
        self.audio_output_file = "output_audio.mp3"
        self.voices = self._list_voices()
        self.set_voice(self.DEFAULT_VOICE)
        self.set_rate(self.DEFAULT_RATE)

    def read_text(self, text, call_before=None, call_back=None, lang="en", tld="co.uk", do_not_interrupt=False):
        try:
            if call_before and callable(call_before):
                # self.logger.debug("read_text() Calling before function...")
                self.logger.debug("[SOUND_PROCESSOR] : read_text.call_before() ...")
                call_before()
            with self.mixer_lock:
                if lang in self.SUPPORTED_LANGUAGES.keys():
                    if self.SUPPORTED_LANGUAGES[lang] != self.current_voice:
                        self.set_voice(self.SUPPORTED_LANGUAGES[lang])
                with self.file_lock:
                    self.engine.save_to_file(text, self.audio_output_file)
                    self.engine.runAndWait()
                if lang in self.SUPPORTED_LANGUAGES.keys():
                    if self.SUPPORTED_LANGUAGES[lang] != self.current_voice:
                        self.set_voice(self.current_voice)
            with self.file_lock:
                with open(self.audio_output_file, "rb") as file:
                    audio_data = file.read()
                self.play_stream(audio_data, call_back=call_back, do_not_interrupt=do_not_interrupt)
        except Exception as e:
            self.logger.debug(f"[SOUND_PROCESSOR] Error generating text-to-speech audio with pyttsx3: {e}, {traceback.format_exc()}")

    def set_voice(self, voice_name):
        if voice_name in self.voices:
            self.engine.setProperty('voice', self.voices[voice_name]['id'])
            self.current_voice = voice_name
            self.logger.debug(f"[SOUND_PROCESSOR] Setting AI voice ...")
            # self.logger.debug(f"[SOUND_PROCESSOR] Voice set to: {voice_name}")
        else:
            self.logger.debug(f"[SOUND_PROCESSOR] Voice '{voice_name}' not found.")

    def set_rate(self, rate):
        self.engine.setProperty('rate', rate)
        self.rate = rate
        self.logger.debug(f"[SOUND_PROCESSOR] Rate set to: {rate}")

    def _list_voices(self):
        """Fetch available voices."""
        voices = self.engine.getProperty('voices')
        voice_dict = {}
        for voice in voices:
            voice_dict[voice.name] = {"id": voice.id, "gender": voice.gender, "languages": voice.languages}
        return voice_dict

    def shutdown(self):
        """Cleanly shut down pyttsx3 engine."""
        with self.mixer_lock:
            self.engine.stop()
            self.logger.debug("[SOUND_PROCESSOR] pyttsx3 engine shut down.")


# Usage
if __name__ == "__main__":
    processor = SpeechProcessorTTSX3()
    from varstore import HELLOS


    def test_playback():

        print("Testing playback...")
        for helo in HELLOS:
            processor.read_text(helo)
        # processor.read_text(". ".join(HELLOS))
        # processor.read_text(" ".join(THINKING_SOUNDS))


    # threading.Thread(target=test_playback, daemon=True).start()
    print("Testing playback...")
    for helo in HELLOS:
        processor.read_text(helo, do_not_interrupt=True)

    while True:
        pass
