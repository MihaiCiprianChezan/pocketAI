from dataclasses import dataclass
import logging
from queue import Queue
import sys
import threading
import time
from time import sleep
import traceback

from better_profanity import profanity
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from assistant.assistant2 import Assistant, HistoryManager
from atention import Attention
from energy_ball import EnergyBall
from speech_processor import SpeechProcessorTTSX3
from utils import *
from varstore import *


class VoiceApp(QObject):
    # Define signals to communicate with the EnergyBall
    send_command_to_ball = Signal(str, dict)  # str: command, dict: optional parameters

    NAME = "Opti"
    HELLO = YELLOW
    GENERATING = MAGENTA
    SPEAKING = BLUE
    OPERATING_TEXT = YELLOW
    INITIAL = TRANSPARENT
    PROFANITY = RED
    UNCERTAIN = ORANGE
    PAUSED = LIGHT_GREY
    TRANSLATE = CYAN

    def __init__(self):
        super().__init__()
        self.logger_instance = AppLogger(file_name="VoiceUtilApp.log", overwrite=True, log_level=logging.DEBUG)
        self.logger = self.logger_instance
        # self.logger = self.logger_instance.get_logger()
        self.speech_processor = SpeechProcessorTTSX3()
        self.chat_assistant = Assistant()
        self.history_manager = HistoryManager()
        self.utils = Utils()

        self.recognized_speech_queue = Queue(1000)  # Shared queue for recognized speech
        self.speech_thread = None
        self.speech_thread_running = False

        self.edit_mode_active = False
        self.buffer_text = ""
        self.chat_mode = True
        self.assistant_speaking = False
        self.previous_expression = ""
        self.saluted = False
        self.name_variants = [self.NAME.lower(), self.NAME.title(), self.NAME.upper()]
        self.general_commands = self.init_commands()
        self.write_commands = self.init_write_commands()

    def init_commands(self):
        # a:bool = prompt is Addressing Assistant
        return {
            # EDIT MODE enter - Will run edit commands
            ("editing", "mode"): lambda a: self.activate_write_mode(a),
            ("enter", "editing"): lambda a: self.activate_write_mode(a),
            ("editing", "please"): lambda a: self.activate_write_mode(a),

            # CHAT MODE enter - will prompt the AI
            ("start", "chat"): lambda a: self.start_chat(a),
            ("resume", "chat"): lambda a: self.start_chat(a),
            ("chat", "again"): lambda a: self.start_chat(a),
            # CHAT MODE pause - will not prompt the AI
            ("pause", "chat"): lambda a: self.pause_chat(a),
            ("have", "pause"): lambda a: self.pause_chat(a),
            ("stop", "chat"): lambda a: self.pause_chat(a),
            # Interruption commands
            ("wait", "stop"): lambda a: self.interrupt_assistant(a),
            ("wait", "wait"): lambda a: self.interrupt_assistant(a),
            ("stop", "stop"): lambda a: self.interrupt_assistant(a),
            ("ok", "thanks"): lambda a: self.interrupt_assistant(a),
            ("thank", "you"): lambda a: self.interrupt_assistant(a),
            ("hold", "on"): lambda a: self.interrupt_assistant(a),
            ("all", "right"): lambda a: self.interrupt_assistant(a),
            ("okay", "okay"): lambda a: self.interrupt_assistant(a),

            ("read", "this"): lambda a: self.read_selected_text(a),
            ("explain", "this"): lambda a: self.explain_selected_text(a),
            ("summarize", "this"): lambda a: self.summarize_selected_text(a),
            ("summary", "of", "this"): lambda a: self.summarize_selected_text(a),

            ("translate", "to", "english"): lambda a: self.translate_selected_text(a, 'en'),
            ("translate", "into", "english"): lambda a: self.translate_selected_text(a, 'en'),
            ("translate", "to", "french"): lambda a: self.translate_selected_text(a, 'fr'),
            ("translate", "into", "french"): lambda a: self.translate_selected_text(a, 'fr'),
            ("translate", "to", "german"): lambda a: self.translate_selected_text(a, 'de'),
            ("translate", "into", "german"): lambda a: self.translate_selected_text(a, 'de'),
            ("translate", "to", "spanish"): lambda a: self.translate_selected_text(a, 'es'),
            ("translate", "into", "spanish"): lambda a: self.translate_selected_text(a, 'es'),
            ("translate", "to", "chinese"): lambda a: self.translate_selected_text(a, 'zh'),
            ("translate", "into", "chinese"): lambda a: self.translate_selected_text(a, 'zh'),
            ("exit", "app"): lambda a: self.exit_app(a),
        }

    def init_write_commands(self):
        # TODO: Maybe translate inline text, correct text, chat inline in the text.
        return {
            ("new", "line"): lambda a: self.edit_action("enter", "New line."),
            ("copy", "text"): lambda a: self.edit_action("ctrl+c", "Copied."),
            ("paste", "text"): lambda a: self.edit_action("ctrl+v", "Pasted."),
            ("cut", "text"): lambda a: self.edit_action("ctrl+x", "Cut."),
            ("delete", "text"): lambda a: self.edit_action("delete", "Deleted."),
            ("select", "all"): lambda a: self.edit_action("ctrl+a", "Selected all."),
            ("select", "word"): lambda a: self.edit_action("ctrl+shift+left", "Selected word."),
            ("select", "line"): lambda a: self.edit_action("ctrl+shift+down", "Selected line."),
            ("select", "paragraph"): lambda a: self.edit_action("ctrl+shift+up", "Selected paragraph."),
            ("move", "up"): lambda a: self.edit_action("up", "Up."),
            ("move", "down"): lambda a: self.edit_action("down", "Down."),
            ("move", "left"): lambda a: self.edit_action("left", "Left."),
            ("move", "right"): lambda a: self.edit_action("right", "Right."),
            ("undo", "please"): lambda a: self.edit_action("ctrl+z", "Undo."),
            ("redo", "please"): lambda a: self.edit_action("ctrl+shift+z", "Redo."),

            # EDIT MODE exit - Will NOT run edit commands
            ("exit", "editing"): lambda a: self.deactivate_write_mode(a),
            ("close", "editing"): lambda a: self.deactivate_write_mode(a),
            ("stop", "editing"): lambda a: self.deactivate_write_mode(a),
            ("stop", "stop"): lambda a: self.deactivate_write_mode(a),

            ("translate", "to", "english"): lambda a: self.edit_translate_text(a, 'en'),
            ("translate", "into", "english"): lambda a: self.edit_translate_text(a, 'en'),
            ("translate", "to", "french"): lambda a: self.edit_translate_text(a, 'fr'),
            ("translate", "into", "french"): lambda a: self.edit_translate_text(a, 'fr'),
            ("translate", "to", "german"): lambda a: self.edit_translate_text(a, 'de'),
            ("translate", "into", "german"): lambda a: self.edit_translate_text(a, 'de'),
            ("translate", "to", "spanish"): lambda a: self.edit_translate_text(a, 'es'),
            ("translate", "into", "spanish"): lambda a: self.edit_translate_text(a, 'es'),
            ("translate", "to", "chinese"): lambda a: self.edit_translate_text(a, 'zh'),
            ("translate", "into", "chinese"): lambda a: self.edit_translate_text(a, 'zh'),
        }

    def edit_action(self, keys, confirmation=""):
        keyboard.send(keys)
        if confirmation:
            self.agent_speak(confirmation)

    # def paste_copied(self, confirmation):
    #     self.utils.paste_at_cursor()
    #     if confirmation:
    #         self.agent_speak(confirmation)

    def is_for_assistant(self, spoken, clean):
        return any(variant in clean or variant in spoken.lower() for variant in self.name_variants)

    def clean_name(self, spoken, clean):
        clean = clean.replace(self.NAME.lower(), "").strip()
        for variant in self.name_variants:
            spoken = spoken.replace(variant, "").strip()
        for_bot_attention = True
        return spoken, clean

    def is_command(self, tokens: list[str], searched_tokens: tuple[str, ...]) -> bool:
        """
        Check if any consecutive subsequence in `tokens` matches `searched_tokens`.

        Args:
            tokens (list[str]): A list of words/tokens to search through.
            searched_tokens (tuple[str, ...]): A tuple of strings representing the command.

        Returns:
            bool: True if `searched_tokens` exists in `tokens` as a consecutive subsequence, False otherwise.
        """
        token_count = len(searched_tokens)
        if token_count == 0 or token_count > len(tokens):
            return False

        # Use zip to slide over the tokens
        for idx, token_window in enumerate(zip(*[tokens[i:] for i in range(token_count)])):
            if tuple(token_window) == searched_tokens:
                self.logger.debug(f"[APP] Match found at index {idx}: {token_window} == {searched_tokens}")
                return True

        return False

    def process_command(self, spoken, clean, tokens, addresses_assistant):
        """Process voice commands and handle them using helper methods.
        ****************************************************************
        """
        is_command = False
        # Write-mode -- to issue 'general commands' exiting from write mode is required
        if self.edit_mode_active:
            if self.handle_commands(tokens, self.write_commands):
                is_command = True
            # Write spoken text (DICTATE)
            if not is_command:
                self.logger.debug(f"[APP] Writing dictated text ...")
                spoken = self.speech_processor.restore_punctuation(f"{spoken}")
                self.utils.write_text(f"{spoken} ")

        # General commands (must NOT be in write mode) -- to issue 'general commands'
        elif self.handle_commands(tokens, self.general_commands, addresses_assistant or self.chat_mode):
            is_command = True
        return is_command

    def process_chat(self, spoken, addresses_assistant):
        # -------------------------------------------------------------------------------------
        # >>> chat with the assistant the spoken text ==========================================
        # We prevent going into self conversation loops, to stop the assistant use 'hold on' type commands
        if not self.assistant_speaking:
            self.logger.debug(f"[APP] AI Assistant <is NOT> currently speaking ...")
        else:
            self.logger.debug(f"[APP] AI Assistant <IS> currently speaking ...")
            self.interrupt_assistant(True)

        prompt = self.speech_processor.restore_punctuation(spoken)
        self.logger.debug(f"[APP] Prompting AI Assistant with: {prompt} ...")
        response = self.get_ai_response(prompt)
        if response:
            self.assistant_speaking = True
            self.agent_speak(response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def handle_commands(self, tokens, commands, for_assistant=None):

        self.logger.debug(f"[APP] for_assistant: {for_assistant}, self.chat_mode: {self.chat_mode}, ...")

        """Handle commands and return True if a command was handled."""
        for command_tokens, action in commands.items():
            # self.debug_match_commands(tokens, command_tokens)
            if self.is_command(tokens, command_tokens):
                action(for_assistant)
                return True  # Command was handled
        return False  # No command was recognized

    def debug_match_commands(self, tokens, command_tokens):
        self.logger.debug(
            f"[APP] [DEBUG] is_command: {self.is_command(tokens, command_tokens)}, "
            f"command_tokens: {command_tokens}, tokens: {tokens}, "
            f"command_tokens in tokens: {command_tokens in tokens}, "
            f"command_tokens == tokens: {command_tokens == tokens}"
        )

    def activate_write_mode(self, is_for_assistant, lang="en"):
        """Activate write mode."""
        with Attention(is_for_assistant, "activate_write_mode()", self.logger):
            if not self.edit_mode_active:
                self.agent_speak("Edit mode activated!", after_color=self.OPERATING_TEXT)
                self.edit_mode_active = True
            else:
                self.agent_speak("Edit mode is already active!")

    def deactivate_write_mode(self, is_for_assistant):
        """Deactivate write mode."""
        if self.edit_mode_active:
            self.agent_speak("Closing Edit mode...", after_color=self.INITIAL)
            self.edit_mode_active = False
        else:
            self.agent_speak("Edit mode is not active!")

    def start_chat(self, is_for_assistant):
        """Start chat mode."""
        if not self.chat_mode:
            self.agent_speak("Chat resumed!", after_color=self.INITIAL)
            self.chat_mode = True
            self.history_manager.clean()
            self.logger.debug("[APP] Chat mode resumed.")
            return
        self.agent_speak("Chat mode is already active!")

    def pause_chat(self, is_for_assistant):
        """Pause chat mode."""
        with Attention(is_for_assistant, "pause_chat()", self.logger):
            if self.chat_mode:
                self.agent_speak("Chat paused!", after_color=self.PAUSED)
                self.chat_mode = False
                self.logger.debug("[APP] Chat mode paused.")
                return
            self.agent_speak("Already Paused! If you want to chat, speak the resume command!")

    def interrupt_assistant(self, is_for_assistant=False):
        """Handle 'stop' or 'hold on' commands."""
        if self.assistant_speaking or is_for_assistant:
            try:
                self.speech_processor.stop_sound()
                self.ball_reset_colorized()
                self.speech_processor.wait(100)
                self.logger.debug("[APP] Assistant IS speaking, <STOPPING> ...")
                self.logger.debug("[APP] Saying a short confirmation ...")
                self.read_unique(SHORT_CONFIRMS, speaking_color=self.SPEAKING, after_color=self.INITIAL)
                self.assistant_speaking = False
                self.logger.debug("[APP] Speaking interrupted ...")
                self.speech_processor.wait(1000)
            except Exception as e:
                self.logger.error(f"[APP]  Error stopping: {e}, {traceback.format_exc()}")
            return
        self.logger.debug("[APP] Assistant IS NOT currently speaking, <NO_REASON_TO_STOP> ...")

    def read_selected_text(self, is_for_assistant):
        """Read the selected text."""
        with Attention(is_for_assistant, "read_selected_text()", self.logger):
            self.read_unique(ACKNOWLEDGEMENTS, do_not_interrupt=True)
            keyboard.send("ctrl+c")  # Copy selected text
            sleep(0.3)
            text = pyperclip.paste()
            self.ball_change_color(self.OPERATING_TEXT)
            self.agent_speak(text, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def explain_selected_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        with Attention(is_for_assistant, "explain_selected_text()", self.logger):
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            copied_text = copied_text.replace('\n', '').strip()
            prompt = f"Explain this: `{copied_text}`"
            ai_response = self.get_ai_response(prompt, lang=lang)
            self.agent_speak(ai_response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def summarize_selected_text(self, is_for_assistant, lang="en"):
        """Summarize the selected text."""
        with Attention(is_for_assistant, "explain_selected_text()", self.logger):
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            copied_text = copied_text.replace('\n', '').strip()
            prompt = f"Summarize `{copied_text}`. Output only the plain text, in a single line, without any formatting or additional comments."
            ai_response = self.get_ai_response(prompt, lang=lang)
            self.agent_speak(ai_response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def translate_selected_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        with Attention(is_for_assistant, "translate_selected_text()", self.logger):
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            language = LANGUAGES[lang]
            self.logger.debug(f"[APP] [Translating to {lang}:language] of: {copied_text}")
            prompt = f"Translate `{copied_text}` to {language} language. Output only the plain translation, in a single line, without any formatting or additional comments."
            self.logger.debug(f"[APP] [Translation prompt]: {prompt}")
            ai_response = self.get_ai_response(prompt, lang=lang, context_free=True)
            self.agent_speak(ai_response, speaking_color=self.TRANSLATE, after_color=self.INITIAL, lang=lang)

    def edit_translate_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        with Attention(is_for_assistant, "translate_selected_text()", self.logger):
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            language = LANGUAGES[lang]
            self.logger.debug(f"[APP] [Editor, translating to {lang}:language] of: {copied_text}")
            prompt = f"Translate `{copied_text}` to {language} language. Output only the plain translation, in a single line, without any formatting or additional comments."
            self.logger.debug(f"[APP] [Editor, translation prompt]: {prompt}")
            ai_response = self.get_ai_response(prompt, lang=lang)
            keyboard.send("end")
            keyboard.send("space")
            sleep(0.3)
            self.utils.write_text(f"{ai_response} ", delay=0.03)
            self.agent_speak(f"Here's the text translated to {language}...", speaking_color=self.SPEAKING, after_color=self.OPERATING_TEXT, lang="en")

    def exit_app(self, is_for_assistant):
        """Exit the application gracefully."""
        with Attention(is_for_assistant, "exit_app()", self.logger):
            try:
                self.logger.debug("[APP] Exiting the program...")
                # Stop the speech recognition thread
                self.stop_speech_recognition_thread()
                # Finish any other cleanup tasks
                self.read_unique(GOODBYES, do_not_interrupt=True)
                self.speech_processor.wait(3000)
                self.speech_processor.stop_sound(call_back=None)
                self.emit_exit()
                self.logger.debug("[APP] Program exited cleanly.")
                sys.exit(0)
            except Exception as e:
                self.logger.debug(f"[APP] Error exiting: {e}, {e.__traceback__}")

    def read_unique(self, expression_list, speaking_color=None, after_color=None, do_not_interrupt=False):
        self.previous_expression = self.utils.get_unique_choice(expression_list, self.previous_expression)
        self.agent_speak(self.previous_expression, speaking_color=speaking_color, after_color=after_color, do_not_interrupt=do_not_interrupt)

    def get_ai_response(self, spoken_prompt, colour=GENERATING, entertain=True, lang="en", context_free=False):
        """
        Will change the color of the ball, start pulsating on the rhythm of the text token responses from the model.
        Will also progressively print in the console the full response from the AI model.
        Once the response from the AI model is received, it will be spoken out loud.
        """
        self.logger.debug(f"(*) [USER] Says to AI Assistant: \"{spoken_prompt}\"")
        self.ball_change_color(colour)
        self.read_unique(SHORT_CONFIRMS)

        last_update_time, random_interval = self._initialize_time() if entertain else (None, None)
        response = ""
        response_iter = self.chat_assistant.get_response(spoken_prompt, context_free=context_free)

        # AI will generate the response ...
        self.logger_instance.pause()
        print(f"[APP][AI_ASSISTANT][REAL_TIME_RESPONSE] (*) >>>")

        # partial prompt results
        for partial_response in response_iter:
            # Handle cases where partial_response is not a string
            if isinstance(partial_response, list):
                # Convert list to a space-joined string
                partial_response = " ".join(str(item) for item in partial_response)
            elif not isinstance(partial_response, str):
                # Fallback: Convert non-string types to a string
                partial_response = str(partial_response)

            # Process the string `partial_response`
            response, diff = self._update_response(response, partial_response.replace("\n", ""), zoom=True)
            diff = diff.replace("\n", "")
            print(f"{diff}", end="", flush=True)

            # Entertain periodically
            if entertain and time.time() - last_update_time >= random_interval:
                last_update_time, random_interval = self.entertain(last_update_time, random_interval)

        print(f"\n", flush=True)

        ai_response = self.utils.clean_response(response)
        self.logger_instance.resume()
        self.logger.debug(f"[APP][AI_ASSISTANT] Response: {response}")
        self.logger.debug(f"[APP][AI_ASSISTANT] Response cleaned: {ai_response}")
        return ai_response

    def entertain(self, min_interval=2, max_interval=5):
        self.read_unique(WAITING_SOUNDS)
        last_update_time, random_interval = self._initialize_time(min_interval, max_interval)
        return last_update_time, random_interval

    @staticmethod
    def _initialize_time(min_interval=2, max_interval=5):
        """Initialize and return the starting time and a random interval."""
        last_update_time = time.time()
        random_interval = random.uniform(min_interval, max_interval)
        return last_update_time, random_interval

    def _update_response(self, response, partial_response, zoom=True):
        """Update the response with the latest partial data and print any new tokens."""
        diff = ""
        if response != partial_response:
            diff = partial_response.replace(response, "")
            response += diff
            if diff and zoom:
                self.ball_zoom_effect()
        return response, diff

    # --------------------- Energy ball related ----------------------------

    def agent_speak(self, speech_script, speaking_color=None, after_color=None, lang="en", do_not_interrupt=False):
        self.logger.debug(f"(*) [APP][AI_ASSISTANT] Says: \"{speech_script}\" [Energy Color: {speaking_color}]")
        call_before = self.speak_call_before
        call_back = self.speak_callback
        if speaking_color:
            call_before = lambda: self.speak_call_before(color=speaking_color)
        if after_color:
            call_back = lambda: self.speak_callback(color=after_color)
        self.speech_processor.read_text(
            speech_script,
            call_before=call_before,
            call_back=call_back,
            lang=lang,
            do_not_interrupt=do_not_interrupt)

    def speak_call_before(self, color=None):
        if color:
            self.ball_change_color(color)
        self.ball_start_pulsating()
        self.assistant_speaking = True

    def speak_callback(self, color=None):
        if color:
            self.ball_change_color(color)
        self.ball_stop_pulsating()
        self.assistant_speaking = False

    # def make_silence(self):
    #     self.speech_processor.stop_sound(call_back=self.ball_stop_pulsating)

    def ball_zoom_effect(self, zoom=FAST_ZOOM):
        self.send_command_to_ball.emit("zoom_effect", zoom)

    def ball_change_color(self, color):
        self.send_command_to_ball.emit("change_color", color)

    def ball_start_pulsating(self):
        # self.send_command_to_ball.emit("change_color", color)
        self.send_command_to_ball.emit("start_pulsating", {})

    def ball_reset_colorized(self):
        self.send_command_to_ball.emit("reset_colorized", {})

    def ball_stop_pulsating(self):
        """Stop pulsating the ball."""
        self.send_command_to_ball.emit("stop_pulsating", {})

    def emit_exit(self):
        """Stop pulsating the ball."""
        self.send_command_to_ball.emit("exit", {})

    def start_speech_recognition_thread(self):
        """Start the speech recognition thread."""
        if self.speech_thread_running:
            self.logger.debug("[APP] Speech recognition thread is already running.")
            return

        self.speech_thread_running = True
        self.speech_thread = threading.Thread(target=self.speech_recognition_task, daemon=True)
        self.speech_thread.start()
        self.logger.debug("[APP] Speech recognition thread has started.")

    def speech_recognition_task(self):
        """The speech recognition loop that runs in its own thread."""
        self.logger.debug("[APP] Speech recognition task running.")
        while self.speech_thread_running:
            try:
                spoken = self.speech_processor.recognize_speech()
                if spoken:
                    self.logger.debug(f"[APP][USER] Recognized: \"{spoken}\"")
                    self.recognized_speech_queue.put(spoken)
                self.speech_processor.wait(100)
            except Exception as e:
                self.logger.error(f"[APP] Error in speech recognition thread: {e}, {traceback.format_exc()}")
                self.speech_processor.wait(1000)
        self.logger.debug("[APP] Speech recognition task has stopped.")

    def stop_speech_recognition_thread(self):
        """Stop the speech recognition thread."""
        if not self.speech_thread_running:
            self.logger.debug("[APP] Speech recognition thread is not running.")
            return
        self.speech_thread_running = False
        if self.speech_thread:
            self.speech_thread.join()
            self.logger.debug("[APP] Speech recognition thread has fully stopped.")

    def run(self):
        """Main application loop to process recognized speech from the queue."""
        self.read_unique(HELLOS, speaking_color=self.HELLO, after_color=self.INITIAL, do_not_interrupt=True)
        self.speech_processor.wait(1000)
        self.start_speech_recognition_thread()
        try:
            while True:
                if not self.recognized_speech_queue.empty():

                    spoken = self.recognized_speech_queue.get()
                    self.logger.debug(f"[APP] Processing speech from queue: \"{spoken}\"")
                    clean = self.utils.clean_text(spoken)
                    tokens = clean.split()
                    prompt_is_valid = self.utils.is_prompt_valid(spoken, clean)

                    # Check if prompt is a profanity / vulgarity
                    if profanity.contains_profanity(spoken):
                        self.logger.debug(f"[APP] Profanity detected: {profanity.censor(spoken)}")
                        self.read_unique(POLITE_RESPONSES, speaking_color=self.PROFANITY, after_color=self.INITIAL)
                        continue

                    # Check if Assistant name was mentioned
                    addresses_assistant = self.is_for_assistant(spoken, clean)
                    if addresses_assistant:
                        self.logger.debug(f"[APP] Prompt mentions AI Assistant name ...")
                        spoken, clean = self.clean_name(spoken, clean)

                    # Check for commands and execute commands
                    is_command = self.process_command(spoken, clean, tokens, addresses_assistant)
                    if is_command:
                        self.logger.debug(f"[APP] COMMAND detected in prompt {tokens} ...")
                        continue

                    # Process chat is no command was given, not in edit mode and prompt is valid
                    if prompt_is_valid and not self.edit_mode_active and (self.chat_mode or addresses_assistant):
                        self.process_chat(spoken, addresses_assistant)
                        continue

                    # Unclear prompts neither commands neither chats, respond only to longer valid prompts when not in editing mode
                    self.logger.debug(f"[APP] Prompt {tokens} is NOT VALID FOR CHAT and is NEITHER A COMMAND: {spoken}")
                    if not self.assistant_speaking and len(tokens) > 1:
                        self.read_unique(UNCLEAR_PROMPT_RESPONSES, speaking_color=self.UNCERTAIN, after_color=self.INITIAL)

                # Add a slight delay to this loop to avoid high CPU usage
                self.speech_processor.wait(100)

        except KeyboardInterrupt:
            # Trigger `exit_app` when exiting via keyboard
            self.exit_app(is_for_assistant=True)


@dataclass
class SpeachStream:
    spoken: str  # Source of the speech stream, e.g., "microphone" or "file"
    language: str  # Language of the stream, e.g., "en-US"
    transcription: str = ""  # Storage for the processed speech transcription
    timestamp: float = 0.0  # Timestamp for the stream data
    is_active: bool = False  # Status to indicate if the stream is currently active


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
    app = QApplication(sys.argv)
    energy_ball = EnergyBall()
    energy_ball.show()
    voice_util_app = VoiceApp()
    voice_thread = VoiceUtilThread(voice_util_app)
    voice_util_app.send_command_to_ball.connect(energy_ball.receive_command)
    voice_thread.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Voice App
    # app = VoiceUtilApp()
    # app.run()

    # Full threaded app mode with light ball
    main()
