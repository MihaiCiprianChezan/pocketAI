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
from assistant.assistant import Assistant
from atention import Attention
from energy_ball import EnergyBall
from entertain import Entertain
from speech_processor import SpeechProcessorTTSX3
from utils import *
from varstore import *
from translation import TranslationService
from prompt import Prompt
from commands import *


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

    # TODO: - Add error check when translation text is various. Make translation a tool.
    #       - Verify check intent for direct commands ...

    def __init__(self):
        super().__init__()
        self.logger_instance = AppLogger(file_name="AI_Assistant_Application.log", overwrite=True, log_level=logging.DEBUG)
        self.logger = self.logger_instance
        self.speech_processor = SpeechProcessorTTSX3()
        self.assistant = Assistant()
        self.assistant.initialize()
        self.utils = Utils()
        self.translations = TranslationService()

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
        # TODO: lists for similar commands...
        return {
            ENTER_EDIT_MODE: lambda a: self.activate_write_mode(a),
            ENTER_CHAT_MODE: lambda a: self.start_chat(a),
            PAUSE_CHAT_MODE: lambda a: self.pause_chat(a),
            INTERRUPT_ASSISTANT: lambda a: self.interrupt_assistant(a),
            READ_THIS: lambda a: self.read_selected_text(a),
            EXPLAIN_THIS: lambda a: self.explain_selected_text(a),
            SUMMARIZE_THIS: lambda a: self.summarize_selected_text(a),
            ("describe", "images"): lambda a: self.describe_images(a),
            ("describe", "video"): lambda a: self.describe_video(a),
            TRANSLATE_TO_ENGLISH: lambda a: self.translate_selected_text(a, 'en'),
            TRANSLATE_TO_FRENCH: lambda a: self.translate_selected_text(a, 'fr'),
            TRANSLATE_TO_GERMAN: lambda a: self.translate_selected_text(a, 'de'),
            TRANSLATE_TO_SPANISH: lambda a: self.translate_selected_text(a, 'es'),
            TRANSLATE_TO_CHINESE: lambda a: self.translate_selected_text(a, 'zh'),
            EXIT: lambda a: self.exit_app(a),
        }

    def init_write_commands(self):
        return {
            NEW_LINE: lambda a: self.edit_action("enter", "New line."),
            COPY_TEXT: lambda a: self.edit_action("ctrl+c", "Copied."),
            PASTE_TEXT: lambda a: self.edit_action("ctrl+v", "Pasted."),
            CUT_TEXT: lambda a: self.edit_action("ctrl+x", "Cut."),
            DELETE_TEXT: lambda a: self.edit_action("delete", "Deleted."),
            SELECT_ALL: lambda a: self.edit_action("ctrl+a", "Selected all."),
            SELECT_WORD: lambda a: self.edit_action("ctrl+shift+left", "Selected word."),
            SELECT_LINE: lambda a: self.edit_action("ctrl+shift+down", "Selected line."),
            SELECT_PARAGRAPH: lambda a: self.edit_action("ctrl+shift+up", "Selected paragraph."),
            MOVE_UP: lambda a: self.edit_action("up", "Up."),
            MOVE_DOWN: lambda a: self.edit_action("down", "Down."),
            MOVE_LEFT: lambda a: self.edit_action("left", "Left."),
            MOVE_RIGHT: lambda a: self.edit_action("right", "Right."),
            UNDO: lambda a: self.edit_action("ctrl+z", "Undo."),
            REDO: lambda a: self.edit_action("ctrl+shift+z", "Redo."),
            EXIT_EDIT_MODE: lambda a: self.deactivate_write_mode(a),
            TRANSLATE_TO_ENGLISH: lambda a: self.edit_translate_text(a, 'en'),
            TRANSLATE_TO_FRENCH: lambda a: self.edit_translate_text(a, 'fr'),
            TRANSLATE_TO_GERMAN: lambda a: self.edit_translate_text(a, 'de'),
            TRANSLATE_TO_SPANISH: lambda a: self.edit_translate_text(a, 'es'),
            TRANSLATE_TO_CHINESE: lambda a: self.edit_translate_text(a, 'zh'),
        }

    def edit_action(self, keys, confirmation=""):
        keyboard.send(keys)
        if confirmation:
            self.agent_speak(confirmation)

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
        # Command is length is greater than the tokens length
        if token_count == 0 or token_count > len(tokens):
            return False
        # Use zip to slide over the tokens
        for idx, token_window in enumerate(zip(*[tokens[i:] for i in range(token_count)])):
            if tuple(token_window) == searched_tokens:
                self.logger.debug(f"[APP] Match found at index {idx}: {token_window} == {searched_tokens}")
                return True
        # No command match
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

        response = self.get_ai_response(Prompt(prompt))
        if response:
            self.assistant_speaking = True
            self.agent_speak(response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def handle_commands(self, tokens, commands, for_assistant=None):
        """
        Handle commands and return True if a command was handled.

        Args:
            tokens (list[str]): A list of tokens extracted from the spoken text.
            commands (dict): A dictionary where keys are command patterns
                             (tuple or tuple of tuples) and values are functions (actions).
            for_assistant (optional): Context for the assistant, passed to the action.

        Returns:
            bool: True if a command was executed, False otherwise.
        """
        self.logger.debug(f"[APP] for_assistant: {for_assistant}, self.chat_mode: {self.chat_mode}")
        for command_tokens, action in commands.items():
            # Normalize the command_tokens to always be a tuple of command combos
            command_combos = (
                command_tokens if isinstance(command_tokens[0], tuple) else (command_tokens,)
            )

            # Check each command combination against the input tokens
            for command_combo in command_combos:
                if self.is_command(tokens, command_combo):  # Match individual tuple
                    # Execute the action function associated with the command combo
                    action(for_assistant)
                    return True
        # No command was recognized
        return False

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
            self.logger.debug("[APP] Chat mode resumed.")
            return
        self.agent_speak("Chat mode is already active!")

    def pause_chat(self, is_for_assistant):
        """Pause chat mode."""
        with Attention(is_for_assistant, "pause_chat()", self.logger):
            if self.chat_mode:
                self.agent_speak("Chat paused!", after_color=self.PAUSED)
                self.assistant.history_mng.clean()
                self.chat_mode = False
                self.logger.debug("[APP] Chat mode paused.")
                return
            self.agent_speak("Already Paused! If you want to chat, speak the resume command!")

    def interrupt_assistant(self, is_for_assistant=False):
        """Handle 'stop' or 'hold on' commands."""
        if self.assistant_speaking or is_for_assistant:
            try:
                # self.speech_processor.stop_sound()
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

    def describe_images(self, is_for_assistant):
        with Attention(is_for_assistant, "process_images()", self.logger):
            prompt = Prompt("Please describe the images.")
            prompt.images = ["./samples/image1.jpg", "./samples/image2.jpg"]
            prompt.pixel_values, prompt.num_patches_list = self.assistant.chat_mng.get_image_pixel_values(prompt.images)
            ai_response = self.get_ai_response(prompt)
            self.agent_speak(ai_response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def describe_video(self, is_for_assistant):
        with Attention(is_for_assistant, "process_images()", self.logger):
            prompt = Prompt("Please describe video, what is the red panda doing?")
            prompt.video = "./samples/red-panda.mp4"
            prompt.video_prefix, prompt.pixel_values, prompt.num_patches_list = self.assistant.chat_mng.get_video_pixel_values(prompt.video)
            ai_response = self.get_ai_response(prompt)
            self.agent_speak(ai_response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def explain_selected_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        with Attention(is_for_assistant, "explain_selected_text()", self.logger):
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            copied_text = copied_text.replace('\n', '').strip()
            ai_response = self.get_ai_response(Prompt(f"Please explain this:\n`{copied_text}`"))
            self.agent_speak(ai_response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def summarize_selected_text(self, is_for_assistant):
        """Summarize the selected text."""
        with Attention(is_for_assistant, "explain_selected_text()", self.logger):
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            copied_text = copied_text.replace('\n', '').strip()
            prompt = f"Please summarize this:\n `{copied_text}`"
            ai_response = self.get_ai_response(Prompt(prompt))
            self.agent_speak(ai_response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def translate_selected_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        with Attention(is_for_assistant, "translate_selected_text()", self.logger):
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            language = LANGUAGES[lang]
            self.logger.info(f"[APP] [Translating to {lang}:{language} language] of: {copied_text}")
            ai_response = self.translations.auto_translate(copied_text, lang)
            self.agent_speak(ai_response, speaking_color=self.TRANSLATE, after_color=self.INITIAL, lang=lang)

    def edit_translate_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        with Attention(is_for_assistant, "translate_selected_text()", self.logger):
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            language = LANGUAGES[lang]
            self.logger.debug(f"[APP] [Editor, translating to {lang}:{language}:language] of: {copied_text}")
            ai_response = self.translations.auto_translate(copied_text, lang)
            keyboard.send("end")
            keyboard.send("enter")
            sleep(0.3)
            self.utils.write_text(f"{ai_response} ", delay=0.005)
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
                self.logger.debug(f"[APP] Error exiting: {e}, {traceback.format_exc()}")

    def read_unique(self, expression_list, speaking_color=None, after_color=None, do_not_interrupt=False):
        self.previous_expression = self.utils.get_unique_choice(expression_list, self.previous_expression)
        self.agent_speak(self.previous_expression, speaking_color=speaking_color, after_color=after_color, do_not_interrupt=do_not_interrupt)

    def get_ai_response(self, user_prompt: Prompt, colour=GENERATING, entertain=True, context_free=False):
        """
        Fetch the AI Response while managing the ball and optionally entertaining the user periodically.

        Args:
            user_prompt: The user's input prompt.
            colour: The colour of the ball during processing.
            entertain: Whether to entertain the user during generation.
            context_free: Disable context-based responses.
        """
        self.logger.info(f"(*) [USER] Says to AI Assistant: \"{user_prompt.message}\"")
        self.ball_change_color(colour)
        self.read_unique(SHORT_CONFIRMS)

        # Create the entertainer
        entertainer = Entertain(action=lambda: self.read_unique(WAITING_SOUNDS), should_entertain=entertain)

        # Thread to run the entertainer
        def entertain_in_background():
            while entertainer.should_entertain:
                entertainer.check_and_entertain()
                self.speech_processor.wait(100)

        # Start entertain thread if entertain=True
        entertain_thread = None
        if entertain:
            entertain_thread = threading.Thread(target=entertain_in_background, daemon=True)
            entertain_thread.start()

        # AI response fetching loop
        response_iter = self.assistant.get_response(user_prompt, context_free=context_free)
        response = ""

        self.logger.pause()
        print("[APP][AI_ASSISTANT][REAL_TIME_RESPONSE] (*) >>>")

        for partial_response in response_iter:
            response += partial_response
            print(self.utils.clean_new_lines(partial_response), end="", flush=True)

        print("\n", flush=True)
        self.logger.resume()

        # Stop the entertainer
        if entertain:
            entertainer.should_entertain = False
            if entertain_thread:
                entertain_thread.join()

        self.logger.debug(f"[APP][AI_ASSISTANT] Raw response: {response}")
        clean_ai_response = self.utils.deep_text_clean(response)
        self.logger.debug(f"[APP][AI_ASSISTANT] Cleaned response: {clean_ai_response}")
        self.assistant.history_mng.add(user_prompt.message, clean_ai_response)
        return clean_ai_response

    # --------------------- Energy ball related ----------------------------

    def agent_speak(self, speech_script, speaking_color=None, after_color=None, lang="en", do_not_interrupt=False):
        self.logger.info(f"(*) [APP][AI_ASSISTANT] Says: \"{speech_script}\" [Energy Color: {speaking_color}]")
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

                    # No command was given and agent is speaking, so we skip (unless agent is not speaking or a command)
                    if self.assistant_speaking:
                        self.logger.debug(f"[APP] Assistant is speaking and no command was issued, <SKIPPING>: `{spoken}` ...")
                        continue

                    # No command was given, not in edit mode and prompt is valid
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
