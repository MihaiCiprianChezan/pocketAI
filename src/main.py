import sys
import threading
import time
import traceback
from time import sleep

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from better_profanity import profanity

from assistant import ChatAssistant
from energy_ball import EnergyBall
from speech_processor import SpeechProcessorTTSX3
from utils import *
from varstore import BLUE, YELLOW, MAGENTA, FAST_ZOOM, SHORT_CONFIRMS, \
    GOODBYES, HELLOS, POLITE_RESPONSES, ACKNOWLEDGEMENTS, LANGUAGES, WAITING_SOUNDS, \
    UNCLEAR_PROMPT_RESPONSES, PINK


class VoiceApp(QObject):
    # Define signals to communicate with the EnergyBall
    send_command_to_ball = Signal(str, dict)  # str: command, dict: optional parameters

    NAME = "Opti"
    HELLO = PINK
    GENERATING = MAGENTA
    SPEAKING = BLUE
    OPERATING_TEXT = YELLOW
    INITIAL = {"color": "initial"}

    def __init__(self):
        super().__init__()
        self.speech_processor = SpeechProcessorTTSX3()
        self.in_write_mode = False
        self.buffer_text = ""
        self.chat_mode = True
        self.chat_assistant = ChatAssistant()
        self.assistant_speaking = False
        self.history = []
        self.previous_expression = ""
        self.saluted = False
        self.name_variants = [self.NAME.lower(), self.NAME.title(), self.NAME.upper()]
        self.general_commands = self.init_commands()
        self.write_commands = self.init_write_commands()

    def init_commands(self):
        return {
            # a = for_assistant
            ("enter", "edit", "mode"): lambda a: self.activate_write_mode(),
            ("exit", "edit", "mode"): lambda a: self.deactivate_write_mode(),
            ("start", "chat", "mode"): lambda a: self.start_chat(),
            ("pause", "chat"): lambda a: self.pause_chat(),
            ("stop", "chat", "mode"): lambda a: self.stop_chat(),
            # Interruption commands
            ("wait", "stop"): lambda a: self.handle_stop_command(),
            ("wait", "wait"): lambda a: self.handle_stop_command(),
            ("stop", "stop"): lambda a: self.handle_stop_command(),
            ("thanks", "thanks"): lambda a: self.handle_stop_command(),
            ("thank", "you"): lambda a: self.handle_stop_command(),
            ("hold", "on"): lambda a: self.handle_stop_command(),
            ("all", "right"): lambda a: self.handle_stop_command(),
            ("okay", "okay"): lambda a: self.handle_stop_command(),
            ("read", "this"): lambda a: self.read_selected_text(a),
            ("explain", "this"): lambda a: self.explain_selected_text(a),
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
            ("exit", "app"): lambda a: self.exit_app,
        }

    @staticmethod
    def init_write_commands():
        return {
            ("new", "line"): lambda: keyboard.write("\n"),
            ("copy", "copy"): lambda: keyboard.send("ctrl+c"),
            ("paste", "paste"): paste_at_cursor,
            ("cut", "cut"): lambda: keyboard.send("ctrl+x"),
            ("delete", "delete"): lambda: keyboard.send("delete"),
            ("select", "all"): lambda: keyboard.send("ctrl+a"),
            ("select", "word"): lambda: keyboard.send("ctrl+shift+left"),
            ("select", "line"): lambda: keyboard.send("ctrl+shift+down"),
            ("select", "paragraph"): lambda: keyboard.send("ctrl+shift+up"),
            ("up", "up"): lambda: keyboard.send("up"),
            ("down", "down"): lambda: keyboard.send("up"),
            ("left", "left"): lambda: keyboard.send("up"),
            ("right", "right"): lambda: keyboard.send("up"),
        }

    def is_for_assistant(self, spoken, clean):
        return any(variant in clean or variant in spoken.lower() for variant in self.name_variants)

    def clean_name(self, spoken, clean):
        clean = clean.replace(self.NAME.lower(), "").strip()
        for variant in self.name_variants:
            spoken = spoken.replace(variant, "").strip()
        for_bot_attention = True
        return spoken, clean

    @staticmethod
    def is_command(tokens, searched_tokens):
        """
        Get the command from the tokens if the searched tokens exist
        consecutively in the token list.
        """
        token_count = len(searched_tokens)
        # Convert each sliding window of 'token_count' size in tokens to compare
        for i in range(len(tokens) - token_count + 1):
            # Convert the current slice to a tuple for consistent comparison
            if tuple(tokens[i:i + token_count]) == searched_tokens:
                print(f"[APP] Comparing {tuple(tokens[i:i + token_count])} with {searched_tokens} ... ")
                return True
        return False

    def process_command(self, spoken, clean):
        """Process voice commands and handle them using helper methods."""
        # check if prompt makes sense

        # check if prompt is a profanity/vulgarity
        prompt_is_profanity = profanity.contains_profanity(spoken)
        if prompt_is_profanity:
            print(f"[APP] <!> Profanity detected: {profanity.censor(spoken)}")
            self.previous_expression = get_unique_choice(POLITE_RESPONSES, self.previous_expression)
            self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
            return

        addresses_assistant = self.is_for_assistant(spoken, clean) or self.chat_mode
        if addresses_assistant:
            print(f"[APP] (i) Prompt is addressed to the Assistant (mentions Assistant name) ...")
            clean, spoken = self.clean_name(spoken, clean)
        tokens = clean.split()

        if self.assistant_speaking:
            print(f"[APP] <!> Assistant is currently speaking ...")

        # >>> write-mode -- to issue 'general commands' exiting from write mode is required ===
        if self.in_write_mode:
            if self.handle_commands(tokens, self.write_commands):
                return
            # write spoken text
            write_text(self.speech_processor.restore_punctuation(spoken))

        # >>> general commands -- to issue 'general commands' must not be in write mode =======
        elif self.handle_commands(tokens, self.general_commands, addresses_assistant):
            return

        # >>> chat with the assistant the spoken text ==========================================
        elif self.chat_mode or addresses_assistant:

            # check if prompt is valid
            prompt_is_valid = is_prompt_valid(spoken, clean)
            if not prompt_is_valid:
                print(f"[APP] <!> Prompt is not valid: {spoken}")
                self.previous_expression = get_unique_choice(UNCLEAR_PROMPT_RESPONSES, self.previous_expression)
                self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
                return
            print(f"[APP] (i) Prompt is VALID, [{len(tokens)} tokens]: {tokens}")

            # We prevent going into self conversation loops, to stop the assistant use 'hold on' type commands
            if not self.assistant_speaking:
                response = self.get_ai_response(self.speech_processor.restore_punctuation(spoken))
                if response:
                    self.assistant_speaking = True
                    self.agent_speak(response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def handle_commands(self, tokens, commands, for_assistant=None):
        """Handle commands and return True if a command was handled."""
        for command_tokens, action in commands.items():
            # self.debug_match_commands(tokens, command_tokens)
            if self.is_command(tokens, command_tokens):
                if for_assistant is not None:
                    action(for_assistant)
                else:
                    action()
                return True  # Command was handled
        return False  # No command was recognized

    def debug_match_commands(self, tokens, command_tokens):
        print(
            f"[APP] [DEBUG] is_command result: {self.is_command(tokens, command_tokens)}, "
            f"command_tokens: {command_tokens}, "
            f"tokens: {tokens}, "
            f"command_tokens in tokens: {command_tokens in tokens}, "
            f"command_tokens == tokens: {command_tokens == tokens}"
        )

    def activate_write_mode(self):
        """Activate write mode."""
        if not self.in_write_mode:
            self.speech_processor.read_text("Edit Mode was activated!", call_before=None, call_back=None)
            self.ball_change_color(self.OPERATING_TEXT)
            self.in_write_mode = True
        else:
            self.speech_processor.read_text("Edit mode is already active!", call_before=None, call_back=None)

    def deactivate_write_mode(self):
        """Deactivate write mode."""
        if self.in_write_mode:
            self.speech_processor.read_text("Closing Edit mode...", call_before=None, call_back=None)
            self.in_write_mode = False
        else:
            self.speech_processor.read_text("Edit mode is not active!", call_before=None, call_back=None)
        self.ball_reset_colorized()

    def start_chat(self):
        """Start chat mode."""
        self.speech_processor.read_text("Chat mode Activated!", call_before=None, call_back=None)
        self.chat_mode = True
        self.history = []
        print("[APP] Chat mode activated.")

    def pause_chat(self):
        """Pause chat mode."""
        self.speech_processor.read_text("Chat mode Paused!", call_before=None, call_back=None)
        self.chat_mode = False
        print("[APP] Chat mode paused.")

    def stop_chat(self):
        """Stop chat mode."""
        self.speech_processor.read_text("Closing Chat mode...", call_before=None, call_back=None)
        self.history = []
        self.chat_mode = False
        print("[APP] Chat mode deactivated.")

    def handle_stop_command(self):
        """Handle 'stop' or 'hold on' commands."""
        if self.assistant_speaking:
            try:
                print("[APP] <!> Assistant is speaking! stopping ...")
                print("[APP] (i) Saying a short confirmation ...")
                self.assistant_speaking = False
                self.speech_processor.stop_sound()
                self.read_unique(SHORT_CONFIRMS)
                print("[APP] (i) Reading text interrupted ...")
            except Exception as e:
                print(f"[APP] Error stopping: {e}, {traceback.format_exc()}")
            return
        print("[APP] (i) Assistant not currently speaking, no reason to stop...")

    def read_selected_text(self, is_for_assistant):
        """Read the selected text."""
        if is_for_assistant:
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")  # Copy selected text
            sleep(0.3)
            text = pyperclip.paste()
            self.ball_change_color(self.OPERATING_TEXT)
            self.agent_speak(text, speaking_color=self.OPERATING_TEXT, after_color=self.INITIAL)
        else:
            print("[APP] (i) Not in chat mode. Please activate chat mode or call Assistant by name.")

    def explain_selected_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        if is_for_assistant:
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            copied_text = copied_text.replace('\n', '').strip()
            prompt = f"[APP] Please explain this: `{copied_text}`"
            ai_response = self.get_ai_response(prompt, lang=lang)
            self.agent_speak(ai_response, speaking_color=self.OPERATING_TEXT, after_color=self.INITIAL)
        else:
            print("[APP] (i) Not in chat mode. Please activate chat mode or call bot by name.")

    def translate_selected_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        if is_for_assistant:
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            language = LANGUAGES[lang]
            print(f"[APP] [Translation to {lang}:language] of: {copied_text}")
            prompt = f"[APP] Detect the language of the text and translate it to {language} language, reply only the translation. Text to translate into {language} language: `{copied_text}`\n"
            print(f"[APP] [Translation prompt]: {prompt}")
            ai_response = self.get_ai_response(prompt, lang=lang)
            self.agent_speak(ai_response, speaking_color=self.OPERATING_TEXT, after_color=self.INITIAL, lang=lang)

        else:
            print("[APP] (i) Not in chat mode. Please activate chat mode or call bot by name.")

    def exit_app(self):
        """Exit the application gracefully."""
        try:
            print("[APP] Exiting the program...")
            self.read_unique(GOODBYES, wait_for_speech=True)
            self.speech_processor.stop_sound(call_back=None)
            self.emit_exit()
            sys.exit(0)
        except Exception as e:
            print(f"[APP] Error exiting: {e}, {e.__traceback__}")

    def read_unique(self, expression_list, speaking_color=None, after_color=None, do_not_interrupt=False):
        self.previous_expression = get_unique_choice(expression_list, self.previous_expression)
        self.agent_speak(self.previous_expression, speaking_color=speaking_color, after_color=after_color, do_not_interrupt=do_not_interrupt)

    def get_ai_response(self, spoken_prompt, colour=GENERATING, entertain=True, lang="en"):
        """
        Will change the color of the ball, start pulsating on the rhythm of the text token responses from the model.
        Will also progressively print in the console the full response from the AI model.
        Once the response from the AI model is received, it will be spoken out loud.
        """
        print(f"[USER prompt]: {spoken_prompt}")
        self.ball_change_color(colour)
        self.read_unique(SHORT_CONFIRMS)
        last_update_time, random_interval = self._initialize_time() if entertain else (None, None)
        response = ""
        response_iter = self.chat_assistant.get_response(self.history, spoken_prompt, lang=lang)
        print(f"\n[APP] [DEBUG] [AI *REAL TIME* response]: ", flush=True)
        # partial prompt results
        for partial_response in response_iter:
            response, diff = self._update_response(response, partial_response.replace("\n", ""), zoom=True)
            print(f"{diff}", end="", flush=True)
            # entertain periodically
            if entertain and time.time() - last_update_time >= random_interval:
                last_update_time, random_interval = self.entertain(last_update_time, random_interval)
        print(f"\n", flush=True)
        self.ball_reset_colorized()
        ai_response = clean_response(response)
        print(f"[APP] [AI *CLEANED* response]: {ai_response}")
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
        print("[APP] Starting agent_speak")
        print(f"[APP] Script: {speech_script}, Color: {speaking_color}")
        self.speech_processor.stop_sound()
        self.speech_processor.read_text(
            speech_script,
            call_before=lambda: self.speak_call_before(color=speaking_color),
            call_back=lambda: self.speak_callback(color=after_color),
            lang=lang,
            do_not_interrupt=do_not_interrupt)

    def speak_call_before(self, color=None):
        if color:
            self.ball_change_color(color)
        self.ball_start_pulsating()
        self.assistant_speaking = True

    def speak_callback(self, color=None):
        if color:
            if color == self.INITIAL:
                self.ball_reset_colorized()
            else:
                self.ball_change_color(color)
        self.ball_stop_pulsating()
        self.assistant_speaking = False

    # def make_silence(self):
    #     self.speech_processor.stop_sound(call_back=self.ball_stop_pulsating)

    def ball_zoom_effect(self, zoom=FAST_ZOOM):
        self.send_command_to_ball.emit("zoom_effect", zoom)

    def ball_change_color(self, color):
        self.send_command_to_ball.emit("change_color", color)
        sleep(0.3)

    def ball_start_pulsating(self):
        # self.send_command_to_ball.emit("change_color", color)
        self.send_command_to_ball.emit("start_pulsating", {})
        sleep(0.3)

    def ball_reset_colorized(self):
        self.send_command_to_ball.emit("reset_colorized", {})

    def ball_stop_pulsating(self):
        """Stop pulsating the ball."""
        self.send_command_to_ball.emit("stop_pulsating", {})
        sleep(0.3)

    def emit_exit(self):
        """Stop pulsating the ball."""
        self.send_command_to_ball.emit("exit", {})

    def run(self):
        """Run the main loop to listen for and process voice commands."""
        self.read_unique(HELLOS, speaking_color=self.HELLO, after_color=self.INITIAL, do_not_interrupt=True)
        self.speech_processor.wait(1000)
        while True:
            self.speech_processor.wait(100)
            spoken = self.speech_processor.recognize_speech()
            if is_recog_glitch(spoken):
                print(f"[APP] Glitch in speech recognition: {spoken}")
                spoken = ""
            if spoken:
                print(f"[APP] Recognized speech: {spoken}")
                cleaned = spoken
                self.process_command(spoken, cleaned)
            else:
                print("[APP] No speech recognized ...")


# if __name__ == "__main__":
#     app = VoiceUtilApp()
#     app.run()


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
    energy_ball = EnergyBall("images/opti100.gif")
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
