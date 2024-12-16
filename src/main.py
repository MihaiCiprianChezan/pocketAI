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
    UNCLEAR_PROMPT_RESPONSES


class VoiceApp(QObject):
    # Define signals to communicate with the EnergyBall
    send_command_to_ball = Signal(str, dict)  # str: command, dict: optional parameters

    NAME = "Opti"

    GENERATING = MAGENTA
    SPEAKING = BLUE
    OPERATING_TEXT = YELLOW
    INITIAL = {"color": "initial"}

    def __init__(self):
        super().__init__()
        self.speech_processor = SpeechProcessorTTSX3()
        self.in_write_mode = False
        self.buffer_text = ""
        self.chatting = True
        self.chat_assistant = ChatAssistant()
        self.assistant_speaking = False
        self.history = []
        self.previous_expression = None
        self.saluted = False

    def is_for_assistant(self, spoken, clean, for_bot_attention=False):
        name_variants = [self.NAME.lower(), self.NAME.title(), self.NAME.upper()]
        is_addressed = any(variant in clean or variant in spoken.lower() for variant in name_variants)
        if is_addressed:
            clean = clean.replace(self.NAME.lower(), "").strip()
            for variant in name_variants:
                spoken = spoken.replace(variant, "").strip()
            for_bot_attention = True
        return spoken, clean, for_bot_attention

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
                print(f"Comparing {tuple(tokens[i:i + token_count])} with {searched_tokens} ... ")
                return True
        return False

    def process_command(self, spoken, clean):
        """Process voice commands and handle them using helper methods."""
        # check if prompt makes sense

        # check if prompt is a profanity/vulgarity
        prompt_is_profanity = profanity.contains_profanity(spoken)
        if prompt_is_profanity:
            print(f"<!> Profanity detected: {profanity.censor(spoken)}")
            self.previous_expression = get_unique_choice(POLITE_RESPONSES, self.previous_expression)
            self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
            return

        prompt_is_valid = is_prompt_valid(spoken, clean)
        if not prompt_is_valid:
            print(f"<!> Prompt is not valid: {spoken}")
            self.previous_expression = get_unique_choice(UNCLEAR_PROMPT_RESPONSES, self.previous_expression)
            self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
            return

        spoken, clean, addresses_assistant = self.is_for_assistant(spoken, clean)
        tokens = clean.split()

        print(f"(i) Prompt is VALID, [{len(tokens)} tokens]: {tokens}")

        if self.assistant_speaking:
            print(f"<!> Assistant is currently speaking. Stopping current speech ...")

        if addresses_assistant:
            print(f"(i) Prompt is addressed to the Assistant (mentions Assistant name) ...")

        # >>> write-mode =======================================================================
        if self.in_write_mode:
            if self.handle_write_mode_commands(tokens, spoken):
                return
            # writing spoken text
            write_text(self.speech_processor.restore_punctuation(spoken))
        else:
            # >>> general commands =============================================================
            if self.handle_general_commands(tokens, self.chatting or addresses_assistant):
                return

            # >>> chatting with the assistant ==================================================
            if self.chatting or addresses_assistant:
                response = self.get_ai_response(self.speech_processor.restore_punctuation(spoken))
                if response:
                    self.assistant_speaking = True
                    print(f"[Assistant final response]: {response}")
                    self.agent_speak(response, speaking_color=self.SPEAKING, after_color=self.INITIAL)

    def handle_write_mode_commands(self, tokens, spoken):
        """Handle commands in write mode and return True if a command was handled."""
        commands = {
            ("new", "line"): lambda: keyboard.write("\n"),
            ("paste", "paste"): paste_at_cursor,
            ("select", "all"): lambda: keyboard.send("ctrl+a"),
            ("select", "word"): lambda: keyboard.send("ctrl+shift+left"),
            ("copy", "copy"): lambda: keyboard.send("ctrl+c"),
            ("cut", "cut"): lambda: keyboard.send("ctrl+x"),
            ("delete", "delete"): lambda: keyboard.send("delete"),
        }

        for command_tokens, action in commands.items():
            if self.is_command(tokens, command_tokens):
                action()
                return True  # Command was handled
        return False  # No command matched, fallback to write the spoken text

    def handle_general_commands(self, tokens, is_for_assistant):
        """Handle commands that are not related to write mode and return True if a command was handled."""
        commands = {
            ("enter", "edit", "mode"): self.activate_write_mode,
            ("exit", "edit", "mode"): self.deactivate_write_mode,
            ("start", "chat", "mode"): self.start_chat,
            ("pause", "chat"): self.pause_chat,
            ("stop", "chat", "mode"): self.stop_chat,
            ("wait", "stop"): self.handle_stop_command,
            ("wait", "wait"): self.handle_stop_command,
            ("stop", "stop"): self.handle_stop_command,
            ("hold", "on"): self.handle_stop_command,
            ("read", "this"): lambda: self.read_selected_text(is_for_assistant),
            ("explain", "this"): lambda: self.explain_selected_text(is_for_assistant),
            ("translate", "to", "english"): lambda: self.translate_selected_text(is_for_assistant, 'en'),
            ("translate", "to", "french"): lambda: self.translate_selected_text(is_for_assistant, 'fr'),
            ("translate", "to", "german"): lambda: self.translate_selected_text(is_for_assistant, 'de'),
            ("translate", "to", "spanish"): lambda: self.translate_selected_text(is_for_assistant, 'es'),
            ("translate", "to", "chinese"): lambda: self.translate_selected_text(is_for_assistant, 'zh'),
            ("translate", "into", "english"): lambda: self.translate_selected_text(is_for_assistant, 'en'),
            ("translate", "into", "french"): lambda: self.translate_selected_text(is_for_assistant, 'fr'),
            ("translate", "into", "german"): lambda: self.translate_selected_text(is_for_assistant, 'de'),
            ("translate", "into", "spanish"): lambda: self.translate_selected_text(is_for_assistant, 'es'),
            ("translate", "into", "chinese"): lambda: self.translate_selected_text(is_for_assistant, 'zh'),
            ("exit", "app"): self.exit_app,
        }

        for command_tokens, action in commands.items():
            # print(
            #     f"[DEBUG] is_command result: {self.is_command(tokens, command_tokens)}, "
            #     f"command_tokens: {command_tokens}, "
            #     f"tokens: {tokens}, "
            #     f"command_tokens in tokens: {command_tokens in tokens}, "
            #     f"command_tokens == tokens: {command_tokens == tokens}"
            # )
            if self.is_command(tokens, command_tokens):
                action()
                return True  # Command was handled
        return False  # No command matched

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
        self.chatting = True
        self.history = []
        print("Chat mode activated.")

    def pause_chat(self):
        """Pause chat mode."""
        self.speech_processor.read_text("Chat mode Paused!", call_before=None, call_back=None)
        self.chatting = False
        print("Chat mode paused.")

    def stop_chat(self):
        """Stop chat mode."""
        self.speech_processor.read_text("Closing Chat mode...", call_before=None, call_back=None)
        self.history = []
        self.chatting = False
        print("Chat mode deactivated.")

    def handle_stop_command(self):
        """Handle 'stop' or 'hold on' commands."""
        if self.assistant_speaking:
            try:
                print("<!> Assistant is speaking! stopping ...")
                print("(i) Saying a short confirmation ...")
                self.assistant_speaking = False
                # self.speech_processor.stop_sound()
                self.read_unique(SHORT_CONFIRMS)
                print("(i) Reading text interrupted ...")
            except Exception as e:
                print(f"Error stopping: {e}, {traceback.format_exc()}")
            return
        print("(i) Assistant not currently speaking, no reason to stop...")

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
            print("(i) Not in chat mode. Please activate chat mode or call Assistant by name.")

    def explain_selected_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        if is_for_assistant:
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            copied_text = copied_text.replace('\n', '').strip()
            prompt = f"Please explain this: `{copied_text}`"
            ai_response = self.get_ai_response(prompt, lang=lang)
            self.agent_speak(ai_response, speaking_color=self.OPERATING_TEXT, after_color=self.INITIAL)
        else:
            print("(i) Not in chat mode. Please activate chat mode or call bot by name.")

    def translate_selected_text(self, is_for_assistant, lang="en"):
        """Explain the selected text."""
        if is_for_assistant:
            self.read_unique(ACKNOWLEDGEMENTS)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            language = LANGUAGES[lang]
            print(f" [Translation to {lang}:language] of: {copied_text}")
            prompt = f"Detect the language of the text and translate it to {language} language, reply only the translation. Text to translate into {language} language: `{copied_text}`\n"
            print(f" [Translation prompt]: {prompt}")
            ai_response = self.get_ai_response(prompt, lang=lang)
            self.agent_speak(ai_response, speaking_color=self.OPERATING_TEXT, after_color=self.INITIAL, lang=lang)

        else:
            print("(i) Not in chat mode. Please activate chat mode or call bot by name.")

    def exit_app(self):
        """Exit the application gracefully."""
        try:
            print("Exiting the program...")
            self.read_unique(GOODBYES)
            self.speech_processor.wait(2000)
            self.speech_processor.stop_sound(call_back=None)
            self.emit_exit()
            sys.exit(0)
        except Exception as e:
            print(f"Error exiting: {e}, {e.__traceback__}")

    def read_unique(self, expression_list, speaking_color=None, after_color=None):
        self.previous_expression = get_unique_choice(expression_list, self.previous_expression)
        self.agent_speak(self.previous_expression, speaking_color=speaking_color, after_color=after_color)

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
        print(f"\n[DEBUG] [AI real time response] >", flush=True)
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
        print(f"[AI final cleaned response]: {ai_response}")
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

    def agent_speak(self, speech_script, speaking_color=None, after_color=None, lang="en"):
        print("[DEBUG] Starting agent_speak")
        print(f"[DEBUG] Script: {speech_script}, Color: {speaking_color}")
        self.speech_processor.stop_sound()
        self.speech_processor.read_text(
            speech_script,
            call_before=lambda: self.speak_call_before(color=speaking_color),
            call_back=lambda: self.speak_callback(color=after_color),
            lang=lang)

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
        self.read_unique(HELLOS)
        self.speech_processor.wait(2000)
        while True:
            spoken = self.speech_processor.recognize_speech()
            if is_recog_glitch(spoken):
                print(f"Glitch in speech recognition: {spoken}")
                spoken = ""
            if spoken:
                print(f"Recognized speech: {spoken}")
                cleaned = spoken
                self.process_command(spoken, cleaned)
            else:
                print("No speech recognized ...")


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
