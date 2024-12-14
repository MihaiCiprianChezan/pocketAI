import random
import sys
import threading
import time
from time import sleep
import keyboard
import pyperclip
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from better_profanity import profanity
from chat_bot import ChatBot
from energy_ball import EnergyBall
from speech_processor import SpeechProcessor
from utils import is_recog_glitch, get_unique_choice, is_prompt_valid
from varstore import NON_ALPHANUMERIC_REGEX, MULTIPLE_SPACES_REGEX, BLUE, YELLOW, MAGENTA, FAST_ZOOM, SHORT_CONFIRMS, \
    GOODBYES, HELLOS, THINKING_SOUNDS, POLITE_RESPONSES, ACKNOWLEDGEMENTS, LANGUAGES


class VoiceApp(QObject):
    # Define signals to communicate with the EnergyBall
    send_command_to_ball = Signal(str, dict)  # str: command, dict: optional parameters

    NAME = "Opti"

    GENERATING = MAGENTA
    SPEAKING = BLUE
    OPERATING_TEXT = YELLOW

    def __init__(self):
        super().__init__()
        self.speech_processor = SpeechProcessor()
        self.in_write_mode = False
        self.buffer_text = ""
        self.chatting = True
        self.chatbot = ChatBot()
        self.chatbot_speaking = False
        self.history = []
        self.previous_expression = None
        self.speech_processor.read_text(random.choice(HELLOS), call_before=None, call_back=None)

    def clean_text(self, text):
        """Clean the input text by removing non-alphanumeric characters."""
        alphanumeric_text = NON_ALPHANUMERIC_REGEX.sub('', text)
        single_spaced_text = MULTIPLE_SPACES_REGEX.sub(' ', alphanumeric_text)
        return single_spaced_text.strip().lower()

    def paste_at_cursor(self):
        """Paste copied text at the cursor."""
        text = pyperclip.paste()
        keyboard.write(text, delay=0.05)

    def write_text(self, text):
        """Write the text dynamically with a slight delay."""
        keyboard.write(text, delay=0.05)

    def is_for_bot(self, spoken, clean, for_bot_attention=False):
        name_variants = [self.NAME.lower(), self.NAME.title(), self.NAME.upper()]
        is_addressed = any(variant in clean or variant in spoken.lower() for variant in name_variants)
        if is_addressed:
            clean = self.clean_text(clean.replace(self.NAME.lower(), "").strip())
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
        if profanity.contains_profanity(spoken):
            print(f"Profanity detected: {spoken}")
            self.previous_expression = get_unique_choice(POLITE_RESPONSES, self.previous_expression)
            self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
            return
        prompt_is_valid = is_prompt_valid(spoken, clean)
        spoken, clean, is_for_bot = self.is_for_bot(spoken, clean)
        tokens = clean.split()
        print(f"[Prompt is valid]: {prompt_is_valid}")
        print(f"[Is for BOT]  : {is_for_bot}")
        print(f"[Prompt tokens]  : {tokens}")
        print(f"[AI is speaking] : {self.chatbot_speaking}")
        # Delegate write-mode or general commands
        if self.in_write_mode:
            if not self.handle_write_mode_commands(tokens, spoken):
                # If no specific write mode command, fallback to writing spoken text
                self.write_text(spoken)
        elif not self.handle_general_commands(tokens, self.chatting or is_for_bot):
            # If no general commands are recognized, handle fallback for general mode
            # pass
            if (not self.chatbot_speaking) and (self.chatting or is_for_bot) and prompt_is_valid:
                self.get_ai_response(spoken)

    def handle_write_mode_commands(self, tokens, spoken):
        """Handle commands in write mode and return True if a command was handled."""
        commands = {
            ("new", "line"): lambda: self.write_text("\n"),
            ("paste", "paste"): self.paste_at_cursor,
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

    def handle_general_commands(self, tokens, is_for_bot):
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
            ("read", "this"): lambda: self.read_selected_text(is_for_bot),
            ("explain", "this"): lambda: self.explain_selected_text(is_for_bot),
            ("translate", "to", "english"): lambda: self.translate_selected_text(is_for_bot, 'en'),
            ("translate", "to", "french"): lambda: self.translate_selected_text(is_for_bot, 'fr'),
            ("translate", "to", "german"): lambda: self.translate_selected_text(is_for_bot, 'de'),
            ("translate", "to", "spanish"): lambda: self.translate_selected_text(is_for_bot, 'es'),
            ("translate", "to", "swedish"): lambda: self.translate_selected_text(is_for_bot, 'sv'),
            ("exit", "exit"): self.exit_app,
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
        if self.chatbot_speaking:
            print("Chatbot is speaking, stopping chatbot ...")
            self.chatbot_speaking = False
            self.speech_processor.stop_sound()
            self.speech_processor.wait(self.speech_processor.FADEOUT_DURATION_MS + 50)
            self.previous_expression = get_unique_choice(SHORT_CONFIRMS, self.previous_expression)
            self.speech_processor.read_text(
                self.previous_expression,
                call_before=None,
                call_back=self.ball_stop_pulsating
            )
            return
        print("Chatbot is not currently speaking, no reason to stop...")

    def read_selected_text(self, is_for_bot):
        """Read the selected text."""
        if is_for_bot:
            self.previous_expression = get_unique_choice(ACKNOWLEDGEMENTS, self.previous_expression)
            self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
            keyboard.send("ctrl+c")  # Copy selected text
            sleep(0.3)
            text = pyperclip.paste()
            self.ball_change_color(self.OPERATING_TEXT)
            self.speak(text)
        else:
            print("(i) Not in chat mode. Please activate chat mode or call Assistant by name.")

    def explain_selected_text(self, is_for_bot):
        """Explain the selected text."""
        if is_for_bot:
            self.previous_expression = get_unique_choice(ACKNOWLEDGEMENTS, self.previous_expression)
            self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            copied_text = copied_text.replace('\n', '').strip()
            prompt = f"Please explain this: `{copied_text}`"
            self.get_ai_response(prompt)
        else:
            print("(i) Not in chat mode. Please activate chat mode or call bot by name.")

    def translate_selected_text(self, is_for_bot, lang="en"):
        """Explain the selected text."""
        if is_for_bot:
            self.previous_expression = get_unique_choice(ACKNOWLEDGEMENTS, self.previous_expression)
            self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
            keyboard.send("ctrl+c")
            sleep(0.3)
            copied_text = pyperclip.paste()
            language = LANGUAGES[lang]
            print(f" [Translation to {lang}:language] of: {copied_text}")
            prompt = f"Detect the language of the text and translate it to {language} language, reply only the translation. Text to translate into {language} language: `{copied_text}`\n"
            print(f" [Translation prompt]: {prompt}")
            self.get_ai_response(prompt, lang=lang)
        else:
            print("(i) Not in chat mode. Please activate chat mode or call bot by name.")

    def exit_app(self):
        """Exit the application gracefully."""
        try:
            self.previous_expression = get_unique_choice(GOODBYES, self.previous_expression)
            self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
            print("Exiting the program...")
            self.speech_processor.stop_sound(call_back=None)
            self.emit_exit()
            sys.exit(0)
        except Exception as e:
            print(f"Error exiting: {e}, {e.__traceback__}")

    def get_ai_response(
            self,
            spoken_prompt,
            colour=SPEAKING,
            min_interval=2.0,  # Minimum interval in seconds
            max_interval=5.0,  # Maximum interval in seconds
            entertain_messages=THINKING_SOUNDS,  # List of random messages
            lang="en"
    ):
        """
        Will change the color of the ball, start pulsating on the rhythm of the text token responses from the model.
        Will also progressively print in the console the full response from the AI model.
        Once the response from the AI model is received, it will be spoken out loud.
        """
        self.ball_change_color(self.GENERATING)
        print(f"[USER prompt]: {spoken_prompt}")
        print(f"[AI response]:", end="")
        response = ""
        response_iter = self.chatbot.chat(self.history, spoken_prompt, return_iter=True)
        # Initialize time tracking
        if entertain_messages:
            last_update_time, random_interval = self._initialize_time(min_interval, max_interval)
        for partial_response in response_iter:
            # Update and print partial response
            response = self._update_response(response, partial_response)
            # Entertain the user periodically
            if entertain_messages:
                if self._should_entertain(last_update_time, random_interval):
                    last_update_time, random_interval = self._entertain_user(
                        entertain_messages, min_interval, max_interval
                    )

        print(f"\n")
        self.ball_change_color(colour)
        self.speak(response, lang=lang)

    # ------------------------- Helper Functions -------------------------

    def _initialize_time(self, min_interval, max_interval):
        """Initialize and return the starting time and a random interval."""
        last_update_time = time.time()
        random_interval = random.uniform(min_interval, max_interval)
        return last_update_time, random_interval

    def _update_response(self, response, partial_response):
        """Update the response with the latest partial data and print any new tokens."""
        if response != partial_response:
            diff = partial_response.replace(response, "")
            response += diff
            if diff:
                self.ball_zoom_effect()
                print(f"{diff}", end="")
        return response

    def _should_entertain(self, last_update_time, random_interval):
        """Check if it is time to entertain the user."""
        return time.time() - last_update_time >= random_interval

    def _entertain_user(self, entertain_messages, min_interval, max_interval):
        """
        Entertain the user with a random message and reinitialize time tracking.
        Returns: tuple: Updated last_update_time and random_interval.
        """
        self.previous_expression = get_unique_choice(entertain_messages, self.previous_expression)
        self.speech_processor.read_text(self.previous_expression, call_before=None, call_back=None)
        return self._initialize_time(min_interval, max_interval)

    # --------------------- Energy ball related ----------------------------

    def speak(self, speech_script, lang="en"):
        self.chatbot_speaking = True
        self.ball_start_pulsating()
        self.speech_processor.read_text(speech_script, call_back=self.speak_callback, lang=lang)

    def speak_callback(self):
        self.speech_processor.stop_sound(call_back=None)
        self.ball_stop_pulsating()
        self.chatbot_speaking = False

    def make_silence(self):
        self.speech_processor.stop_sound(call_back=self.ball_stop_pulsating)

    def ball_zoom_effect(self, zoom=FAST_ZOOM):
        self.send_command_to_ball.emit("zoom_effect", zoom)

    def ball_change_color(self, color):
        self.send_command_to_ball.emit("change_color", color)
        sleep(0.3)

    def ball_start_pulsating(self, color=SPEAKING):
        self.ball_change_color(color)
        sleep(0.3)
        self.send_command_to_ball.emit("start_pulsating", {})

    def ball_reset_colorized(self):
        self.send_command_to_ball.emit("reset_colorized", {})

    def ball_stop_pulsating(self):
        """Stop pulsating the ball."""
        self.send_command_to_ball.emit("stop_pulsating", {})
        sleep(0.3)
        self.ball_reset_colorized()

    def emit_exit(self):
        """Stop pulsating the ball."""
        self.send_command_to_ball.emit("exit", {})

    def run(self):
        """Run the main loop to listen for and process voice commands."""
        while True:
            spoken = self.speech_processor.recognize_speech()
            if is_recog_glitch(spoken):
                print(f"Glitch in speech recognition: {spoken}")
                spoken = ""
            if spoken:
                print(f"Recognized speech: {spoken}")
                cleaned = self.clean_text(spoken)
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
