import random
import re
import sys
import threading
from time import sleep
import keyboard
import pyperclip
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from chat_bot import ChatBot
from energy_ball import EnergyBall
from speech_processor import SpeechProcessor
from varstore import NON_ALPHANUMERIC_REGEX, MULTIPLE_SPACES_REGEX, RED, ORANGE, BROWN, PINK, PURPLE, TURQUOISE, DARK_GREY, LIGHT_GREY, BLUE, GREEN, YELLOW, MAGENTA, GREY, SLOW_GROW, FAST_ZOOM, ZOOM_IN, SHORT_CONFIRMS, GLITCHES, GOODBYES, HELLOS
from utils import is_recog_glitch

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
        self.speech_processor.read_text(random.choice(HELLOS), call_before=None, call_back=None)

    def clean_text(self, text):
        """Clean the input text by removing non-alphanumeric characters."""
        alphanumeric_text = NON_ALPHANUMERIC_REGEX.sub('', text)
        single_spaced_text = MULTIPLE_SPACES_REGEX.sub(' ', alphanumeric_text)
        return single_spaced_text.strip().lower()

    def paste_at_cursor(self):
        """Paste copied text at the cursor."""
        text = pyperclip.paste()
        keyboard.write(text)

    def write_text(self, text):
        """Write the text dynamically with a slight delay."""
        keyboard.write(text, delay=0.005)

    def is_for_bot(self, spoken, clean, for_bot_attention=False):
        if self.NAME.lower() in clean or self.NAME.lower() in spoken.lower():
            clean = self.clean_text(clean.replace("opti", "").strip())
            for variant in [self.NAME.lower(), self.NAME.title(), self.NAME.upper()]:
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
        spoken, clean, is_for_bot = self.is_for_bot(spoken, clean)
        tokens = clean.split()
        print(f"Command tokens: {tokens}")

        # Delegate write-mode or general commands
        if self.in_write_mode:
            if not self.handle_write_mode_commands(tokens, spoken):
                # If no specific write mode command, fallback to writing spoken text
                self.write_text(spoken)
        elif not self.handle_general_commands(tokens, is_for_bot):
            # If no general commands are recognized, handle fallback for general mode
            if self.chatting or is_for_bot:
                self.color_speak(spoken)

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
            ("start", "chat"): self.start_chat,
            ("pause", "chat"): self.pause_chat,
            ("stop", "chat"): self.stop_chat,
            ("wait", "stop"): self.handle_stop_command,
            ("wait", "wait"): self.handle_stop_command,
            ("stop", "stop"): self.handle_stop_command,
            ("hold", "on"): self.handle_stop_command,
            ("read", "this", "please"): lambda: self.read_selected_text(is_for_bot),
            ("explain", "this"): lambda: self.explain_selected_text(is_for_bot),
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
        self.speech_processor.read_text("Edit Mode, write with words!", call_before=None, call_back=None)
        self.ball_change_color(self.OPERATING_TEXT)
        self.in_write_mode = True

    def deactivate_write_mode(self):
        """Deactivate write mode."""
        self.speech_processor.read_text("Closing Edit mode...", call_before=None, call_back=None)
        self.in_write_mode = False
        self.ball_reset_colorized()

    def start_chat(self):
        """Start chat mode."""
        self.chatting = True
        self.history = []
        print("Chat mode activated.")

    def pause_chat(self):
        """Pause chat mode."""
        self.chatting = False
        print("Chat mode paused.")

    def stop_chat(self):
        """Stop chat mode."""
        self.history = []
        self.chatting = False
        print("Chat mode deactivated.")

    def handle_stop_command(self):
        """Handle 'stop' or 'hold on' commands."""
        if self.chatbot_speaking:
            print("Chatbot is speaking, stopping chatbot ...")
            self.chatbot_speaking = False
            self.speech_processor.stop_sound()
            self.speech_processor.wait(650)
            speak = random.choice(SHORT_CONFIRMS)
            self.speech_processor.read_text(
                speak,
                call_before=None,
                call_back=self.ball_stop_pulsating
                )
            return
        print("Chatbot is not currently speaking, no reason to stop...")

    def read_selected_text(self, is_for_bot):
        """Read the selected text."""
        if is_for_bot:
            keyboard.send("ctrl+c")  # Copy selected text
            sleep(0.1)
            text = pyperclip.paste()
            self.ball_change_color(self.OPERATING_TEXT)
            self.speech_processor.read_text(text, call_before=None, call_back=None)
        else:
            print("(i) Not in chat mode. Please activate chat mode or call Assistant by name.")

    def explain_selected_text(self, is_for_bot):
        """Explain the selected text."""
        if is_for_bot:
            keyboard.send("ctrl+c")
            sleep(0.5)
            copied_text = pyperclip.paste()
            prompt = f"Explain this to me: {copied_text}"
            self.color_speak(prompt)
        else:
            print("(i) Not in chat mode. Please activate chat mode or call bot by name.")

    def exit_app(self):
        """Exit the application gracefully."""
        try:
            self.speech_processor.read_text(random.choice(GOODBYES), call_before=None, call_back=None)
            print("Exiting the program...")
            self.speech_processor.stop_sound(call_back=None)
            self.emit_exit()
            sys.exit(0)
        except Exception as e:
            print(f"Error exiting: {e}, {e.__traceback__}")

    def color_speak(self, spoken_prompt, colour=SPEAKING):
        """
        Will change the color of the ball start pulsating on the rhythm of the text token responses from the model.
        Will aloso progressively print in the console the full response from the AI model.
        Once the response from the AI model is received it will be spoken out loud.
        """
        self.ball_change_color(self.GENERATING)
        print(f"[USER prompt]: {spoken_prompt}")
        print(f"[AI response]:", end="")
        response = ""
        response_iter = self.chatbot.chat(self.history, spoken_prompt, return_iter=True)
        for partial_response in response_iter:
            if response != partial_response:
                diff = partial_response.replace(response, "")
                response += diff
                if diff:
                    self.ball_zoom_effect()
                    print(f"{diff}", end="")
        print(f"\n")
        self.ball_change_color(colour)
        # print(f"\n[Full AI response]: {response}")
        self.speak(response)

    # --------------------- Energy ball related ----------------------------

    def speak(self, speech_script):
        self.chatbot_speaking = True
        self.ball_start_pulsating()
        self.speech_processor.read_text(speech_script, call_back=self.speak_callback)

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