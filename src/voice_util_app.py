import random
import re
import sys
import threading
from time import sleep
import keyboard
import pyperclip
from chat_bot import ChatBot
from speech_processor import SpeechProcessor
from floating_energy_ball import FloatingEnergyBall
from PySide6.QtCore import QObject, Signal
# Precompile regex patterns
NON_ALPHANUMERIC_REGEX = re.compile(r'[^a-zA-Z0-9\s]')
MULTIPLE_SPACES_REGEX = re.compile(r'\s+')
from PySide6.QtCore import QCoreApplication



class VoiceUtilApp(QObject):
    # Define signals to communicate with the FloatingEnergyBall
    send_command_to_ball = Signal(str, dict)  # str: command, dict: optional parameters

    RED = {"color": (100, 0, 0)}
    BLUE = {"color": (0, 20, 120)}
    GREEN = {"color": (0, 100, 0)}
    YELLOW = {"color": (80, 80, 0)}
    MAGENTA = {"color": (80, 0, 80)}
    GREY = {"color": (50, 50, 50)}

    SLOW_GROW = {"factor": 1.1, "duration": 1500}

    def __init__(self):
        super().__init__()
        self.speech_processor = SpeechProcessor()
        self.in_write_mode = False
        self.buffer_text = ""
        self.chatting = False
        self.chatbot = ChatBot()
        self.history = []

    @staticmethod
    def clean_text(text):
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

    def is_for_bot(self, spoken, clean, is_for_bot=False):
        if "opti" in clean or "opti" in spoken.lower():
            clean = clean.replace("opti", "").strip()
            clean = self.clean_text(clean)
            spoken = spoken.replace("Opti", "").strip()
            spoken = spoken.replace("opti", "").strip()
            is_for_bot = True
        return spoken, clean, is_for_bot

    @staticmethod
    def get_command(tokens, token_count=2):
        """Get the command from the tokens."""
        command = " ".join(tokens[:token_count])
        return command

    @staticmethod
    def is_command(tokens, searched_tokens):
        """
        Get the command from the tokens if the searched tokens exist
        consecutively in the token list.
        """
        token_count = len(searched_tokens)
        # Convert each sliding window of 'token_count' size in tokens to compare
        for i in range(len(tokens) - token_count + 1):
            # Extract current slice to compare
            if tokens[i:i + token_count] == searched_tokens:
                print(f"Comparing  {tokens[i:i + token_count]} with {searched_tokens} ... ")
                return True
        return False

    def process_command(self, spoken, clean):
        """Process voice commands and issue them to the FloatingEnergyBall."""
        spoken, clean, is_for_bot = self.is_for_bot(spoken, clean)
        tokens = clean.split()

        print(f"Command tokens: {tokens}")

        if self.speech_processor.is_playing():
            self.speech_processor.stop_sound()

        # self.send_command_to_ball.emit("reset_colorized", {})
        self.send_command_to_ball.emit("zoom_effect", {})

        if self.get_command(tokens) == "paste paste":
            self.paste_at_cursor()

        if self.get_command(tokens) == "new line":
            if self.in_write_mode:
                self.write_text("\n")

        # elif self.get_command(tokens) == "clear clear":
        #     self.buffer_text = ""

        elif self.get_command(tokens) == "note note":
            self.speech_processor.read_text("Edit Mode, your words will be written!")
            self.emit_change_color(self.YELLOW)
            self.in_write_mode = True

        elif self.get_command(tokens) == "done note":
            self.speech_processor.read_text("Closing Edit mode...")
            self.in_write_mode = False
            self.emit_reset_colorized()

        elif self.get_command(tokens) == "select all":
            keyboard.send("ctrl+a")

        elif self.get_command(tokens) == "copy copy":
            keyboard.send("ctrl+c")

        elif self.get_command(tokens) == "cut cut":
            keyboard.send("ctrl+x")

        elif self.get_command(tokens) == "delete delete":
            keyboard.send("delete")

        elif self.get_command(tokens) == "start chat":
            self.chatting = True
            self.history = []
            print("Chat mode activated.")

        elif self.get_command(tokens) == "pause chat":
            self.chatting = False
            print("Chat mode paused.")

        elif self.get_command(tokens) == "stop chat":
            self.history = []
            self.chatting = False
            print("Chat mode deactivated.")

        elif self.get_command(tokens) == "stop stop" or self.get_command(tokens) == "hold on":
            self.speech_processor.stop_sound()
            short_confirm = ['aha..', 'ok..', 'yup..', 'yes...', 'al right...']
            random_response = random.choice(short_confirm)
            self.speech_processor.read_text(random_response)

        # --- READ selected texts
        elif self.get_command(tokens) == "read this":
            if is_for_bot:
                keyboard.send("ctrl+c")  # Copy selected text
                sleep(0.5)
                text = pyperclip.paste()
                self.emit_change_color(self.GREEN)
                self.speech_processor.read_text(text, self.emit_start_pulsating, self.emit_stop_pulsating)
            else:
                print("(i) Not in chat mode. Please activate chat mode or call bot by name.")

        # --- EXPLAIN selected text
        elif self.get_command(tokens) == "explain this":
            if is_for_bot:
                keyboard.send("ctrl+c")
                sleep(0.5)
                copied_text = pyperclip.paste()
                prompt = f"Explain this to me: {copied_text}"
                self.color_chatbot(prompt)
            else:
                print("(i) Not in chat mode. Please activate chat mode or call bot by name.")

        elif self.is_command(tokens, ["exit", "exit"]):
            try:
                self.speech_processor.read_text("Good bye!")
                print("Exiting the program...")
                self.speech_processor.stop_sound()
                self.emit_exit()
                sys.exit(0)
            except Exception as e:
                print(f"Error exiting: {e}, {e.__traceback__}")

        else:
            if self.in_write_mode:
                self.write_text(spoken)
            # Normal chat if bot name was mentioned or if active chat is on
            if self.chatting or is_for_bot:
                self.color_chatbot(spoken)

    def color_chatbot(self, spoken_prompt):
        self.emit_zoom_effect(self.SLOW_GROW)
        self.emit_change_color(self.MAGENTA)

        print(f"[USER prompt]: {spoken_prompt}")
        response = self.chatbot.chat(self.history, spoken_prompt)
        print(f"[AI response]: {response}")

        self.speech_processor.read_text(response, self.emit_start_pulsating, self.emit_stop_pulsating)

    def emit_zoom_effect(self, zoom):
        self.send_command_to_ball.emit("zoom_effect", zoom)

    def emit_change_color(self, color):
        self.send_command_to_ball.emit("change_color", color)

    def emit_start_pulsating(self):
        self.emit_change_color(self.BLUE)
        self.send_command_to_ball.emit("start_pulsating", {})

    def emit_reset_colorized(self):
        self.send_command_to_ball.emit("reset_colorized", {})

    def emit_stop_pulsating(self):
        """Stop pulsating the ball."""
        self.send_command_to_ball.emit("stop_pulsating", {})
        self.emit_reset_colorized()

    def emit_exit(self):
        """Stop pulsating the ball."""
        self.send_command_to_ball.emit("exit", {})

    def run(self):
        """Run the main loop to listen for and process voice commands."""

        while True:
            spoken = self.speech_processor.recognize_speech()
            print(f"Recognized speech: {spoken}")

            if spoken:
                cleaned = self.clean_text(spoken)
                self.process_command(spoken, cleaned)


if __name__ == "__main__":
    app = VoiceUtilApp()
    app.run()
