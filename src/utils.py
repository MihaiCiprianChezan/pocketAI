import random
from pathlib import Path
import keyboard
import pyperclip

from app_logger import AppLogger
from varstore import GLITCHES, MULTIPLE_SPACES_REGEX
import re


class Utils:

    def __init__(self):
        self.logger = AppLogger()

    @staticmethod
    def is_recog_glitch(spoken, glitches=GLITCHES):
        """Determine if the recognized speech is a glitch."""
        for g in glitches:
            if spoken.find(g) != -1:
                return True
        return False

    def is_prompt_valid(self, spoken, clean, min_tokens=3, min_chars=10, min_unique_tokens=3, vulgarities_list=None):
        """Check if a spoken prompt is valid based on multiple conditions."""
        result = True
        if not clean.strip():
            result = False
        token_count = len(clean.split())
        if token_count < min_tokens:
            result = False
        char_count = len(clean)
        if char_count < min_chars:
            result = False
        if vulgarities_list:
            clean_lower = clean.lower()
            if any(word in clean_lower for word in vulgarities_list):
                result = False
        if spoken.isalpha() and len(set(spoken.lower())) <= 2:
            result = False
        tokens = clean.split()
        if len(set(tokens)) < min_unique_tokens:
            result = False
        most_common_token = max(set(tokens), key=tokens.count)
        if tokens.count(most_common_token) / token_count > 0.8:
            result = False
        if token_count == 1 and char_count > 20:
            result = False
        if clean.isdigit() or all(not c.isalnum() for c in clean):
            result = False
        self.logger.debug(f"[UTILS][is_prompt_valid()] (i) Prompt does not make sense for a chat ...")
        return result

    @staticmethod
    def get_unique_choice(options, last_choice=None):
        """Get a randomly selected option, avoiding the same choice as the last one."""
        new_choice = random.choice(options)
        while new_choice == last_choice:
            new_choice = random.choice(options)
        return new_choice

    def clean_response(self, llm_response, ensure_punctuation=True):
        """Clean an AI model response string for live chat."""
        if not isinstance(llm_response, str):
            return ""

        # Detect Chinese characters
        chinese_characters_pattern = re.compile(r"[\u4e00-\u9fff]+")
        if chinese_characters_pattern.search(llm_response):
            return llm_response

        # Regex for cleaning unwanted characters
        combined_pattern = re.compile(
            r"\*\*|"  # Double asterisks
            "##|###|"  # Hashes
            "[\U0001F600-\U0001FBFF]+"  # Emojis and special symbols
            "|[\U0000200B-\U0000200D]",  # Zero-width characters
            flags=re.UNICODE,
        )
        cleaned_response = combined_pattern.sub("", llm_response)

        cleaned_text = cleaned_response.replace('\n', ' ')
        cut_part = None
        if ensure_punctuation and "." in cleaned_text:
            if not cleaned_text.endswith((".", "!", "?", ";")):
                last_punctuation = max(cleaned_text.rfind("."), cleaned_text.rfind("!"), cleaned_text.rfind("?"), cleaned_text.rfind(";"))
                if last_punctuation != -1:
                    cut_part = cleaned_text[last_punctuation + 1:].strip()
                    cleaned_text = cleaned_text[:last_punctuation + 1].strip()
        cleaned_text = cleaned_text.strip()
        if cut_part:
            self.logger.debug(f"[UTILS] <!> Cleaned out incomplete part: {cut_part}")
        return cleaned_text

    @staticmethod
    def clean_text(text):
        """Clean the input text by removing non-alphanumeric characters."""
        single_spaced_text = MULTIPLE_SPACES_REGEX.sub(' ', text)
        return single_spaced_text.strip().lower()

    @staticmethod
    def paste_at_cursor():
        """Paste copied text at the cursor."""
        text = pyperclip.paste()
        keyboard.send(text)

    @staticmethod
    def write_text(text, delay=0.03):
        """Write the text dynamically with a slight delay."""
        keyboard.write(text, delay=delay)

    def ensure_path_exists(self, path):
        """
        Ensure the specified path exists:
        - If a file path, ensure the folder exists and create the file.
        - If a directory path, ensure it exists.
        """
        try:
            path_obj = Path(path)

            if path_obj.suffix:  # Check if it's a file
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                if not path_obj.exists():
                    path_obj.touch()
                    self.logger.debug(f"[UTILS] File created: {path_obj.resolve()}")
            else:  # It's a folder
                path_obj.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"[UTILS] Directory ensured: {path_obj.resolve()}")

        except Exception as e:
            self.logger.debug(f"[UTILS] Error ensuring path {path}: {e}")
