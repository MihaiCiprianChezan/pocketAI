from pathlib import Path
import random
import re
import mistune
from bs4 import BeautifulSoup
import keyboard
import pyperclip

from app_logger import AppLogger
from varstore import EMOJIS_REGEX, GLITCHES, MULTIPLE_SPACES_REGEX, CHINESE_CHARACTERS_REGEX, KOREAN_CHARACTERS_REGEX, JAPANESE_CHARACTERS_REGEX

HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
MODELS_DIR = Path(__file__).parent.parent / "models"
ALL_MINI_LM_L6_V2 = str(MODELS_DIR / "all-MiniLM-L6-v2")
VOSK_MODEL_SMALL_EN_US_0_15 = str(MODELS_DIR / "vosk-model-small-en-us-0.15")
VOSK_MODEL_EN_US_0_22_LGRAPH = str(MODELS_DIR / "vosk-model-en-us-0.22-lgraph")
INTERNLM__2_5__1_8_B_CHAT = str(MODELS_DIR / "internlm2_5-1_8b-chat")
INTERN_VL_2_5_2B = str(MODELS_DIR / "InternVL2_5-2B")

THUDM_GLM_EDGE_1_5_B_CHAT = str(MODELS_DIR / "THUDM-glm-edge-1.5b-chat")
CROSS_ENCODER_NLI_DISTILROBERTA_BASE = str(MODELS_DIR / "cross-encoder-nli-distilroberta-base")


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
        self.logger.debug(f"[UTILS][is_prompt_valid()] Prompt does not make sense for a chat ...")
        return result

    @staticmethod
    def get_unique_choice(options, last_choice=None):
        """Get a randomly selected option, avoiding the same choice as the last one."""
        if len(set(options)) <= 1:  # Guard to check if only one unique value exists
            return options[0]
        new_choice = random.choice(options)
        while new_choice == last_choice:
            new_choice = random.choice(options)
        return new_choice

    @staticmethod
    def clean_markdown_to_text(markdown_text):
        """
        Cleans the output from Markdown and extracts plain text efficiently.
        """
        renderer = mistune.create_markdown()
        html = renderer(markdown_text)
        plain_text = BeautifulSoup(html, "html.parser").get_text()
        return plain_text

    @staticmethod
    def clean_new_lines(plain_text):
        """ Remove new lines from a string. """
        return plain_text.replace('\n', ' ')

    @staticmethod
    def clean_multiple_spaces(plain_text):
        """ Remove multiple spaces from a string. """
        return MULTIPLE_SPACES_REGEX.sub(' ', plain_text)

    def clean_incomplete(self, plain_text):
        """ Cleans incomplete parts of a string. """
        punctuation_marks = (".", "!", "?", ";")
        cut_part = None
        if any(mark in plain_text for mark in punctuation_marks):
            if not plain_text.endswith(punctuation_marks):
                last_punctuation = max(plain_text.rfind("."), plain_text.rfind("!"), plain_text.rfind("?"), plain_text.rfind(";"))
                if last_punctuation != -1:
                    cut_part = plain_text[last_punctuation + 1:].strip()
                    plain_text = plain_text[:last_punctuation + 1].strip()
        plain_text = plain_text.strip()
        if cut_part:
            self.logger.debug(f"[UTILS] Cleaned out incomplete part: {cut_part}")
        return plain_text

    @staticmethod
    def clean_emojis(plain_text):
        return EMOJIS_REGEX.sub("", plain_text)

    @staticmethod
    def is_eastern_asian(plain_text):
        return ((CHINESE_CHARACTERS_REGEX.search(plain_text))
                or (JAPANESE_CHARACTERS_REGEX.search(plain_text))
                or (KOREAN_CHARACTERS_REGEX.search(plain_text))
                or (JAPANESE_CHARACTERS_REGEX.search(plain_text)))

    def deep_text_clean(self, plain_text: str):
        if not plain_text:
            return ""
        # Preserve Chinese, Korean and Japanese characters and phrases ...
        if self.is_eastern_asian(plain_text):
            return plain_text
        # Make a deep clean of the text
        plain_text = self.clean_markdown_to_text(plain_text)
        plain_text = self.clean_new_lines(plain_text)
        plain_text = self.clean_multiple_spaces(plain_text)
        plain_text = self.clean_incomplete(plain_text)
        plain_text = self.clean_emojis(plain_text)
        return plain_text.strip()

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
