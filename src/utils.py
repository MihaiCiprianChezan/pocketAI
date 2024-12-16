import random
import re

import keyboard
import pyperclip

from varstore import GLITCHES, NON_ALPHANUMERIC_REGEX, MULTIPLE_SPACES_REGEX


def is_recog_glitch(spoken, glitches=GLITCHES):
    for g in glitches:
        if spoken.find(g) != -1:
            return True
    return False


def is_prompt_valid(spoken, clean, min_tokens=2, min_chars=5, min_unique_tokens=1, vulgarities_list=None):
    if not clean.strip():
        return False
    token_count = len(clean.split())
    if token_count < min_tokens:
        return False
    char_count = len(clean)
    if char_count < min_chars:
        return False
    if vulgarities_list:
        clean_lower = clean.lower()
        if any(word in clean_lower for word in vulgarities_list):
            return False
    if spoken.isalpha() and len(set(spoken.lower())) <= 2:
        return False
    tokens = clean.split()
    if len(set(tokens)) < min_unique_tokens:
        return False
    most_common_token = max(set(tokens), key=tokens.count)
    if tokens.count(most_common_token) / token_count > 0.8:
        return False
    if token_count == 1 and char_count > 20:
        return False
    if clean.isdigit() or all(not c.isalnum() for c in clean):
        return False
    return True


def get_unique_choice(options, last_choice=None):
    """
    Gets a randomly selected item from the list of options
    ensuring it's not the same as the last choice.
    """
    new_choice = random.choice(options)
    # Ensure no consecutive duplicates
    while new_choice == last_choice:
        new_choice = random.choice(options)
    last_choice = new_choice
    return new_choice

import re


import re


def clean_response(llm_response, ensure_punctuation=True):
    """
    Cleans an LLM response for live chat.
    Args:
        llm_response: The LLM-generated text.
        ensure_punctuation: If True, removes the last incomplete sentence.
    """
    if not isinstance(llm_response, str):
        return ""
    # Combine regex for efficiency
    combined_pattern = re.compile(
        r"\*\*|"  # Remove double asterisks
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed Characters
        "]+"
        r"|\s+",
        flags=re.UNICODE,
    )
    cleaned_text = combined_pattern.sub(' ', llm_response)
    cleaned_text = cleaned_text.replace('\n', ' ')
    cut_part = None
    if ensure_punctuation and "." in cleaned_text:
        if not cleaned_text.endswith((".", "!", "?")):
            last_punctuation = max(cleaned_text.rfind("."), cleaned_text.rfind("!"), cleaned_text.rfind("?"))
            if last_punctuation != -1:
                cut_part = cleaned_text[last_punctuation + 1:].strip()  # Store cut part here.
                cleaned_text = cleaned_text[:last_punctuation + 1].strip()
    cleaned_text = cleaned_text.strip()
    if cut_part:
        print(f"<!> Cleaned out incomplete part: {cut_part}")
    return cleaned_text



def clean_text(text):
    """Clean the input text by removing non-alphanumeric characters."""
    # alphanumeric_text = NON_ALPHANUMERIC_REGEX.sub('', text)
    single_spaced_text = MULTIPLE_SPACES_REGEX.sub(' ', text)
    return single_spaced_text.strip().lower()

def paste_at_cursor():
    """Paste copied text at the cursor."""
    text = pyperclip.paste()
    keyboard.send(text)

def write_text(text):
    """Write the text dynamically with a slight delay."""
    keyboard.write(text)


