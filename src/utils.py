import random
import re
from varstore import GLITCHES

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

def clean_response(llm_response):
    """
    Cleans a response from an LLM model by:
    - Removing newlines
    - Stripping emojis
    - Removing leading/trailing whitespaces
    - Optionally performing further text sanitizations

    Parameters:
    - llm_response (str): The response text from the LLM.

    Returns:
    - str: The cleaned response.
    """
    if not isinstance(llm_response, str):
        raise ValueError("Input must be a string")

    # Remove newlines and replace with a single space
    cleaned_text = llm_response.replace('\n', ' ')

    # Strip emojis using a regex pattern
    emoji_pattern = re.compile(
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
        "]+",
        flags=re.UNICODE,
    )
    cleaned_text = emoji_pattern.sub('', cleaned_text)

    # Strip leading and trailing whitespaces
    cleaned_text = cleaned_text.strip()

    return cleaned_text
