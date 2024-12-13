import random

from varstore import GLITCHES

def is_recog_glitch(spoken, glitches=GLITCHES):
    for g in glitches:
        if spoken.find(g) != -1:
            return True
    return False


def is_prompt_valid(spoken, clean, min_tokens=3, min_chars=10, min_unique_tokens=2, vulgarities_list=None):
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