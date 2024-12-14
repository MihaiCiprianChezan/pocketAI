import re

# Precompile regex patterns
NON_ALPHANUMERIC_REGEX = re.compile(r'[^a-zA-Z0-9\s]')
MULTIPLE_SPACES_REGEX = re.compile(r'\s+')

# Colorize colors for ball effects
RED = {"color": (100, 0, 0)}
BLUE = {"color": (0, 20, 120)}
GREEN = {"color": (0, 100, 0)}
YELLOW = {"color": (80, 80, 0)}
MAGENTA = {"color": (80, 0, 80)}
GREY = {"color": (50, 50, 50)}
ORANGE = {"color": (100, 50, 0)}
BROWN = {"color": (60, 30, 10)}
PINK = {"color": (100, 50, 60)}
CYAN = {"color": (0, 80, 80)}
LIME = {"color": (50, 100, 0)}
PURPLE = {"color": (50, 0, 100)}
TURQUOISE = {"color": (0, 100, 100)}
DARK_GREY = {"color": (30, 30, 30)}
LIGHT_GREY = {"color": (80, 80, 80)}

# Zoom effects for the ball
SLOW_GROW = {"factor": 1.1, "duration": 1500}
ZOOM_IN = {"factor": 0.95, "duration": 500}
FAST_ZOOM = {"factor": 0.95, "duration": 100}

# Bot utility words and expressions
SHORT_CONFIRMS = [
    'hmm...',
    'ooh...',
    'ahh..',
    'uhh..',
    'aha...',
    'sure...',
    'oh...',
    'ok...',
    'yup...',
    'yes...',
    'right...'
]

HELLOS = [
    "Hey!",
    "Hello!",
    "Hi there!",
    "Howdy!",
    "Greetings!",
    "What's up?",
    "Good day!",
    "Salutations!",
    "Hiya!",
    "Hey dear!",
    "Yo!"
]

GOODBYES = [
    "See ya!",
    "Take care!",
    "Catch you later!",
    "Bye!",
    "Later!",
    "Cheers!",
    "Adios!",
    "Ciao!",
    "Until next time!",
    "Peace out!",
    "Farewell!"
]

THINKING_SOUNDS = [
    "Hmm...",
    "Let's see...",
    "Interesting...",
    "Just a sec...",
    "Good one...",
    "Hold on...",
    "One sec...",
    "Aha...",
    "Alright...",
    "Wait...",
    "Hmm, okay...",
    "Curious...",
    "Got it...",
    "Okay...",
    "Alrighty...",
    "Looking...",
    "Huh...",
    "Sure...",
    "Well...",
    "Cool...",
    "Okey!",
    "Yep...",
    "Aha...",
    "Right...",
    "Uh-huh...",
    "Yup...",
    "Ah...",
    "Yeah...",
    "Righty...",
    "Yep-yep...",
    "Ooh...",
    "Ah-ha...",
    "Mmm...",
    "Okay then...",
    "So...",
    "Hmm, alright...",
    "Uh...",
    "Eh...",
    "Oh..."
]

WAITING_SOUNDS = [
    "Just a sec...",
    "Still thinking...",
    "Answer is on the way...",
    "Hang tight...",
    "One moment please...",
    "Calculating...",
    "Just a bit longer...",
    "Stay with me...",
    "Preparing your answer...",
    "Almost there...",
    "Don't go anywhere...",
    "Hold on...",
    "Fetching information...",
    "Gathering data...",
    "Let me see...",
    "Processing...",
    "Working on it...",
    "In progress...",
    "Give me a moment...",
    "Coming right up..."
]

POLITE_RESPONSES = [
    "Oh, that's colorful. Let's keep it respectful.",
    "Well, that escalated quickly. Let's tone it down a notch.",
    "I believe we can be civil. How about we try that?",
    "Let's aim for a more polite chat.",
    "Classy! But let's keep it clean.",
    "Such language! Let's steer clear of that.",
    "Let's keep things friendly.",
    "I prefer conversations without the profanities.",
    "Whoa there! Let’s stay classy.",
    "Let’s keep it professional.",
    "Mind your language, please.",
    "Let's keep our conversation respectful.",
    "Let's use kind words.",
    "There's no need for that language.",
    "Keep it polite, please.",
    "Let's stay respectful.",
    "Let's keep it civil.",
    "Let's keep it clean.",
    "Respectful language goes a long way.",
    "We can communicate better with respectful language."
]

ACKNOWLEDGEMENTS = [
    "Got it!",
    "Sure thing!",
    "Okay!",
    "Alright!",
    "Understood!",
    "Noted!",
    "Will do!",
    "Absolutely!",
    "Roger that!",
    "Affirmative!",
    "Yes!",
    "Right away!",
    "Sounds good!",
    "I see!",
    "Acknowledged!",
    "Okay then!",
    "Great!",
    "Perfect!",
    "Sure!"
]

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "it": "Italian",
    "ko": "Korean",
    "hi": "Hindi",
    "bn": "Bengali",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "cs": "Czech",
    "he": "Hebrew"
}

# Glitches in voice recognizer got no clue how those are generated may be some glitches in the model
GLITCHES = [
    "I'm going to go to the next room",
    "1.5% 1.5% 1.5% 1.5% 1.5%"
    "I'm going to get the right"
    "I'm going to get you a little bit more"
    "1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1"
]

VOICES = {
    "Microsoft Aria (Natural)": {"language": "English (United States)"},
    "Microsoft Ryan (Natural)": {"language": "English (United Kingdom)"},
    "Microsoft Sonia (Natural)": {"language": "English (United Kingdom)"},
    "Microsoft Dalia (Natural)": {"language": "Spanish (Mexico)"},
    "Microsoft Denise (Natural)": {"language": "French (France)"},
    "Microsoft Katja (Natural)": {"language": "German (Germany)"},
    "Microsoft Xiaoxiao (Natural)": {"language": "Chinese (Simplified, China)"}
}

UNCLEAR_PROMPT_RESPONSES = [
    "What?",
    "Huh?",
    "Pardon?",
    "Sorry?",
    "Come again?",
    "Explain?",
    "Repeat?",
    "Clarify?",
    "Say again?",
    "Meaning?",
    "Details?",
    "Sorry, what?",
    "Eh?",
    "What now?",
    "Can you?",
    "Repeat that?",
    "More info?",
    "Missed that.",
    "How so?",
    "Once more?"
]
