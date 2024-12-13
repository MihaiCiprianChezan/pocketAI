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

# Glitches in voice recognizer
GLITCHES = [
    "I'm going to go to the next room",
    "1.5% 1.5% 1.5% 1.5% 1.5%"
]

