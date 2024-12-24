import re

# Precompile regex patterns
NON_ALPHANUMERIC_REGEX = re.compile(r'[^a-zA-Z0-9\s]')
MULTIPLE_SPACES_REGEX = re.compile(r'\s+')

# Colorize colors for ball effects
RED = {"color": (80, 0, 0, 150)}
BLUE = {"color": (0, 10, 100, 150)}
GREEN = {"color": (0, 60, 0, 150)}
YELLOW = {"color": (135, 105, 35, 150)}
BRIGHT_YELLOW = {"color": (212, 175, 150)}
MAGENTA = {"color": (60, 0, 60, 150)}
GREY = {"color": (50, 50, 50,150)}
ORANGE = {"color": (100, 50, 0, 150)}
BROWN = {"color": (60, 30, 10, 150)}
PINK = {"color": (70, 20, 30, 150)}
CYAN = {"color": (0, 80, 80, 150)}
LIME = {"color": (50, 100, 0, 150)}
PURPLE = {"color": (50, 0, 100, 150)}
TURQUOISE = {"color": (0, 100, 100, 150)}
DARK_GREY = {"color": (30, 30, 30, 150)}
LIGHT_GREY = {"color": (80, 80, 80, 150)}
TRANSPARENT = {"color": (0, 0, 0)}

# Zoom effects for the ball
SLOW_GROW = {"factor": 1.1, "duration": 1500}
ZOOM_IN = {"factor": 0.95, "duration": 500}
FAST_ZOOM = {"factor": 0.95, "duration": 100}

# Bot utility words and expressions
SHORT_CONFIRMS = [
    'imh!',
    'ooh!',
    'ahh!',
    'uhh!',
    'aha!',
    'sure!',
    'yup!',
    'yes!',
    'right!',
    'ok!',
    'yeah!',
    'got it!',
    'roger!',
    'fine!',
    'done!',
    'okay!',
    'absolutely!',
    'definitely!',
    'for sure!'
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
    "Hello there!",
    "Howdy-do!",
    "Hey friend!",
    "Good to see you!",
    "Hi buddy!",
    "Nice to meet you!",
    "Top of the morning!",
    "Wassup!",
    "Hey hey!",
    "Hiya!",
    "Hi folks!",
    "Hi everyone!",
    "Welcome!",
    "Heya!",
    "Howdy partner!",
    "Long time no see!",
    "What's crackin'?",
    "Ahoy matey!",
    "Hi, sunshine!",
    "Hey champ!",
    "Look who it is!",
    "Hey, hey rockstar!",
    "How's it going?",
    "What's cookin'?",
    "Hi there, stranger!",
]

GOODBYES = [
    "See ya!",
    "Take care!",
    "Catch you later!",
    "Bye!",
    "Later!",
    "Cheers!",
    "Until next time!",
    "Peace out!",
    "Farewell!",
    "Bye-bye!",
    "Catch you on the flip side!",
    "See you later!",
    "Toodles!",
    "Don't be a stranger!",
    "See you soon!",
    "Stay awesome!",
    "Laters!",
    "I'm out!",
    "Keep in touch!",
    "Hasta la vista, baby!",
    "Take it easy!",
    "So long!",
    "See you around!",
    "Take it easy!",
    "Stay safe!",
    "Bye for now!",
    "Over and out!",
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
    "Let me think...",
    "Curious...",
    "Seems intriguing...",
    "Got it...",
    "Okay...",
    "Alrighty...",
    "Looking...",
    "Whoa there..."
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
    "Gotcha...",
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
    "Preparing info...",
    "Almost there...",
    "Don't go anywhere...",
    "Hold on...",
    "Fetching info...",
    "Gathering data...",
    "Let me see...",
    "Processing...",
    "Working on it...",
    "In progress...",
    "Give me a moment...",
    "Coming right up...",
    "Almost done...",
    "Bear with me...",
    "Wait a moment...",
    "Hold tight...",
    "Patience, please...",
    "Finalizing...",
    "One sec...",
    "Right away...",
    "Let me check...",
    "Just a minute...",
    "Hang on...",
    "Crunching numbers...",
    "Hold the line...",
    "Loading...",
    "Just a few seconds...",
    "Stay tuned...",
    "Nearly there...",
    "Let me do the magic...",
    "Brewing the answer...",
    "Cooking up...",
    "Uploading thoughts...",
    "On it, boss!",
    "Almost cooked!",
    "We're almost there...",
    "It's on the way!",
    "You're gonna like this..."
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
    "Respectful language goes a long way.",
    "Words are powerful—let’s use them kindly.",
    "I hear you, but we can express ourselves without profanities.",
    "Yikes! Let’s stay cool and talk it out politely.",
    "Oof! That’s intense. Let’s keep it chill.",
    "Whoa, that's some spicy vocabulary—let’s keep it mild!",
    "Please, let’s use our indoor voices—metaphorically speaking.",
    "Time-out for those words. Let’s keep it clean.",
    "I respect your input, but let's keep this civil and professional.",
    "Our conversation can be more productive without offensive language.",
    "Profanities detected; rebooting polite mode!",
    "Let’s take a deep breath and keep it friendly.",
    "Let’s save the dramatic words for a screenplay!",
    "Mutual respect will ensure a better conversation."
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
    "Sure!",
    "Cool, gotcha!",
    "You bet!",
    "All set!",
    "Yup, on it!",
    "Alrighty!",
    "Certainly!",
    "No problem!",
    "Consider it done!",
    "Happy to help!",
    "Copy that!",
    "Confirmed!",
    "On it like a bot!",
    "Acknowledged, captain!",
    "Done and done!",
    "Of course!",
    "Got this!",
    "Right on it!",
    "Checked and ready!",
    "I'll handle this for you.",
    "You can count on me!",
    "That works for me!"
]

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
    "Once more?",
    "Again?",
    "Missed it?",
    "More?",
    "How?",
    "Why?",
    "Info?"
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
