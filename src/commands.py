ENTER_EDIT_MODE = (  # EDIT MODE enter - Will run edit commands
    ("editing", "mode"),
    ("enter", "editing"),
    ("editing", "please"),
    ("start", "editing"),
    ("enable", "editing"),
    ("switch", "editing"),
    ("go", "editing"),
    ("write", "mode"),
    ("activate", "editing"),
    ("turn", "on", "editing")
)
EXIT_EDIT_MODE = (  # EDIT MODE exit - Will NOT run edit commands
    ("exit", "editing"),
    ("close", "editing"),
    ("stop", "editing"),
    ("stop", "stop")
)
ENTER_CHAT_MODE = (  # CHAT MODE enter, to prompt the AI
    ("start", "chat"),
    ("resume", "chat"),
    ("chat", "again"),
    ("enter", "chat"),
    ("return", "chat"),
    ("switch", "chat"),
    ("open", "chat"),
    ("back", "to", "chat"),
    ("activate", "chat"),
    ("chat", "please")
)
PAUSE_CHAT_MODE = (  # CHAT MODE pause, AI will not be prompted
    ("pause", "chat"),
    ("have", "pause"),
    ("stop", "chat"),
    ("end", "chat"),
    ("hold", "chat"),
    ("disable", "chat"),
    ("turn", "off", "chat"),
    ("pause", "conversation"),
    ("stop", "conversation"),
    ("halt", "chat"),
    ("pause", "for", "now")
)
INTERRUPT_ASSISTANT = (  # Interruption commands for ongoing speeches
    ("wait", "stop"),
    ("wait", "wait"),
    ("stop", "stop"),
    ("ok", "thanks"),
    ("thank", "you"),
    ("hold", "on"),
    ("all", "right"),
    ("okay", "okay"),
    ("stop", "it"),
    ("wait", "up"),
    ("hold", "up"),
    ("stop", "please"),
    ("pause", "now"),
    ("okay", "stop"),
    ("thanks", "anyway"),
    ("forget", "it"),
    ("leave", "it"),
    ("not", "now"),
    ("no", "thanks"),
    ("all", "done"),
    ("that", "is", "enough"),
    ("it's", "okay"),
    ("alright", "then"),
    ("wait", "please"),
    ("never", "mind"),
    ("stop", "that"),
    ("enough", "now"),
    ("move", "on"),
    ("just", "stop"),
)
READ_THIS = ("read", "this")
EXPLAIN_THIS = ("explain", "this")
SUMMARIZE_THIS = (
    ("summarize", "this"),
    ("summary", "of", "this"),
    ("make", "a", "summary"),
    ("create", "a", "summary"),
    ("provide", "a", "summary"),
    ("give", "me", "a", "summary"),
    ("can", "you", "summarize", "this"),
)
EXIT = (
    ("exit", "app"),
    ("quit", "app"),
    ("exit", "now"),
    ("please", "exit"),
    ("quit", "now"),
    ("quit", "please"),
)
TRANSLATE_TO_ENGLISH = (
    ("translate", "to", "english"),
    ("translate", "into", "english")
)
TRANSLATE_TO_FRENCH = (
    ("translate", "to", "french"),
    ("translate", "into", "french")
)
TRANSLATE_TO_GERMAN = (
    ("translate", "to", "german"),
    ("translate", "into", "german")
)
TRANSLATE_TO_SPANISH = (
    ("translate", "to", "spanish"),
    ("translate", "into", "spanish")
)
TRANSLATE_TO_CHINESE = (
    ("translate", "to", "chinese"),
    ("translate", "into", "chinese")
)
NEW_LINE = ("new", "line")
COPY_TEXT = ("copy", "text")
PASTE_TEXT = ("paste", "text")
CUT_TEXT = ("cut", "text")
DELETE_TEXT = ("delete", "text")
SELECT_ALL = ("select", "all")
SELECT_WORD = ("select", "word")
SELECT_LINE = ("select", "line")
SELECT_PARAGRAPH = ("select", "paragraph")
MOVE_UP = ("move", "up")
MOVE_DOWN = ("move", "down")
MOVE_LEFT = ("move", "left")
MOVE_RIGHT = ("move", "right")
UNDO = ("undo", "please")
REDO = ("redo", "please")