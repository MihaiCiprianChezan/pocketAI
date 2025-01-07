from app_logger import AppLogger
from assistant.chat import ChatManager
from assistant.history import HistoryManager
from assistant.model import ModelManager
from assistant.tool import ToolManager
from similarity import Similarity
from utils import INTERN_VL_2_5_2B


class Assistant:
    ASSISTANT_NAME = "Opti"

    MAIN_INSTRUCTIONS = (
        f"You are {ASSISTANT_NAME}, a friendly, helpful, honest, and harmless AI assistant who can understand "
        f"and communicate fluently in the language chosen by the user such as English and 中文.\n"
        f"Always respond clearly.")
    CONVERSATION = (
        f"{MAIN_INSTRUCTIONS}\n"
        f"You are using natural conversational human language, just as if you were speaking directly to someone in live chat.\n"
        f"You respond in plain text paragraphs without any kind of formating.\n"
        f"You do NOT use in outputs: new lines, markdown, headings, bold, bullet points, numbered lists, or any kind of formatting.\n")
    INLINE_TEXT = (
        f"{MAIN_INSTRUCTIONS}\n"
        "You are using ONLY plain text formatting (spaces, tabs, dashed, points etc) instead of markdown or any other formatting."
    )

    # Minimum score to consider the message related wit the previous message
    MIN_MESSAGE_SCORE = 0.3
    # Minimum score to consider the message related with the whole history
    MIN_HISTORY_SCORE = 0.5

    def __init__(self, model_name=INTERN_VL_2_5_2B):
        self.logger = AppLogger()
        self.model_mng = ModelManager(model_name, trust_remote_code=True, logger=self.logger)
        self.tools_mng = ToolManager(logger=self.logger)
        self.history_mng = HistoryManager(logger=self.logger)
        self.chat_mng = ChatManager(self.model_mng, self.history_mng, logger=self.logger)
        self.similarity = Similarity()
        self.name = self.__class__.__name__

    def initialize(self):
        self.logger.info("Initializing Assistant...")
        self.model_mng.initialize()
        self.model_mng.model.system_message = self.CONVERSATION
        self.model_mng.warm_model()

    def clean_history(self):
        self.history_mng.clean()
        self.logger.error(f"[{self.name}] > Cleaning history: {self.history_mng.history}")
        with self.history_mng.history_lock:
            self.history_mng.history = []
        self.logger.error(f"[{self.name}] > History after cleaning: {self.history_mng.history}")

    def regulate_history(self, message, context_free=False):
        self.logger.debug(f"[{self.name}] History before regulation: {self.history_mng.get_formated()}")
        if context_free:
            # TODO: Check if really needs to clear history ...
            self.logger.debug(f"[{self.name}] <CLEANING_HISTORY> because context_free=True.")
            self.history_mng.empty_clean()
        else:
            if len(self.history_mng.history) > 0:
                # Make space in the history for new entries if history is full
                previous_message = self.history_mng.history[-1][0]
                message_score = self.similarity.get_score([message, previous_message])
                history_messages = [message[0] for message in self.history_mng.history]
                # Calculate similarity between current message, previous message and whole history
                history_score = self.similarity.get_score([message, *history_messages])
                self.logger.debug(f"[{self.name}]  User query is similar {message_score} to the previous message and {history_score} similar to all history")
                # Clear history if current message related score is below min message and history (related) scores
                if message_score < self.MIN_MESSAGE_SCORE and history_score < self.MIN_HISTORY_SCORE:
                    self.logger.debug(f"[{self.name}] <CLEANING_HISTORY> because user has started a new topic.")
                    self.history_mng.clean()
        self.logger.debug(f"[{self.name}] History after regulation: {self.history_mng.get_formated()}")

    def get_response(self, user_prompt, context_free=False, return_full_history=False):
        if not user_prompt.message:
            return
        self.logger.debug(f"[{self.name}] User query: `{user_prompt.message}`")
        self.regulate_history(user_prompt.message, context_free)

        tool_result = self.tools_mng.dispatch(user_prompt)

        if tool_result:
            if not return_full_history:
                self.history_mng.add(user_prompt.message, tool_result)
            return tool_result

        if user_prompt.images:
            # self.logger.debug(f"[{self.name}]  user_prompt.message: `{user_prompt.message}`")
            # self.logger.debug(f"[{self.name}]  user_prompt.pixel_values: `{user_prompt.pixel_values}`")
            # self.logger.debug(f"[{self.name}]  user_prompt.num_patches_list: `{user_prompt.num_patches_list}`")

            return self.chat_mng.stream_chat_image(
                user_prompt.message,
                pixel_values=user_prompt.pixel_values,
                num_patches_list=user_prompt.num_patches_list
            )

        if user_prompt.video:
            return self.chat_mng.stream_chat_video(
                user_prompt.message,
                pixel_values=user_prompt.pixel_values,
                video_prefix=user_prompt.video_prefix,
                num_patches_list=user_prompt.num_patches_list
            )

        return self.chat_mng.stream_chat(user_prompt.message, return_full_history=return_full_history)
