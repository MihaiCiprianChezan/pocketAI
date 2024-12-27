from app_logger import AppLogger
from assistant.chat import ChatManager
from assistant.history import HistoryManager
from assistant.model import ModelManager
from assistant.tool import ToolManager
from similarity import Similarity
from utils import INTERNLM__2_5__1_8_B_CHAT

class Assistant:
    def __init__(self, model_name=INTERNLM__2_5__1_8_B_CHAT):
        self.logger = AppLogger()
        self.model_manager = ModelManager(model_name)
        self.tools_manager = ToolManager()
        self.history_manager = HistoryManager()
        self.chat_manager = ChatManager(self.model_manager, self.tools_manager, self.history_manager)
        self.similarity = Similarity()
        self.model_manager.warm_model()

    def get_response(self, message, context_free=False, max_length=150, top_p=0.8, temperature=0.7):
        self.logger.debug(f"[Assistant] History before: {self.history_manager.history}")
        self.logger.debug(f"[Assistant] User query: `{message}`")
        if context_free:
            history = self.history_manager.clean_history
        else:
            if len(self.history_manager.history) > 0:
                previous_message = self.history_manager.history[-1]["content"]
                message_score = self.similarity.get_score([message, previous_message])
                history_messages = [message["content"] for message in self.history_manager.history]
                history_score = self.similarity.get_score([message, *history_messages])
                self.logger.debug(f"[Assistant] User query is similar {message_score} to the previous message and {history_score} similar to all history")
                if message_score < 0.3 and history_score < 0.5:
                    self.logger.debug(f"[Assistant] User has started a new topic, cleaning history.")
                    self.history_manager.clean()
            history = self.history_manager.history
        self.history_manager.add(message, role="user")
        self.logger.debug(f"[Assistant] History after: {history}")

        tool_response, tool_target = self.tools_manager.dispatch(message)

        if tool_response:
            return tool_response[0]["content"]
        else:
            return self.chat_manager.predict(history, max_length, top_p, temperature)
