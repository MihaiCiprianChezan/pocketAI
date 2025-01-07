import json
from threading import Lock
import traceback

from app_logger import AppLogger


class HistoryManager:
    def __init__(self, history=None, history_size=1000, logger=None):
        self.logger = logger if logger else AppLogger()
        self.history_lock = Lock()
        self.history_size = history_size
        self.history = history if history else []
        self.clean_history = []

    @staticmethod
    def validate(history):
        if not isinstance(history, list):
            raise ValueError("History must be a list of tuple entries of format (user_message, assistant_message).")
        for entry in history:
            if not isinstance(entry, tuple) or (len(entry) != 2):
                raise ValueError(f"Invalid history format: {entry}, must be a tuple of  of format (user_message, assistant_message).")

    def add_to(self, target_history, user_message, assistant_message):
        with self.history_lock:
            try:
                history_entry = (user_message, assistant_message)
                if len(target_history) == self.history_size:
                    target_history = target_history[1:]
                target_history.append(history_entry)
                self.logger.debug(f"[HistoryManager] Added history entry: {history_entry}")
            except Exception as e:
                self.logger.debug(f"[HistoryManager] Error adding message to history: {e}, {traceback.format_exc()}")
            finally:
                return target_history

    def add(self, user_message, assistant_message):
        self.history = self.add_to(self.history, user_message, assistant_message)

    def add_clean(self, user_message, assistant_message):
        self.clean_history = self.add_to(self.clean_history, user_message, assistant_message)

    def clean(self):
        # self.logger.error(f"[HistoryManager] >>> Cleaning history: {self.history}")
        with self.history_lock:
            self.history = self.clean_history
        # self.logger.error(f"[HistoryManager] >>> History after cleaning: {self.history}")

    def empty_clean(self):
        self.logger.error(f"[HistoryManager] > Cleaning history: {self.history}")
        with self.history_lock:
            self.history = []
        self.logger.error(f"[HistoryManager] > History after cleaning: {self.history}")

    def empty_all(self):
        with self.history_lock:
            self.history = []
            self.clean_history = []

    def get_formated(self):
        return json.dumps(self.history, indent=2)
