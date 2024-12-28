from threading import Lock
import traceback
from app_logger import AppLogger


class HistoryManager:
    def __init__(self, history=None, history_size=1000):
        self.logger = AppLogger()
        self.history_lock = Lock()
        self.history_size = history_size
        self.history = history if history else []
        self.clean_history = []

    @staticmethod
    def validate(history):
        if not isinstance(history, list):
            raise ValueError("History must be a list of dictionaries.")
        for entry in history:
            if not isinstance(entry, dict) or ("role" not in entry or "content" not in entry):
                raise ValueError(f"Invalid history format: {entry}")

    def add_to(self, target_history, message, role="user"):
        with self.history_lock:
            try:
                if len(target_history) == self.history_size:
                    target_history = target_history[1:]
                history_entry = {"role": role, "content": message}
                target_history.append(history_entry)
                self.logger.error(f"[HistoryManager] Added history entry: {history_entry}")
            except Exception as e:
                self.logger.error(f"[HistoryManager] Error adding message to history: {e}, {traceback.format_exc()}")
            finally:
                return target_history


    def add(self, message, role="user"):
        self.history = self.add_to(self.history, message, role)

    def add_clean(self, message, role="user"):
        self.clean_history = self.add_to(self.clean_history, message, role)

    def clean(self):
        with self.history_lock:
            self.history = []
            self.clean_history = []

    def clean_history(self):
        with self.history_lock:
            self.history = []

    def clean_empty(self):
        with self.history_lock:
            self.clean_history = []
