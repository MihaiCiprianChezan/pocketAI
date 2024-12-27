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

    def add(self, message, role="user"):
        with self.history_lock:
            try:
                if len(self.history) == self.history_size:
                    self.history = self.history[1:]
                history_entry = {"role": role, "content": message}
                self.history.append(history_entry)
                self.logger.error(f"[HistoryManager] Added history entry: {history_entry}")
            except Exception as e:
                self.logger.error(f"[HistoryManager] Error adding message to history: {e}, {traceback.format_exc()}")

    def clean(self):
        with self.history_lock:
            self.history = []
            self.clean_history = []
