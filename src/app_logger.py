import logging

from PySide6.scripts.project import Singleton


class AppLogger:
    _instance = None  # Singleton instance
    _paused = False  # Global paused state for ALL logger instances
    _cached_logs = []  # Shared global log cache

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def __init__(self, file_name: str = "VoiceUtilApp.log", overwrite: bool = True, log_level: int = logging.DEBUG):
        # Instance initialization only adds attributes once
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.log_level = log_level

    def _initialize(self, file_name: str = "VoiceUtilApp.log", overwrite: bool = True, log_level: int = logging.DEBUG):
        """
        Initialize a singleton-style shared logger instance.

        Args:
            file_name (str): Name of the log file.
            overwrite (bool): Overwrite existing log file if True.
            log_level (int): Initial log level (e.g., logging.DEBUG).
        """
        self._logger = logging.getLogger("VoiceUtilLogger")

        # Prevent duplicate handlers
        if not self._logger.hasHandlers():
            self._logger.setLevel(log_level)

            # File handler
            file_mode = "w" if overwrite else "a"
            file_handler = logging.FileHandler(file_name, mode=file_mode, encoding="utf-8", delay=True, errors="ignore")
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
            self._logger.addHandler(file_handler)

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Return the shared logger instance."""
        if cls._instance:
            return cls._instance._logger
        raise RuntimeError("AppLogger has not been initialized.")

    @classmethod
    def pause(cls):
        """Globally pause logging for all instances."""
        cls._paused = True
        # if cls._instance:
        #     # cls._instance._logger.info("[LOGGER PAUSED] Logging is now paused.")

    @classmethod
    def resume(cls):
        """
        Globally resume logging for all instances and flush cached logs.
        Cached log messages are handled in the order they were received.
        """
        if cls._paused:
            cls._paused = False
            # if cls._instance:
            #     cls._instance._logger.info("[LOGGER RESUMED] Logging is now resumed. Flushing cached messages...")

            # Flush cached logs
            for record in cls._cached_logs:
                cls.get_logger().handle(record)
            cls._cached_logs.clear()

    def log(self, level, message):
        """
        Log a message:
        - If logger is globally paused, cache the log message.
        - Otherwise, emit the log immediately.
        """
        if self._paused:
            # Create a log record and cache it
            record = self._logger.makeRecord(self._logger.name, level, None, None, message, None, None)
            self._cached_logs.append(record)
        else:
            self._logger.log(level, message)

    # Convenience methods for commonly used log levels
    def info(self, message):
        self.log(logging.INFO, message)

    def debug(self, message):
        self.log(logging.DEBUG, message)

    def warning(self, message):
        self.log(logging.WARNING, message)

    def error(self, message):
        self.log(logging.ERROR, message)

    def critical(self, message):
        self.log(logging.CRITICAL, message)

    def set_log_level(self, log_level: int):
        """
        Dynamically update the log level for the shared logger.

        Args:
            log_level (int): New logging level (e.g., logging.DEBUG).
        """
        self._logger.setLevel(log_level)
        for handler in self._logger.handlers:
            handler.setLevel(log_level)
        self._logger.info(f"Log level updated to: {logging.getLevelName(log_level)}")

    def get_current_log_level(self) -> str:
        """
        Get the current logging level as a human-readable name.

        Returns:
            str: The current logging level (e.g., "DEBUG", "INFO").
        """
        return logging.getLevelName(self._logger.level)