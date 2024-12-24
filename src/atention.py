from contextlib import ContextDecorator


class Attention(ContextDecorator):
    def __init__(self, is_for_assistant, component=None, logger=None):
        self.is_for_assistant = is_for_assistant
        self.component = component
        self.logger = logger

    def __enter__(self):
        if not self.is_for_assistant:
            # Log or handle the case where it's not for assistant
            comp = f"[{self.component}]" if self.component else ""
            message = f"[APP]{comp} Not in chat mode. Activate [Chat Mode] or call AI Assistant by [Name]."
            self.logger.debug(message)
            return
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Additional cleanup logic (if needed)
        pass
