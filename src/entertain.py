import random
import time


class Entertain:
    """
    A class to manage user entertainment logic during extended tasks.

    Attributes:
        should_entertain (bool): Whether entertainment is enabled.
        min_interval (int): Minimum interval (in seconds) between entertainment actions.
        max_interval (int): Maximum interval (in seconds) between entertainment actions.
        action (callable): Optional callback for the entertainment action.
    """

    def __init__(self, action, should_entertain=True, min_interval=2, max_interval=5):
        """
        Initialize the Entertain object.

        Args:
            action (callable): The action to perform for entertainment.
            should_entertain (bool): Enable or disable entertaining (affects all behavior).
            min_interval (int): Minimum interval in seconds between entertainment actions.
            max_interval (int): Maximum interval in seconds between entertainment actions.
        """
        self.should_entertain = should_entertain
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.action = action
        self._last_update_time = None
        self._random_interval = None

        if self.should_entertain:
            self.reset()

    def reset(self):
        """Resets the timing values for entertainment, but only if enabled."""
        if self.should_entertain:
            self._last_update_time = time.time()
            self._random_interval = random.uniform(self.min_interval, self.max_interval)

    def check_and_entertain(self):
        """
        Executes the entertainment action if enabled and the required interval has passed.
        """
        if self.should_entertain:
            if time.time() - self._last_update_time >= self._random_interval:
                if self.action:
                    self.action()
                self.reset()
