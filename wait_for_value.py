import time
from contextlib import ContextDecorator


class WaitForValue(ContextDecorator):
    def __init__(self, var_getter, timeout, check_interval=0.3):
        """
        Args:
            var_getter (callable): A function or lambda that retrieves the variable to check (should return the variable, e.g., `lambda: some_var`).
            timeout (float): The maximum amount of time to wait (in seconds) before timing out.
            check_interval (float): How often to check the variable (in seconds).
        """
        self.var_getter = var_getter
        self.timeout = timeout
        self.check_interval = check_interval

    def __enter__(self):
        """
        Waits for the variable to be populated, or raises a TimeoutError if the timeout is exceeded.
        """
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            value = self.var_getter()
            if value is not None:  # Stop waiting if the variable is populated
                return value
            time.sleep(self.check_interval)
        # If we reach here, the variable was not populated within the timeout
        raise TimeoutError("Timeout: The variable did not become populated within the specified timeout.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context gracefully if no uncaught exceptions occurred.
        """
        return False  # Do not suppress exceptions
