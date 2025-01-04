from itertools import chain
from app_logger import AppLogger
from assistantm.intent import DetectionResult, GENERIC_INTENTS, Intent
from assistantm.tools import DateTimeTool, get_tools_summary, PythonExecutionTool, WikipediaTool
from prompt import Prompt


class ToolManager:
    def __init__(self, logger=None):
        self.logger = logger if logger else AppLogger()
        self.tools = [
            PythonExecutionTool(),
            DateTimeTool(),
            WikipediaTool(),
        ]
        self.tools_summary = get_tools_summary(self.tools)
        self.intent = Intent(labels=list(chain.from_iterable(tool.intents for tool in self.tools)) + GENERIC_INTENTS)

    def get_tool_for_intent(self, intent):
        """
        Retrieves a tool that matches the given intent.

        This method iterates through a list of pre-defined tools and checks if the
        provided intent's label exists within any tool's supported intents. If a
        match is found, it returns the corresponding tool. Otherwise, it returns
        None.

        Args:
            intent: Intent object containing a label that specifies the purpose or
                requirement for which a tool is sought.

        Returns:
            The tool object that matches the given intent if found, or None if no
            matching tool is available.
        """
        for tool in self.tools:
            if intent.label in tool.intents:
                return tool
        return None  # No matching tool for the detected intent.

    def dispatch(self, user_prompt:Prompt):
        """
        Dispatches the user prompt to the appropriate tool based on the detected or explicitly provided intent.

        This method determines the intent of the user's message and matches it with the appropriate
        tool. If the intent is provided directly in the user prompt, it bypasses the intent detection
        process. The method then attempts to find a corresponding tool for the detected or provided
        intent and forwards the user's message to the tool for processing if a match is found. The
        result of the tool's processing is returned to the caller.

        Args:
            user_prompt (Prompt): The user prompt containing the message and optionally the intent.

        Returns:
            Any: The result of the processing by the matched tool, if a tool is found. If no tool
            matches the intent or the tool does not return a result, None is returned.
        """
        if not user_prompt.intent:
            message_intent = self.intent.detect(user_prompt.message)
            self.logger.debug(f"[ToolManager] <INTENT> detected: `{message_intent}`")
        else:
            message_intent = DetectionResult(label=user_prompt.intent, score=0.99)
            self.logger.debug(f"[ToolManager] Direct <INTENT>: `{message_intent}`")
        tool_match = self.get_tool_for_intent(message_intent)
        # If a tool is matched forward the message to the tool for processing and return the result
        if tool_match:
            self.logger.debug(f"[ToolManager] The <APPROPRIATE_TOOL> is: `{tool_match.__class__.__name__}`")
            tool_result = tool_match.forward(user_prompt.message)
            if tool_result:
                return tool_result
