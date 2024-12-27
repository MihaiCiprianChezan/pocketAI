from itertools import chain
from app_logger import AppLogger
from assistant.intent import GENERIC_INTENTS, Intent
from assistant.tools import DateTimeTool, get_tools_summary, PythonExecutionTool, WikipediaTool


class ToolManager:
    def __init__(self):
        self.logger = AppLogger()
        self.tools = [
            PythonExecutionTool(),
            DateTimeTool(),
            WikipediaTool(),
        ]
        self.tools_summary = get_tools_summary(self.tools)
        self.intent = Intent(labels=list(chain.from_iterable(tool.intents for tool in self.tools)) + GENERIC_INTENTS)

    def get_tool_for_intent(self, intent):
        for tool in self.tools:
            if intent.label in tool.intents:
                return tool
        return None  # No matching tool for the detected intent.

    def dispatch(self, query):
        result = []
        target = None
        message_intent = self.intent.detect(query)
        self.logger.debug(f"[ToolManager] Detected intent: `{message_intent}`")

        tool = self.get_tool_for_intent(message_intent)
        if tool:
            tool_result = tool.forward(query)
            if tool_result:
                target = tool.target
                result = [{"role": f"{tool.target}", "content": tool_result}]
        return result, target
