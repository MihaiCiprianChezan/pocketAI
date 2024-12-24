from datetime import datetime

from transformers.agents import Tool
import wikipedia

from app_logger import AppLogger


class DateTimeTool(Tool):
    name = "datetime-tool"
    target = "direct"
    intents = [
        "current time",
        "what time",
        "time now",
        "current date",
        "what date",
        "date today",
        "current date and time"
    ]

    description = (
        "Provides the current system time in the format 'HH:MM AM/PM, Day Month Date YYYY'. "
        "Ensure correctly formatted, dynamically generated timestamps for time queries."
        "Output example: 'The current time and date is 08:32 AM, Sunday December 22 2024.'"
    )

    inputs = {
        "query": {"type": "string", "description": "A string that specifies either 'date' or 'time' for returning the current date or time."}
    }

    output_type = "string"

    def forward(self, query):
        # Handle the query to provide the current date or time
        # AppLogger().info(f"[DateTimeTool] Invoked - Generated timestamp: {current_timestamp}")
        if "date" in query.lower() and "time" in query.lower():
            return f"The current time and date is {datetime.now().strftime('%I:%M %p, %A %B %d %Y')}."
        elif "date" in query.lower():
            return f"The current date is {datetime.now().strftime('%A %B %d %Y')}."
        elif "time" in query.lower():
            return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
        else:
            return None
            # return "I can only provide the current date or time."


class WikipediaTool(Tool):
    name = "wikipedia-tool"
    target = "assistant"
    intents = [
        "find",
        "search wikipedia",
    ]

    description = (
        "Fetches summaries of topics or concepts from Wikipedia. "
        "Use this tool for informational queries requiring general up to date knowledge. "
        "Output should be concise and answer the query directly."
    )

    inputs = {
        "query": {"type": "string", "description": "A topic or concept to search on Wikipedia."}
    }
    output_type = "string"

    def forward(self, q: str) -> str:
        """Fetches a concise Wikipedia summary for the given query."""
        query = q[:300]  # Wikipedia can handle max 300 chars in a query
        try:
            # Fetch first 2 sentences of the result
            summary = wikipedia.summary(query, sentences=2)
            summary = summary.replace('\n', ' ').strip()
            AppLogger().info(f"[WIKIPEDIA_TOOL] Searched for: {query}, Result: {summary}")
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            AppLogger().error(f"[WIKIPEDIA_TOOL] The query '{query}' is ambiguous. Here are some suggestions: {', '.join(e.options[:5])}. ")
            return None
            # Handle ambiguous terms
            # return (
            #     f"The query '{query}' is ambiguous. Here are some suggestions: {', '.join(e.options[:5])}. "
            #     "Please specify further."
            # )
        except wikipedia.exceptions.PageError:
            # Handle cases where no results are found
            AppLogger().error(f"[WIKIPEDIA_TOOL] No results found")
            return None
            # return f"No page found on Wikipedia for '{query}'. Please try another query."
        except Exception as e:
            AppLogger().error(f"[WIKIPEDIA_TOOL] Error handling query '{query}': {repr(e)}")
            return None
            # return f"An unexpected error occurred while searching Wikipedia for '{query}'."


class PythonExecutionTool(Tool):
    name = "python-exec"
    target = "assistant"
    intents = [
        "request to run a Python code snippet"
    ]

    description = (
        "Executes Python code and returns the result. "
        "Use this tool only for simple Python snippets."
    )

    inputs = {
        "code": {"type": "string", "description": "Python code to execute"},
    }

    output_type = "string"

    def forward(self, code: str) -> str:
        """Executes the provided Python code and returns the result."""
        try:
            # Evaluate a single expression
            result = eval(code)
            return str(result)
        except Exception:
            try:
                # Fallback to handling multi-line code
                locals_dict = {}
                exec(code, {}, locals_dict)
                return str(locals_dict)
            except Exception as e:
                return f"Error executing code: {repr(e)}"
