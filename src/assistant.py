import traceback
from datetime import datetime
from threading import Thread, Lock
import wikipedia
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from transformers.agents import Tool, ReactJsonAgent, HfApiEngine

from app_logger import AppLogger


class DateTimeTool(Tool):
    name = "datetime-tool"
    description = (
        "Provides the current system time in the format 'HH:MM AM/PM, Day Month Date YYYY'. "
        "Ensure correctly formatted, dynamically generated timestamps for time queries."
        "Output example: 'The current time and date is 08:32 AM, Sunday December 22 2024.'"
    )

    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        """Returns the current date and time."""
        current_timestamp = f"The current time and date is {datetime.now().strftime('%I:%M %p, %A %B %d %Y')}."
        AppLogger().info(f"[DateTimeTool] Invoked - Generated timestamp: {current_timestamp}")
        return current_timestamp

class WikipediaTool(Tool):
    name = "wikipedia-tool"
    description = (
        "Fetches summaries of topics or concepts from Wikipedia. "
        "Use this tool for informational queries requiring general up to date knowledge. "
        "Output should be concise and answer the query directly."
    )

    inputs = {
        "query": {"type": "string", "description": "A topic or concept to search on Wikipedia."}
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        """Fetches a concise Wikipedia summary for the given query."""
        try:
            # Fetch first 2 sentences of the result
            summary = wikipedia.summary(query, sentences=2)
            AppLogger().info(f"[WIKIPEDIA_TOOL] Searched for: {query}, Result: {summary}")
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle ambiguous terms
            return (
                f"The query '{query}' is ambiguous. Here are some suggestions: {', '.join(e.options[:5])}. "
                "Please specify further."
            )
        except wikipedia.exceptions.PageError:
            # Handle cases where no results are found
            return f"No page found on Wikipedia for '{query}'. Please try another query."
        except Exception as e:
            AppLogger().error(f"[WIKIPEDIA_TOOL] Error handling query '{query}': {repr(e)}")
            return f"An unexpected error occurred while searching Wikipedia for '{query}'."



class PythonExecutionTool(Tool):
    name = "python-exec"
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


class ChatAssistant:
    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "You are Opti, a friendly AI assistant with access to specialized tools like 'datetime-tool' for handling time-related queries. You have access to real time information through those tools. "
            "Use 'datetime-tool' exclusively for system-time or date-related queries by detecting related intents. "
            "For all other prompts, respond as a general-purpose assistant using your training to generate appropriate answers. "
            "Provide concise, natural responses . "
            "Provide plain, natural responses with correct formatting, ideally under 256 characters when possible."
            "Use plain text without any formatting, prefixes, or suffixes. "
            "Avoid lists, tables, titles, or additional comments. "
            "Limit responses to direct answers, ensuring clarity and seamless assistance. "
            "Your responses should contain, NO italicized, NO bold, No enclosing in parentheses, NO lists, NO tables, NO backticks and NO markdown, NO titles, NO formatting and NO additional comments."
            "When you don't understand a prompt or it does not make sense simply respond only with: <NOT_UNDERSTANDABLE!>",
        )
    }

    # SYSTEM_PROMPT = {
    #     "role": "system",
    #     "content": (
    #         "You are Opti, a friendly AI assistant with access to specialized tools like 'datetime-tool' for time/date queries and 'wikipedia-tool' for real-time informational queries. "
    #         "You must decide when to use these tools to generate the most accurate and up-to-date responses. "
    #         "If a query relates to the current date or time, use 'datetime-tool'. For general factual or real-world information beyond your training, use 'wikipedia-tool'. "
    #         "Always check if a tool is relevant before crafting a response. "
    #         "If tools do not provide the necessary information, rely on your training and knowledge for an appropriate response. "
    #         "If neither your training nor tools can answer a question, respond with: <NOT_UNDERSTANDABLE!>. "
    #         "Generate concise, natural responses below 256 characters when possible, avoiding unnecessary details or formatting. "
    #         "Respond in plain text. Avoid formatting such as lists, tables, bold text, italicized text, markdown, or enclosing responses in any special symbols. "
    #         "Present short answers for simple questions and summarize answers for longer or more complex queries. "
    #         "Do not claim that your knowledge is outdated or that you cannot look up information if tools are available. Rely on the tools provided to retrieve missing or recent data. "
    #         "Only say 'I do not have access to that information' if both your tools and training are unable to provide accurate information. "
    #         "For illogical or nonsensical queries, respond with: <NOT_UNDERSTANDABLE!>."
    #     )
    # }

    MODEL = "THUDM/glm-edge-1.5b-chat"

    TOKEN = "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"

    def __init__(self, model_name=MODEL, device_map="auto", trust_remote_code=True):
        self.history_lock = Lock()
        self.logger = AppLogger()
        self.logger.info("[CHAT_ASSISTANT] Initialization started.")
        self.trust_remote_code = trust_remote_code
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define custom tools
        self.tools = [
            PythonExecutionTool(),
            DateTimeTool(),
            # WikipediaTool(),
        ]

        # Initialize LLM Engine and Agent
        self.llm_engine = HfApiEngine(model=model_name)  # Use HfApiEngine
        self.agent = ReactJsonAgent(
            tools=self.tools,
            llm_engine=self.llm_engine,
        )

        # Load and optimize a local model for causal LM if necessary
        self.tokenizer, self.model = self.initialize_model(model_name, device_map, trust_remote_code)
        self.prewarm_model()

    def initialize_model(self, model_name, device_map, trust_remote_code):
        """Loads and configures the tokenizer and the model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=self.TOKEN, trust_remote_code=True)
            self.logger.debug(f"[CHAT_ASSISTANT] Loading model to device: {self.device}")

            model = (AutoModelForCausalLM.from_pretrained(
                model_name,
                # device_map=device_map,
                local_files_only=True,
                device_map="balanced",
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                revision="main"
            ).to(self.device))

            self.logger.debug("[CHAT_ASSISTANT] Optimizing model for inference...")
            model = torch.compile(model, mode="max-autotune")

            return tokenizer, model
        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {e}, {traceback.format_exc()}")
            raise RuntimeError(
                f"Failed to load model '{model_name}'. Ensure the model exists and has the required files."
            )

    def prewarm_model(self):
        """Pre-warms the model to initialize CUDA kernels."""
        dummy_input = self.tokenizer("Warm-up round!", padding=True, return_tensors="pt").to(self.device)
        self.model.generate(**dummy_input)

    def preprocess_messages(self, history):
        """Converts history into the model's expected message format."""
        chat_messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                chat_messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                chat_messages.append({"role": "user", "content": user_msg})
            if model_msg:
                chat_messages.append({"role": "assistant", "content": model_msg})
        chat_messages.append(ChatAssistant.SYSTEM_PROMPT)
        return chat_messages

    def tools_dispatcher(self, query: str) -> str:
        """
        Dispatches query to the relevant tool based on the intent.
        Returns the tool's response or a fallback message if no tools match.
        """
        tool_mapping = {
            "datetime-tool": ["calendar date", "what time is it", "what time is now", "time now", "current time", "today's date", "date is today", "date now", "current date",
                              "what date is now", "what's the date"],
            "wikipedia-tool": ["explain", "what is", "meaning of", "search on wikipedia", "find on wikipedia", "who is", "what are"]
            # More tools to be added here in future
        }

        # Match query with tool intents
        for tool_name, intents in tool_mapping.items():
            if any(intent in query.lower() for intent in intents):
                # Find and invoke the matched tool
                for tool in self.tools:
                    if tool.name == tool_name:
                        if tool.name == "wikipedia-tool":
                            self.logger.debug(f"Routing query to [WIKIPEDIA_TOOL] tool: {query}")
                            cleaned_query = query.strip()
                            result = tool.forward(cleaned_query)
                            if result:
                                self.logger.debug(f"[WIKIPEDIA_TOOL] tool <RESULT>: {result}")
                                return result  # Return tool response directly if available
                            else:
                                self.logger.debug("[WIKIPEDIA_TOOL] tool returned <NO_RESULTS>.")
                        return tool.forward()
        # return "I'm sorry, I cannot answer that."
        # Fallback: No matching tools, handle query with default behavior (LLM)
        return None  # Indicate no tool matched

    def predict(self, history, max_length=150, top_p=0.8, temperature=0.7):
        """Generates a prediction based on the conversation history."""
        chat_messages = self.preprocess_messages(history)

        # Format the input for the model
        model_inputs = self.tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True
        ).to(self.device)

        # Define the streamer for generating response output
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": 1.2,
            "eos_token_id": self.tokenizer.convert_tokens_to_ids("<|user|>")
        }

        def generate_in_thread():
            try:
                self.model.generate(**generate_kwargs)
            except Exception as e:
                self.logger.debug(f"[CHAT_ASSISTANT] Error during generation: {e}")

        # Start text generation in a separate thread
        t = Thread(target=generate_in_thread)
        t.start()
        generated_response = ""
        for new_token in streamer:
            if new_token:
                with self.history_lock:
                    history[-1][1] += new_token
                generated_response += new_token  # Append newly generated tokens
            yield history
        t.join()
        return generated_response

    def get_response(self, history, message, return_iter=True, lang="en", context_free=False, max_history_length=1):
        """
        Generates a response while managing the conversation history.
        Automatically limits the history to the most recent interactions (up to max_history_length).
        """
        if lang:
            pass  # TODO: Extend for multilingual support if needed

        # Add the new message to the history
        history = history if not context_free else [[message, ""]]
        if not context_free:
            history.append([message, ""])

        # Limit the history length to the most recent interactions
        history = history[-max_history_length:]  # Retain only the last N interactions

        # Check for tool applicability first
        tool_response = self.tools_dispatcher(message)
        if tool_response:
            return [tool_response]  # Immediate tool resolution

        # Fallback: Generate a response using the assistant's language model
        def response_iterator():
            generated_response = ""
            for updated_history in self.predict(history):
                generated_response = updated_history[-1][1]
                yield generated_response

        fallback_response = "<NOT_UNDERSTANDABLE!>"  # Default fallback for invalid/unmatched queries
        return response_iterator() if return_iter else fallback_response
