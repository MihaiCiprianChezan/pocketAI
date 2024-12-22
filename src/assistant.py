from datetime import datetime
from threading import Thread, Lock
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.agents import Tool, ReactJsonAgent, HfApiEngine
from transformers import TextIteratorStreamer
from app_logger import AppLogger
from datetime import datetime




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
        from pytz import timezone
        current_timestamp = f"The current time and date is {datetime.now().strftime('%I:%M %p, %A %B %d %Y')}."
        AppLogger().info(f"[DateTimeTool] Invoked - Generated timestamp: {current_timestamp}")
        return current_timestamp

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
        )
    }
    MODEL = "THUDM/glm-edge-1.5b-chat"
    # MODEL = "THUDM/glm-4-9b-chat"
    MODEL = "THUDM/GLM-Edge-4B-Chat"

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
            DateTimeTool()  # Add the new DateTimeTool here
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
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=self.TOKEN)
            self.logger.debug(f"[CHAT_ASSISTANT] Loading model to device: {self.device}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                revision="main"
            ).to(self.device)
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
            "datetime-tool": ["calendar date", "what time is it", "what time is now", "time now", "current time", "today's date", "date is today", "date now", "current date", "what date is now", "what's the date"]
            # More tools to be added here in future
        }

        # Match query with tool intents
        for tool_name, intents in tool_mapping.items():
            if any(intent in query.lower() for intent in intents):
                # Find and invoke the matched tool
                for tool in self.tools:
                    if tool.name == tool_name:
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

    def get_response(self, history, message, return_iter=True, lang="en", context_free=False):
        """
        Generates a response based on the conversation history or dispatches a tool-based response dynamically.
        If no tool matches, the query is processed by the LLM as a fallback.
        """
        if lang:
            pass  # TODO: Handle multilingual prompts in the future

        history = history if not context_free else [[message, ""]]
        if not context_free:
            history.append([message, ""])

        # Check for potential tool matches
        tool_response = self.tools_dispatcher(message)
        if tool_response:
            return [tool_response]  # Return tool response directly

        # Otherwise, fallback to the LLM for open-ended query handling
        def response_iterator():
            generated_response = ""
            for updated_history in self.predict(history):
                generated_response = updated_history[-1][1]
                yield generated_response

        return response_iterator() if return_iter else "".join(response_iterator())


