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
        """Returns the current date and time with timezone as a string."""

        from pytz import timezone
        current_timestamp = f"The current time and date is {datetime.now().strftime('%I:%M %p, %A %B %d %Y')}."
        AppLogger().info(f"[DateTimeTool] Generated timestamp: {current_timestamp}")
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
            "You are Opti, a friendly AI assistant. "
            "You have access to tools such as 'datetime-tool'. "
            "For any time-related queries, use 'datetime-tool' exclusively. "
            # Modified to allow broader time-query handling
            "You should handle all time-related queries dynamically by detecting related intents."
            "Enhance understanding of user prompts like 'what time is it' or 'tell me the date and time.'"
            "You may directly call tools like 'datetime-tool' for accurate timestamps."
            "Do NOT salute the user; respond with time directly without any prefix or suffix. "
            "Use direct, concise, real-time timestamp for responses."
            "Provide concise, natural responses. "
            "Limit your responses to less than 256 characters when possible. "
            "Output only plain text (NO lists, NO tables, NO backticks and NO markdown), in a single line, "
            "without any formatting or additional comments."
        )
    }
    MODEL = "THUDM/glm-edge-1.5b-chat"
    # MODEL = "THUDM/glm-edge-4b-chat"
    # MODEL = "HuggingFaceTB/SmolLM2-1.7B"

    # MODEL = "rwkv/rwkv-raven-14b"

    # MODEL = "facebook/MobileLLM-1.5B"
    # MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # MODEL = "facebook/opt-2.7b"
    # MODEL = "meta-llama/Llama-2-7b-chat-hf"

    # MODEL = "meta-llama/Llama-2-7b-chat"

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
        Generates a response based on the conversation history or a standalone task-specific prompt.
        """
        if lang:
            pass  # TODO: Handle multilingual prompts in the future

        history = history if not context_free else [[message, ""]]
        if not context_free:
            history.append([message, ""])

        def response_iterator():
            generated_response = ""
            for updated_history in self.predict(history):
                generated_response = updated_history[-1][1]
                yield generated_response

        return response_iterator() if return_iter else "".join(response_iterator())
