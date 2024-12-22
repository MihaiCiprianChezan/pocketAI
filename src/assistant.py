from threading import Thread, Lock
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.agents import Tool, ReactJsonAgent, HfApiEngine
from transformers import TextIteratorStreamer
from app_logger import AppLogger


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
            "DO NOT salute or greet the user (consider salutations have already been done before). "
            "Use a warm, engaging, live chat tone. "
            "Provide concise, natural responses. "
            "Limit your responses to less than 256 characters when possible. "
            "Output only plain text (NO lists, NO tables, NO backticks and NO markdown), in a single line, "
            "without any formatting or additional comments."
        )
    }

    def __init__(self, model_name="THUDM/glm-edge-1.5b-chat", device_map="auto"):
        self.history_lock = Lock()
        self.logger = AppLogger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define custom tools
        self.tools = [PythonExecutionTool()]

        # Initialize LLM Engine and Agent
        self.llm_engine = HfApiEngine(model=model_name)  # Use HfApiEngine
        self.agent = ReactJsonAgent(
            tools=self.tools,
            llm_engine=self.llm_engine,
        )

        # Load and optimize a local model for causal LM if necessary
        self.tokenizer, self.model = self.initialize_model(model_name, device_map)
        self.prewarm_model()

    def initialize_model(self, model_name, device_map):
        """Loads and configures the tokenizer and the model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger.debug(f"[CHAT_ASSISTANT] Loading model to device: {self.device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.logger.debug("[CHAT_ASSISTANT] Optimizing model for inference...")
        model = torch.compile(model, mode="max-autotune")
        return tokenizer, model

    def prewarm_model(self):
        """Pre-warms the model to initialize CUDA kernels."""
        dummy_input = self.tokenizer("Warm-up round!", return_tensors="pt").to(self.device)
        self.model.generate(**dummy_input)

    @staticmethod
    def preprocess_messages(history):
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
