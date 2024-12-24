from itertools import chain
from threading import Lock, Thread
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.agents import HfApiEngine, ReactJsonAgent

from app_logger import AppLogger
from assistant.intent import GENERIC_INTENTS, Intent
from assistant.tools import DateTimeTool, PythonExecutionTool, WikipediaTool
from utils import MODELS_DIR

LLM_MODEL = str(MODELS_DIR / "THUDM-glm-edge-1.5b-chat")  # THUDM/glm-edge-1.5b-chat


# HF_TOKEN = "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"

class ChatAssistant:
    BASE_SYSTEM_PROMPT = ("You are Opti, an friendly AI assistant equipped with tools to handle specialized queries efficiently."
                          "Provide direct concise, natural responses, ensuring clarity and seamless assistance."
                          "Use one line plain text output WITHOUT any formatting, prefixes, suffixes, markup or decorations.")

    def __init__(self, model_name=LLM_MODEL, device_map="auto", trust_remote_code=True):
        self.history_lock = Lock()
        self.logger = AppLogger()

        self.logger.info("[CHAT_ASSISTANT] Initialization started.")
        self.trust_remote_code = trust_remote_code
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define custom tools
        self.tools = [
            PythonExecutionTool(),
            DateTimeTool(),
            WikipediaTool(),
        ]
        # Use all tools intents as labels for intent predictions
        self.intent = Intent(labels=list(chain.from_iterable(tool.intents for tool in self.tools)) + GENERIC_INTENTS)
        # Initialize LLM Engine and Agent
        self.llm_engine = HfApiEngine(model=model_name)  # Use HfApiEngine
        self.agent = ReactJsonAgent(
            tools=self.tools,
            llm_engine=self.llm_engine,
        )
        # Load and optimize a local model for causal LM if necessary
        self.tokenizer, self.model = self.initialize_model(model_name, device_map, trust_remote_code)
        self.prewarm_model()

    def initialize_model(self, model_name, device_map="auto", trust_remote_code=True, use_fp8=False):
        """Loads and configures the tokenizer and the model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                # use_auth_token=HF_TOKEN,
                trust_remote_code=trust_remote_code,
                use_fast=True,
                legacy=False)
            self.logger.debug(f"[CHAT_ASSISTANT] Loading model {model_name} to device: {self.device}")

            # Determine precision (torch.float32, torch.float16, or torch.float8)
            if use_fp8 and torch.cuda.is_available() and hasattr(torch.cuda, "is_fp8_supported") and torch.cuda.is_fp8_supported():
                torch_dtype = torch.float8  # Use FP8 if supported
                self.logger.info("[CHAT_ASSISTANT] Using float8 (FP8) precision for model inference.")
            elif torch.cuda.is_available():
                torch_dtype = torch.float16  # Otherwise, fall back to FP16 if GPU is available
                self.logger.info("[CHAT_ASSISTANT] Using float16 (FP16) precision for model inference.")
            else:
                torch_dtype = torch.float32  # Default to FP32 on CPU
                self.logger.info("[CHAT_ASSISTANT] Using float32 (FP32) precision on CPU.")

            model = (AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                revision="main",
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
        """Pre-warms model for CUDA kernel initialization, skip if model not initialized."""
        if self.model and self.tokenizer:
            try:
                dummy_input = self.tokenizer(
                    "Warm-up round!", padding=True, return_tensors="pt"
                ).to(self.device)
                # Trigger CUDA kernel warming
                self.model.generate(**dummy_input)
            except Exception as e:
                self.logger.debug(f"[CHAT_ASSISTANT] Prewarm error: {e}")
        else:
            self.logger.info("[CHAT_ASSISTANT] Prewarm skipped; model not loaded.")

    @staticmethod
    def preprocess_messages(history):
        """Converts history (dict-based) into the model's expected input format."""
        chat_messages = []
        for message in history:
            chat_messages.append(message)
        chat_messages.append({"role": "system", "content": ChatAssistant.BASE_SYSTEM_PROMPT})
        return chat_messages

    def get_tool_for_intent(self, intent):
        for tool in self.tools:
            if intent.label in tool.intents:  # Check if intent exists in the tool's intents list
                return tool
        return None  # No tool for the detected intent, will use internal training

    def tools_dispatcher(self, query: str) -> (list, str or None):
        """
        Handles query routing to the relevant tool. Returns response messages as a list of dicts.
        """
        # Get user intent from the message
        result = []
        target = None
        message_intent = self.intent.detect(query)
        self.logger.debug(f"[CHAT_ASSISTANT] Detected message intent `{message_intent}` ")
        # Get the appropriate tool for the detected intent (if any of the available tools is matching)
        tool = self.get_tool_for_intent(message_intent)
        if tool:
            self.logger.debug(f"[CHAT_ASSISTANT] The appropriate tool for `{message_intent}` intent is:  {tool.name}")
            # Get knowledge from tool
            tool_result = tool.forward(query)
            if tool_result:
                target = tool.target
                self.logger.debug(f"[CHAT_ASSISTANT] Tool [{tool.name}] result: `{tool_result}` for query: `{query}`")
                result = [
                    {"role": f"{tool.target}", "content": tool_result},
                ]
            else:
                self.logger.debug(f"[CHAT_ASSISTANT] No result from the tool [{tool.name}] found for query: `{query}`")
                # Will use internal training data knowledge
        else:
            self.logger.debug(f"[CHAT_ASSISTANT] No tool found for message intent `{message_intent}`")
        return result, target

    @staticmethod
    def validate_history(history):
        """
        Ensures that 'history' is a list of dictionaries with 'role' and 'content' keys.
        Raises an error if the structure is invalid.
        """
        if not isinstance(history, list):
            raise ValueError("History must be a list of dictionaries.")
        for entry in history:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid history entry: {entry}. Expected a dictionary.")
            if "role" not in entry or "content" not in entry:
                raise ValueError(f"History entry missing keys 'role' or 'content': {entry}")

    def predict(self, history, max_length=150, top_p=0.8, temperature=0.7):
        """
        Predicts the assistant's response based on the chat history.
        Ensures valid response generation and verifies consistent history structure.
        """
        # Validate history structure before proceeding
        self.validate_history(history)
        # Preprocess chat messages for the model
        chat_messages = self.preprocess_messages(history)
        # Model input preparation
        self.logger.debug(f"[CHAT_ASSISTANT] <<< !!! >>> <chat_messages> is: {chat_messages}")
        model_inputs = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.device)
        self.logger.debug(f"[CHAT_ASSISTANT] <<< !!! >>> <model_inputs> is: {model_inputs}")
        # Set up TextIteratorStreamer for streaming tokens
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
        # Initialize cumulative response for appending tokens
        cumulative_response = ""

        # Launch the generation in a thread to avoid blocking
        def generate_in_thread():
            try:
                self.model.generate(**generate_kwargs)
            except Exception as e:
                self.logger.debug(f"[CHAT_ASSISTANT] Error during generation: {e}")

        t = Thread(target=generate_in_thread)
        t.start()
        # Read tokens from the streamer and build the response incrementally
        response_parts = []
        for new_token in streamer:
            if new_token:
                with self.history_lock:
                    response_parts.append(new_token)
                    cumulative_response = "".join(response_parts)  # Accumulate tokens
                    history[-1]["content"] = cumulative_response  # Update assistant's content in history
                # Validate history after each token for correctness
                self.validate_history(history)
                yield history
        # Ensure the thread finishes cleanly
        t.join()
        # Final validation of history to ensure consistency
        self.validate_history(history)

    @staticmethod
    def format_history_for_output(history_segment):
        """
        Formats a segment of the history (list of dicts) into clean, readable plain text.
        """
        output_lines = []
        for message in history_segment:
            role = message["role"].capitalize()
            content = message["content"]
            output_lines.append(f"{role}: {content}")
        return "\n".join(output_lines)

    @staticmethod
    def format_tool_response(tool_response):
        """
        Formats tool-based response (list of dicts) into clean, human-readable text.
        """
        formatted_lines = []
        for entry in tool_response:
            role = entry["role"].capitalize()
            content = entry["content"]
            formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)

    def get_response(self, history, message, return_iter=True, lang="en", context_free=False, max_history_length=5):
        """
        Handles user queries by focusing on the latest prompt first.
        """
        self.logger.debug(f"[CHAT_ASSISTANT] <<< !!! >>> History before: {history}")

        history = history[-max_history_length:]

        # Check for detected tools or actions
        tool_response, tool_target = self.tools_dispatcher(message)
        if tool_response:
            if tool_target == 'direct':
                return tool_response[0]["content"]

            history = [entry for entry in tool_response]
            self.logger.debug(f"[CHAT_ASSISTANT] Tool response handled with context_free={context_free}.")
        else:
            if not isinstance(message, dict):
                message = {"role": "user", "content": str(message)}
                history.append(message)

        self.logger.debug(f"[CHAT_ASSISTANT] <<< !!! >>> History is: {history}")

        # For normal prompts, use prediction (model response)
        def response_iterator():
            """
            Handles history or context-free LLM invocation based on `context_free`.
            """
            # Decide what to send to the LLM
            # active_history = [history[-1]] if context_free else history  # Only use the latest message if context-free
            active_history = history
            self.logger.debug(f"[CHAT_ASSISTANT] <<< !!! >>> Sending to LLM: {self.format_history_for_output(active_history)}")
            for updated_history in self.predict(active_history):
                # Extract LLM output from predictions
                assistant_content = updated_history[-1]["content"] if "content" in updated_history[-1] else "<NO_RESPONSE>"
                yield assistant_content

        return response_iterator() if return_iter else "<NOT_UNDERSTANDABLE!>"
