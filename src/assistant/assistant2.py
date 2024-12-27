from itertools import chain
from threading import Lock, Thread
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app_logger import AppLogger
from assistant.intent import GENERIC_INTENTS, Intent
from assistant.tools import DateTimeTool, get_tools_summary, PythonExecutionTool, WikipediaTool
from utils import MODELS_DIR

LLM_MODEL = str(MODELS_DIR / "internlm-internlm2_5-1_8b-chat")  # OpenGVLab/InternVL2_5-1B
HF_TOKEN = "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"


# Model Manager: Handles model initialization, optimization, and tokenization
class ModelManager:
    def __init__(self, model_name=LLM_MODEL, device_map="auto", trust_remote_code=True, use_fp8=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = AppLogger()
        self.model_name = model_name
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_fp8 = use_fp8
        self.tokenizer, self.model = self.initialize()

    def initialize(self):
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            attn_implementation = None
            self.logger.debug(f"[TORCH] Torch Version: {torch.__version__}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # device_props = torch.cuda.get_device_properties('cuda:0')  # For first CUDA device
                device_props = torch.cuda.get_device_properties('cuda:0')  # For first CUDA device
                # self.logger.debug(f"Device props: {device_props}")
                self.logger.debug(f"[CUDA] Device Name: {device_props.name}")
                self.logger.debug(f"[CUDA] Total Memory: {device_props.total_memory / 1024 ** 3:.2f} GB")
                self.logger.debug(f"[CUDA] Compute Capability: {device_props.major}.{device_props.minor}")
                self.logger.debug(f"[CUDA] Multiprocessors: {device_props.multi_processor_count}")
                self.logger.debug(f"[CUDA] Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")
                attn_implementation = "flash_attention_2"

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=HF_TOKEN,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
                legacy=False
            )
            self.logger.debug(f"[ModelManager] Loading model {self.model_name} to device: {self.device}")

            # Determine precision mode
            if self.use_fp8 and torch.cuda.is_available() and hasattr(torch.cuda, "is_fp8_supported"):
                torch_dtype = torch.float8
                self.logger.info("[ModelManager] Using float8 (FP8) precision for model inference.")
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
                self.logger.info("[ModelManager] Using float16 (FP16) precision for model inference.")
            else:
                torch_dtype = torch.float32
                self.logger.info("[ModelManager] Using float32 (FP32) precision on CPU.")

            # Load the causal language model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                local_files_only=True,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch_dtype,
                attn_implementation = attn_implementation
            ).to(self.device)

            self.logger.debug("[ModelManager] Optimizing model for inference...")
            model = torch.compile(model, mode="max-autotune")
            return tokenizer, model
        except Exception as e:
            self.logger.error(f"[ModelManager] Error loading model '{self.model_name}': {e}, {traceback.format_exc()}")
            raise RuntimeError("Failed to load model. Ensure the model exists and is properly configured.")

    def warm_model(self):
        """Pre-warms the model for CUDA kernel initialization."""
        if self.model and self.tokenizer:
            try:
                dummy_input = self.tokenizer(
                    "Warm-up round!",
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                self.model.generate(**dummy_input)
            except Exception as e:
                self.logger.error(f"[ModelManager] Pre-warm error: {e}")
        else:
            self.logger.info("[ModelManager] Pre-warming skipped; model not loaded.")


# Tool Manager: Handles tools and their interaction via intent recognition
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


# Chat Manager: Handles chat history and response formatting
class ChatManager:
    BASE_SYSTEM_PROMPT = (
        "You are Opti, a friendly AI assistant equipped with tools to handle specialized queries efficiently."
        " Always respond in clear, plain text using natural, conversational human language."
        " Do NOT use markdown, headers, bullet points, numbered lists, or any special formatting in your responses."
        " Respond in a way that's clear and conversational, just as if you were speaking directly to someone in live chat."
    )

    def __init__(self, model_manager, tool_manager, history_manager):
        self.logger = AppLogger()
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.history_manager = history_manager

    @staticmethod
    def preprocess_messages(history):
        chat_messages = []
        for message in history:
            chat_messages.append(message)
        chat_messages.append({"role": "system", "content": ChatManager.BASE_SYSTEM_PROMPT})
        return chat_messages

    def get_kwargs(self, model_inputs, streamer, max_length=150, top_p=0.8, temperature=0.7):
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": 1.2,
            "eos_token_id": self.model_manager.tokenizer.convert_tokens_to_ids("<|user|>")
        }

    def predict(self, history, max_length=150, top_p=0.8, temperature=0.7):
        self.history_manager.validate(history)
        chat_messages = self.preprocess_messages(history)

        applied_template = self.model_manager.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=self.tool_manager.tools_summary,
        )
        self.logger.debug(f"[ChatManager] Message template: `{applied_template}`")

        model_inputs = self.model_manager.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model_manager.device)
        self.logger.debug(f"[ChatManager] Model inputs: `{model_inputs}`")

        streamer = TextIteratorStreamer(self.model_manager.tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = self.get_kwargs(
            model_inputs,
            streamer,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature)

        t = Thread(target=lambda: self.model_manager.model.generate(**generate_kwargs))
        t.start()
        response_parts = []
        for new_token in streamer:
            response_parts.append(new_token)
            yield "".join(response_parts)
        t.join()
        self.logger.debug(f"[ChatManager] Complete answer: `{response_parts}`")

class HistoryManager:
    def __init__(self, history=None, history_size=1000):
        self.logger = AppLogger()
        self.history_lock = Lock()
        self.history_size = history_size
        self.history = history if history else []
        self.clean_history = []

    @staticmethod
    def validate(history):
        if not isinstance(history, list):
            raise ValueError("History must be a list of dictionaries.")
        for entry in history:
            if not isinstance(entry, dict) or ("role" not in entry or "content" not in entry):
                raise ValueError(f"Invalid history format: {entry}")

    def add(self, message, role="user"):
        with self.history_lock:
            try:
                if len(self.history) == self.history_size:
                    self.history = self.history[:len(self.history)-1]
                self.history.append({"role": role, "content": message})
            except Exception as e:
                self.logger.error(f"[HistoryManager] Error adding message to history: {e}, {traceback.format_exc()}")

    def clean(self):
        with self.history_lock:
            self.history = []
            self.clean_history = []


# Assistant: Combines all components into a unified interface
class Assistant:
    def __init__(self, model_name=LLM_MODEL):
        self.logger = AppLogger()
        self.model_manager = ModelManager(model_name)
        self.tools_manager = ToolManager()
        self.history_manager = HistoryManager()
        self.chat_manager = ChatManager(self.model_manager, self.tools_manager, self.history_manager)
        self.model_manager.warm_model()

    def get_response(self, message, context_free=False, max_length=150, top_p=0.8, temperature=0.7):
        if context_free:
            history = self.history_manager.clean_history
        else:
            history = self.history_manager.history
        self.logger.debug(f"[Assistant] History before: {history}")
        self.logger.debug(f"[Assistant] User query: `{message}`")
        self.history_manager.add(message, role="user")
        self.logger.debug(f"[Assistant] History after: {history}")

        tool_response, tool_target = self.tools_manager.dispatch(message)

        if tool_response:
            return tool_response[0]["content"]
        else:
            return self.chat_manager.predict(history, max_length, top_p, temperature)