import traceback

import torch
from transformers import AutoModel, AutoTokenizer

from app_logger import AppLogger
from utils import HF_TOKEN


class ModelManager:
    def __init__(self, model_name, device_map="auto", trust_remote_code=True, use_fp8=False, logger=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger if logger else AppLogger()
        self.model_name = model_name
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_fp8 = use_fp8
        self.tokenizer = None
        self.model = None
        self.name = self.__class__.__name__

    def initialize(self):
        """Initialize the Vision-LLM model and tokenizer."""
        try:
            attn_implementation = self.get_attention()
            torch_dtype = self.get_precision()
            self.model = AutoModel.from_pretrained(
                self.model_name,
                token=HF_TOKEN,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                use_flash_attn=True,
                trust_remote_code=self.trust_remote_code,
            ).eval().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=HF_TOKEN,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
                use_fast=True)
        except Exception as e:
            self.logger.error(f"[{self.name}] Error loading model '{self.model_name}': {e}, {traceback.format_exc()}")
            raise RuntimeError(f"[{self.name}] Failed to load model. Ensure the model exists and is properly configured.")

    def get_attention(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        attn_implementation = None
        self.logger.debug(f"[{self.name}][TORCH] Torch Version: {torch.__version__}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # device_props = torch.cuda.get_device_properties('cuda:0')  # For first CUDA device
            device_props = torch.cuda.get_device_properties('cuda:0')  # For first CUDA device
            # self.logger.debug(f"Device props: {device_props}")
            self.logger.debug(f"[{self.name}][CUDA] Device Name: {device_props.name}")
            self.logger.debug(f"[{self.name}][CUDA] Total Memory: {device_props.total_memory / 1024 ** 3:.2f} GB")
            self.logger.debug(f"[{self.name}][CUDA] Compute Capability: {device_props.major}.{device_props.minor}")
            self.logger.debug(f"[{self.name}][CUDA] Multiprocessors: {device_props.multi_processor_count}")
            self.logger.debug(f"[{self.name}][CUDA] Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")
            attn_implementation = "flash_attention_2"
        return attn_implementation

    def get_precision(self):
        # Determine precision mode
        if self.use_fp8 and torch.cuda.is_available() and hasattr(torch.cuda, "is_fp8_supported"):
            torch_dtype = torch.float8
            self.logger.info(f"[{self.name}] Using float8 (FP8) precision for model inference.")
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
            self.logger.info(f"[{self.name}] Using float16 (FP16) precision for model inference.")
        else:
            torch_dtype = torch.float32
            self.logger.info(f"[{self.name}] Using float32 (FP32) precision on CPU.")
        return torch_dtype

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
