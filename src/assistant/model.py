import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app_logger import AppLogger
from utils import HF_TOKEN, INTERNLM__2_5__1_8_B_CHAT


class ModelManager:
    def __init__(self, model_name=INTERNLM__2_5__1_8_B_CHAT, device_map="auto", trust_remote_code=True, use_fp8=False):
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
                # local_files_only=True,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation
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
