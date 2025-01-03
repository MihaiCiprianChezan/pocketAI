import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from decord import VideoReader, cpu
import torch.nn.functional as F

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLModel:
    def __init__(self, model_path, device="cuda"):
        """
        Initialize the InternVLModel with specified model path and device.

        Args:
            model_path (str): Path or identifier to the pretrained HuggingFace model.
            device (str): Device to load model onto ('cuda' or 'cpu').
        """
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def build_transform(self, input_size=448):
        """
        Build preprocessing transforms for images or single video frames.

        Args:
            input_size (int): Final size to resize input.

        Returns:
            torchvision.transforms.Compose: Transform pipeline.
        """
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def preprocess_video(self, video_path, num_frames=8):
        """
        Preprocess a video file for the model.

        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of evenly spaced frames to extract.

        Returns:
            tuple: Preprocessed video tensor and number of patches.
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = vr.get_batch(frame_indices).asnumpy()

            print(f"[DEBUG] Extracted {frames.shape[0]} frames with shape: {frames.shape}.")

            transform = self.build_transform(input_size=224)
            video_tensor = torch.stack([transform(Image.fromarray(frame)) for frame in frames])
            video_tensor = video_tensor.to(self.device, dtype=torch.bfloat16)

            num_patches = video_tensor.shape[-2] // 16 * video_tensor.shape[-1] // 16
            return video_tensor, [num_patches] * num_frames
        except Exception as e:
            print(f"[ERROR] Video preprocessing failed: {e}")
            raise

    def preprocess_image(self, image_path, input_size=448):
        """
        Preprocess an image for the model.

        Args:
            image_path (str): Path to the image file.
            input_size (int): Target size for resized image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        try:
            transform = self.build_transform(input_size)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
            return image_tensor
        except Exception as e:
            print(f"[ERROR] Image preprocessing failed: {e}")
            raise

    def ask_text_question(self, question, generation_config=None):
        """
        Generate a response for a text-based question.

        Args:
            question (str): Input question.
            generation_config (dict): Optional HuggingFace generation configuration.

        Returns:
            str: Model-generated response.
        """
        if generation_config is None:
            generation_config = {"max_new_tokens": 128}

        try:
            response = self.model.chat(
                self.tokenizer, None, question.strip(), generation_config
            )
            return response
        except Exception as e:
            print(f"[ERROR] Text question processing failed: {e}")
            raise

    def ask_image_question(self, image_tensor, question, generation_config=None):
        """
        Generate a response for an image-based question.

        Args:
            image_tensor (torch.Tensor): Preprocessed image.
            question (str): Input question.

        Returns:
            str: Model-generated response.
        """
        if generation_config is None:
            generation_config = {"max_new_tokens": 128}

        try:
            response = self.model.chat(
                self.tokenizer, image_tensor, question.strip(), generation_config
            )
            return response
        except Exception as e:
            print(f"[ERROR] Image question processing failed: {e}")
            raise

    def ask_video_question(self, video_tensor, num_patches_list, question, generation_config=None):
        """
        Generate a response for a video-based question.

        Args:
            video_tensor (torch.Tensor): Preprocessed video tensor.
            num_patches_list (list): Patch counts for each frame.
            question (str): Input question.

        Returns:
            str: Model-generated response.
        """
        if generation_config is None:
            generation_config = {"max_new_tokens": 128}

        try:
            response, _ = self.model.chat(
                self.tokenizer, video_tensor, question.strip(), generation_config, num_patches_list
            )
            return response
        except Exception as e:
            print(f"[ERROR] Video question processing failed: {e}")
            raise


if __name__ == "__main__":
    model_path = "OpenGVLab/InternVL2_5-2B"
    internvl_model = InternVLModel(model_path=model_path, device="cuda")

    # Example 1: Pure text QA
    text_question = "What is the nature of language models?"
    print(f"User: {text_question}")
    print(f"Assistant: {internvl_model.ask_text_question(text_question)}")

    # Example 2: Image-based QA
    image_path = "./image.jpg"
    image_tensor = internvl_model.preprocess_image(image_path)
    image_question = "What is in the image?"
    print(f"User: {image_question}")
    print(f"Assistant: {internvl_model.ask_image_question(image_tensor, image_question)}")

    # Example 3: Video QA
    video_path = "./red-panda.mp4"
    video_tensor, num_patches = internvl_model.preprocess_video(video_path)
    video_question = "What is the red panda doing?"
    print(f"User: {video_question}")
    print(f"Assistant: {internvl_model.ask_video_question(video_tensor, num_patches, video_question)}")

    # Example 4: Describe entire video
    multi_frame_question = "Describe the entire video in detail."
    multi_frame_response = internvl_model.ask_video_question(video_tensor, num_patches_list, multi_frame_question)
    print(f"User: {multi_frame_question}\nAssistant: {multi_frame_response}")