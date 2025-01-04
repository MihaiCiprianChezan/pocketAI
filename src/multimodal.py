from decord import cpu, VideoReader
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


def extract_patches(video_tensor, patch_size=16):
    """
    Split frames into non-overlapping patches.

    Args:
        video_tensor (torch.Tensor): Shape (N, 3, H, W) - batch of video frames.
        patch_size (int): Size of each patch (default=16 for ViT).

    Returns:
        torch.Tensor: Flattened patches of shape (N * num_patches, C * patch_size * patch_size)
        list: List of patch counts for each frame.
    """
    batch_size, channels, height, width = video_tensor.shape
    # Sanity check for valid dimensions
    assert height % patch_size == 0 and width % patch_size == 0, f"Height and width ({height}x{width}) must be divisible by patch_size ({patch_size})"

    # Number of patches along each dimension (e.g., 14 for 224/16)
    num_patches_per_frame = (height // patch_size) * (width // patch_size)  # 14x14 = 196
    patches = video_tensor.unfold(2, patch_size, patch_size)  # Slide vertically
    patches = patches.unfold(3, patch_size, patch_size)  # Then horizontally

    # Rearrange patches: (N, C, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (N, num_patches_h, num_patches_w, C, patch_size, patch_size)

    # Flatten patches: (N, num_patches_h * num_patches_w, C, patch_size, patch_size)
    patches = patches.flatten(1, 2)  # Combine patch dimensions into a single one
    patches = patches.flatten(2)  # Flatten each patch size

    # Reshape to (N * num_patches, patch_dim)
    return patches.view(-1, patches.size(-1)), [num_patches_per_frame] * batch_size  # Flatten all frames into (N_patches, patch_dim)


def build_transform(self, input_size=224):
    """
    Build a transformation pipeline for frames.

    Args:
        input_size (int): The target size (used for square resizing).

    Returns:
        torchvision.transforms: Transformation pipeline.
    """
    return Compose([
        Resize((input_size, input_size)),  # Resize image to 224x224
        ToTensor(),  # Convert image to PyTorch tensor (HWC -> CHW, scale to [0, 1])
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])


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
            torch_dtype=torch.float16,
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

    def preprocess_video(self, video_path, num_frames=8, patch_size=16):
        """
           Preprocess a video file for model input.

           Args:
               video_path (str): Path to the video file.
               num_frames (int): Number of frames to extract.
               patch_size (int): Patch size for splitting frames.

           Returns:
               tuple: (Raw video tensor, num_patches_list)
           """
        try:
            # Extract video frames
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            idx = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = vr.get_batch(idx).asnumpy()

            # Transform each frame to tensor with normalization
            transform = self.build_transform(input_size=224)
            video_tensor = torch.stack([
                transform(Image.fromarray(frame)) for frame in frames
            ]).to(self.device)

            print(f"[DEBUG] Extracted {video_tensor.shape[0]} frames with shape {video_tensor.shape}")

            # Calculate `num_patches_list` based on the number of frames and patches per frame
            height, width = video_tensor.shape[2], video_tensor.shape[3]
            num_patches_per_frame = (height // patch_size) * (width // patch_size)
            num_patches_list = [num_patches_per_frame] * video_tensor.shape[0]

            print(f"[DEBUG] Num patches list: {num_patches_list}")
            # NOTE: Return raw video tensor, not the patches
            return video_tensor, num_patches_list

        except Exception as e:
            print(f"[ERROR] Video preprocessing failed: {e}")
            raise

    def preprocess_image(self, image_path, input_size=448):
        """
        Preprocess a single image.

        Args:
            image_path (str): Path to the image file.
            input_size (int): Final size to resize the image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        try:
            transform = self.build_transform(input_size)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(self.device, dtype=torch.float16)
            return image_tensor
        except Exception as e:
            print(f"[ERROR] Image preprocessing failed: {e}")
            raise

    def ask_text_question(self, question, generation_config=None):
        """
        Perform text question-and-answer interaction with the model.

        Args:
            question (str): Input question.
            generation_config (dict): Optional generation configuration dictionary.

        Returns:
            str: Model response to the text input.
        """
        if generation_config is None:
            generation_config = {"max_new_tokens": 128}

        try:
            response = self.model.chat(self.tokenizer, None, question.strip(), generation_config)
            return response
        except Exception as e:
            print(f"[ERROR]: Text question processing failed: {e}")
            raise

    def ask_image_question(self, image_tensor, question, generation_config=None):
        """
        Perform question-and-answer interaction with an image.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor.
            question (str): Input question.

        Returns:
            str: Model response to the image-based question.
        """
        if generation_config is None:
            generation_config = {"max_new_tokens": 128}

        try:
            response = self.model.chat(self.tokenizer, image_tensor, question.strip(), generation_config)
            return response
        except Exception as e:
            print(f"[ERROR]: Image question processing failed: {e}")
            raise

    def ask_video_question(self, raw_video_tensor, num_patches_list, question, generation_config=None):
        """
           Ask a video-based question and receive a response from the model.

           Args:
               raw_video_tensor (torch.Tensor): Raw video tensor with shape (N, 3, H, W).
               num_patches_list (list): List of number of patches per frame.
               question (str): User question.
               generation_config (dict): Generation configuration.

           Returns:
               str: Model response.
           """
        if generation_config is None:
            generation_config = {"max_new_tokens": 128}

        try:
            print(f"[DEBUG] Raw video tensor shape: {raw_video_tensor.shape}")
            print(f"[DEBUG] Num patches list: {num_patches_list}")
            print(f"[DEBUG] Num patches sum: {sum(num_patches_list)}")

            # Pass the raw video tensor to the model
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=raw_video_tensor,  # Pass original frames
                question=question.strip(),
                generation_config=generation_config,
                history=[],
                num_patches_list=num_patches_list,
                verbose=False
            )

            print(f"[DEBUG] Model response: {response}")
            return response

        except Exception as e:
            print(f"[ERROR] Video question processing failed: {e}")
            raise


if __name__ == "__main__":
    model_path = "OpenGVLab/InternVL2_5-2B"
    internvl_model = InternVLModel(model_path=model_path, device="cuda")

    # Example 1: Text question
    text_question = "What is the nature of language models?"
    print(f"User: {text_question}")
    print(f"Assistant: {internvl_model.ask_text_question(text_question)}")

    # Example 2: Image-based question
    image_path = "./image.jpg"
    image_tensor = internvl_model.preprocess_image(image_path)
    image_question = "What is in the image?"
    print(f"User: {image_question}")
    print(f"Assistant: {internvl_model.ask_image_question(image_tensor, image_question)}")

    # Example 3: Video-based question
    video_path = "./red-panda.mp4"
    raw_video_tensor, num_patches_list = internvl_model.preprocess_video(video_path)
    video_question = "What is the red panda doing?"
    print(f"User: {video_question}")
    print(f"Assistant: {internvl_model.ask_video_question(raw_video_tensor, num_patches_list, video_question)}")

    # Example 4: Describe entire video
    multi_frame_question = "Describe the entire video in detail."
    multi_frame_response = internvl_model.ask_video_question(raw_video_tensor, num_patches_list, multi_frame_question)
    print(f"User: {multi_frame_question}\nAssistant: {multi_frame_response}")
