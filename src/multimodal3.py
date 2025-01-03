from threading import Thread

from decord import cpu, VideoReader
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Transformations
def build_transform(input_size):
    """Build the image transformation pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Image Preprocessing
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Helper function to find the closest aspect ratio for tiling."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Preprocess images into tiles dynamically."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate possible target aspect ratios
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num},
        key=lambda x: x[0] * x[1]
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions and split into tiles
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = [
        resized_img.crop(
            (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
        )
        for i in range(blocks)
    ]

    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    """Load and preprocess an image."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


# Video Preprocessing
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Generate frame indices for video sampling."""
    start, end = bound if bound else (-100000, 100000)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Load, preprocess, and tile video frames."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = vr.get_avg_fps()

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values_list.append(torch.stack([transform(tile) for tile in tiles]))
        num_patches_list.append(len(tiles))
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# Model Loading and Chat
def initialize_model_and_tokenizer(model_path):
    """Initialize the Vision-LLM model and tokenizer."""
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    return model, tokenizer


def chat(model, tokenizer, question, pixel_values=None, generation_config=None, history=None, num_patches_list=None, return_history=True):
    """Generate a response from the Vision-LLM model."""
    result = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        history=history,
        num_patches_list=num_patches_list,
        return_history=return_history)
    response, history = result if return_history else (result, None)
    return response, history


def stream_chat_with_yield(model, tokenizer, question, pixel_values=None, generation_config=None, history=None, num_patches_list=None, return_history=True):
    """
    Function to execute model chat with streaming text output using `yield` for incremental updates.

    Args:
        model: The loaded language-vision model.
        tokenizer: The tokenizer associated with the model.
        pixel_values: Image or video tensor input (optional).
        question: User's query or text input.
        generation_config: Configuration for model generation.
        history: Conversation history (if applicable).
        num_patches_list: List of patches for video inputs (if applicable).
        return_history: Whether to return the conversation history.

    Yields:
        - Each chunk of streamed output as it is generated.
        - After streaming, returns the final conversation history as part of the last chunk.
    """
    if generation_config is None:
        generation_config = {}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
    generation_config['streamer'] = streamer
    generation_config.setdefault("max_new_tokens", 1024)  # Default max tokens if not set
    generation_config.setdefault("do_sample", False)  # Default sampling behavior
    shared_result = {"response": None, "history": None}

    def chat_thread_function():
        result = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            history=history,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
            return_history=return_history
        )
        if return_history:
            shared_result["response"], shared_result["history"] = result
        else:
            shared_result["response"] = result

    # Start the model execution in a thread
    thread = Thread(target=chat_thread_function)
    thread.start()

    for new_text in streamer:  # Stream chunks from the streamer using a generator
        if new_text == model.conv_template.sep:  # Check for end of stream
            break
        yield new_text  # Yield each chunk as it streams in

    # Ensure the thread joins back
    thread.join()

    # After streaming, optionally yield the final response and history
    yield {"response": shared_result["response"], "history": shared_result["history"]}


def stream_chat(model, tokenizer, question, pixel_values=None, generation_config=None, history=None, num_patches_list=None, stream_print=True, return_history=True):
    """
    Function to execute model chat with streaming text output while returning the final response and updated history.

    Args:
        model: The loaded language-vision model.
        tokenizer: The tokenizer associated with the model.
        pixel_values: Image or video tensor input (optional).
        question: User's query or text input.
        generation_config: Configuration for model generation.
        history: Conversation history (if applicable).

    Returns:
        A tuple containing:
        - Generated response as a string (captured during streaming).
        - Updated conversation history.
    """
    if generation_config is None:
        generation_config = {}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
    generation_config['streamer'] = streamer
    generation_config.setdefault("max_new_tokens", 1024)  # Fallback max tokens if not set
    generation_config.setdefault("do_sample", False)  # Fallback sampling behavior
    shared_result = {"response": None, "history": None}

    def chat_thread_function():
        result = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            history=history,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
            return_history=return_history
        )
        if return_history:
            shared_result["response"], shared_result["history"] = result
        else:
            shared_result["response"] = result

    thread = Thread(target=chat_thread_function)
    thread.start()
    generated_text = ''
    if stream_print:
        print("Streaming output:\n", end='', flush=True)
    for new_text in streamer:
        if new_text == model.conv_template.sep:
            break
        generated_text += new_text
        if stream_print:
            print(new_text, end='', flush=True)
    if stream_print:
        print("", flush=True)
    thread.join()
    return generated_text, shared_result["history"]


# Main Execution
if __name__ == "__main__":
    model_path = "OpenGVLab/InternVL2_5-2B"
    model, tokenizer = initialize_model_and_tokenizer(model_path)
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    question = "Hello, what's up?"

    # Call the generator function
    for output in stream_chat_with_yield(model, tokenizer, question=question, return_history=True):
        if isinstance(output, str):
            print(output, end='', flush=True)  # Streaming output
        elif isinstance(output, dict):  # Final response and history after stream
            print("\n\nFinal Response:", output["response"])
            print("Updated History:", output["history"])

    # Text-only conversation
    question = "How are you 100% functional today?"
    # response, history = chat(model, tokenizer, question, generation_config=generation_config)
    # print(f"User: {question}\nAssistant: {response}")

    # Stream the conversation
    response, updated_history = stream_chat(
        model, tokenizer, pixel_values=None, question=question, generation_config=generation_config
    )
    print("\n\nFull conversation history:")
    print(updated_history)

    # Single image conversation
    pixel_values = load_image("./image.jpg", max_num=12).to(torch.bfloat16).cuda()
    # question = "<image>\nPlease describe the image in detail."
    question = "<image>\nPlease briefly describe the image."
    # response, history = chat(model, tokenizer, question, pixel_values, generation_config, history=None)
    # print(f"User: {question}\nAssistant: {response}")

    response, updated_history = stream_chat(
        model, tokenizer, pixel_values=pixel_values, question=question, generation_config=generation_config, history=updated_history
    )

    print(f'User: {question}\nAssistant: {response}')
    # print("\n\nFull conversation history:")
    # print(updated_history)

    # Video conversation
    pixel_values, num_patches_list = load_video("./red-panda.mp4", num_segments=8, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
    question = video_prefix + "What is the red panda doing?"
    # response, history = chat(
    #     model, tokenizer, question, pixel_values, generation_config, history=None, num_patches_list=num_patches_list
    # )
    # print(f"User: {question}\nAssistant: {response}")

    response, updated_history = stream_chat(model, tokenizer, question, pixel_values, generation_config, history=updated_history, num_patches_list=num_patches_list
                                            )
    print(f"User: {question}\nAssistant: {response}")

    question = 'Describe this video in detail.'
    response, updated_history = stream_chat(model, tokenizer, question, pixel_values, generation_config, history=updated_history, num_patches_list=num_patches_list)
    print(f'User: {question}\nAssistant: {response}')

    print("\n\nFull conversation history:")
    print(updated_history)
