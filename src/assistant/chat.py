from threading import Thread

from decord import cpu, VideoReader
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import TextIteratorStreamer
from app_logger import AppLogger


class ChatManager:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Mean pixel values for the RGB (Red, Green, Blue) channels across the ImageNet dataset
    IMAGENET_STD = (0.229, 0.224, 0.225)  # Standard deviations for the RGB channels across the ImageNet dataset
    MAX_NEW_TOKENS = 1024  # Default max tokens
    SAMPLING_GEN = True  # True:sampling-based method (randomly conversational), False:greedy deterministic method (precise and predictable)

    def __init__(self, model_mng, history_mng, logger=None):
        self.logger = logger if logger else AppLogger()
        self.model_mng = model_mng
        self.history_mng = history_mng
        self.name = self.__class__.__name__

    def get_stream_generation_config(self, streamer:TextIteratorStreamer=None, max_new_tokens=MAX_NEW_TOKENS, do_sample=False):
        if streamer is None:
            streamer = TextIteratorStreamer(self.model_mng.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        return streamer, dict(streamer=streamer, max_new_tokens=max_new_tokens, do_sample=do_sample)

    @staticmethod
    def get_generation_config(max_new_tokens=MAX_NEW_TOKENS, do_sample=True):
        return dict(max_new_tokens=max_new_tokens, do_sample=do_sample)

    def build_transform(self, input_size):
        """Build the image transformation pipeline."""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    @staticmethod
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

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Preprocess images into tiles dynamically."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = sorted(
            {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num},
            key=lambda x: x[0] * x[1]
        )
        target_aspect_ratio = self.find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
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

    def load_image(self, image_file, input_size=448, max_num=12):
        """Load and preprocess an image."""
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(img) for img in images])
        return pixel_values

    @staticmethod
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

    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        """Load, preprocess, and tile video frames."""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = vr.get_avg_fps()
        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size)
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            tiles = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values_list.append(torch.stack([transform(tile) for tile in tiles]))
            num_patches_list.append(len(tiles))
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def chat_with_wait(self, question, pixel_values=None, generation_config=None, num_patches_list=None, return_full_history=False):
        """
        Handles a chat interaction using the provided question, optional visual inputs, and configurations.

        Args:
            question (str): The textual input question to the chat model.
            pixel_values (Optional[Any]): Optional visual input associated with the question. Defaults to None.
            generation_config (Optional[Any]): Configuration for chat generation parameters. Defaults to None.
            num_patches_list (Optional[List[int]]): Optional list of patches or segments for input processing. Defaults to None.
            return_full_history (bool): If True, returns the full conversation history from the model memory. Defaults to False.

        Returns:
            Union[str, Tuple[str, Any]]: The generated response. Includes full history if `return_full_history` is True.
        """
        if not generation_config:
            generation_config = self.get_generation_config()
        result = self.model_mng.model.chat(
            tokenizer=self.model_mng.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            history=self.history_mng.history,
            num_patches_list=num_patches_list,
            return_history=return_full_history,
        )
        response, history = result if return_full_history else (result, None)
        if return_full_history:
            with self.history_mng.history_lock:
                self.logger.debug(f"[{self.name}] Updating history with: `{history}` ...")
                self.history_mng.history = history
        self.logger.debug(f"[ChatManager] Complete answer: `{response}`")
        return response

    def stream_chat(self, question, pixel_values=None, generation_config=None, num_patches_list=None, return_full_history=False):
        """
            Streams the conversation based on the provided question and parameters using threaded execution.
            """
        streamer, gen_config = self.get_stream_generation_config(max_new_tokens=self.MAX_NEW_TOKENS, do_sample=True)
        if generation_config is None:
            generation_config = gen_config
        shared_result = {"response": None, "history": None}

        def chat_thread_function():
            try:
                result = self.model_mng.model.chat(
                    tokenizer=self.model_mng.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    history=self.history_mng.history,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,
                    return_history=return_full_history,
                )
                if return_full_history:
                    shared_result["response"], shared_result["history"] = result
                else:
                    shared_result["response"] = result
            except Exception as e:
                self.logger.error(f"Chat thread encountered an error: {str(e)}")
                shared_result["response"] = f"Error: {str(e)}"

        thread = Thread(target=chat_thread_function)
        thread.start()
        generated_text = ""
        try:
            if streamer:
                for new_text in streamer:
                    try:
                        # Attempt to fetch from the queue
                        if new_text == self.model_mng.model.conv_template.sep:  # Check for end of stream
                            break
                        generated_text += new_text
                        yield new_text
                    except queue.Empty:
                        self.logger.warning("Timeout: No data available in the streamer queue.")
                        break  # Exit if no new data is available for too long
        except Exception as e:
            self.logger.error(f"Error in streamer loop: {str(e)}")
        finally:
            # Ensure the thread completes before exiting
            thread.join()

        if return_full_history:
            with self.history_mng.history_lock:
                self.logger.debug(f"[{self.name}] Updating history with: `{shared_result['history']}` ...")
                self.history_mng.history = shared_result["history"]
        self.logger.debug(f"[{self.name}] Complete answer: `{generated_text}`")

    def get_image_pixel_values(self, image_paths: list[str]):
        """
        Preprocess images and generate pixel_values in float precision to match the model's operations.
        """
        if len(image_paths) == 1:
            # Single image
            pixel_values = self.load_image(image_paths[0], max_num=12).to(torch.float32)  # Changed to float32
            num_patches_list = None
        else:
            # Multiple images
            p_values_list, num_patches_list = [], []
            for image_path in image_paths:
                p_values = self.load_image(image_path, max_num=12).to(torch.float32)  # Changed to float32
                p_values_list.append(p_values)
                num_patches_list.append(p_values.size(0))
            p_values_tuple = tuple(p_values_list)
            pixel_values = torch.cat(p_values_tuple, dim=0)
        return pixel_values, num_patches_list

        # pixel_values1 = load_image('./image.jpg', max_num=12).to(torch.float16).cuda()
        # pixel_values2 = load_image('./image.jpg', max_num=12).to(torch.float16).cuda()
        # pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        # num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
        #
        # question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
        # response, history = model.chat(tokenizer, pixel_values, question, generation_config,
        #                                num_patches_list=num_patches_list,
        #                                history=None, return_history=True)


    def stream_chat_image(self, question, pixel_values=None, num_patches_list=None, return_full_history=True):
        """ Generates a streaming chat response for a specific question related to an image input.
        You need to load the images data only once with get_image_pixel_values() and then use the data for multiple subsequent queries """
        if pixel_values is not None:
            if isinstance(num_patches_list, list) and len(num_patches_list) > 1:
                image_prefix = ''.join([f"Image-{i + 1}: <image>\n" for i in range(len(num_patches_list))])
            else:
                image_prefix = '<image>\n'
            question = f"{image_prefix}{question}"
        return self.stream_chat(question, pixel_values=pixel_values, return_full_history=return_full_history, num_patches_list=num_patches_list)

    def get_video_pixel_values(self, mp4_video_path: str, num_segments=12, max_num=1):
        """ Load and preprocess video frames to be able to attach the video data in chats """
        pixel_values, num_patches_list = self.load_video(mp4_video_path, num_segments=num_segments, max_num=max_num)
        pixel_values = pixel_values.to(torch.float16)
        video_prefix = ''.join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        return video_prefix, pixel_values, num_patches_list

    def stream_chat_video(self, question: str, pixel_values=None, video_prefix=None, num_patches_list=None, return_full_history=True):
        """ Generates a streaming chat response for a specific question related to a video input.
         You need to load the video data only once with get_video_pixel_values() and then use the data for multiple subsequent queries
         """
        if video_prefix:
            question = f"{video_prefix}{question}"
        return self.stream_chat(question, pixel_values=pixel_values, return_full_history=return_full_history, num_patches_list=num_patches_list)
