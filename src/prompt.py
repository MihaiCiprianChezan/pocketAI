from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class Prompt:
    message: str  # Plain text message for the prompt
    lang: Optional[str] = "en"
    images: Optional[List[str]] = None  # List of image URLs or paths
    video: Optional[str] = None  # Video URL or path
    video_prefix: Optional[str] = None  # Video-specific prefix or metadata
    pixel_values: Optional[Any] = None  # Precomputed pixel values (e.g., tensors, embeddings)
    num_patches_list: Optional[List[int]] = None  # Number of patches/segments
    intent: Optional[str] = None

    def update_message(self, new_message: str):
        """
        Updates the message of the prompt but retains existing precomputed properties.
        """
        self.message = new_message

    def clean(self):
        """
        Resets all properties to their default states.
        """
        self.lang = "en"
        self.images = None
        self.video = None
        self.video_prefix = None
        self.pixel_values = None
        self.num_patches_list = None
        self.message = ""