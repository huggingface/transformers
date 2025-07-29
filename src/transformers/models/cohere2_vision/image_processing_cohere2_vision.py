import functools
from typing import Union

import numpy as np
from PIL import Image

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ImageInput


class Cohere2VisionImageProcessor(BaseImageProcessor):
    """Handles image processing operations for Cohere2Vision"""

    def __init__(
        self,
        max_patches: int,
        image_mean: Union[list[float], float],
        image_std: Union[list[float], float],
        img_size: int = 512,
        patch_size: int = 16,
        downsample_factor: int = 2,
        start_of_img_token: str = "<|START_OF_IMG|>",
        end_of_img_token: str = "<|END_OF_IMG|>",
        img_patch_token: str = "<|IMG_PATCH|>",
        img_line_break_token: str = "<|IMG_LINE_BREAK|>",
        **kwargs,
    ):
        self.img_size = img_size
        self.patch_size = patch_size * downsample_factor
        self.max_patches = max_patches
        self.image_mean = [image_mean] * 3 if isinstance(image_mean, float) else image_mean
        self.image_std = [image_std] * 3 if isinstance(image_std, float) else image_std
        self.start_of_img_token = start_of_img_token
        self.end_of_img_token = end_of_img_token
        self.img_patch_token = img_patch_token
        self.img_line_break_token = img_line_break_token

    @property
    def resolutions(self):
        return self.get_all_possible_resolutions(self.max_patches)

    def preprocess(self, images: ImageInput, **kwargs):
        """
        preprocess is not supported in Cohere2VisionImageProcessor.

        The text and image processing is interlinked for interleaved inputs as image aspect ratios are needed in order to
        generate the image token sequence. Hence it isn't useful to preprocess a sequence of images independently
        of text preprocessing. Instead, you should use `Cohere2VisionProcessor.apply_chat_template` or
        `Cohere2VisionProcessor.__call__` to process images and text together. Alternatively, you can use
        `process_image` to preprocess and generate token strings for individual images.
        """
        raise NotImplementedError(
            "Cohere2VisionImageProcessor does not support the `preprocess` method. Use "
            "`self.process_image` or Cohere2VisionProcessor.apply_chat_template instead."
        )

    def process_image(self, img: Image.Image) -> tuple[str, list[np.ndarray], tuple[int, int]]:
        """Main entry point for image processing.

        Args:
            img: Input PIL image

        Returns:
            tuple: (token_string, image_patches, image_size)
                - token_string: String representation of image tokens
                - image_patches: List of normalized image patches as numpy arrays
                - image_size: Original image dimensions (width, height)
        """
        img = img.convert("RGB")
        splits = self.scale_to_optimal_aspect_ratio(img, self.resolutions, self.img_size)

        # Generate token string
        text = self.start_of_img_token
        for split in splits:
            text += self.img_tokens_from_size(*split.size)
            text += self.img_line_break_token
        text += self.end_of_img_token

        # Convert to numpy arrays and normalize
        img_splits = [np.array(img) for img in splits]
        return text, img_splits, img.size

    def get_all_possible_resolutions(self, num_crops: int) -> list[tuple[int]]:
        resolutions = []
        for i in range(1, num_crops + 1):
            for j in range(1, num_crops + 1):
                if i * j <= num_crops:
                    resolutions.append((i, j))
        return sorted(resolutions, key=lambda x: x[0] * x[1])

    def scale_to_optimal_aspect_ratio(
        self, img: Image.Image, all_resolution: list[tuple[int]], image_size: int
    ) -> list[Image.Image]:
        h, w = img.height, img.width

        selected_resolution = self.select_tiling(h, w, image_size, all_resolution)
        if selected_resolution[0] * selected_resolution[1] > 1:
            # means we have a few tiles, need to add thumbnail
            thumbnail_image = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        new_h, new_w = (
            selected_resolution[0] * image_size,
            selected_resolution[1] * image_size,
        )
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        # Now crop the image along width first, then height
        crops = [
            img.crop((x, y, x + image_size, y + image_size))
            for y in range(0, new_h, image_size)
            for x in range(0, new_w, image_size)
        ]
        if len(crops) > 1:
            crops.append(thumbnail_image)
        return crops

    def select_tiling(self, h: int, w: int, base_image_size: int, tilings: list[tuple[int]]) -> np.ndarray:
        """Divide in image of size [w, h] in up to max_num_patches of size patch_size"""
        original_size = np.stack([h, w])  # [1, 2]
        candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
        candidate_resolutions = candidate_tilings * base_image_size  # [n_resolutions, 2]
        # How much we would need to scale the image to fit exactly in each tiling
        original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]

        with np.errstate(divide="ignore"):
            required_scale_d = (candidate_resolutions.astype(np.float32) / original_size,)
        required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
        if np.all(required_scale < 1):
            # We are forced to downscale, so try to minimize the amount of downscaling
            ix = np.argmax(required_scale)
        else:
            # Pick the resolution that required the least upscaling so that it most closely fits the image
            required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
            ix = np.argmin(required_scale)
        return candidate_tilings[ix]

    def img_tokens_from_size(self, width: int, height: int) -> str:
        w_patch = width / self.patch_size
        h_patch = height / self.patch_size
        assert w_patch % 1 == 0 and h_patch % 1 == 0, "height and width doesn't match the patch size"
        return self.create_image_str(int(w_patch), int(h_patch))

    def create_image_str(self, w_patch: int, h_patch: int) -> str:
        # Build each line (row) of patches
        lines = []
        for _ in range(h_patch):
            line = "".join(self.img_patch_token for _ in range(w_patch))
            lines.append(line)
        return "".join(lines)

    def normalize_patches(self, patches: list[np.ndarray]) -> list[np.ndarray]:
        """Normalize image patches to the expected range."""
        return [self._normalize(patch / 255) for patch in patches]

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        return (image - self.image_mean) / self.image_std

    @functools.lru_cache(maxsize=1)
    def _possible_aspect_ratios(self):
        min_num, max_num = 1, self.max_patches
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        return sorted(target_ratios, key=lambda x: x[0] * x[1])


__all__ = ["Cohere2VisionImageProcessor"]
