# coding=utf-8
# Copyright 2025 OpenGVLab and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image processor class for InternVL3.5."""

from typing import Optional, Union

import numpy as np

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    Unpack,
)
from ...image_utils import ImageInput, SizeDict
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_vision_available,
    logging,
)


if is_vision_available():
    from PIL import Image

    from ...image_utils import PILImageResampling

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class InternVL3_5FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    max_patches (`int`, *optional*):
        Maximum number of patches per image for dynamic preprocessing.
    min_patches (`int`, *optional*):
        Minimum number of patches per image for dynamic preprocessing.
    patch_size (`int`, *optional*):
        Size of each patch when using dynamic preprocessing.
    crop_to_patches (`bool`, *optional*):
        Whether to crop images to patches using dynamic preprocessing.
    """

    max_patches: Optional[int]
    min_patches: Optional[int]
    patch_size: Optional[int]
    crop_to_patches: Optional[bool]


@auto_docstring(custom_intro="InternVL3.5 image processor")
class InternVL3_5ImageProcessor(BaseImageProcessorFast):
    r"""
    Constructs an InternVL3.5 image processor with dynamic multi-patch preprocessing.

    This processor supports dynamic aspect ratio handling by splitting images into multiple patches
    based on their aspect ratio, allowing the model to handle images of various dimensions efficiently.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of the output image after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            Standard deviation to use if normalizing the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        max_patches (`int`, *optional*, defaults to 12):
            Maximum number of patches per image for dynamic preprocessing.
        min_patches (`int`, *optional*, defaults to 1):
            Minimum number of patches per image for dynamic preprocessing.
        patch_size (`int`, *optional*, defaults to 448):
            Size of each patch when using dynamic preprocessing.
        crop_to_patches (`bool`, *optional*, defaults to `True`):
            Whether to crop images to patches using dynamic preprocessing.
    """

    resample = PILImageResampling.BICUBIC
    size = {"height": 448, "width": 448}
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    max_patches = 12
    min_patches = 1
    patch_size = 448
    crop_to_patches = True
    valid_kwargs = InternVL3_5FastImageProcessorKwargs
    model_input_names = ["pixel_values", "num_patches"]

    def __init__(self, **kwargs: Unpack[InternVL3_5FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def find_closest_aspect_ratio(
        self, aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
    ) -> tuple[int, int]:
        """
        Find the closest aspect ratio to the target among the available ratios.

        Args:
            aspect_ratio (`float`): The aspect ratio of the input image.
            target_ratios (`List[Tuple[int, int]]`): List of available aspect ratios.
            width (`int`): Width of the input image.
            height (`int`): Height of the input image.
            image_size (`int`): Size of each patch.

        Returns:
            `Tuple[int, int]`: The closest aspect ratio.
        """
        best_ratio_diff = float("inf")
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

    def dynamic_preprocess(
        self,
        image: Union[np.ndarray, "torch.Tensor", Image.Image],
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = True,
    ) -> list[Union[np.ndarray, "torch.Tensor", Image.Image]]:
        """
        Dynamically preprocess an image into patches based on its aspect ratio.

        Args:
            image: The input image.
            min_num (`int`, *optional*, defaults to 1): Minimum number of patches.
            max_num (`int`, *optional*, defaults to 12): Maximum number of patches.
            image_size (`int`, *optional*, defaults to 448): Size of each patch.
            use_thumbnail (`bool`, *optional*, defaults to True): Whether to add a thumbnail.

        Returns:
            List of processed image patches.
        """
        # Handle different image formats
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size
            is_pil = True
        elif is_torch_available() and isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW
                orig_height, orig_width = image.shape[1:]
            elif image.ndim == 3:  # HWC
                orig_height, orig_width = image.shape[:2]
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
            is_pil = False
        else:  # numpy array
            if image.ndim == 3 and image.shape[-1] in [1, 3]:  # HWC
                orig_height, orig_width = image.shape[:2]
            elif image.ndim == 3:  # CHW
                orig_height, orig_width = image.shape[1:]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
            is_pil = False

        aspect_ratio = orig_width / orig_height

        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image based on format
        if is_pil:
            resized_img = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        elif is_torch_available() and isinstance(image, torch.Tensor):
            import torch.nn.functional as F

            # Ensure CHW format for resize
            if image.ndim == 3 and image.shape[-1] in [1, 3]:  # HWC to CHW
                image = image.permute(2, 0, 1)
            resized_img = F.interpolate(
                image.unsqueeze(0), size=(target_height, target_width), mode="bicubic", align_corners=False
            ).squeeze(0)
        else:  # numpy array
            # Convert to PIL for resizing
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW to HWC
                image_pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    if image.max() <= 1.0
                    else image.transpose(1, 2, 0).astype(np.uint8)
                )
            else:  # Already HWC
                image_pil = Image.fromarray(
                    (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                )

            resized_img = image_pil.resize((target_width, target_height), Image.Resampling.BICUBIC)

        # Split into patches
        processed_images = []
        for i in range(blocks):
            if is_pil:
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size,
                )
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            else:
                # Handle tensor cropping
                row = i // (target_width // image_size)
                col = i % (target_width // image_size)
                start_row = row * image_size
                end_row = (row + 1) * image_size
                start_col = col * image_size
                end_col = (col + 1) * image_size

                if isinstance(resized_img, torch.Tensor):
                    split_img = resized_img[:, start_row:end_row, start_col:end_col]
                else:  # PIL was used for numpy
                    box = (start_col, start_row, end_col, end_row)
                    split_img = resized_img.crop(box)
                processed_images.append(split_img)

        assert len(processed_images) == blocks

        # Add thumbnail if requested and we have multiple patches
        if use_thumbnail and len(processed_images) > 1:
            if is_pil:
                thumbnail_img = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
            elif is_torch_available() and isinstance(image, torch.Tensor):
                import torch.nn.functional as F

                if image.ndim == 3 and image.shape[-1] in [1, 3]:  # HWC to CHW
                    image = image.permute(2, 0, 1)
                thumbnail_img = F.interpolate(
                    image.unsqueeze(0), size=(image_size, image_size), mode="bicubic", align_corners=False
                ).squeeze(0)
            else:
                # Convert to PIL for thumbnail
                if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW to HWC
                    image_pil = Image.fromarray(
                        (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                        if image.max() <= 1.0
                        else image.transpose(1, 2, 0).astype(np.uint8)
                    )
                else:  # Already HWC
                    image_pil = Image.fromarray(
                        (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                    )
                thumbnail_img = image_pil.resize((image_size, image_size), Image.Resampling.BICUBIC)

            processed_images.append(thumbnail_img)

        return processed_images

    def get_number_of_image_patches(self, height: int, width: int, processor_kwargs: dict) -> int:
        """
        Calculate the number of patches for a given image size.

        Args:
            height (`int`): Height of the image.
            width (`int`): Width of the image.
            processor_kwargs (`dict`): Processor arguments.

        Returns:
            `int`: Number of patches.
        """
        if not processor_kwargs.get("crop_to_patches", self.crop_to_patches):
            return 1

        max_patches = processor_kwargs.get("max_patches", self.max_patches)
        min_patches = processor_kwargs.get("min_patches", self.min_patches)
        image_size = processor_kwargs.get("patch_size", self.patch_size)

        aspect_ratio = width / height
        target_ratios = set(
            (i, j)
            for n in range(min_patches, max_patches + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_patches and i * j >= min_patches
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)

        patches = target_aspect_ratio[0] * target_aspect_ratio[1]
        # Add 1 for thumbnail if multiple patches
        if patches > 1:
            patches += 1
        return patches

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[InternVL3_5FastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[Union[np.ndarray, "torch.Tensor", Image.Image]],
        size: SizeDict,
        interpolation: Optional["PILImageResampling"],
        max_patches: int,
        min_patches: int,
        patch_size: int,
        crop_to_patches: bool,
        do_rescale: bool,
        rescale_factor: Optional[float],
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        all_processed_images = []
        all_num_patches = []

        for image in images:
            # Dynamic preprocessing if crop_to_patches is enabled
            if crop_to_patches:
                processed_patches = self.dynamic_preprocess(
                    image, min_num=min_patches, max_num=max_patches, image_size=patch_size, use_thumbnail=True
                )
                all_num_patches.append(len(processed_patches))
            else:
                processed_patches = [image]
                all_num_patches.append(1)

            # Process each patch using base class methods
            batch_patches = []
            for patch in processed_patches:
                # Convert to format expected by base class
                if isinstance(patch, Image.Image):
                    patch_processed = self.convert_to_rgb(patch)
                else:
                    patch_processed = patch

                # Use base class processing methods
                patch_processed = self.resize(patch_processed, size, interpolation=interpolation, antialias=False)
                patch_processed = self.rescale_and_normalize(
                    patch_processed, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                batch_patches.append(patch_processed)

            all_processed_images.extend(batch_patches)

        data = {"pixel_values": all_processed_images, "num_patches": all_num_patches}
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["InternVL3_5ImageProcessor"]
