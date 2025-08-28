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

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging


if is_vision_available():
    from PIL import Image


logger = logging.get_logger(__name__)


class InternVL3_5ImageProcessor(BaseImageProcessor):
    r"""
    Constructs an InternVL3.5 image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        max_patches (`int`, *optional*, defaults to 12):
            Maximum number of patches per image for dynamic preprocessing.
        patch_size (`int`, *optional*, defaults to 448):
            Size of each patch when using dynamic preprocessing.
        crop_to_patches (`bool`, *optional*, defaults to `True`):
            Whether to crop images to patches using dynamic preprocessing.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        max_patches: int = 12,
        min_patches: int = 1,
        patch_size: int = 448,
        crop_to_patches: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        size = size if size is not None else {"height": 448, "width": 448}
        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb
        self.max_patches = max_patches
        self.min_patches = min_patches
        self.patch_size = patch_size
        self.crop_to_patches = crop_to_patches
        self.valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "max_patches",
            "min_patches",
            "patch_size",
            "crop_to_patches",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    def find_closest_aspect_ratio(
        self, aspect_ratio: float, target_ratios: List[Tuple[int, int]], width: int, height: int, image_size: int
    ) -> Tuple[int, int]:
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
        image: np.ndarray,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ) -> List[np.ndarray]:
        """
        Dynamically preprocess an image into patches based on its aspect ratio.

        Args:
            image (`np.ndarray`): The input image.
            min_num (`int`, *optional*, defaults to 1): Minimum number of patches.
            max_num (`int`, *optional*, defaults to 12): Maximum number of patches.
            image_size (`int`, *optional*, defaults to 448): Size of each patch.
            use_thumbnail (`bool`, *optional*, defaults to False): Whether to add a thumbnail.

        Returns:
            `List[np.ndarray]`: List of processed image patches.
        """
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size
        else:
            # Handle numpy array
            if image.ndim == 3 and image.shape[-1] == 3:  # HWC
                orig_height, orig_width = image.shape[:2]
            elif image.ndim == 3 and image.shape[0] == 3:  # CHW
                orig_height, orig_width = image.shape[1:]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

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

        # Resize the image
        if isinstance(image, Image.Image):
            resized_img = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        else:
            # Convert numpy array to PIL Image for resizing, then back
            if image.ndim == 3 and image.shape[0] == 3:  # CHW to HWC
                image_pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
            else:  # Already HWC
                if image.max() <= 1.0:
                    image_pil = Image.fromarray((image * 255).astype(np.uint8))
                else:
                    image_pil = Image.fromarray(image.astype(np.uint8))
            
            resized_img = image_pil.resize((target_width, target_height), Image.Resampling.BICUBIC)

        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            if isinstance(image, Image.Image):
                thumbnail_img = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
            else:
                # Convert to PIL for thumbnail
                if image.ndim == 3 and image.shape[0] == 3:  # CHW to HWC
                    image_pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                else:  # Already HWC
                    if image.max() <= 1.0:
                        image_pil = Image.fromarray((image * 255).astype(np.uint8))
                    else:
                        image_pil = Image.fromarray(image.astype(np.uint8))
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
        if not processor_kwargs.get("crop_to_patches", True):
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

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, image_size
        )

        return target_aspect_ratio[0] * target_aspect_ratio[1]

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        max_patches: Optional[int] = None,
        min_patches: Optional[int] = None,
        patch_size: Optional[int] = None,
        crop_to_patches: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images. Copy of the superclass method. Here for typing purposes.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single image, a batch of images, or a list of batches of images. 
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The shortest edge of the image is resized to
                `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
                is > `size["longest_edge"]`, then the image is resized again to make the longest edge equal to
                `size["longest_edge"]`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            max_patches (`int`, *optional*, defaults to `self.max_patches`):
                Maximum number of patches per image.
            min_patches (`int`, *optional*, defaults to `self.min_patches`):
                Minimum number of patches per image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                Size of each patch.
            crop_to_patches (`bool`, *optional*, defaults to `self.crop_to_patches`):
                Whether to crop images to patches using dynamic preprocessing.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        max_patches = max_patches if max_patches is not None else self.max_patches
        min_patches = min_patches if min_patches is not None else self.min_patches
        patch_size = patch_size if patch_size is not None else self.patch_size
        crop_to_patches = crop_to_patches if crop_to_patches is not None else self.crop_to_patches

        images = make_list_of_images(images)

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_processor_keys)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        all_processed_images = []
        all_num_patches = []

        for image in images:
            if do_convert_rgb:
                image = convert_to_rgb(image)

            # Dynamic preprocessing if crop_to_patches is enabled
            if crop_to_patches:
                processed_patches = self.dynamic_preprocess(
                    image, min_num=min_patches, max_num=max_patches, image_size=patch_size, use_thumbnail=True
                )
                all_num_patches.append(len(processed_patches))
            else:
                processed_patches = [image]
                all_num_patches.append(1)

            # Process each patch
            batch_patches = []
            for patch in processed_patches:
                # Convert PIL image back to numpy array if needed
                if isinstance(patch, Image.Image):
                    patch = to_numpy_array(patch)

                if do_resize:
                    patch = resize(patch, size, resample=resample, input_data_format=input_data_format)

                if do_rescale:
                    patch = rescale(patch, scale=rescale_factor, input_data_format=input_data_format)

                if do_normalize:
                    patch = normalize(patch, mean=image_mean, std=image_std, input_data_format=input_data_format)

                patch = to_channel_dimension_format(patch, data_format, input_channel_dim=input_data_format)
                batch_patches.append(patch)

            all_processed_images.extend(batch_patches)

        data = {"pixel_values": all_processed_images, "num_patches": all_num_patches}
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["InternVL3_5ImageProcessor"]