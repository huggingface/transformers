# coding=utf-8
# Copyright 2025 Cohere Inc. team. All rights reserved.
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
"""Image processor class for Aya Vision."""

import functools
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
)
from ...utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


class AyaVisionImageProcessor(BaseImageProcessor):
    r"""
    Constructs a AyaVision image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        patch_size (`int`, *optional*):
            Size of the patches to extract from the image.
        max_splits_per_image (`int`, *optional*, defaults to 1):
            Maximum number of splits to make for a single image.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = True,
        do_convert_rgb: bool = True,
        patch_size: Optional[int] = None,
        max_splits_per_image: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 364}
        self.do_resize = do_resize
        self.size = size
        self.img_size = size.get("shortest_edge", 364)
        self.patch_size = patch_size
        self.max_splits_per_image = max_splits_per_image
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_pad = do_pad
        self.do_convert_rgb = do_convert_rgb

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - `'pt'`: Return PyTorch tensors.
                - `'tf'`: Return TensorFlow tensors.
                - `'np'`: Return NumPy arrays.
                - `None`: Return the processed image as a PIL image.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image.
        """
        # Implement preprocessing logic here
        pass

    def find_closest_aspect_ratio(self, aspect_ratio: float, width: int, height: int) -> Tuple[int, int]:
        """
        Find the closest aspect ratio from the predefined set of aspect ratios.
        
        Args:
            aspect_ratio (`float`):
                The aspect ratio of the input image.
            width (`int`):
                The width of the input image.
            height (`int`):
                The height of the input image.
                
        Returns:
            `Tuple[int, int]`: The closest aspect ratio as (width, height).
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for (rw, rh) in self._possible_aspect_ratios():
            target_aspect_ratio = rw / rh
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = rw, rh
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.img_size * self.img_size * rw * rh:
                    best_ratio = rw, rh
        return best_ratio

    def scale_to_optimal_aspect_ratio(self, img: Image.Image) -> Image.Image:
        """
        Scale the image to the optimal aspect ratio.
        
        Args:
            img (`Image.Image`):
                The input image.
                
        Returns:
            `Image.Image`: The scaled image.
        """
        w, h = img.size
        target_aspect_ratio = self.find_closest_aspect_ratio(w / h, w, h)
        target_width = self.img_size * target_aspect_ratio[0]
        target_height = self.img_size * target_aspect_ratio[1]
        return img.resize((target_width, target_height), resample=3)

    def make_img_splits(self, img: Image.Image) -> List[Image.Image]:
        """
        Split the image into patches.
        
        Args:
            img (`Image.Image`):
                The input image.
                
        Returns:
            `List[Image.Image]`: List of image patches.
        """
        width, height = img.size
        return [img.crop((x, y, x + self.img_size, y + self.img_size))
                for y in range(0, height, self.img_size)
                for x in range(0, width, self.img_size)]
    
    def normalize_image_patches(self, image_patches):
        """
        Normalize image patches.
        
        Args:
            image_patches (`List[np.ndarray]`):
                List of image patches.
                
        Returns:
            `List[np.ndarray]`: List of normalized image patches.
        """
        normalize = lambda x: (x - self.image_mean) / self.image_std
        normalized_patches = []
        for img_patch in image_patches:
            img_patch = normalize(img_patch/255)
            normalized_patches.append(img_patch)
        return normalized_patches

    @functools.lru_cache(maxsize=1)
    def _possible_aspect_ratios(self):
        """
        Get all possible aspect ratios based on max_splits_per_image.
        
        Returns:
            `List[Tuple[int, int]]`: List of possible aspect ratios as (width, height).
        """
        min_num, max_num = 1, self.max_splits_per_image
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        return sorted(target_ratios, key=lambda x: x[0] * x[1])

