# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


from typing import Dict, List, Optional, Tuple, Union

import torch

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    SizeDict,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
)
from ...processing_utils import Unpack
from ...image_processing_utils_fast import BatchFeature
from ...utils import add_start_docstrings, is_torchvision_available, is_vision_available, logging, TensorType


if is_vision_available():
    import PIL

if is_torchvision_available():
    from torchvision.transforms import functional as F


logger = logging.get_logger(__name__)


def get_resize_output_image_size(image: torch.Tensor, size: SizeDict) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Args:
        image (`torch.Tensor`):
            Image to resize.
        size (`SizeDict`):
            Size of the output image containing the keys "shortest_edge" and "longest_edge".

    Returns:
        The output size of the image after resizing.
    """
    height, width = image.size()[-2:]

    min_len = size.shortest_edge
    max_len = size.longest_edge
    aspect_ratio = width / height

    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


def get_max_height_width(images_list: List[List[torch.Tensor]]) -> Tuple[int, int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    image_sizes = []
    for images in images_list:
        for image in images:
            image_sizes.append(image.size()[-2:])

    max_height = max(size[0] for size in image_sizes)
    max_width = max(size[1] for size in image_sizes)
    return (max_height, max_width)


def make_pixel_mask(
    image: torch.Tensor, output_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`torch.Tensor`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = image.size()[-2:]
    mask = torch.zeros(output_size, dtype=torch.int64, device=image.device)
    mask[:input_height, :input_width] = 1
    return mask


def convert_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image to RGB format if it's not already.
    For PIL Images only - tensors are assumed to be already in the correct format.
    """
    if isinstance(image, PIL.Image.Image):
        if image.mode == "RGB":
            return F.pil_to_tensor(image).float() / 255.0
        
        # Use a white background for consistent results with slow processor
        image_rgba = image.convert("RGBA")
        background = PIL.Image.new("RGBA", image_rgba.size, (255, 255, 255))
        alpha_composite = PIL.Image.alpha_composite(background, image_rgba)
        alpha_composite = alpha_composite.convert("RGB")
        return F.pil_to_tensor(alpha_composite).float() / 255.0
    
    return image


class Idefics2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_convert_rgb: Optional[bool]
    do_image_splitting: Optional[bool]
    do_center_crop: Optional[bool]
    crop_size: Optional[Dict[str, int]]


@add_start_docstrings(
    "Constructs a fast Idefics2 image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
            Only has an effect if the input image is in the PIL format.
        do_image_splitting (`bool`, *optional*, defaults to `False`):
            Whether to split the image into a sequence 4 equal sub-images concatenated with the original image.
    """,
)
class Idefics2ImageProcessorFast(BaseImageProcessorFast):
    """
    Fast implementation of Idefics2ImageProcessor that uses torch and torchvision functions for image transformations.
    """
    
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True  
    do_normalize = True
    do_pad = True
    do_convert_rgb = True
    do_image_splitting = False
    size = {"shortest_edge": 378, "longest_edge": 980}
    model_input_names = ["pixel_values", "pixel_attention_mask"]
    valid_kwargs = Idefics2FastImageProcessorKwargs

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Resize an image using torchvision's functional resize.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        
        if size.shortest_edge and size.longest_edge:
            new_size = get_resize_output_image_size(image, size)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys."
            )
            
        image = F.resize(
            image,
            size=new_size,
            interpolation=interpolation,
            **kwargs
        )
        return image

    def _crop(
        self,
        image: torch.Tensor,
        w1: int,
        h1: int,
        w2: int,
        h2: int,
    ) -> torch.Tensor:
        """Crop a region from an image"""
        return image[:, h1:h2, w1:w2]

    def split_image(
        self,
        image: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Split an image into 4 equal sub-images, and concatenate that sequence with the original image.
        That means that a single image becomes a sequence of 5 images.
        """
        height, width = image.size()[-2:]

        mid_width = width // 2
        mid_height = height // 2
        
        return [
            self._crop(image, 0, 0, mid_width, mid_height),
            self._crop(image, mid_width, 0, width, mid_height),
            self._crop(image, 0, mid_height, mid_width, height),
            self._crop(image, mid_width, mid_height, width, height),
            image,
        ]

    def pad(
        self, 
        image: torch.Tensor, 
        padded_size: Tuple[int, int], 
        fill: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad an image to the specified size and create the corresponding pixel mask.
        """
        original_size = image.size()[-2:]
        padding_bottom = padded_size[0] - original_size[0]
        padding_right = padded_size[1] - original_size[1]
        
        if padding_bottom < 0 or padding_right < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded size: {padded_size}, original size: {original_size}."
            )
            
        if original_size != padded_size:
            # Ensure padding is applied consistently with the slow implementation
            padding = [0, 0, padding_right, padding_bottom]
            # Use constant padding to match slow implementation
            image = F.pad(image, padding, fill=fill, padding_mode='constant')
        
        # Create pixel mask to match the slow implementation
        pixel_mask = torch.zeros(padded_size, dtype=torch.int64, device=image.device)
        pixel_mask[:original_size[0], :original_size[1]] = 1

        return image, pixel_mask

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_image_splitting (`bool`, *optional*, defaults to `False`):
            Whether to split the image into a sequence 4 equal sub-images concatenated with the original image.
        """,
    )
    def _preprocess(
        self,
        images: List[torch.Tensor],
        **kwargs
    ) -> BatchFeature:
        """
        Preprocess a batch of images with optimized torch operations.
        """
        # Extract parameters from kwargs
        do_resize = kwargs.get("do_resize", self.do_resize)
        size = kwargs.get("size", self.size)
        interpolation = kwargs.get("interpolation", None)
        do_rescale = kwargs.get("do_rescale", self.do_rescale)
        rescale_factor = kwargs.get("rescale_factor", self.rescale_factor)
        do_normalize = kwargs.get("do_normalize", self.do_normalize)
        image_mean = kwargs.get("image_mean", self.image_mean)
        image_std = kwargs.get("image_std", self.image_std)
        do_convert_rgb = kwargs.get("do_convert_rgb", self.do_convert_rgb)
        do_pad = kwargs.get("do_pad", self.do_pad)
        do_image_splitting = kwargs.get("do_image_splitting", self.do_image_splitting)
        return_tensors = kwargs.get("return_tensors", None)
        
        # The rest of the implementation remains the same
        images_list = []
        for image_group in images:
            if isinstance(image_group, list):
                processed_group = []
                for img in image_group:
                    if do_convert_rgb and isinstance(img, PIL.Image.Image):
                        img = convert_to_rgb(img)
                    processed_group.append(img)
                images_list.append(processed_group)
            else:
                if do_convert_rgb and isinstance(image_group, PIL.Image.Image):
                    image_group = convert_to_rgb(image_group)
                images_list.append([image_group])
                

        if do_image_splitting:
            new_images_list = []
            for images in images_list:
                new_images = []
                for image in images:
                    new_images.extend(self.split_image(image))
                new_images_list.append(new_images)
            images_list = new_images_list


        if do_resize:
            for i, images in enumerate(images_list):
                for j, image in enumerate(images):
                    images_list[i][j] = self.resize(image, size=size, interpolation=interpolation)


        for i, images in enumerate(images_list):
            for j, image in enumerate(images):
                if do_rescale and not (isinstance(image, PIL.Image.Image) or (isinstance(image, torch.Tensor) and image.max() <= 1.0)):
                    images_list[i][j] = image * rescale_factor
                

                if do_normalize:
                    if isinstance(images_list[i][j], PIL.Image.Image):
                        images_list[i][j] = F.pil_to_tensor(images_list[i][j]).float() / 255.0
                    
                    images_list[i][j] = F.normalize(images_list[i][j], mean=image_mean, std=image_std)


        if do_pad:
            padded_size = get_max_height_width(images_list)
            padded_images = []
            pixel_masks = []
            
            for images in images_list:
                batch_padded_images = []
                batch_pixel_masks = []
                
                for image in images:
                    padded_image, pixel_mask = self.pad(image, padded_size)
                    batch_padded_images.append(padded_image)
                    batch_pixel_masks.append(pixel_mask)
                
                padded_images.append(batch_padded_images)
                pixel_masks.append(batch_pixel_masks)
            
            # Make sure to preserve batch dimension
            if len(padded_images) == 1:
                return BatchFeature(
                    {
                        "pixel_values": torch.stack(padded_images[0]).unsqueeze(0) if do_image_splitting else torch.stack(padded_images[0]),
                        "pixel_attention_mask": torch.stack(pixel_masks[0]).unsqueeze(0) if do_image_splitting else torch.stack(pixel_masks[0])
                    },
                    tensor_type=return_tensors
                )
            else:
                batch_pixel_values = [torch.stack(imgs) for imgs in padded_images]
                batch_pixel_masks = [torch.stack(masks) for masks in pixel_masks]
                
                return BatchFeature(
                    {
                        "pixel_values": torch.stack(batch_pixel_values),
                        "pixel_attention_mask": torch.stack(batch_pixel_masks)
                    },
                    tensor_type=return_tensors
                )
        else:
            # Make sure to preserve batch dimension
            if len(images_list) == 1:
                return BatchFeature(
                    {"pixel_values": torch.stack(images_list[0]).unsqueeze(0) if do_image_splitting else torch.stack(images_list[0])},
                    tensor_type=return_tensors
                )
            else:
                processed_images = [torch.stack(images) for images in images_list]
                return BatchFeature(
                    {"pixel_values": torch.stack(processed_images)},
                    tensor_type=return_tensors
                )
            
__all__ = ["Idefics2ImageProcessorFast"]