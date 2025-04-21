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
"""Fast Image processor class for Nougat."""

from typing import List, Optional, Union

import numpy as np

from ...image_processing_utils import BatchFeature, get_patch_output_size, select_best_resolution

from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling, SizeDict

from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)


from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    divide_to_patches,
    group_images_by_shape,
    reorder_images,
)

from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
)

from ...image_transforms import (
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
    to_pil_image,
)

from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)

from ...processing_utils import Unpack

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

from torchvision import transforms


class NougatFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_crop_margin: Optional[bool]
    do_thumbnail: Optional[bool]
    do_align_long_axis: Optional[bool]
    do_pad: Optional[bool]


@add_start_docstrings(
    "Constructs a fast Nougat image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """do_crop_margin (`bool`, *optional*, defaults to `True`):
            Whether to crop the image margins.
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the image using thumbnail method.
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the images to the largest image size in the batch.
    """)

class NougatImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 896, "width": 672}
    do_resize: bool = True, 
    do_normalize: bool = True
    do_thumbnail: bool = True
    do_align_long_axis: bool = False
    do_pad: bool  = True
    do_rescale = True
    do_crop_margin: bool = True
    valid_kwargs = NougatFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[NougatFastImageProcessorKwargs]):
        super().__init__(**kwargs)  

    @add_start_docstrings(
        "Constructs a fast Nougat image processor.",
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
        """do_crop_margin (`bool`, *optional*, defaults to `True`):
                Whether to crop the image margins.
            do_thumbnail (`bool`, *optional*, defaults to `True`):
                Whether to resize the image using thumbnail method.
            do_align_long_axis (`bool`, *optional*, defaults to `False`):
                Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
            do_pad (`bool`, *optional*, defaults to `True`):
                Whether to pad the images to the largest image size in the batch.        
        """)         
     
    def preprocess(self, images: ImageInput, **kwargs: Unpack[NougatFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def python_find_non_zero(self, 
                        image: "torch.Tensor",):
        """This is a reimplementation of a findNonZero function equivalent to cv2."""

        non_zero_indices = torch.nonzero(image, as_tuple = False)
        idxvec = non_zero_indices[:, [2, 1]]
        idxvec = idxvec.reshape(-1, 1, 2)
        return idxvec

    def python_bounding_rect(self,  
                        coordinates):
        """This is a reimplementation of a BoundingRect function equivalent to cv2."""

        min_values = coordinates.min(dim=0).values.min(dim=0).values.to(torch.int) 
        max_values = coordinates.max(dim=0).values.max(dim=0).values.to(torch.int) 
        x_min, y_min = min_values[0], min_values[1]
        width = max_values[0] - x_min + 1
        height = max_values[1] - y_min + 1
        return x_min, y_min, width, height    
      
    def crop_margin(
        self,
        image: "torch.Tensor",
        gray_threshold: int = 200,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        ) -> "torch.Tensor":
        """
        Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the
        threshold).

        Args:
            image (`torch.Tensor`):
                The image to be cropped.
            gray_threshold (`int`, *optional*, defaults to `200`)
                Value below which pixels are considered to be gray.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the output image. If unset, will use the inferred format from the
                input.
            input_data_format (`ChannelDimension`):
                The channel dimension format of the input image. If unset, will use the inferred format from the input.
        """

        transform = transforms.Grayscale(num_output_channels=1)  # 1 for single-channel grayscale
        data = transform(image)  # image can be a PIL image or Tensor
        
        max_val = torch.max(data)
        min_val = torch.min(data)
        
        if max_val == min_val:
            return image
        data = (data - min_val) / (max_val - min_val) * 255
        gray = data < gray_threshold
        coords = self.python_find_non_zero(gray)
        x_min, y_min, width, height = self.python_bounding_rect(coords)
        image = image[:, y_min : y_min + height, x_min : x_min + width]
    
        return image

    def _crop_margin(self, 
                    stacked_images: "torch.Tensor" 
                    ) -> "torch.Tensor":

        return torch.stack([self.crop_margin(image) for image in stacked_images])

    def align_long_axis(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        ) -> "torch.Tensor":
        """
        Align the long axis of the image to the longest axis of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be aligned.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to align the long axis to.
        Returns:
            `torch.Tensor`: The aligned image.
        """
        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):

            image = torch.rot90(image, 3, dims=[height_dim, width_dim])

        return image


    def thumbnail(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        ) -> "torch.Tensor":
        """
        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
        corresponding dimension of the specified size.

        Args:
            image (`torch.tensor`):
                The image to be resized.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to resize the image to.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                The resampling filter to use. 
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BICUBIC

        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        # We always resize to the smallest of either the input or output size.
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return image

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        new_size = (height, width)    

        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)
    

    def pad_image(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        ) -> "torch.Tensor":
        """
        Pad the image to the specified size at the top, bottom, left and right.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
        """
        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        pad_top = delta_height // 2
        pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return  F.pad(image, padding)


    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BICUBIC

        shortest_edge = min(size["height"], size["width"])

        new_size = get_resize_output_image_size(
            image, size=shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
        )
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)
        

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        do_align_long_axis: bool, 
        do_thumbnail: bool, 
        do_pad: bool, 
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_crop_margin: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
        
        ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_crop_margin:
                stacked_images = self._crop_margin(stacked_images)
            if do_align_long_axis:
                stacked_images = self.align_long_axis(image=stacked_images, size=size)
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size)
            if do_thumbnail:
                stacked_images = self.thumbnail(image=stacked_images, size=size)
            if do_pad:
                stacked_images = self.pad_image(image=stacked_images, size=size)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)        

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)



__all__ = ["NougatImageProcessorFast"]
