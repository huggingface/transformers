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
"""Fast Image processor class for EfficientFormer."""

import copy
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any

from ....image_processing_utils import BatchFeature, get_size_dict
from ....image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    SizeDict,
)
from ....image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from ....processing_utils import Unpack
from ....utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)


if is_torch_available():
    import torch

if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
elif is_torchvision_available():
    from torchvision.transforms import functional as F


logger = logging.get_logger(__name__)


class EfficientFormerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_center_crop: Optional[bool]
    crop_size: Optional[Dict[str, int]]


@add_start_docstrings(
    "Constructs a fast EfficientFormer image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class EfficientFormerImageProcessorFast(BaseImageProcessorFast):
    """
    Fast image processor for EfficientFormer models. This processor uses GPU-accelerated operations from
    PyTorch/torchvision instead of PIL/NumPy for better performance, especially when processing batches of images.
    """

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 18, "width": 18}
    crop_size = {"height": 18, "width": 18}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    valid_kwargs = EfficientFormerFastImageProcessorKwargs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override the image_processor_type to match the slow version for serialization compatibility
        self._image_processor_type = "EfficientFormerImageProcessor"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = super().to_dict()
        
        # Convert PILImageResampling enum to int for serialization
        if "resample" in output and hasattr(output["resample"], "value"):
            output["resample"] = output["resample"].value
            
        # Convert ChannelDimension enum to string for serialization
        if "data_format" in output and hasattr(output["data_format"], "value"):
            output["data_format"] = output["data_format"].value
            
        # Remove any keys that are not in the slow processor
        keys_to_remove = ["_processor_class", "default_to_square", "input_data_format"]
        for key in keys_to_remove:
            if key in output:
                del output[key]
                
        return output

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save a image processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~image_processing_utils.ImageProcessingMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the image processor JSON file will be saved (will be created if it does not exist).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        # Convert the image processor to a serializable dictionary
        config_dict = self.to_dict()
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the config
        output_config_file = os.path.join(save_directory, "preprocessor_config.json")
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
            
        return [output_config_file]

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Instantiates a type of [`ImageProcessingMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the image processor object.
            kwargs:
                Additional parameters from which to initialize the image processor object.

        Returns:
            [`ImageProcessingMixin`]: The image processor object instantiated from those parameters.
        """
        config_dict = copy.deepcopy(config_dict)
        
        # Handle the case where resample is an integer
        if "resample" in config_dict and isinstance(config_dict["resample"], int):
            config_dict["resample"] = PILImageResampling(config_dict["resample"])
            
        # Handle the case where data_format is a string
        if "data_format" in config_dict and isinstance(config_dict["data_format"], str):
            config_dict["data_format"] = ChannelDimension(config_dict["data_format"])
            
        return super().from_dict(config_dict, **kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the image.
        crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
            Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
        """,
    )
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[EfficientFormerFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images for use with EfficientFormer models.
        """
        return super().preprocess(images, **kwargs)

    def center_crop(
        self, image: "torch.Tensor", crop_size: Dict[str, int]
    ) -> "torch.Tensor":
        """
        Center crop an image to the given size.
        
        Args:
            image (`torch.Tensor`):
                The image to center crop.
            crop_size (`Dict[str, int]`):
                The size to crop the image to.
                
        Returns:
            `torch.Tensor`: The center cropped image.
        """
        # Get crop size
        crop_height = crop_size["height"]
        crop_width = crop_size["width"]
        
        # Get image dimensions
        height, width = image.shape[-2:]
        
        # Calculate crop coordinates
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        
        # Crop the image
        cropped_image = F.crop(image, top, left, crop_height, crop_width)
        
        return cropped_image

    def resize(
        self, image: "torch.Tensor", size: SizeDict, interpolation: Optional["F.InterpolationMode"] = None
    ) -> "torch.Tensor":
        """
        Resize an image to a specific size.
        
        Args:
            image (`torch.Tensor`):
                The image to resize.
            size (`SizeDict`):
                The size to resize the image to.
            interpolation (`F.InterpolationMode`, *optional*):
                The interpolation method to use when resizing the image.
                
        Returns:
            `torch.Tensor`: The resized image.
        """
        if interpolation is None:
            interpolation = F.InterpolationMode.BICUBIC
            
        # Get target size
        if hasattr(size, "height") and hasattr(size, "width") and size.height is not None and size.width is not None:
            # Explicit height and width
            new_height, new_width = size.height, size.width
        else:
            raise ValueError(f"Unsupported size format: {size}")
        
        # Resize the image
        resized_image = F.resize(image, [new_height, new_width], interpolation=interpolation)
        
        return resized_image

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool = None,
        size: SizeDict = None,
        interpolation: Optional["F.InterpolationMode"] = None,
        do_center_crop: bool = None,
        crop_size: SizeDict = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the EfficientFormer model.
        
        Args:
            images (`List[torch.Tensor]`):
                List of images to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            interpolation (`F.InterpolationMode`, *optional*, defaults to `F.InterpolationMode.BICUBIC`):
                Interpolation method to use when resizing the image.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use when normalizing the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use when normalizing the image.
            return_tensors (`str` or `TensorType`, *optional*):
                Type of tensors to return.
                
        Returns:
            `BatchFeature`: A `BatchFeature` with the following field:
                - **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
                    Pixel values normalized and resized to the model's expected input size.
        """
        data = {}
        
        processed_images = []
        
        for image in images:
            # Resize image if needed
            if do_resize:
                resized_image = self.resize(image, size=size, interpolation=interpolation)
                image = resized_image
                
            # Center crop if needed
            if do_center_crop:
                cropped_image = self.center_crop(image, crop_size=crop_size)
                image = cropped_image
                
            # Rescale and normalize image
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
                
            processed_images.append(image)
            
        # Stack images and create BatchFeature
        data.update({"pixel_values": torch.stack(processed_images, dim=0)})
        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)
            
        return encoded_inputs


__all__ = ["EfficientFormerImageProcessorFast"]
