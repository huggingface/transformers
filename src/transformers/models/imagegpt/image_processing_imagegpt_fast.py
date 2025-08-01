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
"""Fast Image processor class for ImageGPT."""
import PIL
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from transformers.image_transforms import to_channel_dimension_format

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    BatchFeature, logger
)
from ...processing_utils import Unpack
from ...image_utils import (
    PILImageResampling,
    ImageInput,
    ChannelDimension, SizeDict,
    infer_channel_dimension_format, make_list_of_images, valid_images, validate_preprocess_arguments, is_scaled_image,
)
from ...utils import (
    auto_docstring,
    is_torch_available,
    TensorType
)

if is_torch_available():
    import torch

def squared_euclidean_distance_fast(a, b):
    b = b.T
    a2 = torch.sum(a ** 2, dim = 1)
    b2 = torch.sum(b ** 2, dim = 0)
    ab = torch.matmul(a, b)
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d

def color_quantize_fast(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_fast(x, clusters)
    return np.argmin(d, axis=1)

class ImageGPTFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    # TODO: Add documentation for each argument
    do_color_quantize: Optional[bool] = True
    clusters: Optional[np.ndarray] = None
    resample: Optional[PILImageResampling] = PILImageResampling.BILINEAR
    return_tensors: Optional[Union[str, TensorType]] = None,
    data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None

@auto_docstring
class ImageGPTImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BILINEAR
    size = {"height": 256, "width": 256} # import get_size_dict?, can be overridden in preprocess
    do_resize = True
    do_normalize = True

    # Specific Kwargs
    do_color_quantize = True
    clusters = None
    resample = PILImageResampling.BILINEAR

    # not in base ##########
    image_mean = None # not in base, normalize uses a constant factor to divide pixel values
    image_std = None # not in base, normalize uses a constant factor to divide pixel values
    default_to_square = None # not in base
    crop_size = None # not in base
    do_center_crop = None # not in base
    do_rescale = None # not in base
    do_convert_rgb = None # not in base
    ############

    # initialize these arguments, pass it into super constructor
    def __init__(self, **kwargs: Unpack[ImageGPTFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    # _preprocessor has additional kwargs:
        # images, return_tensors, data_format, input_data_format

    # PUBLIC preprocess:
    def preprocess(self, images: ImageInput, **kwargs: Unpack[ImageGPTFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    # PRIVATE preprocess:
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling,
        do_normalize: bool,
        do_color_quantize: Optional[bool],
        clusters: Optional[Union[list[list[int]], np.ndarray]],
        return_tensors: Optional[Union[str, TensorType]],
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_color_quantize = do_color_quantize if do_color_quantize is not None else self.do_color_quantize
        clusters = clusters if clusters is not None else self.clusters
        clusters = np.array(clusters)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # Here, normalize() is using a constant factor to divide pixel values.
        # hence, the method does not need iamge_mean and image_std.
        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_color_quantize and clusters is None:
            raise ValueError("Clusters must be specified if do_color_quantize is True.")


        # TODO:

        # Resize to specific size

        # Normalize pixel values

        # Optionally color quantize into clusters

        # Return processed images in a specified tensor format

        if do_normalize and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If you wish to do this, "
                "make sure to set `do_normalize` to `False` and that pixel values are between [-1, 1].",
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [self.normalize(image=image, input_data_format=input_data_format) for image in images]

        if do_color_quantize:
            images = [to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format) for image in images]
            # color quantize from (batch_size, height, width, 3) to (batch_size, height, width)
            images = np.array(images)
            images = color_quantize_fast(images, clusters).reshape(images.shape[:-1])

            # flatten to (batch_size, height*width)
            batch_size = images.shape[0]
            images = images.reshape(batch_size, -1)

            # We need to convert back to a list of images to keep consistent behaviour across processors.
            images = list(images)
        else:
            images = [
                to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
                for image in images
            ]

        data = {"input_ids": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

__all__ = ["ImageGPTImageProcessorFast"]
