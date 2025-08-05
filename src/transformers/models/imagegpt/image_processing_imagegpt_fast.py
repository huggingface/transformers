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
    BatchFeature, logger,
    group_images_by_shape, reorder_images
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
    is_torchvision_available,
    TensorType
)

if is_torch_available():
    import torch
    if is_torchvision_available():
        from torchvision.transforms import functional as F

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
    return torch.argmin(d, dim=1)

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
    image_mean = [0.5, 0.5, 0.5] # not in base, normalize uses a constant factor to divide pixel values
    image_std = [0.5, 0.5, 0.5] # not in base, normalize uses a constant factor to divide pixel values
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
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        disable_grouping: Optional[bool] = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        resample: Optional[PILImageResampling] = None,
        do_color_quantize: Optional[bool] = None,
        clusters: Optional[Union[list[list[int]], np.ndarray]] = None,
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
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        # 1. Setup. Validate ImageGPT-specific requirements
        # Check for do_color_quantize and clusters.
        if do_color_quantize and clusters is None:
            raise ValueError("Clusters must be specified if do_color_quantize is True.")

        # Clusters come in np arrays. Convert to torch tensors.
        cluster_tensors = None
        if clusters is not None:
            cluster_tensors = torch.tensor(clusters, dtype=torch.float32)
            if len(images) > 0 and images[0].is_cuda:
                # if image is stored on a CUDA GPA, convert tensors to CUDA
                cluster_tensors = cluster_tensors.cuda()

        # 2. Group images into batches of the same shape for more efficient processing.
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        # Loop through shapes and stacked images
        for shape, stacked_images in grouped_images.items():
            # Resize to specific sizes (if do_resize is specified)
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images,
                    size=size,
                    interpolation=interpolation
                )
            # Normalize pixel values (if do_normalize is specified)
            if do_normalize:
                stacked_images = self.normalize(
                    image=stacked_images,
                    mean=image_mean,
                    std=image_std
                )
            resized_images_grouped[shape] = stacked_images

        # 3. Reorder and maintain original image order after processing into batches
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
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

        # 4. Color quantize if specified
        if do_color_quantize:
            quantized_images = []
            for image in processed_images:
                # Convert CHW to HWC for quantization
                image_hwc = image.permute(1, 2, 0)  # (H, W, C)

                # Denormalize back to [0, 255] for quantization
                image_hwc = (image_hwc + 1.0) * 127.5
                image_hwc = torch.clamp(image_hwc, 0, 255)

                # Fast torch-based color quantization
                quantized = color_quantize_fast(image_hwc, cluster_tensors)

                # Flatten to sequence (H*W,)
                quantized_flat = quantized.view(-1)
                quantized_images.append(quantized_flat)

            # Stack all quantized sequences
            input_ids = torch.stack(quantized_images, dim=0)

            return BatchFeature(data={"input_ids": input_ids}, tensor_type=return_tensors)
        else:
        # 5. Standard output without quantizing
            pixel_values = torch.stack(processed_images, dim=0)
            return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

__all__ = ["ImageGPTImageProcessorFast"]
