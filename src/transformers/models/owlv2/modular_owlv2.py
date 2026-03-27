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

import warnings

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import (
    group_images_by_shape,
    pad,
    reorder_images,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...utils import (
    TensorType,
    auto_docstring,
    is_scipy_available,
    requires_backends,
)
from ...utils.import_utils import requires
from ..owlvit.image_processing_owlvit import OwlViTImageProcessor
from ..owlvit.image_processing_pil_owlvit import OwlViTImageProcessorPil


if is_scipy_available():
    from scipy import ndimage as ndi


def _preprocess_resize_output_shape(image, output_shape):
    """Validate resize output shape according to input image.

    Args:
        image (`np.ndarray`):
         Image to be resized.
        output_shape (`iterable`):
            Size of the generated output image `(rows, cols[, ...][, dim])`. If `dim` is not provided, the number of
            channels is preserved.

    Returns
        image (`np.ndarray`):
            The input image, but with additional singleton dimensions appended in the case where `len(output_shape) >
            input.ndim`.
        output_shape (`Tuple`):
            The output shape converted to tuple.

    Raises ------ ValueError:
        If output_shape length is smaller than the image number of dimensions.

    Notes ----- The input image is reshaped if its number of dimensions is not equal to output_shape_length.

    """
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape += (1,) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1],)
    elif output_ndim < image.ndim:
        raise ValueError("output_shape length cannot be smaller than the image number of dimensions")

    return image, output_shape


def _clip_warp_output(input_image, output_image):
    """Clip output image to range of values of input image.

    Note that this function modifies the values of *output_image* in-place.

    Taken from:
    https://github.com/scikit-image/scikit-image/blob/b4b521d6f0a105aabeaa31699949f78453ca3511/skimage/transform/_warps.py#L640.

    Args:
        input_image : ndarray
            Input image.
        output_image : ndarray
            Output image, which is modified in-place.
    """
    min_val = np.min(input_image)
    if np.isnan(min_val):
        # NaNs detected, use NaN-safe min/max
        min_func = np.nanmin
        max_func = np.nanmax
        min_val = min_func(input_image)
    else:
        min_func = np.min
        max_func = np.max
    max_val = max_func(input_image)

    output_image = np.clip(output_image, min_val, max_val)

    return output_image


def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    # for owlv2 image is padded to max size unlike owlvit, that's why we have to scale boxes to max size
    max_size = torch.max(image_height, image_width)

    scale_factor = torch.stack([max_size, max_size, max_size, max_size], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


@auto_docstring
class Owlv2ImageProcessor(OwlViTImageProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 960, "width": 960}
    rescale_factor = 1 / 255
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    crop_size = None
    do_center_crop = None

    def _pad_images(self, images: "torch.Tensor", constant_value: float = 0.0) -> "torch.Tensor":
        """
        Pad an image with zeros to the given size.
        """
        height, width = images.shape[-2:]
        size = max(height, width)
        pad_bottom = size - height
        pad_right = size - width

        padding = (0, 0, pad_right, pad_bottom)
        padded_image = tvF.pad(images, padding, fill=constant_value)
        return padded_image

    def pad(
        self,
        images: list["torch.Tensor"],
        disable_grouping: bool | None,
        constant_value: float = 0.0,
        **kwargs,
    ) -> list["torch.Tensor"]:
        """
        Unlike the Base class `self.pad` where all images are padded to the maximum image size,
        Owlv2 pads an image to square.
        """
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self._pad_images(
                stacked_images,
                constant_value=constant_value,
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        return processed_images

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        anti_aliasing: bool = True,
        anti_aliasing_sigma=None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image as per the original implementation.

        Args:
            image (`Tensor`):
                Image to resize.
            size (`dict[str, int]`):
                Dictionary containing the height and width to resize the image to.
            anti_aliasing (`bool`, *optional*, defaults to `True`):
                Whether to apply anti-aliasing when downsampling the image.
            anti_aliasing_sigma (`float`, *optional*, defaults to `None`):
                Standard deviation for Gaussian kernel when downsampling the image. If `None`, it will be calculated
                automatically.
        """
        output_shape = (size.height, size.width)

        input_shape = image.shape

        # select height and width from input tensor
        factors = torch.tensor(input_shape[2:]).to(image.device) / torch.tensor(output_shape).to(image.device)

        if anti_aliasing:
            if anti_aliasing_sigma is None:
                anti_aliasing_sigma = ((factors - 1) / 2).clamp(min=0)
            else:
                anti_aliasing_sigma = torch.atleast_1d(anti_aliasing_sigma) * torch.ones_like(factors)
                if torch.any(anti_aliasing_sigma < 0):
                    raise ValueError("Anti-aliasing standard deviation must be greater than or equal to zero")
                elif torch.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                    warnings.warn(
                        "Anti-aliasing standard deviation greater than zero but not down-sampling along all axes"
                    )
            if torch.any(anti_aliasing_sigma == 0):
                filtered = image
            else:
                kernel_sizes = 2 * torch.ceil(3 * anti_aliasing_sigma).int() + 1

                filtered = tvF.gaussian_blur(
                    image, (kernel_sizes[0], kernel_sizes[1]), sigma=anti_aliasing_sigma.tolist()
                )

        else:
            filtered = image

        return TorchvisionBackend.resize(filtered, size=size, antialias=False)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_pad: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            # Rescale images before other operations as done in original implementation
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, False, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        if do_pad:
            processed_images = self.pad(processed_images, constant_value=0.0, disable_grouping=disable_grouping)

        grouped_images, grouped_images_index = group_images_by_shape(
            processed_images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                resized_stack = self.resize(image=stacked_images, size=size, resample=resample)
                resized_images_grouped[shape] = resized_stack
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, False, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


@auto_docstring
@requires(backends=("vision", "torch", "torchvision"))
class Owlv2ImageProcessorPil(OwlViTImageProcessorPil):
    resample = PILImageResampling.BILINEAR
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 960, "width": 960}
    rescale_factor = 1 / 255
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    crop_size = None
    do_center_crop = None

    def pad(self, image: "np.ndarray", constant_value: float = 0.0) -> "np.ndarray":
        """
        Pad an image with zeros to the given size.
        """
        height, width = image.shape[-2:]
        size = max(height, width)
        pad_bottom = size - height
        pad_right = size - width
        image = pad(
            image=image,
            padding=((0, pad_bottom), (0, pad_right)),
            constant_values=constant_value,
        )
        return image

    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        anti_aliasing: bool = True,
        anti_aliasing_sigma=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image as per the original implementation.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`dict[str, int]`):
                Dictionary containing the height and width to resize the image to.
            anti_aliasing (`bool`, *optional*, defaults to `True`):
                Whether to apply anti-aliasing when downsampling the image.
            anti_aliasing_sigma (`float`, *optional*, defaults to `None`):
                Standard deviation for Gaussian kernel when downsampling the image. If `None`, it will be calculated
                automatically.
        """
        requires_backends(self, "scipy")

        output_shape = (size["height"], size["width"])
        image = to_channel_dimension_format(image, ChannelDimension.LAST)
        image, output_shape = _preprocess_resize_output_shape(image, output_shape)
        input_shape = image.shape
        factors = np.divide(input_shape, output_shape)

        # Translate modes used by np.pad to those used by scipy.ndimage
        ndi_mode = "mirror"
        cval = 0
        order = 1
        if anti_aliasing:
            if anti_aliasing_sigma is None:
                anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
            else:
                anti_aliasing_sigma = np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
                if np.any(anti_aliasing_sigma < 0):
                    raise ValueError("Anti-aliasing standard deviation must be greater than or equal to zero")
                elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                    warnings.warn(
                        "Anti-aliasing standard deviation greater than zero but not down-sampling along all axes"
                    )
            filtered = ndi.gaussian_filter(image, anti_aliasing_sigma, cval=cval, mode=ndi_mode)
        else:
            filtered = image

        zoom_factors = [1 / f for f in factors]
        out = ndi.zoom(filtered, zoom_factors, order=order, mode=ndi_mode, cval=cval, grid_mode=True)

        image = _clip_warp_output(image, out)

        image = to_channel_dimension_format(image, ChannelDimension.FIRST)
        return image

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_pad: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image in images:
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_pad:
                image = self.pad(image)
            if do_resize:
                image = self.resize(image, size, resample)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["Owlv2ImageProcessor", "Owlv2ImageProcessorPil"]
