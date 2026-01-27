# Copyright 2022 The HuggingFace Inc. team.
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

import math
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache, partial
from typing import Any

import numpy as np
from huggingface_hub.dataclasses import validate_typed_dict

from .image_processing_base import BatchFeature, ImageProcessingMixin
from .image_transforms import (
    center_crop as np_center_crop,
)
from .image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
    group_images_by_shape,
    reorder_images,
)
from .image_transforms import (
    normalize as np_normalize,
)
from .image_transforms import (
    rescale as np_rescale,
)
from .image_transforms import (
    resize as np_resize,
)
from .image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    SizeDict,
    get_image_size,
    get_image_size_for_max_height_width,
    get_image_type,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from .processing_utils import ImagesKwargs, Unpack
from .utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)
from .utils.import_utils import is_rocm_platform, is_torchdynamo_compiling


if is_vision_available():
    from .image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF

    from .image_utils import pil_torch_interpolation_mapping, torch_pil_interpolation_mapping
else:
    pil_torch_interpolation_mapping = None
    torch_pil_interpolation_mapping = None


logger = logging.get_logger(__name__)


INIT_SERVICE_KWARGS = [
    "processor_class",
    "image_processor_type",
]


@lru_cache(maxsize=10)
def validate_fast_preprocess_arguments(
    do_rescale: bool | None = None,
    rescale_factor: float | None = None,
    do_normalize: bool | None = None,
    image_mean: float | list[float] | None = None,
    image_std: float | list[float] | None = None,
    do_center_crop: bool | None = None,
    crop_size: SizeDict | None = None,
    do_resize: bool | None = None,
    size: SizeDict | None = None,
    resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None = None,
    return_tensors: str | TensorType | None = None,
    data_format: ChannelDimension = ChannelDimension.FIRST,
):
    """
    Checks validity of typically used arguments in an `ImageProcessor` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    """
    validate_preprocess_arguments(
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_center_crop=do_center_crop,
        crop_size=crop_size,
        do_resize=do_resize,
        size=size,
        resample=resample,
    )
    # Extra checks for torchvision backend
    if return_tensors is not None and return_tensors != "pt":
        # Only applies when using torchvision backend
        pass

    if data_format != ChannelDimension.FIRST:
        # Only applies when using torchvision backend
        pass


def safe_squeeze(tensor: "torch.Tensor", axis: int | None = None) -> "torch.Tensor":
    """
    Squeezes a tensor, but only if the axis specified has dim 1.
    """
    if axis is None:
        return tensor.squeeze()

    try:
        return tensor.squeeze(axis=axis)
    except ValueError:
        return tensor


def max_across_indices(values: Iterable[Any]) -> list[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_max_height_width(images: list["torch.Tensor" | np.ndarray]) -> tuple[int, ...]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if len(images) == 0:
        return (0, 0)

    if isinstance(images[0], np.ndarray):
        # For NumPy arrays, assume channel-first format
        max_height = max(img.shape[-2] for img in images)
        max_width = max(img.shape[-1] for img in images)
        return (max_height, max_width)
    else:
        # For torch tensors
        _, max_height, max_width = max_across_indices([img.shape for img in images])
        return (max_height, max_width)


def divide_to_patches(image: np.ndarray | "torch.Tensor", patch_size: int) -> list[np.ndarray | "torch.Tensor"]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array | "torch.Tensor"`):
            The input image.
        patch_size (`int`):
            The size of each patch.
    Returns:
        list: A list of `np.array | "torch.Tensor"` representing the patches.
    """
    patches = []
    height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches


# ============ Image Processing Backend Classes ============


class ImageProcessingBackend:
    """
    Abstract base class for image processing backends.

    All backend implementations must inherit from this class and implement
    the required methods. Backends encapsulate backend-specific processing logic
    while maintaining a consistent interface.

    Backends are self-contained and do not hold references to processors to avoid
    circular dependencies. If custom preprocessing logic is needed, processors should
    override `_preprocess()` directly rather than relying on backend methods calling
    processor methods.
    """

    def process_image(self, *args, **kwargs):
        """Process a single image for preprocessing."""
        raise NotImplementedError

    def pad(self, *args, **kwargs):
        """Pad images to specified size."""
        raise NotImplementedError

    def resize(self, *args, **kwargs):
        """Resize an image."""
        raise NotImplementedError

    def rescale(self, *args, **kwargs):
        """Rescale an image by a scale factor."""
        raise NotImplementedError

    def normalize(self, *args, **kwargs):
        """Normalize an image."""
        raise NotImplementedError

    def center_crop(self, *args, **kwargs):
        """Center crop an image."""
        raise NotImplementedError

    def preprocess(self, *args, **kwargs) -> BatchFeature:
        """Main preprocessing pipeline."""
        raise NotImplementedError


class TorchVisionBackend(ImageProcessingBackend):
    """TorchVision backend for GPU-accelerated batched image processing."""

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: "torch.device" | None = None,
    ) -> "torch.Tensor":
        """Process a single image for torchvision backend."""
        image_type = get_image_type(image)
        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")

        if do_convert_rgb:
            image = convert_to_rgb(image)

        if image_type == ImageType.PIL:
            image = tvF.pil_to_tensor(image)
        elif image_type == ImageType.NUMPY:
            image = torch.from_numpy(image).contiguous()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        if input_data_format == ChannelDimension.LAST:
            image = image.permute(2, 0, 1).contiguous()

        if device is not None:
            image = image.to(device)

        return image

    def pad(
        self,
        images: list["torch.Tensor"],
        pad_size: SizeDict = None,
        fill_value: int | None = 0,
        padding_mode: str | None = "constant",
        return_mask: bool = False,
        disable_grouping: bool | None = False,
        is_nested: bool | None = False,
        **kwargs,
    ) -> tuple["torch.Tensor", "torch.Tensor"] | "torch.Tensor":
        """Pad images using TorchVision with batched operations."""
        if pad_size is not None:
            if not (pad_size.height and pad_size.width):
                raise ValueError(f"Pad size must contain 'height' and 'width' keys only. Got pad_size={pad_size}.")
            pad_size = (pad_size.height, pad_size.width)
        else:
            pad_size = get_max_height_width(images)

        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping, is_nested=is_nested
        )
        processed_images_grouped = {}
        processed_masks_grouped = {}
        for shape, stacked_images in grouped_images.items():
            image_size = stacked_images.shape[-2:]
            padding_height = pad_size[0] - image_size[0]
            padding_width = pad_size[1] - image_size[1]
            if padding_height < 0 or padding_width < 0:
                raise ValueError(
                    f"Padding dimensions are negative. Please make sure that the `pad_size` is larger than the "
                    f"image size. Got pad_size={pad_size}, image_size={image_size}."
                )
            if image_size != pad_size:
                padding = (0, 0, padding_width, padding_height)
                stacked_images = tvF.pad(stacked_images, padding, fill=fill_value, padding_mode=padding_mode)
            processed_images_grouped[shape] = stacked_images

            if return_mask:
                stacked_masks = torch.zeros_like(stacked_images, dtype=torch.int64)[..., 0, :, :]
                stacked_masks[..., : image_size[0], : image_size[1]] = 1
                processed_masks_grouped[shape] = stacked_masks

        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=is_nested)
        if return_mask:
            processed_masks = reorder_images(processed_masks_grouped, grouped_images_index, is_nested=is_nested)
            return processed_images, processed_masks

        return processed_images

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize an image using TorchVision."""
        # Convert PIL resample to torchvision interpolation if needed
        if resample is not None:
            if isinstance(resample, (PILImageResampling, int)):
                interpolation = pil_torch_interpolation_mapping[resample]
            else:
                interpolation = resample
        else:
            interpolation = tvF.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                image,
                size=size.shortest_edge,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )

        # Workaround for torch.compile issue with uint8 on AMD GPUs
        if is_torchdynamo_compiling() and is_rocm_platform():
            return self._compile_friendly_resize(image, new_size, interpolation, antialias)
        return tvF.resize(image, new_size, interpolation=interpolation, antialias=antialias)

    @staticmethod
    def _compile_friendly_resize(
        image: "torch.Tensor",
        new_size: tuple[int, int],
        interpolation: "tvF.InterpolationMode" | None = None,
        antialias: bool = True,
    ) -> "torch.Tensor":
        """A wrapper around tvF.resize for torch.compile compatibility with uint8 tensors."""
        if image.dtype == torch.uint8:
            image = image.float() / 256
            image = tvF.resize(image, new_size, interpolation=interpolation, antialias=antialias)
            image = image * 256
            image = torch.where(image > 255, 255, image)
            image = torch.where(image < 0, 0, image)
            image = image.round().to(torch.uint8)
        else:
            image = tvF.resize(image, new_size, interpolation=interpolation, antialias=antialias)
        return image

    def rescale(
        self,
        image: "torch.Tensor",
        scale: float,
        **kwargs,
    ) -> "torch.Tensor":
        """Rescale an image by a scale factor using TorchVision."""
        return image * scale

    def normalize(
        self,
        image: "torch.Tensor",
        mean: float | Iterable[float],
        std: float | Iterable[float],
        **kwargs,
    ) -> "torch.Tensor":
        """Normalize an image using TorchVision."""
        return tvF.normalize(image, mean, std)

    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        device: "torch.device" | None = None,
    ) -> tuple:
        if do_rescale and do_normalize:
            # Fused rescale and normalize
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale

    def _rescale_and_normalize(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
    ) -> "torch.Tensor":
        """Rescale and normalize images using TorchVision (fused for efficiency)."""
        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device,
        )
        if do_normalize:
            images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)
        elif do_rescale:
            images = self.rescale(images, rescale_factor)

        return images

    def center_crop(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        **kwargs,
    ) -> "torch.Tensor":
        """Center crop an image using TorchVision."""
        if size.height is None or size.width is None:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        image_height, image_width = image.shape[-2:]
        crop_height, crop_width = size.height, size.width

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            image = tvF.pad(image, padding_ltrb, fill=0)
            image_height, image_width = image.shape[-2:]
            if crop_width == image_width and crop_height == image_height:
                return image

        crop_top = int((image_height - crop_height) / 2.0)
        crop_left = int((image_width - crop_width) / 2.0)
        return tvF.crop(image, crop_top, crop_left, crop_height, crop_width)

    def preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess using TorchVision backend (fast, GPU-accelerated)."""
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self._rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size, disable_grouping=disable_grouping)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


class PythonBackend(ImageProcessingBackend):
    """Python/NumPy backend for portable CPU-only image processing."""

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: "torch.device" | None = None,
    ) -> np.ndarray:
        """Process a single image for python backend."""
        image_type = get_image_type(image)
        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")

        if do_convert_rgb:
            image = convert_to_rgb(image)

        if image_type == ImageType.PIL:
            image = np.array(image)
        elif image_type == ImageType.TORCH:
            image = image.numpy()

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        if input_data_format == ChannelDimension.LAST:
            # Convert from channels-last to channels-first
            if isinstance(image, np.ndarray):
                image = np.transpose(image, (2, 0, 1))

        return image

    def pad(
        self,
        images: list[np.ndarray],
        pad_size: SizeDict = None,
        fill_value: int | None = 0,
        padding_mode: str | None = "constant",
        return_mask: bool = False,
        disable_grouping: bool | None = False,
        is_nested: bool | None = False,
        **kwargs,
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | list[np.ndarray]:
        """Pad images to specified size using NumPy."""
        if pad_size is not None:
            if not (pad_size.height and pad_size.width):
                raise ValueError(f"Pad size must contain 'height' and 'width' keys only. Got pad_size={pad_size}.")
            target_height, target_width = pad_size.height, pad_size.width
        else:
            target_height, target_width = get_max_height_width(images)

        processed_images = []
        processed_masks = []

        for image in images:
            height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
            padding_height = target_height - height
            padding_width = target_width - width

            if padding_height < 0 or padding_width < 0:
                raise ValueError(
                    f"Padding dimensions are negative. Please make sure that the `pad_size` is larger than the "
                    f"image size. Got pad_size=({target_height}, {target_width}), image_size=({height}, {width})."
                )

            if height != target_height or width != target_width:
                # Pad format: ((before_1, after_1), (before_2, after_2), ...)
                # For CHW format: ((0, 0), (0, padding_height), (0, padding_width))
                pad_width = ((0, 0), (0, padding_height), (0, padding_width))
                if padding_mode == "constant":
                    image = np.pad(image, pad_width, mode="constant", constant_values=fill_value)
                else:
                    image = np.pad(image, pad_width, mode=padding_mode)

            processed_images.append(image)

            if return_mask:
                mask = np.zeros((target_height, target_width), dtype=np.int64)
                mask[:height, :width] = 1
                processed_masks.append(mask)

        if return_mask:
            return processed_images, processed_masks
        return processed_images

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None = None,
        antialias: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Resize an image using PIL/NumPy."""
        # Python backend only supports PILImageResampling
        if resample is not None and not isinstance(resample, (PILImageResampling, int)):
            if torch_pil_interpolation_mapping is not None and resample in torch_pil_interpolation_mapping:
                resample = torch_pil_interpolation_mapping[resample]
            else:
                resample = PILImageResampling.BILINEAR
        resample = resample if resample is not None else PILImageResampling.BILINEAR

        if size.shortest_edge and size.longest_edge:
            height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
            new_size = get_size_with_aspect_ratio(
                (height, width),
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                image,
                size=size.shortest_edge,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        elif size.max_height and size.max_width:
            height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
            new_size = get_image_size_for_max_height_width((height, width), size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )

        return np_resize(
            image,
            size=new_size,
            resample=resample,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def rescale(
        self,
        image: np.ndarray,
        scale: float,
        **kwargs,
    ) -> np.ndarray:
        """Rescale an image by a scale factor using NumPy."""
        return np_rescale(
            image,
            scale=scale,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def normalize(
        self,
        image: np.ndarray,
        mean: float | Iterable[float],
        std: float | Iterable[float],
        **kwargs,
    ) -> np.ndarray:
        """Normalize an image using NumPy."""
        return np_normalize(
            image,
            mean=mean,
            std=std,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def center_crop(
        self,
        image: np.ndarray,
        size: SizeDict,
        **kwargs,
    ) -> np.ndarray:
        """Center crop an image using NumPy."""
        if size.height is None or size.width is None:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")

        return np_center_crop(
            image,
            size=(size.height, size.width),
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess using Python backend (portable, CPU-only)."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


@auto_docstring
class BaseImageProcessor(ImageProcessingMixin):
    r"""
    Base class for image processors supporting both Python and TorchVision backends.

    This class provides a complete implementation for standard image preprocessing operations (resize, crop, rescale,
    normalize) with support for both Python/NumPy (for portability) and TorchVision (for GPU acceleration and speed).
    Most image processors can be implemented by simply setting class attributes; only processors requiring custom
    logic need to override methods.

    Backend Selection
    -----------------

    The processor automatically selects the best backend:
    - `backend="auto"` (default): Uses torchvision if available, otherwise python
    - `backend="torchvision"`: Forces torchvision backend (GPU-accelerated, faster)
    - `backend="python"`: Forces python backend (NumPy/PIL, more portable)

        processor = MyImageProcessor(backend="torchvision")  # Fast, GPU support
        processor = MyImageProcessor(backend="python")       # Portable, CPU-only

    Basic Implementation
    --------------------

    For processors that only need standard operations (resize, center crop, rescale, normalize), define class
    attributes:

        class MyImageProcessor(BaseImageProcessor):
            resample = PILImageResampling.BILINEAR
            image_mean = IMAGENET_DEFAULT_MEAN
            image_std = IMAGENET_DEFAULT_STD
            size = {"height": 224, "width": 224}
            do_resize = True
            do_rescale = True
            do_normalize = True

    Custom Processing
    -----------------

    Override `_preprocess` (most common):
        For custom image processing logic, override `_preprocess`. This method receives images as either torch
        tensors (torchvision backend) or NumPy arrays (python backend), both with channel dimension first.

            def _preprocess(
                self,
                images: list[torch.Tensor] | list[np.ndarray],
                do_resize: bool,
                size: SizeDict,
                # ... other parameters
                **kwargs,
            ) -> BatchFeature:
                if self.backend == "torchvision":
                    # Use group_images_by_shape and reorder_images for efficient batch processing
                    grouped_images, indices = group_images_by_shape(images)
                    processed_groups = {}
                    for shape, stacked_images in grouped_images.items():
                        if do_resize:
                            stacked_images = self.resize(stacked_images, size)
                        processed_groups[shape] = stacked_images
                    processed_images = reorder_images(processed_groups, indices)
                    return BatchFeature(data={"pixel_values": torch.stack(processed_images)})
                else:
                    # Process images one by one for python backend
                    processed_images = []
                    for image in images:
                        if do_resize:
                            image = self.resize(image, size)
                        processed_images.append(image)
                    return BatchFeature(data={"pixel_values": np.stack(processed_images)})

    Override `_preprocess_image_like_inputs` (for additional inputs):
        For processors handling multiple input types (e.g., images + segmentation maps), override this method:

            def _preprocess_image_like_inputs(
                self,
                images: ImageInput,
                segmentation_maps: ImageInput | None = None,
                do_convert_rgb: bool,
                input_data_format: ChannelDimension,
                device: "torch.device" | None = None,
                **kwargs,
            ) -> BatchFeature:
                images = self._prepare_image_like_inputs(images, do_convert_rgb, input_data_format, device)
                batch_feature = self._preprocess(images, **kwargs)

                if segmentation_maps is not None:
                    maps = self._prepare_image_like_inputs(segmentation_maps, ...)
                    batch_feature["labels"] = self._preprocess(maps, ...)

                return batch_feature

    Custom Backend Classes
    ----------------------

    To customize backend behavior, override `_backend_classes`:

        class MyTorchVisionBackend(TorchVisionBackend):
            def resize(self, image, size, **kwargs):
                # Custom resize logic for torchvision
                return super().resize(image, size, **kwargs)

        class MyPythonBackend(PythonBackend):
            def resize(self, image, size, **kwargs):
                # Custom resize logic for python
                return super().resize(image, size, **kwargs)

        class MyImageProcessor(BaseImageProcessor):
            _backend_classes = {
                "torchvision": MyTorchVisionBackend,
                "python": MyPythonBackend,
            }

    To add a new backend, extend both `_backend_classes` and `_backend_availability_checks`:

        class MyNewBackend(ImageProcessingBackend):
            # Implement required methods
            ...

        class MyImageProcessor(BaseImageProcessor):
            _backend_classes = {
                **BaseImageProcessor._backend_classes,
                "my_backend": MyNewBackend,
            }
            _backend_availability_checks = {
                **BaseImageProcessor._backend_availability_checks,
                "my_backend": lambda: check_my_backend_available(),
            }

    Custom Parameters
    -----------------

    To add parameters beyond `ImagesKwargs`, create a custom kwargs class and set it as `valid_kwargs`:

        class MyImageProcessorKwargs(ImagesKwargs):
            custom_param: int | None = None

        class MyImageProcessor(BaseImageProcessor):
            valid_kwargs = MyImageProcessorKwargs
            custom_param = 10  # default value

    Key Notes
    ---------

    - Images in `_preprocess` are torch.Tensor (torchvision backend) or np.ndarray (python backend)
    - All images have channel dimension first, regardless of backend
    - Arguments not provided by users default to class attribute values
    - TorchVision backend supports GPU acceleration and is faster
    - Python backend is more portable and doesn't require PyTorch/TorchVision
    - Backend classes encapsulate backend-specific logic and can be overridden for customization
    """

    resample = None
    image_mean = None
    image_std = None
    size = None
    default_to_square = True
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_pad = None
    pad_size = None
    do_rescale = None
    rescale_factor = 1 / 255
    do_normalize = None
    do_convert_rgb = None
    return_tensors = None
    data_format = ChannelDimension.FIRST
    input_data_format = None
    device = None
    model_input_names = ["pixel_values"]
    image_seq_length = None
    valid_kwargs = ImagesKwargs
    unused_kwargs = None

    # Backend registry: maps backend names to their backend classes
    _backend_classes = {
        "torchvision": TorchVisionBackend,
        "python": PythonBackend,
    }

    # Backend availability checkers: maps backend names to functions that check availability
    _backend_availability_checks = {
        "torchvision": is_torchvision_available,
        "python": lambda: True,  # Python backend is always available
    }

    @classmethod
    def _get_available_backends(cls):
        """
        Get a list of available backend names based on availability checks.

        Returns:
            list[str]: List of available backend names.
        """
        available = []
        for backend_name, check_func in cls._backend_availability_checks.items():
            if check_func():
                available.append(backend_name)
        return available

    @classmethod
    def _get_default_backend(cls):
        """
        Get the default backend name (first available backend in priority order).

        Returns:
            str: Default backend name.
        """
        available = cls._get_available_backends()
        if not available:
            raise RuntimeError("No backends are available. At least 'python' backend should be available.")
        # Priority: torchvision > python
        if "torchvision" in available:
            return "torchvision"
        return available[0]

    def __init__(self, backend="auto", **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

        if backend == "auto":
            backend = self._get_default_backend()
        elif backend not in self._backend_classes:
            available_backends = list(self._backend_classes.keys())
            raise ValueError(f"Invalid backend '{backend}'. Must be one of: {available_backends}, or 'auto'.")
        else:
            # Check if requested backend is available
            check_func = self._backend_availability_checks.get(backend)
            if not check_func():
                fallback_backend = self._get_default_backend()
                logger.warning_once(
                    f"You requested backend='{backend}' but it is not available. "
                    f"Falling back to backend='{fallback_backend}'."
                )
                backend = fallback_backend
        self.backend = backend
        self._backend_instance = self._backend_classes[self.backend]()

        kwargs = self.filter_out_unused_kwargs(kwargs)
        size = kwargs.pop("size", self.size)
        self.size = (
            get_size_dict(size=size, default_to_square=kwargs.pop("default_to_square", self.default_to_square))
            if size is not None
            else None
        )
        crop_size = kwargs.pop("crop_size", self.crop_size)
        self.crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None
        pad_size = kwargs.pop("pad_size", self.pad_size)
        self.pad_size = get_size_dict(size=pad_size, param_name="pad_size") if pad_size is not None else None

        for key in self.valid_kwargs.__annotations__:
            kwarg = kwargs.pop(key, None)
            if kwarg is not None:
                setattr(self, key, kwarg)
            else:
                setattr(self, key, deepcopy(getattr(self, key, None)))

        # get valid kwargs names
        self._valid_kwargs_names = list(self.valid_kwargs.__annotations__.keys())

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Whether or not this image processor is using the fast (TorchVision) backend.
        """
        return self.backend == "torchvision"

    def __call__(self, images: ImageInput, *args, **kwargs: Unpack[ImagesKwargs]) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, *args, **kwargs)

    # Backend-specific methods are now handled by backend classes
    # These methods are kept for backward compatibility but delegate to backends

    # ============ Public Interface Methods (Backend-Agnostic) ============

    def pad(
        self,
        images: list["torch.Tensor"] | list[np.ndarray],
        pad_size: SizeDict | None = None,
        fill_value: int | None = 0,
        padding_mode: str | None = "constant",
        return_mask: bool = False,
        disable_grouping: bool | None = False,
        is_nested: bool | None = False,
        **kwargs,
    ) -> tuple | list:
        """
        Pads images to `(pad_size["height"], pad_size["width"])` or to the largest size in the batch.

        Args:
            images (`list[torch.Tensor]` or `list[np.ndarray]`):
                Images to pad.
            pad_size (`SizeDict`, *optional*):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            fill_value (`int`, *optional*, defaults to `0`):
                The constant value used to fill the padded area.
            padding_mode (`str`, *optional*, defaults to "constant"):
                The padding mode to use.
            return_mask (`bool`, *optional*, defaults to `False`):
                Whether to return a pixel mask to denote padded regions.
            disable_grouping (`bool`, *optional*, defaults to `False`):
                Whether to disable grouping of images by size (torchvision backend only).

        Returns:
            `tuple | list`: The padded images and pixel masks if `return_mask` is `True`.
        """
        return self._backend_instance.pad(
            images, pad_size, fill_value, padding_mode, return_mask, disable_grouping, is_nested, **kwargs
        )

    def resize(
        self,
        image: "torch.Tensor" | np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor" | np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor` or `np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling` or `tvF.InterpolationMode`, *optional*):
                Resampling/interpolation filter to use. Can be a PIL `PILImageResampling` enum or a torchvision
                `tvF.InterpolationMode`. Defaults to `PILImageResampling.BILINEAR` (or equivalent).
            antialias (`bool`, *optional*, defaults to `True`):
                Whether to use antialiasing (torchvision backend only).

        Returns:
            `torch.Tensor` or `np.ndarray`: The resized image.
        """
        return self._backend_instance.resize(image, size, resample, antialias, **kwargs)

    def rescale(
        self,
        image: "torch.Tensor" | np.ndarray,
        scale: float,
        **kwargs,
    ) -> "torch.Tensor" | np.ndarray:
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`torch.Tensor` or `np.ndarray`):
                Image to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.

        Returns:
            `torch.Tensor` or `np.ndarray`: The rescaled image.
        """
        return self._backend_instance.rescale(image, scale, **kwargs)

    def normalize(
        self,
        image: "torch.Tensor" | np.ndarray,
        mean: float | Iterable[float],
        std: float | Iterable[float],
        **kwargs,
    ) -> "torch.Tensor" | np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`torch.Tensor` or `np.ndarray`):
                Image to normalize.
            mean (`float` or `Iterable[float]`):
                Image mean to use for normalization.
            std (`float` or `Iterable[float]`):
                Image standard deviation to use for normalization.

        Returns:
            `torch.Tensor` or `np.ndarray`: The normalized image.
        """
        return self._backend_instance.normalize(image, mean, std, **kwargs)

    def center_crop(
        self,
        image: "torch.Tensor" | np.ndarray,
        size: SizeDict,
        **kwargs,
    ) -> "torch.Tensor" | np.ndarray:
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`torch.Tensor` or `np.ndarray`):
                Image to center crop.
            size (`SizeDict`):
                Size of the output image.

        Returns:
            `torch.Tensor` or `np.ndarray`: The center cropped image.
        """
        return self._backend_instance.center_crop(image, size, **kwargs)

    def convert_to_rgb(
        self,
        image: ImageInput,
    ) -> ImageInput:
        """
        Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
        as is.
        Args:
            image (ImageInput):
                The image to convert.

        Returns:
            ImageInput: The converted image.
        """
        return convert_to_rgb(image)

    def filter_out_unused_kwargs(self, kwargs: dict):
        """
        Filter out the unused kwargs from the kwargs dictionary.
        """
        if self.unused_kwargs is None:
            return kwargs

        for kwarg_name in self.unused_kwargs:
            if kwarg_name in kwargs:
                logger.warning_once(f"This processor does not use the `{kwarg_name}` parameter. It will be ignored.")
                kwargs.pop(kwarg_name)
        return kwargs

    def _prepare_images_structure(
        self,
        images: ImageInput,
        expected_ndims: int = 3,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        images = self.fetch_images(images)
        return make_flat_list_of_images(images, expected_ndims=expected_ndims)

    def _prepare_image_like_inputs(
        self,
        images: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: "torch.device" | None = None,
        expected_ndims: int = 3,
    ) -> list["torch.Tensor"] | list[np.ndarray]:
        """
        Prepare image-like inputs for processing.

        Args:
            images (`ImageInput`):
                The image-like inputs to process.
            do_convert_rgb (`bool`, *optional*):
                Whether to convert the images to RGB.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The input data format of the images.
            device (`torch.device`, *optional*):
                The device to put the processed images on (torchvision backend only).
            expected_ndims (`int`, *optional*):
                The expected number of dimensions for the images.

        Returns:
            List[`torch.Tensor`] or List[`np.ndarray`]: The processed images.
        """
        images = self._prepare_images_structure(images, expected_ndims=expected_ndims)

        process_image_partial = partial(
            self._backend_instance.process_image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device if self.backend == "torchvision" else None,
        )

        has_nested_structure = len(images) > 0 and isinstance(images[0], list | tuple)

        if has_nested_structure:
            processed_images = [[process_image_partial(img) for img in nested_list] for nested_list in images]
        else:
            processed_images = [process_image_partial(img) for img in images]

        return processed_images

    def _further_process_kwargs(
        self,
        size: SizeDict | None = None,
        crop_size: SizeDict | None = None,
        pad_size: SizeDict | None = None,
        default_to_square: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        data_format: ChannelDimension | None = None,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated.
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if kwargs is None:
            kwargs = {}
        if size is not None:
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if crop_size is not None:
            crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size"))
        if pad_size is not None:
            pad_size = SizeDict(**get_size_dict(size=pad_size, param_name="pad_size"))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)
        if data_format is None:
            data_format = ChannelDimension.FIRST

        kwargs["size"] = size
        kwargs["crop_size"] = crop_size
        kwargs["pad_size"] = pad_size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        kwargs["data_format"] = data_format
        kwargs["resample"] = resample

        return kwargs

    def _validate_preprocess_kwargs(
        self,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | tuple[float] | None = None,
        image_std: float | tuple[float] | None = None,
        do_resize: bool | None = None,
        size: SizeDict | None = None,
        do_center_crop: bool | None = None,
        crop_size: SizeDict | None = None,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension | None = None,
        **kwargs,
    ):
        """
        validate the kwargs for the preprocess method.
        """
        validate_fast_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )

        # Backend-specific validation
        if self.backend == "torchvision":
            if return_tensors is not None and return_tensors != "pt":
                raise ValueError("TorchVision backend only supports returning PyTorch tensors (return_tensors='pt').")
            if data_format != ChannelDimension.FIRST:
                raise ValueError("TorchVision backend only supports channel-first data format.")

    @auto_docstring
    def preprocess(self, images: ImageInput, *args, **kwargs: Unpack[ImagesKwargs]) -> BatchFeature:
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_kwargs_names)

        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("data_format")

        return self._preprocess_image_like_inputs(
            images, *args, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device, **kwargs
        )

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        *args,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: str | "torch.device" | None = None,
        **kwargs: Unpack[ImagesKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        To be overridden by subclasses when image-like inputs other than images should be processed.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        return self._backend_instance.preprocess(images, *args, **kwargs)

    def to_dict(self):
        encoder_dict = super().to_dict()

        # Filter out None values that are class defaults
        filtered_dict = {}
        for key, value in encoder_dict.items():
            if value is None:
                class_default = getattr(type(self), key, "NOT_FOUND")
                # Keep None if user explicitly set it (class default is non-None)
                if class_default != "NOT_FOUND" and class_default is not None:
                    filtered_dict[key] = value
            else:
                filtered_dict[key] = value

        filtered_dict.pop("_valid_processor_keys", None)
        filtered_dict.pop("_valid_kwargs_names", None)
        # Don't save backend to config - it should be selected at runtime
        filtered_dict.pop("backend", None)
        filtered_dict.pop("_backend_instance", None)
        return filtered_dict


VALID_SIZE_DICT_KEYS = (
    {"height", "width"},
    {"shortest_edge"},
    {"shortest_edge", "longest_edge"},
    {"longest_edge"},
    {"max_height", "max_width"},
)


def is_valid_size_dict(size_dict):
    if not isinstance(size_dict, dict):
        return False

    size_dict_keys = set(size_dict.keys())
    for allowed_keys in VALID_SIZE_DICT_KEYS:
        if size_dict_keys == allowed_keys:
            return True
    return False


def convert_to_size_dict(
    size, max_size: int | None = None, default_to_square: bool = True, height_width_order: bool = True
):
    # By default, if size is an int we assume it represents a tuple of (size, size).
    if isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError("Cannot specify both size as an int, with default_to_square=True and max_size")
        return {"height": size, "width": size}
    # In other configs, if size is an int and default_to_square is False, size represents the length of
    # the shortest edge after resizing.
    elif isinstance(size, int) and not default_to_square:
        size_dict = {"shortest_edge": size}
        if max_size is not None:
            size_dict["longest_edge"] = max_size
        return size_dict
    # Otherwise, if size is a tuple it's either (height, width) or (width, height)
    elif isinstance(size, (tuple, list)) and height_width_order:
        return {"height": size[0], "width": size[1]}
    elif isinstance(size, (tuple, list)) and not height_width_order:
        return {"height": size[1], "width": size[0]}
    elif size is None and max_size is not None:
        if default_to_square:
            raise ValueError("Cannot specify both default_to_square=True and max_size")
        return {"longest_edge": max_size}

    raise ValueError(f"Could not convert size input to size dict: {size}")


def get_size_dict(
    size: int | Iterable[int] | dict[str, int] | None = None,
    max_size: int | None = None,
    height_width_order: bool = True,
    default_to_square: bool = True,
    param_name="size",
) -> dict:
    """
    Converts the old size parameter in the config into the new dict expected in the config. This is to ensure backwards
    compatibility with the old image processor configs and removes ambiguity over whether the tuple is in (height,
    width) or (width, height) format.

    - If `size` is tuple, it is converted to `{"height": size[0], "width": size[1]}` or `{"height": size[1], "width":
    size[0]}` if `height_width_order` is `False`.
    - If `size` is an int, and `default_to_square` is `True`, it is converted to `{"height": size, "width": size}`.
    - If `size` is an int and `default_to_square` is False, it is converted to `{"shortest_edge": size}`. If `max_size`
      is set, it is added to the dict as `{"longest_edge": max_size}`.

    Args:
        size (`int | Iterable[int] | dict[str, int]`, *optional*):
            The `size` parameter to be cast into a size dictionary.
        max_size (`int | None`, *optional*):
            The `max_size` parameter to be cast into a size dictionary.
        height_width_order (`bool`, *optional*, defaults to `True`):
            If `size` is a tuple, whether it's in (height, width) or (width, height) order.
        default_to_square (`bool`, *optional*, defaults to `True`):
            If `size` is an int, whether to default to a square image or not.
    """
    if not isinstance(size, dict):
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
        logger.info(
            f"{param_name} should be a dictionary on of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size}."
            f" Converted to {size_dict}.",
        )
    else:
        size_dict = size

    if not is_valid_size_dict(size_dict):
        raise ValueError(
            f"{param_name} must have one of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size_dict.keys()}"
        )
    return size_dict


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_patch_output_size(image, target_resolution, input_data_format):
    """
    Given an image and a target resolution, calculate the output size of the image after cropping to the target
    """
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width
