# Copyright 2025 The HuggingFace Inc. team.
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

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, Optional, Union

import numpy as np

from .image_processing_base import BatchFeature
from .image_processing_utils import BaseImageProcessor
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
    get_max_height_width,
    infer_channel_dimension_format,
)
from .processing_utils import ImagesKwargs, Unpack
from .utils import (
    TensorType,
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


class TorchVisionBackend(BaseImageProcessor):
    """TorchVision backend for GPU-accelerated batched image processing."""

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Whether or not this image processor is using the fast (TorchVision) backend.
        The `is_fast` property is deprecated and will be removed in v5.3 of Transformers.
        Use the `backend` attribute instead (e.g., `processor.backend == "torchvision"`).
        """
        logger.warning_once(
            "The `is_fast` property is deprecated and will be removed in v5.3 of Transformers. "
            "Use the `backend` attribute instead (e.g., `processor.backend == 'torchvision'`)."
        )
        return True

    @property
    def backend(self) -> str:
        """
        `str`: The backend used by this image processor.
        """
        return "torchvision"

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: Optional["torch.device"] = None,
        **kwargs: Unpack[ImagesKwargs],
    ) -> "torch.Tensor":
        """Process a single image for torchvision backend."""
        image_type = get_image_type(image)
        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")

        if do_convert_rgb:
            image = self.convert_to_rgb(image)

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

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        """Convert an image to RGB format."""
        return convert_to_rgb(image)

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
    ) -> Union[tuple["torch.Tensor", "torch.Tensor"], "torch.Tensor"]:
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
        resample: Union["PILImageResampling", "tvF.InterpolationMode", int] | None = None,
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
        interpolation: Optional["tvF.InterpolationMode"] = None,
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
        device: Optional["torch.device"] = None,
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

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: Union["PILImageResampling", "tvF.InterpolationMode", int] | None,
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


class PilBackend(BaseImageProcessor):
    """PIL/NumPy backend for portable CPU-only image processing."""

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Whether or not this image processor is using the fast (TorchVision) backend.
        The `is_fast` property is deprecated and will be removed in v5.3 of Transformers.
        Use the `backend` attribute instead (e.g., `processor.backend == "torchvision"`).
        """
        logger.warning_once(
            "The `is_fast` property is deprecated and will be removed in v5.3 of Transformers. "
            "Use the `backend` attribute instead (e.g., `processor.backend == 'torchvision'`)."
        )
        return False

    @property
    def backend(self) -> str:
        """
        `str`: The backend used by this image processor.
        """
        return "pil"

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        **kwargs: Unpack[ImagesKwargs],
    ) -> np.ndarray:
        """Process a single image for PIL backend."""
        image_type = get_image_type(image)
        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")

        if do_convert_rgb:
            image = self.convert_to_rgb(image)

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

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        """Convert an image to RGB format."""
        return convert_to_rgb(image)

    def pad(
        self,
        images: list[np.ndarray],
        pad_size: SizeDict = None,
        fill_value: int | None = 0,
        padding_mode: str | None = "constant",
        return_mask: bool = False,
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
        resample: Union["PILImageResampling", "tvF.InterpolationMode", int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize an image using PIL/NumPy."""
        # PIL backend only supports PILImageResampling
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

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: Union["PILImageResampling", "tvF.InterpolationMode", int] | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess using PIL backend (portable, CPU-only)."""
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

    def to_dict(self) -> dict[str, Any]:
        processor_dict = super().to_dict()
        # Remove the "Pil" suffix from the image processor type
        if processor_dict.get("image_processor_type", "").endswith("Pil"):
            processor_dict["image_processor_type"] = processor_dict["image_processor_type"][:-3]
        return processor_dict
