# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for DPT."""

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Union

import numpy as np

from ...image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    PilBackend,
    TorchVisionBackend,
)
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    logging,
    requires_backends,
)


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput

if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


class DPTImageProcessorKwargs(ImagesKwargs, total=False):
    """
    ensure_multiple_of (`int`, *optional*, defaults to 1):
        If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overridden
        by `ensure_multiple_of` in `preprocess`.
    keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
        If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can
        be overridden by `keep_aspect_ratio` in `preprocess`.
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """

    ensure_multiple_of: int
    size_divisor: int
    keep_aspect_ratio: bool
    do_reduce_labels: bool


def get_resize_output_image_size_dpt(
    input_image: Union["torch.Tensor", np.ndarray],
    output_size: int | Iterable[int],
    keep_aspect_ratio: bool,
    multiple: int,
) -> tuple[int, int] | SizeDict:
    """Calculate output size for DPT resize with aspect ratio and multiple constraints."""

    def constrain_to_multiple_of(val, multiple, min_val=0, max_val=None):
        x = round(val / multiple) * multiple
        if max_val is not None and x > max_val:
            x = math.floor(val / multiple) * multiple
        if x < min_val:
            x = math.ceil(val / multiple) * multiple
        return x

    output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
    input_height, input_width = input_image.shape[-2:]
    output_height, output_width = output_size

    scale_height = output_height / input_height
    scale_width = output_width / input_width

    if keep_aspect_ratio:
        if abs(1 - scale_width) < abs(1 - scale_height):
            scale_height = scale_width
        else:
            scale_width = scale_height

    new_height = constrain_to_multiple_of(scale_height * input_height, multiple=multiple)
    new_width = constrain_to_multiple_of(scale_width * input_width, multiple=multiple)

    return SizeDict(height=new_height, width=new_width)


class DPTTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for DPT with custom resize and pad."""

    def reduce_label(self, labels: list["torch.Tensor"]) -> list["torch.Tensor"]:
        """Reduce label values by 1, replacing 0 with 255."""
        for idx in range(len(labels)):
            label = labels[idx]
            label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype, device=label.device), label)
            label = label - 1
            label = torch.where(label == 254, torch.tensor(255, dtype=label.dtype, device=label.device), label)
            labels[idx] = label
        return labels

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        ensure_multiple_of: int = 1,
        keep_aspect_ratio: bool = False,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize with aspect ratio and multiple constraints."""
        if not size.height or not size.width:
            raise ValueError(f"Size must contain 'height' and 'width' keys. Got {size.keys()}")
        output_size = get_resize_output_image_size_dpt(
            image,
            output_size=(size.height, size.width),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
        )
        return super().resize(image, output_size, resample, **kwargs)

    def pad_image(
        self,
        images: "torch.Tensor",
        size_divisor: int = 1,
    ) -> "torch.Tensor":
        """Center pad images to be a multiple of size_divisor."""

        def _get_pad(size, size_divisor):
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        height, width = images.shape[-2:]
        pad_top, pad_bottom = _get_pad(height, size_divisor)
        pad_left, pad_right = _get_pad(width, size_divisor)
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return tvF.pad(images, padding)

    def preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
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
        do_reduce_labels: bool = False,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        size_divisor: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for DPT."""
        if do_reduce_labels:
            images = self.reduce_label(images)

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    stacked_images,
                    size,
                    resample,
                    ensure_multiple_of=ensure_multiple_of,
                    keep_aspect_ratio=keep_aspect_ratio,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            if do_pad and size_divisor is not None:
                stacked_images = self.pad_image(stacked_images, size_divisor)
            stacked_images = self._rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


class DPTPilBackend(PilBackend):
    """PIL backend for DPT with custom resize and pad."""

    def reduce_label(self, image: np.ndarray) -> np.ndarray:
        """Reduce label values by 1, replacing 0 with 255."""
        image[image == 0] = 255
        image = image - 1
        image[image == 254] = 255
        return image

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        ensure_multiple_of: int = 1,
        keep_aspect_ratio: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Resize with aspect ratio and multiple constraints."""
        if size.height is None or size.width is None:
            raise ValueError(f"Size must contain 'height' and 'width' keys. Got {size.keys()}")
        output_size = get_resize_output_image_size_dpt(
            image,
            output_size=(size.height, size.width),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
        )
        return super().resize(image, output_size, resample, **kwargs)

    def pad_image(
        self,
        image: np.ndarray,
        size_divisor: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Center pad image to be a multiple of size_divisor."""

        def _get_pad(size, size_divisor):
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        height, width = image.shape[-2:]
        pad_size_left, pad_size_right = _get_pad(height, size_divisor)
        pad_size_top, pad_size_bottom = _get_pad(width, size_divisor)
        return np_pad(
            image,
            padding=((pad_size_left, pad_size_right), (pad_size_top, pad_size_bottom)),
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
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
        do_reduce_labels: bool = False,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        size_divisor: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for DPT."""
        processed_images = []
        for image in images:
            if do_reduce_labels:
                image = self.reduce_label(image)
            if do_resize:
                image = self.resize(
                    image,
                    size,
                    resample,
                    ensure_multiple_of=ensure_multiple_of,
                    keep_aspect_ratio=keep_aspect_ratio,
                )
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            if do_pad and size_divisor is not None:
                image = self.pad_image(image, size_divisor)
            processed_images.append(image)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


@auto_docstring
class DPTImageProcessor(BaseImageProcessor):
    _backend_classes = {
        "torchvision": DPTTorchVisionBackend,
        "pil": DPTPilBackend,
    }

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_reduce_labels = False
    do_pad = False
    ensure_multiple_of = 1
    keep_aspect_ratio = False
    valid_kwargs = DPTImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[DPTImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        **kwargs: Unpack[DPTImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess.
        """
        return super().preprocess(images, segmentation_maps, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Union[str, "torch.device"] | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Handle extra inputs beyond images."""
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        images_kwargs = kwargs.copy()
        images_kwargs["do_reduce_labels"] = False
        batch_feature = self._backend_instance.preprocess(images, **images_kwargs)

        if segmentation_maps is not None:
            processed_segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            segmentation_maps_kwargs = kwargs.copy()
            segmentation_maps_kwargs.update({"do_normalize": False, "do_rescale": False})
            processed_segmentation_maps = self._backend_instance.preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            ).pixel_values.squeeze(1)

            if is_torchvision_available() and isinstance(processed_segmentation_maps, torch.Tensor):
                processed_segmentation_maps = processed_segmentation_maps.to(torch.int64)
            else:
                processed_segmentation_maps = processed_segmentation_maps.astype(np.int64)
            batch_feature["labels"] = processed_segmentation_maps

        return batch_feature

    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple] | None = None):
        """Converts the output of [`DPTForSemanticSegmentation`] into semantic segmentation maps."""
        requires_backends(self, "torch")
        logits = outputs.logits
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
            if isinstance(target_sizes, torch.Tensor):
                target_sizes = target_sizes.numpy()
            semantic_segmentation = []
            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]
        return semantic_segmentation

    def post_process_depth_estimation(
        self,
        outputs: "DepthEstimatorOutput",
        target_sizes: TensorType | list[tuple[int, int]] | None = None,
    ) -> list[dict[str, TensorType]]:
        """Converts the raw output of [`DepthEstimatorOutput`] into final depth predictions."""
        requires_backends(self, "torch")
        predicted_depth = outputs.predicted_depth
        if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )
        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0).unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
                ).squeeze()
            results.append({"predicted_depth": depth})
        return results


__all__ = ["DPTImageProcessor", "DPTImageProcessorKwargs"]
