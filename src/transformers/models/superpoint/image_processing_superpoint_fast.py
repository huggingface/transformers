# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Fast Image processor class for Superpoint."""

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    infer_channel_dimension_format,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    requires_backends,
)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from .modeling_superpoint import SuperPointKeypointDescriptionOutput

if is_torchvision_available():
    if is_torchvision_v2_available():
        import torchvision.transforms.v2.functional as F
    else:
        import torchvision.transforms.functional as F

if is_vision_available():
    pass


def is_grayscale(
    image: "torch.Tensor",
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """Checks if an image is grayscale (all RGB channels are identical)."""
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    if isinstance(image, torch.Tensor):
        if input_data_format == ChannelDimension.FIRST:
            if image.ndim < 3 or image.shape[0 if image.ndim == 3 else 1] == 1:
                return True
            return torch.all(image[0, ...] == image[1, ...]) and torch.all(image[1, ...] == image[2, ...])
        elif input_data_format == ChannelDimension.LAST:
            if image.ndim < 3 or image.shape[-1] == 1:
                return True
            return torch.all(image[..., 0] == image[..., 1]) and torch.all(image[..., 1] == image[..., 2])
    else:
        if input_data_format == ChannelDimension.FIRST:
            if image.ndim < 3 or image.shape[0 if image.ndim == 3 else 1] == 1:
                return True
            return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])
        elif input_data_format == ChannelDimension.LAST:
            if image.ndim < 3 or image.shape[-1] == 1:
                return True
            return np.all(image[..., 0] == image[..., 1]) and np.all(image[..., 1] == image[..., 2])
    return False


def convert_to_grayscale(
    image: "torch.Tensor",
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    keep_channels: bool = True,
) -> "torch.Tensor":
    """Convert a PyTorch tensor to grayscale using the NTSC formula.

    Args:
        image: Input image tensor (3D or 4D)
        input_data_format: Format of input tensor (channels first or last)
        keep_channels: Whether to keep the same number of channels in output
    """
    requires_backends(convert_to_grayscale, ["vision"])

    # Determine channel dimension
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    # If already grayscale, just return (possibly with channel duplication)
    if (input_data_format == ChannelDimension.FIRST and image.shape[0 if image.ndim == 3 else 1] == 1) or (
        input_data_format == ChannelDimension.LAST and image.shape[-1] == 1
    ):
        if (
            keep_channels
            and image.shape[
                0
                if input_data_format == ChannelDimension.FIRST and image.ndim == 3
                else 1
                if input_data_format == ChannelDimension.FIRST
                else -1
            ]
            == 1
        ):
            if input_data_format == ChannelDimension.FIRST:
                return image.repeat(3, 1, 1) if image.ndim == 3 else image.repeat(1, 3, 1, 1)
            else:
                return image.repeat(1, 1, 3) if image.ndim == 3 else image.repeat(1, 1, 1, 3)
        return image

    # RGB weights (NTSC formula)
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device, dtype=torch.float32)

    if input_data_format == ChannelDimension.FIRST:
        image = (
            image[:3]
            if image.ndim == 3 and image.shape[0] > 3
            else image[:, :3]
            if image.ndim == 4 and image.shape[1] > 3
            else image
        )

        if image.ndim == 3:
            gray = (image.to(torch.float32) * weights[:, None, None]).sum(dim=0, keepdim=True)
            if keep_channels:
                gray = gray.repeat(3, 1, 1)
        else:
            gray = (image.to(torch.float32) * weights[None, :, None, None]).sum(dim=1, keepdim=True)
            if keep_channels:
                gray = gray.repeat(1, 3, 1, 1)
    else:
        # Take first 3 channels if more than 3
        image = image[..., :3]

        # Convert to grayscale
        gray = (image.to(torch.float32) * weights).sum(dim=-1, keepdim=True)
        if keep_channels:
            gray = gray.repeat(1, 1, 3) if image.ndim == 3 else gray.repeat(1, 1, 1, 3)

    return gray.to(dtype=image.dtype)


class SuperPointFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_grayscale: Optional[bool] = True


@auto_docstring
class SuperPointImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    size = {"height": 480, "width": 640}
    default_to_square = False
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = None
    valid_kwargs = SuperPointFastImageProcessorKwargs

    @classmethod
    def from_dict(cls, image_processor_dict: dict[str, Any], **kwargs):
        """Instantiate an image processor from a python dictionary of parameters."""
        return super().from_dict(image_processor_dict, **kwargs)

    def __init__(self, **kwargs: Unpack[SuperPointFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        result = super().to_dict()
        result.pop("data_format", None)
        result.pop("default_to_square", None)
        result.pop("resample", None)
        return result

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: Optional[Union[dict[str, int], SizeDict]] = None,
        rescale_factor: Optional[float] = None,
        do_rescale: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if size is None:
            size = self.size
        if do_rescale is None:
            do_rescale = self.do_rescale
        if rescale_factor is None:
            rescale_factor = self.rescale_factor
        if do_resize is None:
            do_resize = self.do_resize

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=False)
        processed_images_grouped = {}
        do_grayscale = kwargs.get("do_grayscale", self.do_grayscale)
        input_data_format = kwargs.get("input_data_format", ChannelDimension.FIRST)

        for shape, stacked_images in grouped_images.items():
            if do_grayscale:
                stacked_images = convert_to_grayscale(
                    stacked_images, input_data_format=input_data_format, keep_channels=True
                )
            if do_resize:
                stacked_images = self.resize(stacked_images, size=size, resample=self.resample)
            if do_rescale:
                stacked_images = self.rescale(stacked_images, rescale_factor)
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(data={"pixel_values": processed_images})

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[SuperPointFastImageProcessorKwargs],
    ) -> BatchFeature:
        kwargs = self._further_process_kwargs(**kwargs)

        if kwargs.get("do_resize", self.do_resize):
            kwargs.setdefault("size", self.size)
            kwargs.setdefault("resample", self.resample)

        self._validate_preprocess_kwargs(**kwargs)
        input_data_format = kwargs.pop("input_data_format", None)
        device = kwargs.pop("device", None)

        if not isinstance(images, (list, tuple)):
            images = [images]

        kwargs.pop("default_to_square", None)
        kwargs.pop("data_format", None)

        processed_images = []
        for image in images:
            if isinstance(image, torch.Tensor):
                if image.ndim == 2:
                    image = image.unsqueeze(0)
                processed_images.append(image)
            else:
                processed_images.append(image)

        images = processed_images
        if input_data_format is None and len(images) > 0:
            input_data_format = ChannelDimension.FIRST

        images = self._prepare_input_images(images=images, input_data_format=input_data_format, device=device)

        if input_data_format is not None:
            kwargs["input_data_format"] = input_data_format

        return self._preprocess(
            images=images,
            **kwargs,
        )

    def resize(
        self,
        image: "torch.Tensor",
        size: Union[dict[str, int], SizeDict],
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`Dict[str, int]` or `SizeDict`):
                Dictionary specifying the size with height and width keys.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR

        # Check if size is a dict with height and width
        if isinstance(size, dict) and "height" in size and "width" in size:
            new_size = (size["height"], size["width"])
        else:
            # Attempt to use the parent class's logic
            return super().resize(image, size, interpolation, antialias, **kwargs)

        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)

    def post_process_keypoint_detection(
        self, outputs: "SuperPointKeypointDescriptionOutput", target_sizes: Union[TensorType, list[tuple]]
    ) -> list[dict[str, "torch.Tensor"]]:
        """
        Converts the raw output of [`SuperPointForKeypointDetection`] into lists of keypoints, scores and descriptors
        with coordinates absolute to the original image sizes.

        Args:
            outputs ([`SuperPointKeypointDescriptionOutput`]):
                Raw outputs of the model containing keypoints in a relative (x, y) format, with scores and descriptors.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. This must be the original
                image size (before any processing).
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the keypoints in absolute format according
            to target_sizes, scores and descriptors for an image in the batch as predicted by the model.
        """
        if len(outputs.mask) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the mask")

        if isinstance(target_sizes, list):
            image_sizes = torch.tensor(target_sizes, device=outputs.mask.device)
        else:
            if target_sizes.shape[1] != 2:
                raise ValueError(
                    "Each element of target_sizes must contain the size (h, w) of each image of the batch"
                )
            image_sizes = target_sizes

        # Flip the image sizes to (width, height) and convert keypoints to absolute coordinates
        image_sizes = torch.flip(image_sizes, [1])
        masked_keypoints = outputs.keypoints * image_sizes[:, None]

        # Convert masked_keypoints to int
        masked_keypoints = masked_keypoints.to(torch.int32)

        results = []
        for image_mask, keypoints, scores, descriptors in zip(
            outputs.mask, masked_keypoints, outputs.scores, outputs.descriptors
        ):
            indices = torch.nonzero(image_mask).squeeze(1)
            keypoints = keypoints[indices]
            scores = scores[indices]
            descriptors = descriptors[indices]
            results.append({"keypoints": keypoints, "scores": scores, "descriptors": descriptors})

        return results


__all__ = ["SuperPointImageProcessorFast"]
