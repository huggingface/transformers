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

from typing import TYPE_CHECKING, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from .modeling_superpoint import SuperPointKeypointDescriptionOutput

if is_torchvision_v2_available():
    import torchvision.transforms.v2.functional as F
elif is_torchvision_available():
    import torchvision.transforms.functional as F


def is_grayscale(
    image: "torch.Tensor",
):
    """Checks if an image is grayscale (all RGB channels are identical)."""
    if image.ndim < 3 or image.shape[0 if image.ndim == 3 else 1] == 1:
        return True
    return torch.all(image[..., 0, :, :] == image[..., 1, :, :]) and torch.all(
        image[..., 1, :, :] == image[..., 2, :, :]
    )


class SuperPointFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    do_grayscale (`bool`, *optional*, defaults to `True`):
        Whether to convert the image to grayscale. Can be overridden by `do_grayscale` in the `preprocess` method.
    """

    do_grayscale: Optional[bool] = True


def convert_to_grayscale(
    image: "torch.Tensor",
) -> "torch.Tensor":
    """
    Converts an image to grayscale format using the NTSC formula. Only support torch.Tensor.

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (torch.Tensor):
            The image to convert.
    """
    if is_grayscale(image):
        return image
    return F.rgb_to_grayscale(image, num_output_channels=3)


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

    def __init__(self, **kwargs: Unpack[SuperPointFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: Union[dict[str, int], SizeDict],
        rescale_factor: float,
        do_rescale: bool,
        do_resize: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_grayscale: bool,
        disable_grouping: bool,
        return_tensors: Union[str, TensorType],
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_grayscale:
                stacked_images = convert_to_grayscale(stacked_images)
            if do_resize:
                stacked_images = self.resize(stacked_images, size=size, interpolation=interpolation)
            if do_rescale:
                stacked_images = self.rescale(stacked_images, rescale_factor)
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(data={"pixel_values": processed_images})

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
