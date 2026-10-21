# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Image processor class for SuperPoint."""

from typing import TYPE_CHECKING

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires


if TYPE_CHECKING:
    import torch

    from .modeling_superpoint import SuperPointKeypointDescriptionOutput


def is_grayscale(image: np.ndarray) -> bool:
    """Checks if an image is grayscale (all RGB channels are identical)."""
    if image.shape[0] == 1:
        return True
    return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to grayscale format using the NTSC formula. Only support numpy arrays.

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (np.ndarray):
            The image to convert.
    """
    if is_grayscale(image):
        return image

    gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
    gray_image = np.stack([gray_image] * 3, axis=0)
    return gray_image


# Adapted from transformers.models.superpoint.image_processing_superpoint.SuperPointImageProcessorKwargs
class SuperPointImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_grayscale (`bool`, *optional*, defaults to `self.do_grayscale`):
        Whether to convert the image to grayscale. Can be overridden by `do_grayscale` in the `preprocess` method.
    """

    do_grayscale: bool


@auto_docstring
class SuperPointImageProcessorPil(PilBackend):
    valid_kwargs = SuperPointImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    size = {"height": 480, "width": 640}
    default_to_square = False
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = None
    do_grayscale = False

    def __init__(self, **kwargs: Unpack[SuperPointImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        return_tensors: str | TensorType | None,
        do_grayscale: bool = False,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image in images:
            # Resize (must happen before grayscale for PIL backend to work correctly)
            if do_resize:
                image = self.resize(image, size, resample)
            # Rescale
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            # Apply grayscale conversion after rescale (if requested)
            if do_grayscale:
                image = convert_to_grayscale(image)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    @requires(backends=("torch",))
    def post_process_keypoint_detection(
        self, outputs: "SuperPointKeypointDescriptionOutput", target_sizes: TensorType | list[tuple]
    ) -> list[dict[str, "torch.Tensor"]]:
        """
        Converts the raw output of [`SuperPointForKeypointDetection`] into lists of keypoints, scores and descriptors
        with coordinates absolute to the original image sizes.

        Args:
            outputs ([`SuperPointKeypointDescriptionOutput`]):
                Raw outputs of the model containing keypoints in a relative (x, y) format, with scores and descriptors.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. This must be the original
                image size (before any processing).
        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the keypoints in absolute format according
            to target_sizes, scores and descriptors for an image in the batch as predicted by the model.
        """
        import torch

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


__all__ = ["SuperPointImageProcessorPil"]
