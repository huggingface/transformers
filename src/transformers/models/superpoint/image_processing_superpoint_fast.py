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

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    SizeDict,
    infer_channel_dimension_format,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
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
        pass
    else:
        pass

from ...image_processing_utils import BatchFeature


if is_vision_available():
    import PIL

import numpy as np


def is_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    if input_data_format == ChannelDimension.FIRST:
        if image.shape[0] == 1:
            return True
        return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])
    elif input_data_format == ChannelDimension.LAST:
        if image.shape[-1] == 1:
            return True
        return np.all(image[..., 0] == image[..., 1]) and np.all(image[..., 1] == image[..., 2])


def convert_to_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> ImageInput:
    """
    Converts an image to grayscale format using the NTSC formula. Only support numpy and PIL Image. TODO support torch
    and tensorflow grayscale conversion

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (Image):
            The image to convert.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
    """
    requires_backends(convert_to_grayscale, ["vision"])

    if isinstance(image, np.ndarray):
        if is_grayscale(image, input_data_format=input_data_format):
            return image
        if input_data_format == ChannelDimension.FIRST:
            gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
            gray_image = np.stack([gray_image] * 3, axis=0)
        elif input_data_format == ChannelDimension.LAST:
            gray_image = image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140
            gray_image = np.stack([gray_image] * 3, axis=-1)
        return gray_image

    if not isinstance(image, PIL.Image.Image):
        return image

    image = image.convert("L")
    return image


class SuperPointImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_reduce_labels: Optional[bool]


@add_start_docstrings(
    "Constructs a fast SuperPoint image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class SuperPointImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = None
    size = {"height": 480, "width": 640}
    default_to_square = False
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = None
    do_grayscale = False
    valid_kwargs = SuperPointImageProcessorKwargs

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_grayscale: bool,
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_rescale:
                stacked_images = self.rescale(stacked_images, rescale_factor)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return processed_images

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature:
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))
        do_grayscale = kwargs.pop("do_grayscale")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        if do_grayscale:
            images = [convert_to_grayscale(image, input_data_format=input_data_format) for image in images]

        images = self._prepare_input_images(images=images, input_data_format=input_data_format, device=device)
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        images = self._preprocess(
            images=images,
            **kwargs,
        )

        data = {"pixel_values": images}
        return BatchFeature(data=data)

    def post_process_keypoint_detection(
        self, outputs: "SuperPointKeypointDescriptionOutput", target_sizes: Union[TensorType, List[Tuple]]
    ) -> List[Dict[str, "torch.Tensor"]]:
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

        if isinstance(target_sizes, List):
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
