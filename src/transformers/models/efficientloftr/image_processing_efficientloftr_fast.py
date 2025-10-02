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
"""Fast Image processor class for EfficientLoFTR."""

from typing import TYPE_CHECKING, Optional, Union

import torch
from PIL import Image, ImageDraw

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    ImageInput,
    ImageType,
    PILImageResampling,
    SizeDict,
    get_image_type,
    is_pil_image,
    is_valid_image,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)


if TYPE_CHECKING:
    from .modeling_efficientloftr import KeypointMatchingOutput

import torchvision.transforms.v2.functional as F


def _is_valid_image(image):
    return is_pil_image(image) or (
        is_valid_image(image) and get_image_type(image) != ImageType.PIL and len(image.shape) == 3
    )


def flatten_pair_images(images):
    # Handle the pair validation and flattening similar to slow processor
    if isinstance(images, list):
        if len(images) == 2 and all((_is_valid_image(image) or isinstance(image, torch.Tensor)) for image in images):
            # Single pair of images - keep as is, they'll be processed by the base class
            return images
        elif all(
            isinstance(image_pair, list)
            and len(image_pair) == 2
            and all(_is_valid_image(image) or isinstance(image, torch.Tensor) for image in image_pair)
            for image_pair in images
        ):
            # Multiple pairs - flatten them
            images = [image for image_pair in images for image in image_pair]
            return images
    raise ValueError(
        "Input images must be a one of the following :",
        " - A pair of PIL images.",
        " - A pair of 3D arrays.",
        " - A list of pairs of PIL images.",
        " - A list of pairs of 3D arrays.",
    )


def is_grayscale(
    image: "torch.Tensor",
):
    """Checks if an image is grayscale (all RGB channels are identical)."""
    if image.ndim < 3 or image.shape[0 if image.ndim == 3 else 1] == 1:
        return True
    return torch.all(image[..., 0, :, :] == image[..., 1, :, :]) and torch.all(
        image[..., 1, :, :] == image[..., 2, :, :]
    )


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


class EfficientLoFTRFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    do_grayscale (`bool`, *optional*, defaults to `True`):
        Whether to convert the image to grayscale. Can be overridden by `do_grayscale` in the `preprocess` method.
    """

    do_grayscale: Optional[bool] = True


@auto_docstring
class EfficientLoFTRImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    size = {"height": 480, "width": 640}
    default_to_square = False
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = None
    valid_kwargs = EfficientLoFTRFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[EfficientLoFTRFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[EfficientLoFTRFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
        **kwargs,
    ) -> ImageInput:
        # we need to handle image pairs validation and flattening
        return flatten_pair_images(images)

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
            if do_resize:
                stacked_images = self.resize(stacked_images, size=size, interpolation=interpolation)
            processed_images_grouped[shape] = stacked_images
        resized_images = reorder_images(processed_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_rescale:
                stacked_images = self.rescale(stacked_images, rescale_factor)
            if do_grayscale:
                stacked_images = convert_to_grayscale(stacked_images)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Convert back to pairs format
        image_pairs = [processed_images[i : i + 2] for i in range(0, len(processed_images), 2)]

        # Stack each pair into a single tensor to match slow processor format
        stacked_pairs = [torch.stack(pair, dim=0) for pair in image_pairs]

        # Return in same format as slow processor
        image_pairs = torch.stack(stacked_pairs, dim=0) if return_tensors else stacked_pairs

        return BatchFeature(data={"pixel_values": image_pairs})

    def post_process_keypoint_matching(
        self,
        outputs: "KeypointMatchingOutput",
        target_sizes: Union[TensorType, list[tuple]],
        threshold: float = 0.0,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the raw output of [`KeypointMatchingOutput`] into lists of keypoints, scores and descriptors
        with coordinates absolute to the original image sizes.
        Args:
            outputs ([`KeypointMatchingOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` or `List[Tuple[Tuple[int, int]]]`, *optional*):
                Tensor of shape `(batch_size, 2, 2)` or list of tuples of tuples (`Tuple[int, int]`) containing the
                target size `(height, width)` of each image in the batch. This must be the original image size (before
                any processing).
            threshold (`float`, *optional*, defaults to 0.0):
                Threshold to filter out the matches with low scores.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the keypoints in the first and second image
            of the pair, the matching scores and the matching indices.
        """
        if outputs.matches.shape[0] != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the mask")
        if not all(len(target_size) == 2 for target_size in target_sizes):
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        if isinstance(target_sizes, list):
            image_pair_sizes = torch.tensor(target_sizes, device=outputs.matches.device)
        else:
            if target_sizes.shape[1] != 2 or target_sizes.shape[2] != 2:
                raise ValueError(
                    "Each element of target_sizes must contain the size (h, w) of each image of the batch"
                )
            image_pair_sizes = target_sizes

        keypoints = outputs.keypoints.clone()
        keypoints = keypoints * image_pair_sizes.flip(-1).reshape(-1, 2, 1, 2)
        keypoints = keypoints.to(torch.int32)

        results = []
        for keypoints_pair, matches, scores in zip(keypoints, outputs.matches, outputs.matching_scores):
            # Filter out matches with low scores
            valid_matches = torch.logical_and(scores > threshold, matches > -1)

            matched_keypoints0 = keypoints_pair[0][valid_matches[0]]
            matched_keypoints1 = keypoints_pair[1][valid_matches[1]]
            matching_scores = scores[0][valid_matches[0]]

            results.append(
                {
                    "keypoints0": matched_keypoints0,
                    "keypoints1": matched_keypoints1,
                    "matching_scores": matching_scores,
                }
            )

        return results

    def visualize_keypoint_matching(
        self,
        images,
        keypoint_matching_output: list[dict[str, torch.Tensor]],
    ) -> list["Image.Image"]:
        """
        Plots the image pairs side by side with the detected keypoints as well as the matching between them.

        Args:
            images:
                Image pairs to plot. Same as `EfficientLoFTRImageProcessor.preprocess`. Expects either a list of 2
                images or a list of list of 2 images list with pixel values ranging from 0 to 255.
            keypoint_matching_output (List[Dict[str, torch.Tensor]]]):
                A post processed keypoint matching output

        Returns:
            `List[PIL.Image.Image]`: A list of PIL images, each containing the image pairs side by side with the detected
            keypoints as well as the matching between them.
        """
        from ...image_utils import to_numpy_array
        from .image_processing_efficientloftr import validate_and_format_image_pairs

        images = validate_and_format_image_pairs(images)
        images = [to_numpy_array(image) for image in images]
        image_pairs = [images[i : i + 2] for i in range(0, len(images), 2)]

        results = []
        for image_pair, pair_output in zip(image_pairs, keypoint_matching_output):
            height0, width0 = image_pair[0].shape[:2]
            height1, width1 = image_pair[1].shape[:2]
            plot_image = torch.zeros((max(height0, height1), width0 + width1, 3), dtype=torch.uint8)
            plot_image[:height0, :width0] = torch.from_numpy(image_pair[0])
            plot_image[:height1, width0:] = torch.from_numpy(image_pair[1])

            plot_image_pil = Image.fromarray(plot_image.numpy())
            draw = ImageDraw.Draw(plot_image_pil)

            keypoints0_x, keypoints0_y = pair_output["keypoints0"].unbind(1)
            keypoints1_x, keypoints1_y = pair_output["keypoints1"].unbind(1)
            for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
                keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, pair_output["matching_scores"]
            ):
                color = self._get_color(matching_score)
                draw.line(
                    (keypoint0_x, keypoint0_y, keypoint1_x + width0, keypoint1_y),
                    fill=color,
                    width=3,
                )
                draw.ellipse((keypoint0_x - 2, keypoint0_y - 2, keypoint0_x + 2, keypoint0_y + 2), fill="black")
                draw.ellipse(
                    (keypoint1_x + width0 - 2, keypoint1_y - 2, keypoint1_x + width0 + 2, keypoint1_y + 2),
                    fill="black",
                )

            results.append(plot_image_pil)
        return results

    def _get_color(self, score):
        """Maps a score to a color."""
        r = int(255 * (1 - score))
        g = int(255 * score)
        b = 0
        return (r, g, b)


__all__ = ["EfficientLoFTRImageProcessorFast"]
