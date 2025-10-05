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


from ... import is_vision_available
from ...image_processing_utils_fast import BaseImageProcessorFast, BatchFeature, get_size_dict
from ...image_utils import (
    PILImageResampling,
    ChannelDimension,
    ImageInput,
    ImageType,
    is_pil_image,
    valid_images,
    is_scaled_image,
    infer_channel_dimension_format,
    validate_preprocess_arguments,
    to_numpy_array,
    is_valid_image,
    get_image_type, SizeDict, pil_torch_interpolation_mapping,
)

import torch
from ...image_transforms import to_channel_dimension_format
from typing import Optional, Union
from ...utils import auto_docstring, TensorType, logging, requires_backends, is_torchvision_v2_available
import numpy as np
if is_vision_available():
    import PIL
    from PIL import Image, ImageDraw

if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
else:
    from torchvision.transforms import functional as F, InterpolationMode


def is_grayscale(
    image: torch.Tensor,
    input_data_format: ChannelDimension = None,
):
    if input_data_format == ChannelDimension.FIRST:
        if image.shape[0] == 1:
            return True
        return bool(torch.equal(image[0, ...], image[1, ...]) and torch.equal(image[1, ...], image[2, ...]))

    elif input_data_format == ChannelDimension.LAST:
        if image.shape[-1] == 1:
            return True
        return bool(torch.equal(image[..., 0], image[..., 1]) and torch.equal(image[..., 1], image[..., 2]))
    return None


def convert_to_grayscale(
    image: torch.Tensor,
    input_data_format: ChannelDimension = None,
) -> ImageInput:
    """
    Converts an image to grayscale format. Only support numpy and PIL Image.

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (Image):
            The image to convert.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
    """

    # Already grayscale?
    if is_grayscale(image, input_data_format=input_data_format):
        return image

    if input_data_format == ChannelDimension.LAST:
        # Convert from (H, W, C) → (C, H, W) for torchvision
        image = image.permute(2, 0, 1)

    # torchvision rgb_to_grayscale → returns (1, H, W)
    gray = F.rgb_to_grayscale(image, num_output_channels=3)

    if input_data_format == ChannelDimension.LAST:
        # Convert back to (H, W, C)
        gray = gray.permute(1, 2, 0)

    return gray
"""
    # numpy path → wrap to torch
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
        gray_tensor = convert_to_grayscale(tensor, input_data_format=input_data_format)
        return gray_tensor.numpy()

    # PIL path
    if isinstance(image, PIL.Image.Image):
        return image.convert("L").convert("RGB")  # ensure 3 channels

    return image"""


def pil_resampling_to_interpolation(
        mode: Union[PIL.Image.Resampling, F.InterpolationMode]
) -> F.InterpolationMode:
    """
    Convert a PIL.Image.Resampling or torchvision InterpolationMode
    to torchvision InterpolationMode.
    Args:
        mode: PIL.Image.Resampling, InterpolationMode

    Returns:
        InterpolationMode
    """
    # If it's already a torch InterpolationMode
    if isinstance(mode, F.InterpolationMode):
        return mode
    if mode not in pil_torch_interpolation_mapping:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    return pil_torch_interpolation_mapping[mode]


def validate_and_format_image_pairs(images: ImageInput):
    error_message = (
        "Input images must be a one of the following :",
        " - A pair of PIL images.",
        " - A pair of 3D arrays.",
        " - A list of pairs of PIL images.",
        " - A list of pairs of 3D arrays.",
    )

    def _is_valid_image(image):
        """images is a PIL Image or a 3D array."""
        return is_pil_image(image) or (
            is_valid_image(image) and get_image_type(image) != ImageType.PIL and len(image.shape) == 3
        )

    if isinstance(images, list):
        if len(images) == 2 and all((_is_valid_image(image)) for image in images):
            return images
        if all(
            isinstance(image_pair, list)
            and len(image_pair) == 2
            and all(_is_valid_image(image) for image in image_pair)
            for image_pair in images
        ):
            return [image for image_pair in images for image in image_pair]
    raise ValueError(error_message)


@auto_docstring
class SuperGlueImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BILINEAR
    image_mean = None
    image_std = None
    size = {"height": 480, "width": 640}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = None
    do_convert_rgb = None



    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_grayscale: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image pairs to preprocess. Expects either a list of 2 images or a list of list of 2 images list with
                pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set
                `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the output image after `resize` has been applied. If `size["shortest_edge"]` >= 384, the image
                is resized to `(size["shortest_edge"], size["shortest_edge"])`. Otherwise, the smaller edge of the
                image will be matched to `int(size["shortest_edge"]/ crop_pct)`, after which the image is cropped to
                `(size["shortest_edge"], size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of `PILImageResampling`, filters. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_grayscale (`bool`, *optional*, defaults to `self.do_grayscale`):
                Whether to convert the image to grayscale.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        resample = pil_resampling_to_interpolation(resample)

        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_grayscale = do_grayscale if do_grayscale is not None else self.do_grayscale

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)

        # Validate and convert the input images into a flattened list of images for all subsequent processing steps.
        images = validate_and_format_image_pairs(images)

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size,
            interpolation=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
        )

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        all_images = []
        for image in images:
            deep_copy = torch.from_numpy(np.ascontiguousarray(image))
            if do_resize:
                if input_data_format == ChannelDimension.LAST:
                    deep_copy = deep_copy.permute(2, 0, 1)
                deep_copy = self.resize(deep_copy, size=SizeDict(**size), interpolation=pil_resampling_to_interpolation(resample) )
                if input_data_format == ChannelDimension.LAST:
                    deep_copy = deep_copy.permute(1, 2, 0)

            if do_rescale:
                deep_copy = self.rescale(deep_copy, scale=rescale_factor, input_data_format=input_data_format)

            if do_grayscale:
                deep_copy = convert_to_grayscale(deep_copy, input_data_format=input_data_format)
            image = deep_copy.numpy()

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            all_images.append(image)

        # Convert back the flattened list of images into a list of pairs of images.
        image_pairs = [all_images[i : i + 2] for i in range(0, len(all_images), 2)]

        data = {"pixel_values": image_pairs}

        return BatchFeature(data=data, tensor_type=return_tensors)

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
            target_sizes (`torch.Tensor` or `list[tuple[tuple[int, int]]]`, *optional*):
                Tensor of shape `(batch_size, 2, 2)` or list of tuples of tuples (`tuple[int, int]`) containing the
                target size `(height, width)` of each image in the batch. This must be the original image size (before
                any processing).
            threshold (`float`, *optional*, defaults to 0.0):
                Threshold to filter out the matches with low scores.
        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the keypoints in the first and second image
            of the pair, the matching scores and the matching indices.
        """
        if outputs.mask.shape[0] != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the mask")
        if not all(len(target_size) == 2 for target_size in target_sizes):
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        if isinstance(target_sizes, list):
            image_pair_sizes = torch.tensor(target_sizes, device=outputs.mask.device)
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
        for mask_pair, keypoints_pair, matches, scores in zip(
            outputs.mask, keypoints, outputs.matches[:, 0], outputs.matching_scores[:, 0]
        ):
            mask0 = mask_pair[0] > 0
            mask1 = mask_pair[1] > 0
            keypoints0 = keypoints_pair[0][mask0]
            keypoints1 = keypoints_pair[1][mask1]
            matches0 = matches[mask0]
            scores0 = scores[mask0]

            # Filter out matches with low scores
            valid_matches = torch.logical_and(scores0 > threshold, matches0 > -1)

            matched_keypoints0 = keypoints0[valid_matches]
            matched_keypoints1 = keypoints1[matches0[valid_matches]]
            matching_scores = scores0[valid_matches]

            results.append(
                {
                    "keypoints0": matched_keypoints0,
                    "keypoints1": matched_keypoints1,
                    "matching_scores": matching_scores,
                }
            )

        return results

    # Copied from transformers.models.efficientloftr.image_processing_efficientloftr.EfficientLoFTRImageProcessor.visualize_keypoint_matching with EfficientLoFTR->SuperGlue
    def visualize_keypoint_matching(
        self,
        images: ImageInput,
        keypoint_matching_output: list[dict[str, torch.Tensor]],
    ) -> list["Image.Image"]:
        """
        Plots the image pairs side by side with the detected keypoints as well as the matching between them.

        Args:
            images (`ImageInput`):
                Image pairs to plot. Same as `SuperGlueImageProcessor.preprocess`. Expects either a list of 2
                images or a list of list of 2 images list with pixel values ranging from 0 to 255.
            keypoint_matching_output (List[Dict[str, torch.Tensor]]]):
                A post processed keypoint matching output

        Returns:
            `List[PIL.Image.Image]`: A list of PIL images, each containing the image pairs side by side with the detected
            keypoints as well as the matching between them.
        """
        images = validate_and_format_image_pairs(images)
        images = [to_numpy_array(image) for image in images]
        image_pairs = [images[i : i + 2] for i in range(0, len(images), 2)]

        results = []
        for image_pair, pair_output in zip(image_pairs, keypoint_matching_output):
            height0, width0 = image_pair[0].shape[:2]
            height1, width1 = image_pair[1].shape[:2]
            plot_image = np.zeros((max(height0, height1), width0 + width1, 3), dtype=np.uint8)
            plot_image[:height0, :width0] = image_pair[0]
            plot_image[:height1, width0:] = image_pair[1]

            plot_image_pil = Image.fromarray(plot_image)
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

    # Copied from transformers.models.efficientloftr.image_processing_efficientloftr.EfficientLoFTRImageProcessor._get_color
    def _get_color(self, score):
        """Maps a score to a color."""
        r = int(255 * (1 - score))
        g = int(255 * score)
        b = 0
        return (r, g, b)

__all__ = ["SuperGlueImageProcessorFast"]
