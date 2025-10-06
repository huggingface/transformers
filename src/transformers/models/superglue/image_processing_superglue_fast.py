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
from typing import Optional, Union

import numpy as np
import torch

from ... import is_vision_available
from ...image_processing_utils_fast import BaseImageProcessorFast, BatchFeature, get_size_dict
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    PILImageResampling,
    SizeDict,
    get_image_type,
    infer_channel_dimension_format,
    is_pil_image,
    is_valid_image,
    pil_torch_interpolation_mapping,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, auto_docstring, is_torchvision_v2_available
from .modeling_superglue import KeypointMatchingOutput


if is_vision_available():
    import PIL
    from PIL import Image, ImageDraw

if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
else:
    from torchvision.transforms import functional as F


def is_grayscale(
    images: torch.Tensor,
    input_data_format: ChannelDimension,
) -> bool:
    if input_data_format == ChannelDimension.FIRST:
        if images.shape[1] == 1:
            return True
        result = (images[:, 0, ...] == images[:, 1, ...]) & (images[:, 1, ...] == images[:, 2, ...])
    elif input_data_format == ChannelDimension.LAST:
        if images.shape[-1] == 1:
            return True
        result = (images[..., 0] == images[..., 1]) & (images[..., 1] == images[..., 2])
    else:
        raise ValueError(f"Unsupported input_data_format: {input_data_format}")
    return result.all().item()



def convert_to_grayscale(
    images: torch.Tensor,
    input_data_format: ChannelDimension,
) -> ImageInput:
    """
    Converts an image to grayscale format.

    This function returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        images (torch.Tensor):
            A list of image to convert, the tensor dimension is supposed to be 4 -> (B,C,H,W).
        input_data_format (`ChannelDimension`):
            The channel dimension format of the input image.
    """

    if is_grayscale(images, input_data_format=input_data_format):
        return images

    if input_data_format == ChannelDimension.LAST:
        images = images.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W

    images = F.rgb_to_grayscale(images, num_output_channels=3)

    if input_data_format == ChannelDimension.LAST:
        images = images.permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C

    return images


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

def images_to_torch_tensor(
        images: ImageInput,
        data_format: ChannelDimension = ChannelDimension.LAST,
        input_data_format: Optional[ChannelDimension] = None
   ) -> torch.Tensor:
    """
    Convert a list of images to a list of torch.Tensor in C,H,W format.

    Args:
        images (`List[PIL.Image.Image | np.ndarray | torch.Tensor]`):
            Input images.
        data_format (`str`, optional, default "channels_first"):
            If "channels_first", output tensors will be (C,H,W). If "channels_last", (H,W,C).
        input_data_format (ChannelDimension or None)
            references the input images channel dimension.

    Returns:
        List[torch.Tensor]: Converted images as torch tensors.
    """

    # Assuming they all are the same type
    if isinstance(images[0], np.ndarray):
        all_images = [torch.from_numpy(img) for img in images]
    elif isinstance(images[0], Image.Image) :
        all_images = [F.to_dtype(F.to_image(img), dtype=torch.float32, scale=True) for img in images]
    elif isinstance(images[0], torch.Tensor):
        all_images = images
    else:
        raise ValueError(f"Unsupported image type: {type(images)}")


    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(all_images[0].numpy())

    if input_data_format != data_format:
        if input_data_format == ChannelDimension.LAST and data_format == ChannelDimension.FIRST:
            all_images = [img.permute(2, 0, 1) for img in all_images]
        elif input_data_format == ChannelDimension.FIRST and data_format == ChannelDimension.LAST:
            all_images = [img.permute(1, 2, 0) for img in all_images]
        else:
            raise ValueError(f"Unsupported conversion: {input_data_format} -> {data_format}")

    return all_images

# Copied from transformers.models.efficientloftr.image_processing_efficientloftr.EfficientLoFTRImageProcessor._get_color
def _get_color(score):
    """Maps a score to a color."""
    r = int(255 * (1 - score))
    g = int(255 * score)
    b = 0
    return r, g, b


@auto_docstring
class SuperGlueImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    size = {"height": 480, "width": 640}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_grayscale = False

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

        # Handle input
        do_resize       = do_resize         if do_resize        is not None else self.do_resize
        resample        = resample          if resample         is not None else self.resample
        do_rescale      = do_rescale        if do_rescale       is not None else self.do_rescale
        rescale_factor  = rescale_factor    if rescale_factor   is not None else self.rescale_factor
        do_grayscale    = do_grayscale      if do_grayscale     is not None else self.do_grayscale
        size            = size              if size             is not None else self.size
        data_format     = data_format       if data_format      is not None else self.data_format

        resample = pil_resampling_to_interpolation(resample)
        size = get_size_dict(size, default_to_square=False)
        if isinstance(input_data_format, str):
            input_data_format = ChannelDimension(input_data_format)

        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size,
            interpolation=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
        )

        # Format Images
        images = validate_and_format_image_pairs(images)
        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        # All transformations expect torch tensors, using ChannelDimension.FIRST dataformat for processing.
        images = images_to_torch_tensor(images, data_format=ChannelDimension.FIRST, input_data_format=input_data_format)

        if do_resize:
            images = torch.stack([
                self.resize(img, size=SizeDict(**size), interpolation=resample)
                for img in images
            ], dim=0)
        if do_rescale:
            images = self.rescale(images, scale=rescale_factor)
        if do_grayscale:
            images = convert_to_grayscale(images, input_data_format=ChannelDimension.FIRST)
        if data_format == ChannelDimension.LAST:
            images = images.permute(0, 2, 3, 1)
        images = [img.numpy() for img in images]

        # Convert back the flattened list of images into a list of pairs of images.
        image_pairs = [images[i: i + 2] for i in range(0, len(images), 2)]

        data = {"pixel_values": image_pairs}

        return BatchFeature(data=data, tensor_type=return_tensors)


    def post_process_keypoint_matching(
        self,
        outputs: KeypointMatchingOutput,
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
                color = _get_color(matching_score)
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



__all__ = ["SuperGlueImageProcessorFast"]
