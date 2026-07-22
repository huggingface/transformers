# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

import torch

from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring
from ..superglue.image_processing_superglue import SuperGlueImageProcessor


if TYPE_CHECKING:
    from .modeling_roma import RomaKeypointMatchingOutput


class RomaImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_upsample (`bool`, *optional*, defaults to `self.do_upsample`):
        Whether to additionally resize the pair to `upsample_size` and return it as `pixel_values_upsampled`, used
        by RoMa's high-resolution refinement pass.
    upsample_size (`dict`, *optional*, defaults to `self.upsample_size`):
        The `{"height", "width"}` size of the `pixel_values_upsampled` output.
    """

    do_upsample: bool
    upsample_size: dict


@auto_docstring
class RomaImageProcessor(SuperGlueImageProcessor):
    r"""
    Constructs a RoMa image processor. It resizes each image of a pair to a fixed square resolution (a multiple of 14
    for the DINOv2 backbone), rescales to `[0, 1]` and normalizes with the ImageNet statistics. Unlike SuperGlue/
    EfficientLoFTR, RoMa operates on RGB images (no grayscale conversion). When `do_upsample=True` it also returns a
    higher-resolution `pixel_values_upsampled` for the refinement pass.
    """

    valid_kwargs = RomaImageProcessorKwargs
    # Upstream RoMa resizes the RGB match images with torchvision `Resize` at its BICUBIC default
    # (`TupleResize` in `romatch/utils/utils.py`), so match that to reproduce its preprocessing exactly.
    resample = PILImageResampling.BICUBIC
    size = {"height": 560, "width": 560}
    default_to_square = True
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_grayscale = False
    do_upsample = False
    upsample_size = {"height": 864, "width": 864}
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD

    def __init__(self, **kwargs: Unpack[RomaImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _resize_normalize_pairs(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        disable_grouping: bool | None,
    ) -> list["torch.Tensor"]:
        """Resize (to `size`), rescale/normalize, and re-pair a flat list of images into `(2, C, H, W)` tensors."""
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.resize(stacked_images, size=size, resample=resample)
            processed_images_grouped[shape] = stacked_images
        resized_images = reorder_images(processed_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        image_pairs = [processed_images[i : i + 2] for i in range(0, len(processed_images), 2)]
        return [torch.stack(pair, dim=0) for pair in image_pairs]

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        do_upsample: bool = False,
        upsample_size: dict | None = None,
        **kwargs,
    ) -> BatchFeature:
        data = {
            "pixel_values": self._resize_normalize_pairs(
                images,
                size,
                resample,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
                disable_grouping,
            )
        }
        if do_upsample:
            upsample_size = SizeDict(**(upsample_size if upsample_size is not None else self.upsample_size))
            data["pixel_values_upsampled"] = self._resize_normalize_pairs(
                images,
                upsample_size,
                resample,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
                disable_grouping,
            )
        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_keypoint_matching(
        self,
        outputs: "RomaKeypointMatchingOutput",
        target_sizes: TensorType | list[tuple],
        threshold: float = 0.0,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the raw output of [`RomaKeypointMatchingOutput`] into lists of pixel keypoints and matching scores
        with coordinates absolute to the original image sizes.

        Args:
            outputs ([`RomaKeypointMatchingOutput`]):
                Raw outputs of the model, whose `matches` are normalized `(x0, y0, x1, y1)` correspondences in
                `[-1, 1]`.
            target_sizes (`torch.Tensor` or `list[tuple[tuple[int, int]]]`):
                Tensor of shape `(batch_size, 2, 2)` or list of tuples of tuples (`tuple[int, int]`) containing the
                target size `(height, width)` of each image of the pair. This must be the original image size (before
                any processing).
            threshold (`float`, *optional*, defaults to `0.0`):
                Threshold to filter out the matches with low scores.

        Returns:
            `list[Dict]`: A list of dictionaries, each containing the matched keypoints in the first and second image
            of the pair (`keypoints0`, `keypoints1`) and their `matching_scores`.
        """
        if outputs.matches.shape[0] != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the matches")
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

        # (batch, 2, 2) -> (width, height) per image; map normalized [-1, 1] coords to absolute pixels.
        sizes_wh = image_pair_sizes.flip(-1).to(outputs.matches.dtype)

        results = []
        for matches, scores, sizes in zip(outputs.matches, outputs.matching_scores, sizes_wh):
            valid = scores > threshold
            keypoints0 = (matches[valid, :2] + 1) / 2 * sizes[0]
            keypoints1 = (matches[valid, 2:] + 1) / 2 * sizes[1]
            results.append(
                {
                    "keypoints0": keypoints0.to(torch.int32),
                    "keypoints1": keypoints1.to(torch.int32),
                    "matching_scores": scores[valid],
                }
            )
        return results


__all__ = ["RomaImageProcessor"]
