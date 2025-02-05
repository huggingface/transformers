# coding=utf-8
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
"""Fast Image processor class for DepthPro."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from ...image_processing_base import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
)
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
    requires_backends,
)


if TYPE_CHECKING:
    from .modeling_depth_pro import DepthProDepthEstimatorOutput

logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


if is_torchvision_available():
    from ...image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


@add_start_docstrings(
    "Constructs a fast DepthPro image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class DepthProImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    antialias = False
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 1536, "width": 1536}
    do_resize = True
    do_rescale = True
    do_normalize = True

    # DepthPro resizes image after rescaling and normalizing,
    # which makes it different from BaseImageProcessorFast._preprocess
    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        antialias: bool,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        # Group images by size for batched scaling
        grouped_images, grouped_images_index = group_images_by_shape(images)
        scaled_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            scaled_images_grouped[shape] = stacked_images
        scaled_images = reorder_images(scaled_images_grouped, grouped_images_index)

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(scaled_images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images,
                    size=size,
                    interpolation=interpolation,
                    antialias=antialias,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        processed_images = torch.stack(resized_images, dim=0) if return_tensors else resized_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    # Copied from transformers.models.depth_pro.image_processing_depth_pro.DepthProImageProcessor.post_process_depth_estimation
    def post_process_depth_estimation(
        self,
        outputs: "DepthProDepthEstimatorOutput",
        target_sizes: Optional[Union[TensorType, List[Tuple[int, int]], None]] = None,
    ) -> Dict[str, List[TensorType]]:
        """
        Post-processes the raw depth predictions from the model to generate
        final depth predictions which is caliberated using the field of view if provided
        and resized to specified target sizes if provided.

        Args:
            outputs ([`DepthProDepthEstimatorOutput`]):
                Raw outputs of the model.
            target_sizes (`Optional[Union[TensorType, List[Tuple[int, int]], None]]`, *optional*, defaults to `None`):
                Target sizes to resize the depth predictions. Can be a tensor of shape `(batch_size, 2)`
                or a list of tuples `(height, width)` for each image in the batch. If `None`, no resizing
                is performed.

        Returns:
            `List[Dict[str, TensorType]]`: A list of dictionaries of tensors representing the processed depth
            predictions, and field of view (degrees) and focal length (pixels) if `field_of_view` is given in `outputs`.

        Raises:
            `ValueError`:
                If the lengths of `predicted_depths`, `fovs`, or `target_sizes` are mismatched.
        """
        requires_backends(self, "torch")

        predicted_depth = outputs.predicted_depth
        fov = outputs.field_of_view

        batch_size = len(predicted_depth)

        if target_sizes is not None and batch_size != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many fov values as the batch dimension of the predicted depth"
            )

        results = []
        fov = [None] * batch_size if fov is None else fov
        target_sizes = [None] * batch_size if target_sizes is None else target_sizes
        for depth, fov_value, target_size in zip(predicted_depth, fov, target_sizes):
            focal_length = None
            if target_size is not None:
                # scale image w.r.t fov
                if fov_value is not None:
                    width = target_size[1]
                    focal_length = 0.5 * width / torch.tan(0.5 * torch.deg2rad(fov_value))
                    depth = depth * width / focal_length

                # interpolate
                depth = torch.nn.functional.interpolate(
                    # input should be (B, C, H, W)
                    input=depth.unsqueeze(0).unsqueeze(1),
                    size=target_size,
                    mode=pil_torch_interpolation_mapping[self.resample].value,
                ).squeeze()

            # inverse the depth
            depth = 1.0 / torch.clamp(depth, min=1e-4, max=1e4)

            results.append(
                {
                    "predicted_depth": depth,
                    "field_of_view": fov_value,
                    "focal_length": focal_length,
                }
            )

        return results


__all__ = ["DepthProImageProcessorFast"]
