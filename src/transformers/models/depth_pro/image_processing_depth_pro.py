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
"""Image processor class for DepthPro."""

from typing import TYPE_CHECKING

from ...image_processing_backends import TorchVisionBackend
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
)
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    logging,
    requires_backends,
)


if TYPE_CHECKING:
    from .modeling_depth_pro import DepthProDepthEstimatorOutput

if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF

    from ...image_utils import pil_torch_interpolation_mapping

logger = logging.get_logger(__name__)


class DepthProTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for DepthPro with custom order (rescale+normalize then resize)."""

    # DepthPro resizes image after rescaling and normalizing,
    # which makes it different from TorchVisionBackend.preprocess
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
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for DepthPro: rescale+normalize FIRST, then resize."""
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Rescale and normalize FIRST
            stacked_images = self._rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            # Then resize (using torch interpolation to handle negative values)
            if do_resize:
                stacked_images = self.resize(stacked_images, size, resample, antialias=False)
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


@auto_docstring(custom_intro="Constructs a DepthPro image processor.")
class DepthProImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    _backend_classes = {
        "torchvision": DepthProTorchVisionBackend,
    }

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 1536, "width": 1536}
    do_resize = True
    do_rescale = True
    do_normalize = True

    def post_process_depth_estimation(
        self,
        outputs: "DepthProDepthEstimatorOutput",
        target_sizes: TensorType | list[tuple[int, int]] | None = None,
    ) -> list[dict[str, TensorType]]:
        """Post-processes the raw depth predictions from the model."""
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
                depth = torch.nn.functional.interpolate(
                    # input should be (B, C, H, W)
                    input=depth.unsqueeze(0).unsqueeze(1),
                    size=target_size,
                    mode=pil_torch_interpolation_mapping[self.resample].value,
                ).squeeze()
            depth = 1.0 / torch.clamp(depth, min=1e-4, max=1e4)
            results.append(
                {
                    "predicted_depth": depth,
                    "field_of_view": fov_value,
                    "focal_length": focal_length,
                }
            )
        return results


__all__ = ["DepthProImageProcessor"]
