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
"""Fast Image processor class for DPT."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from ...image_processing_utils_fast import (
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
from ...image_processing_base import BatchFeature
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
    requires_backends,
)
from ...utils.import_utils import requires

if TYPE_CHECKING:
    from .modeling_dpt import DPTDepthEstimatorOutput

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F
    from ...image_utils import pil_torch_interpolation_mapping

@auto_docstring
@requires(backends=("torchvision", "torch"))
class DPTImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = False
    keep_aspect_ratio = False
    ensure_multiple_of = 1

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        do_pad: bool,
        keep_aspect_ratio: bool,
        ensure_multiple_of: int,
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        if not is_torch_available() or not is_torchvision_available():
            raise ImportError("DPTImageProcessorFast requires torch and torchvision.")
        if not size or "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must contain the keys 'height' and 'width'. Got {size}")
        grouped_images, grouped_images_index = group_images_by_shape(images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_rescale:
                stacked_images = stacked_images * rescale_factor
            if do_pad:
                # DPT default is no pad, but if enabled, pad to next multiple
                h, w = stacked_images.shape[-2:]
                pad_h = (ensure_multiple_of - h % ensure_multiple_of) % ensure_multiple_of
                pad_w = (ensure_multiple_of - w % ensure_multiple_of) % ensure_multiple_of
                if pad_h or pad_w:
                    stacked_images = F.pad(stacked_images, [0, 0, pad_w, pad_h], padding_mode="constant", value=0)
            if do_resize:
                target_size = (size["height"], size["width"])
                if keep_aspect_ratio:
                    # Resize keeping aspect ratio, then center crop/pad
                    h, w = stacked_images.shape[-2:]
                    scale = min(target_size[0] / h, target_size[1] / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    stacked_images = F.resize(stacked_images, [new_h, new_w], interpolation=interpolation)
                    pad_h = target_size[0] - new_h
                    pad_w = target_size[1] - new_w
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    stacked_images = F.pad(stacked_images, [pad_left, pad_top, pad_right, pad_bottom], padding_mode="constant", value=0)
                else:
                    stacked_images = F.resize(stacked_images, target_size, interpolation=interpolation)
            if do_normalize:
                mean = torch.tensor(image_mean, device=stacked_images.device).view(-1, 1, 1)
                std = torch.tensor(image_std, device=stacked_images.device).view(-1, 1, 1)
                stacked_images = (stacked_images - mean) / std
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_depth_estimation(
        self,
        outputs: "DPTDepthEstimatorOutput",
        target_sizes: Optional[Union[TensorType, List[Tuple[int, int]], None]] = None,
    ) -> List[Dict[str, TensorType]]:
        requires_backends(self, "torch")
        predicted_depth = outputs.predicted_depth
        if target_sizes is not None and len(predicted_depth) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the predicted depth")
        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                depth = torch.nn.functional.interpolate(
                    input=depth.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode=pil_torch_interpolation_mapping[self.resample].value,
                ).squeeze()
            results.append({"predicted_depth": depth})
        return results

__all__ = ["DPTImageProcessorFast"] 