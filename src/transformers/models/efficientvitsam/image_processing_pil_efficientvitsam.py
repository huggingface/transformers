# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""PIL image processor for EfficientViT-SAM."""

import numpy as np

from ...image_processing_utils import get_size_dict
from ...image_utils import SizeDict
from ...processing_utils import Unpack
from ...utils import is_torch_available
from ..sam.image_processing_pil_sam import SamImageProcessorKwargs, SamImageProcessorPil


class EfficientvitsamImageProcessorPilKwargs(SamImageProcessorKwargs, total=False):
    prompt_size: dict[str, int]


class EfficientvitsamImageProcessorPil(SamImageProcessorPil):
    valid_kwargs = EfficientvitsamImageProcessorPilKwargs
    size = {"longest_edge": 512}
    pad_size = {"height": 512, "width": 512}
    prompt_size = {"longest_edge": 1024}

    def __init__(
        self, prompt_size: dict[str, int] | None = None, **kwargs: Unpack[EfficientvitsamImageProcessorPilKwargs]
    ):
        super().__init__(**kwargs)
        self.prompt_size = prompt_size if prompt_size is not None else {"longest_edge": 1024}

    def _standardize_kwargs(self, prompt_size: int | dict[str, int] | SizeDict | None = None, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        if prompt_size is None:
            prompt_size = self.prompt_size
        if not isinstance(prompt_size, SizeDict):
            prompt_size = SizeDict(**get_size_dict(prompt_size, default_to_square=False, param_name="prompt_size"))
        kwargs["prompt_size"] = prompt_size
        return kwargs

    def post_process_masks(
        self,
        masks,
        original_sizes,
        reshaped_input_sizes,
        mask_threshold=0.0,
        binarize=True,
        prompt_size=None,
    ):
        if not is_torch_available():
            raise ImportError("PyTorch is required for post_process_masks")
        import torch
        import torch.nn.functional as F

        prompt_size = self.prompt_size if prompt_size is None else prompt_size
        prompt_size = prompt_size["longest_edge"] if isinstance(prompt_size, dict) else prompt_size.longest_edge
        target_image_size = (prompt_size, prompt_size)

        if isinstance(original_sizes, torch.Tensor | np.ndarray):
            original_sizes = original_sizes.tolist()
        if isinstance(reshaped_input_sizes, torch.Tensor | np.ndarray):
            reshaped_input_sizes = reshaped_input_sizes.tolist()

        output_masks = []
        for i, original_size in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])
            elif not isinstance(masks[i], torch.Tensor):
                raise TypeError("Input masks should be a list of `torch.tensors` or a list of `np.ndarray`")
            interpolated_mask = F.interpolate(masks[i], target_image_size, mode="bilinear", align_corners=False)
            interpolated_mask = interpolated_mask[..., : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1]]
            interpolated_mask = F.interpolate(interpolated_mask, original_size, mode="bilinear", align_corners=False)
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)

        return output_masks


__all__ = ["EfficientvitsamImageProcessorPil"]
