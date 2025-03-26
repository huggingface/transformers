# coding=utf-8
# Copyright 2025 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for Kosmos2_5."""

from typing import Dict, List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast, DefaultFastImageProcessorKwargs, BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import add_start_docstrings


class Kosmos2_5FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    # Q: should we use `SizeDict`?
    patch_size: Optional[Dict[str, int]]
    max_patches: Optional[int]


@add_start_docstrings(
    "Constructs a fast Kosmos2_5 image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image. According to Kosmos2_5 paper and code, the patch size is 16x16.
        max_patches (`int`, *optional*, defaults to 4096):
            The maximum number of patches to extract from the image as per the
            [KOSMOS 2.5 paper](https://arxiv.org/pdf/2309.11419).
    """,
)
class Kosmos2_5ImageProcessorFast(BaseImageProcessorFast):
    # To be checked against the slow image processor
    # None values left after checking can be removed
    do_normalize = True
    do_convert_rgb = True
    patch_size = {"height": 16, "width": 16}
    max_patches = 4096
    valid_kwargs = Kosmos2_5FastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Kosmos2_5FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image. According to Kosmos2_5 paper and code, the patch size is 16x16.
        max_patches (`int`, *optional*, defaults to 4096):
            The maximum number of patches to extract from the image as per the
            [KOSMOS 2.5 paper](https://arxiv.org/pdf/2309.11419).
        """,
    )
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Kosmos2_5FastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)


__all__ = ["Kosmos2_5ImageProcessorFast"]
