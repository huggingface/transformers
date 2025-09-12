# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
"""
Image/Text processor class for FLAVA
"""

import warnings
from collections.abc import Iterable
from typing import Optional, Union

from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin


class FlavaImagesKwargs(ImagesKwargs):
    # Mask related params
    return_image_mask: Optional[bool]
    input_size_patches: Optional[int]
    total_mask_patches: Optional[int]
    mask_group_min_patches: Optional[int]
    mask_group_max_patches: Optional[int]
    mask_group_min_aspect_ratio: Optional[float]
    mask_group_max_aspect_ratio: Optional[float]
    # Codebook related params
    return_codebook_pixels: Optional[bool]
    codebook_do_resize: Optional[bool]
    codebook_size: Optional[bool]
    codebook_resample: Optional[int]
    codebook_do_center_crop: Optional[bool]
    codebook_crop_size: Optional[int]
    codebook_do_rescale: Optional[bool]
    codebook_rescale_factor: Optional[Union[int, float]]
    codebook_do_map_pixels: Optional[bool]
    codebook_do_normalize: Optional[bool]
    codebook_image_mean: Optional[Union[float, Iterable[float]]]
    codebook_image_std: Optional[Union[float, Iterable[float]]]


class FlavaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: FlavaImagesKwargs
    _defaults = {}


class FlavaProcessor(ProcessorMixin):
    r"""
    Constructs a FLAVA processor which wraps a FLAVA image processor and a FLAVA tokenizer into a single processor.

    [`FlavaProcessor`] offers all the functionalities of [`FlavaImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~FlavaProcessor.__call__`] and [`~FlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`FlavaImageProcessor`], *optional*): The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*): The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FlavaImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    valid_processor_kwargs = FlavaProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor


__all__ = ["FlavaProcessor"]
