# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Image/Text processor class for XCLIP
"""

import warnings

from ...processing_utils import ProcessorMixin


class XCLIPProcessor(ProcessorMixin):
    r"""
    Constructs an X-CLIP processor which wraps a VideoMAE image processor and a CLIP tokenizer into a single processor.

    [`XCLIPProcessor`] offers all the functionalities of [`VideoMAEImageProcessor`] and [`CLIPTokenizerFast`]. See the
    [`~XCLIPProcessor.__call__`] and [`~XCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`VideoMAEImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "VideoMAEImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

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
        self.video_processor = self.image_processor
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


__all__ = ["XCLIPProcessor"]
