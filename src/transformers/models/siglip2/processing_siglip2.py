# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Image/Text processor class for SigLIP2.
"""

from typing import Optional

from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin


class Siglip2ImagesKwargs(ImagesKwargs, total=False):
    max_num_patches: Optional[int]
    patch_size: Optional[int]


class Siglip2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Siglip2ImagesKwargs

    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "truncation": True,
            "max_length": 64,
        },
        "images_kwargs": {
            "max_num_patches": 256,
            "patch_size": 16,
        },
    }


class Siglip2Processor(ProcessorMixin):
    r"""
    Constructs a Siglip2 processor which wraps a Siglip2 image processor and a Gemma tokenizer into a single processor.

    [`Siglip2Processor`] offers all the functionalities of [`Siglip2ImageProcessor`] and [`GemmaTokenizerFast`]. See the
    [`~Siglip2Processor.__call__`] and [`~Siglip2Processor.decode`] for more information.

    Args:
        image_processor ([`Siglip2ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`GemmaTokenizerFast`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_processor_kwargs = Siglip2ProcessorKwargs

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)


__all__ = ["Siglip2Processor"]
