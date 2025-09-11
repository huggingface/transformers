# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Image/Text processor class for SigLIP.
"""

from ...processing_utils import ProcessorMixin


class SiglipProcessor(ProcessorMixin):
    r"""
    Constructs a Siglip processor which wraps a Siglip image processor and a Siglip tokenizer into a single processor.

    [`SiglipProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`SiglipTokenizer`]. See the
    [`~SiglipProcessor.__call__`] and [`~SiglipProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`SiglipTokenizer`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("SiglipImageProcessor", "SiglipImageProcessorFast")
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)


__all__ = ["SiglipProcessor"]
