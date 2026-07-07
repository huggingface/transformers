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
"""
Processor class for DeepSeek-OCR-2.
"""

import math

from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class DeepseekOcr2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
            "min_patches": 2,
            "max_patches": 6,
        },
    }


@auto_docstring
class DeepseekOcr2Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        patch_size=16,
        downsample_ratio=4,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to `16`):
            The patch size used by the vision encoder (SAM ViT patch embedding size).
        downsample_ratio (`int`, *optional*, defaults to `4`):
            The downsampling ratio applied after the vision encoder.
        """
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> TextInput:
        size = self.image_processor.size["height"]
        tile_size = self.image_processor.tile_size

        num_queries_global = math.ceil(size / self.patch_size / self.downsample_ratio)
        global_tokens = num_queries_global * num_queries_global

        num_queries_local = math.ceil(tile_size / self.patch_size / self.downsample_ratio)
        local_tokens = num_queries_local * num_queries_local

        num_crops = image_inputs["num_local_patches"][image_idx]
        num_tokens = global_tokens + local_tokens * num_crops + 1
        return self.image_token * num_tokens


__all__ = ["DeepseekOcr2Processor"]
