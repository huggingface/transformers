# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
Processor class for Phi4Multimodal
"""

from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class Phi4MultimodalProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "device": "cpu",
        },
    }


@auto_docstring
class Phi4MultimodalProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor,
        audio_processor,
        tokenizer,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        self.audio_token = tokenizer.audio_token
        self.audio_token_id = tokenizer.audio_token_id
        super().__init__(image_processor, audio_processor, tokenizer, **kwargs)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        num_img_tokens = image_inputs["num_img_tokens"][image_idx]
        return self.image_token * num_img_tokens

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        audio_embed_sizes = audio_inputs["audio_embed_sizes"][audio_idx]
        return self.audio_token * audio_embed_sizes


__all__ = ["Phi4MultimodalProcessor"]
