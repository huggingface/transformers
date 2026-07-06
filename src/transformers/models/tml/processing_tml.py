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


from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class TmlProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class TmlProcessor(ProcessorMixin):
    valid_processor_kwargs = TmlProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = "<|content_image|>" # tokenizer.image_token
        self.image_token_id = tokenizer.encode(self.image_token, add_special_tokens=False)[0] # tokenizer.image_token_id
        self.audio_token = "<|content_audio|>" # tokenizer.audio_token
        self.audio_token_id = tokenizer.encode(self.audio_token, add_special_tokens=False)[0] # tokenizer.audio_token_id
        self.eoa_token_id = "<|audio_end|>" # tokenizer.eoa_token_id  # where is it used???

        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        num_soft_tokens = image_inputs["num_patches"][image_idx]
        return self.image_token * num_soft_tokens

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        num_soft_tokens = audio_inputs["num_audio_tokens"][audio_idx]
        return self.audio_token * num_soft_tokens


__all__ = ["TmlProcessor"]
