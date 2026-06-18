# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for BLIP-2.
"""

from ...image_utils import ImageInput
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class Blip2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
    }


@auto_docstring
class Blip2Processor(ProcessorMixin):
    valid_processor_kwargs = Blip2ProcessorKwargs

    def __init__(self, image_processor, tokenizer, num_query_tokens=None, **kwargs):
        r"""
        num_query_tokens (`int`, *optional*):
            Number of tokens used by the Qformer as queries, should be same as in model's config.
        """
        tokenizer.return_token_type_ids = False
        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken("<image>", normalized=False, special=True)
            tokenizer.add_tokens([image_token], special_tokens=True)
            self.image_token = image_token.content
        else:
            self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.num_query_tokens = num_query_tokens

        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[Blip2ProcessorKwargs],
    ):
        kwargs["add_special_tokens"] = False
        model_inputs = super().__call__(images=images, text=text, **kwargs)
        return model_inputs

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(images=images, text=text)

        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")

    def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
        images, text, videos, audio = super().prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        if text is not None and images is not None and self.num_query_tokens is not None:
            text = [self.image_token + sample for sample in text]
        return images, text, videos, audio

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        # TODO: add BOS token manualy after adding placeholders
        # since they have to be added after placeholders
        return self.image_token * self.num_query_tokens


__all__ = ["Blip2Processor"]
