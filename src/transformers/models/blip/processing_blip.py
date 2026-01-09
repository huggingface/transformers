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
Processor class for Blip.
"""

from typing import Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import auto_docstring


class BlipProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
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
class BlipProcessor(ProcessorMixin):
    def __init__(self, image_processor, tokenizer, **kwargs):
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[str, list[str], TextInput, PreTokenizedInput]] = None,
        **kwargs: Unpack[BlipProcessorKwargs],
    ) -> BatchEncoding:
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        text_encoding = None

        # add pixel_values encoding. If we also have text_encoding, update image encoding and return it.
        # else, return the text encoding.
        output_kwargs = self._merge_kwargs(
            BlipProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if text is not None:
            text_encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if images is not None:
            encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])

            if text_encoding is not None:
                encoding_image_processor.update(text_encoding)
            return encoding_image_processor

        return text_encoding

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        tokenizer_input_names = [name for name in tokenizer_input_names if name != "token_type_ids"]
        return tokenizer_input_names + image_processor_input_names


__all__ = ["BlipProcessor"]
