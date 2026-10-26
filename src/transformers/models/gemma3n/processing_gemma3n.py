# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class Gemma3nProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_mm_token_type_ids": True},
    }


@auto_docstring
class Gemma3nProcessor(ProcessorMixin):
    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        chat_template=None,
        audio_seq_length: int = 188,
        image_seq_length: int = 256,
        **kwargs,
    ):
        r"""
        audio_seq_length (int, *optional*, defaults to 188):
            The number of audio soft tokens that will be added to the text prompt
        image_seq_length (int, *optional*, defaults to 256):
            The number of image soft tokens that should be added to
        """
        self.audio_seq_length = audio_seq_length
        self.audio_token_id = tokenizer.audio_token_id
        self.boa_token = tokenizer.boa_token
        self.audio_token = tokenizer.audio_token
        audio_tokens_expanded = "".join([tokenizer.audio_token] * audio_seq_length)
        self.full_audio_sequence = f"\n\n{tokenizer.boa_token}{audio_tokens_expanded}{tokenizer.eoa_token}\n\n"

        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.image_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"

        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]] | None = None,
        **kwargs: Unpack[Gemma3nProcessorKwargs],
    ) -> BatchFeature:
        model_inputs = super().__call__(images=images, text=text, audio=audio, **kwargs)
        if "mm_token_type_ids" in model_inputs:
            model_inputs["token_type_ids"] = model_inputs.pop("mm_token_type_ids")
        return model_inputs

    def prepare_inputs_layout(self, images=None, text=None, audio=None, **kwargs):
        images, text, *_ = super().prepare_inputs_layout(images=images, text=text, audio=audio, **kwargs)
        if images is not None:
            images = make_nested_list_of_images(images)
            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.image_token] * len(image_list)) for image_list in images]

        if audio is not None and not text:
            text = " ".join([self.audio_token] * len(audio))
        return images, text, None, None

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        audio_processor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = [name for name in image_processor_input_names if name != "num_crops"]
        return list(tokenizer_input_names + image_processor_input_names + audio_processor_input_names)


__all__ = ["Gemma3nProcessor"]
