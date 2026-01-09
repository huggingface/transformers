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

import re
from typing import Optional, Union

from ...audio_utils import AudioInput
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
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

    @auto_docstring
    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        images: Optional[ImageInput] = None,
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **input_image_embeds** -- Pixel values to be fed to a model.
            - **image_sizes** -- List of tuples specifying the size of each image in `input_image_embeds`.
            - **image_attention_mask** -- List of attention masks for each image in `input_image_embeds`.
            - **input_audio_embeds** -- Audio embeddings to be fed to a model.
            - **audio_embed_sizes** -- List of integers specifying the size of each audio in `input_audio_embeds`.
        """

        output_kwargs = self._merge_kwargs(Phi4MultimodalProcessorKwargs, self.tokenizer.init_kwargs, **kwargs)
        image_kwargs = output_kwargs["images_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]

        image_inputs = self.image_processor(images, **image_kwargs) if images is not None else {}
        audio_inputs = self.audio_processor(audio, **audio_kwargs) if audio is not None else {}

        # We pop here for images as we don't need it later
        num_img_tokens = image_inputs.pop("num_img_tokens", [])
        audio_embed_sizes = audio_inputs.get("audio_embed_sizes", [])

        # Replace certain special tokens for compatibility
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        image_token = self.tokenizer.image_token
        audio_token = self.tokenizer.audio_token

        # Check that the number of special tokens is sound
        concatenated_prompt = "".join(text)
        if concatenated_prompt.count(image_token) != len(num_img_tokens):
            raise ValueError(
                "You should add as much image tokens `<|image|>` in your prompt as you pass `images` to the processor. ",
                f"Input contains {concatenated_prompt.count(image_token)} tokens != {len(num_img_tokens)} images",
            )
        if concatenated_prompt.count(audio_token) != len(audio_embed_sizes):
            raise ValueError(
                "You should add as much audio tokens `<|audio|>` in your prompt as you pass `audios` to the processor. "
                f"Input contains {concatenated_prompt.count(audio_token)} tokens != {len(audio_embed_sizes)} audios"
            )

        # Add appropriate number of image/audio tokens (note that the count of replacement is dynamic)
        image_count_iter = iter(num_img_tokens)
        audio_count_iter = iter(audio_embed_sizes)
        processed_text = [
            re.sub(re.escape(image_token), lambda _: image_token * next(image_count_iter), t) for t in text
        ]
        processed_text = [
            re.sub(re.escape(audio_token), lambda _: audio_token * next(audio_count_iter), t) for t in processed_text
        ]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(processed_text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(processed_text, text_inputs, modalities=["image"])

        # prepare batch feature
        data = {
            **text_inputs,
            **image_inputs,
            **audio_inputs,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Phi4MultimodalProcessor"]
