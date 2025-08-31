# coding=utf-8
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
from typing import Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import AudioKwargs, ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class Gemma3nImagesKwargs(ImagesKwargs):
    do_convert_rgb: Optional[bool]


class Gemma3nProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: AudioKwargs
    images_kwargs: Gemma3nImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Gemma3nProcessor(ProcessorMixin):
    """
    A processor for Gemma 3n, wrapping the full capabilities of a feature extractor, image processor, and tokenizer
    into a single processor.

    Args:
        feature_extractor (`Gemma3nAudioFeatureExtractor`):
            Feature extractor that converts raw audio waveforms into MEL spectrograms for the audio encoder. This
            should return a `BatchFeature` with `input_features` and `input_features_mask` features.
        image_processor (`SiglipImageProcessorFast`):
            Image processor that prepares batches of images for the vision encoder. This should return a `BatchFeature`
            with a `pixel_values` feature.
        tokenizer (`GemmaTokenizerFast`):
            The text tokenizer for the model.
        chat_template (`string`, *optional*):
            A Jinja template for generating text prompts from a set of messages.
        audio_seq_length (int, *optional*, defaults to 188):
            The number of audio soft tokens that will be added to the text prompt
        image_seq_length (int, *optional*, defaults to 256):
            The number of image soft tokens that should be added to
    """

    attributes = ["feature_extractor", "image_processor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

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

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        audio: Optional[Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]]] = None,
        videos=None,
        **kwargs: Unpack[Gemma3nProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None and audio is None:
            raise ValueError("Provide at least one of `text`, `images`, or `audio`.")

        output_kwargs = self._merge_kwargs(
            Gemma3nProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        if audio is not None:
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

            if not text:
                text = [self.audio_token for _ in audio]

            # Expand placeholder audio tokens to the full audio token sequence
            text = [prompt.replace(self.audio_token, self.full_audio_sequence) for prompt in text]
        else:
            audio_inputs = {}

        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(batched_images, **output_kwargs["images_kwargs"])

            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.image_token] * len(images)) for images in batched_images]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Expand placeholder image tokens to the full image token sequence
            text = [prompt.replace(self.image_token, self.full_image_sequence) for prompt in text]
        else:
            image_inputs = {}

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"], return_tensors="np")
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        # Add token type ids manually, as tokenizer can't do arbitrary position token types
        array_ids = text_inputs["input_ids"]
        token_type_ids = np.zeros_like(array_ids)
        token_type_ids[array_ids == self.image_token_id] = 1
        token_type_ids[array_ids == self.audio_token_id] = 3
        text_inputs = {k: v.tolist() for k, v in text_inputs.items()}  # in case user requested list inputs
        text_inputs["token_type_ids"] = token_type_ids.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs, **audio_inputs}, tensor_type=return_tensors)


__all__ = ["Gemma3nProcessor"]
