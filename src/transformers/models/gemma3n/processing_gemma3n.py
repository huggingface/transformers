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
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class Gemma3nProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
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
        if text is None and images is None and audio is None:
            raise ValueError("Provide at least one of `text`, `images`, or `audio`.")

        output_kwargs = self._merge_kwargs(
            Gemma3nProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        return_text_replacement_offsets = output_kwargs["text_kwargs"].pop("return_text_replacement_offsets", False)

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) or not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        if audio is not None:
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

            if not text:
                text = [self.audio_token for _ in audio]

            audio_replacements = [self.full_audio_sequence] * len(audio_inputs["input_features"])
        else:
            audio_inputs = {}
            audio_replacements = []

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

            images_replacements = [self.full_image_sequence] * sum(len(batch) for batch in batched_images)
        else:
            image_inputs = {}
            images_replacements = []

        text, text_replacement_offsets = self.get_text_with_replacements(
            list(text),
            images_replacements=images_replacements,
            audio_replacements=audio_replacements,
        )

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
        if return_text_replacement_offsets:
            text_inputs["text_replacement_offsets"] = text_replacement_offsets
        return BatchFeature(
            data={**text_inputs, **image_inputs, **audio_inputs},
            tensor_type=return_tensors,
            skip_tensor_conversion=self.skip_tensor_conversion,
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, audio_lengths=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            audio_lengths (`list[int]`, *optional*):
                The lengths of audio inputs in number of samples.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            # NOTE: No image cropping supported yet
            num_image_tokens = [self.image_seq_length] * len(image_sizes)
            num_image_patches = [1] * len(image_sizes)

            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if audio_lengths is not None:
            # NOTE: Every audio is padded or truncated to a fixed window
            vision_data["num_audio_tokens"] = [self.audio_seq_length] * len(audio_lengths)

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        audio_processor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = [name for name in image_processor_input_names if name != "num_crops"]
        return list(tokenizer_input_names + image_processor_input_names + audio_processor_input_names)


__all__ = ["Gemma3nProcessor"]
