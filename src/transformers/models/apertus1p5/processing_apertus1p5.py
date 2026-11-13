# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for Apertus 1.5."""

import numpy as np

from ...audio_utils import AudioInput
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring
from ...utils.import_utils import requires


@auto_docstring(custom_intro="Constructs an Apertus 1.5 processor for text, images, and audio.")
@requires(backends=("vision", "torch", "torchvision"))
class Apertus1p5Processor(ProcessorMixin):
    def __init__(self, image_processor=None, feature_extractor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = tokenizer.image_token
        self.audio_token = tokenizer.audio_token
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_wrapper_token = tokenizer.image_wrapper_token
        self.eol_token = tokenizer.eol_token
        self.boa_token = tokenizer.boa_token
        self.eoa_token = tokenizer.eoa_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template)

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos=None,
        audio: AudioInput | None = None,
        **kwargs,
    ):
        """Run the base validation and require the text that holds the media placeholders."""
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if text is None and (images is not None or audio is not None):
            raise ValueError("`text` with media placeholder tokens must be provided when passing images or audio.")

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        spatial_factor = kwargs.get("spatial_factor", self.image_processor.spatial_factor)
        grid_height, grid_width = (int(side) // spatial_factor for side in image_inputs["image_sizes"][image_idx])
        rows = self.eol_token.join([self.image_token * grid_width] * grid_height)
        return f"{self.boi_token}{grid_height}*{grid_width}{self.image_wrapper_token}{rows}{self.eoi_token}"

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_codes = int(audio_inputs["num_audio_codes"][audio_idx])
        return f"{self.boa_token}{self.audio_token * num_codes}{self.eoa_token}"

    def _process_audio(self, audio: AudioInput, **kwargs):
        """Peak-normalize the clips, extract features under the model's input names, and build the replacements."""
        # peak-normalize each clip to -3 dBFS in float32, as in the reference Apertus 1.5 pipeline
        # (the codec feature extractor itself deliberately performs no normalization)
        target_peak = 10.0 ** (-3.0 / 20.0)
        clips = []
        for clip in audio:
            clip = np.asarray(clip, dtype=np.float32)
            if clip.size > 0:
                clip = clip * (target_peak / max(float(np.abs(clip).max()), 1e-10))
            clips.append(clip)  # empty clips fall through to the feature extractor's clear error

        # bare arrays are assumed to already be 24 kHz mono, like the reference pipeline
        kwargs.setdefault("sampling_rate", self.feature_extractor.sampling_rate)
        audio_inputs = self.feature_extractor(clips, **kwargs)
        if "padding_mask" not in audio_inputs:
            # Unpadded waveforms contain only valid samples, including after truncation.
            audio_inputs["padding_mask"] = [
                np.ones(values.shape[-1], dtype=np.int64) for values in audio_inputs["input_values"]
            ]
            audio_inputs.convert_to_tensors(kwargs.get("return_tensors"))
        audio_inputs["input_features"] = audio_inputs.pop("input_values")
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("padding_mask")
        # counts come from the feature-extractor OUTPUT so that truncation/max_length can never desync the
        # placeholder count from the features the model will actually encode
        audio_inputs["num_audio_codes"] = [
            self.feature_extractor.get_num_audio_codes(int(np.asarray(mask).sum()))
            for mask in audio_inputs["feature_attention_mask"]
        ]

        audio_replacements = []
        for idx in range(len(clips)):
            audio_replacements.append(self.replace_audio_token(audio_inputs, audio_idx=idx))
        return audio_inputs, audio_replacements

    def _get_num_multimodal_tokens(self, image_sizes=None, audio_lengths=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Counts cover only the `<|image|>` / `<|audio|>` placeholder tokens (one per discrete code), not the
        surrounding structure tokens or the textual `H*W` size header. Audio counts assume untruncated clips;
        explicit `truncation`/`max_length` audio kwargs passed to `__call__` reduce the actual counts.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            audio_lengths (`list[int]`, *optional*):
                The number of audio samples (at 24 kHz) per each clip.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        data = {}
        if image_sizes is not None:
            images_kwargs = self._merge_kwargs(**kwargs)["images_kwargs"]
            data["num_image_tokens"] = [
                self.image_processor.get_number_of_image_patches(int(height), int(width), images_kwargs)
                for height, width in image_sizes
            ]
            data["num_image_patches"] = [1] * len(image_sizes)
        if audio_lengths is not None:
            data["num_audio_tokens"] = [
                self.feature_extractor.get_num_audio_codes(int(length)) for length in audio_lengths
            ]
        return MultiModalData(**data)

    @property
    def model_input_names(self):
        # the audio names are hardcoded because `_process_audio` renames the feature extractor's
        # `input_values`/`padding_mask` outputs to the model's `input_features`/`feature_attention_mask`
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        audio_input_names = ["input_features", "feature_attention_mask"]
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + audio_input_names))

    @property
    def unused_input_names(self) -> list[str]:
        """Input names returned always by subprocessors but not used in the model's `forward`"""
        return ["num_audio_codes"]


__all__ = ["Apertus1p5Processor"]
