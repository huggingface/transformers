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

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin


class VoxCPM2Processor(ProcessorMixin):
    """Builds mixed text and audio prompts for VoxCPM2 generation."""

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_patch_size=2560,
        audio_placeholder_token_id=0,
        audio_start_token="<|audio_start|>",
        reference_audio_start_token="<|audio_prompt_start|>",
        reference_audio_end_token="<|audio_prompt_end|>",
    ):
        if audio_patch_size <= 0:
            raise ValueError("`audio_patch_size` must be greater than zero")

        self.audio_patch_size = audio_patch_size
        self.audio_placeholder_token_id = audio_placeholder_token_id
        self.audio_start_token = audio_start_token
        self.reference_audio_start_token = reference_audio_start_token
        self.reference_audio_end_token = reference_audio_end_token
        self.audio_start_token_id = tokenizer.convert_tokens_to_ids(audio_start_token)
        self.reference_audio_start_token_id = tokenizer.convert_tokens_to_ids(reference_audio_start_token)
        self.reference_audio_end_token_id = tokenizer.convert_tokens_to_ids(reference_audio_end_token)
        super().__init__(feature_extractor, tokenizer)

    def _validate_generation_inputs(self, text, audio, prompt_text, reference_audio):
        if not isinstance(text, str):
            raise TypeError(
                "`text` must be a single string because VoxCPM2 generation currently supports batch size 1"
            )
        if prompt_text is not None and not isinstance(prompt_text, str):
            raise TypeError("`prompt_text` must be a string")
        if (audio is None) != (prompt_text is None):
            raise ValueError("`audio` and `prompt_text` must be provided together")
        if audio is None and reference_audio is None and not text:
            raise ValueError("`text` cannot be empty for zero-shot generation")

    def _build_generation_sequence(self, text_token_ids, prompt_audio_patches=0, reference_audio_patches=0):
        if prompt_audio_patches < 0 or reference_audio_patches < 0:
            raise ValueError("The number of audio patches must be non-negative")

        input_ids = list(text_token_ids) + [self.audio_start_token_id]
        text_mask = [1] * len(input_ids)
        if reference_audio_patches:
            reference_prefix = [self.reference_audio_start_token_id]
            reference_prefix += [self.audio_placeholder_token_id] * reference_audio_patches
            reference_prefix += [self.reference_audio_end_token_id]
            input_ids = reference_prefix + input_ids
            text_mask = [1] + [0] * reference_audio_patches + [1] + text_mask
        if prompt_audio_patches:
            input_ids += [self.audio_placeholder_token_id] * prompt_audio_patches
            text_mask += [0] * prompt_audio_patches

        return {
            "input_ids": [input_ids],
            "attention_mask": [[1] * len(input_ids)],
            "text_mask": [text_mask],
            "audio_mask": [[1 - value for value in text_mask]],
        }

    def _prepare_audio(self, audio, prefix, sampling_rate, return_tensors):
        if prefix not in {"prompt", "reference"}:
            raise ValueError("`prefix` must be either 'prompt' or 'reference'")

        audio_inputs = self.feature_extractor(
            audio,
            padding=True,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors,
        )
        input_values = audio_inputs["input_values"]
        padding_mask = audio_inputs["padding_mask"]
        if len(padding_mask) != 1:
            raise ValueError("VoxCPM2 generation currently supports one audio sample at a time")

        num_samples = int(np.asarray(padding_mask[0].tolist()).sum())
        if num_samples == 0:
            raise ValueError("Audio inputs must contain at least one sample")
        num_patches = (num_samples + self.audio_patch_size - 1) // self.audio_patch_size
        return {
            f"{prefix}_input_values": input_values,
            f"{prefix}_attention_mask": padding_mask,
        }, num_patches

    def __call__(
        self,
        text,
        audio=None,
        prompt_text=None,
        reference_audio=None,
        sampling_rate=None,
        return_tensors="pt",
    ):
        self._validate_generation_inputs(text, audio, prompt_text, reference_audio)
        text_to_tokenize = f"{prompt_text}{text}" if prompt_text is not None else text
        text_token_ids = self.tokenizer(text_to_tokenize, add_special_tokens=False).input_ids

        prompt_inputs = {}
        prompt_audio_patches = 0
        if audio is not None:
            prompt_inputs, prompt_audio_patches = self._prepare_audio(
                audio,
                prefix="prompt",
                sampling_rate=sampling_rate,
                return_tensors=return_tensors,
            )

        reference_inputs = {}
        reference_audio_patches = 0
        if reference_audio is not None:
            reference_inputs, reference_audio_patches = self._prepare_audio(
                reference_audio,
                prefix="reference",
                sampling_rate=sampling_rate,
                return_tensors=return_tensors,
            )

        sequence_inputs = self._build_generation_sequence(
            text_token_ids,
            prompt_audio_patches=prompt_audio_patches,
            reference_audio_patches=reference_audio_patches,
        )
        return BatchFeature(
            data={**sequence_inputs, **prompt_inputs, **reference_inputs},
            tensor_type=return_tensors,
        )

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "text_mask",
            "audio_mask",
            "prompt_input_values",
            "prompt_attention_mask",
            "reference_input_values",
            "reference_attention_mask",
        ]


__all__ = ["VoxCPM2Processor"]
