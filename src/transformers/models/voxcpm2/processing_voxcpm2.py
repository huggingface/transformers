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

from ...processing_utils import ProcessorMixin


class VoxCPM2Processor(ProcessorMixin):
    """Builds mixed text and audio prompts for VoxCPM2 generation."""

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "VoxCPM2Tokenizer"

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


__all__ = ["VoxCPM2Processor"]
