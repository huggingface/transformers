# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Generation utilities for Nemotron3_5Asr RNN-T models.

Nemotron3_5Asr is the multilingual extension of [`NemotronAsrStreaming`]; the RNN-T generation machinery
(offline + cache-aware streaming) is identical, so this reuses [`NemotronAsrStreamingGenerationMixin`]
wholesale and only adds language-ID prompt conditioning: the target language is fixed for a whole
utterance/stream, so `generate` stashes `prompt_ids` once and `get_audio_features` (in the modeling file)
reads it for both the offline encode and every streaming chunk.
"""

from ..nemotron_asr_streaming.generation_nemotron_asr_streaming import (
    NemotronAsrStreamingGenerationMixin,
    NemotronAsrStreamingRNNTDecoderCache,
)


class Nemotron3_5AsrRNNTDecoderCache(NemotronAsrStreamingRNNTDecoderCache): ...


class Nemotron3_5AsrGenerationMixin(NemotronAsrStreamingGenerationMixin):
    def generate(self, inputs=None, generation_config=None, **kwargs):
        # The target language is fixed for the whole utterance/stream; stash it once so that
        # `get_audio_features` can read it for the offline encode and every streaming chunk.
        self._prompt_ids = kwargs.pop("prompt_ids", None)
        try:
            return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
        finally:
            if hasattr(self, "_prompt_ids"):
                del self._prompt_ids
