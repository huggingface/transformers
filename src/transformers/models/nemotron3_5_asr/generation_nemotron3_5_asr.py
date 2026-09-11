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

from ..nemotron_asr_streaming.generation_nemotron_asr_streaming import (
    NemotronAsrStreamingGenerationMixin,
    NemotronAsrStreamingRNNTDecoderCache,
)


class Nemotron3_5AsrRNNTDecoderCache(NemotronAsrStreamingRNNTDecoderCache): ...


class Nemotron3_5AsrGenerationMixin(NemotronAsrStreamingGenerationMixin):
    """Generation mixin for Nemotron 3.5 ASR RNN-T models.

    Same decoding as [`NemotronAsrStreamingGenerationMixin`], offline and streaming. The language-conditioning
    `prompt_ids` passed to `generate` stay in `model_kwargs` and reach every encoder call through
    `_encoder_kwargs`; nothing is stored on the model.
    """
