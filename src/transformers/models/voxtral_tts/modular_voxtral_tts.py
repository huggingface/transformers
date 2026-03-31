# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralMLP,
    MistralModel,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralRotaryEmbedding,
)


class VoxtralTtsMLP(MistralMLP):
    pass


class VoxtralTtsAttention(MistralAttention):
    pass


class VoxtralTtsRMSNorm(MistralRMSNorm):
    pass


class VoxtralTtsDecoderLayer(MistralDecoderLayer):
    pass


class VoxtralTtsPreTrainedModel(MistralPreTrainedModel):
    pass


class VoxtralTtsRotaryEmbedding(MistralRotaryEmbedding):
    pass


class VoxtralTtsModel(MistralModel):
    pass


class VoxtralTtsForCausalLM(MistralForCausalLM):
    pass


__all__ = [
    "VoxtralTtsForCausalLM",
    "VoxtralTtsModel",
    "VoxtralTtsPreTrainedModel",
]
