# Copyright 2026 KORMo Team and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch KORMo model.

KORMo (Korean Open Reasoning Model) is architecturally identical to Llama, except that the
two decoder-layer RMSNorms are named ``pre_attention_layernorm`` / ``pre_mlp_layernorm``
(Llama uses ``input_layernorm`` / ``post_attention_layernorm``). Keeping these names lets the
existing KORMo checkpoints load with no weight renaming.
"""

import torch
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="KORMo-Team/KORMo-10B-base")
@strict
class KORMoConfig(LlamaConfig):
    r"""
    ```python
    >>> from transformers import KORMoModel, KORMoConfig

    >>> # Initializing a KORMo-10B style configuration
    >>> configuration = KORMoConfig()

    >>> # Initializing a model from the KORMo-10B style configuration
    >>> model = KORMoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kormo"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }


class KORMoRMSNorm(LlamaRMSNorm):
    pass


class KORMoRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class KORMoMLP(LlamaMLP):
    pass


class KORMoAttention(LlamaAttention):
    pass


class KORMoDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: KORMoConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # KORMo names its norms pre_attention_layernorm / pre_mlp_layernorm
        # (Llama: input_layernorm / post_attention_layernorm).
        self.pre_attention_layernorm = KORMoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_layernorm = KORMoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        del self.input_layernorm
        del self.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_attention_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class KORMoPreTrainedModel(LlamaPreTrainedModel):
    pass


class KORMoModel(LlamaModel):
    pass


class KORMoForCausalLM(LlamaForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, KORMoForCausalLM

        >>> model = KORMoForCausalLM.from_pretrained("KORMo-Team/KORMo-10B-sft")
        >>> tokenizer = AutoTokenizer.from_pretrained("KORMo-Team/KORMo-10B-sft")

        >>> messages = [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}]
        >>> inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        >>> generate_ids = model.generate(inputs, max_new_tokens=32)
        >>> tokenizer.batch_decode(generate_ids[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        '대한민국의 수도는 서울입니다.'
        ```"""
        return super().forward(**super_kwargs)


class KORMoForSequenceClassification(LlamaForSequenceClassification):
    pass


class KORMoForTokenClassification(LlamaForTokenClassification):
    pass


class KORMoForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "KORMoConfig",
    "KORMoPreTrainedModel",
    "KORMoModel",
    "KORMoForCausalLM",
    "KORMoForSequenceClassification",
    "KORMoForTokenClassification",
    "KORMoForQuestionAnswering",
]
