# Copyright 2025 the HuggingFace Team. All rights reserved.
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


import torch

from ...cache_utils import Cache
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import Gemma2Attention, Gemma2DecoderLayer, Gemma2ForCausalLM, Gemma2MLP, Gemma2RMSNorm


@auto_docstring(checkpoint="google/vaultgemma-1b")
class VaultGemmaConfig(Gemma2Config):
    def __init__(
        self,
        vocab_size: int | None = 256000,
        hidden_size: int | None = 2304,
        intermediate_size: int | None = 9216,
        num_hidden_layers: int | None = 26,
        num_attention_heads: int | None = 8,
        num_key_value_heads: int | None = 4,
        head_dim: int | None = 256,
        hidden_activation: str | None = "gelu_pytorch_tanh",
        max_position_embeddings: int | None = 8192,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = 0,
        eos_token_id: int | None = 1,
        bos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        query_pre_attn_scalar: int | None = 256,
        sliding_window: int | None = 4096,
        layer_types: list[str] | None = None,
        final_logit_softcapping: float | None = 30.0,
        attn_logit_softcapping: float | None = 50.0,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_activation=hidden_activation,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            query_pre_attn_scalar=query_pre_attn_scalar,
            sliding_window=sliding_window,
            layer_types=layer_types,
            final_logit_softcapping=final_logit_softcapping,
            attn_logit_softcapping=attn_logit_softcapping,
            **kwargs,
        )

        del self.use_bidirectional_attention


class VaultGemmaRMSNorm(Gemma2RMSNorm):
    pass


class VaultGemmaMLP(Gemma2MLP):
    pass


class VaultGemmaAttention(Gemma2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: VaultGemmaConfig, layer_idx: int):
        super().__init__()
        self.is_causal = True


class VaultGemmaDecoderLayer(Gemma2DecoderLayer):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)
        del self.post_attention_layernorm
        del self.post_feedforward_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VaultGemmaForCausalLM(Gemma2ForCausalLM):
    pass


__all__ = [
    "VaultGemmaConfig",
    "VaultGemmaForCausalLM",
    "VaultGemmaModel",  # noqa: F822
    "VaultGemmaPreTrainedModel",  # noqa: F822
]
