# coding=utf-8
# Copyright 2024 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from ..gemma.configuration_gemma import GemmaConfig
from ..gemma.modeling_gemma import (
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
)
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaModel,
    repeat_kv,
)
from ..phi3.modeling_phi3 import (
    Phi3MLP,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
)


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


class GlmConfig(GemmaConfig):
    """
    resid_pdrop (`float`, *optional*, defaults to `0.0`):
        Dropout ratio in the decoder layers.
    linear_bias (`bool`, *optional*, defaults to `False`):
        Whether to use a bias in the MLP layers, as well as the query, key, value and output projection layers during self-attention.
    """

    model_type = "glm"

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=13696,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        hidden_act="silu",
        resid_pdrop=0.0,
        attention_dropout=0.0,
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=0.00000015625,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        pad_token_id=151329,
        eos_token_id=[151329, 151336, 151338],
        bos_token_id=None,
        attention_bias=True,
        linear_bias=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.resid_pdrop = resid_pdrop
        self.linear_bias = linear_bias
        del self.hidden_activation


class GlmRMSNorm(Phi3RMSNorm):
    pass


class GlmRotaryEmbedding(Phi3RotaryEmbedding):
    pass


class GlmMLP(Phi3MLP):
    def __init__(self, config):
        super().__init__(config)

        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=self.config.linear_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=self.config.linear_bias)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    # Keep half for later concatenation
    q, q_pass = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    k, k_pass = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]

    # Apply rotary embeddings on the first half
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class GlmAttention(LlamaAttention):
    def __init__(self, config: GlmConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.linear_bias)
        # Not used in the attention, only for BC
        self.rotary_emb = GlmRotaryEmbedding(
            dim=config.head_dim // 2, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )


class GlmFlashAttention2(GlmAttention, LlamaFlashAttention2):
    pass


class GlmSdpaAttention(GlmAttention, LlamaSdpaAttention):
    pass

GLM_ATTENTION_CLASSES = {
    "eager": GlmAttention,
    "flash_attention_2": GlmFlashAttention2,
    "sdpa": GlmSdpaAttention,
}


class GlmDecoderLayer(nn.Module):
    def __init__(self, config: GlmConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.self_attn = GLM_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)

        self.mlp = GlmMLP(config)
        self.input_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GlmModel(LlamaModel):
    def __init__(self, config: GlmConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GlmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmRotaryEmbedding(
            dim=config.head_dim // 2, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class GlmForCausalLM(GemmaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()


class GlmForSequenceClassification(GemmaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()


class GlmForTokenClassification(GemmaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()
