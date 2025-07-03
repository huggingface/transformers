# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_ernie4_5 import Ernie4_5Config


logger = logging.get_logger(__name__)


# llama copy
class Ernie4_5RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Ernie4_5RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# llama copy
class Ernie4_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Ernie4_5RopeEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation for transformer models.

    RoPE encodes absolute positional information with rotation matrices and
    naturally incorporates relative position information in self-attention.

    Args:
        head_dim (int): Dimension size of each attention head
        compression_ratio (float, optional): Sequence length compression ratio. Defaults to 1.0.
        base (int, optional): Base value for frequency calculation. Defaults to 10000.

    Attributes:
        head_dim (int): Dimension size of each attention head
        compression_ratio (float): Sequence length compression factor
        base (int): Base value for frequency calculation
    """

    def __init__(self, head_dim, compression_ratio=1.0, base=10000):
        """
        Initialize RoPE embedding layer.

        Args:
            head_dim: Dimension of each attention head
            compression_ratio: Scaling factor for position indices
            base: Base value for frequency calculation
        """
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.base = base

    def forward(self, seq_length, position_ids=None):
        """
        Compute rotary position embeddings for given sequence length.

        Args:
            seq_length (int): Maximum sequence length
            position_ids (Tensor, optional): Custom position indices. Defaults to None.

        Returns:
            Tensor: Rotary position embeddings of shape [1, 1, seq_length, head_dim]
        """
        indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        indices = 1 / self.base ** (indices / self.head_dim)
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, 1, dtype=torch.float32
            ).unsqueeze(1)
            position_ids = position_ids / self.compression_ratio
            sinusoid_inp = position_ids * indices.unsqueeze(0)
        else:
            position_ids = position_ids / self.compression_ratio
            seq_length = position_ids.shape[-1]
            sinusoid_inp = position_ids.unsqueeze(-1).to(
                torch.float32
            ) * indices.unsqueeze(0)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(-1, 1, seq_length, self.head_dim)
        pos_emb = pos_emb.detach()
        return pos_emb

    def apply_rotary(self, rp, q, k):
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            rp (Tensor): Rotary position embeddings
            q (Tensor): Query tensor [batch, heads, seq_len, dim]
            k (Tensor): Key tensor [batch, heads, seq_len, dim]

        Returns:
            Tuple[Tensor, Tensor]: Rotated queries and keys
        """
        sin, cos = torch.chunk(rp.to(q.device), 2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape(rp.shape)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape(rp.shape)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = torch.stack(
            [-q[:, :, :, 1::2], q[:, :, :, 0::2]], dim=-1
        ).reshape(q.shape)
        query = (q.to(torch.float32) * cos_pos) + (
            rotate_half_q.to(torch.float32) * sin_pos
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = torch.stack(
            [-k[:, :, :, 1::2], k[:, :, :, 0::2]], dim=-1
        ).reshape(k.shape)
        key = (k.to(torch.float32) * cos_pos) + (
            rotate_half_k.to(torch.float32) * sin_pos
        )
        return query, key


# llama copy
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# llama copy except for the (on the fly) causal mask
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    ###
    # Causal mask
    seq_len = attn_weights.size(-1)
    mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=attn_weights.device),
        diagonal=1,
    )
    attn_weights = attn_weights.masked_fill(mask, float("-inf"))
    ###

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Ernie4_5Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        """Initialize the attention layer.

        Args:
            config: Model configuration.
            layer_idx (int, optional): Index in transformer stack. Defaults to 0.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        if config.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = config.head_dim

        self.is_gqa = (
            self.num_key_value_heads is not None
            and self.num_key_value_heads != self.num_heads
        )

        if self.is_gqa:
            logger.info(
                f"use GQA - num_heads: {self.num_heads}- num_key_value_heads: {self.num_key_value_heads}"
            )
            assert (
                self.num_heads % self.num_key_value_heads == 0
            ), f"num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}"
            kv_hidden_size = self.head_dim * self.num_key_value_heads
            q_hidden_size = self.head_dim * self.num_heads
        else:
            q_hidden_size = kv_hidden_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, q_hidden_size, bias=config.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, kv_hidden_size, bias=config.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, kv_hidden_size, bias=config.use_bias)
        self.o_proj = nn.Linear(q_hidden_size, self.hidden_size, bias=config.use_bias)

        self.rotary_emb = Ernie4_5RopeEmbedding(
            self.head_dim,
            #compression_ratio=config.compression_ratio,
            compression_ratio=1.0,
            base=config.rope_theta,
        )
        self.config = config

        self.attn_func = self.core_attn
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_start_row_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        token_type_ids: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Compute attention outputs.

        Args:
            hidden_states (torch.Tensor): Input tensor [bsz, seq_len, hidden_size]
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached key/value states
            attention_mask (Optional[torch.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[torch.Tensor]): Variable length attention indices
            position_ids (Optional[torch.Tensor]): Position indices for RoPE
            output_attentions (bool): Return attention weights if True
            use_cache (bool): Cache key/value states if True

        Returns:
            Tuple containing:
                - attention_output: [bsz, seq_len, hidden_size]
                - attention_weights: Optional attention probabilities
                - updated_key_value_cache: Optional updated cache
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)

        attn_output, attn_weights, past_key_value = self.rope_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def rope_attn(
        self,
        query_states,
        key_states,
        value_states,
        past_key_value=None,
        use_cache=False,
    ):
        """Attention computation with rotary embeddings.

        Args:
            query_states (torch.Tensor): Query states
            key_states (torch.Tensor): Key states
            value_states (torch.Tensor): Value states
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached states
            use_cache (bool): Cache new states

        Returns:
            Tuple containing:
                - attention_output: Result tensor
                - attention_weights: Optional weights
                - updated_key_value_cache: Optional cache
        """

        ## rope and caching
        query_states_dtype = query_states.dtype

        kv_seq_len = key_states.shape[-3]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-3]
            kv_seq_len += offset

        cos_sin = self.rotary_emb(kv_seq_len).permute(
            [0, 2, 1, 3]
        )  # [b,h,s,d]->[b,s,h,d]
        if offset > 0:
            cos_sin = cos_sin[:, offset:]
        query_states, key_states = self.rotary_emb.apply_rotary(
            cos_sin, query_states, key_states
        )

        query_states = query_states.to(query_states_dtype)
        key_states = key_states.to(query_states_dtype)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        past_key_value = [key_states, value_states] if use_cache else None
        ## rope and caching

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            None,
            self.scaling,
        )

        attn_output = attn_output.contiguous().view(attn_output.size(0), attn_output.size(1), -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class Ernie4_5DecoderLayer(nn.Module):
    """
    A single transformer decoder layer in ERNIE model.
    """

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config: Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config

        self.self_attn = Ernie4_5Attention(config, layer_idx)
        self.mlp = Ernie4_5MLP(config)

        self.input_layernorm = Ernie4_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_start_row_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[torch.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[torch.Tensor]): Indices for variable length attention
            position_ids (Optional[torch.Tensor]): Position indices for rotary embeddings
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_value (Optional[Tuple[torch.Tensor]]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            token_type_ids=token_type_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Ernie4_5PretrainedModel(PreTrainedModel):
    """Base class for ERNIE pretrained models."""

    config_class = Ernie4_5Config
    base_model_prefix = "model"


class Ernie4_5Model(Ernie4_5PretrainedModel):

    def __init__(self, config):
        """Initialize the ERNIE model architecture.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.ModuleList(
            [Ernie4_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        self.norm = Ernie4_5RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        """Get the input embedding layer.

        Returns:
            nn.Embedding: The embedding layer for input tokens
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set new input embeddings.

        Args:
            value (nn.Embedding): New embedding layer to use
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs
            position_ids (Optional[torch.Tensor]): Position indices
            attention_mask (Optional[torch.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[torch.Tensor]): Variable length attention indices
            inputs_embeds (Optional[torch.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Tuple[Tuple[torch.Tensor]]]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(self.embed_tokens.weight.dtype)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                attn_mask_start_row_indices,
                position_ids,
                token_type_ids,
                output_attentions,
                past_key_value,
                use_cache,
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # apply kv cache
            if past_key_value is not None:
                hidden_states = hidden_states[:, -1:, :]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Ernie4_5ForCausalLM(Ernie4_5PretrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        """
        Initializes the ERNIE model for causal language modeling.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

        self.config = config
        self.model = Ernie4_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        """
        Forward pass for causal language modeling.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            position_ids (torch.Tensor): Position IDs.
            attention_mask (torch.Tensor): Attention mask.
            attn_mask_start_row_indices (torch.Tensor): Attention mask start indices.
            inputs_embeds (torch.Tensor): Optional embedded inputs.
            labels (torch.Tensor): Target labels.
            use_cache (bool): Whether to use cached hidden states.
            past_key_values (dict): Pre-computed hidden states.
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.

        Returns:
            CausalLMOutputWithPast: Model outputs.
        """

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["Ernie4_5ForCausalLM", "Ernie4_5Model", "Ernie4_5PretrainedModel"]
