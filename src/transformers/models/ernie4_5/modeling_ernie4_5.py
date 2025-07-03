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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging

from .configuration_ernie4_5 import Ernie4_5Config


logger = logging.get_logger(__name__)


class Ernie4_5RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Ernie4_5RMSNorm) implementation.

    Ernie4_5RMSNorm is a simplified version of LayerNorm that focuses on the root mean square of inputs,
    omitting the mean-centering operation. This provides computational efficiency while maintaining
    good performance.
    """

    def __init__(self, config):
        """
        Initialize Ernie4_5RMSNorm layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.get_default_dtype())
        )
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states):
        """
        Apply RMS normalization to input hidden states.

        Args:
            hidden_states (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: Normalized output tensor of same shape as input

        Note:
            - computes Ernie4_5RMSNorm manually:
                1. Compute variance of features
                2. Apply reciprocal square root normalization
                3. Scale by learned weight parameter
            - Maintains original dtype for numerical stability during computation
        """
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = torch.rsqrt(variance + self.variance_epsilon) * hidden_states
        return hidden_states.to(self.weight.dtype) * self.weight


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


class Ernie4_5FusedDropoutImpl(nn.Module):
    """
    Fused dropout implementation with residual connection support.

    This layer combines dropout and residual addition in a single operation for better performance,
    particularly on GPU devices. The dropout is conditionally applied based on the probability.

    Args:
        prob (float): Dropout probability (between 0 and 1)

    Attributes:
        prob (float): Stores the dropout probability
        dropout (nn.Dropout): The actual dropout layer instance
    """

    def __init__(self, prob):
        """
        Initialize the fused dropout layer.

        Args:
            prob (float): Dropout probability (0 means no dropout)
        """
        super().__init__()
        self.prob = prob
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x, y):
        """
        Forward pass of the fused dropout layer.

        Args:
            x (Tensor): Input tensor to potentially apply dropout
            y (Tensor): Residual tensor to add to the (possibly dropped out) x

        Returns:
            Tensor: Result of x (with optional dropout) + y
        """
        if self.prob > 0:
            x = self.dropout(x)
        output = x + y

        return output


class Ernie4_5MLP(nn.Module):
    """
    Ernie4_5MLP - Gated Multi-Layer Perceptron module used in Ernie model.
    """

    def __init__(self, config, layer_idx=0):
        """
        Initialize the MLP module with configuration options.

        Args:
            config: Model configurations.
            layer_idx (int): Index of current layer (default: 0)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.use_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.use_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.use_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Args:
            x (Tensor): shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: shape [batch_size, seq_len, hidden_size]
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


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

        self.set_attn_func()

    def set_attn_func(self):
        """Configure attention function based on settings.

        Selects between flash/core attention.
        """
        config = self.config
        use_flash_attn = False
        #if config.use_flash_attention:
        if use_flash_attn:
            self.attn_func = self._flash_attention_wrapper
        else:
            self.attn_func = self.core_attn

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
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1]

        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape(
            [bsz, q_len, -1, self.head_dim]
        )
        key_states = self.k_proj(hidden_states).reshape([bsz, q_len, -1, self.head_dim])
        value_states = self.v_proj(hidden_states).reshape(
            [bsz, q_len, -1, self.head_dim]
        )

        attn_output, attn_weights, past_key_value = self.rope_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
        )

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def repeat_kv(self, hidden_states, n_rep):
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _flash_attention_wrapper(
        self,
        q,
        k,
        v,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        seq_length=None,
    ):
        """Wrapper for flash attention implementation.

        Args:
            q (torch.Tensor): Query tensor
            k (torch.Tensor): Key tensor
            v (torch.Tensor): Value tensor
            attention_mask (Optional[torch.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[torch.Tensor]): Variable length indices
            seq_length (Optional[int]): Sequence length

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and weights
        """
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                #dropout_p=self.config.attention_probs_dropout_prob,
                dropout_p=0.0,
                is_causal=attention_mask is None and q.shape[1] != 1,
                scale=1
                / (getattr(self.config, "scale_qk_coeff", 1.0) * self.head_dim**0.5),
                enable_gqa=self.is_gqa,
            )
        out = out.transpose(1, 2)
        out = out.contiguous().view(out.size(0), out.size(1), -1)

        return out, None

    def core_attn(
        self,
        q,
        k,
        v,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        seq_length=None,
    ):
        """Standard self-attention implementation.

        Args:
            q (torch.Tensor): Query tensor
            k (torch.Tensor): Key tensor
            v (torch.Tensor): Value tensor
            attention_mask (Optional[torch.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[torch.Tensor]): Variable length indices
            seq_length (Optional[int]): Sequence length

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and weights
        """
        origin_dtype = q.dtype

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale_qk_coeff = (
            getattr(self.config, "scale_qk_coeff", 1.0) * self.head_dim**0.5
        )

        q = q / scale_qk_coeff

        # Handle GQA case - repeat k and v heads to match q heads
        if self.is_gqa:
            # [batch, num_key_value_heads, seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
            repeat_factor = self.num_heads // self.num_key_value_heads
            k = self.repeat_kv(k, repeat_factor)
            v = self.repeat_kv(v, repeat_factor)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if getattr(self.config, "scale_qk_coeff", 1.0) != 1.0:
            attn_scores = attn_scores * getattr(self.config, "scale_qk_coeff", 1.0)

        # Causal mask
        seq_len = attn_scores.size(-1)
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=attn_scores.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = attn_weights.to(origin_dtype)

        # attention_probs_dropout_prob default 0.0
        if getattr(self.config, "attention_probs_dropout_prob", 0.0) > 0:
            attn_weights = F.dropout(
                attn_weights,
                p=self.config.attention_probs_dropout_prob,
                training=self.training,
            )

        # [batch, num_heads, q_len, k_len] @ [batch, num_heads, k_len, head_dim] -> [batch, num_heads, q_len, head_dim]
        out = torch.matmul(attn_weights, v)

        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        out = out.permute(0, 2, 1, 3)
        # [batch, seq_len, hidden_size]
        out = out.contiguous().view(out.size(0), out.size(1), -1)

        return out, attn_weights

    def rope_attn(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        position_ids,
        output_attentions=False,
        past_key_value=None,
        use_cache=False,
        attn_mask_start_row_indices=None,
    ):
        """Attention computation with rotary embeddings.

        Args:
            query_states (torch.Tensor): Query states
            key_states (torch.Tensor): Key states
            value_states (torch.Tensor): Value states
            attention_mask (Optional[torch.Tensor]): Attention mask
            position_ids (Optional[torch.Tensor]): Position indices
            output_attentions (bool): Return attention weights
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached states
            use_cache (bool): Cache new states
            attn_mask_start_row_indices (Optional[torch.Tensor]): Variable length indices

        Returns:
            Tuple containing:
                - attention_output: Result tensor
                - attention_weights: Optional weights
                - updated_key_value_cache: Optional cache
        """

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

        # shape: [2, b, s, kvh, d]
        past_key_value = [key_states, value_states] if use_cache else None
        seq_length = query_states.shape[1]
        attn_output, attn_weights = self.attn_func(
            query_states,
            key_states,
            value_states,
            attention_mask,
            attn_mask_start_row_indices,
            seq_length,
        )
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

        self.input_layernorm = Ernie4_5RMSNorm(config)
        self.post_attention_layernorm = Ernie4_5RMSNorm(config)

        #self.residual_add1 = Ernie4_5FusedDropoutImpl(config.hidden_dropout_prob)
        #self.residual_add2 = Ernie4_5FusedDropoutImpl(config.hidden_dropout_prob)
        self.residual_add1 = Ernie4_5FusedDropoutImpl(0.0)
        self.residual_add2 = Ernie4_5FusedDropoutImpl(0.0)

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
        hidden_states = self.residual_add1(hidden_states, residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.residual_add2(hidden_states, residual)
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
    base_model_prefix = "ernie"


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

        self.norm = Ernie4_5RMSNorm(config)

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


class Ernie4_5LMHead(nn.Module):
    """Language model head for ERNIE"""

    def __init__(self, config):
        """Initialize the language model head.

        Args:
            config: Model configuration containing:
                - vocab_size: Size of vocabulary
                - hidden_size: Dimension of hidden states
                - tie_word_embeddings: Whether to tie input/output embeddings
                - weight_share_add_bias: Whether to add bias when weight sharing
                - use_bias: Whether to use bias term
        """

        super(Ernie4_5LMHead, self).__init__()
        self.config = config
        vocab_size = config.vocab_size

        if config.tie_word_embeddings:
            # Weight of shape [vocab_size, hidden_size]
            self.weight = nn.Parameter(
                torch.empty(
                    vocab_size, config.hidden_size, dtype=torch.get_default_dtype()
                )
            )
        else:
            # Weight of shape [hidden_size, vocab_size]
            self.weight = nn.Parameter(
                torch.empty(
                    config.hidden_size, vocab_size, dtype=torch.get_default_dtype()
                )
            )
        nn.init.xavier_uniform_(self.weight)

        logger.info(
            f"output-weight: {self.weight.shape}, tie_word_embeddings: {config.tie_word_embeddings}"
        )

        #if config.weight_share_add_bias and config.use_bias:
        if True and config.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(vocab_size, dtype=torch.get_default_dtype())
            )
        else:
            self.bias = None

    def forward(self, hidden_states):
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        return self.calc_lm_head_logits(
            self.config, hidden_states, self.weight, self.bias
        )

    def calc_lm_head_logits(self, config, hidden_states, weight, bias):
        """
        Calculate language model head logits.

        This is the core function that computes the final output logits for a language model.

        Args:
            config: Model configuration.
            hidden_states (Tensor): Hidden states from the transformer layers
            weight (Tensor): Weight matrix for the language model head
            bias (Tensor): Bias vector for the language model head

        Returns:
            Tensor: The computed logits for language modeling.
        """

        if config.tie_word_embeddings:
            logits = torch.matmul(hidden_states, weight.T)
        else:
            logits = torch.matmul(hidden_states, weight)

        if bias is not None:
            logits = logits + bias

        return logits


class Ernie4_5ForCausalLM(Ernie4_5PretrainedModel, GenerationMixin):
    """ERNIE model for causal language modeling."""

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
        self.lm_head = Ernie4_5LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def set_state_dict(self, state_dict, *args, **kwargs):
        """
        Loads the model state dictionary.
        """
        ret = super().set_state_dict(state_dict)
        return ret

    def get_input_embeddings(self):
        """Returns the input embeddings layer."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Sets the input embeddings layer."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Returns the output embeddings (LM head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Sets the output embeddings layer."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """Sets the ERNIE decoder model."""
        self.model = decoder

    def get_decoder(self):
        """Gets the ERNIE decoder model."""
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


__all__ = ["Ernie4_5ForCausalLM", "Ernie4_5Model", "Ernie4_5PreTrainedModel"]
