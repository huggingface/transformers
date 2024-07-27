# coding=utf-8
# Copyright 2024 GLM & ZhipuAI team. All rights reserved.
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
"""PyTorch GLM model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_glm import GLMConfig


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/glm-4-9b-chat"
_CONFIG_FOR_DOC = "GLMConfig"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->GLM
class GLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GLMRMSNorm is equivalent to T5LayerNorm
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


class GLMRotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_theta=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_theta = rope_theta

    def forward_impl(
        self,
        seq_len: int,
        n_elem: int,
        dtype: torch.dtype,
        device: torch.device,
        base: int = 10000,
    ):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_theta
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).to(dtype=dtype)
        return cache

    def forward(self, seq_len):
        return self.forward_impl(
            seq_len,
            self.dim,
            dtype=self.inv_freq.dtype,
            device=self.inv_freq.device,
        )


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class GLMSelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: GLMConfig, layer_number, device=None):
        super(GLMSelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.num_heads = config.num_attention_heads
        self.projection_size = config.kv_channels * self.num_heads
        self.hidden_size_per_attention_head = self.projection_size // self.num_heads
        self.multi_query_group_num = config.multi_query_group_num
        self.multi_query_attention = config.multi_query_attention
        self.hidden_size = config.hidden_size
        self.qkv_hidden_size = 3 * self.projection_size
        self.add_qkv_bias = config.add_qkv_bias

        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = self.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * self.multi_query_group_num
            )
        self.query_key_value = nn.Linear(
            self.hidden_size,
            self.qkv_hidden_size,
            bias=self.add_qkv_bias,
            device=device,
        )

        self.core_attention = GLM_ATTENTION_CLASSES[config._attn_implementation](config, self.layer_number)

        # Output.
        self.dense = nn.Linear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            device=device,
        )

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_heads
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        past_key_value=None,
        use_cache=True,
    ):
        # hidden_states: [b, sq, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [b, sq, h] --> [b, sq, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_heads * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_heads,
                    self.hidden_size_per_attention_head,
                )
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_heads,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if past_key_value is not None:
            key_layer, value_layer = past_key_value.update(key_layer, value_layer, self.layer_number - 1)
        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1,
                -1,
                self.num_heads // self.num_multi_query_groups_per_partition,
                -1,
                -1,
            )
            key_layer = key_layer.contiguous().view(key_layer.size()[:1] + (self.num_heads,) + key_layer.size()[3:])
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1,
                -1,
                self.num_heads // self.num_multi_query_groups_per_partition,
                -1,
                -1,
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_heads,) + value_layer.size()[3:]
            )
        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, past_key_value


class GLMMLP(nn.Module):
    def __init__(self, config: GLMConfig):
        super().__init__()

        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.act = swiglu

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


# Copied from transformers.models.llama.modeling_llama.repeat_kv
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


class GLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper, modified to include features from CoreAttention."""

    def __init__(self, config: GLMConfig, layer_number):
        super(GLMAttention, self).__init__()
        self.config = config
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.num_heads = config.num_attention_heads
        self.kv_channels = config.kv_channels
        self.attention_dropout = config.attention_dropout
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.is_causal = True

        projection_size = self.kv_channels * self.num_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // self.num_heads

        self.layer_scaling_coefficient = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            # Scale the norm factor by the layer number to adjust attention dynamics across layers
            self.layer_scaling_coefficient = self.layer_number
            self.norm_factor *= self.layer_scaling_coefficient

        self.attention_dropout = torch.nn.Dropout(self.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(0),
            query_layer.size(1),
            query_layer.size(2),
            key_layer.size(2),
        )

        # [b, np, sq, hn] -> [b * np, sq, hn]
        query_layer = query_layer.reshape(output_size[0] * output_size[1], output_size[2], -1)
        # [b, np, sk, hn] -> [b * np, sk, hn]
        key_layer = key_layer.reshape(output_size[0] * output_size[1], output_size[3], -1)

        # preallocating input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,  # [b * np, sq, hn]
            key_layer.transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.reshape(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.layer_scaling_coefficient is not None:
            attention_scores = attention_scores * self.layer_scaling_coefficient
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-2]]
            attention_scores = attention_scores + causal_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # query layer shape: [b * np, sq, hn]
        # value layer shape: [b, np, sk, hn]
        # attention shape: [b, np, sq, sk]
        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(0),
            value_layer.size(1),
            query_layer.size(1),
            value_layer.size(3),
        )
        # change view [b * np, sk, hn]
        value_layer = value_layer.reshape(output_size[0] * output_size[1], value_layer.size(2), -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape(output_size[0] * output_size[1], output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)
        # change view [b, np, sq, hn]
        context_layer = context_layer.reshape(*output_size)
        # [b, np, sq, hn] --> [b, sq, np, hn]
        context_layer = context_layer.transpose(1, 2).contiguous()
        # [b, sq, np, hn] --> [b, sq, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn] and hn is not used in here.
    b, np, sq, _ = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class GLMFlashAttention2(GLMAttention):
    """
    GLM flash attention module. This module inherits from `GLMAttention` as the weights of the module stay
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them. Additionally, for sliding window attention,
    we apply SWA only to the bottom config.max_window_layers layers.
    """

    def __init__(self, config: GLMConfig, layer_number):
        super().__init__(config, layer_number)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.hidden_size_per_attention_head).transpose(
            1, 2
        )
        key_states = key_states.view(bsz, q_len, self.num_heads, self.hidden_size_per_attention_head).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.hidden_size_per_attention_head).transpose(
            1, 2
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_number)

        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_pos_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states)

        if past_key_value is not None:
            cache_has_contents = past_key_value.get_seq_length(self.layer_number) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_number][0]
                past_value = past_key_value[self.layer_number][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_number, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_heads)
        value_states = repeat_kv(value_states, self.num_heads)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_number >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_per_partition).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class GLMSdpaAttention(GLMAttention):
    """
    GLM attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GLMAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                is_causal=True,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
        else:
            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


GLM_ATTENTION_CLASSES = {
    "eager": GLMAttention,
    "flash_attention_2": GLMFlashAttention2,
    "sdpa": GLMSdpaAttention,
}

GLM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GLMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare GLM Model outputting raw hidden-states without any specific head on top.",
    GLM_START_DOCSTRING,
)
class GLMPreTrainedModel(PreTrainedModel):
    config_class = GLMConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GLMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    _version = "0.0.5"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method
        # calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted
            # form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: GLMConfig, device=None):
        super(Embedding, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, device=device)
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # If the input flag for fp32 residual connection is set, convert for
        # float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class GLMDecoderLayer(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: GLMConfig, layer_number, device=None):
        super(GLMDecoderLayer, self).__init__()
        self.layer_number = layer_number
        self.fp32_residual_connection = config.fp32_residual_connection
        self.self_attention = GLMSelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        self.mlp = GLMMLP(config)
        self.input_layernorm = GLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        past_key_value=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, past_key_value = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # Residual connection.
        residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, past_key_value


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: GLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection

        # Number of layers.
        self.num_hidden_layers = config.num_hidden_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMDecoderLayer(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_hidden_layers)])

        self.final_layernorm = GLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        past_key_values,
        output_attentions: bool = False,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for index in range(self.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    past_key_values,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

            hidden_states, next_decoder_cache = layer_ret

        if output_attentions:
            all_self_attns += (hidden_states,)

        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, next_decoder_cache, all_hidden_states, all_self_attns


GLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare GLM Model outputting raw hidden-states without any specific head on top.",
    GLM_START_DOCSTRING,
    """
        device ([`str`], *optional*):
            The device on which this model will be run.
        add_lm_head ([`bool`], *optional*, defaults to `False`):
            Whether or not to add a language modeling head on top of the model. The language modeling head is composed
            of two dense layers.
""",
)
class GLMModel(GLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GLMBlock`]

    Args:
        config: GLMConfig
        device: The device on which the model should be run.
        add_lm_head: Whether to add a language modeling head on top of the model.
    """

    def __init__(self, config: GLMConfig, device: str = None, add_lm_head: bool = False):
        super().__init__(config)

        def default_init(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        # Rotary positional embeddings
        self.max_position_embeddings = config.max_position_embeddings
        rotary_dim = config.hidden_size // self.num_heads if self.kv_channels is None else self.kv_channels

        self.rotary_pos_emb = GLMRotaryEmbedding(
            rotary_dim // 2,
            rope_theta=config.rope_theta,
            original_impl=True,
            device=device,
        )
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        if add_lm_head:
            self.output_layer = init_method(
                nn.Linear,
                self.hidden_size,
                self.vocab_size,
                bias=False,
                **init_kwargs,
            )
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_legacy_cache = False
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        rotary_pos_emb = self.rotary_pos_emb(self.max_position_embeddings)

        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        # Run encoder.
        hidden_states, next_cache, all_hidden_states, all_self_attns = self.encoder(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            rotary_pos_emb=rotary_pos_emb,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not use_cache:
            next_cache = None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GLMForCausalLM(GLMPreTrainedModel):
    _tied_weights_keys = ["transformer.output_layer.weight"]

    def __init__(self, config: GLMConfig, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = GLMModel(config, add_lm_head=True, device=device)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.transformer.embedding.word_embeddings = value

    def get_output_embeddings(self):
        return self.transformer.output_layer

    def set_output_embeddings(self, new_embeddings):
        self.transformer.output_layer = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import GLMTokenizer, GLMForCausalLM

        >>> model = GLMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = GLMTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden,
        # dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.transformer.output_layer(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


@add_start_docstrings(
    """
    The GLM Model transformer with a sequence classification head on top (linear layer).

    [`GLMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GLM_START_DOCSTRING,
)
class GLMForSequenceClassification(GLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GLMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.transformer.embedding.word_embeddings = value

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    The GLM Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    GLM_START_DOCSTRING,
)
class GLMForTokenClassification(GLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GLMModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.transformer.embedding.word_embeddings = value

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
