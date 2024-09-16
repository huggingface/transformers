# coding=utf-8
# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Moshi model."""

import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig, GenerationMode
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
    Seq2SeqLMOutput,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ...pytorch_utils import ALL_LAYERNORM_LAYERS

from .configuration_moshi import MoshiConfig


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward

if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MoshiConfig"
_CHECKPOINT_FOR_DOC = "kyutai/moshiko"


@dataclass
class MoshiCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # transpose to get (bsz, num_codebooks, seq_len)
    input_ids = input_ids.transpose(1, 2)
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[..., 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask



# Copied from transformers.models.gemma.modeling_gemma.GemmaRMSNorm with Gemma->Moshi
class MoshiRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Moshi is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


ALL_LAYERNORM_LAYERS.append(MoshiRMSNorm)


class MoshiFlexibleLinear(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        # Stack the weights for N layers into a single tensor (num_layers, output_size, input_size)
        self.weight = nn.Parameter(torch.randn(num_layers, output_size, input_size))

    def forward(self, x, layer_idx=None):
        """
        `MoshiFlexibleLinear` creates one linear layer per codebook. There's multiple ways to use it.
        In the default case, `sequence_length=num_layers`, so each element of the sequence will be matmul to the weights corresponding to its index on the sequence.
        
        For more advanced cases, one can specify which codebook's layer(s) to use with `layer_idx`.
        If `layer_idx` indicates a single integer, all of the element of the sequence will be matmul to this single codebook's layer.
        But if `layer_idx` is a tensor of shape `(seq_length,)`, it will matmul each i-th element of the input sequence to the corresponding layer `weight[i]`.
        
         
        Args:
            x (`torch.FloatTensor): input to the layer of shape `(batch, num_layers, embed_dim)` or of shape `(batch, seq_length, embed_dim)`
            layer_idx (`torch.Tensor`, *optional*):
                Can be used to specify which codebook's layers(s) to use.
                If it's a tensor of shape `(seq_length,)`, will matmul each element of the sequence to the corresponding weights.
                But if `layer_idx` is a tensor of shape `(seq_length,)`, it will matmul each i-th element of the input sequence to the corresponding layer `weight[i]`.
        """
        if layer_idx is not None and layer_idx.dim()==0:
            # Single layer case: select a specific layer (batch_size, 1 , input_size) -> (batch_size, 1, output_size)
            return torch.matmul(x, self.weight[layer_idx].T)
        elif layer_idx is not None:
            # Use torch.gather to select the corresponding weights for each sample
            selected_weights = torch.index_select(self.weight, 0, layer_idx)
            return torch.einsum('bnh,noh->bno', x, selected_weights)
            
        # Multiple layers case: 
        # use einsum to batch the operations (batch_size, num_layers, input_size) -> (batch_size, num_layers, output_size)
        return torch.einsum('bnh,noh->bno', x, self.weight)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Moshi
class MoshiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    # Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    # TODO(joao): add me back asap :)
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
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
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MoshiGatingMLP(nn.Module):
    def __init__(self, config, num_layers=1, is_depth_mlp=False):
        super().__init__()

        self.activation_fn = ACT2FN[config.hidden_act]
        ffn_dim = config.ffn_dim if not is_depth_mlp else config.depth_ffn_dim
        hidden_size = config.hidden_size if not is_depth_mlp else config.depth_hidden_size
        if num_layers == 1:
            self.fc1 = nn.Linear(hidden_size, ffn_dim, bias=False)
            self.fc2 = nn.Linear(ffn_dim // 2, hidden_size, bias=False)
        else:
            self.fc1 = MoshiFlexibleLinear(hidden_size, ffn_dim, num_layers)
            self.fc2 = MoshiFlexibleLinear(ffn_dim // 2, hidden_size, num_layers)

    def forward(self, hidden_states: torch.Tensor, layer_idx:int=None) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) if layer_idx is None else self.fc1(hidden_states, layer_idx)
        
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, sequence_length, 2, -1)
        hidden_states = self.activation_fn(hidden_states[..., 0, :]) * hidden_states[..., 1, :]
        hidden_states = self.fc2(hidden_states) if layer_idx is None else self.fc2(hidden_states, layer_idx)
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


class MoshiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MoshiConfig, layer_idx: Optional[int] = None, is_depth_attention=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_depth_attention = is_depth_attention
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size if not is_depth_attention else config.depth_hidden_size
        self.num_heads = config.num_attention_heads if not is_depth_attention else config.depth_num_attention_heads
        self.head_dim = config.head_dim if not is_depth_attention else config.depth_head_dim
        self.num_key_value_heads = config.num_key_value_heads if not is_depth_attention else config.depth_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings if not is_depth_attention else config.depth_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = 1 / math.sqrt(self.head_dim)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if not is_depth_attention:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        else:
            self.q_proj = MoshiFlexibleLinear(self.hidden_size, self.num_heads * self.head_dim, num_layers=config.num_codebooks)
            self.k_proj = MoshiFlexibleLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, num_layers=config.num_codebooks)
            self.v_proj = MoshiFlexibleLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, num_layers=config.num_codebooks)
            self.o_proj = MoshiFlexibleLinear(self.num_heads * self.head_dim, self.hidden_size, num_layers=config.num_codebooks)

        # rotary embeddings are not used in the depth decoder
        self.rotary_emb = MoshiRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        ) if not is_depth_attention else None

    # Copied from transformers.models.gemma.modeling_gemma.GemmaAttention.forward with Gemma->Moshi
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) if not self.is_depth_attention else self.q_proj(hidden_states, position_ids) # Ignore copy
        key_states = self.k_proj(hidden_states) if not self.is_depth_attention else self.k_proj(hidden_states, position_ids) # Ignore copy
        value_states = self.v_proj(hidden_states) if not self.is_depth_attention else self.v_proj(hidden_states, position_ids) # Ignore copy

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None: # Ignore copy
            cos, sin = self.rotary_emb(value_states, position_ids) # Ignore copy
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) # Ignore copy

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output) if not self.is_depth_attention else self.o_proj(attn_output, position_ids) # Ignore copy

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.gemma.modeling_gemma.GemmaFlashAttention2 with Gemma->Moshi
class MoshiFlashAttention2(MoshiAttention):
    """
    Moshi flash attention module. This module inherits from `MoshiAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None: # Ignore copy
            cos, sin = self.rotary_emb(value_states, position_ids) # Ignore copy
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) # Ignore copy


        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (MoshiRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.gemma.modeling_gemma.GemmaSdpaAttention with Gemma->Moshi
class MoshiSdpaAttention(MoshiAttention):
    """
    Moshi attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MoshiAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MoshiAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MoshiModel is using MoshiSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None: # Ignore copy
            cos, sin = self.rotary_emb(value_states, position_ids) # Ignore copy
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) # Ignore copy


        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


MOSHI_ATTENTION_CLASSES = {
    "eager": MoshiAttention,
    "flash_attention_2": MoshiFlashAttention2,
    "sdpa": MoshiSdpaAttention,
}

class MoshiDecoderLayer(nn.Module):
    def __init__(self, config: MoshiConfig, layer_idx: int, use_flexible_linear: bool=False, is_depth_layer=False):
        super().__init__()
        self.is_depth_layer = is_depth_layer
        self.hidden_size = config.hidden_size if not is_depth_layer else config.depth_hidden_size
        self.use_flexible_linear = use_flexible_linear

        # depth_ffn_dim
        # depth_hidden_size
        # depth doesn't use pos embedding
        
        self.self_attn = MOSHI_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = MoshiGatingMLP(config) if not use_flexible_linear else MoshiGatingMLP(config, config.num_codebooks, is_depth_layer)
        self.input_layernorm = MoshiRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MoshiRMSNorm(self.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
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
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) if not self.use_flexible_linear else self.mlp(hidden_states, cache_position)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenPreTrainedModel with Musicgen->Moshi
class MoshiPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MoshiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoshiDecoderLayer", "MoshiAttention"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    main_input_name = "input_ids"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MOSHI_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MoshiConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# TODO: update
MOSHI_INPUTS_DOCSTRING = r"""
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

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `input_ids` and `inputs_embeds` are both unset, `inputs_embeds` takes the value
            of `inputs_embeds`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
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
"""

MOSHI_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, sequence_length)`):
            Indices of input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.

            Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,
            such as with the [`EncodecModel`]. See [`EncodecModel.encode`] for details.

            [What are input IDs?](../glossary#input-ids)

            <Tip warning={true}>

            The `input_ids` will automatically be converted from shape `(batch_size * num_codebooks,
            target_sequence_length)` to `(batch_size, num_codebooks, target_sequence_length)` in the forward pass. If
            you obtain audio codes from an audio encoding model, such as [`EncodecModel`], ensure that the number of
            frames is equal to 1, and that you reshape the audio codes from `(frames, batch_size, num_codebooks,
            target_sequence_length)` to `(batch_size * num_codebooks, target_sequence_length)` prior to passing them as
            `input_ids`.

            </Tip>

        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
            selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# TODO: DO it as a depth decoder
class MoshiDepthDecoder(MoshiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MoshiTransformerLayer`]

    Args:
        config: MoshiConfig
    """

    def __init__(self, config: MoshiConfig):
        super().__init__(config)

        self.text_embed_tokens = nn.Embedding(config.vocab_size + 1, config.depth_hidden_size)
    
        # the last codebook is never used as input
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size + 1, config.depth_hidden_size) for _ in range(config.num_codebooks - 1)]
        )

        self.input_projections = MoshiFlexibleLinear(config.hidden_size, config.depth_hidden_size, config.num_codebooks)
        
        # TODO: remove if relevant
        # nn.ModuleList(
        #     [nn.Linear(config.hidden_size, config.depth_hidden_size, bias=False) for _ in range(config.num_codebooks)]
        # )
        
        self.layers = nn.ModuleList(
            [MoshiDecoderLayer(config, layer_idx, use_flexible_linear=True, is_depth_layer=True) for layer_idx in range(config.depth_num_hidden_layers)]
        )
        
        self.lm_heads = MoshiFlexibleLinear(config.depth_hidden_size, config.audio_vocab_size, config.num_codebooks)
        
        # TODO: remove if relevant
        # nn.ModuleList(
        #     [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
        # )

        self._attn_implementation = config._attn_implementation

        self.gradient_checkpointing = False
        self.config = config

    def forward( # TODO: update docstrings entirely
        self,
        input_ids: Optional[torch.LongTensor] = None, # sequence of oracle input ids, i.e it must be the input ids that are predicted by the decoder # (B, S)
        audio_codes: Optional[torch.Tensor] = None, # Same, shoud be oracle audio codebooks, but also with one channel less: # (B, C, S) or C-1
        last_hidden_states: torch.LongTensor = None, # use 8 times (B, S, H_in) | (B*S, H_in)
        attention_mask: Optional[torch.BoolTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,

    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Embedded representation that will be contextualized by the model
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
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
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # here, we actually predict a sequence length of C
        # independtly from the batch size and sequence length
        # 1/ input ids is passed through text_embed_tokens -> (B, S, H) H=1024
        # 2/ each codebooks is passed through the embedding layer ase well -> (B, C-1, S, H)
        # 3/ concat the two precedent results and get (B, C, S ,H)
        # 4/ then we also pass the last hidden states through the input projection layers, one for each codebooks
        # we get (B, C, S, H)
        # 5/ sum one and another (B, C, S, H)
        # 6/ pass (B*S, C, H) through the model and get (B*S, C, H_out)
        # 7/ for each codebook, pass it through its own lm heads: (B*S, C, H)
        # 8/ predict the codebook C1, C2 ... -> (B, S, C, H)
        
        # TODO: can we suppose B*S each time instead of B,S
        # in the generation mode, it's different:
        # S=1
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
            
        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        # If inputs_embeds is provided, it has the priority over input_ids and audio_codes, which won't be used
        if inputs_embeds is None:
            if input_ids is None and audio_codes is None:
                raise ValueError("You must provide at least one of `input_ids`, `inputs_embeds`, `input_values` and `audio_codes`.")
            
            if input_ids is not None:
                inputs_embeds = self.text_embed_tokens(input_ids)
            
            # TODO: this should actually use embed_tokens depending on which position ids is asked for
            # We should actually use one codebook embedding per element of the sequence
            if audio_codes is not None: # TODO(YL): make sure it's C-1
                audio_inputs_embeds = sum([self.embed_tokens[codebook](audio_codes[:, codebook]) for codebook in range(audio_codes.shape[1])])
                inputs_embeds = audio_inputs_embeds if inputs_embeds is None else audio_inputs_embeds + inputs_embeds

        # TODO: check position ids shape
        inputs_embeds += self.input_projections(last_hidden_states, position_ids)
            

        causal_mask = None
        if attention_mask is not None:
            causal_mask = self._update_causal_mask(
                attention_mask, hidden_states, cache_position, past_key_values, output_attentions
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        
        # TODO: check position ids shape
        # TODO: remove the float() operation in v4.46
        logits = self.lm_heads(hidden_states, position_ids).float()

        loss = None
        if labels is not None:
            # TODO:  it's probably not the right way to do it
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
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
            return tuple(v for v in [loss, logits, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # TODO(YL): on the first step, `input_ids` is used
        # then every new input_ids are passed as `audio_codes` instead!
        # Make sure cache_positions is correct
        # do we use num_logits_to_keep? 
        
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
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

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
    "The bare Moshi Model outputting raw hidden-states without any specific head on top.",
    MOSHI_START_DOCSTRING,
)
# Copied from transformers.models.gemma.modeling_gemma.GemmaModel with Gemma->Moshi
class MoshiModel(MoshiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MoshiDecoderLayer`]

    Args:
        config: MoshiConfig
    """

    def __init__(self, config: MoshiConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MoshiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MoshiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MOSHI_DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False  # noqa: F841
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True  # noqa: F841
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        # Moshi downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/internal/generation_utils#transformers.Cache)"
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
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

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
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


@add_start_docstrings(
    "The Moshi decoder model with a text language modelling head on top. Only usable for text.",
    MOSHI_START_DOCSTRING,
)
# Copied from transformers.models.musicgen.modeling_gemma.GemmaForCausalLM with GEMMA->MOSHI,Gemma->Moshi,gemma-7b->moshiko,google->kyutai, CausalLM->MoshiCausalLM
class MoshiForCausalLM(MoshiPreTrainedModel):
    _tied_weights_keys = None # Ignore copy

    def __init__(self, config):
        super().__init__(config)
        self.model = MoshiModel(config)
        self.vocab_size = config.vocab_size
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

    @add_start_docstrings_to_model_forward(MOSHI_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoshiCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, MoshiCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MoshiForCausalLM

        >>> model = MoshiForCausalLM.from_pretrained("kyutai/moshiko")
        >>> tokenizer = AutoTokenizer.from_pretrained("kyutai/moshiko")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
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
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
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
        
        return MoshiCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states, # Ignore copy
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
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
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

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
    "The original Moshi model with an audio encoder, a Moshi depth decoder and a Moshi decoder, "
    "for speech-to-speech.",
    MOSHI_START_DOCSTRING,
)
class MoshiForConditionalGeneration(MoshiPreTrainedModel): # TODO(YL): don't think I can initialize like this for a composite model
    config_class = MoshiConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: Optional[MoshiConfig] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[MoshiForCausalLM] = None,
        depth_decoder: Optional[MoshiDepthDecoder] = None,
    ):
        if config is None and (audio_encoder is None or decoder is None or depth_decoder is None):
            raise ValueError(
                "Either a configuration has to be provided, or all three of Moshi depth decoder, audio encoder and Moshi decoder."
            )
        if config is None:
            config = MoshiConfig.from_audio_encoder_config(audio_encoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # TODO: verify decoder and depth decoder not incompatible
        # TODO: does the decoder and depth decoder makes sense?

        # initialize with config
        super().__init__(config)

        if audio_encoder is None:
            from ..auto.modeling_auto import AutoModel

            audio_encoder = AutoModel.from_config(config.audio_encoder)

        if decoder is None:
            decoder = MoshiForCausalLM(config)
            
        if depth_decoder is None:
            depth_decoder = MoshiDepthDecoder(config)

        self.depth_decoder = depth_decoder
        self.decoder = decoder
        self.audio_encoder = audio_encoder

        # We have 2 * num_codebooks audio embedding layers because we have the user input channel and the model output channel.
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size + 1, config.hidden_size) for _ in range(2 * config.num_codebooks)]
        )
        
        if self.decoder.config.to_dict() != self.config.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared config:"
                f" {self.config}"
            )
        if self.depth_decoder.config.to_dict() != self.config.to_dict():
            logger.warning(
                f"Config of the depth decoder: {depth_decoder.decoder.__class__} is overwritten by shared config:"
                f" {self.config}"
            )
        if self.audio_encoder.config.to_dict() != self.config.audio_encoder.to_dict():
            logger.warning(
                f"Config of the audio_encoder: {self.audio_encoder.__class__} is overwritten by shared audio_encoder config:"
                f" {self.config.audio_encoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config
        self.depth_decoder.config = self.config
        
        self.num_codebooks = config.num_codebooks

        # tie text encoder, decoder weights if config set accordingly
        self.tie_weights()

    def get_audio_encoder(self):
        return self.audio_encoder

    def get_depth_decoder(self):
        return self.depth_decoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import MoshiForConditionalGeneration

        >>> model = MoshiForConditionalGeneration.from_pretrained("kyutai/moshiko")
        ```"""

        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for MoshiForConditionalGeneration. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @add_start_docstrings_to_model_forward(MOSHI_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        input_values: Optional[torch.FloatTensor] = None, # audio_codes has priority over input_values - precise it
        padding_mask: Optional[torch.BoolTensor] = None,
        audio_codes: Optional[torch.Tensor] = None, # TODO add to docstrings - and decide if it's 16 codebooks or (8 and another audio_values)
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        text_labels: Optional[torch.LongTensor] = None, #TODO: update do docstrings
        audio_labels: Optional[torch.LongTensor] = None, #TODO: update do docstrings
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from transformers import AutoProcessor, MoshiForConditionalGeneration
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("kyutai/moshiko")
        >>> model = MoshiForConditionalGeneration.from_pretrained("kyutai/moshiko")
        >>> # TODO(YL): update
        >>> inputs = processor(
        ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        ...     padding=True,
        ...     return_tensors="pt",
        ... )

        >>> pad_token_id = model.generation_config.pad_token_id
        >>> input_ids = (
        ...     torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)
        ...     * pad_token_id
        ... )

        >>> logits = model(**inputs, input_ids=input_ids).logits
        >>> logits.shape  # (bsz * num_codebooks, tgt_len, vocab_size)
        torch.Size([8, 1, 2048])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_audio_encoder = {
            argument[len("audio_encoder_")]: value
            for argument, value in kwargs.items()
            if argument.startswith("audio_encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        
        # TODO: encode input_values
        # TODO: how to deal with both streams, we actually two input_values stream
        if input_values is not None and audio_codes is None:
            # TODO: should be 16 codebooks
            audio_codes = self.audio_encoder.encode(input_values, padding_mask, num_quantizers=self.num_codebooks, **kwargs_audio_encoder)[0]
            
        
        if (text_labels is not None) and (input_ids is None and inputs_embeds is None):
            input_ids = shift_tokens_right(
                text_labels, self.config.decoder.pad_token_id, self.config.decoder.decoder_start_token_id
            )
            # TODO: also do it with audio_labels?
            
        # If inputs_embeds is provided, it has the priority over input_ids and audio_codes, which won't be used
        if inputs_embeds is None:
            if input_ids is None and audio_codes is None:
                raise ValueError("You must provide at least one of `input_ids`, `inputs_embeds`, `input_values` and `audio_codes`.")
            
            if input_ids is not None:
                inputs_embeds = self.decoder.model.embed_tokens(input_ids)
            
            if audio_codes is not None:
                audio_inputs_embeds = sum([self.embed_tokens[codebook](audio_codes[:, codebook]) for codebook in range(audio_codes.shape[1])])
                inputs_embeds = audio_inputs_embeds if inputs_embeds is None else audio_inputs_embeds + inputs_embeds

        # Decode
        decoder_outputs = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
            labels=text_labels,
            **kwargs_decoder,
        )
        
        # TODO: how to deal with loss here ? maybe we can do one loss for text
        # and one loss for audio_labels?
        decoder_last_hidden_states = decoder_outputs.last_hidden_state
        # TODO: we want to pass the audio_codes and audio_labels from the model inputs
        
        depth_decoder_outputs = None
        if text_labels is not None and audio_labels is not None:
            # TODO: how to deal with padding mask and attention mask ? 
            
            # To use depth decoder forward here, we actually need oracle input ids since we're supposed to pass the true input ids
            depth_decoder_outputs = self.depth_decoder(
                hidden_states=decoder_last_hidden_states,
                input_ids=text_labels, # probably need to reshape to (B*S) 
                audio_codes=audio_labels, # probably need to reshape to (B*S)
                attention_mask=attention_mask,
                padding_mask=padding_mask,
            )
        
        
        if not return_dict:
            outputs = decoder_outputs.to_tuple()
            if depth_decoder_outputs is not None:
                outputs += depth_decoder_outputs.to_tuple()
            return outputs# TODO + encoder_outputs

        # TODO: change
        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss, # TODO: it's the text loss
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )

    def _prepare_inputs_embeds_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        # If inputs_embeds is provided, it has the priority over input_ids and audio_codes, which won't be used
        if inputs_embeds is None:
            # TODO: here we have to decide how to deal with audio codes from the user
            # also have to decide how to deal with number of channels
        
            if input_values is not None and audio_codes is None:
                # TODO: should be 16 codebooks
                audio_codes = self.audio_encoder.encode(input_values, num_quantizers=self.num_codebooks,)[0]
            
            
            if input_ids is None and audio_codes is None:
                raise ValueError("You must provide at least one of `input_ids`, `inputs_embeds`, `input_values` and `audio_codes`.")
            
            if input_ids is not None:
                inputs_embeds = self.decoder.model.embed_tokens(input_ids)
            
            if audio_codes is not None:
                audio_inputs_embeds = sum([self.embed_tokens[codebook](audio_codes[:, codebook]) for codebook in range(audio_codes.shape[1])])
                inputs_embeds = audio_inputs_embeds if inputs_embeds is None else audio_inputs_embeds + inputs_embeds

        return inputs_embeds
        

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        decoder_delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,
    ):
        if decoder_delay_pattern_mask is None:
            input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                input_ids,
                self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # apply the delay pattern mask
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, decoder_delay_pattern_mask)

        if guidance_scale is not None and guidance_scale > 1:
            # for classifier free guidance we need to replicate the decoder args across the batch dim (we'll split these
            # before sampling)
            input_ids = input_ids.repeat((2, 1))
            if attention_mask is not None:
                attention_mask = attention_mask.repeat((2, 1))

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # TODO encoder_outputs is defined. input_ids not needed
            "past_key_values": past_key_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }


    def _prepare_audio_encoder_kwargs_for_generation(
        self, input_values, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get audio encoder
        encoder = self.get_audio_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.audio_encoder.main_input_name
        encoder_kwargs["return_dict"] = True

        encoder_kwargs[model_input_name] = input_values
        audio_encoder_outputs = encoder.encode(**encoder_kwargs)
        audio_codes = audio_encoder_outputs.audio_codes
        audio_scales = audio_encoder_outputs.audio_scales

        frames, bsz, codebooks, seq_len = audio_codes.shape


        if frames != 1:
            raise ValueError(
                f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                "disabled by setting `chunk_length=None` in the audio encoder."
            )

        input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)

        model_kwargs["input_ids"] = input_ids
        model_kwargs["audio_scales"] = audio_scales
        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.decoder.pad_token_id, self.config.decoder.bos_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def freeze_audio_encoder(self):
        """
        Freeze the audio encoder weights.
        """
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder._requires_grad = False

    def freeze_depth_decoder(self):
        """
        Freeze the depth encoder weights.
        """
        for param in self.depth_decoder.parameters():
            param.requires_grad = False
        self.depth_decoder._requires_grad = False

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _get_decoder_start_token_id(
        self, decoder_start_token_id: Union[int, List[int]] = None, bos_token_id: int = None
    ) -> int:
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())


        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        requires_attention_mask = False # TODO "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=inputs_tensor.device)

        # 4. Define other model kwargs
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )


        if "input_ids" not in model_kwargs and "input_values" in model_kwargs:
            model_kwargs = self._prepare_audio_encoder_kwargs_for_generation(
                model_kwargs["input_values"],
                model_kwargs,
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            bos_token_id=generation_config._bos_token_tensor,
            device=inputs_tensor.device,
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to Moshi)
        input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            pad_token_id=generation_config._decoder_start_token_tensor,
            max_length=generation_config.max_length,
        )
        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

        # input_ids are ready to be placed on the streamer (if used)
        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode()

        # 8. prepare batched CFG externally (to enable coexistance with the unbatched CFG)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None

        # 9. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=input_ids.device,
        )

        # 10. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 11. run sample
            outputs = self._sample(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        # apply the pattern mask to the final ids
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs["decoder_delay_pattern_mask"])

        # revert the pattern delay mask by filtering the pad token id
        output_ids = output_ids[output_ids != generation_config._pad_token_tensor].reshape(
            batch_size, self.decoder.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...]

        audio_scales = model_kwargs.get("audio_scales")
        if audio_scales is None:
            audio_scales = [None] * batch_size

        output_values = self.audio_encoder.decode(
            output_ids,
            audio_scales=audio_scales,
        ).audio_values


        if generation_config.return_dict_in_generate:
            outputs.sequences = output_values
            return outputs
        else:
            return output_values