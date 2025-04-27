# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
# Copyright 2025 ConvAI Innovations. All rights reserved.
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

"""PyTorch Hindi Causal Language Model."""

import math
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache # Keep for type hinting if switching to Cache object later
from ...generation.utils import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
        add_start_docstrings,
        add_start_docstrings_to_model_forward,
        logging,
        replace_return_docstrings,
        can_return_tuple,
)

# Import the configuration class
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"

# List of pretrained model archives available for this model
HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "convaiinnovations/hindi-foundational-model-base",
]


# RMSNorm implementation
class HindiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
         return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Rotary Positional Embedding implementation
class HindiCausalLMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            torch.tensor(self.base, dtype=torch.float32) ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        # Register inv_freq buffer
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # *** Fix: Initialize buffers here, register_buffer should only be called once ***
        self.max_seq_len_cached = -1
        # Register empty tensors initially, they will be resized/filled in _set_cos_sin_cache
        self.register_buffer("cos_cached", torch.empty(0, device=device), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, device=device), persistent=False)


    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # Ensure inv_freq is on the correct device before calculation
        current_inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=current_inv_freq.dtype)
        freqs = torch.outer(t, current_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # *** Fix: Update buffers using direct assignment, not re-registering ***
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            try:
                seq_len = x.shape[-2]
            except IndexError:
                raise ValueError("Could not infer sequence length from input tensor shape.")
        target_device = x.device
        target_dtype = x.dtype

        # Check if cache needs update
        if (self.max_seq_len_cached == -1 or seq_len > self.max_seq_len_cached or
            self.cos_cached.device != target_device or self.cos_cached.dtype != target_dtype):
            self._set_cos_sin_cache(seq_len=seq_len, device=target_device, dtype=target_dtype)

        # Return sliced cache
        # Ensure slicing doesn't go out of bounds if seq_len > max_seq_len_cached after update
        # This should not happen if _set_cos_sin_cache is called correctly above
        current_max_len = self.cos_cached.shape[0]
        slice_len = min(seq_len, current_max_len)

        return (
            self.cos_cached[:slice_len],
            self.sin_cached[:slice_len],
        )


# Helper function for RoPE
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Apply RoPE to query and key tensors
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if position_ids is None or position_ids.numel() == 0:
        logger.warning_once("position_ids provided to apply_rotary_pos_emb is None or empty. Skipping RoPE application.")
        return q, k

    # Ensure cos/sin are long enough for max position_id after clamping
    max_cached_len = cos.shape[0]
    max_pos_id = position_ids.max()
    if max_pos_id >= max_cached_len:
        # This indicates an issue, likely cache wasn't updated correctly in RoPE forward
        raise ValueError(
            f"RoPE cache length {max_cached_len} is insufficient for maximum position ID {max_pos_id}. "
            "Ensure RoPE cache is updated with the correct sequence length."
        )

    clamped_position_ids = torch.clamp(position_ids, 0, max_cached_len - 1)
    try:
        cos_gathered = cos[clamped_position_ids]
        sin_gathered = sin[clamped_position_ids]
    except IndexError as e:
         logger.error(f"Error indexing RoPE cache (size {max_cached_len}) with clamped_position_ids (shape {clamped_position_ids.shape}, min {clamped_position_ids.min()}, max {clamped_position_ids.max()}). Original position_ids shape: {position_ids.shape}")
         raise IndexError(f"Error indexing RoPE cache: {e}") from e
    if cos_gathered.numel() == 0 or sin_gathered.numel() == 0:
        logger.warning_once("RoPE gathered cos/sin tensors are empty after indexing. Skipping RoPE application.")
        return q, k

    cos_final = cos_gathered.unsqueeze(unsqueeze_dim)
    sin_final = sin_gathered.unsqueeze(unsqueeze_dim)

    if q.shape[2] != cos_final.shape[2]:
        raise RuntimeError(
             f"Mismatch in sequence length dimension for RoPE. Query has length {q.shape[2]}, "
             f"but gathered cosine/sine have length {cos_final.shape[2]}. "
             f"Original position_ids shape was {position_ids.shape}."
         )

    q_embed = (q * cos_final) + (rotate_half(q) * sin_final)
    k_embed = (k * cos_final) + (rotate_half(k) * sin_final)
    return q_embed, k_embed


# Function to repeat key/value heads
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Updated mask preparation function
def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    bsz, tgt_len = input_shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    kv_seq_len = past_key_values_length + tgt_len

    if tgt_len > 0:
        causal_mask_bool = torch.ones((tgt_len, kv_seq_len), dtype=torch.bool, device=device)
        rows = torch.arange(tgt_len, device=device).view(-1, 1)
        cols = torch.arange(kv_seq_len, device=device).view(1, -1)
        causal_mask_bool[rows, cols] = cols > (rows + past_key_values_length)
        if sliding_window is not None:
            window_mask_bool = cols < (rows + past_key_values_length - sliding_window + 1)
            causal_mask_bool = causal_mask_bool | window_mask_bool
        causal_4d_mask = torch.where(
            causal_mask_bool, torch.finfo(dtype).min, torch.tensor(0.0, dtype=dtype, device=device)
        )
        causal_4d_mask = causal_4d_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, tgt_len, kv_seq_len)
    else:
        causal_4d_mask = torch.zeros((bsz, 1, 0, kv_seq_len), dtype=dtype, device=device)

    if attention_mask is not None:
        input_pad_mask_len = attention_mask.shape[-1]
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                additive_padding_mask = torch.where(attention_mask, 0.0, torch.finfo(dtype).min)
            else:
                additive_padding_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min
            if input_pad_mask_len < kv_seq_len:
                padding = torch.zeros((bsz, kv_seq_len - input_pad_mask_len), dtype=dtype, device=device)
                padded_mask = torch.cat([additive_padding_mask, padding], dim=1)
            elif input_pad_mask_len > kv_seq_len:
                 padded_mask = additive_padding_mask[:, :kv_seq_len]
            else:
                padded_mask = additive_padding_mask
            expanded_padding_mask = padded_mask.unsqueeze(1).unsqueeze(1).expand(bsz, 1, tgt_len, kv_seq_len)
        elif attention_mask.dim() == 4:
            expanded_padding_mask = attention_mask
            if expanded_padding_mask.shape[-1] != kv_seq_len:
                 logger.warning(f"Provided 4D mask shape[-1] {expanded_padding_mask.shape[-1]} != kv_seq_len {kv_seq_len}")
                 if expanded_padding_mask.shape[-1] < kv_seq_len:
                     pad_len = kv_seq_len - expanded_padding_mask.shape[-1]
                     expanded_padding_mask = nn.functional.pad(expanded_padding_mask, (0, pad_len), value=torch.finfo(dtype).min)
                 else:
                     expanded_padding_mask = expanded_padding_mask[..., :kv_seq_len]
        else:
            raise ValueError(f"Unexpected attention mask dimension: {attention_mask.dim()}")

        if causal_4d_mask.shape != expanded_padding_mask.shape:
             try:
                 combined_mask = torch.maximum(causal_4d_mask, expanded_padding_mask)
             except RuntimeError as e:
                 raise ValueError(
                     f"Shape mismatch combining masks: causal {causal_4d_mask.shape}, padding {expanded_padding_mask.shape}. Error: {e}"
                 ) from e
        else:
             combined_mask = torch.maximum(causal_4d_mask, expanded_padding_mask)

        if combined_mask.shape != (bsz, 1, tgt_len, kv_seq_len):
             raise ValueError(f"Final combined mask shape {combined_mask.shape} incorrect. Expected {(bsz, 1, tgt_len, kv_seq_len)}")
        return combined_mask
    else:
        return causal_4d_mask


# Attention mechanism (Eager implementation)
class HindiCausalLMAttention(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = 1
        self.is_causal = True
        if (self.head_dim * self.num_heads) != self.hidden_size:
             raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout = config.attention_probs_dropout_prob
        self.positional_encoding_type = getattr(config, "positional_encoding_type", "rope")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int):
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self._shape(self.q_proj(hidden_states), q_len, bsz, self.num_heads)
        key_states = self._shape(self.k_proj(hidden_states), q_len, bsz, self.num_key_value_heads)
        value_states = self._shape(self.v_proj(hidden_states), q_len, bsz, self.num_key_value_heads)

        cos, sin = position_embeddings
        if self.positional_encoding_type == "rope":
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.device != key_states.device:
                 past_k = past_k.to(key_states.device)
            if past_v.device != value_states.device:
                 past_v = past_v.to(value_states.device)
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)

        final_kv_seq_len = key_states_rep.shape[-2]
        expected_attn_shape = (bsz, self.num_heads, q_len, final_kv_seq_len)
        if attn_weights.size() != expected_attn_shape:
             raise ValueError(f"Attention weights shape mismatch: expected {expected_attn_shape}, got {attn_weights.size()}")

        if attention_mask is not None:
            expected_mask_shape = (bsz, 1, q_len, final_kv_seq_len)
            if attention_mask.size() != expected_mask_shape:
                 if not (attention_mask.shape[0] == bsz and attention_mask.shape[1] == 1 and attention_mask.shape[2] == q_len and attention_mask.shape[3] == final_kv_seq_len):
                     if not (q_len == 0 and attention_mask.size() == (bsz, 1, 0, final_kv_seq_len)) and \
                        not (attention_mask.shape[3] >= final_kv_seq_len):
                            raise ValueError(
                                f"Attention mask shape {attention_mask.size()} cannot be broadcast to expected shape "
                                f"{expected_mask_shape}. Final KV len is {final_kv_seq_len}."
                            )
                     else:
                         attention_mask = attention_mask[..., :final_kv_seq_len]
            attn_weights = attn_weights + attention_mask

        attn_weights_softmax = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_dropped = nn.functional.dropout(attn_weights_softmax, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights_dropped, value_states_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        returned_attn_weights = attn_weights_softmax if output_attentions else None
        return attn_output, returned_attn_weights, present_key_value


# MLP Layer
class HindiCausalLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, x):
        intermediate_act = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(intermediate_act)
        output = self.dropout(output)
        return output


# Transformer Layer
class HindiCausalLMLayer(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = HindiCausalLMMLP(config)
        norm_class = HindiRMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states_norm,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1]
        present_key_value = attn_outputs[2]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_norm)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# PreTrainedModel Base Class
@add_start_docstrings("...", """Initialize the weights.""")
class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = False
    _supports_static_cache = False
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"model.rotary_emb.inv_freq", r"rotary_emb.inv_freq"]

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
        elif isinstance(module, (HindiRMSNorm, nn.LayerNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                 module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                 module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HindiCausalLMModel):
             module.gradient_checkpointing = value


# Core Model - HindiCausalLMModel
HINDI_CAUSAL_LM_INPUTS_DOCSTRING = r"""...""" # Assume defined
@add_start_docstrings("...", HindiCausalLMPreTrainedModel.__doc__)
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.token_embeddings = self.embed_tokens
        self.layers = nn.ModuleList([HindiCausalLMLayer(config, i) for i in range(config.num_hidden_layers)])
        if config.positional_encoding_type == "rope":
             self.rotary_emb = HindiCausalLMRotaryEmbedding(
                 dim=config.hidden_size // config.num_attention_heads,
                 max_position_embeddings=config.max_position_embeddings,
                 base=config.rope_theta,
             )
        else:
             self.rotary_emb = None
        norm_class = HindiRMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.token_embeddings = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(HINDI_CAUSAL_LM_INPUTS_DOCSTRING)
    def forward(
        self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
             raise ValueError("Specify either input_ids or inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
             raise ValueError("Specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            try:
                 past_key_values_length = past_key_values[0][0].shape[2]
            except (IndexError, TypeError, AttributeError):
                 past_key_values_length = 0

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            ).unsqueeze(0)
        if position_ids.shape[0] != batch_size:
            if position_ids.shape[0] == 1:
                 position_ids = position_ids.expand(batch_size, -1)
            else:
                 raise ValueError(f"Position IDs batch size {position_ids.shape[0]} does not match input batch size {batch_size}")

        if inputs_embeds is None:
             inputs_embeds = self.embed_tokens(input_ids)

        attention_mask_4d = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        hidden_states = inputs_embeds

        if self.rotary_emb:
             kv_seq_len = seq_length + past_key_values_length
             # Ensure rotary_emb is on the correct device only if it exists
             if self.rotary_emb.cos_cached is None or self.rotary_emb.cos_cached.device != hidden_states.device:
                  self.rotary_emb.to(hidden_states.device)
             position_embeddings = self.rotary_emb(hidden_states, seq_len=kv_seq_len)
        else:
             position_embeddings = (None, None)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                 logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
                 use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                 all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(inputs[0], position_embeddings=inputs[1], attention_mask=inputs[2], position_ids=inputs[3], past_key_value=None, output_attentions=False, use_cache=False)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(decoder_layer), hidden_states, position_embeddings, attention_mask_4d, position_ids, use_reentrant=False)
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = decoder_layer(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask_4d, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
                hidden_states = layer_outputs[0]

            if use_cache:
                 next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
             all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=next_cache,
            hidden_states=all_hidden_states, attentions=all_self_attns,
        )


# Causal LM Head Model
@add_start_docstrings("...", HindiCausalLMPreTrainedModel.__doc__)
class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.model = HindiCausalLMModel(config)
        self.hindi_causal_lm = self.model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config.pad_token_id = getattr(config, "pad_token_id", 0)
        self.config.bos_token_id = getattr(config, "bos_token_id", 1)
        self.config.eos_token_id = getattr(config, "eos_token_id", 2)
        self.post_init()

    def get_input_embeddings(self): return self.model.get_input_embeddings()
    def set_input_embeddings(self, value): self.model.set_input_embeddings(value)
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings

    @can_return_tuple
    @add_start_docstrings_to_model_forward(HINDI_CAUSAL_LM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): ...
        Returns:

        """
        if "token_type_ids" in kwargs:
            logger.info_once("token_type_ids provided but not used by HindiCausalLMForCausalLM.")
            kwargs.pop("token_type_ids")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True, cache_position=cache_position, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + tuple(v for v in [outputs.past_key_values, outputs.hidden_states, outputs.attentions] if v is not None)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
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
        **kwargs
    ):
        past_length = 0
        if past_key_values is not None:
            try:
                 past_length = past_key_values[0][0].shape[2]
            except (IndexError, TypeError, AttributeError):
                 past_length = 0

        # Determine current_length based on the actual input being processed
        if inputs_embeds is not None and input_ids is None:
             current_length = inputs_embeds.shape[1]
             # If past is used, inputs_embeds likely represents only the new tokens
             if past_key_values is not None:
                 # No change needed here if inputs_embeds *only* contains the new tokens
                 pass
        elif input_ids is not None:
             current_length = input_ids.shape[1]
             # If past is used, input_ids should already be sliced to new tokens by GenerationMixin
        else:
             raise ValueError("Must provide either input_ids or inputs_embeds")

        # Position IDs: Calculate based on past_length and *actual current_length*
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_length, past_length + current_length, dtype=torch.long, device=device
            ).unsqueeze(0) # Shape [1, current_length]
        else:
            # Ensure provided position_ids correspond to the current tokens
            # GenerationMixin usually handles adjusting position_ids for the next token
            if position_ids.shape[-1] != current_length:
                logger.warning(f"Provided position_ids length ({position_ids.shape[-1]}) != current_length ({current_length}). Taking last {current_length}.")
                position_ids = position_ids[:, -current_length:]

        # Prepare model inputs dictionary
        if inputs_embeds is not None and input_ids is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}


        # Prepare attention mask for the *forward* pass
        # It needs to cover the full expected kv sequence length
        expected_kv_len = past_length + current_length
        if attention_mask is not None:
             # Ensure the mask covers the full length
             if attention_mask.shape[1] < expected_kv_len:
                 # Pad the mask if it's too short (happens during generation steps)
                 padding_length = expected_kv_len - attention_mask.shape[1]
                 pad = torch.ones(
                     (attention_mask.shape[0], padding_length),
                     dtype=attention_mask.dtype,
                     device=attention_mask.device
                 )
                 attention_mask = torch.cat([attention_mask, pad], dim=1)
             elif attention_mask.shape[1] > expected_kv_len:
                 # Trim the mask if it's too long (can happen with left padding)
                 attention_mask = attention_mask[:, :expected_kv_len]


        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask, # Pass the (potentially adjusted) 2D mask
            # cache_position is not used by tuple cache, omit it
            **kwargs,
        })
        return model_inputs

    def generate(self, *args, **kwargs):
        outputs = super().generate(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            sequences_tensor = outputs
            is_output_object = False
        elif hasattr(outputs, "sequences"):
            sequences_tensor = outputs.sequences
            is_output_object = True
        else:
            logger.warning(f"Unexpected output type from super().generate(): {type(outputs)}. Returning original output.")
            return outputs
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
             return outputs
        sequences_copy = sequences_tensor.clone()
        for i in range(sequences_copy.size(0)):
            seq = sequences_copy[i]
            non_pad_indices = (seq != pad_token_id).nonzero(as_tuple=False).squeeze(-1)
            if len(non_pad_indices) > 0:
                last_real_token_idx = non_pad_indices[-1].item()
                pad_start_index = last_real_token_idx + 1
                if pad_start_index < seq.size(0):
                    last_real_token_value = seq[last_real_token_idx].item()
                    sequences_copy[i, pad_start_index:] = last_real_token_value
        if is_output_object:
             outputs.sequences = sequences_copy
             return outputs
        else:
             return sequences_copy

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            if layer_past is None or not isinstance(layer_past, tuple) or len(layer_past) != 2:
                logger.warning("Encountered invalid cache format during reordering. Skipping layer.")
                reordered_past += (None,)
                continue
            reordered_layer_past_states = []
            for past_state in layer_past:
                if past_state is not None:
                    beam_idx_device = beam_idx.to(past_state.device)
                    reordered_state = past_state.index_select(0, beam_idx_device)
                    reordered_layer_past_states.append(reordered_state)
                else:
                    reordered_layer_past_states.append(None)
            while len(reordered_layer_past_states) < 2:
                 reordered_layer_past_states.append(None)
            reordered_past += (tuple(reordered_layer_past_states[:2]),)
        return reordered_past