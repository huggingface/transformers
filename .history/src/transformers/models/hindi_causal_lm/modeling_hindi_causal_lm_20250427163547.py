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
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...generation.utils import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging

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
class RMSNorm(nn.Module):
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


# Rotary Positional Embedding implementation
class HindiCausalLMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            try:
                seq_len = x.shape[-2]
            except IndexError:
                raise ValueError("Could not infer sequence length from input tensor shape.")

        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        # Return cached cos/sin values up to the required sequence length
        # Ensure device and dtype match the input tensor `x` if cache was just created/updated
        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        )


# Helper function for RoPE
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Apply RoPE to query and key tensors
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # cos/sin after gathering by position_ids: [bsz, q_len, dim]
    # Needs shape [bsz, 1, q_len, dim] for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Updated mask preparation function
def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor], # Input mask (padding), shape [bsz, seq_len] or [bsz, kv_seq_len]
    input_shape: Tuple[int, int], # (bsz, tgt_len) - current query sequence length
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a 4D causal attention mask and incorporates the input padding mask.
    Ensures output mask shape is [bsz, 1, tgt_len, kv_seq_len].
    Returns float mask: 0.0 for positions to attend, -inf for masked positions.
    """
    bsz, tgt_len = input_shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    # Total K/V sequence length including past
    kv_seq_len = past_key_values_length + tgt_len

    # 1. Create the base causal mask (lower triangular) for the query sequence length
    # Shape: [tgt_len, kv_seq_len] -> Expanded to [bsz, 1, tgt_len, kv_seq_len]
    if tgt_len > 0:
        # Mask has True where attention is *not* allowed
        causal_mask_bool = torch.ones((tgt_len, kv_seq_len), dtype=torch.bool, device=device)
        rows = torch.arange(tgt_len, device=device).view(-1, 1)
        cols = torch.arange(kv_seq_len, device=device).view(1, -1)
        # Allow attention to positions j <= i + past_key_values_length
        causal_mask_bool[rows, cols] = cols > rows + past_key_values_length

        # Apply sliding window if needed (mask if outside window)
        if sliding_window is not None:
            window_mask_bool = cols < rows + past_key_values_length - sliding_window + 1
            causal_mask_bool = causal_mask_bool | window_mask_bool

        # Convert boolean mask (True=mask) to float mask (0.0=attend, -inf=mask)
        # Expand to 4D [bsz, 1, tgt_len, kv_seq_len]
        causal_4d_mask = torch.where(
            causal_mask_bool, torch.finfo(dtype).min, torch.tensor(0.0, dtype=dtype, device=device)
        ).expand(bsz, 1, tgt_len, kv_seq_len)
    else:
        # If target length is 0, the mask is effectively empty for the query dimension
        causal_4d_mask = torch.zeros((bsz, 1, 0, kv_seq_len), dtype=dtype, device=device)


    # 2. Incorporate the input padding mask (attention_mask)
    if attention_mask is not None:
        # Input mask is typically [bsz, original_input_seq_len]
        # It needs to cover the full kv_seq_len for keys/values
        input_pad_mask_len = attention_mask.shape[-1]

        # Convert padding mask (1=attend, 0=mask) to additive mask (0.0=attend, -inf=mask)
        if attention_mask.dtype == torch.bool: # If input is boolean mask (True=attend)
             additive_padding_mask = torch.where(attention_mask, 0.0, torch.finfo(dtype).min)
        else: # Assume input is float/int mask (1=attend, 0=mask)
             additive_padding_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min

        # Adjust padding mask to cover kv_seq_len
        if additive_padding_mask.dim() == 2: # Shape [bsz, input_pad_mask_len]
             # Expand to [bsz, 1, 1, input_pad_mask_len]
            additive_padding_mask = additive_padding_mask[:, None, None, :]

            # Ensure last dim matches kv_seq_len
            if input_pad_mask_len < kv_seq_len:
                 # Assume missing parts are past keys that should be attended to (pad with 0.0)
                padding = torch.zeros((bsz, 1, 1, kv_seq_len - input_pad_mask_len), dtype=dtype, device=device)
                additive_padding_mask = torch.cat([additive_padding_mask, padding], dim=-1)
            elif input_pad_mask_len > kv_seq_len:
                 # Truncate if input mask is longer than needed
                additive_padding_mask = additive_padding_mask[..., :kv_seq_len]

            # Expand target length dimension to match causal_4d_mask
            # Shape becomes [bsz, 1, tgt_len, kv_seq_len]
            # Only expand if tgt_len > 0
            if tgt_len > 0:
                 additive_padding_mask = additive_padding_mask.expand(bsz, 1, tgt_len, kv_seq_len)
            else:
                 # If tgt_len is 0, the resulting combined mask should also have tgt_len 0
                 # We use the causal_4d_mask which already has shape [bsz, 1, 0, kv_seq_len]
                 additive_padding_mask = torch.empty((bsz, 1, 0, kv_seq_len), dtype=dtype, device=device)


        elif additive_padding_mask.dim() == 4:
             # If 4D, assume it's already [bsz, 1, tgt_len, kv_seq_len] or broadcastable
            # Ensure last dimension matches kv_seq_len
            if additive_padding_mask.shape[-1] != kv_seq_len:
                 # This case is less common for padding masks but handle defensively
                 if additive_padding_mask.shape[-1] < kv_seq_len:
                     pad_len = kv_seq_len - additive_padding_mask.shape[-1]
                     additive_padding_mask = torch.nn.functional.pad(additive_padding_mask, (0, pad_len), value=0.0)
                 else:
                     additive_padding_mask = additive_padding_mask[..., :kv_seq_len]
            # Ensure target length dimension matches
            if additive_padding_mask.shape[2] != tgt_len:
                try:
                    additive_padding_mask = additive_padding_mask.expand(bsz, 1, tgt_len, kv_seq_len)
                except RuntimeError as e:
                    raise ValueError(f"Cannot expand 4D input mask shape {additive_padding_mask.shape} to target shape {(bsz, 1, tgt_len, kv_seq_len)}: {e}") from e

        else:
            raise ValueError(f"Attention mask must be 2D or 4D, but got {attention_mask.dim()}D")

        # Merge masks: using torch.maximum is correct for additive masks (-inf for masked)
        # Make sure shapes are compatible before merging
        if causal_4d_mask.shape == additive_padding_mask.shape:
             combined_mask = torch.maximum(causal_4d_mask, additive_padding_mask)
        else:
             # This indicates an error in the expansion logic above
             raise ValueError(f"Shape mismatch before combining masks: causal {causal_4d_mask.shape}, padding {additive_padding_mask.shape}")

        return combined_mask
    else:
        # No input padding mask provided, return only the causal mask
        return causal_4d_mask


# Attention mechanism
class HindiCausalLMAttention(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended..."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.positional_encoding_type = getattr(config, "positional_encoding_type", "rope")
        if self.positional_encoding_type == "rope":
            self.rotary_emb = HindiCausalLMRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings, base=getattr(config, "rope_theta", 10000)
            )
        else:
            self.rotary_emb = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # Expect shape [bsz, 1, q_len, kv_seq_len]
        position_ids: Optional[torch.LongTensor] = None, # Expect shape [bsz, q_len]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        # Calculate kv_seq_len BEFORE applying RoPE and concatenating cache
        kv_seq_len = key_states.shape[-2] # Current length
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2 and past_key_value[0] is not None:
                kv_seq_len += past_key_value[0].shape[-2] # Add past length

        if self.rotary_emb is not None:
            if position_ids is None:
                raise ValueError("`position_ids` must be provided when using rotary embeddings.")
            # Get cos/sin cache up to the total kv sequence length
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            # Gather the embeddings based on position_ids [bsz, q_len]
            try:
                # Clamp position_ids to be within the cache length
                clamped_position_ids = torch.clamp(position_ids, 0, cos.shape[0] - 1)
                # Gather cos/sin for the specific positions needed for the *query* sequence
                cos_gathered = cos[clamped_position_ids] # Shape: [bsz, q_len, dim]
                sin_gathered = sin[clamped_position_ids] # Shape: [bsz, q_len, dim]
            except IndexError as e:
                 raise IndexError(f"Error indexing rotary embeddings cache (size {cos.shape[0]}) with position_ids (range [{position_ids.min()}, {position_ids.max()}]). Error: {e}") from e

            # Apply RoPE to the *current* query and key states
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_gathered, sin_gathered)

        # Concatenate past K/V states from cache AFTER RoPE is applied to current K
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                try:
                     key_states = torch.cat([past_key_value[0], key_states], dim=2)
                     value_states = torch.cat([past_key_value[1], value_states], dim=2)
                except Exception as e:
                     logger.error(f"Error concatenating past key values: {e}")
                     raise e
            else:
                 logger.warning("Invalid past_key_value format encountered during concatenation.")


        # Store current key/value states in cache if requested
        present_key_value = (key_states, value_states) if use_cache else None

        # --- Attention Calculation ---
        # Update kv_seq_len based on final key_states shape after concatenation
        final_kv_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Check dimensions before applying mask
        expected_attn_shape = (bsz, self.num_heads, q_len, final_kv_seq_len)
        if attn_weights.size() != expected_attn_shape:
            raise ValueError(
                f"Attention weights should be of size {expected_attn_shape}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            # Mask shape MUST be [bsz, 1, q_len, final_kv_seq_len]
            expected_mask_shape = (bsz, 1, q_len, final_kv_seq_len)
            if attention_mask.size() != expected_mask_shape:
                raise ValueError(
                    f"Attention mask shape {attention_mask.size()} does not match expected shape "
                    f"{expected_mask_shape}. Check mask preparation logic."
                )
            # Apply the prepared additive mask
            attn_weights = attn_weights + attention_mask

        # --- Softmax and Output ---
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        expected_output_shape = (bsz, self.num_heads, q_len, self.head_dim)
        if attn_output.size() != expected_output_shape:
            raise ValueError(
                f"`attn_output` should be of size {expected_output_shape}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value

# MLP Layer - unchanged
class HindiCausalLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        intermediate_act = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(intermediate_act)
        output = self.dropout(output)
        return output

# Transformer Layer - unchanged
class HindiCausalLMLayer(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = HindiCausalLMMLP(config)
        norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
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

        hidden_states = residual + attn_output # Add residual before second norm

        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_norm)
        hidden_states = residual + hidden_states # Add residual after MLP

        outputs = (hidden_states,)
        if output_attentions: outputs += (attn_weights,)
        if use_cache: outputs += (present_key_value,)
        return outputs

# PreTrainedModel - add position_ids to ignore list
class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = False
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"position_ids"] # Added position_ids

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (RMSNorm, nn.LayerNorm)):
             if hasattr(module, 'bias') and module.bias is not None: module.bias.data.zero_()
             if hasattr(module, 'weight') and module.weight is not None: module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HindiCausalLMModel): module.gradient_checkpointing = value

# Core Model - Ensure position_ids are expanded correctly
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.token_embeddings = self.embed_tokens # Alias for tests
        self.layers = nn.ModuleList([HindiCausalLMLayer(config, i) for i in range(config.num_hidden_layers)])
        norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self): return self.embed_tokens
    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.token_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if token_type_ids is not None: logger.info_once("token_type_ids provided but not used.")
        if "cache_position" in kwargs: kwargs.pop("cache_position")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None: raise ValueError("Specify either input_ids or inputs_embeds")
        elif input_ids is not None: batch_size, seq_length = input_ids.shape; device = input_ids.device
        elif inputs_embeds is not None: batch_size, seq_length, _ = inputs_embeds.shape; device = inputs_embeds.device
        else: raise ValueError("Specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            if len(past_key_values) > 0 and past_key_values[0] is not None and len(past_key_values[0]) == 2:
                if past_key_values[0][0] is not None and past_key_values[0][0].dim() == 4:
                    past_key_values_length = past_key_values[0][0].shape[2]

        seq_length_with_past = seq_length + past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(past_key_values_length, seq_length_with_past, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # Shape [1, seq_len]

        # Expand position_ids batch dimension AFTER creating the default range
        # This handles the case where batch_size > 1 but position_ids started as [1, seq_len]
        # Crucial for beam search where batch size changes.
        if position_ids.shape[0] == 1 and batch_size > 1:
             position_ids = position_ids.expand(batch_size, -1)

        if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)

        # Prepare 4D attention mask - uses the improved function
        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache: logger.warning_once("`use_cache=True` incompatible with gradient checkpointing..."); use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states: all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None and idx < len(past_key_values) else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                         # inputs[0]=hidden_states, inputs[1]=attention_mask_4d, inputs[2]=position_ids
                        return module(inputs[0], attention_mask=inputs[1], position_ids=inputs[2],
                                      past_key_value=None, output_attentions=False, use_cache=False)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask_4d, position_ids, use_reentrant=False
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=attention_mask_4d, position_ids=position_ids,
                    past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if use_cache: next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                 attn_idx = 1 if len(layer_outputs) > 1 else -1
                 if attn_idx > 0 and layer_outputs[attn_idx] is not None: all_self_attns += (layer_outputs[attn_idx],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=next_cache,
            hidden_states=all_hidden_states, attentions=all_self_attns,
        )

# Causal LM Head Model - Ensure defaults, loss calculation safety
class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"position_ids"]
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.model = HindiCausalLMModel(config)
        self.hindi_causal_lm = self.model # Alias for tests
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Ensure defaults
        self.config.pad_token_id = getattr(config, 'pad_token_id', 0); self.config.pad_token_id = 0 if self.config.pad_token_id is None else self.config.pad_token_id
        self.config.bos_token_id = getattr(config, 'bos_token_id', 1); self.config.bos_token_id = 1 if self.config.bos_token_id is None else self.config.bos_token_id
        self.config.eos_token_id = getattr(config, 'eos_token_id', 2); self.config.eos_token_id = 2 if self.config.eos_token_id is None else self.config.eos_token_id

        self.post_init() # Calls tie_weights if needed

    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value; self.model.token_embeddings = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
             output_embeddings, input_embeddings = self.get_output_embeddings(), self.get_input_embeddings()
             if output_embeddings is not None and input_embeddings is not None:
                 output_embeddings.weight = input_embeddings.weight
                 if output_embeddings.weight.shape != input_embeddings.weight.shape:
                     logger.warning(f"Mismatched tied weights shapes: LM head {output_embeddings.weight.shape}, Embeddings {input_embeddings.weight.shape}")
             elif output_embeddings is None: logger.warning("Output embeddings (lm_head) is None, cannot tie weights.")
             else: logger.warning("Input embeddings is None, cannot tie weights.")
        super().tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if token_type_ids is not None: logger.info_once("token_type_ids provided but not used.")
        if "cache_position" in kwargs: kwargs.pop("cache_position")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        transformer_outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            # token_type_ids=token_type_ids, # Pass if needed by model
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=True, # Use dict internally
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state

        if hidden_states.shape[1] == 0:
             logger.warning("Hidden states sequence length is 0.")
             # Create logits with shape [batch_size, 0, vocab_size] for consistency
             logits = torch.empty((hidden_states.shape[0], 0, self.vocab_size), dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float() # Cast for stability

        loss = None
        if labels is not None:
            if logits.shape[1] > 1 and labels.shape[1] > 1:
                 shift_logits = logits[..., :-1, :].contiguous()
                 shift_labels = labels[..., 1:].contiguous()
                 loss_fct = CrossEntropyLoss()
                 shift_logits_flat = shift_logits.view(-1, self.config.vocab_size)
                 shift_labels_flat = shift_labels.view(-1).to(shift_logits_flat.device)
                 loss = loss_fct(shift_logits_flat, shift_labels_flat)
            elif logits.shape[1] <= 1:
                 logger.debug(f"Cannot compute loss for sequence length {logits.shape[1]}. Setting loss=None.")
                 loss = None

        if not return_dict:
            output_components = (logits, transformer_outputs.past_key_values, transformer_outputs.hidden_states, transformer_outputs.attentions)
            output = tuple(v for v in output_components if v is not None)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions,
        )

    # prepare_inputs_for_generation - Ensure correct batch expansion for position_ids/attn_mask
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Determine effective batch size (could be bsz * num_beams)
        effective_batch_size = input_ids.shape[0]

        # Handle past_key_values and input_ids slicing
        past_length = 0
        if past_key_values is not None:
            input_ids = input_ids[:, -1:] # Only need the last token
            # Safely determine past_length
            if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0 and \
               isinstance(past_key_values[0], (list, tuple)) and len(past_key_values[0]) >= 1 and \
               past_key_values[0][0] is not None and past_key_values[0][0].dim() == 4:
                 past_length = past_key_values[0][0].shape[2]

        # --- Prepare position_ids ---
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            # Calculate position_ids for the current step(s)
            current_length = input_ids.shape[1]
            position_ids = torch.arange(past_length, past_length + current_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0) # Shape [1, current_length]
        # Crucially, expand position_ids to the effective batch size BEFORE returning
        if position_ids.shape[0] != effective_batch_size:
             if position_ids.shape[0] == 1:
                 position_ids = position_ids.expand(effective_batch_size, -1)
             else:
                 # This case should not happen if logic is correct, but raise error if it does
                 raise ValueError(f"Position IDs batch dimension ({position_ids.shape[0]}) does not match effective batch size ({effective_batch_size}) and cannot be expanded from 1.")


        # --- Handle inputs_embeds ---
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
             model_inputs = {"input_ids": input_ids}

        # --- Update Attention Mask ---
        # The attention mask needs to cover the full kv_seq_len
        # It might need to be expanded to the effective_batch_size as well
        final_attention_mask = attention_mask
        if final_attention_mask is not None and final_attention_mask.shape[0] != effective_batch_size:
             if final_attention_mask.shape[0] == 1:
                 final_attention_mask = final_attention_mask.expand(effective_batch_size, -1)
             else:
                 # Handle potential mismatch error if needed
                 logger.warning(f"Attention mask batch size {final_attention_mask.shape[0]} doesn't match effective batch size {effective_batch_size}.")


        # --- Assemble final model inputs ---
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": final_attention_mask, # Use potentially expanded mask
            # Pass token_type_ids if present in kwargs
            **({"token_type_ids": kwargs.get("token_type_ids")} if "token_type_ids" in kwargs else {}),
        })
        return model_inputs


    # generate override - unchanged from previous version (padding fix)
    def generate(self, *args, **kwargs):
        sequences = super().generate(*args, **kwargs)
        pad_token = getattr(self.config, 'pad_token_id', None)
        if pad_token is None: return sequences
        for i in range(sequences.size(0)):
             seq = sequences[i]; last_real_token = None; pad_start_index = -1
             for idx in range(seq.size(0)):
                 token = seq[idx].item()
                 if token != pad_token: last_real_token = token; pad_start_index = -1
                 elif pad_start_index == -1 and token == pad_token: pad_start_index = idx
             if pad_start_index != -1 and last_real_token is not None: sequences[i, pad_start_index:] = last_real_token
        return sequences

    # _reorder_cache - ensure device placement
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            if layer_past is None: reordered_past += (None,); continue
            reordered_layer_past = []
            for past_state in layer_past:
                if past_state is not None:
                     beam_idx_device = beam_idx.to(past_state.device) # Ensure device match
                     reordered_state = past_state.index_select(0, beam_idx_device)
                     reordered_layer_past.append(reordered_state)
                else: reordered_layer_past.append(None)
            while len(reordered_layer_past) < 2: reordered_layer_past.append(None)
            reordered_past += (tuple(reordered_layer_past[:2]),)
        return reordered_past

