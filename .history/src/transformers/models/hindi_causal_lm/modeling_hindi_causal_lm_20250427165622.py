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
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...generation.utils import GenerationMixin # Ensure GenerationMixin is imported
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
        """
        RMSNorm module adopted from LLaMA.
        """
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
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None): # Added device
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Initialize cache
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype()) # Pass device

    def _set_cos_sin_cache(self, seq_len, device, dtype): # Added device
        self.max_seq_len_cached = seq_len
        # Ensure t is created on the correct device
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device)) # Ensure inv_freq is on correct device
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            try:
                seq_len = x.shape[-2]
            except IndexError:
                raise ValueError("Could not infer sequence length from input tensor shape.")

        # Determine target device from input tensor x
        target_device = x.device

        # Update cache if necessary based on seq_len, device or dtype
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != target_device or self.cos_cached.dtype != x.dtype:
             # Pass target_device to cache update
            self._set_cos_sin_cache(seq_len=seq_len, device=target_device, dtype=x.dtype)

        # Return cached cos/sin values sliced and potentially moved to target device
        return (
            self.cos_cached[:seq_len].to(device=target_device),
            self.sin_cached[:seq_len].to(device=target_device),
        )


# Helper function for RoPE
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Apply RoPE to query and key tensors
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    # cos: [seq_len, dim] - Before gathering
    # sin: [seq_len, dim] - Before gathering
    # position_ids: [bsz, seq_len]
    # q: [bsz, num_heads, seq_len, head_dim]
    # k: [bsz, num_heads, seq_len, head_dim]

    # Gather cos/sin based on position_ids
    # Clamp position_ids to be within the cache length
    max_cached_len = cos.shape[0]
    clamped_position_ids = torch.clamp(position_ids, 0, max_cached_len - 1)

    # Gather the embeddings: cos/sin are [seq_len, dim], position_ids is [bsz, seq_len]
    # Output shape: [bsz, seq_len, dim]
    cos_gathered = cos[clamped_position_ids]
    sin_gathered = sin[clamped_position_ids]

    # Unsqueeze for broadcasting: [bsz, 1, seq_len, dim]
    cos_final = cos_gathered.unsqueeze(unsqueeze_dim)
    sin_final = sin_gathered.unsqueeze(unsqueeze_dim)

    # Apply rotation
    q_embed = (q * cos_final) + (rotate_half(q) * sin_final)
    k_embed = (k * cos_final) + (rotate_half(k) * sin_final)
    return q_embed, k_embed


# Updated mask preparation function
def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],  # Input mask (padding), shape [bsz, seq_len]
    input_shape: Tuple[int, int],            # (bsz, tgt_len) - current query sequence length
    inputs_embeds: torch.Tensor,             # Used for dtype and device
    past_key_values_length: int,
    sliding_window: Optional[int] = None,    # Added for potential future use
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

    # 1. Create the base causal mask (lower triangular)
    # Needs to cover the query length (tgt_len) against the key/value length (kv_seq_len)
    if tgt_len > 0:
        # Mask has True where attention is *not* allowed.
        causal_mask_bool = torch.ones((tgt_len, kv_seq_len), dtype=torch.bool, device=device)
        # Create indices for rows (query position) and columns (key/value position)
        rows = torch.arange(tgt_len, device=device).view(-1, 1)
        cols = torch.arange(kv_seq_len, device=device).view(1, -1)

        # Allow attention only to positions j <= i + past_key_values_length
        # This ensures causality relative to the start of the sequence
        causal_mask_bool[rows, cols] = cols > (rows + past_key_values_length)

        # Apply sliding window if needed (Not used by default in this model but kept for compatibility)
        if sliding_window is not None:
            # Mask positions outside the sliding window [i + pkv_len - sw + 1, i + pkv_len]
            window_mask_bool = cols < (rows + past_key_values_length - sliding_window + 1)
            causal_mask_bool = causal_mask_bool | window_mask_bool # Combine masks

        # Convert boolean mask (True=mask) to float mask (0.0=attend, -inf=mask)
        # Expand to 4D [bsz, 1, tgt_len, kv_seq_len]
        causal_4d_mask = torch.where(
            causal_mask_bool, torch.finfo(dtype).min, torch.tensor(0.0, dtype=dtype, device=device)
        )
        causal_4d_mask = causal_4d_mask.expand(bsz, 1, tgt_len, kv_seq_len)

    else:
        # If target length is 0, the mask is effectively empty for the query dimension
        causal_4d_mask = torch.zeros((bsz, 1, 0, kv_seq_len), dtype=dtype, device=device)


    # 2. Incorporate the input padding mask (attention_mask)
    # attention_mask shape is [bsz, original_input_seq_len] (1 = attend, 0 = mask)
    if attention_mask is not None:
        # Ensure attention_mask covers the full kv_seq_len.
        # The provided attention_mask usually corresponds to the *input* sequence length,
        # which might be different from kv_seq_len when using past_key_values.
        input_pad_mask_len = attention_mask.shape[-1]

        # Convert padding mask (1=attend, 0=mask) to additive mask (0.0=attend, -inf=mask)
        if attention_mask.dtype == torch.bool:
            additive_padding_mask = torch.where(attention_mask, 0.0, torch.finfo(dtype).min)
        else:
            additive_padding_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min

        # Adjust mask length to match kv_seq_len
        if input_pad_mask_len < kv_seq_len:
            # Pad the mask. Assumes past keys (not covered by original mask) should be attended to (0.0).
            padding_length = kv_seq_len - input_pad_mask_len
            # Pad on the right (covering the future/current tokens)
            padding = torch.zeros((bsz, padding_length), dtype=dtype, device=device)
            # This assumes the original mask corresponds to the *past* keys.
            # If the original mask covers the *current* input_ids, padding might need adjustment.
            # Let's assume the standard HF way: mask covers the full sequence up to the current input.
            # If using kv cache, the mask passed often only covers the *new* tokens.
            # The logic here might need adjustment based on how attention_mask is passed during generation.
            # --> Sticking to the common case: mask passed is for the full sequence length up to now.
            padded_mask = torch.cat([additive_padding_mask, padding], dim=1)

        elif input_pad_mask_len > kv_seq_len:
            # Truncate the mask if it's longer than needed (e.g., during first step of generation)
            padded_mask = additive_padding_mask[:, :kv_seq_len]
        else:
            padded_mask = additive_padding_mask # Length matches

        # Expand the adjusted padding mask to 4D: [bsz, 1, tgt_len, kv_seq_len]
        # The mask should apply to keys/values, so expand query dimension (tgt_len)
        expanded_padding_mask = padded_mask.unsqueeze(1).unsqueeze(1).expand(bsz, 1, tgt_len, kv_seq_len)

        # Merge masks: using torch.maximum ensures that if either mask says "mask" (-inf), the result is "mask"
        combined_mask = torch.maximum(causal_4d_mask, expanded_padding_mask)
        return combined_mask
    else:
        # No input padding mask provided, return only the causal mask
        return causal_4d_mask


# Attention mechanism
class HindiCausalLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True  # Causal attention is assumed

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Use RoPE if configured
        self.positional_encoding_type = getattr(config, "positional_encoding_type", "rope")
        if self.positional_encoding_type == "rope":
            self.rotary_emb = HindiCausalLMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000), # Use rope_theta from config
                # device=self.q_proj.weight.device # Pass device during init? Or handle in forward? --> Handle in forward
            )
        else:
            self.rotary_emb = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # Expect shape [bsz, 1, q_len, kv_seq_len]
        position_ids: Optional[torch.LongTensor] = None,  # Expect shape [bsz, q_len]
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
        kv_seq_len_pre_cache = key_states.shape[-2] # Current length
        past_kv_len = 0
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2 and past_key_value[0] is not None:
                # k/v cache shape [bsz, num_heads, seq_len_past, head_dim]
                 past_kv_len = past_key_value[0].shape[-2] # Add past length
        kv_seq_len = past_kv_len + kv_seq_len_pre_cache

        if self.rotary_emb is not None:
            if position_ids is None:
                # Should have been created in the main model's forward
                raise ValueError("`position_ids` must be provided when using rotary embeddings.")
            # Ensure rotary embeddings are on the same device as the input
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            # Apply RoPE to the *current* query and key states using their specific position_ids
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Concatenate past K/V states from cache AFTER RoPE is applied to current K/Q
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                # Ensure device consistency before concat
                past_k = past_key_value[0]
                past_v = past_key_value[1]
                if past_k.device != key_states.device:
                   past_k = past_k.to(key_states.device)
                if past_v.device != value_states.device:
                    past_v = past_v.to(value_states.device)

                try:
                    key_states = torch.cat([past_k, key_states], dim=2)
                    value_states = torch.cat([past_v, value_states], dim=2)
                except Exception as e:
                    logger.error(f"Error concatenating past key values: {e}")
                    logger.error(f"Past K shape: {past_k.shape}, Current K shape: {key_states.shape}")
                    logger.error(f"Past V shape: {past_v.shape}, Current V shape: {value_states.shape}")
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
                 # Allow mask shape [bsz, 1, 0, final_kv_seq_len] if q_len is 0 (e.g., first step with cache)
                if not (q_len == 0 and attention_mask.size() == (bsz, 1, 0, final_kv_seq_len)):
                    raise ValueError(
                        f"Attention mask shape {attention_mask.size()} does not match expected shape "
                        f"{expected_mask_shape}. Check mask preparation logic in `HindiCausalLMModel.forward` "
                        f"and `_prepare_4d_causal_attention_mask`."
                    )
            # Apply the prepared additive mask
            attn_weights = attn_weights + attention_mask

        # --- Softmax and Output ---
        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        expected_output_shape = (bsz, self.num_heads, q_len, self.head_dim)
        if attn_output.size() != expected_output_shape:
            raise ValueError(f"`attn_output` should be of size {expected_output_shape}, but is {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


# MLP Layer - Using SiLU
class HindiCausalLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Ensure hidden_act uses SiLU if specified
        if config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif config.hidden_act == "gelu":
             self.act_fn = nn.GELU()
        elif config.hidden_act == "relu":
             self.act_fn = nn.ReLU()
        else:
             # Fallback or raise error for unsupported activations
             logger.warning(f"Unsupported hidden_act '{config.hidden_act}', defaulting to SiLU.")
             self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # Added dropout

    def forward(self, x):
        # Implementation follows Llama structure (SiLU(gate) * up)
        intermediate_act = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(intermediate_act)
        output = self.dropout(output) # Apply dropout
        return output


# Transformer Layer
class HindiCausalLMLayer(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx) # Pass layer_idx
        self.mlp = HindiCausalLMMLP(config)
        # Choose norm class based on config
        norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,         # Expect 4D mask
        position_ids: Optional[torch.LongTensor] = None,      # Expect [bsz, q_len]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        # Pre-normalization for attention
        hidden_states_norm = self.input_layernorm(hidden_states)

        # Self Attention Block
        attn_outputs = self.self_attn(
            hidden_states=hidden_states_norm,
            attention_mask=attention_mask,       # Pass the 4D mask
            position_ids=position_ids,           # Pass position_ids
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1]        # Optional attention weights
        present_key_value = attn_outputs[2]   # Optional cache update

        # Residual connection after attention
        hidden_states = residual + attn_output

        # Fully Connected Block (MLP)
        residual = hidden_states
        # Pre-normalization for MLP
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        # MLP forward pass
        hidden_states = self.mlp(hidden_states_norm)
        # Residual connection after MLP
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,) # Append cache update tuple

        return outputs


# PreTrainedModel
class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "model" # Changed from "hindi_causal_lm" to match common practice and tests
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False # Set to False unless implemented
    _supports_sdpa = True # Set to True if using PyTorch >= 2.0 and standard attention
    _supports_cache_class = False # Set to False as we use tuples for cache
    # Add position_ids to ignore list as it's often generated dynamically
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (RMSNorm, nn.LayerNorm)): # Check both norm types
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            # Initialize weight to 1 for LayerNorm/RMSNorm
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HindiCausalLMModel): # Target the base model class
            module.gradient_checkpointing = value


# Core Model
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`HindiCausalLMLayer`]
    """
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.token_embeddings = self.embed_tokens # Alias for tests
        self.layers = nn.ModuleList([HindiCausalLMLayer(config, i) for i in range(config.num_hidden_layers)])

        # Choose norm class based on config
        norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False # Default gradient checkpointing to False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.token_embeddings = value # Keep alias consistent

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,     # Expect 2D mask [bsz, seq_len]
        position_ids: Optional[torch.LongTensor] = None,   # Expect [bsz, seq_len]
        past_key_values: Optional[List[torch.FloatTensor]] = None, # List of tuples
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None, # Not used, but keep for signature compatibility
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs, # Accept arbitrary kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Handle unused arguments explicitly if they appear in kwargs
        if token_type_ids is not None:
            logger.info_once("token_type_ids provided but not used by HindiCausalLMModel.")
        if "cache_position" in kwargs: # Pop cache_position if present (used by newer HF generate)
            kwargs.pop("cache_position")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Determine sequence length and past length
        past_key_values_length = 0
        if past_key_values is not None:
            # Check format: list/tuple of tuples, each inner tuple (k, v)
            if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                 # Get length from the first layer's key tensor shape: [bsz, num_heads, seq_len_past, head_dim]
                if past_key_values[0] is not None and isinstance(past_key_values[0], tuple) and len(past_key_values[0]) == 2:
                    if past_key_values[0][0] is not None:
                        past_key_values_length = past_key_values[0][0].shape[2]
                    elif past_key_values[0][1] is not None: # Fallback to value tensor if key is None
                         past_key_values_length = past_key_values[0][1].shape[2]

        # Generate position_ids if not provided
        if position_ids is None:
            # Position IDs correspond to the absolute position in the sequence
            # Range starts from past_key_values_length
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0) # Shape [1, seq_len]

        # Expand position_ids if batch size > 1 and position_ids has batch size 1
        if position_ids.shape[0] == 1 and batch_size > 1:
             position_ids = position_ids.expand(batch_size, -1)

        # Get input embeddings if needed
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare 4D attention mask
        # The attention mask passed should be 2D [bsz, seq_len (or full seq len)]
        # _prepare_4d_causal_attention_mask handles merging causal and padding masks
        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        # Expected shape: [bsz, 1, tgt_len, kv_seq_len]

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get past key value for the current layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Custom forward function for gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # inputs: hidden_states, attention_mask_4d, position_ids
                        # past_key_value, output_attentions, use_cache are handled by closure
                        return module(
                            inputs[0], # hidden_states
                            attention_mask=inputs[1], # 4D mask
                            position_ids=inputs[2],   # Correct position ids
                            past_key_value=None,      # Cannot use cache with checkpointing
                            output_attentions=False,  # Cannot output attentions with checkpointing
                            use_cache=False,          # Cannot use cache with checkpointing
                        )
                    return custom_forward

                # Perform checkpointing
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask_4d, # Pass the 4D mask
                    position_ids,      # Pass position ids
                    use_reentrant=False, # Recommended for PyTorch >= 1.11
                )
                # layer_outputs will only contain hidden_states when checkpointing
                hidden_states = layer_outputs[0]

            else:
                # Standard forward pass
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_4d, # Pass the 4D mask
                    position_ids=position_ids,        # Pass position ids
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                # layer_outputs: (hidden_states, Optional[attn_weights], Optional[cache])
                hidden_states = layer_outputs[0]

            # Collect outputs if needed
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],) # Cache is always the last element if use_cache=True
            if output_attentions:
                 # Attention weights are the second element if output_attentions=True
                if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            # Construct tuple output, filtering None values
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Causal LM Head Model
class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel, GenerationMixin): # Inherit GenerationMixin
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"position_ids"]
    # Define keys for automatic weight tying
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        # Instantiate the base model
        self.model = HindiCausalLMModel(config)
        # Alias for tests (matches test structure)
        self.hindi_causal_lm = self.model
        self.vocab_size = config.vocab_size
        # LM Head (Linear layer)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Ensure essential token IDs are present in config for GenerationMixin
        self.config.pad_token_id = getattr(config, "pad_token_id", 0) # Default to 0 if not set
        self.config.bos_token_id = getattr(config, "bos_token_id", 1) # Default to 1
        self.config.eos_token_id = getattr(config, "eos_token_id", 2) # Default to 2

        # Initialize weights and potentially tie embeddings
        self.post_init() # Calls _init_weights and potentially tie_weights

    def get_input_embeddings(self):
        # Delegate to the base model
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # Delegate to the base model
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        # Return the LM head
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # Set the LM head
        self.lm_head = new_embeddings

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        # Check if config requires tying (defaults to True if not present)
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                 # Share the weight parameter directly
                output_embeddings.weight = input_embeddings.weight
                # Log potential shape mismatches (shouldn't happen if config is consistent)
                if output_embeddings.weight.shape != input_embeddings.weight.shape:
                    logger.warning(
                        f"Mismatched tied weights shapes: LM head {output_embeddings.weight.shape}, Embeddings {input_embeddings.weight.shape}"
                    )
            elif output_embeddings is None:
                logger.warning("Output embeddings (lm_head) is None, cannot tie weights.")
            else: # input_embeddings is None
                logger.warning("Input embeddings is None, cannot tie weights.")
        # Call superclass tie_weights if it exists (currently PreTrainedModel doesn't have one)
        # super().tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,      # Expect 2D mask
        position_ids: Optional[torch.LongTensor] = None,   # Expect [bsz, seq_len]
        past_key_values: Optional[List[torch.FloatTensor]] = None, # List of tuples
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None, # Not used, but keep for signature
        labels: Optional[torch.LongTensor] = None,         # For loss calculation
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs, # Accept arbitrary kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if token_type_ids is not None:
            logger.info_once("token_type_ids provided but not used by HindiCausalLMForCausalLM.")
        if "cache_position" in kwargs: # Pop cache_position if present
            kwargs.pop("cache_position")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # --- Pass inputs to the base model ---
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            # token_type_ids=token_type_ids, # Pass if base model uses it
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # Force return_dict from base model
            **kwargs, # Pass remaining kwargs
        )
        hidden_states = transformer_outputs.last_hidden_state # Shape: [bsz, seq_len, hidden_size]

        # --- Calculate LM Logits ---
        if hidden_states.shape[1] == 0:
            # Handle case where sequence length is 0 (e.g., empty input with cache)
            logger.warning("Sequence length is 0. Returning empty logits.")
            # Create logits with shape [batch_size, 0, vocab_size] for consistency
            logits = torch.empty(
                (hidden_states.shape[0], 0, self.vocab_size), dtype=hidden_states.dtype, device=hidden_states.device
            )
        else:
            logits = self.lm_head(hidden_states)
        # Cast logits to float32 for numerical stability during loss calculation
        logits = logits.float()

        # --- Calculate Loss ---
        loss = None
        if labels is not None:
             # Check if sequence length allows for shifting
            if logits.shape[1] > 1 and labels.shape[1] > 1:
                # Shift so tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits_flat = shift_logits.view(-1, self.config.vocab_size)
                shift_labels_flat = shift_labels.view(-1)
                 # Ensure labels are on the same device as logits
                loss = loss_fct(shift_logits_flat, shift_labels_flat.to(shift_logits_flat.device))
            elif logits.shape[1] <= 1:
                # Cannot compute loss for sequence length 1 or 0
                logger.debug(f"Sequence length {logits.shape[1]} is too short to compute loss. Setting loss=None.")
                loss = None # Explicitly set loss to None

        # --- Return Outputs ---
        if not return_dict:
            # Construct tuple output: (loss, logits, cache, hidden_states, attentions)
            output_components = (
                logits,
                transformer_outputs.past_key_values, # Use cache from base model output
                transformer_outputs.hidden_states,   # Use hidden_states from base model output
                transformer_outputs.attentions,      # Use attentions from base model output
            )
            output = tuple(v for v in output_components if v is not None)
            if loss is not None:
                return (loss,) + output # Prepend loss if calculated
            else:
                return output

        # Return CausalLMOutputWithPast dataclass
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # prepare_inputs_for_generation - Crucial for generation with KV caching
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # --- Handle KV Cache ---
        past_length = 0
        if past_key_values is not None:
            # If past_key_values exist, we only need the last token of input_ids
            input_ids = input_ids[:, -1:] # Shape: [bsz, 1]
            # Safely determine past_length from the cache structure
            if (
                isinstance(past_key_values, (list, tuple))
                and len(past_key_values) > 0
                and isinstance(past_key_values[0], (list, tuple))
                and len(past_key_values[0]) >= 1 # Should have at least key
                and past_key_values[0][0] is not None # Check if key exists
                and past_key_values[0][0].dim() == 4 # Check shape [bsz, heads, seq_len, dim]
            ):
                past_length = past_key_values[0][0].shape[2]

        # Determine effective batch size (could be bsz * num_beams)
        # Use input_ids shape *after* potential slicing
        effective_batch_size = input_ids.shape[0]

        # --- Prepare position_ids ---
        # position_ids should correspond to the absolute positions of the *current* input tokens
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            # Calculate position_ids dynamically based on past length and current length
            current_length = input_ids.shape[1] # Length of the *new* tokens (usually 1 when caching)
            # Create range starting from past_length
            position_ids = torch.arange(
                past_length, past_length + current_length, dtype=torch.long, device=input_ids.device
            ) # Shape [current_length]
            position_ids = position_ids.unsqueeze(0) # Shape [1, current_length]
        else:
            # If position_ids are provided, ensure they correspond to the last token(s)
             position_ids = position_ids[:, -input_ids.shape[1]:] # Take the last part

        # Expand position_ids to the effective batch size if needed
        if position_ids.shape[0] != effective_batch_size:
            if position_ids.shape[0] == 1:
                position_ids = position_ids.expand(effective_batch_size, -1)
            else:
                # This case indicates a mismatch, raise error
                raise ValueError(
                    f"Position IDs batch dimension ({position_ids.shape[0]}) does not match "
                    f"effective batch size ({effective_batch_size}) and cannot be expanded from 1."
                )

        # --- Handle inputs_embeds (less common during generation with cache) ---
        if inputs_embeds is not None and past_key_values is None:
            # If providing embeddings for the *full* sequence (no cache)
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # Default case: use input_ids
            model_inputs = {"input_ids": input_ids}

        # --- Update Attention Mask ---
        # The attention mask needs to cover the full kv_seq_len (past + current)
        # It also needs to be expanded to the effective_batch_size
        final_attention_mask = attention_mask
        if final_attention_mask is not None:
             # Expand batch dimension if necessary
            if final_attention_mask.shape[0] != effective_batch_size:
                if final_attention_mask.shape[0] == 1:
                    final_attention_mask = final_attention_mask.expand(effective_batch_size, -1)
                else:
                    logger.warning(
                        f"Attention mask batch size {final_attention_mask.shape[0]} doesn't match "
                        f"effective batch size {effective_batch_size}. Check beam search setup."
                    )
             # Ensure the mask covers the current token(s) being added
             # Typically, the passed mask already covers past+current, or just needs a '1' appended
             # If only the last token ID is passed, we need to append a '1' to the mask
            if past_key_values is not None and final_attention_mask.shape[1] == past_length:
                 mask_extension = torch.ones((effective_batch_size, 1), dtype=torch.long, device=input_ids.device)
                 final_attention_mask = torch.cat([final_attention_mask, mask_extension], dim=1)
            elif final_attention_mask.shape[1] != past_length + input_ids.shape[1]:
                 logger.warning(
                     f"Attention mask length ({final_attention_mask.shape[1]}) doesn't match "
                     f"expected length ({past_length + input_ids.shape[1]}). Ensure it covers past and current tokens."
                 )

        # --- Assemble final model inputs ---
        model_inputs.update(
            {
                "position_ids": position_ids,            # Use the calculated/adjusted position_ids
                "past_key_values": past_key_values,      # Pass the cache
                "use_cache": kwargs.get("use_cache", True), # Default to using cache
                "attention_mask": final_attention_mask,  # Use the potentially expanded/updated mask (2D)
                # Pass token_type_ids if present in kwargs, otherwise don't include it
                **({"token_type_ids": kwargs.get("token_type_ids")} if "token_type_ids" in kwargs else {}),
            }
        )
        return model_inputs


    # generate override - Fixes attribute error and refines padding logic
    def generate(self, *args, **kwargs):
        """
        Generates sequences of token ids for models with a language modeling head.

        This method overrides the default `generate` to potentially apply custom post-processing,
        such as filling padding tokens with the last real token.
        """
        # --- Step 1: Call super().generate() ---
        # This handles the core generation logic (greedy, sampling, beam search)
        # It might return a Tensor or a GenerateOutput object
        outputs = super().generate(*args, **kwargs)

        # --- Step 2: Extract the sequences tensor ---
        if isinstance(outputs, torch.Tensor):
            # super().generate() returned only the tensor
            sequences_tensor = outputs
            is_output_object = False
        elif hasattr(outputs, "sequences"):
            # super().generate() returned a GenerateOutput object (e.g., GenerateDecoderOnlyOutput)
            sequences_tensor = outputs.sequences
            is_output_object = True
        else:
            # Unexpected output type, log and return as is
            logger.warning(f"Unexpected output type from super().generate(): {type(outputs)}. Returning original output.")
            return outputs

        # --- Step 3: Apply custom padding logic (Optional) ---
        # This example fills padding tokens with the last non-padding token.
        # Modify or remove this section if not desired.
        pad_token_id = getattr(self.config, "pad_token_id", None)

        # If no pad_token_id or no padding logic needed, return the original output
        if pad_token_id is None:
            # logger.debug("No pad_token_id found in config or padding logic disabled. Skipping modification.")
            return outputs # Return original tensor or object

        # Clone the tensor to modify it, especially if returning the object
        sequences_copy = sequences_tensor.clone()

        for i in range(sequences_copy.size(0)): # Iterate through batch
            seq = sequences_copy[i] # Get one sequence

            # Find indices of non-padding tokens
            non_pad_indices = (seq != pad_token_id).nonzero().squeeze(-1)

            if len(non_pad_indices) > 0:
                # If there are any non-padding tokens
                last_real_token_idx = non_pad_indices[-1].item() # Index of the last real token
                pad_start_index = last_real_token_idx + 1 # Index where padding potentially starts

                # Check if there is actually padding after the last real token
                if pad_start_index < seq.size(0):
                    last_real_token_value = seq[last_real_token_idx].item() # Value of the last real token
                    # Fill the padding area (from pad_start_index onwards) with the last real token's value
                    sequences_copy[i, pad_start_index:] = last_real_token_value
            # else: # Sequence might be all padding tokens (e.g., empty input generated nothing)
                # In this case, leave the sequence as all padding.

        # --- Step 4: Return the correct type ---
        if is_output_object:
            # Update the sequences attribute in the output object with the modified tensor
            outputs.sequences = sequences_copy
            return outputs # Return the modified GenerateOutput object
        else:
            # Return only the modified tensor
            return sequences_copy


    # _reorder_cache - Essential for beam search
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the KV cache according to the specified beam indices.

        Args:
            past_key_values (tuple(tuple(torch.Tensor))): Cached key and value hidden states.
            beam_idx (torch.LongTensor): Tensor containing the indices of the beams to select. Shape [batch_size*num_beams].

        Returns:
            tuple(tuple(torch.Tensor)): Reordered past key value states.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # layer_past is tuple of (key, value), each shape [bsz*beams, heads, seq_len, dim]
            if layer_past is None:
                reordered_past += (None,) # Handle cases where a layer's cache might be None
                continue

            reordered_layer_past_states = []
            for past_state in layer_past: # Iterate through key and value
                if past_state is not None:
                    # Ensure beam_idx is on the same device as the past state
                    beam_idx_device = beam_idx.to(past_state.device)
                    # Select the tensors corresponding to the chosen beams
                    reordered_state = past_state.index_select(0, beam_idx_device)
                    reordered_layer_past_states.append(reordered_state)
                else:
                    reordered_layer_past_states.append(None) # Keep None placeholders if present

             # Ensure the inner tuple always has length 2 (key, value) even if one was None
            while len(reordered_layer_past_states) < 2:
                reordered_layer_past_states.append(None)

            reordered_past += (tuple(reordered_layer_past_states[:2]),) # Add the reordered (key, value) tuple

        return reordered_past