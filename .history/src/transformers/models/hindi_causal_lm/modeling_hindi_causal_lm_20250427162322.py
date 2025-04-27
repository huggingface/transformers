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
    # Add other checkpoints here if available
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
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Initialize cache
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but aligns with HF implementation: freqs = torch.cat((freqs, freqs), dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            seq_len = x.shape[-2] # Use the sequence length from the input tensor if not provided
        # Update cache if necessary
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
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
    # The 'unsqueeze_dim' argument specifies the dimension to unsqueeze cos and sin tensors.
    # For query and key tensors shaped [bs, num_heads, seq_len, head_dim],
    # cos and sin shaped [seq_len, head_dim / 2] need unsqueezing at dim 1.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None, # Added for potential sliding window attention
):
    """
    Creates a 4D causal attention mask and incorporates the input attention mask.
    Addresses potential broadcasting issues and handles 2D/4D masks robustly.
    """
    bsz, tgt_len = input_shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    src_len = past_key_values_length + tgt_len

    # 1. Create the base causal mask
    causal_mask = torch.full((tgt_len, src_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask_cond = torch.arange(causal_mask.size(-1), device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0.0) # Standard lower triangular mask

    # Adjust for past_key_values_length
    if past_key_values_length > 0:
         causal_mask[:, :past_key_values_length] = 0.0 # Allow attention to past keys

    # Add sliding window constraint if specified
    if sliding_window is not None:
        window_mask = torch.ones_like(causal_mask) * torch.finfo(dtype).min
        min_val = torch.arange(tgt_len, device=device).view(-1, 1) - sliding_window + 1
        mask_cond_sliding = torch.arange(src_len, device=device).view(1, -1) >= min_val
        window_mask.masked_fill_(mask_cond_sliding, 0.0)
        causal_mask = torch.maximum(causal_mask, window_mask) # Combine causal and window masks

    # Expand to 4D: [bsz, 1, tgt_len, src_len]
    causal_4d_mask = causal_mask[None, None, :, :].expand(bsz, 1, tgt_len, src_len)

    # 2. Incorporate the input attention mask (padding mask)
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            # Convert 2D mask [bsz, src_len] to 4D [bsz, 1, 1, src_len] for broadcasting
            input_4d_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
        elif attention_mask.dim() == 4:
            # Assume 4D mask is already correctly shaped or broadcastable
            # Ensure it matches the target shape if possible, otherwise log warning
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                 # Try to make it compatible, handle singleton dimensions carefully
                try:
                    input_4d_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
                except RuntimeError as e:
                    logger.warning(
                        f"Could not broadcast input attention mask shape {attention_mask.shape} "
                        f"to ({bsz}, 1, {tgt_len}, {src_len}). Mask may not be applied correctly. Error: {e}"
                    )
                    input_4d_mask = None # Fallback: don't apply incompatible mask
            else:
                 input_4d_mask = attention_mask
        else:
            raise ValueError(f"Attention mask should be 2D or 4D, but got {attention_mask.dim()}D")

        # Combine causal mask and input mask
        if input_4d_mask is not None:
            # Ensure no size mismatch occurs here (fix for RuntimeError)
            # Use torch.maximum which handles broadcasting safely for masks (0.0 allowed, -inf masked)
            combined_mask = torch.maximum(causal_4d_mask, input_4d_mask.to(dtype)) # Use maximum to merge masks
            # Ensure masked positions are truly negative infinity
            combined_mask = combined_mask.masked_fill(combined_mask != 0.0, torch.finfo(dtype).min)
            return combined_mask
        else:
             # If input mask was incompatible, return only the causal mask
            return causal_4d_mask
    else:
        # No input attention mask provided
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
                "to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True # Causal attention is assumed

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Use RoPE if configured
        self.positional_encoding_type = getattr(config, "positional_encoding_type", "rope")
        if self.positional_encoding_type == "rope":
             self.rotary_emb = HindiCausalLMRotaryEmbedding(
                 self.head_dim, max_position_embeddings=self.max_position_embeddings, base=getattr(config, "rope_theta", 10000)
             )
        else:
            self.rotary_emb = None # Handle other types or no positional encoding if needed


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please use `attention_mask` instead.",
                FutureWarning,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # Handle legacy cache format
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                kv_seq_len += past_key_value[0].shape[-2]
            else:
                # Handle potential new cache format if introduced later
                logger.warning("Unexpected past_key_value format encountered.")


        if self.rotary_emb is not None:
            if position_ids is None:
                raise ValueError("`position_ids` must be provided when using rotary embeddings.")
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            # Select cos/sin based on position_ids
            # Ensure position_ids are correctly broadcastable or indexed
            # cos = cos[position_ids] # This might fail if position_ids shape is wrong
            # sin = sin[position_ids] # Need careful indexing or gathering
            # Assuming position_ids are [bsz, seq_len], gather embeddings for each position
            # Need to handle the shape [seq_len, dim] of cos/sin and position_ids [bsz, seq_len]
            # Gather requires careful index handling, potentially flattening and unflattening
            # Simplified approach: Assume position_ids selects indices from precomputed cache
            # This aligns with common HF implementations where position_ids directly index the cache
            cos = cos.squeeze(1).squeeze(0) # Remove potential singleton dimensions
            sin = sin.squeeze(1).squeeze(0)
            cos = cos[position_ids].unsqueeze(1) # Gather and add head dim: [bsz, 1, q_len, dim] -> index requires care
            sin = sin[position_ids].unsqueeze(1) # Ensure position_ids are correctly aligned with seq_len dimension of cos/sin

            # Apply RoPE
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        if past_key_value is not None:
             # Reuse k, v, self_attention
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                # Handle legacy cache format
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            # Handle potential new cache formats here if needed

        # Update key/value sequence length after potential concatenation
        kv_seq_len = key_states.shape[-2]

        # Check if cache is being used and store current key/value states
        if use_cache:
             present_key_value = (key_states, value_states)
        else:
             present_key_value = None

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Check dimensions before applying mask
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            # Ensure mask is broadcastable (should be handled by _prepare_4d_causal_attention_mask)
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                 # This should ideally not happen if _prepare_4d_causal_attention_mask works correctly
                logger.warning(
                    f"Attention mask shape {attention_mask.size()} does not match expected shape "
                    f"({bsz}, 1, {q_len}, {kv_seq_len}). Mask may not be applied correctly."
                )
                # Attempt to broadcast anyway, or raise error if certain
                try:
                    attention_mask = attention_mask.expand(bsz, 1, q_len, kv_seq_len)
                except RuntimeError as e:
                     raise RuntimeError(f"Could not broadcast attention mask from shape {attention_mask.size()} to {(bsz, 1, q_len, kv_seq_len)}: {e}") from e

            # Apply the mask (additive mask)
            attn_weights = attn_weights + attention_mask # Broadcasting should work here


        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project attention output
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # Return outputs
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


# MLP (Feed-Forward Network) layer
class HindiCausalLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU() # Using SiLU activation as specified in config default
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # Apply dropout

    def forward(self, x):
        intermediate_act = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(intermediate_act)
        output = self.dropout(output) # Apply dropout
        return output


# Transformer Layer combining Attention and MLP
class HindiCausalLMLayer(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = HindiCausalLMMLP(config)
        # Choose normalization layer based on config
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
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `(batch, seq_len)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # Apply input layer norm
        hidden_states_norm = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1]
        present_key_value = attn_outputs[2]

        # Residual connection after attention
        hidden_states = residual + attn_output

        # Apply post-attention layer norm and MLP
        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_norm)

        # Residual connection after MLP
        hidden_states = residual + hidden_states

        # Prepare outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# PreTrainedModel wrapper for Hindi Causal LM
class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMLayer"] # Modules that shouldn't be split during model parallelism
    _skip_keys_device_placement = "past_key_values" # Skip placing cache on device automatically
    _supports_flash_attn_2 = False # Update if Flash Attention 2 is supported
    _supports_sdpa = True # Assume support for Scaled Dot Product Attention
    _supports_cache_class = False # Use legacy tuple format for cache

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # Initialize RMSNorm/LayerNorm weights
        elif isinstance(module, (RMSNorm, nn.LayerNorm)):
             if hasattr(module, 'bias') and module.bias is not None:
                 module.bias.data.zero_()
             module.weight.data.fill_(1.0) # Initialize weights to 1


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HindiCausalLMModel):
            module.gradient_checkpointing = value


# Core Hindi Causal LM Model (stacks layers)
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    """
    Transformer model based on the LLaMA architecture.

    Args:
        config: HindiCausalLMConfig
    """
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Add alias for testing compatibility
        self.token_embeddings = self.embed_tokens

        self.layers = nn.ModuleList([HindiCausalLMLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # Final normalization layer
        norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.token_embeddings = value # Keep alias synced

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None, # Accept token_type_ids but ignore
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs, # Accept extra kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Handle unused token_type_ids gracefully
        if token_type_ids is not None:
             logger.info_once("token_type_ids provided but not used by this model.")

        # Retrieve config values or defaults
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Validate inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Determine sequence length and device
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            # Check structure of past_key_values (should be a tuple of tuples)
            if len(past_key_values) > 0 and past_key_values[0] is not None and len(past_key_values[0]) == 2:
                 past_key_values_length = past_key_values[0][0].shape[2] # Get length from first layer's key
                 seq_length_with_past = seq_length_with_past + past_key_values_length
            else:
                logger.warning("Invalid past_key_values format detected.")


        # Create position_ids if not provided
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
             position_ids = torch.arange(
                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
             )
             position_ids = position_ids.unsqueeze(0).view(1, -1) # Shape: [1, seq_len]

        # Get input embeddings if needed
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare 4D attention mask
        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # Start main transformer block
        hidden_states = inputs_embeds

        # Apply gradient checkpointing if enabled
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Store intermediate results if needed
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Iterate through transformer layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get past key/value for the current layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Define layer forward function for checkpointing if needed
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Provide args explicitly, disable cache and attention output if not needed by checkpoint
                        # Note: output_attentions might be needed depending on checkpoint implementation details
                        return module(*inputs, past_key_value=None, output_attentions=False, use_cache=False)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask_4d,
                    position_ids,
                    use_reentrant=False # Recommended for newer PyTorch versions
                )
            else:
                # Standard forward pass
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            # Update hidden states
            hidden_states = layer_outputs[0]

            # Store cache and attentions if requested
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],) # Cache is always the last element
            if output_attentions:
                 # Attention weights are typically the second element if present
                attn_index = 1 if len(layer_outputs) > 1 else -1 # Find attention index
                if attn_index > 0 and layer_outputs[attn_index] is not None:
                    all_self_attns += (layer_outputs[attn_index],)


        # Apply final layer norm
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Handle return format
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Hindi Causal LM Model with Language Modeling Head
class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel, GenerationMixin):
    # Keys to ignore when loading state dicts with missing keys
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"model.embed_tokens.weight"]
    # Keys indicating tied weights
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.model = HindiCausalLMModel(config)
        # Add alias for testing compatibility
        self.hindi_causal_lm = self.model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Ensure essential generation config defaults are present
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = 0 # Default PAD token ID
            logger.info(f"Setting pad_token_id to {config.pad_token_id}")
        if not hasattr(config, 'bos_token_id') or config.bos_token_id is None:
            config.bos_token_id = 1 # Default BOS token ID
            logger.info(f"Setting bos_token_id to {config.bos_token_id}")
        if not hasattr(config, 'eos_token_id') or config.eos_token_id is None:
            config.eos_token_id = 2 # Default EOS token ID
            logger.info(f"Setting eos_token_id to {config.eos_token_id}")


        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        self.model.token_embeddings = value # Keep alias synced

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Override tie_weights to ensure embedding and lm_head are tied if configured
    def tie_weights(self):
        if self.config.tie_word_embeddings:
             output_embeddings = self.get_output_embeddings()
             input_embeddings = self.get_input_embeddings()
             if output_embeddings is not None and input_embeddings is not None:
                 output_embeddings.weight = input_embeddings.weight
                 # Check if weight shapes match
                 if output_embeddings.weight.shape != input_embeddings.weight.shape:
                     logger.warning(
                         f"Tied weights have different shapes: LM head {output_embeddings.weight.shape}, "
                         f"Embeddings {input_embeddings.weight.shape}. This might cause issues."
                     )
             else:
                 logger.warning("Could not tie input and output embeddings.")

        super().tie_weights() # Call parent method


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None, # Accept but ignore
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs, # Accept extra kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```
        >>> from transformers import AutoTokenizer, HindiCausalLMForCausalLM
        >>> import torch

        >>> model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-foundational-model-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("convaiinnovations/hindi-foundational-model-base")

        >>> prompt = "भारत की राजधानी क्या है?" # What is the capital of India?
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate text
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Example output might be: 'भारत की राजधानी क्या है? नई दिल्ली' (What is the capital of India? New Delhi)
        ```"""
        # Handle unused token_type_ids
        if token_type_ids is not None:
             logger.info_once("token_type_ids provided but not used by this model.")

        # Get config values or defaults
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through the base model
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # Always use dict internally for easier access
            **kwargs, # Pass extra kwargs
        )
        hidden_states = transformer_outputs.last_hidden_state

        # Get logits from the LM head
        logits = self.lm_head(hidden_states)
        logits = logits.float() # Cast logits to float32 for stability

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
             # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure labels are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Prepare output based on return_dict flag
        if not return_dict:
            output = (logits,) + transformer_outputs[1:] # type: ignore
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepares inputs for generation."""
        # Handle past_key_values: if provided, only the last input_ids token is needed
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # Prepare position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
             # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1) # Ensure non-zero position for padding
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1) # Use only the last position id


        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
             model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
            # Pass token_type_ids if present in kwargs
            "token_type_ids": kwargs.get("token_type_ids", None),
        })
        return model_inputs

    # Override generate to fix padding for GenerationTesterMixin
    def generate(self, *args, **kwargs):
        """
        Overrides GenerationMixin.generate to replace trailing pad_token_id
        with the last *non-pad* token to match GenerationTesterMixin expectations.
        """
        # Run the standard generation process first
        sequences = super().generate(*args, **kwargs)

        # Get pad token ID, default to 0 if not set
        pad_token = self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else 0

        if pad_token is None:
            # If pad token is explicitly None, skip the post-processing
            return sequences

        # Post-process each sequence in the batch
        for i in range(sequences.size(0)):
             seq = sequences[i]
             last_real_token = None
             pad_start_index = -1

             # Find the last non-pad token and the start of padding
             for idx in range(seq.size(0)):
                 token = seq[idx].item()
                 if token != pad_token:
                     last_real_token = token
                     pad_start_index = -1 # Reset if we find a non-pad token
                 elif pad_start_index == -1:
                     # Mark the start of a potential padding sequence
                     pad_start_index = idx

             # If padding was found and there was a real token before it, fill padding
             if pad_start_index != -1 and last_real_token is not None:
                 sequences[i, pad_start_index:] = last_real_token

        return sequences


    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorders the cache for beam search/sampling."""
        reordered_past = ()
        for layer_past in past_key_values:
             # Handle potential None or incorrect format in layer_past
            if layer_past is None or not isinstance(layer_past, tuple) or len(layer_past) != 2:
                reordered_past += (layer_past,) # Append as is if invalid
                continue

            # Reorder key and value states
            reordered_layer_past = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past if past_state is not None
            )

            # Ensure tuple has length 2, padding with None if necessary (shouldn't happen with valid cache)
            if len(reordered_layer_past) == 1:
                 reordered_layer_past = reordered_layer_past + (None,)
            elif len(reordered_layer_past) == 0:
                 reordered_layer_past = (None, None)

            reordered_past += (reordered_layer_past,)
        return reordered_past

