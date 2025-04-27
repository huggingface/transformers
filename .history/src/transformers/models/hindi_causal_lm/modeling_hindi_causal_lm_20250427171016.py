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

from ...activations import ACT2FN # Use ACT2FN for flexibility
from ...cache_utils import Cache # Import Cache for type hinting if needed later
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
from ...utils.import_utils import is_torch_fx_available # Example utility import

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
class HindiRMSNorm(nn.Module): # Renamed slightly for clarity
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
        # Note: Hindi model doesn't use (1.0 + weight) like Gemma2
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
        # Precompute inverse frequencies
        # Note: Ensure calculations are done in float32 for precision
        inv_freq = 1.0 / (
            torch.tensor(self.base, dtype=torch.float32) ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Initialize cache
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # Ensure t is created on the correct device and dtype
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq) # Outer product alternative
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            try:
                seq_len = x.shape[-2]
            except IndexError:
                raise ValueError("Could not infer sequence length from input tensor shape.")

        # Determine target device and dtype from input tensor x
        target_device = x.device
        target_dtype = x.dtype

        # Update cache if necessary based on seq_len, device or dtype
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != target_device or self.cos_cached.dtype != target_dtype:
            self._set_cos_sin_cache(seq_len=seq_len, device=target_device, dtype=target_dtype)

        # Return cached cos/sin values sliced and on the correct device/dtype
        return (
            self.cos_cached[:seq_len].to(device=target_device, dtype=target_dtype),
            self.sin_cached[:seq_len].to(device=target_device, dtype=target_dtype),
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
    # Clamp position_ids to be within the cache length before indexing
    max_cached_len = cos.shape[0]
    clamped_position_ids = torch.clamp(position_ids, 0, max_cached_len - 1)

    # Gather the embeddings: cos/sin are [seq_len, dim], clamped_position_ids is [bsz, seq_len]
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


# Updated mask preparation function - Focused on Eager implementation
def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],  # Input mask (padding), shape [bsz, seq_len]
    input_shape: Tuple[int, int],            # (bsz, tgt_len) - current query sequence length
    inputs_embeds: torch.Tensor,             # Used for dtype and device
    past_key_values_length: int,
    sliding_window: Optional[int] = None,    # Keep for potential future use
):
    """
    Creates a 4D causal attention mask for eager attention and incorporates the input padding mask.
    Ensures output mask shape is [bsz, 1, tgt_len, kv_seq_len].
    Returns float mask: 0.0 for positions to attend, -inf for masked positions.
    """
    bsz, tgt_len = input_shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device

    # Total K/V sequence length including past
    kv_seq_len = past_key_values_length + tgt_len

    # 1. Create the base causal mask (lower triangular)
    if tgt_len > 0:
        # Mask has True where attention is *not* allowed. Shape: [tgt_len, kv_seq_len]
        # A query token i (row) can attend to key/value tokens j (column) where j <= i + past_kv_len
        causal_mask_bool = torch.ones((tgt_len, kv_seq_len), dtype=torch.bool, device=device)
        rows = torch.arange(tgt_len, device=device).view(-1, 1)
        cols = torch.arange(kv_seq_len, device=device).view(1, -1)
        causal_mask_bool[rows, cols] = cols > (rows + past_key_values_length)

        # Apply sliding window if needed (Currently not used by Hindi model config)
        if sliding_window is not None:
            window_mask_bool = cols < (rows + past_key_values_length - sliding_window + 1)
            causal_mask_bool = causal_mask_bool | window_mask_bool # Combine masks

        # Convert boolean mask (True=mask) to float mask (0.0=attend, -inf=mask)
        # Expand to 4D [bsz, 1, tgt_len, kv_seq_len]
        causal_4d_mask = torch.where(
            causal_mask_bool, torch.finfo(dtype).min, torch.tensor(0.0, dtype=dtype, device=device)
        )
        # Expand batch dimension
        causal_4d_mask = causal_4d_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, tgt_len, kv_seq_len)
    else:
        # If target length is 0, the mask is effectively empty for the query dimension
        causal_4d_mask = torch.zeros((bsz, 1, 0, kv_seq_len), dtype=dtype, device=device)

    # 2. Incorporate the input padding mask (attention_mask) [bsz, seq_len] (1 = attend, 0 = mask)
    if attention_mask is not None:
        # Ensure attention_mask covers the full kv_seq_len.
        # The mask passed might only cover the current input tokens or the full history.
        # We assume the mask covers the entire sequence up to kv_seq_len for keys/values.
        input_pad_mask_len = attention_mask.shape[-1]

        # Convert padding mask (1=attend, 0=mask) to additive mask (0.0=attend, -inf=mask)
        if attention_mask.dtype == torch.bool:
            additive_padding_mask = torch.where(attention_mask, 0.0, torch.finfo(dtype).min)
        else:
            additive_padding_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min

        # Adjust mask length to match kv_seq_len
        if input_pad_mask_len < kv_seq_len:
            # Pad the mask. Assume past keys (not covered by original mask) should be attended to (0.0).
            padding_length = kv_seq_len - input_pad_mask_len
            # Pad on the right (consistent with how KV cache grows)
            padding = torch.zeros((bsz, padding_length), dtype=dtype, device=device)
            padded_mask = torch.cat([additive_padding_mask, padding], dim=1)
        elif input_pad_mask_len > kv_seq_len:
            # Truncate the mask if it's longer than needed
            padded_mask = additive_padding_mask[:, :kv_seq_len]
        else:
            padded_mask = additive_padding_mask # Length matches

        # Expand the adjusted padding mask to 4D: [bsz, 1, tgt_len, kv_seq_len]
        # Mask applies to keys/values, so it should cover all query positions against the keys/values
        expanded_padding_mask = padded_mask.unsqueeze(1).unsqueeze(1).expand(bsz, 1, tgt_len, kv_seq_len)

        # Merge masks: take the maximum (minimum numerically, since -inf indicates masking)
        combined_mask = torch.maximum(causal_4d_mask, expanded_padding_mask)
        return combined_mask
    else:
        # No input padding mask provided, return only the causal mask
        return causal_4d_mask


# Attention mechanism (Eager implementation)
class HindiCausalLMAttention(nn.Module):
    """Multi-headed attention using PyTorch's eager implementation."""

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
        # Gemma2 uses GQA, Hindi model uses MHA (num_key_value_groups = 1 implicitly)
        self.num_key_value_heads = self.num_heads # For MHA
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # Will be 1 for MHA

        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attention_dropout = config.attention_probs_dropout_prob # Use configured dropout

        # Use RoPE if configured
        self.positional_encoding_type = getattr(config, "positional_encoding_type", "rope")
        if self.positional_encoding_type == "rope":
            self.rotary_emb = HindiCausalLMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000),
            )
        else:
            self.rotary_emb = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int):
        # Reshapes tensor to [bsz, num_heads, seq_len, head_dim]
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], # Passed from main model
        attention_mask: Optional[torch.Tensor] = None,  # Expect 4D mask [bsz, 1, q_len, kv_seq_len]
        position_ids: Optional[torch.LongTensor] = None,  # Expect [bsz, q_len]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Cache is tuple
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs, # Allow arbitrary kwargs (e.g., from GenerationMixin)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape Q, K, V
        query_states = self._shape(query_states, q_len, bsz, self.num_heads)
        key_states = self._shape(key_states, q_len, bsz, self.num_key_value_heads)
        value_states = self._shape(value_states, q_len, bsz, self.num_key_value_heads)

        # Apply RoPE if configured
        cos, sin = position_embeddings # Get precomputed cos/sin
        if self.rotary_emb is not None:
            if position_ids is None:
                raise ValueError("`position_ids` must be provided when using rotary embeddings.")
            # Apply RoPE to the *current* query and key states using their specific position_ids
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # KV Cache handling (before attention calculation)
        if past_key_value is not None:
            # past_key_value format: (past_key, past_value)
            # Each shape: [bsz, num_key_value_heads, seq_len_past, head_dim]
            past_k, past_v = past_key_value
            # Ensure device consistency
            if past_k.device != key_states.device: past_k = past_k.to(key_states.device)
            if past_v.device != value_states.device: past_v = past_v.to(value_states.device)
            # Concatenate along the sequence length dimension (dim=2)
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        # Store updated K/V states in cache if requested
        present_key_value = (key_states, value_states) if use_cache else None

        # --- Eager Attention Calculation ---
        # Repeat K/V heads if using GQA (num_key_value_groups > 1). For MHA, n_rep = 1.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Calculate attention scores: (Query @ Key.T) / sqrt(head_dim)
        # Q: [bsz, num_heads, q_len, head_dim]
        # K: [bsz, num_heads, kv_seq_len, head_dim]
        # Score: [bsz, num_heads, q_len, kv_seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Check dimensions before applying mask
        final_kv_seq_len = key_states.shape[-2]
        expected_attn_shape = (bsz, self.num_heads, q_len, final_kv_seq_len)
        if attn_weights.size() != expected_attn_shape:
            raise ValueError(
                f"Attention weights should be of size {expected_attn_shape}, but is {attn_weights.size()}"
            )

        # Apply the combined causal and padding mask
        if attention_mask is not None:
            expected_mask_shape = (bsz, 1, q_len, final_kv_seq_len)
            if attention_mask.size() != expected_mask_shape:
                if not (q_len == 0 and attention_mask.size() == (bsz, 1, 0, final_kv_seq_len)): # Allow empty q_len mask
                    raise ValueError(
                        f"Attention mask shape {attention_mask.size()} does not match expected shape "
                        f"{expected_mask_shape}. Check mask preparation logic."
                    )
            attn_weights = attn_weights + attention_mask # Additive mask

        # Upcast attention to fp32 for stability
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training) # Use configured dropout

        # Calculate attention output: (Attention @ Value)
        # Attn: [bsz, num_heads, q_len, kv_seq_len]
        # V:    [bsz, num_heads, kv_seq_len, head_dim]
        # Out:  [bsz, num_heads, q_len, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape attention output back to [bsz, q_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Final output projection
        attn_output = self.o_proj(attn_output)

        # Optionally return attention weights
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


# MLP Layer - Using ACT2FN for flexibility (matches Hindi SiLU if config.hidden_act="silu")
class HindiCausalLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act] # Use ACT2FN based on config
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # Use configured dropout

    def forward(self, x):
        # Implementation follows Llama/Gemma structure (ACT(gate) * up)
        intermediate_act = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(intermediate_act)
        output = self.dropout(output) # Apply dropout
        return output


# Transformer Layer (Hindi Pre-Norm Structure)
class HindiCausalLMLayer(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = HindiCausalLMMLP(config)
        # Use pre-normalization structure
        norm_class = HindiRMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], # Passed down
        attention_mask: Optional[torch.Tensor] = None,         # Expect 4D mask
        position_ids: Optional[torch.LongTensor] = None,      # Expect [bsz, q_len]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Tuple cache
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
            position_embeddings=position_embeddings, # Pass RoPE embeddings
            attention_mask=attention_mask,           # Pass 4D mask
            position_ids=position_ids,               # Pass position_ids
            past_key_value=past_key_value,           # Pass tuple cache
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs, # Pass other kwargs like flash attn params if added later
        )
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1]        # Optional attention weights
        present_key_value = attn_outputs[2]   # Optional tuple cache update

        # Residual connection after attention
        hidden_states = residual + attn_output

        # Fully Connected Block (MLP)
        residual = hidden_states
        # Pre-normalization for MLP
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_norm)
        # Residual connection after MLP
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,) # Append cache update tuple

        return outputs


# PreTrainedModel Base Class
@add_start_docstrings(
    "The bare Hindi Causal LM Model outputting raw hidden-states without any specific head on top.",
    """Initialize the weights.""", # Simplified docstring for base model
)
class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMLayer"]
    _skip_keys_device_placement = ["past_key_values"] # KV cache is stored as tuples
    _supports_flash_attn_2 = False # Set based on implementation
    _supports_sdpa = True # Assume True if using PyTorch >= 2.0
    _supports_cache_class = False # Using tuple cache
    _supports_static_cache = False # Not implemented
    # Add position_ids to ignore list as it's often generated dynamically
    _keys_to_ignore_on_load_missing = [r"lm_head.weight", r"model.rotary_emb.inv_freq", r"rotary_emb.inv_freq"]

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
        elif isinstance(module, (HindiRMSNorm, nn.LayerNorm)): # Check both norm types
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0) # Initialize gains to 1

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HindiCausalLMModel): # Target the base model class
            module.gradient_checkpointing = value


# Core Model - HindiCausalLMModel
HINDI_CAUSAL_LM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Tuple of `tuple(torch.FloatTensor)` of length `config.num_hidden_layers`, with each tuple having 2 tensors
            of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.use_cache=True` contains pre-computed hidden-states (key and values in the self-attention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding. If `past_key_values` are
            used, the user can optionally input only the last `input_ids` (those that don't have their past key value
            states given to this model) of shape `(batch_size, 1)` instead of all `input_ids` of shape `(batch_size,
            sequence_length)`.
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
            Indices depicting the position of the input sequence tokens in the sequence. When using tuple-based cache,
            this is used **implicitly** via the `past_key_values_length` calculation. It is not directly used in the
            forward pass like with `Cache` objects, but the concept is important for understanding KV caching.
"""
@add_start_docstrings(
    "The bare Hindi Causal LM Model outputting raw hidden-states without any specific head on top.",
    HindiCausalLMPreTrainedModel.__doc__, # Use base class docstring
)
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`HindiCausalLMLayer`]
    Args:
        config: HindiCausalLMConfig
    """
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.token_embeddings = self.embed_tokens # Alias for tests
        self.layers = nn.ModuleList([HindiCausalLMLayer(config, i) for i in range(config.num_hidden_layers)])

        # Rotary embeddings moved here from Attention layer if shared across layers
        # If RoPE is calculated per-layer, keep it within the Attention class
        if config.positional_encoding_type == "rope":
             self.rotary_emb = HindiCausalLMRotaryEmbedding(
                 dim=self.embed_tokens.embedding_dim // config.num_attention_heads, # Pass head_dim
                 max_position_embeddings=config.max_position_embeddings,
                 base=config.rope_theta,
                 device=self.embed_tokens.weight.device # Attempt to set device, might need adjustment
             )
        else:
            self.rotary_emb = None

        # Final normalization layer
        norm_class = HindiRMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
        self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.token_embeddings = value # Keep alias consistent

    @can_return_tuple # Allow returning tuple or ModelOutput
    @add_start_docstrings_to_model_forward(HINDI_CAUSAL_LM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,     # Expect 2D mask [bsz, seq_len]
        position_ids: Optional[torch.LongTensor] = None,   # Expect [bsz, seq_len]
        past_key_values: Optional[List[torch.FloatTensor]] = None, # List of tuples
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # Keep for potential future Cache object use
        **kwargs, # Accept arbitrary kwargs (like token_type_ids, which are ignored)
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Handle unused arguments explicitly if they appear in kwargs
        if "token_type_ids" in kwargs:
            logger.info_once("token_type_ids provided but not used by HindiCausalLMModel.")
            kwargs.pop("token_type_ids")

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

        # Determine sequence length and past length for KV cache
        past_key_values_length = 0
        if past_key_values is not None:
            try:
                past_key_values_length = past_key_values[0][0].shape[2] # [B, H, S, D] -> S is index 2
            except (IndexError, TypeError, AttributeError) as e:
                 logger.warning(f"Could not determine past_key_values_length: {e}. Assuming 0.")
                 past_key_values_length = 0

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0) # Shape [1, seq_len]

        # Expand position_ids if batch size > 1 and position_ids has batch size 1
        # Note: During generation with batch>1, position_ids might already be expanded
        if position_ids.shape[0] != batch_size:
            if position_ids.shape[0] == 1:
                position_ids = position_ids.expand(batch_size, -1)
            else:
                 # This shouldn't happen if generation logic is correct
                 raise ValueError(f"Position IDs batch size {position_ids.shape[0]} does not match input batch size {batch_size}")

        # Get input embeddings if needed
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # --- Prepare 4D attention mask ---
        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        # Expected shape: [bsz, 1, q_len, kv_seq_len]

        hidden_states = inputs_embeds

        # --- Calculate Rotary Embeddings ---
        # These are shared across layers
        if self.rotary_emb:
             # Pass the full sequence length required for RoPE calculation (past + current)
             kv_seq_len = seq_length + past_key_values_length
             position_embeddings = self.rotary_emb(hidden_states, seq_len=kv_seq_len) # Pass seq_len
        else:
             position_embeddings = (None, None) # Placeholder if not using RoPE

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

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # inputs: hidden_states, position_embeddings, attention_mask_4d, position_ids
                        return module(
                            inputs[0], # hidden_states
                            position_embeddings=inputs[1], # Pass RoPE embeddings
                            attention_mask=inputs[2],      # Pass 4D mask
                            position_ids=inputs[3],        # Pass position_ids
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                        )
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    position_embeddings, # Pass shared RoPE here
                    attention_mask_4d,
                    position_ids,
                    use_reentrant=False,
                )
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings, # Pass shared RoPE
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],) # Append tuple cache
            if output_attentions:
                if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Causal LM Head Model
@add_start_docstrings(
    """
    The Hindi Causal LM model with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    HindiCausalLMPreTrainedModel.__doc__, # Use base class docstring
)
class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.model = HindiCausalLMModel(config)
        self.hindi_causal_lm = self.model # Alias for tests
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Ensure essential token IDs are present in config for GenerationMixin
        # Use getattr with defaults for safety
        self.config.pad_token_id = getattr(config, "pad_token_id", 0)
        self.config.bos_token_id = getattr(config, "bos_token_id", 1)
        self.config.eos_token_id = getattr(config, "eos_token_id", 2)

        # Initialize weights and potentially tie embeddings
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # tie_weights is handled implicitly by PreTrainedModel if _tied_weights_keys is set.
    # Explicit tie_weights can be kept for clarity or custom logic.
    # def tie_weights(self): ...

    @can_return_tuple
    @add_start_docstrings_to_model_forward(HINDI_CAUSAL_LM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,      # Expect 2D mask
        position_ids: Optional[torch.LongTensor] = None,   # Expect [bsz, seq_len]
        past_key_values: Optional[List[torch.FloatTensor]] = None, # List of tuples
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,         # For loss calculation
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # Keep for compatibility
        **kwargs, # Accept arbitrary kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # Handle unused token_type_ids if passed in kwargs
        if "token_type_ids" in kwargs:
            logger.info_once("token_type_ids provided but not used by HindiCausalLMForCausalLM.")
            kwargs.pop("token_type_ids")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # --- Pass inputs to the base model ---
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # Force return_dict from base model
            cache_position=cache_position,
            **kwargs, # Pass remaining kwargs
        )
        hidden_states = outputs.last_hidden_state

        # --- Calculate LM Logits ---
        logits = self.lm_head(hidden_states)
        logits = logits.float() # Cast to float32 for numerical stability

        # --- Calculate Loss ---
        loss = None
        if labels is not None:
            # Shift so tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # Ensure labels are on the same device as logits before viewing
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # --- Return Outputs ---
        if not return_dict:
            output = (logits,) + outputs[1:] # Combine logits with other outputs from base model
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # prepare_inputs_for_generation - Crucial for generation with KV caching
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None, # Accept cache_position
        position_ids=None,   # Accept position_ids
        use_cache=True,      # Add use_cache argument
        **kwargs
    ):
        # --- Handle KV Cache ---
        past_length = 0
        if past_key_values is not None:
            # If past_key_values exist, we only need the last token of input_ids
            if input_ids.shape[1] > 1:
                 input_ids = input_ids[:, -1:] # Shape: [bsz * beams, 1]
            # Safely determine past_length from the tuple cache structure
            try:
                 past_length = past_key_values[0][0].shape[2]
            except (IndexError, TypeError, AttributeError):
                 past_length = 0 # Reset if cache is invalid

        # --- Handle Position IDs ---
        if position_ids is None:
            # Calculate position_ids based on past length and current length
            current_length = input_ids.shape[1]
            position_ids = torch.arange(
                past_length, past_length + current_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0) # Shape [1, current_length]
        else:
             # If provided, ensure they correspond to the current token(s)
             position_ids = position_ids[:, -input_ids.shape[1]:]

        # --- Handle cache_position (for compatibility with HF generate) ---
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)
        else:
            cache_position = cache_position[-input_ids.shape[1] :]


        # --- Handle inputs_embeds (less common during generation with cache) ---
        if inputs_embeds is not None and past_key_values is None:
            # If providing embeddings for the *full* sequence (no cache)
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # Default case: use input_ids
            model_inputs = {"input_ids": input_ids}

        # --- Attention Mask ---
        # The attention mask needs to cover the full kv_seq_len (past + current)
        # For generation, typically we just need to extend it by one '1' if using cache.
        if attention_mask is not None:
            if past_key_values is not None:
                 # If using cache and mask length matches past length, append '1' for the new token
                 if attention_mask.shape[1] == past_length:
                      mask_extension = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                      attention_mask = torch.cat([attention_mask, mask_extension], dim=1)
                 # Ensure mask length covers past + current
                 elif attention_mask.shape[1] != past_length + input_ids.shape[1]:
                      logger.warning(
                         f"Attention mask length ({attention_mask.shape[1]}) does not match "
                         f"expected kv_seq_len ({past_length + input_ids.shape[1]}). Generation may be unexpected."
                      )
            # Else (no cache), the mask should cover the input_ids length

        # --- Assemble final model inputs ---
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position, # Pass cache_position for potential future use
                # Pass other kwargs directly (they might be generation config args)
                **kwargs,
            }
        )
        return model_inputs


    # generate override - Fixes attribute error and refines padding logic
    def generate(self, *args, **kwargs):
        """
        Generates sequences of token ids for models with a language modeling head.
        Overrides default generate to handle potential custom padding logic.
        """
        # --- Step 1: Call super().generate() ---
        outputs = super().generate(*args, **kwargs)

        # --- Step 2: Extract the sequences tensor ---
        if isinstance(outputs, torch.Tensor):
            sequences_tensor = outputs
            is_output_object = False
        elif hasattr(outputs, "sequences"):
            sequences_tensor = outputs.sequences
            is_output_object = True
        else:
            logger.warning(f"Unexpected output type from super().generate(): {type(outputs)}. Returning original output.")
            return outputs

        # --- Step 3: Apply custom padding logic (Optional) ---
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            return outputs # Return original tensor or object if no padding logic needed

        # Clone the tensor to modify it
        sequences_copy = sequences_tensor.clone()

        for i in range(sequences_copy.size(0)): # Iterate through batch
            seq = sequences_copy[i]
            non_pad_indices = (seq != pad_token_id).nonzero(as_tuple=False).squeeze(-1) # Use as_tuple=False

            if len(non_pad_indices) > 0:
                last_real_token_idx = non_pad_indices[-1].item()
                pad_start_index = last_real_token_idx + 1
                if pad_start_index < seq.size(0):
                    last_real_token_value = seq[last_real_token_idx].item()
                    sequences_copy[i, pad_start_index:] = last_real_token_value

        # --- Step 4: Return the correct type ---
        if is_output_object:
            outputs.sequences = sequences_copy
            return outputs
        else:
            return sequences_copy


    # _reorder_cache - Essential for beam search with tuple cache
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the KV cache (stored as tuples) according to the specified beam indices.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # layer_past is tuple of (key, value)
            # Each shape: [bsz*beams, num_heads, seq_len, dim]
            if layer_past is None or not isinstance(layer_past, tuple) or len(layer_past) != 2:
                # Handle cases where cache might be None or malformed for a layer
                logger.warning("Encountered invalid cache format during reordering. Skipping layer.")
                reordered_past += (None,)
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
                    reordered_layer_past_states.append(None) # Keep None placeholders

            # Ensure the inner tuple always has length 2
            while len(reordered_layer_past_states) < 2:
                reordered_layer_past_states.append(None)

            reordered_past += (tuple(reordered_layer_past_states[:2]),) # Add the reordered (key, value) tuple

        return reordered_past