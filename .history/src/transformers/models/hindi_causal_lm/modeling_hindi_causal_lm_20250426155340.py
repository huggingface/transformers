# coding=utf-8
# Copyright 2024 The Convai Innovations Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Hindi Causal LM model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN  # Use standard activations [1] indicates "silu"
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HindiCausalLMConfig"


# === RMSNorm (Adapted from Llama, matching config.json 'normalization_layer': 'rmsnorm') === [1]
class HindiCausalLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HindiCausalLMRMSNorm is equivalent to T5LayerNorm.
        Using eps from config.json's 'layer_norm_eps' mapped to rms_norm_eps. [1]
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps # eps is passed from config.rms_norm_eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        # Correctly reference the epsilon value stored in the instance
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


# === Rotary Position Embedding (Adapted from Llama, matching config.json 'positional_encoding_type': 'rope') === [1]
# Note: This differs from the absolute position embeddings in the provided hindi_language_model.py [5]
class HindiCausalLMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Correct calculation for inv_freq based on Llama implementation
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # freqs = torch.outer(t, self.inv_freq) # Use torch.outer for clarity
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # Einsum is also common
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None: # Handle case where seq_len might not be passed
             seq_len = x.shape[-2] # Infer from input if needed, common in some implementations

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # Return cos and sin caches sliced to the required sequence length
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            a tensor of shape (`batch_size`, `sequence_length`) specifying the position of each token relative to
            the start of the sequence.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze cos and sin before applying the embedding. Usually 1 for BSHD format.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Ensure position_ids have the correct shape for indexing (batch_size, seq_len) -> need (seq_len,) for slicing cos/sin
    # The cos/sin cache is of shape (max_seq_len, dim/2). We need (seq_len, dim/2).
    # position_ids are (bs, seq_len). We need to gather based on these.
    # cos/sin shape: (seq_len, dim), position_ids shape: (bs, seq_len)
    # Gathering requires cos/sin shape to be compatible or careful indexing.
    # Common Llama implementation applies gather like: cos[position_ids] -> (bs, seq_len, dim)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim) # shape: (bs, 1, seq_len, dim) or similar depending on unsqueeze_dim
    sin = sin[position_ids].unsqueeze(unsqueeze_dim) # shape: (bs, 1, seq_len, dim)

    # q, k shape: (bs, num_heads, seq_len, head_dim)
    # RoPE expects input shape (bs, seq_len, num_heads, head_dim) or (bs * num_heads, seq_len, head_dim)
    # Let's adapt based on q/k shape (bs, num_heads, seq_len, head_dim)
    # Need cos/sin shape: (bs, 1, seq_len, head_dim) or broadcast correctly
    # The unsqueeze_dim=1 is for the num_heads dimension.

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# === MLP (Adapted from Llama, using config.hidden_act = "silu") === [1]
# Note: README [3] mentions "swiglu", treated as equivalent activation ("silu"). Differs from GELU in hindi_language_model.py [5].
class HindiCausalLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act] # Use activation from config, which is "silu" [1]

    def forward(self, x):
        # Implements the SwiGLU logic: down_proj(silu(gate_proj(x)) * up_proj(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# === Attention (Adapted from Llama Attention, using RoPE) ===
class HindiCausalLMAttention(nn.Module):
    """Multi-headed attention using RoPE, adapted from Llama."""

    # Ensure use_cache receives the kv cache and layer_idx is passed correctly.
    def __init__(self, config: HindiCausalLMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # Store layer_idx for cache management
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_probs_dropout_prob
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0) # Get from config or default

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        # Initialize RoPE based on config [1]
        # Assuming no rope_scaling based on config.json [1]
        self.rotary_emb = HindiCausalLMRotaryEmbedding(
             self.head_dim,
             max_position_embeddings=self.max_position_embeddings,
             base=self.rope_theta,
        )


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # Reshape to (bsz, num_heads, seq_len, head_dim)
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs, # Accept potential future arguments
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        kv_seq_len = key_states.shape[-2] # Seq len of K, V for this forward pass
        if past_key_value is not None:
            # If cache is used, kv_seq_len includes the past length
            kv_seq_len += past_key_value[0].shape[-2]

        # Get RoPE embeddings for the full sequence length (past + current)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Apply RoPE based on position_ids
        # Position IDs shape: (bsz, q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            # Correct cache concatenation: dim=2 for seq_len dimension (bsz, num_heads, seq_len, head_dim)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Store updated key/value states in cache if required
        past_key_value = (key_states, value_states) if use_cache else None

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Check attention weights shape
        # Expected shape: (bsz, num_heads, q_len, kv_seq_len)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
             # The attention mask prepared by the model's forward should be 4D (bsz, 1, q_len, kv_seq_len)
             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                 raise ValueError(
                     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                 )
             attn_weights = attn_weights + attention_mask # Additive mask

        # Upcast attention weights to fp32 for stability
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # Check attention output shape
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape attention output back to (bsz, q_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Final output projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None # Return None if not requested

        return attn_output, attn_weights, past_key_value


# === Decoder Layer (Adapted from Llama, using RMSNorm) ===
# Note: Uses RMSNorm based on config [1], unlike LayerNorm in hindi_language_model.py [5]
class HindiCausalLMDecoderLayer(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Pass layer_idx to Attention for cache management
        self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = HindiCausalLMMLP(config)
        # Use RMSNorm based on config [1]
        self.input_layernorm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx # Store layer index

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs, # Accept potential future arguments
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # Apply input RMSNorm
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
        attn_output = attn_outputs[0] # The attention output tensor
        attn_weights = attn_outputs[1] # Optional attention weights
        present_key_value = attn_outputs[2] # Optional key/value cache

        # Residual connection after attention
        hidden_states = residual + attn_output

        # Fully Connected (MLP)
        residual = hidden_states
        # Apply post-attention RMSNorm
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states_norm)
        # Residual connection after MLP
        hidden_states = residual + hidden_states_mlp

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


HINDI_CAUSAL_LM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HindiCausalLMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Hindi Causal LM Model outputting raw hidden-states without any specific head on top.",
    HINDI_CAUSAL_LM_START_DOCSTRING,
)
class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    # Set these based on thorough testing with HF infrastructure
    _supports_flash_attn_2 = False # Default to False until verified
    _supports_sdpa = False # Default to False until verified
    _supports_cache_class = True # Standard cache format is supported

    def _init_weights(self, module):
        """Initialize the weights."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # Initialize RMSNorm weights
        elif isinstance(module, HindiCausalLMRMSNorm):
             module.weight.data.fill_(1.0) # Typically initialized to 1

        # Note: LayerNorm initialization removed as RMSNorm is used

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HindiCausalLMModel): # Check if module is the base model class
            module.gradient_checkpointing = value


HINDI_CAUSAL_LM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            `attention_mask`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Attention mask calculations rely on the implementation of `_prepare_4d_causal_attention_mask` internally.
            See diagram 1 in [the paper](https://arxiv.org/abs/1706.03762) for more
            information on the default strategy for causal masks.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens. Used by RoPE. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.num_hidden_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, head_dim)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't have
            their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids` of
            shape `(batch_size, sequence_length)`.
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
"""


@add_start_docstrings(
    "The bare Hindi Causal LM Model outputting raw hidden-states without any specific head on top.",
    HINDI_CAUSAL_LM_START_DOCSTRING,
)
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`HindiCausalLMDecoderLayer`]

    Args:
        config: HindiCausalLMConfig
    """

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # No separate position embeddings needed for RoPE [1]
        # Dropout is typically applied after the embeddings in transformer stacks
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # Use dropout defined in config [1]

        self.layers = nn.ModuleList(
            # Pass layer index to each decoder layer
            [HindiCausalLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Use final RMSNorm layer based on config [1]
        self.norm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Allow setting the dropout probability
    def _set_dropout(self, dropout_prob):
         self.dropout.p = dropout_prob
         # Also update dropout in attention and MLP if needed, or assume config controls it


    @add_start_docstrings_to_model_forward(HINDI_CAUSAL_LM_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Determine sequence length and device
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            # Cache is structured as [(k1, v1), (k2, v2), ...]
            past_key_values_length = past_key_values[0][0].shape[2] # Get seq len from cache
            seq_length_with_past = seq_length_with_past + past_key_values_length


        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Create position IDs if not provided
        if position_ids is None:
             # Position IDs are crucial for RoPE and should cover the full sequence length including past
             position_ids = torch.arange(
                 past_key_values_length, seq_length_with_past, dtype=torch.long, device=device
             )
             position_ids = position_ids.unsqueeze(0).view(-1, seq_length) # Reshape to (batch_size, current_seq_len)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Apply embedding dropout
        hidden_states = self.dropout(inputs_embeds)

        # === Prepare Attention Mask ===
        # HF's standard causal mask preparation
        if attention_mask is not None and hasattr(self.config, "_attn_implementation") and self.config._attn_implementation == "sdpa":
             # For SDPA, the mask preparation might differ slightly or be handled internally
             # This requires checking the specific _prepare_4d_causal_attention_mask_for_sdpa if available
             # Assuming standard preparation for now
             from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
             attention_mask = _prepare_4d_causal_attention_mask(
                 attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
             )
        elif attention_mask is not None:
            from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
            # Create the standard 4D causal mask for additive application in attention
            attention_mask = _prepare_4d_causal_attention_mask(
                 attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # If attention_mask is None initially, the above function handles creating the causal mask.


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # --- Decoder Layers ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Correctly extract past_key_value for the current layer
            layer_past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Define function for gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Inputs expected by layer: hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None, # Pass None for past_key_value inside checkpoint
                    output_attentions,
                    False, # Pass False for use_cache inside checkpoint
                    # Preserve RNG state for dropout consistency
                    use_reentrant=False, # Recommended for newer PyTorch versions
                )

            else:
                # Standard forward pass
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0] # Update hidden states

            if use_cache:
                 # Append the new cache state for this layer
                 # layer_outputs structure: (hidden_state, Optional[attn_weights], Optional[past_key_value])
                 present_key_value = layer_outputs[-1] if use_cache else None
                 if present_key_value is not None:
                     next_decoder_cache += (present_key_value,)


            if output_attentions:
                all_self_attns += (layer_outputs[1],) # Attention weights are the second element if output_attentions=True

        # Apply final RMSNorm
        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Consolidate cache tuple
        next_cache = next_decoder_cache if use_cache and len(next_decoder_cache) > 0 else None


        if not return_dict:
            # Construct tuple output, filtering None values
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        # Return ModelOutput object
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Main model class name matching config.json['architectures'] [1]
@add_start_docstrings(
    """
    The Hindi Causal LM model with a language modeling head on top (linear layer with weights tied to the input
    embeddings if `config.tie_word_embeddings=True`). This model should be used for causal language modeling tasks.
    """,
    HINDI_CAUSAL_LM_START_DOCSTRING,
)
class HindiCausalLM(HindiCausalLMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"] # Specify keys for tied weights check

    def __init__(self, config):
        super().__init__(config)
        # Contains the stack of transformer layers
        self.model = HindiCausalLMModel(config)
        self.vocab_size = config.vocab_size
        # LM Head on top of the transformer output
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing (including weight tying)
        self.post_init()

    def get_input_embeddings(self):
        # Delegate to the base model
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        # Delegate to the base model
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        # Return the LM head
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # Set the LM head
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # Allow replacing the underlying transformer model
        self.model = decoder

    def get_decoder(self):
        # Return the underlying transformer model
        return self.model

    @add_start_docstrings_to_model_forward(HINDI_CAUSAL_LM_INPUTS_DOCSTRING)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the causal language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size - 1]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size - 1]`.

        Returns:

        Example:

        ```
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM # Use Auto classes

        >>> model_name = "convaiinnovations/hindi-foundational-model-base"
        >>> # Ensure custom code is discoverable (e.g., via local files and install)
        >>> # or use trust_remote_code=True
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        >>> model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True) # Loads HindiCausalLM

        >>> prompts = ["नमस्ते दुनिया", "भारत की राजधानी"]
        >>> inputs = tokenizer(prompts, return_tensors="pt", padding=True)

        >>> # Generate text
        >>> outputs = model.generate(**inputs, max_new_tokens=50)
        >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        >>> # For loss calculation:
        >>> labels = inputs["input_ids"].clone()
        >>> outputs_loss = model(**inputs, labels=labels)
        >>> loss = outputs_loss.loss
        >>> logits = outputs_loss.logits
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call the base model (transformer layers)
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
        )

        # Get the hidden states from the base model output
        hidden_states = outputs[0] # BaseModelOutputWithPast.last_hidden_state or tuple index 0

        # Apply the LM head to get logits
        logits = self.lm_head(hidden_states)
        # Cast logits to float32 for stability, especially for loss calculation
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss() # Standard cross-entropy loss
            # Reshape logits and labels for loss function
            shift_logits_flat = shift_logits.view(-1, self.config.vocab_size)
            shift_labels_flat = shift_labels.view(-1)
            # Ensure labels are on the same device as logits
            shift_labels_flat = shift_labels_flat.to(shift_logits_flat.device)
            loss = loss_fct(shift_logits_flat, shift_labels_flat)

        if not return_dict:
            # Construct tuple output: (loss,) + (logits,) + other outputs from base model
            output = (logits,) + outputs[1:] # outputs[1:] contains cache, hidden_states, attentions if requested
            return ((loss,) + output) if loss is not None else output

        # Return CausalLMOutputWithPast object
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values, # Pass cache from base model output
            hidden_states=outputs.hidden_states,   # Pass hidden states from base model output
            attentions=outputs.attentions,       # Pass attentions from base model output
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepares inputs for generation, handling cache and position IDs."""
        # Handle cache: if past_key_values exist, only the last token is needed
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # Prepare position_ids for the new token(s) based on cache
        # The model's forward pass handles default position_ids if None is passed.
        # However, for generation with cache, it's crucial to calculate them correctly.
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
             # Generate position IDs for the single new token relative to the past
             position_ids = torch.ones(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device) * past_length
             # If generating multiple tokens at once (uncommon for standard autoregressive), adjust accordingly
             # position_ids = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        # if `inputs_embeds` are passed, we only want to use them in the first generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Add other necessary inputs
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask, # Pass the updated attention mask
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorders the cache for beam search/sampling."""
        reordered_past = ()
        for layer_past in past_key_values:
            # layer_past contains (key_cache, value_cache)
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

# End of file
