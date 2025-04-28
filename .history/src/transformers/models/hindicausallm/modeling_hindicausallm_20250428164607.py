# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HindiCausalLM model."""

import math
from functools import partial # For gradient checkpointing with kwargs
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache # Import cache types
from ...modeling_attn_mask_utils import ( # Use new attn mask utils
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput, # Added for completeness
)
from ...modeling_rope_utils import get_rope_buffer # Simplified RoPE handling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings, # For adding examples
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_hindicausallm import HindiCausalLMConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = is_flash_attn_greater_or_equal_2_10()


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HindiCausalLMConfig"
_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-causal-lm" # Add checkpoint for docs


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm and adapted for HindiCausalLM
class HindiCausalLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HindiCausalLMRMSNorm is equivalent to T5LayerNorm or LlamaRMSNorm without bias.
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

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding and adapted for HindiCausalLM
class HindiCausalLMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        cos = self.cos_cached.to(dtype=x.dtype)
        sin = self.sin_cached.to(dtype=x.dtype)

        # Dynamic Cache scaling requires buffers to be initialized using the expected maximum sequence length
        if x.shape[-2] > self.max_seq_len_cached:
             self._set_cos_sin_cache(seq_len=x.shape[-2], device=x.device, dtype=x.dtype)
             cos = self.cos_cached.to(dtype=x.dtype)
             sin = self.sin_cached.to(dtype=x.dtype)
        cos, sin = get_rope_buffer((cos, sin), position_ids)
        return cos, sin


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
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and sin before applying
            the embedding. It should be 1 for inside the attention layer and 2 for outside.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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

# Adapted from transformers.models.llama.modeling_llama.LlamaAttention
class HindiCausalLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper, with GQA and RoPE"""

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

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None, # Added cache_position
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE rotation needs the cos/sin cached values generated by the model
        # Note: This requires the `HindiCausalLMModel` to pass the `position_embeddings`
        # which are generated once by `self.rotary_emb`
        cos, sin = past_key_value.get_seq_length(self.layer_idx) # Assuming model passes rotary_emb output somehow
        # Or alternatively, recompute here if `position_ids` are passed
        # cos, sin = self.rotary_emb(value_states, position_ids=position_ids)

        # Apply RoPE
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        if past_key_value is not None:
            # Update the cache and get the full key/value sequences
            # Needs the `cos` and `sin` for RoPE along with `cache_position`
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention calculation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, key_states.size(-2)): # Use key_states seq length
             raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, key_states.size(-2))}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
             if attention_mask.size() != (bsz, 1, q_len, key_states.size(-2)): # Use key_states seq length
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, key_states.size(-2))}, but is {attention_mask.size()}"
                )
             attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# Adapted from transformers.models.llama.modeling_llama.LlamaMLP
class HindiCausalLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Adapted from transformers.models.llama.modeling_llama.LlamaDecoderLayer
class HindiCausalLMDecoderLayer(nn.Module):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx)

        self.mlp = HindiCausalLMMLP(config)
        self.input_layernorm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
         **kwargs, # Accept arbitrary kwargs for flash usage
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `(batch, seq_len)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.max_position_embeddings - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. Contrarily to
                `position_ids`, this tensor is not affected by padding. It is used to update the cache in the correct
                position and calculate the cached mask.
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
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        # During inference, the Cache object (slot 2) is returned without conditions
        if use_cache:
            outputs += (present_key_value,)


        return outputs


HINDICAUSALLM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HindiCausalLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare HindiCausalLM Model outputting raw hidden-states without any specific head on top.",
    HINDICAUSALLM_START_DOCSTRING,
)
class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = is_flash_attn_2_available() # Support FA2 if installed
    _supports_sdpa = True
    _supports_cache_class = True

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

    def _setup_cache(self, cache_cls: type, max_batch_size: Optional[int], max_cache_len: Optional[int] = None):
        if self.config.cache_implementation == "static":
            if max_cache_len is None:
                 max_cache_len = self.config.max_position_embeddings
            self.model.past_key_values = cache_cls(
                self.config, max_batch_size, max_cache_len, device=self.device, dtype=self.dtype
            )
        elif self.config.cache_implementation == "dynamic":
             self.model.past_key_values = cache_cls()
        else:
            raise ValueError(f"Unsupported cache implementation: {self.config.cache_implementation}")


HINDICAUSALLM_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read specifications of the chosen attention mechanism.
            See the documentation of the chosen attention mechanism for more information regarding the expected padding
            strategy and corresponding attributes (`padding_side`, `padding_strategy`). For more information on how
            `attention_mask` interacts with the chosen attention mechanism, please refer to the documentation of the
            chosen attention mechanism.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up
            sequential decoding. The `Cache` class instance handles the metadata internally, following a key-value format.
            You can initialize this object with the `backend` keyword argument. It contains pre-computed hidden-states (key
            and values in the self-attention blocks) that can be used to speed up sequential decoding. If `past_key_values`
            are used, the user can optionally input only the last `input_ids` (those that don't have their past key value states given to this model)
            of shape `(batch_size, 1)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

            If `past_key_values` are passed, the attention mask needs to be adjusted accordingly when using static cache.

            See [`Cache`] for more details.
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
            this tensor is not affected by padding. It is used to update the cache in the correct position and calculate
            the cached mask. Non-mutable.
"""


@add_start_docstrings(
    "The bare HindiCausalLM Model outputting raw hidden-states without any specific head on top.",
    HINDICAUSALLM_START_DOCSTRING,
)
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`HindiCausalLMDecoderLayer`]

    Args:
        config: HindiCausalLMConfig
    """
    # Adapted from transformers.models.llama.modeling_llama.LlamaModel
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HindiCausalLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Initialize rotary embeddings here
        self.rotary_emb = HindiCausalLMRotaryEmbedding(
            config.hidden_size // config.num_attention_heads, # dim = head_dim
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(HINDICAUSALLM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Compute rotary embeddings for the sequence
        # The call to self.rotary_emb needs the current sequence length determined by cache_position
        cos, sin = self.rotary_emb(inputs_embeds, position_ids=position_ids) # Pass inputs_embeds to get dtype/device
        position_embeddings = (cos, sin) # Pass this tuple to layers

        # Attention mask handling
        attn_implementation = self.config._attn_implementation
        # TODO: @Arthur -- need to check if this works for all attn implementations
        # Eager and SDPA require 4D attention mask
        if attn_implementation in ["eager", "sdpa"] or output_attentions:
             # 4d mask is passed through the layers
             attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
             )
        # FA2 implementation requires 2D mask for kv-cache handling
        elif attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0 not in attention_mask:
                # No padding, assuming causal mask generation is handled inside FA2
                attention_mask = None
            elif past_key_values_length > 0:
                 # Handle padding for FA2 with KV cache
                attention_mask = AttentionMaskConverter._unpad_unattended(attention_mask, past_key_values_length)


        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if use_cache and past_key_values is None:
             # Initialize cache based on config
            cache_cls = StaticCache if self.config.cache_implementation == "static" else DynamicCache
            past_key_values = cache_cls(
                 self.config, inputs_embeds.shape[0], self.config.max_position_embeddings, device=self.device, dtype=self.dtype
            )


        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                 # Helper function for gradient checkpointing with kwargs
                def create_custom_forward(module):
                     def custom_forward(*inputs):
                         # None for past_key_value
                         return module(*inputs)
                     return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None, # Pass None for past_key_value
                    output_attentions,
                    use_cache=False, # Cache must be disabled for GC
                    cache_position=cache_position,
                    use_reentrant=False, # Recommended for PyTorch >= 1.11
                )
                # Update state manually outside checkpointed function if needed (e.g., cache)
                if use_cache:
                     # Recompute attention to get the KV state for caching if GC was used
                     # This is inefficient but necessary if cache is desired with GC
                     with torch.no_grad():
                          noop_outputs = decoder_layer(
                               hidden_states,
                               position_embeddings=position_embeddings, # Pass computed embeddings
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               past_key_value=past_key_values,
                               output_attentions=False, # No need for attentions here
                               use_cache=True,
                               cache_position=cache_position,
                          )
                     # The cache (present_key_value) is the last element if use_cache=True
                     present_key_value = noop_outputs[-1]

            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings, # Pass computed embeddings
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values, # Pass Cache object
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                 # layer_outputs includes hidden_states, (optional attentions), present_key_value
                 next_decoder_cache = layer_outputs[-1] # Cache is the last element

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # `next_cache` is the updated Cache object or None
        next_cache = next_decoder_cache if use_cache else None


        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    """
    The HindiCausalLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    HINDICAUSALLM_START_DOCSTRING,
)
class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HindiCausalLMModel(config)
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

    @add_start_docstrings_to_model_forward(HINDICAUSALLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None, # Use Cache type hint
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # Added cache_position
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the causal language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size - 1]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size - 1]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, HindiCausalLMForCausalLM

        >>> model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-causal-lm")
        >>> tokenizer = AutoTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")

        >>> prompt = "भारत एक विशाल देश है"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        'भारत एक विशाल देश है जो दुनिया के सबसे बड़े लोकतंत्रों में से एक है।'
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
        logits = self.lm_head(hidden_states)
        logits = logits.float() # Cast logits to float32 for stability

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

    # Adapted from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is fixed in shape and cache_position is used
        use_static_cache = isinstance(past_key_values, StaticCache)

        # if the cache is static, the `inputs_ids` should be padded to the full sequence length
        if use_static_cache and attention_mask is not None:
             input_ids = input_ids[:, -1:] # Only need the last token for generation with cache
             if cache_position[0] == 0: # First step
                 # Create a new attention mask for the full sequence length
                 # `cache_position` holds the single index for the new token's position
                 # `past_key_values` holds the max length for the static cache
                 new_mask = torch.ones(
                     (attention_mask.shape[0], past_key_values.get_max_length()),
                     dtype=attention_mask.dtype,
                     device=attention_mask.device,
                 )
                 # Set the single position for the new token to 1
                 new_mask[:, cache_position] = 1
                 attention_mask = new_mask
             else:
                 # Subsequent steps: extend the existing mask
                 new_mask = torch.cat(
                      [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                 )
                 attention_mask = new_mask[:, -past_key_values.get_max_length() :] # Keep the mask size fixed


        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if use_static_cache
                    else None
                )
            else:
                # Handle tuple cache for backward compatibility (although Cache is preferred)
                past_length = past_key_values[0][0].shape[2]
                max_cache_length = None


            # Keep only the unprocessed tokens:
            # 1. If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the input sequence tokens have been cached (i.e. greedy generation).
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -1:]
            # 2. If the past_length is larger than input_ids', then input_ids holds all points and the cache is empty.
            elif past_length < input_ids.shape[1]:
                 input_ids = input_ids[:, past_length:]
            # 3. Where input_ids holds the next token with static cache.
            elif use_static_cache and input_ids.shape[1] == 1:
                 pass # Already handled slicing for static cache above
            # 4. tmp case for dynamic cache fallback
            elif not use_static_cache:
                 input_ids = input_ids[:, -1:]


            if cache_position is None:
                 cache_position = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)
            elif use_static_cache:
                 cache_position = cache_position[-1:] + 1 # Increment position for static cache

            # If the cache is static, the attention mask is fixed size, so we don't need to handle it here as it's already prepared
            # if not use_static_cache and attention_mask is not None:
                 # attention_mask = attention_mask[:, past_length:] # Slice dynamic mask


        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, past_length:] # Slice position_ids for dynamic cache
            elif use_static_cache:
                 position_ids = cache_position.unsqueeze(0) # Use the single cache position for static cache


        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static graph during trace
            model_inputs = {"input_ids": input_ids.contiguous()}

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

    @staticmethod
    # Adapted from transformers.models.llama.modeling_llama.LlamaForCausalLM._reorder_cache
    def _reorder_cache(past_key_values: Cache, beam_idx: torch.LongTensor) -> Cache:
        if past_key_values is None:
            logger.warning("You are attempting to reorder past_key_values (`past_key_values`), but `past_key_values` is None. ")
            return None
        return past_key_values.reorder_cache(beam_idx)



@add_start_docstrings(
    """
    The HindiCausalLM Model transformer with a sequence classification head on top (linear layer).

    [`HindiCausalLMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    HINDICAUSALLM_START_DOCSTRING,
)
class HindiCausalLMForSequenceClassification(HindiCausalLMPreTrainedModel):
    # Adapted from transformers.models.llama.modeling_llama.LlamaForSequenceClassification
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = HindiCausalLMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(HINDICAUSALLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None, # Use Cache type hint
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # Added cache_position
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
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
                # Find the last non-padding token index
                # Clamp indices to avoid -1 index if all tokens are padding
                sequence_lengths = torch.max(
                    (input_ids != self.config.pad_token_id).int().cumsum(dim=1).argmax(dim=1),
                    torch.zeros(1, device=input_ids.device, dtype=torch.long)
                )
            else:
                sequence_lengths = -1
                logger.warning_once(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        # Gather logits for the last token of each sequence
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
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# TokenClassification head can also be added if needed, following Llama pattern:
# class HindiCausalLMForTokenClassification(HindiCausalLMPreTrainedModel): ...