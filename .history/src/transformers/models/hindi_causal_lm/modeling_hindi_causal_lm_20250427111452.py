# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ...utils import is_torch_available, logging
from ...utils.import_utils import requires_backends
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

# Set constants
_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"

HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "convaiinnovations/hindi-foundational-model-base",
    # See all Hindi Causal LM models at https://huggingface.co/models?filter=hindi_causal_lm
]


# Define classes at module level
class HindiCausalLMPreTrainedModel:
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HindiCausalLMConfig
    base_model_prefix = "hindi_causal_lm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMLayer"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    """
    The Hindi Causal LM base model.
    """

    def __init__(self, config=None):
        requires_backends(self, ["torch"])


class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel):
    """
    Hindi Causal LM model with a language modeling head.
    """

    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config=None):
        requires_backends(self, ["torch"])


# Override with actual implementations when torch is available
if is_torch_available():
    import torch
    import torch.utils.checkpoint
    from torch import nn
    from torch.nn import CrossEntropyLoss

    from ...activations import ACT2FN
    from ...cache_utils import Cache, DynamicCache # Import Cache utilities if using KV Caching
    from ...generation.configuration_utils import GenerationConfig
    from ...generation.utils import GenerationMixin
    from ...modeling_outputs import (
        BaseModelOutputWithPast,
        CausalLMOutputWithPast,
    )
    from ...modeling_utils import PreTrainedModel
    from ...utils import logging # Already imported, but ensure logger is available

    logger = logging.get_logger(__name__) # Re-initialize logger inside the conditional block

    class RMSNorm(nn.Module):
        """
        Root Mean Square Layer Normalization, variant of LayerNorm.
        """

        def __init__(self, hidden_size, eps=1e-6):
            """
            Initialize RMSNorm.

            Args:
                hidden_size: The size of the input tensors
                eps: Small value to avoid division by zero
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, hidden_states):
            """
            Apply RMSNorm to input hidden states.

            Args:
                hidden_states: Input tensor

            Returns:
                Normalized tensor
            """
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return (self.weight * hidden_states).to(input_dtype)

    # Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
    class HindiCausalLMRotaryEmbedding(nn.Module):
        def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
            super().__init__()

            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )
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

        def forward(self, x, seq_len=None):
            # x: [bs, num_attention_heads, seq_len, head_size]
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

            # Return cos and sin caches sliced to the correct seq_len
            return (
                 self.cos_cached[:seq_len].to(dtype=x.dtype),
                 self.sin_cached[:seq_len].to(dtype=x.dtype),
             )

    # Copied from transformers.models.llama.modeling_llama.rotate_half
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    class HindiCausalLMAttention(nn.Module):
        """Multi-headed attention with causal mask and RoPE, adapted from LlamaAttention."""

        def __init__(
            self, config: HindiCausalLMConfig, layer_idx: Optional[int] = None
        ):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            if layer_idx is None:
                 logger.warning_once(
                     f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                     "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                     "when creating this class."
                 )

            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.max_position_embeddings = config.max_position_embeddings
            self.is_causal = True

            if self.head_dim * self.num_heads != self.hidden_size:
                raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                    f" and `num_heads`: {self.num_heads})."
                )

            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

            self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

            self.positional_encoding_type = getattr(config, "positional_encoding_type", "absolute")
            if self.positional_encoding_type == "rope":
                self.rotary_emb = HindiCausalLMRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings
                )
            else:
                self.rotary_emb = None

        def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None, # Use Cache type hint
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            if "padding_mask" in kwargs:
                warnings.warn(
                    "Passing `padding_mask` is deprecated and will be removed in v4.37. Please pass `attention_mask` instead.`",
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
                if self.layer_idx is None:
                     raise ValueError(
                         f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                         "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                         "with a layer index."
                     )
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)


            if self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                # Update K V cache
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)


            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                     raise ValueError(
                         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                     )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = self.attention_dropout(attn_weights)

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

            # Use present_key_value for consistency with other models
            present_key_value = past_key_value if use_cache else None

            return attn_output, attn_weights, present_key_value


    class HindiCausalLMLayer(nn.Module):
        """Transformer layer, adapted from LlamaDecoderLayer."""

        def __init__(self, config: HindiCausalLMConfig, layer_idx: int):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx)
            self.mlp = nn.Sequential(
                nn.Linear(self.hidden_size, config.intermediate_size),
                ACT2FN[config.hidden_act],
                nn.Linear(config.intermediate_size, self.hidden_size),
                nn.Dropout(config.hidden_dropout_prob),
            )

            norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
            self.input_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
            self.layer_idx = layer_idx # Store layer_idx


        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None, # Use Cache type hint
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
             **kwargs,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = attn_outputs[0]
            attn_weights = attn_outputs[1]
            present_key_value = attn_outputs[2] # Use present_key_value


            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (attn_weights,)

            if use_cache:
                outputs += (present_key_value,) # Return present_key_value

            return outputs


    class HindiCausalLMPreTrainedModel(PreTrainedModel): # Define base class again for context
        """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
        """
        config_class = HindiCausalLMConfig
        base_model_prefix = "model"
        supports_gradient_checkpointing = True
        _no_split_modules = ["HindiCausalLMLayer"]
        _skip_keys_device_placement = "past_key_values"
        _supports_flash_attn_2 = False # Update if implemented
        _supports_sdpa = True # Assuming RoPE works with SDPA
        _supports_cache_class = True

        def _init_weights(self, module):
            """Initialize the weights"""
            std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)

        def _set_gradient_checkpointing(self, module, value=False):
             if isinstance(module, HindiCausalLMModel): # Check against the main model class
                 module.gradient_checkpointing = value


    class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
        """
        The Hindi Causal LM base model transformer.
        """

        def __init__(self, config: HindiCausalLMConfig):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size

            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [HindiCausalLMLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
            self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

            self.gradient_checkpointing = False
            self.post_init()

        def get_input_embeddings(self):
            return self.embed_tokens

        def set_input_embeddings(self, value):
            self.embed_tokens = value

        # Copied from transformers.models.llama.modeling_llama.LlamaModel.forward
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None, # Updated type hint
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

            if (input_ids is None) ^ (inputs_embeds is None):
                 raise ValueError(
                     "You cannot specify both input_ids and inputs_embeds at the same time, but must specify either one"
                 )

            if self.gradient_checkpointing and self.training:
                 if use_cache:
                     logger.warning_once(
                         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                     )
                     use_cache = False

            if inputs_embeds is None:
                 inputs_embeds = self.embed_tokens(input_ids)

            # Llama-like KV cache handling
            use_legacy_cache = False
            if use_cache and not isinstance(past_key_values, Cache):
                 past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                 use_legacy_cache = True

            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2" and use_cache:
                is_padding_right = attention_mask[:, -1].sum().item() != attention_mask.shape[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Llama. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )

            if self.config._attn_implementation == "flash_attention_2":
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.config._attn_implementation == "sdpa" and not output_attentions:
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # With respect to computation, this is equivalent to passing a causal mask in addition to the attention_mask
                # input, which is handled by SDPA internally anyway.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (inputs_embeds.shape[0], inputs_embeds.shape[1]), # Use shape from embeds
                    inputs_embeds,
                    past_key_values,
                 )
            else:
                 # 4d mask is passed through the layers
                 attention_mask = _prepare_4d_causal_attention_mask(
                     attention_mask, (inputs_embeds.shape[0], inputs_embeds.shape[1]), inputs_embeds, past_key_values # Use shape from embeds
                 )


            hidden_states = inputs_embeds

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
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1] # Correct indexing

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = None
            if use_cache:
                next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache


            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

    # Helper functions for mask preparation (copied from Llama)
    # Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask
    def _prepare_4d_causal_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Creates a causal 4D mask."""
        # input_shape: (bsz, seq_len)
        bsz, tgt_len = input_shape
        dtype = inputs_embeds.dtype

        # Check if attention_mask is already 4D, otherwise expand
        if attention_mask is not None and attention_mask.dim() == 4:
            # Ensure correct shape if 4D is provided
            expected_shape = (bsz, 1, tgt_len, past_key_values_length + tgt_len)
            if attention_mask.shape != expected_shape:
                raise ValueError(f"Provided 4D attention mask shape {attention_mask.shape} does not match expected {expected_shape}")
            # Ensure it's additive
            if (attention_mask.dtype != torch.float32) and (attention_mask.dtype != torch.float16) and (attention_mask.dtype != torch.bfloat16):
                 raise ValueError("4D attention mask must be additive (float type with 0.0 for attended positions and -inf for masked)")
            return attention_mask

        # Create the combined causal and padding mask
        # [bsz, seq_len] -> [bsz, 1, tgt_len, src_len]
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=inputs_embeds.device)
        mask_cond = torch.arange(mask.size(-1), device=inputs_embeds.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=inputs_embeds.device), mask], dim=-1)

        # Expand padding mask and combine
        if attention_mask is not None: # Should be 2D [bsz, src_len]
            # Expand padding mask to 4D [bsz, 1, tgt_len, src_len]
            padding_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, past_key_values_length + tgt_len)
            # Where padding_mask is False (0 or False means padding), set mask to -inf
            mask = mask[None, None, :, :].expand(bsz, 1, -1, -1) # Expand causal mask
            mask = mask.masked_fill(padding_mask == 0, torch.finfo(dtype).min)

        # Expand final mask for heads: [bsz, 1, tgt_len, src_len]
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, past_key_values_length + tgt_len) if attention_mask is None else mask


    # Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_for_sdpa
    def _prepare_4d_causal_attention_mask_for_sdpa(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Creates a causal 4D mask suitable for SDPA."""
        bsz, query_pos = input_shape # input_shape is (bsz, q_len)
        kv_pos = past_key_values_length + query_pos # k/v seq len

        # If attention_mask is passed, expand it to 4D format expected by SDPA if needed
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                 # [bsz, seq_len] -> [bsz, 1, q_len, kv_len]
                 attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, query_pos, kv_pos).to(torch.bool)
            elif attention_mask.dim() == 4:
                 # Ensure it's boolean and has the correct shape
                 attention_mask = attention_mask.to(torch.bool)
                 expected_shape = (bsz, 1, query_pos, kv_pos)
                 if attention_mask.shape != expected_shape:
                      raise ValueError(f"Provided 4D attention mask shape {attention_mask.shape} does not match expected {expected_shape} for SDPA")
            else:
                 raise ValueError("Attention mask must be 2D or 4D for SDPA")

            # Create the causal mask for SDPA (True means attend)
            causal_mask = torch.tril(torch.ones((query_pos, kv_pos), dtype=torch.bool, device=inputs_embeds.device))
            # Expand causal mask to 4D: [1, 1, q_len, kv_len]
            causal_mask = causal_mask[None, None, :, :]
            # Combine with padding mask: attend only where both masks allow (True)
            attention_mask = attention_mask & causal_mask
        else:
            # No padding mask provided, just create the causal mask for SDPA
            attention_mask = torch.tril(torch.ones((query_pos, kv_pos), dtype=torch.bool, device=inputs_embeds.device))
            # Expand to 4D [bsz, 1, q_len, kv_len]
            attention_mask = attention_mask[None, None, :, :].expand(bsz, 1, -1, -1)


        return attention_mask


    class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel, GenerationMixin):
        _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
        _tied_weights_keys = ["lm_head.weight"]

        def __init__(self, config: HindiCausalLMConfig):
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

        def tie_weights(self):
            if self.config.tie_word_embeddings:
                output_embeddings = self.get_output_embeddings()
                input_embeddings = self.get_input_embeddings()
                if output_embeddings is not None and input_embeddings is not None:
                    output_embeddings.weight = input_embeddings.weight
            super().tie_weights()

        def _untie_weights(self):
            if self.config.tie_word_embeddings:
                output_embeddings = self.get_output_embeddings()
                input_embeddings = self.get_input_embeddings()
                if output_embeddings is not None and input_embeddings is not None:
                    if output_embeddings.weight is input_embeddings.weight:
                        output_embeddings.weight = nn.Parameter(output_embeddings.weight.clone())

        def save_pretrained(
            self,
            save_directory,
            is_main_process=True,
            state_dict=None,
            save_function=None,
            push_to_hub=False,
            max_shard_size="5GB",
            safe_serialization=True,
            variant=None,
            save_peft_format=False,
            **kwargs,
        ):
            weights_were_tied = self.config.tie_word_embeddings and self.get_output_embeddings() is not None
            untied_for_save = False

            if safe_serialization and weights_were_tied:
                try:
                    self._untie_weights()
                    untied_for_save = True
                    logger.info("Untied weights for safe serialization.")
                except Exception as e:
                    logger.warning(f"Failed to untie weights for saving: {e}. Proceeding without untying.")

            result = super().save_pretrained(
                save_directory=save_directory,
                is_main_process=is_main_process,
                state_dict=state_dict,
                save_function=save_function,
                push_to_hub=push_to_hub,
                max_shard_size=max_shard_size,
                safe_serialization=safe_serialization,
                variant=variant,
                save_peft_format=save_peft_format,
                **kwargs,
            )

            if untied_for_save:
                try:
                    self.tie_weights()
                    logger.info("Re-tied weights after saving.")
                except Exception as e:
                    logger.warning(f"Failed to re-tie weights after saving: {e}.")

            return result


        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
            # Omit tokens covered by past_key_values
             past_length = 0
             if past_key_values is not None:
                 if isinstance(past_key_values, Cache):
                     past_length = past_key_values.get_seq_length()
                     max_cache_length = past_key_values.get_max_length()
                 else: # Legacy cache format
                     past_length = past_key_values[0][0].shape[2]
                     max_cache_length = None

                 # Keep only ids that are not already in the cache
                 input_ids_length = input_ids.shape[1]
                 if input_ids_length < past_length:
                      # If the input is smaller than the cache, take the last token (likely inference)
                      input_ids = input_ids[:, -1:]
                 elif input_ids_length > past_length:
                     # If the input is larger, take the part not in the cache
                     input_ids = input_ids[:, past_length:]

             position_ids = kwargs.get("position_ids", None)
             if attention_mask is not None and position_ids is None:
                 # create position_ids on the fly for batch generation
                 position_ids = attention_mask.long().cumsum(-1) - 1
                 position_ids.masked_fill_(attention_mask == 0, 1)
                 if past_key_values:
                      # Adjust position IDs for the new tokens based on cache length
                      position_ids = position_ids[:, past_length:]


             # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
             if inputs_embeds is not None and past_key_values is None:
                 model_inputs = {"inputs_embeds": inputs_embeds}
             else:
                 model_inputs = {"input_ids": input_ids}

             # Update KV cache type if needed
             if isinstance(past_key_values, Cache) and use_legacy_cache:
                 past_key_values = past_key_values.to_legacy_cache()


             model_inputs.update(
                 {
                     "position_ids": position_ids,
                     "past_key_values": past_key_values,
                     "use_cache": kwargs.get("use_cache"),
                     "attention_mask": attention_mask,
                 }
             )
             return model_inputs

        def get_generation_config(self):
            if hasattr(self, "generation_config") and self.generation_config is not None:
                config = self.generation_config
            else:
                config = GenerationConfig()

            config.pad_token_id = self.config.pad_token_id
            config.bos_token_id = self.config.bos_token_id
            config.eos_token_id = self.config.eos_token_id
            config.max_length = getattr(self.config, "max_length", 20)
            config.do_sample = getattr(self.config, "do_sample", True)

            return config

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None, # Keep legacy type hint for now
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
                    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
                    ignored (masked), the loss is only computed for the tokens with labels in `[0, ...,
                    config.vocab_size]`.

            Returns:

            Example:

            ```python
            >>> from transformers import AutoTokenizer, HindiCausalLMForCausalLM
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("convaiinnovations/hindi-foundational-model-base")
            >>> model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-foundational-model-base")

            >>> # Batch size 1
            >>> prompt = "नमस्ते दुनिया"
            >>> inputs = tokenizer(prompt, return_tensors="pt")

            >>> # Generate text
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            '...' # Generated Hindi text
            ```"""
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

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

        # Keep the _reorder_cache staticmethod for beam search
        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            # Handle both legacy and Cache objects
            if isinstance(past_key_values, Cache):
                 return past_key_values.reorder_cache(beam_idx)
            else: # Legacy cache format
                 reordered_past = ()
                 for layer_past in past_key_values:
                     reordered_layer_past = tuple(
                         past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past
                     )
                     reordered_past += (reordered_layer_past,)
                 return reordered_past