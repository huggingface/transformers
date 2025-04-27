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


# Define abstract classes at module level (required by AutoClasses)
class HindiCausalLMPreTrainedModel:
    config_class = HindiCausalLMConfig
    base_model_prefix = "model" # Use "model" for consistency with Llama-like tests
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiCausalLMLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    def __init__(self, config=None):
        requires_backends(self, ["torch"])


class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel):
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
    from ...cache_utils import Cache, DynamicCache
    from ...generation.configuration_utils import GenerationConfig
    from ...generation.utils import GenerationMixin
    from ...modeling_outputs import (
        BaseModelOutputWithPast,
        CausalLMOutputWithPast,
    )
    from ...modeling_utils import PreTrainedModel # Only import PreTrainedModel now
    from ...utils import logging

    logger = logging.get_logger(__name__)

    class RMSNorm(nn.Module):
        """
        Root Mean Square Layer Normalization, variant of LayerNorm.
        """
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, hidden_states):
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
            self._set_cos_sin_cache(
                seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
            )

        def _set_cos_sin_cache(self, seq_len, device, dtype):
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

        def forward(self, x, seq_len=None):
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            return (
                 self.cos_cached[:seq_len].to(dtype=x.dtype),
                 self.sin_cached[:seq_len].to(dtype=x.dtype),
             )

    # Copied from transformers.models.llama.modeling_llama.rotate_half
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # ---- ADD MASK HELPERS HERE ----
    # Adapted from transformers.modeling_utils._prepare_4d_causal_attention_mask
    def _prepare_4d_causal_attention_mask(
        attention_mask: Optional[torch.Tensor], input_shape: Tuple[int, int], inputs_embeds: torch.Tensor, past_key_values_length: int, is_causal: bool = True
    ):
        bsz, tgt_len = input_shape
        dtype = inputs_embeds.dtype
        src_len = past_key_values_length + tgt_len

        if attention_mask is not None and attention_mask.dim() == 4:
            expected_shape = (bsz, 1, tgt_len, src_len)
            if attention_mask.shape != expected_shape: raise ValueError(f"Provided 4D attention mask shape {attention_mask.shape} does not match expected {expected_shape}")
            if not torch.is_floating_point(attention_mask) and attention_mask.dtype != torch.bool: raise ValueError("4D attention mask must be additive (float type with 0.0/-inf) or boolean")
            if torch.is_floating_point(attention_mask): attention_mask = attention_mask.to(dtype)
            return attention_mask

        if attention_mask is None and not is_causal: return None

        # Uses torch.ones to create the lower triangle
        if is_causal:
            mask = torch.ones((tgt_len, src_len), dtype=torch.bool, device=inputs_embeds.device)
            mask = torch.tril(mask, diagonal=past_key_values_length) # Create lower triangular mask
            # Expand to 4D: [1, 1, tgt_len, src_len] and convert True->0, False->-inf
            expanded_mask = torch.zeros_like(mask, dtype=dtype)
            expanded_mask.masked_fill_(~mask, torch.finfo(dtype).min) # Invert mask for additive format
            expanded_mask = expanded_mask[None, None, :, :]
        else:
             # If not causal but mask is needed (e.g., padding), start with zeros
            expanded_mask = torch.zeros((bsz, 1, tgt_len, src_len), dtype=dtype, device=inputs_embeds.device)

        if attention_mask is not None: # 2D padding mask [bsz, src_len]
            if attention_mask.dim() != 2: raise ValueError(f"Attention mask should be 2D but is {attention_mask.dim()}D")
            padding_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
            expanded_mask = expanded_mask.masked_fill(padding_mask == 0, torch.finfo(dtype).min)

        return expanded_mask

    # Adapted from transformers.modeling_utils._prepare_4d_causal_attention_mask_for_sdpa
    def _prepare_4d_causal_attention_mask_for_sdpa(
        attention_mask: Optional[torch.Tensor], input_shape: Tuple[int, int], inputs_embeds: torch.Tensor, past_key_values_length: int
    ):
        bsz, query_pos = input_shape
        kv_pos = past_key_values_length + query_pos
        dtype = torch.bool
        device = inputs_embeds.device

        # Start with causal mask (True means attend)
        attn_mask = torch.ones((query_pos, kv_pos), dtype=dtype, device=device)
        if query_pos > 0 and kv_pos > past_key_values_length:
             attn_mask = torch.tril(attn_mask[:, past_key_values_length:], diagonal=past_key_values_length)
             if past_key_values_length > 0:
                 attn_mask = torch.cat([torch.ones((query_pos, past_key_values_length), dtype=dtype, device=device), attn_mask], dim=1)

        attn_mask = attn_mask[None, None, :, :].expand(bsz, 1, query_pos, kv_pos)

        # Combine with padding mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                 attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, query_pos, kv_pos).to(dtype)
            elif attention_mask.dim() == 4:
                 attention_mask = attention_mask.to(dtype)
                 expected_shape = (bsz, 1, query_pos, kv_pos)
                 if attention_mask.shape != expected_shape: raise ValueError(...)
            else:
                 raise ValueError("Attention mask must be 2D or 4D for SDPA")
            # Combine: attend only where both masks are True
            attn_mask = attn_mask & attention_mask

        return attn_mask
    # ---- END MASK HELPERS ----

    class HindiCausalLMAttention(nn.Module):
        """Multi-headed attention with causal mask and RoPE, adapted from LlamaAttention."""
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

            if self.head_dim * self.num_heads != self.hidden_size:
                raise ValueError("hidden_size must be divisible by num_heads")

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
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
            if "padding_mask" in kwargs:
                warnings.warn("Passing `padding_mask` is deprecated...", FutureWarning)

            bsz, q_len, _ = hidden_states.size()
            query_states = self._shape(self.q_proj(hidden_states), q_len, bsz)
            key_states = self._shape(self.k_proj(hidden_states), q_len, bsz)
            value_states = self._shape(self.v_proj(hidden_states), q_len, bsz)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if self.layer_idx is None:
                     raise ValueError("Layer index needed for cache")
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

            if self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                cos = cos[position_ids]
                sin = sin[position_ids]
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

            # --- SDPA handling or standard matmul ---
            # Using standard matmul for simplicity now
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(f"Attention weights shape error {attn_weights.size()} vs expected")
            if attention_mask is not None:
                 # Ensure mask has correct shape for addition
                 if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                      # Maybe it's boolean and needs conversion? (Should be handled upstream)
                      raise ValueError(f"Attention mask shape error {attention_mask.size()} vs expected {(bsz, 1, q_len, kv_seq_len)}")
                 attn_weights = attn_weights + attention_mask # Additive mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                 raise ValueError(f"Attention output shape error {attn_output.size()} vs expected")

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            present_key_value = past_key_value if use_cache else None
            attn_weights_output = attn_weights if output_attentions else None

            return attn_output, attn_weights_output, present_key_value

    class HindiCausalLMLayer(nn.Module):
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
            self.layer_idx = layer_idx

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
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
            present_key_value = attn_outputs[2]
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                 outputs += (attn_weights,)
            if use_cache:
                 outputs += (present_key_value,)
            return outputs

    class HindiCausalLMPreTrainedModel(PreTrainedModel):
        config_class = HindiCausalLMConfig
        base_model_prefix = "model"
        supports_gradient_checkpointing = True
        _no_split_modules = ["HindiCausalLMLayer"]
        _skip_keys_device_placement = "past_key_values"
        _supports_flash_attn_2 = False
        _supports_sdpa = True
        _supports_cache_class = True

        def _init_weights(self, module):
            std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                 if hasattr(module, 'bias') and module.bias is not None:
                      module.bias.data.zero_()
                 module.weight.data.fill_(1.0)

        def _set_gradient_checkpointing(self, module, value=False):
             if isinstance(module, HindiCausalLMModel):
                 module.gradient_checkpointing = value


    class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
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

        def get_input_embeddings(self): return self.embed_tokens
        def set_input_embeddings(self, value): self.embed_tokens = value

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            # token_type_ids: Optional[torch.LongTensor] = None, # REMOVED from this signature
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

            if (input_ids is None) == (inputs_embeds is None):
                 raise ValueError("You must specify either input_ids or inputs_embeds")

            if self.gradient_checkpointing and self.training:
                 if use_cache:
                     logger.warning_once("use_cache=True is incompatible with gradient checkpointing...")
                     use_cache = False

            past_key_values_length = 0
            if use_cache:
                if past_key_values is None:
                     if inputs_embeds is not None:
                         batch_size, _, hidden_size = inputs_embeds.shape
                         device, dtype = inputs_embeds.device, inputs_embeds.dtype
                     else:
                         batch_size, _ = input_ids.shape
                         device, dtype = input_ids.device, self.embed_tokens.weight.dtype
                         hidden_size = self.config.hidden_size
                     past_key_values = DynamicCache(self.config, batch_size, hidden_size, device=device, dtype=dtype)

                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_seq_length()
            else:
                past_key_values = None
                use_legacy_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            batch_size, seq_length = inputs_embeds.shape[:2]

            if position_ids is None:
                device = inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            # Use the locally defined mask helpers
            if self.config._attn_implementation == "flash_attention_2":
                # Flash attention specific mask preparation
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif attention_mask is not None:
                is_causal = past_key_values_length == 0
                # SDPA expects boolean mask
                is_sdpa = self.config._attn_implementation == "sdpa" and not output_attentions
                if is_sdpa:
                     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                         attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                     )
                else: # Standard implementation expects float additive mask
                     attention_mask = _prepare_4d_causal_attention_mask(
                         attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, is_causal=is_causal
                     )
            # If attention_mask is None, the attention layer handles causal masking

            hidden_states = inputs_embeds
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                # Do NOT pass token_type_ids to the layer forward call
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None, # past_key_value
                        output_attentions,
                        False, # use_cache
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
                    next_decoder_cache = past_key_values # Cache object is updated inplace
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            next_cache = next_decoder_cache.to_legacy_cache() if use_cache and use_legacy_cache else next_decoder_cache

            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


    class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel, GenerationMixin):
        _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
        _tied_weights_keys = ["lm_head.weight"]

        def __init__(self, config: HindiCausalLMConfig):
            super().__init__(config)
            self.model = HindiCausalLMModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def get_input_embeddings(self): return self.model.embed_tokens
        def set_input_embeddings(self, value): self.model.embed_tokens = value
        def get_output_embeddings(self): return self.lm_head
        def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings
        def tie_weights(self): # Method remains the same
            if self.config.tie_word_embeddings:
                output_embeddings, input_embeddings = self.get_output_embeddings(), self.get_input_embeddings()
                if output_embeddings is not None and input_embeddings is not None: output_embeddings.weight = input_embeddings.weight
            super().tie_weights()
        def _untie_weights(self): # Method remains the same
            if self.config.tie_word_embeddings:
                 output_embeddings, input_embeddings = self.get_output_embeddings(), self.get_input_embeddings()
                 if output_embeddings is not None and input_embeddings is not None:
                     if output_embeddings.weight is input_embeddings.weight: output_embeddings.weight = nn.Parameter(output_embeddings.weight.clone())
        def save_pretrained(self, *args, **kwargs): return super().save_pretrained(*args, **kwargs) # Inherit method
        def prepare_inputs_for_generation(self, *args, **kwargs): return super().prepare_inputs_for_generation(*args, **kwargs) # Inherit method
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
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None, # Keep in signature for tests
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:

            # Warn if token_type_ids is provided, as it's ignored by the base model
            if token_type_ids is not None:
                warnings.warn("The `token_type_ids` argument is deprecated and will be ignored for HindiCausalLM.")

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
                # Do NOT pass token_type_ids here to self.model
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
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

            if not return_dict:
                # Keep legacy format for tuple output
                output = (logits,) + outputs[1:] # Use outputs[1:] which contains past_key_values, hidden_states, attentions
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values, # Pass the cache/legacy tuple from base model output
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            if isinstance(past_key_values, Cache):
                 return past_key_values.reorder_cache(beam_idx)
            else:
                 reordered_past = ()
                 for layer_past in past_key_values:
                     reordered_layer_past = tuple(
                         past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past
                     )
                     reordered_past += (reordered_layer_past,)
                 return reordered_past