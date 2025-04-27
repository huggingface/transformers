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
import warnings # <--- ADDED IMPORT
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
    _tied_weights_keys = ["lm_head.weight"] # Added for clarity

    def __init__(self, config=None):
        requires_backends(self, ["torch"])


# Override with actual implementations when torch is available
if is_torch_available():
    import torch
    import torch.utils.checkpoint
    from torch import nn
    from torch.nn import CrossEntropyLoss

    from ...activations import ACT2FN
    from ...generation.configuration_utils import GenerationConfig
    from ...generation.utils import GenerationMixin
    from ...modeling_outputs import (
        BaseModelOutputWithPast, # Use BaseModelOutputWithPast instead of PastAndCrossAttentions
        CausalLMOutputWithPast, # Use CausalLMOutputWithPast instead of PastAndCrossAttentions
    )
    from ...modeling_utils import PreTrainedModel

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
            return (self.weight * hidden_states).to(input_dtype) # Cast back to original dtype

    # Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->HindiCausalLM
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

        def forward(self, x, seq_len=None):
            # x: [bs, num_attention_heads, seq_len, head_size]
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

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
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The dimension along which to unsqueeze cos[position_ids] and sin[position_ids] before applying the term
                wise multiplication. Useful mainly for Conditionnal Generation with beams search applied along the batch
                dimension.

        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        # The 'unsqueeze_dim' argument attempts to handle same logic for graph/non-graph modes USE_GRAPH=False
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


    class HindiCausalLMAttention(nn.Module):
        """Multi-headed attention with causal mask and RoPE, adapted from LlamaAttention."""

        def __init__(self, config: HindiCausalLMConfig, layer_idx: Optional[int] = None): # Add layer_idx if needed for specific init
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx # Store layer index if provided
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.max_position_embeddings = config.max_position_embeddings

            if self.head_dim * self.num_heads != self.hidden_size:
                raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                    f" and `num_heads`: {self.num_heads})."
                )

            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

            self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob) # Added dropout layer

            self.is_causal = True # Standard causal attention

            # RoPE Initialization
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
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs, # Catch potential extra kwargs like head_mask
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            if "padding_mask" in kwargs: # Support older padding_mask argument
                warnings.warn(
                    "Passing `padding_mask` is deprecated and will be removed in v4.37. Please pass `attention_mask` instead.`",
                    FutureWarning,
                )


            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # Reshape q, k, v for multi-head attention
            query_states = self._shape(query_states, q_len, bsz)
            key_states = self._shape(key_states, q_len, bsz)
            value_states = self._shape(value_states, q_len, bsz)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]

            # Apply RoPE if configured
            if self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                # Apply RoPE using position_ids
                # Select the cached cos/sin values using position_ids
                cos = cos[position_ids] # Shape: [bs, seq_len, dim]
                sin = sin[position_ids] # Shape: [bs, seq_len, dim]
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # Scaled Dot-Product Attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            # Check sequence lengths
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            # Apply attention mask (combines causal mask and padding mask)
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                     # Older format might be [bsz, kv_seq_len] -> expand
                    if attention_mask.dim() == 2:
                         attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, q_len, kv_seq_len)
                    else:
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                attn_weights = attn_weights + attention_mask # Additive mask (0 for attend, -inf for mask)


            # Upcast attention to fp32 (recommended for stability)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = self.attention_dropout(attn_weights) # Apply dropout

            attn_output = torch.matmul(attn_weights, value_states)

            # Check output shape after matmul
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            # Reshape output
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            # Final projection
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value


    class HindiCausalLMLayer(nn.Module):
        """Transformer layer, adapted from LlamaDecoderLayer."""

        def __init__(self, config: HindiCausalLMConfig, layer_idx: int): # Add layer_idx
            super().__init__()
            self.hidden_size = config.hidden_size
            self.self_attn = HindiCausalLMAttention(config=config, layer_idx=layer_idx) # Pass layer_idx
            self.mlp = nn.Sequential( # Simplified MLP for now, can enhance later
                nn.Linear(self.hidden_size, config.intermediate_size),
                 ACT2FN[config.hidden_act],
                 nn.Linear(config.intermediate_size, self.hidden_size),
                 nn.Dropout(config.hidden_dropout_prob) # Added dropout
             )

            # Use RMSNorm or LayerNorm based on config
            norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
            self.input_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
             **kwargs, # Catch potential extra kwargs like head_mask
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            """
            Args:
                hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                    `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                position_ids (`torch.LongTensor` of shape `(batch, seq_len)`, *optional*):
                    Indices of positions of each input sequence tokens in the position embeddings. Used for RoPE.
                output_attentions (`bool`, *optional*):
                    Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                    returned tensors for more detail.
                use_cache (`bool`, *optional*):
                    If set to `True`, `past_key_values` key value states are returned and can be used to speed up
                    decoding (see `past_key_values`).
                past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            """
            residual = hidden_states

            # Pre-normalization for self-attention
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            attention_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = attention_outputs[0] # Attention output
            attn_weights = attention_outputs[1] if output_attentions else None
            present_key_value = attention_outputs[2] if use_cache else None

            # First residual connection
            hidden_states = residual + hidden_states

            # Pre-normalization for MLP
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

            # MLP
            hidden_states = self.mlp(hidden_states)

            # Second residual connection
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs

    class HindiCausalLMEncoder(nn.Module): # Renamed to HindiCausalLMDecoder for clarity, though acting as encoder here
        """
        Transformer decoder consisting of `config.num_hidden_layers` layers. Adapted from LlamaModel.
        Acts as the main stack of transformer layers for this Causal LM.
        """

        def __init__(self, config: HindiCausalLMConfig):
            super().__init__()
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.config = config # Store config

            self.layers = nn.ModuleList([HindiCausalLMLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

            # Use RMSNorm or LayerNorm based on config for the final norm
            norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
            self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

            self.gradient_checkpointing = False

        def forward(
            self,
            hidden_states: torch.Tensor, # Changed from input_ids/inputs_embeds; Embeddings handled upstream
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
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

            # Retrieve input_ids and inputs_embeds -> Now handled by HindiCausalLMModel

            # Ensure inputs are on the correct device -> Handled by main model .to(device)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = () if use_cache else None

            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None, # past_key_value is handled by recomputation
                        output_attentions,
                        use_cache, # Check if use_cache needs to be passed here
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],) # KV cache is last or second-last

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states) # Apply final norm

            # add hidden states from the last decoder layer
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

    class HindiCausalLMPreTrainedModel(PreTrainedModel):
        """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
        """
        config_class = HindiCausalLMConfig
        base_model_prefix = "model" # Changed to 'model' to match Llama-like structure
        supports_gradient_checkpointing = True
        _no_split_modules = ["HindiCausalLMLayer"]
        _skip_keys_device_placement = "past_key_values" # Required for Llama-like KV cache handling
        _supports_flash_attn_2 = False # Set to True if Flash Attention is implemented
        _supports_sdpa = True # Set to True if SDP Attention is implemented and preferred
        _supports_cache_class = True # Required for Llama-like KV cache handling

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
             # Enable/disable gradient checkpointing in the main layer stack
             if isinstance(module, HindiCausalLMEncoder): # Keep original class name here for check
                 module.gradient_checkpointing = value


    class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
        """
        The Hindi Causal LM base model transformer. Changed `base_model_prefix` to `model`.

        Args:
            config ([`HindiCausalLMConfig`]): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the
                configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        """

        def __init__(self, config: HindiCausalLMConfig):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size

            # Initialize token embeddings
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )
            # No separate dropout layer needed here if applied within HindiCausalLMEncoder/Layers

            # Initialize encoder (Transformer layers) - Renamed to 'layers'
            self.layers = nn.ModuleList(
                 [HindiCausalLMLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
             )

            # Final normalization layer
            norm_class = RMSNorm if getattr(config, "normalization_layer", "rmsnorm") == "rmsnorm" else nn.LayerNorm
            self.norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

            self.gradient_checkpointing = False # Initialize gradient checkpointing flag

            # Initialize weights and apply final processing
            self.post_init()

        def get_input_embeddings(self):
            return self.embed_tokens

        def set_input_embeddings(self, value):
            self.embed_tokens = value

        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            """
            for layer, heads in heads_to_prune.items():
                self.layers[layer].self_attn.prune_heads(heads) # Access self_attn via layer list


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
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            # --- Prepare inputs ---
            past_key_values_length = 0
            if past_key_values is not None:
                 # Correctly get past length from the cache (shape is [bs, num_heads, seq_len, head_dim])
                 past_key_values_length = past_key_values[0][0].shape[2]


            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0) # Shape: [1, seq_len]

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            # --- Handle attention mask ---
            # Llama-style attention mask handling (for causal mask + padding)
            if attention_mask is not None and attention_mask.dim() == 2:
                # 4d mask is passed through the layers
                 attention_mask = self._update_causal_mask(attention_mask, inputs_embeds, past_key_values_length)


            hidden_states = inputs_embeds # No separate embedding dropout here, handled in layers if needed

            # --- Pass through layers ---
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = () if use_cache else None

            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None, # past_key_value handled by recomputation
                        output_attentions,
                        use_cache,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states) # Apply final norm

            # add hidden states from the last decoder layer
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

        # Helper function for attention mask creation, adapted from Llama
        def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # TODO: make specialization cleaner
            if self.config._attn_implementation == "sdpa":
                # MHA needs a bool mask
                dtype, device = input_tensor.dtype, input_tensor.device
                min_dtype = torch.finfo(dtype).min
                sequence_length = input_tensor.shape[1]
                if attention_mask is None:
                    attention_mask = torch.ones((input_tensor.shape[0], sequence_length + cache_position.shape[-1]), device=device, dtype=torch.bool)
                else:
                    # Ensure attention_mask is bool
                    attention_mask = attention_mask.bool()

                if attention_mask.dim() != 4: # Expand if necessary
                     if attention_mask.dim() == 2:
                         attention_mask = attention_mask[:, None, None, :].expand(-1, 1, sequence_length, -1)
                     else:
                         raise ValueError("Invalid attention mask shape for SDPA")


                # Combine causal mask
                if cache_position.shape[-1] == 0 : # No cache
                    causal_mask = torch.triu(
                        torch.full((sequence_length, sequence_length), min_dtype, dtype=dtype, device=device),
                        diagonal=1,
                    )
                    causal_mask = (causal_mask == 0.0) # Convert to bool
                    # Add batch and head dims
                    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
                    # Apply padding mask
                    attention_mask = attention_mask & causal_mask

                else: # With cache
                     # Need to adjust the causal mask size based on cache position
                     pass # SDPA causal mask handling might need specific logic with KV cache


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
            past_length = 0
            if past_key_values is not None:
                # Cache is not empty, retrieve past length
                past_length = past_key_values[0][0].shape[2] # k_states shape: [bs, num_heads, seq_len, head_dim]

                # Only return last token ids
                input_ids = input_ids[:, -1:]


            position_ids = kwargs.get("position_ids", None)
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1) # Fill masked positions
                if past_key_values:
                    position_ids = position_ids[:, past_length:] # Adjust for past length
            elif position_ids is None:
                 # Generate position ids if not provided
                 position_ids = torch.arange(past_length, input_ids.shape[1] + past_length, dtype=torch.long, device=input_ids.device)
                 position_ids = position_ids.unsqueeze(0) #.expand(input_ids.shape[0], -1)


            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}

            # Add remaining inputs
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
                shift_labels = shift_labels.to(shift_logits.device) # Ensure labels are on correct device
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
            reordered_past = ()
            for layer_past in past_key_values:
                 reordered_layer_past = tuple(
                     past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past
                 )
                 reordered_past += (reordered_layer_past,)
            return reordered_past