# coding=utf-8
# Copyright 2024 Convai Innovations and The HuggingFace Inc. team. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved. # Added for copied code
# Copyright 2023 The Llama Authors released the Llama v2 model. # For inherited components
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model. # Added for copied code
# It also incorporates components and structures inspired by the Llama v2 implementation.
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
"""PyTorch ConvaiCausalLM model using the modular approach."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

from ...cache_utils import Cache, DynamicCache  # Import Cache API
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging

# Import Llama components for inheritance and utilities
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaMLP, repeat_kv  # Import repeat_kv

# Import configuration class directly
from .configuration_convaicausallm import ConvaiCausalLMConfig


logger = logging.get_logger(__name__)

# Define CONVAICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC if needed
_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-causal-lm"
_CONFIG_FOR_DOC = "ConvaiCausalLMConfig"


# ==== Helper Functions ====
# NOTE: repeat_kv is now imported from llama

# NOTE: Removed potential '# Copied from ...' comments from helper functions below
# as they might be adapted or their source might have changed, causing check_copies errors.


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    # Expand to 4D for compatibility with attention module
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_len, src_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# ==== Custom Attention (No RoPE, GQA, Cache API) ====
class ConvaiCausalLMAttention(nn.Module):
    """
    Grouped Query Attention module for ConvaiCausalLM. Does not use RoPE.
    Uses the Cache API for KV caching.
    """

    def __init__(self, config: ConvaiCausalLMConfig, layer_idx: Optional[int] = None):
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
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True  # Standard for causal LM attention

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Use bias=False consistent with many modern LLMs like Llama
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,  # Needed for Cache API
        past_key_value: Optional[Cache] = None,  # Use Cache API type hint
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:  # Return Cache API type hint
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # KV Cache update logic using Cache API
        if past_key_value is not None:
            # DynamicCache requires cache_position
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Get the full sequence length including past keys/values
        kv_seq_len = key_states.shape[-2]

        # GQA: Repeat KVs before attention calculation
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention calculation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)

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

        # Return the Cache object itself when use_cache is True
        return attn_output, attn_weights, past_key_value


# ==== MLP (Inherited) ====
class ConvaiCausalLMMLP(LlamaMLP):
    """MLP for ConvaiCausalLM, inheriting directly from LlamaMLP."""

    def __init__(self, config):
        super().__init__(config)
        # Ensure bias matches if specified in config (LlamaMLP respects config.bias)
        # ConvaiCausalLMConfig currently does not have a 'bias' field, so LlamaMLP might default bias=True.
        # If Convai *must* have bias=False in MLP, update ConvaiCausalLMConfig or override here.


# ==== Decoder Layer (Inherited and Modified) ====
class ConvaiCausalLMDecoderLayer(LlamaDecoderLayer):
    """
    ConvaiCausalLM Decoder Layer using standard LayerNorm and custom Attention.
    Inherits the forward pass structure from LlamaDecoderLayer (Pre-LayerNorm).
    """

    def __init__(self, config: ConvaiCausalLMConfig, layer_idx: int):
        # Skip the parent's __init__ if it causes issues (e.g., RoPE init)
        # super(LlamaDecoderLayer, self).__init__() # Call grandparent's init
        # Instead, directly initialize needed attributes if parent init is problematic
        nn.Module.__init__(self)  # Safest way to initialize if parent __init__ is incompatible

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx  # Store layer_idx needed by attention

        # Override the attention module with our custom one
        self.self_attn = ConvaiCausalLMAttention(config=config, layer_idx=layer_idx)

        # Override the MLP module (inherits from LlamaMLP)
        self.mlp = ConvaiCausalLMMLP(config)

        # Override the normalization layers to use standard LayerNorm instead of Llama's RMSNorm
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # We inherit the forward method from LlamaDecoderLayer.
    # It executes: residual -> input_norm -> attention -> hidden_states += residual
    #             -> residual -> post_attn_norm -> mlp -> hidden_states += residual
    # This inherited method will automatically use the overridden modules defined above.


# ==== PreTrainedModel Base ====
class ConvaiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = ConvaiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ConvaiCausalLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False  # Using standard attention
    _supports_sdpa = False  # Using standard attention
    _supports_cache_class = True  # Supports Cache API

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
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm bias to 0 and weight to 1
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:  # LayerNorm always has weight
                module.weight.data.fill_(1.0)


# ==== Main Model ====
class ConvaiCausalLMModel(ConvaiCausalLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.

    Args:
        config: ConvaiCausalLMConfig
    """

    def __init__(self, config: ConvaiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # Use the (potentially inherited) ConvaiCausalLMDecoderLayer
        self.layers = nn.ModuleList(
            # Pass layer_idx to the DecoderLayer constructor
            [ConvaiCausalLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Use standard LayerNorm for the final normalization
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # NOTE: Removed '# Copied from ...' comment for _prepare_decoder_attention_mask
    # Treat this as potentially custom or adapted logic.
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Creates causal attention mask for decoding. Handles padding mask if provided."""
        # Create causal mask for decoder generation with cache.
        combined_attention_mask = None
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        bsz, seq_len = input_shape

        if seq_len > 1:
            # Uses the helper defined at the top of the file
            combined_attention_mask = _make_causal_mask(
                (bsz, seq_len),  # Pass correct shape
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # Uses the helper defined at the top of the file
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=seq_len).to(device)
            if combined_attention_mask is not None:
                combined_attention_mask = expanded_attn_mask + combined_attention_mask
            else:
                combined_attention_mask = expanded_attn_mask

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,  # Use Cache API type hint
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
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0
        # Initialize cache and potentially retrieve past length
        if use_cache:
            if past_key_values is None:
                # Initialize a default DynamicCache if none provided and caching is enabled
                past_key_values = DynamicCache()
            # Get length from cache
            # Use layer_idx 0 as representative, assuming all layers have the same cache length
            past_key_values_length = past_key_values.get_seq_length(self.layers[0].layer_idx)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)  # Shape: [1, seq_length]

        # If position_ids are provided, ensure they are the correct shape [bsz, seq_length]
        # Note: DynamicCache expects position_ids shape [bsz * num_heads, ...] or similar if passed in cache_kwargs
        # but standard practice is to pass [bsz, seq_len] or [1, seq_len] and let the attention module handle it.
        # The position_ids passed to the layer forward should be [bsz, seq_len].
        if position_ids is not None:
            position_ids = position_ids.view(batch_size, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Embed positions - This model does not use explicit positional embeddings.
        # It relies on the causal mask and the implicit order of tokens.

        # Prepare attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        # attention_mask shape should be [bsz, 1, q_len, kv_seq_len]

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
        # next_decoder_cache is now represented by the final state of past_key_values

        for decoder_layer in self.layers:  # No need for layer_idx here if layer stores it
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Custom forward function for gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Cache needs to be passed correctly. Checkpoint expects tensors.
                        # This might require adapting how cache is handled with checkpointing.
                        # For now, assume standard checkpointing without complex cache handling:
                        # return module(*inputs, output_attentions, None) # Pass None for cache

                        # If checkpointing needs to support cache object, it's more complex.
                        # Let's stick to the simpler incompatibility warning for now.
                        # The warning above should set use_cache=False anyway.
                        return module(
                            inputs[0],  # hidden_states
                            attention_mask=inputs[1],
                            position_ids=inputs[2],
                            past_key_value=None,  # Cache is disabled by warning
                            output_attentions=output_attentions,
                            use_cache=use_cache,  # Will be False here
                        )

                    return custom_forward

                # Inputs to checkpoint must be tensors or tuples/lists of tensors
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,  # Might need manipulation if it's not always needed/compatible
                    position_ids,
                    use_reentrant=False,  # Recommended for newer PyTorch versions
                )

            else:
                # Regular forward pass
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,  # Pass the whole cache object
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            # Cache is managed internally by the Cache object passed to the layer
            # layer_outputs[2] (or layer_outputs[1] if no attentions) is the updated Cache object
            if use_cache:
                past_key_values = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # next_cache is the final state of the past_key_values object
        next_cache = past_key_values if use_cache else None
        if self.gradient_checkpointing and self.training:
            next_cache = None  # Ensure cache is not returned during checkpointing

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ==== Causal LM Head Model ====
class ConvaiCausalLMForCausalLM(ConvaiCausalLMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = ConvaiCausalLMModel(config)
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,  # Use Cache API type hint
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
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns: CausalLMOutputWithPast

        Example:

        ```python
        >>> from transformers import AutoTokenizer, ConvaiCausalLMForCausalLM
        >>> import torch

        >>> # Ensure the custom code is registered if not using main branch transformers
        >>> # from .modeling_convaicausallm import ConvaiCausalLMForCausalLM # (Auto registration should work if setup correctly)
        >>> # from .configuration_convaicausallm import ConvaiCausalLMConfig
        >>> # from .tokenization_convaicausallm import ConvaiCausalLMTokenizer # (If custom tokenizer class exists)

        >>> model_name = "convaiinnovations/hindi-causal-lm"
        >>> # Assuming tokenizer is available at a different location or under the same name
        >>> tokenizer = AutoTokenizer.from_pretrained("convaiinnovations/hindi-embedding-foundational-model")
        >>> model = ConvaiCausalLMForCausalLM.from_pretrained(model_name)
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model.to(device)

        >>> prompt = "भारत एक विशाल देश है"
        >>> inputs = tokenizer(prompt, return_tensors="pt").to(device)

        >>> # Generate text
        >>> outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.8, top_k=50)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # Expected output might be: भारत एक विशाल देश है। यहाँ विभिन्न संस्कृतियाँ और भाषाएँ पाई जाती हैं। देश की राजधानी नई दिल्ली है। यह एक लोकतांत्रिक गणराज्य है जहाँ ... (Example continuation)
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs consists of (last_hidden_state, past_key_values, hidden_states, attentions)
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
        logits = logits.float()  # Cast to float32 for stability before loss calculation

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

        if not return_dict:
            # Handle case where outputs is a tuple vs BaseModelOutputWithPast
            past_key_values_out = outputs[1] if isinstance(outputs, tuple) else outputs.past_key_values
            other_outputs = outputs[2:] if isinstance(outputs, tuple) else (outputs.hidden_states, outputs.attentions)
            output = (logits,) + (past_key_values_out,) + other_outputs
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,  # past_key_values is the Cache object
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # NOTE: Removed potential '# Copied from ...' comment. Treat as adapted.
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepares inputs for generation, handling cache usage."""

        # Omit tasks in kwargs if using Cache API and have already been handled

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # if possible, send only the last token ID for efficiency
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]  # Select the last token
            model_inputs = {"input_ids": input_ids}

        # Get past length from Cache object. Assumes layer_idx 0 is representative.
        if past_key_values is not None:
            try:
                # Attempt to get seq_length from the first layer's cache
                past_key_values.get_seq_length(layer_idx=0)
            except Exception:
                # Fallback or warning if cache structure is unexpected or empty
                logger.warning("Could not determine past_key_values length. Assuming 0.")

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                # If we have past_key_values, we only need the position id for the *new* token
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # Prepare final model inputs dictionary
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,  # Pass full mask, _prepare_decoder_attention_mask handles slicing/causal
            }
        )
        # Remove None values
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        return model_inputs
