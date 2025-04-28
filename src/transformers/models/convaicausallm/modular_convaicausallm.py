# coding=utf-8
# Copyright 2024 Convai Innovations and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ConvaiCausalLM model using the modular approach."""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging

# Import LlamaMLP for inheritance
from ..llama.modeling_llama import LlamaMLP

# Import configuration class directly
from .configuration_convaicausallm import ConvaiCausalLMConfig


logger = logging.get_logger(__name__)

# Define CONVAICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC if needed
# e.g. _CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-causal-lm"
#      _CONFIG_FOR_DOC = "ConvaiCausalLMConfig"


# ==== Helper Functions ====

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
# Removed # Copied from comment for make_causal_mask as it might not exist standalone in target llama version
def make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    # Use torch.finfo(dtype).min for FP16/BF16 compatibility
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    # Expand mask to [bsz, 1, tgt_len, src_len]
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Removed # Copied from comment for expand_mask as it might not exist standalone in target llama version
def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    # Use torch.finfo(dtype).min for FP16/BF16 compatibility
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# ==== Custom Attention (No RoPE, GQA) ====
class ConvaiCausalLMAttention(nn.Module):
    """
    Grouped Query Attention module for ConvaiCausalLM. Does not use RoPE.

    Args:
        config (`ConvaiCausalLMConfig`): Model configuration.
        layer_idx (`int`, *optional*):
            Layer index for KV cache. Required if `use_cache=True`.
    """
    def __init__(self, config: ConvaiCausalLMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # Useful for KV caching layer index
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
        # Add attention_dropout from config if it exists, otherwise default to 0.0
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    # Note: repeat_kv method is DEFINED OUTSIDE this class now

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # No RoPE, so position_ids not used for embedding here
        past_key_value: Optional[Tuple[torch.Tensor]] = None, # Assume structure is (key_states, value_states)
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs, # Handles potential cache_position argument if using Cache class later
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape QKV states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # KV Cache update
        # N.B.: Assumes past_key_value is a tuple (key_states, value_states). Adapt if using Cache class.
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KVs for GQA - Call the standalone helper function
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention calculation (Standard Scaled Dot-Product Attention)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)

        # Check shapes before mask application
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
             raise ValueError(
                 f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                 f" {attn_weights.size()}"
             )

        # Apply causal attention mask
        if attention_mask is not None:
             # Expected mask shape: [bsz, 1, q_len, kv_seq_len]
             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                 raise ValueError(
                     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                 )
             attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32 for stability
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training) # Apply dropout

        attn_output = torch.matmul(attn_weights, value_states)

        # Check output shape after matmul
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
             raise ValueError(
                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                 f" {attn_output.size()}"
             )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# ==== MLP (Inherited) ====
class ConvaiCausalLMMLP(LlamaMLP):
    """MLP for ConvaiCausalLM, inheriting directly from LlamaMLP."""
    pass


# ==== Decoder Layer ====
class ConvaiCausalLMDecoderLayer(nn.Module):
    def __init__(self, config: ConvaiCausalLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Use standard LayerNorm
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # Assumes layer_norm_eps in config
        self.self_attn = ConvaiCausalLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = ConvaiCausalLMMLP(config) # Use the explicitly defined inherited MLP
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # Passed to attention, but not used for RoPE
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs, # Handles potential cache_position argument
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # --- Start: Pre-LayerNorm Application ---
        hidden_states_norm = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states_norm, # Apply attention to normalized states
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states_attn = attn_outputs[0]
        self_attn_weights = attn_outputs[1]
        present_key_value = attn_outputs[2]

        hidden_states = residual + hidden_states_attn # First residual connection

        # --- Start: MLP Block ---
        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states) # Normalize before MLP
        hidden_states_mlp = self.mlp(hidden_states_norm)
        hidden_states = residual + hidden_states_mlp # Second residual connection

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ==== PreTrainedModel Base ====
class ConvaiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = ConvaiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ConvaiCausalLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
             if hasattr(module, 'bias') and module.bias is not None:
                  module.bias.data.zero_()
             if hasattr(module, 'weight') and module.weight is not None:
                  module.weight.data.fill_(1.0)


# ==== Main Model ====
class ConvaiCausalLMModel(ConvaiCausalLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    """
    def __init__(self, config: ConvaiCausalLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ConvaiCausalLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
         combined_attention_mask = None
         device = inputs_embeds.device
         dtype = inputs_embeds.dtype
         if input_shape[-1] > 1:
             combined_attention_mask = make_causal_mask(
                 input_shape,
                 dtype,
                 device=device,
                 past_key_values_length=past_key_values_length,
             )

         if attention_mask is not None:
             expanded_attn_mask = expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
             combined_attention_mask = (
                 expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
             )

         return combined_attention_mask

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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = 0
        if past_key_values is not None:
             if hasattr(past_key_values, "__len__") and len(past_key_values) > 0 and hasattr(past_key_values[0], "__len__") and len(past_key_values[0]) > 0:
                 try:
                     past_key_values_length = past_key_values[0][0].shape[2]
                 except (AttributeError, IndexError):
                      logger.warning("Could not determine past_key_values length from structure.")
                      past_key_values_length = 0
             else:
                 logger.warning("past_key_values structure is unexpected.")
                 past_key_values_length = 0

        if position_ids is None:
             position_ids = torch.arange(
                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
             )
             position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

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
                         return module(*inputs, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
                     return custom_forward

                 layer_outputs = torch.utils.checkpoint.checkpoint(
                     create_custom_forward(decoder_layer),
                     hidden_states,
                     attention_mask,
                     position_ids,
                     use_reentrant=False
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
                 present_key_value_index = 2 if output_attentions else 1
                 if len(layer_outputs) > present_key_value_index:
                     next_decoder_cache += (layer_outputs[present_key_value_index],)
                 else:
                     if self.gradient_checkpointing and self.training:
                         pass
                     else:
                         logger.warning_once("KV Cache not found in layer outputs.")

            if output_attentions:
                 attn_weights_index = 1
                 if len(layer_outputs) > attn_weights_index:
                      all_self_attns += (layer_outputs[attn_weights_index],)
                 else:
                    if self.gradient_checkpointing and self.training:
                        pass
                    else:
                        logger.warning_once("Attention weights not found in layer outputs.")

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if self.gradient_checkpointing and self.training:
            next_cache = None

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
                Labels for computing the masked language modeling loss.
        """
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
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            past_key_values_out = outputs.past_key_values if isinstance(outputs, BaseModelOutputWithPast) else outputs[1]
            other_outputs = outputs[2:] if isinstance(outputs, BaseModelOutputWithPast) else outputs[2:]
            output = (logits,) + (past_key_values_out,) + other_outputs
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # <<< REMOVED "# Copied from..." comment for prepare_inputs_for_generation
    def prepare_inputs_for_generation(
         self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
     ):
         past_length = 0
         if past_key_values is not None:
             if hasattr(past_key_values, "__len__"):
                  if len(past_key_values) > 0 and hasattr(past_key_values[0], "__len__") and len(past_key_values[0]) > 0:
                       try:
                            past_length = past_key_values[0][0].shape[2]
                       except (AttributeError, IndexError):
                            logger.warning("Could not determine past_key_values length.")
                            past_length = 0
                  else:
                       logger.warning("past_key_values structure is unexpected.")
             else:
                 logger.warning("past_key_values structure is unexpected.")

         if inputs_embeds is not None and past_key_values is None:
             model_inputs = {"inputs_embeds": inputs_embeds}
         else:
              if input_ids.shape[1] > past_length:
                  input_ids = input_ids[:, past_length:]
              model_inputs = {"input_ids": input_ids}

         position_ids = kwargs.get("position_ids", None)
         if attention_mask is not None and position_ids is None:
             position_ids = attention_mask.long().cumsum(-1) - 1
             position_ids.masked_fill_(attention_mask == 0, 1)
             if past_key_values:
                  position_ids = position_ids[:, past_length:] + past_length

         model_inputs.update(
             {
                 "position_ids": position_ids,
                 "past_key_values": past_key_values,
                 "use_cache": kwargs.get("use_cache"),
                 "attention_mask": attention_mask,
             }
         )
         model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
         return model_inputs
