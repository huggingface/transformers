# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
# (License header...)
"""PyTorch HindiCausalLM model, adapted to match hindi_language_model.py"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, functional as F

# Import HF utilities and base classes
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
# --- !! ADD GenerationMixin IMPORT !! ---
from transformers.generation import GenerationMixin
# --- !! END ADD !! ---
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

# Import the adapted configuration
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"


# --- CausalSelfAttention class (remains the same) ---
class CausalSelfAttention(nn.Module):
    # ... (Keep the code from the previous correct version) ...
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.config = config # Store config
        if config.hidden_size % config.num_attention_heads != 0:
             raise ValueError(f"hidden_size {config.hidden_size} not divisible by num_attention_heads {config.num_attention_heads}")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Sequential(
            nn.Linear(self.all_head_size, config.hidden_size),
            nn.Dropout(config.attention_probs_dropout_prob)
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((config.max_position_embeddings, config.max_position_embeddings), -float('inf')), diagonal=1)
        )
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.size()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if seq_length > self.config.max_position_embeddings:
             raise ValueError(f"Sequence length ({seq_length}) > max_position_embeddings ({self.config.max_position_embeddings})")
        causal_mask = self.causal_mask[None, None, :seq_length, :seq_length].to(attention_scores.device, dtype=attention_scores.dtype)
        attention_scores = attention_scores + causal_mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                 attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
            attention_mask = attention_mask.to(attention_scores.device, dtype=attention_scores.dtype)
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = F.dropout(attention_probs, p=self.config.attention_probs_dropout_prob, training=self.training)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        output = self.output(context_layer)
        outputs = (output,)
        if output_attentions:
            outputs += (attention_probs,)
        return outputs


# --- TransformerBlock class (remains the same) ---
class TransformerBlock(nn.Module):
    # ... (Keep the code from the previous correct version) ...
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.config = config
        self.attention = CausalSelfAttention(config)
        self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.ffn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        residual = hidden_states
        attn_outputs = self.attention(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.attention_layernorm(residual + attn_output)
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_layernorm(residual + ffn_output)
        outputs = (hidden_states,)
        if output_attentions:
            if len(attn_outputs) > 1: outputs += (attn_outputs[1],)
            else: outputs += (None,)
        return outputs


# --- Head Model matching hindi_language_model.py (FLATTENED STRUCTURE) ---
@add_start_docstrings(
    # ... (docstring)
    _CONFIG_FOR_DOC,
)
# --- !! EXPLICITLY ADD GenerationMixin !! ---
class HindiCausalLMHeadModel(PreTrainedModel, GenerationMixin):
# --- !! END ADD !! ---
    config_class = HindiCausalLMConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _skip_keys_device_placement = "past_key_values"

    # --- Ensure __init__ is correct ---
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config) # Calls PreTrainedModel init
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init() # Calls _init_weights and tie_weights

    # --- Embedding getters/setters ---
    def get_input_embeddings(self) -> nn.Embedding: return self.token_embeddings
    def set_input_embeddings(self, value: nn.Embedding): self.token_embeddings = value
    def get_output_embeddings(self) -> nn.Linear: return self.lm_head
    def set_output_embeddings(self, new_embeddings: nn.Linear): self.lm_head = new_embeddings

    def tie_weights(self):
        # ... (tie_weights code remains the same) ...
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            input_embeddings = self.get_input_embeddings()
            output_embeddings.weight = input_embeddings.weight
        super().tie_weights()


    # --- Forward method (remains the same) ---
    @add_start_docstrings_to_model_forward(_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None,
        use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # ... (forward implementation remains the same) ...
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if past_key_values is not None or use_cache: use_cache = False
        if labels is not None and use_cache: use_cache = False
        if input_ids is not None and inputs_embeds is not None: raise ValueError("Specify either input_ids or inputs_embeds")
        elif input_ids is not None: batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None: batch_size, seq_length, _ = inputs_embeds.shape
        else: raise ValueError("Specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
        elif position_ids.shape[0] == 1 and batch_size > 1 : position_ids = position_ids.expand(batch_size, -1)
        if inputs_embeds is None: inputs_embeds = self.token_embeddings(input_ids)
        if self.config.positional_encoding_type in ["absolute", "learned"]:
            if seq_length > self.position_embeddings.num_embeddings:
                 position_ids_to_use = torch.arange(self.position_embeddings.num_embeddings, dtype=torch.long, device=device).unsqueeze(0)
                 position_embeds = self.position_embeddings(position_ids_to_use)
                 padding_needed = seq_length - position_embeds.shape[1]
                 if padding_needed > 0:
                     padding_tensor = torch.zeros((batch_size, padding_needed, self.config.hidden_size), dtype=position_embeds.dtype, device=device)
                     position_embeds = torch.cat([position_embeds.expand(batch_size, -1, -1), padding_tensor], dim=1)
                 else: position_embeds = position_embeds.expand(batch_size, -1, -1)
                 logger.warning_once(f"Input seq length {seq_length} > max pos embeddings {self.position_embeddings.num_embeddings}.")
            else:
                 position_ids_to_use = torch.clamp(position_ids, 0, self.position_embeddings.num_embeddings - 1)
                 position_embeds = self.position_embeddings(position_ids_to_use)
            hidden_states = inputs_embeds + position_embeds
        else: hidden_states = inputs_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        prepared_attention_mask = attention_mask
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                 layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, prepared_attention_mask, output_attentions,)
            else: layer_outputs = layer_module(hidden_states, attention_mask=prepared_attention_mask, output_attentions=output_attentions,)
            hidden_states = layer_outputs[0]
            if output_attentions: all_self_attns += (layer_outputs[1],)
        if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if not return_dict:
            outputs_list = [lm_logits, None]
            if output_hidden_states: outputs_list.append(all_hidden_states)
            if output_attentions: outputs_list.append(all_self_attns)
            output = tuple(outputs_list)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=None, hidden_states=all_hidden_states, attentions=all_self_attns,)


    # --- prepare_inputs_for_generation (remains the same) ---
    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[torch.Tensor]] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> dict:
        # ... (Keep the code from the previous correct version) ...
        if past_key_values is not None: input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
             position_ids = attention_mask.long().cumsum(-1) - 1
             position_ids.masked_fill_(attention_mask == 0, 1)
             if past_key_values is not None: position_ids = position_ids[:, -1].unsqueeze(-1)
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None and past_key_values is None: model_inputs = {"inputs_embeds": inputs_embeds}
        else: model_inputs = {"input_ids": input_ids}
        model_inputs.update({"attention_mask": attention_mask, "position_ids": position_ids, "use_cache": kwargs.get("use_cache"),})