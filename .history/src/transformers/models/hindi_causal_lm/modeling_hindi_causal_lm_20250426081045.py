# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HindiCausalLM model, adapted to match hindi_language_model.py"""

import math
from typing import List, Optional, Tuple, Union

# Import HF utilities and base classes FIRST
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_available, # Import the utility
    logging,
)
from .configuration_hindi_causal_lm import HindiCausalLMConfig

# --- Conditionally import PyTorch and define base class ---
_torch_available = is_torch_available()
if _torch_available:
    import torch
    import torch.utils.checkpoint
    from torch import nn
    from torch.nn import CrossEntropyLoss, functional as F
    _MODEL_BASE_CLASS = nn.Module # Base for custom modules if torch available
else:
    # Define a dummy base class if torch is not available
    _MODEL_BASE_CLASS = object

logger = logging.get_logger(__name__)
# print("--- EXECUTING modeling_hindi_causal_lm.py (v_syntax_fix) ---") # Optional debug print

_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"


# --- CausalSelfAttention matching hindi_language_model.py ---
class CausalSelfAttention(_MODEL_BASE_CLASS):
    """
    Causal self-attention layer adapted from hindi_language_model.py.
    Uses exact layer names and static causal mask buffer (persistent).
    Conditional PyTorch usage.
    """
    def __init__(self, config: HindiCausalLMConfig):
        if not _torch_available: self.config = config; return
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0: raise ValueError(f"hidden_size {config.hidden_size} not divisible by num_attention_heads {config.num_attention_heads}")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Sequential(nn.Linear(self.all_head_size, config.hidden_size), nn.Dropout(config.attention_probs_dropout_prob))
        self.register_buffer("causal_mask", torch.triu(torch.full((config.max_position_embeddings, config.max_position_embeddings), -float('inf')), diagonal=1))

    def transpose_for_scores(self, x: "torch.Tensor") -> "torch.Tensor":
        if not _torch_available: raise ImportError("PyTorch not available")
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size); x = x.view(new_shape); return x.permute(0, 2, 1, 3)

    def forward( self, hidden_states: "torch.Tensor", attention_mask: Optional["torch.Tensor"]=None, output_attentions: bool=False ) -> Union[Tuple["torch.Tensor"], Tuple["torch.Tensor", "torch.Tensor"]]:
        if not _torch_available: raise ImportError("PyTorch not available")
        batch_size, seq_length, _ = hidden_states.size(); query_layer = self.transpose_for_scores(self.query(hidden_states)); key_layer = self.transpose_for_scores(self.key(hidden_states)); value_layer = self.transpose_for_scores(self.value(hidden_states)); attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)); attention_scores = attention_scores / math.sqrt(self.attention_head_size);
        if seq_length > self.config.max_position_embeddings: raise ValueError(f"Seq len {seq_length} > max pos emb {self.config.max_position_embeddings}");
        causal_mask = self.causal_mask[None, None, :seq_length, :seq_length].to(attention_scores.device, dtype=attention_scores.dtype); attention_scores = attention_scores + causal_mask;
        if attention_mask is not None:
            if attention_mask.dim() == 2: attention_mask = attention_mask[:, None, None, :];
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min; attention_mask = attention_mask.to(attention_scores.device, dtype=attention_scores.dtype); attention_scores = attention_scores + attention_mask;
        attention_probs = F.softmax(attention_scores, dim=-1); attention_probs = F.dropout(attention_probs, p=self.config.attention_probs_dropout_prob, training=self.training); context_layer = torch.matmul(attention_probs, value_layer); context_layer = context_layer.permute(0, 2, 1, 3).contiguous(); new_shape = context_layer.size()[:-2] + (self.all_head_size,); context_layer = context_layer.view(new_shape); output = self.output(context_layer); outputs = (output,);
        if output_attentions: outputs += (attention_probs,);
        return outputs


# --- TransformerBlock ---
# Inherit from the conditional base class
class TransformerBlock(_MODEL_BASE_CLASS):
    """
    Transformer block adapted from hindi_language_model.py.
    Uses Post-LN, hardcoded GELU FFN, and exact layer names.
    Conditional PyTorch usage.
    """
    def __init__(self, config: HindiCausalLMConfig):
        if not _torch_available:
            self.config = config
            return
        super().__init__() # Initialize the nn.Module base
        self.config = config
        self.attention = CausalSelfAttention(config)
        self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(), # Hardcoded GELU
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.ffn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # Ensure correct signature and type hints for forward method
    def forward(
        self,
        hidden_states: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None, # Explicitly Optional
        output_attentions: bool = False,
    ) -> Union[Tuple["torch.Tensor"], Tuple["torch.Tensor", Optional["torch.Tensor"]]]:
        """Forward pass for the transformer block using Post-LN."""
        if not _torch_available:
            raise ImportError("PyTorch not available to run forward pass")

        # --- Post-LN implementation ---
        residual = hidden_states
        # Pass arguments explicitly
        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        hidden_states = self.attention_layernorm(residual + attn_output)

        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_layernorm(residual + ffn_output)
        # --- End Post-LN implementation ---

        outputs = (hidden_states,) # Start tuple with hidden_states
        if output_attentions:
            # Append attention tensor if it was returned (attn_outputs has length > 1)
            if len(attn_outputs) > 1:
                outputs += (attn_outputs[1],)
            else:
                 # Append None if attentions weren't requested or returned
                 outputs += (None,)
        # If output_attentions is False, attn_outputs only had length 1,
        # so the final outputs tuple correctly only contains hidden_states.
        return outputs


# --- Head Model (FLATTENED STRUCTURE) ---
@add_start_docstrings( /* ... docstring ... */ _CONFIG_FOR_DOC )
class HindiCausalLMHeadModel(PreTrainedModel, GenerationMixin):
    config_class = HindiCausalLMConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _skip_keys_device_placement = "past_key_values"
    main_input_name = "input_ids"

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config) # MUST BE FIRST
        self.config = config

        if _torch_available:
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        if not hasattr(self, 'token_embeddings'): raise ImportError("PyTorch not available or model not fully initialized.")
        return self.token_embeddings
    def set_input_embeddings(self, value):
        if not _torch_available: raise ImportError("PyTorch not available")
        self.token_embeddings = value
    def get_output_embeddings(self):
        if not hasattr(self, 'lm_head'): return None
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        if not _torch_available: raise ImportError("PyTorch not available")
        self.lm_head = new_embeddings
    def tie_weights(self):
        if not _torch_available: return
        output_embeddings=self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            if hasattr(self, 'token_embeddings'): input_embeddings=self.get_input_embeddings(); output_embeddings.weight=input_embeddings.weight;
            else: logger.warning("Could not tie weights: input embeddings not found.")
        super().tie_weights()

    @add_start_docstrings_to_model_forward(_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional["torch.LongTensor"] = None,
        attention_mask: Optional["torch.Tensor"] = None,
        position_ids: Optional["torch.LongTensor"] = None,
        token_type_ids: Optional["torch.LongTensor"] = None, # Accept token_type_ids
        past_key_values: Optional[List["torch.FloatTensor"]] = None,
        inputs_embeds: Optional["torch.FloatTensor"] = None,
        labels: Optional["torch.LongTensor"] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r""" Forward pass """
        if not _torch_available: raise ImportError("PyTorch not available to run forward pass")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is not None or use_cache: use_cache = False
        if labels is not None and use_cache: use_cache = False

        # --- Input Processing & Get Batch Size/Seq Length ---
        if input_ids is not None and inputs_embeds is not None: raise ValueError("Specify either input_ids or inputs_embeds")
        elif input_ids is not None: batch_size, seq_length = input_ids.shape; self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is not None: batch_size, seq_length, _ = inputs_embeds.shape
        else: raise ValueError("Specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # --- Prepare Positional IDs ---
        if position_ids is None: position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
        # Expand here to handle potential batch size differences early
        if position_ids.shape[0] == 1 and batch_size > 1: position_ids = position_ids.expand(batch_size, -1)


        # --- Get Token Embeddings ---
        if inputs_embeds is None: inputs_embeds = self.token_embeddings(input_ids) # Shape: [batch_size, seq_length, hidden_size]

        # --- Get & Add Positional Embeddings ---
        if self.config.positional_encoding_type in ["absolute", "learned"]:
            effective_seq_length = min(seq_length, self.config.max_position_embeddings)
            position_ids_to_lookup = position_ids[:, :effective_seq_length] # Use potentially expanded position_ids
            position_ids_to_lookup = torch.clamp(position_ids_to_lookup, 0, self.position_embeddings.num_embeddings - 1)
            position_embeds = self.position_embeddings(position_ids_to_lookup) # Shape [batch_size, effective_seq_len, hidden]

            if seq_length > effective_seq_length:
                 padding_needed = seq_length - effective_seq_length
                 padding_tensor = torch.zeros((batch_size, padding_needed, self.config.hidden_size), dtype=position_embeds.dtype, device=device)
                 position_embeds = torch.cat([position_embeds, padding_tensor], dim=1) # Shape [batch_size, seq_length, hidden]
                 logger.warning_once(f"Input seq len {seq_length} > max pos emb {self.config.max_position_embeddings}.")

            # --- Batch sizes should now match after position_ids expansion ---
            hidden_states = inputs_embeds + position_embeds
        else:
             hidden_states = inputs_embeds

        hidden_states = self.embedding_dropout(hidden_states)

        # --- Transformer Layers ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        prepared_attention_mask = attention_mask

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)
            # Use self.is_gradient_checkpointing to check
            if self.is_gradient_checkpointing and self.training:
                 layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, prepared_attention_mask, output_attentions,)
            else: layer_outputs = layer_module(hidden_states, attention_mask=prepared_attention_mask, output_attentions=output_attentions,)
            hidden_states = layer_outputs[0]
            if output_attentions:
                 if len(layer_outputs) > 1: all_self_attns += (layer_outputs[1],)
                 else: all_self_attns += (None,)
        if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)

        # --- Apply LM head ---
        lm_logits = self.lm_head(hidden_states)

        # --- Compute Loss ---
        loss = None
        if labels is not None: shift_logits=lm_logits[..., :-1, :].contiguous(); shift_labels=labels[..., 1:].contiguous(); loss_fct=CrossEntropyLoss(); loss=loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1));

        # --- Prepare Output ---
        if not return_dict:
            outputs_list = [lm_logits, None];
            if output_hidden_states: outputs_list.append(all_hidden_states);
            if output_attentions: outputs_list.append(all_self_attns);
            output = tuple(outputs_list); return ((loss,) + output) if loss is not None else output;
        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=None, hidden_states=all_hidden_states, attentions=all_self_attns,)


    # prepare_inputs_for_generation remains the same
    def prepare_inputs_for_generation( self, input_ids: torch.LongTensor, past_key_values: Optional[List[torch.Tensor]] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs ) -> dict:
        batch_size, seq_length = input_ids.shape
        if past_key_values is not None: input_ids = input_ids[:, -1:]; past_length = past_key_values[0][0].shape[2]; seq_length = past_length + 1; # Illustrative KV cache logic
        else: past_length = 0;
        position_ids = kwargs.get("position_ids", None);
        if attention_mask is not None and position_ids is None:
             position_ids = attention_mask.long().cumsum(-1) - 1;
             position_ids.masked_fill_(attention_mask == 0, 0); # Use 0 for padding
             if past_key_values is not None: position_ids = position_ids[:, -1].unsqueeze(-1);
        # Create position IDs if still None (initial step without attention mask, or not derived)
        if position_ids is None:
            new_seq_len = input_ids.shape[1] # Length of current input_ids
            position_ids = torch.arange(past_length, past_length + new_seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0) # Shape [1, new_seq_len]

        inputs_embeds = kwargs.get("inputs_embeds");
        if inputs_embeds is not None and past_key_values is None: model_inputs = {"inputs_embeds": inputs_embeds}
        else: model_inputs = {"input_ids": input_ids}
        model_inputs.update({"position_ids": position_ids, "attention_mask": attention_mask, "use_cache": kwargs.get("use_cache"),})
        return model_inputs