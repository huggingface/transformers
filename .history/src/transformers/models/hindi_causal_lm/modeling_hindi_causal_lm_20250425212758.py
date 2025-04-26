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

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

# Import HF utilities and base classes
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

# Import the adapted configuration
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

# Update checkpoint name if you upload your fine-tuned model
_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"


# --- CausalSelfAttention matching hindi_language_model.py ---
class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer adapted from hindi_language_model.py.
    Uses exact layer names and static causal mask buffer.
    """

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.config = config  # Store config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} not divisible by num_attention_heads {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # --- Use exact layer names from original ---
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Use nn.Sequential for output projection to match original
        self.output = nn.Sequential(
            nn.Linear(self.all_head_size, config.hidden_size),
            nn.Dropout(config.attention_probs_dropout_prob),  # Use config dropout prob
        )
        # --- End exact layer names ---

        # Use static causal mask buffer from original
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((config.max_position_embeddings, config.max_position_embeddings), -float("inf")), diagonal=1
            ),
            # Using -inf is safer than -1e10 for mixed precision
            persistent=False,
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention score calculation."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # [bsz, n_heads, seq_len, head_size]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # External mask (for padding)
        output_attentions: bool = False,  # Add HF standard arg
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:  # Return type hint
        """Forward pass for causal self-attention."""
        batch_size, seq_length, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply static causal mask
        if seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length ({seq_length}) cannot be greater than max_position_embeddings "
                f"({self.config.max_position_embeddings}) when using static causal mask."
            )
        # Ensure mask dtype matches scores dtype for addition
        causal_mask = self.causal_mask[None, None, :seq_length, :seq_length].to(attention_scores.dtype)
        attention_scores = attention_scores + causal_mask

        # Apply external attention mask (for padding) if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expected mask format [bsz, seq_len] -> broadcast format [bsz, 1, 1, seq_len]
                attention_mask = attention_mask[:, None, None, :]
            # Convert 1/0 mask to 0/-inf mask
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores + attention_mask

        # Normalize attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        # Apply dropout *after* softmax (matches original code)
        attention_probs = F.dropout(
            attention_probs, p=self.config.attention_probs_dropout_prob, training=self.training
        )

        # Calculate context layer
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape context layer back to original dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        # Apply final output projection using the nn.Sequential layer 'output'
        output = self.output(context_layer)

        outputs = (output,)
        if output_attentions:
            outputs += (attention_probs,)
        return outputs


# --- TransformerBlock matching hindi_language_model.py ---
class TransformerBlock(nn.Module):
    """
    Transformer block adapted from hindi_language_model.py.
    Uses Post-LN, hardcoded GELU FFN, and exact layer names.
    """

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.config = config
        # --- Use exact layer names ---
        self.attention = CausalSelfAttention(config)
        # Only nn.LayerNorm is used, ignore config.normalization_layer here
        self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # --- FFN: Use nn.Sequential with nn.GELU() exactly as in original ---
        # This ignores config.hidden_act because the original code used GELU
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),  # Hardcoded GELU to match original code's implementation
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),  # Use configured dropout
        )
        # Only nn.LayerNorm is used
        self.ffn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # --- End exact layer names and structure ---

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,  # Add HF standard arg
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]:  # Type hint for output
        """Forward pass for the transformer block using Post-LN."""
        # --- Post-LN implementation ---
        # Self-attention block
        residual = hidden_states
        attn_outputs = self.attention(
            hidden_states,  # Input to attention is pre-norm
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        # Apply layer norm *after* adding residual
        hidden_states = self.attention_layernorm(residual + attn_output)

        # Feed-forward block
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)  # Input to FFN is pre-norm
        # Apply layer norm *after* adding residual
        hidden_states = self.ffn_layernorm(residual + ffn_output)
        # --- End Post-LN implementation ---

        outputs = (hidden_states,)
        if output_attentions:
            outputs += attn_outputs[1:]  # Pass attention probs if generated

        return outputs


# --- Base Model matching hindi_language_model.py ---
class HindiCausalLMModel(PreTrainedModel):
    """
    The base Hindi Causal LM transformer model adapted from hindi_language_model.py.
    Uses standard positional embeddings and Post-LN blocks with GELU FFN.
    """

    config_class = HindiCausalLMConfig  # Link to the updated config
    base_model_prefix = "transformer"  # Standard HF prefix for the base model component
    supports_gradient_checkpointing = True  # Enable gradient checkpointing support
    _no_split_modules = ["TransformerBlock"]  # Hint for model parallelism libraries

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # --- Use exact layer names from original ---
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        # Use standard positional embeddings as in original
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Use list of adapted TransformerBlocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        # --- End exact layer names ---

        # No final LayerNorm in the original base model structure before the LM head

        # Initialize weights and apply final processing based on PreTrainedModel defaults
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Returns the token embedding module."""
        return self.token_embeddings

    def set_input_embeddings(self, value: nn.Embedding):
        """Sets the token embedding module."""
        self.token_embeddings = value

    @add_start_docstrings_to_model_forward(_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # Keep for HF compatibility, but not used
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,  # Keep for HF compatibility, but not used
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        )  # Not implemented here
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is not None or use_cache:
            logger.warning_once("KV Caching is not implemented in this adapted model version.")
            use_cache = False

        # --- Input Processing ---
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # --- Positional IDs ---
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)  # Shape [1, seq_len] -> broadcasts to [bsz, seq_len]

        # --- Get Embeddings ---
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        # Add standard positional embeddings
        if self.config.positional_encoding_type in ["absolute", "learned"]:
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            # Should not happen based on config validation, but fallback
            logger.warning(
                f"Unsupported positional encoding {self.config.positional_encoding_type}. Using only token embeddings."
            )
            hidden_states = inputs_embeds

        hidden_states = self.embedding_dropout(hidden_states)

        # --- Prepare Attention Mask ---
        # The adapted CausalSelfAttention handles the internal causal mask.
        # We only need to pass the padding mask if provided.
        # The attention layer expects the mask to be 0 for padding, 1 for non-padding.
        prepared_attention_mask = attention_mask

        # --- Transformer Layers ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # --- Handle Gradient Checkpointing ---
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # inputs will be hidden_states, attention_mask, output_attentions
                        return module(*inputs)

                    return custom_forward

                # Pass only args required by the layer's forward method
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    prepared_attention_mask,
                    output_attentions,
                    use_reentrant=False,  # Recommended for PyTorch >= 1.11
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=prepared_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # Attention probs are at index 1

        # No final LayerNorm in the original base model structure

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=None,  # KV cache not implemented
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# --- Head Model matching hindi_language_model.py ---
@add_start_docstrings(
    """
    The Hindi Causal LM model with a language modeling head on top (linear layer with weights tied to the input
    embeddings). Matches the structure from hindi_language_model.py.
    """,
    _CONFIG_FOR_DOC,
)
class HindiCausalLMHeadModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "transformer"  # Matches the attribute name below
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _skip_keys_device_placement = "past_key_values"  # Standard HF attribute

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        # --- Use the adapted base model class ---
        self.transformer = HindiCausalLMModel(config)
        self.vocab_size = config.vocab_size
        # --- Use exact layer name from original ---
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # --- End exact layer name ---

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Returns the base model's token embedding module."""
        return self.transformer.get_input_embeddings()  # Delegate to base

    def set_input_embeddings(self, value: nn.Embedding):
        """Sets the base model's token embedding module."""
        self.transformer.set_input_embeddings(value)  # Delegate to base

    def get_output_embeddings(self) -> nn.Linear:
        """Returns the language modeling head."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Sets the language modeling head."""
        self.lm_head = new_embeddings

    # Override tie_weights to use correct embedding layer name from base model
    def tie_weights(self):
        """Ties the weights of the output layer and the input embeddings."""
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            # Use the correct base model embedding layer name
            input_embeddings = self.transformer.token_embeddings
            output_embeddings.weight = input_embeddings.weight
        # Ensure the tied status is updated in the superclass
        super().tie_weights()

    @add_start_docstrings_to_model_forward(_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # Keep for compatibility
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,  # Keep for compatibility
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100.
                Tokens with indices set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with `labels` != None. Setting `use_cache=False`...")
            use_cache = False

        # --- Call the adapted base model ---
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,  # Pass along, though base model ignores it
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,  # Pass along, though base model ignores it
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # --- Apply LM head ---
        lm_logits = self.lm_head(hidden_states)

        # --- Compute Loss ---
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure labels are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # --- Prepare Output ---
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]  # Append hidden states and attentions if returned
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,  # Will be None here
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Prepares inputs for generation. Basic version without KV caching."""

        # Cut input_ids if past_key_values is used (standard HF behavior)
        if past_key_values is not None:
            # In a real KV cache implementation, past_key_values length would be used
            input_ids = input_ids[:, -1:]

        # Prepare position_ids (needed for standard embeddings)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)  # Ensure valid position index 1 for padding
            if past_key_values is not None:
                # If KV cache were used, only the last position ID is needed
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # Assemble model inputs dictionary
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            # "past_key_values": past_key_values, # Add if KV implemented
            "use_cache": kwargs.get("use_cache"),  # Standard HF generation arg
        }
        return model_inputs
