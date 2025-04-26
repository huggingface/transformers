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
"""PyTorch HindiCausalLM model."""

import math

from ...utils import is_torch_available, logging
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HindiCausalLMConfig"
_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"

# Check PyTorch availability
if is_torch_available():
    import torch
    import torch.nn.functional as F
    import torch.utils.checkpoint
    from torch import nn
    from torch.nn import CrossEntropyLoss, LayerNorm

    from ...generation import GenerationMixin
    from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from ...modeling_utils import PreTrainedModel
else:
    # Define dummy base classes when PyTorch isn't available
    class PreTrainedModel:
        """Dummy PreTrainedModel class for when PyTorch isn't available."""

        config_class = None
        base_model_prefix = "transformer"

        def __init__(self, config=None, *args, **kwargs):
            self.config = config

        def __call__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install PyTorch to use this model.")

    class GenerationMixin:
        """Dummy GenerationMixin class for when PyTorch isn't available."""

        pass


if is_torch_available():

    class CausalSelfAttention(nn.Module):
        """Causal self-attention layer"""

        def __init__(self, config):
            super().__init__()
            assert config.hidden_size % config.num_attention_heads == 0

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.hidden_size // config.num_attention_heads
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # Query, Key, Value projections
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

            # Output projection
            self.output = nn.Sequential(
                nn.Linear(self.all_head_size, config.hidden_size), nn.Dropout(config.attention_probs_dropout_prob)
            )

            # Causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.triu(
                    torch.ones(config.max_position_embeddings, config.max_position_embeddings) * -1e10, diagonal=1
                ),
            )

        def transpose_for_scores(self, x):
            # Reshape from [batch_size, seq_length, hidden_size] to [batch_size, seq_length, num_heads, head_size]
            new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_shape)
            # Transpose to [batch_size, num_heads, seq_length, head_size]
            return x.permute(0, 2, 1, 3)

        def forward(self, hidden_states, attention_mask=None, output_attentions=False):
            batch_size, seq_length = hidden_states.size()[:2]

            # Project inputs to queries, keys, and values
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            # Scale dot-product attention
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Apply causal mask - prevents attending to future tokens
            causal_mask = self.causal_mask[:seq_length, :seq_length]
            attention_scores = attention_scores + causal_mask

            # Apply attention mask if provided
            if attention_mask is not None:
                # Expand mask to match attention_scores shape
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
                attention_scores = attention_scores + attention_mask

            # Softmax normalization
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = F.dropout(attention_probs, p=0.1, training=self.training)

            # Apply attention to values
            context_layer = torch.matmul(attention_probs, value_layer)

            # Reshape back to [batch_size, seq_length, hidden_size]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_shape)

            # Final output projection
            output = self.output(context_layer)

            outputs = (output,)
            if output_attentions:
                outputs = outputs + (attention_probs,)

            return outputs

    class TransformerBlock(nn.Module):
        """Transformer block with causal attention for language modeling"""

        def __init__(self, config):
            super().__init__()
            self.attention = CausalSelfAttention(config)
            self.attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob),
            )
            self.ffn_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        def forward(self, hidden_states, attention_mask=None, output_attentions=False):
            # Self-attention block with residual connection and layer norm
            attn_outputs = self.attention(
                hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
            )
            attn_output = attn_outputs[0]
            hidden_states = self.attention_layernorm(hidden_states + attn_output)

            # Feed-forward block with residual connection and layer norm
            ffn_output = self.ffn(hidden_states)
            hidden_states = self.ffn_layernorm(hidden_states + ffn_output)

            outputs = (hidden_states,)
            if output_attentions:
                outputs = outputs + (attn_outputs[1] if len(attn_outputs) > 1 else None,)

            return outputs


class HindiCausalLMModel(PreTrainedModel):
    """Hindi Causal Language Model backbone."""

    config_class = HindiCausalLMConfig
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)

        if not is_torch_available():
            raise ImportError(
                "PyTorch must be installed to use HindiCausalLMModel. Please install torch first: pip install torch"
            )

        self.config = config

        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        # Final layer norm
        self.final_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.token_embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not is_torch_available():
            raise ImportError("PyTorch must be installed to use this model.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Process inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        if input_ids is not None:
            device = input_ids.device
            batch_size, seq_length = input_ids.size()
        elif inputs_embeds is not None:
            device = inputs_embeds.device
            batch_size, seq_length = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You must provide either input_ids or inputs_embeds")

        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        # Add position embeddings
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions and len(layer_outputs) > 1:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Apply final layer norm
        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class HindiCausalLMHeadModel(PreTrainedModel):
    """Hindi Causal Language Model with a language modeling head."""

    config_class = HindiCausalLMConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, config):
        super().__init__(config)

        if not is_torch_available():
            raise ImportError(
                "PyTorch must be installed to use HindiCausalLMHeadModel. "
                "Please install torch first: pip install torch"
            )

        self.transformer = HindiCausalLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.token_embeddings.weight

        # Initialize weights
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Prepare inputs for text generation."""
        # Only keep inputs needed for forward pass
        inputs = {
            "input_ids": input_ids,
        }

        # Add attention mask if provided
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        return inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not is_torch_available():
            raise ImportError("PyTorch must be installed to use this model.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=transformer_outputs.hidden_states if return_dict else None,
            attentions=transformer_outputs.attentions if return_dict else None,
        )
