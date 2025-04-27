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

from ...generation.configuration_utils import GenerationConfig
from ...generation.utils import GenerationMixin
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
    _keys_to_ignore_on_load_missing = []

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

    def __init__(self, config=None):
        requires_backends(self, ["torch"])


# Override with actual implementations when torch is available
if is_torch_available():
    import torch
    import torch.utils.checkpoint
    from torch import nn
    from torch.nn import CrossEntropyLoss

    from ...activations import ACT2FN
    from ...modeling_outputs import (
        BaseModelOutputWithPastAndCrossAttentions,
        CausalLMOutputWithCrossAttentions,
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
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            # Convert to half-precision if necessary
            if self.weight.dtype != hidden_states.dtype:
                hidden_states = hidden_states.to(self.weight.dtype)
            return self.weight * hidden_states


    class HindiCausalLMAttention(nn.Module):
        """Multi-headed attention with causal mask specifically for Hindi Causal LM."""

        def __init__(self, config):
            super().__init__()
            if config.hidden_size % config.num_attention_heads != 0:
                raise ValueError(
                    f"The hidden size ({config.hidden_size}) is not divisible by the number of attention heads "
                    f"({config.num_attention_heads})"
                )

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # Query, Key, and Value projections
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

            # Output projection
            self.output = nn.Linear(self.all_head_size, config.hidden_size)
            self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
            self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

            # Causal mask to ensure attention only attends to previous tokens
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(config.max_position_embeddings, config.max_position_embeddings) * -1e10, diagonal=1),
            )

            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
            self.max_position_embeddings = config.max_position_embeddings
            self.positional_encoding_type = getattr(config, "positional_encoding_type", "absolute")

            # Initialize for RoPE if used
            if self.positional_encoding_type == "rope":
                # Ensure rotary dimension is not greater than attention head size
                self.rotary_dim = min(self.attention_head_size, getattr(config, "rotary_dim", self.attention_head_size))
                # Create and cache positions
                inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
                self.register_buffer("inv_freq", inv_freq)

        def _transpose_for_scores(self, x):
            """Reshape from [batch_size, seq_length, hidden_size] to [batch_size, num_heads, seq_length, head_size]"""
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        def _rotate_half(self, x):
            """Rotary position embedding helper function"""
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def _apply_rotary_pos_emb(self, q, k, cos, sin):
            """Apply rotary position embeddings to query and key tensors safely."""
            # Ensure our dimensions align - handle case where rotary_dim < attention_head_size
            if self.rotary_dim < self.attention_head_size:
                # Only apply rotation to subset of dimensions
                q_rot = q[..., :self.rotary_dim]
                q_pass = q[..., self.rotary_dim:]
                k_rot = k[..., :self.rotary_dim]
                k_pass = k[..., self.rotary_dim:]

                # Apply rotation only to the subset
                q_rot_embed = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
                k_rot_embed = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

                # Concatenate back with unchanged part
                q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
                k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)

                return q_embed, k_embed
            else:
                # Standard case - apply to full vectors
                q_embed = (q * cos) + (self._rotate_half(q) * sin)
                k_embed = (k * cos) + (self._rotate_half(k) * sin)
                return q_embed, k_embed

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,
        ):
            mixed_query_layer = self.query(hidden_states)
            batch_size, seq_length = hidden_states.shape[:2]

            # If past key value is provided, only process the new tokens
            if past_key_value is not None:
                # Reuse k, v from past
                key_layer = self.key(hidden_states) if past_key_value[0] is None else torch.cat([past_key_value[0], self.key(hidden_states)], dim=1)
                value_layer = self.value(hidden_states) if past_key_value[1] is None else torch.cat([past_key_value[1], self.value(hidden_states)], dim=1)
                mixed_query_layer = mixed_query_layer[:, -hidden_states.size(1):, :]
            else:
                key_layer = self.key(hidden_states)
                value_layer = self.value(hidden_states)

            # Transpose for attention [batch, num_heads, seq_len, head_dim]
            query_layer = self._transpose_for_scores(mixed_query_layer)
            key_layer = self._transpose_for_scores(key_layer)
            value_layer = self._transpose_for_scores(value_layer)

            # Apply RoPE if configured
            if self.positional_encoding_type == "rope":
                # Generate position-dependent rotation
                seq_len = key_layer.shape[2]
                position = torch.arange(seq_len, device=hidden_states.device).unsqueeze(1)
                # [seq_len, dim/2]
                cos = torch.cos(position * self.inv_freq).unsqueeze(0).unsqueeze(0)
                sin = torch.sin(position * self.inv_freq).unsqueeze(0).unsqueeze(0)
                # Extend to match dimensions [1, 1, seq_len, dim/2]
                cos = cos.expand(batch_size, self.num_attention_heads, -1, -1)
                sin = sin.expand(batch_size, self.num_attention_heads, -1, -1)
                # Apply rotary embeddings
                query_layer, key_layer = self._apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

            kv_seq_len = key_layer.shape[-2]
            if past_key_value is not None:
                kv_seq_len = kv_seq_len + past_key_value[0].shape[-2]

            # Get the attention scores
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Apply causal mask (if needed and no explicit attention mask given)
            if attention_mask is None and seq_length > 1:
                attention_mask = self.causal_mask[:seq_length, :kv_seq_len]
                # Add mask to attention scores
                attention_scores = attention_scores + attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask is not None:
                # Process the user-provided attention mask
                if attention_mask.dim() == 2:
                    # [batch, seq_len] -> [batch, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    attention_mask = (1.0 - attention_mask) * -10000.0
                    attention_scores = attention_scores + attention_mask
                elif attention_mask.dim() == 3:
                    # [batch, 1, seq_len] -> [batch, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1)
                    attention_mask = (1.0 - attention_mask) * -10000.0
                    attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            context_layer = self.output(context_layer)
            context_layer = self.output_dropout(context_layer)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            # Cache past key and value for faster decoding
            if use_cache:
                outputs = outputs + ((key_layer, value_layer),)

            return outputs


    class HindiCausalLMLayer(nn.Module):
        """Transformer layer for Hindi Causal LM with attention and feed-forward networks."""

        def __init__(self, config):
            super().__init__()
            self.chunk_size_feed_forward = config.chunk_size_feed_forward if hasattr(config, "chunk_size_feed_forward") else 0
            self.seq_len_dim = 1

            # Use RMSNorm or LayerNorm based on config
            norm_class = RMSNorm if getattr(config, "normalization_layer", "layernorm") == "rmsnorm" else nn.LayerNorm
            self.attention_norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
            self.attention = HindiCausalLMAttention(config)

            self.ffn_norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)
            self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
            self.output = nn.Linear(config.intermediate_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,
        ):
            # Pre-norm architecture
            norm_hidden_states = self.attention_norm(hidden_states)

            # Self-attention
            attention_outputs = self.attention(
                norm_hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            attention_output = attention_outputs[0]

            # Residual connection
            hidden_states = hidden_states + attention_output

            # Feed-forward network with pre-norm
            ffn_norm_hidden = self.ffn_norm(hidden_states)

            # Feed-forward computation
            intermediate_output = self.intermediate(ffn_norm_hidden)
            intermediate_output = self.intermediate_act_fn(intermediate_output)
            ffn_output = self.output(intermediate_output)
            ffn_output = self.dropout(ffn_output)

            # Residual connection
            layer_output = hidden_states + ffn_output

            outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

            return outputs


    class HindiCausalLMEncoder(nn.Module):
        """
        Transformer encoder consisting of `config.num_hidden_layers` self attention layers.
        Each layer is a [`HindiCausalLMLayer`].
        """

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.layers = nn.ModuleList([HindiCausalLMLayer(config) for _ in range(config.num_hidden_layers)])

            # Use RMSNorm or LayerNorm based on config
            norm_class = RMSNorm if getattr(config, "normalization_layer", "layernorm") == "rmsnorm" else nn.LayerNorm
            self.final_layer_norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ):
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None

            next_decoder_cache = () if use_cache else None

            for i, layer_module in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # Apply final norm after last layer
            hidden_states = self.final_layer_norm(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions] if v is not None)

            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=None,
            )


    class HindiCausalLMPreTrainedModel(PreTrainedModel):
        """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
        """

        config_class = HindiCausalLMConfig
        base_model_prefix = "hindi_causal_lm"
        supports_gradient_checkpointing = True

        def _init_weights(self, module):
            """Initialize the weights"""
            if isinstance(module, nn.Linear):
                # Slightly different from the original implementation
                # which used truncated_normal for initialization
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                module.bias.data.zero_() if hasattr(module, "bias") else None
                module.weight.data.fill_(1.0)

        def _set_gradient_checkpointing(self, module, value=False):
            if isinstance(module, HindiCausalLMEncoder):
                module.gradient_checkpointing = value


    class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
        """
        The Hindi Causal LM base model.
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config

            # Initialize token and position embeddings
            self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

            # Use position embeddings based on config
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
            if self.position_embedding_type == "absolute" or getattr(config, "positional_encoding_type", "absolute") == "learned":
                self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

            self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

            # Initialize encoder
            self.encoder = HindiCausalLMEncoder(config)

            # Initialize weights and apply final processing
            self.post_init()

        def get_input_embeddings(self):
            return self.token_embeddings

        def set_input_embeddings(self, value):
            self.token_embeddings = value

        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            """
            for layer, heads in heads_to_prune.items():
                self.encoder.layers[layer].attention.prune_heads(heads)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                batch_size, seq_length = input_shape
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size, seq_length = input_shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # past_key_values_length
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)

            # Prepare head mask if needed
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            if inputs_embeds is None:
                inputs_embeds = self.token_embeddings(input_ids)

            # Add position embeddings if using absolute or learned position embeddings
            if (self.position_embedding_type == "absolute" or
                getattr(self.config, "positional_encoding_type", "absolute") == "learned"):
                if position_ids is None:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

                position_embeds = self.position_embeddings(position_ids)
                inputs_embeds = inputs_embeds + position_embeds

            embeddings = self.embedding_dropout(inputs_embeds)

            # RoPE is handled inside the attention layer, not here

            encoder_outputs = self.encoder(
                embeddings,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = encoder_outputs[0]

            if not return_dict:
                return (sequence_output,) + encoder_outputs[1:]

            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=sequence_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
            )


    class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel, GenerationMixin):
        _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.weight"]
        _tied_weights_keys = ["lm_head.weight"]

        def __init__(self, config):
            super().__init__(config)
            self.config = config

            # Initialize the base model
            self.hindi_causal_lm = HindiCausalLMModel(config)

            # LM head
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            # Initialize weights and apply final processing
            self.post_init()

            # Tie weights if configured
            if config.tie_word_embeddings:
                self._tie_weights()

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            self.lm_head = new_embeddings

        def _tie_weights(self):
            """
            Tie the weights between the input embeddings and the output embeddings.
            """
            if self.config.tie_word_embeddings:
                self._tied_weights_keys = ["lm_head.weight"]
                self.lm_head.weight = self.hindi_causal_lm.token_embeddings.weight

        def _untie_weights(self):
            """
            Untie the weights between the input embeddings and the output embeddings.
            
            This method should be called before serialization with safetensors when using weight tying.
            """
            if hasattr(self, "_tied_weights_keys"):
                for key in self._tied_weights_keys:
                    if key == "lm_head.weight":
                        self.lm_head.weight = nn.Parameter(self.lm_head.weight.clone())

        def save_pretrained(
            self,
            save_directory,
            is_main_process=True,
            state_dict=None,
            save_function=None,
            push_to_hub=False,
            max_shard_size="10GB",
            safe_serialization=True,
            **kwargs,
        ):
            """
            Save model with special handling for weight tying
            """
            # Temporarily untie weights if using safe serialization
            if safe_serialization and hasattr(self, "_untie_weights"):
                self._untie_weights()

            # Save normally using parent method
            result = super().save_pretrained(
                save_directory=save_directory,
                is_main_process=is_main_process,
                state_dict=state_dict,
                save_function=save_function,
                push_to_hub=push_to_hub,
                max_shard_size=max_shard_size,
                safe_serialization=safe_serialization,
                **kwargs,
            )

            # Re-tie weights if they were untied
            if safe_serialization and hasattr(self, "_tie_weights"):
                self._tie_weights()

            return result

        def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
            """Prepare model inputs for generation"""
            # Only keep inputs needed for forward pass
            inputs = {
                "input_ids": input_ids,
            }

            # Add attention mask if provided
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask

            # Adjust attention mask for past key values
            if past_key_values is not None:
                inputs["past_key_values"] = past_key_values

                # Only use last input token when using past key values
                inputs["input_ids"] = input_ids[:, -1].unsqueeze(-1)

                # Extend attention mask if present
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    one_hot_positions = torch.ones((attention_mask.shape[0], 1),
                                                   dtype=attention_mask.dtype,
                                                   device=attention_mask.device)
                    inputs["attention_mask"] = torch.cat([attention_mask, one_hot_positions], dim=1)

            return inputs

        def get_generation_config(self):
            """Return the default generation configuration"""
            if hasattr(self, "generation_config") and self.generation_config is not None:
                return self.generation_config
            return GenerationConfig(
                pad_token_id=self.config.pad_token_id,
                bos_token_id=self.config.bos_token_id,
                eos_token_id=self.config.eos_token_id
            )

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            outputs = self.hindi_causal_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]
            lm_logits = self.lm_head(sequence_output)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
