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
    from ...generation.configuration_utils import GenerationConfig
    from ...generation.utils import GenerationMixin
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
                torch.triu(
                    torch.ones(config.max_position_embeddings, config.max_position_embeddings) * -1e10, diagonal=1
                ),
                persistent=False # Avoid saving the mask in state_dict
            )

            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
            self.max_position_embeddings = config.max_position_embeddings
            self.positional_encoding_type = getattr(config, "positional_encoding_type", "absolute")

            # Initialize for RoPE if used
            if self.positional_encoding_type == "rope":
                self.rotary_dim = min(
                    self.attention_head_size, getattr(config, "rotary_dim", self.attention_head_size)
                )
                # Ensure rotary_dim is even
                if self.rotary_dim % 2 != 0:
                    self.rotary_dim = max(2, self.rotary_dim - 1) # Adjust if odd

                # Cache frequency - use float() for compatibility
                inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
                self.register_buffer("inv_freq", inv_freq, persistent=False) # Avoid saving in state_dict

                # Precompute rotary embeddings if needed (or compute on the fly)
                self._set_cos_sin_cache(seq_len=config.max_position_embeddings, device="cpu", dtype=torch.float32) # Initialize cache

        def _set_cos_sin_cache(self, seq_len, device, dtype):
            """Helper to compute rotary embeddings."""
            if self.positional_encoding_type != "rope":
                return

            # Ensure inv_freq is on the correct device
            self.inv_freq = self.inv_freq.to(device)

            # `torch.arange` creates integers, need float for `outer`
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)

            # Different from Llama, shape is (1, seq_len, 1, rotary_dim).
            # Cat is on the last dimension.
            emb = torch.cat((freqs, freqs), dim=-1)

            # Reshape for broadcasting: [1, seq_len, 1, rotary_dim]
            cos = emb.cos().unsqueeze(1)
            sin = emb.sin().unsqueeze(1)

            # Update buffer directly, ensuring correct dtype
            self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
            self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

        def _transpose_for_scores(self, x):
            """Reshape from [batch_size, seq_length, hidden_size] to [batch_size, num_heads, seq_length, head_size]"""
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        def _apply_rotary_pos_emb(self, q, k, cos, sin):
            """Applies Rotary Position Embedding to the query and key tensors."""
            # q, k: [bs, num_heads, seq_len, head_dim]
            # cos, sin: [1, seq_len, 1, rotary_dim]

            # Extract the dimensions to be rotated
            q_rot = q[..., : self.rotary_dim]
            k_rot = k[..., : self.rotary_dim]

            # Split the rotation dimension into two halves
            q_rot1, q_rot2 = torch.chunk(q_rot, 2, dim=-1)
            k_rot1, k_rot2 = torch.chunk(k_rot, 2, dim=-1)

            # Also split cos and sin
            cos1, cos2 = torch.chunk(cos, 2, dim=-1) # cos1 applies to rot1, cos2 to rot2
            sin1, sin2 = torch.chunk(sin, 2, dim=-1) # sin1 applies to rot1, sin2 to rot2

            # Apply the rotation:
            # q_rotated = [q1*cos - q2*sin, q1*sin + q2*cos]
            # The formula uses the same cos/sin for both parts but applied differently
            # Correct application: x_rot = x * cos + rotate_half(x) * sin
            # Where rotate_half([x1, x2]) = [-x2, x1]
            # So, q_rot_result = q_rot * cos + torch.cat([-q_rot2, q_rot1], dim=-1) * sin

            # Using the standard implementation approach:
            q_rot_result = torch.cat(
                [q_rot1 * cos1 - q_rot2 * sin1, q_rot2 * cos1 + q_rot1 * sin1], dim=-1
            )
            k_rot_result = torch.cat(
                [k_rot1 * cos1 - k_rot2 * sin1, k_rot2 * cos1 + k_rot1 * sin1], dim=-1
            )


            # If only rotating a subset of dimensions, concatenate back the unrotated part
            if self.rotary_dim < self.attention_head_size:
                q_out = torch.cat([q_rot_result, q[..., self.rotary_dim:]], dim=-1)
                k_out = torch.cat([k_rot_result, k[..., self.rotary_dim:]], dim=-1)
            else:
                q_out = q_rot_result
                k_out = k_rot_result

            return q_out, k_out

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None, # Add position_ids for RoPE
            head_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,
        ):
            batch_size, seq_length, _ = hidden_states.shape

            # Query, Key, Value projections
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            # Transpose for attention [batch, num_heads, seq_len, head_dim]
            query_layer = self._transpose_for_scores(mixed_query_layer)
            key_layer = self._transpose_for_scores(mixed_key_layer)
            value_layer = self._transpose_for_scores(mixed_value_layer)

            # Apply RoPE if configured
            kv_seq_len = key_layer.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]

            if self.positional_encoding_type == "rope":
                # Ensure cos/sin cache is ready and on the right device/dtype
                cos = self.cos_cached[:, :kv_seq_len, ...].to(dtype=query_layer.dtype)
                sin = self.sin_cached[:, :kv_seq_len, ...].to(dtype=query_layer.dtype)

                # Apply RoPE based on position_ids
                query_layer, key_layer = self._apply_rotary_pos_emb(query_layer, key_layer, cos, sin)


            # Handle past_key_value for efficient generation
            if past_key_value is not None:
                # Reuse k, v, self_attention
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

            # Update past_key_value tuple if needed
            past_key_value = (key_layer, value_layer) if use_cache else None

            # Get the attention scores
            # q: [bs, n_heads, seq_len, head_dim]
            # k: [bs, n_heads, kv_seq_len, head_dim]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Apply causal mask (ensure correct slice for kv_seq_len)
            current_causal_mask = self.causal_mask[None, None, :seq_length, :kv_seq_len].to(attention_scores.device)
            attention_scores = attention_scores + current_causal_mask

            # Apply attention mask if provided
            if attention_mask is not None:
                 # Format should be [bs, 1, query_len, key_len]
                 # We might need to expand the mask from [bs, seq_len] or [bs, key_len]
                 if attention_mask.dim() == 2: # [bs, key_len]
                     attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, kv_seq_len)
                 elif attention_mask.dim() == 3: # [bs, query_len, key_len] -> should be rare for causal
                     attention_mask = attention_mask[:, None, :, :]
                 elif attention_mask.dim() != 4:
                     raise ValueError(f"Unexpected attention mask shape: {attention_mask.shape}")

                 # Apply the mask (usually 0 for allowed, 1 for masked)
                 # PyTorch needs -inf or large negative number for masked positions
                 # Invert mask: 0 -> 1, 1 -> 0; then multiply by large negative
                 inverted_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
                 attention_scores = attention_scores + inverted_mask


            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # Dropout on attention probabilities
            attention_probs = self.attention_dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # Weighted sum of values
            context_layer = torch.matmul(attention_probs, value_layer)

            # Reshape back to [batch_size, seq_length, hidden_size]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            # Final output projection and dropout
            context_layer = self.output(context_layer)
            context_layer = self.output_dropout(context_layer)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            # Include past_key_value if caching is enabled
            if use_cache:
                outputs = outputs + (past_key_value,)

            return outputs

    class HindiCausalLMLayer(nn.Module):
        """Transformer layer for Hindi Causal LM with attention and feed-forward networks."""

        def __init__(self, config):
            super().__init__()
            self.chunk_size_feed_forward = (
                config.chunk_size_feed_forward if hasattr(config, "chunk_size_feed_forward") else 0
            )
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
            position_ids=None, # Pass position_ids for RoPE
            head_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,
        ):
            # Pre-norm architecture for Attention
            norm_hidden_states = self.attention_norm(hidden_states)

            # Self-attention
            attention_outputs = self.attention(
                norm_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids, # Pass to attention
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            attention_output = attention_outputs[0]
            present_key_value = attention_outputs[-1] if use_cache else None # Get kv cache if returned

            # Residual connection for Attention
            hidden_states = hidden_states + attention_output

            # Pre-norm architecture for Feed-Forward
            ffn_norm_hidden = self.ffn_norm(hidden_states)

            # Feed-forward computation
            intermediate_output = self.intermediate(ffn_norm_hidden)
            intermediate_output = self.intermediate_act_fn(intermediate_output)
            ffn_output = self.output(intermediate_output)
            ffn_output = self.dropout(ffn_output)

            # Residual connection for Feed-Forward
            layer_output = hidden_states + ffn_output

            outputs = (layer_output,)
            if output_attentions:
                outputs += (attention_outputs[1],) # Add attention probabilities

            if use_cache:
                outputs += (present_key_value,) # Add kv cache

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
            self.gradient_checkpointing = False # Initialize gradient checkpointing flag

            # Use RMSNorm or LayerNorm based on config for the final norm
            norm_class = RMSNorm if getattr(config, "normalization_layer", "layernorm") == "rmsnorm" else nn.LayerNorm
            self.final_layer_norm = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None, # Pass position_ids for RoPE
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

                # Handle gradient checkpointing
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, past_key_value=None, output_attentions=output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        layer_head_mask,
                        use_cache=use_cache, # Pass use_cache here
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids, # Pass position_ids
                        head_mask=layer_head_mask,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],) # KV cache is always the last element when use_cache=True

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # Apply final norm after last layer
            hidden_states = self.final_layer_norm(hidden_states)

            # Add last hidden state
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions]
                    if v is not None
                )

            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=None, # This model does not use cross-attention
            )

    class HindiCausalLMPreTrainedModel(PreTrainedModel):
        """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
        """

        config_class = HindiCausalLMConfig
        base_model_prefix = "hindi_causal_lm"
        supports_gradient_checkpointing = True
        _no_split_modules = ["HindiCausalLMLayer"] # Modules that shouldn't be split for model parallelism

        # Keys to ignore when loading weights with missing keys (useful for fine-tuning)
        _keys_to_ignore_on_load_missing = [r"position_ids"] # position_ids usually not saved

        # Keys to ignore when loading weights with unexpected keys (useful when removing heads)
        _keys_to_ignore_on_load_unexpected = [r"decoder\.final_layer_norm\.weight"]

        def _init_weights(self, module):
            """Initialize the weights"""
            std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
            if isinstance(module, nn.Linear):
                # Slightly different from the original paper which uses truncated_normal
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
                # RMSNorm only has weight
                module.weight.data.fill_(1.0)

        def _set_gradient_checkpointing(self, module, value=False):
            if isinstance(module, HindiCausalLMEncoder):
                module.gradient_checkpointing = value

    class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
        """
        The Hindi Causal LM base model transformer.

        Args:
            config ([`HindiCausalLMConfig`]): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the
                configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        """

        def __init__(self, config: HindiCausalLMConfig):
            super().__init__(config)
            self.config = config
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size

            # Initialize token embeddings
            self.token_embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
            )

            # Use position embeddings based on config
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
            if (
                self.position_embedding_type == "absolute"
                or getattr(config, "positional_encoding_type", "absolute") == "learned" # Treat 'learned' as 'absolute' here
            ):
                self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            else:
                self.position_embeddings = None # RoPE is handled in attention layer

            self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

            # Initialize encoder (Transformer layers)
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
            Used by the pruning logic in modeling_utils.
            """
            for layer, heads in heads_to_prune.items():
                self.encoder.layers[layer].attention.prune_heads(heads)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
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
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # Retrieve input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            # Create position_ids if not provided
            if position_ids is None:
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                # Ensure position_ids are the correct shape and type
                position_ids = position_ids.view(-1, seq_length).long()


            # Get token embeddings
            if inputs_embeds is None:
                inputs_embeds = self.token_embeddings(input_ids)

            # Add absolute/learned position embeddings if configured
            if self.position_embeddings is not None:
                position_embeds = self.position_embeddings(position_ids)
                inputs_embeds = inputs_embeds + position_embeds

            hidden_states = self.embedding_dropout(inputs_embeds)

            # Create default attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length + past_key_values_length), dtype=torch.bool, device=hidden_states.device
                 )

            # Expand attention mask [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # This is needed for the attention layer's masking mechanism
            # combined_attention_mask = None # Handled inside attention layer now

            # Prepare head mask if needed
            # [layer_num] -> [bsz x n_heads x seq_len x seq_len]
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            # Pass inputs through the encoder (Transformer layers)
            encoder_outputs = self.encoder(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids, # Pass position_ids for RoPE
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
        _keys_to_ignore_on_load_missing = [r"lm_head.weight"] # Often tied, might be missing if only base is saved
        _tied_weights_keys = ["lm_head.weight"]

        def __init__(self, config: HindiCausalLMConfig):
            super().__init__(config)
            self.config = config

            # Initialize the base model
            self.hindi_causal_lm = HindiCausalLMModel(config)

            # LM head (output layer)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            # Initialize weights and apply final processing
            self.post_init()

            # Tie weights if configured (done after post_init)
            if config.tie_word_embeddings:
                self.tie_weights() # Use the built-in method

        def get_input_embeddings(self):
            return self.hindi_causal_lm.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.hindi_causal_lm.set_input_embeddings(value)

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            self.lm_head = new_embeddings

        # Override tie/untie methods for clarity and correctness
        def tie_weights(self):
            """
            Tie the weights between the input embeddings and the output embeddings.
            """
            if self.config.tie_word_embeddings:
                output_embeddings = self.get_output_embeddings()
                input_embeddings = self.get_input_embeddings()
                if output_embeddings is not None and input_embeddings is not None:
                    output_embeddings.weight = input_embeddings.weight
            super().tie_weights() # Call parent method if needed

        def _untie_weights(self):
            """
            Untie the weights between the input embeddings and the output embeddings.
            Needed for saving with safetensors when weights are tied.
            """
            if self.config.tie_word_embeddings:
                output_embeddings = self.get_output_embeddings()
                input_embeddings = self.get_input_embeddings()
                if output_embeddings is not None and input_embeddings is not None:
                    # Check if they are currently the same object
                    if output_embeddings.weight is input_embeddings.weight:
                        # Clone the weight to create a separate parameter
                        output_embeddings.weight = nn.Parameter(output_embeddings.weight.clone())


        # Override save_pretrained to handle untying/retying for safetensors
        def save_pretrained(
            self,
            save_directory,
            is_main_process=True,
            state_dict=None,
            save_function=None, # Let parent handle save_function
            push_to_hub=False,
            max_shard_size="5GB", # Adjust default if needed
            safe_serialization=True, # Default to True
            variant=None,
            save_peft_format=False,
            **kwargs,
        ):
            """
            Save model with special handling for weight tying with safetensors.
            """
            weights_were_tied = self.config.tie_word_embeddings and self.get_output_embeddings() is not None
            untied_for_save = False

            # Untie weights before saving if using safe serialization and weights are tied
            if safe_serialization and weights_were_tied:
                try:
                    self._untie_weights()
                    untied_for_save = True
                    logger.info("Untied weights for safe serialization.")
                except Exception as e:
                    logger.warning(f"Failed to untie weights for saving: {e}. Proceeding without untying.")

            # Use parent's save_pretrained method
            result = super().save_pretrained(
                save_directory=save_directory,
                is_main_process=is_main_process,
                state_dict=state_dict,
                save_function=save_function, # Pass it along
                push_to_hub=push_to_hub,
                max_shard_size=max_shard_size,
                safe_serialization=safe_serialization,
                variant=variant,
                save_peft_format=save_peft_format,
                **kwargs,
            )

            # Re-tie weights if they were untied for saving
            if untied_for_save:
                try:
                    self.tie_weights()
                    logger.info("Re-tied weights after saving.")
                except Exception as e:
                    logger.warning(f"Failed to re-tie weights after saving: {e}.")

            return result

        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
            """Prepare model inputs for generation with support for KV caching."""
            # If past_key_values are provided, only the last token needs to be processed
            if past_key_values:
                input_ids = input_ids[:, -1:]
                # The `past_key_values` already contains the attention information for previous tokens
                # We only need the attention mask for the *new* token.
                # However, the position_ids need to be incremented correctly.
                past_length = past_key_values[0][0].shape[2]
                attention_mask = attention_mask[:, -1:] # Mask for the new token
            else:
                 past_length = 0

            # Calculate position_ids dynamically based on past length
            position_ids = torch.arange(
                past_length, input_ids.shape[1] + past_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            # Prepare the final inputs dictionary
            model_inputs = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

            # Add other potential kwargs needed by the model's forward method
            # (e.g., token_type_ids if the model uses them, although uncommon for causal LM)
            # model_inputs.update(kwargs) # Be careful not to overwrite essential keys

            return model_inputs

        # Override get_generation_config if specific defaults are needed
        def get_generation_config(self):
            """Return the default generation configuration"""
            # Start with the base config if available
            if hasattr(self, "generation_config") and self.generation_config is not None:
                config = self.generation_config
            else:
                config = GenerationConfig()

            # Set model-specific defaults based on HindiCausalLMConfig
            config.pad_token_id = self.config.pad_token_id
            config.bos_token_id = self.config.bos_token_id
            config.eos_token_id = self.config.eos_token_id
            config.max_length = getattr(self.config, "max_length", 20) # Default max_length if not set
            config.do_sample = getattr(self.config, "do_sample", True) # Default do_sample if not set

            return config

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
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
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to
                `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            # Pass inputs through the base model
            outputs = self.hindi_causal_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # Get the last hidden state (sequence output)
            sequence_output = outputs[0]

            # Compute LM logits using the LM head
            lm_logits = self.lm_head(sequence_output)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens and compute the cross-entropy loss
                loss_fct = CrossEntropyLoss()
                # Move labels to the same device as logits before computing loss
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions, # Keep even if None
            )

        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            """
            Reorders the `past_key_values` cache according to the `beam_idx` for beam search generation.
            """
            reordered_past = ()
            for layer_past in past_key_values:
                # Each layer_past is likely a tuple (key, value)
                # Ensure both key and value tensors are indexed
                reordered_layer_past = tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past
                 )
                reordered_past += (reordered_layer_past,)
            return reordered_past