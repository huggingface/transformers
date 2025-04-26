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
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss  # Import functional
from torch.nn import functional as F

# Ensure ACT2FN is imported correctly relative to your project structure
# Assuming it's in the parent directory's 'activations.py' file:
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging

# Import the UPDATED configuration class
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-embedding-foundational-model"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"


# --- HindiCausalSelfAttention remains the same ---
class HindiCausalSelfAttention(nn.Module):
    """Causal self-attention layer for Hindi Causal LM."""

    def __init__(self, config: HindiCausalLMConfig):  # Add type hint
        super().__init__()
        self.hidden_size = config.hidden_size  # Store hidden_size
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query, Key, Value projections
        self.q_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=False
        )  # Use common naming q_proj, k_proj, v_proj and often bias=False
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.all_head_size, config.hidden_size, bias=False)  # Use common naming o_proj
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)  # Use nn.Dropout module
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)  # Dropout after output projection (resid)

        # Causal mask buffer removed - typically generated dynamically or handled differently (e.g., in forward)
        # Especially important if using flash attention or kv caching

        # Placeholder for RoPE Cache if using RoPE - not implemented here
        self.rotary_emb = None  # Add placeholder if RoPE is used
        if config.positional_encoding_type == "rope":
            # You would initialize your RoPE embedding cache here
            # from ...modeling_attn_mask_utils import AttentionMaskConverter # Needed for dynamic mask
            # self.attention_mask_converter = AttentionMaskConverter(is_causal=True)
            # Example: self.rotary_emb = RotaryEmbedding(...)
            logger.warning("RoPE positional encoding selected but not fully implemented in this example.")
            pass

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2).contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,  # Add position_ids for RoPE
        # past_key_value: Optional[Tuple[torch.Tensor]] = None, # Add for KV caching
        output_attentions: bool = False,
        use_cache: bool = False,  # Add for KV caching
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        # --- RoPE ---
        # if self.rotary_emb is not None and position_ids is not None:
        #     cos, sin = self.rotary_emb(value_states, seq_len=q_len) # Assuming RoPE needs value states shape
        #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # --- KV Caching ---
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # present_key_value = (key_states, value_states) if use_cache else None
        # else:
        #     present_key_value = (key_states, value_states) if use_cache else None

        # --- Attention Calculation ---
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.attention_head_size)

        # Apply causal mask + attention mask
        if attention_mask is not None:
            if q_len != attention_mask.size(-1):
                raise ValueError(
                    f"Attention mask sequence length ({attention_mask.size(-1)}) "
                    f"doesn't match query sequence length ({q_len})"
                )
            # The attention mask logic here needs to correctly combine causal and padding masks.
            # Often handled by AttentionMaskConverter or similar utility.
            # Simplified version assuming attention_mask is pre-computed correctly for causal LM:
            attn_weights = (
                attn_weights + attention_mask
            )  # Assume mask adds large negative values where attention is prevented

        # Upcast attention to fp32 (important for stability)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)
        # attn_output = self.resid_dropout(attn_output) # Apply residual dropout *after* adding residual in the block

        if not output_attentions:
            attn_weights = None

        # return attn_output, attn_weights, present_key_value # With KV Cache
        return attn_output, attn_weights, None  # Without KV Cache


# --- UPDATED HindiMLP (formerly part of HindiTransformerBlock) ---
class HindiMLP(nn.Module):
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Determine if we are using a GLU variant based on hidden_act
        # Common GLU activations: silu (SwiGLU), gelu variants (GeGLU)
        self.is_glu_activation = config.hidden_act in [
            "silu",
            "swish",
            "gelu_pytorch_tanh",
            "gelu_fast",
        ]  # Add others if needed

        if self.is_glu_activation:
            # GLU Structure (SwiGLU/GeGLU)
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            # The down projection merges the gated output
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            self.act_fn = ACT2FN[config.hidden_act]  # Get the specific activation (SiLU, GeLU etc.)
        else:
            # Standard FFN Structure
            self.wi = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # Often called wi or fc1
            self.wo = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # Often called wo or fc2
            self.act_fn = ACT2FN[config.hidden_act]

        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # Define dropout once

    def forward(self, x):
        if self.is_glu_activation:
            # SwiGLU/GeGLU path: gate(x) * up(x)
            # Activatio is applied to the gate_proj output
            hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        else:
            # Standard FFN path: wo(act(wi(x)))
            hidden_states = self.act_fn(self.wi(x))

        hidden_states = self.down_proj(hidden_states) if self.is_glu_activation else self.wo(hidden_states)
        hidden_states = self.dropout(hidden_states)  # Apply dropout *before* residual connection in the block
        return hidden_states


# --- UPDATED HindiTransformerBlock ---
class HindiTransformerBlock(nn.Module):
    """Transformer block using HindiMLP and HindiCausalSelfAttention."""

    def __init__(self, config: HindiCausalLMConfig, layer_idx: int):  # Add layer_idx if needed for some features
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HindiCausalSelfAttention(config)
        self.mlp = HindiMLP(config)

        # Layer Normalization - Choose based on config
        if config.normalization_layer == "rmsnorm":
            # Import RMSNorm if not standard in torch.nn yet
            # from .rms_norm import RMSNorm # Assuming you have rms_norm.py
            # self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            # self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            # Using LayerNorm as placeholder if RMSNorm not available
            logger.warning("RMSNorm selected but using nn.LayerNorm as placeholder.")
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif config.normalization_layer == "layernorm":
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            raise ValueError(f"Unsupported normalization layer: {config.normalization_layer}")

        # Dropout for residual connections is often applied *inside* the sub-modules (attn, mlp)
        # Or applied just before adding the residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Residual connection start
        residual = hidden_states

        # Apply input LayerNorm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_output = attn_outputs  # Rename for clarity

        # Residual connection: Add before the MLP's LayerNorm
        # Dropout is usually applied *before* the residual add in attn/mlp outputs
        hidden_states = residual + attn_output

        # MLP block
        residual = hidden_states  # Start new residual connection
        hidden_states = self.post_attention_layernorm(hidden_states)  # Norm before MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # Add residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        # if use_cache:
        #     outputs += (present_key_value,)

        # return outputs # With KV Cache
        return outputs  # Without KV Cache


# --- HindiCausalLMModel remains largely the same, just passes config down ---
# --- Make sure positional encoding logic uses config.positional_encoding_type ---
class HindiCausalLMModel(PreTrainedModel):
    """Hindi Causal Language Model backbone."""

    config_class = HindiCausalLMConfig
    base_model_prefix = "transformer"  # Changed from "model" to common "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiTransformerBlock"]  # Helps with FSDP/DeepSpeed if needed

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(  # Use common name embed_tokens
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )

        # Positional encoding - RoPE handled in attention, others here
        self.position_embeddings = None
        if config.positional_encoding_type == "absolute" or config.positional_encoding_type == "learned":
            # Use 'absolute' for sinusoidal, 'learned' for nn.Embedding
            if config.positional_encoding_type == "learned":
                self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            else:
                # Implement sinusoidal absolute embeddings if needed
                logger.warning("Absolute sinusoidal positional encoding not fully implemented.")
                pass  # Placeholder
        elif config.positional_encoding_type != "rope":
            logger.warning(
                f"Unsupported positional encoding type: {config.positional_encoding_type}. No positional embeddings applied."
            )

        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)  # Renamed from embedding_dropout

        # Transformer layers
        self.layers = nn.ModuleList(
            [HindiTransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Final layer norm
        if config.normalization_layer == "rmsnorm":
            # from .rms_norm import RMSNorm
            # self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            logger.warning("RMSNorm selected but using nn.LayerNorm as placeholder.")
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Final norm often called 'norm'
        elif config.normalization_layer == "layernorm":
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            # Already validated in block, but belt-and-suspenders
            raise ValueError(f"Unsupported normalization layer: {config.normalization_layer}")

        self.gradient_checkpointing = False  # Add attribute for checkpointing control

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Removed RoPE application here - should be inside attention mechanism

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None, # Add for KV caching
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,  # Add for KV caching
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:  # Update return type hint
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )  # Get use_cache from config if needed
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Process inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # --- KV Caching Input Processing ---
        # past_key_values_length = 0
        # if past_key_values is not None:
        #     past_key_values_length = past_key_values[0][0].shape[2] # k_proj shape [bsz, num_heads, seq_len, head_size]

        # Create position ids if needed
        if position_ids is None:
            # position_ids = torch.arange(
            #     past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            # )
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)  # Without KV Cache
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)  # Shape [1, seq_len]
        else:
            # Ensure position_ids are correctly shaped if provided
            position_ids = position_ids.view(-1, seq_length).long()

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Add position embeddings if applicable (learned/absolute)
        if self.position_embeddings is not None:
            if self.config.positional_encoding_type != "rope":  # RoPE applied in attention
                position_embeds = self.position_embeddings(position_ids)
                hidden_states = inputs_embeds + position_embeds
            else:
                hidden_states = inputs_embeds  # RoPE applied later
        else:
            hidden_states = inputs_embeds  # No explicit pos embeddings other than potential RoPE

        hidden_states = self.embed_dropout(hidden_states)

        # --- Prepare attention mask ---
        # Attention mask logic needs careful handling for causal LM + padding + KV cache
        # Using simplified version here. A dedicated utility is better.
        if attention_mask is not None:
            # Assume attention_mask is [bsz, seq_len] with 1 for valid tokens, 0 for padding
            # Create combined causal and padding mask for attention layer
            # Shape expected by attention layer: [bsz, 1, q_len, k_len] or broadcastable
            combined_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states.dtype,
                0,  # past_key_values_length
            )
        else:
            combined_mask = None  # Let attention layer handle causal mask internally if needed

        # --- Transformer Layers ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # next_decoder_cache = () if use_cache else None # For KV caching

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # --- Handle Gradient Checkpointing ---
            if self.gradient_checkpointing and self.training:
                # Define layer_outputs based on use_cache etc.
                # layer_outputs = self._gradient_checkpointing_func(
                #     decoder_layer.__call__,
                #     hidden_states,
                #     combined_mask, # Pass combined_mask
                #     position_ids,
                #     # past_key_values[idx] if past_key_values is not None else None,
                #     output_attentions,
                #     use_cache,
                # )
                # Placeholder for non-checkpointed execution
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_mask,  # Pass combined_mask
                    position_ids=position_ids,
                    # past_key_value=past_key_values[idx] if past_key_values is not None else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_mask,  # Pass combined_mask
                    position_ids=position_ids,
                    # past_key_value=past_key_values[idx] if past_key_values is not None else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            # if use_cache:
            #     next_decoder_cache += (layer_outputs[2 if output_attentions else 1],) # Append present_key_value

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Apply final layer norm
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # next_cache = next_decoder_cache if use_cache else None # For KV caching

        if not return_dict:
            # return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None) # With KV
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)  # Without KV

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            # past_key_values=next_cache, # Add if using KV cache
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Add _prepare_decoder_attention_mask helper (simplified example)
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length):
        # Create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        bsz, seq_len = input_shape
        # Causal mask based on target sequence length
        causal_mask = torch.full((seq_len, seq_len), fill_value=torch.finfo(dtype).min, device=attention_mask.device)
        mask_cond = torch.arange(causal_mask.size(-1), device=attention_mask.device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
        causal_mask = causal_mask.to(dtype)

        if past_key_values_length > 0:
            # If using KV cache, extend causal mask
            causal_mask = torch.cat(
                [
                    torch.zeros((seq_len, past_key_values_length), dtype=dtype, device=attention_mask.device),
                    causal_mask,
                ],
                dim=-1,
            )

        # Combine with padding mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand padding mask: [bsz, src_len] -> [bsz, 1, tgt_len, src_len]
                expanded_mask = (
                    attention_mask[:, None, None, :]
                    .expand(bsz, 1, seq_len, seq_len + past_key_values_length)
                    .to(dtype)
                )
            else:
                # Assume mask is already correctly shaped
                expanded_mask = attention_mask.to(dtype)

            # Values in expanded_mask are 0 for padding, 1 for non-padding.
            # Invert mask: 0 for non-padding, large negative for padding
            inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min

            # Add masks: causal mask handles future tokens, inverted_mask handles padding
            combined_mask = causal_mask[None, None, :, :] + inverted_mask
            return combined_mask
        else:
            # Return only the causal mask if no padding mask provided
            return causal_mask[None, None, :, :]  # Add batch and head dims


# --- HindiCausalLMHeadModel remains largely the same ---
# --- Ensure it uses the updated HindiCausalLMModel ---
class HindiCausalLMHeadModel(PreTrainedModel):
    """Hindi Causal Language Model with a language modeling head."""

    config_class = HindiCausalLMConfig
    base_model_prefix = "transformer"  # Changed to match base model
    supports_gradient_checkpointing = True
    _no_split_modules = ["HindiTransformerBlock"]  # Match base model

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.transformer = HindiCausalLMModel(config)  # Uses the updated base model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Tie weights method (optional but good practice)
    def tie_weights(self):
        if self.config.tie_word_embeddings:
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            output_embeddings.weight = input_embeddings.weight

            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = torch.nn.functional.pad(
                    output_embeddings.bias.data,
                    (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                    "constant",
                    0,
                )
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # KV Caching logic
        # if past_key_values:
        #     input_ids = input_ids[:, -1:] # Only need the last token if past is provided

        position_ids = kwargs.get("position_ids", None)
        # if attention_mask is not None and position_ids is None:
        #     # create position_ids on the fly for batch generation
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if past_key_values:
        #         position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                # "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:  # Updated type hint
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            # return ((loss,) + output) if loss is not None else output # With KV Cache loss calculation
            # Without KV Cache:
            if loss is not None:
                # The tuple format depends on whether past_key_values are returned by the base model
                # Assuming no KV cache returned by base model here:
                return (loss,) + (lm_logits,) + transformer_outputs[1:]  # loss, logits, hidden_states, attentions
            else:
                return (lm_logits,) + transformer_outputs[1:]  # logits, hidden_states, attentions

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            # past_key_values=transformer_outputs.past_key_values, # Add if using KV cache
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
