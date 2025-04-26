# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the MIT License.
#

"""PyTorch HindiCausalLM model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"

HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "convaiinnovations/hindi-foundational-model-base",
    # See all HindiCausalLM models at https://huggingface.co/models?filter=hindi_causal_lm
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class HindiCausalLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        HindiCausalLMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HindiCausalLMRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=512, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies rotary position embedding to queries and keys."""
    if position_ids is None:
        cos = cos[:, :, : q.shape[2], :]  # [bs, 1, seq_len, dim]
        sin = sin[:, :, : q.shape[2], :]  # [bs, 1, seq_len, dim]
    else:
        cos = cos.squeeze(0).squeeze(0)[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin.squeeze(0).squeeze(0)[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class HindiCausalLMSelfAttention(nn.Module):
    """Causal self-attention layer"""
    def __init__(self, config: HindiCausalLMConfig):
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
        self.output = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # RoPE
        if config.positional_encoding_type == "rope":
            self.rotary_emb = HindiCausalLMRotaryEmbedding(self.attention_head_size, max_position_embeddings=config.max_position_embeddings)
        else:
            self.rotary_emb = None
        
        self.max_position_embeddings = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_position_embeddings, config.max_position_embeddings) * -1e10,
                diagonal=1
            )
        )
        
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        batch_size, seq_length = hidden_states.size()[:2]
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Apply rotary embeddings if configured
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_length)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply causal mask - prevents attending to future tokens
        causal_mask = self.causal_mask[:seq_length, :seq_length]
        attention_scores = attention_scores + causal_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Add the attention mask to the raw attention scores
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to [batch_size, seq_length, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Final output projection
        output = self.output(context_layer)
        
        outputs = (output, attention_probs) if output_attentions else (output,)

        if use_cache:
            outputs = outputs + ((key_layer, value_layer),)
        return outputs


class HindiCausalLMTransformerBlock(nn.Module):
    """Transformer block with causal attention for language modeling"""
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.attention = HindiCausalLMSelfAttention(config)
        
        # Use the appropriate normalization layer
        if config.normalization_layer == "rmsnorm":
            self.attention_layernorm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ffn_layernorm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ffn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            ACT2FN[config.hidden_act],
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        # Self-attention block with residual connection and layer norm
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        
        # Self-attention
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attention_output = attn_outputs[0]
        
        # Add residual connection
        hidden_states = residual + attention_output
        
        # Feed-forward block with residual connection and layer norm
        residual = hidden_states
        hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_outputs[1],)
        
        if use_cache:
            outputs += (attn_outputs[-1],)
        
        return outputs


class HindiCausalLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.weight"]
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, HindiCausalLMRMSNorm):
            module.weight.data.fill_(1.0)


HINDI_CAUSAL_LM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HindiCausalLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

HINDI_CAUSAL_LM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
            `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape `(batch_size, 1)`
            instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up
            decoding (see `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare HindiCausalLM Model outputting raw hidden-states without any specific head on top.",
    HINDI_CAUSAL_LM_START_DOCSTRING,
)
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    """
    Hindi Causal Language Model for text generation
    """
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Use different position embedding based on config
        if config.positional_encoding_type == "learned":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings,
                config.hidden_size
            )
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand(1, -1))
        
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            HindiCausalLMTransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        if config.normalization_layer == "rmsnorm":
            self.final_layer_norm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.token_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        self.token_embeddings = new_embeddings
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = torch.triu(
                torch.full(
                    (input_shape[-1], input_shape[-1] + past_key_values_length),
                    torch.tensor(torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device),
                    device=inputs_embeds.device,
                ),
                diagonal=1 + past_key_values_length,
            )[:input_shape[-1], :]
            combined_attention_mask = combined_attention_mask.unsqueeze(0).expand(input_shape[0], -1, -1)
        
        # Apply the user-provided attention mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, seq_len, 1]
            expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        
        return combined_attention_mask
    
    @add_start_docstrings_to_model_forward(HINDI_CAUSAL_LM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input shape and device
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Past key values length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        # Get position IDs
        if position_ids is None:
            if self.config.positional_encoding_type == "learned":
                position_ids = self.position_ids[:, past_key_values_length : input_shape[-1] + past_key_values_length]
            else:
                position_ids = torch.arange(
                    past_key_values_length, input_shape[-1] + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).expand(input_shape[0], -1)
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)
        
        # Add position embeddings if needed
        if self.config.positional_encoding_type == "learned":
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        
        # Apply dropout
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Default attention mask (all tokens can be attended to)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        # Prepare attention mask
        if self.config.positional_encoding_type == "rope":
            # No need to adjust the attention_mask for RoPE
            pass
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        
        # Init previous hidden states if not empty
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Apply transformer layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_
