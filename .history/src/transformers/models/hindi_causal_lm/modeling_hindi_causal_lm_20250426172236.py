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
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if position_ids is None:
        cos = cos[:, :, : q.shape[2], :]
        sin = sin[:, :, : q.shape[2], :]
    else:
        cos = cos.squeeze(0).squeeze(0)[position_ids].unsqueeze(1)
        sin = sin.squeeze(0).squeeze(0)[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class HindiCausalLMSelfAttention(nn.Module):
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        if config.positional_encoding_type == "rope":
            self.rotary_emb = HindiCausalLMRotaryEmbedding(
                self.attention_head_size, 
                max_position_embeddings=config.max_position_embeddings
            )
        else:
            self.rotary_emb = None
            
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
        
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_length)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        causal_mask = self.causal_mask[:seq_length, :seq_length]
        attention_scores = attention_scores + causal_mask
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        output = self.output(context_layer)
        outputs = (output, attention_probs) if output_attentions else (output,)
        if use_cache:
            outputs += ((key_layer, value_layer),)
            
        return outputs

class HindiCausalLMTransformerBlock(nn.Module):
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.attention = HindiCausalLMSelfAttention(config)
        
        if config.normalization_layer == "rmsnorm":
            self.attention_layernorm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ffn_layernorm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ffn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
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
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        
        attn_outputs = self.attention(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )
        attention_output = attn_outputs[0]
        hidden_states = residual + attention_output
        
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
    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.weight"]

    def _init_weights(self, module):
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
    HindiCausalLM model with a language modeling head on top.
    This class is compatible with HuggingFace's `PreTrainedModel` and contains the main
    architecture for the Hindi language model.
"""

HINDI_CAUSAL_LM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens
        attention_mask (`torch.FloatTensor`, *optional*):
            Mask to avoid attention on padding tokens
        position_ids (`torch.LongTensor`, *optional*):
            Position indices for positional embeddings
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Cached key/value states from previous decoding steps
        use_cache (`bool`, *optional*):
            Whether to use cached states
        output_attentions (`bool`, *optional*):
            Whether to return attention tensors
        output_hidden_states (`bool`, *optional*):
            Whether to return hidden states
        return_dict (`bool`, *optional*):
            Whether to return a ModelOutput instead of tuple
"""

@add_start_docstrings(
    "Base Hindi Causal LM Model",
    HINDI_CAUSAL_LM_START_DOCSTRING,
)
class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.config = config
        
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        if config.positional_encoding_type == "learned":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings,
                config.hidden_size
            )
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand(1, -1))
            
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([
            HindiCausalLMTransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        if config.normalization_layer == "rmsnorm":
            self.final_layer_norm = HindiCausalLMRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.token_embeddings = new_embeddings

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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
            
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("Need input_ids or inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        if position_ids is None:
            if self.config.positional_encoding_type == "learned":
                position_ids = self.position_ids[:, past_key_values_length : input_shape[-1] + past_key_values_length]
            else:
                position_ids = torch.arange(
                    past_key_values_length, 
                    input_shape[-1] + past_key_values_length, 
                    dtype=torch.long, 
                    device=device
                ).unsqueeze(0).expand(input_shape[0], -1)

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)
            
        if self.config.positional_encoding_type == "learned":
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
            
        hidden_states = self.embedding_dropout(hidden_states)
        attention_mask = attention_mask if attention_mask is not None else torch.ones(input_shape, device=device)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
                
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions
            ] if v is not None)
            
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

@add_start_docstrings(
    "HindiCausalLM Model with Language Modeling Head",
    HINDI_CAUSAL_LM_START_DOCSTRING,
)
class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HindiCausalLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.token_embeddings.weight

    def get_input_embeddings(self):
        return self.model.token_embeddings

    def set_input_embeddings(self, value):
        self.model.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(HINDI_CAUSAL_LM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
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
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
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
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        input_shape = input_ids.shape

        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1].unsqueeze(-1)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = {"input_ids": input_ids}
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )
