# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from ...modeling_utils import PreTrainedModel

from ..moshi.modeling_moshi import MoshiForCausalLM, MoshiForConditionalGeneration

from ..llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaForCausalLM, CausalLMOutputWithPast
from ..llama.configuration_llama import LlamaConfig
from ..auto.modeling_auto import AutoModel

from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import rope_config_validation
from ...processing_utils import Unpack
from ...cache_utils import Cache
from ...utils import (
    ModelOutput,
)


class ConversationalSpeechModelDepthDecoderConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=128256,
        audio_vocab_size=2048,
        audio_num_codebooks=32,
        hidden_size=1024,
        intermediate_size=8192,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000,
        rope_scaling=None, #TODO: to change
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ConversationalSpeechModelConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=128256,
        audio_vocab_size=2048,
        audio_num_codebooks=32,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000,
        rope_scaling=None, #TODO: to change
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    

class ConversationalSpeechModelEmbeddingsWithProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding((config.audio_vocab_size + 3) * config.audio_num_codebooks, config.hidden_size)
        self.projector = nn.Linear(config.audio_vocab_size, config.hidden_size)

    def forward(self, input_ids):
        return self.projector(self.embed_tokens(input_ids))


def _codebook_head_selector(module, args, kwargs):
    # select the correct lm_head depending on the position

    input_ids = kwargs.get("input_ids", None)
    cache_position = kwargs.get("cache_position", None)
    position_ids = kwargs.get("position_ids", None)
    past_key_values = kwargs.get("past_key_values", None)
    inputs_embeds = kwargs.get("inputs_embeds", None)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if inputs_embeds is not None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        else:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
            )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    codebook_idx = position_ids[:, -1]
    module.lm_head = module.codebooks_heads[codebook_idx - 1]
    return args, kwargs


class ConversationalSpeechModelDepthDecoder(LlamaForCausalLM):
    config_class = ConversationalSpeechModelDepthDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechModelEmbeddingsWithProjector(config)
        self.codebooks_heads = [
            nn.Linear(config.hidden_size, config.audio_vocab_size + 3, bias=False) for _ in range(config.audio_num_codebooks - 1)
        ]
        self.lm_head = None 
        self.register_forward_pre_hook(_codebook_head_selector, with_kwargs=True)


class ConversationalSpeechModelEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_codebooks = config.audio_num_codebooks
        self.embed_text_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_audio_tokens = nn.Embedding((config.audio_vocab_size + 3) * config.audio_num_codebooks, config.hidden_size)

        audio_tokens_offsets = torch.arange(self.num_codebooks) * (config.audio_vocab_size + 3)
        self.register_buffer("audio_tokens_offsets", audio_tokens_offsets)

    def forward(self, input_ids):
        # input_ids should be of shape (batch_size, seq_length, num_codebooks)
        text_tokens = input_ids[:, :, -1:]
        audio_tokens = input_ids[:, :, :-1]

        # id 0 means it should be masked in the sum
        text_embeds = self.embed_text_tokens(text_tokens)
        text_embeds = text_embeds * text_tokens.bool().unsqueeze(-1)

        audio_embeds = self.embed_audio_tokens(audio_tokens + self.audio_tokens_offsets)
        audio_embeds = audio_embeds * audio_tokens.bool().unsqueeze(-1)

        inputs_embeds = torch.cat([text_embeds, audio_embeds], dim=-2)
        inputs_embeds = inputs_embeds.sum(dim=-2)

        return inputs_embeds


class ConversationalSpeechModelModel(LlamaModel):
    # TODO: change docstring input_ids, should have shape (batch_size, seq_length, num_codebooks) and not (batch_size, seq_length)
    # Likewise, the labels should have shape (batch_size, num_codebooks)
    # when doing the forward pass, 
    
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechModelEmbeddings(config)
    

class ConversationalSpeechModelForCausalLM(LlamaForCausalLM):
    pass


@dataclass
class ConversationalSpeechModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    depth_loss: Optional[torch.FloatTensor] = None
    audio_logits: torch.FloatTensor = None
    depth_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    depth_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    depth_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ConversationalSpeechModelForConditionalGeneration(LlamaForCausalLM):
    # TODO: not sure about the class naming here...
    # TODO: modifications to the forward could be simply replaced by a forward hook

    def __init__(self, config):
        super().__init__(config)
        self.audio_encoder = AutoModel.from_config(config.audio_encoder_config)
        self.depth_decoder = ConversationalSpeechModelDepthDecoder(config.depth_decoder_config)
        self.embed_text_tokens = nn.Linear(config.audio_encoder_config.hidden_size, config.depth_decoder_config.hidden_size)
        self.first_codebook_head = nn.Linear(config.depth_decoder_config.hidden_size, config.audio_vocab_size + 3, bias=False)
        self.model = ConversationalSpeechModelModel(config.depth_decoder_config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs, #: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #TODO: add docstring

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        if labels is not None:
            last_hidden_state = hidden_states[:, -1, :]
            projected_last_hidden_state = self.depth_decoder.embed_tokens.projector(last_hidden_state)

            codebook_embeds = self.depth_decoder.embed_tokens(labels)
            depth_decoder_inputs_embeds = torch.cat([projected_last_hidden_state, codebook_embeds], dim=-2)

            depth_decoder_outputs = self.depth_decoder(
                inputs_embeds=depth_decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss += depth_decoder_outputs.loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        return ConversationalSpeechModelOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            depth_loss=None if depth_decoder_outputs is None else depth_decoder_outputs.loss,
            audio_logits=None if depth_decoder_outputs is None else depth_decoder_outputs.logits,
            depth_past_key_values=None if outputs is None else outputs.past_key_values,
            depth_hidden_states=None if outputs is None else outputs.hidden_states,
            depth_attentions=None if outputs is None else outputs.attentions,
        ) 
        

__all__ = [
    "ConversationalSpeechModelDepthDecoderConfig",
    "ConversationalSpeechModelConfig",
    "ConversationalSpeechModelDepthDecoder",
    "ConversationalSpeechModelModel",
    "ConversationalSpeechModelForCausalLM",
    "ConversationalSpeechModelForConditionalGeneration",
]