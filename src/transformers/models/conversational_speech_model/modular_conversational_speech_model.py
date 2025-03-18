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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modeling_utils import PreTrainedModel

from ..moshi.modeling_moshi import MoshiForCausalLM, MoshiForConditionalGeneration

from ..llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaForCausalLM, KwargsForCausalLM
from ..llama.configuration_llama import LlamaConfig
from ..auto.modeling_auto import AutoModel

from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import rope_config_validation
from ...processing_utils import Unpack
from ...cache_utils import Cache
from ...utils import (
    ModelOutput,
    logging
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from ...cache_utils import Cache, DynamicCache


logger = logging.get_logger(__name__)


class ConversationalSpeechModelDepthDecoderConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=2051,
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
        vocab_size=2051,
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


class ConversationalSpeechModelEmbeddings(nn.Embedding):
    def __init__(self, *args, audio_vocab_size=None, audio_num_codebooks=None, **kwargs):
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        super().__init__(*args, **kwargs)
        

    def reset_parameters(self):
        for i in range(self.audio_num_codebooks):
            nn.init.normal_(self.weight[i * (self.audio_vocab_size + 3): (i + 1) * (self.audio_vocab_size + 3)])
        self._fill_padding_idx_with_zero()

    def forward(self, input, codebook_idxs):
        offsets = codebook_idxs * (self.audio_vocab_size + 3)
        return F.embedding(
            input + offsets,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

class ConversationalSpeechModelDepthDecoder(LlamaModel):
    config_class = ConversationalSpeechModelDepthDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechModelEmbeddings(
            num_embeddings=(config.audio_vocab_size + 3) * config.audio_num_codebooks,
            embedding_dim=config.audio_vocab_size,
            audio_vocab_size=config.audio_vocab_size,
            audio_num_codebooks=config.audio_num_codebooks,
        )
        self.inputs_embeds_projector = nn.Linear(config.audio_vocab_size, config.hidden_size, bias=False)

    def _get_last_position_id(self, input_ids, inputs_embeds, past_key_values, position_ids, cache_position):
        if position_ids is not None:
            return position_ids[-1]
                                
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if inputs_embeds is not None:
                last_cache_position = past_seen_tokens + inputs_embeds.shape[1]
            else:
                last_cache_position = past_seen_tokens + input_ids.shape[1]

        # what about the else? at least position_ids/ cache_position should be provided

        return last_cache_position

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # input_ids shape (batch_size, seq_length, num_codebooks)
        # we need to have backbone_hidden_states shape (batch_size, seq_length, hidden_size)
        # we can run it like (batch_size * seq_length, num_codebooks)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            inputs_seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_seq_length, device=device
            )
        
        codebook_idxs = cache_position.clone()
        is_first_codebook = codebook_idxs[0] == 0
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids, cache_position.unsqueeze(0))
            if is_first_codebook:
                if backbone_last_hidden_states is None:
                    raise ValueError("backbone_last_hidden_states must be provided when input_ids is provided.")
                inputs_embeds = torch.cat([backbone_last_hidden_states.unsqueeze(1), inputs_embeds], dim=1)
                cache_position = torch.cat([cache_position, cache_position[-1:] + 1], dim=-1)
                if position_ids is not None:
                    position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=-1)
        
        if not is_first_codebook:
            cache_position += 1
            position_ids += 1
                
        inputs_embeds = self.inputs_embeds_projector(inputs_embeds)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class ConversationalSpeechModelCodebooksHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_vocab_size = config.audio_vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.weight = nn.Parameter(
            torch.empty(self.audio_num_codebooks, config.hidden_size, config.audio_vocab_size + 3)
        )
        # we only need audio_num_codebooks - 1
        # can be shortcutted to avoid last computation (codebook 32)
        self.reset_parameters()


    def reset_parameters(self):
        for i in range(self.audio_num_codebooks - 1):
            nn.init.kaiming_uniform_(
                self.weight[i], a=math.sqrt(5)
            )

    def forward(self, hidden_states, last_cache_position):
        n_kept_logits = hidden_states.shape[1]
        codebook_weight = self.weight[last_cache_position - n_kept_logits : last_cache_position + 1]

        return torch.einsum('bsh,sho->bso', hidden_states, codebook_weight)


class ConversationalSpeechModelDepthDecoderForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.codebooks_head = ConversationalSpeechModelCodebooksHead(config)
        self.model = ConversationalSpeechModelDepthDecoder(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_states: Optional[torch.FloatTensor] = None, 
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
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            backbone_last_hidden_states=backbone_last_hidden_states,
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
        if isinstance(logits_to_keep, int):
            if logits_to_keep == 0:
                slice_indices = slice(1, None)  # Skip the first logit
            else:
                slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep

        # cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if inputs_embeds is not None:
                last_cache_position = past_seen_tokens + inputs_embeds.shape[1]
            else:
                last_cache_position = past_seen_tokens + input_ids.shape[1]
        else:
            last_cache_position = cache_position[-1]

        logits = self.codebooks_head(hidden_states[:, slice_indices, :], last_cache_position)
        logits = logits.contiguous()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ConversationalSpeechBackboneModelEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_codebooks = config.audio_num_codebooks
        self.embed_audio_tokens = nn.Embedding((config.audio_vocab_size + 3) * config.audio_num_codebooks, config.hidden_size)
        audio_tokens_offsets = torch.arange(self.num_codebooks) * (config.audio_vocab_size + 3)
        self.register_buffer("audio_tokens_offsets", audio_tokens_offsets)

    def forward(self, input_ids):
        # input_ids should be of shape (batch_size, seq_length, num_codebooks)
        audio_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        inputs_embeds = audio_embeds.sum(dim=-2)

        return inputs_embeds


class ConversationalSpeechModelBackboneModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechBackboneModelEmbeddings(config)


# there is no reason for the class to exist by itself I think since it's not able to generate without the depth decoder
# class ConversationalSpeechModelBackboneModelForCausalLM(LlamaForCausalLM):
#     # # TODO: change docstring input_ids, should have shape (batch_size, seq_length, num_codebooks) and not (batch_size, seq_length) 
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = ConversationalSpeechModelBackboneModel(config)


class ConversationalSpeechModelForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.depth_decoder = ConversationalSpeechModelDepthDecoderForCausalLM(config.depth_decoder_config)
        self.backbone_model = ConversationalSpeechModelBackboneModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None, # shape (batch_size, seq_length, num_codebooks)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # shape (batch_size, seq_length, num_codebooks)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # first codeboook label is for the backbone model
        backbone_labels = labels[:, :, 0] if labels is not None else None  # shape (batch_size, seq_length)
        depth_decoder_labels = labels[:, :, 1:] if labels is not None else None  # shape (batch_size, seq_length, num_codebooks - 1)

        backbone_outputs = self.backbone_model(
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

        backbone_hidden_states = backbone_outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        backbone_logits = self.lm_head(backbone_hidden_states[:, slice_indices, :])

        backbone_loss = None
        if labels is not None:
            backbone_loss = self.loss_function(logits=backbone_logits, labels=backbone_labels, vocab_size=self.config.vocab_size, **kwargs)
        
        if depth_decoder_labels is not None:
            projected_hidden_states = self.depth_decoder.embed_tokens.projector(backbone_hidden_states)
            codebook_embeds = self.depth_decoder.embed_tokens(depth_decoder_labels)
            depth_decoder_inputs_embeds = torch.cat([projected_hidden_states, codebook_embeds], dim=-2)

            depth_decoder_outputs = self.depth_decoder(
                inputs_embeds=depth_decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=depth_decoder_labels,
            )

        return 
        

__all__ = [
    "ConversationalSpeechModelDepthDecoderConfig",
    "ConversationalSpeechModelConfig",
    "ConversationalSpeechModelDepthDecoder",
    "ConversationalSpeechModelDepthDecoderForCausalLM",
    "ConversationalSpeechModelBackboneModel",
    "ConversationalSpeechModelBackboneModelForCausalLM",
    "ConversationalSpeechModelForConditionalGeneration",
]