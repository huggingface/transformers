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

from typing import Optional, Union, List, Tuple, Dict
from dataclasses import dataclass

import math
import os

import torch
import torch.nn as nn

from ..llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaForCausalLM, KwargsForCausalLM
from ..llama.configuration_llama import LlamaConfig

from ...generation import GenerationMixin
from ...generation import GenerationMixin, GenerateDecoderOnlyOutput, GenerationConfig
from ...generation.logits_process import (
    LogitsProcessorList,
)
from ...generation.stopping_criteria import (
    StoppingCriteriaList,
    StoppingCriteria
)
from ...configuration_utils import PretrainedConfig
from ...generation.stopping_criteria import StoppingCriteriaList
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
        num_codebooks=32,
        backbone_hidden_size=2048,
        vocab_size=2051,
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
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None, 
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
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.backbone_hidden_size = backbone_hidden_size
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


class ConversationalSpeechModelBackboneConfig(LlamaConfig):
    def __init__(
        self,
        num_codebooks=32,
        codebook_vocab_size=2051,
        vocab_size=128256,
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
        codebook_pad_token_id=0,
        bos_token_id=None,
        eos_token_id=0,
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
        self.num_codebooks = num_codebooks
        self.codebook_vocab_size = codebook_vocab_size
        self.vocab_size = vocab_size
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
        self.codebook_pad_token_id = codebook_pad_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ConversationalSpeechModelConfig(PretrainedConfig):
    model_type = "conversational_speech_model"

    def __init__(
        self,
        backbone_config: Dict=None,
        depth_decoder_config: Dict=None,
        initializer_range=0.02,
        tie_codebook_embeddings=True,
        eos_token_id=None,
        **kwargs,
    ):
        if backbone_config is None:
            backbone_config = {}
            logger.info("backbone_config is None. Initializing the backbone with default values.")

        if depth_decoder_config is None:
            depth_decoder_config = {}
            logger.info("depth_decoder_config is None. Initializing the depth decoder with default values.")

        self.backbone_config = ConversationalSpeechModelBackboneConfig(**backbone_config)
        self.depth_decoder_config = ConversationalSpeechModelDepthDecoderConfig(**depth_decoder_config)
        
        if self.backbone_config.num_codebooks != self.depth_decoder_config.num_codebooks:
            raise ValueError(
                f"{self.backbone_config.__class__.__name__} and {self.depth_decoder_config.__class__.__name__} "
                "`num_codebooks` must be the same."
            )
        
        if self.backbone_config.codebook_vocab_size != self.depth_decoder_config.vocab_size:
            raise ValueError(
                f"{self.backbone_config.__class__.__name__} `codebook_vocab_size` and {self.depth_decoder_config.__class__.__name__} "
                "`vocab_size` must be the same."
            )
        
        if self.backbone_config.hidden_size != self.depth_decoder_config.backbone_hidden_size:
            raise ValueError(
                f"{self.backbone_config.__class__.__name__} `hidden_size` and {self.depth_decoder_config.__class__.__name__} "
                "`backbone_hidden_size` must be the same."
            )

        self.initializer_range = initializer_range
        self.tie_codebook_embeddings = tie_codebook_embeddings
        self.vocab_size = self.backbone_config.codebook_vocab_size
        self.hidden_size = self.backbone_config.hidden_size
        self.num_codebooks = self.backbone_config.num_codebooks

        # disable tie_word_embeddings as it does not apply here
        kwargs["tie_word_embeddings"] = False

        super().__init__(**kwargs)
     


@dataclass
class ConversationalSpeechModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    depth_decoder_loss: Optional[torch.FloatTensor] = None
    depth_decoder_logits: torch.FloatTensor = None
    depth_decoder_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    depth_decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    depth_decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    backbone_loss: Optional[torch.FloatTensor] = None


class ConversationalSpeechModelEmbeddings(nn.Module):
    def __init__(self, num_codebooks, codebook_vocab_size, backbone_hidden_size, codebook_padding_idx):
        super().__init__()
        self.codebook_vocab_size = codebook_vocab_size
        self.embed_audio_tokens = nn.Embedding((num_codebooks * codebook_vocab_size), backbone_hidden_size, codebook_padding_idx)   

    def forward(self, input_ids, codebook_idxs):
        """
        Args:
            input_ids (`torch.Tensor`): 
                Codebooks ids of shape (batch_size, seq_length)
            codebook_idxs (`torch.Tensor`): 
                Corresponding codebook indices of shape (batch_size, seq_length)
        """
        offset = codebook_idxs * self.codebook_vocab_size
        return self.embed_audio_tokens(input_ids + offset)


class ConversationalSpeechModelDepthDecoder(LlamaModel):
    config_class = ConversationalSpeechModelDepthDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechModelEmbeddings(config.num_codebooks, config.vocab_size, config.backbone_hidden_size, self.padding_idx)
        self.inputs_embeds_projector = nn.Linear(config.backbone_hidden_size, config.hidden_size, bias=False)

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if backbone_last_hidden_states is None:
            raise ValueError("You must be provided backbone_last_hidden_states.")

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
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_seq_length, device=device) 

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids, cache_position.unsqueeze(0))
        
        codebook_idxs = cache_position.clone()
        has_first_codebook = codebook_idxs[0] == 0
        if has_first_codebook:
            # condition on backbone_last_hidden_states by concatenating it to the first position
            inputs_embeds = torch.cat([backbone_last_hidden_states.unsqueeze(1), inputs_embeds], dim=1)
            cache_position = torch.cat([cache_position, cache_position[-1:] + 1], dim=-1)
            if position_ids is not None:
                position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=-1)
        else:
            cache_position = cache_position.clone()
            position_ids = position_ids.clone()
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
    def __init__(self, hidden_size, num_codebooks, vocab_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.weight = nn.Parameter(
            torch.empty(self.num_codebooks, hidden_size, vocab_size)
        )

    def reset_parameters(self):
        for i in range(self.num_codebooks - 1):
            nn.init.kaiming_uniform_(
                self.weight[i], a=math.sqrt(5)
            )

    def forward(self, hidden_states, last_cache_position):
        if last_cache_position is None:
            codebook_weight = self.weight
        else:
            n_kept_logits = hidden_states.shape[1]
            codebook_weight = self.weight[last_cache_position - n_kept_logits + 1 : last_cache_position + 1]

        return torch.einsum("bsh,sho->bso", hidden_states, codebook_weight)     


class ConversationalSpeechModelDepthDecoderForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = None

    def __init__(self, config):
        super().__init__(config)
        self.codebooks_head = ConversationalSpeechModelCodebooksHead(config.hidden_size, config.num_codebooks, config.vocab_size)
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
                # skip idx 0 logits since it's for the concatenated backbone last hidden state
                slice_indices = slice(1, None) 
            else:
                slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep

        logits = self.codebooks_head(hidden_states[:, slice_indices, :], cache_position)
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
    def __init__(self, hidden_size, vocab_size, num_codebooks, codebook_vocab_size, padding_idx, codebook_padding_idx):
        super().__init__()
        self.codebook_padding_idx = codebook_padding_idx
        self.embed_text_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.embed_audio_tokens = nn.Embedding(
            (num_codebooks * codebook_vocab_size), hidden_size, codebook_padding_idx
        )

        audio_tokens_offsets = torch.arange(num_codebooks) * codebook_vocab_size
        self.register_buffer("audio_tokens_offsets", audio_tokens_offsets)

    def forward(self, input_ids):
        """
        Args:
            input_ids (`torch.Tensor` of shape (batch_size, seq_length, num_codebooks + 1)):
                On last dimension, 32 first values are codebook tokens, and last value is a text token.
        Returns:
            `torch.Tensor` of shape (batch_size, seq_length, hidden_size):
                Embedded tokens, summed over the last dimension according to input_ids_mask.
        """
        text_tokens = input_ids[:, :, -1:]

        audio_tokens = input_ids[:, :, :-1]
        # apply the offset only to the non-padded codebook tokens
        audio_tokens_mask = (audio_tokens != self.codebook_padding_idx)
        audio_tokens = audio_tokens + self.audio_tokens_offsets
        audio_tokens *= audio_tokens_mask 

        text_embeds = self.embed_text_tokens(text_tokens)
        audio_embeds = self.embed_audio_tokens(audio_tokens)

        inputs_embeds = torch.cat([text_embeds, audio_embeds], dim=-2)
        inputs_embeds = inputs_embeds.sum(dim=-2)

        return inputs_embeds


class ConversationalSpeechModelBackboneModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechBackboneModelEmbeddings(
            config.hidden_size,
            config.vocab_size,
            config.num_codebooks,
            config.codebook_vocab_size,
            self.padding_idx,
            config.codebook_pad_token_id,
        )


class ConversationalSpeechModelForCausalLM(LlamaForCausalLM, GenerationMixin):
    _tied_weights_keys = ["backbone_model.embed_tokens.embed_audio_tokens.weight", "depth_decoder.model.embed_tokens.embed_audio_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        del self.model
        self.depth_decoder = ConversationalSpeechModelDepthDecoderForCausalLM(config.depth_decoder_config)
        self.backbone_model = ConversationalSpeechModelBackboneModel(config.backbone_config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_mask: Optional[torch.Tensor] = None,
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

        backbone_outputs = self.backbone_model(
            input_ids=input_ids,
            input_ids_mask=input_ids_mask,
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

        loss = None
        backbone_loss = None
        depth_decoder_loss = None
        depth_decoder_outputs = None
        if labels is not None:
            backbone_labels = labels[:, :, 0] if labels is not None else None
            backbone_loss = self.loss_function(
                logits=backbone_logits, labels=backbone_labels, vocab_size=self.config.vocab_size, **kwargs
            )

            depth_decoder_input_ids = input_ids[:, :, :self.config.num_codebooks].view(-1, self.config.num_codebooks)
            backbone_last_hidden_states = backbone_hidden_states.view(-1, self.config.hidden_size)
            depth_decoder_labels = labels[:, :, :self.config.num_codebooks].view(-1, self.config.num_codebooks)

            depth_decoder_outputs = self.depth_decoder(
                input_ids=depth_decoder_input_ids,
                backbone_last_hidden_states=backbone_last_hidden_states,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                labels=depth_decoder_labels,
            )
            depth_decoder_loss = depth_decoder_outputs.loss
            loss = backbone_loss + depth_decoder_loss

        return ConversationalSpeechModelOutputWithPast(
            loss=loss,
            backbone_loss=backbone_loss,
            depth_decoder_loss=depth_decoder_loss,
            logits=backbone_logits,
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,  
            depth_decoder_logits=depth_decoder_outputs.logits if depth_decoder_outputs is not None else None,
            depth_decoder_past_key_values=depth_decoder_outputs.past_key_values if depth_decoder_outputs is not None else None,
            depth_decoder_hidden_states=depth_decoder_outputs.hidden_states if depth_decoder_outputs is not None else None,
            depth_decoder_attentions=depth_decoder_outputs.attentions if depth_decoder_outputs is not None else None,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ):
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # expand input_ids to (batch_size, seq_length, num_codebooks)
        input_ids = input_ids.reshape(batch_size, 0, self.config.num_codebooks + 1)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            is_compileable = is_compileable and not self.generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
 
            # infer the depth decoder
            first_codebook_ids = next_tokens[:, None]
            last_backbone_hidden_state = outputs.hidden_states[-1][:, -1, :]
            depth_decoder_outputs = self.depth_decoder.generate(
                input_ids=first_codebook_ids,
                backbone_last_hidden_states=last_backbone_hidden_state,
                return_dict_in_generate=True,
                min_new_tokens=31,
                max_new_tokens=31,
            )
            codebook_ids = depth_decoder_outputs.sequences
            next_tokens = torch.cat([codebook_ids, torch.zeros((codebook_ids.shape[0], 1), dtype=torch.long, device=codebook_ids.device)], dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None, :]], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # for the eos stopping criteria, is it expected that the eos token is the same for each codebook !!!!
            unfinished_sequences = unfinished_sequences & ~(input_ids[:, -1, :] == self.config.backbone_config.eos_token_id).all(-1) 
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
            del depth_decoder_outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids

    def generate(self, input_ids, **kwargs):
        inputs_embeds = self.backbone_model.get_input_embeddings()(input_ids)
        return GenerationMixin.generate(self,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            **kwargs
        )


__all__ = [
    "ConversationalSpeechModelDepthDecoderConfig",
    "ConversationalSpeechModelConfig",
    "ConversationalSpeechModelDepthDecoder",
    "ConversationalSpeechModelDepthDecoderForCausalLM",
    "ConversationalSpeechModelBackboneModel",
    "ConversationalSpeechModelForCausalLM",
]