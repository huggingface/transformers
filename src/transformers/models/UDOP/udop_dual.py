# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from transformers import T5Config, T5PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Block, T5ForConditionalGeneration, T5LayerNorm

from embedding.cell_embed import CellEmbeddings
from embedding.relative.relative import (
    RelativePositionBias1D,
    RelativePositionBiasAggregated,
    RelativePositionBiasBase,
    create_relative_bias,
)
from mae.build import mae_model

logger = logging.getLogger(__name__)

        
@dataclass
class BaseModelOutputWithVisionEmbeds(BaseModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_embeds: torch.FloatTensor = None

        
@dataclass
class VisSeq2SeqLMOutput(BaseModelOutput):

    loss: Optional[torch.FloatTensor] = None
    image_output: Optional[Tuple[torch.FloatTensor]] = None
    image_target: Optional[Tuple[torch.FloatTensor]] = None  
    image_mask_label: Optional[Tuple[torch.FloatTensor]] = None 
        
         
class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


class T52dStack(T5PreTrainedModel):
    """
    Almost exact copy of transformers T5Stack with the modification
    of passing `position_bias` in the forward method
    """

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        setattr(config, 'output_attentions', True)
        if self.is_decoder:
            self.num_layers = (
                config.truncate_decoder_after_layer if config.truncate_decoder_after_layer else config.num_layers
            )
        else:
            self.num_layers = (
                config.truncate_encoder_after_layer if config.truncate_encoder_after_layer else config.num_layers
            )

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
        if not self.is_decoder:
            self.cell2dembedding = CellEmbeddings(config.max_2d_position_embeddings, config.hidden_size)

        self.init_weights()

        if not self.is_decoder:
            self.vision_encoder = mae_model(config.mae_version, os.path.join(config.data_dir, config.mae_checkpoint), config.image_size, config.vocab_size, config.max_2d_position_embeddings)
        else:
            self.vision_fc = nn.Linear(config.d_model, config.d_model)
            self.vision_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
            

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        vision_embeds=None,
        head_mask=None,
        past_key_values=None,
        image=None,
        ids_keep=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        position_bias=None,  # modified line,
        seg_data=None,  # modified line
        cross_attn_head_mask = None,
        token_type_ids = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = True #False #True #output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if image is not None:
            vision_embeds = self.vision_encoder.forward_encoder(image, ids_keep=ids_keep)
        else:
            vision_embeds = None
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and torch.numel(input_ids) == 0:
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            seg_data['tokens']['bboxes'] = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            logger.warning('Empty batch')
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
            if seg_data is not None:
                inputs_embeds += self.cell2dembedding(seg_data)
            if token_type_ids is not None:
                inputs_embeds += self.token_type_embedding(token_type_ids)

        batch_size, seq_length = input_shape
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(self)         
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )
            
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        if self.is_decoder:  # modified lines
            position_bias = None
        else:
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None
        
        
        hidden_states = inputs_embeds
        hidden_states = self.dropout(hidden_states)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:     # MP fixes
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]
            # We share the position biases between the layers - the first layer store them

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
                
            # append next layer key value states
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    vision_embeds,
                ]
                if v is not None
            )
        
        return BaseModelOutputWithVisionEmbeds(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            vision_embeds=vision_embeds,
        )


class UdopDualForConditionalGeneration(T5ForConditionalGeneration):
    """
    Copied from original T5ForConditionalGeneration class with signature extended with 2D data.
    :param config: a `T5Config` instance
    """

    def __init__(self, config):
        super(UdopDualForConditionalGeneration, self).__init__(config)

        # get max length of decoder part, for T5 decoder lenght depends
        # on the task and it can be modified by passing `_max_decoder_length` to the model/config
        self._max_decoder_length = config.max_decoder_length if hasattr(config, "max_decoder_length") else 256
        self.config.decoder_start_token_id = self.config.pad_token_id

        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

        # tie weights of original position bias of encoder
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
                )
        self.init_weights()
        
        self.encoder = T52dStack(self.encoder.config, self.shared)
        self.decoder = T52dStack(self.decoder.config, self.shared)

    @staticmethod
    def get_required_segment_levels() -> Sequence[str]:
        return ["tokens"]

    @staticmethod
    def _get_relative_bias(config: T5Config) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        ids_keep: Optional[Tensor] = None,
        ids_restore: Optional[Tensor] = None,
        image_mask_label: Optional[Tensor] = None,
        mask_ratio: Optional[Tensor] = None,
        seg_data: Dict[str, Any] = None,
        labels: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        char_ids: Optional[Tensor] = None,
        char_seg_data: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_dict: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Tensor, ...]:

        if seg_data is not None:
            seg_data = torch.clip(seg_data, 0.0, 1.0)
        if input_dict is not None:
            return_task_outputs = []
            for task in input_dict:
                return_task_outputs.append(self.forward(**input_dict[task]))
            return return_task_outputs
        if encoder_outputs is None:
            # compute positional bias (can be aggregation of 1D and 2D biases)
            encoder_position_bias = self.relative_bias(
                attention_mask=attention_mask, seg_data=seg_data
            )
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                position_bias=encoder_position_bias,
                seg_data=seg_data,
                image=image,
                ids_keep=ids_keep,
            )

        if encoder_outputs is None:
            return None

        if ids_restore is not None:
            # print(char_ids.size(), char_seg_data.size(), input_ids.size())
            image_output = self.encoder.vision_encoder.forward_decoder(encoder_outputs.vision_embeds, ids_restore, context=encoder_outputs.last_hidden_state, char_inputs=[char_ids, char_seg_data])
            loss = self.encoder.vision_encoder.forward_loss(image, image_output, image_mask_label)
            return VisSeq2SeqLMOutput(
                loss=loss,
                image_output=image_output,
                image_target=image,
                image_mask_label=image_mask_label
            )
        else:
            if masked_lm_labels is not None:
                labels = masked_lm_labels
                
            if decoder_input_ids is None and labels is not None:
                decoder_input_ids = self._shift_right(labels)
                
            if encoder_outputs.vision_embeds is not None:
                vision_embeds = self.decoder.vision_fc(encoder_outputs.vision_embeds)
                vision_embeds = self.decoder.vision_norm(vision_embeds)
                hidden_states = torch.cat([encoder_outputs.last_hidden_state, vision_embeds], 1)
                encoder_outputs.last_hidden_state = hidden_states
                encoder_outputs.vision_embeds = None                
            
            # ugly hack for model to work as an encoder
            if decoder_input_ids is None and labels is None:
                return encoder_outputs
            
            attention_mask = torch.cat([attention_mask, torch.ones_like(encoder_outputs.last_hidden_state[:, :encoder_outputs.last_hidden_state.size(1)-attention_mask.size(1), 0])], 1)
    
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            return outputs  # type: ignore
    
    def get_encoder(self):
        return self
    
    
