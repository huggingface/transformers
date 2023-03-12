# coding=utf-8
# Copyright 2023 Microsoft Research and HuggingFace Inc. team.
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
""" PyTorch UDOP model."""


import logging
from typing import Any, Dict, Optional, Sequence, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor

from transformers import T5Config, T5PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Block, T5ForConditionalGeneration, T5LayerNorm

from core.models.embedding.cell_embed import CellEmbeddings
from core.models.embedding.relative.relative import (
    RelativePositionBias1D,
    RelativePositionBiasAggregated,
    RelativePositionBiasBase,
    create_relative_bias,
)
from core.models.mae.build import mae_model

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
    attention_mask: torch.FloatTensor = None
    seg_data: torch.FloatTensor = None

        
@dataclass
class VisSeq2SeqLMOutput(BaseModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    image_output: Optional[Tuple[torch.FloatTensor]] = None
    image_target: Optional[Tuple[torch.FloatTensor]] = None  
    image_mask_label: Optional[Tuple[torch.FloatTensor]] = None 


def pad_sequence(seq, target_len, pad_value=0):
    if isinstance(seq, torch.Tensor):
        n = seq.shape[0]
    else:
        n = len(seq)
        seq = torch.tensor(seq)
    m = target_len - n
    if m > 0:
        ret = torch.stack([pad_value] * m).to(seq)
        seq = torch.cat([seq, ret], dim=0)
    return seq[:target_len]


def collate_vlembed(inputs_patches, inputs_embeds, seg_data, visual_segdata, vis_special_token=None, attention_mask=None, num_patches=14, max_len=0):
    
    L = num_patches
    ocr_points_x = torch.clip(torch.floor((seg_data[:, :, 0]+seg_data[:, :, 2])/2.0 * L).long(), 0, L-1)
    ocr_points_y = torch.clip(torch.floor((seg_data[:, :, 1]+seg_data[:, :, 3])/2.0 * L).long(), 0, L-1) * L
    ocr_points = ocr_points_x + ocr_points_y
    target_seg = (seg_data.mean(-1) == 0.0) | (seg_data.mean(-1) == 1.0)
    repeated_vision_embeds = torch.gather(inputs_patches, 1, ocr_points.unsqueeze(-1).repeat(1, 1, inputs_patches.size(-1)))
    repeated_vision_embeds[target_seg] = 0.0
    inputs_embeds += repeated_vision_embeds
    
    
    patch_inds = torch.full_like(inputs_patches[:, :, 0], True).bool()
    ind = torch.cat([torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points), ocr_points[:, :, None]], -1).flatten(0,1)
    rows, cols = zip(*ind)
    patch_inds[rows, cols] = False
    
    
    input_vision_patches = [inputs_patches[i][patch_inds[i]] for i in range(len(patch_inds))]
    visual_segdata = [visual_segdata[i][patch_inds[i]] for i in range(len(patch_inds))]
    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_segdata]
        
    if max_len == 0:
        max_len = inputs_patches.size(1)
    else:
        max_len = max_len - inputs_embeds.size(1)    
    inputs_vision_patches = torch.stack([pad_sequence(item, max_len, torch.zeros_like(inputs_patches[0, 0])) for item in input_vision_patches])
    visual_segdata = torch.stack([pad_sequence(item, max_len, torch.zeros_like(seg_data[0, 0])) for item in visual_segdata])
    if attention_mask is not None:
        visual_attention_mask = torch.stack([pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask])
        
    if vis_special_token is not None:
        inputs_vision_patches += vis_special_token
    
    
    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)
    seg_data = torch.cat([seg_data, visual_segdata], 1)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)
    return inputs_embeds, seg_data, attention_mask

         
class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual
    

# Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel with T5->Udop
class UdopPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UdopConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class UdopStack(T5PreTrainedModel):
    """
    Almost exact copy of transformers T5Stack with the modification
    of passing `position_bias` in the forward method
    """

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self._max_length = config.max_length
        
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

            
        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

        # tie weights of original position bias of encoder
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias
                )
                
        self.init_weights()

    @staticmethod
    def _get_relative_bias(config: T5Config) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)
    
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
        head_mask=None,
        past_key_values=None,
        ids_keep=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cross_attn_head_mask = None,
        position_bias=None,  # modified line,
        inputs_patches=None, # modified line,
        seg_data=None, # modified line,
        visual_seg_data=None, # modified line,
        num_patches=None, # modified line,
        special_vis_token=None, # modified line,
    ):
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = True #False #True #output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                      
            
            
        #======================================================    
        # input embeddings processing

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
            seg_data = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(
                self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
            )
            # encoder_attention_mask = attention_mask
            logger.warning('Empty batch')
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
                
        if inputs_patches is not None:
            #===========================    
            # combine OCR text and visual embed
            inputs_embeds, seg_data, attention_mask = collate_vlembed(inputs_patches, inputs_embeds, seg_data, visual_seg_data, special_vis_token, attention_mask, num_patches, 0)
            input_shape = inputs_embeds.size()[:-1]
            
        if not self.is_decoder:
            inputs_embeds += self.cell2dembedding(seg_data)

        batch_size, seq_length = input_shape
        
        
        #======================================================    
        # input masking/pos embed processing

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

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

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
            position_bias = self.relative_bias(
                    attention_mask=attention_mask, seg_data=seg_data
                )
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None

        
        
        #======================================================    
        # model inferencing 
        
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
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

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
                ]
                if v is not None
            )
        
        return BaseModelOutputWithVisionEmbeds(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask=attention_mask,
            seg_data=seg_data,
        )


class UdopForConditionalGeneration(T5ForConditionalGeneration):
    """
    Copied from original T5ForConditionalGeneration class with signature extended with 2D data.
    """

    def __init__(self, config):
        super(UdopForConditionalGeneration, self).__init__(config)

        # get max length of decoder part, for T5 decoder lenght depends
        # on the task and it can be modified by passing `_max_decoder_length` to the model/config
        self._max_decoder_length = config.max_decoder_length if hasattr(config, "max_decoder_length") else 256

        self.config.decoder_start_token_id = self.config.pad_token_id

        self.encoder = UdopStack(self.encoder.config, self.shared)
        self.decoder = UdopStack(self.decoder.config, self.shared)
        
        self.init_weights()
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics

        mae_model_tmp = mae_model(config.mae_version, config.mae_checkpoint, config.image_size, config.vocab_size, config.max_2d_position_embeddings)

        self.patch_embed = mae_model_tmp.patch_embed
        self.embed_dim = mae_model_tmp.embed_dim
        self.pos_embed = mae_model_tmp.pos_embed
        self.special_vis_token = mae_model_tmp.special_vis_token
        
    @staticmethod
    def get_required_segment_levels() -> Sequence[str]:
        return ["tokens"]

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def mae_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
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
        visual_seg_data: Dict[str, Any] = None,
        masked_lm_labels: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
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
        
        if input_dict is not None:
            return_task_outputs = []
            for task in input_dict:
                return_task_outputs.append(self.forward(**input_dict[task]))
            return return_task_outputs

        if encoder_outputs is None:
            inputs_patches=None
            if image is not None:
                assert visual_seg_data is not None
                x = self.patch_embed(image)
                num_patches = image.size(2) // 16
                if ids_keep is not None:
                    x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))
                    pad_tokens = self.pad_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
                    x_padded = torch.cat([x, pad_tokens], dim=1)  # no cls token
                    x_padded = torch.gather(x_padded, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_padded.shape[2]))
                    inputs_patches = x_padded
                else:
                    inputs_patches = x

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                seg_data=seg_data,
                visual_seg_data=visual_seg_data,
                inputs_patches=inputs_patches,
                num_patches=num_patches,
                special_vis_token=self.special_vis_token,
                ids_keep=ids_keep,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )           

        if encoder_outputs is None:
            return None

        if masked_lm_labels is not None and labels is None:
            labels = masked_lm_labels

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        # ugly hack for model to work as an encoder
        if decoder_input_ids is None and masked_lm_labels is None:
            return encoder_outputs

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=encoder_outputs.attention_mask,
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