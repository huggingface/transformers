# coding=utf-8
# Copyright 2024 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ProPainter model."""

import collections.abc
import itertools
import math
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torchvision
from functools import reduce

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.modules.utils import _pair, _single
from torch.cuda.amp import autocast
from torch.nn.functional import normalize


from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel,TORCH_INIT_FUNCTIONS
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_propainter import ProPainterConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ProPainterConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "ruffy369/propainter"
_EXPECTED_OUTPUT_SHAPE = [80, 240, 432, 3] #****************************TO FILL

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "ruffy369/propainter"


# Copied from transformers.models.vit.modeling_vit.ViTEmbeddings with ViT->ProPainter
# class ProPainterEmbeddings(nn.Module):
#     """
#     Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
#     """

#     def __init__(self, config: ProPainterConfig, use_mask_token: bool = False) -> None:
#         super().__init__()

#         # self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
#         # self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
#         # self.patch_embeddings = ProPainterPatchEmbeddings(config)
#         # num_patches = self.patch_embeddings.num_patches
#         # self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.config = config

#     def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
#         """
#         This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
#         resolution images.

#         Source:
#         https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
#         """

#         num_patches = embeddings.shape[1] - 1
#         num_positions = self.position_embeddings.shape[1] - 1
#         if num_patches == num_positions and height == width:
#             return self.position_embeddings
#         class_pos_embed = self.position_embeddings[:, 0]
#         patch_pos_embed = self.position_embeddings[:, 1:]
#         dim = embeddings.shape[-1]
#         h0 = height // self.config.patch_size
#         w0 = width // self.config.patch_size
#         # we add a small number to avoid floating point error in the interpolation
#         # see discussion at https://github.com/facebookresearch/dino/issues/8
#         h0, w0 = h0 + 0.1, w0 + 0.1
#         patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
#         patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed,
#             scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
#             mode="bicubic",
#             align_corners=False,
#         )
#         assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#         return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

#     def forward(
#         self,
#         pixel_values: torch.Tensor,
#         bool_masked_pos: Optional[torch.BoolTensor] = None,
#         interpolate_pos_encoding: bool = False,
#     ) -> torch.Tensor:
#         batch_size, num_channels, height, width = pixel_values.shape
#         embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

#         if bool_masked_pos is not None:
#             seq_length = embeddings.shape[1]
#             mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
#             # replace the masked visual tokens by mask_tokens
#             mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
#             embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

#         # add the [CLS] token to the embedded patch tokens
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         embeddings = torch.cat((cls_tokens, embeddings), dim=1)

#         # add positional encoding to each token
#         if interpolate_pos_encoding:
#             embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
#         else:
#             embeddings = embeddings + self.position_embeddings

#         embeddings = self.dropout(embeddings)

#         return embeddings


# # Copied from transformers.models.vit.modeling_vit.ViTPatchEmbeddings with ViT->ProPainter
# class ProPainterPatchEmbeddings(nn.Module):
#     """
#     This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
#     `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
#     Transformer.
#     """

#     def __init__(self, config):
#         super().__init__()
#         image_size, patch_size = config.image_size, config.patch_size
#         num_channels, hidden_size = config.num_channels, config.hidden_size

#         image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
#         patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
#         num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.num_channels = num_channels
#         self.num_patches = num_patches

#         self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

#     def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
#         batch_size, num_channels, height, width = pixel_values.shape
#         if num_channels != self.num_channels:
#             raise ValueError(
#                 "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
#                 f" Expected {self.num_channels} but got {num_channels}."
#             )
#         if not interpolate_pos_encoding:
#             if height != self.image_size[0] or width != self.image_size[1]:
#                 raise ValueError(
#                     f"Input image size ({height}*{width}) doesn't match model"
#                     f" ({self.image_size[0]}*{self.image_size[1]})."
#                 )
#         embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
#         return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->ProPainter
class ProPainterSelfAttention(nn.Module):
    def __init__(self, config: ProPainterConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->ProPainter
class ProPainterAttention(nn.Module):
    def __init__(self, config: ProPainterConfig) -> None:
        super().__init__()
        self.attention = ProPainterSelfAttention(config)
        # self.output = ProPainterSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        # self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # attention_output = self.output(self_outputs[0], hidden_states)

        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return None



# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->ProPainter
class ProPainterIntermediate(nn.Module):
    def __init__(self, config: ProPainterConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->ProPainter
class ProPainterOutput(nn.Module):
    def __init__(self, config: ProPainterConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with VIT->PROPAINTER,ViT->ProPainter
class ProPainterLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ProPainterConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # self.attention = PROPAINTER_ATTENTION_CLASSES[config._attn_implementation](config)
        self.intermediate = ProPainterIntermediate(config)
        self.output = ProPainterOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ProPainter, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ProPainter, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->ProPainter
# class ProPainterEncoder(nn.Module):
#     def __init__(self, config: ProPainterConfig) -> None:
#         super().__init__()
#         self.config = config
#         self.layer = nn.ModuleList([ProPainterLayer(config) for _ in range(config.num_hidden_layers)])
#         self.gradient_checkpointing = False

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#     ) -> Union[tuple, BaseModelOutput]:
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None

#         for i, layer_module in enumerate(self.layer):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_head_mask = head_mask[i] if head_mask is not None else None

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     layer_module.__call__,
#                     hidden_states,
#                     layer_head_mask,
#                     output_attentions,
#                 )
#             else:
#                 layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

#             hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         )

############################***************####################ALL MODULES TO ADDDDDDDDDDDDDd################**********##########################################

##################################################ProPainterRaftOpticalFlow MODULES STARTS HERE######################################################


#made transformers compliant
class ProPainterResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, norm_fn='group', stride=1):
        super(ProPainterResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = channels // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(channels)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(channels)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride), self.norm3)


    def forward(self, hidden_states):
        residual = hidden_states
        residual = self.relu(self.norm1(self.conv1(residual)))
        residual = self.relu(self.norm2(self.conv2(residual)))

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.relu(hidden_states+residual)
        return hidden_states

class ProPainterBasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(ProPainterBasicEncoder, self).__init__()

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        in_channels = (64,64,96)
        channels = (64,96,128)
        strides = (1,2,2)

        self.resblocks = [[ProPainterResidualBlock(in_channel, channel, norm_fn, stride),ProPainterResidualBlock(channel, channel, norm_fn, stride=1)] for in_channel,channel,stride in zip(in_channels,channels,strides)]
        #using itertools makes flattening a little faster :)
        self.resblocks = nn.ModuleList(list(itertools.chain.from_iterable(self.resblocks))) 

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image):

        # if input is list, combine batch dimension
        is_list = isinstance(image, tuple) or isinstance(image, list)
        if is_list:
            batch_dim = image[0].shape[0]
            image = torch.cat(image, dim=0)

        hidden_states = self.conv1(image)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.relu1(hidden_states)

        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.training and self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        if is_list:
            hidden_states = torch.split(hidden_states, [batch_dim, batch_dim], dim=0)

        return hidden_states

class ProPainterBasicMotionEncoder(nn.Module):
    def __init__(self, config):
        super(ProPainterBasicMotionEncoder, self).__init__()
        corr_planes = config.corr_levels * (2*config.corr_radius + 1)**2
        self.conv_corr1 = nn.Conv2d(corr_planes, 256, 1, padding=0)
        self.conv_corr2 = nn.Conv2d(256, 192, 3, padding=1)
        self.conv_flow1 = nn.Conv2d(2, 128, 7, padding=3)
        self.conv_flow2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        hidden_states_corr = F.relu(self.conv_corr1(corr))
        hidden_states_corr = F.relu(self.conv_corr2(hidden_states_corr))
        hidden_states_flow = F.relu(self.conv_flow1(flow))
        hidden_states_flow = F.relu(self.conv_flow2(hidden_states_flow))

        hidden_states = torch.cat([hidden_states_corr, hidden_states_flow], dim=1)
        hidden_states = F.relu(self.conv(hidden_states))
        return torch.cat([hidden_states, flow], dim=1)

class ProPainterSepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ProPainterSepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, hidden_states, motion_features):
        # horizontal
        hidden_states_motion_features = torch.cat([hidden_states, motion_features], dim=1)
        z = torch.sigmoid(self.convz1(hidden_states_motion_features))
        r = torch.sigmoid(self.convr1(hidden_states_motion_features))
        q = torch.tanh(self.convq1(torch.cat([r*hidden_states, motion_features], dim=1)))        
        hidden_states = (1-z) * hidden_states + z * q

        # vertical
        hidden_states_motion_features = torch.cat([hidden_states, motion_features], dim=1)
        z = torch.sigmoid(self.convz2(hidden_states_motion_features))
        r = torch.sigmoid(self.convr2(hidden_states_motion_features))
        q = torch.tanh(self.convq2(torch.cat([r*hidden_states, motion_features], dim=1)))       
        hidden_states = (1-z) * hidden_states + z * q

        return hidden_states

class ProPainterFlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(ProPainterFlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hidden_states):
        return self.conv2(self.relu(self.conv1(hidden_states)))

class ProPainterBasicUpdateBlock(nn.Module):
    def __init__(self, config, hidden_dim=128, input_dim=128):
        super(ProPainterBasicUpdateBlock, self).__init__()
        self.config = config
        self.encoder = ProPainterBasicMotionEncoder(config)
        self.gru = ProPainterSepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = ProPainterFlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


def coords_grid(batch_size, height, width):
    coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)


def sample_point(img, coords):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img


class ProPainterCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = ProPainterCorrBlock.corr(fmap1, fmap2)

        batch_size, height1, width1, dimension, height2, width2 = corr.shape
        corr = corr.reshape(batch_size*height1*width1, dimension, height2, width2)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        radius = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch_size, height1, width1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-radius, radius, 2*radius+1)
            dy = torch.linspace(-radius, radius, 2*radius+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch_size*height1*width1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*radius+1, 2*radius+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = sample_point(corr, coords_lvl)
            corr = corr.view(batch_size, height1, width1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch_size, dimension, height, width = fmap1.shape
        fmap1 = fmap1.view(batch_size, dimension, height*width)
        fmap2 = fmap2.view(batch_size, dimension, height*width)

        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch_size, height, width, 1, height, width)
        return corr  / torch.sqrt(torch.tensor(dimension).float())


class ProPainterRaftOpticalFlow(nn.Module):
    def __init__(self, config):
        super(ProPainterRaftOpticalFlow, self).__init__()
        self.config = config
        self.hidden_dim = 128
        self.context_dim = 128
        
        self.feature_network = ProPainterBasicEncoder(output_dim=256, norm_fn='instance', dropout=self.config.dropout)
        self.context_network = ProPainterBasicEncoder(output_dim=self.hidden_dim+self.context_dim, norm_fn='batch', dropout=self.config.dropout)
        self.update_block = ProPainterBasicUpdateBlock(self.config, hidden_dim=self.hidden_dim)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, image):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = image.shape
        coords0 = coords_grid(N, H//8, W//8).to(image.device)
        coords1 = coords_grid(N, H//8, W//8).to(image.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def _forward(self, image1, image2, iters=12, flow_init=None):
        """ Estimate optical flow between pair of frames """

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        with autocast(enabled=False):
            fmap1, fmap2 = self.feature_network([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        
        corr_fn = ProPainterCorrBlock(fmap1, fmap2, radius=self.config.corr_radius)

        # run the context network
        with autocast(enabled=False):
            context_network_out = self.context_network(image1)
            net, inp = torch.split(context_network_out, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=False):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                new_size = (8 * (coords1 - coords0).shape[2], 8 * (coords1 - coords0).shape[3])
                flow_up =   8 * F.interpolate((coords1 - coords0), size=new_size, mode='bilinear', align_corners=True)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)


        return coords1 - coords0, flow_up

    def forward(self, gt_local_frames, iters=20):
        batch_size, temporal_length, channel, height, width = gt_local_frames.size()

        gt_local_frames_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, channel, height, width)
        gt_local_frames_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, channel, height, width)

        _, gt_flows_forward = self._forward(gt_local_frames_1, gt_local_frames_2, iters=iters)
        _, gt_flows_backward = self._forward(gt_local_frames_2, gt_local_frames_1, iters=iters)

        
        gt_flows_forward = gt_flows_forward.view(batch_size, temporal_length-1, 2, height, width)
        gt_flows_backward = gt_flows_backward.view(batch_size, temporal_length-1, 2, height, width)

        return gt_flows_forward, gt_flows_backward


##################################################ProPainterRaftOpticalFlow MODULES ENDS HERE######################################################

class ProPainterP3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_residual=0, bias=True):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size),
                                    stride=(1, stride, stride), padding=(0, padding, padding), bias=bias),
                        nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                    padding=(2, 0, 0), dilation=(2, 1, 1), bias=bias)
        )
        self.use_residual = use_residual

    def forward(self, hidden_states):
        features1 = self.conv1(hidden_states)
        features2 = self.conv2(features1)
        if self.use_residual:
            hidden_states = hidden_states + features2
        else:
            hidden_states = features2
        return hidden_states


class ProPainterEdgeDetection(nn.Module):
    def __init__(self, in_channel=2, out_channel=1, intermediate_channel=16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channel, intermediate_channel, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.intermediate_layer_1 = nn.Sequential(
            nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.intermediate_layer_2 = nn.Sequential(
            nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1)
        )        

        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self.out_layer = nn.Conv2d(intermediate_channel, out_channel, 1, 1, 0)

    def forward(self, flow):
        flow = self.projection(flow)
        edge = self.intermediate_layer_1(flow)
        edge = self.intermediate_layer_2(edge)
        edge = self.relu(flow + edge)
        edge = self.out_layer(edge)
        edge = torch.sigmoid(edge)
        return edge

class ProPainterBidirectionalPropagationFlowComplete(nn.Module):
    def __init__(self, channel):
        super(ProPainterBidirectionalPropagationFlowComplete, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        for i, module in enumerate(modules):
            self.deform_align[module] = ProPainterSecondOrderDeformableAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16)

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, hidden_states):
        """
        hidden_states shape : [batch_size, timesteps, channel, height, width]
        return [batch_size, timesteps, channel, height, width]
        """
        batch_size, timesteps, _, height, width = hidden_states.shape
        features = {}
        features['spatial'] = [hidden_states[:, i, :, :, :] for i in range(0, timesteps)]

        for module_name in ['backward_', 'forward_']:

            features[module_name] = []

            frame_idx = range(0, timesteps)
            mapping_idx = list(range(0, len(features['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]

            feature_propagation = hidden_states.new_zeros(batch_size, self.channel, height, width)
            for i, idx in enumerate(frame_idx):
                feat_current = features['spatial'][mapping_idx[idx]]
                if i > 0:
                    cond_n1 = feature_propagation

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feature_propagation)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:  # second-order features
                        feat_n2 = features[module_name][-2]
                        cond_n2 = feat_n2

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1) # condition information, cond(flow warped 1st/2nd feature)
                    feature_propagation = torch.cat([feature_propagation, feat_n2], dim=1) # two order feature_propagation -1 & -2
                    feature_propagation = self.deform_align[module_name](feature_propagation, cond)

                # fuse current features
                feat = [feat_current] + \
                    [features[k][idx] for k in features if k not in ['spatial', module_name]] \
                    + [feature_propagation]

                feat = torch.cat(feat, dim=1)
                # embed current features
                feature_propagation = feature_propagation + self.backbone[module_name](feat)

                features[module_name].append(feature_propagation)

            # end for
            if 'backward' in module_name:
                features[module_name] = features[module_name][::-1]

        outputs = []
        for i in range(0, timesteps):
            align_feats = [features[k].pop(0) for k in features if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + hidden_states

def flow_warp(features,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        features (Tensor): Tensor with size (n, c, height, width).
        flow (Tensor): Tensor with size (n, height, width, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if features.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({features.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, height, width = features.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height, device=device), torch.arange(0, width, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(features)  # (width, height, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(width - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(height - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(features,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output

def fbConsistencyCheck(flow_forward, flow_backward, alpha1=0.01, alpha2=0.5):
    flow_backward_warped = flow_warp(flow_backward, flow_forward.permute(0, 2, 3, 1))
    flow_diff_forward = flow_forward + flow_backward_warped

    normalized_forward = torch.norm(flow_forward, p=2, dim=1, keepdim=True)**2 + torch.norm(flow_backward_warped, p=2, dim=1, keepdim=True)**2  # |wf| + |wb(wf(x))|
    occ_thresh_forward = alpha1 * normalized_forward + alpha2

    fb_valid_forward = (torch.norm(flow_diff_forward, p=2, dim=1, keepdim=True)**2 < occ_thresh_forward).to(flow_forward)
    return fb_valid_forward


class ProPainterBidirectionalPropagationInPaint(nn.Module):
    def __init__(self, channel, learnable=True):
        super(ProPainterBidirectionalPropagationInPaint, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.propagation_list = ['backward_1', 'forward_1']
        self.learnable = learnable

        if self.learnable:
            for _, module in enumerate(self.propagation_list):
                self.deform_align[module] = ProPainterDeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2*channel+2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                    nn.Conv2d(2*channel+2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                ) 
            
    def binary_mask(self, mask, th=0.1):
        return torch.where(mask > th, 1, 0).to(mask)

    def forward(self, masked_frames, flows_forward, flows_backward, mask, interpolation='bilinear'):
        """
        masked_frames shape : [batch_size, timesteps, channel, height, width]
        return [batch_size, timesteps, channel, height, width]
        """

        # For backward warping, pred_flows_forward for backward feature propagation, pred_flows_backward for forward feature propagation
        batch_size, timesteps, channel, height, width = masked_frames.shape
        features, masks = {}, {}
        features['input'] = [masked_frames[:, i, :, :, :] for i in range(0, timesteps)]
        masks['input'] = [mask[:, i, :, :, :] for i in range(0, timesteps)]

        propagation_list = ['backward_1', 'forward_1']
        cache_list = ['input'] +  propagation_list

        for p_i, module_name in enumerate(propagation_list):
            features[module_name] = []
            masks[module_name] = []

            if 'backward' in module_name:
                frame_idx = range(0, timesteps)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, timesteps)
                flow_idx = range(-1, timesteps - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = features[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask, mask_current], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                        mask_prop_valid = self.binary_mask(mask_prop_valid)

                        union_vaild_mask = self.binary_mask(mask_current*flow_vaild_mask*(1-mask_prop_valid))
                        feat_prop = union_vaild_mask * feat_warped + (1-union_vaild_mask) * feat_current
                        # update mask
                        mask_prop = self.binary_mask(mask_current*(1-(flow_vaild_mask*(1-mask_prop_valid))))
                
                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)

                features[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if 'backward' in module_name:
                features[module_name] = features[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(features['backward_1'], dim=1).view(-1, channel, height, width)
        outputs_f = torch.stack(features['forward_1'], dim=1).view(-1, channel, height, width)

        if self.learnable:
            mask_in = mask.view(-1, 2, height, width)
            masks_f = None
            outputs = self.fuse(torch.cat([outputs_b, outputs_f, mask_in], dim=1)) + masked_frames.view(-1, channel, height, width)
        else:
            masks_f = torch.stack(masks['forward_1'], dim=1)
            outputs = outputs_f

        return outputs_b.view(batch_size, -1, channel, height, width), outputs_f.view(batch_size, -1, channel, height, width), \
               outputs.view(batch_size, -1, channel, height, width), masks_f

class ProPainterDeconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(hidden_states)

class ProPainterModulatedDeformConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
        super(ProPainterModulatedDeformConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, hidden_states, offset, mask):
        pass

class ProPainterDeformableAlignment(ProPainterModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)

        super(ProPainterDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

    def forward(self, features_propagation, cond_features, flow):
        output = self.conv_offset(cond_features)
        output1, output2, mask = torch.chunk(output, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((output1, output2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(features_propagation, offset, self.weight, self.bias, 
                                             self.stride, self.padding,
                                             self.dilation, mask)

class ProPainterSecondOrderDeformableAlignment(ProPainterModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 5)

        super(ProPainterSecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

    def forward(self, features, extra_features):
        output = self.conv_offset(extra_features)
        output1, output2, mask = torch.chunk(output, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((output1, output2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(features, offset, self.weight, self.bias, 
                                             self.stride, self.padding,
                                             self.dilation, mask)


class ProPainterRecurrentFlowCompleteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
                        nn.Conv3d(3, 32, kernel_size=(1, 5, 5), stride=(1, 2, 2), 
                                        padding=(0, 2, 2), padding_mode='replicate'),
                        nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder1 = nn.Sequential(
            ProPainterP3DBlock(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterP3DBlock(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 4x

        self.encoder2 = nn.Sequential(
            ProPainterP3DBlock(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterP3DBlock(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 8x

        self.intermediate_dilation = nn.Sequential(
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 3, 3), dilation=(1, 3, 3)), # p = d*(k-1)/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # feature propagation module
        self.feature_propagation_module = ProPainterBidirectionalPropagationFlowComplete(128)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterDeconv(128, 64, 3, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 4x

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterDeconv(64, 32, 3, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 2x

        self.upsample = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterDeconv(32, 2, 3, 1)
        )

        # edge loss
        self.edgeDetector = ProPainterEdgeDetection(in_channel=2, out_channel=1, intermediate_channel=16)

    def forward(self, masked_flows, masks):
        batch_size, timesteps, _, height, width = masked_flows.size()
        masked_flows = masked_flows.permute(0,2,1,3,4)
        masks = masks.permute(0,2,1,3,4)

        inputs = torch.cat((masked_flows, masks), dim=1)
        
        downsample_inputs = self.downsample(inputs)

        features_enc1 = self.encoder1(downsample_inputs)
        features_enc2 = self.encoder2(features_enc1) # batch_size channel timesteps height width
        features_intermediate = self.intermediate_dilation(features_enc2) # batch_size channel timesteps height width
        features_intermediate = features_intermediate.permute(0,2,1,3,4) # batch_size timesteps channel height width

        features_prop = self.feature_propagation_module(features_intermediate)
        features_prop = features_prop.view(-1, 128, height//8, width//8) # batch_size*timesteps channel height width

        _, c, _, h_f, w_f = features_enc1.shape
        features_enc1 = features_enc1.permute(0,2,1,3,4).contiguous().view(-1, c, h_f, w_f) # batch_size*timesteps channel height width
        features_dec2 = self.decoder2(features_prop) + features_enc1

        _, c, _, h_f, w_f = downsample_inputs.shape
        downsample_inputs = downsample_inputs.permute(0,2,1,3,4).contiguous().view(-1, c, h_f, w_f) # batch_size*timesteps channel height width

        features_dec1 = self.decoder1(features_dec2)

        flow = self.upsample(features_dec1)
        if self.training:
            edge = self.edgeDetector(flow)
            edge = edge.view(batch_size, timesteps, 1, height, width)
        else:
            edge = None

        flow = flow.view(batch_size, timesteps, 2, height, width)

        return flow, edge
        

    def forward_bidirect_flow(self, masked_flows_bi, masks):
        """
        Args:
            masked_flows_bi: [masked_flows_f, masked_flows_b] | (batch_size, timesteps-1, 2, height, width), (batch_size, timesteps-1, 2, height, width)
            masks: batch_size, timesteps, 1, height, width
        """
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        # mask flow
        masked_flows_forward = masked_flows_bi[0] * (1-masks_forward)
        masked_flows_backward = masked_flows_bi[1] * (1-masks_backward)
        
        # -- completion --
        pred_flows_forward, pred_edges_forward = self.forward(masked_flows_forward, masks_forward)

        # backward
        masked_flows_backward = torch.flip(masked_flows_backward, dims=[1])
        masks_backward = torch.flip(masks_backward, dims=[1])
        pred_flows_backward, pred_edges_backward = self.forward(masked_flows_backward, masks_backward)
        pred_flows_backward = torch.flip(pred_flows_backward, dims=[1])
        if self.training:
            pred_edges_backward = torch.flip(pred_edges_backward, dims=[1])

        return [pred_flows_forward, pred_flows_backward], [pred_edges_forward, pred_edges_backward]


    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1-masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1-masks_backward)

        return pred_flows_forward, pred_flows_backward

#########################################################RECURRENT FLOW NETWORK FINISH#####################################################

# class ProPainterBaseNetwork(nn.Module):
#     def __init__(self):
#         super(ProPainterBaseNetwork, self).__init__()

    # def print_network(self):
    #     if isinstance(self, list):
    #         self = self[0]
    #     num_params = 0
    #     for param in self.parameters():
    #         num_params += param.numel()
    #     print(
    #         'Network [%s] was created. Total number of parameters: %.1f million. '
    #         'To see the architecture, do print(network).' %
    #         (type(self).__name__, num_params / 1000000))

    # def init_weights(self, init_type='normal', gain=0.02):
    #     '''
    #     initialize network's weights
    #     init_type: normal | xavier | kaiming | orthogonal
    #     https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    #     '''
    #     def init_func(m):
    #         classname = m.__class__.__name__
    #         if classname.find('InstanceNorm2d') != -1:
    #             if hasattr(m, 'weight') and m.weight is not None:
    #                 nn.init.constant_(m.weight.data, 1.0)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 nn.init.constant_(m.bias.data, 0.0)
    #         elif hasattr(m, 'weight') and (classname.find('Conv') != -1
    #                                        or classname.find('Linear') != -1):
    #             if init_type == 'normal':
    #                 nn.init.normal_(m.weight.data, 0.0, gain)
    #             elif init_type == 'xavier':
    #                 nn.init.xavier_normal_(m.weight.data, gain=gain)
    #             elif init_type == 'xavier_uniform':
    #                 nn.init.xavier_uniform_(m.weight.data, gain=1.0)
    #             elif init_type == 'kaiming':
    #                 nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 nn.init.orthogonal_(m.weight.data, gain=gain)
    #             elif init_type == 'none':  # uses pytorch's default init method
    #                 m.reset_parameters()
    #             else:
    #                 raise NotImplementedError(
    #                     'initialization method [%s] is not implemented' %
    #                     init_type)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 nn.init.constant_(m.bias.data, 0.0)

    #     self.apply(init_func)

    #     # propagate to children
    #     for m in self.children():
    #         if hasattr(m, 'init_weights'):
    #             m.init_weights(init_type, gain)


class ProPainterEncoder(nn.Module):
    def __init__(self):
        super(ProPainterEncoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        negative_slope = 0.2
        self.layers = nn.ModuleList([
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(negative_slope, inplace=True)
        ])

    def forward(self, masked_inputs):
        batch_size, _, _, _ = masked_inputs.size()
        features = masked_inputs
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = features
                _, _, height, width = x0.size()
            if i > 8 and i % 2 == 0:
                group = self.group[(i - 8) // 2]
                masked_inputs = x0.view(batch_size, group, -1, height, width)
                feature = features.view(batch_size, group, -1, height, width)
                features = torch.cat([masked_inputs, feature], 2).view(batch_size, -1, height, width)
            features = layer(features)
        return features

class ProPainterSoftSplit(nn.Module):
    def __init__(self, channel, hidden_size, kernel_size, stride, padding):
        super(ProPainterSoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        input_features = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(input_features, hidden_size)

    def forward(self, hidden_states, batch_size, output_size):
        features_height = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        features_width = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.embedding(hidden_states)
        hidden_states = hidden_states.view(batch_size, -1, features_height, features_width, hidden_states.size(2))
        return hidden_states

class ProPainterSoftComp(nn.Module):
    def __init__(self, channel, hidden_size, kernel_size, stride, padding):
        super(ProPainterSoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        output_features = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden_size, output_features)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, hidden_states, timestep, output_size):
        num_batch_, _, _, _, channel_ = hidden_states.shape
        hidden_states = hidden_states.view(num_batch_, -1, channel_)
        hidden_states = self.embedding(hidden_states)
        batch_size, _, channel = hidden_states.size()
        hidden_states = hidden_states.view(batch_size * timestep, -1, channel).permute(0, 2, 1)
        hidden_states = F.fold(hidden_states,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        hidden_states = self.bias_conv(hidden_states)
        return hidden_states

# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size, num_head):
    """
    Args:
        input_feature: shape is (batch_size, timesteps, height, width, num_channels)
        window_size (tuple[int]): window size
    Returns:
        windows: (batch_size, num_windows_h, num_windows_w, num_head, timesteps, window_size, window_size, num_channels//num_head)
    """
    batch_size, timesteps, height, width, num_channels= input_feature.shape
    input_feature = input_feature.view(
        batch_size, timesteps, height // window_size[0], window_size[0], width // window_size[1], window_size[1], num_head, num_channels//num_head)
    windows = input_feature.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows


class ProPainterSparseWindowAttention(nn.Module):
    def __init__(self, dimension, num_head, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0., 
                pooling_token=True):
        super().__init__()
        assert dimension % num_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dimension, dimension, qkv_bias)
        self.query = nn.Linear(dimension, dimension, qkv_bias)
        self.value = nn.Linear(dimension, dimension, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(dimension, dimension)
        self.num_head = num_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            kernel_size, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dimension, dimension, kernel_size=kernel_size, stride=stride, padding=(0, 0), groups=dimension)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        # self.expand_size = tuple(i // 2 for i in window_size)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))


    def forward(self, hidden_states, mask=None, token_indices=None):
        batch_size, timesteps, height, width, num_channels = hidden_states.shape # 20 36
        window_height, window_width = self.window_size[0], self.window_size[1]
        channel_head = num_channels // self.num_head
        n_window_height = math.ceil(height / self.window_size[0])
        n_window_width = math.ceil(width / self.window_size[1])
        new_height = n_window_height * self.window_size[0] # 20
        new_width = n_window_width * self.window_size[1] # 36
        pad_r = new_width - width
        pad_b = new_height - height
        # reverse order
        if pad_r > 0 or pad_b > 0:
            hidden_states = F.pad(hidden_states,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 
            mask = F.pad(mask,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dimension
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        window_query = window_partition(query.contiguous(), self.window_size, self.num_head).view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, window_height*window_width, channel_head)
        window_key = window_partition(key.contiguous(), self.window_size, self.num_head).view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, window_height*window_width, channel_head)
        window_value = window_partition(value.contiguous(), self.window_size, self.num_head).view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, window_height*window_width, channel_head)
        # roll_k and roll_v
        if any(i > 0 for i in self.expand_size):
            (key_top_left, value_top_left) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (key, value))
            (key_top_right, value_top_right) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3)), (key, value))
            (key_bottom_left, value_bottom_left) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (key, value))
            (key_bottom_right, value_bottom_right) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3)), (key, value))

            (key_top_left_windows, key_top_right_windows, key_bottom_left_windows, key_bottom_right_windows) = map(
                lambda a: window_partition(a, self.window_size, self.num_head).view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, window_height*window_width, channel_head), 
                (key_top_left, key_top_right, key_bottom_left, key_bottom_right))
            (value_top_left_windows, value_top_right_windows, value_bottom_left_windows, value_bottom_right_windows) = map(
                lambda a: window_partition(a, self.window_size, self.num_head).view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, window_height*window_width, channel_head), 
                (value_top_left, value_top_right, value_bottom_left, value_bottom_right))
            rool_key = torch.cat((key_top_left_windows, key_top_right_windows, key_bottom_left_windows, key_bottom_right_windows), 4).contiguous()
            rool_value = torch.cat((value_top_left_windows, value_top_right_windows, value_bottom_left_windows, value_bottom_right_windows), 4).contiguous() # [batch_size, n_window_height*n_window_width, num_head, timesteps, window_height*window_width, channel_head]
            # mask output tokens in current window
            rool_key = rool_key[:, :, :, :, self.valid_ind_rolled]
            rool_value = rool_value[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_key.shape[4]
            rool_key = rool_key.view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, roll_N, num_channels // self.num_head)
            rool_value = rool_value.view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, roll_N, num_channels // self.num_head)
            window_key = torch.cat((window_key, rool_key), dim=4)
            window_value = torch.cat((window_value, rool_value), dim=4)
        else:
            window_key = window_key
            window_value = window_value
        
        # pool_k and pool_v
        if self.pooling_token:
            pool_x = self.pool_layer(hidden_states.view(batch_size*timesteps, new_height, new_width, num_channels).permute(0,3,1,2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0,2,3,1).view(batch_size, timesteps, p_h, p_w, num_channels)
            # pool_k
            pool_k = self.key(pool_x).unsqueeze(1).repeat(1, n_window_height*n_window_width, 1, 1, 1, 1) # [batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, num_channels]
            pool_k = pool_k.view(batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, self.num_head, channel_head).permute(0,1,5,2,3,4,6)
            pool_k = pool_k.contiguous().view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, p_h*p_w, channel_head)
            window_key = torch.cat((window_key, pool_k), dim=4)
            # pool_v
            pool_v = self.value(pool_x).unsqueeze(1).repeat(1, n_window_height*n_window_width, 1, 1, 1, 1) # [batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, num_channels]
            pool_v = pool_v.view(batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, self.num_head, channel_head).permute(0,1,5,2,3,4,6)
            pool_v = pool_v.contiguous().view(batch_size, n_window_height*n_window_width, self.num_head, timesteps, p_h*p_w, channel_head)
            window_value = torch.cat((window_value, pool_v), dim=4)

        # [batch_size, n_window_height*n_window_width, num_head, timesteps, window_height*window_width, channel_head]
        output = torch.zeros_like(window_query)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(batch_size * l_t, new_height, new_width))
        mask = mask.view(batch_size, l_t, n_window_height*n_window_width)
        mask = torch.sum(mask, dim=1) # [batch_size, n_window_height*n_window_width]
        for i in range(window_query.shape[0]):
            ### For masked windows
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            # mask output quary in current window
            # [batch_size, n_window_height*n_window_width, num_head, timesteps, window_height*window_width, channel_head]
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                window_query_t = window_query[i, mask_ind_i].view(mask_n, self.num_head, timesteps*window_height*window_width, channel_head)
                window_key_t = window_key[i, mask_ind_i] 
                window_value_t = window_value[i, mask_ind_i] 
                # mask output key and value
                if token_indices is not None:
                    # key [n_window_height*n_window_width, num_head, timesteps, window_height*window_width, channel_head]
                    window_key_t = window_key_t[:, :, token_indices.view(-1)].view(mask_n, self.num_head, -1, channel_head)
                    # value
                    window_value_t = window_value_t[:, :, token_indices.view(-1)].view(mask_n, self.num_head, -1, channel_head)
                else:
                    window_key_t = window_key_t.view(n_window_height*n_window_width, self.num_head, timesteps*window_height*window_width, channel_head)
                    window_value_t = window_value_t.view(n_window_height*n_window_width, self.num_head, timesteps*window_height*window_width, channel_head)

                att_t = (window_query_t @ window_key_t.transpose(-2, -1)) * (1.0 / math.sqrt(window_query_t.size(-1)))
                att_t = F.softmax(att_t, dim=-1)
                att_t = self.attn_drop(att_t)
                y_t = att_t @ window_value_t 
                
                output[i, mask_ind_i] = y_t.view(-1, self.num_head, timesteps, window_height*window_width, channel_head)

            ### For unmasked windows
            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            # mask output quary in current window
            # [batch_size, n_window_height*n_window_width, num_head, timesteps, window_height*window_width, channel_head]
            window_query_s = window_query[i, unmask_ind_i]
            window_key_s = window_key[i, unmask_ind_i, :, :, :window_height*window_width]
            window_value_s = window_value[i, unmask_ind_i, :, :, :window_height*window_width]

            att_s = (window_query_s @ window_key_s.transpose(-2, -1)) * (1.0 / math.sqrt(window_query_s.size(-1)))
            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            y_s = att_s @ window_value_s
            output[i, unmask_ind_i] = y_s

        # re-assemble all head outputs side by side
        output = output.view(batch_size, n_window_height, n_window_width, self.num_head, timesteps, window_height, window_width, channel_head)
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(batch_size, timesteps, new_height, new_width, num_channels)


        if pad_r > 0 or pad_b > 0:
            output = output[:, :, :height, :width, :]

        # output projection
        output = self.proj_drop(self.proj(output))
        return output

class ProPainterFusionFeedForward(nn.Module):
    def __init__(self, dimension, hidden_dim=1960, token_to_token_params=None):
        super(ProPainterFusionFeedForward, self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(dimension, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dimension))
        assert token_to_token_params is not None
        self.token_to_token_params = token_to_token_params
        self.kernel_shape = reduce((lambda x, y: x * y), token_to_token_params['kernel_size']) # 49

    def forward(self, hidden_states, output_size):
        num_vecs = 1
        for i, d in enumerate(self.token_to_token_params['kernel_size']):
            num_vecs *= int((output_size[i] + 2 * self.token_to_token_params['padding'][i] -
                           (d - 1) - 1) / self.token_to_token_params['stride'][i] + 1)

        hidden_states = self.fc1(hidden_states)
        batch_size, timestep, num_channel = hidden_states.size()
        normalizer = hidden_states.new_ones(batch_size, timestep, self.kernel_shape).view(-1, num_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.token_to_token_params['kernel_size'],
                            padding=self.token_to_token_params['padding'],
                            stride=self.token_to_token_params['stride'])

        hidden_states = F.fold(hidden_states.view(-1, num_vecs, num_channel).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.token_to_token_params['kernel_size'],
                   padding=self.token_to_token_params['padding'],
                   stride=self.token_to_token_params['stride'])

        hidden_states = F.unfold(hidden_states / normalizer,
                     kernel_size=self.token_to_token_params['kernel_size'],
                     padding=self.token_to_token_params['padding'],
                     stride=self.token_to_token_params['stride']).permute(
                         0, 2, 1).contiguous().view(batch_size, timestep, num_channel)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class ProPainterTemporalSparseTransformer(nn.Module):
    def __init__(self, dimension, num_head, window_size, pool_size,
                layer_norm=nn.LayerNorm, token_to_token_params=None):
        super().__init__()
        self.window_size = window_size
        self.attention = ProPainterSparseWindowAttention(dimension, num_head, window_size, pool_size)
        self.layer_norm1 = layer_norm(dimension)
        self.layer_norm2 = layer_norm(dimension)
        self.mlp = ProPainterFusionFeedForward(dimension, token_to_token_params=token_to_token_params)

    def forward(self, image_tokens, fold_x_size, mask=None, token_indices=None):
        """
        Args:
            image_tokens: shape [batch_size, timesteps, height, width, channel]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [batch_size, timesteps, height, width, 1]
        Returns:
            out_tokens: shape [batch_size, timesteps, height, width, 1]
        """
        batch_size, timesteps, height, width, channel = image_tokens.shape # 20 36

        shortcut = image_tokens
        image_tokens = self.layer_norm1(image_tokens)
        att_x = self.attention(image_tokens, mask, token_indices)

        # FFN
        image_tokens = shortcut + att_x
        y = self.layer_norm2(image_tokens)
        image_tokens = image_tokens + self.mlp(y.view(batch_size, timesteps * height * width, channel), fold_x_size).view(batch_size, timesteps, height, width, channel)

        return image_tokens

class ProPainterTemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dimension, num_head, window_size, pool_size, depths, token_to_token_params=None):
        super().__init__()
        blocks = []
        for _ in range(depths):
             blocks.append(
                ProPainterTemporalSparseTransformer(dimension, num_head, window_size, pool_size, token_to_token_params=token_to_token_params)
             )
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths

    def forward(self, image_tokens, fold_x_size, local_mask=None, t_dilation=2):
        """
        Args:
            image_tokens: shape [batch_size, timesteps, height, width, channel]
            fold_x_size: fold feature size, shape [60 108]
            local_mask: local mask tokens, shape [batch_size, timesteps, height, width, 1]
        Returns:
            out_tokens: shape [batch_size, timesteps, height, width, channel]
        """
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        timesteps = image_tokens.size(1)
        token_indices = [torch.arange(i, timesteps, t_dilation) for i in range(t_dilation)] * (self.depths // t_dilation)

        for i in range(0, self.depths):
            image_tokens = self.transformer[i](image_tokens, fold_x_size, local_mask, token_indices[i])

        return image_tokens

class ProPainterInpaintGenerator(nn.Module):
    def __init__(self):
        super(ProPainterInpaintGenerator, self).__init__()
        channel = 128
        hidden_size = 512

        # ProPainterEncoder
        self.encoder = ProPainterEncoder()

        # decoder
        self.decoder = nn.Sequential(
            ProPainterDeconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterDeconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        token_to_token_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.soft_split = ProPainterSoftSplit(channel, hidden_size, kernel_size, stride, padding)
        self.soft_comp = ProPainterSoftComp(channel, hidden_size, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        # feature propagation module
        self.img_prop_module = ProPainterBidirectionalPropagationInPaint(3, learnable=False)
        self.feature_propagation_module = ProPainterBidirectionalPropagationInPaint(128, learnable=True)
        
        
        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = ProPainterTemporalSparseTransformerBlock(dimension=hidden_size,
                                                num_head=num_heads,
                                                window_size=window_size,
                                                pool_size=pool_size,
                                                depths=depths,
                                                token_to_token_params=token_to_token_params)

    def img_propagation(self, masked_frames, completed_flows, masks, interpolation='nearest'):
        _, _, prop_frames, updated_masks = self.img_prop_module(masked_frames, completed_flows[0], completed_flows[1], masks, interpolation)
        return prop_frames, updated_masks

    def forward(self, masked_frames, completed_flows, masks_in, masks_updated, num_local_frames, interpolation='bilinear', t_dilation=2):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """

        local_timestep = num_local_frames
        batch_size, timestep, _, original_height, original_width = masked_frames.size()

        # extracting features
        encoder_hidden_states = self.encoder(torch.cat([masked_frames.view(batch_size * timestep, 3, original_height, original_width),
                                        masks_in.view(batch_size * timestep, 1, original_height, original_width),
                                        masks_updated.view(batch_size * timestep, 1, original_height, original_width)], dim=1))
        _, channel, height, width = encoder_hidden_states.size()
        local_features = encoder_hidden_states.view(batch_size, timestep, channel, height, width)[:, :local_timestep, ...]
        ref_features = encoder_hidden_states.view(batch_size, timestep, channel, height, width)[:, local_timestep:, ...]
        fold_feat_size = (height, width)

        ds_flows_f = F.interpolate(completed_flows[0].view(-1, 2, original_height, original_width), scale_factor=1/4, mode='bilinear', align_corners=False).view(batch_size, local_timestep-1, 2, height, width)/4.0
        ds_flows_b = F.interpolate(completed_flows[1].view(-1, 2, original_height, original_width), scale_factor=1/4, mode='bilinear', align_corners=False).view(batch_size, local_timestep-1, 2, height, width)/4.0
        ds_mask_in = F.interpolate(masks_in.reshape(-1, 1, original_height, original_width), scale_factor=1/4, mode='nearest').view(batch_size, timestep, 1, height, width)
        ds_mask_in_local = ds_mask_in[:, :local_timestep]
        ds_mask_updated_local =  F.interpolate(masks_updated[:,:local_timestep].reshape(-1, 1, original_height, original_width), scale_factor=1/4, mode='nearest').view(batch_size, local_timestep, 1, height, width)


        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, height, width))
            mask_pool_l = mask_pool_l.view(batch_size, timestep, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.view(-1, 1, height, width))
            mask_pool_l = mask_pool_l.view(batch_size, local_timestep, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))


        prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
        _, _, local_features, _ = self.feature_propagation_module(local_features, ds_flows_f, ds_flows_b, prop_mask_in, interpolation)
        encoder_hidden_states = torch.cat((local_features, ref_features), dim=1)

        trans_feat = self.soft_split(encoder_hidden_states.view(-1, channel, height, width), batch_size, fold_feat_size)
        mask_pool_l = mask_pool_l.permute(0,1,3,4,2).contiguous()
        trans_feat = self.transformers(trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation)
        trans_feat = self.soft_comp(trans_feat, timestep, fold_feat_size)
        trans_feat = trans_feat.view(batch_size, timestep, -1, height, width)

        encoder_hidden_states = encoder_hidden_states + trans_feat

        if self.training:
            output = self.decoder(encoder_hidden_states.view(-1, channel, height, width))
            output = torch.tanh(output).view(batch_size, timestep, 3, original_height, original_width)
        else:
            output = self.decoder(encoder_hidden_states[:, :local_timestep].view(-1, channel, height, width))
            output = torch.tanh(output).view(batch_size, local_timestep, 3, original_height, original_width)

        return output


# ######################################################################
#  ProPainterDiscriminator for Temporal Patch GAN
# ######################################################################
class ProPainterSpectralNorm(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1

    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(self, name='weight', num_power_iterations=1, dimension=0, eps=1e-12):
        self.name = name
        self.dimension = dimension
        if num_power_iterations <= 0:
            raise ValueError(
                'Expected num_power_iterations to be positive, but '
                'got num_power_iterations={}'.format(num_power_iterations))
        self.num_power_iterations = num_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dimension != 0:
            # permute dimension to front
            weight_mat = weight_mat.permute(
                self.dimension,
                *[d for d in range(weight_mat.dim()) if d != self.dimension])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.num_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u),
                                  dim=0,
                                  eps=self.eps,
                                  out=v)
                    u = normalize(torch.mv(weight_mat, v),
                                  dim=0,
                                  eps=self.eps,
                                  out=u)
                if self.num_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma#🐺
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name,
                                  torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(
            module, self.name,
            self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(),
                               weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, num_power_iterations, dimension, eps):
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, ProPainterSpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name))

        func = ProPainterSpectralNorm(name, num_power_iterations, dimension, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = func.reshape_weight_to_matrix(weight)

            height, width = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=func.eps)
            v = normalize(weight.new_empty(width).normal_(0, 1), dim=0, eps=func.eps)

        delattr(module, func.name)
        module.register_parameter(func.name + "_orig", weight)
        # We still need to assign weight back as func.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, func.name, weight.data)
        module.register_buffer(func.name + "_u", u)
        module.register_buffer(func.name + "_v", v)

        module.register_forward_pre_hook(func)

        module._register_state_dict_hook(ProPainterSpectralNormStateDictHook(func))
        module._register_load_state_dict_pre_hook(
            ProPainterSpectralNormLoadStateDictPreHook(func))
        return func

# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class ProPainterSpectralNormLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, func):
        self.func = func

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        func = self.func
        version = local_metadata.get('spectral_norm',
                                     {}).get(func.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + func.name + '_orig']
                weight_mat = func.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + func.name + '_u']


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class ProPainterSpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, func):
        self.func = func

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.func.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError(
                "Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.func._version


def spectral_norm(module,
                  name='weight',
                  num_power_iterations=1,
                  eps=1e-12,
                  dimension=None):
    r"""Applies spectral normalization to a parameter in the given module.

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        num_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dimension (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dimension is None:
        if isinstance(module,
                      (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
                       torch.nn.ConvTranspose3d)):
            dimension = 1
        else:
            dimension = 0
    ProPainterSpectralNorm.apply(module, name, num_power_iterations, dimension, eps)
    return module

class ProPainterDiscriminator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 ):
        super(ProPainterDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        num_features = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=num_features * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(num_features * 1,
                          num_features * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(num_features * 2,
                          num_features * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(num_features * 4,
                          num_features * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(num_features * 4,
                          num_features * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_features * 4,
                      num_features * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))


    def forward(self, completed_frames):
        completed_frames_t = torch.transpose(completed_frames, 1, 2)
        hidden_states = self.conv(completed_frames_t)
        if self.use_sigmoid:
            hidden_states = torch.sigmoid(hidden_states)
        hidden_states = torch.transpose(hidden_states, 1, 2)  # batch_size, timesteps, channel, height, width
        return hidden_states




#########################################propainter moduelsss finsihes here##########################################################################




# Copied from transformers.models.vit.modeling_vit.ViTPreTrainedModel with ViT->ProPainter,vit->propainter
class ProPainterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProPainterConfig
    base_model_prefix = "propainter"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["ProPainterEmbeddings", "ProPainterLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ProPainterSecondOrderDeformableAlignment) or isinstance(module, ProPainterDeformableAlignment):
            num_channels = module.in_channels
            for k in module.kernel_size:
                num_channels *= k
            stdv = 1. / math.sqrt(num_channels)
            module.weight.data.uniform_(-stdv, stdv)
            if module.bias is not None:
                module.bias.data.zero_()     
            if hasattr(module.conv_offset[-1], 'weight') and module.conv_offset[-1].weight is not None:
                TORCH_INIT_FUNCTIONS["constant_"](module.conv_offset[-1].weight, 0)
            if hasattr(module.conv_offset[-1], 'bias') and module.conv_offset[-1].bias is not None:
                TORCH_INIT_FUNCTIONS["constant_"](module.conv_offset[-1].bias, 0)   
        elif isinstance(module, ProPainterInpaintGenerator) or isinstance(module, ProPainterDiscriminator):
            for child in module.children():
                classname = child.__class__.__name__
                if classname.find('InstanceNorm2d') != -1:
                    if hasattr(child, 'weight') and child.weight is not None:
                        nn.init.constant_(child.weight.data, 1.0)
                    if hasattr(child, 'bias') and child.bias is not None:
                        nn.init.constant_(child.bias.data, 0.0)
                elif hasattr(child, 'weight') and (classname.find('Conv') != -1
                                            or classname.find('Linear') != -1):
                    nn.init.normal_(child.weight.data, 0.0, 0.02)
                    if hasattr(child, 'bias') and child.bias is not None:
                        nn.init.constant_(child.bias.data, 0.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


PROPAINTER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ProPainterConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PROPAINTER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ProPainterImageProcessor.__call__`]
            for details.

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
    "The bare ProPainter Model outputting composed frames without any specific head on top.",
    PROPAINTER_START_DOCSTRING,
)
# Copied from transformers.models.vit.modeling_vit.ViTModel with VIT->PROPAINTER,ViT->ProPainter
class ProPainterModel(ProPainterPreTrainedModel):
    def __init__(self, config: ProPainterConfig):
        super().__init__(config)
        self.config = config
        self.optical_flow_model = ProPainterRaftOpticalFlow(config)
        self.flow_completion_net = ProPainterRecurrentFlowCompleteNet()
        self.inpaint_generator = ProPainterInpaintGenerator()
        self.discriminator = ProPainterDiscriminator(use_sigmoid=self.config.GAN_LOSS != 'hinge')
        # Initialize weights and apply final processing
        self.post_init()

    def _get_ref_index(self,mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    def _get_short_clip_len(self, width):
        if width <= 640:
            return 12
        elif width <= 720:
            return 8
        elif width <= 1280:
            return 4
        else:
            return 2

    @add_start_docstrings_to_model_forward(PROPAINTER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )

    def compute_flow(self,pixel_values):
        short_clip_len = self._get_short_clip_len(pixel_values.size(-1))
        if pixel_values.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = zip(*[
                self.optical_flow_model(
                    pixel_values[:, max(f - 1, 0):end_f], iters=self.config.raft_iter
                ) for f, end_f in
                [(f, min(self.video_length, f + short_clip_len)) for f in range(0, self.video_length, short_clip_len)]
            ])
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = self.optical_flow_model(pixel_values, iters=self.config.raft_iter)

        torch.cuda.empty_cache()
        return gt_flows_bi
    
    def complete_flow(self,gt_flows_bi, flow_masks):
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > self.config.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, self.config.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.config.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.config.subvideo_length)
                pred_flows_bi_sub, _ = self.flow_completion_net.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = self.flow_completion_net.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = self.flow_completion_net.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = self.flow_completion_net.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()

        return pred_flows_bi        

    def image_propagation(self,pixel_values,masks_dilated, pred_flows_bi):
        width, height = self.size
        masked_frames = pixel_values * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, self.config.subvideo_length) # ensure a minimum of 100 frames for image propagation
        if self.video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, self.video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(self.video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(self.video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = self.inpaint_generator.img_propagation(masked_frames[:, s_f:e_f], 
                                                                    pred_flows_bi_sub, 
                                                                    masks_dilated[:, s_f:e_f], 
                                                                    'nearest')
                updated_frames_sub = pixel_values[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, height, width) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, height, width)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = self.inpaint_generator.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = pixel_values * (1 - masks_dilated) + prop_imgs.view(b, t, 3, height, width) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, height, width)
            torch.cuda.empty_cache()

        return updated_frames,updated_masks

    def feature_propagation(self,updated_frames,updated_masks,masks_dilated,pred_flows_bi,original_frames):
        width, height = self.size
        comp_frames = [None] * self.video_length

        neighbor_stride = self.config.neighbor_length // 2
        if self.video_length > self.config.subvideo_length:
            ref_num = self.config.subvideo_length // self.config.ref_stride
        else:
            ref_num = -1
        
        # ---- feature propagation + transformer ----
        for f in range(0, self.video_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(self.video_length, f + neighbor_stride + 1))
            ]
            ref_ids = self._get_ref_index(f, neighbor_ids, self.video_length, self.config.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                
                # pred_img = selected_imgs # results of image propagation
                pred_img = self.inpaint_generator(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                
                pred_img = pred_img.view(-1, 3, height, width)

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + original_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else: 
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

        return comp_frames
    
 
    def forward(
        self,
        pixel_values_inp: Optional[List[np.ndarray]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        flow_masks: Optional[torch.BoolTensor] = None,
        masks_dilated: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int,int]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        

        self.size = size
        
        self.video_length = pixel_values.size(1)
        with torch.no_grad():
            # ---- compute flow ----
            
            # # use fp32 for RAFT
            # if pixel_values.size(1) > short_clip_len:
            #     gt_flows_f_list, gt_flows_b_list = [], []
            #     for f in range(0, video_length, short_clip_len):
            #         end_f = min(video_length, f + short_clip_len)
            #         if f == 0:
            #             flows_f, flows_b = self.optical_flow_model(pixel_values[:,f:end_f], iters=self.config.raft_iter)
            #         else:
            #             flows_f, flows_b = self.optical_flow_model(pixel_values[:,f-1:end_f], iters=self.config.raft_iter)
                    
            #         gt_flows_f_list.append(flows_f)
            #         gt_flows_b_list.append(flows_b)
            #         torch.cuda.empty_cache()
                    
            #     gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            #     gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            #     gt_flows_bi = (gt_flows_f, gt_flows_b)
            # else:
            #     gt_flows_bi = self.optical_flow_model(pixel_values, iters=self.config.raft_iter)
            #     torch.cuda.empty_cache()

            gt_flows_bi = self.compute_flow(pixel_values)
            
            # ---- complete flow ----
            pred_flows_bi = self.complete_flow(gt_flows_bi,flow_masks)
                

            # ---- image propagation ----
            updated_frames,updated_masks = self.image_propagation(pixel_values,masks_dilated,pred_flows_bi)
                
        comp_frames = self.feature_propagation(updated_frames, updated_masks, masks_dilated, pred_flows_bi, pixel_values_inp)
        
        # if not return_dict:
        #     head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        #     return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=comp_frames,
            hidden_states=None,
            attentions=None,
        )


# @add_start_docstrings(
#     """ProPainter Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

#     <Tip>

#     Note that we provide a script to pre-train this model on custom data in our [examples
#     directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

#     </Tip>
#     """,
#     PROPAINTER_START_DOCSTRING,
# )

# class ProPainterModelForVideoInPainting(ProPainterPreTrainedModel):
#     def __init__(self, config: ProPainterConfig) -> None:
#         super().__init__(config)

#         self.propainter = ProPainterModel(config, add_pooling_layer=False, use_mask_token=True)

#         self.decoder = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=config.hidden_size,
#                 out_channels=config.encoder_stride**2 * config.num_channels,
#                 kernel_size=1,
#             ),
#             nn.PixelShuffle(config.encoder_stride),
#         )

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(PROPAINTER_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_IMAGE_CLASS_CHECKPOINT,
#         output_type=MaskedImageModelingOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_EXPECTED_OUTPUT_SHAPE,
#     )
#     @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         bool_masked_pos: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, MaskedImageModelingOutput]:
#         r"""
#         bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
#             Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

#         Returns:

#         Examples:
#         ```python
#         >>> from transformers import AutoImageProcessor, ProPainterForMaskedImageModeling
#         >>> import torch
#         >>> from PIL import Image
#         >>> import requests

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> image_processor = AutoImageProcessor.from_pretrained("ruffy369/propainter")
#         >>> model = ProPainterForMaskedImageModeling.from_pretrained("ruffy369/propainter")

#         >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
#         >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
#         >>> # create random boolean mask of shape (batch_size, num_patches)
#         >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

#         >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
#         >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
#         >>> list(reconstructed_pixel_values.shape)
#         [1, 3, 224, 224]
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
#             raise ValueError(
#                 "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
#                 "the reconstructed image has the same dimensions as the input. "
#                 f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
#             )

#         outputs = self.propainter(
#             pixel_values,
#             bool_masked_pos=bool_masked_pos,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             interpolate_pos_encoding=interpolate_pos_encoding,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         # Reshape to (batch_size, num_channels, height, width)
#         sequence_output = sequence_output[:, 1:]
#         batch_size, sequence_length, num_channels = sequence_output.shape
#         height = width = math.floor(sequence_length**0.5)
#         sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

#         # Reconstruct pixel values
#         reconstructed_pixel_values = self.decoder(sequence_output)

#         masked_im_loss = None
#         if bool_masked_pos is not None:
#             size = self.config.image_size // self.config.patch_size
#             bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
#             mask = (
#                 bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
#                 .repeat_interleave(self.config.patch_size, 2)
#                 .unsqueeze(1)
#                 .contiguous()
#             )
#             reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
#             masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

#         if not return_dict:
#             output = (reconstructed_pixel_values,) + outputs[1:]
#             return ((masked_im_loss,) + output) if masked_im_loss is not None else output

#         return MaskedImageModelingOutput(
#             loss=masked_im_loss,
#             reconstruction=reconstructed_pixel_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     ProPainter Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
#     the [CLS] token) e.g. for ImageNet.

#     <Tip>

#         Note that it's possible to fine-tune ProPainter on higher resolution images than the ones it has been trained on, by
#         setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
#         position embeddings to the higher resolution.

#     </Tip>
#     """,
#     PROPAINTER_START_DOCSTRING,
# )
# # Copied from transformers.models.vit.modeling_vit.ViTForImageClassification with VIT->PROPAINTER,ViT->ProPainter,vit->propainter
# class ProPainterModelForVideoOutPainting(ProPainterPreTrainedModel):
#     def __init__(self, config: ProPainterConfig) -> None:
#         super().__init__(config)

#         self.num_labels = config.num_labels
#         self.propainter = ProPainterModel(config, add_pooling_layer=False)

#         # Classifier head
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(PROPAINTER_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_IMAGE_CLASS_CHECKPOINT,
#         output_type=MaskedImageModelingOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_EXPECTED_OUTPUT_SHAPE,
#     )
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, MaskedImageModelingOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.propainter(
#             pixel_values,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             interpolate_pos_encoding=interpolate_pos_encoding,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         logits = self.classifier(sequence_output[:, 0, :])

#         loss = None
#         if labels is not None:
#             # move labels to correct device to enable model parallelism
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return MaskedImageModelingOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )