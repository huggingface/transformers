# coding=utf-8
# Copyright 2023 The Meta AI Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch MOBILESAM model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mobilesam import MobileSamConfig, MobileSamMaskDecoderConfig, MobileSamPromptEncoderConfig, MobileSamVisionConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MobileSamConfig"
_CHECKPOINT_FOR_DOC = "facebook/mobilesam"

MOBILESAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mobilesam",
    # See all MobileSam models at https://huggingface.co/models?filter=mobilesam
]



@dataclass
# Copied from transformers.models.sam.modeling_sam.SamVisionEncoderOutput with Sam->MobileSam,sam->mobilesam
class MobileSamVisionEncoderOutput(ModelOutput):
    """
    Base class for mobilesam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.sam.modeling_sam.SamImageSegmentationOutput with Sam->MobileSam
class MobileSamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    iou_scores: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class MobileSamVisionConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, pad=0, dilation=1, groups=1):
        super().__init__()

        self.conv =  nn.Conv2d(
            input_channels, 
            output_channels,
            kernel_size, 
            stride, 
            pad, 
            dilation, 
            groups, 
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(output_channels)
    
    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        return hidden_states
    

class MobileSamVisionConvBlock(nn.Module):
    def __init__(self, config, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_chans = int(input_channels * config.expand_ratio)
        self.output_channels = output_channels

        self.conv1 = MobileSamVisionConv(input_channels, self.hidden_chans, kernel_size=1)
        self.act = ACT2FN[config.hidden_act]

        self.conv2 = MobileSamVisionConv(
            self.hidden_chans, 
            self.hidden_chans,
            kernel_size=3, 
            stride=1, 
            pad=1, 
            groups=self.hidden_chans
        )

        self.conv3 = MobileSamVisionConv(self.hidden_chans, self.output_channels, kernel_size=1)


    def forward(self, hidden_states):
        shortcut = hidden_states

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.act(hidden_states)

        hidden_states = self.conv3(hidden_states)

        hidden_states += shortcut
        hidden_states = self.act(hidden_states)

        return hidden_states    
        

class MobileSamVisionMLPBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_dim = config.embed_dims[layer_idx]
        mlp_dim = int(config.mlp_ratio * hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, hidden_dim)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


class MobileSamVisionBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        hidden_dim = config.embed_dims[layer_idx]

        self.attn = MobileSamVisionAttention(config, layer_idx)
        self.mlp = MobileSamVisionMLPBlock(config, layer_idx)
        self.local_conv = MobileSamVisionConv(hidden_dim, hidden_dim, kernel_size=3, stride=1, pad=1, groups=hidden_dim)

    def forward(self, hidden_states):
        hidden_states = self.attn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.local_conv(hidden_states)
        return hidden_states


class MobileSamVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.image_size

        self.patches_resolution = (image_size // 4, image_size // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.input_channels = config.input_channels
        self.embed_dim = config.embed_dims[0]

        self.conv1 = MobileSamVisionConv(self.input_channels, self.embed_dim // 2, 3, 2, 1)

        self.act = ACT2FN[config.hidden_act]

        self.conv2 = MobileSamVisionConv(self.embed_dim // 2, self.embed_dim, 3, 2, 1)

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class MobileSamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""
    def __init__(self, config, layer_idx):
        super().__init__()

        self.hidden_dim = config.embed_dims[layer_idx]

        self.num_attention_heads = config.num_attention_heads[layer_idx]
        self.attention_offsets = config.attention_offsets[layer_idx]

        head_dim = self.hidden_dim // config.num_attention_heads[layer_idx]
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.post_attention_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.pre_attention_layer_norm = nn.LayerNorm(self.hidden_dim)

        self.attention_biases = torch.nn.Parameter(
            torch.zeros(self.num_attention_heads, self.attention_offsets)
        )

    # TODO: completely refactor this as it has been copied from the authors code directly
    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.pre_attention_layer_norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_attention_heads, -
                           1).split([self.hidden_dim, self.hidden_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        x = self.post_attention_layer_norm(x)
        return x


# Copied from transformers.models.sam.modeling_sam.SamMLPBlock with Sam->MobileSam
class MobileSamMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->MobileSam
class MobileSamLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# Copied from transformers.models.sam.modeling_sam.SamAttention with SAM->MOBILESAM,Sam->MobileSam,sam->mobilesam
class MobileSamAttention(nn.Module):
    """
    MOBILESAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(self, config, downmobilesample_rate=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        downmobilesample_rate = config.attention_downmobilesample_rate if downmobilesample_rate is None else downmobilesample_rate

        self.internal_dim = config.hidden_size // downmobilesample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # MobileSamAttention
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # batch_size * point_batch_size  x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out


# Copied from transformers.models.sam.modeling_sam.SamTwoWayAttentionBlock with Sam->MobileSam,sam->mobilesam
class MobileSamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downmobilesample_rate: int = 2, skip_first_layer_pe: bool = False):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`MobileSamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downmobilesample_rate (*optionalk*, int, defaults to 2):
                The downmobilesample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = MobileSamAttention(config, downmobilesample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.cross_attn_token_to_image = MobileSamAttention(config, downmobilesample_rate=attention_downmobilesample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.mlp = MobileSamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = MobileSamAttention(config, downmobilesample_rate=attention_downmobilesample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        output_attentions: bool = False,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out

        queries = self.layer_norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        keys = self.layer_norm4(keys)

        outputs = (queries, keys)

        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)

        return outputs


# Copied from transformers.models.sam.modeling_sam.SamTwoWayTransformer with Sam->MobileSam
class MobileSamTwoWayTransformer(nn.Module):
    def __init__(self, config: MobileSamMaskDecoderConfig):
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            self.layers.append(MobileSamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = MobileSamAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_attentions = ()

        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)

        # Prepare queries
        queries = point_embeddings
        keys = image_embeddings

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding

            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)

        # Apply the final attenion layer from the points to the image
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys, all_attentions


# Copied from transformers.models.sam.modeling_sam.SamFeedForward with Sam->MobileSam
class MobileSamFeedForward(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


# Copied from transformers.models.sam.modeling_sam.SamMaskDecoder with Sam->MobileSam
class MobileSamMaskDecoder(nn.Module):
    def __init__(self, config: MobileSamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = MobileSamTwoWayTransformer(config)

        # should we create a new class for this?
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = MobileSamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [MobileSamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = MobileSamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


# Copied from transformers.models.sam.modeling_sam.SamPositionalEmbedding with Sam->MobileSam
class MobileSamPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.hidden_size // 2
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, config.num_pos_feats)))

    def forward(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()

        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]

        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        # outputs d_1 x ... x d_n x channel shape
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


# Copied from transformers.models.sam.modeling_sam.SamMaskEmbedding with Sam->MobileSam
class MobileSamMaskEmbedding(nn.Module):
    def __init__(self, config: MobileSamPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = MobileSamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = MobileSamLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


# Copied from transformers.models.sam.modeling_sam.SamPromptEncoder with Sam->MobileSam
class MobileSamPromptEncoder(nn.Module):
    def __init__(self, config: MobileSamPromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = MobileSamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # This is required for the ONNX export. The dtype, device need to be explicitely
        # specificed as otherwise torch.onnx.export interprets as double
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device),
        )

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(
        self,
        input_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

        return sparse_embeddings, dense_embeddings


class MobileSamVisionPatchMerging(nn.Module):
    def __init__(self, config, input_resolution, input_channels, output_channels):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = input_channels
        self.output_channels = output_channels
        self.act = ACT2FN[config.hidden_act]
        self.conv1 = MobileSamVisionConv(input_channels, output_channels, kernel_size=1, stride=1, pad=0)
        
        stride = 2
        if(output_channels==320 or output_channels==448 or output_channels==576):
            stride =1
        
        self.conv2 = MobileSamVisionConv(output_channels, output_channels, kernel_size=3, stride=stride, pad=1, groups=output_channels)
        self.conv3 = MobileSamVisionConv(output_channels, output_channels, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MobileSamVisionConvLayer(nn.Module):
    def __init__(self, config, layer_idx):

        super().__init__()
        patches_resolution = config.image_size // 4

        self.hidden_dim = config.embed_dims[layer_idx]
        self.output_hidden_dim = config.embed_dims[min(layer_idx + 1, len(config.embed_dims) - 1)]

        self.input_resolution = (patches_resolution // (2 ** (layer_idx - 1 if layer_idx == 3 else layer_idx)))
        self.depth = config.depths[layer_idx]
        self.expand_ratio = config.expand_ratio

        # build blocks
        self.layers = nn.ModuleList([MobileSamVisionConvBlock(config, self.hidden_dim, self.hidden_dim) for _ in range(self.depth)])

        # patch merging layer
        self.downsample = MobileSamVisionPatchMerging(
            config,
            (patches_resolution, patches_resolution), 
            input_channels=self.hidden_dim, 
            output_channels=self.output_hidden_dim,
        )


    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class MobileSamVisionLayer(nn.Module):
    def __init__(self, config, layer_idx):

        super().__init__()
        patches_resolution = config.image_size // 4

        self.hidden_dim = config.embed_dims[layer_idx]
        self.output_hidden_dim = config.embed_dims[min(layer_idx + 1, len(config.embed_dims) - 1)]

        self.input_resolution = (patches_resolution // (2 ** (layer_idx - 1 if layer_idx == 3 else layer_idx)))
        self.depth = config.depths[layer_idx]
        self.window_size = config.window_sizes[layer_idx]

        # build blocks
        self.layers = nn.ModuleList([MobileSamVisionBlock(config, layer_idx) for _ in range(self.depth)])

        # patch merging layer
        if layer_idx < config.num_hidden_layers - 1:
            self.downsample = MobileSamVisionPatchMerging(
                config,
                (patches_resolution, patches_resolution), 
                input_channels=self.hidden_dim, 
                output_channels=self.output_hidden_dim,
            )
        else:
            self.downsample = None


    def forward(self, x):
        for block in self.layers:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# Copied from transformers.models.sam.modeling_sam.SamVisionNeck with Sam->MobileSam
class MobileSamVisionNeck(nn.Module):
    def __init__(self, config: MobileSamVisionConfig):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = MobileSamLayerNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = MobileSamLayerNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


# Copied from transformers.models.sam.modeling_sam.SamVisionEncoder with Sam->MobileSam
class MobileSamVisionEncoder(nn.Module):
    def __init__(self, config: MobileSamVisionConfig):
        super().__init__()
        self.patch_embed = MobileSamVisionPatchEmbed(config)

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx == 0:
                target_cls = MobileSamVisionConvLayer
            else:
                target_cls = MobileSamVisionLayer
            self.layers.append(target_cls(config, layer_idx))

        self.layer_norm_head = nn.LayerNorm(config.embed_dims[-1])
        self.head = nn.Linear(
            config.embed_dims[-1], config.num_classes
        )

        self.neck = MobileSamVisionNeck(config)


        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MobileSamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_hidden_states = ()
        all_self_attentions = ()

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return MobileSamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.sam.modeling_sam.SamPreTrainedModel with Sam->MobileSam,sam->mobilesam
class MobileSamPreTrainedModel(PreTrainedModel):
    config_class = MobileSamConfig
    base_model_prefix = "mobilesam"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MOBILESAM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MobileSamConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


MOBILESAM_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SamProcessor`]. See [`SamProcessor.__call__`] for
            details.
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and botton right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box

        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            MOBILESAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).

        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerMOBILESAM](https://arxiv.org/abs/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerMOBILESAM](https://arxiv.org/abs/2305.03048).
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
    "Segment Anything Model (MOBILESAM) for generating segmentation masks, given an input image and ",
    " optional 2D location and bounding boxes.",
    MOBILESAM_START_DOCSTRING,
)
# Copied from transformers.models.sam.modeling_sam.SamModel with SAM->MOBILESAM,Sam->MobileSam,sam->mobilesam
class MobileSamModel(MobileSamPreTrainedModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        super().__init__(config)
        self.shared_image_embedding = MobileSamPositionalEmbedding(config.vision_config)

        self.vision_encoder = MobileSamVisionEncoder(config.vision_config)
        self.prompt_encoder = MobileSamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        self.mask_decoder = MobileSamMaskDecoder(config.mask_decoder_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    @add_start_docstrings_to_model_forward(MOBILESAM_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/mobilesam-vit-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/mobilesam-vit-base")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mobilesam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the mobilesame. ",
                "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
                " if you want to pass multiple points for the mobilesame image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        if not return_dict:
            output = (iou_predictions, low_res_masks)
            if output_hidden_states:
                output = output + (vision_hidden_states,)

            if output_attentions:
                output = output + (vision_attentions, mask_decoder_attentions)
            return output

        return MobileSamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        )
