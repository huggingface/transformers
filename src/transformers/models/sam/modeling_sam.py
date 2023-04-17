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
""" PyTorch SAM model."""

import collections
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
from ...utils import ModelOutput, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/sam-vit-h"

SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/sam-vit-h",
    "facebook/sam-vit-l",
    "facebook/sam-vit-b",
    # See all SAM models at https://huggingface.co/models?filter=sam
]


@dataclass
class SamVisionEncoderOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

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
class SamPromptEncoderOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
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

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SamMaskEncoderOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
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

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        low_resolution_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
    """

    iou_scores: torch.FloatTensor = None
    low_resolution_masks: torch.FloatTensor = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from src.models.modeling_vit_mae.ViTMAEPatchEmbeddings with ViTMAEPatchEmbeddings->SamVisionEmbeddings,x->embeddings
class SamPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class SamMLPBlock(nn.Module):
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


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->Sam
class SamLayerNorm(nn.Module):
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


class SamTwoWayTransformerAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(self, config, downsample_rate=None) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        b, n, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b, n, num_attention_heads, c // num_attention_heads)
        return hidden_states.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, hidden_states: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Separate into heads
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # SamAttention
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ value
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class SamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse inputs, (2) cross attention of sparse inputs
        to dense inputs, (3) mlp block on sparse inputs, and (4) cross attention of dense inputs to sparse inputs.

        Arguments:
          hidden_size (int): the channel dimension of the embeddings
          num_attention_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = SamTwoWayTransformerAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.cross_attn_token_to_image = SamTwoWayTransformerAttention(
            config, downsample_rate=attention_downsample_rate
        )
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamTwoWayTransformerAttention(
            config, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
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

        attn_out = self.cross_attn_token_to_image(query=query, key=key, value=keys)
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


class SamTwoWayTransformer(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = SamTwoWayTransformerAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        image_embedding: Tensor,
        image_position_embedding: Tensor,
        point_embedding: Tensor,
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

        if image_embedding is None:
            raise ValueError("You have to specify image_embedding")

        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_position_embedding = image_position_embedding.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embedding,
                key_point_embedding=image_position_embedding,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)

        # Apply the final attenion layer from the points to the image
        query = queries + point_embedding
        key = keys + image_position_embedding

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys, all_attentions


class SamFeedForward(nn.Module):
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
        for i, layer in enumerate(self.layers):
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class SamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamTwoWayTransformer(config)

        # should we create a new class for this?
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embedding: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_positional_embedding (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks torch.Tensor: batched predictions of mask quality
        """
        # Concatenate output tokens
        batch_size = image_embeddings.shape[0]
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        tokens = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings
        src = src + dense_prompt_embeddings

        pos_src = image_positional_embedding
        batch_size, num_channels, height, width = src.shape

        # Run the transformer
        hs, src, attentions = self.transformer(
            src,
            pos_src,
            tokens,
            output_attentions=output_attentions,
        )
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(batch_size, num_channels, height, width)

        upscaled_embedding = self.upscale_conv1(src)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        # let's define a class for this: probably the dynamic_prediction_head?
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        batch_size, num_channels, height, width = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(batch_size, num_channels, height * width)).view(
            batch_size, -1, height, width
        )

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.hidden_size // 2
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, config.num_pos_feats)))

    def forward(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()

        if input_shape is not None:
            coordinates[:, :, 0] = coordinates[:, :, 0] / input_shape[1]
            coordinates[:, :, 1] = coordinates[:, :, 1] / input_shape[0]

        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


class SamMaskEmbedding(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = SamLayerNorm(self.mask_input_channels)
        self.layer_norm2 = SamLayerNorm(self.mask_input_channels * 4)

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


class SamPromptEncoder(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.input_image_size

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight

        point_embedding[labels == -10] = 0.0  # ignore points

        point_embedding[labels == 0] += self.point_embed[0].weight
        point_embedding[labels == 1] += self.point_embed[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, 0, :] += self.point_embed[2].weight
        corner_embedding[:, 1, :] += self.point_embed[3].weight
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

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(hidden_size), where N is determined by the number of input points and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(hidden_size)x(embed_H)x(embed_W)
        """
        sparse_embeddings = None
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size = input_points[0].shape[0] if len(input_points.shape) == 2 else input_points.shape[0]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = torch.empty((batch_size, 0, self.hidden_size), device=target_device)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = torch.empty((batch_size, 0, self.hidden_size), device=target_device)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        if sparse_embeddings is None:
            sparse_embeddings = torch.empty((dense_embeddings.shape[0], 0, self.hidden_size), device=target_device)

        return sparse_embeddings, dense_embeddings


class SamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size) -> None:
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )

        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int): size of query q.
            k_size (int): size of key k.
            rel_pos (Tensor): relative position embeddings (L, C).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https:
            //github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
        # noqa B950
            attn (Tensor): attention map. q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis. rel_pos_w (Tensor): relative
            position embeddings (Lw, C) for width axis. q_size (Tuple): spatial sequence size of query q with (q_h,
            q_w). k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = self.get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = query.shape
        r_q = query.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
            B, q_h * q_w, k_h * k_w
        )

        return attn

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        B, H, W, _ = hidden_states.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(hidden_states).reshape(B, H * W, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_attention_heads, H * W, -1).unbind(0)

        attn_weights = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(attn_weights, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(q.dtype)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (
            (attn_probs @ v).view(B, self.num_attention_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        )
        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        return outputs


class SamVisionLayer(nn.Module):
    def __init__(self, config, window_size) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.window_size = window_size

    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [B, H, W, C]. window_size (int): window size.

        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C]. (Hp, Wp): padded
            height and width before partition
        """
        B, H, W, C = hidden_states.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        hidden_states = hidden_states.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (Hp, Wp)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor): input tokens with [B * num_windows, window_size, window_size, C]. window_size
            (int): window size. pad_hw (Tuple): padded height and width (Hp, Wp). hw (Tuple): original height and width
            (H, W) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [B, H, W, C].
        """
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        hidden_states = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            hidden_states = hidden_states[:, :H, :W, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # Window partition
        if self.window_size > 0:
            H, W = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_hw = self.window_partition(hidden_states, self.window_size)

        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, pad_hw, (H, W))

        hidden_states = residual + hidden_states
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SamVisionEncoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck_conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.neck_layer_norm1 = SamLayerNorm(config.output_channels, data_format="channels_first")
        self.neck_conv2 = nn.Conv2d(
            config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False
        )
        self.neck_layer_norm2 = SamLayerNorm(config.output_channels, data_format="channels_first")
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Should neck have nn module of its own?
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.neck_conv1(hidden_states)
        hidden_states = self.neck_layer_norm1(hidden_states)

        hidden_states = self.neck_conv2(hidden_states)
        hidden_states = self.neck_layer_norm2(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SamPreTrainedModel(PreTrainedModel):
    config_class = SamConfig
    base_model_prefix = "sam"
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


SAM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SamConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SAM_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SamProcessor`]. See [`SamProcessor.__call__`] for
            details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

SAM_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

SAM_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SamProcessor`]. See [`SamProcessor.__call__`] for
            details.

        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
            provided to serve as text prompt, which the language model can continue.

            Indices can be obtained using [`SamProcessor`]. See [`SamProcessor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary of the language model. Only relevant in case an
            encoder-decoder language model (like T5) is used.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are decoder input IDs?](../glossary#decoder-input-ids)

        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            Only relevant in case an encoder-decoder language model (like T5) is used.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class SamForMaskGeneration(SamPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)

        self.vision_encoder = SamVisionEncoder(config.vision_config)
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        self.mask_decoder = SamMaskDecoder(config.mask_decoder_config)

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
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # C x H x W

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
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
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict=True,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Args:
            multimask_output (`bool`, *optional*):
                Whether or not to return 3 scores and binary mask for each image as described in the original paper.

        Returns:
            iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
                The predicted iou scores of the 3 predicted masks
            low_resolution_masks (`torch.FloatTensor`):
                Raw low resolution masks
        Example:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        image_position_embedding = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_position_embedding = image_position_embedding.repeat(batch_size, 1, 1, 1)

        all_attentions = ()

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (vision_outputs[-1],)
            else:
                all_attentions = all_attentions + (None,)
        else:
            all_attentions = all_attentions + (None,)

        if input_points is not None and input_labels is None:
            # TODO: @younesbelkada fix this
            input_labels = torch.ones(input_points.shape[0], dtype=torch.int, device=input_points.device)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embedding=image_position_embedding,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            output_attentions=output_attentions,
        )

        if output_attentions:
            all_attentions = all_attentions + (mask_decoder_attentions,)
        else:
            all_attentions = all_attentions + (None,)

        if not return_dict:
            output = (iou_predictions, low_res_masks)
            if output_attentions:
                output = output + (all_attentions[0], all_attentions[1])
            return output

        return SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            low_resolution_masks=low_res_masks,
            vision_attentions=all_attentions[0],
            mask_decoder_attentions=all_attentions[1],
        )
