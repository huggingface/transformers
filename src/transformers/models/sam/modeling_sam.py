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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
    logging,
)
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/sam-vit-h"

SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/sam-vit-h",
    # See all SAM models at https://huggingface.co/models?filter=sam
]


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
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
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


class SamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

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

        self.self_attn = SamTwoWayTransformerAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)

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

    def forward(self, queries: Tensor, keys: Tensor, query_point_embedding: Tensor, key_point_embedding: Tensor):
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
        return queries, keys


class SamTwoWayTransformerAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self, config, downsample_rate: int = 1) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(self, x: Tensor, num_attention_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_attention_heads, c // num_attention_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

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


class SamTwoWayTransformer(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = SamTwoWayTransformerAttention(config)
        self.norm_final_attn = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        image_embedding: Tensor,
        image_position_embedding: Tensor,
        point_embedding: Tensor,
    ):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_position_embedding = image_position_embedding.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embedding,
                key_point_embedding=image_position_embedding,
            )

        # Apply the final attenion layer from the points to the image
        query = queries + point_embedding
        key = keys + image_position_embedding

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


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
        for _, layer in enumerate(self.layers):
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class SamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = self.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamTwoWayTransformer(config)

        # should we create a new class for this?
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upsacle_layer_norm = SamLayerNorm(self.hidden_size // 4)
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
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        tokens = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding = self.upscale_conv1(src)
        upscaled_embedding = self.activation(self.upsacle_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        # let's define a class for this: probably the dynamic_prediction_head?
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class SamPositionalEmbedding(nn.Embedding):
    def __init__(self, config):
        super().__init__()
        self.register_buffer(
            "positional_embedding",
            scale * torch.randn((2, num_pos_feats)),
        )

    def forward(self, input_coords):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()
        coordinates[:, :, 0] = coordinates[:, :, 0] / image_size[1]
        coordinates[:, :, 1] = coordinates[:, :, 1] / image_size[0]
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
        self.mask_input_size = (4 * self.image_embedding_size[0], 4 * self.image_embedding_size[1])
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = SamLayerNorm(self.mask_input_channels)
        self.layer_norm2 = SamLayerNorm(self.mask_input_channels)

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
        self.shared_emebdding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, self.hidden_size)

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, self.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.not_a_point_embed = nn.Embedding(1, self.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.shared_emebdding(points)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.shared_emebdding(coords)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(hidden_size), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(hidden_size)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.hidden_size), device=self._get_device())
        if points is not None:
            if labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            coords = points
            point_embeddings = self.point_embedding(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self.box_emebdding(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self.mask_embed(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse_embeddings, dense_embeddings


class SamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        head_dim = dim // num_attention_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_attention_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = torch.nn.functional.softmax(attn, dtype=torch.float32, dim=-1).to(q.dtype)
        x = (attn @ v).view(B, self.num_attention_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class SamVisionLayer(nn.Module):
    def __init__(
        self,
        config,
        window_size,
        # dim: int,
        # num_attention_heads: int,
        # mlp_ratio: float = 4.0,
        # qkv_bias: bool = True,
        # activation: str = "gelu",
        # use_rel_pos: bool = False,
        # window_size: int = 0,
        # input_size: Optional[Tuple[int, int]] = None,
        # layer_norm_eps: float = 1e-6,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_attention_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        input_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)

        self.attn = SamVisionAttention(
            config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            qkv_bias=config.qkv_bias,
            use_rel_pos=config.use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SamMLPBlock(
            hidden_size=config.hidden_size,
            mlp_dim=int(config.hidden_size * config.mlp_ratio),
            hidden_act=config.hidden_act,
        )

        self.window_size = window_size

    def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (tensor): input tokens with [B, H, W, C].
            window_size (int): window size.

        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
        """
        B, H, W, C = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (Hp, Wp)

    def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Window unpartition into original sequences and removing padding.
        Args:
            x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
            window_size (int): window size.
            pad_hw (Tuple): padded height and width (Hp, Wp).
            hw (Tuple): original height and width (H, W) before padding.

        Returns:
            x: unpartitioned sequences with [B, H, W, C].
        """
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x

    def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
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
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
        Args:
            attn (Tensor): attention map.
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = query.shape
        r_q = query.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
            B, q_h * q_w, k_h * k_w
        )

        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


# class SamVisionLayer(nn.Module):
#     """This corresponds to the Block class in the timm implementation."""

#     def __init__(self, config: SamVisionConfig) -> None:
#         super().__init__()
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward
#         self.seq_len_dim = 1
#         self.attention = SamVisionAttention(config)
#         self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.out_proj = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#         self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#         self_attention_outputs = self.attention(
#             self.layernorm_before(hidden_states),  # in ViTMAE, layernorm is applied before self-attention
#             head_mask,
#             output_attentions=output_attentions,
#         )
#         attention_output = self_attention_outputs[0]
#         outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

#         # first residual connection
#         hidden_states = attention_output + hidden_states

#         # in ViTMAE, layernorm is also applied after self-attention
#         layer_output = self.layernorm_after(hidden_states)

#         layer_output = self.intermediate(layer_output)

#         # second residual connection
#         layer_output = self.out_proj(layer_output)
#         layer_output = self.dropout(layer_output)
#         layer_output = layer_output + hidden_states

#         outputs = (layer_output,) + outputs

#         return outputs


class SamVisionEncoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        # self.activation = ACT2FN[self.config.hidden_act]
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
                # dim=self.config.hidden_size,
                # num_attention_heads=self.config.num_attention_heads,
                # mlp_ratio=self.config.mlp_ratio,
                # qkv_bias=self.config.qkv_bias,
                # activation=self.config.hidden_act,
                # use_rel_pos=self.config.use_rel_pos,
                # input_size=(self.config.image_size // self.config.patch_size, self.config.image_size // self.config.patch_size),
            )
            self.layers.append(layer)

        self.neck_conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.neck_layer_norm1 = SamLayerNorm(config.output_channels)
        self.neck_conv2 = nn.Conv2d(
            config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False
        )
        self.neck_layer_norm2 = SamLayerNorm(config.output_channels)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(hidden_states)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, mask, ids_restore = self.embeddings(pixel_values, noise=noise)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

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
            return (hidden_states, mask, ids_restore) + all_hidden_states

        return ViTMAEModelOutput(
            last_hidden_state=hidden_states,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

        return hidden_states


class SamPreTrainedModel(PreTrainedModel):
    config_class = SamConfig
    base_model_prefix = "sam"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            pass


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


class SamForImageSegmentation(SamPreTrainedModel):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, config) -> None:
        super().__init__(config)
        if scale is None or scale <= 0.0:
            pass
        self.shared_image_embedding = SamPositionalEmbedding(config)

        self.vision_encoder = SamVisionEncoder(config.vision_config)
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        self.mask_decoder = SamMaskDecoder(config.mask_decoder_config)

    def get_image_wide_positional_embeddings(self, size):
        h, w = size
        device: Any = self.shared_image_embedding.device
        grid = torch.ones((h, w), device=device, dtype=self.shared_image_embedding.dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1)  # C x H x W

    def forward(
        self,
        pixel_values,
        input_points=None,
        input_labels=None,
        input_image_sizes=None,
        input_boxes=None,
        mask_inputs=None,
        original_sizes=None,
        multimask_output: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        image_position_embedding = self.get_image_positional_embeddings(None)
        image_embeddings = self.vision_encoder(pixel_values)[0]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=input_points,
            labels=input_labels,
            boxes=input_boxes,
            masks=mask_inputs,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_position_embedding,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=pixel_values.shape[-2:] if input_image_sizes is None else input_image_sizes[0],
            original_size=original_sizes,
        )
        masks = masks > self.mask_threshold
        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: torch.Tensor,
    ):
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        image_size = (self.vision_encoder.image_size, self.vision_encoder.image_size)
        masks = F.interpolate(masks, image_size, mode="bilinear", align_corners=False)
        masks = masks[..., : input_size[0], : input_size[1]]

        # check if original size is not the same across batches
        output_masks = []
        if original_size.shape[0] > 1:
            for i in range(original_size.shape[0]):
                output_masks.append(
                    F.interpolate(
                        masks[i].unsqueeze(0),
                        (original_size[i][0], original_size[i][1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            output_masks = torch.cat(output_masks, dim=0)
        else:
            output_masks = F.interpolate(
                masks, (original_size[0][0], original_size[0][1]), mode="bilinear", align_corners=False
            )
        return output_masks
