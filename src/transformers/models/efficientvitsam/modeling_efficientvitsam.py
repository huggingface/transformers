# Copyright 2026 The HuggingFace Team. All rights reserved.
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


from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ... import initialization as init
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, auto_docstring, logging
from ...utils.generic import TransformersKwargs, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from .configuration_efficientvitsam import (
    EfficientvitsamConfig,
    EfficientvitsamMaskDecoderConfig,
    EfficientvitsamPromptEncoderConfig,
    EfficientvitsamVisionConfig,
)


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for efficientvitsam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.
    """
)
class EfficientvitsamVisionEncoderOutput(ModelOutput):
    r"""
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The image embeddings obtained by applying the projection layer to the pooler_output.
    """

    image_embeds: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Segment-Anything model's output
    """
)
class EfficientvitsamImageSegmentationOutput(ModelOutput):
    r"""
    iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
        The iou scores of the predicted masks.
    pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
        The predicted low resolutions masks. Needs to be post-processed by the processor
    vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

        Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
    vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
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

    iou_scores: torch.FloatTensor | None = None
    pred_masks: torch.FloatTensor | None = None
    vision_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    vision_attentions: tuple[torch.FloatTensor, ...] | None = None
    mask_decoder_attentions: tuple[torch.FloatTensor, ...] | None = None


class EfficientvitsamMLPBlock(nn.Module):
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


class EfficientvitsamLayerNorm(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, *, eps=1e-6, data_format="channels_last", **kwargs):
        super().__init__(normalized_shape, eps=eps, **kwargs)
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.data_format = data_format

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_first":
            features = features.permute(0, 2, 3, 1)
            features = super().forward(features)
            features = features.permute(0, 3, 1, 2)
        else:
            features = super().forward(features)
        return features


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class EfficientvitsamAttention(nn.Module):
    def __init__(self, config, downsample_rate=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")
        self.scaling = (self.internal_dim // config.num_attention_heads) ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

        self.is_causal = False

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_tokens, n_heads, c_per_head = hidden_states.shape
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_similarity: Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_similarity,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = self._recombine_heads(attn_output, point_batch_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class EfficientvitsamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = EfficientvitsamAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.cross_attn_token_to_image = EfficientvitsamAttention(config, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.mlp = EfficientvitsamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = EfficientvitsamAttention(config, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if self.skip_first_layer_pe:
            queries, _ = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out, _ = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out, _ = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out

        queries = self.layer_norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out, _ = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        keys = self.layer_norm4(keys)
        return queries, keys, attn_out


class EfficientvitsamTwoWayTransformer(nn.Module):
    def __init__(self, config: EfficientvitsamMaskDecoderConfig):
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            self.layers.append(EfficientvitsamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = EfficientvitsamAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput:
        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)

        queries = point_embeddings
        keys = image_embeddings

        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding

            queries, keys, _ = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                **kwargs,
            )
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        attn_out, _ = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys


class EfficientvitsamFeedForward(nn.Module):
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


class EfficientvitsamMaskDecoder(nn.Module):
    def __init__(self, config: EfficientvitsamMaskDecoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = EfficientvitsamTwoWayTransformer(config)

        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = EfficientvitsamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [EfficientvitsamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = EfficientvitsamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        attention_similarity: torch.Tensor | None = None,
        target_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1] if sparse_prompt_embeddings is not None else 1
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings is not None:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        point_embedding, image_embeddings = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

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

        iou_pred = self.iou_prediction_head(iou_token_out)

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]
        return masks, iou_pred


class EfficientvitsamPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.scale
        self.positional_embedding = nn.Parameter(self.scale * torch.randn((2, config.num_pos_feats)))

    def forward(self, input_coords, input_shape=None):
        coordinates = input_coords.clone()

        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]

        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


class EfficientvitsamMaskEmbedding(nn.Module):
    def __init__(self, config: EfficientvitsamPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = EfficientvitsamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = EfficientvitsamLayerNorm(
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


class EfficientvitsamPromptEncoder(nn.Module):
    def __init__(self, config: EfficientvitsamConfig):
        super().__init__()
        self.shared_embedding = EfficientvitsamPositionalEmbedding(config.vision_config)
        config = config.prompt_encoder_config
        self.prompt_encoder_patch_size = config.patch_size
        self.mask_embed = EfficientvitsamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        points = points + 0.5
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        point_embedding = torch.where(labels[..., None] != -10, point_embedding, torch.zeros_like(point_embedding))

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
        boxes = boxes + 0.5
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(
        self,
        input_points: tuple[torch.Tensor, torch.Tensor] | None,
        input_labels: torch.Tensor | None,
        input_boxes: torch.Tensor | None,
        input_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_embeddings = None
        batch_size = 1
        if input_points is not None:
            batch_size = input_points.shape[0]
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

        return sparse_embeddings, dense_embeddings


@auto_docstring
class EfficientvitsamPreTrainedModel(PreTrainedModel):
    config_class = EfficientvitsamConfig
    base_model_prefix = "efficientvitsam"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["EfficientvitsamVisionBackbone"]
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        if isinstance(module, EfficientvitsamPositionalEmbedding):
            init.normal_(module.positional_embedding, std=module.scale)


def list_sum(values: list[torch.Tensor]) -> torch.Tensor:
    return values[0] if len(values) == 1 else values[0] + list_sum(values[1:])


def val2list(value, repeat_time: int = 1) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value for _ in range(repeat_time)]


def val2tuple(value, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    value = val2list(value)
    if len(value) > 0:
        value[idx_repeat:idx_repeat] = [value[idx_repeat] for _ in range(min_len - len(value))]
    return tuple(value)


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple(get_same_padding(k) for k in kernel_size)
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd.")
    return kernel_size // 2


def resize(
    x: torch.Tensor,
    size=None,
    scale_factor=None,
    mode: str = "bicubic",
    align_corners: bool | None = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)


def build_act(name: str | None, **kwargs) -> nn.Module | None:
    if name is None:
        return None
    if name == "relu":
        return nn.ReLU(**kwargs)
    if name == "relu6":
        return nn.ReLU6(**kwargs)
    if name == "hswish":
        return nn.Hardswish(**kwargs)
    if name == "silu":
        return nn.SiLU(**kwargs)
    if name == "gelu":
        return nn.GELU(approximate="tanh", **kwargs)
    raise ValueError(f"Unsupported activation: {name}")


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


def build_norm(name: str | None = "bn2d", num_features: int | None = None) -> nn.Module | None:
    if name is None:
        return None
    if name == "bn2d":
        return nn.BatchNorm2d(num_features)
    if name == "ln":
        return nn.LayerNorm(num_features)
    if name == "ln2d":
        return LayerNorm2d(num_features)
    raise ValueError(f"Unsupported norm: {name}")


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0.0,
        norm: str | None = "bn2d",
        act_func: str | None = "relu",
    ) -> None:
        super().__init__()
        padding = get_same_padding(kernel_size)
        padding = padding * dilation
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode: str = "bicubic",
        size: int | tuple[int, int] | list[int] | None = None,
        factor=2,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == tuple(self.size)) or self.factor == 1:
            return x
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module | None]) -> None:
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module | None,
        shortcut: nn.Module | None,
        post_act: str | None = None,
        pre_norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) if self.pre_norm is None else self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act is not None:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: nn.Module | None,
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ) -> None:
        super().__init__()
        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input
        self.middle = middle
        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.cat(feat, dim=1)
        else:
            raise NotImplementedError(f"Unsupported merge: {self.merge}")

        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ) -> None:
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(x))


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ) -> None:
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        return self.point_conv(x)


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ) -> None:
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.spatial_conv(x))


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ) -> None:
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class LiteMLA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func: str = "relu",
        scales: tuple[int, ...] = (5,),
        eps: float = 1e-15,
    ) -> None:
        super().__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = list(qkv.size())
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        qkv = torch.reshape(qkv, (batch_size, -1, 3 * self.dim, height * width))
        q, k, v = qkv[:, :, : self.dim], qkv[:, :, self.dim : 2 * self.dim], qkv[:, :, 2 * self.dim :]
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return torch.reshape(out, (batch_size, -1, height, width))

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = list(qkv.size())
        qkv = torch.reshape(qkv, (batch_size, -1, 3 * self.dim, height * width))
        q, k, v = qkv[:, :, : self.dim], qkv[:, :, self.dim : 2 * self.dim], qkv[:, :, 2 * self.dim :]
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        att_map = torch.matmul(k.transpose(-1, -2), q)
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)
        return torch.reshape(out, (batch_size, -1, height, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)
        height, width = list(qkv.size())[-2:]
        if height * width > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        return self.proj(out)


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim: int = 32,
        expand_ratio: float = 4.0,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
    ) -> None:
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            ),
            IdentityLayer(),
        )
        self.local_module = ResidualBlock(
            MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
            ),
            IdentityLayer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        return self.local_module(x)


class EfficientvitsamVisionBackbone(nn.Module):
    def __init__(self, config: EfficientvitsamVisionConfig) -> None:
        super().__init__()
        self.variant = config.variant
        width_list = list(config.width_list)
        depth_list = list(config.depth_list)
        block_list = list(config.block_list)
        expand_list = list(config.expand_list)
        fewer_norm_list = list(config.fewer_norm_list)
        self.width_list = []
        self.stages = []

        stage0 = [
            ConvLayer(
                in_channels=config.num_channels,
                out_channels=width_list[0],
                stride=2,
                norm=config.norm,
                act_func=config.act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=config.norm,
                act_func=config.act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (width, depth) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block_name = "mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id]
            block = self.build_local_block(
                block=block_name,
                in_channels=in_channels,
                out_channels=width,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=config.norm,
                act_func=config.act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = width
            for _ in range(depth):
                if block_list[stage_id].startswith("att"):
                    scales = (3,) if block_list[stage_id] == "att@3" else (5,)
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=config.qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=scales,
                            norm=config.norm,
                            act_func=config.act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=config.norm,
                        act_func=config.act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    stage.append(ResidualBlock(block, IdentityLayer()))
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if block == "res":
            return ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        if block == "fmb":
            return FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        if block == "mb":
            return MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        raise ValueError(block)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict[f"stage{stage_id}"] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


class EfficientvitsamSamNeck(DAGBlock):
    def __init__(self, config: EfficientvitsamVisionConfig):
        inputs = {}
        for fid, in_channel in zip(config.neck_feature_names, config.neck_hidden_sizes):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(in_channel, config.head_width, 1, norm=config.norm, act_func=None),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        middle = []
        for _ in range(config.head_depth):
            if config.head_middle_op == "mb":
                block = MBConv(
                    config.head_width,
                    config.head_width,
                    expand_ratio=config.head_expand_ratio,
                    norm=config.norm,
                    act_func=(config.act_func, config.act_func, None),
                )
            elif config.head_middle_op == "fmb":
                block = FusedMBConv(
                    config.head_width,
                    config.head_width,
                    expand_ratio=config.head_expand_ratio,
                    norm=config.norm,
                    act_func=(config.act_func, None),
                )
            elif config.head_middle_op == "res":
                block = ResBlock(
                    config.head_width,
                    config.head_width,
                    expand_ratio=config.head_expand_ratio,
                    norm=config.norm,
                    act_func=(config.act_func, None),
                )
            else:
                raise NotImplementedError(f"Unsupported neck op: {config.head_middle_op}")
            middle.append(ResidualBlock(block, IdentityLayer()))

        outputs = {
            "sam_encoder": OpSequential(
                [
                    ConvLayer(
                        config.head_width,
                        config.output_channels,
                        1,
                        use_bias=True,
                        norm=None,
                        act_func=None,
                    )
                ]
            )
        }
        super().__init__(inputs, "add", None, OpSequential(middle), outputs)


class EfficientvitsamVisionEncoder(nn.Module):
    def __init__(self, config: EfficientvitsamVisionConfig):
        super().__init__()
        self.config = config
        self.backbone = EfficientvitsamVisionBackbone(config)
        self.neck = EfficientvitsamSamNeck(config)
        self.norm = build_norm("ln2d", config.output_channels)
        self._set_norm_eps(config.layer_norm_eps)

    def _set_norm_eps(self, eps: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
                module.eps = eps

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.stages[0].op_list[0].conv

    def forward(self, pixel_values: torch.FloatTensor | None = None, **kwargs) -> EfficientvitsamVisionEncoderOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        feed_dict = self.backbone(pixel_values)
        feed_dict = self.neck(feed_dict)
        output = self.norm(feed_dict["sam_encoder"])
        return EfficientvitsamVisionEncoderOutput(last_hidden_state=output)


@auto_docstring(
    custom_intro="""
    The vision model from EfficientViT-SAM without any head or projection on top.
    """
)
class EfficientvitsamVisionModel(EfficientvitsamPreTrainedModel):
    config_class = EfficientvitsamVisionConfig
    config: EfficientvitsamVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: EfficientvitsamVisionConfig):
        super().__init__(config)
        self.vision_encoder = EfficientvitsamVisionEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_encoder.get_input_embeddings()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | EfficientvitsamVisionEncoderOutput:
        return self.vision_encoder(pixel_values, **kwargs)


@auto_docstring(
    custom_intro="""
    EfficientViT-SAM model for generating segmentation masks, using an EfficientViT image encoder
    with SAM prompt and mask decoders.
    """
)
class EfficientvitsamModel(EfficientvitsamPreTrainedModel):
    config_class = EfficientvitsamConfig
    input_modalities = ("image", "text")
    _can_record_outputs = {"mask_decoder_attentions": OutputRecorder(EfficientvitsamTwoWayAttentionBlock, index=2)}
    _tied_weights_keys = {
        "prompt_encoder.shared_embedding.positional_embedding": "shared_image_embedding.positional_embedding"
    }

    def __init__(self, config: EfficientvitsamConfig):
        super().__init__(config)
        self.shared_image_embedding = EfficientvitsamPositionalEmbedding(config.vision_config)

        self.vision_encoder = EfficientvitsamVisionEncoder(config.vision_config)
        self.prompt_encoder = EfficientvitsamPromptEncoder(config)
        config.mask_decoder_config._attn_implementation = config._attn_implementation
        self.mask_decoder = EfficientvitsamMaskDecoder(config.mask_decoder_config)
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
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def get_image_embeddings(self, pixel_values, **kwargs: Unpack[TransformersKwargs]):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
        """
        vision_output = self.vision_encoder(
            pixel_values,
            **kwargs,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: torch.FloatTensor | None = None,
        input_labels: torch.LongTensor | None = None,
        input_boxes: torch.FloatTensor | None = None,
        input_masks: torch.LongTensor | None = None,
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

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_points: torch.FloatTensor | None = None,
        input_labels: torch.LongTensor | None = None,
        input_boxes: torch.FloatTensor | None = None,
        input_masks: torch.LongTensor | None = None,
        image_embeddings: torch.FloatTensor | None = None,
        multimask_output: bool = True,
        attention_similarity: torch.FloatTensor | None = None,
        target_embedding: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EfficientvitsamImageSegmentationOutput:
        r"""
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
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box
        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            EfficientViT-SAM also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
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
            model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("mit-han-lab/efficientvit-sam")
        >>> processor = AutoProcessor.from_pretrained("mit-han-lab/efficientvit-sam")

        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/efficientvitsam-car.png"
        >>> with httpx.stream("GET", url) as response:
        ...     raw_image = Image.open(BytesIO(response.read())).convert("RGB")
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
        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                f" got {input_points.shape}.",
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                f" got {input_boxes.shape}.",
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    f"You should provide as many bounding boxes as input points per box. Got {point_batch_size} and {box_batch_size}."
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs: EfficientvitsamVisionEncoderOutput = self.vision_encoder(pixel_values, **kwargs)
            image_embeddings = vision_outputs.last_hidden_state
            vision_hidden_states = vision_outputs.hidden_states
            vision_attentions = vision_outputs.attentions

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                f"Got {image_embeddings.shape[0]} and {input_points.shape[0]} respectively.",
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
        )

        return EfficientvitsamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
        )


__all__ = [
    "EfficientvitsamImageSegmentationOutput",
    "EfficientvitsamModel",
    "EfficientvitsamPreTrainedModel",
    "EfficientvitsamVisionEncoderOutput",
    "EfficientvitsamVisionModel",
]
