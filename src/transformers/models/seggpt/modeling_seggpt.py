# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch SegGPT model."""


import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_seggpt import SegGPTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SegGPTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "EduardoPacheco/seggpt-vit-large"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]


SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EduardoPacheco/seggpt-vit-large",
    # See all SegGPT models at https://huggingface.co/models?filter=seggpt
]


def patchify(self, tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size = tensor.shape[0]
    patch_height = tensor.shape[2] // patch_size
    patch_width = tensor.shape[3] // patch_size

    tensor = tensor.reshape(shape=(batch_size, 3, patch_height, patch_size, patch_width, patch_size))
    tensor = torch.einsum("nchpwq->nhwpqc", tensor)
    tensor = tensor.reshape(shape=(batch_size, patch_height * patch_width, patch_size**2 * 3))

    return tensor


def unpatchify(self, tensor: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    batch_size = tensor.shape[0]
    patch_size = int((tensor.shape[-1] / 3) ** 0.5)
    assert patch_height * patch_width == tensor.shape[1]

    tensor = tensor.reshape(shape=(batch_size, patch_height, patch_width, patch_size, patch_size, 3))
    tensor = torch.einsum("nhwpqc->nchpwq", tensor)
    tensor = tensor.reshape(shape=(batch_size, 3, patch_height * patch_size, patch_width * patch_size))

    return tensor


@dataclass
class SegGPTEncoderOutput(BaseModelOutput):
    """
    Output type of [`SegGPTEncoderOutput`].
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, patch_height, patch_width, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`Tuple[torch.FloatTensor]`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, patch_height, patch_width, hidden_size)`.
        attentions (`Tuple[torch.FloatTensor]`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, seq_len, seq_len)`.
        intermediate_features (`Tuple[torch.FloatTensor]`, `optional`, returned when ``config.encoder_output_indicies`` is set):
            Tuple of `torch.FloatTensor` of shape `(batch_size, patch_height, patch_width, hidden_size)`.
            Each element in the Tuple corresponds to the output of the layer specified in ``config.encoder_output_indicies``.
            Additionaly, each feature passes through a LayerNorm.
    """

    intermediate_features: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SegGPTImageSegmentationOutput(ModelOutput):
    """
    Output type of [`SegGPTImageSegmentationOutput`].

    Args:
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The predicted masks.
        hidden_states (`Tuple[torch.FloatTensor]`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, patch_height, patch_width, hidden_size)`.
        attentions (`Tuple[torch.FloatTensor]`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, seq_len, seq_len)`.
        loss (`torch.FloatTensor`, `optional`, returned when ``labels`` is provided):
            The loss value.

    """

    pred_masks: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class SegGPTEmbeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings for input and prompt.
    """

    def __init__(self, config: SegGPTConfig) -> None:
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.segment_token_input = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.segment_token_prompt = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        # token for seg types
        self.type_token_semantic = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.type_token_instance = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))

        self.patch_embeddings = SegGPTPatchEmbeddings(config)

        num_positions = (config.pretrain_img_size // config.patch_size) ** 2 + 1
        self.position_embeddings = nn.Parameter(torch.randn(1, num_positions, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        patch_pos_embed = self.position_embeddings[:, 1:]
        num_patches = patch_pos_embed.shape[1]
        pretrain_patch_size = int(math.sqrt(num_patches))
        assert pretrain_patch_size * pretrain_patch_size == num_patches

        if pretrain_patch_size != height or pretrain_patch_size != width:
            patch_pos_embed = F.interpolate(
                patch_pos_embed.reshape(1, pretrain_patch_size, pretrain_patch_size, -1).permute(0, 3, 1, 2),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )

            return patch_pos_embed.permute(0, 2, 3, 1)
        else:
            return patch_pos_embed.reshape(1, height, width, -1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        embedding_type: Optional[str] = None,
    ) -> torch.Tensor:
        input_embeddings = self.patch_embeddings(pixel_values)
        prompt_embeddings = self.patch_embeddings(prompt_pixel_values)

        batch_size, patch_height, patch_width, _ = input_embeddings.shape

        if bool_masked_pos is not None:
            mask_token = self.mask_token.expand(batch_size, patch_height, patch_width, -1)
            # replace the masked visual tokens by mask_token
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, patch_height, patch_width, 1)
            prompt_embeddings = prompt_embeddings * (1 - w) + mask_token * w

        embedding_type = embedding_type if embedding_type is not None else "instance"

        # add positional encoding to each token
        pos_embed = self.interpolate_pos_encoding(patch_height, patch_width)

        # add segment token
        input_embeddings = input_embeddings + self.segment_token_input
        prompt_embeddings = prompt_embeddings + self.segment_token_prompt

        # add position embedding skipping CLS
        input_embeddings = input_embeddings + pos_embed
        prompt_embeddings = prompt_embeddings + pos_embed

        # add type embedding to each token
        if embedding_type == "semantic":
            type_embedding = self.type_token_semantic
        elif embedding_type == "instance":
            type_embedding = self.type_token_instance
        else:
            raise ValueError(f"Embedding type should be either 'semantic' or 'instance', but got {embedding_type}")

        input_embeddings = input_embeddings + type_embedding
        prompt_embeddings = prompt_embeddings + type_embedding

        embeddings = torch.cat((input_embeddings, prompt_embeddings), dim=0)

        return embeddings


# Copied from transformers.models.sam.modeling_sam.SamPatchEmbeddings with Sam->SegGPT
class SegGPTPatchEmbeddings(nn.Module):
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


# Modified from transformers.models.sam.modeling_sam.SamVisionAttention
class SegGPTAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        input_size = (image_size[0] // config.patch_size, image_size[1] // config.patch_size)
        head_dim = config.hidden_size // config.num_attention_heads

        self.num_attention_heads = config.num_attention_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")

            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

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
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_attention_heads, height * width, -1)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_attention_heads, height * width, -1)
        else:
            attn_weights_reshaped = None

        attn_output = (attn_weights @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights_reshaped)
        else:
            outputs = (attn_output, None)

        return outputs


class SegGPTMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop(hidden_states)
        return hidden_states


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->SegGPT
class SegGPTDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SegGPTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: SegGPTConfig, drop_path_rate: float) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SegGPTAttention(config)
        self.mlp = SegGPTMlp(config)
        self.drop_path = SegGPTDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in SegGPT, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # TODO: Add feature_ensemble

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states
        residual = hidden_states

        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.drop_path(hidden_states)

        outputs = (hidden_states,) + outputs

        return outputs


class SegGPTEncoder(nn.Module):
    def __init__(self, config: SegGPTConfig) -> None:
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([SegGPTLayer(config, dpr[i]) for i in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, SegGPTEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        intermediate_features = []

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if i == self.config.merge_index:
                hidden_states = (
                    hidden_states[: hidden_states.shape[0] // 2] + hidden_states[hidden_states.shape[0] // 2 :]
                ) * 0.5

            if i in self.config.encoder_output_indicies:
                intermediate_features.append(self.layernorm(hidden_states))

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, intermediate_features]
                if v is not None
            )
        return SegGPTEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            intermediate_features=intermediate_features,
        )


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->SegGPT
class SegGPTLayerNorm(nn.Module):
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


class SegGPTDecoderHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(
            config.decoder_hidden_size,
            config.decoder_hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.layernorm = SegGPTLayerNorm(
            normalized_shape=config.decoder_hidden_size, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.act_fct = ACT2FN[config.hidden_act]
        self.head = nn.Conv2d(config.decoder_hidden_size, 3, kernel_size=1, bias=True)  # decoder to patch

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.act_fct(hidden_states)
        hidden_states = self.head(hidden_states)

        return hidden_states


class SegGPTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder_embed = nn.Linear(
            config.hidden_size * len(config.encoder_output_indicies),
            config.patch_size**2 * config.decoder_hidden_size,
            bias=True,
        )
        self.decoder_pred = SegGPTDecoderHead(config)
        self.patch_size = config.patch_size
        self.decoder_hidden_size = config.decoder_hidden_size
        self.config = config

    def _reshape_hidden_states(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, patch_height, patch_width, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, patch_height, patch_width, self.patch_size, self.patch_size, self.decoder_hidden_size
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = hidden_states.reshape(
            shape=(batch_size, -1, patch_height * self.patch_size, patch_width * self.patch_size)
        )

        return hidden_states

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states = self.decoder_embed(hidden_states)
        hidden_states = self._reshape_hidden_states(hidden_states)
        hidden_states = self.decoder_pred(hidden_states)

        return hidden_states


class SegGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SegGPTConfig
    base_model_prefix = "seggpt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SegGPTEmbeddings", "SegGPTLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=std).to(
                module.weight.dtype
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, SegGPTAttention):
            module.rel_pos_h.data = nn.init.trunc_normal_(
                module.rel_pos_h.data.to(torch.float32),
                mean=0.0,
                std=std,
            ).to(module.rel_pos_h.dtype)

            module.rel_pos_w.data = nn.init.trunc_normal_(
                module.rel_pos_w.data.to(torch.float32),
                mean=0.0,
                std=std,
            ).to(module.rel_pos_w.dtype)

        elif isinstance(module, SegGPTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=std,
            ).to(module.position_embeddings.dtype)

            torch.nn.init.normal_(module.mask_token, std=std)
            torch.nn.init.normal_(module.segment_token_input, std=std)
            torch.nn.init.normal_(module.segment_token_prompt, std=std)
            torch.nn.init.normal_(module.type_token_semantic, std=std)
            torch.nn.init.normal_(module.type_token_instance, std=std)


SEGGPT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SegGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEGGPT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`SegGPTImageProcessor.__call__`]
            for details.

        prompt_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Prompt pixel values. Prompt pixel values can be obtained using [`AutoImageProcessor`]. See
            [`SegGPTImageProcessor.__call__`] for details.

        prompt_masks (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Prompt mask. Prompt mask can be obtained using [`AutoImageProcessor`]. See [`SegGPTImageProcessor.__call__`] for
            details.

        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        embedding_type (`str`, *optional*):
            Embedding type. Indicates whether the prompt is a semantic or instance embedding. Can be either
            instance or semantic.

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
    "The bare SegGPT Model transformer outputting raw hidden-states without any specific head on top.",
    SEGGPT_START_DOCSTRING,
)
class SegGPTModel(SegGPTPreTrainedModel):
    def __init__(self, config: SegGPTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = SegGPTEmbeddings(config)
        self.encoder = SegGPTEncoder(config)
        self.decoder = SegGPTDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> SegGPTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _prepare_input(
        self,
        pixel_values: torch.FloatTensor,
        prompt_pixel_values: torch.FloatTensor,
        prompt_masks: torch.FloatTensor,
    ) -> Tuple[torch.Tensor]:
        pixel_values = torch.cat((prompt_pixel_values, pixel_values), dim=2)
        prompt_pixel_values = torch.cat((prompt_masks, prompt_masks), dim=2)

        return pixel_values, prompt_pixel_values

    @add_start_docstrings_to_model_forward(SEGGPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SegGPTImageSegmentationOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        prompt_pixel_values: Optional[torch.Tensor] = None,
        prompt_masks: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        embedding_type: Optional[str] = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegGPTImageSegmentationOutput]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, `optional`):
            Ground truth mask for input images.

        Returns:

        Examples:

        ```python
        >>> from transformers import SegGPTImageProcessor, SegGPTModel
        >>> from PIL import Image
        >>> import requests

        >>> image_input_url = (
        "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_2.jpg"
        )
        >>> image_prompt_url = (
            "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1.jpg"
        )
        >>> mask_prompt_url = (
            "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1_target.png"
        )

        >>> image_input = Image.open(requests.get(image_input_url, stream=True).raw)
        >>> image_prompt = Image.open(requests.get(image_prompt_url, stream=True).raw)
        >>> mask_prompt = Image.open(requests.get(mask_prompt_url, stream=True).raw)

        >>> checkpoint = "EduardoPacheco/seggpt-vit-large
        >>> model = SegGPTModel.from_pretrained(checkpoint)
        >>> image_processor = SegGPTImageProcessor.from_checkpoint(checkpoint)

        >>> inputs = image_processor(image_input, image_prompt, mask_prompt)
        >>> outputs = model(**inputs)
        >>> list(outputs.pred_masks)
        [1, 3, 896, 448]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        if prompt_pixel_values is None:
            raise ValueError("You have to specify prompt_pixel_values")
        if prompt_masks is None:
            raise ValueError("You have to specify prompt_masks")

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        if prompt_pixel_values.dtype != expected_dtype:
            prompt_pixel_values = prompt_pixel_values.to(expected_dtype)

        pixel_values, prompt_pixel_values = self._prepare_input(pixel_values, prompt_pixel_values, prompt_masks)

        if bool_masked_pos is None:
            num_patches = self.embeddings.patch_embeddings.num_patches
            bool_masked_pos = torch.zeros(num_patches, dtype=torch.bool).to(pixel_values.device)
            bool_masked_pos[num_patches // 2 :] = 1
            bool_masked_pos = bool_masked_pos.unsqueeze(0)

        embedding_output = self.embeddings(
            pixel_values, prompt_pixel_values, embedding_type=embedding_type, bool_masked_pos=bool_masked_pos
        )

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        intermediate_features = encoder_outputs[-1]
        intermediate_features = torch.cat(intermediate_features, dim=-1)

        pred_masks = self.decoder(intermediate_features)

        loss = None
        if labels is not None:
            loss_fn = SegGPTLoss(beta=self.config.beta)
            loss = loss_fn(pixel_values, prompt_pixel_values, pred_masks, labels, bool_masked_pos)

        if not return_dict:
            output = (pred_masks,)
            if output_hidden_states:
                output = output + (encoder_outputs[1],)

            if output_attentions:
                idx = 2 if output_hidden_states else 1
                output = output + (encoder_outputs[idx],)

            if loss is not None:
                output = output + (loss,)
            return output

        return SegGPTImageSegmentationOutput(
            pred_masks=pred_masks,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            loss=loss,
        )


class SegGPTLoss(nn.Module):
    def __init__(self, beta: float = 0.01):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        prompt_pixel_values: torch.FloatTensor,
        pred_masks: torch.FloatTensor,
        labels: torch.FloatTensor,
        bool_masked_pos: torch.BoolTensor,
    ):
        patch_size = self.config.patch_size
        mask = bool_masked_pos[:, :, None].repeat(1, 1, patch_size**2 * 3)
        mask = unpatchify(mask, pixel_values.shape[1] // patch_size, pixel_values.shape[2] // patch_size)
        # Changing dummy mask in prompt_pixel_values to labels values
        prompt_pixel_values[:, :, prompt_pixel_values.shape[2] // 2 :, :] = labels
        loss = F.smooth_l1_loss(pred_masks, prompt_pixel_values, reduction="none", beta=0.01)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss
