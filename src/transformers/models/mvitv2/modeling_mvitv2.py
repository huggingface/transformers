# coding=utf-8
# Copyright 2025 Meta and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MViTV2 model."""

import operator
from functools import reduce
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from .configuration_mvitv2 import MViTV2Config


logger = logging.get_logger(__name__)


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/6a621b5b1cd88d5286067acdbef2057adb3cef27/timm/layers/drop.py#L150
def drop_path(
    hidden_states: torch.Tensor, drop_rate: float = 0.0, training: bool = False, scale_by_keep: bool = True
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_rate == 0.0 or not training:
        return hidden_states
    keep_prob = 1 - drop_rate
    shape = (hidden_states.shape[0],) + (1,) * (
        hidden_states.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = hidden_states.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return hidden_states * random_tensor


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/6a621b5b1cd88d5286067acdbef2057adb3cef27/timm/layers/drop.py#L170
class MViTV2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_rate: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_rate, self.training, self.scale_by_keep)


def product(iterable):
    return reduce(operator.mul, iterable, 1)


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/6a621b5b1cd88d5286067acdbef2057adb3cef27/timm/models/mvitv2.py#L148
def calculate_relative_positional_embeddings(
    attention: torch.Tensor,
    query: torch.Tensor,
    use_cls_token: bool,
    query_size: tuple[int, int],
    key_size: tuple[int, int],
    relative_positional_embeddings_height: torch.Tensor,
    relative_positional_embeddings_width: torch.Tensor,
) -> torch.Tensor:
    """
    Spatial Relative Positional Embeddings (offsets added to attention scores)
    """
    cls_token_offset = 1 if use_cls_token else 0
    query_height, query_width = query_size
    key_height, key_width = key_size

    # Scale up relative positions if shapes for query and key are different.
    query_height_ratio = max(key_height / query_height, 1.0)
    key_height_ratio = max(query_height / key_height, 1.0)
    relative_distances_height = (
        torch.arange(query_height, device=query.device).unsqueeze(-1) * query_height_ratio
        - torch.arange(key_height, device=query.device).unsqueeze(0) * key_height_ratio
    )
    relative_distances_height += (key_height - 1) * key_height_ratio
    query_width_ratio = max(key_width / query_width, 1.0)
    key_width_ratio = max(query_width / key_width, 1.0)
    relative_distances_width = (
        torch.arange(query_width, device=query.device).unsqueeze(-1) * query_width_ratio
        - torch.arange(key_width, device=query.device).unsqueeze(0) * key_width_ratio
    )
    relative_distances_width += (key_width - 1) * key_width_ratio

    relative_embeddings_height = relative_positional_embeddings_height[relative_distances_height.long()]
    relative_embeddings_width = relative_positional_embeddings_width[relative_distances_width.long()]

    batch_size, num_heads, _, feature_dim = query.shape

    reshaped_query = query[:, :, cls_token_offset:].reshape(
        batch_size, num_heads, query_height, query_width, feature_dim
    )
    relative_embeddings_height = torch.einsum("byhwc,hkc->byhwk", reshaped_query, relative_embeddings_height)
    relative_embeddings_width = torch.einsum("byhwc,wkc->byhwk", reshaped_query, relative_embeddings_width)

    attention[:, :, cls_token_offset:, cls_token_offset:] = (
        attention[:, :, cls_token_offset:, cls_token_offset:].view(
            batch_size, -1, query_height, query_width, key_height, key_width
        )
        + relative_embeddings_height.unsqueeze(-1)
        + relative_embeddings_width.unsqueeze(-2)
    ).view(batch_size, -1, query_height * query_width, key_height * key_width)

    return attention


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/6a621b5b1cd88d5286067acdbef2057adb3cef27/timm/models/mvitv2.py#L119
def reshape_pre_pool(
    representations: torch.Tensor, feature_size: tuple[int, int], use_cls_token: bool = True
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    height, width = feature_size
    if use_cls_token:
        cls_token, representations = representations[:, :, :1, :], representations[:, :, 1:, :]
    else:
        cls_token = None
    representations = (
        representations.reshape(-1, height, width, representations.shape[-1]).permute(0, 3, 1, 2).contiguous()
    )
    return representations, cls_token


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/6a621b5b1cd88d5286067acdbef2057adb3cef27/timm/models/mvitv2.py#L134
def reshape_post_pool(
    representations: torch.Tensor, num_heads: int, cls_token: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, tuple[int, int]]:
    feature_size = [representations.shape[2], representations.shape[3]]
    num_patches = representations.shape[2] * representations.shape[3]
    representations = representations.reshape(-1, num_heads, representations.shape[1], num_patches).transpose(2, 3)
    if cls_token is not None:
        representations = torch.cat((cls_token, representations), dim=2)
    return representations, feature_size


class MViTV2PatchEmbeddings(nn.Module):
    def __init__(self, config: MViTV2Config) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_kernel_size,
            stride=config.patch_stride_size,
            padding=config.patch_padding_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.proj(pixel_values)
        return embeddings.flatten(2).transpose(1, 2)


class MViTV2Embeddings(nn.Module):
    def __init__(self, config: MViTV2Config, num_patches: int) -> None:
        super().__init__()
        self.patch_embeddings = MViTV2PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_cls_token else None
        self.absolute_positional_embeddings = (
            nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
            if config.use_absolute_positional_embeddings
            else None
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if self.absolute_positional_embeddings is not None:
            embeddings = embeddings + self.absolute_positional_embeddings
        return embeddings


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/6a621b5b1cd88d5286067acdbef2057adb3cef27/timm/models/mvitv2.py#L198
class MViTV2SelfAttentionPoolingFirst(nn.Module):
    def __init__(
        self,
        config: MViTV2Config,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        feature_size: tuple[int, int],
        kernel_q: tuple[int, int],
        kernel_kv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        # Logic specific to Attention Pooling First. In this layer the feature dimension is expanded after pooling, so the initial
        # number of heads should be smaller.
        self.initial_num_heads = num_heads if in_dim == out_dim else num_heads // 2
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim**-0.5

        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        self.query_projection = nn.Linear(in_dim, out_dim, bias=config.qkv_bias)
        self.key_projection = nn.Linear(in_dim, out_dim, bias=config.qkv_bias)
        self.value_projection = nn.Linear(in_dim, out_dim, bias=config.qkv_bias)

        # Skip pooling with kernel and stride size of (1, 1).
        if product(kernel_q) == 1 and product(stride_q) == 1:
            kernel_q = None
        if product(kernel_kv) == 1 and product(stride_kv) == 1:
            kernel_kv = None

        self.pool_query, self.pool_key, self.pool_value = None, None, None
        self.norm_query, self.norm_key, self.norm_value = None, None, None
        if config.mode in ("avg", "max"):
            pool_operation = nn.MaxPool2d if config.mode == "max" else nn.AvgPool2d
            if kernel_q:
                self.pool_query = pool_operation(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_key = pool_operation(kernel_kv, stride_kv, padding_kv)
                self.pool_value = pool_operation(kernel_kv, stride_kv, padding_kv)
        elif config.mode in ("conv", "conv_unshared"):
            head_dim = in_dim // self.initial_num_heads if config.mode == "conv" else in_dim
            if kernel_q:
                self.pool_query = nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                self.norm_query = nn.LayerNorm(head_dim)
            if kernel_kv:
                self.pool_key = nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                self.norm_key = nn.LayerNorm(head_dim)
                self.pool_value = nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                self.norm_value = nn.LayerNorm(head_dim)
        else:
            raise NotImplementedError(f"Unsupported mode {config.mode}")

        # relative positional embedding
        if self.config.relative_positional_embeddings_type == "spatial":
            if feature_size[0] != feature_size[1]:
                raise ValueError(
                    "Relative positional embeddings can only be used with square feature maps (height must equal width)"
                )
            size = feature_size[0]
            q_size = size // stride_q[0]
            kv_size = size // stride_kv[0]
            relative_distance_max_range = 2 * max(q_size, kv_size) - 1

            self.relative_positional_embeddings_height = nn.Parameter(
                torch.zeros(relative_distance_max_range, self.head_dim)
            )
            self.relative_positional_embeddings_width = nn.Parameter(
                torch.zeros(relative_distance_max_range, self.head_dim)
            )

    def forward(
        self, hidden_states: torch.Tensor, feature_size: tuple[int, int], output_attentions: Optional[bool] = None
    ) -> tuple[torch.Tensor, tuple[int, int], Optional[torch.Tensor]]:
        batch_size, num_patches, _ = hidden_states.shape

        num_folds = 1 if self.config.mode == "conv_unshared" else self.initial_num_heads
        hidden_states = hidden_states.reshape(batch_size, num_patches, num_folds, -1).permute(0, 2, 1, 3)
        query = key = value = hidden_states

        if self.pool_query is not None:
            query, query_cls_token = reshape_pre_pool(query, feature_size, self.config.use_cls_token)
            query = self.pool_query(query)
            query, query_size = reshape_post_pool(query, num_folds, query_cls_token)
        else:
            query_size = feature_size
        if self.norm_query is not None:
            query = self.norm_query(query)

        if self.pool_key is not None:
            key, key_cls_token = reshape_pre_pool(key, feature_size, self.config.use_cls_token)
            key = self.pool_key(key)
            key, key_size = reshape_post_pool(key, num_folds, key_cls_token)
        else:
            key_size = feature_size
        if self.norm_key is not None:
            key = self.norm_key(key)

        if self.pool_value is not None:
            value, value_cls_token = reshape_pre_pool(value, feature_size, self.config.use_cls_token)
            value = self.pool_value(value)
            value, value_size = reshape_post_pool(value, num_folds, value_cls_token)
        else:
            value_size = feature_size
        if self.norm_value is not None:
            value = self.norm_value(value)

        query_num_patches = query_size[0] * query_size[1] + int(self.config.use_cls_token)
        query = query.transpose(1, 2).reshape(batch_size, query_num_patches, -1)
        query = self.query_projection(query).reshape(batch_size, query_num_patches, self.num_heads, -1).transpose(1, 2)

        key_num_patches = key_size[0] * key_size[1] + int(self.config.use_cls_token)
        key = key.transpose(1, 2).reshape(batch_size, key_num_patches, -1)
        key = self.key_projection(key).reshape(batch_size, key_num_patches, self.num_heads, -1).transpose(1, 2)

        value_num_patches = value_size[0] * value_size[1] + int(self.config.use_cls_token)
        value = value.transpose(1, 2).reshape(batch_size, value_num_patches, -1)
        value = self.value_projection(value).reshape(batch_size, value_num_patches, self.num_heads, -1).transpose(1, 2)

        attention = (query * self.scale) @ key.transpose(-2, -1)
        if self.config.relative_positional_embeddings_type == "spatial":
            attention = calculate_relative_positional_embeddings(
                attention,
                query,
                self.config.use_cls_token,
                query_size,
                key_size,
                self.relative_positional_embeddings_height,
                self.relative_positional_embeddings_width,
            )
        attention = attention.softmax(dim=-1)
        hidden_states = attention @ value

        if self.config.residual_pooling:
            # as in the original implementation, the cls token is skipped in the residual connection
            if self.config.use_cls_token:
                hidden_states[:, :, 1:, :] = hidden_states[:, :, 1:, :] + query[:, :, 1:, :]
            else:
                hidden_states = hidden_states + query

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.out_dim)
        outputs = (hidden_states, query_size)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)
        return outputs


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/6a621b5b1cd88d5286067acdbef2057adb3cef27/timm/models/mvitv2.py#L368
class MViTV2SelfAttention(nn.Module):
    def __init__(
        self,
        config: MViTV2Config,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        feature_size: tuple[int, int],
        kernel_q: tuple[int, int],
        kernel_kv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
    ) -> None:
        super().__init__()
        self.config = config
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        self.qkv = nn.Linear(in_dim, out_dim * 3, bias=config.qkv_bias)

        # Skip pooling with kernel and stride size of (1, 1).
        if product(kernel_q) == 1 and product(stride_q) == 1:
            kernel_q = None
        if product(kernel_kv) == 1 and product(stride_kv) == 1:
            kernel_kv = None

        mode = config.mode
        self.norm_query, self.norm_key, self.norm_value = None, None, None
        self.pool_query, self.pool_key, self.pool_value = None, None, None
        if mode in ("avg", "max"):
            pool_operation = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            if kernel_q:
                self.pool_query = pool_operation(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_key = pool_operation(kernel_kv, stride_kv, padding_kv)
                self.pool_value = pool_operation(kernel_kv, stride_kv, padding_kv)
        elif mode == "conv":
            head_dim = out_dim // num_heads
            if kernel_q:
                self.pool_query = nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                self.norm_query = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)
            if kernel_kv:
                self.pool_key = nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                self.norm_key = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)
                self.pool_value = nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                self.norm_value = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)
        else:
            raise NotImplementedError(f"Unsupported mode {mode}")

        if self.config.relative_positional_embeddings_type == "spatial":
            assert feature_size[0] == feature_size[1]
            size = feature_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 1 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 1 else size
            relative_distance_max_range = 2 * max(q_size, kv_size) - 1

            self.relative_positional_embeddings_height = nn.Parameter(
                torch.zeros(relative_distance_max_range, self.head_dim)
            )
            self.relative_positional_embeddings_width = nn.Parameter(
                torch.zeros(relative_distance_max_range, self.head_dim)
            )

    def forward(
        self, hidden_states: torch.Tensor, feature_size: tuple[int, int], output_attentions: Optional[bool] = None
    ) -> tuple[torch.Tensor, tuple[int, int], Optional[torch.Tensor]]:
        batch_size, num_patches, _ = hidden_states.shape

        qkv = self.qkv(hidden_states).reshape(batch_size, num_patches, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        if self.pool_query is not None:
            query, query_token = reshape_pre_pool(query, feature_size, self.config.use_cls_token)
            query = self.pool_query(query)
            query, query_size = reshape_post_pool(query, self.num_heads, query_token)
        else:
            query_size = feature_size
        if self.norm_query is not None:
            query = self.norm_query(query)

        if self.pool_key is not None:
            key, key_token = reshape_pre_pool(key, feature_size, self.config.use_cls_token)
            key = self.pool_key(key)
            key, key_size = reshape_post_pool(key, self.num_heads, key_token)
        else:
            key_size = feature_size
        if self.norm_key is not None:
            key = self.norm_key(key)

        if self.pool_value is not None:
            value, value_token = reshape_pre_pool(value, feature_size, self.config.use_cls_token)
            value = self.pool_value(value)
            value, _ = reshape_post_pool(value, self.num_heads, value_token)
        if self.norm_value is not None:
            value = self.norm_value(value)

        attention = (query * self.scale) @ key.transpose(-2, -1)
        if self.config.relative_positional_embeddings_type == "spatial":
            attention = calculate_relative_positional_embeddings(
                attention,
                query,
                self.config.use_cls_token,
                query_size,
                key_size,
                self.relative_positional_embeddings_height,
                self.relative_positional_embeddings_width,
            )
        attention = attention.softmax(dim=-1)
        hidden_states = attention @ value

        if self.config.residual_pooling:
            # as in the original implementation, the cls token is skipped in the residual connection
            if self.config.use_cls_token:
                hidden_states[:, :, 1:, :] = hidden_states[:, :, 1:, :] + query[:, :, 1:, :]
            else:
                hidden_states = hidden_states + query

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.out_dim)
        outputs = (hidden_states, query_size)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)
        return outputs


# minimal class for compatibility with HF
class MViTV2SelfOutput(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states


class MViTV2AttentionPoolingFirst(nn.Module):
    def __init__(
        self,
        config: MViTV2Config,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        feature_size: tuple[int, int],
        kernel_q: tuple[int, int],
        kernel_kv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
    ) -> None:
        super().__init__()
        self.attention = MViTV2SelfAttentionPoolingFirst(
            config, in_dim, out_dim, num_heads, feature_size, kernel_q, kernel_kv, stride_q, stride_kv
        )
        self.output = MViTV2SelfOutput(out_dim)

    def forward(
        self, hidden_states: torch.Tensor, feature_size: tuple[int, int], output_attentions: Optional[bool] = None
    ) -> tuple[torch.Tensor, tuple[int, int], Optional[torch.Tensor]]:
        hidden_states, new_feature_size, attentions = self.attention(hidden_states, feature_size, output_attentions)
        hidden_states = self.output(hidden_states)
        return hidden_states, new_feature_size, attentions


class MViTV2Attention(nn.Module):
    def __init__(
        self,
        config: MViTV2Config,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        feature_size: tuple[int, int],
        kernel_q: tuple[int, int],
        kernel_kv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
    ) -> None:
        super().__init__()
        self.attention = MViTV2SelfAttention(
            config, in_dim, out_dim, num_heads, feature_size, kernel_q, kernel_kv, stride_q, stride_kv
        )
        self.output = MViTV2SelfOutput(out_dim)

    def forward(
        self, hidden_states: torch.Tensor, feature_size: tuple[int, int], output_attentions: Optional[bool] = None
    ) -> tuple[torch.Tensor, tuple[int, int], Optional[torch.Tensor]]:
        hidden_states, new_feature_size, attentions = self.attention(hidden_states, feature_size, output_attentions)
        hidden_states = self.output(hidden_states)
        return hidden_states, new_feature_size, attentions


class MViTV2Intermediate(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, config: MViTV2Config) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, hidden_dim)
        if isinstance(config.hidden_activation_function, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_activation_function]
        else:
            self.intermediate_act_fn = config.hidden_activation_function

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# minimal class for compatibility with HF
class MViTV2Output(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/6a621b5b1cd88d5286067acdbef2057adb3cef27/timm/models/mvitv2.py#L521
class MViTV2Block(nn.Module):
    def __init__(
        self,
        config: MViTV2Config,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        feature_size: tuple[int, int],
        kernel_qkv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.config = config
        projection_needed = in_dim != out_dim
        # two shortcuts as per the original implementation
        self.shortcut_project_feature_dim = nn.Linear(in_dim, out_dim) if projection_needed else None
        if stride_q and product(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(s // 2) for s in kernel_skip]
            self.shortcut_pool_features = nn.MaxPool2d(kernel_skip, stride_skip, padding_skip)
        else:
            self.shortcut_pool_features = None

        self.layernorm_before = nn.LayerNorm(in_dim, eps=config.layer_norm_epsilon)
        attention_class = MViTV2AttentionPoolingFirst if config.attention_pool_first else MViTV2Attention
        attention_out_dim = out_dim if config.expand_feature_dimension_in_attention else in_dim
        self.attention = attention_class(
            config, in_dim, attention_out_dim, num_heads, feature_size, kernel_qkv, kernel_qkv, stride_q, stride_kv
        )

        self.layernorm_after = nn.LayerNorm(attention_out_dim, eps=config.layer_norm_epsilon)
        hidden_intermediate_dim = attention_out_dim * config.mlp_ratio
        self.intermediate = MViTV2Intermediate(attention_out_dim, hidden_intermediate_dim, config)
        self.output = MViTV2Output(hidden_intermediate_dim, out_dim)

        self.drop_path = MViTV2DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def _shortcut_pool(self, hidden_states: torch.Tensor, feature_size: tuple[int, int]) -> torch.Tensor:
        if self.shortcut_pool_features is None:
            return hidden_states
        if self.config.use_cls_token:
            cls_token, hidden_states = hidden_states[:, :1, :], hidden_states[:, 1:, :]
        else:
            cls_token = None
        batch_size, _, feature_dim = hidden_states.shape
        height, width = feature_size
        hidden_states = hidden_states.reshape(batch_size, height, width, feature_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.shortcut_pool_features(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, feature_dim, -1).transpose(1, 2)
        if cls_token is not None:
            hidden_states = torch.cat((cls_token, hidden_states), dim=1)
        return hidden_states

    def forward(
        self, hidden_states: torch.Tensor, feature_size: tuple[int, int], output_attentions: Optional[bool] = None
    ) -> tuple[torch.Tensor, tuple[int, int], Optional[torch.Tensor]]:
        # like in the original implementation, if no projection is needed, unnormalized inputs are used as shortcuts
        hidden_states_normalized = self.layernorm_before(hidden_states)
        if self.shortcut_project_feature_dim and self.config.expand_feature_dimension_in_attention:
            hidden_states_shortcut = self.shortcut_project_feature_dim(hidden_states_normalized)
        else:
            hidden_states_shortcut = hidden_states
        hidden_states_shortcut = self._shortcut_pool(hidden_states_shortcut, feature_size)
        attention_output, new_feature_size, attentions = self.attention(
            hidden_states_normalized, feature_size, output_attentions
        )
        hidden_states = hidden_states_shortcut + self.drop_path(attention_output)

        hidden_states_normalized = self.layernorm_after(hidden_states)
        if self.shortcut_project_feature_dim and not self.config.expand_feature_dimension_in_attention:
            hidden_states_shortcut = self.shortcut_project_feature_dim(hidden_states_normalized)
        else:
            hidden_states_shortcut = hidden_states
        hidden_states = self.intermediate(hidden_states_normalized)
        hidden_states = self.output(hidden_states)
        hidden_states = hidden_states_shortcut + self.drop_path(hidden_states)
        return hidden_states, new_feature_size, attentions


class MViTV2Stage(nn.Module):
    def __init__(
        self,
        config: MViTV2Config,
        depth: int,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        feature_size: tuple[int, int],
        kernel_qkv: tuple[int, int],
        stride_q: tuple[int, int],
        stride_kv: tuple[int, int],
        drop_path_rate: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False

        self.blocks = nn.ModuleList()
        if config.expand_feature_dimension_in_attention:
            out_dims = [out_dim] * depth
        else:
            out_dims = [in_dim] * (depth - 1) + [out_dim]
        for i in range(depth):
            block = MViTV2Block(
                config=config,
                in_dim=in_dim,
                out_dim=out_dims[i],
                num_heads=num_heads,
                feature_size=feature_size,
                kernel_qkv=kernel_qkv,
                stride_q=stride_q if i == 0 else (1, 1),
                stride_kv=stride_kv,
                drop_path_rate=drop_path_rate[i],
            )
            if i == 0:
                feature_size = tuple(feature_size[j] // stride_q[j] for j in range(len(feature_size)))

            in_dim = out_dims[i]
            self.blocks.append(block)

    def forward(
        self,
        hidden_states: torch.Tensor,
        feature_size: tuple[int, int],
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> tuple[torch.Tensor, tuple[int, int], Optional[tuple[torch.Tensor, ...]], Optional[tuple[torch.Tensor, ...]]]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states, feature_size, attentions = self._gradient_checkpointing_func(
                    block.__call__, hidden_states, feature_size, output_attentions
                )
            else:
                hidden_states, feature_size, attentions = block(hidden_states, feature_size, output_attentions)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if output_attentions:
                all_self_attentions += (attentions,)
        return hidden_states, feature_size, all_self_attentions, all_hidden_states


class MViTV2Encoder(nn.Module):
    def __init__(self, config: MViTV2Config, feature_size: tuple[int, int]) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        drop_path_rate = [
            x.tolist()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu").split(config.depths)
        ]

        num_stages = len(config.depths)
        num_heads = tuple([config.num_heads * (2**i) for i in range(num_stages)])
        base_dims = tuple([config.hidden_size * (2**i) for i in range(num_stages)])
        if config.expand_feature_dimension_in_attention:
            in_dims = (base_dims[0],) + base_dims[:-1]
            out_dims = base_dims
        else:
            in_dims = base_dims
            out_dims = base_dims[1:] + (base_dims[-1],)

        if config.stride_kv_adaptive is not None and config.stride_kv is None:
            _stride_kv = config.stride_kv_adaptive
            pool_kv_stride = []
            for i in range(num_stages):
                if min(config.stride_q[i]) > 1:
                    _stride_kv = [max(_stride_kv[d] // config.stride_q[i][d], 1) for d in range(len(_stride_kv))]
                pool_kv_stride.append(tuple(_stride_kv))
            stride_kv = tuple(pool_kv_stride)

        for i in range(num_stages):
            stage = MViTV2Stage(
                config=config,
                depth=config.depths[i],
                in_dim=in_dims[i],
                out_dim=out_dims[i],
                num_heads=num_heads[i],
                feature_size=feature_size,
                kernel_qkv=tuple(config.kernel_qkv),
                stride_q=tuple(config.stride_q[i]),
                stride_kv=stride_kv[i],
                drop_path_rate=drop_path_rate[i],
            )
            feature_size = tuple(feature_size[j] // config.stride_q[i][j] for j in range(len(feature_size)))
            self.stages.append(stage)
        # the final LayerNorm is in the encoder for simplicity (its output is considered an intermediate hidden state in MViTv2)
        self.layer_norm = nn.LayerNorm(out_dims[-1], eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        feature_size: tuple[int, int],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> Union[tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], BaseModelOutput]:
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for stage in self.stages:
            hidden_states, feature_size, attentions, intermediate_hidden_states = stage(
                hidden_states,
                feature_size,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if all_self_attentions is not None:
                all_self_attentions = all_self_attentions + attentions
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + intermediate_hidden_states
        hidden_states = self.layer_norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring
class MViTV2PreTrainedModel(PreTrainedModel):
    config_class = MViTV2Config
    base_model_prefix = "mvitv2"
    main_input_name = "pixel_values"
    _no_split_modules = ["MViTV2Block"]
    supports_gradient_checkpointing = True

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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MViTV2SelfAttention):
            if hasattr(module, "relative_positional_embeddings_height"):
                module.relative_positional_embeddings_height.data = nn.init.trunc_normal_(
                    module.relative_positional_embeddings_height.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.relative_positional_embeddings_height.dtype)
            if hasattr(module, "relative_positional_embeddings_width"):
                module.relative_positional_embeddings_width.data = nn.init.trunc_normal_(
                    module.relative_positional_embeddings_width.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.relative_positional_embeddings_width.dtype)
        elif isinstance(module, MViTV2Embeddings):
            if module.cls_token is not None:
                module.cls_token.data = nn.init.trunc_normal_(
                    module.cls_token.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.cls_token.dtype)
            if module.absolute_positional_embeddings is not None:
                module.absolute_positional_embeddings.data = nn.init.trunc_normal_(
                    module.absolute_positional_embeddings.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.absolute_positional_embeddings.dtype)


@auto_docstring
class MViTV2Model(MViTV2PreTrainedModel):
    def __init__(self, config: MViTV2Config) -> None:
        super().__init__(config)

        # If expansion doesn't happen in attention, it happens in the FFN of the previous block, so in the next block we can't possibly
        # pool before expanding (our input is already expanded)
        if not config.expand_feature_dimension_in_attention and config.attention_pool_first:
            raise ValueError(
                "It's not possible to set config.expand_feature_dimension_in_attention to False and config.attention_pool_first to True."
            )

        self.feature_size = tuple(
            config.image_size[i] // config.patch_stride_size[i] for i in range(len(config.image_size))
        )
        num_patches = product(self.feature_size) + int(config.use_cls_token)
        self.embeddings = MViTV2Embeddings(config, num_patches)
        self.encoder = MViTV2Encoder(config, self.feature_size)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], BaseModelOutput]:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoModel, AutoImageProcessor
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download
        >>> import torch

        >>> processor = AutoImageProcessor.from_pretrained("KamilaMila/mvitv2-base")
        >>> model = AutoModel.from_pretrained("KamilaMila/mvitv2-base")

        >>> # prepare an image
        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_nlvr2", filename="image2.jpeg", repo_type="dataset"
        ... )
        >>> image = Image.open(file)
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> # Inference
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ... last_hidden_states = outputs.last_hidden_state
        ... print(list(last_hidden_states.shape))
        [1, 49, 768]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output,
            self.feature_size,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # they should be properly fomatted at this point
        return encoder_outputs


@auto_docstring(
    custom_intro="""
        MViTV2 Model transformer with an image classification head on top (a linear layer on top of the mean of the final hidden states / the [CLS] token) e.g. for ImageNet.
    """
)
class MViTV2ForImageClassification(MViTV2PreTrainedModel):
    def __init__(self, config: MViTV2Config) -> None:
        super().__init__(config)
        self.mvitv2 = MViTV2Model(config)
        self.classifier_dropout = nn.Dropout(p=config.drop_rate) if config.drop_rate > 0.0 else nn.Identity()
        classifier_hidden_size = config.hidden_size * 2 ** (len(config.depths) - 1)
        self.classifier = (
            nn.Linear(classifier_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> from transformers import AutoModelForImageClassification, AutoImageProcessor
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download
        >>> import torch

        >>> processor = AutoImageProcessor.from_pretrained("KamilaMila/mvitv2-base")
        >>> model = AutoModelForImageClassification.from_pretrained("KamilaMila/mvitv2-base")

        >>> # prepare an image
        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_nlvr2", filename="image2.jpeg", repo_type="dataset"
        ... )
        >>> image = Image.open(file)
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> # Inference
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ... logits = outputs.logits
        ... predicted_class = torch.argmax(logits, dim=1).item()
        ... print(f"Predicted class: {model.config.id2label[predicted_class]}")
        Predicted class: dingo
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mvitv2(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.config.use_cls_token:
            pooler_output = sequence_output[:, 0]
        else:
            pooler_output = sequence_output.mean(1)
        pooler_output = self.classifier_dropout(pooler_output)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["MViTV2ForImageClassification", "MViTV2Model", "MViTV2PreTrainedModel"]
