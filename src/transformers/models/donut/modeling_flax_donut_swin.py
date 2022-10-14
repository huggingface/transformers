# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
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

from typing import Iterable, Optional, Tuple, Union
import math
import collections.abc
import numpy as np
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_donut_swin import DonutSwinConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "DonutSwinConfig"
_FEAT_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "https://huggingface.co/naver-clova-ix/donut-base"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]

DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "naver-clova-ix/donut-base",
    # See all Donut Swin models at https://huggingface.co/models?filter=donut
]


@flax.struct.dataclass
class DonutSwinEncoderOutput(ModelOutput):
    """
    DonutSwin encoder's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    last_hidden_state: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    reshaped_hidden_states: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
# Copied from transformers.models.swin.modeling_swin.SwinModelOutput with Swin->DonutSwin
class DonutSwinModelOutput(ModelOutput):
    """
    DonutSwin model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    last_hidden_state: jnp.ndarray = None
    pooler_output: Optional[jnp.ndarray] = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    reshaped_hidden_states: Optional[Tuple[jnp.ndarray]] = None


def window_partition(input_feature: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = jnp.reshape(input_feature, (
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    ))
    windows = jnp.transpose(input_feature, (0, 1, 3, 2, 4, 5))
    windows = jnp.reshape(windows, (-1, window_size, window_size, num_channels))
    return windows

# TODO: find a way in which you can use just one function instead of two
def window_partition_np(input_feature: np.ndarray, window_size: int) -> np.ndarray:
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = np.reshape(input_feature, (
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    ))
    windows = np.transpose(input_feature, (0, 1, 3, 2, 4, 5))
    windows = np.reshape(windows, (-1, window_size, window_size, num_channels))
    return windows


def window_reverse(windows: jnp.ndarray, window_size: int, height: int, width: int) -> jnp.ndarray:
    """
    Merges windows to produce higher resolution features.
    """
    batch_size = math.floor(windows.shape[0] / (height * width / window_size / window_size))
    windows = jnp.reshape(windows, (batch_size, height // window_size, width // window_size, window_size, window_size, -1))
    windows = jnp.transpose(windows, (0, 1, 3, 2, 4, 5))
    windows = jnp.reshape(windows, (batch_size, height, width, -1))
    return windows

class AdaptiveAveragePool1D(nn.Module):
    output_size: Union[int, Iterable[int]]

    @nn.compact
    def __call__(self, inputs):
        output_size = (
            (self.output_size,)
            if isinstance(self.output_size, int)
            else self.output_size
        )
        split = jnp.split(inputs, output_size[0], axis=1)
        stack = jnp.stack(split, axis=1)
        return jnp.mean(stack, axis=2)

class FlaxDonutSwinPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    config: DonutSwinConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        image_size, patch_size = self.config.image_size, self.config.patch_size
        num_channels, hidden_size = self.config.num_channels, self.config.embed_dim
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.projection = nn.Conv(hidden_size, kernel_size=patch_size, strides=patch_size)
    
    # TODO: use pad_values everywhere and make 0. as a constant
    def maybe_pad(self, pixel_values: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
        pad_values = [(0, 0, 0,), (0, 0, 0,), (0, 0, 0), (0, 0, 0)]

        if height % self.patch_size[0] != 0:
            pad_values[1] = (0, self.patch_size[0] - (height % self.patch_size[0]), 0)

        if width % self.patch_size[1] != 0:
            pad_values[2] = (0, self.patch_size[1] - (width % self.patch_size[1]), 0)

        if pad_values[2][1] != 0 or pad_values[3][1] != 0:
            pixel_values = jax.lax.pad(pixel_values, 0., pad_values)

        return pixel_values


    def __call__(self, pixel_values: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int]]:
        _, height, width, num_channels = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        
        embeddings = self.projection(pixel_values)
        batch_size, height, width, channels = embeddings.shape
        
        return jnp.reshape(embeddings, (batch_size, -1, channels)), (height, width)


class DonutSwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """
    config: DonutSwinConfig
    use_mask_token: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.patch_embeddings = FlaxDonutSwinPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        if self.use_mask_token:
            self.mask_token = self.param("mask_token", nn.initializers.zeros, (1, 1, self.config.embed_dim))
        else:
            self.mask_token = None

        if self.config.use_absolute_embeddings:
            self.position_embeddings = self.param("position_embeddings", nn.initializers.zeros, (1, num_patches + 1, self.config.embed_dim)) 
        else:
            self.position_embeddings = None

        self.norm = nn.LayerNorm(self.config.embed_dim)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(
        self, pixel_values: jnp.ndarray, bool_masked_pos: Optional[jnp.ndarray] = None, deterministic:bool=True
    ) -> Tuple[jnp.ndarray]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, _ = embeddings.shape

        if bool_masked_pos is not None:
            mask_tokens = jnp.broadcast_to(self.mask_token, (batch_size, seq_len, -1))
            # replace the masked visual tokens by mask_tokens
            mask = jnp.expand_dims(bool_masked_pos, -1).astype(mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings.astype(self.dtype)

        embeddings = self.dropout(embeddings, deterministic=deterministic)

        return embeddings, output_dimensions


# Copied from transformers.models.swin.modeling_swin.SwinPatchMerging
class DonutSwinPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """
    input_resolution: Tuple[int]
    layer_norm_eps: float
    dim: int
    norm_layer: nn.Module = nn.LayerNorm
    dtype: jnp.dtype = jnp.float32
    
    def setup(self) -> None:
        self.reduction = nn.Dense(2 * self.dim, use_bias=False)
        self.norm = self.norm_layer(self.layer_norm_eps)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_config = [(0, 0, 0), (width % 2, 0, 0), (0, height % 2, 0), (0, 0, 0)]
            input_feature = jax.lax.pad(input_feature, 0., pad_config)

        return input_feature

    def __call__(self, input_feature: jnp.ndarray, input_dimensions: Tuple[int, int]) -> jnp.ndarray:
        height, width = input_dimensions
        
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = jnp.reshape(input_feature, (batch_size, height, width, num_channels))
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        
        input_feature = jnp.concatenate([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = jnp.reshape(input_feature, (batch_size, -1, 4 * num_channels))  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


# Copied from transformers.models.beit.modeling_flax_beit.FlaxBeitDropPath
class FlaxDonutSwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    rate: float

    @nn.module.compact
    def __call__(self, inputs, deterministic: Optional[bool] = True):
        if self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        if deterministic:
            return inputs
        else:
            shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
            rng = self.make_rng("droppath")
            random_tensor = keep_prob + jax.random.uniform(rng, shape=shape, dtype=inputs.dtype)
            binary_tensor = jnp.floor(random_tensor)
            output = inputs / keep_prob * binary_tensor
            return output


def relative_position_index_init(window_size: Tuple[int, int]) -> jnp.ndarray:
    """
    get pair-wise relative position index for each token inside the window
    """
    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
    coords_flatten = np.reshape(coords, (2, -1))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1

    relative_position_index = relative_coords.sum(-1)
    return jnp.array(relative_position_index)

class FlaxDonutSwinRelativePositionBias(nn.Module):
    config: DonutSwinConfig
    window_size: Tuple[int, int]
    num_attention_heads: int
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        num_relative_distance = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = self.param(
            "relative_position_bias_table",
            nn.initializers.zeros,
            (num_relative_distance, self.num_attention_heads),
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        self.relative_position_index = relative_position_index_init(self.window_size)

    def __call__(self):
        index = self.relative_position_index.reshape(-1)
        shape = (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = jnp.reshape(self.relative_position_bias_table[index], shape)
        return jnp.transpose(relative_position_bias, (2, 0, 1))

class FlaxDonutSwinSelfAttention(nn.Module):
    config: DonutSwinConfig
    dim: int
    num_attention_heads: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.dim % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.dim}) is not a multiple of the number of attention heads ({self.num_attention_heads})"
            )
        self.attention_head_size = int(self.dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            self.config.window_size if isinstance(self.config.window_size, collections.abc.Iterable) else (self.config.window_size, self.config.window_size)
        )
        self.relative_position_bias = FlaxDonutSwinRelativePositionBias(self.config, self.window_size, self.num_attention_heads, dtype=self.dtype)
        self.query = nn.Dense(self.all_head_size, use_bias=self.config.qkv_bias)
        self.key = nn.Dense(self.all_head_size, use_bias=self.config.qkv_bias)
        self.value = nn.Dense(self.all_head_size, use_bias=self.config.qkv_bias)

        self.dropout = nn.Dropout(self.config.attention_probs_dropout_prob)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = False,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray]:
        batch_size, _, _ = hidden_states.shape

        query_states = jnp.reshape(self.query(hidden_states), (
            hidden_states.shape[:2] + (self.num_attention_heads, self.attention_head_size)
        ))
        value_states = jnp.reshape(self.value(hidden_states), (
            hidden_states.shape[:2] + (self.num_attention_heads, self.attention_head_size)
        ))
        key_states = jnp.reshape(self.key(hidden_states), (
            hidden_states.shape[:2] + (self.num_attention_heads, self.attention_head_size)
        ))

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")
        
        # Add relative position bias if present.
        attention_bias = jnp.expand_dims(self.relative_position_bias(), 0)
        attention_bias = attention_bias.astype(query_states.dtype)

        
        # calculate attention matrix
        depth = query_states.shape[-1]
        query_states = query_states / jnp.sqrt(depth).astype(self.dtype)
        # attn weight shape is (batch..., num_heads, q_length, kv_length)
        attn_weights = jnp.einsum('...qhd,...khd->...hqk', query_states, key_states)

        if attention_bias is not None:
            attn_weights = attn_weights + attention_bias
        
        if attention_mask is not None:
            mask_shape = attention_mask.shape[0]
            keep_as_is_shape = attn_weights.shape[1:]
            attn_weights = jnp.reshape(attn_weights,
                (batch_size // mask_shape, mask_shape) + keep_as_is_shape
            )
            attn_weights += jnp.expand_dims(jnp.expand_dims(attention_mask, 1), 0).astype(self.dtype)
            attn_weights = jnp.reshape(attn_weights, (-1,) + keep_as_is_shape)
        
        attn_weights = jax.nn.softmax(attn_weights).astype(self.dtype)

        #TODO: add attention dropout

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput
class FlaxDonutSwinSelfOutput(nn.Module):
    config: DonutSwinConfig
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self): 
        self.dense = nn.Dense(self.dim)
        self.dropout = nn.Dropout(self.config.attention_probs_dropout_prob)

    def __call__(self, hidden_states: jnp.ndarray, deterministic:bool = True) -> jnp.ndarray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        return hidden_states


class FlaxDonutSwinAttention(nn.Module):
    config: DonutSwinConfig
    dim: int
    num_heads: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self = FlaxDonutSwinSelfAttention(self.config, self.dim, self.num_heads, dtype=self.dtype)
        self.output = FlaxDonutSwinSelfOutput(self.config, self.dim, dtype=self.dtype)
    

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray]:
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], deterministic)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate
class FlaxDonutSwinIntermediate(nn.Module):
    mlp_ratio: float
    dim: int
    hidden_act: str
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.dense = nn.Dense(int(self.mlp_ratio * self.dim))
        if isinstance(self.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[self.hidden_act]
        else:
            self.intermediate_act_fn = self.hidden_act

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput
class FlaxDonutSwinOutput(nn.Module):
    mlp_ratio: float
    dim: int
    hidden_dropout_prob: float
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.dim)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def __call__(self, hidden_states: jnp.ndarray, deterministic:Optional[bool] = True) -> jnp.ndarray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# TODO: convert from torch to jax
class FlaxDonutSwinLayer(nn.Module):
    config: DonutSwinConfig
    dim: int
    input_resolution: Tuple[int, int]
    num_heads: int
    shift_size: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        
        # if window size is larger than input resolution, we don't partition windows
        self.window_size = min(self.input_resolution) if min(self.input_resolution) <= self.config.window_size else self.config.window_size
        self.layernorm_before = nn.LayerNorm(self.config.layer_norm_eps)
        self.attention = FlaxDonutSwinAttention(self.config, self.dim, self.num_heads)
        self.drop_path = FlaxDonutSwinDropPath(self.config.drop_path_rate) if self.config.drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
        self.intermediate = FlaxDonutSwinIntermediate(self.config.mlp_ratio, self.dim, self.config.hidden_act, dtype=self.dtype)
        self.output = FlaxDonutSwinOutput(self.config.mlp_ratio, self.dim, self.config.hidden_dropout_prob, dtype=self.dtype)
        

    def get_attn_mask(self, height:int, width:int, shift_size:int, window_size:int, dtype:jnp.dtype) -> jnp.ndarray:
        if shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = np.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            width_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition_np(img_mask, window_size)
            mask_windows = np.reshape(mask_windows, (-1, window_size * window_size))
            attn_mask = np.expand_dims(mask_windows, 1) - np.expand_dims(mask_windows, 2)
            attn_mask = np.select([attn_mask != 0, attn_mask == 0], [np.broadcast_to(float(-100.0), attn_mask.shape), np.broadcast_to(float(0.0), attn_mask.shape)])
            attn_mask = jnp.asarray(attn_mask, dtype=dtype)
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states:jnp.ndarray, height: int, width:int, window_size:int):
        pad_right = (window_size - width % window_size) % window_size
        pad_bottom = (window_size - height % window_size) % window_size
        pad_config = [(0, 0, 0), (0, pad_right, 0), (0, pad_bottom, 0), (0, 0, 0)]
        hidden_states = jax.lax.pad(hidden_states, 0., pad_config)
        return hidden_states, pad_config

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        deterministic: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.shape
        shortcut = hidden_states
        
        input_resolution = min(height, width)
        if input_resolution <= self.window_size:
            shift_size = 0
            window_size = input_resolution
        else:
            shift_size = self.shift_size
            window_size = self.window_size

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = jnp.reshape(hidden_states, (batch_size, height, width, channels))
        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width, window_size)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if shift_size > 0:
            shifted_hidden_states = jnp.roll(hidden_states, shift=(-shift_size, -shift_size), axis=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, window_size)
        hidden_states_windows = jnp.reshape(hidden_states_windows, (-1, window_size * window_size, channels))
        attn_mask = self.get_attn_mask(height_pad, width_pad, shift_size, window_size, dtype=hidden_states.dtype)

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, deterministic=deterministic, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = jnp.reshape(attention_output, (-1, window_size, window_size, channels))
        shifted_windows = window_reverse(attention_windows, window_size, height_pad, width_pad)

        # reverse cyclic shift
        if shift_size > 0:
            attention_windows = jnp.roll(shifted_windows, shift=(shift_size, shift_size), axis=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[1][1] > 0 or pad_values[2][1] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]

        attention_windows = jnp.reshape(attention_windows, (batch_size, height * width, channels))

        hidden_states = shortcut + self.drop_path(attention_windows, deterministic=deterministic)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output, deterministic=deterministic)

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


class FlaxDonutSwinStage(nn.Module):
    config: DonutSwinConfig
    dim: int
    input_resolution: Tuple[int, int]
    depth: int
    num_heads: int
    downsample_module: DonutSwinPatchMerging
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks =[
                FlaxDonutSwinLayer(
                    config=self.config,
                    dim=self.dim,
                    input_resolution=self.input_resolution,
                    num_heads=self.num_heads,
                    shift_size=0 if (i % 2 == 0) else self.config.window_size // 2,
                    dtype=self.dtype
                )
                for i in range(self.depth)
            ]

        # patch merging layer
        self.downsample = None
        if self.downsample_module is not None:
            self.downsample = self.downsample_module(self.input_resolution, self.config.layer_norm_eps, dim=self.dim, norm_layer=nn.LayerNorm)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        determinstic: Optional[bool] = True,
        output_attentions: Optional[bool] = False
    ) -> Tuple[jnp.ndarray]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            layer_outputs = layer_module(hidden_states, input_dimensions, determinstic, output_attentions)
            hidden_states = layer_outputs[0]

        if self.downsample_module is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(layer_outputs[0], input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# Copied from transformers.models.swin.modeling_swin.SwinEncoder with Swin->DonutSwin
class DonutSwinEncoder(nn.Module):
    config: DonutSwinConfig
    grid_size: Tuple[int, int]
    # TODO: support gradient checkpointing
    dtype: jnp.dtype = jnp.float32
    

    def setup(self):
        self.num_layers = len(self.config.depths)
        self.layers = [
                FlaxDonutSwinStage(
                    config=self.config,
                    dim=int(self.config.embed_dim * 2**i_layer),
                    input_resolution=(self.grid_size[0] // (2**i_layer), self.grid_size[1] // (2**i_layer)),
                    depth=self.config.depths[i_layer],
                    num_heads=self.config.num_heads[i_layer],
                    downsample_module=DonutSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        deterministic: bool = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, DonutSwinEncoderOutput]:
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = jnp.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
            reshaped_hidden_state = jnp.transpose(reshaped_hidden_state, (0, 3, 1, 2))
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, input_dimensions, deterministic, output_attentions)

            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)

            if output_hidden_states:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = jnp.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
                reshaped_hidden_state = jnp.transpose(reshaped_hidden_state, (0, 3, 1, 2))
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[2:]

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return DonutSwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )

SWIN_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`DonutSwinConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

SWIN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class FlaxDonutSwinModule(nn.Module):
    config: DonutSwinConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer:bool=True
    use_mask_token:bool=False

    def setup(self):
        self.num_layers = len(self.config.depths)
        self.num_features = int(self.config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = DonutSwinEmbeddings(self.config, use_mask_token=self.use_mask_token)
        self.encoder = DonutSwinEncoder(self.config, self.embeddings.patch_grid)

        self.pooler = AdaptiveAveragePool1D(1) if self.add_pooling_layer else None


    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    def __call__(   
        self,
        pixel_values: Optional[jnp.ndarray] = None,
        bool_masked_pos: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, DonutSwinModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, deterministic=deterministic)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = None
        if self.pooler is not None:
            # TODO: check torch implementation of avg_pool_global_1d, ensure jax and torch implementation matches
            pooled_output = self.pooler(sequence_output)
            batch_size = pixel_values.shape[0]
            pooled_output = jnp.reshape(pooled_output, (batch_size, -1))

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        return DonutSwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )

# Copied from transformers.models.swin.modeling_swin.SwinPreTrainedModel with Swin->DonutSwin
class FlaxDonutSwinPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # TODO: support gradient checkpointing
    # TODO: add deterministic to all the __call__ functions

    config_class = DonutSwinConfig
    base_model_prefix = "swin"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: DonutSwinConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)
        dropout_rng, droppath_rng = jax.random.split(dropout_rng)
        rngs = {"params": params_rng, "dropout": dropout_rng, "droppath": droppath_rng}

        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
    
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        bool_masked_pos: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            bool_masked_pos,
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


@add_start_docstrings(
    "The bare Donut Swin Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN_START_DOCSTRING,
)
class FlaxDonutSwinModel(FlaxDonutSwinPreTrainedModel):
    module_class = FlaxDonutSwinModule