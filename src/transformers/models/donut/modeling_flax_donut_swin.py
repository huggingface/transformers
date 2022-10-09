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

from typing import Optional, Tuple, Union
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
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
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


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = jnp.transpose(input_feature, (0, 1, 3, 2, 4, 5))
    windows = jnp.reshape(windows, (-1, window_size, window_size, num_channels))
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows: jnp.ndarray, window_size: int, height: int, width: int) -> jnp.ndarray:
    """
    Merges windows to produce higher resolution features.
    """
    batch_size = math.floor(windows.shape[0] / (height * width / window_size / window_size))
    windows = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    windows = jnp.transpose(windows, (0, 1, 3, 2, 4, 5))
    windows = jnp.reshape(windows, (batch_size, height, width, -1))
    return windows


# Copied from transformers.models.swin.modeling_swin.SwinEmbeddings with Swin->DonutSwin
class DonutSwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """
    config: DonutSwinConfig
    use_mask_token: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.patch_embeddings = DonutSwinPatchEmbeddings(self.config, dtype=self.dtype)
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
        self, pixel_values: jnp.ndarray, bool_masked_pos: Optional[jnp.ndarray] = None
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

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


# Copied from transformers.models.swin.modeling_swin.SwinPatchEmbeddings
class DonutSwinPatchEmbeddings(nn.Module):
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

        self.projection = nn.Conv(hidden_size, kernel_size=patch_size, stride=patch_size)

    def maybe_pad(self, pixel_values: jnp.ndarray, height: int, width: int):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = jax.lax.pad(pixel_values, 0, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = jax.lax.pad(pixel_values, 0, pad_values)
        return pixel_values

    def __call__(self, pixel_values: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int]]:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels)), output_dimensions


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
    config: DonutSwinConfig
    input_resolution: Tuple[int]
    dim: int
    norm_layer: nn.Module = nn.LayerNorm
    dtype: jnp.dtype = jnp.float32
    
    def setup(self) -> None:
        self.reduction = nn.Dense(2 * self.dim, use_bias=False)
        self.norm = self.norm_layer(self.config.layer_norm_eps)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = jax.lax.pad(input_feature, 0, pad_values)

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
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
    coords_flatten = np.reshape(coords, (2, -1))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1

    relative_position_index = np.zeros(shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return jnp.array(relative_position_index)

class FlaxDonutSwinRelativePositionBias(nn.Module):
    config: DonutSwinConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        num_relative_distance = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) + 3
        self.relative_position_bias_table = self.param(
            "relative_position_bias_table",
            nn.initializers.zeros,
            (num_relative_distance, self.config.num_attention_heads),
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        self.relative_position_index = relative_position_index_init(self.window_size)

    def __call__(self):
        index = self.relative_position_index.reshape(-1)
        shape = (self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
        relative_position_bias = self.relative_position_bias_table[index].reshape(shape)  # Wh*Ww,Wh*Ww,nH
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
        self.relative_position_bias = FlaxDonutSwinRelativePositionBias(self.config, self.window_size, dtype=self.dtype)
        self.query = nn.Dense(self.all_head_size, use_bias=self.config.qkv_bias)
        self.key = nn.Dense(self.all_head_size, use_bias=self.config.qkv_bias)
        self.value = nn.Dense(self.all_head_size, use_bias=self.config.qkv_bias)

        self.dropout = nn.Dropout(self.config.attention_probs_dropout_prob)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = False,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray]:
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")
        
        # Add relative position bias if present.
        attention_bias = jnp.expand_dims(self.relative_position_bias(), 0)
        attention_bias = attention_bias.astype(query_states.dtype)

        if attention_mask is not None:
            attention_bias += attention_mask.astype(self.dtype)

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            attention_bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        if head_mask is not None:
            attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, head_mask)

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

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

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
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
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
        self.dense = nn.Dense(int(self.mlp_ratio * self.dim), self.dim)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
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
        
        self.window_size = self.config.window_size
        self.set_shift_and_window_size(self.input_resolution)
        self.layernorm_before = nn.LayerNorm(self.config.layer_norm_eps)
        self.attention = FlaxDonutSwinAttention(self.config, self.dim, self.num_heads)
        self.drop_path = FlaxDonutSwinDropPath(self.config.drop_path_rate) if self.config.drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
        self.intermediate = FlaxDonutSwinIntermediate(self.config.mlp_ratio, self.dim, self.config.hidden_act, dtype=self.dtype)
        self.output = FlaxDonutSwinOutput(self.config.mlp_ratio, self.dim, self.config.hidden_dropout_prob, dtype=self.dtype)

    def set_shift_and_window_size(self, input_resolution:Tuple[int, int]):
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(input_resolution)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = jnp.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        self.set_shift_and_window_size(input_dimensions)
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = jnp.roll(hidden_states, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = jnp.roll(shifted_windows, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(attention_windows)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


# Copied from transformers.models.swin.modeling_swin.SwinStage with Swin->DonutSwin
class FlaxDonutSwinStage(nn.Module):
    config: DonutSwinConfig
    dim: int
    input_resolution: Tuple[int, int]
    depth: int
    num_heads: int
    downsample: DonutSwinPatchMerging
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
        if self.downsample is not None:
            self.downsample = self.downsample(self.input_resolution, dim=self.dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[jnp.ndarray] = None,
        determinstic: Optional[bool] = True,
        output_attentions: Optional[bool] = False
    ) -> Tuple[jnp.ndarray]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

        if self.downsample is not None:
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
    gradient_checkpointing: bool = False
    dtype: jnp.dtype = jnp.float32
    

    def setup(self):
        self.num_layers = len(self.config.depths)
        dpr = [x.item() for x in jnp.linspace(0, self.config.drop_path_rate, sum(self.config.depths))]
        self.layers = [
                FlaxDonutSwinStage(
                    config=self.config,
                    dim=int(self.config.embed_dim * 2**i_layer),
                    input_resolution=(self.grid_size[0] // (2**i_layer), self.grid_size[1] // (2**i_layer)),
                    depth=self.config.depths[i_layer],
                    num_heads=self.config.num_heads[i_layer],
                    drop_path=dpr[sum(self.config.depths[:i_layer]): sum(self.config.depths[: i_layer + 1])],
                    downsample=DonutSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[jnp.ndarray] = None,
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
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hidden_states, input_dimensions, layer_head_mask
                )
            else:
                layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)

            if output_hidden_states:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
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


# Copied from transformers.models.swin.modeling_swin.SwinPreTrainedModel with Swin->DonutSwin
class DonutSwinPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DonutSwinConfig
    base_model_prefix = "swin"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DonutSwinEncoder):
            module.gradient_checkpointing = value


SWIN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DonutSwinConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SWIN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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
    "The bare Donut Swin Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN_START_DOCSTRING,
)
class FlaxDonutSwinModel(DonutSwinPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = DonutSwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = DonutSwinEncoder(config, self.embeddings.patch_grid)

        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=DonutSwinModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DonutSwinModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

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
